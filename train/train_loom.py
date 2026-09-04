"""
Stage 2 fit: make the looming detectors (LC4, LPLC2) respond to an approaching
object and not to wide-field motion, on top of the milestone 3 fit.

  python train_loom.py --device cuda --steps 800                # on the 5090 box
  python train_loom.py --device cpu --steps 2 --max-el 20 --T 20 # smoke test

Everything from train_optic.py stays frozen (flyvis types, flyvis pairs).
Trainable: tau and bias of LC4 / LPLC2, and the strength of every input pair
onto them (T2 -> LC4, T4/T5 -> LPLC2, ...), initialised at the browser's
pooling scale. Stimuli on the real column lattice, one per batch slot:

  loom      dark disc expanding at one eye with the l/v profile
  recede    the same disc shrinking
  translate a fixed-size disc sweeping across one eye
  grating   wide-field drifting grating (what the optomotor loop sees)

Loss per (type, side): the top-k cells' rate during the scored window of an
ipsilateral loom should exceed every other stimulus by a margin, with a
selectivity index toward 1. Top-k, not the population mean: only the cells whose
receptive field sits at the loom centre see all four expanding edges, and that
is what the giant fiber reads. Output: out/fitted-params.json with LC4/LPLC2 types (tau, bias,
restV) and their input pairs appended; the browser picks them up as fitted.
"""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import numpy as np
import torch
from graph_torch import Graph, FlyvisModel
from train_optic import PD

HERE = Path(__file__).parent
DEG = math.pi / 180
LC_TYPES = ["LC4", "LPLC2"]
KINDS = ["loomL", "loomR", "recedeL", "recedeR", "transL", "transR", "grating", "grating"]
# l/v of the looming object, seconds (Drosophila escapes are tested at 10-80 ms; slower here so
# the edge speed stays inside what the stage 1 fit tuned T4/T5 for), and the scored window (last fraction).
LV_MIN, LV_MAX, WIN = 0.1, 0.4, 0.4


def angular_distance(az1, el1, az2, el2):
    """Great-circle distance between directions given as (az, el), broadcasting."""
    cosd = torch.sin(el1) * torch.sin(el2) + torch.cos(el1) * torch.cos(el2) * torch.cos(az1 - az2)
    return torch.acos(cosd.clamp(-1, 1))



class Stimuli:
    """Lamina drive for the trainer's stimulus kinds on the real column lattice."""

    def __init__(self, g: Graph, fv: dict, dev: str, T: int, dt: float, lv=(LV_MIN, LV_MAX)):
        self.g, self.dev, self.T, self.dt, self.lv = g, dev, T, dt, lv
        tn = np.array([g.types[t] for t in g.unit_type])
        self.tn = tn
        lam_w = fv["photoreceptor"]["laminaInput"]
        lam_units = np.where(np.isin(tn, list(lam_w.keys())) & (g.unit_col >= 0))[0]
        self.lam_cols = torch.tensor(g.unit_col[lam_units], device=dev)
        self.lam_wt = torch.tensor([lam_w[tn[i]] for i in lam_units], device=dev, dtype=torch.float32)
        self.lam_units_t = torch.tensor(lam_units, device=dev)
        self.n_lam = len(lam_units)
        self.col_az = torch.tensor(g.col_az, device=dev)
        self.col_el = torch.tensor(g.col_el, device=dev)
        self.pr = fv["photoreceptor"]
        self.t_axis = torch.arange(T, device=dev, dtype=torch.float32) * dt
        self.dur = T * dt

    def disc_centre(self, side: int):
        """Random RF-ish centre within one eye: az 25-110 deg on that side, el -30..30."""
        dev = self.dev
        az = (25 + 85 * torch.rand((), device=dev)) * DEG * (1 if side == 1 else -1)
        el = (torch.rand((), device=dev) - 0.5) * 60 * DEG
        return az, el

    def to_ext(self, lum: torch.Tensor) -> torch.Tensor:
        """Column luminance (T, B, C) -> lamina ext (T, B, n)."""
        rR = self.pr["restOffset"] + self.pr["stimGain"] * lum
        ext = torch.zeros(lum.shape[0], lum.shape[1], self.g.n, device=self.dev)
        ext[:, :, self.lam_units_t] = rR[:, :, self.lam_cols] * self.lam_wt[None, None, :]
        return ext

    def kinds(self, kinds: list[str]) -> torch.Tensor:
        """One stimulus per entry of `kinds` (loomL/R, recedeL/R, transL/R, grating)."""
        T, dev, t_axis, dur = self.T, self.dev, self.t_axis, self.dur
        col_az, col_el = self.col_az, self.col_el
        lum = torch.full((T, len(kinds), len(self.g.col_az)), 0.5, device=dev)
        for b, kind in enumerate(kinds):
            if kind == "grating":
                theta = torch.rand((), device=dev) * 2 * math.pi
                lam = (15 + 25 * torch.rand((), device=dev)) * DEG
                tf = 0.5 + 2.5 * torch.rand((), device=dev)
                proj = col_az * torch.cos(theta) + col_el * torch.sin(theta)
                phase = 2 * math.pi * (proj[None] / lam - tf * t_axis[:, None])
                lum[:, b] = 0.5 + 0.5 * torch.sin(phase)
                continue
            side = 1 if kind.endswith("R") else 0
            az0, el0 = self.disc_centre(side)
            if kind.startswith("trans"):
                # fixed 12 deg disc sweeping 60 deg in azimuth over the trial
                direction = 1.0 if torch.rand(()) < 0.5 else -1.0
                az_t = az0 + direction * (t_axis / dur - 0.5) * 60 * DEG
                radius = torch.full((T,), 12 * DEG, device=dev)
                d = angular_distance(az_t[:, None], el0, col_az[None, :], col_el[None, :])
            else:
                # l/v looming: half-angle theta(t) = atan(l / (v (t_c - t)))
                lv = self.lv[0] + (self.lv[1] - self.lv[0]) * torch.rand((), device=dev)
                tc = dur * 1.02
                radius = torch.atan(lv / (tc - t_axis).clamp_min(1e-3))
                if kind.startswith("recede"):
                    radius = radius.flip(0)
                radius = radius.clamp(2 * DEG, 70 * DEG)
                d = angular_distance(az0, el0, col_az[None, :], col_el[None, :]).expand(T, -1)
            inside = (d < radius[:, None]).float()
            lum[:, b] = 0.5 * (1 - inside) + 0.05 * inside
        return self.to_ext(lum)

    def gratings(self, Bg: int):
        """Random drifting gratings as in train_optic.py. Returns ext, theta."""
        dev, T, t_axis = self.dev, self.T, self.t_axis
        theta = torch.rand(Bg, device=dev) * 2 * math.pi
        lam = (15 + 25 * torch.rand(Bg, device=dev)) * DEG
        tf = 0.5 + 2.5 * torch.rand(Bg, device=dev)
        c = 0.5 + 0.5 * torch.rand(Bg, device=dev)
        proj = self.col_az[None, :] * torch.cos(theta)[:, None] + self.col_el[None, :] * torch.sin(theta)[:, None]
        phase = 2 * math.pi * (proj[None] / lam[None, :, None] - tf[None, :, None] * t_axis[:, None, None])
        lum = 0.5 + 0.5 * c[None, :, None] * torch.sin(phase)
        return self.to_ext(lum), theta

    def grey(self, B: int = 1) -> torch.Tensor:
        return self.to_ext(torch.full((self.T, B, len(self.g.col_az)), 0.5, device=self.dev))


def load_model(graph: str, params: str, dev: str, max_el: float = 90):
    g0 = Graph(Path(graph))
    fv = json.load(open(params))
    tn0 = np.array([g0.types[t] for t in g0.unit_type])
    keep = (g0.unit_role == g0.roles.index("optic")) | np.isin(tn0, LC_TYPES)
    if max_el < 90:
        col_ok = np.abs(g0.col_el) <= max_el * DEG
        keep &= ((g0.unit_col >= 0) & col_ok[np.clip(g0.unit_col, 0, None)]) | np.isin(tn0, LC_TYPES)
    g = g0.subgraph(keep)
    model = FlyvisModel(g, fv, device=dev, extra_post_types=LC_TYPES, extra_scale=0.005)
    return g, fv, model


def topk_rate(resp: torch.Tensor, pos: torch.Tensor, k: int) -> torch.Tensor:
    """(B, nrec) -> (B,): mean of the k most active recorded units in `pos`."""
    return resp[:, pos].topk(min(k, len(pos)), dim=1).values.mean(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--T", type=int, default=200, help="time steps per stimulus")
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--max-el", type=float, default=90)
    ap.add_argument("--graph", default=str(HERE.parent / "data" / "out" / "optic-v2"))
    ap.add_argument("--params", default=str(HERE / "out" / "fitted-params.json"), help="stage 1 output")
    ap.add_argument("--out", default=str(HERE / "out" / "fitted-params.json"))
    ap.add_argument("--margin", type=float, default=1.0, help="loom minus control rate margin")
    ap.add_argument("--reg", type=float, default=0.1)
    ap.add_argument("--topk", type=int, default=5, help="cells per group that carry the readout")
    ap.add_argument("--joint", action="store_true",
                    help="unfreeze everything and keep the stage 1 grating DS loss alongside (lets the OFF pathway adapt)")
    ap.add_argument("--ds-weight", type=float, default=1.0)
    ap.add_argument("--ds-batch", type=int, default=4)
    args = ap.parse_args()
    dev = args.device

    g0 = Graph(Path(args.graph))
    fv = json.load(open(args.params))
    tn0 = np.array([g0.types[t] for t in g0.unit_type])
    keep = (g0.unit_role == g0.roles.index("optic")) | np.isin(tn0, LC_TYPES)
    if args.max_el < 90:
        col_ok = np.abs(g0.col_el) <= args.max_el * DEG
        keep &= ((g0.unit_col >= 0) & col_ok[np.clip(g0.unit_col, 0, None)]) | np.isin(tn0, LC_TYPES)
    g = g0.subgraph(keep)
    print(f"model: {g.n} units, {len(g.pre)} edges on {dev}")
    model = FlyvisModel(g, fv, device=dev, extra_post_types=LC_TYPES, extra_scale=0.001)
    masks = model.grad_masks(None if args.joint else LC_TYPES, train_extra_pairs_only=not args.joint)
    init = {k: v.detach().clone() for k, v in model.named_parameters()}
    n_extra = len(model.pairs) - model.n_fv_pairs
    print(f"trainable: {n_extra} input pairs onto {LC_TYPES}, plus their tau/bias")

    stim = Stimuli(g, fv, dev, args.T, args.dt)
    tn = stim.tn
    groups = {}
    for st in LC_TYPES:
        for side, sidx in [("L", 0), ("R", 1)]:
            u = np.where((tn == st) & (g.unit_side == sidx))[0]
            if len(u):
                groups[(st, side)] = torch.tensor(u, device=dev)
    rec = torch.tensor(np.concatenate([v.cpu().numpy() for v in groups.values()]), device=dev)
    rec_pos, off = {}, 0
    for k, v in groups.items():
        rec_pos[k] = torch.arange(off, off + len(v), device=dev)
        off += len(v)
    print(f"readout: {[(k, len(v)) for k, v in groups.items()]}; lamina inputs {stim.n_lam}")

    opt = torch.optim.Adam([p for p in model.parameters()], lr=args.lr)
    T, dt = args.T, args.dt

    # stage 1 readout (T4/T5 subtypes) for the joint mode
    ds_groups = {}
    for st in PD:
        for side, sidx in [("L", 0), ("R", 1)]:
            u = np.where((tn == st) & (g.unit_side == sidx))[0]
            if len(u):
                ds_groups[(st, side)] = torch.tensor(u, device=dev)
    ds_rec = torch.tensor(np.concatenate([v.cpu().numpy() for v in ds_groups.values()]), device=dev) if ds_groups else None
    ds_pos, off = {}, 0
    for k, v in ds_groups.items():
        ds_pos[k] = torch.arange(off, off + len(v), device=dev)
        off += len(v)

    def ds_loss(resp, theta):
        total = 0.0
        for (st, side), pos in ds_pos.items():
            r = resp[:, pos].mean(1)
            cosv = torch.cos(theta - PD[st][side])
            rc, cc = r - r.mean(), cosv - cosv.mean()
            corr = (rc * cc).sum() / (rc.norm() * cc.norm() + 1e-6)
            depth = r.std() / (r.mean() + 1e-3)
            total = total + (1 - corr) + 0.5 * torch.relu(0.5 - depth) - 0.2 * torch.log(r.mean() + 1e-3).clamp(max=0)
        return total

    kind_idx = {k: [b for b, kk in enumerate(KINDS) if kk == k] for k in set(KINDS)}
    t0 = time.time()
    for step in range(args.steps):
        ext = stim.kinds(KINDS)
        rates = model(ext, dt, record=rec.tolist())            # (T, B, nrec)
        resp = rates[-int(T * WIN):].mean(0)                    # (B, nrec)
        loss_sel = 0.0
        report = {}
        for (st, side), pos in rec_pos.items():
            r = topk_rate(resp, pos, args.topk)                 # (B,)
            ipsi = r[kind_idx[f"loom{side}"]].mean()
            others = torch.cat([r[kind_idx[k]] for k in KINDS if k != f"loom{side}"])
            worst = others.max()
            sel = (ipsi - worst) / (ipsi + worst + 1e-3)
            loss_sel = loss_sel + torch.relu(args.margin - (ipsi - worst)) + (1 - sel) + torch.relu(0.3 - ipsi)
            report[f"{st}{side}"] = (ipsi.item(), worst.item())
        loss_rate = torch.relu(rates - 5.0).pow(2).mean() * 10
        loss_reg = sum(((p - init[k]) ** 2 * masks[k]).mean() for k, p in model.named_parameters()) * args.reg
        loss_ds = torch.zeros((), device=dev)
        if args.joint and ds_rec is not None:
            gext, theta = stim.gratings(args.ds_batch)
            grates = model(gext, dt, record=ds_rec.tolist())
            loss_ds = ds_loss(grates[T // 3:].mean(0), theta) * args.ds_weight
        loss = loss_sel + loss_rate + loss_reg + loss_ds
        opt.zero_grad()
        loss.backward()
        for k, p in model.named_parameters():
            if p.grad is not None:
                p.grad *= masks[k]
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 10 == 0 or step == args.steps - 1:
            rep = " ".join(f"{k} {a:.2f}/{b:.2f}" for k, (a, b) in report.items())
            print(f"step {step:5d} loss {loss.item():.4f} sel {float(loss_sel):.4f} ds {loss_ds.item():.3f} | loom/worst: {rep}  {time.time() - t0:.0f}s", flush=True)

    # resting membrane under grey for the browser's homeostat target
    with torch.no_grad():
        ext = stim.grey()
        w = model.weights()
        tau = torch.exp(model.log_tau)[model.unit_type_t].clamp_min(dt)
        bias = model.bias[model.unit_type_t]
        x = bias.expand(1, g.n).clone()
        for t in range(2 * T):
            r = model.r_max * torch.tanh(torch.relu(x) / model.r_max)
            drive = torch.zeros(1, g.n, device=dev).index_add_(1, model.e_post, r[:, model.e_pre] * w[None, :])
            x = x + (dt / tau) * (-x + bias + drive + ext[0])
        rest_v = {st: float(x[0, torch.tensor(np.where(tn == st)[0], device=dev)].mean()) for st in LC_TYPES}
    print("rest membrane under grey:", rest_v)
    model.export(Path(args.out), fv, rest_v=rest_v, source="fitted on MaleCNS optic-v2 (train_optic.py + train_loom.py)")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
