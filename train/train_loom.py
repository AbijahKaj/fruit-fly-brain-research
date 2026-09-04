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

Loss per (type, side): population rate during the last third of an ipsilateral
loom should exceed every other stimulus by a margin, with a selectivity index
toward 1. Output: out/fitted-params.json with LC4/LPLC2 types (tau, bias,
restV) and their input pairs appended; the browser picks them up as fitted.
"""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import numpy as np
import torch
from graph_torch import Graph, FlyvisModel

HERE = Path(__file__).parent
DEG = math.pi / 180
LC_TYPES = ["LC4", "LPLC2"]
KINDS = ["loomL", "loomR", "recedeL", "recedeR", "transL", "transR", "grating", "grating"]


def angular_distance(az1, el1, az2, el2):
    """Great-circle distance between directions given as (az, el), broadcasting."""
    cosd = torch.sin(el1) * torch.sin(el2) + torch.cos(el1) * torch.cos(el2) * torch.cos(az1 - az2)
    return torch.acos(cosd.clamp(-1, 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--T", type=int, default=100, help="time steps per stimulus")
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--max-el", type=float, default=90)
    ap.add_argument("--graph", default=str(HERE.parent / "data" / "out" / "optic-v2"))
    ap.add_argument("--params", default=str(HERE / "out" / "fitted-params.json"), help="stage 1 output")
    ap.add_argument("--out", default=str(HERE / "out" / "fitted-params.json"))
    ap.add_argument("--margin", type=float, default=1.0, help="loom minus control rate margin")
    ap.add_argument("--reg", type=float, default=0.1)
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
    masks = model.grad_masks(LC_TYPES, train_extra_pairs_only=True)
    init = {k: v.detach().clone() for k, v in model.named_parameters()}
    n_extra = len(model.pairs) - model.n_fv_pairs
    print(f"trainable: {n_extra} input pairs onto {LC_TYPES}, plus their tau/bias")

    tn = np.array([g.types[t] for t in g.unit_type])
    lam_w = fv["photoreceptor"]["laminaInput"]
    lam_units = np.where(np.isin(tn, list(lam_w.keys())) & (g.unit_col >= 0))[0]
    lam_cols = torch.tensor(g.unit_col[lam_units], device=dev)
    lam_wt = torch.tensor([lam_w[tn[i]] for i in lam_units], device=dev, dtype=torch.float32)
    lam_units_t = torch.tensor(lam_units, device=dev)
    col_az = torch.tensor(g.col_az, device=dev)
    col_el = torch.tensor(g.col_el, device=dev)
    col_side = torch.tensor(g.col_side, device=dev)
    pr = fv["photoreceptor"]

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
    print(f"readout: {[(k, len(v)) for k, v in groups.items()]}; lamina inputs {len(lam_units)}")

    opt = torch.optim.Adam([p for p in model.parameters()], lr=args.lr)
    T, dt = args.T, args.dt
    B = len(KINDS)
    t_axis = torch.arange(T, device=dev, dtype=torch.float32) * dt
    dur = T * dt

    def disc_centre(side: int):
        """Random RF-ish centre within one eye: az 25-110 deg on that side, el -30..30."""
        az = (25 + 85 * torch.rand((), device=dev)) * DEG * (1 if side == 1 else -1)
        el = (torch.rand((), device=dev) - 0.5) * 60 * DEG
        return az, el

    def make_ext():
        lum = torch.full((T, B, len(g.col_az)), 0.5, device=dev)
        for b, kind in enumerate(KINDS):
            if kind == "grating":
                theta = torch.rand((), device=dev) * 2 * math.pi
                lam = (15 + 25 * torch.rand((), device=dev)) * DEG
                tf = 0.5 + 2.5 * torch.rand((), device=dev)
                proj = col_az * torch.cos(theta) + col_el * torch.sin(theta)
                phase = 2 * math.pi * (proj[None] / lam - tf * t_axis[:, None])
                lum[:, b] = 0.5 + 0.5 * torch.sin(phase)
                continue
            side = 1 if kind.endswith("R") else 0
            az0, el0 = disc_centre(side)
            if kind.startswith("trans"):
                # fixed 12 deg disc sweeping 60 deg in azimuth over the trial
                direction = 1.0 if torch.rand(()) < 0.5 else -1.0
                az_t = az0 + direction * (t_axis / dur - 0.5) * 60 * DEG
                radius = torch.full((T,), 12 * DEG, device=dev)
                d = angular_distance(az_t[:, None], el0, col_az[None, :], col_el[None, :])
            else:
                # l/v looming: half-angle theta(t) = atan(l / (v (t_c - t))); l/v in 20-80 ms
                lv = 0.02 + 0.06 * torch.rand((), device=dev)
                tc = dur * 1.02
                if kind.startswith("loom"):
                    radius = torch.atan(lv / (tc - t_axis).clamp_min(1e-3))
                else:
                    radius = torch.atan(lv / (tc - t_axis).clamp_min(1e-3)).flip(0)
                radius = radius.clamp(2 * DEG, 70 * DEG)
                d = angular_distance(az0, el0, col_az[None, :], col_el[None, :]).expand(T, -1)
            inside = (d < radius[:, None]).float()
            lum[:, b] = 0.5 * (1 - inside) + 0.05 * inside
        rR = pr["restOffset"] + pr["stimGain"] * lum
        ext = torch.zeros(T, B, g.n, device=dev)
        ext[:, :, lam_units_t] = rR[:, :, lam_cols] * lam_wt[None, None, :]
        return ext

    kind_idx = {k: [b for b, kk in enumerate(KINDS) if kk == k] for k in set(KINDS)}
    t0 = time.time()
    for step in range(args.steps):
        ext = make_ext()
        rates = model(ext, dt, record=rec.tolist())            # (T, B, nrec)
        resp = rates[-T // 3:].mean(0)                          # (B, nrec)
        loss_sel = 0.0
        report = {}
        for (st, side), pos in rec_pos.items():
            r = resp[:, pos].mean(1)                            # (B,)
            ipsi = r[kind_idx[f"loom{side}"]].mean()
            others = torch.cat([r[kind_idx[k]] for k in KINDS if k != f"loom{side}"])
            worst = others.max()
            sel = (ipsi - worst) / (ipsi + worst + 1e-3)
            loss_sel = loss_sel + torch.relu(args.margin - (ipsi - worst)) + (1 - sel) + torch.relu(0.3 - ipsi)
            report[f"{st}{side}"] = (ipsi.item(), worst.item())
        loss_rate = torch.relu(rates - 5.0).pow(2).mean() * 10
        loss_reg = sum(((p - init[k]) ** 2 * masks[k]).mean() for k, p in model.named_parameters()) * args.reg
        loss = loss_sel + loss_rate + loss_reg
        opt.zero_grad()
        loss.backward()
        for k, p in model.named_parameters():
            if p.grad is not None:
                p.grad *= masks[k]
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 10 == 0 or step == args.steps - 1:
            rep = " ".join(f"{k} {a:.2f}/{b:.2f}" for k, (a, b) in report.items())
            print(f"step {step:5d} loss {loss.item():.4f} sel {float(loss_sel):.4f} | loom/worst: {rep}  {time.time() - t0:.0f}s", flush=True)

    # resting membrane under grey for the browser's homeostat target
    with torch.no_grad():
        ext = torch.zeros(T, 1, g.n, device=dev)
        ext[:, :, lam_units_t] = (pr["restOffset"] + pr["stimGain"] * 0.5) * lam_wt[None, None, :]
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
