"""
Stage 2 fit: make the looming detectors (LC4, LPLC2) respond to an approaching
object and not to wide-field motion, on top of the grating (stage 1) fit.

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
  static    a static grating: motion detectors and looming detectors must stay quiet
            (the rendered scene is full of static contrast; without this term the
            fitted T4/T5 sit at the rate ceiling at rest)

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
from graph_torch import Graph, FlyvisModel, pooling_types
from train_optic import PD

HERE = Path(__file__).parent
DEG = math.pi / 180
LC_TYPES = ["LC4", "LPLC2"]
HS_TYPES = ["HSE", "HSN", "HSS"]
FLOW_KINDS = ["static", "rotCCW", "rotCW", "transFwd", "transAsym"]
KINDS = ["loomL", "loomR", "recedeL", "recedeR", "transL", "transR", "grating", "grating", "static"]
# Half of the object trials play over a static high-contrast grating instead of grey, as in the scene.
P_STRUCTURED_BG = 0.5
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
        """Random RF-ish centre within one eye: az 5-110 deg on that side (frontal included, so
        head-on approaches are covered), el -30..30."""
        dev = self.dev
        az = (5 + 105 * torch.rand((), device=dev)) * DEG * (1 if side == 1 else -1)
        el = (torch.rand((), device=dev) - 0.5) * 60 * DEG
        return az, el

    def to_ext(self, lum: torch.Tensor) -> torch.Tensor:
        """Column luminance (T, B, C) -> lamina ext (T, B, n)."""
        return self.ext_from_photoreceptor(self.pr["restOffset"] + self.pr["stimGain"] * lum)

    def ext_from_photoreceptor(self, rR: torch.Tensor) -> torch.Tensor:
        """Photoreceptor rate per column (T, B, C) -> lamina ext (T, B, n)."""
        ext = torch.zeros(rR.shape[0], rR.shape[1], self.g.n, device=self.dev)
        ext[:, :, self.lam_units_t] = rR[:, :, self.lam_cols] * self.lam_wt[None, None, :]
        return ext

    def kinds(self, kinds: list[str]) -> torch.Tensor:
        """One stimulus per entry of `kinds` (loomL/R, recedeL/R, transL/R, grating)."""
        T, dev, t_axis, dur = self.T, self.dev, self.t_axis, self.dur
        col_az, col_el = self.col_az, self.col_el
        lum = torch.full((T, len(kinds), len(self.g.col_az)), 0.5, device=dev)

        def grating(tf, contrast):
            theta = torch.rand((), device=dev) * 2 * math.pi
            lam = (15 + 25 * torch.rand((), device=dev)) * DEG
            proj = col_az * torch.cos(theta) + col_el * torch.sin(theta)
            phase = 2 * math.pi * (proj[None] / lam - tf * t_axis[:, None])
            return 0.5 + 0.5 * contrast * torch.sin(phase)

        for b, kind in enumerate(kinds):
            if kind == "grating":
                lum[:, b] = grating(0.5 + 2.5 * torch.rand((), device=dev), 0.5 + 0.5 * torch.rand((), device=dev))
                continue
            if kind == "static":
                lum[:, b] = grating(0.0, 0.5 + 0.5 * torch.rand((), device=dev))
                continue
            if torch.rand(()) < P_STRUCTURED_BG:
                lum[:, b] = grating(0.0, 0.5 + 0.5 * torch.rand((), device=dev))
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
            lum[:, b] = lum[:, b] * (1 - inside) + 0.05 * inside
        return self.to_ext(lum)

    def flow(self, kinds: list[str], radius: float = 10.0) -> torch.Tensor:
        """Self-motion flow fields by ray casting a striped cylinder wall (radius `radius`, the
        drum) and a checkered ground 2 units below the fly. Kinds: static, rotCCW / rotCW (yaw,
        0.5-2 rad/s), transFwd (forward flight, 1-3 units/s, centred), transAsym (same, but the
        wall is 3 units nearer on one side). Returns lamina ext (T, B, n)."""
        T, dev, t_axis = self.T, self.dev, self.t_axis
        C = len(self.g.col_az)
        lum = torch.full((T, len(kinds), C), 0.5, device=dev)
        # column directions in the fly frame
        dx0 = torch.sin(self.col_az) * torch.cos(self.col_el)
        dy = torch.sin(self.col_el)
        dz0 = -torch.cos(self.col_az) * torch.cos(self.col_el)
        for b, kind in enumerate(kinds):
            omega, speed, x_off = 0.0, 0.0, 0.0
            if kind.startswith("rot"):
                omega = (0.5 + 1.5 * torch.rand(()).item()) * (1 if kind == "rotCCW" else -1)
            elif kind.startswith("trans"):
                speed = 1.0 + 2.0 * torch.rand(()).item()
                if kind == "transAsym":
                    x_off = 3.0 * (1 if torch.rand(()) < 0.5 else -1)
            period = (15 + 15 * torch.rand(()).item()) * DEG          # wall stripe period
            checker = 1.0 + 1.5 * torch.rand(()).item()               # ground cell size
            for t in range(T):
                tt = t_axis[t].item()
                yaw = omega * tt                                       # positive = left
                fx, fz = x_off, -speed * tt
                c, sn = math.cos(yaw), math.sin(yaw)
                dx = c * dx0 - sn * dz0                                # fly -> world
                dz = sn * dx0 + c * dz0
                # cylinder |p + s d| = R in the xz plane
                a = dx * dx + dz * dz
                bq = 2 * (fx * dx + fz * dz)
                cq = fx * fx + fz * fz - radius * radius
                disc = (bq * bq - 4 * a * cq).clamp_min(0)
                s_wall = (-bq + torch.sqrt(disc)) / (2 * a + 1e-9)
                s_ground = torch.where(dy < -1e-3, -2.0 / dy, torch.full_like(dy, 1e9))
                on_ground = s_ground < s_wall
                s = torch.where(on_ground, s_ground, s_wall)
                hx, hz = fx + s * dx, fz + s * dz
                phi = torch.atan2(hx, -hz)
                wall_lum = 0.5 + 0.5 * torch.sin(2 * math.pi * phi / period)
                parity = (torch.floor(hx / checker) + torch.floor(hz / checker)) % 2
                ground_lum = 0.3 + 0.4 * parity
                lum[t, b] = torch.where(on_ground, ground_lum, wall_lum)
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


def load_model(graph: str, params: str, dev: str, max_el: float = 90, extra_types: list[str] | None = None,
               dt: float = 0.005, T: int = 160):
    """Graph (optic lobe + every pooling cell), params, model. Pooling types flyvis does not cover
    become extra types: their inputs at the browser's pooling scale, their bias set like the
    browser's homeostat (rest membrane 0.3 under grey) and exported, so the browser runs the same
    operating point instead of re-centring them itself."""
    extra_types = list(extra_types or LC_TYPES)
    g0 = Graph(Path(graph))
    fv = json.load(open(params))
    pool = [t for t in pooling_types(g0) if t not in fv["types"]]
    extra_types = extra_types + [t for t in pool if t not in extra_types]
    tn0 = np.array([g0.types[t] for t in g0.unit_type])
    keep = (g0.unit_role == g0.roles.index("optic")) | np.isin(tn0, extra_types)
    if max_el < 90:
        col_ok = np.abs(g0.col_el) <= max_el * DEG
        keep &= ((g0.unit_col >= 0) & col_ok[np.clip(g0.unit_col, 0, None)]) | np.isin(tn0, extra_types)
    g = g0.subgraph(keep)
    # new input pairs start at the browser's pooling scale (lptcScale)
    model = FlyvisModel(g, fv, device=dev, extra_post_types=extra_types, extra_scale=0.001, pool_scale=0.001)
    if pool:
        stim = Stimuli(g, fv, dev, T, dt)
        rest = model.init_type_bias(stim.grey()[0, 0], dt, pool)
        print(f"pooling types without fitted params: {len(pool)}; homeostat-style bias set, rest membrane e.g. "
              + ", ".join(f"{t} {v:.2f}" for t, v in list(rest.items())[:6]))
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
    ap.add_argument("--static-weight", type=float, default=2.0, help="penalty on T4/T5 rate under a static grating")
    ap.add_argument("--sym-weight", type=float, default=1.0, help="left/right response symmetry of T4/T5 under gratings")
    ap.add_argument("--hs", action="store_true",
                    help="stage 3: fit the HS cells' inputs, tau and bias so each side is bidirectional "
                         "(up for progressive, below rest for regressive motion); L-R then reads rotation")
    ap.add_argument("--hs-weight", type=float, default=1.0)
    ap.add_argument("--hs-tau-max", type=float, default=0.05, help="upper bound on the HS membrane time constant (s)")
    ap.add_argument("--scene", default=None,
                    help="HS stage stimuli from eye input recorded in the browser (app: fly.record) instead of the ray-cast world")
    args = ap.parse_args()
    dev = args.device

    extra = LC_TYPES + (HS_TYPES if args.hs else [])
    g, fv, model = load_model(args.graph, args.params, dev, args.max_el, extra_types=extra, dt=args.dt, T=args.T)
    print(f"model: {g.n} units, {len(g.pre)} edges on {dev}")
    if args.hs and not args.joint:
        # stage 3 alone: only the HS cells' inputs, tau and bias move
        masks = model.grad_masks(HS_TYPES, train_extra_pairs_only=True)
    else:
        masks = model.grad_masks(None if args.joint else LC_TYPES, train_extra_pairs_only=not args.joint)
    init = {k: v.detach().clone() for k, v in model.named_parameters()}
    n_extra = len(model.pairs) - model.n_fv_pairs
    print(f"trainable: {n_extra} new input pairs onto {extra}, plus their tau/bias")

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
        side_mean = {}
        for (st, side), pos in ds_pos.items():
            r = resp[:, pos].mean(1)
            cosv = torch.cos(theta - PD[st][side])
            rc, cc = r - r.mean(), cosv - cosv.mean()
            corr = (rc * cc).sum() / (rc.norm() * cc.norm() + 1e-6)
            depth = r.std() / (r.mean() + 1e-3)
            total = total + (1 - corr) + 0.5 * torch.relu(0.5 - depth) - 0.2 * torch.log(r.mean() + 1e-3).clamp(max=0)
            side_mean.setdefault(st, {})[side] = r.mean()
        # the two eyes should respond alike (directions are uniform over the batch, PDs mirrored);
        # otherwise the HS left-right readout drifts during straight flight
        for st, m in side_mean.items():
            if "L" in m and "R" in m:
                total = total + args.sym_weight * (m["L"] - m["R"]).abs() / (m["L"] + m["R"] + 1e-3)
        return total

    hs_groups = {}
    if args.hs:
        for side, sidx in [("L", 0), ("R", 1)]:
            u = np.where(np.isin(tn, HS_TYPES) & (g.unit_side == sidx))[0]
            hs_groups[side] = torch.tensor(u, device=dev)
        print(f"HS readout: L {len(hs_groups['L'])} cells, R {len(hs_groups['R'])} cells")
    hs_rec = torch.tensor(np.concatenate([v.cpu().numpy() for v in hs_groups.values()]), device=dev) if hs_groups else None
    scene = None
    if args.hs and args.scene:
        from scene_episodes import SceneEpisodes
        scene = SceneEpisodes(args.scene, dev, dt, fv)
        print(f"scene episodes: {len(scene.kinds)} ({ {k: len(v) for k, v in scene.by_kind.items()} })")
    # record T4/T5 too, so the static control can penalise motion detectors that respond to static contrast
    all_rec = torch.cat([rec, ds_rec]) if ds_rec is not None else rec
    ds_off = len(rec)
    kind_idx = {k: [b for b, kk in enumerate(KINDS) if kk == k] for k in set(KINDS)}
    t0 = time.time()
    for step in range(args.steps):
        do_loom = not args.hs or args.joint      # the HS-only stage skips the (frozen) looming batch
        loss_static = torch.zeros((), device=dev)
        loss_sel = torch.zeros((), device=dev)
        loss_rate = torch.zeros((), device=dev)
        report = {}
        if do_loom:
            ext = stim.kinds(KINDS)
            rates = model(ext, dt, record=all_rec.tolist())        # (T, B, nrec + nds)
            resp = rates[-int(T * WIN):].mean(0)                    # (B, nrec + nds)
            loss_static = torch.zeros((), device=dev)
            if ds_rec is not None:
                # T4/T5 population under the static grating: mean rate above 0.15 is penalised
                for (st, side), pos in ds_pos.items():
                    rs = resp[kind_idx["static"]][:, ds_off + pos].mean()
                    loss_static = loss_static + torch.relu(rs - 0.15) * args.static_weight
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
        loss_hs = torch.zeros((), device=dev)
        if args.hs and hs_rec is not None:
            fext = stim.ext_from_photoreceptor(scene.batch(FLOW_KINDS, T)) if scene else stim.flow(FLOW_KINDS)
            frates = model(fext, dt, record=hs_rec.tolist())              # (T, Bf, nhs)
            fresp = frates[-int(T * WIN):].mean(0)                          # (Bf, nhs)
            nl = len(hs_groups["L"])
            mL = fresp[:, :nl].mean(1)                                      # per stimulus
            mR = fresp[:, nl:].mean(1)
            k = {name: i for i, name in enumerate(FLOW_KINDS)}
            restL, restR = mL[k["static"]], mR[k["static"]]
            dL = mL - restL
            dR = mR - restR
            m = 1.0
            # each side: the two rotations move it in opposite directions, one of them below rest
            rotL = dL[k["rotCCW"]] - dL[k["rotCW"]]
            rotR = dR[k["rotCCW"]] - dR[k["rotCW"]]
            loss_hs = (torch.relu(m - rotL.abs()) + torch.relu(m - rotR.abs())
                       + torch.relu(rotL * rotR) * 2                          # opposite signs across eyes
                       + torch.relu(0.25 * m + torch.min(dL[k["rotCCW"]], dL[k["rotCW"]]))
                       + torch.relu(0.25 * m + torch.min(dR[k["rotCCW"]], dR[k["rotCW"]]))
                       + torch.relu(0.8 - restL) + torch.relu(0.8 - restR)    # room to go down
                       + torch.relu(-dL[k["transFwd"]]) + torch.relu(-dR[k["transFwd"]])  # translation: not below rest
                       ) * args.hs_weight
            # fast membranes (real HS cells respond within tens of ms) and matched eyes
            hs_t = torch.tensor([i for i, t in enumerate(g.types) if t in HS_TYPES], device=dev)
            tau_hs = torch.exp(model.log_tau)[hs_t]
            loss_hs = loss_hs + (torch.relu(tau_hs - args.hs_tau_max).sum() * 20
                                 + (rotL.abs() - rotR.abs()).abs() + (restL - restR).abs()) * args.hs_weight
            rot = (rotL - rotR).abs() / 2
            trans = (dL - dR)[[k["transFwd"], k["transAsym"]]]
            report["HS rot/trans"] = (rot.item(), trans.abs().max().item())
            report["HS restL/R"] = (restL.item(), restR.item())
            report["HS dL ccw/cw"] = (dL[k["rotCCW"]].item(), dL[k["rotCW"]].item())
        loss = loss_sel + loss_rate + loss_reg + loss_ds + loss_static + loss_hs
        opt.zero_grad()
        loss.backward()
        for k, p in model.named_parameters():
            if p.grad is not None:
                p.grad *= masks[k]
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 10 == 0 or step == args.steps - 1:
            rep = " ".join(f"{k} {a:.2f}/{b:.2f}" for k, (a, b) in report.items())
            print(f"step {step:5d} loss {loss.item():.4f} sel {float(loss_sel):.4f} ds {loss_ds.item():.3f} static {loss_static.item():.3f} hs {loss_hs.item():.3f} | loom/worst: {rep}  {time.time() - t0:.0f}s", flush=True)

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
        rest_v = {st: float(x[0, torch.tensor(np.where(tn == st)[0], device=dev)].mean()) for st in model.extra_post_types
                  if (tn == st).any()}
    print("rest membrane under grey:", rest_v)
    model.export(Path(args.out), fv, rest_v=rest_v, source="fitted on MaleCNS optic-v2 (train_optic.py + train_loom.py)")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
