"""
Fit the MaleCNS optic-lobe model so T4/T5 subtypes are direction selective.

  python train_optic.py --device cuda --steps 2000            # on the 5090 box
  python train_optic.py --device cpu --steps 3 --max-el 20    # smoke test on the Mac

Model: train/graph_torch.py (same graph the browser runs, flyvis init).
Stimulus: drifting sine gratings on the real column lattice (az/el from the
connectome-calibrated retinotopy), random direction, wavelength, temporal
frequency, contrast; one virtual photoreceptor per column drives L1/L2/L3.
Loss: for each T4/T5 subtype and side, the mean response across grating
directions should follow 1 + cos(theta - PD); plus rate bounds and an L2
pull toward the flyvis init. Trainable: per-type tau and bias, per-pair
synapse strength (signs fixed). Output: out/fitted-params.json, drop into
app/public/graphs/ and point the browser at it.
"""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import numpy as np
import torch
from graph_torch import Graph, FlyvisModel, fv_name

HERE = Path(__file__).parent
DEG = math.pi / 180
# Preferred directions in the fly's visual field, per side: az positive = right.
# Front-to-back on the right eye is +az, on the left eye -az.
PD = {
    "T4a": {"R": 0.0, "L": math.pi}, "T5a": {"R": 0.0, "L": math.pi},
    "T4b": {"R": math.pi, "L": 0.0}, "T5b": {"R": math.pi, "L": 0.0},
    "T4c": {"R": math.pi / 2, "L": math.pi / 2}, "T5c": {"R": math.pi / 2, "L": math.pi / 2},
    "T4d": {"R": -math.pi / 2, "L": -math.pi / 2}, "T5d": {"R": -math.pi / 2, "L": -math.pi / 2},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=8, help="gratings per step")
    ap.add_argument("--T", type=int, default=120, help="time steps per grating")
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--max-el", type=float, default=90, help="keep columns with |el| <= this (deg) to shrink the model")
    ap.add_argument("--graph", default=str(HERE.parent / "data" / "out" / "optic-v2"))
    ap.add_argument("--params", default=str(HERE / "out" / "flyvis-params.json"))
    ap.add_argument("--out", default=str(HERE / "out" / "fitted-params.json"))
    args = ap.parse_args()
    dev = args.device

    g0 = Graph(Path(args.graph))
    fv = json.load(open(args.params))
    # optic-lobe units only, optionally a band of columns
    keep = g0.unit_role == g0.roles.index("optic")
    if args.max_el < 90:
        col_ok = np.abs(g0.col_el) <= args.max_el * DEG
        keep &= (g0.unit_col >= 0) & col_ok[np.clip(g0.unit_col, 0, None)]
    g = g0.subgraph(keep)
    print(f"model: {g.n} units, {len(g.pre)} edges on {dev}")
    model = FlyvisModel(g, fv, device=dev)
    init = {k: v.detach().clone() for k, v in model.named_parameters()}

    # lamina injection: unit -> (column, weight)
    lam_w = fv["photoreceptor"]["laminaInput"]
    tn = np.array([g.types[t] for t in g.unit_type])
    lam_units = np.where(np.isin(tn, list(lam_w.keys())) & (g.unit_col >= 0))[0]
    lam_cols = torch.tensor(g.unit_col[lam_units], device=dev)
    lam_wt = torch.tensor([lam_w[tn[i]] for i in lam_units], device=dev, dtype=torch.float32)
    lam_units_t = torch.tensor(lam_units, device=dev)
    col_az = torch.tensor(g.col_az, device=dev)
    col_el = torch.tensor(g.col_el, device=dev)
    pr = fv["photoreceptor"]

    # readout groups: (subtype, side) -> unit indices
    groups = {}
    for st in PD:
        for side, sidx in [("L", 0), ("R", 1)]:
            u = np.where((tn == st) & (g.unit_side == sidx))[0]
            if len(u):
                groups[(st, side)] = torch.tensor(u, device=dev)
    rec = torch.tensor(np.concatenate([v.cpu().numpy() for v in groups.values()]), device=dev)
    rec_pos = {}
    off = 0
    for k, v in groups.items():
        rec_pos[k] = torch.arange(off, off + len(v), device=dev)
        off += len(v)
    print(f"readout: {len(groups)} groups, {len(rec)} units; lamina inputs {len(lam_units)}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    T, B, dt = args.T, args.batch, args.dt
    t_axis = torch.arange(T, device=dev, dtype=torch.float32) * dt

    def make_ext():
        """Random drifting gratings -> photoreceptor rate -> lamina ext. Returns ext (T,B,n), theta (B,)."""
        theta = torch.rand(B, device=dev) * 2 * math.pi
        lam = (15 + 25 * torch.rand(B, device=dev)) * DEG     # wavelength 15-40 deg
        tf = 0.5 + 2.5 * torch.rand(B, device=dev)           # 0.5-3 Hz
        c = 0.5 + 0.5 * torch.rand(B, device=dev)
        # project column directions onto the grating axis (small-angle planar approx on az/el)
        proj = col_az[None, :] * torch.cos(theta)[:, None] + col_el[None, :] * torch.sin(theta)[:, None]  # (B, C)
        phase = 2 * math.pi * (proj[None] / lam[None, :, None] - tf[None, :, None] * t_axis[:, None, None])  # (T,B,C)
        lum = 0.5 + 0.5 * c[None, :, None] * torch.sin(phase)
        stim = lum  # adapted: mean 0.5 already
        rR = pr["restOffset"] + pr["stimGain"] * stim                                   # (T,B,C)
        ext = torch.zeros(T, B, g.n, device=dev)
        ext[:, :, lam_units_t] = rR[:, :, lam_cols] * lam_wt[None, None, :]
        return ext, theta

    t0 = time.time()
    for step in range(args.steps):
        ext, theta = make_ext()
        rates = model(ext, dt, record=rec.tolist())          # (T, B, nrec)
        resp = rates[T // 3:].mean(0)                          # (B, nrec) steady part
        loss_ds = 0.0
        for (st, side), pos in rec_pos.items():
            r = resp[:, pos].mean(1)                           # (B,)
            target = 1 + torch.cos(theta - PD[st][side])       # (B,)
            rn = r / (r.mean() + 1e-3)
            loss_ds = loss_ds + ((rn - target) ** 2).mean() - 0.5 * torch.log(r.mean() + 1e-3).clamp(max=0)
        rate_all = rates.mean()
        loss_rate = torch.relu(rates - 3.0).pow(2).mean() * 10 + torch.relu(0.02 - rate_all) * 10
        loss_reg = sum(((p - init[k]) ** 2).mean() for k, p in model.named_parameters()) * 1.0
        loss = loss_ds + loss_rate + loss_reg
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 10 == 0 or step == args.steps - 1:
            # DS index on the batch for T4a right
            with torch.no_grad():
                k = ("T4a", "R") if ("T4a", "R") in rec_pos else next(iter(rec_pos))
                r = resp[:, rec_pos[k]].mean(1)
                pd_ = torch.cos(theta - PD[k[0]][k[1]])
                dsi = float((r * pd_).sum() / (r.abs().sum() + 1e-6))
            print(f"step {step:5d} loss {loss.item():.4f} ds {loss_ds.item():.4f} rate {rate_all.item():.3f} "
                  f"DSI[{k[0]}{k[1]}] {dsi:+.3f}  {time.time() - t0:.0f}s", flush=True)
    model.export(Path(args.out), fv)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
