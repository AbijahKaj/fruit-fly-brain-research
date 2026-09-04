"""Direction selectivity of the model per T4/T5 subtype and eye, with given params.

  python eval_ds.py --device cuda --params runs/fit2-params.json
Reports DSI = (r_PD - r_ND) / (r_PD + r_ND) and the correlation of the tuning
curve with cos(theta - PD) over 8 grating directions.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import torch
from graph_torch import Graph, FlyvisModel
from train_optic import PD, DEG

HERE = Path(__file__).parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--params", default=str(HERE / "out" / "fitted-params.json"))
    ap.add_argument("--graph", default=str(HERE.parent / "app" / "public" / "graphs" / "optic-v2"))
    ap.add_argument("--T", type=int, default=120)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--max-el", type=float, default=90)
    ap.add_argument("--wavelength", type=float, default=25)
    ap.add_argument("--tf", type=float, default=1.5)
    args = ap.parse_args()
    dev = args.device
    g0 = Graph(Path(args.graph))
    fv = json.load(open(args.params))
    keep = g0.unit_role == g0.roles.index("optic")
    if args.max_el < 90:
        ok = np.abs(g0.col_el) <= args.max_el * DEG
        keep &= (g0.unit_col >= 0) & ok[np.clip(g0.unit_col, 0, None)]
    g = g0.subgraph(keep)
    model = FlyvisModel(g, fv, device=dev)
    tn = np.array([g.types[t] for t in g.unit_type])
    lam_w = fv["photoreceptor"]["laminaInput"]
    lam_units = np.where(np.isin(tn, list(lam_w.keys())) & (g.unit_col >= 0))[0]
    lam_cols = torch.tensor(g.unit_col[lam_units], device=dev)
    lam_wt = torch.tensor([lam_w[tn[i]] for i in lam_units], device=dev, dtype=torch.float32)
    lam_units_t = torch.tensor(lam_units, device=dev)
    col_az = torch.tensor(g.col_az, device=dev); col_el = torch.tensor(g.col_el, device=dev)
    pr = fv["photoreceptor"]
    dirs = torch.arange(8, device=dev, dtype=torch.float32) * (2 * math.pi / 8)
    B, T = len(dirs), args.T
    t_axis = torch.arange(T, device=dev, dtype=torch.float32) * args.dt
    proj = col_az[None] * torch.cos(dirs)[:, None] + col_el[None] * torch.sin(dirs)[:, None]
    phase = 2 * math.pi * (proj[None] / (args.wavelength * DEG) - args.tf * t_axis[:, None, None])
    lum = 0.5 + 0.5 * torch.sin(phase)
    rR = pr["restOffset"] + pr["stimGain"] * lum
    ext = torch.zeros(T, B, g.n, device=dev)
    ext[:, :, lam_units_t] = rR[:, :, lam_cols] * lam_wt[None, None, :]
    groups = {}
    for st in PD:
        for side, sidx in [("L", 0), ("R", 1)]:
            u = np.where((tn == st) & (g.unit_side == sidx))[0]
            if len(u): groups[(st, side)] = torch.tensor(u, device=dev)
    rec = torch.cat(list(groups.values()))
    with torch.no_grad():
        rates = model(ext, args.dt, record=rec.tolist())
    resp = rates[T // 3:].mean(0).cpu().numpy()  # (B, nrec)
    off = 0
    print(f"{'group':8s} {'DSI':>6s} {'corr':>6s} {'r_PD':>6s} {'r_ND':>6s}   tuning over 8 dirs (0=+az ... )")
    rows = {}
    for (st, side), u in groups.items():
        r = resp[:, off:off + len(u)].mean(1); off += len(u)
        pd_ = PD[st][side]
        cosv = np.cos(dirs.cpu().numpy() - pd_)
        ipd = int(np.argmax(cosv)); ind = int(np.argmin(cosv))
        dsi = (r[ipd] - r[ind]) / (r[ipd] + r[ind] + 1e-6)
        corr = float(np.corrcoef(r, cosv)[0, 1]) if r.std() > 1e-6 else 0.0
        rows[f"{st}{side}"] = {"dsi": float(dsi), "corr": corr, "rPD": float(r[ipd]), "rND": float(r[ind])}
        print(f"{st}{side:3s} {dsi:+6.2f} {corr:+6.2f} {r[ipd]:6.3f} {r[ind]:6.3f}   " + " ".join(f"{v:.2f}" for v in r))
    print("mean DSI:", round(float(np.mean([v['dsi'] for v in rows.values()])), 3), " mean corr:", round(float(np.mean([v['corr'] for v in rows.values()])), 3))


if __name__ == "__main__":
    main()
