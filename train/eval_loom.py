"""Looming selectivity of LC4 / LPLC2 with a parameter file: top-k response per
stimulus kind (mean over reps), and a selectivity index per (type, side).

  python eval_loom.py --params out/fitted-params.json --device cuda
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
from train_loom import HERE, LC_TYPES, KINDS, WIN, Stimuli, load_model, topk_rate
from train_optic import PD

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--graph", default=str(HERE.parent / "app" / "public" / "graphs" / "optic-v2"))
ap.add_argument("--params", default=str(HERE / "out" / "fitted-params.json"))
ap.add_argument("--T", type=int, default=200)
ap.add_argument("--dt", type=float, default=0.005)
ap.add_argument("--reps", type=int, default=4)
ap.add_argument("--topk", type=int, default=5)
ap.add_argument("--max-el", type=float, default=90)
args = ap.parse_args()
dev = args.device
g, fv, model = load_model(args.graph, args.params, dev, args.max_el)
stim = Stimuli(g, fv, dev, args.T, args.dt)
tn = stim.tn
groups = {}
for st in LC_TYPES:
    for side, sidx in [("L", 0), ("R", 1)]:
        u = np.where((tn == st) & (g.unit_side == sidx))[0]
        if len(u):
            groups[(st, side)] = torch.tensor(u, device=dev)
# T4/T5 population means per kind too (they must be quiet under the static grating)
ds = {}
for st in PD:
    u = np.where(tn == st)[0]
    if len(u):
        ds[st] = torch.tensor(u, device=dev)
rec = torch.tensor(np.concatenate([v.cpu().numpy() for v in list(groups.values()) + list(ds.values())]), device=dev)
pos, off = {}, 0
for k, v in list(groups.items()) + list(ds.items()):
    pos[k] = torch.arange(off, off + len(v), device=dev)
    off += len(v)
kinds = sorted(set(KINDS))
acc = {k: [] for k in list(groups) + list(ds)}
with torch.no_grad():
    for rep in range(args.reps):
        ext, score = stim.kinds_scored(kinds)
        rates = model(ext, args.dt, record=rec.tolist())
        resp = (rates * score[:, :, None]).sum(0) / score.sum(0)[:, None]
        for k, p in pos.items():
            acc[k].append((topk_rate(resp, p, args.topk) if k in groups else resp[:, p].mean(1)).cpu().numpy())
print(f"{'group':8s} " + " ".join(f"{k:>8s}" for k in kinds) + "   sel")
for k in groups:
    v = np.mean(acc[k], 0)
    ipsi = v[kinds.index(f"loom{k[1]}")]
    worst = max(x for i, x in enumerate(v) if kinds[i] != f"loom{k[1]}")
    sel = (ipsi - worst) / (ipsi + worst + 1e-3)
    print(f"{k[0] + k[1]:8s} " + " ".join(f"{x:8.2f}" for x in v) + f"  {sel:+.2f}")
print("T4/T5 population mean per kind:")
for k in ds:
    v = np.mean(acc[k], 0)
    print(f"{k:8s} " + " ".join(f"{x:8.2f}" for x in v))
