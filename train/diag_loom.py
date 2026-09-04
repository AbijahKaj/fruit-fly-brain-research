"""Diagnostic: per pair type, how much presynaptic drive (count x rate) reaches
LC4 / LPLC2 under each looming-trainer stimulus. Says whether any weighting of
the inputs could separate loom from the controls before spending GPU time."""
from __future__ import annotations
import argparse
import numpy as np
import torch
from train_loom import HERE, LC_TYPES, KINDS, Stimuli, load_model

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--graph", default=str(HERE.parent / "app" / "public" / "graphs" / "optic-v2"))
ap.add_argument("--params", default=str(HERE / "out" / "fitted-params.json"))
ap.add_argument("--T", type=int, default=200)
ap.add_argument("--lv", type=float, nargs=2, default=[0.1, 0.4])
ap.add_argument("--win", type=float, default=0.4)
ap.add_argument("--pre", default="T4a,T4b,T4c,T4d,T5a,T5b,T5c,T5d,T2,T3,Tm3,Tm4,TmY3,Tm5Y,TmY5a,LPi43,LPLC2,LC4", help="pre types to print")
ap.add_argument("--dt", type=float, default=0.005)
ap.add_argument("--reps", type=int, default=3)
args = ap.parse_args()
dev = args.device
g, fv, model = load_model(args.graph, args.params, dev)
stim = Stimuli(g, fv, dev, args.T, args.dt, lv=tuple(args.lv))
tn = stim.tn
lc_units = {st: np.where(tn == st)[0] for st in LC_TYPES}
side = g.unit_side
pre_t, post_t = tn[g.pre], tn[g.post]
B = len(KINDS)
with torch.no_grad():
    acc = {}
    for rep in range(args.reps):
        rates = model(stim.kinds(KINDS), args.dt)
        r = rates[-int(args.T * args.win):].mean(0)
        for st in LC_TYPES:
            em = np.where(post_t == st)[0]
            for pt in sorted(set(pre_t[em])):
                if pt not in args.pre.split(","):
                    continue
                e = em[pre_t[em] == pt]
                cnt = torch.tensor(g.count[e], device=dev)
                contrib = torch.zeros(B, g.n, device=dev).index_add_(
                    1, torch.tensor(g.post[e], device=dev), r[:, torch.tensor(g.pre[e], device=dev)] * cnt)
                for sd, sname in [(0, "L"), (1, "R")]:
                    u = torch.tensor([i for i in lc_units[st] if side[i] == sd], device=dev)
                    acc.setdefault((st, sname, pt), []).append(contrib[:, u].mean(1).cpu().numpy())
    print(f"{'post':7s} {'pre':8s} " + " ".join(f"{k:>8s}" for k in KINDS))
    for (st, sname, pt), vs in sorted(acc.items()):
        print(f"{st + sname:7s} {pt:8s} " + " ".join(f"{x:8.1f}" for x in np.mean(vs, 0)))
