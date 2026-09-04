"""Diagnostic: per pair type, how much presynaptic drive (count x rate) reaches
LC4 / LPLC2 under each looming-trainer stimulus, with the current (frozen)
optic-lobe parameters. Says whether any weighting of the inputs could separate
loom from the controls before we spend GPU time fitting."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import torch
from graph_torch import Graph, FlyvisModel
import train_loom as tl

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--graph", default=str(tl.HERE.parent / "app" / "public" / "graphs" / "optic-v2"))
ap.add_argument("--params", default=str(tl.HERE / "out" / "fitted-params.json"))
ap.add_argument("--T", type=int, default=200)
ap.add_argument("--lv", type=float, nargs=2, default=[0.1, 0.4])
ap.add_argument("--win", type=float, default=0.4)
ap.add_argument("--pre", default="T4a,T4b,T4c,T4d,T5a,T5b,T5c,T5d,T2,T3,Tm3,Tm4,TmY3,Tm5Y,TmY5a,LPi43,LPLC2,LC4", help="pre types to print")
ap.add_argument("--dt", type=float, default=0.005)
ap.add_argument("--reps", type=int, default=3)
args = ap.parse_args()
dev = args.device
g0 = Graph(Path(args.graph))
fv = json.load(open(args.params))
tn0 = np.array([g0.types[t] for t in g0.unit_type])
keep = (g0.unit_role == g0.roles.index("optic")) | np.isin(tn0, tl.LC_TYPES)
g = g0.subgraph(keep)
model = FlyvisModel(g, fv, device=dev, extra_post_types=tl.LC_TYPES, extra_scale=0.001)
tn = np.array([g.types[t] for t in g.unit_type])
lam_w = fv["photoreceptor"]["laminaInput"]
lam_units = np.where(np.isin(tn, list(lam_w.keys())) & (g.unit_col >= 0))[0]
lam_cols = torch.tensor(g.unit_col[lam_units], device=dev)
lam_wt = torch.tensor([lam_w[tn[i]] for i in lam_units], device=dev, dtype=torch.float32)
lam_units_t = torch.tensor(lam_units, device=dev)
col_az = torch.tensor(g.col_az, device=dev)
col_el = torch.tensor(g.col_el, device=dev)
pr = fv["photoreceptor"]
T, dt, B = args.T, args.dt, len(tl.KINDS)
t_axis = torch.arange(T, device=dev, dtype=torch.float32) * dt
dur = T * dt

# reuse the trainer's stimulus code by binding its closure variables
ns = dict(T=T, B=B, g=g, dev=dev, col_az=col_az, col_el=col_el, t_axis=t_axis, dur=dur, pr=pr,
          lam_units_t=lam_units_t, lam_cols=lam_cols, lam_wt=lam_wt, DEG=tl.DEG, KINDS=tl.KINDS,
          angular_distance=tl.angular_distance, torch=torch, math=math,
          LV_MIN=args.lv[0], LV_MAX=args.lv[1])
src = open(tl.__file__).read()
start = src.index("    def disc_centre")
end = src.index("    kind_idx =")
body = "\n".join(line[4:] for line in src[start:end].splitlines())
exec(body, ns)
make_ext = ns["make_ext"]

lc_units = {st: np.where(tn == st)[0] for st in tl.LC_TYPES}
side = g.unit_side
pre_t = tn[g.pre]
post_t = tn[g.post]
with torch.no_grad():
    acc = {}
    for rep in range(args.reps):
        ext = make_ext()
        rates = model(ext, dt)                       # (T, B, n)
        r = rates[-int(T * args.win):].mean(0)       # (B, n)
        for st in tl.LC_TYPES:
            em = np.where(post_t == st)[0]
            pre_types = sorted(set(pre_t[em]))
            for pt in pre_types:
                e = em[pre_t[em] == pt]
                cnt = torch.tensor(g.count[e], device=dev)
                contrib = torch.zeros(B, g.n, device=dev).index_add_(1, torch.tensor(g.post[e], device=dev), r[:, torch.tensor(g.pre[e], device=dev)] * cnt)
                for sd, sname in [(0, "L"), (1, "R")]:
                    u = torch.tensor([i for i in lc_units[st] if side[i] == sd], device=dev)
                    v = contrib[:, u].mean(1).cpu().numpy()      # (B,)
                    acc.setdefault((st, sname, pt), []).append(v)
    print(f"{'post':7s} {'pre':8s} " + " ".join(f"{k:>8s}" for k in tl.KINDS))
    for (st, sname, pt), vs in sorted(acc.items()):
        v = np.mean(vs, 0)
        if pt not in args.pre.split(","):
            continue
        print(f"{st + sname:7s} {pt:8s} " + " ".join(f"{x:8.1f}" for x in v))
