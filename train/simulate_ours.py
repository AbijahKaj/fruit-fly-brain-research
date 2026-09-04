"""Steady state of our MaleCNS optic graph with flyvis parameters under uniform grey,
per type, next to the flyvis reference. Diagnoses where the two regimes diverge."""
import json, numpy as np, scipy.sparse as sp
hdr = json.load(open("../data/out/optic-v2.json")); buf = open("../data/out/optic-v2.bin","rb").read()
DT = {"int8":np.int8,"int16":np.int16,"int32":np.int32,"float32":np.float32,"float64":np.float64}
def arr(n):
    d = hdr["arrays"][n]; return np.frombuffer(buf, dtype=DT[d["dtype"]], count=d["length"], offset=d["offset"])
types = [t["name"] for t in hdr["types"]]; tau_g = np.array([t["tau"] for t in hdr["types"]])
ut = arr("units.type"); sign = arr("units.sign").astype(np.float32); pre = arr("edges.pre"); post = arr("edges.post"); cnt = arr("edges.weight")
n = hdr["units"]["count"]
fv = json.load(open("out/flyvis-params.json")); ref = json.load(open("out/flyvis-reference.json"))["rest"]
alias = {"TmY9a":"TmY9","TmY9b":"TmY9"}
fvn = lambda t: alias.get(t,t)
pairs = {(p["pre"],p["post"]):(p["strength"],p["sign"]) for p in fv["pairs"]}
tname = np.array(types)
tp = tname[ut[pre]]; tq = tname[ut[post]]
w = np.empty(len(pre), np.float32); covered = np.zeros(len(pre), bool)
for e in range(len(pre)):
    k = (fvn(tp[e]), fvn(tq[e]))
    if k in pairs: s_, sg = pairs[k]; w[e] = cnt[e]*sg*s_; covered[e] = True
    else: w[e] = cnt[e]*sign[pre[e]]*0.02
W = sp.csr_matrix((w, (post, pre)), shape=(n, n))
tau = np.array([fv["types"][fvn(t)]["tau"] if fvn(t) in fv["types"] else tau_g[i] for i,t in enumerate(types)])[ut]
bias = np.array([fv["types"][fvn(t)]["bias"] if fvn(t) in fv["types"] else 0.0 for t in types])[ut]
# photoreceptor drive at grey
pr = fv["photoreceptor"]; rR = pr["restOffset"] + pr["stimGain"]*0.5
ext = np.zeros(n, np.float32)
for L, wl in pr["laminaInput"].items():
    ext[tname[ut] == L] = wl * rR
x = bias.copy().astype(np.float32); dt = 0.004
RMAX = 50.0
for step in range(int(2.0/dt)):
    r = np.minimum(RMAX, np.maximum(0, x))
    x += (dt/np.maximum(tau, dt)) * (-x + bias + W @ r + ext)
    x = np.clip(x, -1e3, 1e3)
r = np.minimum(RMAX, np.maximum(0, x))
print("units at ceiling:", int((r >= RMAX).sum()), " types at ceiling:", sorted(set(tname[ut][r >= RMAX]))[:30])
print(f"{'type':7s} {'ours':>8s} {'flyvis':>8s}   n")
for t in ["L1","L2","L3","L4","L5","Lawf1","Lawf2","C2","C3","Mi1","Tm3","Mi4","Mi9","Mi2","Mi10","Mi13","Mi14","Mi15","T1","T2","T2a","T3","Tm1","Tm2","Tm4","Tm9","Tm5Y","Tm5a","Tm16","Tm20","Tm30","TmY3","TmY5a","TmY18","T4a","T4b","T4c","T4d","T5a","T5b","T5c","T5d"]:
    sel = tname[ut] == t
    if sel.sum(): print(f"{t:7s} {r[sel].mean():8.3f} {ref.get(fvn(t), float('nan')):8.3f}   {sel.sum()}")
# L1 input decomposition
sel = np.where(tname[ut]=="L1")[0][:200]
rows = {}
for i in sel:
    a, b = W.indptr[i], W.indptr[i+1]
    for j, ww in zip(W.indices[a:b], W.data[a:b]):
        rows[tname[ut[j]]] = rows.get(tname[ut[j]], 0) + ww*r[j]
print("\nL1 mean input by pre type (ours):")
for k, v in sorted(rows.items(), key=lambda kv: -abs(kv[1]))[:12]: print(f"  {k:8s} {v/len(sel):+.3f}")
print(f"  ext(R)   {ext[sel].mean():+.3f}   bias {bias[sel].mean():+.3f}")
