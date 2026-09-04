"""flyvis reference: central-column responses to a horizontally drifting sine grating."""
import json, numpy as np, torch, h5py
from flyvis import NetworkView
net = NetworkView("flow/0000/000").init_network()
ct = net.connectome
node_type = np.array([x.decode() for x in h5py.File(ct.path/"nodes"/"type.h5")["data"][:]])
u = h5py.File(ct.path/"nodes"/"u.h5")["data"][:]; v = h5py.File(ct.path/"nodes"/"v.h5")["data"][:]
central = (u == 0) & (v == 0)
# input hexals: the stimulus layout uses the receptor lattice; get its coordinates via the Stimulus object
stim = net.stimulus
n_hex = stim.n_hexals if hasattr(stim, "n_hexals") else None
r1 = node_type == "R1"
uu, vv = u[r1], v[r1]
print("R1 cells:", r1.sum(), "n_hexals attr:", n_hex)
x = uu + 0.5 * vv   # axial hex -> cartesian x (columns ~ 5.8 deg apart)
dt = 1/100
T = 200
def grating(direction, period_cols=4.0, tf=1.0):
    frames = np.zeros((T, len(uu)), np.float32)
    for t in range(T):
        phase = 2*np.pi*(x/period_cols - direction*tf*t*dt)
        frames[t] = 0.5 + 0.5*np.sin(phase)
    return torch.tensor(frames)[None, :, None, :]  # (1, T, 1, hexals)
with torch.no_grad():
    ss = net.steady_state(t_pre=1.0, dt=dt, batch_size=1, value=0.5)
    res = {}
    for name, d in [("right(+x)", +1), ("left(-x)", -1)]:
        r = net.simulate(grating(d), dt=dt, initial_state=ss).squeeze().numpy()  # (T, cells)
        res[name] = r
out = {}
for t in ["L1","L2","L3","Mi1","Tm3","Mi4","Mi9","Tm1","Tm2","Tm9","T4a","T4b","T4c","T4d","T5a","T5b","T5c","T5d"]:
    sel = (node_type == t) & central
    row = {}
    for name, r in res.items():
        a = np.maximum(0, r[50:, sel]).mean(axis=1)
        row[name] = (float(a.mean()), float(a.max()))
    out[t] = row
    print(f"{t:5s} " + "  ".join(f"{k}: mean={m:.3f} peak={p:.3f}" for k,(m,p) in row.items()))
json.dump(out, open("out/flyvis-grating.json","w"), indent=1)
