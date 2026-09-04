"""Reference activities from the flyvis network: steady state under grey, and
responses to a horizontal drifting grating in both directions, per cell type
(central column). Used to calibrate our MaleCNS-based model."""
import json, numpy as np, torch
from flyvis import NetworkView
from flyvis.datasets.moving_bar import MovingBar

net = NetworkView("flow/0000/000").init_network()
dt = 1 / 100
ct = net.connectome
types = [t.decode() for t in ct.unique_cell_types[:]] if hasattr(ct, "unique_cell_types") else None
import h5py
with h5py.File(ct.path / "unique_cell_types.h5") as f: types = [x.decode() for x in f["data"][:]]
node_type = np.array([x.decode() for x in h5py.File(ct.path/"nodes"/"type.h5")["data"][:]])
u = h5py.File(ct.path/"nodes"/"u.h5")["data"][:]; v = h5py.File(ct.path/"nodes"/"v.h5")["data"][:]
central = (u == 0) & (v == 0)

with torch.no_grad():
    ss = net.steady_state(t_pre=1.0, dt=dt, batch_size=1, value=0.5)
    act = ss.nodes.activity.squeeze().numpy()
rest = {t: float(np.mean(np.maximum(0, act[(node_type == t) & central]))) for t in types}
print("REST (grey 0.5, central column):")
for t in ["L1","L2","L3","L5","Mi1","Tm3","Mi4","Mi9","Tm1","Tm2","Tm9","C2","C3","T4a","T4b","T4c","T5a","T5b","Lawf2","Mi15","TmY18"]:
    print(f"  {t:6s} {rest[t]:.3f}")

# moving bars/gratings: MovingBar dataset with angles 0 and 180 (rightward/leftward), speed 19 deg/s?
ds = MovingBar(widths=[4], offsets=(-10, 11), intensities=[1], speeds=[13], height=9, dt=dt, device="cpu", angles=[0, 180], post_pad_mode="value", t_pre=1.0, t_post=0.5)
print("n stimuli:", len(ds))
resp = {}
with torch.no_grad():
    for i in range(len(ds)):
        x = ds[i]  # (n_frames, n_hexals)
        r = net.simulate(x[None], dt=dt, initial_state=ss).squeeze().numpy()  # (T, n_cells)
        resp[i] = r
    ang = [ds.arg_df.iloc[i].angle for i in range(len(ds))]
print("angles:", ang)
out = {"rest": rest, "grating": {}}
for t in ["T4a","T4b","T4c","T4d","T5a","T5b","T5c","T5d","Mi1","Mi4","Mi9","Tm3","L1","Tm1","Tm9"]:
    sel = (node_type == t) & central
    row = {}
    for i, a in enumerate(ang):
        r = np.maximum(0, resp[i][:, sel]).mean(axis=1)
        row[str(a)] = {"peak": float(r.max()), "mean": float(r.mean())}
    out["grating"][t] = row
    print(f"  {t:5s} " + "  ".join(f"ang{a}: peak={row[str(a)]['peak']:.3f} mean={row[str(a)]['mean']:.3f}" for a in ang))
json.dump(out, open("out/flyvis-reference.json", "w"), indent=1)
