"""Load the binary graph (data/graphio.py format) into numpy / torch and assemble
the flyvis-parameterised sparse weight matrix with trainable pieces."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch

DT = {"int8": np.int8, "int16": np.int16, "int32": np.int32, "float32": np.float32, "float64": np.float64}
ALIAS = {"TmY9a": "TmY9", "TmY9b": "TmY9"}


def fv_name(t: str) -> str:
    return ALIAS.get(t, t)


class Graph:
    def __init__(self, stem: Path):
        self.hdr = json.load(open(stem.with_suffix(".json")))
        buf = open(stem.with_suffix(".bin"), "rb").read()

        def arr(name):
            d = self.hdr["arrays"][name]
            return np.frombuffer(buf, dtype=DT[d["dtype"]], count=d["length"], offset=d["offset"]).copy()

        self.types = [t["name"] for t in self.hdr["types"]]
        self.unit_type = arr("units.type")
        self.unit_side = arr("units.side")
        self.unit_role = arr("units.role")
        self.unit_sign = arr("units.sign")
        self.unit_col = arr("units.col")
        self.pre = arr("edges.pre")
        self.post = arr("edges.post")
        self.count = arr("edges.weight")
        self.col_side = arr("columns.side")
        self.col_az = arr("columns.az")
        self.col_el = arr("columns.el")
        self.roles = self.hdr["roles"]
        self.n = self.hdr["units"]["count"]

    def type_name(self, i: int) -> str:
        return self.types[self.unit_type[i]]

    def subgraph(self, keep: np.ndarray) -> "Graph":
        """Restrict to units where keep[i] is True (re-indexes edges)."""
        idx = -np.ones(self.n, np.int64)
        idx[keep] = np.arange(keep.sum())
        g = Graph.__new__(Graph)
        g.hdr, g.types, g.roles = self.hdr, self.types, self.roles
        for a in ["unit_type", "unit_side", "unit_role", "unit_sign", "unit_col"]:
            setattr(g, a, getattr(self, a)[keep])
        em = keep[self.pre] & keep[self.post]
        g.pre, g.post, g.count = idx[self.pre[em]], idx[self.post[em]], self.count[em]
        g.col_side, g.col_az, g.col_el = self.col_side, self.col_az, self.col_el
        g.n = int(keep.sum())
        return g


class FlyvisModel(torch.nn.Module):
    """
    tau_i dx_i/dt = -x_i + bias_type(i) + sum_j w_ij relu(x_j) + ext_i
    w_ij = count_ij * sign_pair * strength_pair        (pairs flyvis knows)
         = count_ij * sign_nt(j) * default_scale         (otherwise, fixed)
    Trainable: log tau per type, bias per type, log strength per pair.
    """

    def __init__(self, g: Graph, fv: dict, default_scale: float = 0.02, device: str = "cpu", r_max: float = 10.0):
        super().__init__()
        self.g = g
        self.device = device
        # Soft rate ceiling: r = r_max * tanh(relu(x) / r_max). Keeps the untrained
        # network bounded (the browser uses a hard clamp) while gradients survive.
        self.r_max = r_max
        n = g.n
        tnames = g.types
        # per-type params, init from flyvis where known
        tau0 = np.array([fv["types"].get(fv_name(t), {}).get("tau", 0.03) for t in tnames], np.float32)
        bias0 = np.array([fv["types"].get(fv_name(t), {}).get("bias", 0.0) for t in tnames], np.float32)
        self.log_tau = torch.nn.Parameter(torch.log(torch.tensor(tau0)))
        self.bias = torch.nn.Parameter(torch.tensor(bias0))
        # per-pair strengths
        pair_key = {(p["pre"], p["post"]): k for k, p in enumerate(fv["pairs"])}
        strength0 = np.array([p["strength"] for p in fv["pairs"]], np.float32)
        pair_sign = np.array([p["sign"] for p in fv["pairs"]], np.float32)
        self.log_strength = torch.nn.Parameter(torch.log(torch.tensor(strength0) + 1e-4))
        self.register_buffer("pair_sign", torch.tensor(pair_sign))
        # edge -> pair index (or -1)
        tp = np.array([fv_name(tnames[t]) for t in g.unit_type[g.pre]])
        tq = np.array([fv_name(tnames[t]) for t in g.unit_type[g.post]])
        eidx = np.array([pair_key.get((a, b), -1) for a, b in zip(tp, tq)], np.int64)
        self.register_buffer("edge_pair", torch.tensor(eidx))
        self.register_buffer("edge_count", torch.tensor(g.count, dtype=torch.float32))
        fixed = torch.tensor(g.count * g.unit_sign[g.pre] * default_scale, dtype=torch.float32)
        fixed[torch.tensor(eidx >= 0)] = 0.0
        self.register_buffer("edge_fixed", fixed)
        self.register_buffer("indices", torch.tensor(np.stack([g.post, g.pre]).astype(np.int64)))
        self.register_buffer("unit_type_t", torch.tensor(g.unit_type.astype(np.int64)))
        self.n = n
        self.to(device)

    def weights(self) -> torch.Tensor:
        s = torch.exp(self.log_strength)
        ep = self.edge_pair
        known = ep >= 0
        w = self.edge_fixed.clone()
        w[known] = self.edge_count[known] * self.pair_sign[ep[known]] * s[ep[known]]
        return w

    def forward(self, ext: torch.Tensor, dt: float, x0: torch.Tensor | None = None, record: list[int] | None = None):
        """ext: (T, B, n). Returns rates (T, B, n) or only for `record` unit indices."""
        T, B, n = ext.shape
        w = self.weights()
        W = torch.sparse_coo_tensor(self.indices, w, (n, n)).coalesce()
        tau = torch.exp(self.log_tau)[self.unit_type_t].clamp_min(dt)
        bias = self.bias[self.unit_type_t]
        x = x0 if x0 is not None else bias.expand(B, n).clone()
        out = []
        rec = torch.tensor(record, device=ext.device) if record is not None else None
        for t in range(T):
            r = self.r_max * torch.tanh(torch.relu(x) / self.r_max)
            drive = torch.sparse.mm(W, r.T).T  # (B, n)
            x = x + (dt / tau) * (-x + bias + drive + ext[t])
            out.append(r[:, rec] if rec is not None else r)
        return torch.stack(out)

    def export(self, path: Path, fv: dict) -> None:
        """Write params in the flyvis-params.json format the browser loads."""
        out = json.loads(json.dumps(fv))
        tau = torch.exp(self.log_tau).detach().cpu().numpy()
        bias = self.bias.detach().cpu().numpy()
        for k, t in enumerate(self.g.types):
            key = fv_name(t)
            if key in out["types"]:
                out["types"][key]["tau"] = float(tau[k])
                out["types"][key]["bias"] = float(bias[k])
        s = torch.exp(self.log_strength).detach().cpu().numpy()
        for k, p in enumerate(out["pairs"]):
            p["strength"] = float(s[k])
        out["source"] = "fitted on MaleCNS optic-v2 (train/train_optic.py), init " + fv.get("source", "")
        path.write_text(json.dumps(out, indent=1))
