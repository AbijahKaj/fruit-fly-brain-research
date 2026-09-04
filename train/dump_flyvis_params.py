"""
Dump flyvis (Lappalainen et al. 2024) trained parameters for the types in
our optic-v2 graph: per-type time constant and bias, per (pre,post)-type
synapse strength and sign. Averages over the ensemble models given.

Output: out/flyvis-params.json, loaded by the browser as an override for the
graph's type table (tau, bias) and as a per-type-pair weight scale.
"""
from __future__ import annotations
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from flyvis import NetworkView

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
MODELS = [f"flow/0000/{i:03d}" for i in range(int(sys.argv[1]) if len(sys.argv) > 1 else 10)]

tau: dict[str, list[float]] = defaultdict(list)
bias: dict[str, list[float]] = defaultdict(list)
strength: dict[tuple[str, str], list[float]] = defaultdict(list)
sign: dict[tuple[str, str], float] = {}
count: dict[tuple, list[float]] = defaultdict(list)
for m in MODELS:
    net = NetworkView(m).init_network()
    tc = net.node_params.time_const
    for k, v in zip(tc.keys, tc.semantic_values.detach().numpy().tolist()):
        tau[k].append(v)
    b = net.node_params.bias
    for k, v in zip(b.keys, b.semantic_values.detach().numpy().tolist()):
        bias[k].append(v)
    ss = net.edge_params.syn_strength
    for k, v in zip(ss.keys, ss.semantic_values.detach().numpy().tolist()):
        strength[tuple(k)].append(v)
    sg = net.edge_params.sign
    for k, v in zip(sg.keys, sg.semantic_values.detach().numpy().tolist()):
        sign[tuple(k)] = v
    sc = net.edge_params.syn_count
    for k, v in zip(sc.keys, sc.semantic_values.detach().numpy().tolist()):
        count[tuple(k)].append(v)

def alias(t: str) -> str:
    return {"CT1(Lo1)": "CT1", "CT1(M10)": "CT1", "Am": "Am1"}.get(t, t)

# Photoreceptor -> lamina drive per column (central filter element, summed over R1..R6).
lamina_input: dict[str, float] = defaultdict(float)
for k, v in count.items():
    src, tgt = k[0], k[1]
    rest = tuple(k[2:])
    if src in {"R1", "R2", "R3", "R4", "R5", "R6"} and tgt in {"L1", "L2", "L3", "L4", "L5"} and all(x == 0 for x in rest):
        lamina_input[tgt] += sign[(src, tgt)] * float(np.mean(strength[(src, tgt)])) * float(np.mean(v))
print("syn_count key example:", next(iter(count.keys())))
out = {
    "source": "flyvis 1.2 pretrained ensemble flow/0000, models " + ",".join(MODELS),
    "citation": "Lappalainen et al., Nature 2024",
    "types": {alias(k): {"tau": float(np.mean(v)), "bias": float(np.mean(bias[k])), "tauSd": float(np.std(v))} for k, v in tau.items()},
    "photoreceptor": {"tau": float(np.mean(tau["R1"])), "bias": float(np.mean(bias["R1"])), "laminaInput": dict(lamina_input)},
    "pairs": [{"pre": alias(p), "post": alias(q), "strength": float(np.mean(v)), "sign": sign[(p, q)]} for (p, q), v in strength.items()],
}
(OUT / "flyvis-params.json").write_text(json.dumps(out, indent=1))
print(f"types={len(out['types'])} pairs={len(out['pairs'])}")
for t in ["L1", "L2", "L3", "Mi1", "Tm3", "Mi4", "Mi9", "Tm1", "Tm2", "Tm9", "C3", "T4a", "T5a"]:
    d = out["types"].get(t)
    if d: print(f"{t:5s} tau={d['tau']*1000:6.1f} ms  bias={d['bias']:+.3f}")
for pair in [("L1", "Mi1"), ("Mi1", "T4a"), ("Mi9", "T4a"), ("Mi4", "T4a"), ("Tm3", "T4a"), ("C3", "T4a"), ("Tm9", "T5a"), ("Tm1", "T5a")]:
    r = [p for p in out["pairs"] if (p["pre"], p["post"]) == pair]
    if r: print(f"{pair}: strength={r[0]['strength']:+.4f} sign={r[0]['sign']:+.0f}")
