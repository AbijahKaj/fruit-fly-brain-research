"""
Extract the milestone-2 flight subgraph from MaleCNS v1.0 flat tables into
the shared graph JSON that app/src/brain/graph.ts loads.

Input (data/raw/, from gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/):
  body-annotations.feather          bodyId, type, superclass, subclass, somaSide, status, ...
  body-neurotransmitters.feather    body, consensus_nt, ...
  connectome-weights-traced.feather body_pre, body_post, weight, type_pre, type_post

Output:
  data/out/flight-v1.json           the graph
  data/out/flight-v1.report.md      what was pulled and the static sanity checks

Graph = seed populations + one-hop bridges:
  visual inputs (LPTCs, looming LCs)  ->  [brain bridge]  ->  DNs  ->  [VNC bridge]  ->  wing / haltere MNs
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.feather as feather

RAW = Path(__file__).parent / "raw"
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- selection
MIN_EDGE = 5          # synapses; drop weaker connections
MIN_BRIDGE_IN = 10    # a bridge neuron must get >= this many synapses from the upstream set
MIN_BRIDGE_OUT = 10   # ... and send >= this many to the downstream set
MAX_BRIDGE_BRAIN = 300
MAX_BRIDGE_VNC = 300

INPUT_TYPES = re.compile(r"^(HSE|HSN|HSS|HST|VS|VST1|VST2|VSm|H2|DCH|VCH|LPLC2|LC4)$")
DN_TYPES = re.compile(r"^(DNg02_[a-g]|DNp0[1-6])$")

SIGN = {"acetylcholine": +1, "gaba": -1, "glutamate": -1, "histamine": -1,
        "dopamine": 0, "serotonin": 0, "octopamine": 0, "unclear": 0}
TAU_BY_SUPERCLASS = {"vnc_motor": 0.02, "descending_neuron": 0.02, "visual_projection": 0.03,
                     "visual_centrifugal": 0.03, "cb_intrinsic": 0.03, "vnc_intrinsic": 0.02}


def main() -> None:
    ann = pd.read_feather(RAW / "body-annotations.feather")
    ann = ann[ann.status == "Traced"].copy()
    ann["type"] = ann["type"].fillna("")
    ann["superclass"] = ann["superclass"].fillna("")
    ann["subclass"] = ann["subclass"].fillna("")
    ann["somaSide"] = ann["somaSide"].fillna("M")
    ann = ann.set_index("bodyId", drop=False)

    nt = pd.read_feather(RAW / "body-neurotransmitters.feather").set_index("body")
    nt_of = nt["consensus_nt"].to_dict()

    inputs = ann[ann["type"].str.match(INPUT_TYPES)]
    dns = ann[ann["type"].str.match(DN_TYPES)]
    mns = ann[ann.subclass.isin(["wm", "hm"]) & (ann.superclass == "vnc_motor")]
    print(f"seeds: inputs={len(inputs)} DNs={len(dns)} MNs={len(mns)}", file=sys.stderr)

    seed_ids = set(inputs.bodyId) | set(dns.bodyId) | set(mns.bodyId)

    # Load only the edges we might need: pre or post in the seed set, above threshold.
    tbl = feather.read_table(RAW / "connectome-weights-traced.feather",
                             columns=["body_pre", "body_post", "weight"])
    tbl = tbl.filter(pc.greater_equal(tbl["weight"], MIN_EDGE))
    seed_arr = np.fromiter(seed_ids, dtype=np.int64)
    touch = pc.or_(pc.is_in(tbl["body_pre"], value_set=__import__("pyarrow").array(seed_arr)),
                   pc.is_in(tbl["body_post"], value_set=__import__("pyarrow").array(seed_arr)))
    edges = tbl.filter(touch).to_pandas()
    print(f"edges touching seeds (w>={MIN_EDGE}): {len(edges)}", file=sys.stderr)

    # ---- bridges: brain (inputs -> X -> DNs), VNC (DNs -> Y -> MNs)
    def bridge(up: set[int], down: set[int], cap: int, allowed_superclass: set[str]) -> list[int]:
        w_in = edges[edges.body_pre.isin(up) & ~edges.body_post.isin(seed_ids)].groupby("body_post").weight.sum()
        w_out = edges[edges.body_post.isin(down) & ~edges.body_pre.isin(seed_ids)].groupby("body_pre").weight.sum()
        cand = w_in[w_in >= MIN_BRIDGE_IN].index.intersection(w_out[w_out >= MIN_BRIDGE_OUT].index)
        cand = [b for b in cand if b in ann.index and ann.at[b, "superclass"] in allowed_superclass]
        score = {b: float(min(w_in[b], w_out[b])) for b in cand}
        return sorted(cand, key=lambda b: -score[b])[:cap]

    brain_bridge = bridge(set(inputs.bodyId), set(dns.bodyId), MAX_BRIDGE_BRAIN,
                          {"cb_intrinsic", "visual_projection", "visual_centrifugal", "descending_neuron"})
    vnc_bridge = bridge(set(dns.bodyId), set(mns.bodyId), MAX_BRIDGE_VNC,
                        {"vnc_intrinsic", "ascending_neuron", "vnc_motor"})
    print(f"bridges: brain={len(brain_bridge)} vnc={len(vnc_bridge)}", file=sys.stderr)

    keep_ids = seed_ids | set(brain_bridge) | set(vnc_bridge)
    # Edges among kept bodies need a second pass (bridge->bridge edges were not loaded).
    keep_arr = __import__("pyarrow").array(np.fromiter(keep_ids, dtype=np.int64))
    both = pc.and_(pc.is_in(tbl["body_pre"], value_set=keep_arr), pc.is_in(tbl["body_post"], value_set=keep_arr))
    sub = tbl.filter(both).to_pandas()

    # ---- units
    kept = ann.loc[sorted(keep_ids)].copy()
    role = {}
    for b in kept.bodyId:
        if b in set(inputs.bodyId):
            role[b] = "input"
        elif b in set(mns.bodyId):
            role[b] = "output"
        elif b in set(dns.bodyId):
            role[b] = "dn"
        elif b in set(brain_bridge):
            role[b] = "brain"
        else:
            role[b] = "vnc"
    kept["role"] = kept.bodyId.map(role)
    kept["nt"] = kept.bodyId.map(lambda b: nt_of.get(b, "unclear"))
    kept["sign"] = kept.nt.map(SIGN).fillna(0).astype(int)
    kept["label"] = np.where(kept["type"] != "", kept["type"], "untyped_" + kept.superclass)

    type_names = sorted(kept["label"].unique())
    type_index = {t: i for i, t in enumerate(type_names)}
    idx_of = {b: i for i, b in enumerate(kept.bodyId)}

    sub = sub[sub.body_pre.isin(idx_of) & sub.body_post.isin(idx_of)]
    pre = sub.body_pre.map(idx_of).to_numpy()
    post = sub.body_post.map(idx_of).to_numpy()
    wgt = sub.weight.to_numpy()

    types = []
    for t in type_names:
        rows = kept[kept.label == t]
        sc = rows.superclass.mode().iat[0] if len(rows) else ""
        types.append({
            "name": t,
            "superclass": sc,
            "count": int(len(rows)),
            "nt": rows.nt.mode().iat[0] if len(rows) else "unclear",
            "tau": TAU_BY_SUPERCLASS.get(sc, 0.03),
        })

    graph = {
        "version": 1,
        "source": "male-cns:v1.0 flat-connectome, traced-only, minconf-0.5",
        "license": "CC-BY; cite Berg et al., Cell 2026",
        "extract": {"minEdge": MIN_EDGE, "minBridgeIn": MIN_BRIDGE_IN, "minBridgeOut": MIN_BRIDGE_OUT,
                    "inputTypes": INPUT_TYPES.pattern, "dnTypes": DN_TYPES.pattern, "mnSubclass": ["wm", "hm"]},
        "units": {
            "count": int(len(kept)),
            "bodyId": kept.bodyId.astype(int).tolist(),
            "type": kept.label.map(type_index).astype(int).tolist(),
            "side": kept.somaSide.tolist(),
            "role": kept.role.tolist(),
            "sign": kept.sign.astype(int).tolist(),
            "nt": kept.nt.tolist(),
        },
        "types": types,
        "edges": {
            "count": int(len(sub)),
            "pre": pre.astype(int).tolist(),
            "post": post.astype(int).tolist(),
            "weight": wgt.astype(int).tolist(),
        },
    }
    payload = json.dumps(graph, separators=(",", ":"))
    (OUT / "flight-v1.json").write_text(payload)
    app_graphs = Path(__file__).parent.parent / "app" / "public" / "graphs"
    if app_graphs.is_dir():
        (app_graphs / "flight-v1.json").write_text(payload)
    print(f"wrote units={len(kept)} edges={len(sub)} types={len(types)}", file=sys.stderr)

    # ---- report + static checks
    lines = ["# flight-v1 extraction report", "",
             f"Units: {len(kept)}  Edges: {len(sub)} (w >= {MIN_EDGE})  Types: {len(types)}", "",
             "## Units by role", "", kept.role.value_counts().to_string(), "",
             "## Transmitter sign by role", "",
             kept.groupby(["role", "nt"]).size().unstack(fill_value=0).to_string(), ""]

    def partners(mask_pre, mask_post, by="label", top=25):
        e = sub[mask_pre(sub.body_pre) & mask_post(sub.body_post)]
        pre_lab = e.body_pre.map(kept[by])
        post_lab = e.body_post.map(kept[by])
        return pd.DataFrame({"pre": pre_lab.values, "post": post_lab.values, "w": e.weight.values}) \
            .groupby(["pre", "post"]).w.sum().sort_values(ascending=False).head(top)

    is_dng02 = kept.label.str.startswith("DNg02")
    dng02_ids = set(kept.bodyId[is_dng02])
    lines += ["## Static check 1: what drives DNg02 (top pre-types, summed synapses)", "",
              partners(lambda s: ~s.isin(dng02_ids), lambda s: s.isin(dng02_ids)).to_string(), ""]
    lines += ["## Static check 2: what DNg02 drives (top post-types)", "",
              partners(lambda s: s.isin(dng02_ids), lambda s: ~s.isin(dng02_ids)).to_string(), ""]

    # Laterality: DNg02 soma side vs wing MN soma side, direct + via one VNC hop.
    mn_ids = set(mns.bodyId) & keep_ids
    side = kept.somaSide.to_dict()
    lat = Counter()
    d = sub[sub.body_pre.isin(dng02_ids) & sub.body_post.isin(mn_ids)]
    for r in d.itertuples():
        lat[(side[r.body_pre], side[r.body_post], "direct")] += r.weight
    vnc_ids = set(vnc_bridge)
    first = sub[sub.body_pre.isin(dng02_ids) & sub.body_post.isin(vnc_ids)]
    second = sub[sub.body_pre.isin(vnc_ids) & sub.body_post.isin(mn_ids)]
    out_by_mid = defaultdict(Counter)
    for r in second.itertuples():
        out_by_mid[r.body_pre][side[r.body_post]] += r.weight
    for r in first.itertuples():
        for ms, w2 in out_by_mid[r.body_post].items():
            lat[(side[r.body_pre], ms, "via-vnc")] += r.weight * w2 / 100.0
    lines += ["## Static check 3: DNg02 laterality onto wing/haltere MNs", "",
              "(DN soma side, MN soma side, path) -> synapse weight (via-vnc is w1*w2/100)", ""]
    lines += [f"- {k}: {v:.0f}" for k, v in sorted(lat.items())] + [""]

    lines += ["## Static check 4: LPTC -> DN direct edges", "",
              partners(lambda s: s.isin(set(inputs.bodyId)), lambda s: s.isin(set(dns.bodyId))).to_string(), ""]

    lines += ["## Wing / haltere MNs present", "", mns[mns.bodyId.isin(keep_ids)].groupby(["type", "somaSide"]).size()
              .unstack(fill_value=0).to_string(), ""]
    (OUT / "flight-v1.report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
