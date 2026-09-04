"""
Milestone 3 extraction: the real per-column optic lobe joined to the flight graph.

  lamina (L1,L2,L3,L5) -> medulla (Mi1,Tm3,Mi4,Mi9,Tm1,Tm2,Tm4,Tm9,C2,C3)
    -> T4a-d / T5a-d -> lobula plate (LPi*, HS, VS, H2, CH) -> [flight-v1 part]
    -> posterior slope bridge -> DNg02 / DNp -> VNC bridge -> wing/haltere MNs

Plus LPLC2 / LC4 looming inputs and their DNp targets.

Column assignment: annotations carry (assignedOlHex1, assignedOlHex2) for
lamina/medulla types. T4, T5 and Tm3 are unassigned; they take the
synapse-weighted mode of their column-assigned presynaptic partners.

Retinotopy: the visual-field axes are calibrated from the wiring itself.
For each T4 subtype, the offset from its Mi9 input centroid to its Mi4
input centroid in hex space is its preferred direction. T4a prefers
front-to-back and T4c prefers upward, which fixes a rotation (and mirror)
from hex space to (azimuth, elevation) per eye.

Output: out/optic-v2.json + out/optic-v2.bin (see graphio.py), copied to
app/public/graphs/.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather

from graphio import write_graph

RAW = Path(__file__).parent / "raw"
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

# Every flyvis (Lappalainen et al. 2024) cell type that exists in MaleCNS, minus the
# photoreceptors (emulated in the browser) and the two non-columnar giants Am1 and CT1.
COLUMNAR = ["L1", "L2", "L3", "L4", "L5", "Lawf1", "Lawf2", "C2", "C3",
            "Mi1", "Mi2", "Mi4", "Mi9", "Mi10", "Mi13", "Mi14", "Mi15",
            "T1", "T2", "T2a", "T3", "T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d",
            "Tm1", "Tm2", "Tm3", "Tm4", "Tm5Y", "Tm5a", "Tm5b", "Tm5c", "Tm9", "Tm16", "Tm20", "Tm30",
            "TmY3", "TmY4", "TmY5a", "TmY9a", "TmY9b", "TmY10", "TmY13", "TmY14", "TmY15", "TmY18"]
LOBULA_PLATE = r"^(LPi|HS[ENST]$|VS$|VST1$|VST2$|VSm$|H2$|DCH$|VCH$)"
LOOMING = r"^(LPLC2|LC4)$"
DN_TYPES = r"^(DNg02_[a-g]|DNp0[1-6])$"

MIN_EDGE_OL = 2
MIN_EDGE_CB = 5
MIN_BRIDGE = 10
MAX_BRIDGE = 300

SIGN = {"acetylcholine": 1, "gaba": -1, "glutamate": -1, "histamine": -1}
# Hand-set time constants (s). Slow inhibitory arms (Mi4, Mi9, Tm9) vs fast excitatory
# centre (Mi1, Tm3, Tm1, Tm2). These are the first parameters the GPU fit replaces.
TAU = {"L1": 0.010, "L2": 0.010, "L3": 0.015, "L5": 0.010,
       "Mi1": 0.010, "Tm3": 0.010, "Mi4": 0.050, "Mi9": 0.050,
       "Tm1": 0.010, "Tm2": 0.010, "Tm4": 0.015, "Tm9": 0.060, "C2": 0.020, "C3": 0.020,
       "T4a": 0.010, "T4b": 0.010, "T4c": 0.010, "T4d": 0.010,
       "T5a": 0.010, "T5b": 0.010, "T5c": 0.010, "T5d": 0.010}
# Types without a hand-set tau fall back to the superclass default; flyvis overrides all of these in the app.
TAU_BY_SUPERCLASS = {"vnc_motor": 0.02, "descending_neuron": 0.02, "visual_projection": 0.03,
                     "visual_centrifugal": 0.03, "cb_intrinsic": 0.03, "vnc_intrinsic": 0.02, "ol_intrinsic": 0.02}
ROLES = ["input", "brain", "dn", "vnc", "output", "optic"]
SIDES = {"L": 0, "R": 1, "M": 2}
SPACING_DEG = 5.0


def main() -> None:
    ann = pd.read_feather(RAW / "body-annotations.feather")
    ann = ann[ann.status == "Traced"].copy()
    for c in ["type", "superclass", "subclass"]:
        ann[c] = ann[c].fillna("")
    ann["somaSide"] = ann["somaSide"].fillna("M")
    ann = ann.set_index("bodyId", drop=False)
    nt = pd.read_feather(RAW / "body-neurotransmitters.feather").set_index("body")["consensus_nt"].to_dict()

    optic = ann[ann.type.isin(COLUMNAR)]
    lp = ann[ann.type.str.match(LOBULA_PLATE)]
    loom = ann[ann.type.str.match(LOOMING)]
    dns = ann[ann.type.str.match(DN_TYPES)]
    mns = ann[ann.subclass.isin(["wm", "hm"]) & (ann.superclass == "vnc_motor")]
    inputs = pd.concat([lp, loom])
    seed_ids = set(optic.bodyId) | set(inputs.bodyId) | set(dns.bodyId) | set(mns.bodyId)
    print(f"seeds: optic={len(optic)} lp={len(lp)} loom={len(loom)} dn={len(dns)} mn={len(mns)}", file=sys.stderr)

    tbl = feather.read_table(RAW / "connectome-weights-traced.feather", columns=["body_pre", "body_post", "weight"])
    tbl = tbl.filter(pc.greater_equal(tbl["weight"], MIN_EDGE_OL))
    seed_arr = pa.array(np.fromiter(seed_ids, dtype=np.int64))
    touch = pc.or_(pc.is_in(tbl["body_pre"], value_set=seed_arr), pc.is_in(tbl["body_post"], value_set=seed_arr))
    edges = tbl.filter(touch).to_pandas()

    # ---- bridges (same recipe as flight-v1), on the central-brain threshold
    cb_edges = edges[edges.weight >= MIN_EDGE_CB]

    def bridge(up: set[int], down: set[int], allowed: set[str]) -> list[int]:
        w_in = cb_edges[cb_edges.body_pre.isin(up) & ~cb_edges.body_post.isin(seed_ids)].groupby("body_post").weight.sum()
        w_out = cb_edges[cb_edges.body_post.isin(down) & ~cb_edges.body_pre.isin(seed_ids)].groupby("body_pre").weight.sum()
        cand = w_in[w_in >= MIN_BRIDGE].index.intersection(w_out[w_out >= MIN_BRIDGE].index)
        cand = [b for b in cand if b in ann.index and ann.at[b, "superclass"] in allowed]
        return sorted(cand, key=lambda b: -min(w_in[b], w_out[b]))[:MAX_BRIDGE]

    brain_bridge = bridge(set(inputs.bodyId), set(dns.bodyId),
                          {"cb_intrinsic", "visual_projection", "visual_centrifugal", "descending_neuron"})
    vnc_bridge = bridge(set(dns.bodyId), set(mns.bodyId), {"vnc_intrinsic", "ascending_neuron", "vnc_motor"})
    keep_ids = seed_ids | set(brain_bridge) | set(vnc_bridge)
    keep_arr = pa.array(np.fromiter(keep_ids, dtype=np.int64))
    both = pc.and_(pc.is_in(tbl["body_pre"], value_set=keep_arr), pc.is_in(tbl["body_post"], value_set=keep_arr))
    sub = tbl.filter(both).to_pandas()
    print(f"kept units={len(keep_ids)} edges={len(sub)}", file=sys.stderr)

    kept = ann.loc[sorted(keep_ids)].copy()
    optic_ids = set(optic.bodyId)
    role = {}
    for b in kept.bodyId:
        if b in optic_ids:
            role[b] = "optic"
        elif b in set(inputs.bodyId):
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
    kept["nt"] = kept.bodyId.map(lambda b: nt.get(b, "unclear"))
    kept["sign"] = kept.nt.map(SIGN).fillna(0).astype(int)
    kept["label"] = np.where(kept["type"] != "", kept["type"], "untyped_" + kept.superclass)

    # ---- column assignment
    h1 = kept.assignedOlHex1.copy()
    h2 = kept.assignedOlHex2.copy()
    assigned = kept.bodyId[h1.notna() & kept.type.isin(COLUMNAR)]
    ah1 = h1[assigned].to_dict()
    ah2 = h2[assigned].to_dict()
    need = set(kept.bodyId[kept.type.isin(COLUMNAR) & h1.isna()])
    e_in = sub[sub.body_post.isin(need) & sub.body_pre.isin(assigned)]
    e_out = sub[sub.body_pre.isin(need) & sub.body_post.isin(assigned)]
    votes: dict[int, dict[tuple[float, float], float]] = defaultdict(lambda: defaultdict(float))
    for r in e_in.itertuples():
        votes[r.body_post][(ah1[r.body_pre], ah2[r.body_pre])] += r.weight
    for r in e_out.itertuples():
        votes[r.body_pre][(ah1[r.body_post], ah2[r.body_post])] += r.weight
    for b, v in votes.items():
        (c1, c2), _ = max(v.items(), key=lambda kv: kv[1])
        h1[b] = c1
        h2[b] = c2
    kept["h1"] = h1
    kept["h2"] = h2
    n_col = int(kept.h1.notna().sum())
    print(f"column-assigned units: {n_col} / {len(kept)}; unassigned columnar: "
          f"{int((kept.type.isin(COLUMNAR) & kept.h1.isna()).sum())}", file=sys.stderr)

    # ---- retinotopy calibration from T4 subtypes
    def pd_vector(side: str, sub_t: str) -> np.ndarray:
        t4 = kept.bodyId[(kept.type == sub_t) & (kept.somaSide == side)]
        e = sub[sub.body_post.isin(t4) & sub.body_pre.isin(assigned)].copy()
        e["tp"] = e.body_pre.map(kept.type)
        e = e[e.tp.isin(["Mi4", "Mi9"])]
        e["c1"] = e.body_pre.map(ah1)
        e["c2"] = e.body_pre.map(ah2)
        cent = e.groupby(["body_post", "tp"]).apply(
            lambda g: pd.Series({"c1": np.average(g.c1, weights=g.weight), "c2": np.average(g.c2, weights=g.weight)})
        ).unstack("tp")
        d = cent.xs("Mi4", axis=1, level=1) - cent.xs("Mi9", axis=1, level=1)
        return d.dropna().mean().to_numpy()

    def hex_to_cart(v: np.ndarray) -> np.ndarray:
        # axial hex with 120 deg between the two axes
        return np.array([v[0] - 0.5 * v[1], 0.8660254 * v[1]])

    maps = {}
    for side in ["L", "R"]:
        a = hex_to_cart(pd_vector(side, "T4a"))
        c = hex_to_cart(pd_vector(side, "T4c"))
        # rotate so a points to +x (front-to-back = +az on the right eye, -az on the left)
        ang = np.arctan2(a[1], a[0])
        R = np.array([[np.cos(-ang), -np.sin(-ang)], [np.sin(-ang), np.cos(-ang)]])
        c_rot = R @ c
        flip_y = -1.0 if c_rot[1] < 0 else 1.0
        M = np.diag([1.0 if side == "R" else -1.0, flip_y]) @ R
        ortho = np.degrees(np.arctan2(abs(c_rot[1]), abs(c_rot[0])))
        maps[side] = M
        print(f"{side}: T4a PD hex->cart {a.round(2)}, T4c {c.round(2)}, T4c angle after align {ortho:.0f} deg, flip_y={flip_y}",
              file=sys.stderr)

    # column table per side
    col_rows = []
    col_index: dict[tuple[str, float, float], int] = {}
    for side in ["L", "R"]:
        M = maps[side]
        cols = kept[(kept.somaSide == side) & kept.h1.notna()][["h1", "h2"]].drop_duplicates()
        xy = np.array([M @ hex_to_cart(np.array([r.h1, r.h2])) for r in cols.itertuples()])
        xy -= xy.mean(axis=0)
        # scale so the median nearest-neighbour distance is one interommatidial angle
        from scipy.spatial import cKDTree  # type: ignore
        d, _ = cKDTree(xy).query(xy, k=2)
        xy *= SPACING_DEG / np.median(d[:, 1])
        az0 = 80.0 if side == "R" else -80.0
        for (r, (x, y)) in zip(cols.itertuples(), xy):
            az = np.radians(az0 + x)
            el = np.radians(np.clip(y, -80, 80))
            col_index[(side, r.h1, r.h2)] = len(col_rows)
            col_rows.append((SIDES[side], int(r.h1), int(r.h2), az, el))
    col_arr = np.array(col_rows, dtype=object)
    print(f"columns: {len(col_rows)}", file=sys.stderr)

    kept["col"] = [col_index.get((s, a, b), -1) if not np.isnan(a) else -1
                   for s, a, b in zip(kept.somaSide, kept.h1, kept.h2)]

    # ---- arrays
    type_names = sorted(kept.label.unique())
    type_index = {t: i for i, t in enumerate(type_names)}
    idx_of = {b: i for i, b in enumerate(kept.bodyId)}
    types = []
    for t in type_names:
        rows = kept[kept.label == t]
        sc = rows.superclass.mode().iat[0]
        types.append({"name": t, "superclass": sc, "count": int(len(rows)),
                      "nt": rows.nt.mode().iat[0], "tau": TAU.get(t, TAU_BY_SUPERCLASS.get(sc, 0.03))})

    arrays = {
        "units.bodyId": kept.bodyId.to_numpy().astype(np.float64),
        "units.type": kept.label.map(type_index).to_numpy().astype(np.int32),
        "units.side": kept.somaSide.map(SIDES).to_numpy().astype(np.int8),
        "units.role": kept.role.map(ROLES.index).to_numpy().astype(np.int8),
        "units.sign": kept.sign.to_numpy().astype(np.int8),
        "units.col": kept.col.to_numpy().astype(np.int32),
        "edges.pre": sub.body_pre.map(idx_of).to_numpy().astype(np.int32),
        "edges.post": sub.body_post.map(idx_of).to_numpy().astype(np.int32),
        "edges.weight": sub.weight.to_numpy().astype(np.float32),
        "columns.side": col_arr[:, 0].astype(np.int8),
        "columns.h1": col_arr[:, 1].astype(np.int16),
        "columns.h2": col_arr[:, 2].astype(np.int16),
        "columns.az": col_arr[:, 3].astype(np.float32),
        "columns.el": col_arr[:, 4].astype(np.float32),
    }
    meta = {
        "version": 2,
        "source": "male-cns:v1.0 flat-connectome, traced-only, minconf-0.5",
        "license": "CC-BY; cite Berg et al., Cell 2026",
        "units": {"count": int(len(kept))},
        "edges": {"count": int(len(sub))},
        "columns": {"count": int(len(col_rows))},
        "roles": ROLES,
        "sides": ["L", "R", "M"],
        "types": types,
        "extract": {"minEdgeOptic": MIN_EDGE_OL, "minEdgeCentral": MIN_EDGE_CB, "columnar": COLUMNAR,
                    "spacingDeg": SPACING_DEG},
    }
    write_graph(OUT / "optic-v2", meta, arrays)
    app_graphs = Path(__file__).parent.parent / "app" / "public" / "graphs"
    if app_graphs.is_dir():
        write_graph(app_graphs / "optic-v2", meta, arrays)

    # ---- report
    lines = ["# optic-v2 extraction report", "",
             f"Units: {len(kept)}  Edges: {len(sub)}  Types: {len(types)}  Columns: {len(col_rows)}", "",
             "## Units by role", "", kept.role.value_counts().to_string(), "",
             "## Columnar types: count, with column", ""]
    for t in COLUMNAR:
        r = kept[kept.type == t]
        lines.append(f"- {t}: {len(r)} units, {int(r.h1.notna().sum())} with column, nt={r.nt.mode().iat[0] if len(r) else '-'}")
    lines += ["", "## Retinotopy calibration", ""]
    for side, M in maps.items():
        lines.append(f"- {side}: M = {M.round(3).tolist()}")
    lab = kept.label
    sub2 = sub.assign(tp=sub.body_pre.map(lab), tq=sub.body_post.map(lab))
    for target in ["T4a", "T5a", "HSE", "VS", "LPi21"]:
        top = sub2[sub2.tq == target].groupby("tp").weight.sum().sort_values(ascending=False).head(10)
        lines += ["", f"## Inputs to {target}", "", top.to_string()]
    (OUT / "optic-v2.report.md").write_text("\n".join(lines))
    print("\n".join(lines[:40]), file=sys.stderr)


if __name__ == "__main__":
    main()
