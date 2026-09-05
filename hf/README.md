---
license: cc-by-4.0
language:
- en
tags:
- connectome
- drosophila
- neuroscience
- computational-neuroscience
- webgpu
- rate-model
- flyvis
library_name: fruit-fly-brain
pipeline_tag: other
---

# Fruit fly brain: the MaleCNS optic lobe as a runnable network

A simulated fruit fly that sees a 3D scene and steers through its own connectome. This repo holds
the graph and the fitted parameters that the browser app runs: the optic-lobe-to-wing-motor
circuit cut from the 2026 male *Drosophila* central nervous system connectome (MaleCNS, Berg et al.,
*Cell* 2026), with per-type dynamics and per-type-pair synapse strengths fitted so that the real
per-cell wiring detects motion and looming objects.

Code, extraction scripts, trainer and the app: https://github.com/AbijahKaj/fruit-fly-brain-research

## Files

| file | what | size |
| --- | --- | --- |
| `optic.json` + `optic.bin` | the graph: JSON header and typed arrays (per unit: type, side, role, transmitter sign, column, body id; per edge: pre, post, synapse count; per column: side, hex coordinates, azimuth, elevation) | 25 MB |
| `fitted-params.json` | per-type tau, bias and resting level for 93 types, per-type-pair strength for 1,545 pairs, and the photoreceptor model. What the app runs. | 164 kB |
| `flyvis-params.json` | the same fields transferred from one released flyvis model (Lappalainen et al. 2024), the starting point of the fit | 67 kB |
| `runs/` | training logs of the shipped stages | |

## The graph

65,799 units and 1,967,771 edges: every flyvis cell type that exists in MaleCNS for every column of
both eyes (1,771 columns), the lobula plate tangential cells (HS, VS), the looming cells LC4 and
LPLC2, the posterior-slope relays, the descending neurons DNg02 and DNp01–06, VNC relays, and wing
and haltere motor neurons. Every unit is one traced cell. Edge weights are synapse counts; the sign
comes from the presynaptic cell's predicted transmitter. Retinotopy (which column a cell belongs to)
is taken from the wiring itself, calibrated on the T4 Mi9 → Mi4 offsets.

## The model

Each unit is a leaky rate unit:

    tau_i dx_i/dt = -x_i + wScale * sum_j w_ij r_j + ext_i + bias_i,   r = clamp(x, 0, rMax)

with `w_ij = synapse count × sign(pre) × pair strength` and per-type `tau`, `bias`. Photoreceptors
see the scene through Weber adaptation (tau 1 s) and drive the lamina with flyvis's input weights.
Inputs onto pooling cells (tangential cells, LPi, looming LCs) are scaled by 0.001, other
non-fitted edges by 0.02. Substep 4 ms.

## What was fitted, and how

flyvis parameters transfer to this graph (synapse counts agree within about 30%) but give zero
direction selectivity on the real per-cell wiring. Three stages, PyTorch on one RTX 5090, all in
the GitHub repo's `train/`:

1. **Direction selectivity** (`train_optic.py`): per-type tau and bias and per-pair strength on
   moving gratings, loss on the T4/T5 subtypes' tuning against their known preferred directions.
2. **Looming** (`train_loom.py --joint`): LC4/LPLC2 top-5 population selectivity for approaching
   discs against receding and translating ones, gratings and static patterns, plus the stage 1
   objective, a static-contrast penalty on T4/T5 and a left/right symmetry term. Looms scored while
   the disc is 8–35° wide.
3. **HS cells** (`train_loom.py --hs --scene`): the bidirectional wide-field objective (progressive
   motion up, regressive below rest, rest held on every static view) on eye input recorded from the
   app's own scene, played through the app's photoreceptor pipeline.

Results on the trainer:

| | value |
| --- | --- |
| mean direction selectivity index, 16 T4/T5 subtype × eye groups | 0.67 (0.00 before fitting) |
| tuning correlation with the known preferred directions | 0.96 |
| looming selectivity, LC4 and LPLC2, both eyes | 0.89–0.97 |

In the closed loop (browser, WebGPU): the fly follows a rotating striped drum in both directions,
forward flight no longer reads as rotation, it crosses a 40-pillar course without a collision, and
a hovering fly sideslips away from a sphere approaching at 2 units/s from 45°. Open: head-on
approaches, a rest wobble in the closed loop, and the dependence of looming detection on the
background stripe behind the object.

## Loading

```python
import json, numpy as np
h = json.load(open("optic.json"))
buf = open("optic.bin", "rb").read()
def arr(name):
    d = h["arrays"][name]
    return np.frombuffer(buf, dtype=d["dtype"], count=d["length"], offset=d["offset"])
pre, post, count = arr("edges.pre"), arr("edges.post"), arr("edges.weight")
```

The TypeScript loader (`app/src/brain/graph.ts`) and the PyTorch model (`train/graph_torch.py`)
in the GitHub repo read the same format.

## License and credit

The graph and the fitted parameters are derived from MaleCNS and are released under CC BY 4.0,
the license of the source data. Please cite:

> Berg, S. et al. (2026). *A connectome of the male Drosophila melanogaster central nervous system.*
> Cell. FlyEM Project Team, HHMI Janelia Research Campus and Google Research. https://male-cns.janelia.org/

`flyvis-params.json` is derived from flyvis (MIT):

> Lappalainen, J. K. et al. (2024). *Connectome-constrained networks predict neural activity across
> the fly visual system.* Nature. https://github.com/TuragaLab/flyvis

The code is MIT.
