# app — milestones 1 to 3

Closed loop with a choice of brains. See the root README roadmap.

```
npm install
npm run dev      # http://localhost:5173
npm run build    # typecheck + bundle
```

| Layer | Path | What |
| --- | --- | --- |
| 1 world + body | `src/world/scene.ts`, `src/world/fly.ts` | Striped drum, ground, obstacles, fly rigid body |
| 2 eye | `src/eye/ommatidia.ts`, `src/eye/eye.ts`, `src/eye/photoreceptor.ts` | ~750 ommatidia per eye, six-camera render + gather, log luminance + high-pass |
| 3 brain | `src/brain/types.ts` | `Brain` interface: photoreceptor arrays in, wing amplitudes out |
| | `src/brain/hr.ts` | Hassenstein–Reichardt correlator (stands in for T4/T5 until milestone 3) |
| | `src/brain/stub.ts` | Milestone 1: correlator + P controller, no connectome |
| | `src/brain/graph.ts`, `src/brain/rate-net.ts` | Shared graph JSON loader, CSR build, rate-model runtime on flat typed arrays |
| | `src/brain/connectome.ts` | Milestone 2: correlator → HS/H2 drive → MaleCNS graph → DNg02 L/R readout |
| | `src/brain/optic.ts` | Milestone 3: per-column optic lobe. Virtual photoreceptor per column → lamina → real graph → T4/T5 → lobula plate → DNg02 |
| | `src/brain/flyvis.ts` | Applies flyvis (Lappalainen 2024) per-type tau/bias and per-pair strengths to the graph |
| | `src/brain/net.worker.ts`, `src/brain/remote-net.ts` | The 65k-unit network runs in a Web Worker; homeostatic bias procedure lives there |
| 4 motor | `src/motor/wings.ts` | Wing amplitudes → thrust, yaw torque, bank |
| glue | `src/sim/loop.ts`, `src/ui/hud.ts`, `src/main.ts` | Frame loop, eye + network HUD, wiring |

Graph data in `public/graphs/`: `flight-v1.json` (milestone 2), `optic-v2.json` + `.bin` (milestone 3, from `data/extract_v2.py`), `flyvis-params.json` (from `train/dump_flyvis_params.py`); `fitted-params.json` is picked up automatically when present.

Controls: `[` `]` drum speed, space stops the drum, `b` cycles off / stub / connectome / optic, `r` resets, `v` cycles the eye HUD: luminance, photoreceptor high-pass, T4a − T4b per column. Sliders: net gain (global weight multiplier), out gain, readout level (DNg02 or steering MNs), side mapping.

## Runtime shape

`RateNet.step` is one loop over `indptr / pre / w` (Int32Array, Float32Array) with per-unit `tau` and a clamp. Nothing else touches the arrays, so the same kernel can move to a Web Worker, WASM, or a WebGPU compute pass without changing the graph format. 1072 units and 26k edges at 1 ms substeps costs well under a millisecond per frame in plain JS.

## Milestone 3 result

The 65.8k-unit optic lobe runs in a worker at 120 fps (4 ms substeps). The eye samples the scene at the real 1,771 column directions, one virtual photoreceptor per column drives L1/L2/L3, and the network runs with parameters fitted on the GPU box (`public/graphs/fitted-params.json`, from `train/train_optic.py`).

Closed loop, HS readout, out gain 4:

| drum ω (rad/s) | fly yaw rate |
| --- | --- |
| 1.0 | 0.40 |
| 2.0 | 1.08 |
| −1.0 | −0.80 |
| −2.0 | −0.84 |

Open loop the HS cells lateralise (left 1.5 / right 0 for one rotation, 1.9 / 3.4 for the other), and the T4 view (`v` twice) shows per-column T4a − T4b responses flipping with drum direction on both eyes.

What is hand-written here: the virtual photoreceptor, Weber adaptation, the pooling-cell input scale (`lptcScale`, cells that receive thousands of T4/T5 or LC synapses), the homeostatic rest for those pooling cells, and the HS → wing mapping. Everything from the lamina to the lobula plate is synapse counts, fitted per-type and per-pair parameters, and the connectome's own retinotopy.

**Open:** with the DNg02 readout the fly does not turn. In this graph DNg02 sits at a constant rate: the central-brain bridge units run at the default synapse scale and its posterior-slope relay (PS080) is nearly silent under strong inhibition from PLP034. Calibrating that hop (the milestone-2 recipe, at this graph's operating point) is the next piece of work.

## Milestone 2 result

With the connectome brain, drum rotation direction predicts the DNg02 left/right rate difference and the closed loop stabilizes:

| drum ω (rad/s) | DNg02 L − R | fly yaw rate |
| --- | --- | --- |
| 0.5 | +0.005 | 0.15 |
| 1.0 | +0.026 | 0.49 |
| 2.0 | +0.066 | 1.12 |
| −1.0 | −0.034 | −0.49 |
| −2.0 | −0.079 | −1.19 |

Hand-written pieces: correlator, LPTC injection, tonic DNg02 drive, readout gain and side mapping. Everything between HS and DNg02 is synapse counts and predicted signs. See `src/brain/connectome.ts` header for the side-mapping assumption.
