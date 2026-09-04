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
| | `src/brain/net-backend.ts`, `src/brain/gpu-net.ts` | Network runtime interface; WebGPU backend (two WGSL compute passes per Euler step, state stays on the GPU) |
| | `src/brain/net.worker.ts`, `src/brain/remote-net.ts` | CPU fallback: the same RateNet in a Web Worker |
| 4 motor | `src/motor/wings.ts` | Wing amplitudes → thrust, yaw torque, bank |
| glue | `src/sim/loop.ts`, `src/ui/hud.ts`, `src/main.ts` | Frame loop, eye + network HUD, wiring |

Graph data in `public/graphs/`: `flight-v1.json` (milestone 2), `optic-v2.json` + `.bin` (milestone 3, from `data/extract_v2.py`), `flyvis-params.json` (from `train/dump_flyvis_params.py`); `fitted-params.json` is picked up automatically when present.

Controls: `[` `]` drum speed, space stops the drum, `b` cycles off / stub / connectome / optic, `r` resets, `v` cycles the eye HUD: luminance, photoreceptor high-pass, T4a − T4b per column. Sliders: net gain (global weight multiplier), out gain, readout level (DNg02 or steering MNs), side mapping. `net on` picks WebGPU or the CPU worker for the optic brain.

## Runtime shape

`RateNet.step` is one loop over `indptr / pre / w` (Int32Array, Float32Array) with per-unit `tau` and a clamp. Nothing else touches the arrays, so the same kernel runs on three devices behind one interface (`NetBackend`, `src/brain/net-backend.ts`):

| Backend | Where | Per Euler step, optic-v2 (65.8k units, 1.97M edges) |
| --- | --- | --- |
| `gpu` (default) | WebGPU compute, `gpu-net.ts` | 0.86 ms on an Apple M-series GPU while the scene renders; identical to the CPU result to float32 rounding (max rate difference 3e-5 after 100 steps) |
| `worker` | Web Worker, `net.worker.ts` | 2.9–3.5 ms in V8, i.e. 1.4x headroom at 4 ms substeps, none once the browser is busy |
| main thread | `rate-net.ts` | milestone 2 only (1k units, well under a millisecond) |

The GPU backend keeps `x`, `r`, `ext`, `bias`, CSR and weights on the device. Each frame's substeps are one command buffer with two dispatches per step: `drive` sums in-edges in chunks of 64 (so pooling cells with tens of thousands of inputs do not stall a workgroup), `integrate` updates one unit per thread. Submits never wait; three staging buffers rotate for readback of `r`, so the main thread sees rates one to two frames old while the network itself stays in real time. `settle` and the homeostat take the device exclusively and wait. GPU time comes from timestamp queries when the adapter has them.

Why it matters for the closed loop: the worker, when it cannot keep up, integrates less sim time than passes on screen (`pendingDt` is capped) and the fly's brain runs slow and late. In the Playwright test browser this made the optic brain turn the wrong way at drum ω = 2. On WebGPU the same model, same parameters, follows the drum at every speed tested.

TensorFlow.js was considered and skipped: the model is a sparse graph, not dense tensors. A dense weight matrix would be 65.8k² floats (17 GB); the gather/segment-sum route through tfjs costs several dispatches plus a sync per step. A 60-line WGSL kernel is faster and keeps the graph format.

## Milestone 3 result

The 65.8k-unit optic lobe runs on WebGPU (or in a worker) at 4 ms substeps in real time. The eye samples the scene at the real 1,771 column directions, one virtual photoreceptor per column drives L1/L2/L3, and the network runs with parameters fitted on the GPU box (`public/graphs/fitted-params.json`, from `train/train_optic.py`).

Closed loop, HS readout, out gain 4 (fly reset before each speed, mean yaw over 2 s after 3 s):

| drum ω (rad/s) | fly yaw rate, WebGPU | worker (lagging, see above) |
| --- | --- | --- |
| 0 | −0.05 | 0.07 |
| 1.0 | 0.44 | 0.59 |
| 2.0 | 0.49 | −0.77 |
| −1.0 | −0.94 | −0.35 |
| −2.0 | −1.23 | −0.84 |

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
