# app — the closed loop

The 3D scene, the fly's eye, the per-column MaleCNS optic lobe (optic-v2) and the wing model, closed into one loop in the browser. See the root README for the roadmap.

```
npm install
npm run dev      # http://localhost:5173
npm run build    # typecheck + bundle
```

| Layer | Path | What |
| --- | --- | --- |
| 1 world + body | `src/world/scene.ts`, `src/world/fly.ts`, `src/world/loom.ts` | Striped drum, ground, pillars and a 40-pillar course, fly rigid body, an approaching or stationary object (the looming stimulus) |
| 2 eye | `src/eye/ommatidia.ts`, `src/eye/eye.ts` | One ommatidium per optic-lobe column at the connectome's own directions (~885 per eye), six-camera render + gather |
| 3 brain | `src/brain/types.ts` | `Brain` interface: luminance per column in, wing amplitudes out |
| | `src/brain/optic.ts` | The optic-v2 brain: virtual photoreceptor per column → lamina → real graph → T4/T5 → lobula plate (HS readout) and → LC4/LPLC2 (looming readout) |
| | `src/brain/graph.ts`, `src/brain/rate-net.ts` | Binary graph loader, CSR build, rate-model runtime on flat typed arrays |
| | `src/brain/flyvis.ts` | Applies the fitted per-type tau/bias and per-pair strengths (flyvis format) to the graph |
| | `src/brain/net-backend.ts`, `src/brain/gpu-net.ts` | Network runtime interface; WebGPU backend (two WGSL compute passes per Euler step, state stays on the GPU) |
| | `src/brain/net.worker.ts`, `src/brain/remote-net.ts` | CPU fallback: the same RateNet in a Web Worker |
| 4 motor | `src/motor/wings.ts` | Wing amplitudes → thrust, yaw torque, bank |
| glue | `src/sim/loop.ts`, `src/ui/hud.ts`, `src/main.ts` | Frame loop, eye + network HUD, wiring |

Graph data in `public/graphs/`: `optic-v2.json` + `.bin` (from `data/extract_v2.py`), `fitted-params.json` (from `train/`), `flyvis-params.json` (the raw flyvis transfer, used only if no fitted file is present).

Controls: `[` `]` drum speed, space stops the drum, `b` brain on/off (off = hover), `r` resets, `v` switches the eye HUD between luminance and T4a − T4b per column, `l` launches or stops a looming object. Sliders: net gain (global weight multiplier), loom gain, out gain (steering readout). `net on` picks WebGPU or the CPU worker. Console: `fly.loom(azDeg, opts)`, `fly.cruise(0.8)` (forward flight through the pillar field, `fly.collisions` counts contacts), `fly.calibrateSides()`, `fly.brain.params`.

## Runtime shape

`RateNet.step` is one loop over `indptr / pre / w` (Int32Array, Float32Array) with per-unit `tau` and a clamp. Nothing else touches the arrays, so the same kernel runs on three devices behind one interface (`NetBackend`, `src/brain/net-backend.ts`):

| Backend | Where | Per Euler step, optic-v2 (65.8k units, 1.97M edges) |
| --- | --- | --- |
| `gpu` (default) | WebGPU compute, `gpu-net.ts` | 0.86 ms on an Apple M-series GPU while the scene renders; identical to the CPU result to float32 rounding (max rate difference 3e-5 after 100 steps) |
| `worker` | Web Worker, `net.worker.ts` | 2.9–3.5 ms in V8, i.e. 1.4x headroom at 4 ms substeps, none once the browser is busy |

The GPU backend keeps `x`, `r`, `ext`, `bias`, CSR and weights on the device. Each frame's substeps are one command buffer with two dispatches per step: `drive` sums in-edges in chunks of 64 (so pooling cells with tens of thousands of inputs do not stall a workgroup), `integrate` updates one unit per thread. Submits never wait; three staging buffers rotate for readback of `r`, so the main thread sees rates one to two frames old while the network itself stays in real time. `settle` and the homeostat take the device exclusively and wait. GPU time comes from timestamp queries when the adapter has them.

Why it matters for the closed loop: the worker, when it cannot keep up, integrates less sim time than passes on screen (`pendingDt` is capped) and the fly's brain runs slow and late. In the Playwright test browser this made the optic brain turn the wrong way at drum ω = 2. On WebGPU the same model, same parameters, follows the drum at every speed tested.

TensorFlow.js was considered and skipped: the model is a sparse graph, not dense tensors. A dense weight matrix would be 65.8k² floats (17 GB); the gather/segment-sum route through tfjs costs several dispatches plus a sync per step. A 60-line WGSL kernel is faster and keeps the graph format.

## Optomotor response

The 65.8k-unit optic lobe runs on WebGPU (or in a worker) at 4 ms substeps in real time. The eye samples the scene at the real 1,771 column directions, one virtual photoreceptor per column drives L1/L2/L3, and the network runs with parameters fitted on the GPU box (`public/graphs/fitted-params.json`, from `train/`).

Closed loop, HS readout, out gain 4 (fly reset before each speed, mean yaw over 2 s after 3 s):

| drum ω (rad/s) | fly yaw rate, WebGPU | worker (lagging, see above) |
| --- | --- | --- |
| 0 | −0.05 | 0.07 |
| 1.0 | 0.44 | 0.59 |
| 2.0 | 0.49 | −0.77 |
| −1.0 | −0.94 | −0.35 |
| −2.0 | −1.23 | −0.84 |

Open loop the HS cells lateralise (left 1.5 / right 0 for one rotation, 1.9 / 3.4 for the other), and the T4 view (`v` twice) shows per-column T4a − T4b responses flipping with drum direction on both eyes.

What is hand-written here: the virtual photoreceptor, Weber adaptation, the pooling-cell input scale (`lptcScale`, cells that receive thousands of T4/T5 or LC synapses), the homeostatic rest for those pooling cells, the tonic drive on DNg02, and the HS → wing mapping with its drum-calibrated per-side gains. Everything from the lamina to the lobula plate is synapse counts, fitted per-type and per-pair parameters, and the connectome's own retinotopy.

**Open:** the steering readout sits at HS rather than at the descending neurons. In this graph DNg02 sits at a constant rate: the central-brain bridge units run at the default synapse scale and its posterior-slope relay (PS080, GABAergic) is nearly silent under strong inhibition from PLP034. `params.readout = "dng02"` selects that readout for the calibration work.

## Looming avoidance

The looming pathway of the connectome, LC4 and LPLC2 converging on the giant fiber (DNp01) and DNp02–06, is in `optic-v2`. Each cell's receptive-field centre is derived from its presynaptic columns (synapse-weighted mean direction): 165 left / 146 right cells, azimuth 17–150° per side.

With only the grating fit the LC cells did nothing on approach: their inputs sat at the pooling scale, and worse, the fitted T4/T5 answered *static* contrast (hundreds of cells at the rate ceiling in the rendered scene at rest). Two more fitting stages on the 5090 fixed both (`train/train_loom.py`, see `train/README.md`): looming / receding / translating discs and static gratings over structured backgrounds, jointly with the grating objective. Result on the trainer's stimuli: looming selectivity 0.97–1.0 for all four LC groups, zero response to gratings, receding and translating objects, T4/T5 quiet under static patterns, direction selectivity kept (mean DSI 0.66, tuning correlation 0.96).

In the scene (WebGPU, `fitted-params.json` = stage `loom6`), open loop, retinal looming sphere at 5 units/s, top-5 LC readout above rest:

| stimulus | left readout | right readout |
| --- | --- | --- |
| rest | 0 | 0 |
| loom from −45° | 0.49 | 0 |
| loom from +45° | 0 | 0.62 |
| drum ±1 rad/s | 0 | 0 |

Closed loop, fly cruising at 1.1 units/s toward a stationary sphere (radius 0.6) 8 units ahead, optomotor output off, loom gain 5–10:

| object azimuth | closest approach, gain 0 | gain on | turn |
| --- | --- | --- | --- |
| ±4° | 0.72–0.78 (graze) | 0.75–0.78 | 70–78° away |
| ±6° | 0.84 | 1.03–1.05 | 66° away |
| ±8° | 1.3 | 1.3–1.36 | 45–58° away |
| 0° | 0 (hit) | 0.02–0.2 (hit) | −5 to 16° |

Head-on approaches are the open case: the LC receptive fields start 17° off the midline, so a centred object only enters them when it already subtends ~20° (1.5 s before impact at this speed), and both eyes then fire equally. The readout has a tie-break (a bilateral signal turns right and lowers the wing amplitude) but the signal comes too late. Pillar course (40 pillars over 70 × 70, 45 s at cruise): 0 collisions with or without the optomotor output.

Hand-written here: the top-k population readout per eye (the trainer scores the same top-5), the loom gain / tie-break / brake, the collision counter. LC4/LPLC2 tau, bias and every input synapse type are fitted; their receptive fields are the wiring's.

**Rotation vs translation, solved by training on the scene.** During forward flight both HS populations rise, more on the side with nearer objects, and an unfitted HS readout took that for a rotation: the fly turned toward the pillars (110–135° over 7 s of cruise). The fix that transferred was to fit the HS cells on eye input recorded from this scene (`fly.record`, 50 episodes of the fly yawing, cruising or sitting still, played through the browser's own photoreceptor pipeline in the trainer), with the classic bidirectional objective: progressive motion up, regressive below rest, rest level held on every static view. Three things had to be true first, all found on the way: the eye must stay level while the body banks (the head does that in a real fly; a banked eye collapsed both HS sides and locked the loop in a spin), the readout offsets must be taken after a 2.5 s live warm-up in the scene, and the trainer had to run the pooling cells at the browser's synapse scale with a fitted bias instead of the browser re-centring them (see `train/README.md`). The readout is now each side's deviation relative to its own rest, so a fast self-rotation that silences both sides cancels instead of leaving the difference of two rest levels as a permanent turn.

With that (`fitted-params.json` = stage `hs7`, out gain 0.25):

| | before | now |
| --- | --- | --- |
| drum ω = +1 / −1 / +2 / −2 | 0.7 / −0.3 / 1.6 / 0.0 | 0.55 / −0.62 / 1.33 / −1.61 |
| cruise drift over 7 s | 110–135° | 10–50° |
| pillar course, 45 s at cruise, loom gain 5 | 0–1 collisions, circling | 0 collisions, 53 units travelled |

Still open: a rest wobble with the loop closed (mean |yaw| ≈ 0.2 rad/s at out gain 0.25, more at higher gain; the readout is sensitive and the wing-to-yaw dynamics add delay), and the residual cruise drift.
