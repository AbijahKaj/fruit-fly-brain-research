# Fruit fly brain

A workspace for understanding the 2026 male *Drosophila* connectome, playing with its public data in the browser, and eventually **simulating enough of the fly’s nervous system in TypeScript to make it fly**.

The map is not a mind. It is a wiring diagram. The bet of this project is that a small, well-chosen subgraph — vision in, wing motor out — is enough to start, and that a web app is the right place to see that graph think.

## Goal

1. **Understand the research** — what was mapped, why the male CNS (brain + nerve cord) is a milestone, where male and female flies actually differ.
2. **Play with the data** — a web app that loads real MaleCNS cells and synapses, not screenshots.
3. **Model the network in TypeScript** — leaky neurons + weighted edges from the connectome, running in the browser.
4. **Close the loop in a 3D scene** — a simulated fly renders what it sees, the connectome graph turns that image into wing commands, and the fly steers: stabilize, avoid looming objects, approach a bar.

We will not try to run all ~166,000 neurons as biophysically detailed cells. We extract the flight circuit, give it simple dynamics, and grow from there.

## Why this dataset can support flight

Earlier fly maps were often brain-only. The [male CNS connectome](https://male-cns.janelia.org/) (Berg et al., *Cell*, 2026) includes:

- **Central brain** — decisions, courtship, multimodal integration
- **Optic lobes** — visual motion (the Hoeller companion paper)
- **Ventral nerve cord** — the spinal-cord analog that actually drives wings and legs
- **Intact neck connective** — the cables between brain and body stay in one specimen

That is a sensory-to-motor path in one animal. Flight is a good first behavior because the bottleneck is known: a population of descending neurons (**DNg02**, ~15 pairs) carries visual motion from the brain into the dorsal flight neuropil and sets wingbeat amplitude by a population code ([Namiki et al., 2022](https://pubmed.ncbi.nlm.nih.gov/35090590/)). Downstream are wing motor neurons and the flight CPG in the VNC.

The connectome is static. Dynamics, neurotransmitters, and biomechanics are not in the EM volume. We will borrow those: predicted transmitters from the MaleCNS tables, a simple neuron model, and a cartoon (later physical) body.

## How the pieces fit

Four layers, one loop. Only the brain layer carries connectome wiring; the others are deliberately simple.

```
   1. WORLD + BODY          2. EYE                 3. BRAIN                 4. MOTOR
   three.js scene      ->   offscreen render   ->  connectome graph    ->   DN rates -> wing
   rigid-body fly           ~885 columns/eye       lamina in, DNs out       amplitude L/R
   thrust, roll, yaw        log lum + high-pass    WebGPU (worker fallback) thrust, roll, yaw
         ^                                                                       |
         +---------------------------- pose update <-----------------------------+

   MaleCNS v1.0 (neuPrint / feather)  ->  shared graph JSON  ->  PyTorch (train, 5090)
                                                              ->  TypeScript (run, browser)
```

The brain layer is the only one that carries the scientific claim. The body has no aerodynamics, the eye is a sampled cube map, and the motor map is the DNg02 population code. What we test is whether the published wiring, with borrowed signs and fitted time constants, turns an image stream into a sensible wing command.

[Neuroglancer](https://github.com/google/neuroglancer) is Google’s TypeScript/WebGL viewer for the EM volume, meshes, and skeletons. Use it to *see* neurons. Notes and controls: [`sources/neuroglancer.md`](sources/neuroglancer.md). Official tutorial: [connectomics.readthedocs.io — Neuroglancer](https://connectomics.readthedocs.io/en/latest/external/neuroglancer.html).

Hosted MaleCNS scene (Chrome):

[neuroglancer-demo.appspot.com — male-cns-v1.0](https://neuroglancer-demo.appspot.com/#!gs://flyem-male-cns/v1.0/male-cns-v1.0.jso)

## Repo

| Path | What |
| --- | --- |
| [`a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain.md`](a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain.md) | Saved [Google Research blog](https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/) (2026-09-03) |
| [`research-notes.md`](research-notes.md) | Paper, numbers, companion papers, download URLs |
| [`sources/neuroglancer.md`](sources/neuroglancer.md) | How to use Neuroglancer on this data |
| [`app/`](app/) | TypeScript web app: 3D scene, eye sampling, graph runtime on WebGPU with a Web Worker fallback, HUD |
| [`data/`](data/) | Python extraction from the public MaleCNS tables → graph files, with reports of static checks |
| [`train/`](train/) | PyTorch side: flyvis parameter transfer, reference simulations, the DS trainer for the GPU box |

MaleCNS is **CC-BY**. Cite Berg et al., *Cell* 2026, and the [FlyEM project](https://www.janelia.org/project-team/flyem/male-cns-connectome).

## Explore the data

1. Read the saved blog, then the headline results in [`research-notes.md`](research-notes.md).
2. Open the [Cell Type Explorer](https://reiserlab.github.io/celltype-explorer-drosophila-male-cns/) and look up **AOTU008** (dimorphic example from the blog) and **DNg02** (flight descending neurons).
3. Open the [Neuroglancer scene](https://neuroglancer-demo.appspot.com/#!gs://flyem-male-cns/v1.0/male-cns-v1.0.jso). Double-click a neuron, press `H` for controls.
4. [neuPrint](https://neuprint.janelia.org) dataset `male-cns:v1.0` — query upstream/downstream of a cell type (free account + API token).

The scripts in [`data/`](data/) pull the annotation, transmitter, and 1.1 GB weight tables from the [download page](https://male-cns.janelia.org/download/) and cut the graphs the app runs. Do not pull the EM volume into this repo.

## Roadmap

One system, optic-v2: the per-column MaleCNS optic lobe with its path to the wings, fitted on the GPU box, running in the browser. What follows is what it does, what it is missing, and what comes after.

**Done**

- **World, eye, body.** three.js scene with a striped drum, pillars and an approaching object; the eye samples a six-face render at the connectome's own 1,771 column directions; a rigid-body fly with wing amplitude L/R → thrust, yaw, bank. Runs at 100–120 fps.
- **The graph.** 65.8k units, 1.97M edges: every flyvis cell type that exists in MaleCNS for every column of both eyes, the lobula plate, the looming LCs, the posterior-slope relays, DNg02 and DNp, VNC relays, wing and haltere motor neurons. Retinotopy is calibrated from the wiring (T4 Mi9→Mi4 offsets). Extraction in [`data/`](data/).
- **The runtime.** The rate model on WebGPU, 0.86 ms per 4 ms substep with the scene rendering (CPU worker fallback, 3 ms). One graph format for the browser and the PyTorch trainer.
- **Direction selectivity.** flyvis parameters transfer (synapse counts agree within ~30%) but give zero direction selectivity on the real per-cell wiring. Fitting per-type tau/bias and per-pair strength on the 5090 gives DSI 0.66 and tuning correlation 0.96 across all 16 T4/T5 subtype × eye groups, with the correct preferred directions, and motion detectors that stay quiet under static contrast. The fly follows the drum with the turn read at the HS cells.
- **Looming avoidance.** LC4/LPLC2 fitted in the same way: selectivity ~1.0 for approaching objects, nothing for gratings, receding or translating ones. In the scene the cruising fly turns 45–78° away from objects 4–8° off its heading. Details in [`train/README.md`](train/README.md) and [`app/README.md`](app/README.md).

**Open**

- **Head-on objects still hit.** The LC receptive fields start 17° off the midline, so a centred object enters them about 1.5 s before impact and both eyes fire equally.
- **Rotation vs translation.** The optomotor loop reads the translation flow of forward flight as rotation and turns toward nearby pillars. In this graph the HS cells receive only ipsilateral input, so a rotation-selective HS cell is not expressible; the classic bidirectional HS is, but two fits of it did not transfer from the trainer's ray-cast world to the scene. Next try: train on eye input recorded from the scene itself.
- **The DNg02 hop.** The steering readout sits at HS. HS → posterior slope (PS080, GABAergic) → DNg02 is not calibrated: the relay is nearly silent under PLP034 inhibition and DNg02 sits at a constant rate. Findings from the wiring are in [`data/README.md`](data/README.md).

**Next**

- bar fixation / approach: LC10 and related types (not yet extracted)
- altitude and speed: ventral optic flow
- tune the few free gains on the GPU with the body simulated in PyTorch, against a behavioral objective; export, run in browser
- done when: the fly flies through the scene, avoids obstacles, and approaches a target using only rendered images

**Later** — richer biomechanics ([NeuroMechFly](https://github.com/NeLy-EPFL/NeuroMechFly), DeepMind [flybody](https://github.com/TuragaLab/flybody)), haltere feedback, walking, or the dimorphic courtship switches the *Cell* paper is actually about. Whole-brain graph controllers already exist in research ([FlyGM](https://arxiv.org/html/2602.17997v3)); this project stays small, inspectable, and in-browser.

## Compute

| Job | Where |
| --- | --- |
| three.js scene, eye sampling, graph inference (WebGPU compute), viewer | Mac, browser |
| neuPrint exploration, small feather files | Mac |
| filtering the 1.1 GB weight table, per-column adjacency | GPU box (RAM/disk), CPU work |
| flyvis-style parameter fitting, gain tuning | GPU box, RTX 5090 (32 GB) |

One graph schema (JSON: units, types, edges, weights, per-type params) is shared by the PyTorch trainer and the TypeScript runtime so both provably run the same network. The browser never trains.

## Known risks

- **Raw weights do not give sane dynamics.** Every published whole-graph model rescales globally and saturates. Expect to fit two or three global gains before anything works.
- **Motion detection needs temporal structure.** Uniform time constants give no direction selectivity; that is why per-type parameters are fitted instead of hoped for.
- **Flight is not in the graph.** The wingbeat CPG, power-muscle mechanics, and haltere feedback are not in the EM volume. The body is a cartoon and stays one until a real body model is worth it.
- **No emergent navigation.** Real goal-seeking needs the central complex, state, and odor. Expect reflexes (stabilize, avoid, approach); chained in a 3D world they already look like a fly deciding where to go.

## Neuron model

TypeScript, no Python in the hot loop. Implemented in [`app/src/brain/rate-net.ts`](app/src/brain/rate-net.ts) and the worker; the PyTorch trainer runs the same equation.

```ts
// tau dx/dt = -x + wScale * sum_j w_ji * r_j + I_ext + bias
// r      = clamp(x, 0, rMax)          browser
// r      = rMax * tanh(relu(x)/rMax)  trainer (smooth ceiling for gradients)
// w_ji   = synapse_count(j->i) * sign(transmitter_j) * strength(type_j, type_i)
```

Per-type tau and bias, and per-type-pair strength, come from flyvis and then from the fit; edges keep their connectome counts. Units without fitted parameters get a homeostatic bias that holds them at the flyvis resting level.

That is a cartoon of a neuron. It is enough to ask: does the published wiring turn a rendered image stream into a left/right wing command that keeps the fly in the air and away from walls? If yes, we made the insect fly in the only sense this repo claims. If no, the graph, the signs, the time constants, or the missing CPG is the next experiment.

## Related reading

- Berg et al., [*Sexual dimorphism in the complete connectome of the Drosophila male central nervous system*](https://doi.org/10.1016/j.cell.2026.08.015), *Cell* 2026
- Hoeller et al., visual pathways companion, *Cell* 2026 — [DOI](https://doi.org/10.1016/j.cell.2026.08.014)
- Namiki et al., [DNg02 flight descending neurons](https://pubmed.ncbi.nlm.nih.gov/35090590/), 2022
- Lappalainen et al., [Connectome-constrained networks predict neural activity across the fly visual system](https://www.nature.com/articles/s41586-024-07939-3), *Nature* 2024 — code: [flyvis](https://github.com/TuragaLab/flyvis)
- Shiu et al., [A Drosophila computational brain model reveals sensorimotor processing](https://www.nature.com/articles/s41586-024-07763-9), *Nature* 2024 — whole-connectome LIF simulation
- [Male CNS project](https://male-cns.janelia.org/) · [neuPrint](https://neuprint.janelia.org) · [Neuroglancer](https://github.com/google/neuroglancer)
