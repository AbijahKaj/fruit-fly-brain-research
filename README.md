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

Four layers. Every milestone keeps the whole loop running; a layer starts as a stub and is later replaced by real wiring.

```
   1. WORLD + BODY          2. EYE                 3. BRAIN                 4. MOTOR
   three.js scene      ->   offscreen render   ->  connectome graph    ->   DN rates -> wing
   rigid-body fly           ~750 ommatidia/eye     lamina in, DNs out       amplitude L/R
   thrust, roll, yaw        log lum + high-pass    Worker / WebGPU          thrust, roll, yaw
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
| [`app/`](app/) | TypeScript web app: 3D scene, eye sampling, graph runtime, HUD (milestones 1–2 done) |
| [`data/`](data/) | Python extraction from the public MaleCNS tables → graph files, with reports of static checks |
| [`train/`](train/) | PyTorch side: flyvis parameter transfer, reference simulations, the DS trainer for the GPU box |

MaleCNS is **CC-BY**. Cite Berg et al., *Cell* 2026, and the [FlyEM project](https://www.janelia.org/project-team/flyem/male-cns-connectome).

## Explore now (no app yet)

1. Read the saved blog, then the headline results in [`research-notes.md`](research-notes.md).
2. Open the [Cell Type Explorer](https://reiserlab.github.io/celltype-explorer-drosophila-male-cns/) and look up **AOTU008** (dimorphic example from the blog) and **DNg02** (flight descending neurons).
3. Open the [Neuroglancer scene](https://neuroglancer-demo.appspot.com/#!gs://flyem-male-cns/v1.0/male-cns-v1.0.jso). Double-click a neuron, press `H` for controls.
4. [neuPrint](https://neuprint.janelia.org) dataset `male-cns:v1.0` — query upstream/downstream of a cell type (free account + API token).

Lightest file worth downloading later: `body-annotations-male-cns-v1.0-minconf-0.5.feather` (~13 MB) from the [download page](https://male-cns.janelia.org/download/). The 1.1 GB weight table is the first graph we would actually simulate. Do not pull the EM volume into this repo.

## Roadmap

**0. Research base** — this folder. Papers, Neuroglancer, what “flight circuit” means.

**1. Loop with a stub brain** — world, eye, and motor layers with a hand-written controller in place of the connectome.

- three.js scene, fly as a rigid body, stripe drum and a few obstacles
- two wide-field offscreen cameras (or a cube map) sampled at ommatidia directions into ~750 values per eye
- photoreceptor stage: log luminance + temporal high-pass
- wing amplitude L/R → thrust, roll, yaw → pose update
- done when: the stub controller holds heading against drum rotation at a stable frame budget
- **status: done.** Yaw rate follows the drum at ~70% of its speed, 115–120 fps.

**2. Half-real brain** — hand-coded motion detection feeds real wiring.

- Hassenstein–Reichardt detectors on neighbouring columns stand in for T4/T5
- their output is injected into the real HS, VS, and LPLC2 cells pulled from MaleCNS
- from there the actual graph: lobula plate → **DNg02** and neighbours → wing motor neurons + VNC interneurons
- static check first: DNg02 must receive lobula-plate input and project to wing/haltere motor neurons ([Namiki et al., 2022](https://pubmed.ncbi.nlm.nih.gov/35090590/)); if not, the extraction is wrong
- done when: visual rotation direction predicts the DNg02 left/right rate difference, and the closed loop stabilizes
- **status: done.** 1072 units, 26.5k edges. Findings: no direct LPTC → DNg02 synapses; the relay (PS080) is GABAergic so DNg02 needs a tonic flight-state drive; DNg02 → wing MN output is bilateral so the readout sits at DNg02. Details in [`data/README.md`](data/README.md) and [`app/README.md`](app/README.md).

**3. Real optic lobe** — replace the detectors with the male per-column graph.

- extract a subset of columnar types (L1, L2, Mi1, Tm3, Mi4, Mi9, T4a–d, T5a–d, Tm9, …) for every column: ~800 columns × ~20 types ≈ 16k units, ~1M edges
- write the same graph as a PyTorch model, initialize from [flyvis](https://github.com/TuragaLab/flyvis) parameters ([Lappalainen et al., 2024](https://www.nature.com/articles/s41586-024-07939-3)), train briefly on optic flow on the 5090
- trainable set stays small: per-type time constants and rest, one scale per synapse type; individual edges keep their connectome weight
- export params, swap into the browser runtime (Worker at ~100 Hz substeps, or WebGPU)
- done when: T4/T5 units are direction selective and milestone 2 still passes
- **status: done (via the lobula plate).** 65.8k units / 1.97M edges / 1,771 columns run in a Web Worker at 120 fps; retinotopy is calibrated from the wiring (T4 Mi9→Mi4 offsets). flyvis parameters transfer (synapse counts agree within ~30%) but give **zero** direction selectivity on the real per-cell wiring; 1,500 steps of fitting per-type tau/bias and per-pair strength on the 5090 (11 min) give mean DSI 0.47 and tuning correlation 0.96 across all 16 T4/T5 subtype × eye groups, with the correct preferred directions. In the browser the fly follows the drum in both directions with the turn read at the HS cells. Open: the HS → posterior slope → DNg02 hop is not yet calibrated in the big graph. Details in [`train/README.md`](train/README.md) and [`app/README.md`](app/README.md).

**4. Deciding where to go** — the known visual reflex pathways, chained.

- looming avoidance: LPLC2 / LC4 → escape descending neurons
- bar fixation / approach: LC10 and related types
- altitude and speed: ventral optic flow
- tune the few free gains on the GPU with the body simulated in PyTorch, against a behavioral objective; export, run in browser
- done when: the fly flies through the scene, avoids obstacles, and approaches a target using only rendered images

**5. Later** — richer biomechanics ([NeuroMechFly](https://github.com/NeLy-EPFL/NeuroMechFly), DeepMind [flybody](https://github.com/TuragaLab/flybody)), haltere feedback, walking, or the dimorphic courtship switches the *Cell* paper is actually about. Whole-brain graph controllers already exist in research ([FlyGM](https://arxiv.org/html/2602.17997v3)); this project stays small, inspectable, and in-browser.

## Compute

| Job | Where |
| --- | --- |
| three.js scene, eye sampling, graph inference, viewer | Mac, browser |
| neuPrint exploration, small feather files | Mac |
| filtering the 1.1 GB weight table, per-column adjacency | GPU box (RAM/disk), CPU work |
| flyvis-style parameter fitting, gain tuning | GPU box, RTX 5090 (32 GB) |

One graph schema (JSON: units, types, edges, weights, per-type params) is shared by the PyTorch trainer and the TypeScript runtime so both provably run the same network. The browser never trains.

## Known risks

- **Raw weights do not give sane dynamics.** Every published whole-graph model rescales globally and saturates. Expect to fit two or three global gains before anything works.
- **Motion detection needs temporal structure.** Uniform time constants give no direction selectivity; that is why milestone 3 trains per-type parameters instead of hoping.
- **Flight is not in the graph.** The wingbeat CPG, power-muscle mechanics, and haltere feedback are not in the EM volume. The body is a cartoon and stays one until milestone 5.
- **No emergent navigation.** Real goal-seeking needs the central complex, state, and odor. Expect reflexes (stabilize, avoid, approach); chained in a 3D world they already look like a fly deciding where to go.

## Neuron model (intended)

TypeScript, no Python in the hot loop.

```ts
// dx/dt = (-x + I_syn + I_ext) / tau
// I_syn = sum_j w_ji * f(x_j)
// w_ji  = synapse_count(j→i) * sign(transmitter_j)
// f     = ReLU or logistic
```

That is a cartoon of a neuron. It is enough to ask: does the published wiring turn a rendered image stream into a left/right wing command that keeps the fly in the air and away from walls? If yes, we made the insect fly in the only sense this repo claims. If no, the graph, the signs, the time constants, or the missing CPG is the next experiment.

## Related reading

- Berg et al., [*Sexual dimorphism in the complete connectome of the Drosophila male central nervous system*](https://doi.org/10.1016/j.cell.2026.08.015), *Cell* 2026
- Hoeller et al., visual pathways companion, *Cell* 2026 — [DOI](https://doi.org/10.1016/j.cell.2026.08.014)
- Namiki et al., [DNg02 flight descending neurons](https://pubmed.ncbi.nlm.nih.gov/35090590/), 2022
- Lappalainen et al., [Connectome-constrained networks predict neural activity across the fly visual system](https://www.nature.com/articles/s41586-024-07939-3), *Nature* 2024 — code: [flyvis](https://github.com/TuragaLab/flyvis)
- Shiu et al., [A Drosophila computational brain model reveals sensorimotor processing](https://www.nature.com/articles/s41586-024-07763-9), *Nature* 2024 — whole-connectome LIF simulation
- [Male CNS project](https://male-cns.janelia.org/) · [neuPrint](https://neuprint.janelia.org) · [Neuroglancer](https://github.com/google/neuroglancer)
