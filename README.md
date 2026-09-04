# Fruit fly brain

A workspace for understanding the 2026 male *Drosophila* connectome, playing with its public data in the browser, and eventually **simulating enough of the fly’s nervous system in TypeScript to make it fly**.

The map is not a mind. It is a wiring diagram. The bet of this project is that a small, well-chosen subgraph — vision in, wing motor out — is enough to start, and that a web app is the right place to see that graph think.

## Goal

1. **Understand the research** — what was mapped, why the male CNS (brain + nerve cord) is a milestone, where male and female flies actually differ.
2. **Play with the data** — a web app that loads real MaleCNS cells and synapses, not screenshots.
3. **Model the network in TypeScript** — leaky neurons + weighted edges from the connectome, running in the browser.
4. **Close the loop on flight** — visual motion in, descending neurons, wing motor out, a fly that banks and holds altitude.

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

```
  eyes / optic flow          TypeScript graph              wings / body
  ----------------     ----------------------------     -----------------
  synthetic visual  →  optic-lobe cells (subset)    →   DNg02 / other DNs
  or webcam motion     leaky-integrator units            wing motor neurons
                       edges = synapse weights           beat amplitude, roll

                       ↑
                       MaleCNS v1.0 via neuPrint / feather tables

  Neuroglancer  =  look at the real cells in EM
  This webapp   =  run and poke the graph
```

[Neuroglancer](https://github.com/google/neuroglancer) is Google’s TypeScript/WebGL viewer for the EM volume, meshes, and skeletons. Use it to *see* neurons. Notes and controls: [`sources/neuroglancer.md`](sources/neuroglancer.md). Official tutorial: [connectomics.readthedocs.io — Neuroglancer](https://connectomics.readthedocs.io/en/latest/external/neuroglancer.html).

Hosted MaleCNS scene (Chrome):

[neuroglancer-demo.appspot.com — male-cns-v1.0](https://neuroglancer-demo.appspot.com/#!gs://flyem-male-cns/v1.0/male-cns-v1.0.jso)

## Repo

| Path | What |
| --- | --- |
| [`a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain.md`](a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain.md) | Saved [Google Research blog](https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/) (2026-09-03) |
| [`research-notes.md`](research-notes.md) | Paper, numbers, companion papers, download URLs |
| [`sources/neuroglancer.md`](sources/neuroglancer.md) | How to use Neuroglancer on this data |
| `app/` | Planned TypeScript web app (viewer + simulator) |
| `data/` | Planned local subsets (annotations, flight subgraph — not the EM volume) |

MaleCNS is **CC-BY**. Cite Berg et al., *Cell* 2026, and the [FlyEM project](https://www.janelia.org/project-team/flyem/male-cns-connectome).

## Explore now (no app yet)

1. Read the saved blog, then the headline results in [`research-notes.md`](research-notes.md).
2. Open the [Cell Type Explorer](https://reiserlab.github.io/celltype-explorer-drosophila-male-cns/) and look up **AOTU008** (dimorphic example from the blog) and **DNg02** (flight descending neurons).
3. Open the [Neuroglancer scene](https://neuroglancer-demo.appspot.com/#!gs://flyem-male-cns/v1.0/male-cns-v1.0.jso). Double-click a neuron, press `H` for controls.
4. [neuPrint](https://neuprint.janelia.org) dataset `male-cns:v1.0` — query upstream/downstream of a cell type (free account + API token).

Lightest file worth downloading later: `body-annotations-male-cns-v1.0-minconf-0.5.feather` (~13 MB) from the [download page](https://male-cns.janelia.org/download/). The 1.1 GB weight table is the first graph we would actually simulate. Do not pull the EM volume into this repo.

## Roadmap

**0. Research base** — this folder. Papers, Neuroglancer, what “flight circuit” means.

**1. Flight subgraph** — from neuPrint / feather files, extract:

- a thin visual-motion path (lobula / lobula plate → visual projection neurons)
- **DNg02** (and a few neighbor DN types if they dominate the same wing neuropil)
- wing motor neurons and a handful of VNC interneurons

Target size: hundreds to a few thousand units, not 166k. Export JSON the browser can load.

**2. Web app** — TypeScript. Browse the subgraph, color by type / side / transmitter, click a cell to see partners, deep-link that cell into Neuroglancer.

**3. Dynamics** — each neuron is a leaky integrator (or rate unit). Synapse count × sign (from transmitter predictions) is the weight. Step the graph at ~1 kHz in a Web Worker. Visualize activity on the same graph.

**4. Fly** — map DNg02 / motor output onto left/right wingbeat amplitude (the population-code result). Drive a simple 3D fly: thrust, roll, yaw. First demo is optomotor: a moving stripe world, the model turns with it.

**5. Later** — richer biomechanics ([NeuroMechFly](https://github.com/NeLy-EPFL/NeuroMechFly), DeepMind [flybody](https://github.com/TuragaLab/flybody)), walking, or the dimorphic courtship switches the *Cell* paper is actually about. Whole-brain graph controllers already exist in research ([FlyGM](https://arxiv.org/html/2602.17997v3)); this project stays small, inspectable, and in-browser.

## Neuron model (intended)

TypeScript, no Python in the hot loop.

```ts
// dx/dt = (-x + I_syn + I_ext) / tau
// I_syn = sum_j w_ji * f(x_j)
// w_ji  = synapse_count(j→i) * sign(transmitter_j)
// f     = ReLU or logistic
```

That is a cartoon of a neuron. It is enough to ask: does the published wiring turn optic flow into a left/right wing command? If yes, we made the insect fly in the only sense this repo claims. If no, the graph, the signs, or the missing CPG is the next experiment.

## Related reading

- Berg et al., [*Sexual dimorphism in the complete connectome of the Drosophila male central nervous system*](https://doi.org/10.1016/j.cell.2026.08.015), *Cell* 2026
- Hoeller et al., visual pathways companion, *Cell* 2026 — [DOI](https://doi.org/10.1016/j.cell.2026.08.014)
- Namiki et al., [DNg02 flight descending neurons](https://pubmed.ncbi.nlm.nih.gov/35090590/), 2022
- [Male CNS project](https://male-cns.janelia.org/) · [neuPrint](https://neuprint.janelia.org) · [Neuroglancer](https://github.com/google/neuroglancer)
