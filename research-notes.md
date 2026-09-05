# Research notes: Male *Drosophila* CNS connectome

Local briefing pulled from the [Google Research announcement](https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/) and official project pages.

## What was released

A fully proofread, annotated wiring diagram of an adult **male** fruit fly (*Drosophila melanogaster*) **central nervous system**: central brain + optic lobes + ventral nerve cord (the spinal-cord analog), with the neck connective intact.

That last point matters. Earlier fly maps were often brain-only, or one sex only. This one lets you follow a circuit from eyes/antennae all the way to motor neurons that drive the body.

| Source | Neurons | Synapses / connections | Cell types |
| --- | --- | --- | --- |
| Google Research blog | >166,000 | 125 million synaptic connections | — |
| *Cell* / bioRxiv abstract | 166,691 | — | 11,691 types |
| Cambridge / MRC LMB news | 166,700 | — | 11,710 types |
| Male CNS Cell Type Explorer (v1.0 snapshot) | 164,838 | 171,901,440 synapses | 11,751 types |

Numbers differ slightly by what is counted (glia filtered or not, confidence cutoff, brain-only vs full CNS). Treat **~166k neurons, ~11.7k types, hundreds of millions of synapses** as the working scale.

License: **CC-BY**. Dataset version: **male-cns:v1.0** (released 2026-06-08; paper published 2026-09-03).

## The paper

**Berg, S. et al.** *Sexual dimorphism in the complete connectome of the Drosophila male central nervous system.* *Cell* **189**(18): 5504–5526.e15 (2026).

- [Cell DOI](https://doi.org/10.1016/j.cell.2026.08.015)
- [Google Research publication page](https://research.google/pubs/sexual-dimorphism-in-the-complete-connectome-of-the-drosophila-male-central-nervous-system/)
- [bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2025.10.09.680999v2) (posted 2025-10-09, v2 2025-10-30)

Co-led by Greg Jefferis’s group (Cambridge Zoology / MRC LMB) with FlyEM at HHMI Janelia, Google Research, and others. Reconstruction used electron microscopy + AI segmentation, then the equivalent of **~44 years of human proofreading**.

### Headline scientific result

Most of the nervous system is **isomorphic** (same in both sexes). Sex differences are concentrated in higher brain centers, not in the sensory or motor periphery.

From the paper abstract / Janelia writeup (central-brain cross-match):

- **7,205** isomorphic types
- **114** dimorphic types (present in both sexes, wired or shaped differently)
- **262** male-specific types
- **69** female-specific types

That is only **~4.8% of male neurons** and **~2.4% of female neurons**, but those cells make enough extra connections that about **12% of male neurons** sit in circuits that differ by sex. Dimorphic / sex-specific neurons often express the sex-determination genes *fruitless* and *doublesex*. They form “hotspots” and **circuit switches** that reroute the same sensory input into antagonistic, sex-specific behaviors (e.g. male aggression vs female courtship in response to the same odor).

The blog’s example neuron, **AOTU008**, is dimorphic: the male version has two extra projections compared with the female (FlyWire) map.

## Companion papers (same week)

| Paper | Journal | DOI | Topic |
| --- | --- | --- | --- |
| Hoeller et al., *The organization of visual pathways in the Drosophila brain* | *Cell* 189(18): 5552–5570.e10 | [10.1016/j.cell.2026.08.014](https://doi.org/10.1016/j.cell.2026.08.014) | Vision |
| Tastekin et al., *The complete gustatory connectome of adult Drosophila…* | *Cell* 189(18): 5527–5551.e5 | [10.1016/j.cell.2026.08.016](https://doi.org/10.1016/j.cell.2026.08.016) | Taste → feeding / foraging / social |
| Rubin et al., *Networks of sexually dimorphic neurons that regulate social behaviors in Drosophila* | *Current Biology* | [10.1016/j.cub.2026.08.013](https://doi.org/10.1016/j.cub.2026.08.013) | Courtship / aggression circuits |

## How to look at the data

Official hub: [https://male-cns.janelia.org/](https://male-cns.janelia.org/)

| Tool | URL | Use for |
| --- | --- | --- |
| Cell Type Explorer | [reiserlab.github.io/celltype-explorer-drosophila-male-cns](https://reiserlab.github.io/celltype-explorer-drosophila-male-cns/) | Browse types, morphology, type-to-type connectivity |
| Dimorphism explorer | [male-cns.janelia.org dimorphism overview](https://male-cns.janelia.org/build/dimorphism_overview/#__tabbed_2_1) | Male vs female (FlyWire) comparison |
| neuPrint | [neuprint.janelia.org](https://neuprint.janelia.org) dataset `male-cns:v1.0` | Connectivity queries (needs a free account + API token) |
| Neuroglancer | [demo scene](https://neuroglancer-demo.appspot.com/#!gs://flyem-male-cns/v1.0/male-cns-v1.0.jso) | EM volume, segmentation, synapses, neuropil ROIs |
| Downloads | [male-cns.janelia.org/download](https://male-cns.janelia.org/download/) | Annotations, graphs, skeletons, volumes |
| NeuronBridge | Janelia NeuronBridge | Light-microscopy matches to FlyLight |
| Female map (FlyWire) | [flywire.ai](https://flywire.ai/) | The female brain used for comparison |

Python entry point (after creating a neuPrint token):

```python
from neuprint import Client, fetch_neurons, fetch_adjacencies

client = Client("https://neuprint.janelia.org", dataset="male-cns:v1.0", token=token)
neurons, syndist = fetch_neurons("DNge104")
outgoing, info = fetch_adjacencies("DNge104")
```

Bulk files live under `gs://flyem-male-cns/v1.0/` (Google Cloud Storage). The lighter annotation tables are tens of MB; the full synapse tables are multi-GB. EM + segmentation volumes are huge (the CLAHE JPEG volume is on the order of `94088 × 78317 × 134576` voxels at 8 nm).

Practical starting files from the download page:

- `body-annotations-male-cns-v1.0-minconf-0.5.feather` (~13 MB) — classes, types, sides
- `body-neurotransmitters-male-cns-v1.0.feather` (~42 MB)
- `connectome-weights-male-cns-v1.0-minconf-0.5.feather` (~1.1 GB) — segment-to-segment graph

Related Python/R stack: `neuprint-python`, [`navis`](https://navis.readthedocs.io/) + `navis-flybrains`, R `neuprintr` / natverse.

## How the map was made (Google’s role)

1. Slice the CNS into millions of thin sections.
2. Image each section with serial-section electron microscopy.
3. Align / stitch the stack.
4. Segment neurons with AI (historically **flood-filling networks**; current system **PATHFINDER**).
5. Detect synapses and assign cell types / *fruitless*–*doublesex* labels.
6. Human proofreading and annotation (the expensive step).

Google’s Connectomics team built Neuroglancer and much of the reconstruction stack. PATHFINDER was recently improved by training on **synthetic neurons** (MoGen), which cut merge errors and proofreading load.

Prior Google + Janelia fly maps:

- 2019 — automated full female brain reconstruction (shapes, not a finished connectome)
- 2020 — **hemibrain**: ~25,000 neurons, 21 million connections, human-verified

## Why this is a milestone

Connectomics lineage (from MRC LMB’s writeup):

1. 1986 — *C. elegans* (302 neurons), the first connectome
2. 2023 — *Drosophila* larva (~3,000 neurons)
3. 2024 — female adult *Drosophila* brain (FlyWire)
4. 2026 — male adult CNS, brain + nerve cord, synaptic-resolution sex comparison

It is the **largest brain map by neuron count** so far, and the first finished male fly CNS. Together with the female maps, it is the first synaptic-resolution, whole-brain male/female comparison in an adult animal with complex behavior.

Google’s longer-term path: zebrafish (structure + activity; Fire&Wire / ZAPBench), elephantnose fish cerebellum-like circuit (*Nature*, same week), then portions of mouse, eventually human-scale brains (~86 billion neurons — still out of reach).

## Press and project pages

- [University of Cambridge](https://www.cam.ac.uk/research/news/comparison-of-male-and-female-fly-brains-is-unlocking-the-secret-workings-of-the-mind)
- [MRC Laboratory of Molecular Biology](https://mrclmb.ac.uk/news-events/articles/first-complete-connectome-of-male-fly-central-nervous-system-allows-for-unprecedented-male-female-brain-comparison/)
- [Janelia project page](https://www.janelia.org/project-team/flyem/male-cns-connectome)
- [Google News post](https://blog.google/innovation-and-ai/technology/research/male-fruit-fly-brain-map/)
- [PATHFINDER / synthetic neurons blog](https://research.google/blog/ai-generated-synthetic-neurons-speed-up-brain-mapping/)

## Good first experiments for this repo

1. Open the [Cell Type Explorer](https://reiserlab.github.io/celltype-explorer-drosophila-male-cns/) and look up **AOTU008** (the blog’s dimorphic example) and a descending neuron such as **DNge104**.
2. Open the [Neuroglancer scene](https://neuroglancer-demo.appspot.com/#!gs://flyem-male-cns/v1.0/male-cns-v1.0.jso) and fly through EM + segmentation.
3. Query `male-cns:v1.0` on neuPrint for one cell type’s upstream/downstream partners.
4. Download the ~13 MB annotation feather file and plot type counts, *fruitless*/*doublesex* labels, or brain vs VNC membership — no huge volumes required.
