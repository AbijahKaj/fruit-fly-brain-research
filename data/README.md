# data — MaleCNS extraction

Turns the public MaleCNS v1.0 flat tables into the small graph JSON the app loads. Python only lives here; the browser never sees it.

```
cd data
uv venv && uv pip install pyarrow pandas numpy
# ~570 MB, public bucket, no account needed
curl -o raw/body-annotations.feather        https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather
curl -o raw/body-neurotransmitters.feather  https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-neurotransmitters-male-cns-v1.0.feather
curl -o raw/connectome-weights-traced.feather https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather
.venv/bin/python extract_flight.py
```

Output: `out/flight-v1.json` (also copied to `app/public/graphs/`) and `out/flight-v1.report.md` with the static sanity checks.

## flight-v1

| Role | What | Count |
| --- | --- | --- |
| input | lobula plate tangential cells (HS, VS, H2, CH) and looming LCs (LPLC2, LC4) | 359 |
| brain | one-hop bridges between inputs and DNs (posterior slope, etc.) | 291 |
| dn | DNg02 a–g and DNp01–06 | 41 |
| vnc | one-hop bridges between DNs and motor neurons | 298 |
| output | wing (`wm`) and haltere (`hm`) motor neurons | 83 |

Edges with ≥ 5 synapses, 26,544 total. Sign per unit from the consensus transmitter prediction: acetylcholine +1, GABA / glutamate / histamine −1, amines and unclear 0.

## Graph JSON

```
units: { count, bodyId[], type[], side[], role[], sign[], nt[] }
types: [{ name, superclass, count, nt, tau }]
edges: { count, pre[], post[], weight[] }     # weight = synapse count, unsigned
```

The TypeScript runtime builds compressed-sparse-row by post unit and multiplies each weight by `sign[pre]`. A PyTorch trainer can consume the same file; per-type `tau` is the first thing to fit.

## Findings from the extraction

- No direct LPTC → DNg02 synapses. The route is HS → posterior slope (PS080, PS126, PS311) → DNg02, and **PS080 is GABAergic**: left-eye progressive motion inhibits the right DNg02. That is why the brain model needs a tonic flight-state drive on DNg02.
- DNg02 → wing MN output is bilateral (roughly equal synapse weight to left and right MNs, directly and via VNC interneurons). A rate readout at the MNs is therefore not lateralized; the readout sits at DNg02.
- LC4 and LPLC2 hit DNp01–06 directly with thousands of synapses, ready for the looming reflex in milestone 4.

## optic-v2 (milestone 3)

`extract_v2.py` adds the per-column optic lobe: every flyvis cell type that exists in MaleCNS (51 types), both eyes, plus the lobula plate (LPi, HS, VS, H2, CH) and the flight-v1 seeds and bridges. 65,799 units, 1.97M edges (≥ 2 synapses in the optic lobe, ≥ 5 elsewhere), 1,771 columns. Binary format from `graphio.py`: a JSON header with typed-array descriptors and a `.bin` (25 MB).

- **Columns.** Lamina and medulla types carry `assignedOlHex1/2` in the annotations (~890 columns per side). T4, T5, Tm3 and the rest take the synapse-weighted mode of their column-assigned partners.
- **Retinotopy from the wiring.** For each T4 subtype the offset from its Mi9 input centroid to its Mi4 input centroid in hex space is its preferred direction. T4a (front-to-back) and T4c (upward) fix a rotation and mirror per eye; the two axes come out 81–82° apart on both sides. Columns are placed 5° apart, centred at ±80° azimuth.
- Report: `out/optic-v2.report.md`.
