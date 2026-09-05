# Data license

The code in this repository is under the MIT license (`LICENSE`). The data files below are
derived works and carry the licenses of their sources.

## Derived from MaleCNS (CC BY 4.0)

- `app/public/graphs/optic-v2.json`, `app/public/graphs/optic-v2.bin`: the optic-lobe-to-wing-motor
  graph cut from the public MaleCNS v1.0 tables (cell annotations, predicted transmitters, synapse
  counts) by the scripts in `data/`.
- `app/public/graphs/fitted-params.json`: per-type and per-type-pair parameters fitted on that graph
  by the scripts in `train/`. The parameters are meaningful only together with the graph.
- `data/out/`, `train/out/`, `train/data/`: intermediate and derived tables of the same origin.

These files are released under [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/),
the license of the source data. Reuse must credit:

> Berg, S. et al. (2026). *A connectome of the male Drosophila melanogaster central nervous system.*
> Cell. FlyEM Project Team, HHMI Janelia Research Campus and Google Research.
> https://male-cns.janelia.org/

## Derived from flyvis

- `app/public/graphs/flyvis-params.json`: per-type time constants, biases and resting levels
  transferred from the released flyvis models (TuragaLab/flyvis, MIT license). Credit:

> Lappalainen, J. K. et al. (2024). *Connectome-constrained networks predict neural activity across
> the fly visual system.* Nature. https://github.com/TuragaLab/flyvis
