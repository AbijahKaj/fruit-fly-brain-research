# train — parameter fitting (GPU box)

The browser runs the graph; this fits the numbers it runs with.

```
cd train
uv venv && uv pip install torch numpy scipy flyvis
.venv/bin/flyvis download-pretrained --skip_large_files   # ~6 MB, Lappalainen et al. 2024 models
.venv/bin/python dump_flyvis_params.py 1                  # -> out/flyvis-params.json (ONE model; averaging the ensemble breaks it)
.venv/bin/python simulate_ours.py                          # steady state of our graph vs the flyvis reference, per type
.venv/bin/python train_optic.py --device cuda --steps 2000 # -> out/fitted-params.json
cp out/fitted-params.json ../app/public/graphs/            # the app prefers it over flyvis-params.json
```

Smoke test on a laptop: `train_optic.py --device cpu --steps 3 --max-el 12 --batch 2 --T 40` (13k units, ~2 s/step).

| File | What |
| --- | --- |
| `dump_flyvis_params.py` | Per-type tau and bias, per-pair strength and sign, photoreceptor→lamina weights, from one pretrained flyvis model |
| `flyvis_reference.py` | flyvis steady state under grey per type (`out/flyvis-reference.json`); also stores the resting membrane values the browser's homeostat targets |
| `flyvis_grating.py` | flyvis central-column responses to a drifting grating, the DS magnitude to expect |
| `simulate_ours.py` | Our graph + flyvis params in numpy: steady state per type next to the reference, L1 input decomposition |
| `graph_torch.py` | Binary graph loader; `FlyvisModel`: sparse rate model with trainable per-type tau/bias and per-pair strength |
| `train_optic.py` | Stage 1: drifting gratings on the real column lattice → DS objective for T4/T5 a–d per eye → `fitted-params.json` |
| `train_loom.py` | Stage 2 (`--joint`): looming / receding / translating discs, static gratings, and the stage 1 objective together → LC4/LPLC2 selectivity, quiet T4/T5 under static contrast. Stage 3 (`--hs`, experimental): HS cells on ray-cast self-motion flow fields. `Stimuli` class shared with the tools below |
| `eval_loom.py`, `diag_loom.py`, `eval_ds.py` | Selectivity tables per stimulus kind; per-pair presynaptic drive onto the LC cells; DSI over 8 directions |

## Fit result (fit2, RTX 5090)

`train_optic.py --steps 1500 --batch 16 --T 100 --reg 0.3 --loss corr`, 11 minutes, 22 GB. Log in `runs/fit2.log`. `eval_ds.py` over 8 grating directions:

| | flyvis init | fitted |
| --- | --- | --- |
| mean DSI (16 groups) | 0.00 | 0.47 |
| mean tuning correlation with cos(θ − PD) | 0.09 | 0.96 |

Every T4/T5 subtype on both eyes peaks at its expected direction (T4a right eye at front-to-back, left eye mirrored; T4c up; T4d down). The first loss (normalised MSE to 1 + cos) sat at the trivial flat solution; the correlation loss with a modulation-depth term is what worked.

## Looming fit, RTX 5090

Runs are in `runs/loom*.log`. Sequence, each starting from the previous stage's parameters:

| run | what | outcome |
| --- | --- | --- |
| loom1–3 | only LC input pairs + LC tau/bias trainable, population-mean readout, l/v 20–80 ms | no selectivity: the presynaptic drive onto LC4/LPLC2 is the same for every stimulus (see `diag_loom.py`); the dark disc even *lowers* the ON-pathway drive, and with fixed synapse signs nothing can go up |
| loom4 | `--joint`: everything trainable, stage 1 grating loss kept, top-5 readout, slower l/v (0.1–0.4 s) | selectivity 0.98–1.0, DSI up to 0.74; but in the scene hundreds of T4/T5 sat at the ceiling at rest |
| loom5 | + static-grating control (T4/T5 penalised above 0.15 under static contrast), object trials over structured backgrounds | T4/T5 quiet under static patterns; LC selectivity kept |
| loom6 | + frontal loom positions (az 5–110°), L/R symmetry term on T4/T5 | selectivity 0.97–1.0, DSI 0.66, corr 0.96, eyes within ~30% |
| loom7 | looms scored while the disc is 8–35° wide (was the last 40% of the trial, 30–70°), HS frozen (`--freeze`), on top of hs7 | shipped: selectivity 0.89–0.97, DSI 0.67; in the scene the LC response starts at ~2 units instead of ~0.8, and a hovering fly sideslips out of the way |

`train_loom.py --device cuda --steps 600 --T 160 --joint --ds-batch 4`, 6.5 minutes, 27 GB (two forward passes keep every per-edge product for the backward pass; `--T 200` with a batch of 8 gratings went out of memory).

Lessons: score the best-placed cells (top-k), not the population mean, because only cells whose receptive field sits at the loom centre see all four expanding edges; train with static controls, otherwise motion detectors learn to answer contrast; put stimuli where the readout will need them (frontal looms).

### HS stage (hs1–hs7)

Goal: HS_L − HS_R should read yaw rotation and not the translation flow of forward flight, which in the scene turned the optomotor loop into a bias toward nearby pillars.

- In `optic-v2` the HS cells receive only ipsilateral input (T4a, T5a, LPi21, TmY5a; no H2 or other contralateral partner), so a rotation-selective single HS cell is not expressible. The graph does support the classic bidirectional HS (up for progressive, below rest for regressive motion via LPi21), which is what `--hs` fits.
- hs1–hs2: sparse dot flow fields did not drive the fitted T4/T5 at all (0.11 vs 0.10 static). hs3–hs4: a ray-cast striped wall plus checkered ground does (T4a 1.35 for the preferred rotation vs 0.18 static) and the fit converged, but did not transfer: in the scene every drum direction produced a left turn.
- hs5: **stimuli recorded from the scene** (`--scene data/scene-episodes.json`, from `fly.record` in the app; `scene_episodes.py` replays them through the browser's photoreceptor pipeline). Transferred halfway: directions right, loop still oscillating.
- The trainer and the browser were not running the same network: the browser scales inputs onto pooling cells (LPTCs, LPi, LCs) by `lptcScale` = 0.001 and gives them a homeostatic bias; the trainer scaled them by 0.02 with no bias. LPi21 feeds both HS and T4/T5. `graph_torch.py` now uses the pooling scale, and every pooling type flyvis does not cover gets a homeostat-style bias in the trainer (rest membrane 0.3 under grey) that is exported; the browser treats those types as fitted and runs no homeostat at all. Looming selectivity and DS were unchanged by the switch.
- hs6 on the matched model: rotation separation 1.2–1.4, translation 0.05–0.13, rest 0.82/0.82. In the scene the right HS rest still swung from 0 to 0.8 with the view (the constraint held only on average over two static episodes).
- hs7: 16 static views, rest ≥ 0.6 and sides matched on every one. Shipped. In the scene: rest 0.9 / 0.5–0.7 across views, drum followed in all four directions, cruise drift 10–50° over 7 s (from 110–135°), pillar course 45 s with no collision.

`train_loom.py --device cuda --steps 600 --T 160 --lr 0.02 --hs --scene data/scene-episodes.json --params out/fitted-loom6.json`, 4.5 minutes. The recording (32 MB) is not in git; re-record with `fly.record` (see `app/README.md`).

## What we learned transferring flyvis

- **Synapse counts agree.** Per-pair totals in MaleCNS are within ~30% of the flyvis (hemibrain-era) table, so per-pair strengths transfer.
- **Use one model, not the ensemble mean.** Averaged parameters are not a solution of anything; the network runs away.
- **Photoreceptor activity is 0.78 + stimulus**, not `stimulus + bias`: L2/L4/Am feedback adds ~1. Measured from the steady state and stored as `restOffset` / `stimGain`.
- **Light adaptation is required.** flyvis trains around grey 0.5; a rendered scene averages ~0.2, below the photoreceptor threshold. The browser divides each column by its slow running mean.
- **The resting point matters more than the taus.** L1/L2 rest silent (membrane −1.8), Mi4 at 1.6, T5 at 0. The browser's homeostat now targets flyvis's resting membrane value per type.
- **Still missing** in our graph: CT1 (the big GABAergic input to T4/T5, one cell per side, not splittable per column without synapse positions), Am, R7/R8, Mi3, Mi11, Mi12, Tm28. The fit has to absorb their absence.
