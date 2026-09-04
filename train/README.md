# train — parameter fitting (GPU box)

Python side of milestone 3. The browser runs the graph; this fits the numbers it runs with.

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
| `train_optic.py` | Drifting gratings on the real column lattice → DS objective for T4/T5 a–d per eye → `fitted-params.json` |

## Fit result (fit2, RTX 5090)

`train_optic.py --steps 1500 --batch 16 --T 100 --reg 0.3 --loss corr`, 11 minutes, 22 GB. Log in `runs/fit2.log`. `eval_ds.py` over 8 grating directions:

| | flyvis init | fitted |
| --- | --- | --- |
| mean DSI (16 groups) | 0.00 | 0.47 |
| mean tuning correlation with cos(θ − PD) | 0.09 | 0.96 |

Every T4/T5 subtype on both eyes peaks at its expected direction (T4a right eye at front-to-back, left eye mirrored; T4c up; T4d down). The first loss (normalised MSE to 1 + cos) sat at the trivial flat solution; the correlation loss with a modulation-depth term is what worked.

## What we learned transferring flyvis

- **Synapse counts agree.** Per-pair totals in MaleCNS are within ~30% of the flyvis (hemibrain-era) table, so per-pair strengths transfer.
- **Use one model, not the ensemble mean.** Averaged parameters are not a solution of anything; the network runs away.
- **Photoreceptor activity is 0.78 + stimulus**, not `stimulus + bias`: L2/L4/Am feedback adds ~1. Measured from the steady state and stored as `restOffset` / `stimGain`.
- **Light adaptation is required.** flyvis trains around grey 0.5; a rendered scene averages ~0.2, below the photoreceptor threshold. The browser divides each column by its slow running mean.
- **The resting point matters more than the taus.** L1/L2 rest silent (membrane −1.8), Mi4 at 1.6, T5 at 0. The browser's homeostat now targets flyvis's resting membrane value per type.
- **Still missing** in our graph: CT1 (the big GABAergic input to T4/T5, one cell per side, not splittable per column without synapse positions), Am, R7/R8, Mi3, Mi11, Mi12, Tm28. The fit has to absorb their absence.
