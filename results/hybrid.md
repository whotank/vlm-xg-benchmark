# Hybrid: the models read the geometry, the map-only logistic model scores it

Each model reads three numbers off the same image as the zero-shot runs (distance to the centre of the goal, the angle between the posts, opponents in the shooting triangle, goalkeeper included; scale given in the prompt from the pitch markings; no xG asked). The map-only logistic model scores the readings with the fold model that never saw the shot's match, exactly as it scores the true values. Brackets: 95% match-clustered bootstrap intervals; ΔBrier paired against the map-only baseline on the same shots.

## How well do the models read the image?

Error = reading − true value. Opponents: share read exactly, and within ±1.

| Model | Band | n | Distance bias (yd) | Distance MAE | Distance median AE | r | Angle bias (°) | Angle MAE (°) | Opponents exact | within ±1 | Opponents bias |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Claude Opus 5.5 | all | 298 | -0.09 | 0.12 | 0.11 | 1.000 | +0.0 | 0.4 | 83% | 99% | -0.01 |
| Claude Opus 5.5 | <6 yd | 8 | -0.07 | 0.15 | 0.15 | | +0.8 | 3.2 | 88% | 100% | +0.12 |
| Claude Opus 5.5 | 6–12 yd | 77 | -0.06 | 0.11 | 0.10 | | +0.0 | 0.8 | 82% | 99% | -0.04 |
| Claude Opus 5.5 | 12–18 yd | 79 | -0.08 | 0.12 | 0.12 | | +0.1 | 0.2 | 82% | 100% | +0.10 |
| Claude Opus 5.5 | 18–25 yd | 69 | -0.09 | 0.11 | 0.09 | | -0.0 | 0.2 | 86% | 100% | -0.06 |
| Claude Opus 5.5 | 25+ yd | 65 | -0.11 | 0.13 | 0.13 | | +0.0 | 0.1 | 83% | 97% | -0.08 |
| GPT-6.2 Astra | all | 298 | -0.00 | 0.07 | 0.06 | 1.000 | +0.0 | 0.2 | 91% | 100% | +0.02 |
| GPT-6.2 Astra | <6 yd | 8 | +0.00 | 0.10 | 0.11 | | -0.2 | 1.4 | 100% | 100% | +0.00 |
| GPT-6.2 Astra | 6–12 yd | 77 | -0.00 | 0.07 | 0.06 | | +0.1 | 0.4 | 88% | 100% | +0.01 |
| GPT-6.2 Astra | 12–18 yd | 79 | -0.01 | 0.07 | 0.05 | | +0.0 | 0.1 | 92% | 100% | +0.05 |
| GPT-6.2 Astra | 18–25 yd | 69 | +0.00 | 0.07 | 0.05 | | +0.0 | 0.1 | 88% | 100% | +0.03 |
| GPT-6.2 Astra | 25+ yd | 65 | +0.01 | 0.07 | 0.06 | | +0.0 | 0.0 | 92% | 100% | -0.02 |

## xG from the readings

| Model | Scored by | AUC | Brier | Mean xG | ΔBrier vs map-only | p |
|---|---|---|---|---|---|---|
| Claude Opus 5.5 | Map-only baseline (true geometry) | 0.721 [0.636, 0.808] | 0.110 [0.078, 0.147] | 0.111 |  | |
| Claude Opus 5.5 | Hybrid: all three read by the model | 0.695 [0.603, 0.788] | 0.112 [0.079, 0.150] | 0.111 | +0.002 [−0.000, +0.004] | 0.071 |
| Claude Opus 5.5 | Hybrid: distance read, rest true | 0.720 [0.637, 0.807] | 0.110 [0.078, 0.147] | 0.112 | −0.000 [−0.000, +0.000] | 0.511 |
| Claude Opus 5.5 | Hybrid: angle read, rest true | 0.721 [0.636, 0.807] | 0.110 [0.078, 0.147] | 0.111 | +0.000 [−0.000, +0.000] | 0.919 |
| Claude Opus 5.5 | Hybrid: opponents read, rest true | 0.696 [0.603, 0.789] | 0.112 [0.079, 0.149] | 0.110 | +0.002 [−0.000, +0.003] | 0.053 |
| Claude Opus 5.5 | Direct: the model's own xG (zero-shot, image) | 0.721 [0.624, 0.808] | 0.111 [0.079, 0.147] | 0.109 | +0.001 [−0.006, +0.008] | 0.820 |
| Claude Opus 5.5 | StatsBomb xG | 0.792 [0.704, 0.867] | 0.100 [0.069, 0.136] | 0.100 | −0.011 [−0.018, −0.003] | 0.003 |
| GPT-6.2 Astra | Map-only baseline (true geometry) | 0.721 [0.636, 0.808] | 0.110 [0.078, 0.147] | 0.111 |  | |
| GPT-6.2 Astra | Hybrid: all three read by the model | 0.708 [0.622, 0.796] | 0.111 [0.078, 0.148] | 0.110 | +0.001 [−0.000, +0.002] | 0.147 |
| GPT-6.2 Astra | Hybrid: distance read, rest true | 0.720 [0.636, 0.807] | 0.110 [0.078, 0.147] | 0.111 | +0.000 [+0.000, +0.000] | 0.031 |
| GPT-6.2 Astra | Hybrid: angle read, rest true | 0.720 [0.636, 0.807] | 0.110 [0.078, 0.147] | 0.111 | +0.000 [−0.000, +0.000] | 0.124 |
| GPT-6.2 Astra | Hybrid: opponents read, rest true | 0.709 [0.623, 0.797] | 0.111 [0.078, 0.148] | 0.110 | +0.000 [−0.000, +0.002] | 0.334 |
| GPT-6.2 Astra | Direct: the model's own xG (zero-shot, image) | 0.733 [0.642, 0.816] | 0.108 [0.080, 0.139] | 0.148 | −0.002 [−0.011, +0.006] | 0.613 |
| GPT-6.2 Astra | StatsBomb xG | 0.792 [0.704, 0.867] | 0.100 [0.069, 0.136] | 0.100 | −0.011 [−0.018, −0.003] | 0.003 |

## Decomposition of each model's Brier gap to the map-only baseline

Direct − map-only = (hybrid − map-only: misreading the image) + (direct − hybrid: turning what it reads into a probability).

| Model | n | Direct − map-only | p | Misreading: hybrid − map-only | p | Judgment: direct − hybrid | p |
|---|---|---|---|---|---|---|---|
| Claude Opus 5.5 | 298 | +0.001 [−0.006, +0.008] | 0.820 | +0.002 [−0.000, +0.004] | 0.071 | −0.001 [−0.007, +0.005] | 0.792 |
| GPT-6.2 Astra | 298 | −0.002 [−0.011, +0.006] | 0.613 | +0.001 [−0.000, +0.002] | 0.147 | −0.003 [−0.012, +0.005] | 0.504 |

## Mean xG by distance band

| Model | Band (yd) | n | Observed | StatsBomb | Map-only (true geometry) | Hybrid (model's readings) | Direct (model's xG) |
|---|---|---|---|---|---|---|---|
| Claude Opus 5.5 | <6 | 8 | 0.50 | 0.480 | 0.528 | 0.521 | 0.351 |
| Claude Opus 5.5 | 6–12 | 77 | 0.18 | 0.168 | 0.204 | 0.204 | 0.221 |
| Claude Opus 5.5 | 12–18 | 79 | 0.23 | 0.103 | 0.101 | 0.098 | 0.102 |
| Claude Opus 5.5 | 18–25 | 69 | 0.06 | 0.049 | 0.051 | 0.053 | 0.041 |
| Claude Opus 5.5 | 25+ | 65 | 0.03 | 0.024 | 0.027 | 0.027 | 0.029 |
| GPT-6.2 Astra | <6 | 8 | 0.50 | 0.480 | 0.528 | 0.527 | 0.547 |
| GPT-6.2 Astra | 6–12 | 77 | 0.18 | 0.168 | 0.204 | 0.202 | 0.298 |
| GPT-6.2 Astra | 12–18 | 79 | 0.23 | 0.103 | 0.101 | 0.100 | 0.141 |
| GPT-6.2 Astra | 18–25 | 69 | 0.06 | 0.049 | 0.051 | 0.051 | 0.055 |
| GPT-6.2 Astra | 25+ | 65 | 0.03 | 0.024 | 0.027 | 0.026 | 0.029 |

## Reading

- Claude Opus 5.5 (298 shots): distance read with a mean absolute error of 0.12 yd (bias at 6–12 yd -0.06 yd), angle 0.4°, opponents exact in 83%. Scored by the logistic model, the readings cost +0.0016 Brier against the true geometry (not significant; AUC 0.695 vs 0.721). The model's own xG differs from the hybrid by -0.0008 (not significant; AUC 0.721). Per shot, distance misreading and the direct xG's gap to the map-only baseline correlate at ρ = -0.10 (6–12 yd: -0.22).
- GPT-6.2 Astra (298 shots): distance read with a mean absolute error of 0.07 yd (bias at 6–12 yd -0.00 yd), angle 0.2°, opponents exact in 91%. Scored by the logistic model, the readings cost +0.0007 Brier against the true geometry (not significant; AUC 0.708 vs 0.721). The model's own xG differs from the hybrid by -0.0028 (not significant; AUC 0.733). Per shot, distance misreading and the direct xG's gap to the map-only baseline correlate at ρ = -0.14 (6–12 yd: -0.16).

Caveat: the readings come from a separate request that asks only for measurements; in the xG request the model may look less carefully (it thinks far less there: results/cost_summary.md). The hybrid shows what the model can read off the image, not what it used when it gave its xG.
