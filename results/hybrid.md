# Hybrid: the models read the geometry, the map-only logistic model scores it

Each model reads three numbers off the same image as the zero-shot runs (distance to the centre of the goal, the angle between the posts, opponents in the shooting triangle, goalkeeper included; scale given in the prompt from the pitch markings; no xG asked). The map-only logistic model scores the readings with the fold model that never saw the shot's match, exactly as it scores the true values. Only perception runs that finished are used; Gemini 4 Carbon measured the first 150 shots of the 300 (cost), the others all 300. Brackets: 95% match-clustered bootstrap intervals; ΔBrier paired against the map-only baseline on the same shots.

Opus's "Direct" row is its original zero-shot run, whose images were rendered in the authors' environment; "Direct, rerun" is the same zero-shot setting queried again (`claude_opus_5_5_rerun`) with images rendered in this environment, like the images the readings came from. Carbon's and GPT's zero-shot runs were already rendered here.

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
| Gemini 4 Carbon | all | 150 | -0.03 | 0.13 | 0.10 | 1.000 | +0.0 | 0.3 | 83% | 100% | +0.11 |
| Gemini 4 Carbon | <6 yd | 6 | +0.00 | 0.10 | 0.10 | | -0.4 | 2.2 | 83% | 100% | +0.17 |
| Gemini 4 Carbon | 6–12 yd | 41 | -0.02 | 0.10 | 0.09 | | +0.0 | 0.5 | 90% | 100% | +0.10 |
| Gemini 4 Carbon | 12–18 yd | 42 | +0.02 | 0.15 | 0.11 | | +0.0 | 0.3 | 83% | 100% | +0.12 |
| Gemini 4 Carbon | 18–25 yd | 30 | -0.05 | 0.13 | 0.09 | | +0.0 | 0.1 | 73% | 100% | +0.20 |
| Gemini 4 Carbon | 25+ yd | 31 | -0.10 | 0.13 | 0.11 | | +0.0 | 0.1 | 84% | 100% | +0.03 |
| GPT-6.2 Astra | all | 298 | -0.00 | 0.07 | 0.06 | 1.000 | +0.0 | 0.2 | 91% | 100% | +0.02 |
| GPT-6.2 Astra | <6 yd | 8 | +0.00 | 0.10 | 0.11 | | -0.2 | 1.4 | 100% | 100% | +0.00 |
| GPT-6.2 Astra | 6–12 yd | 77 | -0.00 | 0.07 | 0.06 | | +0.1 | 0.4 | 88% | 100% | +0.01 |
| GPT-6.2 Astra | 12–18 yd | 79 | -0.01 | 0.07 | 0.05 | | +0.0 | 0.1 | 92% | 100% | +0.05 |
| GPT-6.2 Astra | 18–25 yd | 69 | +0.00 | 0.07 | 0.05 | | +0.0 | 0.1 | 88% | 100% | +0.03 |
| GPT-6.2 Astra | 25+ yd | 65 | +0.01 | 0.07 | 0.06 | | +0.0 | 0.0 | 92% | 100% | -0.02 |

## Thinking per shot: measuring vs estimating xG

Thinking tokens per answered call, from each run's `calls.jsonl` (the xG run is the zero-shot run rendered in the same environment).

| Model | Measuring: mean | median | max | cost per shot | xG request: mean thinking tokens |
|---|---|---|---|---|---|
| Claude Opus 5.5 | 568 | 564 | 909 | $0.0148 | 159 |
| Gemini 4 Carbon | 3,718 | 2,102 | 25,622 | $0.0403 | 107 |
| GPT-6.2 Astra | 196 | 151 | 936 | $0.0131 | 25 |

## xG from the readings

| Model | Scored by | AUC | Brier | Mean xG | ΔBrier vs map-only | p |
|---|---|---|---|---|---|---|
| Claude Opus 5.5 | Map-only baseline (true geometry) | 0.721 [0.636, 0.808] | 0.110 [0.078, 0.147] | 0.111 |  | |
| Claude Opus 5.5 | Hybrid: all three read by the model | 0.695 [0.603, 0.788] | 0.112 [0.079, 0.150] | 0.111 | +0.002 [−0.000, +0.004] | 0.071 |
| Claude Opus 5.5 | Hybrid: distance read, rest true | 0.720 [0.637, 0.807] | 0.110 [0.078, 0.147] | 0.112 | −0.000 [−0.000, +0.000] | 0.511 |
| Claude Opus 5.5 | Hybrid: angle read, rest true | 0.721 [0.636, 0.807] | 0.110 [0.078, 0.147] | 0.111 | +0.000 [−0.000, +0.000] | 0.919 |
| Claude Opus 5.5 | Hybrid: opponents read, rest true | 0.696 [0.603, 0.789] | 0.112 [0.079, 0.149] | 0.110 | +0.002 [−0.000, +0.003] | 0.053 |
| Claude Opus 5.5 | Direct: the model's own xG (zero-shot, image) | 0.721 [0.624, 0.808] | 0.111 [0.079, 0.147] | 0.109 | +0.001 [−0.006, +0.008] | 0.820 |
| Claude Opus 5.5 | Direct, rerun (images rendered like the readings') | 0.717 [0.630, 0.800] | 0.113 [0.080, 0.150] | 0.110 | +0.003 [−0.003, +0.008] | 0.294 |
| Claude Opus 5.5 | StatsBomb xG | 0.792 [0.704, 0.867] | 0.100 [0.069, 0.136] | 0.100 | −0.011 [−0.018, −0.003] | 0.003 |
| Gemini 4 Carbon | Map-only baseline (true geometry) | 0.717 [0.587, 0.841] | 0.114 [0.078, 0.157] | 0.117 |  | |
| Gemini 4 Carbon | Hybrid: all three read by the model | 0.697 [0.562, 0.826] | 0.116 [0.079, 0.161] | 0.112 | +0.002 [−0.000, +0.005] | 0.152 |
| Gemini 4 Carbon | Hybrid: distance read, rest true | 0.717 [0.590, 0.840] | 0.114 [0.078, 0.157] | 0.117 | +0.000 [−0.000, +0.000] | 0.367 |
| Gemini 4 Carbon | Hybrid: angle read, rest true | 0.718 [0.587, 0.841] | 0.114 [0.078, 0.157] | 0.117 | −0.000 [−0.000, +0.000] | 0.708 |
| Gemini 4 Carbon | Hybrid: opponents read, rest true | 0.697 [0.561, 0.826] | 0.116 [0.079, 0.161] | 0.112 | +0.002 [−0.000, +0.005] | 0.136 |
| Gemini 4 Carbon | Direct: the model's own xG (zero-shot, image) | 0.723 [0.610, 0.828] | 0.112 [0.075, 0.154] | 0.115 | −0.003 [−0.008, +0.003] | 0.343 |
| Gemini 4 Carbon | StatsBomb xG | 0.762 [0.638, 0.862] | 0.114 [0.076, 0.157] | 0.102 | −0.001 [−0.009, +0.007] | 0.862 |
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
| Gemini 4 Carbon | 150 | −0.003 [−0.008, +0.003] | 0.343 | +0.002 [−0.000, +0.005] | 0.152 | −0.005 [−0.011, +0.001] | 0.124 |
| GPT-6.2 Astra | 298 | −0.002 [−0.011, +0.006] | 0.613 | +0.001 [−0.000, +0.002] | 0.147 | −0.003 [−0.012, +0.005] | 0.504 |

## All three models on the same 150 shots (22 goals)

The common shots that every perception run measured, so the models can be compared with each other directly.

| Model | Distance MAE (yd) | Angle MAE (°) | Opponents exact | Brier: map-only / hybrid / direct | AUC: map-only / hybrid / direct | Hybrid − map-only [95% CI] | p | Direct − hybrid [95% CI] | p |
|---|---|---|---|---|---|---|---|---|---|
| Claude Opus 5.5 | 0.13 | 0.5 | 85% | 0.114 / 0.116 / 0.119 | 0.717 / 0.687 / 0.704 | +0.002 [−0.001, +0.006] | 0.229 | +0.002 [−0.006, +0.011] | 0.612 |
| Gemini 4 Carbon | 0.13 | 0.3 | 83% | 0.114 / 0.116 / 0.112 | 0.717 / 0.697 / 0.723 | +0.002 [−0.000, +0.005] | 0.152 | −0.005 [−0.011, +0.001] | 0.124 |
| GPT-6.2 Astra | 0.07 | 0.2 | 92% | 0.114 / 0.116 / 0.111 | 0.717 / 0.701 / 0.737 | +0.001 [−0.000, +0.003] | 0.144 | −0.005 [−0.016, +0.005] | 0.365 |

## Mean xG by distance band and body part

The drawing does not show the body part, so neither the map-only model nor the readings can tell a header from a foot shot.

| Model | Band (yd) | Body part | n | Observed | StatsBomb | Map-only (true geometry) | Hybrid (model's readings) | Direct (model's xG) |
|---|---|---|---|---|---|---|---|---|
| Claude Opus 5.5 | <6 | all | 8 | 0.50 | 0.480 | 0.528 | 0.521 | 0.351 |
| Claude Opus 5.5 | <6 | header | 4 | 0.50 | 0.322 | 0.500 | 0.513 | 0.362 |
| Claude Opus 5.5 | <6 | foot/other | 4 | 0.50 | 0.637 | 0.555 | 0.528 | 0.340 |
| Claude Opus 5.5 | 6–12 | all | 77 | 0.18 | 0.168 | 0.204 | 0.204 | 0.221 |
| Claude Opus 5.5 | 6–12 | header | 45 | 0.11 | 0.095 | 0.196 | 0.198 | 0.225 |
| Claude Opus 5.5 | 6–12 | foot/other | 32 | 0.28 | 0.271 | 0.215 | 0.213 | 0.215 |
| Claude Opus 5.5 | 12–18 | all | 79 | 0.23 | 0.103 | 0.101 | 0.098 | 0.102 |
| Claude Opus 5.5 | 12–18 | header | 13 | 0.00 | 0.042 | 0.137 | 0.130 | 0.129 |
| Claude Opus 5.5 | 12–18 | foot/other | 66 | 0.27 | 0.115 | 0.095 | 0.092 | 0.096 |
| Claude Opus 5.5 | 18–25 | all | 69 | 0.06 | 0.049 | 0.051 | 0.053 | 0.041 |
| Claude Opus 5.5 | 18–25 | header | 1 | 0.00 | 0.026 | 0.058 | 0.052 | 0.040 |
| Claude Opus 5.5 | 18–25 | foot/other | 68 | 0.06 | 0.050 | 0.051 | 0.053 | 0.041 |
| Claude Opus 5.5 | 25+ | all | 65 | 0.03 | 0.024 | 0.027 | 0.027 | 0.029 |
| Claude Opus 5.5 | 25+ | foot/other | 65 | 0.03 | 0.024 | 0.027 | 0.027 | 0.029 |
| Gemini 4 Carbon | <6 | all | 6 | 0.50 | 0.463 | 0.499 | 0.475 | 0.465 |
| Gemini 4 Carbon | <6 | header | 3 | 0.67 | 0.339 | 0.493 | 0.450 | 0.380 |
| Gemini 4 Carbon | <6 | foot/other | 3 | 0.33 | 0.586 | 0.505 | 0.501 | 0.550 |
| Gemini 4 Carbon | 6–12 | all | 41 | 0.17 | 0.139 | 0.193 | 0.187 | 0.174 |
| Gemini 4 Carbon | 6–12 | header | 23 | 0.13 | 0.076 | 0.183 | 0.172 | 0.160 |
| Gemini 4 Carbon | 6–12 | foot/other | 18 | 0.22 | 0.220 | 0.205 | 0.205 | 0.192 |
| Gemini 4 Carbon | 12–18 | all | 42 | 0.24 | 0.107 | 0.103 | 0.099 | 0.117 |
| Gemini 4 Carbon | 12–18 | header | 6 | 0.00 | 0.045 | 0.125 | 0.125 | 0.133 |
| Gemini 4 Carbon | 12–18 | foot/other | 36 | 0.28 | 0.117 | 0.099 | 0.094 | 0.114 |
| Gemini 4 Carbon | 18–25 | all | 30 | 0.07 | 0.055 | 0.054 | 0.050 | 0.053 |
| Gemini 4 Carbon | 18–25 | header | 1 | 0.00 | 0.026 | 0.058 | 0.057 | 0.060 |
| Gemini 4 Carbon | 18–25 | foot/other | 29 | 0.07 | 0.056 | 0.054 | 0.050 | 0.052 |
| Gemini 4 Carbon | 25+ | all | 31 | 0.00 | 0.023 | 0.024 | 0.023 | 0.027 |
| Gemini 4 Carbon | 25+ | foot/other | 31 | 0.00 | 0.023 | 0.024 | 0.023 | 0.027 |
| GPT-6.2 Astra | <6 | all | 8 | 0.50 | 0.480 | 0.528 | 0.527 | 0.547 |
| GPT-6.2 Astra | <6 | header | 4 | 0.50 | 0.322 | 0.500 | 0.506 | 0.475 |
| GPT-6.2 Astra | <6 | foot/other | 4 | 0.50 | 0.637 | 0.555 | 0.548 | 0.620 |
| GPT-6.2 Astra | 6–12 | all | 77 | 0.18 | 0.168 | 0.204 | 0.202 | 0.298 |
| GPT-6.2 Astra | 6–12 | header | 45 | 0.11 | 0.095 | 0.196 | 0.193 | 0.298 |
| GPT-6.2 Astra | 6–12 | foot/other | 32 | 0.28 | 0.271 | 0.215 | 0.215 | 0.298 |
| GPT-6.2 Astra | 12–18 | all | 79 | 0.23 | 0.103 | 0.101 | 0.100 | 0.141 |
| GPT-6.2 Astra | 12–18 | header | 13 | 0.00 | 0.042 | 0.137 | 0.134 | 0.184 |
| GPT-6.2 Astra | 12–18 | foot/other | 66 | 0.27 | 0.115 | 0.095 | 0.093 | 0.133 |
| GPT-6.2 Astra | 18–25 | all | 69 | 0.06 | 0.049 | 0.051 | 0.051 | 0.055 |
| GPT-6.2 Astra | 18–25 | header | 1 | 0.00 | 0.026 | 0.058 | 0.058 | 0.100 |
| GPT-6.2 Astra | 18–25 | foot/other | 68 | 0.06 | 0.050 | 0.051 | 0.051 | 0.054 |
| GPT-6.2 Astra | 25+ | all | 65 | 0.03 | 0.024 | 0.027 | 0.026 | 0.029 |
| GPT-6.2 Astra | 25+ | foot/other | 65 | 0.03 | 0.024 | 0.027 | 0.026 | 0.029 |

## Reading

- Claude Opus 5.5 (298 shots): distance read with a mean absolute error of 0.12 yd (bias at 6–12 yd −0.06 yd), angle 0.4°, opponents exact in 83%. Scored by the logistic model, the readings cost +0.0016 Brier against the true geometry (not significant; AUC 0.695 vs 0.721). The model's own xG differs from the hybrid by −0.0008 (not significant; AUC 0.721). At 6–12 yd, headers (45): StatsBomb 0.095, map-only 0.196, hybrid 0.198, the model 0.225; foot shots (32): StatsBomb 0.271, map-only 0.215, hybrid 0.213, the model 0.215. Per shot, distance misreading and the direct xG's gap to the map-only baseline correlate at ρ = −0.10 (6–12 yd: −0.22).
- Gemini 4 Carbon (150 shots): distance read with a mean absolute error of 0.13 yd (bias at 6–12 yd −0.02 yd), angle 0.3°, opponents exact in 83%. Scored by the logistic model, the readings cost +0.0018 Brier against the true geometry (not significant; AUC 0.697 vs 0.717). The model's own xG differs from the hybrid by −0.0045 (not significant; AUC 0.723). At 6–12 yd, headers (23): StatsBomb 0.076, map-only 0.183, hybrid 0.172, the model 0.160; foot shots (18): StatsBomb 0.220, map-only 0.205, hybrid 0.205, the model 0.192. Per shot, distance misreading and the direct xG's gap to the map-only baseline correlate at ρ = −0.10 (6–12 yd: 0.02).
- GPT-6.2 Astra (298 shots): distance read with a mean absolute error of 0.07 yd (bias at 6–12 yd −0.00 yd), angle 0.2°, opponents exact in 91%. Scored by the logistic model, the readings cost +0.0007 Brier against the true geometry (not significant; AUC 0.708 vs 0.721). The model's own xG differs from the hybrid by −0.0028 (not significant; AUC 0.733). At 6–12 yd, headers (45): StatsBomb 0.095, map-only 0.196, hybrid 0.193, the model 0.298; foot shots (32): StatsBomb 0.271, map-only 0.215, hybrid 0.215, the model 0.298. Per shot, distance misreading and the direct xG's gap to the map-only baseline correlate at ρ = −0.14 (6–12 yd: −0.16).
- Thinking: measuring takes far more thinking than estimating xG (Claude Opus 5.5 568 tokens per shot on average; Gemini 4 Carbon 3,718 tokens per shot on average; GPT-6.2 Astra 196 tokens per shot on average; the xG requests in the table above). The hybrid shows what a model can read off the image when asked to measure, not what it used when it gave its xG.
- Close range: at 6–12 yd the over-estimate sits in the headers. Claude Opus 5.5, Gemini 4 Carbon, GPT-6.2 Astra rate headers above StatsBomb, and so does the map-only model from the true geometry, because nothing in the drawing marks a header. On foot shots from the same range Claude Opus 5.5, Gemini 4 Carbon are below StatsBomb; GPT-6.2 Astra is at or above it (per-model numbers in the bullets above and the band table).
