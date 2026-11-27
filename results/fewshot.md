# Few-shot in-context learning

Evaluation set: the 298 common shots of compare_models.py (42 goals; observed rate 0.141). Examples come from shots outside the 300-shot test sample and are labelled with the actual outcome only ("Outcome: GOAL" / "Outcome: NO GOAL"); sets are nested (k=1 ⊂ k=5 ⊂ k=10). Brackets: 95% match-clustered bootstrap intervals (2,000 resamples of matches, seed 0); Δ vs k=0 is paired on the same resamples, p = two-sided bootstrap p-value. API runs are not deterministic: compare any difference with the gap between the two k=0 rows where a rerun exists.

Claude Opus 5.5's "zero-shot" k=0 row is the original run, whose images were rendered in the authors' environment; every few-shot run and the "zero-shot rerun" row (`claude_opus_5_5_rerun`) used images rendered in this environment with the same code. The last column compares each few-shot run with that same-environment rerun. Gemini 4 Carbon's k=0 run was already rendered here.

## Claude Opus 5.5

| k | Examples | AUC | Brier | Log loss | Mean xG | ΔBrier vs k=0 | p | ΔAUC vs k=0 | ΔBrier vs zero-shot rerun | p |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | zero-shot | 0.721 [0.624, 0.808] | 0.111 [0.079, 0.147] | 0.378 [0.283, 0.487] | 0.109 [0.097, 0.121] |  |  |  |  |  |
| 0 | zero-shot rerun | 0.717 [0.630, 0.800] | 0.113 [0.080, 0.150] | 0.382 [0.285, 0.490] | 0.110 [0.098, 0.122] | +0.002 [−0.001, +0.006] | 0.259 | −0.004 [−0.036, +0.030] |  |  |
| 1 | set A | 0.721 [0.630, 0.810] | 0.110 [0.079, 0.145] | 0.376 [0.282, 0.484] | 0.117 [0.104, 0.129] | −0.001 [−0.005, +0.003] | 0.722 | −0.000 [−0.030, +0.029] | −0.003 [−0.006, +0.000] | 0.073 |
| 5 | set A | 0.726 [0.637, 0.812] | 0.110 [0.080, 0.144] | 0.372 [0.283, 0.476] | 0.119 [0.107, 0.133] | −0.001 [−0.005, +0.002] | 0.615 | +0.005 [−0.023, +0.030] | −0.003 [−0.007, +0.001] | 0.148 |
| 10 | set A | 0.722 [0.634, 0.804] | 0.113 [0.081, 0.149] | 0.377 [0.287, 0.483] | 0.116 [0.104, 0.129] | +0.002 [−0.002, +0.006] | 0.344 | +0.001 [−0.032, +0.034] | −0.000 [−0.004, +0.003] | 0.892 |
| 1 | set B | 0.707 [0.618, 0.794] | 0.114 [0.082, 0.151] | 0.388 [0.293, 0.498] | 0.115 [0.102, 0.127] | +0.003 [+0.000, +0.006] | 0.026 | −0.014 [−0.040, +0.011] | +0.001 [−0.001, +0.004] | 0.297 |
| 5 | set B | 0.715 [0.633, 0.796] | 0.113 [0.081, 0.149] | 0.381 [0.288, 0.487] | 0.118 [0.105, 0.130] | +0.002 [−0.001, +0.005] | 0.119 | −0.007 [−0.029, +0.021] | +0.000 [−0.004, +0.004] | 0.945 |
| 10 | set B | 0.711 [0.625, 0.792] | 0.113 [0.081, 0.149] | 0.382 [0.290, 0.488] | 0.116 [0.104, 0.128] | +0.002 [−0.001, +0.006] | 0.181 | −0.010 [−0.045, +0.024] | +0.000 [−0.004, +0.004] | 0.937 |

Mean predicted xG by distance band (yards):

| k | Examples | <6 (n=8) | 6–12 (n=77) | 12–18 (n=79) | 18–25 (n=69) | 25+ (n=65) |
|---|---|---|---|---|---|---|
| | Observed rate | 0.50 | 0.18 | 0.23 | 0.06 | 0.03 |
| | StatsBomb xG | 0.48 | 0.17 | 0.10 | 0.05 | 0.02 |
| 0 | zero-shot | 0.35 [0.29, 0.39] | 0.22 [0.19, 0.25] | 0.10 [0.09, 0.11] | 0.04 [0.04, 0.04] | 0.03 [0.03, 0.03] |
| 0 | zero-shot rerun | 0.34 [0.29, 0.39] | 0.23 [0.20, 0.26] | 0.10 [0.09, 0.11] | 0.04 [0.04, 0.04] | 0.03 [0.03, 0.03] |
| 1 | set A | 0.36 [0.33, 0.41] | 0.24 [0.22, 0.26] | 0.11 [0.10, 0.12] | 0.04 [0.04, 0.05] | 0.03 [0.03, 0.03] |
| 5 | set A | 0.35 [0.29, 0.42] | 0.24 [0.22, 0.27] | 0.12 [0.10, 0.13] | 0.04 [0.04, 0.05] | 0.03 [0.03, 0.03] |
| 10 | set A | 0.35 [0.32, 0.39] | 0.23 [0.21, 0.26] | 0.11 [0.10, 0.12] | 0.05 [0.04, 0.06] | 0.03 [0.03, 0.03] |
| 1 | set B | 0.36 [0.33, 0.40] | 0.24 [0.21, 0.26] | 0.11 [0.09, 0.12] | 0.04 [0.04, 0.05] | 0.03 [0.03, 0.03] |
| 5 | set B | 0.36 [0.29, 0.41] | 0.23 [0.21, 0.26] | 0.11 [0.10, 0.13] | 0.05 [0.04, 0.05] | 0.03 [0.03, 0.03] |
| 10 | set B | 0.32 [0.24, 0.39] | 0.23 [0.20, 0.26] | 0.12 [0.10, 0.13] | 0.05 [0.04, 0.05] | 0.03 [0.03, 0.03] |

## Gemini 4 Carbon

| k | Examples | AUC | Brier | Log loss | Mean xG | ΔBrier vs k=0 | p | ΔAUC vs k=0 | ΔBrier vs zero-shot rerun | p |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | zero-shot | 0.743 [0.657, 0.819] | 0.105 [0.074, 0.141] | 0.356 [0.266, 0.459] | 0.111 [0.097, 0.126] |  |  |  |  |  |
| 0 | zero-shot rerun | 0.738 [0.657, 0.817] | 0.106 [0.075, 0.141] | 0.359 [0.269, 0.460] | 0.109 [0.095, 0.124] | +0.001 [−0.001, +0.003] | 0.485 | −0.005 [−0.022, +0.011] |  |  |
| 1 | set A | 0.745 [0.662, 0.821] | 0.106 [0.075, 0.141] | 0.358 [0.269, 0.462] | 0.109 [0.095, 0.123] | +0.001 [−0.002, +0.004] | 0.429 | +0.002 [−0.015, +0.016] | +0.000 [−0.003, +0.004] | 0.861 |
| 5 | set A | 0.745 [0.663, 0.822] | 0.105 [0.074, 0.139] | 0.355 [0.266, 0.454] | 0.113 [0.098, 0.128] | −0.000 [−0.004, +0.003] | 0.898 | +0.002 [−0.017, +0.020] | −0.001 [−0.004, +0.002] | 0.508 |
| 10 | set A | 0.740 [0.655, 0.819] | 0.107 [0.075, 0.142] | 0.360 [0.269, 0.463] | 0.113 [0.100, 0.127] | +0.002 [−0.002, +0.005] | 0.336 | −0.003 [−0.022, +0.016] | +0.001 [−0.002, +0.005] | 0.628 |
| 1 | set B | 0.742 [0.656, 0.822] | 0.107 [0.075, 0.142] | 0.361 [0.268, 0.465] | 0.105 [0.092, 0.118] | +0.002 [−0.001, +0.006] | 0.211 | −0.001 [−0.020, +0.018] | +0.001 [−0.002, +0.006] | 0.483 |
| 5 | set B | 0.749 [0.668, 0.825] | 0.107 [0.075, 0.143] | 0.358 [0.267, 0.461] | 0.109 [0.096, 0.122] | +0.002 [−0.002, +0.007] | 0.425 | +0.006 [−0.014, +0.026] | +0.001 [−0.003, +0.007] | 0.725 |
| 10 | set B | 0.743 [0.662, 0.818] | 0.107 [0.075, 0.142] | 0.359 [0.269, 0.462] | 0.108 [0.095, 0.121] | +0.002 [−0.003, +0.007] | 0.569 | −0.000 [−0.021, +0.019] | +0.001 [−0.004, +0.007] | 0.836 |

Mean predicted xG by distance band (yards):

| k | Examples | <6 (n=8) | 6–12 (n=77) | 12–18 (n=79) | 18–25 (n=69) | 25+ (n=65) |
|---|---|---|---|---|---|---|
| | Observed rate | 0.50 | 0.18 | 0.23 | 0.06 | 0.03 |
| | StatsBomb xG | 0.48 | 0.17 | 0.10 | 0.05 | 0.02 |
| 0 | zero-shot | 0.49 [0.36, 0.60] | 0.20 [0.17, 0.23] | 0.11 [0.10, 0.12] | 0.05 [0.04, 0.06] | 0.03 [0.03, 0.03] |
| 0 | zero-shot rerun | 0.48 [0.36, 0.59] | 0.19 [0.16, 0.22] | 0.11 [0.10, 0.12] | 0.05 [0.04, 0.06] | 0.03 [0.03, 0.03] |
| 1 | set A | 0.45 [0.35, 0.55] | 0.20 [0.17, 0.23] | 0.11 [0.10, 0.13] | 0.05 [0.04, 0.06] | 0.03 [0.02, 0.03] |
| 5 | set A | 0.49 [0.35, 0.62] | 0.20 [0.17, 0.23] | 0.12 [0.11, 0.13] | 0.05 [0.04, 0.06] | 0.03 [0.03, 0.03] |
| 10 | set A | 0.44 [0.31, 0.56] | 0.20 [0.18, 0.24] | 0.12 [0.11, 0.14] | 0.05 [0.04, 0.05] | 0.03 [0.03, 0.03] |
| 1 | set B | 0.44 [0.33, 0.54] | 0.18 [0.16, 0.21] | 0.11 [0.10, 0.12] | 0.05 [0.04, 0.06] | 0.03 [0.02, 0.03] |
| 5 | set B | 0.46 [0.33, 0.59] | 0.19 [0.16, 0.22] | 0.12 [0.10, 0.13] | 0.05 [0.04, 0.05] | 0.03 [0.02, 0.03] |
| 10 | set B | 0.43 [0.32, 0.55] | 0.19 [0.16, 0.22] | 0.12 [0.10, 0.13] | 0.05 [0.04, 0.05] | 0.03 [0.02, 0.03] |

## Examples close to a test shot

Few-shot examples from the same match and half as an evaluation shot, within 120 seconds (dt = example time − test-shot time). An example image that shows nearly the same situation, labelled with its outcome, can inform the test shot directly. Each example enters at the k shown and stays in the larger sets.

| Set | Example (prompt order, first k) | Test shot | dt | Same team | Match |
|---|---|---|---|---|---|
| A | `c56eba74` Kieffer Roberto Francisco Moore (Wales), 52:19, no goal (order 2, k=5) | `4938ed57` Ali Gholizadeh (Iran), 51:08, no goal | +71 s | no | Wales vs Iran |
| A | `c56eba74` Kieffer Roberto Francisco Moore (Wales), 52:19, no goal (order 2, k=5) | `9a788179` Sardar Azmoun (Iran), 51:09, no goal | +70 s | no | Wales vs Iran |
| A | `7ce22041` Christian Dannemann Eriksen (Denmark), 68:26, no goal (order 10, k=10) | `641eb264` Andreas Christensen (Denmark), 69:15, no goal | -49 s | yes | Denmark vs Tunisia |
| B | `483f71ea` Mohammed Kudus (Ghana), 67:52, goal (order 8, k=10) | `9d2721df` Iñaki Williams Arthuer (Ghana), 67:50, no goal | +2 s | yes | South Korea vs Ghana |

Predictions for the affected test shots (zero-shot, the zero-shot rerun, and every run that includes the example), and the change in each run's ΔBrier vs k=0 when the affected shots are dropped:

| Test shot | Model | Run | xG | Zero-shot xG | StatsBomb |
|---|---|---|---|---|---|
| `4938ed57` | Claude Opus 5.5 | Claude Opus 5.5 (rerun) | 0.030 | 0.040 | 0.043 |
| `4938ed57` | Claude Opus 5.5 | Claude Opus 5.5, k=5 | 0.030 | 0.040 | 0.043 |
| `4938ed57` | Claude Opus 5.5 | Claude Opus 5.5, k=10 | 0.030 | 0.040 | 0.043 |
| `4938ed57` | Gemini 4 Carbon | Gemini 4 Carbon (rerun) | 0.030 | 0.030 | 0.043 |
| `4938ed57` | Gemini 4 Carbon | Gemini 4 Carbon, k=5 | 0.030 | 0.030 | 0.043 |
| `4938ed57` | Gemini 4 Carbon | Gemini 4 Carbon, k=10 | 0.030 | 0.030 | 0.043 |
| `9a788179` | Claude Opus 5.5 | Claude Opus 5.5 (rerun) | 0.350 | 0.300 | 0.214 |
| `9a788179` | Claude Opus 5.5 | Claude Opus 5.5, k=5 | 0.300 | 0.300 | 0.214 |
| `9a788179` | Claude Opus 5.5 | Claude Opus 5.5, k=10 | 0.300 | 0.300 | 0.214 |
| `9a788179` | Gemini 4 Carbon | Gemini 4 Carbon (rerun) | 0.320 | 0.340 | 0.214 |
| `9a788179` | Gemini 4 Carbon | Gemini 4 Carbon, k=5 | 0.300 | 0.340 | 0.214 |
| `9a788179` | Gemini 4 Carbon | Gemini 4 Carbon, k=10 | 0.350 | 0.340 | 0.214 |
| `641eb264` | Claude Opus 5.5 | Claude Opus 5.5 (rerun) | 0.300 | 0.380 | 0.271 |
| `641eb264` | Claude Opus 5.5 | Claude Opus 5.5, k=10 | 0.350 | 0.380 | 0.271 |
| `641eb264` | Gemini 4 Carbon | Gemini 4 Carbon (rerun) | 0.450 | 0.520 | 0.271 |
| `641eb264` | Gemini 4 Carbon | Gemini 4 Carbon, k=10 | 0.420 | 0.520 | 0.271 |
| `9d2721df` | Claude Opus 5.5 | Claude Opus 5.5 (rerun) | 0.080 | 0.090 | 0.106 |
| `9d2721df` | Claude Opus 5.5 | Claude Opus 5.5, k=10 (set B) | 0.160 | 0.090 | 0.106 |
| `9d2721df` | Gemini 4 Carbon | Gemini 4 Carbon (rerun) | 0.120 | 0.160 | 0.106 |
| `9d2721df` | Gemini 4 Carbon | Gemini 4 Carbon, k=10 (set B) | 0.140 | 0.160 | 0.106 |

| Run | Shots dropped | ΔBrier vs k=0, all shots | without the affected shots |
|---|---|---|---|
| Claude Opus 5.5, k=5 | 2 | −0.0009 | −0.0010 |
| Claude Opus 5.5, k=10 | 3 | +0.0018 | +0.0019 |
| Gemini 4 Carbon, k=5 | 2 | −0.0003 | −0.0002 |
| Gemini 4 Carbon, k=10 | 3 | +0.0016 | +0.0019 |
| Claude Opus 5.5, k=10 (set B) | 1 | +0.0022 | +0.0022 |
| Gemini 4 Carbon, k=10 (set B) | 1 | +0.0015 | +0.0015 |

