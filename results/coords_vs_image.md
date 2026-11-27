# Perception vs judgment: image vs coordinates as text

Evaluation set: the 298 common shots of compare_models.py (42 goals). Each model sees the same freeze frame either as the rendered image (zero-shot run) or as (x, y) coordinates in text (`--input coords`: pitch size, goal and post positions, shooter, goalkeeper, other opponents, teammates; no body part, no roles). Brackets: 95% match-clustered bootstrap intervals; Δ = coordinates − image, paired; p = two-sided bootstrap p-value.

## Overall

| Model | Input | AUC | Brier | Mean xG | ΔBrier (coords − image) | p | ΔAUC |
|---|---|---|---|---|---|---|---|
| Claude Opus 5.5 | image | 0.721 [0.624, 0.808] | 0.111 [0.079, 0.147] | 0.109 [0.097, 0.121] | | | |
| Claude Opus 5.5 | coordinates | 0.721 [0.632, 0.805] | 0.111 [0.077, 0.150] | 0.082 [0.072, 0.093] | +0.000 [−0.005, +0.005] | 0.975 | +0.000 [−0.040, +0.036] |
| Gemini 4 Carbon | image | 0.743 [0.657, 0.819] | 0.105 [0.074, 0.141] | 0.111 [0.097, 0.126] | | | |
| Gemini 4 Carbon | coordinates | 0.754 [0.672, 0.831] | 0.105 [0.073, 0.142] | 0.104 [0.089, 0.119] | +0.000 [−0.004, +0.005] | 0.980 | +0.011 [−0.012, +0.035] |
| Map-only baseline | exact numbers | 0.721 [0.636, 0.808] | 0.110 [0.078, 0.147] | 0.111 | | | |
| StatsBomb xG | exact numbers | 0.792 [0.704, 0.867] | 0.100 [0.069, 0.136] | 0.100 | | | |

## By distance band

Gap = mean predicted xG − StatsBomb mean in the band. Change = coordinates − image (paired); negative at 6–12 yd means the over-estimate shrinks.

### <6 yd (n=8, 4 goals, observed 0.50, StatsBomb 0.480)

| Model | Image: mean xG (gap) | Coordinates: mean xG (gap) | Change in mean / gap | p | Brier image → coords | ΔBrier | p |
|---|---|---|---|---|---|---|---|
| Claude Opus 5.5 | 0.351 (-0.128) | 0.335 (-0.145) | −0.016 [−0.067, +0.031] | 0.541 | 0.291 → 0.300 | +0.009 [−0.039, +0.070] | 0.735 |
| Gemini 4 Carbon | 0.491 (+0.012) | 0.518 (+0.038) | +0.026 [−0.037, +0.079] | 0.375 | 0.234 → 0.252 | +0.017 [−0.038, +0.070] | 0.495 |

### 6–12 yd (n=77, 14 goals, observed 0.18, StatsBomb 0.168)

| Model | Image: mean xG (gap) | Coordinates: mean xG (gap) | Change in mean / gap | p | Brier image → coords | ΔBrier | p |
|---|---|---|---|---|---|---|---|
| Claude Opus 5.5 | 0.221 (+0.053) | 0.154 (-0.014) | −0.067 [−0.082, −0.049] | <0.001 | 0.125 → 0.115 | −0.010 [−0.024, +0.003] | 0.132 |
| Gemini 4 Carbon | 0.199 (+0.030) | 0.188 (+0.020) | −0.011 [−0.026, +0.008] | 0.231 | 0.114 → 0.106 | −0.007 [−0.022, +0.008] | 0.322 |

### 12–18 yd (n=79, 18 goals, observed 0.23, StatsBomb 0.103)

| Model | Image: mean xG (gap) | Coordinates: mean xG (gap) | Change in mean / gap | p | Brier image → coords | ΔBrier | p |
|---|---|---|---|---|---|---|---|
| Claude Opus 5.5 | 0.102 (-0.001) | 0.071 (-0.032) | −0.030 [−0.041, −0.019] | <0.001 | 0.195 → 0.204 | +0.009 [−0.000, +0.020] | 0.056 |
| Gemini 4 Carbon | 0.110 (+0.007) | 0.103 (-0.000) | −0.007 [−0.015, +0.000] | 0.055 | 0.189 → 0.195 | +0.006 [+0.001, +0.012] | 0.024 |

### 18–25 yd (n=69, 4 goals, observed 0.06, StatsBomb 0.049)

| Model | Image: mean xG (gap) | Coordinates: mean xG (gap) | Change in mean / gap | p | Brier image → coords | ΔBrier | p |
|---|---|---|---|---|---|---|---|
| Claude Opus 5.5 | 0.041 (-0.009) | 0.038 (-0.011) | −0.002 [−0.004, −0.000] | 0.021 | 0.055 → 0.055 | +0.000 [−0.001, +0.002] | 0.893 |
| Gemini 4 Carbon | 0.049 (-0.000) | 0.038 (-0.011) | −0.011 [−0.017, −0.007] | <0.001 | 0.056 → 0.056 | −0.001 [−0.003, +0.002] | 0.531 |

### 25+ yd (n=65, 2 goals, observed 0.03, StatsBomb 0.024)

| Model | Image: mean xG (gap) | Coordinates: mean xG (gap) | Change in mean / gap | p | Brier image → coords | ΔBrier | p |
|---|---|---|---|---|---|---|---|
| Claude Opus 5.5 | 0.029 (+0.006) | 0.027 (+0.003) | −0.003 [−0.004, −0.001] | 0.001 | 0.030 → 0.030 | −0.000 [−0.000, −0.000] | 0.019 |
| Gemini 4 Carbon | 0.027 (+0.003) | 0.023 (-0.001) | −0.004 [−0.005, −0.003] | <0.001 | 0.029 → 0.030 | +0.000 [−0.000, +0.001] | 0.734 |

## Headers (neither input has the body part)

n=63, 7 goals, observed 0.111, StatsBomb 0.098.

| Model | Image | Coordinates | Change [95% CI] |
|---|---|---|---|
| Claude Opus 5.5 | 0.211 | 0.142 | −0.070 [−0.084, −0.055] |
| Gemini 4 Carbon | 0.185 | 0.179 | −0.006 [−0.021, +0.012] |

## By distance band and body part

Most 6–12 yd shots are headers, which StatsBomb rates far lower than foot shots from the same distance and neither input can identify. Gap = mean xG − StatsBomb mean in the cell; change = coordinates − image (paired).

| Model | Band (yd) | Body part | n (goals) | StatsBomb | Image: mean (gap) | Coordinates: mean (gap) | Change [95% CI] | p |
|---|---|---|---|---|---|---|---|---|
| Claude Opus 5.5 | <6 | header | 4 (2) | 0.322 | 0.362 (+0.040) | 0.302 (−0.020) | −0.060 [−0.110, +0.050] | 0.115 |
| Claude Opus 5.5 | <6 | foot/other | 4 (2) | 0.637 | 0.340 (−0.297) | 0.368 (−0.270) | +0.028 [−0.020, +0.070] | 0.120 |
| Claude Opus 5.5 | 6–12 | header | 45 (5) | 0.095 | 0.225 (+0.130) | 0.143 (+0.048) | −0.082 [−0.099, −0.066] | <0.001 |
| Claude Opus 5.5 | 6–12 | foot/other | 32 (9) | 0.271 | 0.215 (−0.055) | 0.170 (−0.101) | −0.046 [−0.073, −0.011] | 0.013 |
| Claude Opus 5.5 | 12–18 | header | 13 (0) | 0.042 | 0.129 (+0.088) | 0.094 (+0.052) | −0.035 [−0.051, −0.016] | 0.003 |
| Claude Opus 5.5 | 12–18 | foot/other | 66 (18) | 0.115 | 0.096 (−0.019) | 0.067 (−0.048) | −0.029 [−0.041, −0.017] | <0.001 |
| Claude Opus 5.5 | 18–25 | header | 1 (0) | 0.026 | 0.040 (+0.014) | 0.050 (+0.024) | +0.010 [+0.010, +0.010] | <0.001 |
| Claude Opus 5.5 | 18–25 | foot/other | 68 (4) | 0.050 | 0.041 (−0.009) | 0.038 (−0.012) | −0.002 [−0.005, −0.000] | 0.017 |
| Claude Opus 5.5 | 25+ | foot/other | 65 (2) | 0.024 | 0.029 (+0.006) | 0.027 (+0.003) | −0.003 [−0.004, −0.001] | 0.001 |
| Gemini 4 Carbon | <6 | header | 4 (2) | 0.322 | 0.415 (+0.093) | 0.417 (+0.095) | +0.002 [−0.140, +0.140] | 0.980 |
| Gemini 4 Carbon | <6 | foot/other | 4 (2) | 0.637 | 0.568 (−0.070) | 0.618 (−0.020) | +0.050 [+0.000, +0.100] | 0.066 |
| Gemini 4 Carbon | 6–12 | header | 45 (5) | 0.095 | 0.187 (+0.092) | 0.172 (+0.077) | −0.015 [−0.031, +0.003] | 0.099 |
| Gemini 4 Carbon | 6–12 | foot/other | 32 (9) | 0.271 | 0.214 (−0.056) | 0.210 (−0.061) | −0.004 [−0.031, +0.027] | 0.762 |
| Gemini 4 Carbon | 12–18 | header | 13 (0) | 0.042 | 0.115 (+0.073) | 0.138 (+0.097) | +0.024 [+0.002, +0.049] | 0.031 |
| Gemini 4 Carbon | 12–18 | foot/other | 66 (18) | 0.115 | 0.110 (−0.005) | 0.096 (−0.019) | −0.014 [−0.021, −0.006] | <0.001 |
| Gemini 4 Carbon | 18–25 | header | 1 (0) | 0.026 | 0.060 (+0.034) | 0.050 (+0.024) | −0.010 [−0.010, −0.010] | <0.001 |
| Gemini 4 Carbon | 18–25 | foot/other | 68 (4) | 0.050 | 0.049 (−0.001) | 0.038 (−0.012) | −0.011 [−0.017, −0.007] | <0.001 |
| Gemini 4 Carbon | 25+ | foot/other | 65 (2) | 0.024 | 0.027 (+0.003) | 0.023 (−0.001) | −0.004 [−0.005, −0.003] | <0.001 |

## Reading (rule fixed before the runs, plus a level-shift check)

- Claude Opus 5.5: at 6–12 yd the gap to StatsBomb goes from +0.053 (image) to −0.014 (coordinates); change −0.067 [−0.082, −0.049]. Outside 6–12 yd the model's mean xG moves by −0.013 [−0.017, −0.008], so the close-range change beyond that level shift is −0.054 [−0.071, −0.036] (specific to close range). By body part at 6–12 yd: headers +0.130 → +0.048, foot shots −0.055 → −0.101 (gaps to StatsBomb, image → coordinates). The change is not a correction: shots the image run already rated below StatsBomb move further away, so coordinates shift the whole close-range level rather than fixing the over-rated shots. Overall Brier changes by +0.0001 [−0.0048, +0.0052] and the band's Brier by −0.0101 [−0.0239, +0.0035]. By the rule, how the model used the picture affects its close-range level, but the change does not significantly improve accuracy, and the over-estimate itself is on headers, which coordinates cannot identify either.
- Gemini 4 Carbon: at 6–12 yd the gap to StatsBomb goes from +0.030 (image) to +0.020 (coordinates); change −0.011 [−0.026, +0.008]. Inconclusive: the interval includes no change, but its lower end is a reduction of 85% of the image run's over-estimate, so this run can neither confirm nor rule out that reading the picture contributed. It is not evidence that the error is judgment.

Caveat: with coordinates the model still has to turn raw positions into distance, angle and blockers; the `--all-features` runs (results/features_ablation.md), which state those quantities directly, separate that geometric step from the judgment itself.
