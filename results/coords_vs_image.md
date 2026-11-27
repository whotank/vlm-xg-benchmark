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

## Reading (rule fixed before the runs, plus a level-shift check)

- Claude Opus 5.5: at 6–12 yd the gap to StatsBomb goes from +0.053 (image) to −0.014 (coordinates); change −0.067 [−0.082, −0.049], so the over-estimate shrinks significantly. But the change is not specific to close range: the model's mean xG over all shots moves by −0.027 [−0.033, −0.021] and overall Brier by +0.0001 [−0.0048, +0.0052]. Coordinates lower the model's level across bands rather than fixing its reading of close-range shots, so this is weak evidence that the picture was the problem.
- Gemini 4 Carbon: at 6–12 yd the gap to StatsBomb goes from +0.030 (image) to +0.020 (coordinates); change −0.011 [−0.026, +0.008], so the over-estimate does not change significantly: by the rule, the close-range error is in the judgment, not in reading the image.

Caveat: with coordinates the model still has to turn raw positions into distance, angle and blockers; the `--all-features` runs (results/features_ablation.md), which state those quantities directly, separate that geometric step from the judgment itself.
