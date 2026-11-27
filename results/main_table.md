# Main results table

Evaluation set: 298 shots every included run answered (42 goals, 63 matches; observed goal rate 0.141 [0.095, 0.193]).
Brackets: 95% percentile intervals from a match-clustered bootstrap (2,000 resamples of matches, seed 0). ΔBrier = Brier(model) − Brier(reference); negative favours the model. p = two-sided bootstrap p-value for ΔBrier = 0.

## zero-shot

| Model | AUC | Brier | Log loss | Mean xG | ΔBrier vs StatsBomb | p | ΔBrier vs baseline | p |
|---|---|---|---|---|---|---|---|---|
| Gemma 4 (26B) | 0.597 [0.493, 0.696] | 0.190 [0.169, 0.213] | 0.572 [0.522, 0.629] | 0.379 [0.359, 0.400] | +0.091 [+0.062, +0.118] | <0.001 | +0.084 [+0.057, +0.111] | <0.001 |
| Claude Sonnet 5.5 | 0.683 [0.601, 0.760] | 0.121 [0.085, 0.163] | 0.408 [0.301, 0.532] | 0.085 [0.077, 0.094] | +0.022 [+0.010, +0.034] | <0.001 | +0.015 [+0.005, +0.028] | 0.004 |
| Claude Opus 5.5 | 0.721 [0.624, 0.808] | 0.111 [0.079, 0.147] | 0.378 [0.283, 0.487] | 0.109 [0.097, 0.121] | +0.012 [+0.001, +0.022] | 0.022 | +0.005 [−0.005, +0.016] | 0.314 |
| Gemini 3.8 Flash | 0.721 [0.639, 0.801] | 0.116 [0.088, 0.147] | 0.380 [0.304, 0.465] | 0.167 [0.147, 0.188] | +0.016 [+0.003, +0.028] | 0.012 | +0.010 [−0.002, +0.021] | 0.104 |

## reference

| Model | AUC | Brier | Log loss | Mean xG | ΔBrier vs StatsBomb | p | ΔBrier vs baseline | p |
|---|---|---|---|---|---|---|---|---|
| StatsBomb xG | 0.792 [0.704, 0.867] | 0.100 [0.069, 0.136] | 0.340 [0.249, 0.447] | 0.100 [0.084, 0.118] |  |  | −0.006 [−0.012, −0.001] | 0.020 |
| Logistic baseline | 0.768 [0.694, 0.843] | 0.106 [0.073, 0.143] | 0.351 [0.258, 0.462] | 0.108 [0.095, 0.123] | +0.006 [+0.001, +0.012] | 0.020 |  |  |

## Sensitivity: shot-level vs match-clustered bootstrap for ΔBrier

The abstract used a shot-level bootstrap. Match clustering usually widens the intervals a little.

| Model | ΔBrier vs baseline | shot-level 95% CI | match-clustered 95% CI | ΔBrier vs StatsBomb | shot-level 95% CI | match-clustered 95% CI |
|---|---|---|---|---|---|---|
| Gemma 4 (26B) | +0.0844 | [+0.059, +0.111] | [+0.057, +0.111] | +0.0906 | [+0.065, +0.118] | [+0.062, +0.118] |
| Claude Sonnet 5.5 | +0.0153 | [+0.004, +0.027] | [+0.005, +0.028] | +0.0215 | [+0.009, +0.036] | [+0.010, +0.034] |
| Claude Opus 5.5 | +0.0053 | [−0.002, +0.013] | [−0.005, +0.016] | +0.0115 | [+0.003, +0.021] | [+0.001, +0.022] |
| Gemini 3.8 Flash | +0.0100 | [+0.000, +0.020] | [−0.002, +0.021] | +0.0162 | [+0.006, +0.027] | [+0.003, +0.028] |
