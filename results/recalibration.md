# Recalibration (side analysis)

Evaluation set: the 298 common shots of compare_models.py (42 goals, 63 matches). Main results stay raw (zero-shot); this table shows what a monotone recalibration learned from labelled shots would change.

Method: Platt scaling (unpenalised logistic regression of goal on logit p) and isotonic regression, each fitted with 5-fold cross-validation grouped by match (GroupKFold), so every shot is recalibrated by a map that never saw its match. Brackets: 95% match-clustered bootstrap intervals (2,000 resamples of matches, seed 0) of the out-of-fold scores; Δ = recalibrated − raw (negative = recalibration helps), p = two-sided bootstrap p-value for Δ = 0. The bootstrap does not refit the maps, so the intervals cover evaluation sampling only; the last table shows the spread over random fold assignments.

StatsBomb xG and the logistic baseline are reference rows: they are already close to calibrated, so their change shows how much the procedure alone moves a good model with only 42 goals to learn from.

## Brier score

| Model | Raw | Platt | Δ Platt | p | Isotonic | Δ isotonic | p |
|---|---|---|---|---|---|---|---|
| Gemma 4 (26B) | 0.190 [0.169, 0.213] | 0.123 [0.089, 0.161] | −0.067 [−0.094, −0.038] | <0.001 | 0.121 [0.088, 0.158] | −0.069 [−0.095, −0.041] | <0.001 |
| Claude Sonnet 5.5 | 0.121 [0.085, 0.163] | 0.121 [0.090, 0.158] | +0.000 [−0.005, +0.006] | 0.925 | 0.121 [0.090, 0.156] | −0.000 [−0.009, +0.007] | 0.925 |
| Claude Opus 5.5 | 0.111 [0.079, 0.147] | 0.115 [0.085, 0.150] | +0.004 [+0.000, +0.008] | 0.031 | 0.108 [0.078, 0.143] | −0.003 [−0.011, +0.004] | 0.400 |
| Gemini 3.8 Flash | 0.116 [0.088, 0.147] | 0.114 [0.084, 0.148] | −0.001 [−0.009, +0.007] | 0.710 | 0.116 [0.086, 0.150] | +0.001 [−0.007, +0.009] | 0.847 |
| StatsBomb xG | 0.100 [0.069, 0.136] | 0.101 [0.073, 0.135] | +0.002 [−0.002, +0.005] | 0.407 | 0.103 [0.075, 0.137] | +0.004 [−0.002, +0.009] | 0.205 |
| Logistic baseline | 0.106 [0.073, 0.143] | 0.108 [0.078, 0.143] | +0.002 [−0.002, +0.005] | 0.303 | 0.110 [0.080, 0.146] | +0.005 [−0.002, +0.012] | 0.192 |

## Log loss

| Model | Raw | Platt | Δ Platt | p | Isotonic | Δ isotonic | p |
|---|---|---|---|---|---|---|---|
| Gemma 4 (26B) | 0.572 [0.522, 0.629] | 0.417 [0.326, 0.519] | −0.155 [−0.222, −0.076] | <0.001 | 0.422 [0.325, 0.533] | −0.150 [−0.222, −0.070] | <0.001 |
| Claude Sonnet 5.5 | 0.408 [0.301, 0.532] | 0.401 [0.320, 0.498] | −0.007 [−0.036, +0.020] | 0.628 | 0.396 [0.312, 0.490] | −0.012 [−0.049, +0.020] | 0.492 |
| Claude Opus 5.5 | 0.378 [0.283, 0.487] | 0.388 [0.305, 0.485] | +0.011 [−0.008, +0.027] | 0.241 | 0.369 [0.282, 0.471] | −0.008 [−0.036, +0.017] | 0.539 |
| Gemini 3.8 Flash | 0.380 [0.304, 0.465] | 0.381 [0.303, 0.471] | +0.002 [−0.016, +0.021] | 0.870 | 0.398 [0.311, 0.497] | +0.018 [−0.015, +0.064] | 0.376 |
| StatsBomb xG | 0.340 [0.249, 0.447] | 0.341 [0.263, 0.432] | +0.001 [−0.017, +0.017] | 0.909 | 0.354 [0.261, 0.459] | +0.014 [−0.017, +0.046] | 0.407 |
| Logistic baseline | 0.351 [0.258, 0.462] | 0.359 [0.275, 0.456] | +0.008 [−0.007, +0.021] | 0.320 | 0.396 [0.282, 0.532] | +0.045 [−0.005, +0.110] | 0.084 |

## What the Platt map does (fitted on all common shots, for interpretation only)

Slope < 1: the model's probabilities are too spread out (over-confident ranking of extremes); slope > 1: too compressed. The mean predicted xG columns show how recalibration moves the overall level (observed rate 0.141).

| Model | Platt intercept | Platt slope | Mean xG raw | Mean xG Platt (OOF) | Mean xG isotonic (OOF) |
|---|---|---|---|---|---|
| Gemma 4 (26B) | −1.62 | 0.42 | 0.379 | 0.141 | 0.141 |
| Claude Sonnet 5.5 | −0.10 | 0.68 | 0.085 | 0.141 | 0.140 |
| Claude Opus 5.5 | −0.09 | 0.78 | 0.109 | 0.141 | 0.140 |
| Gemini 3.8 Flash | −0.66 | 0.67 | 0.167 | 0.141 | 0.142 |
| StatsBomb xG | +0.37 | 0.94 | 0.100 | 0.141 | 0.143 |
| Logistic baseline | +0.24 | 0.94 | 0.108 | 0.142 | 0.142 |

## Sensitivity to the fold assignment (50 random grouped 5-fold splits)

Brier after recalibration: value with the fixed GroupKFold split used above, then the mean and the 2.5–97.5% range over 50 random grouped splits (match-level shuffles).

| Model | Raw | Platt (fixed split) | Platt over splits | Isotonic (fixed split) | Isotonic over splits |
|---|---|---|---|---|---|
| Gemma 4 (26B) | 0.1901 | 0.1227 | 0.1219 [0.1200, 0.1246] | 0.1210 | 0.1216 [0.1193, 0.1247] |
| Claude Sonnet 5.5 | 0.1210 | 0.1213 | 0.1197 [0.1178, 0.1230] | 0.1206 | 0.1200 [0.1172, 0.1241] |
| Claude Opus 5.5 | 0.1110 | 0.1151 | 0.1135 [0.1116, 0.1163] | 0.1078 | 0.1073 [0.1033, 0.1117] |
| Gemini 3.8 Flash | 0.1157 | 0.1144 | 0.1142 [0.1122, 0.1173] | 0.1165 | 0.1168 [0.1133, 0.1210] |
| StatsBomb xG | 0.0995 | 0.1012 | 0.1004 [0.0986, 0.1027] | 0.1033 | 0.1033 [0.0990, 0.1100] |
| Logistic baseline | 0.1057 | 0.1076 | 0.1072 [0.1047, 0.1114] | 0.1104 | 0.1090 [0.1056, 0.1168] |
