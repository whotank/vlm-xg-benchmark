# Recalibration (side analysis)

Evaluation set: the 298 common shots of compare_models.py (42 goals, 63 matches). Main results stay raw (zero-shot); this table shows what a monotone recalibration learned from labelled shots would change.

Method: Platt scaling (unpenalised logistic regression of goal on logit p) and isotonic regression, each fitted with 5-fold cross-validation grouped by match (the baseline's frozen match folds, results/baseline_folds.json), so every shot is recalibrated by a map that never saw its match. Brackets: 95% match-clustered bootstrap intervals (2,000 resamples of matches, seed 0) of the out-of-fold scores; Δ = recalibrated − raw (negative = recalibration helps), p = two-sided bootstrap p-value for Δ = 0. The bootstrap does not refit the maps, so the intervals cover evaluation sampling only; the last table shows the spread over random fold assignments.

StatsBomb xG and the logistic baseline are reference rows: they are already close to calibrated, so their change shows how much the procedure alone moves a good model with only 42 goals to learn from.

## Brier score

| Model | Raw | Platt | Δ Platt | p | Isotonic | Δ isotonic | p |
|---|---|---|---|---|---|---|---|
| Gemma 4 (26B) | 0.190 [0.169, 0.213] | 0.121 [0.087, 0.158] | −0.069 [−0.096, −0.040] | <0.001 | 0.121 [0.087, 0.158] | −0.070 [−0.095, −0.042] | <0.001 |
| Claude Sonnet 5.5 | 0.121 [0.085, 0.163] | 0.119 [0.087, 0.155] | −0.002 [−0.008, +0.003] | 0.408 | 0.119 [0.089, 0.154] | −0.002 [−0.010, +0.006] | 0.721 |
| Claude Opus 5.5 | 0.111 [0.079, 0.147] | 0.113 [0.082, 0.147] | +0.002 [−0.001, +0.005] | 0.311 | 0.104 [0.075, 0.137] | −0.007 [−0.015, +0.000] | 0.065 |
| Gemini 3.8 Flash | 0.116 [0.088, 0.147] | 0.113 [0.083, 0.147] | −0.003 [−0.010, +0.005] | 0.480 | 0.113 [0.083, 0.148] | −0.002 [−0.010, +0.006] | 0.580 |
| Claude Opus 5.5, k=1 | 0.110 [0.079, 0.145] | 0.112 [0.082, 0.147] | +0.002 [−0.000, +0.005] | 0.097 | 0.109 [0.077, 0.146] | −0.001 [−0.012, +0.009] | 0.844 |
| StatsBomb xG | 0.100 [0.069, 0.136] | 0.101 [0.073, 0.134] | +0.001 [−0.003, +0.006] | 0.545 | 0.105 [0.077, 0.137] | +0.006 [−0.002, +0.012] | 0.135 |
| Logistic baseline | 0.106 [0.073, 0.143] | 0.109 [0.079, 0.144] | +0.004 [−0.001, +0.008] | 0.103 | 0.109 [0.078, 0.143] | +0.003 [−0.004, +0.010] | 0.388 |

## Log loss

| Model | Raw | Platt | Δ Platt | p | Isotonic | Δ isotonic | p |
|---|---|---|---|---|---|---|---|
| Gemma 4 (26B) | 0.572 [0.522, 0.629] | 0.408 [0.319, 0.505] | −0.164 [−0.232, −0.090] | <0.001 | 0.417 [0.322, 0.523] | −0.154 [−0.223, −0.076] | <0.001 |
| Claude Sonnet 5.5 | 0.408 [0.301, 0.532] | 0.392 [0.311, 0.486] | −0.016 [−0.046, +0.011] | 0.260 | 0.389 [0.309, 0.479] | −0.019 [−0.058, +0.014] | 0.289 |
| Claude Opus 5.5 | 0.378 [0.283, 0.487] | 0.376 [0.295, 0.470] | −0.001 [−0.019, +0.015] | 0.893 | 0.356 [0.272, 0.450] | −0.022 [−0.052, +0.004] | 0.095 |
| Gemini 3.8 Flash | 0.380 [0.304, 0.465] | 0.375 [0.296, 0.467] | −0.005 [−0.021, +0.014] | 0.580 | 0.373 [0.292, 0.465] | −0.007 [−0.026, +0.013] | 0.528 |
| Claude Opus 5.5, k=1 | 0.376 [0.282, 0.484] | 0.377 [0.294, 0.472] | +0.001 [−0.013, +0.015] | 0.914 | 0.369 [0.281, 0.470] | −0.007 [−0.036, +0.019] | 0.652 |
| StatsBomb xG | 0.340 [0.249, 0.447] | 0.339 [0.259, 0.431] | −0.001 [−0.021, +0.016] | 0.908 | 0.366 [0.272, 0.474] | +0.026 [−0.012, +0.068] | 0.212 |
| Logistic baseline | 0.351 [0.258, 0.462] | 0.362 [0.276, 0.461] | +0.011 [−0.005, +0.026] | 0.174 | 0.361 [0.266, 0.473] | +0.009 [−0.020, +0.045] | 0.599 |

## What the Platt map does (fitted on all common shots, for interpretation only)

Slope < 1: the model's probabilities are too spread out (over-confident ranking of extremes); slope > 1: too compressed. The mean predicted xG columns show how recalibration moves the overall level (observed rate 0.141).

| Model | Platt intercept | Platt slope | Mean xG raw | Mean xG Platt (OOF) | Mean xG isotonic (OOF) |
|---|---|---|---|---|---|
| Gemma 4 (26B) | −1.62 | 0.42 | 0.379 | 0.141 | 0.142 |
| Claude Sonnet 5.5 | −0.10 | 0.68 | 0.085 | 0.141 | 0.142 |
| Claude Opus 5.5 | −0.09 | 0.78 | 0.109 | 0.141 | 0.141 |
| Gemini 3.8 Flash | −0.66 | 0.67 | 0.167 | 0.141 | 0.137 |
| Claude Opus 5.5, k=1 | −0.16 | 0.77 | 0.117 | 0.141 | 0.136 |
| StatsBomb xG | +0.37 | 0.94 | 0.100 | 0.142 | 0.140 |
| Logistic baseline | +0.24 | 0.94 | 0.108 | 0.144 | 0.143 |

## Sensitivity to the fold assignment (50 random grouped 5-fold splits)

Brier after recalibration: value with the fixed match folds used above, then the mean and the 2.5–97.5% range over 50 random grouped splits (seeded permutations of matches).

| Model | Raw | Platt (fixed split) | Platt over splits | Isotonic (fixed split) | Isotonic over splits |
|---|---|---|---|---|---|
| Gemma 4 (26B) | 0.1901 | 0.1207 | 0.1216 [0.1196, 0.1247] | 0.1206 | 0.1209 [0.1185, 0.1240] |
| Claude Sonnet 5.5 | 0.1210 | 0.1188 | 0.1196 [0.1178, 0.1223] | 0.1194 | 0.1199 [0.1170, 0.1225] |
| Claude Opus 5.5 | 0.1110 | 0.1127 | 0.1134 [0.1114, 0.1163] | 0.1041 | 0.1076 [0.1043, 0.1138] |
| Gemini 3.8 Flash | 0.1157 | 0.1129 | 0.1137 [0.1121, 0.1169] | 0.1135 | 0.1166 [0.1136, 0.1216] |
| Claude Opus 5.5, k=1 | 0.1103 | 0.1124 | 0.1129 [0.1112, 0.1156] | 0.1091 | 0.1078 [0.1029, 0.1130] |
| StatsBomb xG | 0.0995 | 0.1010 | 0.0998 [0.0987, 0.1015] | 0.1050 | 0.1026 [0.0989, 0.1064] |
| Logistic baseline | 0.1057 | 0.1095 | 0.1066 [0.1051, 0.1087] | 0.1086 | 0.1081 [0.1047, 0.1114] |
