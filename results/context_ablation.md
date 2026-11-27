# Body-part context ablation

Evaluation set: the 298 common shots of compare_models.py (42 goals, 63 matches). The context runs add one prompt line, "Body part used: <part>."; everything else (image, prompt, settings) is unchanged. Brackets: 95% match-clustered bootstrap intervals (2,000 resamples of matches, seed 0); Δ = with context − without; p = two-sided bootstrap p-value. Foot = left or right foot.

Claude Opus 5.5's no-context run is the original zero-shot run, whose images were rendered in the authors' environment; its context run and its no-context rerun (`claude_opus_5_5_rerun`, the same-environment zero-shot) used images rendered in this environment with the same code. The rerun rows below show how much the no-context answers move between those two runs. Gemini 4 Carbon's runs were all rendered here.

## Claude Opus 5.5

Runs: `claude_opus_5_5` (no context) vs `claude_opus_5_5_context` (body part given); control rerun without context: `claude_opus_5_5_rerun`.

### Mean predicted xG by body part

| Shots | n (goals) | Observed rate | StatsBomb | No context | Body part given | Δ mean xG |
|---|---|---|---|---|---|---|
| Head | 63 (7) | 0.111 [0.019, 0.222] | 0.098 | 0.211 [0.184, 0.239] | 0.141 [0.117, 0.169] | −0.070 [−0.083, −0.058] |
| Foot | 232 (35) | 0.151 [0.101, 0.206] | 0.097 | 0.079 [0.068, 0.090] | 0.082 [0.071, 0.094] | +0.003 [+0.001, +0.006] |
| Other | 3 (0) | 0.000 [0.000, 0.000] | 0.379 | 0.320 [0.200, 0.380] | 0.330 [0.300, 0.360] | +0.010 [−0.050, +0.100] |
| All shots | 298 (42) | 0.141 [0.095, 0.193] | 0.100 | 0.109 [0.097, 0.121] | 0.097 [0.087, 0.108] | −0.012 [−0.017, −0.007] |

### Brier score by body part

| Shots | n (goals) | No context | Body part given | Δ Brier | p |
|---|---|---|---|---|---|
| Head | 63 (7) | 0.093 [0.056, 0.136] | 0.086 [0.036, 0.145] | −0.007 [−0.023, +0.012] | 0.447 |
| Foot | 232 (35) | 0.116 [0.078, 0.158] | 0.116 [0.078, 0.158] | +0.000 [−0.002, +0.002] | 0.720 |
| Other | 3 (0) | 0.110 [0.040, 0.144] | 0.110 [0.090, 0.130] | −0.000 [−0.035, +0.050] | 0.874 |
| All shots | 298 (42) | 0.111 [0.079, 0.147] | 0.110 [0.076, 0.147] | −0.001 [−0.005, +0.003] | 0.553 |

Overall: AUC 0.721 [0.624, 0.808] without context vs 0.729 [0.637, 0.813] with; log loss 0.378 vs 0.372.

### Control: the same no-context setting queried again

| Shots | No context (original) | No context (rerun) | Δ mean xG | Δ Brier | p |
|---|---|---|---|---|---|
| Head | 0.211 | 0.214 | +0.003 [−0.009, +0.014] | +0.005 [−0.005, +0.014] | 0.307 |
| Foot | 0.079 | 0.079 | +0.000 [−0.003, +0.003] | +0.001 [−0.002, +0.005] | 0.420 |
| Other | 0.320 | 0.310 | −0.010 [−0.050, +0.020] | −0.001 [−0.018, +0.016] | 0.909 |
| All shots | 0.109 | 0.110 | +0.001 [−0.003, +0.004] | +0.002 [−0.001, +0.006] | 0.259 |

Overall AUC of the rerun: 0.717 [0.630, 0.800].

## Gemini 4 Carbon

Runs: `gemini_4_carbon` (no context) vs `gemini_4_carbon_context` (body part given); control rerun without context: `gemini_4_carbon_rerun`.

### Mean predicted xG by body part

| Shots | n (goals) | Observed rate | StatsBomb | No context | Body part given | Δ mean xG |
|---|---|---|---|---|---|---|
| Head | 63 (7) | 0.111 [0.019, 0.222] | 0.098 | 0.185 [0.154, 0.219] | 0.117 [0.097, 0.139] | −0.068 [−0.082, −0.054] |
| Foot | 232 (35) | 0.151 [0.101, 0.206] | 0.097 | 0.087 [0.073, 0.103] | 0.088 [0.073, 0.105] | +0.001 [−0.002, +0.005] |
| Other | 3 (0) | 0.000 [0.000, 0.000] | 0.379 | 0.403 [0.110, 0.550] | 0.363 [0.120, 0.620] | −0.040 [−0.200, +0.070] |
| All shots | 298 (42) | 0.141 [0.095, 0.193] | 0.100 | 0.111 [0.097, 0.126] | 0.097 [0.084, 0.112] | −0.014 [−0.020, −0.008] |

### Brier score by body part

| Shots | n (goals) | No context | Body part given | Δ Brier | p |
|---|---|---|---|---|---|
| Head | 63 (7) | 0.087 [0.045, 0.133] | 0.085 [0.029, 0.153] | −0.002 [−0.022, +0.022] | 0.835 |
| Foot | 232 (35) | 0.109 [0.072, 0.149] | 0.108 [0.072, 0.147] | −0.001 [−0.004, +0.001] | 0.373 |
| Other | 3 (0) | 0.206 [0.012, 0.303] | 0.174 [0.014, 0.384] | −0.032 [−0.180, +0.082] | 0.731 |
| All shots | 298 (42) | 0.105 [0.074, 0.141] | 0.104 [0.072, 0.141] | −0.002 [−0.006, +0.004] | 0.500 |

Overall: AUC 0.743 [0.657, 0.819] without context vs 0.758 [0.672, 0.836] with; log loss 0.356 vs 0.352.

### Control: the same no-context setting queried again

| Shots | No context (original) | No context (rerun) | Δ mean xG | Δ Brier | p |
|---|---|---|---|---|---|
| Head | 0.185 | 0.181 | −0.004 [−0.009, +0.002] | −0.002 [−0.006, +0.001] | 0.145 |
| Foot | 0.087 | 0.086 | −0.001 [−0.004, +0.001] | +0.002 [−0.001, +0.004] | 0.149 |
| Other | 0.403 | 0.387 | −0.017 [−0.030, +0.000] | −0.012 [−0.032, +0.000] | 0.165 |
| All shots | 0.111 | 0.109 | −0.002 [−0.004, −0.000] | +0.001 [−0.001, +0.003] | 0.485 |

Overall AUC of the rerun: 0.738 [0.657, 0.817].

