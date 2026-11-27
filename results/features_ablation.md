# Information ablation: what the image does not show

Evaluation set: the 298 common shots of compare_models.py (42 goals). Brackets: 95% match-clustered bootstrap intervals; differences are paired on the same resamples (p = two-sided bootstrap p-value). The main benchmark is the image-only VLM; the other rows add information as text.

## Who sees what

| Model | Image pixels | Exact coordinates | Body part | Other pre-shot attributes | Trained on |
|---|---|---|---|---|---|
| StatsBomb xG | no | yes (event location, freeze frame) | yes | yes (technique, shot type, assist, pressure, ...) | StatsBomb's own large shot database |
| Logistic, all pre-shot features | no | yes | yes | yes (the same list the VLMs get with `--all-features`) | ~1,140 WC 2022 shots per fold |
| Logistic baseline (original) | no | yes | header flag | no | ~1,140 WC 2022 shots per fold |
| Map-only baseline | no | yes (distance, angle, defenders in the cone) | no | no | ~1,140 WC 2022 shots per fold |
| VLM, image only (main benchmark) | yes | no, read from the image | no | no | no football-specific training; zero-shot prompt |
| VLM + body part | yes | no | yes (text) | no | as above |
| VLM + all pre-shot features | yes | yes (text: distance, angle, defenders, keeper, pressure) | yes (text) | yes (text) | as above |

## Brier score (and AUC) by information and model

| Information | Logistic (exact numbers) | Claude Opus 5.5 | Gemini 4 Carbon | GPT-6.2 Astra |
|---|---|---|---|---|
| image geometry only | 0.110 [0.078, 0.147]<br>AUC 0.721 [0.636, 0.808] | 0.111 [0.079, 0.147]<br>AUC 0.721 [0.624, 0.808] | 0.105 [0.074, 0.141]<br>AUC 0.743 [0.657, 0.819] | 0.108 [0.080, 0.139]<br>AUC 0.733 [0.642, 0.816] |
| + body part | 0.106 [0.073, 0.143]<br>AUC 0.768 [0.694, 0.843] | 0.110 [0.076, 0.147]<br>AUC 0.729 [0.637, 0.813] | 0.104 [0.072, 0.141]<br>AUC 0.758 [0.672, 0.836] | not run |
| + all pre-shot attributes | 0.098 [0.068, 0.133]<br>AUC 0.815 [0.744, 0.881] | 0.104 [0.071, 0.142]<br>AUC 0.758 [0.672, 0.841] | 0.099 [0.068, 0.136]<br>AUC 0.785 [0.700, 0.860] | 0.099 [0.069, 0.133]<br>AUC 0.781 [0.698, 0.856] |
| StatsBomb xG (reference) | 0.100 [0.069, 0.136]<br>AUC 0.792 [0.704, 0.867] | | | |

## Does the VLM judge better than a logistic model with the same information?

ΔBrier = VLM − logistic in the same row (negative = the VLM is better).

| Information | Claude Opus 5.5 | Gemini 4 Carbon | GPT-6.2 Astra |
|---|---|---|---|
| image geometry only | +0.001 [−0.006, +0.008] (p 0.820) | −0.005 [−0.010, −0.001] (p 0.013) | −0.002 [−0.011, +0.006] (p 0.613) |
| + body part | +0.004 [−0.004, +0.013] (p 0.355) | −0.002 [−0.007, +0.002] (p 0.362) |  |
| + all pre-shot attributes | +0.006 [−0.001, +0.013] (p 0.116) | +0.001 [−0.004, +0.008] (p 0.707) | +0.001 [−0.005, +0.007] (p 0.882) |

## How much does the extra information help each model?

ΔBrier relative to the same model with the image (or image features) only; negative = better.

| Added information | Logistic | Claude Opus 5.5 | Gemini 4 Carbon | GPT-6.2 Astra |
|---|---|---|---|---|
| + body part | −0.005 [−0.009, +0.001] (p 0.088) | −0.001 [−0.005, +0.003] (p 0.553) | −0.002 [−0.006, +0.004] (p 0.500) |  |
| + all pre-shot attributes | −0.012 [−0.021, −0.004] (p 0.003) | −0.007 [−0.016, +0.002] (p 0.116) | −0.006 [−0.010, −0.001] (p 0.027) | −0.009 [−0.016, −0.002] (p 0.016) |

## Distance to StatsBomb

ΔBrier = model − StatsBomb xG (negative = better than StatsBomb).

| Information | Logistic | Claude Opus 5.5 | Gemini 4 Carbon | GPT-6.2 Astra |
|---|---|---|---|---|
| image geometry only | +0.011 [+0.003, +0.018] (p 0.003) | +0.012 [+0.001, +0.022] (p 0.022) | +0.006 [−0.002, +0.012] (p 0.114) | +0.009 [−0.002, +0.018] (p 0.109) |
| + body part | +0.006 [+0.001, +0.012] (p 0.020) | +0.010 [+0.002, +0.019] (p 0.012) | +0.004 [+0.000, +0.008] (p 0.045) |  |
| + all pre-shot attributes | −0.001 [−0.008, +0.004] (p 0.737) | +0.004 [−0.000, +0.009] (p 0.064) | −0.000 [−0.004, +0.004] (p 0.984) | −0.001 [−0.005, +0.003] (p 0.781) |

## Mean predicted xG where the information matters

| Model | headers (n=63, observed 0.11) | 6–12 yd (n=77, observed 0.18) | all shots (n=298, observed 0.14) |
|---|---|---|---|
| StatsBomb xG | 0.098 | 0.168 | 0.100 |
| Map-only baseline | 0.201 | 0.204 | 0.111 |
| Logistic baseline | 0.117 | 0.183 | 0.108 |
| Logistic, all pre-shot features | 0.103 | 0.181 | 0.112 |
| Claude Opus 5.5 | 0.211 | 0.221 | 0.109 |
| Gemini 4 Carbon | 0.185 | 0.199 | 0.111 |
| GPT-6.2 Astra | 0.283 | 0.298 | 0.148 |
| Claude Opus 5.5 + body part | 0.141 | 0.188 | 0.097 |
| Gemini 4 Carbon + body part | 0.117 | 0.164 | 0.097 |
| Claude Opus 5.5 + all pre-shot features | 0.110 | 0.150 | 0.087 |
| Gemini 4 Carbon + all pre-shot features | 0.113 | 0.168 | 0.098 |
| GPT-6.2 Astra + all pre-shot features | 0.154 | 0.209 | 0.117 |

Hybrid design (VLM reads the geometry from the image, the logistic model scores it) is the missing cell that separates perception from judgment; it is the next step in CLAUDE.md.
