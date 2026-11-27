# Error analysis: which shots do the VLMs get wrong?

Evaluation set: the 298 common shots (42 goals). Frontier = Gemini 4 Carbon, Gemini 4 Neon, GPT-6.2 Astra, Claude Opus 5.5, Gemini 4 Hydrogen (zero-shot, image only). Gap = frontier mean xG − StatsBomb xG. Excess Brier = (frontier mean − goal)² − (StatsBomb − goal)²; its mean is the Brier gap of the averaged frontier prediction to StatsBomb: +0.0078 [-0.0008, +0.0156]. A group's contribution is the sum of its shots' excess divided by all 298 shots (contributions add up to that total); its share is the contribution divided by the total. The share is a ratio of two noisy sums, so its interval is wide: read the mean gaps and contributions first. Brackets: 95% match-clustered bootstrap intervals.

## Distance band (yards)

| Group | n (goals) | StatsBomb | Frontier VLMs | Gap [95% CI] | Contribution to ΔBrier [95% CI] | Share of ΔBrier [95% CI] |
|---|---|---|---|---|---|---|
| <6 | 8 (4) | 0.480 | 0.429 | −0.051 [−0.176, +0.070] | +0.0012 [−0.0018, +0.0043] | +15% [−75%, +69%] |
| 6–12 | 77 (14) | 0.168 | 0.227 | +0.059 [+0.030, +0.085] | +0.0042 [−0.0017, +0.0094] | +54% [−27%, +115%] |
| 12–18 | 79 (18) | 0.103 | 0.112 | +0.009 [−0.008, +0.026] | +0.0028 [+0.0004, +0.0058] | +37% [−49%, +187%] |
| 18–25 | 69 (4) | 0.049 | 0.045 | −0.004 [−0.011, +0.003] | −0.0005 [−0.0013, +0.0002] | −6% [−56%, +17%] |
| 25+ | 65 (2) | 0.024 | 0.027 | +0.004 [+0.002, +0.006] | +0.0001 [+0.0000, +0.0002] | +1% [−2%, +5%] |

## Distance band and body part

| Group | n (goals) | StatsBomb | Frontier VLMs | Gap [95% CI] | Contribution to ΔBrier [95% CI] | Share of ΔBrier [95% CI] |
|---|---|---|---|---|---|---|
| 12–18, foot/other | 66 (18) | 0.115 | 0.107 | −0.008 [−0.022, +0.005] | +0.0019 [−0.0005, +0.0048] | +25% [−32%, +119%] |
| 12–18, header | 13 (0) | 0.042 | 0.137 | +0.095 [+0.065, +0.129] | +0.0009 [+0.0003, +0.0018] | +12% [−25%, +73%] |
| 18–25, foot/other | 68 (4) | 0.050 | 0.045 | −0.005 [−0.013, +0.002] | −0.0005 [−0.0013, +0.0002] | −6% [−57%, +17%] |
| 18–25, header | 1 (0) | 0.026 | 0.064 | +0.038 [+0.038, +0.038] | +0.0000 [+0.0000, +0.0000] | +0% [+0%, +1%] |
| 25+, foot/other | 65 (2) | 0.024 | 0.027 | +0.004 [+0.002, +0.006] | +0.0001 [+0.0000, +0.0002] | +1% [−2%, +5%] |
| 6–12, foot/other | 32 (9) | 0.271 | 0.231 | −0.040 [−0.092, +0.004] | +0.0024 [−0.0011, +0.0062] | +31% [−42%, +174%] |
| 6–12, header | 45 (5) | 0.095 | 0.224 | +0.129 [+0.119, +0.139] | +0.0018 [−0.0031, +0.0058] | +23% [−132%, +105%] |
| <6, foot/other | 4 (2) | 0.637 | 0.466 | −0.172 [−0.307, −0.055] | +0.0011 [−0.0007, +0.0038] | +14% [−30%, +66%] |
| <6, header | 4 (2) | 0.322 | 0.392 | +0.070 [−0.143, +0.160] | +0.0000 [−0.0022, +0.0017] | +0% [−93%, +60%] |

## Shot kind

| Group | n (goals) | StatsBomb | Frontier VLMs | Gap [95% CI] | Contribution to ΔBrier [95% CI] | Share of ΔBrier [95% CI] |
|---|---|---|---|---|---|---|
| Foot, open play | 165 (25) | 0.109 | 0.094 | −0.015 [−0.025, −0.006] | +0.0048 [+0.0005, +0.0096] | +62% [−75%, +284%] |
| Foot, set piece | 70 (10) | 0.081 | 0.081 | −0.001 [−0.013, +0.012] | +0.0002 [−0.0020, +0.0030] | +3% [−67%, +57%] |
| Header | 63 (7) | 0.098 | 0.214 | +0.117 [+0.103, +0.130] | +0.0028 [−0.0041, +0.0079] | +36% [−162%, +152%] |

## Keeper distance to the shooter

| Group | n (goals) | StatsBomb | Frontier VLMs | Gap [95% CI] | Contribution to ΔBrier [95% CI] | Share of ΔBrier [95% CI] |
|---|---|---|---|---|---|---|
| < 4 yd | 14 (4) | 0.325 | 0.285 | −0.041 [−0.101, +0.024] | +0.0015 [−0.0004, +0.0038] | +20% [−29%, +99%] |
| 4–8 yd | 48 (16) | 0.250 | 0.266 | +0.016 [−0.027, +0.059] | +0.0042 [−0.0036, +0.0110] | +54% [−114%, +172%] |
| 8–12 yd | 72 (8) | 0.087 | 0.149 | +0.062 [+0.044, +0.080] | +0.0020 [−0.0008, +0.0046] | +26% [−48%, +152%] |
| 12+ yd | 164 (14) | 0.043 | 0.044 | +0.001 [−0.003, +0.004] | +0.0000 [−0.0009, +0.0010] | +0% [−28%, +26%] |

## Opponents in the shooting cone

| Group | n (goals) | StatsBomb | Frontier VLMs | Gap [95% CI] | Contribution to ΔBrier [95% CI] | Share of ΔBrier [95% CI] |
|---|---|---|---|---|---|---|
| 0 | 8 (5) | 0.489 | 0.321 | −0.168 [−0.277, −0.060] | +0.0034 [+0.0008, +0.0069] | +43% [−65%, +224%] |
| 1 | 136 (26) | 0.134 | 0.157 | +0.023 [+0.004, +0.043] | +0.0041 [−0.0024, +0.0102] | +53% [−48%, +134%] |
| 2 | 104 (8) | 0.056 | 0.079 | +0.023 [+0.013, +0.034] | +0.0000 [−0.0024, +0.0021] | +0% [−109%, +68%] |
| 3+ | 50 (3) | 0.039 | 0.050 | +0.011 [+0.004, +0.018] | +0.0003 [+0.0000, +0.0006] | +3% [−6%, +19%] |

## What drives the gap (OLS on standardised features)

Gap regressed on 9 shot features, each scaled to one standard deviation; R² = 0.53. Positive = the VLMs rate such shots higher than StatsBomb does, holding the other features fixed. CIs from the match-clustered bootstrap.

| Feature (+1 SD) | Effect on gap [95% CI] |
|---|---|
| header | +0.052 [+0.040, +0.063] |
| keeper off his line (yd) | −0.019 [−0.028, −0.011] |
| distance (yd) | +0.016 [+0.006, +0.027] |
| opponents within 2 yd | +0.014 [+0.006, +0.023] |
| opponents in the shooting cone | +0.010 [+0.002, +0.019] |
| from a corner or free kick | −0.003 [−0.011, +0.004] |
| one-on-one | −0.002 [−0.014, +0.012] |
| under pressure | −0.001 [−0.011, +0.008] |
| first-time shot | +0.001 [−0.006, +0.008] |

## Shared or model-specific?

Correlation of the five models' gaps to StatsBomb, shot by shot: median 0.82 (range 0.73–0.93). The averaged gap carries 85% of a single model's gap variance, so most disagreement with StatsBomb is shared across models rather than idiosyncratic.

| | G4 Carbon | G4 Neon | GPT-6.2 Astra | Opus 5.5 | G4 Hydrogen |
|---|---|---|---|---|---|
| Gemini 4 Carbon | 1.00 | 0.88 | 0.76 | 0.81 | 0.84 |
| Gemini 4 Neon | 0.88 | 1.00 | 0.73 | 0.90 | 0.93 |
| GPT-6.2 Astra | 0.76 | 0.73 | 1.00 | 0.79 | 0.77 |
| Claude Opus 5.5 | 0.81 | 0.90 | 0.79 | 1.00 | 0.92 |
| Gemini 4 Hydrogen | 0.84 | 0.93 | 0.77 | 0.92 | 1.00 |

All five above StatsBomb on 140 shots, all five below on 50, split on 108.

## Largest consensus disagreements (results/figure_error_gallery.png)

| | Shot | Outcome | StatsBomb | Frontier mean (range) | Body part, distance, play | Gemini 4 Carbon's reason |
|---|---|---|---|---|---|---|
| over | `661531f6` | no goal | 0.21 | 0.45 (0.38–0.54) | Head, 8 yd, From Free Kick | The shot is taken from very close range at the edge of the six-yard box with no defenders blocking the path to goal, facing only the centrally positioned goalkeeper. |
| over | `519a5012` | no goal | 0.08 | 0.31 (0.25–0.38) | Head, 11 yd, From Free Kick | The shot is taken from a central position around nine yards out with a clear line to goal, though heavy pressure from an adjacent defender reduces the conversion probability. |
| over | `96265fa7` | goal | 0.23 | 0.45 (0.42–0.52) | Head, 8 yd, Regular Play | The shot is taken from point-blank, central range right at the edge of the six-yard box with no defenders blocking the path to goal, though tight pressure from behind slightly limits the finish. |
| over | `79a770c1` | no goal | 0.09 | 0.30 (0.24–0.34) | Head, 13 yd, From Throw In | The shooter is centrally located right next to the penalty spot with an unobstructed sight of goal against only the goalkeeper, despite close side pressure from a defender. |
| over | `78a794a1` | no goal | 0.05 | 0.25 (0.16–0.34) | Head, 10 yd, From Throw In | The shot is taken from a central position about ten yards out, but the shooter is under heavy, immediate physical pressure from multiple defenders with the goalkeeper well-positioned on the line. |
| over | `4c292e31` | no goal | 0.12 | 0.32 (0.28–0.42) | Head, 7 yd, From Corner | The shot is taken from point-blank central range on the six-yard line, though heavy defensive crowding from an apparent set piece reduces the conversion probability. |
| under | `4b8db9ec` | goal | 0.78 | 0.45 (0.30–0.62) | Right Foot, 10 yd, Regular Play | The shooter is centrally positioned close to goal with a largely open net due to the goalkeeper being pulled out of position to the left. |
| under | `be17203f` | goal | 0.90 | 0.57 (0.40–0.77) | Right Foot, 6 yd, Regular Play | The shot is taken centrally from the edge of the six-yard box with the goalkeeper pulled wide out of position, leaving most of the goal exposed despite close defensive pressure. |
| under | `70fe857d` | goal | 0.73 | 0.46 (0.38–0.59) | Right Foot, 6 yd, Regular Play | The shot is taken from very close range on the six-yard line with a clear path to goal and the goalkeeper positioned slightly toward the far post, though the crowded box suggests an aerial attempt or tight pressure. |
| under | `d53ccaf1` | goal | 0.79 | 0.54 (0.42–0.65) | Right Foot, 4 yd, From Counter | The shooter is centrally positioned inside the six-yard box with all outfield defenders behind them, though the goalkeeper is close enough to potentially smother the effort. |
| under | `eb01f465` | goal | 0.41 | 0.18 (0.15–0.22) | Left Foot, 13 yd, From Goal Kick | The shot is taken from close range slightly right of center, though a well-positioned goalkeeper off his line and nearby defenders significantly narrow the shooting angle. |
| under | `1d0a8857` | goal | 0.52 | 0.31 (0.25–0.38) | Right Foot, 10 yd, Regular Play | The shot is taken from very close range in a central area with only the rushing goalkeeper to beat, though a nearby defender applies close pressure. |

## Rationale phrases and the direction of the gap

Share of each model's rationales that use a phrase, and that model's mean gap on those shots (model xG − StatsBomb).

| Phrase | G4 Carbon | G4 Neon | GPT-6.2 Astra | Claude Opus 5.5 | G4 Hydrogen |
|---|---|---|---|---|---|
| point-blank | 3% (+0.058) | 1% (−0.112) | 0% (+0.111) | 4% (−0.058) | 1% (+0.023) |
| tight / narrow angle | 9% (+0.010) | 19% (−0.006) | 14% (+0.030) | 16% (+0.008) | 10% (+0.010) |
| crowded / congested | 21% (+0.010) | 32% (+0.006) | 14% (+0.067) | 59% (+0.018) | 32% (+0.014) |
| blocked / blocking | 52% (+0.012) | 43% (+0.003) | 16% (+0.011) | 50% (+0.012) | 31% (+0.001) |
| keeper well positioned / set | 12% (−0.020) | 39% (+0.005) | 37% (+0.063) | 44% (+0.006) | 38% (+0.016) |
| keeper off line / out of position | 2% (−0.151) | 1% (−0.079) | 0% (−0.127) | 0% (−0.237) | 1% (−0.142) |
| header / heading | 2% (+0.052) | 1% (+0.040) | 0% | 6% (+0.060) | 0% |
| open / unmarked / clear sight | 1% (−0.119) | 2% (+0.118) | 1% (+0.081) | 1% (−0.073) | 2% (+0.044) |
