# Contamination probes: do the models know these shots from memory?

Sample: 84 shots (`results/probe_sample.json`): all 42 goals among the 298 common shots and 42 non-goals drawn with seed 2026. Settings as in the xG runs (Gemini and GPT: thinking/reasoning medium; Opus: no thinking parameter, max_tokens 1500). Brackets: 95% match-clustered bootstrap intervals.

## Probe A: can the model tell the match from the rendered image?

Same image and legend lines as the zero-shot xG prompt, then: is this a shot you recognise, and your best guess of competition, year, teams and shooter. Hit rates count a best guess as right when it names the 2022 World Cup / the year 2022 / one or both of the two teams / the shooting team / the shooter's surname. Chance = the same guesses scored against the facts of randomly permuted shots (2,000 permutations).

| Model | n | Claims to recognise | Mean confidence | World Cup | 2022 | Either team | Both teams | Shooting team | Shooter |
|---|---|---|---|---|---|---|---|---|---|
| Claude Opus 5.5 | 84 | 0 | 0.01 | 0% (chance 0%) | 0% (chance 0%) | 0% (chance 0%) | 0% (chance 0%) | 0% (chance 0%) | 0% (chance 0%) |
| Gemini 4 Carbon | 84 | 3 | 0.07 | 30% (chance 30%) | 18% (chance 18%) | 7% (chance 5%) | 2% (chance 0%) | 5% (chance 2%) | 5% (chance 1%) |
| GPT-6.2 Astra | 84 | 0 | 0.01 | 57% (chance 57%) | 18% (chance 18%) | 11% (chance 9%) | 1% (chance 1%) | 6% (chance 4%) | 1% (chance 1%) |

- Claude Opus 5.5: most frequent competition guesses unknown (84); most frequent teams none
- Gemini 4 Carbon: most frequent competition guesses FIFA World Cup (25), UEFA Champions League (16), La Liga (16); most frequent teams barcelona (18), manchester city (11), france (10), liverpool (9)
- GPT-6.2 Astra: most frequent competition guesses FIFA World Cup (48), La Liga (15), UEFA Champions League (12); most frequent teams france (25), argentina (23), barcelona (18), england (10)
- Gemini 4 Carbon claims to recognise 3 shot(s): Liverpool vs AC Milan, Xabi Alonso (UEFA Champions League 2005; confidence 0.75): wrong, it was Enner Remberto Valencia Lastra for Ecuador; France vs Argentina, Randal Kolo Muani (FIFA World Cup 2022; confidence 0.95): wrong, it was João Félix Sequeira for Portugal; Manchester City vs Queens Park Rangers, Sergio Agüero (Premier League 2012; confidence 0.95): wrong, it was Mehdi Taremi for Iran
- Gemini 4 Carbon names the exact match and shooter of `70e7f361` (Lionel Messi for Argentina vs Mexico, goal) with confidence 0.15, without claiming to recognise it; its image-only xG for this shot was 0.03 (StatsBomb 0.034).
- Gemini 4 Carbon names the exact match and shooter of `d71bc1d5` (Takuma Asano for Japan vs Germany, goal) with confidence 0.12, without claiming to recognise it; its image-only xG for this shot was 0.04 (StatsBomb 0.027).
- Dropping the 2 exactly identified shot(s) from the 298 common shots changes ΔBrier vs StatsBomb by +0.0005 (Claude Opus 5.5), +0.0001 (Gemini 4 Carbon), +0.0004 (GPT-6.2 Astra).

## Probe B: given the match, the clock and the shooter (text only), does the model remember the shot?

No image and no positions. Sensitivity = share of goals answered "scored: yes"; specificity = share of non-goals answered "no"; "unknown" counts as neither. AUC of the model's goal probability against the actual outcome, next to StatsBomb xG's AUC on the same shots (its knowledge of the shot before the strike) and the model's own image-only xG from its zero-shot run.

| Model | n | Remembers (goals / non-goals) | Sensitivity | Specificity | Unknown | AUC of p(goal) | StatsBomb xG AUC | Image-only xG AUC | ΔAUC p(goal) − StatsBomb | p |
|---|---|---|---|---|---|---|---|---|---|---|
| Claude Opus 5.5 | 84 | 100% / 10% | 100% | 100% | 0% | 1.000 [1.000, 1.000] | 0.790 [0.678, 0.879] | 0.696 [0.566, 0.806] | +0.210 [+0.121, +0.322] | <0.001 |
| Gemini 4 Carbon | 84 | 100% / 57% | 100% | 100% | 0% | 1.000 [1.000, 1.000] | 0.790 [0.678, 0.879] | 0.705 [0.579, 0.812] | +0.210 [+0.121, +0.322] | <0.001 |
| GPT-6.2 Astra | 84 | 100% / 26% | 100% | 100% | 0% | 1.000 [1.000, 1.000] | 0.790 [0.678, 0.879] | 0.705 [0.584, 0.811] | +0.210 [+0.121, +0.322] | <0.001 |

The xG the model gives when it is told which shot it is ("the quality of the chance, regardless of the outcome"), against StatsBomb xG and the model's image-only xG on the same shots:

| Model | AUC of recalled xG | ΔAUC vs StatsBomb xG | p | Mean recalled xG, goals | non-goals | StatsBomb, goals | non-goals | Image-only, goals | non-goals |
|---|---|---|---|---|---|---|---|---|---|
| Claude Opus 5.5 | 0.898 [0.810, 0.965] | +0.108 [+0.034, +0.193] | 0.002 | 0.278 | 0.072 | 0.250 | 0.077 | 0.186 | 0.103 |
| Gemini 4 Carbon | 0.779 [0.667, 0.869] | −0.010 [−0.077, +0.055] | 0.757 | 0.271 | 0.098 | 0.250 | 0.077 | 0.221 | 0.103 |
| GPT-6.2 Astra | 0.834 [0.742, 0.914] | +0.044 [−0.013, +0.117] | 0.166 | 0.283 | 0.089 | 0.250 | 0.077 | 0.271 | 0.140 |

## Reading

- Probe A: the rendered maps do not reliably give the match away (Claude Opus 5.5: 0 of 84 claimed (0 right), a correct team in 0% of best guesses (chance 0%); Gemini 4 Carbon: 3 of 84 claimed (0 right), a correct team in 7% of best guesses (chance 5%); GPT-6.2 Astra: 0 of 84 claimed (0 right), a correct team in 11% of best guesses (chance 9%)). Every claimed recognition is wrong, and best guesses of the teams are about as good as chance. The exceptions: Lionel Messi for Argentina vs Mexico (Gemini 4 Carbon, confidence 0.15); Takuma Asano for Japan vs Germany (Gemini 4 Carbon, confidence 0.12), where the best guess names the exact match and shooter although the model did not claim to recognise the shot. The model's image-only xG for them vs StatsBomb: 0.03 vs 0.034; 0.04 vs 0.027; dropping them moves every model's ΔBrier vs StatsBomb by at most 0.0005. The image-only results do not rest on remembered outcomes.
- Probe B: told only the match, the minute and the shooter, the models know whether the shot went in (Claude Opus 5.5: sensitivity 100%, specificity 100%, AUC 1.000 vs StatsBomb xG 0.790; Gemini 4 Carbon: sensitivity 100%, specificity 100%, AUC 1.000 vs StatsBomb xG 0.790; GPT-6.2 Astra: sensitivity 100%, specificity 100%, AUC 1.000 vs StatsBomb xG 0.790). They claim to remember most goals but few misses (Claude Opus 5.5 100% of goals, 10% of misses; Gemini 4 Carbon 100% of goals, 57% of misses; GPT-6.2 Astra 100% of goals, 26% of misses) and still label the misses correctly: knowing a match's goals and when they were scored is enough to mark every other shot as a miss.
- Consequence for real broadcast frames: wherever a frame lets a model identify the match and the moment (score bar with teams and clock, kits, faces, stadium, commentary graphics), a high score can be recall rather than estimation. Cropping the score bar removes the clock and the team names but not the kits or the players, so a frame benchmark needs its own identification probe.
- Recalled xG: asked for the quality of the chance "regardless of the outcome", the xG the models give once they know which shot it is has AUC 0.898 (Claude Opus 5.5), 0.779 (Gemini 4 Carbon), 0.834 (GPT-6.2 Astra) against StatsBomb's 0.790 on the same shots; significantly higher for Claude Opus 5.5. StatsBomb's model has detailed pre-shot information; separating goals from misses better than it does suggests that the remembered outcome leaks into the estimate.
