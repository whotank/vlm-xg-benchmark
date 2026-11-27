# Draft text: hybrid perception test, contamination probes, broadcast-frame pilot

Draft paragraphs for the full paper. Every number comes from `results/hybrid.md` and
`results/contamination_probe.md`; rerun `hybrid.py` / `contamination_probe.py` if a run changes.

## Methods

**Perception vs judgment (hybrid design).** To separate reading the image from judging the chance, each model is
shown the same image as in the zero-shot runs and asked only to measure three quantities:

- the distance from the shooter to the centre of the goal, in yards
- the angle between the posts, in degrees
- the number of opponents inside the triangle formed by the shooter and the posts

The prompt states the pitch markings' real dimensions for scale and asks for no xG. The map-only logistic model then
scores these readings exactly as it scores the true values: the same five match-grouped fold models, each shot scored
by the model that never saw its match. The decomposition is:

- hybrid − map-only = the cost of misreading the image
- model's own xG − hybrid = its judgment, given what it can read

Opus 5.5 and GPT-6.2 Astra measured all 300 shots. Gemini 4 Carbon measured the first 150: it spends about 2,700
thinking tokens per measurement.

**Contamination probes.** Frontier models may have read match reports of the 2022 World Cup. We probe this on 84
shots: all 42 goals among the 298 evaluation shots and 42 randomly drawn misses.

- Probe A shows the zero-shot image and asks whether the model recognises the shot. It also asks for a best guess of
  competition, year, teams and shooter. Guesses are scored against the true facts and, for chance levels, against
  the facts of randomly permuted shots.
- Probe B shows no image and no positions. It gives only the match (stage, teams, date), the match clock, the
  half and the minute as reported, and the shooter. It asks:
  - whether the model remembers the shot
  - whether it was a goal
  - the probability that it was a goal
  - its xG "regardless of the outcome"

## Results

**The models read the drawing almost exactly.** Asked to measure, Opus and GPT-6.2 Astra read the distance with mean
absolute errors of 0.12 and 0.07 yards (r = 1.000) and the angle with errors of 0.4° and 0.2°. They count the
opponents in the shooting triangle exactly in 83% and 91% of shots (97–100% within one). [Gemini 4 Carbon,
150 shots: still running when this was drafted.]

Scored by the map-only model, the readings perform like the true geometry: ΔBrier +0.002 [−0.000, +0.004] for Opus
and +0.001 [−0.000, +0.002] for GPT, both n.s. The small loss comes from miscounted opponents. Reading the picture
therefore does not explain why the VLMs fall short of the statistical models. Their own xG differs from these
hybrids by −0.001 (Opus) and −0.003 (GPT), both n.s.

At 6–12 yards the geometry-only model itself averages 0.204 against StatsBomb's 0.168, because it cannot tell
headers from shots. Opus's own xG there, 0.221, is close to it. GPT's, 0.298, is far above what its near-perfect
readings imply: an over-estimate in judgment, not in perception.

One caveat: the readings come from a request that asks only for measurements, and in the xG request the models
think far less. What the hybrid establishes is that the geometry is readable from the image, not that the models
use it when they estimate xG.

**The models remember the World Cup, but not from our images.** Told only the match, the minute and the shooter
(probe B), Opus, Carbon and GPT-6.2 Astra classify all 84 shots correctly: 42 of 42 goals and 42 of 42 misses (AUC
1.000, against 0.790 for StatsBomb xG on the same shots). They claim to remember every goal but only 10–57% of the
misses, and still label every miss correctly. Their notes show how: knowing a match's goals and when they were
scored rules out every other shot ("Argentina's goals came from Messi (35') and Álvarez (57'), and Australia's from
Enzo Fernández's own goal (77'), so this…"). Asked for the xG of the shot "regardless of the outcome", Opus gives
xG values that separate goals from misses better than StatsBomb does (AUC 0.898 vs 0.790, p = 0.002): knowing the
outcome leaks into the estimate.

From the rendered maps, however, no model identifies the match (probe A):

- Opus answers "unknown" every time.
- GPT-6.2 Astra's best guesses name a correct team in 11% of shots, against 9% for the same guesses scored against
  other shots (chance).
- Gemini 4 Carbon claims to recognise 3 shots, all wrongly: it mistakes them for famous club goals. Its best
  guesses do name the exact match and shooter of two famous goals, Messi vs Mexico and Asano vs Germany. It reports
  low confidence (0.15 and 0.12) and does not claim to recognise them. Its image-only xG for these two goals (0.03
  and 0.04) matches StatsBomb's (0.034 and 0.027), and removing them changes no model's ΔBrier by more than 0.0005.

The image-only benchmark therefore measures estimation, not recall. Benchmarks built on real footage, where the
score bar, kits, faces and stadium can reveal the match and the moment, need the same identification check.

## Broadcast-frame pilot (methods; results pending)

Thirty of the evaluation shots, the twelve largest consensus errors and eighteen random shots stratified by
distance, are captured by hand from the television broadcast at the moment of the strike, with the score bar
cropped. The models receive the zero-shot prompt with the description of the drawing replaced by one line saying
that the image is a broadcast frame. On the same frames, the identification probe checks whether a model recognises
the match before any frame-based estimate is interpreted.
