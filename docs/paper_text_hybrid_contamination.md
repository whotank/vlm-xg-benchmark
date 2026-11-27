# Draft text: hybrid perception test, contamination probes, broadcast-frame pilot

Draft paragraphs for the full paper. Every number comes from `results/hybrid.md`, `results/contamination_probe.md`,
`results/distance_bands.md` and `results/error_analysis.md`. Rerun the scripts if a run changes.

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

Opus 5.5 and GPT-6.2 Astra measured all 300 shots. Gemini 4 Carbon measured the first 150, because it spends 3,718
thinking tokens per measurement on average. As a sanity check, Opus re-measured 30 shots with the image rescaled to
0.7 times its size.

**Contamination probes.** Frontier models may have read match reports of the 2022 World Cup. We probe this on 84
shots: all 42 goals among the 298 evaluation shots and 42 randomly drawn misses.

- Probe A shows the zero-shot image and asks whether the model recognises the shot. It also asks for a best guess of
  competition, year, teams and shooter. Under the null hypothesis the guesses have nothing to do with which shot they
  are about, so we match them to the shots by a random permutation. The exact distribution of the number of hits
  under that null follows from the rook polynomial of the guess-by-shot hit matrix, which gives exact p-values.
- Probe B shows no image and no positions. It gives only the match (stage, teams, date), the match clock, the
  half and the minute as reported, and the shooter. It asks:
  - whether the model remembers the shot
  - whether it was a goal
  - the probability that it was a goal
  - its xG "regardless of the outcome"

## Results

**The models read the drawing almost exactly.** Asked to measure, Opus and GPT-6.2 Astra read the distance with mean
absolute errors of 0.12 and 0.07 yards (r = 1.000) and the angle with errors of 0.4° and 0.2°. They count the
opponents in the shooting triangle exactly in 83% and 91% of shots (97–100% within one). Gemini 4 Carbon, on 150
shots, reads them as accurately: 0.13 yards, 0.3°, and 83% of opponents counted exactly.

The precision is real. The images have 5.9 pixels per yard. On the rescaled images (4.1 pixels per yard), Opus's
distance error grew from 0.11 to 0.17 yards, about two-thirds of a pixel at both sizes. The models locate the
markers to under a pixel and compute the geometry from them.

Scored by the map-only model, the readings perform like the true geometry: ΔBrier +0.002 [−0.000, +0.004] for Opus,
+0.001 [−0.000, +0.002] for GPT and +0.002 [−0.000, +0.005] for Carbon, all n.s. The small loss comes from
miscounted opponents. Reading the picture therefore does not explain why the VLMs fall short of the statistical
models. Their own xG differs from these hybrids by −0.001 (Opus), −0.003 (GPT) and −0.005 (Carbon, p = 0.12),
none significant.

**Close range is a header problem.** At 6–12 yards, 45 of the 77 shots are headers. StatsBomb rates them 0.095 on
average (observed 0.111) and rates the 32 foot shots from the same range 0.271 (observed 0.281). The drawing does not
show the body part. So every image-only model rates the headers above StatsBomb: the frontier mean is 0.224, the
range Sonnet 0.16 to Gemma 0.40. So does the map-only model on the true geometry (0.196). On the foot shots, most
frontier models are below StatsBomb (Carbon 0.21, Opus 0.22, mean 0.231).

The readings-based hybrids behave like the map-only model, and Opus's own xG follows them (headers 0.225, foot shots
0.215). GPT-6.2 Astra rates both kinds of shot at 0.298, above what its near-perfect readings imply, an over-estimate
in judgment rather than perception.

In Brier terms the header over-rating costs less than it looks: its contribution to the frontier's gap to StatsBomb
is +0.0028 [−0.0041, +0.0079], n.s. The significant contributions come from under-rated foot shots in open play
(+0.0048 [+0.0005, +0.0096]) and from shots with no opponent in the shooting cone (+0.0034 [+0.0008, +0.0069]).

One caveat: the readings come from a request that asks only for measurements, and measuring takes far more thinking.
Thinking tokens per shot when measuring vs in the xG request: Opus 568 vs 159, Carbon 3,718 vs 107, GPT 196 vs 25.
What the hybrid establishes is that the geometry is readable from the image, not that the models use it when they
estimate xG.

**Measuring first does not help.** We therefore asked GPT-6.2 Astra to measure the three quantities and then give
its xG, in one request. It read the geometry as precisely as before (0.07 yards, 0.2°, 90% of opponents counted
exactly) and thought seven times as long (183 vs 25 tokens per shot). Its estimates did not improve: ΔBrier against
its zero-shot run +0.002 [−0.000, +0.005], AUC 0.731 vs 0.733, mean xG 0.150 vs 0.148. At 6–12 yards it still rated
headers 0.31 and foot shots 0.30, where its own readings, scored by the map-only model, give 0.19 and 0.21 (StatsBomb
0.095 and 0.271). With the correct geometry in the same answer, the close-range error lies in how the model turns
geometry into a probability, together with the missing body part, not in what it sees.

**The models remember the World Cup; our images rarely give the match away.** Told only the match, the minute and
the shooter (probe B), Opus, Carbon and GPT-6.2 Astra classify all 84 shots correctly: 42 of 42 goals and 42 of 42
misses (AUC 1.000, against 0.790 for StatsBomb xG on the same shots). They claim to remember every goal but only
10–57% of the misses, and still label every miss correctly. Their notes show how: knowing a match's goals and when
they were scored rules out every other shot ("Argentina's goals came from Messi (35') and Álvarez (57'), and
Australia's from Enzo Fernández's own goal (77'), so this…"). Asked for the xG of the shot "regardless of the
outcome", Opus gives values that separate goals from misses better than StatsBomb does (AUC 0.898 vs 0.790,
p = 0.002): knowing the outcome leaks into the estimate.

From the rendered maps (probe A):

- Opus answers "unknown" every time.
- GPT-6.2 Astra's best guesses are at chance on every criterion (permutation p ≥ 0.2).
- Gemini 4 Carbon claims to recognise 3 shots, all wrongly. It mistakes them for two famous club goals and for a
  moment of the 2022 World Cup final.
- Carbon's best guesses are nonetheless above chance. It names the shooter in 4 of 84 shots against 1% expected by
  chance (p = 0.002), and both teams in 2 (p = 0.024). All four are goals: Messi twice, Giroud, and Asano's winner
  against Germany. It reports low confidence (at most 0.15) and never claims to recognise them.
- Its image-only xG for those goals stayed close to StatsBomb's (0.05, 0.03, 0.24 and 0.04 vs 0.07, 0.03, 0.17 and
  0.03).

Dropping every shot with a claim or an identification hit (8 shots) changes no probed model's ΔBrier against
StatsBomb by more than 0.0011. Dropping every shot with any hit at all (18 shots) changes it by no more than 0.0020.

A model can occasionally place a famous goal from the drawing, but the image-only benchmark measures estimation,
not recall. Benchmarks built on real footage, where the score bar, kits, faces and stadium can reveal the match and
the moment, need the same identification check.

## Broadcast-frame pilot (methods; results pending)

Thirty of the evaluation shots, from 26 matches, are captured by hand from the television broadcast at the moment
of the strike, with the score bar cropped: the twelve largest consensus errors and eighteen random shots
stratified by distance. The models receive the zero-shot prompt with the description of the drawing replaced by one
line saying that the image is a broadcast frame.

Because the twelve gallery shots were chosen for the drawing runs' largest errors, any new query of them regresses
toward StatsBomb. The frames are therefore compared with a fresh drawing query of the same thirty shots, and the
eighteen random shots are reported separately. On the same frames, the identification probe checks whether a model
recognises the match before any frame-based estimate is interpreted.

## Caveats to state in the paper

- No correction for multiple comparisons. The main table alone has 99 paired ΔBrier tests, 34 of them below 0.05;
  single p-values near 0.05 are descriptive.
- Gemini 4 Carbon was selected as the best Gemini 4 on the same 298 shots it is evaluated on.
- "Not significantly worse than StatsBomb" is fragile. Carbon is +0.006 [−0.002, +0.012]; Carbon with the body
  part, although better (Brier 0.104), is +0.004 [+0.000, +0.008] and significant.
- The 298-shot sample is goal-rich (0.141 vs 0.106) and favours the all-pre-shot-features logistic model. It beats
  StatsBomb on these shots (AUC 0.815 vs 0.792), while StatsBomb beats it on all 1,430 shots (AUC 0.815 vs 0.790).
- Opus's original zero-shot run used images rendered in the authors' environment. Its rerun with images rendered
  like those of the other runs is reported alongside.
