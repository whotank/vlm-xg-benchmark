# Project context: VLM-based expected goals (xG) benchmark

## What this project is
We test whether general-purpose vision-language models (VLMs) can estimate expected goals (xG)
from an image of a shot situation alone, and where they diverge from established xG models.

- Abstract submitted to the MIT Sloan Sports Analytics Conference 2027 (SSAC27) Research Paper
  Competition, Soccer track, on Oct 1, 2026. Title: "Can Vision-Language Models Estimate Expected
  Goals? A Benchmark Against Statistical xG" (check `docs/abstract.md` for the submitted version).
- SSAC timeline: full-paper invitations late Oct 2026. Full paper due Tue Dec 1, 2026 (confirmed by the
  authors on Nov 27; an earlier version of this file said Dec 4).
- Longer-term plan: archival paper at the CVsports workshop (CVPR 2027), then an extended journal
  version (target: Computer Vision and Image Understanding). See `docs/xG_project_collaboration_brief.md`.

## Data and setup
- StatsBomb open data, 2022 FIFA World Cup (competition 43, season 106), via `statsbombpy`.
- 1,430 non-penalty shots with freeze frames, 152 goals.
- Each freeze frame is rendered as a half-pitch image (`mplsoccer.VerticalPitch`): gold star = shooter,
  blue = teammates, red = opponents, orange square = goalkeeper. No body part or play context is given.
- Evaluation sample: 300 shots drawn with `random_state=7`. All models are compared on the 298 shots
  every model answered (42 goals, 63 matches, observed goal rate 0.141; the full dataset's rate is 0.106,
  so this sample is goal-rich by chance).
- References: StatsBomb xG, and a logistic baseline (distance, angle, header, defenders in the
  shooting cone) with 5-fold cross-validation grouped by match (no leakage). The match folds are frozen
  in `results/baseline_folds.json` (see gotchas).
- Map-only baseline: the same logistic model without the header flag (distance, angle, defenders in the
  cone; same frozen folds; `baselines.py`). It uses only information drawn in the image, so it is the fair
  statistical counterpart of the image-only VLMs: it is in fact a subset of what the VLM sees (the VLM also
  sees the keeper and every other player), but with exact values where the VLM has to read the pixels. The
  original baseline and StatsBomb also use information the image does not show (header flag; StatsBomb also
  technique, set-piece type, assist type, ...), so comparisons against them favour the statistical models.

## Files
- `xg_vlm_pilot.py`: full pipeline (load, render, query VLM, evaluate). One run per model and setting.
  Backends: anthropic, gemini, openai, ollama. Options: `--shots k`, `--example-set A|B`, `--context`,
  `--all-features`, `--input image|coords`, `--thinking-level`, `--workers`, `--tag`, `--stage`; budget guard
  and cost logging are built in.
- `make_fewshot_examples.py`: chooses the fixed, nested few-shot example sets.
- `baselines.py`: map-only baseline and the all-pre-shot-features logistic model (frozen folds) ->
  `results/baselines.{csv,md}`; `compare_models.py` adds both as reference rows.
- `compare_models.py`: all finished runs on the common shots, match-clustered bootstrap CIs, paired differences
  vs StatsBomb, the baseline and the map-only baseline, distance bands; `--figures` redraws Figure 1,
  `figure_frontier.png` and `figure_forest.png`.
- `recalibrate.py`, `fewshot_analysis.py`, `context_ablation.py`, `features_ablation.py`, `coords_analysis.py`,
  `error_analysis.py`, `cost_summary.py`: side analyses (one markdown table each in `results/`).
- `probe_runs.py`: queries that are not an xG, with the pipeline's backends, cost log, budget guard and run_config
  rails. Tasks:
  - `--task identify`: contamination probe A, "which match is this image?"
  - `--task recall`: probe B. Match, clock and shooter as text; remembered? goal? xG?
  - `--task perception`: the hybrid design. Read distance, angle and opponents in the triangle off the image.
  - `--view`: runs identify on captured broadcast frames.

  Folders: `probe_identify_<model>`, `probe_recall_<model>`, `<model>_perception`,
  `probe_identify_frames_<view>_<model>`. Each has `answers.csv` (parsed answers plus the shot's facts).
- `contamination_probe.py` -> `results/contamination_probe.md` (+ `_shots.csv`); `hybrid.py` -> `results/hybrid.md`
  (+ `hybrid_shots.csv`). The probes' shots: `results/probe_sample.json`.
- Broadcast-frame pilot:
  - `make_frame_list.py` -> `docs/frame_capture_list.csv` (30 shots); capture guide `docs/frame_capture.md`
  - `xg_vlm_pilot.py --input frame --view main` (folders `<model>_frames_<view>`)
  - `frames_analysis.py` -> `results/frames_pilot.md`
  - Frames live in `data_frames/`, which is gitignored: never commit them (broadcast footage).
- `results/<run>/`: `summary.txt`, `shots.csv` (per-shot predictions and rationales), `vlm_cache.jsonl`
  (answers; newer runs also store token usage, cost and the SHA-1 of the image sent), and for newer runs
  `calls.jsonl` (every API call, including failures) and `run_config.json` (settings and invocations).
- `results/main_table.{md,csv}`, `paired_differences.csv`, `distance_bands.{md,csv}`, `recalibration.{md,csv}`,
  `fewshot.{md,csv}`, `context_ablation.{md,csv}`, `features_ablation.{md,csv}`, `coords_vs_image.{md,csv}`,
  `error_analysis.md` (+ `error_analysis_shots.csv`, `figure_error_gallery.png`), `baselines.{md,csv}`,
  `baseline_coefficients.md` (the baseline's coefficients and odds ratios per fold and on all shots),
  `cost_summary.md`, `cost_log.csv`.
- `results/fewshot_examples.json` (set A, seed 1), `results/fewshot_examples_setB.json` (set B, seed 2).
- `results/figure1.png`: Figure 1 of the abstract (unchanged). Full-paper figures: `figure_frontier.png`,
  `figure_forest.png` (AUC and Brier with CIs; Reference group with StatsBomb and the logistic baselines on top),
  `figure_paired.png` (paired ΔBrier vs StatsBomb and vs the baseline for every row), `figure_error_gallery.png`.
- `docs/`: abstract, literature review, collaboration brief, `colab_gemma_fewshot.md` (Gemma few-shot runs),
  `paper_notes.md` (data-section draft and discussion points; numbers checked against the data).

## How to run
```
pip install statsbombpy mplsoccer scikit-learn scipy anthropic google-genai openai requests
export XG_ANTHROPIC_API_KEY=...   # or ANTHROPIC_API_KEY (Colab); never commit keys
export GEMINI_API_KEY=...
export OPENAI_API_KEY=...
python xg_vlm_pilot.py --backend anthropic --model claude-opus-5-5 --n 300 --workers 4
python xg_vlm_pilot.py --backend gemini --model gemini-4-carbon --thinking-level medium --n 300 --workers 6
python xg_vlm_pilot.py --backend openai --model gpt-6.2-astra --thinking-level medium --n 300 --workers 6
python xg_vlm_pilot.py --backend ollama --model gemma4:26b --n 300   # needs a GPU + Ollama
python xg_vlm_pilot.py --backend anthropic --model claude-opus-5-5 --shots 5            # few-shot, set A
python xg_vlm_pilot.py --backend anthropic --model claude-opus-5-5 --shots 5 --example-set B
python xg_vlm_pilot.py --backend anthropic --model claude-opus-5-5 --context            # body part given
python xg_vlm_pilot.py --backend anthropic --model claude-opus-5-5 --all-features       # + every pre-shot attribute
python xg_vlm_pilot.py --backend anthropic --model claude-opus-5-5 --input coords       # freeze frame as text, no image
python baselines.py && python compare_models.py && python recalibrate.py && python fewshot_analysis.py
python context_ablation.py && python features_ablation.py && python coords_analysis.py
python error_analysis.py && python cost_summary.py
python probe_runs.py --task perception --backend anthropic --model claude-opus-5-5 --workers 4   # hybrid readings
python probe_runs.py --task identify --backend anthropic --model claude-opus-5-5 --workers 4     # contamination A
python probe_runs.py --task recall --backend anthropic --model claude-opus-5-5 --workers 4       # contamination B
python xg_vlm_pilot.py --backend gemini --model gemini-4-carbon --thinking-level medium --input frame --view main
python hybrid.py && python contamination_probe.py && python make_frame_list.py && python frames_analysis.py
```
Answers are cached per run in `results/<run>/vlm_cache.jsonl`; reruns only query missing shots.
Folder names: `<model>` + `_context` / `_allfeat` / `_coords` / `_frames_<view>` + `_k{n}` (+ `_setB`) + `_<tag>`.

## Results (298 common shots; 95% match-clustered bootstrap CIs; full tables in `results/*.md`)
Zero-shot, image only:
| Model | AUC | Brier | Mean xG | ΔBrier vs StatsBomb [95% CI] | ΔBrier vs baseline [95% CI] | ΔBrier vs map-only [95% CI] |
|---|---|---|---|---|---|---|
| Gemma 4 (26B) | 0.597 | 0.190 | 0.379 | +0.091 [+0.062, +0.118] | +0.084 [+0.057, +0.111] | +0.080 [+0.052, +0.106] |
| Claude Sonnet 5.5 | 0.683 | 0.121 | 0.085 | +0.022 [+0.010, +0.034] | +0.015 [+0.005, +0.028] | +0.011 [+0.002, +0.021] |
| Claude Opus 5.5 | 0.721 | 0.111 | 0.109 | +0.012 [+0.001, +0.022] | +0.005 [-0.005, +0.016] | +0.001 [-0.006, +0.008] |
| Gemini 3.8 Flash | 0.721 | 0.116 | 0.167 | +0.016 [+0.003, +0.028] | +0.010 [-0.002, +0.021] | +0.005 [-0.006, +0.016] |
| Gemini 4 Carbon | 0.743 | 0.105 | 0.111 | +0.006 [-0.002, +0.012] | -0.001 [-0.009, +0.006] | -0.005 [-0.010, -0.001] |
| Gemini 4 Neon | 0.719 | 0.108 | 0.101 | +0.009 [-0.000, +0.019] | +0.003 [-0.007, +0.012] | -0.002 [-0.008, +0.005] |
| Gemini 4 Hydrogen | 0.721 | 0.111 | 0.113 | +0.012 [+0.002, +0.021] | +0.005 [-0.004, +0.014] | +0.001 [-0.005, +0.006] |
| GPT-6.2 Astra | 0.733 | 0.108 | 0.148 | +0.009 [-0.002, +0.018] | +0.002 [-0.009, +0.012] | -0.002 [-0.011, +0.006] |
| Map-only baseline | 0.721 | 0.110 | 0.111 | +0.011 [+0.003, +0.018] | +0.005 [-0.001, +0.009] | |
| Logistic baseline | 0.768 | 0.106 | 0.108 | +0.006 [+0.001, +0.012] | | -0.005 [-0.009, +0.001] |
| Logistic, all pre-shot features | 0.815 | 0.098 | 0.112 | -0.001 [-0.008, +0.004] | -0.007 [-0.015, -0.001] | -0.012 [-0.021, -0.004] |
| StatsBomb xG | 0.792 | 0.100 | 0.100 | | -0.006 [-0.012, -0.001] | -0.011 [-0.018, -0.003] |

The abstract (Oct 2026) used a shot-level bootstrap; `results/main_table.md` keeps both for comparison.

Key findings:
- Newest models close most of the gap. Gemini 4 Carbon is the best VLM (AUC 0.743, Brier 0.105), ties the
  4-feature baseline and is not significantly worse than StatsBomb (p = 0.11); GPT-6.2 Astra (p = 0.11)
  and Gemini 4 Neon (p = 0.05) are borderline. Opus 5.5, Hydrogen, Sonnet, Gemini 3.8 Flash and Gemma
  remain significantly worse than StatsBomb. "Not significantly worse" is not equivalence: the upper CI
  bounds still allow gaps of 0.012-0.019 Brier.
- Carbon was picked as "best Gemini 4" on the same 298 shots. Paired, it beats Opus 5.5 (ΔBrier -0.006
  [-0.012, -0.001], p = 0.02) and Hydrogen (-0.006, p = 0.006), but not Neon (-0.003, p = 0.14) or
  GPT-6.2 Astra (-0.003, p = 0.37); treat the top three as one tier.
- Against the fair comparator, the map-only baseline (exact geometry from the image, no header flag), the
  frontier VLMs tie or win: Opus +0.001, Hydrogen +0.001, Neon -0.002, GPT-6.2 Astra -0.002 (all n.s.), and
  Carbon -0.005 [-0.010, -0.001] (p = 0.013), from pixels alone. Most of the gap to the 4-feature baseline and
  StatsBomb is information the image does not contain: the header flag alone is worth -0.005 Brier to the
  logistic model, and the map-only baseline over-rates headers (0.201 vs StatsBomb 0.098) just like the VLMs.
- Given every pre-shot attribute and the exact geometry as text (`_allfeat`), Gemini 4 Carbon (Brier 0.099,
  AUC 0.785) and GPT-6.2 Astra (0.099, 0.781) tie StatsBomb (-0.000 / -0.001) and a logistic model on the
  same attributes (+0.001 / +0.001); Opus reaches 0.104 (0.758). The extra information helps every model
  (Carbon -0.006, GPT -0.009, Opus -0.007 n.s., logistic -0.012), but VLM judgment is not better than a
  statistical model with the same inputs. With the numbers in hand the models barely reason (thinking tokens
  per shot: Opus ~16 vs ~160 image-only; GPT ~1).
- Coordinates as text instead of the image (`_coords`; perception vs judgment): Carbon is unchanged (overall
  ΔBrier +0.000; 6-12 yd over-estimate +0.030 -> +0.020, n.s.; headers 0.185 -> 0.179), so its close-range
  error is judgment or missing information, not reading the picture. Opus's 6-12 yd over-estimate disappears
  (+0.053 -> -0.014), but its whole level drops (mean 0.109 -> 0.082, 12-18 yd now 0.032 below StatsBomb,
  headers 0.211 -> 0.142 although neither input has the body part) with unchanged Brier (+0.000) and AUC
  (0.721 both): a level shift, not better reading. `results/coords_vs_image.md`.
- Hybrid (`results/hybrid.md`): the models read the geometry off the image, the map-only logistic model scores the
  readings.
  - Asked only to measure, Opus and GPT-6.2 Astra read the distance to within 0.12 / 0.07 yd on average (r = 1.000)
    and the angle to within 0.4° / 0.2°. They count the opponents in the shooting triangle exactly in 83% / 91% of
    shots (97–100% within one). Gemini 4 Carbon, on the first 150 shots: 0.13 yd, 0.3°, opponents exact in 83%
    (100% within one).
  - Scored by the map-only model, the readings match the true geometry: ΔBrier +0.002 / +0.001 (n.s.); Carbon
    +0.002 (n.s.). The small loss comes from miscounted opponents (AUC 0.695 / 0.708 vs 0.721).
  - So reading the picture is not what separates the VLMs from the statistical models. Their own xG differs from the
    hybrid by −0.001 / −0.003 (n.s.), Carbon −0.005 (p = 0.12): that is judgment.
  - At 6–12 yd the map-only model itself averages 0.204 (StatsBomb 0.168), because it does not know headers. Opus's
    own xG there is 0.221, GPT's 0.298: GPT's extra close-range level is judgment, not perception.
  - Caveat: the readings come from a separate request that only asks for measurements. In the xG request the models
    think much less.
- Contamination (`results/contamination_probe.md`; 84 shots: 42 goals, 42 misses):
  - Told the match, the minute and the shooter as text, Opus, Carbon and GPT label every shot correctly: sensitivity
    and specificity 100%, AUC 1.000 vs StatsBomb 0.790. They claim to remember every goal but only 10–57% of the
    misses; they infer the misses from knowing each match's goals.
  - From the rendered maps no model recognises the match. Opus never guesses, and GPT's best guesses are at chance
    (a correct team 11% vs 9%). Carbon's 3 claimed recognitions are all wrong (famous club goals).
  - Without claiming recognition, Carbon's best guesses name two famous goals: Messi vs Mexico and Asano vs Germany
    (confidence 0.15 / 0.12). Its image-only xG for them matches StatsBomb (0.03 / 0.04 vs 0.034 / 0.027), and
    dropping them moves any ΔBrier by ≤ 0.0005.
  - So the image-only results are not recall, but real footage, where the match and the moment can be identified,
    needs the identify probe.
  - Told the shot and asked for an xG "regardless of the outcome", Opus's xG separates goals from misses better than
    StatsBomb (AUC 0.898 vs 0.790, p = 0.002): the known outcome leaks into the estimate.
- Error analysis (`results/error_analysis.md`): 54% of the frontier models' Brier gap to StatsBomb accrues at
  6-12 yd and 36% on headers (gap +0.117); the largest consensus over-estimates are all headers from 7-13 yd,
  the largest under-estimates close-range foot shots that went in, usually with the keeper out of position
  (models say so in their rationale and still rate them 0.08-0.24 below StatsBomb). Shots with no opponent in
  the cone (8 shots, 5 goals) are rated 0.32 vs StatsBomb 0.49. Errors are shared: the five models' gaps
  correlate at a median of 0.82, so the hard shots are hard for every model.
- With match clustering, Gemini 3.8 Flash vs the baseline is no longer significant (abstract: +0.010
  [+0.001, +0.020] shot-level; now [-0.002, +0.021]).
- Ranking (AUC) and level diverge: among models with AUC 0.72-0.74, mean predicted xG ranges from 0.101 (Neon)
  to 0.167 (Gemini 3.8 Flash), against an observed 0.141 in this goal-rich sample (0.106 over all shots;
  StatsBomb 0.100). GPT-6.2 Astra (0.148) sits near the sample's rate, above the population's.
- Errors concentrate at short range. At 6-12 yards (77 shots, 18% scored, StatsBomb 0.17): Sonnet 0.15,
  Neon 0.19, Carbon 0.20, Opus 0.22, Hydrogen 0.22, GPT-6.2 Astra 0.30, Gemini 3.8 Flash 0.32, Gemma 0.45.
  Beyond 18 yards frontier models match StatsBomb. Carbon tracks StatsBomb in every band.
- Few-shot examples labelled with outcomes do not help: for Opus and Carbon, with example sets A and B, every
  k = 1/5/10 run is within ±0.003 Brier of zero-shot (one Opus set B run reaches p = 0.03 in the wrong
  direction), the same size as a plain rerun of the zero-shot setting (Opus +0.002, p = 0.26; Carbon +0.001,
  p = 0.49). Run-to-run noise, not the examples, explains the differences.
- Body part fixes the header level, not the score: header mean xG falls 0.211 -> 0.141 (Opus) and
  0.185 -> 0.117 (Carbon), toward the observed 0.111 and StatsBomb 0.098; foot shots do not move; overall
  Brier changes by -0.001 / -0.002 (n.s.; only 7 header goals). Carbon + body part: AUC 0.758, Brier 0.104.
- Recalibration (side analysis): only Gemma improves substantially (Brier 0.190 -> 0.121); every frontier
  model moves by at most ~0.004 either way, like StatsBomb itself (+0.001 Platt / +0.006 isotonic).
- Rationales: Gemini 3.8 Flash calls 7-11 yard shots "point-blank"; Sonnet describes high-xG close shots as
  "very tight" angles. Inside 6 yards there are only 8 shots, so treat that band as suggestive.
- The 12-18 yard band has an unusually high observed rate (23%); every model including StatsBomb is
  below it. That is sampling variation, not a VLM failure.
- Cost (see `results/cost_summary.md`): all runs since Nov 27 cost $80.71 in total (Anthropic $26.22, Gemini
  $45.83, OpenAI $8.66; 8,763 calls, none failed). Of that:
  - the overnight session: $54.48
  - the all-features and coordinates runs: $8.41
  - the contamination probes and perception runs: $17.81

  Left under the limits: Anthropic $3.78, Gemini $4.17, OpenAI $21.34 ($19.29 in total). That is enough for the
  frame pilot. A 300-shot zero-shot run costs $1.2-$4.0; thinking
  depth at the same "medium" setting differs a lot (thinking tokens per shot: Carbon ~107, Neon ~593,
  Hydrogen ~572, GPT-6.2 Astra ~25; Opus at its default ~160). Few-shot k=10 costs $2.2 on Opus (cached
  prefix) but $7.2 on Carbon (no explicit caching on Gemini 4).

## Design decisions for the full paper (overnight session, Nov 27, 2026)
Recorded so the paper's methods section and later runs stay consistent.

### Evaluation set and statistics (Stage 1)
- Tables use the shots every included run answered (the same 298 shots as the abstract). A run is included
  only once its latest invocation covered the full 300-shot sample and finished (`compare_models.finished()`).
- Every metric (AUC, Brier, log loss, mean xG, each distance-band mean, observed rates) has a 95% percentile
  interval from a match-clustered bootstrap: resample the 63 matches with replacement, keep all shots of each
  drawn match; 2,000 resamples, seed 0. Paired differences (vs StatsBomb, vs baseline, few-shot vs k = 0,
  context vs none, recalibrated vs raw) use the same resamples; p = two-sided bootstrap p-value.
- The abstract's shot-level bootstrap is kept as a sensitivity column. AUC uses the Mann-Whitney rank formula
  (identical to sklearn); log loss clips predictions to [0.001, 0.999] as in the pilot.

### Recalibration (Stage 2, side analysis; main results stay raw)
- Platt = unpenalised logistic regression of goal on logit(p); isotonic = non-decreasing, clipped to [0, 1].
  Fitted out-of-fold, 5 folds grouped by match (the baseline's frozen match folds). The bootstrap resamples
  out-of-fold predictions (maps are not refitted inside it); a sensitivity table repeats the CV over 50
  seeded match permutations. StatsBomb and the baseline are recalibrated the same way as reference rows.

### Gemini 4 zero-shot (Stage 3)
- gemini-4-carbon, gemini-4-neon, gemini-4-hydrogen (gemini-4-argon exists, not in the plan), all with
  `--thinking-level medium`, JSON response mime type as in the Gemini 3.8 Flash run, no output cap, default
  sampling. Carbon ran first (its price rises from $2/$10 to $4/$20 on Dec 1).
- Best Gemini 4 for Stages 4-6 = lowest Brier on the 298 common shots, AUC tie-break: gemini-4-carbon
  (Brier 0.1052; Neon 0.1084, Hydrogen 0.1110).

### Few-shot in-context learning (Stage 4; set B in Stage 6)
- Pool: the 1,130 shots not in the 300-shot test sample (110 goals). Distance bands as in the paper.
- Nested sets from `make_fewshot_examples.py` with a recorded seed (set A seed 1, set B seed 2; the sets differ
  only by seed): k=1 one 12-18 yd miss; k=5 adds one example per other band, one of them a goal drawn uniformly
  from pool goals in those bands; k=10 adds one more per band, one goal from bands without a goal yet (2 goals,
  8 misses, two per band). Prompt order: k=1 example, k=5 additions shuffled, k=10 additions shuffled, so each
  smaller set is a prefix of the larger one. Files store shot_ids, labels, bands, seed and rules.
- Labels are actual outcomes ("Outcome: GOAL" / "Outcome: NO GOAL"), never xG. Each example image is followed
  by its label, then the test image and the zero-shot prompt with exactly one added line ("The earlier images
  are example shots drawn the same way, each followed by what actually happened; the last image is the shot to
  evaluate."). With --context, example labels would also carry their body part (not used).
- Caching: Anthropic `cache_control` on the last example label (5-minute TTL); Opus 5.5 caches only prefixes of
  >= 512 tokens, so k=1 is never cached, while k=5/10 hit the cache on 99-100% of calls (the first few-shot call
  is sent alone before the parallel calls). Gemini explicit context caching is tried first, but Gemini 4 models
  do not offer it (`createCachedContent` is not among their supported actions; `caches.create` returns 404), so
  the examples are sent in full as a separate user turn; implicit caching hit only 17-19% of k=5/10 calls.
- Ollama/Gemma: few-shot prompts use `/api/chat`, one user turn per example (image + label), `num_ctx` 16384;
  zero-shot still uses `/api/generate` exactly as before. Commands: `docs/colab_gemma_fewshot.md`.
- Output folders: `<model>_k{n}` (set A), `<model>_k{n}_setB`. API runs are not deterministic, so part of any
  set A vs set B or k-to-k difference is run-to-run noise; `<model>_rerun` folders (zero-shot setting queried
  again, added as a control) measure it.

### Body-part context (Stage 5)
- `--context` adds one line, "Body part used: <part>.", before the estimate request; nothing else changes.
  Compared within model for Opus 5.5 and Gemini 4 Carbon (zero-shot): `results/context_ablation.md`.

### All pre-shot features (VLM judgment with perfect perception; added Nov 27)
- Question: given everything StatsBomb's model knows before the strike, does a VLM judge better than a
  statistical model? `--all-features` keeps the image and the zero-shot prompt and puts a text block in the
  context slot: body part, technique, shot type, play pattern; first-time / one-on-one / open goal / under
  pressure / aerial won / after a dribble; the assist (none, height, cross, cut-back, through ball, switch,
  set-piece delivery); distance, goal angle, opponents in the cone, outfield opponents within 2 yd; keeper
  distance from the goal line and from the shooter. Nothing recorded after the strike (end location,
  deflection, save details, outcome) and never StatsBomb's xG.
- Control: `Logistic, all pre-shot features` (same attributes, standardised, L2, C = 1, frozen folds).
- Runs: Opus 5.5, Gemini 4 Carbon (thinking medium), GPT-6.2 Astra (reasoning medium), 300 shots,
  `<model>_allfeat`. Analysis: `results/features_ablation.md` (who-sees-what table, information x model grid).

### Coordinates as text (perception vs judgment; added Nov 27)
- `--input coords` drops the image and replaces the three prompt lines that describe it with the same freeze
  frame as text: pitch 120 x 80 yards, goal at x = 120 with posts at y = 36 and y = 44, then the shooter, the
  opposing goalkeeper, the other opponents and the teammates as (x, y) to one decimal, in freeze-frame order.
  No player roles, no body part, nothing derived (no distances or angles). The role line, the task lines and
  the JSON line are identical to the image prompt.
- Runs: Opus 5.5 (no thinking parameter, max_tokens 600) and Gemini 4 Carbon (thinking medium), 300 shots,
  `<model>_coords`. Analysis: `results/coords_vs_image.md`, overall and by distance band. Rule fixed before
  the runs: if the 6-12 yd over-estimate shrinks with coordinates, reading the picture was the problem; if
  not, it is judgment (or the step from raw coordinates to geometry, which the all-features runs remove).

### Error analysis (added Nov 27, no API calls)
- Frontier = the five zero-shot models with Brier <= 0.111 (Carbon, Neon, GPT-6.2 Astra, Opus, Hydrogen).
  Per shot: gap = frontier mean - StatsBomb; excess Brier = (frontier mean - goal)^2 - (StatsBomb - goal)^2.
  Group shares of the excess, an OLS of the gap on standardised features (goal angle and keeper-to-shooter
  distance left out: collinear with distance and keeper position) with match-clustered CIs, correlation of the
  models' gaps, a gallery of the largest consensus disagreements, rationale phrases. `results/error_analysis.md`.

### Contamination probes, hybrid perception, broadcast-frame pilot (added late Nov 27)
- Match facts: `load_shots()` also keeps minute, second, period, player, team and the match's teams, date and stage.
  The cache was rebuilt: same 1,430 shots in the same order, the same 300-shot sample, and every earlier column
  identical. None of these facts is ever sent in an xG prompt.
- Probe sample (`results/probe_sample.json`): all 42 goals among the 298 common shots plus 42 non-goals (seed 2026),
  queried in a seeded random order. Settings as in the xG runs (Gemini/GPT thinking or reasoning medium, no output
  cap; Opus without a thinking parameter) except Opus `max_tokens` 1500 (thinking counts toward it).
- Probe A (identify): the xG prompt's role and legend lines and the same image, then:
  - asks whether the model recognises the shot, and for a best guess of competition, year, both teams, shooter
    and its confidence (JSON)
  - a best guess counts as right for:
    - World Cup: the competition names the World Cup and not a club, women's, youth or qualifying competition
    - 2022
    - either team / both teams, with aliases for USA, Korea Republic, IR Iran and Holland
    - the shooting team
    - the shooter: the last word of the guess is a word of his full name or nickname
  - chance = the same guesses scored against the facts of permuted shots (2,000 permutations)
- Probe B (recall): text only, no positions. The prompt gives:
  - stage, teams and date
  - the match clock as mm:ss (StatsBomb's event clock) and the half
  - the minute as in match reports: clock minute + 1, stoppage time as 45+n / 90+n. Checked on Messi 64',
    Richarlison 62' / 73', Mbappé 81' and Di María 36'.
  - the shooter's common name (StatsBomb lineup nickname, else full name) and team

  It asks: remembered, scored (yes / no / unknown), p_goal, xg ("the quality of the chance, regardless of the
  outcome") and a note.
- Perception (hybrid): role and legend lines, the scale from the pitch markings (pitch 80 yd wide, penalty area
  18 x 44, six-yard box 6 x 20, penalty spot 12 yd, goal 8 yd), then distance to the centre of the goal (yd), the
  angle between the posts (degrees) and the opponents (red and orange) inside the shooter–posts triangle.
  - The hybrid xG applies the map-only baseline's five fold models (refitted; checked to reproduce
    `baseline_image_xg` to 1e-9) to the readings, angle converted to radians, each shot scored by the fold model that
    never saw its match. Partial hybrids swap in one reading at a time.
  - Opus and GPT-6.2 Astra ran all 300 shots. Gemini 4 Carbon thinks about 2,700 tokens per measurement (one call
    25,600, $0.26), about $0.035 per shot, so it ran the first 150 shots of the sample. This keeps Gemini budget for
    the frame pilot.
- Frame pilot: 30 shots, the 12 error-gallery shots plus 18 random shots stratified by distance band (seed 2027).
  - Frame prompt = the zero-shot prompt with the three image lines replaced by "The image is a frame from the
    television broadcast of a professional match at the moment a shot is taken (the player striking the ball is the
    shooter)."
  - Frames are converted to PNG with the long side at most 1280 px. The score bar is cropped by hand.
  - Runs query only the shots that have a frame; the identify probe also runs on the frames (`--view`).

### Model settings
- Claude Opus 5.5: same request as the original run (no thinking or effort parameter, `max_tokens` 600; Opus 5.5
  always thinks at its default effort, medium, and the thinking counts toward `max_tokens`). No server-side
  refusal fallback, so every answer comes from the named model. No answer was cut off at 600 tokens.
- GPT-6.2 Astra (optional run): Responses API, `reasoning.effort` medium (to match Gemini 4), JSON mode
  (`text.format = json_object`), no output cap; non-`completed` responses are flagged (none occurred).
- Gemma (Ollama): temperature 0, seed 7. API runs use provider default sampling.

### Images
- `render()` is unchanged. Runs from this session used images rendered here (matplotlib 3.11.2, mplsoccer 1.8.1,
  539 x 421 px); the four original runs used images rendered in the authors' environment. New answers store the
  SHA-1 of the image sent (`image_sha1`); images are written atomically so parallel runs never read a partial file.

### Cost logging and budget
- `results/cost_log.csv`: one row per run folder (plus one setup row per provider for the one-call API checks),
  rebuilt from `calls.jsonl` after every invocation. Columns: date, stage, run, provider, model, thinking_level,
  k, example_set, context, max_output_tokens, shots_queried, shots_answered, calls, failed_calls,
  incomplete_flagged, input_tokens (uncached, base price), cache_write_tokens, cached_input_tokens,
  output_tokens (all billed output, thinking included), thinking_tokens (the thinking/reasoning part, as
  reported by the provider), price per 1M tokens (input, cache write, cached input, output), price_source,
  price_date, cost_usd, status, notes. Each `vlm_cache.jsonl` line stores that call's usage and cost.
- Prices (USD per 1M tokens): Opus 5.5 $4 in / $20 out, cache write $5, cache read $0.20 (Anthropic pricing
  page, fetched 2026-11-27). Gemini 4 Carbon $2/$10 to Nov 30 then $4/$20, Neon $3/$15, Hydrogen $1.50/$7.50,
  cached input 5% of input (third-party sources via web search; Google's page is not reachable from the session).
  GPT-6.2 Astra $8/$40, cached $0.40 (project notes; openai.com not reachable, unverified).
- Budget guard: after the first 20 shots of a run the script projects the full run's cost and stops if it would
  exceed the provider limit (Anthropic $30, Gemini $50, OpenAI $30) or $100 in total, counting every other row of
  the cost log. A billing/no-credit error stops the run and marks the provider blocked
  (`data_cache/blocked_<provider>.txt`).

### Safety rails
- `run_config.json` per run: a folder is never reused with different settings; folders from before this file
  existed (the four original runs) are refused unless `--allow-legacy-folder` is given. A per-run lock stops two
  processes writing one folder; if a shot was answered twice anyway, the first answer counts (all calls billed
  and logged; happened once, 6 shots in `claude_opus_5_5_k1`).
- `compare_models.py` only redraws figures with `--figures`; Figure 1 keeps its four-model design.

## Known gotchas
- Never commit API keys. The original Colab notebook contained one; use env vars or Colab Secrets.
- An early parser bug grabbed stray decimals (e.g. "26.9 yards" -> 0.9). It is fixed: `parse()` now only
  accepts the number after `"xg":` and returns NaN otherwise. Delete a model's cache if it was created
  before the fix.
- Claude responses need `max_tokens >= 600` or the JSON gets truncated.
- scikit-learn >= 1.6 changed `GroupKFold`'s fold assignment, which silently moved baseline predictions by up to
  0.25 (AUC 0.762 instead of 0.768). `add_baseline()` now uses the original folds frozen in
  `results/baseline_folds.json`; do not switch back to `GroupKFold`.
- Gemini 3.8 Flash often returns 503 (overloaded) and the free tier allows 20 requests/day; the script
  retries with backoff, and reruns fill gaps. Gemini 4 Carbon has a long latency tail (90th percentile
  ~40 s, some calls > 4 min) and occasional server disconnects; retries handle them.
- Gemini 4 models offer no explicit context caching (404); few-shot prompts are sent in full.
- From the cloud session, ai.google.dev and openai.com are blocked, so Gemini and OpenAI prices could not be
  checked against the official pages. `git push` of a new branch was refused once (GitHub App access); the
  branch was created through the GitHub API, after which pushes worked.
- Only the Gemma run is deterministic (temperature 0). API runs used default sampling.
- Running analyses while runs are in progress is safe: unfinished runs are skipped.
- Gemini 4 Carbon can think for a very long time on measuring tasks (perception: median ~2,200 thinking tokens, one call
  25,600). Project the cost from the first 20 shots before running it on many shots.
- The models remember 2022 World Cup outcomes (`results/contamination_probe.md`). Never put match, date, minute or
  player names in an xG prompt, and treat any real-footage result as possibly recalled unless the identify probe on
  the same frames comes back empty.

## Next steps (for the full paper)
1. Hybrid design: done for distance, angle and opponents in the triangle (`results/hybrid.md`). Possible follow-ups:
   - "Measure, then estimate" in one request: the models read the geometry almost exactly when asked, but think far
     less in the xG request.
   - Keeper position as a fourth reading.
2. Broadcast-frame pilot: capture the 30 frames (`docs/frame_capture.md`), then run the frame xG and identify
   commands there and `frames_analysis.py`.
3. Gemma few-shot runs in Colab (`docs/colab_gemma_fewshot.md`).
4. More repeated runs and prompt variants to measure answer stability (one zero-shot rerun per model so far).
5. Rendering check: lower priority now. Asked to measure, the models read the drawing almost exactly
   (`results/hybrid.md`).
6. More data: other StatsBomb open competitions; later, broadcast video and pose. The models remember 2022 World Cup
   outcomes, so footage needs the identify probe.
