# Project context: VLM-based expected goals (xG) benchmark

## What this project is
We test whether general-purpose vision-language models (VLMs) can estimate expected goals (xG)
from an image of a shot situation alone, and where they diverge from established xG models.

- Abstract submitted to the MIT Sloan Sports Analytics Conference 2027 (SSAC27) Research Paper
  Competition, Soccer track, on Oct 1, 2026. Title: "Can Vision-Language Models Estimate Expected
  Goals? A Benchmark Against Statistical xG" (check `docs/abstract.md` for the submitted version).
- SSAC timeline: full-paper invitations late Oct 2026. Full paper: working deadline Tue Dec 1, 2026
  (this file earlier listed the official due date as Dec 4, 2026; confirm which applies).
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
- `results/<run>/`: `summary.txt`, `shots.csv` (per-shot predictions and rationales), `vlm_cache.jsonl`
  (answers; newer runs also store token usage, cost and the SHA-1 of the image sent), and for newer runs
  `calls.jsonl` (every API call, including failures) and `run_config.json` (settings and invocations).
- `results/main_table.{md,csv}`, `paired_differences.csv`, `distance_bands.{md,csv}`, `recalibration.{md,csv}`,
  `fewshot.{md,csv}`, `context_ablation.{md,csv}`, `features_ablation.{md,csv}`, `coords_vs_image.{md,csv}`,
  `error_analysis.md` (+ `error_analysis_shots.csv`, `figure_error_gallery.png`), `baselines.{md,csv}`,
  `cost_summary.md`, `cost_log.csv`.
- `results/fewshot_examples.json` (set A, seed 1), `results/fewshot_examples_setB.json` (set B, seed 2).
- `results/figure1.png`: Figure 1 of the abstract (unchanged). `results/figure_frontier.png`,
  `results/figure_forest.png`: full-paper figures.
- `docs/`: abstract, literature review, collaboration brief, `colab_gemma_fewshot.md` (Gemma few-shot runs).

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
```
Answers are cached per run in `results/<run>/vlm_cache.jsonl`; reruns only query missing shots.
Folder names: `<model>` + `_context` / `_allfeat` / `_coords` + `_k{n}` (+ `_setB`) + `_<tag>`.

## Results (298 common shots; 95% match-clustered bootstrap CIs; full tables in `results/*.md`)
Zero-shot, image only:
| Model | AUC | Brier | Mean xG | ΔBrier vs StatsBomb [95% CI] | ΔBrier vs baseline [95% CI] |
|---|---|---|---|---|---|
| Gemma 4 (26B) | 0.597 | 0.190 | 0.379 | +0.091 [+0.062, +0.118] | +0.084 [+0.057, +0.111] |
| Claude Sonnet 5.5 | 0.683 | 0.121 | 0.085 | +0.022 [+0.010, +0.034] | +0.015 [+0.005, +0.028] |
| Claude Opus 5.5 | 0.721 | 0.111 | 0.109 | +0.012 [+0.001, +0.022] | +0.005 [-0.005, +0.016] |
| Gemini 3.8 Flash | 0.721 | 0.116 | 0.167 | +0.016 [+0.003, +0.028] | +0.010 [-0.002, +0.021] |
| Gemini 4 Carbon | 0.743 | 0.105 | 0.111 | +0.006 [-0.002, +0.012] | -0.001 [-0.009, +0.006] |
| Gemini 4 Neon | 0.719 | 0.108 | 0.101 | +0.009 [-0.000, +0.019] | +0.003 [-0.007, +0.012] |
| Gemini 4 Hydrogen | 0.721 | 0.111 | 0.113 | +0.012 [+0.002, +0.021] | +0.005 [-0.004, +0.014] |
| GPT-6.2 Astra | 0.733 | 0.108 | 0.148 | +0.009 [-0.002, +0.018] | +0.002 [-0.009, +0.012] |
| Logistic baseline | 0.768 | 0.106 | 0.108 | +0.006 [+0.001, +0.012] | |
| StatsBomb xG | 0.792 | 0.100 | 0.100 | | |

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
- Cost (see `results/cost_summary.md`): the overnight runs cost $54.48 in total (Anthropic $17.44, Gemini
  $35.25, OpenAI $1.79; 6,009 calls, none failed). A 300-shot zero-shot run costs $1.2-$4.0; thinking
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

## Next steps (for the full paper)
1. Hybrid design: ask the VLM to extract geometric features (distance, angle, defenders, keeper
   position), feed them to the logistic model, and compare: true features vs VLM-read features vs
   direct VLM xG. This separates perception errors from judgment errors.
2. Gemma few-shot runs in Colab (`docs/colab_gemma_fewshot.md`).
3. More repeated runs and prompt variants to measure answer stability (one zero-shot rerun per model so far).
4. Rendering check: does the close-range misreading change with smaller markers or a different view?
5. More data: other StatsBomb open competitions; later, broadcast video and pose.
