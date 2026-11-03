# Project context: VLM-based expected goals (xG) benchmark

## What this project is
We test whether general-purpose vision-language models (VLMs) can estimate expected goals (xG)
from an image of a shot situation alone, and where they diverge from established xG models.

- Abstract submitted to the MIT Sloan Sports Analytics Conference 2027 (SSAC27) Research Paper
  Competition, Soccer track, on Oct 1, 2026. Title: "Can Vision-Language Models Estimate Expected
  Goals? A Benchmark Against Statistical xG" (check `docs/abstract.md` for the submitted version).
- SSAC timeline: full-paper invitations late Oct 2026; full paper due Dec 4, 2026 (if invited).
- Longer-term plan: archival paper at the CVsports workshop (CVPR 2027), then an extended journal
  version (target: Computer Vision and Image Understanding). See `docs/xG_project_collaboration_brief.md`.

## Data and setup
- StatsBomb open data, 2022 FIFA World Cup (competition 43, season 106), via `statsbombpy`.
- 1,430 non-penalty shots with freeze frames, 152 goals.
- Each freeze frame is rendered as a half-pitch image (`mplsoccer.VerticalPitch`): gold star = shooter,
  blue = teammates, red = opponents, orange square = goalkeeper. No body part or play context is given.
- Evaluation sample: 300 shots drawn with `random_state=7`. All models are compared on the 298 shots
  every model answered (42 goals, observed goal rate 0.141; the full dataset's rate is 0.106, so this
  sample is goal-rich by chance).
- References: StatsBomb xG, and a logistic baseline (distance, angle, header, defenders in the
  shooting cone) with 5-fold GroupKFold by match (no leakage).

## Files
- `xg_vlm_pilot.py`: full pipeline (load, render, query VLM, evaluate). One run per model.
- `compare_models.py`: four-model comparison on common shots, bootstrap CIs, distance bands, Figure 1.
- `results/<model>/`: `summary.txt`, `shots.csv` (per-shot predictions and rationales), `vlm_cache.jsonl`.
- `results/figure1.png`: Figure 1 used in the abstract.
- `docs/`: abstract, literature review, collaboration brief.

## How to run
```
pip install statsbombpy mplsoccer scikit-learn scipy anthropic google-genai requests
export ANTHROPIC_API_KEY=...   # never commit keys
export GEMINI_API_KEY=...
python xg_vlm_pilot.py --backend anthropic --model claude-opus-5-5 --n 300
python xg_vlm_pilot.py --backend gemini --model gemini-3.8-flash --n 300
python xg_vlm_pilot.py --backend ollama --model gemma4:26b --n 300   # needs a GPU + Ollama
python compare_models.py
```
Answers are cached per model in `results/<model>/vlm_cache.jsonl`; reruns only query missing shots.
Use `--context` to add body part to the prompt (saved to a separate folder).

## Results (298 common shots)
| Model | AUC | Brier | Mean xG | Brier vs baseline [95% CI] |
|---|---|---|---|---|
| Gemma 4 (26B) | 0.597 | 0.190 | 0.379 | +0.084 [+0.059, +0.111] |
| Claude Sonnet 5.5 | 0.683 | 0.121 | 0.085 | +0.015 [+0.004, +0.027] |
| Claude Opus 5.5 | 0.721 | 0.111 | 0.109 | +0.005 [-0.002, +0.013] |
| Gemini 3.8 Flash | 0.721 | 0.116 | 0.167 | +0.010 [+0.001, +0.020] |
| Logistic baseline | 0.768 | 0.106 | 0.108 | |
| StatsBomb xG | 0.792 | 0.100 | 0.100 | |

Key findings:
- Every VLM is significantly worse than StatsBomb; Opus is statistically tied with the 4-feature baseline.
- Ranking (AUC) and calibration diverge: Opus and Gemini rank equally well but miss the overall level
  in opposite directions.
- Errors concentrate at short range. At 6-12 yards (77 shots, 18% scored, StatsBomb 0.17):
  Sonnet 0.15, Opus 0.22, Gemini 0.32, Gemma 0.45. Beyond 18 yards frontier models match StatsBomb.
- Rationales: Gemini calls 7-11 yard shots "point-blank"; Sonnet describes high-xG close shots as
  "very tight" angles. Inside 6 yards there are only 8 shots, so treat that band as suggestive.
- The 12-18 yard band has an unusually high observed rate (23%); every model including StatsBomb is
  below it. That is sampling variation, not a VLM failure.

## Known gotchas
- Never commit API keys. The original Colab notebook contained one; use env vars or Colab Secrets.
- An early parser bug grabbed stray decimals (e.g. "26.9 yards" -> 0.9). It is fixed: `parse()` now only
  accepts the number after `"xg":` and returns NaN otherwise. Delete a model's cache if it was created
  before the fix.
- Claude responses need `max_tokens >= 600` or the JSON gets truncated.
- Gemini 3.8 Flash often returns 503 (overloaded) and the free tier allows 20 requests/day; the script
  retries with backoff, and reruns fill gaps. The "VLM 300/300" line is a loop counter; check
  "VLM sample: N shots" for how many actually have answers.
- Only the Gemma run is deterministic (temperature 0). API runs used default sampling.

## Next steps (for the full paper)
1. `--context` ablation (body part), since headers are overrated without it.
2. Hybrid design: ask the VLM to extract geometric features (distance, angle, defenders, keeper
   position), feed them to the logistic model, and compare: true features vs VLM-read features vs
   direct VLM xG. This separates perception errors from judgment errors.
3. Recalibration (e.g. isotonic/Platt on a held-out split), especially for Gemini.
4. Repeated runs and prompt variants to measure answer stability.
5. Rendering check: does the close-range misreading change with smaller markers or a different view?
6. More data: other StatsBomb open competitions; later, broadcast video and pose.
