# Can Vision-Language Models Estimate Expected Goals? A Benchmark Against Statistical xG

Code and results for our MIT Sloan Sports Analytics Conference 2027 research paper.

We ask whether general-purpose vision-language models (VLMs) can estimate expected goals (xG)
from an image of a shot situation alone, and where they diverge from established xG models.

## Data
StatsBomb open data, 2022 FIFA World Cup (competition 43, season 106), loaded via `statsbombpy`.
Each shot's freeze frame is rendered as a half-pitch image (shooter, teammates, opponents, goalkeeper).
All models are scored on the same 298 shots (42 goals) drawn with `random_state=7`.

## Models
Zero-shot, image only (one standardized prompt, see `PROMPT` in `xg_vlm_pilot.py`):
- Gemma 4 (26B), open-weight, run locally via Ollama
- Claude Sonnet 5.5 and Claude Opus 5.5 (Anthropic API)
- Gemini 3.8 Flash, and Gemini 4 Carbon, Neon and Hydrogen at thinking level "medium" (Gemini API)
- GPT-6.2 Astra at reasoning effort "medium" (OpenAI API)

Ablations: few-shot in-context learning (k = 1, 5, 10 examples labelled with their actual outcome, two fixed
example sets), body-part context, all pre-shot event attributes, the freeze frame as coordinates in text
instead of an image (perception vs judgment), post-hoc recalibration (Platt and isotonic, side analysis),
and an error analysis of the shots the models get wrong.
Comparisons: StatsBomb xG, a logistic baseline (distance, angle, header, defenders in shooting cone), a
map-only baseline (the same without the header flag: only what the image shows) and a logistic model on all
pre-shot attributes, all trained with match-grouped cross-validation. Every metric has a 95% match-clustered
bootstrap interval.

## Reproduce
```bash
pip install statsbombpy mplsoccer scikit-learn scipy anthropic google-genai openai requests
```

API keys are read from the environment only: `XG_ANTHROPIC_API_KEY` (or `ANTHROPIC_API_KEY`),
`GEMINI_API_KEY` and `OPENAI_API_KEY`.

Zero-shot runs (one folder per model under `results/`; answers are cached, so reruns only query missing shots):
```bash
python xg_vlm_pilot.py --backend ollama --model gemma4:26b --n 300
```
```bash
python xg_vlm_pilot.py --backend anthropic --model claude-opus-5-5 --n 300
```
```bash
python xg_vlm_pilot.py --backend gemini --model gemini-4-carbon --thinking-level medium --n 300
```
```bash
python xg_vlm_pilot.py --backend openai --model gpt-6.2-astra --thinking-level medium --n 300
```

Few-shot (example sets are in `results/fewshot_examples*.json`) and body-part context:
```bash
python xg_vlm_pilot.py --backend anthropic --model claude-opus-5-5 --shots 5 --n 300
```
```bash
python xg_vlm_pilot.py --backend anthropic --model claude-opus-5-5 --shots 10 --example-set B --n 300
```
```bash
python xg_vlm_pilot.py --backend anthropic --model claude-opus-5-5 --context --n 300
```

All pre-shot attributes as text next to the image, and the freeze frame as coordinates instead of the image:
```bash
python xg_vlm_pilot.py --backend anthropic --model claude-opus-5-5 --all-features --n 300
```
```bash
python xg_vlm_pilot.py --backend anthropic --model claude-opus-5-5 --input coords --n 300
```

Analysis (tables in `results/*.md`):
```bash
python baselines.py
```
```bash
python compare_models.py
```
```bash
python recalibrate.py
```
```bash
python fewshot_analysis.py
```
```bash
python context_ablation.py
```
```bash
python features_ablation.py
```
```bash
python coords_analysis.py
```
```bash
python error_analysis.py
```
```bash
python cost_summary.py
```

`compare_models.py --figures` also redraws the figures. Gemma few-shot runs need a GPU; see
`docs/colab_gemma_fewshot.md`.

## Files
- `xg_vlm_pilot.py`: full pipeline (data loading, rendering, VLM queries, evaluation); one run per model and setting
- `make_fewshot_examples.py`: chooses the fixed, nested few-shot example sets
- `compare_models.py`: all runs on the common shots, match-clustered bootstrap CIs, distance bands, figures
- `baselines.py`: map-only baseline and the all-pre-shot-features logistic model (same frozen folds)
- `recalibrate.py`, `fewshot_analysis.py`, `context_ablation.py`, `features_ablation.py`, `coords_analysis.py`,
  `error_analysis.py`, `cost_summary.py`: ablation, error and cost tables
- `CLAUDE.md`: project context, results, design decisions and known gotchas
- `results/<run>/`: `summary.txt`, `shots.csv` (per-shot predictions and rationales), `vlm_cache.jsonl` (answers
  with token usage), and for newer runs `calls.jsonl` (every API call) and `run_config.json` (settings)
- `results/main_table.md`, `distance_bands.md`, `recalibration.md`, `fewshot.md`, `context_ablation.md`,
  `features_ablation.md`, `coords_vs_image.md`, `error_analysis.md`, `baselines.md`, `cost_summary.md` (and
  matching CSVs); `results/cost_log.csv` (one row per run); `results/baseline_coefficients.md`
- `results/figure_frontier.png`, `figure_forest.png`, `figure_paired.png`, `figure_error_gallery.png`: full-paper figures
- `results/fewshot_examples.json`, `results/fewshot_examples_setB.json`: example sets (shot ids, labels, seed)
- `results/baseline_folds.json`: frozen match-to-fold assignment of the logistic baseline
- `results/figure1.png`: Figure 1 of the abstract; `results/calibration_all_models.png`: reliability curves
- `docs/`: abstract, literature review, collaboration brief, Colab guide for Gemma few-shot runs, paper notes

## Authors
- Houman Kamran Habibkhani, University of the Pacific
- Hootan Kamran Habibkhani, Northeastern University
