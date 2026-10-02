# Seeing Is Not Scoring: Benchmarking Vision-Language Models Against Expected Goals

Code and results for our MIT Sloan Sports Analytics Conference 2027 research abstract.

We ask whether general-purpose vision-language models (VLMs) can estimate expected goals (xG)
from an image of a shot situation alone, and where they diverge from established xG models.

## Data
StatsBomb open data, 2022 FIFA World Cup (competition 43, season 106), loaded via `statsbombpy`.
Each shot's freeze frame is rendered as a half-pitch image (shooter, teammates, opponents, goalkeeper).

## Models
- Gemma 4 (26B), open-weight, run locally via Ollama
- Claude Sonnet 5.5 and Claude Opus 5.5 (Anthropic API)
- Gemini 3.8 Flash (Gemini API)

All models receive the same zero-shot, image-only prompt (see `PROMPT` in `xg_vlm_pilot.py`).
Comparisons: StatsBomb xG and a logistic baseline (distance, angle, header, defenders in shooting cone)
trained with match-grouped cross-validation.

## Reproduce
pip install statsbombpy mplsoccer scikit-learn scipy anthropic google-genai requests

python xg_vlm_pilot.py --backend ollama --model gemma4:26b --n 300
python xg_vlm_pilot.py --backend anthropic --model claude-sonnet-5-5 --n 300
python xg_vlm_pilot.py --backend anthropic --model claude-opus-5-5 --n 300
python xg_vlm_pilot.py --backend gemini --model gemini-3.8-flash --n 300

API keys are read from the ANTHROPIC_API_KEY and GEMINI_API_KEY environment variables.

## Files
- `xg_vlm_pilot.py`: full pipeline (data loading, rendering, VLM queries, evaluation)
- `*_summary.txt`: per-model results
- `*_shots.csv`: per-shot predictions, StatsBomb xG, baseline xG, and model rationales
- `calibration_all_models.png`: calibration of all models vs StatsBomb xG

## Authors
[Your name], Northeastern University
[Co-author name], [Institution]
