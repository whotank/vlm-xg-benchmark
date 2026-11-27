# Gemma 4 few-shot runs in Colab (Stage 6)

Gemma runs locally through Ollama, so it needs a GPU. These cells reproduce the few-shot design used for
the API models (example set A, k = 1, 5, 10; outcome labels only). Set B is for API models only.

Use a GPU runtime with enough memory for `gemma4:26b` plus a 16k-token context (an A100 or L4 class GPU;
the original zero-shot Gemma run used the same model tag).

## 1. Ollama and the model

```bash
!curl -fsSL https://ollama.com/install.sh | sh
!nohup ollama serve > /content/ollama.log 2>&1 &
!sleep 5 && ollama pull gemma4:26b
```

## 2. Code and dependencies

```bash
!git clone -b claude/intelligent-carson-f0stps https://github.com/whotank/vlm-xg-benchmark.git
%cd vlm-xg-benchmark
!pip install -q statsbombpy mplsoccer scikit-learn scipy requests
```

No API key is needed for Ollama. The first run downloads the StatsBomb data (about a minute) and caches it
in `data_cache/` (not committed).

## 3. Few-shot runs (example set A)

```bash
!python xg_vlm_pilot.py --backend ollama --model gemma4:26b --shots 1 --n 300 --stage "6: Gemma few-shot set A"
!python xg_vlm_pilot.py --backend ollama --model gemma4:26b --shots 5 --n 300 --stage "6: Gemma few-shot set A"
!python xg_vlm_pilot.py --backend ollama --model gemma4:26b --shots 10 --n 300 --stage "6: Gemma few-shot set A"
```

What the script does for these runs (no extra flags needed):

- Same settings as the original zero-shot Gemma run: `temperature 0`, `seed 7`, `format: json`, `think: false`.
- Few-shot prompts go to `/api/chat` as one user turn per example (image + `Outcome: GOAL` / `Outcome: NO GOAL`),
  then the shot to evaluate with the zero-shot prompt plus one line explaining the examples.
- `num_ctx` is set to 16384 for few-shot runs so the 11 images of k = 10 are not truncated
  (override with `--num-ctx`, e.g. `--num-ctx 8192` if the GPU runs out of memory and the check below passes).
- Answers are cached in `results/gemma4_26b_k{1,5,10}/vlm_cache.jsonl`; rerun the same command to fill gaps.
- The original zero-shot folder `results/gemma4_26b` is never touched (new runs get new folders).

## 4. Check that the examples reached the model

Ollama silently drops the start of a prompt that exceeds `num_ctx`. The prompt token count is stored per
answer (`usage.input_tokens`, from Ollama's `prompt_eval_count`). It should grow by roughly one image per
example:

```bash
!python -c "import json; t={k: min(json.loads(l)['usage']['input_tokens'] for l in open(f'results/gemma4_26b_k{k}/vlm_cache.jsonl')) for k in (1,5,10)}; print(t, 'tokens per example: k1->k5', (t[5]-t[1])/4, 'k5->k10', (t[10]-t[5])/5)"
```

The two "tokens per example" numbers should be about equal (one image plus a short label each). If the
k5→k10 number is much smaller, the start of the k = 10 prompt was cut: delete that run's folder and rerun
with a larger `--num-ctx`.

## 5. Save the results

```bash
!python compare_models.py && python fewshot_analysis.py
!git add results/gemma4_26b_k1 results/gemma4_26b_k5 results/gemma4_26b_k10 results/cost_log.csv
!git commit -m "Gemma 4 few-shot runs (set A, k = 1, 5, 10)" && git push
```

`compare_models.py` and `fewshot_analysis.py` pick the new folders up automatically (they only count a run
once it covered all 300 shots). Ollama runs have no API cost; the cost log records their token counts with
the price source "local model, no API cost".
