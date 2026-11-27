# Gemma 4 few-shot runs in Colab (Stage 6)

Gemma runs locally through Ollama, so it needs a GPU. These cells reproduce the few-shot design used for the API
models (example set A, k = 1, 5, 10; outcome labels only). Set B is for API models only. The same session also
reruns Gemma zero-shot (`--tag colab`), so few-shot runs are compared with a zero-shot run from the same
environment: same Ollama build, same model digest, same rendered images.

Use a GPU runtime with enough memory for `gemma4:26b` plus a 16k-token context (an A100 or L4 class GPU; the
original zero-shot Gemma run used the same model tag).

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
!pip install -q statsbombpy mplsoccer==1.8.1 matplotlib==3.11.2 scikit-learn scipy requests
```

matplotlib and mplsoccer are pinned to the versions that rendered the images of every newer API run
(539 x 421 px). The check in step 5 confirms that the images match. No API key is needed for Ollama. The first
run downloads the StatsBomb data (about a minute) and caches it in `data_cache/` (not committed).

Record the environment, including the exact model build (the digest changes if the tag is re-published):

```bash
!(ollama --version; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; curl -s http://localhost:11434/api/tags | python -c "import json,sys; [print(m['name'], m['digest'], m.get('size')) for m in json.load(sys.stdin)['models']]"; pip freeze | grep -iE "^(matplotlib|mplsoccer|statsbombpy|numpy|pandas|scikit-learn)==") > results/gemma_colab_environment.txt; cat results/gemma_colab_environment.txt
```

## 3. Zero-shot rerun in this session

```bash
!python xg_vlm_pilot.py --backend ollama --model gemma4:26b --n 300 --tag colab --stage "6: Gemma zero-shot rerun (Colab)"
```

This writes `results/gemma4_26b_colab`. The original zero-shot folder `results/gemma4_26b` is never touched.
Compare the two (`compare_models.py` lists the Colab run in the "rerun" group): their difference sizes what changed
between environments (Ollama build, model digest, rendering) before any examples are added.

## 4. Few-shot runs (example set A)

```bash
!python xg_vlm_pilot.py --backend ollama --model gemma4:26b --shots 1 --n 300 --stage "6: Gemma few-shot set A"
!python xg_vlm_pilot.py --backend ollama --model gemma4:26b --shots 5 --n 300 --stage "6: Gemma few-shot set A"
!python xg_vlm_pilot.py --backend ollama --model gemma4:26b --shots 10 --n 300 --stage "6: Gemma few-shot set A"
```

What the script does for these runs (no extra flags needed):

- Same settings as the original zero-shot Gemma run: `temperature 0`, `seed 7`, `format: json`, `think: false`.
- Few-shot prompts go to `/api/chat` as one user turn per example (image + `Outcome: GOAL` / `Outcome: NO GOAL`),
  then the shot to evaluate with the zero-shot prompt plus one line explaining the examples.
- `num_ctx` is set to 16384 for few-shot runs so the 11 images of k = 10 are not truncated (override with
  `--num-ctx`, e.g. `--num-ctx 8192` if the GPU runs out of memory and the check below passes).
- Answers are cached in `results/gemma4_26b_k{1,5,10}/vlm_cache.jsonl`; rerun the same command to fill gaps.

## 5. Checks

Truncation. When a prompt is longer than `num_ctx`, Ollama cuts its start without an error and writes a
"truncating input prompt" warning to its server log. This must print 0:

```bash
!grep -ci "truncat" /content/ollama.log
```

If it is not 0, find which run was affected (the log lines carry the prompt length and the limit). Delete that run's
folder and rerun it with a larger `--num-ctx`. For information, the prompt token counts per k (from Ollama's
`prompt_eval_count`) should grow by roughly one image per example:

```bash
!python -c "import json, statistics as s; [print(k, s.median(json.loads(l)['usage']['input_tokens'] for l in open(f'results/gemma4_26b_k{k}/vlm_cache.jsonl'))) for k in (1, 5, 10)]"
```

Images. The rendered images should be byte-identical to those of the API runs. This compares the SHA-1 stored with
each answer and should report 300 of 300:

```bash
!python -c "import json; a={json.loads(l)['shot_id']: json.loads(l)['image_sha1'] for l in open('results/gemini_4_carbon/vlm_cache.jsonl')}; b={json.loads(l)['shot_id']: json.loads(l)['image_sha1'] for l in open('results/gemma4_26b_colab/vlm_cache.jsonl')}; print(sum(a.get(k) == v for k, v in b.items()), 'of', len(b), 'images identical')"
```

## 6. Save the results

Do not commit or push from Colab: the session has no write access to the repository. Copy the run folders to
Google Drive and/or download a zip, then add them to the repository from your own machine.

```bash
!python compare_models.py && python fewshot_analysis.py
!zip -qr /content/gemma_colab_results.zip results/gemma4_26b_colab results/gemma4_26b_k1 results/gemma4_26b_k5 results/gemma4_26b_k10 results/gemma_colab_environment.txt /content/ollama.log
```

```python
from google.colab import drive, files
drive.mount("/content/drive")
!cp /content/gemma_colab_results.zip /content/drive/MyDrive/
files.download("/content/gemma_colab_results.zip")
```

On your machine, unzip into the repository root. `ollama.log` lands in `content/`: keep it with the results, or
delete it after the truncation check. The zip leaves out `results/cost_log.csv` so that it cannot overwrite newer
rows. Rerun `python cost_summary.py` (it adds and rebuilds the cost-log rows from each folder's `calls.jsonl`),
`compare_models.py` and `fewshot_analysis.py`, and commit. `compare_models.py` and
`fewshot_analysis.py` pick up the new folders automatically: they count a run only once it covers all 300 shots,
and `fewshot_analysis.py` uses the Colab zero-shot run as Gemma's same-environment k = 0. Ollama runs have no API
cost; the cost log records their token counts with the price source "local model, no API cost".
