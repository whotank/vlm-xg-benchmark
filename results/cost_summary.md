# API spend

From `results/cost_log.csv` (37 rows). Output tokens include thinking/reasoning tokens, which all three providers bill as output; the thinking column is the part the provider reports as thinking. Prices and sources per row are in the log. Gemma runs locally (no API cost). Latency = seconds per API call (median / 90th percentile) with parallel workers, including provider-side queueing.

## Per run

| Stage | Run | Settings | Shots | Calls (failed) | Input | Cache write | Cache read | Output (thinking) | Cost | Per shot | Latency s | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3: Gemini 4 zero-shot | gemini_4_carbon | thinking medium | 300 | 300 (0) | 385,800 | 0 | 0 | 45,924 (32,013) | $1.23 | $0.0041 | 7 / 42 | complete |
| 3: Gemini 4 zero-shot | gemini_4_hydrogen | thinking medium | 300 | 300 (0) | 385,800 | 0 | 0 | 185,776 (171,738) | $1.97 | $0.0066 | 5 / 6 | complete |
| 3: Gemini 4 zero-shot | gemini_4_neon | thinking medium | 300 | 300 (0) | 386,700 | 0 | 0 | 191,519 (177,869) | $4.03 | $0.0134 | 9 / 12 | complete |
| 4: few-shot set A | claude_opus_5_5_k1 | k=1 set A | 300 | 306 (0) | 269,892 | 0 | 0 | 60,789 (39,730) | $2.30 | $0.0077 | 3 / 4 | complete |
| 4: few-shot set A | claude_opus_5_5_k10 | k=10 set A | 300 | 300 (0) | 163,200 | 1,673 | 1,003,027 | 71,480 (51,241) | $2.29 | $0.0076 | 3 / 4 | complete |
| 4: few-shot set A | claude_opus_5_5_k5 | k=5 set A | 300 | 300 (0) | 163,200 | 3,352 | 499,448 | 70,807 (50,703) | $2.19 | $0.0073 | 3 / 4 | complete |
| 4: few-shot set A | gemini_4_carbon_k1 | thinking medium, k=1 set A | 300 | 300 (0) | 717,900 | 0 | 0 | 43,367 (30,110) | $1.87 | $0.0062 | 7 / 41 | complete |
| 4: few-shot set A | gemini_4_carbon_k10 | thinking medium, k=10 set A | 300 | 300 (0) | 3,214,989 | 0 | 415,611 | 68,006 (54,493) | $7.15 | $0.0238 | 13 / 51 | complete |
| 4: few-shot set A | gemini_4_carbon_k5 | thinking medium, k=5 set A | 300 | 300 (0) | 1,762,038 | 0 | 250,362 | 52,896 (39,294) | $4.08 | $0.0136 | 9 / 47 | complete |
| 5: body-part context | claude_opus_5_5_context | body part | 300 | 300 (0) | 155,502 | 0 | 0 | 69,086 (47,955) | $2.00 | $0.0067 | 3 / 4 | complete |
| 5: body-part context | gemini_4_carbon_context | thinking medium, body part | 300 | 300 (0) | 388,134 | 0 | 0 | 42,395 (28,784) | $1.20 | $0.0040 | 7 / 37 | complete |
| 6: few-shot set B | claude_opus_5_5_k10_setB | k=10 set B | 300 | 300 (0) | 163,200 | 3,349 | 1,001,351 | 67,996 (47,684) | $2.23 | $0.0074 | 3 / 4 | complete |
| 6: few-shot set B | claude_opus_5_5_k1_setB | k=1 set B | 300 | 300 (0) | 264,600 | 0 | 0 | 63,639 (43,378) | $2.33 | $0.0078 | 3 / 4 | complete |
| 6: few-shot set B | claude_opus_5_5_k5_setB | k=5 set B | 300 | 300 (0) | 163,200 | 1,676 | 501,124 | 67,487 (47,421) | $2.11 | $0.0070 | 3 / 4 | complete |
| 6: few-shot set B | gemini_4_carbon_k10_setB | thinking medium, k=10 set B | 300 | 300 (0) | 3,053,406 | 0 | 577,194 | 50,382 (36,824) | $6.67 | $0.0222 | 12 / 65 | complete |
| 6: few-shot set B | gemini_4_carbon_k1_setB | thinking medium, k=1 set B | 300 | 300 (0) | 717,900 | 0 | 0 | 45,843 (32,406) | $1.89 | $0.0063 | 8 / 57 | complete |
| 6: few-shot set B | gemini_4_carbon_k5_setB | thinking medium, k=5 set B | 300 | 300 (0) | 1,719,952 | 0 | 292,448 | 45,477 (31,958) | $3.92 | $0.0131 | 9 / 48 | complete |
| coords: freeze frame as text (perception vs judgment) | claude_opus_5_5_coords | coordinates as text (no image) | 300 | 300 (0) | 110,020 | 0 | 0 | 79,909 (58,266) | $2.04 | $0.0068 | 3 / 4 | complete |
| coords: freeze frame as text (perception vs judgment) | gemini_4_carbon_coords | thinking medium, coordinates as text (no image) | 300 | 300 (0) | 132,641 | 0 | 0 | 144,653 (129,953) | $1.71 | $0.0057 | 12 / 51 | complete |
| extra: zero-shot rerun (noise control) | claude_opus_5_5_rerun | zero-shot | 300 | 300 (0) | 152,100 | 0 | 0 | 68,851 (47,820) | $1.99 | $0.0066 | 3 / 4 | complete |
| extra: zero-shot rerun (noise control) | gemini_4_carbon_rerun | thinking medium | 300 | 300 (0) | 385,800 | 0 | 0 | 45,261 (31,259) | $1.22 | $0.0041 | 7 / 47 | complete |
| features: VLM + all pre-shot features | claude_opus_5_5_allfeat | all pre-shot features | 300 | 300 (0) | 222,301 | 0 | 0 | 28,852 (4,854) | $1.47 | $0.0049 | 1 / 2 | complete |
| features: VLM + all pre-shot features | gemini_4_carbon_allfeat | thinking medium, all pre-shot features | 300 | 300 (0) | 434,972 | 0 | 0 | 40,209 (25,207) | $1.27 | $0.0042 | 7 / 45 | complete |
| features: VLM + all pre-shot features | gpt_6.2_astra_allfeat | thinking medium, all pre-shot features | 300 | 300 (0) | 168,418 | 0 | 0 | 14,464 (247) | $1.93 | $0.0064 | 3 / 4 | complete |
| hybrid: perception | claude_opus_5_5_perception | perception: read distance, angle, opponents in triangle from the image | 300 | 300 (0) | 205,500 | 0 | 0 | 181,067 (170,389) | $4.44 | $0.0148 | 6 / 8 | complete |
| hybrid: perception | gemini_4_carbon_perception | thinking medium, perception: read distance, angle, opponents in triangle from the image | 150 | 150 (0) | 212,250 | 0 | 0 | 562,612 (557,651) | $6.05 | $0.0403 | 36 / 162 | complete |
| hybrid: perception | gpt_6.2_astra_perception | thinking medium, perception: read distance, angle, opponents in triangle from the image | 300 | 300 (0) | 155,700 | 0 | 0 | 67,355 (58,673) | $3.94 | $0.0131 | 7 / 14 | complete |
| optional: GPT-6.2 Astra zero-shot | gpt_6.2_astra | thinking medium | 300 | 300 (0) | 120,300 | 0 | 0 | 20,487 (7,384) | $1.78 | $0.0059 | 4 / 6 | complete |
| probe: contamination | probe_identify_claude_opus_5_5 | probe: identify the match from the image | 84 | 84 (0) | 53,928 | 0 | 0 | 6,483 (603) | $0.35 | $0.0041 | 1 / 2 | complete |
| probe: contamination | probe_recall_claude_opus_5_5 | probe: recall the shot from match, clock and player (text only) | 84 | 84 (0) | 28,552 | 0 | 0 | 18,771 (10,504) | $0.49 | $0.0058 | 3 / 4 | complete |
| probe: contamination | probe_identify_gemini_4_carbon | thinking medium, probe: identify the match from the image | 84 | 84 (0) | 117,180 | 0 | 0 | 99,879 (94,542) | $1.23 | $0.0147 | 19 / 47 | complete |
| probe: contamination | probe_recall_gemini_4_carbon | thinking medium, probe: recall the shot from match, clock and player (text only) | 84 | 84 (0) | 29,045 | 0 | 0 | 24,788 (18,593) | $0.31 | $0.0036 | 7 / 10 | complete |
| probe: contamination | probe_identify_gpt_6.2_astra | thinking medium, probe: identify the match from the image | 84 | 84 (0) | 42,588 | 0 | 0 | 6,699 (2,251) | $0.61 | $0.0072 | 4 / 6 | complete |
| probe: contamination | probe_recall_gpt_6.2_astra | thinking medium, probe: recall the shot from match, clock and player (text only) | 84 | 84 (0) | 20,230 | 0 | 0 | 5,908 (985) | $0.40 | $0.0047 | 3 / 4 | complete |
| setup: smoke test | setup_smoke_anthropic | zero-shot | 1 | 1 (0) | 507 | 0 | 0 | 151 (81) | $0.01 | $0.0050 |  | complete |
| setup: smoke test | setup_smoke_gemini | thinking medium | 1 | 1 (0) | 1,286 | 0 | 0 | 606 (561) | $0.01 | $0.0065 |  | complete |
| setup: smoke test | setup_smoke_openai | thinking medium | 1 | 1 (0) | 401 | 0 | 0 | 43 (0) | $0.00 | $0.0049 |  | complete |

## Per provider

| Provider | Rows | Calls | Cost | Limit |
|---|---|---|---|---|
| anthropic | 14 | 3,475 | $26.22 | $30 |
| gemini | 17 | 4,219 | $45.83 | $50 |
| openai | 6 | 1,069 | $8.66 | $30 |
| **Total** | 37 | 8,763 | **$80.71** | $100 |

