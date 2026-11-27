"""
Summarise results/cost_log.csv for the paper's discussion section: spend per run and per provider, tokens,
and cost per answered shot. Prices and their sources are in the log itself (one row per run).

Output: results/cost_summary.md
Run: python cost_summary.py
"""
import os
import pandas as pd

LOG, OUT = os.path.join("results", "cost_log.csv"), os.path.join("results", "cost_summary.md")


def main():
    d = pd.read_csv(LOG)
    num = ["shots_answered", "calls", "failed_calls", "input_tokens", "cache_write_tokens", "cached_input_tokens",
           "output_tokens", "cost_usd"]
    d[num] = d[num].apply(pd.to_numeric, errors="coerce").fillna(0)
    d["thinking_tokens"] = pd.to_numeric(d.thinking_tokens, errors="coerce")
    d["per_shot"] = d.cost_usd / d.shots_answered.where(d.shots_answered > 0)
    d["settings"] = [", ".join(x for x in [f"thinking {t}" if t != "not set" else "",
                                           f"k={k}" + (f" set {e}" if isinstance(e, str) and e else "") if int(k) else "",
                                           "body part" if str(c) == "True" else ""] if x) or "zero-shot"
                     for t, k, e, c in zip(d.thinking_level, d.k, d.example_set, d.context)]
    d = d.sort_values(["stage", "provider", "run"])
    L = ["# API spend", "", f"From `results/cost_log.csv` ({len(d)} rows). Output tokens include thinking/reasoning "
         "tokens, which all three providers bill as output; the thinking column is the part the provider reports as "
         "thinking. Prices and sources per row are in the log. Gemma runs locally (no API cost).", "",
         "## Per run", "",
         "| Stage | Run | Settings | Shots | Calls (failed) | Input | Cache write | Cache read | Output (thinking) | Cost | Per shot | Status |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in d.iterrows():
        th = "" if pd.isna(r.thinking_tokens) else f" ({int(r.thinking_tokens):,})"
        ps = "" if pd.isna(r.per_shot) else f"${r.per_shot:.4f}"
        L.append(f"| {r.stage} | {r.run} | {r.settings} | {int(r.shots_answered)} | {int(r.calls)} ({int(r.failed_calls)}) | "
                 f"{int(r.input_tokens):,} | {int(r.cache_write_tokens):,} | {int(r.cached_input_tokens):,} | "
                 f"{int(r.output_tokens):,}{th} | ${r.cost_usd:.2f} | {ps} | {r.status} |")
    p = d.groupby("provider").agg(runs=("run", "size"), calls=("calls", "sum"), cost=("cost_usd", "sum"))
    L += ["", "## Per provider", "", "| Provider | Rows | Calls | Cost | Limit |", "|---|---|---|---|---|"]
    limits = {"anthropic": 30, "gemini": 50, "openai": 30}
    for prov, r in p.iterrows():
        L.append(f"| {prov} | {int(r.runs)} | {int(r.calls):,} | ${r.cost:.2f} | ${limits.get(prov, 0)} |")
    L += [f"| **Total** | {int(p.runs.sum())} | {int(p.calls.sum()):,} | **${p.cost.sum():.2f}** | $100 |", ""]
    with open(OUT, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
