"""
Spend per run, per stage and per provider for the paper's discussion section.

Every row of results/cost_log.csv that has a run folder is first rebuilt from that folder's calls.jsonl and
run_config.json (xg_vlm_pilot.cost_row: tokens and cost summed over every logged API call, status from the run's
last invocation), so a row written before a run finished cannot go stale. Rows without a folder (the one-call setup
checks) are kept as logged. The rebuilt log is written back to results/cost_log.csv; rows that changed are listed.

Output: results/cost_summary.md (and the rebuilt results/cost_log.csv)
Run: python cost_summary.py
"""
import os, re, json
import numpy as np
import pandas as pd
import xg_vlm_pilot as P

OUT = os.path.join(P.RESULTS, "cost_summary.md")
DUP_NOTE = "shots were answered twice"
NUM = ["shots_answered", "calls", "failed_calls", "input_tokens", "cache_write_tokens", "cached_input_tokens",
       "output_tokens", "cost_usd"]


def rebuild():
    """Rebuild every row that has calls.jsonl + run_config.json; returns (log, list of changed runs)."""
    with P.FileLock("cost_log"):
        old = P.read_cost_log()
        runs = list(old.run) + [d for d in sorted(os.listdir(P.RESULTS))
                                if d not in set(old.run) and os.path.exists(os.path.join(P.RESULTS, d, "calls.jsonl"))
                                and os.path.exists(os.path.join(P.RESULTS, d, "run_config.json"))]
        rows, changed = [], []
        for run in runs:
            prev = old[old.run == run].iloc[0].to_dict() if run in set(old.run) else {}
            out_dir = os.path.join(P.RESULTS, run)
            if not (os.path.exists(os.path.join(out_dir, "calls.jsonl"))
                    and os.path.exists(os.path.join(out_dir, "run_config.json"))):
                rows.append(prev)
                continue
            with open(os.path.join(out_dir, "run_config.json")) as f:
                cfg = json.load(f)
            last = cfg["invocations"][-1]
            status = last.get("status") or "unfinished (no end recorded)"
            # cost_row re-adds the duplicate-answer note itself; keep the other notes (e.g. Gemini cache attempts)
            notes = re.sub(r"(; )?\d+ " + DUP_NOTE + r".*?all calls are billed and counted", "",
                           str(prev.get("notes", ""))).strip("; ")
            row = {k: ("" if v is None else str(v)) for k, v in P.cost_row(run, cfg, out_dir, status, notes).items()}
            if prev and any(str(prev.get(c, "")) != row.get(c, "") for c in ["calls", "shots_answered", "cost_usd",
                                                                             "output_tokens", "status"]):
                changed.append(f"{run}: calls {prev.get('calls')} -> {row['calls']}, cost {prev.get('cost_usd')} -> "
                               f"{row['cost_usd']}, status {prev.get('status')} -> {row['status']}")
            rows.append(row)
        log = pd.DataFrame([{c: r.get(c, "") for c in P.COST_COLS} for r in rows])
        log.to_csv(P.COST_LOG, index=False)
    return log, changed


def main():
    log, changed = rebuild()
    d = log.copy()
    d[NUM] = d[NUM].apply(pd.to_numeric, errors="coerce").fillna(0)
    d["thinking_tokens"] = pd.to_numeric(d.thinking_tokens, errors="coerce")
    d["per_shot"] = d.cost_usd / d.shots_answered.where(d.shots_answered > 0)
    d["settings"] = [", ".join(x for x in [f"thinking {t}" if t != "not set" else "",
                                           f"k={k}" + (f" set {e}" if isinstance(e, str) and e else "") if int(k) else "",
                                           "body part" if str(c) == "True" else (str(c) if str(c) not in ("False", "nan", "") else "")]
                                          if x) or "zero-shot"
                     for t, k, e, c in zip(d.thinking_level, d.k, d.example_set, d.context)]
    lat = {}
    for run in d.run:
        calls = P.read_jsonl(os.path.join(P.RESULTS, run, "calls.jsonl"))
        v = [c.get("latency_s") for c in calls if c.get("latency_s") is not None]
        if v:
            lat[run] = np.percentile(v, [50, 90])
    d["lat"] = [f"{lat[r][0]:.0f} / {lat[r][1]:.0f}" if r in lat else "" for r in d.run]
    d = d.sort_values(["stage", "provider", "run"])
    L = ["# API spend", "", f"From `results/cost_log.csv` ({len(d)} rows), rebuilt from each run's `calls.jsonl` every time "
         "this script runs (rows without a run folder, the one-call setup checks, are kept as logged). Output tokens "
         "include thinking/reasoning tokens, which all three providers bill as output; the thinking column is the part the "
         "provider reports as thinking. Prices and sources per row are in the log. Gemma runs locally (no API cost). "
         "Latency = seconds per API call (median / 90th percentile) with parallel workers, including provider-side "
         "queueing.", "",
         "## Per run", "",
         "| Stage | Run | Settings | Shots | Calls (failed) | Input | Cache write | Cache read | Output (thinking) | Cost | Per shot | Latency s | Status |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in d.iterrows():
        th = "" if pd.isna(r.thinking_tokens) else f" ({int(r.thinking_tokens):,})"
        ps = "" if pd.isna(r.per_shot) else f"${r.per_shot:.4f}"
        L.append(f"| {r.stage} | {r.run} | {r.settings} | {int(r.shots_answered)} | {int(r.calls)} ({int(r.failed_calls)}) | "
                 f"{int(r.input_tokens):,} | {int(r.cache_write_tokens):,} | {int(r.cached_input_tokens):,} | "
                 f"{int(r.output_tokens):,}{th} | ${r.cost_usd:.2f} | {ps} | {r.lat} | {r.status} |")
    s = d.groupby("stage").agg(runs=("run", "size"), calls=("calls", "sum"), cost=("cost_usd", "sum"))
    L += ["", "## Per stage", "", "| Stage | Rows | Calls | Cost |", "|---|---|---|---|"]
    for stage, r in s.iterrows():
        L.append(f"| {stage} | {int(r.runs)} | {int(r.calls):,} | ${r.cost:.2f} |")
    p = d.groupby("provider").agg(runs=("run", "size"), calls=("calls", "sum"), cost=("cost_usd", "sum"),
                                  failed=("failed_calls", "sum"))
    L += ["", "## Per provider", "", "| Provider | Rows | Calls (failed) | Cost | Limit | Left |", "|---|---|---|---|---|---|"]
    for prov, r in p.iterrows():
        lim = P.LIMITS.get(prov, 0)
        L.append(f"| {prov} | {int(r.runs)} | {int(r.calls):,} ({int(r.failed)}) | ${r.cost:.2f} | ${lim:.0f} | "
                 f"${lim - r.cost:.2f} |")
    L += [f"| **Total** | {int(p.runs.sum())} | {int(p.calls.sum()):,} ({int(p.failed.sum())}) | **${p.cost.sum():.2f}** | "
          f"${P.TOTAL_LIMIT:.0f} | ${P.TOTAL_LIMIT - p.cost.sum():.2f} |", ""]
    with open(OUT, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print("\nRows that changed in the rebuild:" + ("".join(f"\n  {c}" for c in changed) if changed else " none"))


if __name__ == "__main__":
    main()
