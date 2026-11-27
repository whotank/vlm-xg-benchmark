"""
Few-shot in-context learning (Stage 4/6). For each model with few-shot runs, compare k = 0 (zero-shot),
1, 5 and 10 labelled examples (example set A, and set B where it exists) on compare_models.py's common shots.

Reported with match-clustered bootstrap 95% CIs (2,000 resamples of matches, seed 0):
  AUC, Brier, log loss, mean predicted xG, and the paired change in Brier and AUC relative to k = 0;
  mean predicted xG by distance band, where the zero-shot errors concentrate.
Examples are labelled with actual outcomes only (see results/fewshot_examples*.json).

Notes for interpretation:
  - API runs are not deterministic, so part of any k-to-k or set A vs set B difference is run-to-run
    noise. If results/<model>_rerun exists (the zero-shot setting queried again), it is shown as a second
    k = 0 row; the gap between the two k = 0 rows sizes that noise.
  - The original zero-shot Opus run used images rendered in the authors' environment; all few-shot runs
    used images re-rendered here with the same code (the rerun row uses the re-rendered images too).

Outputs: results/fewshot.md, results/fewshot.csv
Run: python fewshot_analysis.py
"""
import os
import numpy as np
import pandas as pd
import compare_models as C


WINDOW_S = 120   # an example within this many seconds of a test shot (same match and half) is disclosed


def k_of(order):
    """Smallest k whose nested example set contains the example with this prompt order."""
    return 1 if order == 1 else 5 if order <= 5 else 10


def leakage_section(base, preds, by_folder, y, idxs):
    """Examples that come from the same match and half as a test shot, within WINDOW_S seconds: the example's image
    can show nearly the same situation, with its outcome. Lists each pair, the affected predictions, and the change
    in ΔBrier vs k=0 when the affected test shots are left out."""
    import json
    import xg_vlm_pilot as P
    shots = P.load_shots_cached(43, 106).set_index("shot_id")
    common = list(base.index)
    t_common = shots.loc[common]
    pairs = []
    for set_, path in sorted(P.EXAMPLE_FILES.items()):
        if not os.path.exists(path):
            continue
        with open(path) as f:
            spec = json.load(f)
        for e in spec["examples"]:
            ex = shots.loc[e["shot_id"]]
            same = t_common[(t_common.match_id == ex.match_id) & (t_common.period == ex.period)]
            for sid, s in same.iterrows():
                dt = int((ex.minute * 60 + ex.second) - (s.minute * 60 + s.second))
                if abs(dt) <= WINDOW_S:
                    pairs.append(dict(set=set_, order=e["order"], k=k_of(e["order"]), example=e["shot_id"],
                                      ex_desc=f"{ex.player} ({ex.team}), {ex.minute}:{ex.second:02d}, "
                                              f"{'goal' if ex.goal else 'no goal'}",
                                      test=sid, test_desc=f"{s.player} ({s.team}), {s.minute}:{s.second:02d}, "
                                                          f"{'goal' if s.goal else 'no goal'}",
                                      dt=dt, same_team=ex.team == s.team, match=f"{ex.home_team} vs {ex.away_team}"))
    L = ["## Examples close to a test shot", "",
         f"Few-shot examples from the same match and half as an evaluation shot, within {WINDOW_S} seconds (dt = example "
         "time − test-shot time). An example image that shows nearly the same situation, labelled with its outcome, "
         "can inform the test shot directly. Each example enters at the k shown and stays in the larger sets.", ""]
    if not pairs:
        return L + ["None.", ""]
    L += ["| Set | Example (prompt order, first k) | Test shot | dt | Same team | Match |", "|---|---|---|---|---|---|"]
    for p in pairs:
        L.append(f"| {p['set']} | `{p['example'][:8]}` {p['ex_desc']} (order {p['order']}, k={p['k']}) | "
                 f"`{p['test'][:8]}` {p['test_desc']} | {p['dt']:+d} s | {'yes' if p['same_team'] else 'no'} | {p['match']} |")
    L += ["", "Predictions for the affected test shots (zero-shot, the zero-shot rerun, and every run that includes the "
          "example), and the change in each run's ΔBrier vs k=0 when the affected shots are dropped:", "",
          "| Test shot | Model | Run | xG | Zero-shot xG | StatsBomb |", "|---|---|---|---|---|---|"]
    pos = {s: i for i, s in enumerate(common)}
    drop = {}
    for p in pairs:
        i = pos[p["test"]]
        for lab, fol in C.BASE:
            runs = [f"{fol}_k{k}" + ("" if p["set"] == "A" else f"_set{p['set']}") for k in (1, 5, 10) if k >= p["k"]]
            runs = [r for r in runs if r in by_folder]
            if not runs or fol not in by_folder:
                continue
            z = preds[by_folder[fol]][i]
            for r in [f"{fol}_rerun"] + runs:
                if r in by_folder:
                    L.append(f"| `{p['test'][:8]}` | {lab} | {by_folder[r]} | {preds[by_folder[r]][i]:.3f} | {z:.3f} | "
                             f"{preds['StatsBomb xG'][i]:.3f} |")
            for r in runs:
                drop.setdefault((lab, fol, r), set()).add(p["test"])
    L += ["", "| Run | Shots dropped | ΔBrier vs k=0, all shots | without the affected shots |", "|---|---|---|---|"]
    for (lab, fol, r), ids in drop.items():
        keep = ~np.isin(np.array(common), sorted(ids))
        p0, p1 = preds[by_folder[fol]], preds[by_folder[r]]
        full = C.brier(y, p1) - C.brier(y, p0)
        part = C.brier(y[keep], p1[keep]) - C.brier(y[keep], p0[keep])
        L.append(f"| {by_folder[r]} | {len(ids)} | {full:+.4f} | {part:+.4f} |".replace("-", "−"))
    return L + [""]


def main():
    runs = C.load_runs(verbose=False)
    common = C.common_shots(runs, verbose=False)
    base, preds, groups, folders = C.assemble(runs, common)
    by_folder = {f: lab for lab, f in folders.items()}
    y = base.goal.values.astype(int)
    idxs = C.cluster_bootstrap(base.match_id.values)
    bands = pd.cut(base.distance, C.BANDS, labels=False).astype(int).values
    nb = len(C.BAND_LABELS)

    def boot(fn, p):
        return np.array([fn(y[ix], p[ix]) for ix in idxs])

    def boot_bands(p):
        out = np.full((len(idxs), nb), np.nan)
        for b, ix in enumerate(idxs):
            cnt = np.bincount(bands[ix], minlength=nb)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[b] = np.bincount(bands[ix], weights=p[ix], minlength=nb) / np.where(cnt > 0, cnt, np.nan)
        return out

    rows, band_rows, models = [], [], []
    for lab, fol in C.BASE:
        variants = [("0", "zero-shot", fol), ("0", "zero-shot rerun", f"{fol}_rerun"),
                    ("0", "zero-shot rerun (Colab)", f"{fol}_colab")]
        variants += [(str(k), "set A", f"{fol}_k{k}") for k in (1, 5, 10)]
        variants += [(str(k), "set B", f"{fol}_k{k}_setB") for k in (1, 5, 10)]
        variants = [(k, s, f) for k, s, f in variants if f in by_folder]
        if not any(k != "0" for k, _, _ in variants):
            continue
        models.append(lab)
        p0 = preds[by_folder[fol]]
        b0 = {m: boot(C.METRICS[m], p0) for m in ("brier", "auc")}
        for k, s, f in variants:
            p = preds[by_folder[f]]
            r = dict(model=lab, k=int(k), examples=s, folder=f)
            bt = {}
            for m in ("auc", "brier", "logloss", "mean_xg"):
                bt[m] = boot(C.METRICS[m], p)
                r[m] = C.METRICS[m](y, p)
                r[f"{m}_lo"], r[f"{m}_hi"] = C.ci(bt[m])
            if f != fol:
                for m in ("brier", "auc"):
                    d = bt[m] - b0[m]
                    r[f"d{m}_vs_k0"] = r[m] - C.METRICS[m](y, p0)
                    r[f"d{m}_vs_k0_lo"], r[f"d{m}_vs_k0_hi"] = C.ci(d)
                    r[f"d{m}_vs_k0_p"] = C.boot_p(d)
            # against the zero-shot rerun from the few-shot runs' environment (Gemma: the Colab session's rerun)
            rerun = f"{fol}_colab" if f"{fol}_colab" in by_folder else f"{fol}_rerun"
            if k != "0" and rerun in by_folder:
                pr = preds[by_folder[rerun]]
                d = bt["brier"] - boot(C.brier, pr)
                r["dbrier_vs_rerun"] = r["brier"] - C.brier(y, pr)
                r["dbrier_vs_rerun_lo"], r["dbrier_vs_rerun_hi"] = C.ci(d)
                r["dbrier_vs_rerun_p"] = C.boot_p(d)
            rows.append(r)
            bb = boot_bands(p)
            for j, bl in enumerate(C.BAND_LABELS):
                lo, hi = C.ci(bb[:, j])
                band_rows.append(dict(model=lab, k=int(k), examples=s, band=bl, n=int((bands == j).sum()),
                                      mean=p[bands == j].mean(), lo=lo, hi=hi))
    if not rows:
        raise SystemExit("no few-shot runs found")
    df, bdf = pd.DataFrame(rows), pd.DataFrame(band_rows)
    df.to_csv(os.path.join(C.RES, "fewshot.csv"), index=False)

    f3 = lambda x, lo, hi: C.fmt(x, lo, hi)
    fd = lambda x, lo, hi: C.fmt(x, lo, hi, sign=True)
    obs = [y[bands == j].mean() for j in range(nb)]
    cnt = [int((bands == j).sum()) for j in range(nb)]
    L = ["# Few-shot in-context learning", "",
         f"Evaluation set: the {len(y)} common shots of compare_models.py ({y.sum()} goals; observed rate "
         f"{y.mean():.3f}). Examples come from shots outside the 300-shot test sample and are labelled with the "
         "actual outcome only (\"Outcome: GOAL\" / \"Outcome: NO GOAL\"); sets are nested (k=1 ⊂ k=5 ⊂ k=10). "
         "Brackets: 95% match-clustered bootstrap intervals (2,000 resamples of matches, seed 0); Δ vs k=0 is "
         "paired on the same resamples, p = two-sided bootstrap p-value. API runs are not deterministic: compare "
         "any difference with the gap between the two k=0 rows where a rerun exists.", "",
         "Claude Opus 5.5's \"zero-shot\" k=0 row is the original run, whose images were rendered in the authors' "
         "environment; every few-shot run and the \"zero-shot rerun\" row (`claude_opus_5_5_rerun`) used images "
         "rendered in this environment with the same code. The last column compares each few-shot run with that "
         "same-environment rerun. Gemini 4 Carbon's k=0 run was already rendered here.", ""]
    for lab in models:
        g = df[df.model == lab]
        has_rr = "dbrier_vs_rerun" in g and g.dbrier_vs_rerun.notna().any()
        L += [f"## {lab}", "",
              "| k | Examples | AUC | Brier | Log loss | Mean xG | ΔBrier vs k=0 | p | ΔAUC vs k=0 |"
              + (" ΔBrier vs zero-shot rerun | p |" if has_rr else ""),
              "|---|---|---|---|---|---|---|---|---|" + ("---|---|" if has_rr else "")]
        for _, r in g.iterrows():
            has_d = not pd.isna(r.get("dbrier_vs_k0"))
            rr = ""
            if has_rr:
                rr = (f" {fd(r.dbrier_vs_rerun, r.dbrier_vs_rerun_lo, r.dbrier_vs_rerun_hi)} | {C.fmt_p(r.dbrier_vs_rerun_p)} |"
                      if not pd.isna(r.get("dbrier_vs_rerun")) else "  |  |")
            L.append(f"| {r.k} | {r.examples} | {f3(r.auc, r.auc_lo, r.auc_hi)} | {f3(r.brier, r.brier_lo, r.brier_hi)} | "
                     f"{f3(r.logloss, r.logloss_lo, r.logloss_hi)} | {f3(r.mean_xg, r.mean_xg_lo, r.mean_xg_hi)} | "
                     + (f"{fd(r.dbrier_vs_k0, r.dbrier_vs_k0_lo, r.dbrier_vs_k0_hi)} | {C.fmt_p(r.dbrier_vs_k0_p)} | "
                        f"{fd(r.dauc_vs_k0, r.dauc_vs_k0_lo, r.dauc_vs_k0_hi)} |" if has_d else " |  |  |") + rr)
        L += ["", "Mean predicted xG by distance band (yards):", "",
              "| k | Examples | " + " | ".join(f"{b} (n={c})" for b, c in zip(C.BAND_LABELS, cnt)) + " |",
              "|---|---" + "|---" * nb + "|",
              "| | Observed rate | " + " | ".join(f"{o:.2f}" for o in obs) + " |",
              "| | StatsBomb xG | " + " | ".join(f"{preds['StatsBomb xG'][bands == j].mean():.2f}" for j in range(nb)) + " |"]
        for (k, s), gb in bdf[bdf.model == lab].groupby(["k", "examples"], sort=False):
            gb = gb.set_index("band")
            L.append(f"| {k} | {s} | " + " | ".join(C.fmt(gb.loc[b, 'mean'], gb.loc[b, 'lo'], gb.loc[b, 'hi'], nd=2)
                                                  for b in C.BAND_LABELS) + " |")
        L.append("")
    L += leakage_section(base, preds, by_folder, y, idxs)
    with open(os.path.join(C.RES, "fewshot.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print("Saved results/fewshot.md and results/fewshot.csv")


if __name__ == "__main__":
    main()
