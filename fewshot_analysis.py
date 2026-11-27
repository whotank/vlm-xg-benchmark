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
        variants = [("0", "zero-shot", fol), ("0", "zero-shot rerun", f"{fol}_rerun")]
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
         "any difference with the gap between the two k=0 rows where a rerun exists.", ""]
    for lab in models:
        g = df[df.model == lab]
        L += [f"## {lab}", "",
              "| k | Examples | AUC | Brier | Log loss | Mean xG | ΔBrier vs k=0 | p | ΔAUC vs k=0 |",
              "|---|---|---|---|---|---|---|---|---|"]
        for _, r in g.iterrows():
            has_d = not pd.isna(r.get("dbrier_vs_k0"))
            L.append(f"| {r.k} | {r.examples} | {f3(r.auc, r.auc_lo, r.auc_hi)} | {f3(r.brier, r.brier_lo, r.brier_hi)} | "
                     f"{f3(r.logloss, r.logloss_lo, r.logloss_hi)} | {f3(r.mean_xg, r.mean_xg_lo, r.mean_xg_hi)} | "
                     + (f"{fd(r.dbrier_vs_k0, r.dbrier_vs_k0_lo, r.dbrier_vs_k0_hi)} | {C.fmt_p(r.dbrier_vs_k0_p)} | "
                        f"{fd(r.dauc_vs_k0, r.dauc_vs_k0_lo, r.dauc_vs_k0_hi)} |" if has_d else " |  |  |"))
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
    with open(os.path.join(C.RES, "fewshot.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print("Saved results/fewshot.md and results/fewshot.csv")


if __name__ == "__main__":
    main()
