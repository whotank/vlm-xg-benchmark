"""
Recalibration side analysis (Stage 2). The main results stay raw (zero-shot); this asks how much of each
model's Brier / log-loss gap is miscalibration that a simple monotone map, learned from a few hundred
labelled shots, could remove.

For every run that compare_models.py picks up (same common shots), plus StatsBomb xG and the logistic
baseline as reference rows:
  - Platt scaling: unpenalised logistic regression of goal on logit(p)
  - Isotonic regression: monotone, non-decreasing step function of p
Both are fitted with 5-fold cross-validation grouped by match_id, so each shot is recalibrated by a map
that never saw its match. The folds are the logistic baseline's frozen match-to-fold assignment
(results/baseline_folds.json), so the split does not depend on the scikit-learn version. The out-of-fold predictions are scored
with Brier and log loss; 95% intervals come from the same match-clustered bootstrap as compare_models.py
(2,000 resamples of matches, seed 0), including paired (recalibrated - raw) differences.

Caveats written into the report:
  - The bootstrap resamples the out-of-fold predictions; it does not refit the maps, so intervals cover
    evaluation-set sampling, not fitting variability. A second table repeats the whole procedure over
    50 random grouped 5-fold splits (seeded permutations of matches) to show how much the fold
    assignment alone moves the result.
  - With 42 goals, each training fold has ~34 goals, so even a well-calibrated model (StatsBomb) can get
    slightly worse after recalibration. The StatsBomb and baseline rows show the size of that effect.

Outputs: results/recalibration.md, results/recalibration.csv
Run: python recalibrate.py
"""
import os, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import compare_models as C

EPS = 1e-3
N_SPLITS, N_REPEATS = 5, 50


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_platt(p, y):
    return LogisticRegression(C=np.inf, max_iter=1000).fit(logit(p)[:, None], y)   # C=inf: no penalty


def platt_predict(m, p):
    return m.predict_proba(logit(p)[:, None])[:, 1]


def fit_iso(p, y):
    return IsotonicRegression(y_min=0, y_max=1, increasing=True, out_of_bounds="clip").fit(p, y)


class MatchFolds:
    """Grouped K-fold from the frozen match -> fold map (seed=None) or a seeded permutation of matches."""
    def __init__(self, groups, seed=None, n_splits=N_SPLITS):
        if seed is None:
            with open(os.path.join(C.RES, "baseline_folds.json")) as f:
                m2f = {int(k): v for k, v in json.load(f)["folds"].items()}
        else:
            uniq = np.random.default_rng(seed).permutation(np.unique(groups))
            m2f = {int(m): i % n_splits for i, m in enumerate(uniq)}
        self.fold = np.array([m2f[int(g)] for g in groups])
        self.n_splits = n_splits

    def split(self):
        for k in range(self.n_splits):
            yield np.flatnonzero(self.fold != k), np.flatnonzero(self.fold == k)


def oof(p, y, folds, method):
    out = np.empty_like(p, dtype=float)
    for tr, te in folds.split():
        if method == "platt":
            out[te] = platt_predict(fit_platt(p[tr], y[tr]), p[te])
        else:
            out[te] = fit_iso(p[tr], y[tr]).predict(p[te])
    return np.clip(out, EPS, 1 - EPS)


def main():
    runs = C.load_runs()
    common = C.common_shots(runs)
    base, preds, groups, _ = C.assemble(runs, common)
    y = base.goal.values.astype(int)
    g = base.match_id.values
    idxs = C.cluster_bootstrap(g)
    print(f"Recalibrating {len(preds)} prediction sets on {len(y)} shots ({y.sum()} goals, {len(set(g))} matches)")

    rows = []
    for m, p in preds.items():
        fixed = MatchFolds(g)
        variants = {"raw": p, "platt": oof(p, y, fixed, "platt"), "isotonic": oof(p, y, fixed, "isotonic")}
        boots = {v: {k: np.array([C.METRICS[k](y[ix], q[ix]) for ix in idxs]) for k in ("brier", "logloss")}
                 for v, q in variants.items()}
        full = fit_platt(p, y)
        r = dict(model=m, group=groups[m], n=len(y), goals=int(y.sum()),
                 platt_intercept=float(full.intercept_[0]), platt_slope=float(full.coef_[0, 0]))
        for v, q in variants.items():
            r[f"mean_xg_{v}"] = q.mean()
            for k in ("brier", "logloss"):
                r[f"{k}_{v}"] = C.METRICS[k](y, q)
                r[f"{k}_{v}_lo"], r[f"{k}_{v}_hi"] = C.ci(boots[v][k])
                if v != "raw":
                    d = boots[v][k] - boots["raw"][k]
                    r[f"d{k}_{v}"] = r[f"{k}_{v}"] - r[f"{k}_raw"]
                    r[f"d{k}_{v}_lo"], r[f"d{k}_{v}_hi"] = C.ci(d)
                    r[f"d{k}_{v}_p"] = C.boot_p(d)
        # sensitivity: repeat the grouped CV over random fold assignments
        for v in ("platt", "isotonic"):
            reps = [C.brier(y, oof(p, y, MatchFolds(g, seed=s), v)) for s in range(N_REPEATS)]
            r[f"brier_{v}_rep_mean"] = float(np.mean(reps))
            r[f"brier_{v}_rep_lo"], r[f"brier_{v}_rep_hi"] = np.percentile(reps, [2.5, 97.5])
        rows.append(r)
        print(f"  {m:<34} Brier raw {r['brier_raw']:.4f}  Platt {r['brier_platt']:.4f}  isotonic {r['brier_isotonic']:.4f}")

    df = pd.DataFrame(rows)
    order = {k: i for i, k in enumerate(["zero-shot", "few-shot set A", "few-shot set B", "context", "rerun", "reference"])}
    df = df.sort_values("group", key=lambda s: s.map(order), kind="stable").reset_index(drop=True)
    df.to_csv(os.path.join(C.RES, "recalibration.csv"), index=False)

    f3 = lambda x, lo, hi: C.fmt(x, lo, hi, nd=3)
    fd = lambda x, lo, hi: C.fmt(x, lo, hi, nd=3, sign=True)
    L = ["# Recalibration (side analysis)", "",
         f"Evaluation set: the {len(y)} common shots of compare_models.py ({y.sum()} goals, {len(set(g))} matches). "
         "Main results stay raw (zero-shot); this table shows what a monotone recalibration learned from labelled "
         "shots would change.", "",
         "Method: Platt scaling (unpenalised logistic regression of goal on logit p) and isotonic regression, each "
         f"fitted with {N_SPLITS}-fold cross-validation grouped by match (the baseline's frozen match folds, "
         "results/baseline_folds.json), so every shot is recalibrated by a map that never saw its match. Brackets: 95% match-clustered bootstrap intervals (2,000 resamples of "
         "matches, seed 0) of the out-of-fold scores; Δ = recalibrated − raw (negative = recalibration helps), "
         "p = two-sided bootstrap p-value for Δ = 0. The bootstrap does not refit the maps, so the intervals cover "
         "evaluation sampling only; the last table shows the spread over random fold assignments.", "",
         "StatsBomb xG and the logistic baseline are reference rows: they are already close to calibrated, so their "
         "change shows how much the procedure alone moves a good model with only 42 goals to learn from.", "",
         "## Brier score", "",
         "| Model | Raw | Platt | Δ Platt | p | Isotonic | Δ isotonic | p |", "|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        L.append(f"| {r.model} | {f3(r.brier_raw, r.brier_raw_lo, r.brier_raw_hi)} | "
                 f"{f3(r.brier_platt, r.brier_platt_lo, r.brier_platt_hi)} | "
                 f"{fd(r.dbrier_platt, r.dbrier_platt_lo, r.dbrier_platt_hi)} | {C.fmt_p(r.dbrier_platt_p)} | "
                 f"{f3(r.brier_isotonic, r.brier_isotonic_lo, r.brier_isotonic_hi)} | "
                 f"{fd(r.dbrier_isotonic, r.dbrier_isotonic_lo, r.dbrier_isotonic_hi)} | {C.fmt_p(r.dbrier_isotonic_p)} |")
    L += ["", "## Log loss", "",
          "| Model | Raw | Platt | Δ Platt | p | Isotonic | Δ isotonic | p |", "|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        L.append(f"| {r.model} | {f3(r.logloss_raw, r.logloss_raw_lo, r.logloss_raw_hi)} | "
                 f"{f3(r.logloss_platt, r.logloss_platt_lo, r.logloss_platt_hi)} | "
                 f"{fd(r.dlogloss_platt, r.dlogloss_platt_lo, r.dlogloss_platt_hi)} | {C.fmt_p(r.dlogloss_platt_p)} | "
                 f"{f3(r.logloss_isotonic, r.logloss_isotonic_lo, r.logloss_isotonic_hi)} | "
                 f"{fd(r.dlogloss_isotonic, r.dlogloss_isotonic_lo, r.dlogloss_isotonic_hi)} | {C.fmt_p(r.dlogloss_isotonic_p)} |")
    L += ["", "## What the Platt map does (fitted on all common shots, for interpretation only)", "",
          "Slope < 1: the model's probabilities are too spread out (over-confident ranking of extremes); slope > 1: "
          "too compressed. The mean predicted xG columns show how recalibration moves the overall level "
          f"(observed rate {y.mean():.3f}).", "",
          "| Model | Platt intercept | Platt slope | Mean xG raw | Mean xG Platt (OOF) | Mean xG isotonic (OOF) |",
          "|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        L.append(f"| {r.model} | {r.platt_intercept:+.2f} | {r.platt_slope:.2f} | {r.mean_xg_raw:.3f} | "
                 f"{r.mean_xg_platt:.3f} | {r.mean_xg_isotonic:.3f} |".replace("-", "−"))
    L += ["", f"## Sensitivity to the fold assignment ({N_REPEATS} random grouped {N_SPLITS}-fold splits)", "",
          "Brier after recalibration: value with the fixed match folds used above, then the mean and the "
          f"2.5–97.5% range over {N_REPEATS} random grouped splits (seeded permutations of matches).", "",
          "| Model | Raw | Platt (fixed split) | Platt over splits | Isotonic (fixed split) | Isotonic over splits |",
          "|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        L.append(f"| {r.model} | {r.brier_raw:.4f} | {r.brier_platt:.4f} | {r.brier_platt_rep_mean:.4f} "
                 f"[{r.brier_platt_rep_lo:.4f}, {r.brier_platt_rep_hi:.4f}] | {r.brier_isotonic:.4f} | "
                 f"{r.brier_isotonic_rep_mean:.4f} [{r.brier_isotonic_rep_lo:.4f}, {r.brier_isotonic_rep_hi:.4f}] |")
    with open(os.path.join(C.RES, "recalibration.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("Saved results/recalibration.md and results/recalibration.csv")


if __name__ == "__main__":
    main()
