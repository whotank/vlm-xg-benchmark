"""
Statistical baselines that bracket the VLMs' information (used by compare_models.py and features_ablation.py).

All three are logistic regressions with out-of-fold predictions for every shot, using the frozen match folds of
the original baseline (results/baseline_folds.json), so no match is in its own training data:

  baseline_xg        the original 4-feature baseline: distance, angle, header, defenders in the shooting cone
  baseline_image_xg  "Map-only baseline": the same model without the header flag, so it uses only information
                     drawn in the rendered image (distance, angle, defenders in the cone). It is a subset of what
                     the VLM sees (the VLM also sees the keeper and every other player) but with exact values,
                     where the VLM has to read positions off the pixels.
  baseline_full_xg   every pre-shot attribute the all-features VLM runs receive: geometry (distance, angle,
                     defenders in the cone, keeper distance to goal line and shooter, outfield opponents within
                     2 yards), body part, technique, shot type, play pattern, first-time / one-on-one / open goal /
                     under pressure / aerial won / after a dribble, and the assist (none, height, cross, cut-back,
                     through ball, switch, set-piece delivery). Standardised, L2-regularised (C = 1).
Nothing recorded after the strike (end location, deflection, save details, outcome) is used.

Outputs: results/baselines.csv (all 1,430 shots), results/baselines.md
Run: python baselines.py
"""
import os, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
import xg_vlm_pilot as P

NUMERIC = ["distance", "angle", "defenders_in_cone", "gk_dist_goal_line", "gk_dist_shooter", "close_opponents"]
FLAGS = ["first_time", "one_on_one", "open_goal", "under_pressure", "aerial_won", "follows_dribble",
         "assist_pass", "assist_cross", "assist_cutback", "assist_through", "assist_switch"]
CATEGORICAL = ["body_part", "technique", "shot_type", "play_pattern", "assist_height", "assist_setpiece"]


def folds(shots):
    with open(P.BASELINE_FOLDS) as f:
        m2f = {int(k): v for k, v in json.load(f)["folds"].items()}
    fold = shots.match_id.map(m2f).values
    return [(np.flatnonzero(fold != k), np.flatnonzero(fold == k)) for k in range(5)]


def full_design(shots):
    X = shots[NUMERIC + FLAGS].astype(float)
    dummies = pd.get_dummies(shots[CATEGORICAL].fillna("none").replace("", "none"), drop_first=False).astype(float)
    return pd.concat([X, dummies], axis=1)


def coefficients(shots, cv):
    """results/baseline_coefficients.md: the original baseline's fitted coefficients per fold and on all shots."""
    feats = ["distance", "angle", "header", "defenders_in_cone"]
    y = shots.goal.values
    fits = []
    for k, (tr, te) in enumerate(cv, 1):
        m = LogisticRegression(max_iter=1000).fit(shots[feats].iloc[tr], y[tr])
        fits.append((f"Fold {k} (predicts {len(te)} shots, {int(y[te].sum())} goals; trained on the other "
                     f"{len(tr)} shots, {int(y[tr].sum())} goals)", m))
    fits.append((f"All {len(shots)} shots ({int(y.sum())} goals)", LogisticRegression(max_iter=1000).fit(shots[feats], y)))
    gk_in = sum(any(not q["teammate"] and (q.get("position") or {}).get("name") == "Goalkeeper"
                    and P.in_triangle(q["location"][:2], (r.x, r.y), P.POST_L, P.POST_R) for q in r.ff)
                for r in shots.itertuples())
    L = ["# Logistic baseline: coefficients", "",
         "Model: scikit-learn `LogisticRegression(max_iter=1000)` (default L2 penalty, C = 1, lbfgs) on raw, unscaled "
         "features. Target: goal = 1 if `shot_outcome` is \"Goal\", else 0 (penalties excluded). Features:", "",
         "- `distance`: yards from the shot location to the centre of the goal (120, 40) on StatsBomb's 120 x 80 pitch",
         "- `angle`: angle in radians between the lines from the shot location to the two posts (y = 36 and y = 44)",
         "- `header`: 1 if `shot_body_part` is \"Head\", else 0",
         "- `defenders_in_cone`: number of opponents in the triangle formed by the shot location and the two posts, "
         f"counting every opponent in the freeze frame, goalkeeper included (the keeper is inside the triangle in "
         f"{gk_in} of {len(shots)} shots)", "",
         "The five fold models are the ones whose out-of-fold predictions form `baseline_xg` (folds frozen in "
         "`results/baseline_folds.json`; fold k = model trained without fold k's matches). The all-shots fit is for "
         "interpretation only; it matches an independent check (intercept −0.358, distance −0.093, angle +1.265, "
         "header −1.163, defenders in cone −0.512).", "",
         "## Coefficients (log-odds)", "", "| Fit | Intercept | distance (per yd) | angle (per rad) | header | defenders in cone |",
         "|---|---|---|---|---|---|"]
    sgn = lambda v: f"{v:+.3f}".replace("-", "−")
    for name, m in fits:
        L.append(f"| {name} | {sgn(m.intercept_[0])} | " + " | ".join(sgn(c) for c in m.coef_[0]) + " |")
    L += ["", "## Odds ratios (exp of the coefficient)", "",
          "Per unit of each feature: one yard further out, one radian more goal angle (1 rad = 57°; a 0.1 rad "
          "change multiplies the odds by the tenth root of the value shown), a header instead of a foot shot, one "
          "more opponent in the cone.", "", "| Fit | Intercept (odds) | distance | angle | header | defenders in cone |",
          "|---|---|---|---|---|---|"]
    for name, m in fits:
        L.append(f"| {name} | {np.exp(m.intercept_[0]):.3f} | " + " | ".join(f"{np.exp(c):.3f}" for c in m.coef_[0]) + " |")
    with open(os.path.join(P.RESULTS, "baseline_coefficients.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    return fits


def main():
    shots = P.add_baseline(P.load_shots_cached(43, 106))   # original baseline_xg, frozen folds
    cv = folds(shots)
    coefficients(shots, cv)
    y = shots.goal.values
    shots["baseline_image_xg"] = cross_val_predict(
        LogisticRegression(max_iter=1000), shots[["distance", "angle", "defenders_in_cone"]], y,
        cv=cv, method="predict_proba")[:, 1]
    X = full_design(shots)
    shots["baseline_full_xg"] = cross_val_predict(
        make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)), X, y,
        cv=cv, method="predict_proba")[:, 1]
    out = shots[["shot_id", "match_id", "goal", "sb_xg", "baseline_xg", "baseline_image_xg", "baseline_full_xg"]]
    out.to_csv(os.path.join(P.RESULTS, "baselines.csv"), index=False)

    brier = lambda p: float(np.mean((p - y) ** 2))
    L = ["# Statistical baselines", "",
         f"Out-of-fold predictions for all {len(shots)} non-penalty shots ({int(y.sum())} goals), 5 folds grouped by "
         "match (the frozen folds in `results/baseline_folds.json`). Scores below are over all shots; "
         "`results/main_table.md` reports them on the 298 common shots with match-clustered CIs.", "",
         "| Model | Inputs | AUC | Brier | Mean xG |", "|---|---|---|---|---|"]
    for col, name, inputs in [
            ("baseline_image_xg", "Map-only baseline", "distance, angle, defenders in the cone (no header)"),
            ("baseline_xg", "Logistic baseline (original)", "the above + header flag"),
            ("baseline_full_xg", "Logistic, all pre-shot features", f"{X.shape[1]} columns: geometry incl. keeper "
             "position, body part, technique, shot type, play pattern, situation flags, assist"),
            ("sb_xg", "StatsBomb xG", "StatsBomb's commercial model")]:
        p = shots[col].values
        L.append(f"| {name} | {inputs} | {roc_auc_score(y, p):.3f} | {brier(p):.4f} | {p.mean():.3f} |")
    with open(os.path.join(P.RESULTS, "baselines.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
