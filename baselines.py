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


def main():
    shots = P.add_baseline(P.load_shots_cached(43, 106))   # original baseline_xg, frozen folds
    cv = folds(shots)
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
