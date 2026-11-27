# Logistic baseline: coefficients

Model: scikit-learn `LogisticRegression(max_iter=1000)` (default L2 penalty, C = 1, lbfgs) on raw, unscaled features. Target: goal = 1 if `shot_outcome` is "Goal", else 0 (penalties excluded). Features:

- `distance`: yards from the shot location to the centre of the goal (120, 40) on StatsBomb's 120 x 80 pitch
- `angle`: angle in radians between the lines from the shot location to the two posts (y = 36 and y = 44)
- `header`: 1 if `shot_body_part` is "Head", else 0
- `defenders_in_cone`: number of opponents in the triangle formed by the shot location and the two posts, counting every opponent in the freeze frame, goalkeeper included (the keeper is inside the triangle in 1387 of 1430 shots)

The five fold models are the ones whose out-of-fold predictions form `baseline_xg` (folds frozen in `results/baseline_folds.json`; fold k = model trained without fold k's matches). The all-shots fit is for interpretation only; it matches an independent check (intercept −0.358, distance −0.093, angle +1.265, header −1.163, defenders in cone −0.512).

## Coefficients (log-odds)

| Fit | Intercept | distance (per yd) | angle (per rad) | header | defenders in cone |
|---|---|---|---|---|---|
| Fold 1 (predicts 288 shots, 29 goals; trained on the other 1142 shots, 123 goals) | −0.171 | −0.100 | +1.062 | −1.205 | −0.470 |
| Fold 2 (predicts 288 shots, 32 goals; trained on the other 1142 shots, 120 goals) | −0.337 | −0.103 | +1.184 | −1.278 | −0.379 |
| Fold 3 (predicts 287 shots, 36 goals; trained on the other 1143 shots, 116 goals) | −0.339 | −0.094 | +1.253 | −1.006 | −0.564 |
| Fold 4 (predicts 278 shots, 36 goals; trained on the other 1152 shots, 116 goals) | −0.660 | −0.075 | +1.247 | −1.095 | −0.525 |
| Fold 5 (predicts 289 shots, 19 goals; trained on the other 1141 shots, 133 goals) | −0.051 | −0.099 | +1.349 | −1.129 | −0.634 |
| All 1430 shots (152 goals) | −0.358 | −0.093 | +1.265 | −1.163 | −0.512 |

## Odds ratios (exp of the coefficient)

Per unit of each feature: one yard further out, one radian more goal angle (1 rad = 57°; a 0.1 rad change multiplies the odds by the tenth root of the value shown), a header instead of a foot shot, one more opponent in the cone.

| Fit | Intercept (odds) | distance | angle | header | defenders in cone |
|---|---|---|---|---|---|
| Fold 1 (predicts 288 shots, 29 goals; trained on the other 1142 shots, 123 goals) | 0.843 | 0.905 | 2.892 | 0.300 | 0.625 |
| Fold 2 (predicts 288 shots, 32 goals; trained on the other 1142 shots, 120 goals) | 0.714 | 0.902 | 3.268 | 0.278 | 0.685 |
| Fold 3 (predicts 287 shots, 36 goals; trained on the other 1143 shots, 116 goals) | 0.713 | 0.910 | 3.501 | 0.366 | 0.569 |
| Fold 4 (predicts 278 shots, 36 goals; trained on the other 1152 shots, 116 goals) | 0.517 | 0.928 | 3.479 | 0.335 | 0.592 |
| Fold 5 (predicts 289 shots, 19 goals; trained on the other 1141 shots, 133 goals) | 0.950 | 0.905 | 3.852 | 0.323 | 0.530 |
| All 1430 shots (152 goals) | 0.699 | 0.912 | 3.542 | 0.312 | 0.600 |
