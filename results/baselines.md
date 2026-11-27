# Statistical baselines

Out-of-fold predictions for all 1430 non-penalty shots (152 goals), 5 folds grouped by match (the frozen folds in `results/baseline_folds.json`). Scores below are over all shots; `results/main_table.md` reports them on the 298 common shots with match-clustered CIs.

| Model | Inputs | AUC | Brier | Mean xG |
|---|---|---|---|---|
| Map-only baseline | distance, angle, defenders in the cone (no header) | 0.764 | 0.0848 | 0.106 |
| Logistic baseline (original) | the above + header flag | 0.781 | 0.0817 | 0.106 |
| Logistic, all pre-shot features | 49 columns: geometry incl. keeper position, body part, technique, shot type, play pattern, situation flags, assist | 0.790 | 0.0805 | 0.104 |
| StatsBomb xG | StatsBomb's commercial model | 0.815 | 0.0762 | 0.096 |
