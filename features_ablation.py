"""
Information ablation: how much of the gap between VLMs and StatsBomb is information the image does not contain,
and does a VLM judge better than a statistical model given the same information?

Rows add information; columns compare a logistic model (exact numbers, frozen match folds; baselines.py) with
VLMs (zero-shot prompt; the extra information is added as text, the image is always shown):
  image geometry only            logistic on distance, angle, defenders in the cone  vs  VLM image only
  + body part                    original 4-feature baseline (header flag)          vs  VLM + "Body part used"
  + all pre-shot attributes      logistic on every pre-shot attribute               vs  VLM + all features
StatsBomb xG is the reference throughout. Everything is scored on compare_models.py's common shots with the
same match-clustered bootstrap (2,000 resamples, seed 0); differences are paired.

Note: in the first row the logistic model gets exact coordinates while the VLM must read them from pixels, so
that comparison still favours the statistical model. With --all-features the VLM also gets the exact geometry
as text, so the last row compares judgment with (nearly) the same inputs.

Outputs: results/features_ablation.md, results/features_ablation.csv
Run: python features_ablation.py   (after baselines.py and the *_allfeat runs)
"""
import os
import numpy as np
import pandas as pd
import compare_models as C

ROWS = [("image geometry only", "Map-only baseline", "", "image only"),
        ("+ body part", "Logistic baseline", " + body part", "body part given"),
        ("+ all pre-shot attributes", "Logistic, all pre-shot features", " + all pre-shot features",
         "all pre-shot attributes + exact geometry")]
VLMS = ["Claude Opus 5.5", "Gemini 4 Carbon", "GPT-6.2 Astra"]

INPUTS = """| Model | Image pixels | Exact coordinates | Body part | Other pre-shot attributes | Trained on |
|---|---|---|---|---|---|
| StatsBomb xG | no | yes (event location, freeze frame) | yes | yes (technique, shot type, assist, pressure, ...) | StatsBomb's own large shot database |
| Logistic, all pre-shot features | no | yes | yes | yes (the same list the VLMs get with `--all-features`) | ~1,140 WC 2022 shots per fold |
| Logistic baseline (original) | no | yes | header flag | no | ~1,140 WC 2022 shots per fold |
| Map-only baseline | no | yes (distance, angle, defenders in the cone) | no | no | ~1,140 WC 2022 shots per fold |
| VLM, image only (main benchmark) | yes | no, read from the image | no | no | no football-specific training; zero-shot prompt |
| VLM + body part | yes | no | yes (text) | no | as above |
| VLM + all pre-shot features | yes | yes (text: distance, angle, defenders, keeper, pressure) | yes (text) | yes (text) | as above |"""


def main():
    runs = C.load_runs(verbose=False)
    common = C.common_shots(runs, verbose=False)
    base, preds, groups, folders = C.assemble(runs, common)
    y = base.goal.values.astype(int)
    idx = C.cluster_bootstrap(base.match_id.values)
    boot = {m: {k: np.array([C.METRICS[k](y[i], preds[m][i]) for i in idx]) for k in ("auc", "brier")}
            for m in preds}

    def cell(m):
        if m not in preds:
            return None
        return dict(model=m, auc=C.auc(y, preds[m]), brier=C.brier(y, preds[m]), mean_xg=preds[m].mean(),
                    auc_ci=C.ci(boot[m]["auc"]), brier_ci=C.ci(boot[m]["brier"]))

    def diff(a, b, k="brier"):
        if a not in preds or b not in preds:
            return None
        d = boot[a][k] - boot[b][k]
        return C.METRICS[k](y, preds[a]) - C.METRICS[k](y, preds[b]), C.ci(d), C.boot_p(d)

    fmt_cell = lambda c: "not run" if c is None else (f"{C.fmt(c['brier'], *c['brier_ci'])}<br>AUC "
                                                       f"{C.fmt(c['auc'], *c['auc_ci'])}")
    fmt_d = lambda d: "" if d is None else f"{C.fmt(d[0], *d[1], sign=True)} (p {C.fmt_p(d[2])})"
    rows_csv = []
    L = ["# Information ablation: what the image does not show", "",
         f"Evaluation set: the {len(y)} common shots of compare_models.py ({y.sum()} goals). Brackets: 95% "
         "match-clustered bootstrap intervals; differences are paired on the same resamples (p = two-sided "
         "bootstrap p-value). The main benchmark is the image-only VLM; the other rows add information as text.", "",
         "## Who sees what", "", INPUTS, "",
         "## Brier score (and AUC) by information and model", "",
         "| Information | Logistic (exact numbers) | " + " | ".join(VLMS) + " |",
         "|---|---|" + "---|" * len(VLMS)]
    for name, logit, suffix, _ in ROWS:
        cells = [fmt_cell(cell(logit))] + [fmt_cell(cell(v + suffix if suffix else v)) for v in VLMS]
        L.append(f"| {name} | " + " | ".join(cells) + " |")
        for col, m in [("Logistic", logit)] + [(v, v + suffix if suffix else v) for v in VLMS]:
            c = cell(m)
            if c:
                rows_csv.append(dict(information=name, column=col, model=m, auc=c["auc"], auc_lo=c["auc_ci"][0],
                                     auc_hi=c["auc_ci"][1], brier=c["brier"], brier_lo=c["brier_ci"][0],
                                     brier_hi=c["brier_ci"][1], mean_xg=c["mean_xg"]))
    sb = cell("StatsBomb xG")
    L += [f"| StatsBomb xG (reference) | {fmt_cell(sb)} |" + " |" * len(VLMS), ""]

    L += ["## Does the VLM judge better than a logistic model with the same information?", "",
          "ΔBrier = VLM − logistic in the same row (negative = the VLM is better).", "",
          "| Information | " + " | ".join(VLMS) + " |", "|---|" + "---|" * len(VLMS)]
    for name, logit, suffix, _ in ROWS:
        L.append(f"| {name} | " + " | ".join(fmt_d(diff(v + suffix if suffix else v, logit)) for v in VLMS) + " |")
    L += ["", "## How much does the extra information help each model?", "",
          "ΔBrier relative to the same model with the image (or image features) only; negative = better.", "",
          "| Added information | Logistic | " + " | ".join(VLMS) + " |", "|---|---|" + "---|" * len(VLMS)]
    for name, logit, suffix, _ in ROWS[1:]:
        L.append(f"| {name} | {fmt_d(diff(logit, ROWS[0][1]))} | " +
                 " | ".join(fmt_d(diff(v + suffix, v)) for v in VLMS) + " |")
    L += ["", "## Distance to StatsBomb", "", "ΔBrier = model − StatsBomb xG (negative = better than StatsBomb).", "",
          "| Information | Logistic | " + " | ".join(VLMS) + " |", "|---|---|" + "---|" * len(VLMS)]
    for name, logit, suffix, _ in ROWS:
        L.append(f"| {name} | {fmt_d(diff(logit, 'StatsBomb xG'))} | " +
                 " | ".join(fmt_d(diff(v + suffix if suffix else v, "StatsBomb xG")) for v in VLMS) + " |")

    # where the information matters: headers and close range
    bands = pd.cut(base.distance, C.BANDS, labels=C.BAND_LABELS).astype(str).values
    head = base.body_part.values == "Head"
    groups_ = [("headers", head), ("6–12 yd", bands == "6–12"), ("all shots", np.ones(len(y), bool))]
    L += ["", "## Mean predicted xG where the information matters", "",
          "| Model | " + " | ".join(f"{g} (n={m.sum()}, observed {y[m].mean():.2f})" for g, m in groups_) + " |",
          "|---|" + "---|" * len(groups_)]
    order = ["StatsBomb xG"] + [r[1] for r in ROWS] + [v + (s if s else "") for _, _, s, _ in ROWS for v in VLMS]
    for m in order:
        if m in preds:
            L.append(f"| {m} | " + " | ".join(f"{preds[m][mask].mean():.3f}" for _, mask in groups_) + " |")
    L += ["", "Hybrid design (VLM reads the geometry from the image, the logistic model scores it) is the missing cell "
          "that separates perception from judgment; it is the next step in CLAUDE.md."]
    with open(os.path.join(C.RES, "features_ablation.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    pd.DataFrame(rows_csv).to_csv(os.path.join(C.RES, "features_ablation.csv"), index=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()
