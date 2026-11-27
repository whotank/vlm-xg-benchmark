"""
Perception vs judgment: the same freeze frame given as an image (zero-shot run) or as (x, y) coordinates in text
(--input coords run). Nothing else differs: same role line, task and JSON lines, no body part, no player roles.

If the close-range errors of the image runs shrink with coordinates, reading the picture was the problem;
if they stay, the problem is the models' judgment (or turning raw coordinates into geometry: the coordinates
still have to be converted into distance, angle and blockers, which the --all-features runs hand over directly).

Reported on compare_models.py's common shots, match-clustered bootstrap (2,000 resamples, seed 0), paired:
  overall AUC / Brier / mean xG, image vs coordinates, with paired differences
  mean predicted xG by distance band vs StatsBomb and the observed rate; the change in the gap to StatsBomb
  (= the change in mean xG, since StatsBomb is the same) and in the band's Brier score
  headers (neither input carries the body part)
Outputs: results/coords_vs_image.md, results/coords_vs_image.csv
Run: python coords_analysis.py
"""
import os
import numpy as np
import pandas as pd
import compare_models as C

PAIRS = [("Claude Opus 5.5", "Claude Opus 5.5, coordinates as text"),
         ("Gemini 4 Carbon", "Gemini 4 Carbon, coordinates as text")]


def main():
    runs = C.load_runs(verbose=False)
    common = C.common_shots(runs, verbose=False)
    base, preds, groups, folders = C.assemble(runs, common)
    pairs = [(a, b) for a, b in PAIRS if a in preds and b in preds]
    if not pairs:
        raise SystemExit("no image/coordinates pairs finished yet")
    y = base.goal.values.astype(int)
    idx = C.cluster_bootstrap(base.match_id.values)
    bands = pd.cut(base.distance, C.BANDS, labels=False).astype(int).values
    head = base.body_part.values == "Head"
    sb = preds["StatsBomb xG"]

    def boot(fn, p, mask=None):
        out = []
        for ix in idx:
            if mask is not None:
                ix = ix[mask[ix]]
            out.append(fn(y[ix], p[ix]) if len(ix) else np.nan)
        return np.array(out)

    fmt, fs = (lambda x, lo, hi: C.fmt(x, lo, hi)), (lambda x, lo, hi: C.fmt(x, lo, hi, sign=True))
    rows = []
    L = ["# Perception vs judgment: image vs coordinates as text", "",
         f"Evaluation set: the {len(y)} common shots of compare_models.py ({y.sum()} goals). Each model sees the same "
         "freeze frame either as the rendered image (zero-shot run) or as (x, y) coordinates in text (`--input coords`: "
         "pitch size, goal and post positions, shooter, goalkeeper, other opponents, teammates; no body part, no roles). "
         "Brackets: 95% match-clustered bootstrap intervals; Δ = coordinates − image, paired; p = two-sided bootstrap "
         "p-value.", "", "## Overall", "",
         "| Model | Input | AUC | Brier | Mean xG | ΔBrier (coords − image) | p | ΔAUC |", "|---|---|---|---|---|---|---|---|"]
    for a, b in pairs:
        ba, bb = boot(C.brier, preds[a]), boot(C.brier, preds[b])
        aa, ab = boot(C.auc, preds[a]), boot(C.auc, preds[b])
        ma, mb = boot(C.mean_xg, preds[a]), boot(C.mean_xg, preds[b])
        d, dauc = bb - ba, ab - aa
        name = a
        L.append(f"| {name} | image | {fmt(C.auc(y, preds[a]), *C.ci(aa))} | {fmt(C.brier(y, preds[a]), *C.ci(ba))} | "
                 f"{fmt(preds[a].mean(), *C.ci(ma))} | | | |")
        L.append(f"| {name} | coordinates | {fmt(C.auc(y, preds[b]), *C.ci(ab))} | {fmt(C.brier(y, preds[b]), *C.ci(bb))} | "
                 f"{fmt(preds[b].mean(), *C.ci(mb))} | {fs(C.brier(y, preds[b]) - C.brier(y, preds[a]), *C.ci(d))} | "
                 f"{C.fmt_p(C.boot_p(d))} | {fs(C.auc(y, preds[b]) - C.auc(y, preds[a]), *C.ci(dauc))} |")
    for ref in ["Map-only baseline", "StatsBomb xG"]:
        if ref in preds:
            L.append(f"| {ref} | exact numbers | {fmt(C.auc(y, preds[ref]), *C.ci(boot(C.auc, preds[ref])))} | "
                     f"{fmt(C.brier(y, preds[ref]), *C.ci(boot(C.brier, preds[ref])))} | {preds[ref].mean():.3f} | | | |")

    L += ["", "## By distance band", "",
          "Gap = mean predicted xG − StatsBomb mean in the band. Change = coordinates − image (paired); negative at "
          "6–12 yd means the over-estimate shrinks.", ""]
    for j, bl in enumerate(C.BAND_LABELS):
        m = bands == j
        L += [f"### {bl} yd (n={m.sum()}, {int(y[m].sum())} goals, observed {y[m].mean():.2f}, StatsBomb {sb[m].mean():.3f})", "",
              "| Model | Image: mean xG (gap) | Coordinates: mean xG (gap) | Change in mean / gap | p | Brier image → coords | ΔBrier | p |",
              "|---|---|---|---|---|---|---|---|"]
        for a, b in pairs:
            mean_a, mean_b = boot(C.mean_xg, preds[a], m), boot(C.mean_xg, preds[b], m)
            bri_a, bri_b = boot(C.brier, preds[a], m), boot(C.brier, preds[b], m)
            dm, db = mean_b - mean_a, bri_b - bri_a
            ga, gb = preds[a][m].mean() - sb[m].mean(), preds[b][m].mean() - sb[m].mean()
            L.append(f"| {a} | {preds[a][m].mean():.3f} ({ga:+.3f}) | {preds[b][m].mean():.3f} ({gb:+.3f}) | "
                     f"{fs(gb - ga, *C.ci(dm))} | {C.fmt_p(C.boot_p(dm))} | {C.brier(y[m], preds[a][m]):.3f} → "
                     f"{C.brier(y[m], preds[b][m]):.3f} | {fs(C.brier(y[m], preds[b][m]) - C.brier(y[m], preds[a][m]), *C.ci(db))} | "
                     f"{C.fmt_p(C.boot_p(db))} |".replace("+-", "−"))
            rows.append(dict(model=a, band=bl, n=int(m.sum()), mean_image=preds[a][m].mean(),
                             mean_coords=preds[b][m].mean(), statsbomb=sb[m].mean(), observed=y[m].mean(),
                             change=gb - ga, change_lo=C.ci(dm)[0], change_hi=C.ci(dm)[1], p_change=C.boot_p(dm),
                             brier_image=C.brier(y[m], preds[a][m]), brier_coords=C.brier(y[m], preds[b][m]),
                             dbrier_lo=C.ci(db)[0], dbrier_hi=C.ci(db)[1], p_dbrier=C.boot_p(db)))
        L.append("")

    L += ["## Headers (neither input has the body part)", "",
          f"n={head.sum()}, {int(y[head].sum())} goals, observed {y[head].mean():.3f}, StatsBomb {sb[head].mean():.3f}.", "",
          "| Model | Image | Coordinates | Change [95% CI] |", "|---|---|---|---|"]
    for a, b in pairs:
        d = boot(C.mean_xg, preds[b], head) - boot(C.mean_xg, preds[a], head)
        L.append(f"| {a} | {preds[a][head].mean():.3f} | {preds[b][head].mean():.3f} | "
                 f"{fs(preds[b][head].mean() - preds[a][head].mean(), *C.ci(d))} |")

    # reading by the rule set in advance: does the 6-12 yd over-estimate shrink?
    L += ["", "## Reading (rule fixed before the runs)", ""]
    for r in [r for r in rows if r["band"] == "6–12"]:
        shrinks = r["change_hi"] < 0
        grows = r["change_lo"] > 0
        verdict = ("shrinks significantly: reading the picture was part of the problem" if shrinks else
                   "grows significantly with coordinates" if grows else
                   "does not change significantly: the close-range error is in the judgment, not in reading the image")
        L.append(f"- {r['model']}: at 6–12 yd the gap to StatsBomb goes from {r['mean_image'] - r['statsbomb']:+.3f} "
                 f"(image) to {r['mean_coords'] - r['statsbomb']:+.3f} (coordinates); change "
                 f"{r['change']:+.3f} [{r['change_lo']:+.3f}, {r['change_hi']:+.3f}], so the over-estimate {verdict}."
                 .replace("+-", "−"))
    L += ["", "Caveat: with coordinates the model still has to turn raw positions into distance, angle and blockers; "
          "the `--all-features` runs (results/features_ablation.md), which state those quantities directly, separate "
          "that geometric step from the judgment itself."]
    with open(os.path.join(C.RES, "coords_vs_image.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    pd.DataFrame(rows).to_csv(os.path.join(C.RES, "coords_vs_image.csv"), index=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()
