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

    L += ["", "## By distance band and body part", "",
          "Most 6–12 yd shots are headers, which StatsBomb rates far lower than foot shots from the same distance and "
          "neither input can identify. Gap = mean xG − StatsBomb mean in the cell; change = coordinates − image (paired).", "",
          "| Model | Band (yd) | Body part | n (goals) | StatsBomb | Image: mean (gap) | Coordinates: mean (gap) | Change [95% CI] | p |",
          "|---|---|---|---|---|---|---|---|---|"]
    for a, b in pairs:
        for j, bl in enumerate(C.BAND_LABELS):
            for body, mb in [("header", (bands == j) & head), ("foot/other", (bands == j) & ~head)]:
                if not mb.sum():
                    continue
                dm = boot(C.mean_xg, preds[b], mb) - boot(C.mean_xg, preds[a], mb)
                ma, mc, ms = preds[a][mb].mean(), preds[b][mb].mean(), sb[mb].mean()
                L.append(f"| {a} | {bl} | {body} | {mb.sum()} ({int(y[mb].sum())}) | {ms:.3f} | {ma:.3f} ({ma - ms:+.3f}) | "
                         f"{mc:.3f} ({mc - ms:+.3f}) | {fs(mc - ma, *C.ci(dm))} | {C.fmt_p(C.boot_p(dm))} |"
                         .replace("(-", "(−"))

    # reading by the rule set in advance: does the 6-12 yd over-estimate shrink? Plus a check that the change is
    # specific to close range rather than a shift of the model's whole level (which would not mean better reading):
    # the level shift is measured on the shots outside the 6-12 band, so the band's own change does not inflate it.
    L += ["", "## Reading (rule fixed before the runs, plus a level-shift check)", ""]
    sgn = lambda v: f"{v:+.3f}".replace("-", "−")
    rng = lambda lo, hi: f"[{sgn(lo)}, {sgn(hi)}]"
    for r in [r for r in rows if r["band"] == "6–12"]:
        a, b = [p for p in pairs if p[0] == r["model"]][0]
        close = bands == C.BAND_LABELS.index("6–12")
        d_in = boot(C.mean_xg, preds[b], close) - boot(C.mean_xg, preds[a], close)
        d_out = boot(C.mean_xg, preds[b], ~close) - boot(C.mean_xg, preds[a], ~close)
        out_shift = preds[b][~close].mean() - preds[a][~close].mean()
        spec = d_in - d_out   # close-range change beyond the change elsewhere (paired, same resamples)
        spec_v = r["change"] - out_shift
        d_brier = boot(C.brier, preds[b]) - boot(C.brier, preds[a])
        img_gap = r["mean_image"] - r["statsbomb"]
        shrinks, grows = r["change_hi"] < 0, r["change_lo"] > 0
        line = (f"- {r['model']}: at 6–12 yd the gap to StatsBomb goes from {sgn(img_gap)} (image) to "
                f"{sgn(r['mean_coords'] - r['statsbomb'])} (coordinates); change {sgn(r['change'])} "
                f"{rng(r['change_lo'], r['change_hi'])}.")
        if shrinks or grows:
            line += (f" Outside 6–12 yd the model's mean xG moves by {sgn(out_shift)} {rng(*C.ci(d_out))}, so the "
                     f"close-range change beyond that level shift is {sgn(spec_v)} {rng(*C.ci(spec))}"
                     + (" (specific to close range)." if C.ci(spec)[1] < 0 or C.ci(spec)[0] > 0 else
                        " (not distinguishable from the level shift)."))
            gh = [preds[k][close & head].mean() - sb[close & head].mean() for k in (a, b)]
            gf = [preds[k][close & ~head].mean() - sb[close & ~head].mean() for k in (a, b)]
            line += (f" By body part at 6–12 yd: headers {sgn(gh[0])} → {sgn(gh[1])}, foot shots {sgn(gf[0])} → "
                     f"{sgn(gf[1])} (gaps to StatsBomb, image → coordinates).")
            if (gf[1] < gf[0] < 0) or (gh[1] > gh[0] > 0):
                line += (" The change is not a correction: shots the image run already rated "
                         f"{'below' if gf[0] < 0 else 'above'} StatsBomb move further away, so coordinates shift the "
                         "whole close-range level rather than fixing the over-rated shots.")
            line += (f" Overall Brier changes by {C.brier(y, preds[b]) - C.brier(y, preds[a]):+.4f} "
                     f"[{C.ci(d_brier)[0]:+.4f}, {C.ci(d_brier)[1]:+.4f}] and the band's Brier by "
                     f"{r['brier_coords'] - r['brier_image']:+.4f} [{r['dbrier_lo']:+.4f}, {r['dbrier_hi']:+.4f}]. By "
                     "the rule, how the model used the picture affects its close-range level, but the change does not "
                     "significantly improve accuracy, and the over-estimate itself is on headers, which coordinates "
                     "cannot identify either.")
        else:
            line += (f" Inconclusive: the interval includes no change, but its lower end is a reduction of "
                     f"{-r['change_lo'] / img_gap:.0%} of the image run's over-estimate, so this run can neither confirm "
                     "nor rule out that reading the picture contributed. It is not evidence that the error is judgment.")
        L.append(line.replace("+-", "−").replace("[-", "[−").replace(" -0", " −0"))
    L += ["", "Caveat: with coordinates the model still has to turn raw positions into distance, angle and blockers; "
          "the `--all-features` runs (results/features_ablation.md), which state those quantities directly, separate "
          "that geometric step from the judgment itself."]
    with open(os.path.join(C.RES, "coords_vs_image.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    pd.DataFrame(rows).to_csv(os.path.join(C.RES, "coords_vs_image.csv"), index=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()
