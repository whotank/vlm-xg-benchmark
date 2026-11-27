"""
Error analysis: which shots do the VLMs get most wrong, and is it the same shots for every model?

"Frontier" = the five zero-shot models with Brier <= 0.111 (Gemini 4 Carbon, Gemini 4 Neon, GPT-6.2 Astra,
Claude Opus 5.5, Gemini 4 Hydrogen). Per shot, on compare_models.py's common shots:
  gap    = frontier mean xG - StatsBomb xG (disagreement with the reference; low noise, but not "error")
  excess = (frontier mean - goal)^2 - (StatsBomb - goal)^2, the shot's share of the Brier gap to StatsBomb
           (sums to n x ΔBrier; noisy per shot because each outcome is 0 or 1)
Analyses:
  1. where the excess Brier accumulates, by shot group (share of the total, mean gap with match-clustered CIs)
  2. which shot features drive the gap: OLS of the gap on standardised features, match-clustered bootstrap CIs
  3. shared vs model-specific error: correlation of the five models' gaps; share of variance that is common
  4. gallery of the largest consensus over- and under-estimates (all five models on the same side), with rationales
  5. rationale phrases and the direction of the gap
Outputs: results/error_analysis.md, results/error_analysis_shots.csv, results/figure_error_gallery.png
Run: python error_analysis.py
"""
import os, re
import numpy as np
import pandas as pd
import compare_models as C
import xg_vlm_pilot as P

FRONTIER = ["Gemini 4 Carbon", "Gemini 4 Neon", "GPT-6.2 Astra", "Claude Opus 5.5", "Gemini 4 Hydrogen"]
# goal angle and keeper-to-shooter distance are left out of the regression: both are nearly functions of distance
# and keeper position, and their collinearity made the coefficients swing in opposite directions
FEATURES = {"distance": "distance (yd)", "header": "header", "defenders_in_cone": "opponents in the shooting cone",
            "close_opponents": "opponents within 2 yd", "gk_dist_goal_line": "keeper off his line (yd)",
            "set_piece": "from a corner or free kick", "first_time": "first-time shot", "one_on_one": "one-on-one",
            "under_pressure": "under pressure"}
PHRASES = {"point-blank": r"point[- ]blank", "tight / narrow angle": r"(tight|narrow|acute) angle",
           "crowded / congested": r"crowd|congest|cluster|packed", "blocked / blocking": r"block",
           "keeper well positioned / set": r"(keeper|goalkeeper)[^.]{0,40}(set|positioned|well placed|on his line)",
           "keeper off line / out of position": r"(off (his|the) line|out of position|rushed out|stranded|committed)",
           "header / heading": r"\bhead", "open / unmarked / clear sight": r"\bunmarked|open goal|clear (sight|view|path)|free shot"}


def load():
    runs = C.load_runs(verbose=False)
    common = C.common_shots(runs, verbose=False)
    base, preds, groups, folders = C.assemble(runs, common)
    shots = P.load_shots_cached(43, 106).set_index("shot_id").loc[common]
    df = base[["match_id", "goal", "sb_xg", "baseline_xg", "body_part", "play_pattern", "distance", "angle",
               "header", "defenders_in_cone"]].copy()
    for col in ["close_opponents", "gk_dist_goal_line", "gk_dist_shooter", "first_time", "one_on_one",
                "under_pressure", "technique"]:
        df[col] = shots[col].values
    df["angle_deg"] = np.degrees(df.angle)
    df["set_piece"] = df.play_pattern.isin(["From Corner", "From Free Kick"]).astype(int)
    for m in FRONTIER:
        df[m] = preds[m]
    reasons = {}
    for lab, fol, _, d in runs:
        if lab in FRONTIER:
            reasons[lab] = d.loc[common].vlm_reason.fillna("").values
    df["frontier_mean"] = df[FRONTIER].mean(axis=1)
    df["frontier_min"], df["frontier_max"] = df[FRONTIER].min(axis=1), df[FRONTIER].max(axis=1)
    df["gap"] = df.frontier_mean - df.sb_xg
    df["excess"] = (df.frontier_mean - df.goal) ** 2 - (df.sb_xg - df.goal) ** 2
    df["n_above_sb"] = (df[FRONTIER].values > df.sb_xg.values[:, None]).sum(axis=1)
    return df, reasons


def group_table(df, idx, col, labels=None):
    """Per group: mean gap (CI), the group's additive contribution to the Brier gap (sum of its excess / all shots;
    contributions sum to the total ΔBrier) and its share of the total, each with match-clustered bootstrap CIs.
    The share is a ratio of two noisy sums, so its interval is wide."""
    rows = []
    n, total = len(df), df.excess.sum()
    ex, gap = df.excess.values, df.gap.values
    for g, sub in df.groupby(col, observed=True):
        mask = (df[col] == g).values
        b_gap, b_con, b_share = [], [], []
        for ix in idx:
            m = mask[ix]
            b_gap.append(gap[ix][m].mean() if m.any() else np.nan)
            b_con.append(ex[ix][m].sum() / len(ix))
            b_share.append(ex[ix][m].sum() / ex[ix].sum())
        rows.append(dict(group=labels.get(g, g) if labels else g, n=len(sub), goals=int(sub.goal.sum()),
                         sb=sub.sb_xg.mean(), vlm=sub.frontier_mean.mean(), gap=sub.gap.mean(), ci_gap=C.ci(b_gap),
                         contrib=sub.excess.sum() / n, ci_contrib=C.ci(b_con), share=sub.excess.sum() / total,
                         ci_share=C.ci(b_share)))
    return pd.DataFrame(rows)


def regression(df, idx):
    X = df[list(FEATURES)].astype(float)
    Xs = (X - X.mean()) / X.std()
    A = np.column_stack([np.ones(len(df)), Xs.values])
    yv = df.gap.values
    beta = np.linalg.lstsq(A, yv, rcond=None)[0]
    boots = np.array([np.linalg.lstsq(A[ix], yv[ix], rcond=None)[0] for ix in idx])
    r2 = 1 - np.sum((yv - A @ beta) ** 2) / np.sum((yv - yv.mean()) ** 2)
    out = [dict(feature=FEATURES[f], coef=beta[i + 1], lo=np.percentile(boots[:, i + 1], 2.5),
                hi=np.percentile(boots[:, i + 1], 97.5)) for i, f in enumerate(FEATURES)]
    return pd.DataFrame(out), r2


def gallery_picks(df, n_each=6):
    over = df[df.n_above_sb == len(FRONTIER)].nlargest(n_each, "gap")
    under = df[df.n_above_sb == 0].nsmallest(n_each, "gap")
    return [("over", s) for s in over.index] + [("under", s) for s in under.index]


def draw_gallery(df, picks, path):
    """Draws the gallery from the rendered shot images; images missing from results/images/ (it is not committed)
    are rendered first with the pipeline's render()."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    os.makedirs(P.IMG_DIR, exist_ok=True)
    missing = [s for _, s in picks if not os.path.exists(os.path.join(P.IMG_DIR, f"{s}.png"))]
    if missing:
        shots = P.load_shots_cached(43, 106).set_index("shot_id")
        for s in missing:
            P.render(shots.loc[s].rename(s), os.path.join(P.IMG_DIR, f"{s}.png"))
    fig, axes = plt.subplots(4, 3, figsize=(13, 12.5))
    for ax, (kind, sid) in zip(axes.ravel(), picks):
        r = df.loc[sid]
        im = Image.open(os.path.join(P.IMG_DIR, f"{sid}.png"))
        ax.imshow(im.crop((0, 0, im.width, int(im.height * 0.55))))   # attacking area only, for legibility
        ax.set_axis_off()
        outcome = "GOAL" if r.goal else "no goal"
        ax.set_title(f"{'Over' if kind == 'over' else 'Under'}-estimate · {outcome}\n"
                     f"StatsBomb {r.sb_xg:.2f} · VLMs {r.frontier_mean:.2f} ({r.frontier_min:.2f}–{r.frontier_max:.2f})\n"
                     f"{r.body_part}, {r.distance:.0f} yd, {r.play_pattern}", fontsize=8.5, loc="left")
    fig.suptitle("Largest consensus disagreements with StatsBomb xG (all five frontier VLMs on the same side).\n"
                 "Top two rows: over-estimates; bottom two rows: under-estimates. Images cropped to the attacking "
                 "area for display; the models saw the full half pitch.", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    df, reasons = load()
    idx = C.cluster_bootstrap(df.match_id.values)
    n, y = len(df), df.goal.values
    d_total = df.excess.mean()
    df["band"] = pd.cut(df.distance, C.BANDS, labels=C.BAND_LABELS)
    df["shot_kind"] = np.where(df.header == 1, "Header", np.where(df.set_piece == 1, "Foot, set piece", "Foot, open play"))
    df["gk_bin"] = pd.cut(df.gk_dist_shooter, [0, 4, 8, 12, 99], labels=["< 4 yd", "4–8 yd", "8–12 yd", "12+ yd"])
    df["cone_bin"] = pd.cut(df.defenders_in_cone, [-1, 0, 1, 2, 99], labels=["0", "1", "2", "3+"])
    df.drop(columns=[]).to_csv(os.path.join(C.RES, "error_analysis_shots.csv"))

    d_ci = C.ci([df.excess.values[ix].mean() for ix in idx])
    kinds = [f"{b}, {k}" for b in C.BAND_LABELS for k in ("header", "foot/other")]
    df["band_kind"] = pd.Categorical([f"{b}, {'header' if h else 'foot/other'}" for b, h in zip(df.band, df.header)],
                                     categories=kinds, ordered=True)
    L = ["# Error analysis: which shots do the VLMs get wrong?", "",
         f"Evaluation set: the {n} common shots ({y.sum()} goals). Frontier = {', '.join(FRONTIER)} (zero-shot, image "
         "only). Gap = frontier mean xG − StatsBomb xG. Excess Brier = (frontier mean − goal)² − (StatsBomb − goal)²; "
         f"its mean is the Brier gap of the averaged frontier prediction to StatsBomb: "
         + f"{d_total:+.4f} [{d_ci[0]:+.4f}, {d_ci[1]:+.4f}]".replace("-", "−")
         + ". A group's contribution is the sum of its shots' excess divided by all "
         f"{n} shots (contributions add up to that total); its share is the contribution divided by the total. The "
         "share is a ratio of two noisy sums, so its interval is wide: read the mean gaps and contributions first. "
         "Brackets: 95% match-clustered bootstrap intervals.", ""]
    for title, col in [("Distance band (yards)", "band"), ("Distance band and body part", "band_kind"),
                       ("Shot kind", "shot_kind"), ("Keeper distance to the shooter", "gk_bin"),
                       ("Opponents in the shooting cone", "cone_bin")]:
        t = group_table(df, idx, col)
        L += [f"## {title}", "", "| Group | n (goals) | StatsBomb | Frontier VLMs | Gap [95% CI] | "
              "Contribution to ΔBrier [95% CI] | Share of ΔBrier [95% CI] |", "|---|---|---|---|---|---|---|"]
        for _, r in t.iterrows():
            L.append(f"| {r.group} | {r.n} ({r.goals}) | {r.sb:.3f} | {r.vlm:.3f} | "
                     f"{C.fmt(r.gap, *r.ci_gap, sign=True)} | {C.fmt(r.contrib, *r.ci_contrib, nd=4, sign=True)} | "
                     f"{r.share:+.0%} [{r.ci_share[0]:+.0%}, {r.ci_share[1]:+.0%}] |".replace("-", "−"))
        L.append("")

    reg, r2 = regression(df, idx)
    L += ["## What drives the gap (OLS on standardised features)", "",
          f"Gap regressed on {len(FEATURES)} shot features, each scaled to one standard deviation; R² = {r2:.2f}. "
          "Positive = the VLMs rate such shots higher than StatsBomb does, holding the other features fixed. "
          "CIs from the match-clustered bootstrap.", "", "| Feature (+1 SD) | Effect on gap [95% CI] |", "|---|---|"]
    for _, r in reg.sort_values("coef", key=abs, ascending=False).iterrows():
        L.append(f"| {r.feature} | {C.fmt(r.coef, r.lo, r.hi, sign=True)} |")

    gaps = df[FRONTIER].sub(df.sb_xg, axis=0)
    corr = gaps.corr()
    shared = np.var(gaps.mean(axis=1)) / np.mean(np.var(gaps, axis=0))
    off = corr.values[np.triu_indices(len(FRONTIER), 1)]
    L += ["", "## Shared or model-specific?", "",
          f"Correlation of the five models' gaps to StatsBomb, shot by shot: median {np.median(off):.2f} "
          f"(range {off.min():.2f}–{off.max():.2f}). The averaged gap carries {shared:.0%} of a single model's gap "
          "variance, so most disagreement with StatsBomb is shared across models rather than idiosyncratic.", "",
          "| | " + " | ".join(m.replace("Gemini 4 ", "G4 ").replace("Claude ", "") for m in FRONTIER) + " |",
          "|---|" + "---|" * len(FRONTIER)]
    for m in FRONTIER:
        L.append(f"| {m} | " + " | ".join(f"{corr.loc[m, k]:.2f}" for k in FRONTIER) + " |")
    agree_over = int((df.n_above_sb == len(FRONTIER)).sum())
    agree_under = int((df.n_above_sb == 0).sum())
    L += ["", f"All five above StatsBomb on {agree_over} shots, all five below on {agree_under}, split on "
          f"{n - agree_over - agree_under}."]

    picks = gallery_picks(df)
    L += ["", "## Largest consensus disagreements (results/figure_error_gallery.png)", "",
          "| | Shot | Outcome | StatsBomb | Frontier mean (range) | Body part, distance, play | Gemini 4 Carbon's reason |",
          "|---|---|---|---|---|---|---|"]
    pos = {s: i for i, s in enumerate(df.index)}
    for kind, sid in picks:
        r = df.loc[sid]
        why = reasons["Gemini 4 Carbon"][pos[sid]].replace("|", "/")
        L.append(f"| {kind} | `{sid[:8]}` | {'goal' if r.goal else 'no goal'} | {r.sb_xg:.2f} | {r.frontier_mean:.2f} "
                 f"({r.frontier_min:.2f}–{r.frontier_max:.2f}) | {r.body_part}, {r.distance:.0f} yd, {r.play_pattern} | {why} |")

    L += ["", "## Rationale phrases and the direction of the gap", "",
          "Share of each model's rationales that use a phrase, and that model's mean gap on those shots "
          "(model xG − StatsBomb).", "", "| Phrase | " + " | ".join(m.replace("Gemini 4 ", "G4 ") for m in FRONTIER) + " |",
          "|---|" + "---|" * len(FRONTIER)]
    for name, pat in PHRASES.items():
        cells = []
        for m in FRONTIER:
            hit = np.array([bool(re.search(pat, t, re.I)) for t in reasons[m]])
            g = (df[m] - df.sb_xg).values
            cells.append(f"{hit.mean():.0%} ({g[hit].mean():+.3f})".replace("-", "−") if hit.any() else "0%")
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    with open(os.path.join(C.RES, "error_analysis.md"), "w") as f:   # tables first: the figure needs the images
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    draw_gallery(df, picks, os.path.join(C.RES, "figure_error_gallery.png"))


if __name__ == "__main__":
    main()
