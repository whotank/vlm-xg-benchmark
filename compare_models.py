"""
Compare VLM runs on the shots every included run answered.

Statistics: every reported metric (AUC, Brier, log loss, mean xG, and each distance-band mean) gets a
95% percentile interval from a match-clustered bootstrap: the matches that contain evaluation shots are
resampled with replacement and every shot of a drawn match is kept (2,000 resamples, seed 0). Shots from
the same match share teams, keepers and game states, so resampling matches rather than shots gives
honest intervals. Paired differences (each model vs StatsBomb xG and vs the logistic baseline) are
computed on the same resamples, so they account for the shared shots. A shot-level (i.i.d.) bootstrap of
the Brier differences is kept as a sensitivity check; the abstract reported shot-level intervals.

Outputs (written every run):
  results/main_table.csv / .md          AUC, Brier, log loss, mean xG with CIs, plus paired ΔBrier tests
  results/paired_differences.csv        ΔAUC, ΔBrier, Δlog loss vs StatsBomb and vs baseline (CI, bootstrap p)
  results/distance_bands.csv / .md      mean predicted xG by distance band with CIs, observed rate with CIs
Figures (only with --figures, because rewriting the PNGs changes their bytes even when the data has not):
  results/figure1.png                   Figure 1 of the abstract (the four original models), unchanged design
  results/figure_zero_shot_all.png      same design for every zero-shot run

Run after xg_vlm_pilot.py has produced results/<run>/shots.csv for each run:
  python compare_models.py              # tables only
  python compare_models.py --figures    # also redraw the figures
"""
import os, json, argparse
import numpy as np
import pandas as pd
from scipy.stats import rankdata

RES = "results"
MIN_N = 300            # a run counts only after an invocation over the full 300-shot sample
N_BOOT, BOOT_SEED = 2000, 0
BANDS = [0, 6, 12, 18, 25, 200]
BAND_LABELS = ["<6", "6–12", "12–18", "18–25", "25+"]
REFS = ["StatsBomb xG", "Logistic baseline"]

# ---------------------------------------------------------------- run registry
# (label, folder, group). Folders without shots.csv yet (or never run) are skipped automatically.
BASE = [  # zero-shot, image only
    ("Gemma 4 (26B)", "gemma4_26b"),
    ("Claude Sonnet 5.5", "claude_sonnet_5_5"),
    ("Claude Opus 5.5", "claude_opus_5_5"),
    ("Gemini 3.8 Flash", "gemini_3.8_flash"),
    ("Gemini 4 Carbon", "gemini_4_carbon"),
    ("Gemini 4 Neon", "gemini_4_neon"),
    ("Gemini 4 Hydrogen", "gemini_4_hydrogen"),
    ("GPT-6.2 Astra", "gpt_6.2_astra"),
]
ORIGINAL4 = ["Gemma 4 (26B)", "Claude Sonnet 5.5", "Claude Opus 5.5", "Gemini 3.8 Flash"]


def registry():
    runs = [(lab, fol, "zero-shot") for lab, fol in BASE]
    for lab, fol in BASE:
        runs.append((f"{lab} + body part", f"{fol}_context", "context"))
    for set_ in ["", "_setB"]:
        for lab, fol in BASE:
            for k in (1, 5, 10):
                runs.append((f"{lab}, k={k}{' (set B)' if set_ else ''}", f"{fol}_k{k}{set_}",
                             "few-shot" + (" set B" if set_ else " set A")))
    # same settings as an earlier run, queried again (run-to-run noise / rendering check)
    for lab, fol in BASE:
        runs.append((f"{lab} (rerun)", f"{fol}_rerun", "rerun"))
    return runs


COLORS = {"Gemma 4 (26B)": "#E69F00", "Claude Sonnet 5.5": "#56B4E9", "Claude Opus 5.5": "#0072B2",
          "Gemini 3.8 Flash": "#009E73", "Gemini 4 Carbon": "#CC79A7", "Gemini 4 Neon": "#D55E00",
          "Gemini 4 Hydrogen": "#F0E442", "GPT-6.2 Astra": "#999999", "StatsBomb xG": "black"}


# ---------------------------------------------------------------- metrics
def auc(y, p):
    """ROC AUC via the Mann-Whitney rank formula (ties count 1/2; identical to sklearn's roc_auc_score)."""
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(p)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def brier(y, p):
    return np.mean((p - y) ** 2)


def logloss(y, p):
    p = np.clip(p, 1e-3, 1 - 1e-3)   # same clipping as xg_vlm_pilot.scores()
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def mean_xg(y, p):
    return np.mean(p)


METRICS = {"auc": auc, "brier": brier, "logloss": logloss, "mean_xg": mean_xg}


def cluster_bootstrap(groups, n_boot=N_BOOT, seed=BOOT_SEED):
    """Index arrays for a cluster bootstrap: draw as many clusters as there are, with replacement."""
    codes, uniq = pd.factorize(np.asarray(groups))
    members = [np.flatnonzero(codes == g) for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    return [np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))]) for _ in range(n_boot)]


def iid_bootstrap(n, n_boot=N_BOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n, n) for _ in range(n_boot)]


def ci(a):
    a = np.asarray(a, float)
    return tuple(np.nanpercentile(a, [2.5, 97.5])) if np.isfinite(a).any() else (np.nan, np.nan)


def boot_p(d):
    """Two-sided bootstrap p-value for 'difference = 0' (share of resamples on the other side, doubled)."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    return float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0))))


def fmt(x, lo, hi, nd=3, sign=False):
    f = (lambda v: f"{v:+.{nd}f}") if sign else (lambda v: f"{v:.{nd}f}")
    return f"{f(x)} [{f(lo)}, {f(hi)}]".replace("-", "−")


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# ---------------------------------------------------------------- loading
def finished(fol):
    """A run counts once its latest invocation covered the full sample and ended complete or partial
    (a few shots unanswered). Runs in progress, stopped, or tested on a handful of shots are skipped.
    Folders from before run_config.json existed (the four original runs) always count."""
    path = os.path.join(RES, fol, "run_config.json")
    if not os.path.exists(path):
        return True
    with open(path) as f:
        last = json.load(f)["invocations"][-1]
    return last.get("n", 0) >= MIN_N and str(last.get("status", "")).startswith(("complete", "partial"))


def load_runs(verbose=True):
    runs, missing = [], []
    for lab, fol, grp in registry():
        path = os.path.join(RES, fol, "shots.csv")
        if os.path.exists(path) and finished(fol):
            df = pd.read_csv(path).set_index("shot_id")
            runs.append((lab, fol, grp, df))
        elif os.path.isdir(os.path.join(RES, fol)):
            missing.append(fol)
    if verbose and missing:
        print(f"Skipped (not finished on the full sample yet): {', '.join(missing)}")
    return runs


def common_shots(runs, verbose=True):
    sets = {lab: set(df.dropna(subset=["vlm_xg"]).index) for lab, _, _, df in runs}
    common = sorted(set.intersection(*sets.values()))
    if verbose:
        orig = [s for lab, s in sets.items() if lab in ORIGINAL4]
        ref = set.intersection(*orig) if orig else set(common)
        short = {lab: len(ref - s) for lab, s in sets.items() if ref - s}
        if short:
            print("WARNING: these runs miss some of the original common shots, so every table uses fewer shots:",
                  ", ".join(f"{k} (-{v})" for k, v in short.items()))
    return common


def assemble(runs, common):
    # shot-level columns (goal, sb_xg, baseline_xg, distance, match_id) are identical in every shots.csv
    base = runs[0][3].loc[common]
    preds = {lab: df.loc[common].vlm_xg.values.astype(float) for lab, _, _, df in runs}
    preds["StatsBomb xG"] = base.sb_xg.values.astype(float)
    preds["Logistic baseline"] = base.baseline_xg.values.astype(float)
    groups = {lab: grp for lab, _, grp, _ in runs}
    groups.update({r: "reference" for r in REFS})
    folders = {lab: fol for lab, fol, _, _ in runs}
    return base, preds, groups, folders


# ---------------------------------------------------------------- analysis
def bootstrap_all(y, preds, match_ids, bands):
    idxs = cluster_bootstrap(match_ids)
    iid = iid_bootstrap(len(y))
    nb = len(BAND_LABELS)
    out = {m: {k: np.empty(len(idxs)) for k in METRICS} for m in preds}
    band_means = {m: np.full((len(idxs), nb), np.nan) for m in preds}
    obs_rate, obs_band = np.empty(len(idxs)), np.full((len(idxs), nb), np.nan)
    for b, ix in enumerate(idxs):
        yb, bc = y[ix], bands[ix]
        cnt = np.bincount(bc, minlength=nb)
        with np.errstate(invalid="ignore", divide="ignore"):
            obs_rate[b] = yb.mean()
            obs_band[b] = np.bincount(bc, weights=yb, minlength=nb) / np.where(cnt > 0, cnt, np.nan)
            for m, p in preds.items():
                pb = p[ix]
                for k, f in METRICS.items():
                    out[m][k][b] = f(yb, pb)
                band_means[m][b] = np.bincount(bc, weights=pb, minlength=nb) / np.where(cnt > 0, cnt, np.nan)
    iid_brier = {m: np.array([brier(y[ix], p[ix]) for ix in iid]) for m, p in preds.items()}
    return dict(boot=out, band=band_means, obs=obs_rate, obs_band=obs_band, iid_brier=iid_brier, n_boot=len(idxs))


def tables(y, preds, groups, folders, B, bands):
    rows, diffs = [], []
    for m, p in preds.items():
        r = dict(group=groups[m], model=m, folder=folders.get(m, ""), n=len(y), goals=int(y.sum()))
        for k, f in METRICS.items():
            r[k] = f(y, p)
            r[f"{k}_lo"], r[f"{k}_hi"] = ci(B["boot"][m][k])
        for ref, tag in [("StatsBomb xG", "sb"), ("Logistic baseline", "base")]:
            if m == ref:
                continue
            for k in ["auc", "brier", "logloss"]:
                d = B["boot"][m][k] - B["boot"][ref][k]
                lo, hi = ci(d)
                diffs.append(dict(model=m, group=groups[m], vs=ref, metric=k,
                                  diff=METRICS[k](y, p) - METRICS[k](y, preds[ref]), lo=lo, hi=hi, p_boot=boot_p(d)))
                if k == "brier":
                    r[f"dbrier_vs_{tag}"] = diffs[-1]["diff"]
                    r[f"dbrier_vs_{tag}_lo"], r[f"dbrier_vs_{tag}_hi"] = lo, hi
                    r[f"dbrier_vs_{tag}_p"] = diffs[-1]["p_boot"]
                    r[f"dbrier_vs_{tag}_iid_lo"], r[f"dbrier_vs_{tag}_iid_hi"] = ci(B["iid_brier"][m] - B["iid_brier"][ref])
        rows.append(r)
    main = pd.DataFrame(rows)
    order = {g: i for i, g in enumerate(["zero-shot", "few-shot set A", "few-shot set B", "context", "rerun", "reference"])}
    main = main.sort_values("group", key=lambda s: s.map(order), kind="stable").reset_index(drop=True)

    band_rows = []
    nb = len(BAND_LABELS)
    cnt = np.bincount(bands, minlength=nb)
    goals = np.bincount(bands, weights=y, minlength=nb).astype(int)
    for j, lab in enumerate(BAND_LABELS):
        lo, hi = ci(B["obs_band"][:, j])
        band_rows.append(dict(model="Observed goal rate", group="observed", band=lab, n=cnt[j], goals=goals[j],
                              mean=goals[j] / cnt[j], lo=lo, hi=hi,
                              empty_resamples=int(np.isnan(B["obs_band"][:, j]).sum())))
        for m, p in preds.items():
            lo, hi = ci(B["band"][m][:, j])
            band_rows.append(dict(model=m, group=groups[m], band=lab, n=cnt[j], goals=goals[j],
                                  mean=p[bands == j].mean(), lo=lo, hi=hi,
                                  empty_resamples=int(np.isnan(B["band"][m][:, j]).sum())))
    return main, pd.DataFrame(diffs), pd.DataFrame(band_rows)


def write_markdown(main, bands_df, y, B, n_matches, path_main, path_bands):
    lo, hi = ci(B["obs"])
    head = (f"Evaluation set: {len(y)} shots every included run answered ({int(y.sum())} goals, "
            f"{n_matches} matches; observed goal rate {fmt(y.mean(), lo, hi)}).\n"
            f"Brackets: 95% percentile intervals from a match-clustered bootstrap "
            f"({B['n_boot']:,} resamples of matches, seed {BOOT_SEED}). ΔBrier = Brier(model) − Brier(reference); "
            f"negative favours the model. p = two-sided bootstrap p-value for ΔBrier = 0.\n")
    L = ["# Main results table", "", head]
    for grp, g in main.groupby("group", sort=False):
        L += [f"## {grp}", "",
              "| Model | AUC | Brier | Log loss | Mean xG | ΔBrier vs StatsBomb | p | ΔBrier vs baseline | p |",
              "|---|---|---|---|---|---|---|---|---|"]
        for _, r in g.iterrows():
            def d(tag):
                if pd.isna(r.get(f"dbrier_vs_{tag}")):
                    return "", ""
                return (fmt(r[f"dbrier_vs_{tag}"], r[f"dbrier_vs_{tag}_lo"], r[f"dbrier_vs_{tag}_hi"], sign=True),
                        fmt_p(r[f"dbrier_vs_{tag}_p"]))
            (dsb, psb), (dba, pba) = d("sb"), d("base")
            L.append(f"| {r.model} | {fmt(r.auc, r.auc_lo, r.auc_hi)} | {fmt(r.brier, r.brier_lo, r.brier_hi)} | "
                     f"{fmt(r.logloss, r.logloss_lo, r.logloss_hi)} | {fmt(r.mean_xg, r.mean_xg_lo, r.mean_xg_hi)} | "
                     f"{dsb} | {psb} | {dba} | {pba} |")
        L.append("")
    L += ["## Sensitivity: shot-level vs match-clustered bootstrap for ΔBrier", "",
          "The abstract used a shot-level bootstrap. Match clustering usually widens the intervals a little.", "",
          "| Model | ΔBrier vs baseline | shot-level 95% CI | match-clustered 95% CI | ΔBrier vs StatsBomb | shot-level 95% CI | match-clustered 95% CI |",
          "|---|---|---|---|---|---|---|"]
    for _, r in main[main.group != "reference"].iterrows():
        f1 = lambda a: f"{a:+.4f}".replace("-", "−")
        f2 = lambda a, b: f"[{a:+.3f}, {b:+.3f}]".replace("-", "−")
        L.append(f"| {r.model} | {f1(r.dbrier_vs_base)} | {f2(r.dbrier_vs_base_iid_lo, r.dbrier_vs_base_iid_hi)} | "
                 f"{f2(r.dbrier_vs_base_lo, r.dbrier_vs_base_hi)} | {f1(r.dbrier_vs_sb)} | "
                 f"{f2(r.dbrier_vs_sb_iid_lo, r.dbrier_vs_sb_iid_hi)} | {f2(r.dbrier_vs_sb_lo, r.dbrier_vs_sb_hi)} |")
    with open(path_main, "w") as f:
        f.write("\n".join(L) + "\n")

    # distance bands: one row per model, one column per band
    nrow = bands_df[bands_df.model == "Observed goal rate"].set_index("band")
    cols = [f"{b} yd (n={nrow.loc[b, 'n']}, {nrow.loc[b, 'goals']} goals)" for b in BAND_LABELS]
    L = ["# Mean predicted xG by distance band", "", head.split("\n")[0],
         "Brackets: 95% match-clustered bootstrap intervals for the band mean. The <6 yd band has only "
         f"{nrow.loc['<6', 'n']} shots; treat it as suggestive.", "",
         "| Model | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    order = ["Observed goal rate"] + list(main.model)
    for m in order:
        g = bands_df[bands_df.model == m].set_index("band")
        L.append(f"| {m} | " + " | ".join(fmt(g.loc[b, 'mean'], g.loc[b, 'lo'], g.loc[b, 'hi'], nd=2)
                                           for b in BAND_LABELS) + " |")
    with open(path_bands, "w") as f:
        f.write("\n".join(L) + "\n")


# ---------------------------------------------------------------- figures
def figure(base, preds, labels, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    y = base.goal.values
    bands = pd.cut(base.distance, BANDS, labels=BAND_LABELS)
    plot_preds = {k: pd.Series(preds[k], index=base.index) for k in labels + ["StatsBomb xG"]}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6))
    x = np.arange(5)
    obs = base.goal.groupby(bands, observed=True).mean()
    n = base.goal.groupby(bands, observed=True).size()
    a.bar(x, obs.values, color="lightgray", width=0.6, label="Observed goal rate")
    for k, p in plot_preds.items():
        ref = k == "StatsBomb xG"
        a.plot(x, p.groupby(bands, observed=True).mean().values, "o-", lw=2.5 if ref else 2, ms=6,
               color=COLORS.get(k), ls="--" if ref else "-", label=k)
    a.set_xticks(x, [f"{l}\n(n={c})" for l, c in zip(obs.index, n.values)])
    a.set_xlabel("Shot distance (yards)"); a.set_ylabel("Mean predicted xG")
    a.set_title("(a) Predicted xG by shot distance", loc="left", fontweight="bold")
    a.legend(frameon=False, fontsize=8.5)

    b.plot([0, 0.8], [0, 0.8], color="gray", lw=1, ls=":")
    for k, p in plot_preds.items():
        ref = k == "StatsBomb xG"
        s = pd.DataFrame({"p": p.values, "y": y})
        g = s.groupby(pd.qcut(s.p.rank(method="first"), 5), observed=True).agg(pred=("p", "mean"), obs=("y", "mean"))
        b.plot(g.pred, g.obs, "o-", lw=2.5 if ref else 2, ms=6, color=COLORS.get(k), ls="--" if ref else "-", label=k)
    b.set_xlim(0, 0.8); b.set_ylim(0, 0.6)
    b.set_xlabel("Mean predicted xG (quintile bins)"); b.set_ylabel("Observed goal rate")
    b.set_title("(b) Calibration (dotted line = perfect)", loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true", help="also redraw results/figure1.png and figure_zero_shot_all.png")
    a = ap.parse_args()

    runs = load_runs()
    common = common_shots(runs)
    base, preds, groups, folders = assemble(runs, common)
    y = base.goal.values.astype(int)
    bands = pd.cut(base.distance, BANDS, labels=False).astype(int).values
    n_matches = base.match_id.nunique()
    print(f"Runs: {len(runs)}. Common shots: {len(common)}, goals: {y.sum()}, matches: {n_matches}, "
          f"observed rate: {y.mean():.3f}\n")

    B = bootstrap_all(y, preds, base.match_id.values, bands)
    main_t, diffs, bands_t = tables(y, preds, groups, folders, B, bands)
    main_t.to_csv(os.path.join(RES, "main_table.csv"), index=False)
    diffs.to_csv(os.path.join(RES, "paired_differences.csv"), index=False)
    bands_t.to_csv(os.path.join(RES, "distance_bands.csv"), index=False)
    write_markdown(main_t, bands_t, y, B, n_matches,
                   os.path.join(RES, "main_table.md"), os.path.join(RES, "distance_bands.md"))

    print(f"{'model':<34}{'AUC':>22}{'Brier':>25}{'MeanXG':>22}   ΔBrier vs StatsBomb [CI]     ΔBrier vs baseline [CI]")
    for _, r in main_t.iterrows():
        line = (f"{r.model:<34}{fmt(r.auc, r.auc_lo, r.auc_hi):>22}{fmt(r.brier, r.brier_lo, r.brier_hi, 4):>25}"
                f"{fmt(r.mean_xg, r.mean_xg_lo, r.mean_xg_hi):>22}")
        if r.group != "reference":
            line += (f"   {fmt(r.dbrier_vs_sb, r.dbrier_vs_sb_lo, r.dbrier_vs_sb_hi, 4, True)}"
                     f"   {fmt(r.dbrier_vs_base, r.dbrier_vs_base_lo, r.dbrier_vs_base_hi, 4, True)}")
        print(line)
    print("\nMean predicted xG by distance band (yards), match-clustered 95% CI:")
    piv = bands_t.assign(cell=[fmt(m, l, h, 2) for m, l, h in zip(bands_t["mean"], bands_t.lo, bands_t.hi)])
    print(piv.pivot(index="model", columns="band", values="cell")[BAND_LABELS]
          .reindex(["Observed goal rate"] + list(main_t.model)).to_string())
    print("\nSaved results/main_table.csv, main_table.md, paired_differences.csv, distance_bands.csv, distance_bands.md")

    if a.figures:
        figure(base, preds, [m for m in ORIGINAL4 if m in preds], os.path.join(RES, "figure1.png"))
        zs = [m for m, g in groups.items() if g == "zero-shot"]
        figure(base, preds, zs, os.path.join(RES, "figure_zero_shot_all.png"))


if __name__ == "__main__":
    main()
