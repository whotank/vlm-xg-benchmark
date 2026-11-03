"""
Compare all VLM runs on the shots every model answered.
Prints AUC / Brier / mean xG with paired bootstrap CIs, mean xG by distance band,
and saves Figure 1 (results/figure1.png).
Run after xg_vlm_pilot.py has produced results/<model>/shots.csv for each model.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, brier_score_loss

RUNS = {"Gemma 4 (26B)": "gemma4_26b", "Claude Sonnet 5.5": "claude_sonnet_5_5",
        "Claude Opus 5.5": "claude_opus_5_5", "Gemini 3.8 Flash": "gemini_3.8_flash"}
COLORS = {"Gemma 4 (26B)": "#E69F00", "Claude Sonnet 5.5": "#56B4E9",
          "Claude Opus 5.5": "#0072B2", "Gemini 3.8 Flash": "#009E73", "StatsBomb xG": "black"}


def boot_diff(y, p1, p2, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = [np.mean((p1[i] - y[i]) ** 2) - np.mean((p2[i] - y[i]) ** 2)
         for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return np.mean((p1 - y) ** 2) - np.mean((p2 - y) ** 2), np.percentile(d, [2.5, 97.5])


def main():
    dfs = {k: pd.read_csv(f"results/{v}/shots.csv").set_index("shot_id") for k, v in RUNS.items()}
    common = sorted(set.intersection(*[set(d.index) for d in dfs.values()]))
    base = dfs["Claude Opus 5.5"].loc[common]
    y = base.goal.values
    print(f"Common shots: {len(common)}, goals: {y.sum()}, observed rate: {y.mean():.3f}\n")

    preds = {k: d.loc[common].vlm_xg.values for k, d in dfs.items()}
    preds["StatsBomb xG"] = base.sb_xg.values
    preds["Logistic baseline"] = base.baseline_xg.values

    print(f"{'model':<20}{'AUC':>7}{'Brier':>8}{'MeanXG':>8}   dBrier vs StatsBomb [95% CI]     dBrier vs baseline [95% CI]")
    for k, p in preds.items():
        line = f"{k:<20}{roc_auc_score(y, p):>7.3f}{brier_score_loss(y, p):>8.4f}{p.mean():>8.3f}"
        if k in RUNS:
            d1, c1 = boot_diff(y, p, preds["StatsBomb xG"])
            d2, c2 = boot_diff(y, p, preds["Logistic baseline"])
            line += f"   {d1:+.4f} [{c1[0]:+.4f}, {c1[1]:+.4f}]   {d2:+.4f} [{c2[0]:+.4f}, {c2[1]:+.4f}]"
        print(line)

    bands = pd.cut(base.distance, [0, 6, 12, 18, 25, 200], labels=["<6", "6–12", "12–18", "18–25", "25+"])
    tab = pd.DataFrame({k: pd.Series(p, index=base.index) for k, p in preds.items()})
    tab["n"], tab["goals"] = 1, base.goal
    print("\nMean predicted xG by distance band (yards):")
    print(tab.groupby(bands, observed=True).agg({**{k: "mean" for k in preds}, "n": "sum", "goals": "sum"}).round(3).T)

    # ---- Figure 1
    plot_preds = {k: pd.Series(v, index=base.index) for k, v in preds.items() if k != "Logistic baseline"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6))
    x = np.arange(5)
    obs = base.goal.groupby(bands, observed=True).mean()
    n = base.goal.groupby(bands, observed=True).size()
    a.bar(x, obs.values, color="lightgray", width=0.6, label="Observed goal rate")
    for k, p in plot_preds.items():
        ref = k == "StatsBomb xG"
        a.plot(x, p.groupby(bands, observed=True).mean().values, "o-", lw=2.5 if ref else 2, ms=6,
               color=COLORS[k], ls="--" if ref else "-", label=k)
    a.set_xticks(x, [f"{l}\n(n={c})" for l, c in zip(obs.index, n.values)])
    a.set_xlabel("Shot distance (yards)"); a.set_ylabel("Mean predicted xG")
    a.set_title("(a) Predicted xG by shot distance", loc="left", fontweight="bold")
    a.legend(frameon=False, fontsize=8.5)

    b.plot([0, 0.8], [0, 0.8], color="gray", lw=1, ls=":")
    for k, p in plot_preds.items():
        ref = k == "StatsBomb xG"
        s = pd.DataFrame({"p": p.values, "y": y})
        g = s.groupby(pd.qcut(s.p.rank(method="first"), 5), observed=True).agg(pred=("p", "mean"), obs=("y", "mean"))
        b.plot(g.pred, g.obs, "o-", lw=2.5 if ref else 2, ms=6, color=COLORS[k], ls="--" if ref else "-", label=k)
    b.set_xlim(0, 0.8); b.set_ylim(0, 0.6)
    b.set_xlabel("Mean predicted xG (quintile bins)"); b.set_ylabel("Observed goal rate")
    b.set_title("(b) Calibration (dotted line = perfect)", loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig("results/figure1.png", dpi=300, bbox_inches="tight")
    print("\nSaved results/figure1.png")


if __name__ == "__main__":
    main()
