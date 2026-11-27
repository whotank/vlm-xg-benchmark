"""
Hybrid design: perception vs judgment, using the models' own readings of the image.

Each model reads the distance, the angle and the number of opponents in the shooting triangle off the zero-shot image
(probe_runs.py --task perception; no xG asked). The map-only logistic model (distance, angle, defenders in the cone;
frozen match folds, as in baselines.py) then scores those readings: each shot is scored by the fold model that never
saw its match, exactly as the map-only baseline scores the true values.

  Map-only baseline   true geometry               -> logistic
  Hybrid              geometry read by the model  -> logistic   hybrid - map-only = cost of the misreadings
  Direct              image -> the model's own xG (zero-shot)  direct - hybrid   = the model's judgment, given what
                                                                                   it can read
Partial hybrids swap in one reading at a time (which misreading matters).

Evaluated on compare_models.py's common shots that the perception run answered (all 298 when it covered the full
sample); match-clustered bootstrap (2,000 resamples, seed 0), paired; p = two-sided bootstrap p-value.
Outputs: results/hybrid.md, results/hybrid_shots.csv
Run: python hybrid.py
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
import compare_models as C
import xg_vlm_pilot as P
import baselines as B

MODELS = [("Claude Opus 5.5", "claude_opus_5_5"), ("Gemini 4 Carbon", "gemini_4_carbon"),
          ("GPT-6.2 Astra", "gpt_6.2_astra")]
FEATS = ["distance", "angle", "defenders_in_cone"]


def fold_models(shots):
    """The five map-only fold models of baselines.py (cross_val_predict fits the same models on the same folds)."""
    y = shots.goal.values
    models, fold_of = [], {}
    for k, (tr, te) in enumerate(B.folds(shots)):
        models.append(LogisticRegression(max_iter=1000).fit(shots[FEATS].iloc[tr], y[tr]))
        fold_of.update({s: k for s in shots.shot_id.iloc[te]})
    return models, fold_of


def score(models, fold_of, ids, dist, angle_rad, count):
    X = pd.DataFrame({"distance": np.asarray(dist, float), "angle": np.asarray(angle_rad, float),
                      "defenders_in_cone": np.asarray(count, float)})
    k = np.array([fold_of[s] for s in ids])
    out = np.full(len(ids), np.nan)
    for j, m in enumerate(models):
        if (k == j).any():
            out[k == j] = m.predict_proba(X[k == j])[:, 1]
    return out


def main():
    shots = P.add_baseline(P.load_shots_cached(43, 106))
    models, fold_of = fold_models(shots)
    runs = C.load_runs(verbose=False)
    common = C.common_shots(runs, verbose=False)
    base, preds, _, _ = C.assemble(runs, common)
    truth = shots.set_index("shot_id")
    t = truth.loc[common]
    check = score(models, fold_of, common, t.distance, t.angle, t.defenders_in_cone)
    assert np.allclose(check, preds["Map-only baseline"], atol=1e-9), "fold models do not reproduce the map-only baseline"

    fs, f3 = (lambda x, lo, hi: C.fmt(x, lo, hi, sign=True)), (lambda x, lo, hi: C.fmt(x, lo, hi))
    acc_rows, xg_rows, band_rows, shot_tabs, summary = [], [], [], [], {}
    for lab, fol in MODELS:
        path = os.path.join(C.RES, f"{fol}_perception", "answers.csv")
        if not os.path.exists(path) or lab not in preds:
            continue
        a = pd.read_csv(path).dropna(subset=["read_distance", "read_angle_deg", "read_opponents"]).set_index("shot_id")
        mask = np.array([s in a.index for s in common])
        ids = [s for s in common if s in a.index]
        if len(ids) < 30:   # a run still in progress (or a quick test)
            continue
        r, tt = a.loc[ids], truth.loc[ids]
        y = base.goal.values[mask].astype(int)
        idx = C.cluster_bootstrap(base.match_id.values[mask])
        rd, ra, rn = r.read_distance.values, np.radians(r.read_angle_deg.values), r.read_opponents.values
        td, ta, tn = tt.distance.values, tt.angle.values, tt.defenders_in_cone.values
        p = {"Map-only baseline (true geometry)": preds["Map-only baseline"][mask],
             f"Hybrid: all three read by the model": score(models, fold_of, ids, rd, ra, rn),
             "Hybrid: distance read, rest true": score(models, fold_of, ids, rd, ta, tn),
             "Hybrid: angle read, rest true": score(models, fold_of, ids, td, ra, tn),
             "Hybrid: opponents read, rest true": score(models, fold_of, ids, td, ta, rn),
             "Direct: the model's own xG (zero-shot, image)": preds[lab][mask],
             "StatsBomb xG": preds["StatsBomb xG"][mask]}
        keys = list(p)
        mapk, hyb, direct = keys[0], keys[1], keys[5]
        bb = {k: np.array([C.brier(y[ix], v[ix]) for ix in idx]) for k, v in p.items()}
        ba = {k: np.array([C.auc(y[ix], v[ix]) for ix in idx]) for k, v in p.items()}
        for k, v in p.items():
            d = bb[k] - bb[mapk]
            xg_rows.append(f"| {lab} | {k} | {f3(C.auc(y, v), *C.ci(ba[k]))} | {f3(C.brier(y, v), *C.ci(bb[k]))} | "
                           f"{v.mean():.3f} | " + ("" if k == mapk else
                           f"{fs(C.brier(y, v) - C.brier(y, p[mapk]), *C.ci(d))} | {C.fmt_p(C.boot_p(d))}") +
                           (" | |" if k == mapk else " |"))
        # decomposition of the direct model's gap to the map-only baseline
        perc, judg = bb[hyb] - bb[mapk], bb[direct] - bb[hyb]
        tot = bb[direct] - bb[mapk]
        val = lambda k1, k2: C.brier(y, p[k1]) - C.brier(y, p[k2])
        summary[lab] = dict(n=len(ids), perc=(val(hyb, mapk), *C.ci(perc), C.boot_p(perc)),
                            judg=(val(direct, hyb), *C.ci(judg), C.boot_p(judg)),
                            tot=(val(direct, mapk), *C.ci(tot), C.boot_p(tot)),
                            auc_h=C.auc(y, p[hyb]), auc_m=C.auc(y, p[mapk]), auc_d=C.auc(y, p[direct]))
        # perception accuracy
        de, ae, ne = rd - td, np.degrees(ra - ta), rn - tn
        bands = pd.cut(td, C.BANDS, labels=C.BAND_LABELS)
        acc_rows.append(f"| {lab} | all | {len(ids)} | {de.mean():+.2f} | {np.abs(de).mean():.2f} | "
                        f"{np.median(np.abs(de)):.2f} | {np.corrcoef(rd, td)[0, 1]:.3f} | {ae.mean():+.1f} | "
                        f"{np.abs(ae).mean():.1f} | {np.mean(ne == 0):.0%} | {np.mean(np.abs(ne) <= 1):.0%} | {ne.mean():+.2f} |")
        for bl in C.BAND_LABELS:
            m = np.asarray(bands == bl)
            if m.sum():
                acc_rows.append(f"| {lab} | {bl} yd | {m.sum()} | {de[m].mean():+.2f} | {np.abs(de[m]).mean():.2f} | "
                                f"{np.median(np.abs(de[m])):.2f} | | {ae[m].mean():+.1f} | {np.abs(ae[m]).mean():.1f} | "
                                f"{np.mean(ne[m] == 0):.0%} | {np.mean(np.abs(ne[m]) <= 1):.0%} | {ne[m].mean():+.2f} |")
                band_rows.append(f"| {lab} | {bl} | {m.sum()} | {y[m].mean():.2f} | {p['StatsBomb xG'][m].mean():.3f} | "
                                 f"{p[mapk][m].mean():.3f} | {p[hyb][m].mean():.3f} | {p[direct][m].mean():.3f} |")
        gap = p[direct] - p[mapk]
        rho = spearmanr(de, gap).correlation
        close = np.asarray(bands == "6–12")
        rho_c = spearmanr(de[close], gap[close]).correlation if close.sum() > 5 else np.nan
        summary[lab].update(rho=rho, rho_close=rho_c, mae_d=np.abs(de).mean(), mae_a=np.abs(ae).mean(),
                            exact_n=np.mean(ne == 0), bias_close=de[close].mean())
        shot_tabs.append(pd.DataFrame(dict(model=lab, shot_id=ids, goal=y, true_distance=td, read_distance=rd,
                                           true_angle_deg=np.degrees(ta), read_angle_deg=np.degrees(ra),
                                           true_opponents=tn, read_opponents=rn, map_only_xg=p[mapk],
                                           hybrid_xg=p[hyb], direct_xg=p[direct], statsbomb_xg=p["StatsBomb xG"])))

    L = ["# Hybrid: the models read the geometry, the map-only logistic model scores it", "",
         "Each model reads three numbers off the same image as the zero-shot runs (distance to the centre of the goal, "
         "the angle between the posts, opponents in the shooting triangle, goalkeeper included; scale given in the "
         "prompt from the pitch markings; no xG asked). The map-only logistic model scores the readings with the fold "
         "model that never saw the shot's match, exactly as it scores the true values. Brackets: 95% match-clustered "
         "bootstrap intervals; ΔBrier paired against the map-only baseline on the same shots.", "",
         "## How well do the models read the image?", "",
         "Error = reading − true value. Opponents: share read exactly, and within ±1.", "",
         "| Model | Band | n | Distance bias (yd) | Distance MAE | Distance median AE | r | Angle bias (°) | Angle MAE (°) | Opponents exact | within ±1 | Opponents bias |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"] + acc_rows
    L += ["", "## xG from the readings", "",
          "| Model | Scored by | AUC | Brier | Mean xG | ΔBrier vs map-only | p |", "|---|---|---|---|---|---|---|"] + xg_rows
    L += ["", "## Decomposition of each model's Brier gap to the map-only baseline", "",
          "Direct − map-only = (hybrid − map-only: misreading the image) + (direct − hybrid: turning what it reads into "
          "a probability).", "",
          "| Model | n | Direct − map-only | p | Misreading: hybrid − map-only | p | Judgment: direct − hybrid | p |",
          "|---|---|---|---|---|---|---|---|"]
    for lab, s in summary.items():
        L.append(f"| {lab} | {s['n']} | {fs(*s['tot'][:3])} | {C.fmt_p(s['tot'][3])} | {fs(*s['perc'][:3])} | "
                 f"{C.fmt_p(s['perc'][3])} | {fs(*s['judg'][:3])} | {C.fmt_p(s['judg'][3])} |")
    L += ["", "## Mean xG by distance band", "",
          "| Model | Band (yd) | n | Observed | StatsBomb | Map-only (true geometry) | Hybrid (model's readings) | Direct (model's xG) |",
          "|---|---|---|---|---|---|---|---|"] + band_rows
    L += ["", "## Reading", ""]
    for lab, s in summary.items():
        tot, perc, judg = s["tot"], s["perc"], s["judg"]
        sig = lambda q: "significant" if q[1] > 0 or q[2] < 0 else "not significant"
        L.append(f"- {lab} ({s['n']} shots): distance read with a mean absolute error of {s['mae_d']:.2f} yd "
                 f"(bias at 6–12 yd {s['bias_close']:+.2f} yd), angle {s['mae_a']:.1f}°, opponents exact in "
                 f"{s['exact_n']:.0%}. Scored by the logistic model, the readings cost {perc[0]:+.4f} Brier against "
                 f"the true geometry ({sig(perc)}; AUC {s['auc_h']:.3f} vs {s['auc_m']:.3f}). The model's own xG differs "
                 f"from the hybrid by {judg[0]:+.4f} ({sig(judg)}; AUC {s['auc_d']:.3f}). Per shot, distance misreading "
                 f"and the direct xG's gap to the map-only baseline correlate at ρ = {s['rho']:.2f} "
                 f"(6–12 yd: {s['rho_close']:.2f}).")
    L += ["", "Caveat: the readings come from a separate request that asks only for measurements; in the xG request the "
          "model may look less carefully (it thinks far less there: results/cost_summary.md). The hybrid shows what "
          "the model can read off the image, not what it used when it gave its xG."]
    with open(os.path.join(C.RES, "hybrid.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    if shot_tabs:
        pd.concat(shot_tabs).to_csv(os.path.join(C.RES, "hybrid_shots.csv"), index=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()
