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
Partial hybrids swap in one reading at a time (which misreading matters). Opus's zero-shot run is the original run
(images rendered in the authors' environment); its rerun (claude_opus_5_5_rerun, images rendered like the readings')
is shown next to it.

Only perception runs whose last invocation finished are used. Each model is evaluated on compare_models.py's common
shots that its run answered (Gemini 4 Carbon measured 150 of the 300), and all three again on the shots every one of
them measured. Also reported: thinking tokens per shot (from calls.jsonl), the band table split by body part, a
sanity check with a rescaled image (probe_runs.py --scale), and the measure-then-estimate run (xg_vlm_pilot.py
--measure-first) when it exists.
Match-clustered bootstrap (2,000 resamples, seed 0), paired; p = two-sided bootstrap p-value.
Outputs: results/hybrid.md, results/hybrid_shots.csv
Run: python hybrid.py
"""
import os, json
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
# the zero-shot xG run whose images were rendered in the same environment as the readings' (for thinking tokens)
SAME_ENV_XG = {"Claude Opus 5.5": "claude_opus_5_5_rerun", "Gemini 4 Carbon": "gemini_4_carbon",
               "GPT-6.2 Astra": "gpt_6.2_astra"}
RERUN = {"Claude Opus 5.5": "Claude Opus 5.5 (rerun)"}
SCALED = [("Claude Opus 5.5", "claude_opus_5_5_perception", "claude_opus_5_5_perception_scale07", 0.7)]
MEASURE = [("GPT-6.2 Astra", "GPT-6.2 Astra, measure first", "gpt_6.2_astra_measure")]
PX_PER_YARD = 5.875   # render(): VerticalPitch drawn at figsize 5 x 5, saved at dpi 110 (transData, checked by hand)


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


def finished(out_dir):
    """The run's last invocation ended (complete or partial); runs in progress or stopped are left out."""
    path = os.path.join(out_dir, "run_config.json")
    if not os.path.exists(path):
        return False
    with open(path) as f:
        last = json.load(f)["invocations"][-1]
    return "finished" in last and str(last.get("status", "")).startswith(("complete", "partial"))


def readings(fol):
    out_dir = os.path.join(C.RES, fol)
    path = os.path.join(out_dir, "answers.csv")
    if not (os.path.exists(path) and finished(out_dir)):
        return None
    return pd.read_csv(path).dropna(subset=["read_distance", "read_angle_deg", "read_opponents"]).set_index("shot_id")


def thinking(fol):
    calls = [c for c in P.read_jsonl(os.path.join(C.RES, fol, "calls.jsonl")) if c.get("kind", "call") == "call"
             and c.get("ok")]
    if not calls:
        return None
    th = np.array([(c.get("usage") or {}).get("thinking_tokens") or 0 for c in calls], float)
    return dict(n=len(calls), mean=th.mean(), median=np.median(th), max=th.max(),
                cost=np.mean([c.get("cost_usd") or 0 for c in calls]))


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
    header = base.header.values.astype(bool)
    R = {lab: a for lab, fol in MODELS if lab in preds and (a := readings(f"{fol}_perception")) is not None}

    def evaluate(lab, mask):
        ids = [s for s, m in zip(common, mask) if m]
        r, tt = R[lab].loc[ids], truth.loc[ids]
        g = dict(rd=r.read_distance.values, ra=np.radians(r.read_angle_deg.values), rn=r.read_opponents.values,
                 td=tt.distance.values, ta=tt.angle.values, tn=tt.defenders_in_cone.values)
        p = {"Map-only baseline (true geometry)": preds["Map-only baseline"][mask],
             "Hybrid: all three read by the model": score(models, fold_of, ids, g["rd"], g["ra"], g["rn"]),
             "Hybrid: distance read, rest true": score(models, fold_of, ids, g["rd"], g["ta"], g["tn"]),
             "Hybrid: angle read, rest true": score(models, fold_of, ids, g["td"], g["ra"], g["tn"]),
             "Hybrid: opponents read, rest true": score(models, fold_of, ids, g["td"], g["ta"], g["rn"]),
             "Direct: the model's own xG (zero-shot, image)": preds[lab][mask]}
        if lab in RERUN and RERUN[lab] in preds:
            p["Direct, rerun (images rendered like the readings')"] = preds[RERUN[lab]][mask]
        p["StatsBomb xG"] = preds["StatsBomb xG"][mask]
        return ids, p, g

    MAPK, HYB, DIR = "Map-only baseline (true geometry)", "Hybrid: all three read by the model", \
        "Direct: the model's own xG (zero-shot, image)"
    fs, f3 = (lambda x, lo, hi: C.fmt(x, lo, hi, sign=True)), (lambda x, lo, hi: C.fmt(x, lo, hi))
    acc_rows, xg_rows, band_rows, shot_tabs, summary = [], [], [], [], {}
    for lab, a in R.items():
        mask = np.array([s in a.index for s in common])
        ids, p, g = evaluate(lab, mask)
        y = base.goal.values[mask].astype(int)
        idx = C.cluster_bootstrap(base.match_id.values[mask])
        bb = {k: np.array([C.brier(y[ix], v[ix]) for ix in idx]) for k, v in p.items()}
        ba = {k: np.array([C.auc(y[ix], v[ix]) for ix in idx]) for k, v in p.items()}
        for k, v in p.items():
            d = bb[k] - bb[MAPK]
            xg_rows.append(f"| {lab} | {k} | {f3(C.auc(y, v), *C.ci(ba[k]))} | {f3(C.brier(y, v), *C.ci(bb[k]))} | "
                           f"{v.mean():.3f} | " + ("" if k == MAPK else
                           f"{fs(C.brier(y, v) - C.brier(y, p[MAPK]), *C.ci(d))} | {C.fmt_p(C.boot_p(d))}") +
                           (" | |" if k == MAPK else " |"))
        perc, judg, tot = bb[HYB] - bb[MAPK], bb[DIR] - bb[HYB], bb[DIR] - bb[MAPK]
        val = lambda k1, k2: C.brier(y, p[k1]) - C.brier(y, p[k2])
        summary[lab] = dict(n=len(ids), perc=(val(HYB, MAPK), *C.ci(perc), C.boot_p(perc)),
                            judg=(val(DIR, HYB), *C.ci(judg), C.boot_p(judg)),
                            tot=(val(DIR, MAPK), *C.ci(tot), C.boot_p(tot)),
                            auc_h=C.auc(y, p[HYB]), auc_m=C.auc(y, p[MAPK]), auc_d=C.auc(y, p[DIR]))
        de, ae, ne = g["rd"] - g["td"], np.degrees(g["ra"] - g["ta"]), g["rn"] - g["tn"]
        bands = np.asarray(pd.cut(g["td"], C.BANDS, labels=C.BAND_LABELS))
        head = header[mask]
        acc_rows.append(f"| {lab} | all | {len(ids)} | {de.mean():+.2f} | {np.abs(de).mean():.2f} | "
                        f"{np.median(np.abs(de)):.2f} | {np.corrcoef(g['rd'], g['td'])[0, 1]:.3f} | {ae.mean():+.1f} | "
                        f"{np.abs(ae).mean():.1f} | {np.mean(ne == 0):.0%} | {np.mean(np.abs(ne) <= 1):.0%} | {ne.mean():+.2f} |")
        for bl in C.BAND_LABELS:
            m = bands == bl
            if not m.sum():
                continue
            acc_rows.append(f"| {lab} | {bl} yd | {m.sum()} | {de[m].mean():+.2f} | {np.abs(de[m]).mean():.2f} | "
                            f"{np.median(np.abs(de[m])):.2f} | | {ae[m].mean():+.1f} | {np.abs(ae[m]).mean():.1f} | "
                            f"{np.mean(ne[m] == 0):.0%} | {np.mean(np.abs(ne[m]) <= 1):.0%} | {ne[m].mean():+.2f} |")
            for body, mb in [("all", m), ("header", m & head), ("foot/other", m & ~head)]:
                if mb.sum():
                    band_rows.append(f"| {lab} | {bl} | {body} | {mb.sum()} | {y[mb].mean():.2f} | "
                                     f"{p['StatsBomb xG'][mb].mean():.3f} | {p[MAPK][mb].mean():.3f} | "
                                     f"{p[HYB][mb].mean():.3f} | {p[DIR][mb].mean():.3f} |")
        gap = p[DIR] - p[MAPK]
        close = bands == "6–12"
        summary[lab].update(rho=spearmanr(de, gap).correlation,
                            rho_close=spearmanr(de[close], gap[close]).correlation if close.sum() > 5 else np.nan,
                            mae_d=np.abs(de).mean(), mae_a=np.abs(ae).mean(), exact_n=np.mean(ne == 0),
                            bias_close=de[close].mean(),
                            close=dict((k, (p[k][close & head].mean(), p[k][close & ~head].mean()))
                                       for k in [MAPK, HYB, DIR, "StatsBomb xG"]),
                            n_close=(int((close & head).sum()), int((close & ~head).sum())))
        shot_tabs.append(pd.DataFrame(dict(model=lab, shot_id=ids, goal=y, header=head.astype(int),
                                           true_distance=g["td"], read_distance=g["rd"],
                                           true_angle_deg=np.degrees(g["ta"]), read_angle_deg=np.degrees(g["ra"]),
                                           true_opponents=g["tn"], read_opponents=g["rn"], map_only_xg=p[MAPK],
                                           hybrid_xg=p[HYB], direct_xg=p[DIR], statsbomb_xg=p["StatsBomb xG"])))

    L = ["# Hybrid: the models read the geometry, the map-only logistic model scores it", "",
         "Each model reads three numbers off the same image as the zero-shot runs (distance to the centre of the goal, "
         "the angle between the posts, opponents in the shooting triangle, goalkeeper included; scale given in the "
         "prompt from the pitch markings; no xG asked). The map-only logistic model scores the readings with the fold "
         "model that never saw the shot's match, exactly as it scores the true values. Only perception runs that "
         "finished are used; Gemini 4 Carbon measured the first 150 shots of the 300 (cost), the others all 300. "
         "Brackets: 95% match-clustered bootstrap intervals; ΔBrier paired against the map-only baseline on the same "
         "shots.", "",
         "Opus's \"Direct\" row is its original zero-shot run, whose images were rendered in the authors' environment; "
         "\"Direct, rerun\" is the same zero-shot setting queried again (`claude_opus_5_5_rerun`) with images rendered "
         "in this environment, like the images the readings came from. Carbon's and GPT's zero-shot runs were already "
         "rendered here.", "",
         "## How well do the models read the image?", "",
         "Error = reading − true value. Opponents: share read exactly, and within ±1.", "",
         "| Model | Band | n | Distance bias (yd) | Distance MAE | Distance median AE | r | Angle bias (°) | Angle MAE (°) | Opponents exact | within ±1 | Opponents bias |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"] + acc_rows

    # thinking effort: measuring request vs the xG request
    L += ["", "## Thinking per shot: measuring vs estimating xG", "",
          "Thinking tokens per answered call, from each run's `calls.jsonl` (the xG run is the zero-shot run rendered in "
          "the same environment).", "",
          "| Model | Measuring: mean | median | max | cost per shot | xG request: mean thinking tokens |", "|---|---|---|---|---|---|"]
    think = {}
    for lab, fol in MODELS:
        tp, tx = thinking(f"{fol}_perception"), thinking(SAME_ENV_XG[lab])
        if tp:
            think[lab] = tp
            L.append(f"| {lab} | {tp['mean']:,.0f} | {tp['median']:,.0f} | {tp['max']:,.0f} | ${tp['cost']:.4f} | "
                     + (f"{tx['mean']:,.0f} |" if tx else "n/a |"))

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

    # the three models on the same shots (the ones every perception run measured)
    if len(R) > 1:
        both = np.array([all(s in a.index for a in R.values()) for s in common])
        y = base.goal.values[both].astype(int)
        idx = C.cluster_bootstrap(base.match_id.values[both])
        L += ["", f"## All three models on the same {both.sum()} shots ({y.sum()} goals)", "",
              "The common shots that every perception run measured, so the models can be compared with each other "
              "directly.", "",
              "| Model | Distance MAE (yd) | Angle MAE (°) | Opponents exact | Brier: map-only / hybrid / direct | AUC: map-only / hybrid / direct | Hybrid − map-only [95% CI] | p | Direct − hybrid [95% CI] | p |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for lab in R:
            ids, p, g = evaluate(lab, both)
            bb = {k: np.array([C.brier(y[ix], p[k][ix]) for ix in idx]) for k in (MAPK, HYB, DIR)}
            d1, d2 = bb[HYB] - bb[MAPK], bb[DIR] - bb[HYB]
            L.append(f"| {lab} | {np.abs(g['rd'] - g['td']).mean():.2f} | {np.degrees(np.abs(g['ra'] - g['ta'])).mean():.1f} | "
                     f"{np.mean(g['rn'] == g['tn']):.0%} | " + " / ".join(f"{C.brier(y, p[k]):.3f}" for k in (MAPK, HYB, DIR)) +
                     " | " + " / ".join(f"{C.auc(y, p[k]):.3f}" for k in (MAPK, HYB, DIR)) +
                     f" | {fs(C.brier(y, p[HYB]) - C.brier(y, p[MAPK]), *C.ci(d1))} | {C.fmt_p(C.boot_p(d1))} | "
                     f"{fs(C.brier(y, p[DIR]) - C.brier(y, p[HYB]), *C.ci(d2))} | {C.fmt_p(C.boot_p(d2))} |")

    L += ["", "## Mean xG by distance band and body part", "",
          "The drawing does not show the body part, so neither the map-only model nor the readings can tell a header "
          "from a foot shot.", "",
          "| Model | Band (yd) | Body part | n | Observed | StatsBomb | Map-only (true geometry) | Hybrid (model's readings) | Direct (model's xG) |",
          "|---|---|---|---|---|---|---|---|---|"] + band_rows

    # sanity check: the same images rescaled (is the precision tied to the pixel grid?)
    for lab, full, fol, scale in SCALED:
        a_s, a_f = readings(fol), readings(full)
        if a_s is None or a_f is None:
            continue
        ids = [s for s in a_s.index if s in a_f.index]
        tt = truth.loc[ids]
        rows, mae = [], []
        for name, a in [("full size", a_f), (f"{scale}x", a_s)]:
            r = a.loc[ids]
            de = r.read_distance.values - tt.distance.values
            ae = r.read_angle_deg.values - np.degrees(tt.angle.values)
            ne = r.read_opponents.values - tt.defenders_in_cone.values
            mae.append(np.abs(de).mean())
            rows.append(f"| {name} | {np.abs(de).mean():.2f} | {np.median(np.abs(de)):.2f} | {np.abs(de).max():.2f} | "
                        f"{de.mean():+.2f} | {np.abs(ae).mean():.2f} | {np.mean(ne == 0):.0%} |".replace("| -", "| −"))
        rows += ["", f"Distance error in pixels: {mae[0] * PX_PER_YARD:.2f} px at full size, "
                     f"{mae[1] * PX_PER_YARD * scale:.2f} px at {scale}x; in yards it grew by a factor of "
                     f"{mae[1] / mae[0]:.2f} (1/{scale} = {1 / scale:.2f}). "
                 + ("The error follows the pixel grid: the model locates the markers to under a pixel at both sizes, so "
                    "the readings are measured off the image." if 0.8 / scale <= mae[1] / mae[0] <= 1.6 / scale else
                    "The error does not scale like the pixel grid; see the per-shot readings.")]
        L += ["", f"## Sanity check: {lab} reading the same {len(ids)} images rescaled to {scale}x", "",
              f"The rendered images have {PX_PER_YARD:.1f} pixels per yard (from the renderer's transform; 539 x 421 px), "
              f"the rescaled ones {PX_PER_YARD * scale:.1f}. Rescaling re-encodes the PNG (no metadata is carried over). "
              f"If the readings' precision came from the pixel grid alone, their errors would grow by about 1/{scale}.", "",
              "| Image | Distance MAE (yd) | median AE | max AE | bias | Angle MAE (°) | Opponents exact |",
              "|---|---|---|---|---|---|---|"] + rows

    L += ["", "## Reading", ""]
    for lab, s in summary.items():
        tot, perc, judg = s["tot"], s["perc"], s["judg"]
        sig = lambda q: "significant" if q[1] > 0 or q[2] < 0 else "not significant"
        ch, cf = s["n_close"]
        L.append(f"- {lab} ({s['n']} shots): distance read with a mean absolute error of {s['mae_d']:.2f} yd "
                 f"(bias at 6–12 yd {s['bias_close']:+.2f} yd), angle {s['mae_a']:.1f}°, opponents exact in "
                 f"{s['exact_n']:.0%}. Scored by the logistic model, the readings cost {perc[0]:+.4f} Brier against "
                 f"the true geometry ({sig(perc)}; AUC {s['auc_h']:.3f} vs {s['auc_m']:.3f}). The model's own xG differs "
                 f"from the hybrid by {judg[0]:+.4f} ({sig(judg)}; AUC {s['auc_d']:.3f}). At 6–12 yd, headers ({ch}): "
                 f"StatsBomb {s['close']['StatsBomb xG'][0]:.3f}, map-only {s['close'][MAPK][0]:.3f}, hybrid "
                 f"{s['close'][HYB][0]:.3f}, the model {s['close'][DIR][0]:.3f}; foot shots ({cf}): StatsBomb "
                 f"{s['close']['StatsBomb xG'][1]:.3f}, map-only {s['close'][MAPK][1]:.3f}, hybrid "
                 f"{s['close'][HYB][1]:.3f}, the model {s['close'][DIR][1]:.3f}. Per shot, distance misreading and the "
                 f"direct xG's gap to the map-only baseline correlate at ρ = {s['rho']:.2f} (6–12 yd: {s['rho_close']:.2f})."
                 .replace("-0.", "−0."))
    if think:
        L.append("- Thinking: measuring takes far more thinking than estimating xG ("
                 + "; ".join(f"{lab} {v['mean']:,.0f} tokens per shot on average" for lab, v in think.items())
                 + "; the xG requests in the table above). The hybrid shows what a model can read off the image when "
                 "asked to measure, not what it used when it gave its xG.")
    if summary:
        over_h = [lab for lab, s in summary.items() if s["close"][DIR][0] > s["close"]["StatsBomb xG"][0]]
        map_over = all(s["close"][MAPK][0] > s["close"]["StatsBomb xG"][0] for s in summary.values())
        below_f = [lab for lab, s in summary.items() if s["close"][DIR][1] < s["close"]["StatsBomb xG"][1]]
        above_f = [lab for lab, s in summary.items() if s["close"][DIR][1] >= s["close"]["StatsBomb xG"][1]]
        L.append("- Close range: at 6–12 yd the over-estimate sits in the headers. "
                 + (f"{', '.join(over_h)} rate headers above StatsBomb" if over_h else "No model rates headers above StatsBomb")
                 + (", and so does the map-only model from the true geometry, because nothing in the drawing marks a "
                    "header." if map_over else ".")
                 + (f" On foot shots from the same range {', '.join(below_f)} {'is' if len(below_f) == 1 else 'are'} "
                    "below StatsBomb" if below_f else "")
                 + (f"{'; ' if below_f else ' On foot shots '}{', '.join(above_f)} "
                    f"{'is' if len(above_f) == 1 else 'are'} at or above it" if above_f else "")
                 + " (per-model numbers in the bullets above and the band table).")
    with open(os.path.join(C.RES, "hybrid.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    if shot_tabs:
        pd.concat(shot_tabs).to_csv(os.path.join(C.RES, "hybrid_shots.csv"), index=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()
