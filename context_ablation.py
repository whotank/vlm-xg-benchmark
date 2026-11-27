"""
Body-part context ablation (Stage 5). The zero-shot prompt shows only the image, so a model cannot tell a
header from a foot shot. Runs made with --context add one line, "Body part used: <part>.", and nothing
else changes. This script pairs each zero-shot run with its _context run and asks whether the extra line
fixes the header over-estimate.

For every pair found among compare_models.py's runs (same common shots), by body-part group (Head, Foot =
left or right foot; "Other" has 3 shots and is shown for completeness) and overall:
  mean predicted xG without and with context, observed goal rate, StatsBomb mean,
  Brier without and with context, and the paired changes, all with match-clustered bootstrap 95% CIs
  (2,000 resamples of matches, seed 0, the same resamples for every quantity).
Overall AUC / Brier / log loss with and without context are also reported.

Caveat: the original zero-shot Opus run used images rendered in the authors' environment; every newer run
(including the context runs) used images re-rendered for this work with the same code. If a rerun of the
zero-shot setting exists (results/<model>_rerun), it is shown as a no-context control: its difference from
the original run sizes the run-to-run (and rendering) noise.

Outputs: results/context_ablation.md, results/context_ablation.csv
Run: python context_ablation.py
"""
import os
import numpy as np
import pandas as pd
import compare_models as C

GROUPS = {"Head": ["Head"], "Foot": ["Left Foot", "Right Foot"], "Other": ["Other"], "All shots": None}


def main():
    runs = C.load_runs(verbose=False)
    common = C.common_shots(runs, verbose=False)
    base, preds, groups, folders = C.assemble(runs, common)
    by_folder = {f: lab for lab, f in folders.items()}
    y = base.goal.values.astype(int)
    idxs = C.cluster_bootstrap(base.match_id.values)
    body = base.body_part.values

    pairs = []
    for lab, fol in C.BASE:
        if fol in by_folder and f"{fol}_context" in by_folder:
            ctrl = by_folder.get(f"{fol}_rerun")
            pairs.append((lab, by_folder[fol], by_folder[f"{fol}_context"], ctrl))
    if not pairs:
        raise SystemExit("no zero-shot/_context pairs found")

    rows = []
    for lab, m0, m1, ctrl in pairs:
        variants = [("no context", m0), ("body part given", m1)] + ([("no context, rerun", ctrl)] if ctrl else [])
        for gname, parts in GROUPS.items():
            mask = np.ones(len(y), bool) if parts is None else np.isin(body, parts)
            if mask.sum() == 0:
                continue
            sel = np.flatnonzero(mask)

            def boot(fn, p):
                out = []
                for ix in idxs:
                    ix = ix[mask[ix]]
                    out.append(fn(y[ix], p[ix]) if len(ix) else np.nan)
                return np.array(out)

            r = dict(model=lab, group=gname, n=int(mask.sum()), goals=int(y[sel].sum()),
                     observed=y[sel].mean(), statsbomb=preds["StatsBomb xG"][sel].mean())
            r["observed_lo"], r["observed_hi"] = C.ci(boot(lambda yy, pp: yy.mean(), preds["StatsBomb xG"]))
            bm, bb = {}, {}
            for vname, m in variants:
                p = preds[m]
                tag = {"no context": "nc", "body part given": "ctx", "no context, rerun": "rerun"}[vname]
                bm[tag] = boot(C.mean_xg, p)
                bb[tag] = boot(C.brier, p)
                r[f"mean_{tag}"], r[f"brier_{tag}"] = p[sel].mean(), C.brier(y[sel], p[sel])
                r[f"mean_{tag}_lo"], r[f"mean_{tag}_hi"] = C.ci(bm[tag])
                r[f"brier_{tag}_lo"], r[f"brier_{tag}_hi"] = C.ci(bb[tag])
                if gname == "All shots":
                    r[f"auc_{tag}"] = C.auc(y, p)
                    r[f"auc_{tag}_lo"], r[f"auc_{tag}_hi"] = C.ci(np.array([C.auc(y[ix], p[ix]) for ix in idxs]))
                    r[f"logloss_{tag}"] = C.logloss(y, p)
            for tag in [t for t in ("ctx", "rerun") if t in bm]:
                d_mean, d_brier = bm[tag] - bm["nc"], bb[tag] - bb["nc"]
                r[f"dmean_{tag}"] = r[f"mean_{tag}"] - r["mean_nc"]
                r[f"dmean_{tag}_lo"], r[f"dmean_{tag}_hi"] = C.ci(d_mean)
                r[f"dbrier_{tag}"] = r[f"brier_{tag}"] - r["brier_nc"]
                r[f"dbrier_{tag}_lo"], r[f"dbrier_{tag}_hi"] = C.ci(d_brier)
                r[f"dbrier_{tag}_p"] = C.boot_p(d_brier)
            r["pair"] = f"{m0} -> {m1}" + (f" (control: {ctrl})" if ctrl else "")
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(C.RES, "context_ablation.csv"), index=False)

    f3 = lambda x, lo, hi: C.fmt(x, lo, hi)
    fd = lambda x, lo, hi: C.fmt(x, lo, hi, sign=True)
    L = ["# Body-part context ablation", "",
         f"Evaluation set: the {len(y)} common shots of compare_models.py ({y.sum()} goals, "
         f"{base.match_id.nunique()} matches). The context runs add one prompt line, \"Body part used: <part>.\"; "
         "everything else (image, prompt, settings) is unchanged. Brackets: 95% match-clustered bootstrap intervals "
         "(2,000 resamples of matches, seed 0); Δ = with context − without; p = two-sided bootstrap p-value. "
         "Foot = left or right foot.", ""]
    for lab, m0, m1, ctrl in pairs:
        g = df[df.model == lab]
        L += [f"## {lab}", "", f"Runs: `{folders[m0]}` (no context) vs `{folders[m1]}` (body part given)"
              + (f"; control rerun without context: `{folders[ctrl]}`" if ctrl else "") + ".", "",
              "### Mean predicted xG by body part", "",
              "| Shots | n (goals) | Observed rate | StatsBomb | No context | Body part given | Δ mean xG |",
              "|---|---|---|---|---|---|---|"]
        for _, r in g.iterrows():
            L.append(f"| {r.group} | {r.n} ({r.goals}) | {f3(r.observed, r.observed_lo, r.observed_hi)} | "
                     f"{r.statsbomb:.3f} | {f3(r.mean_nc, r.mean_nc_lo, r.mean_nc_hi)} | "
                     f"{f3(r.mean_ctx, r.mean_ctx_lo, r.mean_ctx_hi)} | {fd(r.dmean_ctx, r.dmean_ctx_lo, r.dmean_ctx_hi)} |")
        L += ["", "### Brier score by body part", "",
              "| Shots | n (goals) | No context | Body part given | Δ Brier | p |", "|---|---|---|---|---|---|"]
        for _, r in g.iterrows():
            L.append(f"| {r.group} | {r.n} ({r.goals}) | {f3(r.brier_nc, r.brier_nc_lo, r.brier_nc_hi)} | "
                     f"{f3(r.brier_ctx, r.brier_ctx_lo, r.brier_ctx_hi)} | "
                     f"{fd(r.dbrier_ctx, r.dbrier_ctx_lo, r.dbrier_ctx_hi)} | {C.fmt_p(r.dbrier_ctx_p)} |")
        a = g[g.group == "All shots"].iloc[0]
        L += ["", f"Overall: AUC {f3(a.auc_nc, a.auc_nc_lo, a.auc_nc_hi)} without context vs "
                  f"{f3(a.auc_ctx, a.auc_ctx_lo, a.auc_ctx_hi)} with; log loss {a.logloss_nc:.3f} vs {a.logloss_ctx:.3f}."]
        if ctrl:
            L += ["", "### Control: the same no-context setting queried again", "",
                  "| Shots | No context (original) | No context (rerun) | Δ mean xG | Δ Brier | p |",
                  "|---|---|---|---|---|---|"]
            for _, r in g.iterrows():
                L.append(f"| {r.group} | {r.mean_nc:.3f} | {r.mean_rerun:.3f} | "
                         f"{fd(r.dmean_rerun, r.dmean_rerun_lo, r.dmean_rerun_hi)} | "
                         f"{fd(r.dbrier_rerun, r.dbrier_rerun_lo, r.dbrier_rerun_hi)} | {C.fmt_p(r.dbrier_rerun_p)} |")
            L.append("")
            L.append(f"Overall AUC of the rerun: {f3(a.auc_rerun, a.auc_rerun_lo, a.auc_rerun_hi)}.")
        L.append("")
    with open(os.path.join(C.RES, "context_ablation.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print("Saved results/context_ablation.md and results/context_ablation.csv")


if __name__ == "__main__":
    main()
