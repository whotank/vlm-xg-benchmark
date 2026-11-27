"""
Broadcast-frame pilot: the same shots as a drawing and as a captured television frame (xg_vlm_pilot.py --input
frame; frames listed in docs/frame_capture_list.csv, captured as in docs/frame_capture.md).

Regression to the mean: rows 1-12 of the list (the error gallery) were chosen because the original zero-shot drawing
runs got them most wrong, so any new query of those shots (frame or drawing) will tend to land closer to StatsBomb.
The comparison is therefore
  - frame vs a fresh drawing query of the same 30 shots (xg_vlm_pilot.py --shot-list docs/frame_capture_list.csv
    --tag framelist, folder <model>_framelist), when it exists; it shares the selection effect, so frame - drawing is
    fair on all 30 shots;
  - otherwise frame vs the original drawing run, read on the 18 random shots only (the gallery rows are shown but
    flagged).
Everything is reported for all shots, the 18 random shots and the 12 gallery shots separately. The random shots hold
only one goal, so they support comparisons of level (mean xG) and Brier, not of ranking (AUC).
Headers are listed separately (a frame shows the body part, a drawing does not). When the identify probe was run on
the frames (probe_runs.py --task identify --view <view>), whether the model recognised the match is shown per shot: a
frame that gives the match away makes a good score suspect (results/contamination_probe.md).
Paired differences: match-clustered bootstrap (2,000 resamples, seed 0); with 30 shots, read them as descriptive.
Output: results/frames_pilot.md
Run: python frames_analysis.py
"""
import os, json
import numpy as np
import pandas as pd
import compare_models as C
import contamination_probe as CP

MODELS = [("Claude Opus 5.5", "claude_opus_5_5"), ("Gemini 4 Carbon", "gemini_4_carbon"),
          ("GPT-6.2 Astra", "gpt_6.2_astra")]


def finished(folder):
    path = os.path.join(C.RES, folder, "run_config.json")
    if not os.path.exists(path):
        return False
    with open(path) as f:
        last = json.load(f)["invocations"][-1]
    return str(last.get("status", "")).startswith(("complete", "partial"))


def main():
    plan = pd.read_csv(os.path.join("docs", "frame_capture_list.csv")).set_index("shot_id")
    refs = pd.read_csv(os.path.join(C.RES, "baselines.csv")).set_index("shot_id")
    runs = [(lab, fol, d[len(fol) + len("_frames_"):]) for lab, fol in MODELS for d in sorted(os.listdir(C.RES))
            if d.startswith(f"{fol}_frames_") and os.path.exists(os.path.join(C.RES, d, "shots.csv")) and finished(d)]
    L = ["# Broadcast-frame pilot", ""]
    if not runs:
        L.append("No finished frame runs yet: capture frames (docs/frame_capture.md), then run xg_vlm_pilot.py --input frame.")
    fs = lambda x, lo, hi: C.fmt(x, lo, hi, sign=True)
    for lab, fol, view in runs:
        fr = pd.read_csv(os.path.join(C.RES, f"{fol}_frames_{view}", "shots.csv")).set_index("shot_id")
        orig = pd.read_csv(os.path.join(C.RES, fol, "shots.csv")).set_index("shot_id").vlm_xg
        fresh_dir = f"{fol}_framelist"
        fresh = (pd.read_csv(os.path.join(C.RES, fresh_dir, "shots.csv")).set_index("shot_id").vlm_xg
                 if os.path.exists(os.path.join(C.RES, fresh_dir, "shots.csv")) and finished(fresh_dir) else None)
        ids = [s for s in fr.index if s in orig.index and s in refs.index and (fresh is None or s in fresh.index)]
        d = fr.loc[ids]
        y = d.goal.values.astype(int)
        p = {"frame": d.vlm_xg.values}
        if fresh is not None:
            p["drawing, fresh query"] = fresh.loc[ids].values
        p["drawing, original run"] = orig.loc[ids].values
        p["StatsBomb"] = d.sb_xg.values
        p["map-only"] = refs.loc[ids].baseline_image_xg.values
        comp = "drawing, fresh query" if fresh is not None else "drawing, original run"
        grp = np.array([str(plan.group.get(s, "")) for s in ids])
        subsets = [("all shots with a frame", np.ones(len(ids), bool)),
                   ("random shots", np.char.startswith(grp.astype(str), "random")),
                   ("gallery shots (selected on the original drawing run's errors)",
                    np.char.startswith(grp.astype(str), "gallery"))]
        L += [f"## {lab}, view `{view}` ({len(ids)} shots, {y.sum()} goals)", "",
              f"Comparator: {comp}." + ("" if fresh is not None else
              " No fresh drawing query yet: frame − drawing is only fair on the random shots, because the gallery "
              "shots were chosen for the original run's largest errors and will regress toward StatsBomb on any new "
              "query (run the --shot-list command in docs/frame_capture.md)."), ""]
        for name, m in subsets:
            if not m.any():
                continue
            ym = y[m]
            idx = C.cluster_bootstrap(d.match_id.values[m])
            two = 0 < ym.sum() < len(ym) and min(ym.sum(), len(ym) - ym.sum()) >= 3
            L += [f"### {name.capitalize()} ({m.sum()} shots, {ym.sum()} goal{'' if ym.sum() == 1 else 's'})", "",
                  "| Input | Mean xG | Brier | AUC | Δ mean xG vs comparator [95% CI] | ΔBrier vs comparator [95% CI] | p |",
                  "|---|---|---|---|---|---|---|"]
            bc = np.array([C.brier(ym[ix], p[comp][m][ix]) for ix in idx])
            mc = np.array([p[comp][m][ix].mean() for ix in idx])
            for k, v in p.items():
                vm = v[m]
                auc = f"{C.auc(ym, vm):.3f}" if two else "n/a (too few goals)"
                if k == comp:
                    L.append(f"| {k} (comparator) | {vm.mean():.3f} | {C.brier(ym, vm):.3f} | {auc} | | | |")
                    continue
                bk = np.array([C.brier(ym[ix], vm[ix]) for ix in idx])
                mk = np.array([vm[ix].mean() for ix in idx])
                L.append(f"| {k} | {vm.mean():.3f} | {C.brier(ym, vm):.3f} | {auc} | "
                         f"{fs(vm.mean() - p[comp][m].mean(), *C.ci(mk - mc))} | "
                         f"{fs(C.brier(ym, vm) - C.brier(ym, p[comp][m]), *C.ci(bk - bc))} | {C.fmt_p(C.boot_p(bk - bc))} |")
            L.append("")
        head = (d.body_part == "Head").values
        if head.any():
            L += [f"Headers ({head.sum()}, {y[head].sum()} goals): mean xG " + ", ".join(
                f"{k} {v[head].mean():.3f}" for k, v in p.items()) + f"; observed {y[head].mean():.3f}.", ""]
        idp = os.path.join(C.RES, f"probe_identify_frames_{view}_{fol}", "answers.csv")
        rec, team_hit = None, None
        if os.path.exists(idp):
            a = pd.read_csv(idp, keep_default_na=False, na_values=[""]).set_index("shot_id").loc[ids].reset_index()
            facts = a.assign(player_shown=[plan.shooter.get(s, pl) for s, pl in zip(a.shot_id, a.player)])
            hits = CP.identify_hits(a, facts)
            rec = pd.Series(a.recognised.astype(str).str.lower().eq("true").values, index=ids)
            team_hit = pd.Series(hits.either_team.values, index=ids)
            L += [f"Identify probe on the same frames: claims to recognise {int(rec.sum())} of {len(ids)}; best "
                  f"guess names the World Cup {hits.competition.mean():.0%}, 2022 {hits.year.mean():.0%}, a correct "
                  f"team {hits.either_team.mean():.0%}, both teams {hits.both_teams.mean():.0%}, the shooter "
                  f"{hits.shooter.mean():.0%}.", ""]
        L += ["| Shot | Group | Outcome | StatsBomb | Map-only | " + " | ".join(k.capitalize() for k in p
                                                                         if k not in ("StatsBomb", "map-only"))
              + " | Recognised |", "|---|---|---|---|---|" + "---|" * (len(p) - 2) + "---|"]
        for i, s in enumerate(ids):
            L.append(f"| `{s[:8]}` | {grp[i]} | {'goal' if y[i] else 'no goal'} | {p['StatsBomb'][i]:.2f} | "
                     f"{p['map-only'][i]:.2f} | " + " | ".join(f"{v[i]:.2f}" for k, v in p.items()
                                                             if k not in ("StatsBomb", "map-only"))
                     + f" | {'' if rec is None else ('yes' if rec[s] else 'no') + (', team right' if team_hit[s] else '')} |")
        L.append("")
    with open(os.path.join(C.RES, "frames_pilot.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
