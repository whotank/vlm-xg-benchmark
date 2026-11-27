"""
Broadcast-frame pilot: the same shots as a drawing (zero-shot runs) and as a captured television frame
(xg_vlm_pilot.py --input frame; frames listed in docs/frame_capture_list.csv, captured as in docs/frame_capture.md).

Per model and view: xG from the frame next to xG from the drawing, StatsBomb and the map-only baseline, shot by shot;
paired differences on the shots with a frame (match-clustered bootstrap, 2,000 resamples, seed 0: with about 30 shots,
read them as descriptive); headers separately (a frame shows the body part, a drawing does not); and, when the
identify probe was run on the frames (probe_runs.py --task identify --view <view>), whether the model recognised the
match: a frame that gives the match away makes a good score suspect (results/contamination_probe.md).
Output: results/frames_pilot.md
Run: python frames_analysis.py
"""
import os
import numpy as np
import pandas as pd
import compare_models as C
import contamination_probe as CP

MODELS = [("Claude Opus 5.5", "claude_opus_5_5"), ("Gemini 4 Carbon", "gemini_4_carbon"),
          ("GPT-6.2 Astra", "gpt_6.2_astra")]


def main():
    plan = pd.read_csv(os.path.join("docs", "frame_capture_list.csv")).set_index("shot_id")
    refs = pd.read_csv(os.path.join(C.RES, "baselines.csv")).set_index("shot_id")
    runs = [(lab, fol, d[len(fol) + len("_frames_"):]) for lab, fol in MODELS for d in sorted(os.listdir(C.RES))
            if d.startswith(f"{fol}_frames_") and os.path.exists(os.path.join(C.RES, d, "shots.csv"))]
    L = ["# Broadcast-frame pilot", ""]
    if not runs:
        L.append("No frame runs yet: capture frames (docs/frame_capture.md), then run xg_vlm_pilot.py --input frame.")
    fs = lambda x, lo, hi: C.fmt(x, lo, hi, sign=True)
    for lab, fol, view in runs:
        fr = pd.read_csv(os.path.join(C.RES, f"{fol}_frames_{view}", "shots.csv")).set_index("shot_id")
        mp = pd.read_csv(os.path.join(C.RES, fol, "shots.csv")).set_index("shot_id").vlm_xg
        ids = [s for s in fr.index if s in mp.index and s in refs.index]
        d = fr.loc[ids]
        y = d.goal.values.astype(int)
        p = {"frame": d.vlm_xg.values, "drawing": mp.loc[ids].values, "StatsBomb": d.sb_xg.values,
             "map-only": refs.loc[ids].baseline_image_xg.values}
        idx = C.cluster_bootstrap(d.match_id.values)
        L += [f"## {lab}, view `{view}` ({len(ids)} shots, {y.sum()} goals)", "",
              "| Input | Mean xG | Brier | AUC | ΔBrier vs drawing [95% CI] | p |", "|---|---|---|---|---|---|"]
        bd = np.array([C.brier(y[ix], p["drawing"][ix]) for ix in idx])
        for k, v in p.items():
            bk = np.array([C.brier(y[ix], v[ix]) for ix in idx])
            diff = "" if k == "drawing" else f"{fs(C.brier(y, v) - C.brier(y, p['drawing']), *C.ci(bk - bd))} | {C.fmt_p(C.boot_p(bk - bd))}"
            L.append(f"| {k} | {v.mean():.3f} | {C.brier(y, v):.3f} | {C.auc(y, v):.3f} | {diff or ' | '} |")
        head = (d.body_part == "Head").values
        if head.any():
            L += ["", f"Headers ({head.sum()}): mean xG frame {p['frame'][head].mean():.3f}, drawing "
                      f"{p['drawing'][head].mean():.3f}, StatsBomb {p['StatsBomb'][head].mean():.3f}, observed "
                      f"{y[head].mean():.3f}."]
        idp = os.path.join(C.RES, f"probe_identify_frames_{view}_{fol}", "answers.csv")
        rec = None
        if os.path.exists(idp):
            a = pd.read_csv(idp, keep_default_na=False, na_values=[""]).set_index("shot_id").loc[ids].reset_index()
            facts = a.assign(player_shown=[plan.shooter.get(s, pl) for s, pl in zip(a.shot_id, a.player)])
            hits = CP.identify_hits(a, facts)
            rec = pd.Series(a.recognised.astype(str).str.lower().eq("true").values, index=ids)
            L += ["", f"Identify probe on the same frames: claims to recognise {int(rec.sum())} of {len(ids)}; best "
                      f"guess names the World Cup {hits.competition.mean():.0%}, 2022 {hits.year.mean():.0%}, a correct "
                      f"team {hits.either_team.mean():.0%}, both teams {hits.both_teams.mean():.0%}, the shooter "
                      f"{hits.shooter.mean():.0%}."]
            team_hit = pd.Series(hits.either_team.values, index=ids)
        L += ["", "| Shot | Group | Outcome | StatsBomb | Map-only | Drawing | Frame | Recognised |", "|---|---|---|---|---|---|---|---|"]
        for i, s in enumerate(ids):
            L.append(f"| `{s[:8]}` | {plan.group.get(s, '')} | {'goal' if y[i] else 'no goal'} | {p['StatsBomb'][i]:.2f} | "
                     f"{p['map-only'][i]:.2f} | {p['drawing'][i]:.2f} | {p['frame'][i]:.2f} | "
                     f"{'' if rec is None else ('yes' if rec[s] else 'no') + (', team right' if team_hit[s] else '')} |")
        L.append("")
    with open(os.path.join(C.RES, "frames_pilot.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
