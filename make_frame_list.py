"""
Shot list for the broadcast-frame pilot: which World Cup moments to capture by hand (instructions:
docs/frame_capture.md).

30 shots from compare_models.py's common shots:
  12  the error gallery (results/figure_error_gallery.png): the six largest consensus over-estimates (mostly headers
      from 7-13 yards) and the six largest consensus under-estimates (close-range goals), as chosen in
      error_analysis.py
  18  drawn at random from the other common shots (seed 2027), stratified by distance band in proportion to the bands'
      sizes (largest-remainder rounding)
Each row says where to find the moment: date, stage, teams, half, the match clock (StatsBomb's event clock, which
matches the broadcast clock) and the minute as written in match reports; the shooter (common name and shirt number);
and what to expect (body part, distance, goal or not) to confirm it is the right shot.
Output: docs/frame_capture_list.csv
Run: python make_frame_list.py
"""
import os
import numpy as np
import pandas as pd
import compare_models as C
import xg_vlm_pilot as P
import probe_runs as R

SEED, N_RANDOM = 2027, 18
OUT = os.path.join("docs", "frame_capture_list.csv")


def main():
    shots = P.load_shots_cached(43, 106).set_index("shot_id")
    names = R.player_names(shots.reset_index())
    common = C.common_shots(C.load_runs(verbose=False), verbose=False)
    err = pd.read_csv(os.path.join(C.RES, "error_analysis_shots.csv"), index_col=0)
    k = 5   # frontier models: all five on the same side of StatsBomb
    over = err[err.n_above_sb == k].nlargest(6, "gap").index.tolist()
    under = err[err.n_above_sb == 0].nsmallest(6, "gap").index.tolist()
    groups = {s: "gallery: over-estimate" for s in over} | {s: "gallery: under-estimate" for s in under}

    rest = shots.loc[[s for s in common if s not in groups]]
    band = pd.cut(rest.distance, C.BANDS, labels=C.BAND_LABELS)
    sizes = band.value_counts().reindex(C.BAND_LABELS)
    quota = sizes / sizes.sum() * N_RANDOM
    alloc = np.floor(quota).astype(int)
    for b in (quota - alloc).sort_values(ascending=False).index[: N_RANDOM - alloc.sum()]:
        alloc[b] += 1
    rng = np.random.default_rng(SEED)
    for b in C.BAND_LABELS:
        pool = sorted(rest.index[band == b])
        for s in rng.choice(pool, size=int(alloc[b]), replace=False):
            groups[str(s)] = f"random ({b} yd)"

    rows = []
    for i, (s, grp) in enumerate(groups.items(), 1):
        r = shots.loc[s]
        who = names.get(r.player, {})
        rows.append(dict(
            order=i, group=grp, shot_id=s, match_date=r.match_date, stage=R.STAGE.get(r.competition_stage),
            home_team=r.home_team, away_team=r.away_team, shooting_team=r.team, half=R.PERIOD[int(r.period)],
            match_clock=f"{int(r.minute):02d}:{int(r.second):02d}", report_minute=R.reported_minute(r.period, r.minute),
            shooter=who.get("name", r.player), shirt_number=who.get("number", ""), body_part=r.body_part,
            distance_yd=round(float(r.distance), 1), play_pattern=r.play_pattern,
            outcome="goal" if r.goal else "no goal", main_file=f"{s}_main.png", replay_file=f"{s}_replay.png"))
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(out[["order", "group", "match_date", "home_team", "away_team", "match_clock", "shooter", "outcome"]]
          .to_string(index=False))
    n_matches = out[["home_team", "away_team", "match_date"]].drop_duplicates().shape[0]
    print(f"\n{len(out)} shots ({(out.outcome == 'goal').sum()} goals) from {n_matches} matches on "
          f"{out.match_date.nunique()} dates -> {OUT}")


if __name__ == "__main__":
    main()
