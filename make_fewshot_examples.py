"""
Choose a fixed, nested few-shot example set and save it for xg_vlm_pilot.py --shots k.

Pool: every non-penalty shot with a freeze frame that is NOT in the 300-shot test sample
      (shots.sample(n=300, random_state=7)), so no test shot is ever shown as an example.
Bands (yards, as in the paper): <6, 6-12, 12-18, 18-25, 25+ (pd.cut on [0, 6, 12, 18, 25, 200]).
Rules (only the random draws depend on the seed):
  k=1   one miss from the middle band (12-18 yd, "mid-range").
  k=5   the k=1 example + one example from each of the other four bands, exactly one of them a goal.
        The goal is drawn uniformly from all pool goals in those four bands (so its band follows where
        goals actually happen); each remaining band gets a random miss.
  k=10  the k=5 set + one more example from each of the five bands, exactly one of them a goal, drawn
        uniformly from pool goals in the bands that do not have a goal yet; the other bands get a
        random miss. Result: 2 goals and 8 misses, two examples per band.
  Prompt order: the k=1 example, then the k=5 additions in random order, then the k=10 additions in
  random order, so each smaller set is exactly the first k examples of the larger one.
Labels are the actual outcome ("Outcome: GOAL" / "Outcome: NO GOAL"), never xG.

Usage:
  python make_fewshot_examples.py --seed 1 --set A     # -> results/fewshot_examples.json
  python make_fewshot_examples.py --seed 2 --set B     # -> results/fewshot_examples_setB.json
"""
import argparse, datetime, json, os
import numpy as np
import pandas as pd
import xg_vlm_pilot as P

BANDS = [0, 6, 12, 18, 25, 200]
BAND_LABELS = ["<6", "6-12", "12-18", "18-25", "25+"]


def choose(pool, seed):
    rng = np.random.default_rng(seed)
    pool = pool.sort_values("shot_id").reset_index(drop=True)
    pick = lambda df: df.iloc[int(rng.integers(len(df)))]
    taken, rows = set(), []

    def add(r, k):
        taken.add(r.shot_id)
        rows.append(dict(r, added_at_k=k))

    free = lambda: pool[~pool.shot_id.isin(taken)]
    # k = 1: a mid-range miss
    add(pick(free()[(free().band == "12-18") & (free().goal == 0)]), 1)
    # k = 5: one per remaining band, one of them a goal
    others = [b for b in BAND_LABELS if b != "12-18"]
    g = pick(free()[free().band.isin(others) & (free().goal == 1)])
    group = [g]
    for b in others:
        if b != g.band:
            group.append(pick(free()[(free().band == b) & (free().goal == 0)]))
    for i in rng.permutation(len(group)):
        add(group[i], 5)
    # k = 10: one more per band, one goal in a band without a goal yet
    goal_bands = {r["band"] for r in rows if r["goal"] == 1}
    g = pick(free()[~free().band.isin(goal_bands) & (free().goal == 1)])
    group = [g]
    for b in BAND_LABELS:
        if b != g.band:
            group.append(pick(free()[(free().band == b) & (free().goal == 0)]))
    for i in rng.permutation(len(group)):
        add(group[i], 10)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--set", default="A", choices=list(P.EXAMPLE_FILES))
    ap.add_argument("--n", type=int, default=300, help="size of the test sample to exclude")
    ap.add_argument("--sample-seed", type=int, default=7)
    a = ap.parse_args()
    out = P.EXAMPLE_FILES[a.set]
    if os.path.exists(out):
        raise SystemExit(f"{out} exists; runs depend on it, so it is never overwritten.")

    shots = P.load_shots_cached(43, 106)
    test = set(shots.sample(n=a.n, random_state=a.sample_seed).shot_id)
    pool = shots[~shots.shot_id.isin(test)].copy()
    pool["band"] = pd.cut(pool.distance, BANDS, labels=BAND_LABELS).astype(str)
    rows = choose(pool[["shot_id", "match_id", "goal", "distance", "band", "body_part", "play_pattern", "sb_xg"]], a.seed)

    ex = [dict(order=i + 1, shot_id=r["shot_id"], match_id=int(r["match_id"]), goal=int(r["goal"]),
               label=P.OUTCOME[int(r["goal"])], band=r["band"], distance_yards=round(float(r["distance"]), 2),
               body_part=r["body_part"], play_pattern=r["play_pattern"], statsbomb_xg=round(float(r["sb_xg"]), 4),
               added_at_k=int(r["added_at_k"])) for i, r in enumerate(rows)]
    spec = dict(
        set=a.set, seed=a.seed, created=datetime.date.today().isoformat(),
        pool=f"{len(pool)} non-penalty shots with freeze frames not in the {a.n}-shot test sample "
             f"(shots.sample(n={a.n}, random_state={a.sample_seed})); {int(pool.goal.sum())} goals",
        bands_yards="pd.cut(distance, [0, 6, 12, 18, 25, 200]) -> <6, 6-12, 12-18, 18-25, 25+",
        rules=["k=1: one miss from 12-18 yd",
               "k=5: k=1 + one per other band; one goal drawn uniformly from pool goals in those bands, misses elsewhere",
               "k=10: k=5 + one more per band; one goal drawn uniformly from pool goals in bands without a goal, "
               "misses elsewhere (2 goals, 8 misses, two per band)",
               "order: k=1 example, then k=5 additions shuffled, then k=10 additions shuffled (nested prefixes)",
               "labels: actual outcome ('Outcome: GOAL' / 'Outcome: NO GOAL'), never xG"],
        k1=[e["shot_id"] for e in ex[:1]], k5=[e["shot_id"] for e in ex[:5]], k10=[e["shot_id"] for e in ex[:10]],
        examples=ex)
    assert sum(e["goal"] for e in ex[:1]) == 0 and sum(e["goal"] for e in ex[:5]) == 1
    assert sum(e["goal"] for e in ex[:10]) == 2 and not set(spec["k10"]) & test
    with open(out, "w") as f:
        json.dump(spec, f, indent=1)
    print(pd.DataFrame(ex)[["order", "added_at_k", "label", "band", "distance_yards", "body_part",
                            "statsbomb_xg", "shot_id"]].to_string(index=False))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
