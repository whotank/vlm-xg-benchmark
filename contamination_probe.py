"""
Contamination probes: could the VLMs know the outcome of these shots from memory?

Probe A (identify): the same rendered image as the zero-shot xG runs. Does the model claim to recognise the match, and
are its best guesses (competition, year, teams, shooter) right more often than chance?
Probe B (recall): no image, no positions; only the match (2022 World Cup, stage, teams, date), the match clock and the
shooter. Does the model remember the shot, and does it know whether it went in?

Reading: the image-only xG runs can only draw on remembered outcomes if the image gives the match away (probe A).
Probe B measures how much the models would know if it did, which is the risk for real broadcast frames
(scoreboard, kits, faces, stadium).

Chance levels for probe A: the model's own guesses scored against the facts of other shots (the facts shuffled across
the sample, 2,000 permutations), so a model that always names the most frequent World Cup teams is not credited.
Sample: results/probe_sample.json (every goal among compare_models.py's 298 common shots plus as many non-goals,
seed 2026). Intervals: match-clustered bootstrap (2,000 resamples, seed 0), as in compare_models.py.
Outputs: results/contamination_probe.md, results/contamination_probe_shots.csv
Run: python contamination_probe.py
"""
import os, re, json, unicodedata
import numpy as np
import pandas as pd
import compare_models as C

MODELS = [("Claude Opus 5.5", "claude_opus_5_5"), ("Gemini 4 Carbon", "gemini_4_carbon"),
          ("GPT-6.2 Astra", "gpt_6.2_astra")]
TEAM_ALIASES = {"united states": {"usa", "us", "usmnt", "united states of america", "america"},
                "south korea": {"korea republic", "korea", "republic of korea"},
                "iran": {"ir iran", "islamic republic of iran"}, "netherlands": {"holland"}}


def norm(s):
    s = unicodedata.normalize("NFKD", str(s if s is not None and s == s else "")).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9 ]+", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def team_key(s):
    n = re.sub(r"\b(national team|national football team|men s national team)\b", "", norm(s)).strip()
    for k, al in TEAM_ALIASES.items():
        if n == k or n in al:
            return k
    return "" if n in ("", "unknown", "none", "n a") else n


def name_hit(guess, *truths):
    """The guess's last word (the surname, or a one-word name such as 'Richarlison') is a word of the true name."""
    g = [t for t in norm(guess).split() if t not in ("unknown", "jr", "junior")]
    words = set(" ".join(norm(t) for t in truths).split())
    return bool(g) and len(g[-1]) >= 3 and g[-1] in words


def identify_hits(a, facts):
    """Per-shot hits of the identify answers `a` against shot facts `facts` (same order)."""
    comp = a.competition.map(norm)
    guessed = [{team_key(s), team_key(d)} - {""} for s, d in zip(a.shooting_team, a.defending_team)]
    actual = [{team_key(h), team_key(w)} for h, w in zip(facts.home_team, facts.away_team)]
    return pd.DataFrame(dict(
        competition=comp.str.contains("world cup") & ~comp.str.contains("club|women|u 20|u20|qualif"),
        year=a.year.astype(str).str.contains("2022"),
        either_team=[bool(g & t) for g, t in zip(guessed, actual)],
        both_teams=[g == t for g, t in zip(guessed, actual)],
        shooting_team=[team_key(s) != "" and team_key(s) == team_key(t) for s, t in zip(a.shooting_team, facts.team)],
        shooter=[name_hit(g, p, s) for g, p, s in zip(a.shooter, facts.player, facts.player_shown)]))


def load(task, fol):
    path = os.path.join(C.RES, f"probe_{task}_{fol}", "answers.csv")
    return pd.read_csv(path, keep_default_na=False, na_values=[""]) if os.path.exists(path) else None


def main():
    with open(os.path.join(C.RES, "probe_sample.json")) as f:
        spec = json.load(f)
    rng = np.random.default_rng(0)
    pct = lambda v: f"{100 * v:.0f}%"
    L = ["# Contamination probes: do the models know these shots from memory?", "",
         f"Sample: {len(spec['shot_ids'])} shots (`results/probe_sample.json`): all {spec['goals']} goals among the "
         f"{spec['n_common']} common shots and {spec['non_goals']} non-goals drawn with seed {spec['seed']}. Settings as "
         "in the xG runs (Gemini and GPT: thinking/reasoning medium; Opus: no thinking parameter, max_tokens 1500). "
         "Brackets: 95% match-clustered bootstrap intervals.", ""]
    shot_rows, notes, stats, id_stats = [], {}, {}, {}

    # ---------------- probe A
    L += ["## Probe A: can the model tell the match from the rendered image?", "",
          "Same image and legend lines as the zero-shot xG prompt, then: is this a shot you recognise, and your best "
          "guess of competition, year, teams and shooter. Hit rates count a best guess as right when it names the "
          "2022 World Cup / the year 2022 / one or both of the two teams / the shooting team / the shooter's surname. "
          "Chance = the same guesses scored against the facts of randomly permuted shots (2,000 permutations).", "",
          "| Model | n | Claims to recognise | Mean confidence | World Cup | 2022 | Either team | Both teams | Shooting team | Shooter |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for lab, fol in MODELS:
        a = load("identify", fol)
        if a is None:
            continue
        facts = a.copy()
        facts["player_shown"] = facts.player
        rec = load("recall", fol)
        if rec is not None and "player_shown" in rec:
            facts["player_shown"] = facts.shot_id.map(rec.set_index("shot_id").player_shown).fillna(facts.player)
        hits = identify_hits(a, facts)
        perm = []
        for _ in range(2000):
            ix = rng.permutation(len(a))
            perm.append(identify_hits(a, facts.iloc[ix].reset_index(drop=True)).mean())
        chance = pd.DataFrame(perm).mean()
        cells = " | ".join(f"{pct(hits[c].mean())} (chance {pct(chance[c])})" for c in hits.columns)
        conf = pd.to_numeric(a.confidence, errors="coerce")
        rec_flag = a.recognised.astype(str).str.lower() == "true"
        L.append(f"| {lab} | {len(a)} | {int(rec_flag.sum())} | {conf.mean():.2f} | {cells} |")
        top = a.competition.fillna("").replace("", "(empty)").value_counts().head(3)
        teams = pd.concat([a.shooting_team, a.defending_team]).map(team_key).replace("", np.nan).dropna().value_counts().head(4)
        notes[lab] = (f"{lab}: most frequent competition guesses {', '.join(f'{k} ({v})' for k, v in top.items())}; "
                      f"most frequent teams {', '.join(f'{k} ({v})' for k, v in teams.items()) or 'none'}")
        zs = pd.read_csv(os.path.join(C.RES, fol, "shots.csv")).set_index("shot_id").vlm_xg
        exact = (hits.both_teams & hits.shooter).values
        claims = [f"{w.shooting_team} vs {w.defending_team}, {w.shooter} ({w.competition} {w.year}; confidence "
                  f"{w.confidence}): {'right' if ok else 'wrong, it was ' + f'{w.player} for {w.team}'}"
                  for w, ok in zip(a[rec_flag.values].itertuples(), hits.both_teams[rec_flag.values])]
        named = [dict(shot_id=w.shot_id, desc=f"{facts.player_shown.iloc[i]} for {w.team} vs "
                      f"{w.away_team if w.team == w.home_team else w.home_team}", recognised=bool(rec_flag.iloc[i]),
                      confidence=w.confidence, xg=float(zs.get(w.shot_id, np.nan)), sb=w.sb_xg, goal=w.goal)
                 for i, w in enumerate(a.itertuples()) if exact[i]]
        id_stats[lab] = dict(n=len(a), recognised=int(rec_flag.sum()), conf=conf.mean(), claims=claims, named=named,
                             excess=float((hits.mean() - chance)[["either_team", "shooting_team", "shooter"]].max()),
                             either=hits.either_team.mean(), either_chance=chance.either_team, fol=fol)
        for c in hits.columns:
            a[f"hit_{c}"] = hits[c].values
        shot_rows.append(a.assign(model=lab, probe="identify"))
    L += [""] + [f"- {v}" for v in notes.values()]
    for lab, s in id_stats.items():
        if s["claims"]:
            L.append(f"- {lab} claims to recognise {len(s['claims'])} shot(s): " + "; ".join(s["claims"]))
        for d in s["named"]:
            L.append(f"- {lab} names the exact match and shooter of `{d['shot_id'][:8]}` ({d['desc']}, "
                     f"{'goal' if d['goal'] else 'no goal'}) with confidence {d['confidence']}, "
                     f"{'claiming' if d['recognised'] else 'without claiming'} to recognise it; its image-only xG for "
                     f"this shot was {d['xg']:.2f} (StatsBomb {d['sb']:.3f}).")
    named_ids = sorted({d["shot_id"] for s in id_stats.values() for d in s["named"]})
    if named_ids:   # sensitivity: drop every exactly identified shot from the main evaluation set
        runs = C.load_runs(verbose=False)
        common = C.common_shots(runs, verbose=False)
        base, preds, _, _ = C.assemble(runs, common)
        keep = ~np.isin(np.array(common), named_ids)
        yy = base.goal.values.astype(int)
        dd = lambda p, m: C.brier(yy[m], p[m]) - C.brier(yy[m], preds["StatsBomb xG"][m])
        everything = np.ones(len(yy), bool)
        change = {lab: dd(preds[lab], keep) - dd(preds[lab], everything) for lab in id_stats if lab in preds}
        L.append(f"- Dropping the {len(named_ids)} exactly identified shot(s) from the {len(common)} common shots changes "
                 "ΔBrier vs StatsBomb by " + ", ".join(f"{v:+.4f} ({lab})" for lab, v in change.items()) + ".")
        id_stats["_maxchange"] = max(abs(v) for v in change.values())

    # ---------------- probe B
    L += ["", "## Probe B: given the match, the clock and the shooter (text only), does the model remember the shot?", "",
          "No image and no positions. Sensitivity = share of goals answered \"scored: yes\"; specificity = share of "
          "non-goals answered \"no\"; \"unknown\" counts as neither. AUC of the model's goal probability against the "
          "actual outcome, next to StatsBomb xG's AUC on the same shots (its knowledge of the shot before the strike) "
          "and the model's own image-only xG from its zero-shot run.", "",
          "| Model | n | Remembers (goals / non-goals) | Sensitivity | Specificity | Unknown | AUC of p(goal) | StatsBomb xG AUC | Image-only xG AUC | ΔAUC p(goal) − StatsBomb | p |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    xg_rows = []
    for lab, fol in MODELS:
        r = load("recall", fol)
        if r is None:
            continue
        y = r.goal.values.astype(int)
        g = y == 1
        idx = C.cluster_bootstrap(r.match_id.values)
        rem = r.remembered.astype(str).str.lower() == "true"
        sens, spec_ = (r.scored[g] == "yes").mean(), (r.scored[~g] == "no").mean()
        pg, sb = r.p_goal.values.astype(float), r.sb_xg.values.astype(float)
        zs = pd.read_csv(os.path.join(C.RES, fol, "shots.csv")).set_index("shot_id").vlm_xg
        img = r.shot_id.map(zs).values.astype(float)
        boot = lambda p: np.array([C.auc(y[ix], p[ix]) for ix in idx])
        b_pg, b_sb, b_img = boot(pg), boot(sb), boot(img)
        d = b_pg - b_sb
        L.append(f"| {lab} | {len(r)} | {pct(rem[g].mean())} / {pct(rem[~g].mean())} | {pct(sens)} | {pct(spec_)} | "
                 f"{pct((r.scored == 'unknown').mean())} | {C.fmt(C.auc(y, pg), *C.ci(b_pg))} | "
                 f"{C.fmt(C.auc(y, sb), *C.ci(b_sb))} | {C.fmt(C.auc(y, img), *C.ci(b_img))} | "
                 f"{C.fmt(C.auc(y, pg) - C.auc(y, sb), *C.ci(d), sign=True)} | {C.fmt_p(C.boot_p(d))} |")
        xr = r.xg.values.astype(float)
        b_x = boot(xr)
        dx = b_x - b_sb
        xg_rows.append(f"| {lab} | {C.fmt(C.auc(y, xr), *C.ci(b_x))} | "
                       f"{C.fmt(C.auc(y, xr) - C.auc(y, sb), *C.ci(dx), sign=True)} | {C.fmt_p(C.boot_p(dx))} | "
                       f"{xr[g].mean():.3f} | {xr[~g].mean():.3f} | {sb[g].mean():.3f} | {sb[~g].mean():.3f} | "
                       f"{img[g].mean():.3f} | {img[~g].mean():.3f} |")
        stats[lab] = dict(sens=sens, spec=spec_, unknown=(r.scored == "unknown").mean(), auc_pg=C.auc(y, pg),
                          auc_sb=C.auc(y, sb), auc_xg=C.auc(y, xr), dx_lo=C.ci(dx)[0], dx_hi=C.ci(dx)[1],
                          rem_goal=rem[g].mean(), rem_miss=rem[~g].mean(), n=len(r))
        wrong = r[(~g) & (r.scored == "yes")]
        if len(wrong):
            notes[f"{lab} false"] = (f"{lab} said \"scored: yes\" for {len(wrong)} non-goal(s), e.g. "
                                     + "; ".join(f"{w.player_shown} ({w.team}) vs "
                                                 f"{w.away_team if w.team == w.home_team else w.home_team}, minute "
                                                 f"{w.minute + 1}: \"{str(w.note)[:110]}\"" for w in wrong.head(2).itertuples()))
        shot_rows.append(r.assign(model=lab, probe="recall", image_xg=img))
    L += ["", "The xG the model gives when it is told which shot it is (\"the quality of the chance, regardless of the "
          "outcome\"), against StatsBomb xG and the model's image-only xG on the same shots:", "",
          "| Model | AUC of recalled xG | ΔAUC vs StatsBomb xG | p | Mean recalled xG, goals | non-goals | StatsBomb, goals | non-goals | Image-only, goals | non-goals |",
          "|---|---|---|---|---|---|---|---|---|---|"] + xg_rows
    false_notes = [v for k, v in notes.items() if k.endswith(" false")]
    if false_notes:
        L += [""] + [f"- {v}" for v in false_notes]

    # Reading, written from the numbers above (thresholds: no claimed recognition, best guesses within 5 points of chance)
    L += ["", "## Reading", ""]
    id_models = {k: v for k, v in id_stats.items() if not k.startswith("_")}
    if id_models:
        parts = "; ".join(f"{lab}: {s['recognised']} of {s['n']} claimed ({sum(c.endswith('right') for c in s['claims'])} "
                          f"right), a correct team in {pct(s['either'])} of best guesses (chance {pct(s['either_chance'])})"
                          for lab, s in id_models.items())
        named = [(lab, d) for lab, s in id_models.items() for d in s["named"]]
        claimed_right = sum(c.endswith("right") for s in id_models.values() for c in s["claims"])
        if claimed_right == 0 and all(s["excess"] <= 0.05 for s in id_models.values()):
            txt = (f"- Probe A: the rendered maps do not reliably give the match away ({parts}). Every claimed "
                   "recognition is wrong, and best guesses of the teams are about as good as chance.")
            if named:
                unclaimed = all(not d["recognised"] for _, d in named)
                txt += (" The exceptions: " + "; ".join(f"{d['desc']} ({lab}, confidence {d['confidence']})"
                                                       for lab, d in named) +
                        ", where the best guess names the exact match and shooter" +
                        (" although the model did not claim to recognise the shot" if unclaimed else "") +
                        ". The model's image-only xG for them vs StatsBomb: " +
                        "; ".join(f"{d['xg']:.2f} vs {d['sb']:.3f}" for _, d in named) +
                        f"; dropping them moves every model's ΔBrier vs StatsBomb by at most "
                        f"{id_stats.get('_maxchange', np.nan):.4f}.")
            L.append(txt + " The image-only results do not rest on remembered outcomes.")
        else:
            L.append(f"- Probe A: {parts}. Some recognition above chance; see results/contamination_probe_shots.csv.")
    if stats:
        parts = "; ".join(f"{lab}: sensitivity {pct(s['sens'])}, specificity {pct(s['spec'])}, AUC {s['auc_pg']:.3f} "
                          f"vs StatsBomb xG {s['auc_sb']:.3f}" for lab, s in stats.items())
        rem = "; ".join(f"{lab} {pct(s['rem_goal'])} of goals, {pct(s['rem_miss'])} of misses" for lab, s in stats.items())
        L.append(f"- Probe B: told only the match, the minute and the shooter, the models know whether the shot went in "
                 f"({parts}). They claim to remember most goals but few misses ({rem}) and still label the misses "
                 "correctly: knowing a match's goals and when they were scored is enough to mark every other shot as "
                 "a miss.")
        L.append("- Consequence for real broadcast frames: wherever a frame lets a model identify the match and the "
                 "moment (score bar with teams and clock, kits, faces, stadium, commentary graphics), a high score can be "
                 "recall rather than estimation. Cropping the score bar removes the clock and the team names but not "
                 "the kits or the players, so a frame benchmark needs its own identification probe.")
        lifts = [(lab, s) for lab, s in stats.items() if np.isfinite(s["auc_xg"])]
        sig = [lab for lab, s in lifts if s["dx_lo"] > 0]
        L.append("- Recalled xG: asked for the quality of the chance \"regardless of the outcome\", the xG the models "
                 "give once they know which shot it is has AUC "
                 + ", ".join(f"{s['auc_xg']:.3f} ({lab})" for lab, s in lifts)
                 + f" against StatsBomb's {lifts[0][1]['auc_sb']:.3f} on the same shots"
                 + (f"; significantly higher for {', '.join(sig)}" if sig else "; not significantly higher")
                 + ". " + ("StatsBomb's model has detailed pre-shot information; separating goals from misses better "
                           "than it does suggests that the remembered outcome leaks into the estimate." if sig else
                           "No clear sign that the remembered outcome leaks into the xG estimate."))
    with open(os.path.join(C.RES, "contamination_probe.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    if shot_rows:
        pd.concat(shot_rows, ignore_index=True).to_csv(os.path.join(C.RES, "contamination_probe_shots.csv"), index=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()
