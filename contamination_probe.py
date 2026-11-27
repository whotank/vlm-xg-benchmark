"""
Contamination probes: could the VLMs know the outcome of these shots from memory?

Probe A (identify): the same rendered image as the zero-shot xG runs. Does the model claim to recognise the match, and
are its best guesses (competition, year, teams, shooter) right more often than chance?
Probe B (recall): no image, no positions; only the match (2022 World Cup, stage, teams, date), the match clock and the
shooter. Does the model remember the shot, and does it know whether it went in?

Reading: the image-only xG runs can only draw on remembered outcomes if the image gives the match away (probe A).
Probe B measures how much the models would know if it did, which is the risk for real broadcast frames
(scoreboard, kits, faces, stadium).

Chance for probe A: under the null the guesses are unrelated to which shot they are about, so the model's guesses are
matched to the shots' facts by a uniformly random permutation. For each criterion (a correct team, both teams, the
shooting team, the shooter) the number of hits under that null is the hit distribution of the 84 x 84 matrix "guess i
scored against the facts of shot j"; its p-value P(hits >= observed) is computed exactly from the matrix's rook
polynomial (big-integer arithmetic), or with 200,000 random permutations when a connected block of the matrix is too
large for the exact computation (the table says which). Chance rate = the mean of that matrix. Competition and year are
scored against facts that are the same for every shot (all 2022 World Cup), so permuting cannot change them: they are
reported without a chance level.
Sample: results/probe_sample.json (every goal among compare_models.py's 298 common shots plus as many non-goals,
seed 2026). Intervals: match-clustered bootstrap (2,000 resamples, seed 0), as in compare_models.py.
Outputs: results/contamination_probe.md, results/contamination_probe_shots.csv
Run: python contamination_probe.py
"""
import os, re, json, unicodedata
from fractions import Fraction
from math import comb, factorial
import numpy as np
import pandas as pd
import compare_models as C

MODELS = [("Claude Opus 5.5", "claude_opus_5_5"), ("Gemini 4 Carbon", "gemini_4_carbon"),
          ("GPT-6.2 Astra", "gpt_6.2_astra")]
TEAM_ALIASES = {"united states": {"usa", "us", "usmnt", "united states of america", "america"},
                "south korea": {"korea republic", "korea", "republic of korea"},
                "iran": {"ir iran", "islamic republic of iran"}, "netherlands": {"holland"}}
CRITERIA = ["either_team", "both_teams", "shooting_team", "shooter"]
CRIT_LABEL = {"either_team": "a correct team", "both_teams": "both teams", "shooting_team": "the shooting team",
              "shooter": "the shooter"}
IDENT = ["both_teams", "shooter"]   # hits that identify the match or the player, not just a frequently named team
MAX_EXACT_COLS, N_MC = 16, 200_000


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


def hit_matrices(a, facts):
    """H[c][i, j] = guess i scored against the facts of shot j (the diagonal holds the observed hits)."""
    n = len(a)
    H = {c: np.zeros((n, n), bool) for c in CRITERIA}
    for j in range(n):
        h = identify_hits(a, facts.iloc[[j] * n].reset_index(drop=True))
        for c in CRITERIA:
            H[c][:, j] = h[c].values
    return H


def rook_numbers(H, max_cols=MAX_EXACT_COLS):
    """r_k = number of ways to place k non-attacking rooks on the ones of H, as the product of the rook polynomials of
    H's connected blocks (each by dynamic programming over its rows with a bitmask of used columns). None if a block
    has more than max_cols columns."""
    nr, nc = H.shape
    parent = list(range(nr + nc))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    ones = list(zip(*np.nonzero(H)))
    for i, j in ones:
        ra, rb = find(int(i)), find(nr + int(j))
        if ra != rb:
            parent[ra] = rb
    blocks = {}
    for i, j in ones:
        rows, cols = blocks.setdefault(find(int(i)), (set(), set()))
        rows.add(int(i))
        cols.add(int(j))
    poly = [1]
    for rows, cols in blocks.values():
        if len(cols) > max_cols:
            return None
        bit = {c: 1 << k for k, c in enumerate(sorted(cols))}
        dp = {0: 1}
        for i in sorted(rows):
            cand = [bit[int(j)] for j in np.nonzero(H[i])[0]]
            new = dict(dp)
            for mask, cnt in dp.items():
                for b in cand:
                    if not mask & b:
                        new[mask | b] = new.get(mask | b, 0) + cnt
            dp = new
        rk = [0] * (len(cols) + 1)
        for mask, cnt in dp.items():
            rk[bin(mask).count("1")] += cnt
        out = [0] * (len(poly) + len(rk) - 1)
        for x, u in enumerate(poly):
            for z, v in enumerate(rk):
                out[x + z] += u * v
        poly = out
    return poly


def perm_pvalue(H, rng):
    """P(hits >= observed hits) when guesses are matched to shots by a uniformly random permutation.
    Exact from the rook polynomial when possible (hit numbers N_m = sum_j (-1)^(j-m) C(j,m) r_j (n-j)!),
    otherwise Monte Carlo. Returns (p, method)."""
    n, obs = H.shape[0], int(np.trace(H.astype(int)))
    if obs == 0:
        return 1.0, "exact"
    r = rook_numbers(H)
    if r is not None:
        r = r + [0] * (n + 1 - len(r))
        hits = [sum((-1) ** (j - m) * comb(j, m) * r[j] * factorial(n - j) for j in range(m, n + 1)) for m in range(n + 1)]
        assert sum(hits) == factorial(n) and min(hits) >= 0
        return float(Fraction(sum(hits[obs:]), factorial(n))), "exact"
    stats = []
    for _ in range(N_MC // 20_000):
        perms = np.argsort(rng.random((20_000, n)), axis=1)
        stats.append(H[np.arange(n), perms].sum(axis=1))
    stats = np.concatenate(stats)
    return float((1 + np.sum(stats >= obs)) / (1 + len(stats))), f"Monte Carlo ({len(stats):,} permutations)"


def load(task, fol):
    path = os.path.join(C.RES, f"probe_{task}_{fol}", "answers.csv")
    return pd.read_csv(path, keep_default_na=False, na_values=[""]) if os.path.exists(path) else None


def opponent(w):
    return w.away_team if w.team == w.home_team else w.home_team


def main():
    with open(os.path.join(C.RES, "probe_sample.json")) as f:
        spec = json.load(f)
    rng = np.random.default_rng(0)
    pct = lambda v: f"{100 * v:.0f}%"
    fp = lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"
    L = ["# Contamination probes: do the models know these shots from memory?", "",
         f"Sample: {len(spec['shot_ids'])} shots (`results/probe_sample.json`): all {spec['goals']} goals among the "
         f"{spec['n_common']} common shots and {spec['non_goals']} non-goals drawn with seed {spec['seed']}. Settings as "
         "in the xG runs (Gemini and GPT: thinking/reasoning medium; Opus: no thinking parameter, max_tokens 1500). "
         "Brackets: 95% match-clustered bootstrap intervals.", ""]
    shot_rows, notes, stats, id_stats = [], {}, {}, {}

    # ---------------- probe A
    L += ["## Probe A: can the model tell the match from the rendered image?", "",
          "Same image and legend lines as the zero-shot xG prompt, then: is this a shot you recognise, and your best "
          "guess of competition, year, teams and shooter. A best guess is right when it names the 2022 World Cup / the "
          "year 2022 / one or both of the two teams / the shooting team / the shooter's surname. Chance and p: the "
          "model's own guesses matched to the shots by a random permutation (hit distribution exact from the rook "
          "polynomial of the guess-by-shot hit matrix, or 200,000 permutations where marked); p = P(at least the "
          "observed hits). World Cup and 2022 have no chance level: every shot is from the 2022 World Cup, so "
          "permuting cannot change them.", "",
          "| Model | n | Claims to recognise (right) | Mean confidence | World Cup | 2022 | "
          + " | ".join(CRIT_LABEL[c].capitalize() + " (chance; p)" for c in CRITERIA) + " |",
          "|---|---|---|---|---|---|" + "---|" * len(CRITERIA)]
    methods = set()
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
        H = hit_matrices(a, facts)
        assert all((np.diag(H[c]) == hits[c].values).all() for c in CRITERIA)
        pv = {}
        for c in CRITERIA:
            pv[c] = perm_pvalue(H[c], rng)
            methods.add(pv[c][1])
        chance = {c: H[c].mean() for c in CRITERIA}
        conf = pd.to_numeric(a.confidence, errors="coerce")
        rec_flag = a.recognised.astype(str).str.lower() == "true"
        right = int((rec_flag & hits.both_teams).sum())
        cells = " | ".join(f"{int(hits[c].sum())} = {pct(hits[c].mean())} ({pct(chance[c])}; "
                           f"{fp(pv[c][0])}{'' if pv[c][1] == 'exact' else '*'})" for c in CRITERIA)
        L.append(f"| {lab} | {len(a)} | {int(rec_flag.sum())} ({right}) | {conf.mean():.2f} | "
                 f"{pct(hits.competition.mean())} | {pct(hits.year.mean())} | {cells} |")
        top = a.competition.fillna("").replace("", "(empty)").value_counts().head(3)
        teams = pd.concat([a.shooting_team, a.defending_team]).map(team_key).replace("", np.nan).dropna().value_counts().head(4)
        notes[lab] = (f"{lab}: most frequent competition guesses {', '.join(f'{k} ({v})' for k, v in top.items())}; "
                      f"most frequent teams {', '.join(f'{k} ({v})' for k, v in teams.items()) or 'none'}")
        zs = pd.read_csv(os.path.join(C.RES, fol, "shots.csv")).set_index("shot_id").vlm_xg
        claims = [f"{w.shooting_team} vs {w.defending_team}, {w.shooter} ({w.competition} {w.year}; confidence "
                  f"{w.confidence}): {'right' if ok else f'wrong, it was {w.player_shown} for {w.team} vs {opponent(w)}'}"
                  for w, ok in zip(facts[rec_flag.values].assign(shooting_team=a.shooting_team[rec_flag.values],
                                                                  defending_team=a.defending_team[rec_flag.values],
                                                                  shooter=a.shooter[rec_flag.values]).itertuples(),
                                   hits.both_teams[rec_flag.values])]
        kinds = lambda i: [CRIT_LABEL[c] for c in CRITERIA if hits[c].iloc[i]
                           and not (c == "either_team" and hits.both_teams.iloc[i])]
        hit_shots = [dict(shot_id=w.shot_id, desc=f"{facts.player_shown.iloc[i]} for {w.team} vs {opponent(w)}",
                          player=facts.player_shown.iloc[i], kinds=kinds(i), recognised=bool(rec_flag.iloc[i]),
                          guess=f"{a.shooting_team.iloc[i]} vs {a.defending_team.iloc[i]}, {a.shooter.iloc[i]}",
                          confidence=w.confidence, xg=float(zs.get(w.shot_id, np.nan)), sb=w.sb_xg, goal=w.goal)
                     for i, w in enumerate(a.itertuples()) if hits[IDENT].iloc[i].any()]
        id_stats[lab] = dict(n=len(a), recognised=int(rec_flag.sum()), right=right, claims=claims, hit_shots=hit_shots,
                             p={c: pv[c][0] for c in CRITERIA}, rate={c: hits[c].mean() for c in CRITERIA},
                             count={c: int(hits[c].sum()) for c in CRITERIA}, chance=chance,
                             ident_ids={w.shot_id for i, w in enumerate(a.itertuples())
                                        if rec_flag.iloc[i] or hits[IDENT].iloc[i].any()},
                             wide_ids={w.shot_id for i, w in enumerate(a.itertuples())
                                       if rec_flag.iloc[i] or hits[CRITERIA].iloc[i].any()})
        for c in hits.columns:
            a[f"hit_{c}"] = hits[c].values
        shot_rows.append(a.assign(model=lab, probe="identify"))
    if any(m != "exact" for m in methods):
        L += ["", "\\* Monte Carlo p-value (200,000 permutations): a block of that hit matrix was too large for the exact "
              "computation."]
    L += [""] + [f"- {v}" for v in notes.values()]
    for lab, s in id_stats.items():
        if s["claims"]:
            L.append(f"- {lab} claims to recognise {len(s['claims'])} shot(s): " + "; ".join(s["claims"]))
        for d in s["hit_shots"]:
            L.append(f"- {lab} names {' and '.join(d['kinds'])} of `{d['shot_id'][:8]}` ({d['desc']}, "
                     f"{'goal' if d['goal'] else 'no goal'}; guessed \"{d['guess']}\", confidence {d['confidence']}, "
                     f"{'claiming' if d['recognised'] else 'not claiming'} to recognise it); its image-only xG for this "
                     f"shot was {d['xg']:.2f} (StatsBomb {d['sb']:.3f}).")
    # sensitivity: drop the shots a model may have identified from the main evaluation set
    sets = {"every shot with a claim or a hit on both teams or the shooter":
            sorted(set().union(*[s["ident_ids"] for s in id_stats.values()])) if id_stats else [],
            "every shot with a claim or any hit (including a single correct team)":
            sorted(set().union(*[s["wide_ids"] for s in id_stats.values()])) if id_stats else []}
    sens = {}
    if any(sets.values()):
        runs = C.load_runs(verbose=False)
        common = C.common_shots(runs, verbose=False)
        base, preds, _, _ = C.assemble(runs, common)
        yy = base.goal.values.astype(int)
        dd = lambda p, m: C.brier(yy[m], p[m]) - C.brier(yy[m], preds["StatsBomb xG"][m])
        everything = np.ones(len(yy), bool)
        for name, ids in sets.items():
            keep = ~np.isin(np.array(common), ids)
            change = {lab: dd(preds[lab], keep) - dd(preds[lab], everything) for lab in id_stats if lab in preds}
            sens[name] = (len(ids), int((~keep).sum()), max(abs(v) for v in change.values()))
            L.append(f"- Dropping {name} ({len(ids)} shots across the probed models) from the {len(common)} common "
                     "shots changes ΔBrier vs StatsBomb by " + ", ".join(f"{v:+.4f} ({lab})".replace("-", "−")
                                                                        for lab, v in change.items()) + ".")

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
        sens_, spec_ = (r.scored[g] == "yes").mean(), (r.scored[~g] == "no").mean()
        pg, sb = r.p_goal.values.astype(float), r.sb_xg.values.astype(float)
        zs = pd.read_csv(os.path.join(C.RES, fol, "shots.csv")).set_index("shot_id").vlm_xg
        img = r.shot_id.map(zs).values.astype(float)
        boot = lambda p: np.array([C.auc(y[ix], p[ix]) for ix in idx])
        b_pg, b_sb, b_img = boot(pg), boot(sb), boot(img)
        d = b_pg - b_sb
        L.append(f"| {lab} | {len(r)} | {pct(rem[g].mean())} / {pct(rem[~g].mean())} | {pct(sens_)} | {pct(spec_)} | "
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
        stats[lab] = dict(sens=sens_, spec=spec_, unknown=(r.scored == "unknown").mean(), auc_pg=C.auc(y, pg),
                          auc_sb=C.auc(y, sb), auc_xg=C.auc(y, xr), dx_lo=C.ci(dx)[0], dx_hi=C.ci(dx)[1],
                          rem_goal=rem[g].mean(), rem_miss=rem[~g].mean(), n=len(r))
        wrong = r[(~g) & (r.scored == "yes")]
        if len(wrong):
            notes[f"{lab} false"] = (f"{lab} said \"scored: yes\" for {len(wrong)} non-goal(s), e.g. "
                                     + "; ".join(f"{w.player_shown} ({w.team}) vs {opponent(w)}, minute "
                                                 f"{w.minute + 1}: \"{str(w.note)[:110]}\"" for w in wrong.head(2).itertuples()))
        shot_rows.append(r.assign(model=lab, probe="recall", image_xg=img))
    L += ["", "The xG the model gives when it is told which shot it is (\"the quality of the chance, regardless of the "
          "outcome\"), against StatsBomb xG and the model's image-only xG on the same shots:", "",
          "| Model | AUC of recalled xG | ΔAUC vs StatsBomb xG | p | Mean recalled xG, goals | non-goals | StatsBomb, goals | non-goals | Image-only, goals | non-goals |",
          "|---|---|---|---|---|---|---|---|---|---|"] + xg_rows
    false_notes = [v for k, v in notes.items() if k.endswith(" false")]
    if false_notes:
        L += [""] + [f"- {v}" for v in false_notes]

    # ---------------- reading, written from the numbers above
    L += ["", "## Reading", ""]
    if id_stats:
        parts = []
        for lab, s in id_stats.items():
            above = [c for c in CRITERIA if s["p"][c] < 0.05]
            parts.append(f"{lab}: {s['recognised']} claimed ({s['right']} right)"
                         + (", best guesses above chance for " + ", ".join(
                             f"{CRIT_LABEL[c]} ({s['count'][c]} of {s['n']}, chance {pct(s['chance'][c])}, "
                             f"p = {fp(s['p'][c])})" for c in above) if above else ", best guesses at chance"))
        hit_list = [(lab, d) for lab, s in id_stats.items() for d in s["hit_shots"]]
        any_above = any(s["p"][c] < 0.05 for s in id_stats.values() for c in CRITERIA)
        txt = f"- Probe A: {'; '.join(parts)}."
        if hit_list:
            txt += (" The shots behind the identification hits: " + "; ".join(
                f"{d['desc']} ({lab}: {' and '.join(d['kinds'])}; {'goal' if d['goal'] else 'no goal'}; "
                f"image-only xG {d['xg']:.2f} vs StatsBomb {d['sb']:.3f})" for lab, d in hit_list) + ".")
        if sens:
            txt += (" Dropping " + "; dropping ".join(f"{name} ({v[0]} shots) moves every probed model's ΔBrier vs "
                                                     f"StatsBomb by at most {v[2]:.4f}" for name, v in sens.items()) + ".")
        sig_players = sorted({d["player"] for lab, d in hit_list if any(id_stats[lab]["p"][c] < 0.05 for c in IDENT)})
        no_claim = all(not d["recognised"] for _, d in hit_list)
        max_move = max(v[2] for v in sens.values()) if sens else np.nan
        txt += (f" So the rendered maps let a model place a few shots at better than chance (here {', '.join(sig_players)})"
                + (", at low confidence and without claiming to recognise them," if no_claim else ",")
                + f" but not the matches in general; leaving those shots out moves the image-only results by at most "
                  f"{max_move:.4f} Brier." if any_above else
                " The rendered maps do not give the match away, and the image-only results do not rest on remembered "
                "outcomes.")
        L.append(txt)
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
