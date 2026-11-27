"""
Side tasks that ask the VLMs something other than an xG. Same backends, request settings, cost log, budget guard and
safety rails as xg_vlm_pilot.py (one folder per task and model, never reused with other settings).

  identify    Contamination probe A. The zero-shot image (same rendering, same legend lines as the xG prompt).
              Does the model recognise the match? Either way: its best guess of the competition, year, teams and
              shooter, and its confidence.
  recall      Contamination probe B. Text only: no image and no positions. The match (2022 World Cup, stage,
              teams, date), the match clock and the shooter. Does the model remember the shot, was it a goal, how
              likely was a goal, and what was its xG?
  perception  Hybrid design. The zero-shot image; the model reads the distance, the angle and the number of
              opponents in the shooting triangle off the picture (no xG asked). hybrid.py scores the readings with
              the map-only logistic model.

Samples: identify and recall use the probe sample (results/probe_sample.json: every goal among compare_models.py's
common shots plus the same number of non-goals drawn with seed 2026); perception uses the 300-shot sample of the
xG runs (random_state=7).
Settings: as in the xG runs (Gemini and OpenAI: --thinking-level medium, no output cap; Anthropic: no thinking
parameter), except Anthropic's max_tokens, 1500 instead of 600 (thinking counts toward it, and these answers are
longer than an xG).
Folders: results/probe_identify_<model>, results/probe_recall_<model>, results/<model>_perception; each holds
run_config.json, calls.jsonl, vlm_cache.jsonl, answers.csv (parsed answers with the shot's facts) and summary.txt.
Run:      python probe_runs.py --task recall --backend anthropic --model claude-opus-5-5 --workers 4
          python probe_runs.py --task perception --backend gemini --model gemini-4-carbon --thinking-level medium
Analysis: contamination_probe.py, hybrid.py
"""
import os, sys, re, json, argparse, datetime, hashlib, time
import concurrent.futures as cf
import numpy as np
import pandas as pd
import xg_vlm_pilot as P

TASKS = ("identify", "recall", "perception")
TASK_LABEL = {"identify": "probe: identify the match from the image",
              "recall": "probe: recall the shot from match, clock and player (text only)",
              "perception": "perception: read distance, angle, opponents in triangle from the image"}
PROBE_SAMPLE = os.path.join(P.RESULTS, "probe_sample.json")
PROBE_SEED = 2026
ANTHROPIC_MAX_TOKENS = 1500

LEGEND = P.PROMPT.split("{context}")[0]   # the xG prompt's role line and the three lines describing the image

IDENTIFY = LEGEND + (
    "This is a real shot from a professional football match.\n"
    "Do you recognise this specific shot, that is, the match and the player taking it? Set \"recognised\" to true "
    "only if you actually recognise it.\n"
    "Either way, give your best guess of the competition, the year, the team taking the shot, the opposing team "
    "and the shooter (write \"unknown\" if you have no idea at all), and how confident you are, from 0 to 1, that "
    "you have the right match.\n"
    'Reply with JSON only: {"recognised": <true or false>, "competition": "<name>", "year": "<year>", '
    '"shooting_team": "<team>", "defending_team": "<team>", "shooter": "<player>", "confidence": <number>}')

RECALL = (
    "You are a football (soccer) analyst with a detailed memory of past matches.\n"
    "Match: 2022 FIFA World Cup, {stage}: {home} vs {away}, played on {date}.\n"
    "At {clock} on the match clock ({period}; minute {reported} in match reports), {player} of {team} took a shot "
    "(not a penalty).\n"
    "Answer from memory, without looking anything up:\n"
    '- "remembered": do you remember this specific shot? (true or false)\n'
    '- "scored": was this shot a goal? ("yes", "no" or "unknown")\n'
    '- "p_goal": your probability, between 0 and 1, that this shot was a goal\n'
    '- "xg": your estimate of the shot\'s expected goals value: the quality of the chance, regardless of the outcome\n'
    '- "note": one short sentence on what you remember about it\n'
    'Reply with JSON only: {{"remembered": <true or false>, "scored": "<yes, no or unknown>", "p_goal": <number>, '
    '"xg": <number>, "note": "<one short sentence>"}}')

PERCEPTION = LEGEND + (
    "Read the geometry of the shot off the image. For scale: the pitch is 80 yards wide, the penalty area is 18 "
    "yards deep and 44 yards wide, the six-yard box is 6 yards deep and 20 yards wide, the penalty spot is 12 yards "
    "from the goal line, and the goal is 8 yards wide.\n"
    '- "distance": the distance from the shooter to the centre of the goal, in yards\n'
    '- "angle": the angle at the shooter between the lines to the two goalposts, in degrees\n'
    '- "opponents_in_triangle": the number of opponents (red circles and the orange square) inside the triangle '
    "formed by the shooter and the two goalposts\n"
    'Reply with JSON only: {"distance": <number>, "angle": <number>, "opponents_in_triangle": <integer>}')
TEMPLATE = {"identify": IDENTIFY, "recall": RECALL, "perception": PERCEPTION}
# identify on a captured broadcast frame (--view): the frame line of xg_vlm_pilot --input frame, then the same question
IDENTIFY_FRAME = LEGEND.splitlines(keepends=True)[0] + P.FRAME_LINES + IDENTIFY[len(LEGEND):]

STAGE = {"Group Stage": "group stage", "Round of 16": "round of 16", "Quarter-finals": "quarter-final",
         "Semi-finals": "semi-final", "Final": "final", "3rd Place Final": "third-place play-off"}
PERIOD = {1: "first half", 2: "second half", 3: "first half of extra time", 4: "second half of extra time"}


# ---------------------------------------------------------------- match facts
def reported_minute(period, minute):
    """The minute as written in match reports: the clock's minute + 1 (63:32 -> '64'), and stoppage time counted
    from the end of the period (first half 46:10 -> '45+2'). Checked: Messi vs Mexico 63:32 -> 64', Richarlison vs
    Serbia 61:58 / 72:54 -> 62' / 73', Mbappe in the final 80:59 -> 81', Di Maria 35:22 -> 36'."""
    end = {1: 45, 2: 90, 3: 105, 4: 120}[int(period)]
    return f"{end}+{minute - end + 1}" if minute >= end else str(minute + 1)


def player_names(shots, comp=43, season=106):
    """StatsBomb's lineup nickname (the name used in match reports, e.g. 'Lionel Messi') where it has one, else the
    full name, and the shirt number. Cached in data_cache/ (one lineup download per match)."""
    path = os.path.join(P.DATA_CACHE, f"player_names_{comp}_{season}.json")
    if os.path.exists(path):
        with open(path) as f:
            names = json.load(f)
        if set(shots.player) <= set(names):
            return names
    from statsbombpy import sb
    names = {}
    for mid in sorted(set(shots.match_id)):
        for df in sb.lineups(match_id=int(mid)).values():
            for r in df.itertuples():
                nick = r.player_nickname if isinstance(r.player_nickname, str) and r.player_nickname else None
                names[r.player_name] = dict(name=nick or r.player_name, number=int(r.jersey_number))
    os.makedirs(P.DATA_CACHE, exist_ok=True)
    with open(path, "w") as f:
        json.dump(names, f, indent=0, ensure_ascii=False)
    return names


def probe_sample(shots):
    """Every goal among compare_models.py's common shots plus as many non-goals, drawn with PROBE_SEED; written
    once to results/probe_sample.json (in a seeded random query order) and read from there afterwards."""
    if os.path.exists(PROBE_SAMPLE):
        with open(PROBE_SAMPLE) as f:
            return json.load(f)["shot_ids"]
    import compare_models as C
    common = C.common_shots(C.load_runs(verbose=False), verbose=False)
    pool = shots.set_index("shot_id").loc[common]
    goals = sorted(pool.index[pool.goal == 1])
    rng = np.random.default_rng(PROBE_SEED)
    misses = sorted(rng.choice(sorted(pool.index[pool.goal == 0]), size=len(goals), replace=False).tolist())
    ids = [str(s) for s in rng.permutation(goals + misses)]
    spec = dict(seed=PROBE_SEED, rule="every goal among compare_models.py's common shots (the shots every included "
                "run answered) plus the same number of non-goals drawn uniformly without replacement "
                "(numpy default_rng(seed).choice on the sorted non-goal ids); listed in a seeded random query order",
                n_common=len(common), goals=len(goals), non_goals=len(misses), shot_ids=ids)
    with open(PROBE_SAMPLE, "w") as f:
        json.dump(spec, f, indent=1)
    return ids


# ---------------------------------------------------------------- requests and parsers
def scaled_b64(row, scale):
    """The zero-shot image resized by `scale` (Lanczos) and re-encoded as a new PNG (no metadata carried over)."""
    import io, base64
    from PIL import Image
    im = Image.open(io.BytesIO(base64.b64decode(P.image_b64(row))))
    im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def build_request(task, row, names, frames=None, scale=None):
    if frames:
        return P.Req(image=P.frame_b64(frames[row.shot_id]), prompt=IDENTIFY_FRAME)
    if scale:
        return P.Req(image=scaled_b64(row, scale), prompt=TEMPLATE[task])
    if task == "recall":
        who = names.get(row.player, {}).get("name", row.player)
        prompt = RECALL.format(stage=STAGE.get(row.competition_stage, row.competition_stage), home=row.home_team,
                               away=row.away_team, date=row.match_date, clock=f"{row.minute:02d}:{row.second:02d}",
                               period=PERIOD[row.period], reported=reported_minute(row.period, row.minute),
                               player=who, team=row.team)
        return P.Req(image=None, prompt=prompt)
    return P.Req(image=P.image_b64(row), prompt=TEMPLATE[task])


def extract_json(txt):
    """The first JSON object in the answer (text around it is ignored), or None."""
    txt = txt or ""
    dec, i = json.JSONDecoder(), txt.find("{")
    while i != -1:
        try:
            d, _ = dec.raw_decode(txt[i:])
            if isinstance(d, dict):
                return d
        except json.JSONDecodeError:
            pass
        i = txt.find("{", i + 1)
    return None


def to_bool(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    s = str(v).strip().lower()
    return True if s in ("true", "yes") else False if s in ("false", "no") else None


def to_num(v):
    if isinstance(v, bool):
        return None
    try:
        x = float(str(v).strip().rstrip("%"))
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def to_prob(v):
    x = to_num(v)
    if x is None or x < 0 or x > 100:
        return None
    return x / 100 if x > 1 else x   # answered as a percentage


def parse_identify(txt):
    d = extract_json(txt)
    rec = to_bool(d.get("recognised", d.get("recognized"))) if d else None
    if rec is None:
        return None
    s = lambda k: str(d.get(k) if d.get(k) is not None else "").strip()
    return dict(recognised=rec, competition=s("competition"), year=s("year"), shooting_team=s("shooting_team"),
                defending_team=s("defending_team"), shooter=s("shooter"), confidence=to_prob(d.get("confidence")))


def parse_recall(txt):
    d = extract_json(txt)
    if not d:
        return None
    pg, xg = to_prob(d.get("p_goal")), to_prob(d.get("xg"))
    if pg is None or xg is None:
        return None
    sc = str(d.get("scored", "")).strip().lower()
    return dict(remembered=to_bool(d.get("remembered")), scored=sc if sc in ("yes", "no") else "unknown",
                p_goal=pg, xg=xg, note=str(d.get("note") or "")[:400])


def parse_perception(txt):
    d = extract_json(txt)
    if not d:
        return None
    dist, ang, n = to_num(d.get("distance")), to_num(d.get("angle")), to_num(d.get("opponents_in_triangle"))
    if dist is None or ang is None or n is None or dist <= 0 or not 0 <= ang <= 180 or n < 0:
        return None
    return dict(read_distance=dist, read_angle_deg=ang, read_opponents=int(round(n)))


PARSERS = {"identify": parse_identify, "recall": parse_recall, "perception": parse_perception}
FACTS = ["shot_id", "match_id", "goal", "sb_xg", "baseline_xg", "body_part", "distance", "angle", "defenders_in_cone",
         "x", "y", "home_team", "away_team", "team", "player", "match_date", "competition_stage", "period", "minute",
         "second"]
ANSWER_FIELDS = {"identify": ["recognised", "competition", "year", "shooting_team", "defending_team", "shooter",
                              "confidence"],
                 "recall": ["remembered", "scored", "p_goal", "xg", "note"],
                 "perception": ["read_distance", "read_angle_deg", "read_opponents"]}


def summarise(task, df):
    if task == "identify":
        return [f"Claimed to recognise the shot: {int(df.recognised.sum())} of {len(df)}",
                f"Mean confidence: {pd.to_numeric(df.confidence, errors='coerce').mean():.2f}",
                "Best-guess competitions: " + "; ".join(f"{k} ({v})" for k, v in df.competition.value_counts().head(6).items())]
    if task == "recall":
        g, m = df[df.goal == 1], df[df.goal == 0]
        return [f"Claimed to remember the shot: goals {int(g.remembered.fillna(False).sum())}/{len(g)}, "
                f"non-goals {int(m.remembered.fillna(False).sum())}/{len(m)}",
                f"'scored' = yes: goals {(g.scored == 'yes').mean():.2f}, non-goals {(m.scored == 'yes').mean():.2f}; "
                f"'unknown': {(df.scored == 'unknown').mean():.2f}",
                f"Mean p_goal: goals {g.p_goal.mean():.3f}, non-goals {m.p_goal.mean():.3f}; mean xg: goals "
                f"{g.xg.mean():.3f}, non-goals {m.xg.mean():.3f} (StatsBomb {g.sb_xg.mean():.3f} / {m.sb_xg.mean():.3f})"]
    e = df.read_distance - df.distance
    a = df.read_angle_deg - np.degrees(df.angle)
    return [f"Distance read - true: mean {e.mean():+.2f} yd, mean absolute {e.abs().mean():.2f} yd",
            f"Angle read - true: mean {a.mean():+.1f} deg, mean absolute {a.abs().mean():.1f} deg",
            f"Opponents in triangle: exact {np.mean(df.read_opponents == df.defenders_in_cone):.2f}, mean read - true "
            f"{(df.read_opponents - df.defenders_in_cone).mean():+.2f}"]


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=TASKS, required=True)
    ap.add_argument("--backend", choices=[b for b in P.ASK if b != "ollama"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--thinking-level", choices=["minimal", "low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--max-tokens", type=int, help=f"output cap (default: {ANTHROPIC_MAX_TOKENS} for anthropic, none "
                                                   "for the others)")
    ap.add_argument("--n", type=int, help="only the first n shots of the task's sample (quick test)")
    ap.add_argument("--view", help="identify only: use the captured broadcast frames <frames-dir>/<shot_id>_<view>.png "
                                   "instead of the drawing (docs/frame_capture.md)")
    ap.add_argument("--frames-dir", default="data_frames")
    ap.add_argument("--scale", type=float, help="perception only: send the image resized by this factor (sanity check "
                                                "that the readings' precision is not tied to the pixel grid)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--tag", default="")
    ap.add_argument("--stage", default="")
    ap.add_argument("--budget-check-after", type=int, default=20)
    ap.add_argument("--provider-limit", type=float)
    ap.add_argument("--total-limit", type=float, default=P.TOTAL_LIMIT)
    a = ap.parse_args()

    if a.view and (a.task != "identify" or not re.fullmatch(r"[A-Za-z0-9]+", a.view)):
        sys.exit("--view works with --task identify only, and must be letters and digits")
    if a.scale and (a.task != "perception" or not 0 < a.scale < 4):
        sys.exit("--scale works with --task perception only, with a factor between 0 and 4")
    mslug = re.sub(r"[^A-Za-z0-9.]+", "_", a.model)
    slug = (f"{mslug}_perception" + (f"_scale{str(a.scale).replace('.', '')}" if a.scale else "")
            if a.task == "perception" else
            f"probe_identify_frames_{a.view}_{mslug}" if a.view else f"probe_{a.task}_{mslug}") + (f"_{a.tag}" if a.tag else "")
    out_dir = os.path.join(P.RESULTS, slug)
    cfg_path = os.path.join(out_dir, "run_config.json")
    blocked = os.path.join(P.DATA_CACHE, f"blocked_{a.backend}.txt")
    if os.path.exists(blocked):
        sys.exit(f"{a.backend} is marked as blocked after a billing error ({blocked}); delete that file to retry.")
    if os.path.isdir(out_dir) and os.listdir(out_dir) and not os.path.exists(cfg_path):
        sys.exit(f"{out_dir} exists without run_config.json; refusing to touch it")
    os.makedirs(out_dir, exist_ok=True)
    import fcntl
    os.makedirs(P.DATA_CACHE, exist_ok=True)
    run_lock = open(os.path.join(P.DATA_CACHE, f"run_{slug}.lock"), "w")
    try:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)   # held until the process exits
    except BlockingIOError:
        sys.exit(f"another process is already running {slug}")

    shots = P.add_baseline(P.load_shots_cached(43, 106))
    frames = P.find_frames(a.frames_dir, a.view) if a.view else {}
    if a.view:
        sample = shots[shots.shot_id.isin(frames)].sort_values("shot_id").copy()
        sample_desc = f"captured broadcast frames, view {a.view} ({len(sample)} shots)"
        if sample.empty:
            sys.exit(f"no frames named <shot_id>_{a.view}.png in {a.frames_dir}/")
    elif a.task == "perception":
        sample = shots.sample(n=300, random_state=7).copy()
        sample_desc = "the xG runs' 300-shot sample (random_state=7)"
    else:
        ids = probe_sample(shots)
        sample = shots.set_index("shot_id").loc[ids].reset_index()
        sample_desc = f"results/probe_sample.json ({len(ids)} shots)"
    if a.n:
        sample = sample.head(a.n)
    names = player_names(shots) if a.task == "recall" else {}

    max_out = a.max_tokens or (ANTHROPIC_MAX_TOKENS if a.backend == "anthropic" else None)
    template = IDENTIFY_FRAME if a.view else TEMPLATE[a.task]
    cfg = dict(task=a.task, backend=a.backend, model=a.model, thinking_level=a.thinking_level,
               max_output_tokens=max_out or "none", sample=sample_desc.split(" (")[0], shots=0,
               context=TASK_LABEL[a.task] + (f" (broadcast frame, {a.view})" if a.view else ""),
               prompt_sha=hashlib.sha1(template.encode()).hexdigest()[:12])
    if a.view:
        cfg.update(input="frame", view=a.view)
    if a.scale:
        cfg.update(scale=a.scale, context=cfg["context"] + f" (image resized to {a.scale}x)")
    fixed = list(cfg)
    old = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            old = json.load(f)
        diff = {k: (old.get(k), cfg[k]) for k in fixed if old.get(k) != cfg[k]}
        if diff:
            sys.exit(f"{out_dir} was created with different settings {diff}; use --tag for a new folder.")
    cfg["invocations"] = old.get("invocations", [])
    cfg.update(stage=a.stage or old.get("stage", ""), n=len(sample))
    cfg["invocations"].append(dict(started=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                                   n=len(sample), workers=a.workers, stage=a.stage))
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=1)

    cache = {}
    for d in P.read_jsonl(os.path.join(out_dir, "vlm_cache.jsonl")):
        cache.setdefault(d["shot_id"], d)
    todo = [r for r in sample.itertuples() if r.shot_id not in cache]
    price = P.price_for(a.model)
    limit = a.provider_limit if a.provider_limit is not None else P.LIMITS[a.backend]
    provider_room = limit - P.spent(a.backend, exclude_run=slug)
    total_room = a.total_limit - P.spent(exclude_run=slug)
    print(f"{slug}: {len(sample) - len(todo)} cached, {len(todo)} to query; budget left for this run "
          f"${min(provider_room, total_room):.2f} ({a.backend} ${provider_room:.2f}, total ${total_room:.2f})")
    st = P.RunState(out_dir, price, len(todo), provider_room, total_room, a.budget_check_after)
    opt = P.Opts(thinking_level=a.thinking_level, max_tokens=max_out)
    reqs = [(r.shot_id, build_request(a.task, r, names, frames, a.scale)) for r in todo]   # renders in the main thread

    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        futs = [ex.submit(P.query_shot, a.backend, a.model, opt, req, sid, st, cache, a.max_attempts,
                          PARSERS[a.task]) for sid, req in reqs]
        for i, fu in enumerate(cf.as_completed(futs), 1):
            if fu.cancelled():
                continue
            fu.result()
            st.shot_done()
            if i % 10 == 0 or i == len(futs):
                print(f"  done {i}/{len(futs)}  answered {st.answered}  failed calls {st.failed_calls}  "
                      f"cost ${st.prior_cost + st.cost:.3f}", flush=True)
            if st.stop:
                for f2 in futs:
                    f2.cancel()

    missing = [s for s in sample.shot_id if s not in cache]
    if st.stop:
        status = "stopped (budget)" if st.stop.startswith("budget") else (
            "blocked (billing)" if st.stop.startswith("billing") else "stopped (errors)")
    else:
        status = "complete" if not missing else f"partial ({len(missing)} shots unanswered)"
    cfg["invocations"][-1].update(finished=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                                  status=status, elapsed_s=round(time.time() - t0))
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=1)
    row = P.cost_row(slug, cfg, out_dir, status, st.stop or "")
    P.write_cost_row(row)
    print(f"Cost log: {row['calls']} calls, {row['failed_calls']} failed, cost ${row['cost_usd']}")
    if st.stop:
        sys.exit(3)

    facts = sample[[c for c in FACTS if c in sample.columns]].copy()
    if a.task == "recall":
        facts["player_shown"] = [names.get(p, {}).get("name", p) for p in facts.player]
    for k in ANSWER_FIELDS[a.task]:
        facts[k] = [cache[s].get(k) if s in cache else None for s in facts.shot_id]
    facts.to_csv(os.path.join(out_dir, "answers.csv"), index=False)
    done = facts[facts.shot_id.isin(cache)]
    L = [f"Task: {a.task} ({TASK_LABEL[a.task]})", f"Model: {a.backend}/{a.model}; thinking/reasoning level: "
         f"{a.thinking_level or 'not set'}; max output tokens: {cfg['max_output_tokens']}",
         f"Sample: {sample_desc}; answered {len(done)} of {len(sample)}",
         f"API usage: {row['calls']} calls, {row['failed_calls']} failed; cost ${row['cost_usd']}", ""]
    L += summarise(a.task, done) if len(done) else []
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
