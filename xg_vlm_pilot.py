"""
Pilot: Can a vision-language model (VLM) estimate expected goals (xG) from a shot image?

Data:   StatsBomb open data (default: FIFA World Cup 2022, competition 43, season 106)
Steps:  1) load non-penalty shots with freeze frames (cached in data_cache/ after the first download)
        2) render each freeze frame as a half-pitch image
        3) ask a VLM for a goal probability (zero-shot, or few-shot with --shots k)
        4) compare VLM vs StatsBomb xG vs a logistic baseline (distance, angle, header, defenders in cone)
        5) report AUC / Brier / log loss, bootstrap CI, calibration plot, and where the VLM disagrees

Install:  pip install statsbombpy mplsoccer scikit-learn scipy anthropic google-genai openai requests
Keys:     XG_ANTHROPIC_API_KEY (falls back to ANTHROPIC_API_KEY), GEMINI_API_KEY, OPENAI_API_KEY.
          Read from the environment only; never commit keys.
Run:      python xg_vlm_pilot.py --backend anthropic --model claude-opus-5-5 --n 20          # quick test first
          python xg_vlm_pilot.py --backend anthropic --model claude-opus-5-5 --shots 5     # few-shot, example set A
          python xg_vlm_pilot.py --backend gemini --model gemini-4-carbon --thinking-level medium
          python xg_vlm_pilot.py --backend openai --model gpt-6.2-astra --thinking-level medium
          python xg_vlm_pilot.py --backend ollama --model gemma4:26b --shots 5             # needs a GPU + Ollama
Outputs:  results/<run>/summary.txt, shots.csv, calibration.png
          results/<run>/vlm_cache.jsonl   one answer per shot, with that call's token usage and cost
          results/<run>/calls.jsonl       every API call (including failed ones), for the cost log
          results/<run>/run_config.json   settings; a rerun with different settings is refused
          results/cost_log.csv            one row per run (tokens, prices, cost), rebuilt from calls.jsonl
          results/images/*.png            rendered shots, shared across runs (not committed)
"""
import os, sys, json, base64, re, math, argparse, warnings, time, threading, hashlib, datetime, csv, pickle
import concurrent.futures as cf
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsbombpy import sb
from mplsoccer import VerticalPitch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
GOAL, POST_L, POST_R = (120.0, 40.0), (120.0, 36.0), (120.0, 44.0)  # StatsBomb pitch: 120 x 80
RESULTS = "results"
IMG_DIR = os.path.join(RESULTS, "images")      # rendered images are shared across models
COST_LOG = os.path.join(RESULTS, "cost_log.csv")
DATA_CACHE = "data_cache"                      # local, not committed: StatsBomb shots pickle, lock files

PROMPT = """You are a professional football (soccer) analyst.
The image shows a shot at the moment it is taken, drawn on a half pitch with the goal at the top.
Gold star = shooter. Blue circles = shooter's teammates. Red circles = opponents.
Orange square = opposing goalkeeper.
{context}Estimate the probability that this shot results in a goal (expected goals, xG),
assuming an average professional finisher.
Reply with JSON only: {{"xg": <number between 0 and 1>, "reason": "<one short sentence>"}}"""

# Few-shot runs add exactly one line (before the optional context line); the rest of the prompt is identical.
FEWSHOT_LINE = ("The earlier images are example shots drawn the same way, each followed by what actually happened; "
                "the last image is the shot to evaluate.\n")
OUTCOME = {1: "Outcome: GOAL", 0: "Outcome: NO GOAL"}   # labels are actual outcomes, never xG
EXAMPLE_FILES = {"A": os.path.join(RESULTS, "fewshot_examples.json"),
                 "B": os.path.join(RESULTS, "fewshot_examples_setB.json")}

# ---------------------------------------------------------------- prices (USD per 1M tokens)
# Thinking / reasoning tokens are billed as output tokens by all three providers.
# 'schedule' switches prices from a date (UTC). Models without an entry are logged without a cost.
_ANTHROPIC_SRC = "Anthropic pricing page (official): platform.claude.com/docs/en/about-claude/pricing"
PRICES = {
    "claude-opus-5-5": dict(input=4.00, cache_write=5.00, cached_input=0.20, output=20.00,
                            source=_ANTHROPIC_SRC, date="2026-11-27"),
    "claude-sonnet-5-5": dict(input=2.00, cache_write=2.50, cached_input=0.10, output=10.00,
                              source=_ANTHROPIC_SRC, date="2026-11-27"),
    "gemini-4-carbon": dict(input=2.00, cached_input=0.10, output=10.00,
                            schedule=[("2026-12-01", dict(input=4.00, cached_input=0.20, output=20.00))],
                            source="third-party (layer3labs.io, aipricing.guru, braindetox.kr via web search; Google's "
                                   "pricing page unreachable here): $2/$10 to 2026-11-30, $4/$20 from 2026-12-01; "
                                   "cached input assumed at 5% of input", date="2026-11-27"),
    "gemini-4-neon": dict(input=3.00, cached_input=0.15, output=15.00,
                          source="third-party (apidog.com, requesty.ai via web search; Google's pricing page "
                                 "unreachable here)", date="2026-11-27"),
    "gemini-4-hydrogen": dict(input=1.50, cached_input=0.075, output=7.50,
                              source="third-party (aipricing.guru, productbuilder.net via web search; Google's pricing "
                                     "page unreachable here)", date="2026-11-27"),
    "gpt-6.2-astra": dict(input=8.00, cached_input=0.40, output=40.00,
                          source="third-party prices from the project notes (web search); openai.com is unreachable "
                                 "from this environment, so unverified", date="2026-11-27"),
}
GEMINI_CACHE_STORAGE = 4.50   # USD per 1M tokens per hour, assumed (Gemini Pro-tier rate); explicit caches only

# Spending limits used by the budget guard (USD). Override with --provider-limit / --total-limit.
LIMITS = {"anthropic": 30.0, "gemini": 50.0, "openai": 30.0, "ollama": float("inf")}
TOTAL_LIMIT = 100.0


def price_for(model, when=None):
    p = PRICES.get(model)
    if p is None:
        return None
    p = dict(p)
    day = (when or datetime.datetime.now(datetime.timezone.utc)).strftime("%Y-%m-%d")
    for start, upd in p.pop("schedule", []):
        if day >= start:
            p.update(upd)
    p.setdefault("cache_write", p["input"])
    return p


def call_cost(usage, price):
    if price is None or usage is None:
        return None
    return (usage["input_tokens"] * price["input"] + usage["cache_write_tokens"] * price["cache_write"]
            + usage["cached_input_tokens"] * price["cached_input"] + usage["output_tokens"] * price["output"]) / 1e6


# ---------------------------------------------------------------- features
def geometry(x, y):
    dist = math.hypot(GOAL[0] - x, GOAL[1] - y)
    a = math.atan2(POST_L[1] - y, POST_L[0] - x)
    b = math.atan2(POST_R[1] - y, POST_R[0] - x)
    return dist, abs(b - a)


def in_triangle(p, a, b, c):
    def s(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1, d2, d3 = s(p, a, b), s(p, b, c), s(p, c, a)
    return not ((d1 < 0 or d2 < 0 or d3 < 0) and (d1 > 0 or d2 > 0 or d3 > 0))


def load_shots(comp, season):
    matches = sb.matches(competition_id=comp, season_id=season)
    rows = []
    for i, mid in enumerate(matches.match_id, 1):
        print(f"  loading match {i}/{len(matches)}", end="\r")
        ev = sb.events(match_id=mid)
        for _, r in ev[ev["type"] == "Shot"].iterrows():
            ff = r.get("shot_freeze_frame")
            if r.get("shot_type") == "Penalty" or not isinstance(ff, list):
                continue
            x, y = r["location"][:2]
            dist, angle = geometry(x, y)
            opp = [p for p in ff if not p["teammate"]]
            cone = sum(in_triangle(p["location"][:2], (x, y), POST_L, POST_R) for p in opp)
            rows.append(dict(
                match_id=mid, shot_id=r["id"], x=x, y=y, ff=ff,
                goal=int(r["shot_outcome"] == "Goal"), sb_xg=float(r["shot_statsbomb_xg"]),
                body_part=r.get("shot_body_part"), play_pattern=r.get("play_pattern"),
                distance=dist, angle=angle, header=int(r.get("shot_body_part") == "Head"),
                defenders_in_cone=cone))
    print()
    return pd.DataFrame(rows)


def load_shots_cached(comp, season, refresh=False):
    """load_shots() takes ~1 min of downloads; keep a local pickle (data_cache/ is not committed)."""
    path = os.path.join(DATA_CACHE, f"shots_{comp}_{season}.pkl")
    if os.path.exists(path) and not refresh:
        with open(path, "rb") as f:
            return pickle.load(f)
    shots = load_shots(comp, season)
    os.makedirs(DATA_CACHE, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(shots, f)
    return shots


BASELINE_FOLDS = os.path.join(RESULTS, "baseline_folds.json")


def legacy_group_folds(groups, n_splits=5):
    """GroupKFold as implemented before scikit-learn 1.6: groups sorted by size with numpy's default
    (unstable) argsort, largest first, each assigned to the currently lightest fold. Newer scikit-learn
    assigns folds differently, which silently changes the baseline; results/baseline_folds.json freezes
    the assignment used for the published runs."""
    uniq, inv = np.unique(groups, return_inverse=True)
    n_per = np.bincount(inv)
    order = np.argsort(n_per)[::-1]
    load, g2f = np.zeros(n_splits), np.zeros(len(uniq), int)
    for gi in order:
        f = int(np.argmin(load))
        load[f] += n_per[gi]
        g2f[gi] = f
    return dict(zip(uniq.tolist(), g2f.tolist()))


def add_baseline(shots, comp=43, season=106, n_splits=5):
    """Transparent baseline, out-of-fold predictions grouped by match (no leakage)."""
    feats = ["distance", "angle", "header", "defenders_in_cone"]
    m2f = None
    if os.path.exists(BASELINE_FOLDS):
        with open(BASELINE_FOLDS) as f:
            spec = json.load(f)
        if spec["comp"] == comp and spec["season"] == season and spec["n_splits"] == n_splits:
            m2f = {int(k): v for k, v in spec["folds"].items()}
    if m2f is None or not set(shots.match_id) <= set(m2f):
        m2f = legacy_group_folds(shots.match_id.values, n_splits)
    fold = shots.match_id.map(m2f).values
    cv = [(np.flatnonzero(fold != k), np.flatnonzero(fold == k)) for k in range(n_splits)]
    shots["baseline_xg"] = cross_val_predict(
        LogisticRegression(max_iter=1000), shots[feats], shots.goal, cv=cv, method="predict_proba")[:, 1]
    return shots


# ---------------------------------------------------------------- rendering
def render(row, path):
    pitch = VerticalPitch(half=True, pitch_type="statsbomb", line_color="black")
    fig, ax = pitch.draw(figsize=(5, 5))
    for p in row.ff:
        px, py = p["location"][:2]
        is_gk = (p.get("position") or {}).get("name") == "Goalkeeper"
        if p["teammate"]:
            pitch.scatter(px, py, ax=ax, s=120, c="royalblue", edgecolors="black")
        elif is_gk:
            pitch.scatter(px, py, ax=ax, s=170, c="orange", marker="s", edgecolors="black")
        else:
            pitch.scatter(px, py, ax=ax, s=120, c="red", edgecolors="black")
    pitch.scatter(row.x, row.y, ax=ax, s=320, c="gold", marker="*", edgecolors="black")
    tmp = f"{path}.{os.getpid()}.tmp.png"   # write then rename: other processes never see a partial file
    fig.savefig(tmp, dpi=110, bbox_inches="tight")
    plt.close(fig)
    os.replace(tmp, path)


def image_b64(row):
    """Render (main thread only: matplotlib is not thread-safe) and return the PNG as base64."""
    path = os.path.join(IMG_DIR, f"{row.shot_id}.png")
    if not os.path.exists(path):
        render(row, path)
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ---------------------------------------------------------------- VLM backends
@dataclass
class Req:
    image: str                                    # base64 PNG of the shot to evaluate
    prompt: str
    examples: list = field(default_factory=list)  # [(base64 PNG, label text)] shown before the shot

    @property
    def image_sha1(self):
        return hashlib.sha1(base64.b64decode(self.image)).hexdigest()[:16]


@dataclass
class Resp:
    text: str
    usage: dict          # normalised: input / cache_write / cached_input / output (incl. thinking) / thinking
    usage_raw: dict      # the provider's own usage object
    finish: str
    incomplete: bool
    model_version: str = ""


@dataclass
class Opts:
    thinking_level: str = None
    max_tokens: int = None
    num_ctx: int = None
    gemini_cache: str = None


def usage_dict(inp=0, cache_write=0, cached=0, out=0, thinking=None):
    return dict(input_tokens=int(inp or 0), cache_write_tokens=int(cache_write or 0),
                cached_input_tokens=int(cached or 0), output_tokens=int(out or 0),
                thinking_tokens=None if thinking is None else int(thinking))


_clients, _client_lock = {}, threading.Lock()


def get_client(backend):
    with _client_lock:
        if backend not in _clients:
            if backend == "anthropic":
                import anthropic
                key = (os.environ.get("XG_ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or "").strip()
                if not key:
                    sys.exit("Set XG_ANTHROPIC_API_KEY (or ANTHROPIC_API_KEY).")
                _clients[backend] = anthropic.Anthropic(api_key=key, max_retries=6)
            elif backend == "gemini":
                from google import genai
                _clients[backend] = genai.Client(api_key=os.environ["GEMINI_API_KEY"].strip())
            elif backend == "openai":
                import openai
                _clients[backend] = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"].strip(), max_retries=6)
        return _clients[backend]


def ask_anthropic(req, model, opt):
    client = get_client("anthropic")
    img = lambda b64: {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}}
    content = []
    for b64, label in req.examples:
        content += [img(b64), {"type": "text", "text": label}]
    if content:   # cache the fixed example prefix (5-minute TTL; prefixes under 512 tokens are not cached)
        content[-1]["cache_control"] = {"type": "ephemeral"}
    content += [img(req.image), {"type": "text", "text": req.prompt}]
    # Same request as the original zero-shot runs: no thinking/effort parameter unless --thinking-level is given.
    kw = dict(model=model, max_tokens=opt.max_tokens or 600, messages=[{"role": "user", "content": content}])
    if opt.thinking_level:
        kw["output_config"] = {"effort": opt.thinking_level}
    msg = client.messages.create(**kw)
    u = msg.usage
    thinking = u.output_tokens_details.thinking_tokens if getattr(u, "output_tokens_details", None) else None
    return Resp(text="".join(b.text for b in msg.content if b.type == "text"),
                usage=usage_dict(u.input_tokens, u.cache_creation_input_tokens, u.cache_read_input_tokens,
                                 u.output_tokens, thinking),
                usage_raw=u.model_dump(mode="json", exclude_none=True), finish=str(msg.stop_reason),
                incomplete=msg.stop_reason != "end_turn", model_version=msg.model)


def gemini_parts(examples):
    from google.genai import types
    parts = []
    for b64, label in examples:
        parts += [types.Part.from_bytes(data=base64.b64decode(b64), mime_type="image/png"),
                  types.Part.from_text(text=label)]
    return parts


def ask_gemini(req, model, opt):
    from google.genai import types
    client = get_client("gemini")
    image = types.Part.from_bytes(data=base64.b64decode(req.image), mime_type="image/png")
    cfg = dict(response_mime_type="application/json")   # as in the original Gemini 3.8 Flash run; no output cap
    if opt.thinking_level:
        cfg["thinking_config"] = types.ThinkingConfig(thinking_level=opt.thinking_level.upper())
    if opt.max_tokens:
        cfg["max_output_tokens"] = opt.max_tokens
    if req.examples:
        test = types.Content(role="user", parts=[image, types.Part.from_text(text=req.prompt)])
        if opt.gemini_cache:      # explicit context cache holds the example turn
            cfg["cached_content"] = opt.gemini_cache
            contents = [test]
        else:                     # same two turns, sent in full (implicit caching may still apply)
            contents = [types.Content(role="user", parts=gemini_parts(req.examples)), test]
    else:
        contents = [image, req.prompt]   # identical to the original zero-shot call
    resp = client.models.generate_content(model=model, contents=contents, config=types.GenerateContentConfig(**cfg))
    um = resp.usage_metadata
    prompt_t = um.prompt_token_count or 0
    cached = um.cached_content_token_count or 0
    thoughts = um.thoughts_token_count or 0
    cand = resp.candidates[0] if resp.candidates else None
    finish = cand.finish_reason.name if cand is not None and cand.finish_reason is not None else "NO_CANDIDATE"
    return Resp(text=resp.text or "",
                usage=usage_dict(prompt_t - cached, 0, cached, (um.candidates_token_count or 0) + thoughts, thoughts),
                usage_raw=um.model_dump(mode="json", exclude_none=True), finish=finish,
                incomplete=finish != "STOP", model_version=resp.model_version or "")


def ask_openai(req, model, opt):
    client = get_client("openai")
    img = lambda b64: {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"}
    content = []
    for b64, label in req.examples:
        content += [img(b64), {"type": "input_text", "text": label}]
    content += [img(req.image), {"type": "input_text", "text": req.prompt}]
    kw = dict(model=model, input=[{"role": "user", "content": content}],
              text={"format": {"type": "json_object"}})   # JSON mode, like the Gemini and Ollama runs
    if opt.thinking_level:
        kw["reasoning"] = {"effort": opt.thinking_level}
    if opt.max_tokens:                                     # default: no output-token cap
        kw["max_output_tokens"] = opt.max_tokens
    r = client.responses.create(**kw)
    u = r.usage
    det = u.input_tokens_details
    cached = getattr(det, "cached_tokens", 0) or 0
    cw = getattr(det, "cache_write_tokens", 0) or 0
    reasoning = getattr(u.output_tokens_details, "reasoning_tokens", None)
    finish = r.status + (f":{r.incomplete_details.reason}" if r.incomplete_details else "")
    return Resp(text=r.output_text or "", usage=usage_dict(u.input_tokens - cached - cw, cw, cached, u.output_tokens,
                                                           reasoning),
                usage_raw=u.model_dump(mode="json", exclude_none=True), finish=finish,
                incomplete=r.status != "completed", model_version=r.model)


def ask_ollama(req, model, opt):
    import requests
    options = {"temperature": 0, "seed": 7}
    if req.examples:
        # One user turn per example (image + outcome label), then the shot; images need a long context window.
        options["num_ctx"] = opt.num_ctx or 16384
        messages = [{"role": "user", "content": label, "images": [b64]} for b64, label in req.examples]
        messages.append({"role": "user", "content": req.prompt, "images": [req.image]})
        r = requests.post("http://localhost:11434/api/chat", timeout=900,
                          json={"model": model, "messages": messages, "stream": False, "format": "json",
                                "think": False, "options": options}).json()
        text = r["message"]["content"]
    else:   # identical to the original zero-shot Gemma run
        if opt.num_ctx:
            options["num_ctx"] = opt.num_ctx
        r = requests.post("http://localhost:11434/api/generate", timeout=600,
                          json={"model": model, "prompt": req.prompt, "images": [req.image],
                                "stream": False, "format": "json", "think": False, "options": options}).json()
        text = r["response"]
    finish = r.get("done_reason", "")
    raw = {k: r.get(k) for k in ("prompt_eval_count", "eval_count", "total_duration") if k in r}
    return Resp(text=text, usage=usage_dict(r.get("prompt_eval_count"), 0, 0, r.get("eval_count")), usage_raw=raw,
                finish=finish, incomplete=finish not in ("stop", ""), model_version=model)


ASK = {"anthropic": ask_anthropic, "gemini": ask_gemini, "openai": ask_openai, "ollama": ask_ollama}


class BillingError(RuntimeError):
    pass


def classify_error(e, backend):
    """'billing' (stop using the provider), 'transient' (retry) or 'fatal' (do not retry this call)."""
    msg = str(e).lower()
    code = getattr(e, "status_code", None) or getattr(e, "code", None)
    try:
        code = int(code)
    except (TypeError, ValueError):
        code = None
    if backend == "openai" and "insufficient_quota" in msg:
        return "billing"
    if any(s in msg for s in ("credit balance", "billing account", "payment required", "no credits",
                              "prepayment credits are depleted")):
        return "billing"
    if code in (402,):
        return "billing"
    if code is None or code in (408, 409, 429, 500, 502, 503, 504, 529):
        return "transient"   # includes connection errors / timeouts (no status code)
    return "fatal"


def ask_with_retries(backend, req, model, opt, retries=6):
    """The Anthropic and OpenAI SDKs already retry 429/5xx six times; this adds slower outer retries for all."""
    for attempt in range(retries):
        try:
            return ASK[backend](req, model, opt)
        except Exception as e:
            kind = classify_error(e, backend)
            if kind == "billing":
                raise BillingError(f"{type(e).__name__}: {str(e)[:300]}") from e
            if kind == "fatal" or attempt == retries - 1:
                raise
            wait = min(90, 5 * 2 ** attempt)   # 5, 10, 20, 40, 80 s
            print(f"  {backend} busy ({type(e).__name__}: {str(e)[:80]}), retrying in {wait}s")
            time.sleep(wait)


def parse(txt):
    m = re.search(r'"xg"\s*:\s*"?([0-9]*\.?[0-9]+)', txt or "")
    if not m:
        return np.nan, ""
    xg = float(m.group(1))
    if xg > 1:
        xg = xg / 100   # answered as a percentage
    r = re.search(r'"reason"\s*:\s*"([^"]*)', txt)
    return float(np.clip(xg, 0.001, 0.999)), (r.group(1) if r else "")


# ---------------------------------------------------------------- Gemini explicit context cache
def gemini_create_cache(model, examples, ttl_s):
    """Returns (cache name or None, token count, note). Gemini 4 models do not offer explicit caching (404)."""
    from google.genai import types
    try:
        c = get_client("gemini").caches.create(model=model, config=types.CreateCachedContentConfig(
            contents=[types.Content(role="user", parts=gemini_parts(examples))], ttl=f"{ttl_s}s",
            display_name=f"xg-fewshot-k{len(examples)}"))
        tokens = getattr(c.usage_metadata, "total_token_count", 0) or 0
        return c.name, tokens, f"explicit context cache {c.name} ({tokens} tokens)"
    except Exception as e:
        return None, 0, f"explicit context cache unavailable ({type(e).__name__}: {str(e)[:120]}); sent in full"


def gemini_delete_cache(name):
    try:
        get_client("gemini").caches.delete(name=name)
    except Exception as e:
        print(f"  could not delete cache {name}: {e}")


# ---------------------------------------------------------------- cost log
COST_COLS = ["date", "stage", "run", "provider", "model", "thinking_level", "k", "example_set", "context",
             "max_output_tokens", "shots_queried", "shots_answered", "calls", "failed_calls", "incomplete_flagged",
             "input_tokens", "cache_write_tokens", "cached_input_tokens", "output_tokens", "thinking_tokens",
             "price_input_per_mtok", "price_cache_write_per_mtok", "price_cached_input_per_mtok",
             "price_output_per_mtok", "price_source", "price_date", "cost_usd", "status", "notes"]


class FileLock:
    def __init__(self, name):
        os.makedirs(DATA_CACHE, exist_ok=True)
        self.path = os.path.join(DATA_CACHE, f"{name}.lock")

    def __enter__(self):
        import fcntl
        self.f = open(self.path, "w")
        fcntl.flock(self.f, fcntl.LOCK_EX)
        return self

    def __exit__(self, *a):
        import fcntl
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


def read_cost_log():
    if not os.path.exists(COST_LOG):
        return pd.DataFrame(columns=COST_COLS)
    return pd.read_csv(COST_LOG, dtype=str, keep_default_na=False)


def write_cost_row(row):
    """Insert or replace the row for row['run'] in results/cost_log.csv."""
    with FileLock("cost_log"):
        df = read_cost_log()
        df = df[df.run != row["run"]]
        df = pd.concat([df, pd.DataFrame([{c: row.get(c, "") for c in COST_COLS}])], ignore_index=True)
        df.to_csv(COST_LOG, index=False, quoting=csv.QUOTE_MINIMAL)


def spent(provider=None, exclude_run=None):
    df = read_cost_log()
    if exclude_run is not None:
        df = df[df.run != exclude_run]
    if provider is not None:
        df = df[df.provider == provider]
    return float(pd.to_numeric(df.cost_usd, errors="coerce").fillna(0).sum())


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def cost_row(run, cfg, out_dir, status, notes=""):
    calls = read_jsonl(os.path.join(out_dir, "calls.jsonl"))
    first, lines = {}, 0
    for d in read_jsonl(os.path.join(out_dir, "vlm_cache.jsonl")):
        first.setdefault(d["shot_id"], d)
        lines += 1
    answers = list(first.values())
    if lines > len(answers):
        dup = (f"{lines - len(answers)} shots were answered twice (a second process ran briefly on this folder); "
               f"the first answer is used, all calls are billed and counted")
        notes = f"{notes}; {dup}" if notes else dup
    api = [c for c in calls if c.get("kind", "call") == "call"]
    tot = lambda k: int(sum((c.get("usage") or {}).get(k) or 0 for c in calls))
    think = [(c.get("usage") or {}).get("thinking_tokens") for c in api]
    days = sorted({c["ts"][:10] for c in calls if c.get("ts")})
    price = price_for(cfg["model"])
    cost = sum(c.get("cost_usd") or 0 for c in calls) if price else None
    return dict(
        date=(days[0] if len(days) == 1 else f"{days[0]}..{days[-1]}") if days else "",
        stage=cfg.get("stage", ""), run=run, provider=cfg["backend"], model=cfg["model"],
        thinking_level=cfg.get("thinking_level") or "not set", k=cfg.get("shots", 0),
        example_set=cfg.get("example_set") if cfg.get("shots") else "", context=cfg.get("context", False),
        max_output_tokens=cfg.get("max_output_tokens"), shots_queried=len({c["shot_id"] for c in api}),
        shots_answered=len(answers), calls=len(api), failed_calls=sum(not c.get("ok") for c in api),
        incomplete_flagged=sum(bool(a.get("incomplete")) for a in answers),
        input_tokens=tot("input_tokens"), cache_write_tokens=tot("cache_write_tokens"),
        cached_input_tokens=tot("cached_input_tokens"), output_tokens=tot("output_tokens"),
        thinking_tokens=int(sum(t or 0 for t in think)) if any(t is not None for t in think) else "",
        price_input_per_mtok=price["input"] if price else "", price_cache_write_per_mtok=price["cache_write"] if price else "",
        price_cached_input_per_mtok=price["cached_input"] if price else "",
        price_output_per_mtok=price["output"] if price else "", price_source=price["source"] if price else
        ("local model, no API cost" if cfg["backend"] == "ollama" else "no price on file"),
        price_date=price["date"] if price else "", cost_usd=round(cost, 4) if cost is not None else "",
        status=status, notes=notes)


# ---------------------------------------------------------------- run state (shared by worker threads)
class RunState:
    def __init__(self, out_dir, price, n_todo, provider_room, total_room, check_after):
        self.lock = threading.Lock()
        self.cache_path = os.path.join(out_dir, "vlm_cache.jsonl")
        self.calls_path = os.path.join(out_dir, "calls.jsonl")
        self.price, self.n_todo = price, n_todo
        self.provider_room, self.total_room, self.check_after = provider_room, total_room, check_after
        prior = read_jsonl(self.calls_path)
        self.prior_cost = sum(c.get("cost_usd") or 0 for c in prior)            # earlier invocations of this run
        self.prior_shots = len({c["shot_id"] for c in prior if c.get("kind", "call") == "call"})
        self.cost, self.done, self.answered, self.failed_calls, self.consec_fail = 0.0, 0, 0, 0, 0
        self.stop, self.projection = None, None

    def halt(self, why):
        if self.stop is None:
            self.stop = why
            print(f"\n  STOPPING: {why}")

    def log_call(self, rec):
        rec["ts"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        with self.lock:
            with open(self.calls_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            self.cost += rec.get("cost_usd") or 0
            if rec.get("kind", "call") == "call":
                if rec["ok"]:
                    self.consec_fail = 0
                else:
                    self.failed_calls += 1
                    self.consec_fail += 1

    def save_answer(self, rec, cache):
        with self.lock:
            cache[rec["shot_id"]] = rec
            with open(self.cache_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            self.answered += 1

    def shot_done(self):
        with self.lock:
            self.done += 1
            if self.price is None or self.prior_shots + self.done < self.check_after:
                return
            per_shot = self.cost / self.done
            projected = self.prior_cost + self.cost + per_shot * (self.n_todo - self.done)
            self.projection = projected
            if projected > self.provider_room or projected > self.total_room:
                self.halt(f"budget: finishing this run would cost about ${projected:.2f} in total "
                          f"(${per_shot:.4f} per shot), but only ${min(self.provider_room, self.total_room):.2f} "
                          f"is left under the provider/total limits")


def query_shot(backend, model, opt, req, shot_id, st, cache, max_attempts):
    for attempt in range(1, max_attempts + 1):
        if st.stop:
            return
        t0 = time.time()
        try:
            resp = ask_with_retries(backend, req, model, opt)
        except BillingError as e:
            st.log_call(dict(kind="call", shot_id=shot_id, attempt=attempt, ok=False, error=f"BILLING {e}",
                             usage=None, cost_usd=0))
            st.halt(f"billing/no-credit error from {backend}: {e}")
            os.makedirs(DATA_CACHE, exist_ok=True)
            with open(os.path.join(DATA_CACHE, f"blocked_{backend}.txt"), "w") as f:
                f.write(f"{datetime.datetime.now().isoformat()} {e}\n")
            return
        except Exception as e:
            st.log_call(dict(kind="call", shot_id=shot_id, attempt=attempt, ok=False,
                             error=f"{type(e).__name__}: {str(e)[:300]}", usage=None, cost_usd=0,
                             latency_s=round(time.time() - t0, 1)))
            print(f"  shot {shot_id[:8]}: error {type(e).__name__}: {str(e)[:120]}")
            if st.consec_fail >= 12:
                st.halt(f"{st.consec_fail} consecutive failed calls; last: {type(e).__name__}: {str(e)[:150]}")
            time.sleep(5)
            continue
        cost = call_cost(resp.usage, st.price)
        xg, why = parse(resp.text)
        ok = not np.isnan(xg)
        st.log_call(dict(kind="call", shot_id=shot_id, attempt=attempt, ok=ok, finish=resp.finish,
                         incomplete=resp.incomplete, usage=resp.usage, cost_usd=cost,
                         latency_s=round(time.time() - t0, 1), **({} if ok else {"raw": resp.text[:500]})))
        if ok:
            st.save_answer(dict(shot_id=shot_id, vlm_xg=xg, reason=why, image_sha1=req.image_sha1,
                                raw=resp.text[:2000], finish=resp.finish,
                                incomplete=resp.incomplete, usage=resp.usage, usage_raw=resp.usage_raw,
                                cost_usd=cost, attempts=attempt, model_version=resp.model_version,
                                ts=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")), cache)
            return
        print(f"  shot {shot_id[:8]}: unparseable answer (finish={resp.finish}); attempt {attempt}/{max_attempts}")
    if st.consec_fail >= 12:
        st.halt(f"{st.consec_fail} consecutive failed calls")


# ---------------------------------------------------------------- evaluation
def scores(y, p):
    p = np.clip(np.asarray(p, float), 1e-3, 1 - 1e-3)
    y = np.asarray(y, int)
    two_classes = 0 < y.sum() < len(y)   # tiny test runs can contain no goal
    return dict(auc=roc_auc_score(y, p) if two_classes else np.nan, brier=float(np.mean((p - y) ** 2)),
                logloss=log_loss(y, p, labels=[0, 1]), mean_pred=float(p.mean()))


def bootstrap_brier_diff(y, p1, p2, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    y, p1, p2 = map(np.asarray, (y, p1, p2))
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        diffs.append(np.mean((p1[i] - y[i]) ** 2) - np.mean((p2[i] - y[i]) ** 2))
    return float(np.mean((p1 - y) ** 2) - np.mean((p2 - y) ** 2)), np.percentile(diffs, [2.5, 97.5])


def calibration_plot(df, cols, path, bins=8):
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    for c in cols:
        q = pd.qcut(df[c], q=bins, duplicates="drop")
        g = df.groupby(q, observed=True).agg(pred=(c, "mean"), obs=("goal", "mean"))
        plt.plot(g.pred, g.obs, "o-", label=c)
    plt.xlabel("Mean predicted xG")
    plt.ylabel("Observed goal rate")
    plt.legend()
    plt.title("Reliability curve")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


# ---------------------------------------------------------------- few-shot examples
def load_examples(example_set, k, shots, sample, context):
    path = EXAMPLE_FILES[example_set]
    if not os.path.exists(path):
        sys.exit(f"{path} not found; create it with make_fewshot_examples.py")
    with open(path) as f:
        spec = json.load(f)
    ex = sorted(spec["examples"], key=lambda e: e["order"])[:k]
    ids = [e["shot_id"] for e in ex]
    if f"k{k}" in spec and set(spec[f"k{k}"]) != set(ids):
        sys.exit(f"{path}: the first {k} examples do not match its recorded k={k} set")
    if set(ids) & set(sample.shot_id):
        sys.exit("few-shot examples overlap the test sample")
    rows = shots.set_index("shot_id").loc[ids].reset_index()
    out = []
    for r in rows.itertuples():
        label = OUTCOME[int(r.goal)]
        if context:   # if the test shot gets its body part, so do the examples
            label = f"Body part used: {r.body_part}. {label}"
        out.append((image_b64(r), label))
    return out, spec, ids


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp", type=int, default=43)
    ap.add_argument("--season", type=int, default=106)
    ap.add_argument("--n", type=int, default=300, help="shots sent to the VLM")
    ap.add_argument("--backend", choices=list(ASK), default="anthropic")
    ap.add_argument("--model", default="claude-sonnet-5-5",
                    help="e.g. claude-opus-5-5, gemini-4-carbon, gpt-6.2-astra, or a local tag like gemma4:26b")
    ap.add_argument("--context", action="store_true", help="also tell the VLM the body part")
    ap.add_argument("--shots", type=int, default=0, help="few-shot examples per prompt (0 = zero-shot)")
    ap.add_argument("--example-set", default="A", choices=list(EXAMPLE_FILES), help="which fixed example set")
    ap.add_argument("--thinking-level", choices=["minimal", "low", "medium", "high", "xhigh", "max"],
                    help="gemini: thinking_level; openai: reasoning effort; anthropic: output_config.effort "
                         "(leave unset for Anthropic runs to match the original Opus/Sonnet runs)")
    ap.add_argument("--max-tokens", type=int, help="output cap (default: 600 for anthropic, none for the others)")
    ap.add_argument("--num-ctx", type=int, help="ollama context window (default 16384 for few-shot runs)")
    ap.add_argument("--workers", type=int, default=1, help="parallel API calls")
    ap.add_argument("--max-attempts", type=int, default=3, help="tries per shot when an answer cannot be parsed")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag", default="", help="suffix for the output folder (e.g. rerun)")
    ap.add_argument("--stage", default="", help="label written to results/cost_log.csv")
    ap.add_argument("--budget-check-after", type=int, default=20, help="project the run's cost after this many shots")
    ap.add_argument("--provider-limit", type=float, help="USD limit for this provider (default from LIMITS)")
    ap.add_argument("--total-limit", type=float, default=TOTAL_LIMIT, help="USD limit across all providers")
    ap.add_argument("--allow-legacy-folder", action="store_true",
                    help="write into a results folder created before run_config.json existed")
    ap.add_argument("--refresh-data", action="store_true", help="re-download StatsBomb data instead of data_cache/")
    a = ap.parse_args()

    slug = re.sub(r"[^A-Za-z0-9.]+", "_", a.model) + ("_context" if a.context else "")
    if a.shots:
        slug += f"_k{a.shots}" + ("" if a.example_set == "A" else f"_set{a.example_set}")
    if a.tag:
        slug += f"_{a.tag}"
    out_dir = os.path.join(RESULTS, slug)
    cfg_path = os.path.join(out_dir, "run_config.json")
    blocked = os.path.join(DATA_CACHE, f"blocked_{a.backend}.txt")
    if os.path.exists(blocked):
        sys.exit(f"{a.backend} is marked as blocked after a billing error ({blocked}); delete that file to retry.")
    if os.path.isdir(out_dir) and not os.path.exists(cfg_path) and os.listdir(out_dir) and not a.allow_legacy_folder:
        sys.exit(f"{out_dir} exists from an earlier run without run_config.json; refusing to touch it. "
                 f"Pass --allow-legacy-folder to fill gaps in that run, or --tag to start a new folder.")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)
    import fcntl
    os.makedirs(DATA_CACHE, exist_ok=True)
    run_lock = open(os.path.join(DATA_CACHE, f"run_{slug}.lock"), "w")
    try:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)   # held until the process exits
    except BlockingIOError:
        sys.exit(f"another process is already running {slug}")

    print("Loading StatsBomb shots...")
    shots = add_baseline(load_shots_cached(a.comp, a.season, a.refresh_data), a.comp, a.season)
    print(f"{len(shots)} non-penalty shots with freeze frames, {shots.goal.sum()} goals")
    sample = shots.sample(n=min(a.n, len(shots)), random_state=a.seed).copy()

    examples, ex_spec, ex_ids = [], None, []
    if a.shots:
        examples, ex_spec, ex_ids = load_examples(a.example_set, a.shots, shots, sample, a.context)
    prompt_tmpl = PROMPT.replace("{context}", (FEWSHOT_LINE if a.shots else "") + "{context}")

    import importlib.metadata as md
    versions = {}
    for pkg in ["anthropic", "google-genai", "openai", "matplotlib", "mplsoccer", "statsbombpy"]:
        try:
            versions[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            pass
    max_out = a.max_tokens or (600 if a.backend == "anthropic" else None)
    cfg = dict(backend=a.backend, model=a.model, shots=a.shots, example_set=a.example_set if a.shots else None,
               example_ids=ex_ids, example_seed=ex_spec["seed"] if ex_spec else None, context=a.context,
               thinking_level=a.thinking_level, max_output_tokens=max_out if max_out else "none",
               seed=a.seed, comp=a.comp, season=a.season,
               prompt_sha=hashlib.sha1((prompt_tmpl + json.dumps(OUTCOME)).encode()).hexdigest()[:12])
    fixed = list(cfg)
    old = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            old = json.load(f)
        diff = {k: (old.get(k), cfg[k]) for k in fixed if old.get(k) != cfg[k]}
        if diff:
            sys.exit(f"{out_dir} was created with different settings {diff}; use --tag for a new folder.")
        cfg["invocations"] = old.get("invocations", [])
    else:
        cfg["invocations"] = []
    cfg.update(stage=a.stage or old.get("stage", ""), n=a.n)
    started = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    cfg["invocations"].append(dict(started=started, n=a.n, workers=a.workers, stage=a.stage, versions=versions))
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=1)

    cache = {}
    for d in read_jsonl(os.path.join(out_dir, "vlm_cache.jsonl")):
        cache.setdefault(d["shot_id"], d)   # if a shot was answered twice, the first answer counts
    legacy = sum("usage" not in d for d in cache.values())
    if legacy:
        notes_legacy = f"{legacy} answers predate cost logging (their tokens and cost are not in this row)"
    todo = [r for r in sample.itertuples() if r.shot_id not in cache]
    price = price_for(a.model)
    if price is None and a.backend != "ollama":
        print(f"WARNING: no price on file for {a.model}; cost is not logged and the budget guard is off.")
    limit = a.provider_limit if a.provider_limit is not None else LIMITS[a.backend]
    provider_room = limit - spent(a.backend, exclude_run=slug)
    total_room = a.total_limit - spent(exclude_run=slug)
    print(f"{len(sample) - len(todo)} shots cached, {len(todo)} to query. Budget left for this run: "
          f"${min(provider_room, total_room):.2f} ({a.backend} ${provider_room:.2f}, total ${total_room:.2f})")
    st = RunState(out_dir, price, len(todo), provider_room, total_room, a.budget_check_after)
    opt = Opts(thinking_level=a.thinking_level, max_tokens=a.max_tokens, num_ctx=a.num_ctx)
    notes = [notes_legacy] if legacy else []

    reqs = []   # render in the main thread (matplotlib is not thread-safe)
    for row in todo:
        ctx = f"Body part used: {row.body_part}.\n" if a.context else ""
        reqs.append((row.shot_id, Req(image=image_b64(row), prompt=prompt_tmpl.format(context=ctx), examples=examples)))

    cache_name, cache_note = None, ""
    if reqs and examples and a.backend == "gemini":
        cache_name, cache_tokens, note = gemini_create_cache(a.model, examples, ttl_s=4 * 3600)
        notes.append(note)
        cache_note = note
        print("  " + note)
        if cache_name:
            opt.gemini_cache = cache_name
            cache_t0 = time.time()
            st.log_call(dict(kind="cache_create", shot_id="", ok=True,
                             usage=usage_dict(cache_tokens), cost_usd=call_cost(usage_dict(cache_tokens), price)))

    t_start = time.time()
    try:
        if reqs and examples and a.backend in ("anthropic", "openai", "gemini"):
            # first call alone, so the example prefix is in the provider's cache before the parallel calls
            sid, req = reqs.pop(0)
            query_shot(a.backend, a.model, opt, req, sid, st, cache, a.max_attempts)
            st.shot_done()
        with cf.ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
            futs = [ex.submit(query_shot, a.backend, a.model, opt, req, sid, st, cache, a.max_attempts)
                    for sid, req in reqs]
            for i, fu in enumerate(cf.as_completed(futs), 1):
                if fu.cancelled():
                    continue
                fu.result()   # re-raises unexpected errors (bugs); API errors are handled in query_shot
                st.shot_done()
                if i % 10 == 0 or i == len(futs):
                    proj = f", projected run cost ${st.projection:.2f}" if st.projection else ""
                    print(f"  done {i}/{len(futs)}  answered {st.answered}  failed calls {st.failed_calls}  "
                          f"cost ${st.prior_cost + st.cost:.3f}{proj}", flush=True)
                if st.stop:
                    for f2 in futs:
                        f2.cancel()
    finally:
        if cache_name:
            hours = max((time.time() - cache_t0) / 3600, 1 / 60)
            gemini_delete_cache(cache_name)
            storage = cache_tokens * hours * GEMINI_CACHE_STORAGE / 1e6
            st.log_call(dict(kind="cache_storage", shot_id="", ok=True, usage=None, cost_usd=storage,
                             hours=round(hours, 3)))
    elapsed = time.time() - t_start

    missing = [r.shot_id for r in sample.itertuples() if r.shot_id not in cache]
    if st.stop:
        status = "stopped (budget)" if st.stop.startswith("budget") else (
            "blocked (billing)" if st.stop.startswith("billing") else "stopped (errors)")
        notes.append(st.stop)
    else:
        status = "complete" if not missing else f"partial ({len(missing)} shots unanswered)"
    cfg["invocations"][-1].update(finished=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                                  status=status, elapsed_s=round(elapsed))
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=1)
    row = cost_row(slug, cfg, out_dir, status, "; ".join(notes))
    write_cost_row(row)
    print(f"Cost log: {row['calls']} calls, {row['failed_calls']} failed, cost ${row['cost_usd']} -> {COST_LOG}")
    if st.stop:
        print(f"Run not finished ({status}); no summary written. Cached answers are kept; rerun to resume.")
        sys.exit(3)

    # ---- assemble answers
    sample["vlm_xg"] = [cache[s]["vlm_xg"] if s in cache else np.nan for s in sample.shot_id]
    sample["vlm_reason"] = [cache[s]["reason"] if s in cache else "" for s in sample.shot_id]
    sample = sample.dropna(subset=["vlm_xg"])
    sample["gap"] = sample.vlm_xg - sample.sb_xg
    answers = [cache[s] for s in sample.shot_id]
    flagged = [d for d in answers if d.get("incomplete")]

    # ---- report
    L = []
    L.append(f"Data: StatsBomb open data, competition {a.comp}, season {a.season}")
    L.append(f"All non-penalty shots with freeze frames: {len(shots)} ({shots.goal.sum()} goals)")
    L.append(f"VLM sample: {len(sample)} shots ({sample.goal.sum()} goals), "
             f"observed goal rate {sample.goal.mean():.3f}")
    mode = "zero-shot" if not a.shots else (f"few-shot k={a.shots} (example set {a.example_set}, seed "
                                            f"{ex_spec['seed']}, {EXAMPLE_FILES[a.example_set]})")
    L.append(f"VLM: {a.backend}/{a.model}, {mode}, context={'body part' if a.context else 'image only'}")
    if a.backend == "anthropic" and not a.thinking_level:
        think = ("not set: no thinking or effort parameter sent, as in the original Opus/Sonnet runs "
                 "(Opus 5.5 always thinks at its default effort, medium; thinking counts toward max_tokens)")
    else:
        think = a.thinking_level or "not set (provider default)"
    L.append(f"Settings: thinking/reasoning level = {think}; max output tokens = {cfg['max_output_tokens']}; "
             f"sampling = provider default{' (temperature 0, seed 7)' if a.backend == 'ollama' else ''}")
    if a.backend == "gemini":
        L.append("Gemini: response_mime_type=application/json" + (f"; {cache_note}" if cache_note else ""))
    if a.backend == "openai":
        L.append("OpenAI: Responses API, JSON mode (text.format=json_object)")
    if a.shots:
        L.append(f"Few-shot examples (in prompt order): {', '.join(f'{i[:8]}' for i in ex_ids)}; labels = actual "
                 f"outcome ('Outcome: GOAL' / 'Outcome: NO GOAL'), never xG")
    L.append(f"API usage, all invocations of this run: {row['calls']} calls, {row['failed_calls']} failed, "
             f"{row['incomplete_flagged']} accepted answers flagged incomplete; tokens: input {row['input_tokens']}, "
             f"cache writes {row['cache_write_tokens']}, cached input {row['cached_input_tokens']}, output "
             f"{row['output_tokens']} (of which thinking {row['thinking_tokens'] or 'n/a'})")
    L.append(f"Cost: ${row['cost_usd']} at {row['price_input_per_mtok']}/{row['price_output_per_mtok']} USD per 1M "
             f"input/output tokens ({row['price_source']}, {row['price_date']})" if price else
             f"Cost: {row['price_source']}")
    if flagged:
        L.append(f"Incomplete responses kept (answer parsed but the response was cut short): "
                 f"{', '.join(d['shot_id'][:8] + ' (' + d['finish'] + ')' for d in flagged)}")
    L.append("")
    L.append(f"{'model':<14}{'AUC':>8}{'Brier':>9}{'LogLoss':>9}{'MeanXG':>9}")
    for c in ["vlm_xg", "sb_xg", "baseline_xg"]:
        s = scores(sample.goal, sample[c])
        L.append(f"{c:<14}{s['auc']:>8.3f}{s['brier']:>9.4f}{s['logloss']:>9.4f}{s['mean_pred']:>9.3f}")
    d, ci = bootstrap_brier_diff(sample.goal, sample.vlm_xg, sample.sb_xg)
    L.append(f"\nBrier(VLM) - Brier(StatsBomb) = {d:+.4f}  (95% shot-level bootstrap CI {ci[0]:+.4f} to {ci[1]:+.4f})")
    d2, ci2 = bootstrap_brier_diff(sample.goal, sample.vlm_xg, sample.baseline_xg)
    L.append(f"Brier(VLM) - Brier(baseline)  = {d2:+.4f}  (95% shot-level bootstrap CI {ci2[0]:+.4f} to {ci2[1]:+.4f})")
    L.append("(match-clustered intervals for all runs: results/main_table.md, from compare_models.py)")
    rho = spearmanr(sample.vlm_xg, sample.sb_xg).correlation
    L.append(f"Spearman rho, VLM vs StatsBomb xG: {rho:.3f}")
    L.append(f"Mean gap (VLM - StatsBomb): {sample.gap.mean():+.3f}; "
             f"mean absolute gap: {sample.gap.abs().mean():.3f}\n")

    sample["cone_bin"] = pd.cut(sample.defenders_in_cone, [-1, 0, 1, 2, 99], labels=["0", "1", "2", "3+"])
    sample["dist_bin"] = pd.cut(sample.distance, [0, 6, 12, 18, 25, 200],
                                labels=["<6", "6-12", "12-18", "18-25", "25+"])
    for g in ["body_part", "play_pattern", "cone_bin", "dist_bin"]:
        t = sample.groupby(g, observed=True).agg(
            n=("goal", "size"), goals=("goal", "sum"), vlm=("vlm_xg", "mean"),
            statsbomb=("sb_xg", "mean"), mean_gap=("gap", "mean")).round(3)
        L.append(f"Disagreement by {g}:\n{t.to_string()}\n")

    cols = ["shot_id", "goal", "sb_xg", "vlm_xg", "gap", "body_part", "distance", "defenders_in_cone", "vlm_reason"]
    L.append("Largest VLM over-estimates:\n" + sample.nlargest(8, "gap")[cols].round(3).to_string(index=False))
    L.append("\nLargest VLM under-estimates:\n" + sample.nsmallest(8, "gap")[cols].round(3).to_string(index=False))

    report = "\n".join(L)
    print(report)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(report + "\n")
    sample.drop(columns=["ff"]).to_csv(os.path.join(out_dir, "shots.csv"), index=False)
    calibration_plot(sample, ["vlm_xg", "sb_xg", "baseline_xg"], os.path.join(out_dir, "calibration.png"))
    print(f"\nSaved results to ./{out_dir}/")


if __name__ == "__main__":
    main()
