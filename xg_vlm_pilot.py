"""
Pilot: Can a vision-language model (VLM) estimate expected goals (xG) from a shot image?
Data: StatsBomb open data (default: FIFA World Cup 2022, competition 43, season 106)
Backends: anthropic | ollama | gemini
"""
import os, json, base64, re, math, argparse, warnings
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
GOAL, POST_L, POST_R = (120.0, 40.0), (120.0, 36.0), (120.0, 44.0)
OUT = "results"
IMG_DIR = os.path.join("results", "images")
CACHE = None

PROMPT = """You are a professional football (soccer) analyst.
The image shows a shot at the moment it is taken, drawn on a half pitch with the goal at the top.
Gold star = shooter. Blue circles = shooter's teammates. Red circles = opponents.
Orange square = opposing goalkeeper.
{context}Estimate the probability that this shot results in a goal (expected goals, xG),
assuming an average professional finisher.
Reply with JSON only: {{"xg": <number between 0 and 1>, "reason": "<one short sentence>"}}"""


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
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


_client = None


def ask_anthropic(img_b64, prompt, model):
    global _client
    import anthropic
    _client = _client or anthropic.Anthropic(max_retries=6)
    msg = _client.messages.create(
        model=model, max_tokens=600,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}},
            {"type": "text", "text": prompt}]}])
    return "".join(b.text for b in msg.content if b.type == "text")


def ask_ollama(img_b64, prompt, model):
    import requests
    r = requests.post("http://localhost:11434/api/generate", timeout=600,
                      json={"model": model, "prompt": prompt, "images": [img_b64],
                            "stream": False, "format": "json", "think": False,
                            "options": {"temperature": 0, "seed": 7}})
    return r.json()["response"]


def ask_gemini(img_b64, prompt, model, retries=6):
    global _client
    import time
    from google import genai
    from google.genai import types, errors
    _client = _client or genai.Client(api_key=os.environ["GEMINI_API_KEY"].strip())
    for attempt in range(retries):
        try:
            resp = _client.models.generate_content(
                model=model,
                contents=[types.Part.from_bytes(data=base64.b64decode(img_b64), mime_type="image/png"), prompt],
                config=types.GenerateContentConfig(response_mime_type="application/json"))
            return resp.text or ""
        except errors.APIError as e:
            if getattr(e, "code", None) in (429, 500, 503) and attempt < retries - 1:
                wait = min(60, 4 * 2 ** attempt)
                print(f"  busy ({e.code}), retrying in {wait}s")
                time.sleep(wait)
                continue
            raise


def parse(txt):
    m = re.search(r'"xg"\s*:\s*"?([0-9]*\.?[0-9]+)', txt or "")
    if not m:
        return np.nan, ""
    xg = float(m.group(1))
    if xg > 1:
        xg = xg / 100   # answered as a percentage
    r = re.search(r'"reason"\s*:\s*"([^"]*)', txt)
    return float(np.clip(xg, 0.001, 0.999)), (r.group(1) if r else "")


def scores(y, p):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return dict(auc=roc_auc_score(y, p), brier=brier_score_loss(y, p),
                logloss=log_loss(y, p), mean_pred=float(p.mean()))


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
    plt.xlabel("Mean predicted xG"); plt.ylabel("Observed goal rate")
    plt.legend(); plt.title("Reliability curve"); plt.tight_layout()
    plt.savefig(path, dpi=130); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp", type=int, default=43)
    ap.add_argument("--season", type=int, default=106)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--backend", choices=["anthropic", "ollama", "gemini"], default="anthropic")
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--context", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    global OUT, CACHE
    slug = re.sub(r"[^A-Za-z0-9.]+", "_", a.model) + ("_context" if a.context else "")
    OUT = os.path.join("results", slug)
    CACHE = os.path.join(OUT, "vlm_cache.jsonl")
    os.makedirs(OUT, exist_ok=True); os.makedirs(IMG_DIR, exist_ok=True)

    print("Loading StatsBomb shots...")
    shots = load_shots(a.comp, a.season)
    print(f"{len(shots)} non-penalty shots with freeze frames, {shots.goal.sum()} goals")

    feats = ["distance", "angle", "header", "defenders_in_cone"]
    shots["baseline_xg"] = cross_val_predict(
        LogisticRegression(max_iter=1000), shots[feats], shots.goal,
        groups=shots.match_id, cv=GroupKFold(n_splits=5), method="predict_proba")[:, 1]

    sample = shots.sample(n=min(a.n, len(shots)), random_state=a.seed).copy()
    cache = {}
    if os.path.exists(CACHE):
        with open(CACHE) as f:
            for line in f:
                d = json.loads(line); cache[d["shot_id"]] = d
    ask = {"anthropic": ask_anthropic, "ollama": ask_ollama, "gemini": ask_gemini}[a.backend]

    vlm_xg, reasons = [], []
    for i, row in enumerate(sample.itertuples(), 1):
        if row.shot_id not in cache:
            path = os.path.join(IMG_DIR, f"{row.shot_id}.png")
            if not os.path.exists(path):
                render(row, path)
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            ctx = f"Body part used: {row.body_part}.\n" if a.context else ""
            try:
                txt = ask(b64, PROMPT.format(context=ctx), a.model)
            except Exception as e:
                print(f"  shot {i}: error {e} (not cached; rerun to retry)")
                import time; time.sleep(5)
                vlm_xg.append(np.nan); reasons.append(""); continue
            xg, why = parse(txt)
            cache[row.shot_id] = dict(shot_id=row.shot_id, vlm_xg=xg, reason=why)
            with open(CACHE, "a") as f:
                f.write(json.dumps(cache[row.shot_id]) + "\n")
        vlm_xg.append(cache[row.shot_id]["vlm_xg"]); reasons.append(cache[row.shot_id]["reason"])
        print(f"  VLM {i}/{len(sample)}", end="\r")
    print()
    sample["vlm_xg"], sample["vlm_reason"] = vlm_xg, reasons
    sample = sample.dropna(subset=["vlm_xg"])
    sample["gap"] = sample.vlm_xg - sample.sb_xg

    L = [f"Data: StatsBomb open data, competition {a.comp}, season {a.season}",
         f"All non-penalty shots with freeze frames: {len(shots)} ({shots.goal.sum()} goals)",
         f"VLM sample: {len(sample)} shots ({sample.goal.sum()} goals), observed goal rate {sample.goal.mean():.3f}",
         f"VLM: {a.backend}/{a.model}, zero-shot, context={'body part' if a.context else 'image only'}\n",
         f"{'model':<14}{'AUC':>8}{'Brier':>9}{'LogLoss':>9}{'MeanXG':>9}"]
    for c in ["vlm_xg", "sb_xg", "baseline_xg"]:
        s = scores(sample.goal, sample[c])
        L.append(f"{c:<14}{s['auc']:>8.3f}{s['brier']:>9.4f}{s['logloss']:>9.4f}{s['mean_pred']:>9.3f}")
    d, ci = bootstrap_brier_diff(sample.goal, sample.vlm_xg, sample.sb_xg)
    L.append(f"\nBrier(VLM) - Brier(StatsBomb) = {d:+.4f}  (95% bootstrap CI {ci[0]:+.4f} to {ci[1]:+.4f})")
    d2, ci2 = bootstrap_brier_diff(sample.goal, sample.vlm_xg, sample.baseline_xg)
    L.append(f"Brier(VLM) - Brier(baseline)  = {d2:+.4f}  (95% bootstrap CI {ci2[0]:+.4f} to {ci2[1]:+.4f})")
    L.append(f"Spearman rho, VLM vs StatsBomb xG: {spearmanr(sample.vlm_xg, sample.sb_xg).correlation:.3f}")
    L.append(f"Mean gap (VLM - StatsBomb): {sample.gap.mean():+.3f}; mean absolute gap: {sample.gap.abs().mean():.3f}\n")

    sample["cone_bin"] = pd.cut(sample.defenders_in_cone, [-1, 0, 1, 2, 99], labels=["0", "1", "2", "3+"])
    sample["dist_bin"] = pd.cut(sample.distance, [0, 6, 12, 18, 25, 200], labels=["<6", "6-12", "12-18", "18-25", "25+"])
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
    with open(os.path.join(OUT, "summary.txt"), "w") as f:
        f.write(report)
    sample.drop(columns=["ff"]).to_csv(os.path.join(OUT, "shots.csv"), index=False)
    calibration_plot(sample, ["vlm_xg", "sb_xg", "baseline_xg"], os.path.join(OUT, "calibration.png"))
    print(f"\nSaved results to ./{OUT}/")


if __name__ == "__main__":
    main()
