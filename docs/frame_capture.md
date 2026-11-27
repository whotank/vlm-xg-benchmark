# Broadcast-frame pilot: capturing the frames

Goal: show the models the real television picture of shots we already have as drawings, with the same prompt
(only the line describing the image changes), and compare frame xG with drawing xG, StatsBomb and the map-only
baseline on the same shots. It also checks for "cheating": told the match, the minute and the shooter, the models
know the outcome of every shot we probed (`results/contamination_probe.md`), so a frame that lets them recognise
the match can turn estimation into recall.

## What to capture

`docs/frame_capture_list.csv` lists 30 shots (8 goals, 17 matches), made by `make_frame_list.py`:

- rows 1–12: the error gallery (`results/figure_error_gallery.png`). Six headers from 7–13 yd that every frontier
  model over-rated, and six close-range goals that every frontier model under-rated.
- rows 13–30: drawn at random from the other evaluation shots (seed 2027), spread over the distance bands.

Each row has what you need to find the moment:

- `match_date`, `stage`, `home_team`, `away_team`, `half`
- `match_clock`: the broadcast clock, mm:ss. In the second half it starts at 45:00, in extra time at 90:00 and 105:00.
- `report_minute`: the minute as written in match reports (64 for 63:32, 90+5 for 94:40). In stoppage time some
  broadcasts hold the clock at 45:00 or 90:00 and show the added minutes separately; use this column then.
- the shooter: `shooter`, `shirt_number`, `shooting_team`
- what to expect, to confirm it is the right shot: `body_part`, `distance_yd`, `play_pattern`, `outcome`

## How

You need full-match replays you can legally watch, for example on FIFA+ or a broadcaster's archive (check what is
available where you are). Highlight packages show every goal but miss many of the misses.

1. Go to the half and the clock time in the list, and find the shot by the shooter and the expected body part and
   distance.
2. Pause on the frame where the foot or head meets the ball (the moment StatsBomb's freeze frame describes).
3. Take a screenshot of the video picture only: no player controls, no browser.
4. Crop away the score bar (team codes, score, clock) and any caption with a name. Do not resize or otherwise edit
   the frame. Kits, faces and the stadium will still be visible. That is expected, and the identify probe measures
   it.
5. Save the live camera's frame as `data_frames/<shot_id>_main.png`. If the first replay shows the strike from
   another angle (behind the goal, high camera), save that frame too as `data_frames/<shot_id>_replay.png`.
   JPEG also works (`.jpg`); frames are converted to PNG and scaled to at most 1280 px on the long side when sent.
6. Note each shot in `docs/frame_notes.csv` (columns `shot_id, view, found, at_strike, shooter_visible,
   goal_visible, notes`), especially when the live camera did not show the strike and you took the nearest usable
   frame.

Never commit the frames: they are broadcast footage, and `data_frames/` is in `.gitignore`. Only the shot list, the
notes and the model answers go into the repository. Expect about 2–3 minutes per shot, so 1–1.5 hours for all 30.
A partial set also works: the scripts use whichever frames exist.

## Running the models on the frames

Each command queries only the shots with a frame. The cost is roughly $0.5–1 per model for 30 frames.

```bash
python xg_vlm_pilot.py --backend gemini --model gemini-4-carbon --thinking-level medium --input frame --view main --workers 4
```
```bash
python xg_vlm_pilot.py --backend anthropic --model claude-opus-5-5 --input frame --view main --workers 4
```
```bash
python xg_vlm_pilot.py --backend openai --model gpt-6.2-astra --thinking-level medium --input frame --view main --workers 4
```

The same question as the contamination probe, on the frames (does the model recognise the match?):

```bash
python probe_runs.py --task identify --view main --backend gemini --model gemini-4-carbon --thinking-level medium
```
```bash
python probe_runs.py --task identify --view main --backend anthropic --model claude-opus-5-5
```
```bash
python probe_runs.py --task identify --view main --backend openai --model gpt-6.2-astra --thinking-level medium
```

Then:

```bash
python frames_analysis.py
```

`results/frames_pilot.md` then has, per model and view:

- frame xG next to drawing xG, StatsBomb and the map-only baseline, shot by shot, with paired differences
- headers separately: a frame shows the body part, the drawing does not
- whether the model recognised the match from the frame

Output folders are `<model>_frames_<view>`; use `--view replay` for the replay frames.

## What it can and cannot show

With 30 shots this is a pilot. It can show whether real pictures move the estimates in the directions the error
analysis predicts:

- headers rated lower once the model can see they are headers
- close-range goals with the keeper out of position rated higher

It can also show whether the models recognise the matches. It cannot rank models or test significance. If a model
recognises the match and its frame xG tracks the outcome better than StatsBomb, treat that as recall, not estimation.
