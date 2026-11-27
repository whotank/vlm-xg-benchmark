# Paper notes

## Data section (draft)

The StatsBomb Open Data for the 2022 FIFA World Cup covers all 64 matches and all 32 teams, released free of charge by StatsBomb and widely used in football analytics and expected goals (xG) research.

Dataset scope and key figures
- Coverage: all 64 matches and 32 national teams, with every on-ball event. StatsBomb 360 data (positions of all players visible in the broadcast frame, for every event) are also available for this tournament as separate files.
- Shots and goals: excluding penalties (23 in-game penalties and 41 shootout kicks), there are 1,430 shots and 152 goals, a conversion rate of 10.6%. All 1,430 have a shot freeze frame. With 17 scored penalties and 3 own goals, the tournament total was 172 goals.
- Spatial distribution: shots spread well outside the penalty area, but goals concentrate centrally and close to goal: 54% of non-penalty goals came from within 12 yards, and 79% from the central strip of the penalty area (y between 30 and 50 on StatsBomb's 120 x 80 pitch).

Freeze-frame and event features
- Shot freeze frames: each shot includes a snapshot of player locations at the moment of the shot: the shooter plus the teammates, opponents and goalkeeper that StatsBomb recorded (on average 14 of the 22 players; range 3 to 20). We use these freeze frames, not the 360 data.
- Shot event fields: shot location (x, y), end location, body part, technique (e.g. normal, volley), shot type (open play, free kick, corner, penalty), outcome, and an under-pressure flag. Distance and angle to goal are not recorded; we compute them from the shot location.
- Play patterns: how the possession began: regular play, counterattack, free kick, corner, throw-in, goal kick, goalkeeper, kick-off, or other.

Accessing the dataset

| Access method | Package / source | Notes |
|---|---|---|
| Python | statsbombpy | Official Python wrapper for match, event and 360 data. |
| R | StatsBombR | Official R package for loading events, freeze frames and 360 data. |
| Raw JSON | StatsBomb Open Data on GitHub | Raw event and 360 JSON files, subject to StatsBomb's user agreement and attribution rules. |

## Discussion points

- Three comparisons with increasing information: the map-only baseline sees the same information as the VLM; the four-feature baseline adds header; StatsBomb xG also uses technique, ball height, assist type and play pattern, and was trained on a far larger proprietary dataset (we don't know whether these matches were in it). The VLM's gap to the map-only baseline is its own reading and judgment error.
- Baseline vs StatsBomb on the 300-shot sample: per-shot correlation 0.86 (Spearman 0.80); 75% of shots within 0.05 of each other; 17 shots (6%) differ by more than 0.15, almost all 6–13 yards out with 0–1 opponents in the cone, and 6 of the 8 largest disagreements were goals that StatsBomb rated higher. Geometry explains most of xG on average; the remaining room for improvement is in close-range chances.
- Our images are drawn from StatsBomb coordinates; the open data contain no broadcast video. Real broadcast frames (e.g. SoccerNet, xGHub) are future work.
