# Research Collaboration Brief
## Vision-Language Models for Expected Goals: estimating xG from broadcast footage and characterizing where it diverges from event-based models

**Prepared for:** PI planning meeting
**Team (proposed):** Hootan (co-PI, Northeastern University) · [Co-PI — name / institution] · [Graduate research assistant — to be hired]
**Status:** Draft v1 for discussion · Date: ___________

---

### 1. Summary

We propose to build a multimodal / vision-language pipeline that estimates expected goals (xG) directly from shot imagery and video, and then to systematically compare those estimates against established event-based xG in order to characterize *where and why* the two diverge. The scientifically valuable outcome is not "our number differs from theirs," but a defensible demonstration that a vision/VLM model can capture shot-quality signal that conventional event-based xG misses. We plan a two-tier publication: a lean, focused paper at the **CVsports workshop at CVPR 2027** (Seattle, June 2027), followed by an extended full version at a journal (recommended: **Computer Vision and Image Understanding**, the venue where the closest prior work was published). The effort is scoped to one graduate student over roughly one year under joint supervision.

---

### 2. Motivation and opportunity

Expected goals is the central shot-quality metric in football analytics, but it is normally computed from hand-tagged event data plus, at best, freeze-frame player positions. Two things have changed. First, computer vision can now recover player and ball positions, poses, and game state directly from a single broadcast feed — the SoccerNet "Game State Reconstruction" benchmark and codebase have made this a standard, actively developed task. Second, vision-language and multimodal models (e.g., open-weight Gemma 4, released 2026, with native detection and reasoning) can extract *semantic* context from a scene that coordinates alone do not capture.

The closest existing work is **xGHub / "Once Upon a Goal"** (Computer Vision and Image Understanding, 2026), which builds an open dataset of shots enriched with pose and body-orientation features from broadcast video and feeds them into a standard xG framework. This both validates the direction and defines the bar: our contribution must be clearly distinct from a pose-and-orientation-to-xG pipeline. Our differentiation is (a) using VLMs/multimodal models for semantic and contextual features (and, as a stretch, for reasoning/explanation), and (b) framing the core contribution as a rigorous *gap analysis* against reference xG rather than as one more shot-quality regressor.

---

### 3. Research questions and intended contributions

1. **Can a multimodal / VLM pipeline estimate xG from shot footage** at least as well as event-based xG, using features (semantic context, pose, occlusion, goalkeeper state) extracted or reasoned from the image rather than hand-tagged?
2. **Where does vision-derived xG diverge from established event-based xG,** and are those divergences noise, artifacts, or genuine new signal?
3. **(Stretch) Can the VLM produce faithful natural-language explanations** of a shot's xG that reflect the model's actual drivers?

**Intended contributions:** (i) a multimodal vision→xG pipeline that goes beyond geometry/pose-only approaches; (ii) a calibrated, validated comparison of vision-derived xG against reference xG across shots and leagues; (iii) a systematic characterization of the gaps, with a defensible claim of new signal; and (iv) optionally, an explainability component with a faithfulness evaluation.

---

### 4. Approach (pipeline stages)

1. **Perception** — detection, tracking, jersey/team assignment, and camera calibration/homography to recover player and ball positions on a 2D pitch from broadcast clips (building on SoccerNet GSR tooling and the Roboflow *Supervision* library).
2. **Feature extraction** — geometric features (distance, angle), plus VLM/multimodal features: pose and body orientation, defender occlusion of the goal, goalkeeper position/set, and semantic/contextual cues that event data lacks.
3. **xG estimation** — a calibrated model producing a per-shot goal probability, with uncertainty where feasible.
4. **Gap analysis** — a pre-registered comparison protocol between vision-derived xG and a reference event-based xG, including calibration diagnostics, error propagation from perception noise, and stratified analysis of where the two disagree.

---

### 5. Data, tools, and compute

- **Data:** StatsBomb open data (events + freeze frames + shot context) as the primary reference and label source; broadcast shot clips / SoccerNet data for the vision side. Decide the starting competition/league by data coverage.
- **Reference xG:** choose one source (e.g., StatsBomb xG) as the comparison baseline — this choice materially shapes how "gaps" are interpreted (see Risks).
- **Tooling:** Python; PyTorch; Roboflow *Supervision* (detection/tracking/annotation); SoccerNet `sn-gamestate`; `socceraction`/`statsbombpy`/`mplsoccer` for xG and data handling.
- **Models:** open-weight VLMs (Gemma 4) that can run locally, with hosted VLMs used as comparison points.
- **Compute:** VLM inference and any fine-tuning need GPU access — decide local vs. cloud and set a budget.

---

### 6. Publication plan (two-tier)

**Tier 1 — Conference (lean, first out): CVsports workshop @ CVPR 2027**, Seattle, 20–25 June 2027. CVsports is the natural home for this topic and is CVF/IEEE-indexed (Google Scholar / Scopus / DBLP). Content: the pipeline plus the core gap-analysis result on one competition.

- Workshop paper deadline: **to be confirmed** when the CVsports 2027 call posts (historically ~2–3 months before the conference, i.e., roughly March 2027).
- *Alternative (stretch, not recommended for this timeline):* the CVPR **main track**, deadline **16 November 2026**, ~25% acceptance. With the student not yet hired, this is not realistic; noted for completeness.

**Tier 2 — Journal (full version): Computer Vision and Image Understanding (CVIU).** Elsevier, indexed; it published the closest prior work (xGHub), so the extension story is natural and the audience is right. Content: additional leagues/data sources, ablations over the vision and VLM components, error-propagation and calibration analysis, and the deeper "what the gaps mean" study. *Sports-analytics alternatives* if we prefer that audience: Journal of Quantitative Analysis in Sports, or Frontiers in Sports and Active Living.

**Sequencing rule (important).** Per CVPR's author guidelines, an extended version of a CVPR paper *with sufficiently new material* may be submitted to a journal any time after the CVPR submission deadline. What is prohibited is having two *substantially similar* papers in review at once — CVPR defines "substantially similar" as ~20% or more content overlap, and its dual-submission review window for CVPR 2027 is **16 Nov 2026 – 22 Feb 2027**. Practically: submit the workshop paper first; make the CVIU version a genuine extension (well beyond the 20% overlap line) that **cites** the workshop paper; and confirm whether the CVsports edition is archival (recent editions have been), since that determines how much extension is required.

*(Deadline and policy facts above are from the CVPR 2027 Call for Papers and CVPR author guidelines; confirm the CVsports workshop date when its call is posted.)*

---

### 7. Timeline (anchored to a ~March 2027 workshop deadline)

| Window | Milestone | Lead |
|---|---|---|
| Now – early Oct 2026 | Finalize scope & novelty line vs. xGHub; confirm data/compute; **hire and onboard student** | Both PIs |
| Oct 2026 | Assemble datasets; reproduce a baseline event-based xG; stand up perception (detection/tracking/homography → positions) | Student + CV lead |
| Nov – Dec 2026 | Integrate VLM/multimodal features; first vision-derived xG on one league; define the reference-xG comparison protocol | Student + both |
| Jan 2027 | Calibration & validation; initial gap analysis; perception-error propagation checks | Student + xG lead |
| Feb 2027 | Ablations, results, figures; draft the CVsports paper | Student + both |
| ~March 2027 (confirm) | Finalize and submit CVsports workshop paper | Both PIs |
| After the CVPR deadline (policy-permitted) | Begin journal extension: add leagues/vendors, deeper bias/uncertainty analysis | Student + both |
| 20–25 June 2027 | Present at CVPR / CVsports (Seattle) | Presenting author |
| ~Summer–Fall 2027 | Submit full version to CVIU | Both PIs |

Internal deadlines should sit ~2 weeks ahead of each external one.

---

### 8. Team, roles, and staffing

**Proposed division of labor (to confirm in the meeting):**

| Person | Role | Primary responsibilities |
|---|---|---|
| Hootan (co-PI, Northeastern) | xG modeling & evaluation lead; sports-analytics domain | Reference-xG baselines, calibration/validation, gap-analysis design and interpretation, evaluation protocol, sports-analytics framing, co-writing |
| [Co-PI] | Vision/CV & multimodal lead; **primary student advisor (hiring)** | Perception pipeline and VLM/multimodal integration, compute, day-to-day supervision, co-writing |
| [Graduate RA — to be hired] | Primary implementer | Build the pipeline, run experiments, produce results and figures, manage data/code, contribute to writing |

**Student profile for the hire:** MS or PhD student comfortable with Python and deep learning (PyTorch) and with computer vision (object detection/tracking; camera calibration/homography a plus); interest in sports data or willingness to ramp; owns implementation under joint supervision. Suggested appointment: ~[X] hrs/week RA over ~10–12 months to cover the workshop push and the journal extension.

**To settle:** funding source, appointment length and start date, and which PI is the student's advisor of record.

---

### 9. Risks and mitigations

- **Novelty overlap with xGHub.** Differentiate explicitly: the VLM/multimodal features and the gap-analysis framing are ours; a pose-to-xG regressor is not. Lock this in before writing.
- **Broadcast-position noise.** Vision-derived positions carry meters of error (reported RMSE up to ~16 m, concentrated when players are undetected), which propagates into features. Mitigate with explicit error-propagation analysis, and consider anchoring the core result on StatsBomb freeze-frame/360 data to reduce dependence on raw broadcast tracking.
- **The "gap" interpretation trap.** A discrepancy from reference xG can mean our model is wrong, the reference is limited, or we capture real new signal. Only the last is a contribution, so we must calibrate/validate hard, treat the reference as fallible (not ground truth), and claim only defensible new signal.
- **Timeline.** The student is not yet hired and the CVPR main-track deadline is ~8 weeks out; target the workshop, not the main track.
- **Compute / VLM cost.** Prefer open-weight models that run locally (Gemma 4) for the core system; use hosted VLMs only as comparison points; set a budget.

---

### 10. Decisions to make in this meeting

1. Confirm the **novelty line vs. xGHub** and the headline contribution.
2. Choose the **starting competition/league** (by data coverage) and the **reference-xG source**.
3. Decide **compute**: local GPUs vs. cloud, and budget.
4. Confirm **student funding**, appointment length/start date, and advisor of record.
5. Approve the **job description** for the hire (Section 8 profile).
6. Agree on **authorship** (e.g., student first-author on both papers; PI order).
7. Confirm the **conference target** (CVsports workshop) and whether to keep a sports-analytics journal as a backup.
8. Set **meeting cadence** (recommend weekly once the student starts).

---

### 11. Immediate next steps

- [ ] Confirm roles, funding, and student start date.
- [ ] Post/confirm the student job description.
- [ ] Lock the starting dataset and reference-xG source.
- [ ] Watch for the CVsports 2027 call; record its deadline and set internal deadlines two weeks early.
- [ ] Stand up a shared code repository and data-access plan.
