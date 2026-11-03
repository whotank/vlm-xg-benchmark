# A Critical Literature Review of Expected Goals (xG) in Association Football

*Prepared as a foundation for original ML/DS research intended for sports-analytics venues (e.g., MIT Sloan SSAC, the ECML-PKDD Machine Learning and Data Mining for Sports Analytics workshop, KDD, the StatsBomb Conference, and journals such as the Journal of Quantitative Analysis in Sports, the Journal of Sports Analytics, and Frontiers in Sports and Active Living).*

---

## 1. How to read this review

This document is organized to do two things at once: to map the expected-goals (xG) literature comprehensively, and to expose the seams in it — the assumptions, unresolved problems, and methodological disputes where a genuinely novel contribution can be located. Sections 2–3 fix the problem definition and its history. Sections 4–6 cover the modeling core (features, model classes, data sources) and the evaluation machinery, which is where most ML/DS reviewers will concentrate their scrutiny. Sections 7–8 are the two most active research frontiers: the critique of xG as a measure of finishing skill, and the family of "beyond-xG" possession-value models that increasingly subsume the shot-level problem. Section 9 covers applications. Section 10 is the payoff — a structured account of open problems and what a reviewer will demand of a paper that claims to "improve xG." Section 11 covers the practical publication landscape (data, reproducibility, venues), and Section 12 is an annotated bibliography grouped by theme.

A caveat on sourcing. A large and influential fraction of xG knowledge lives outside peer review — in club R&D, data-vendor blogs (StatsBomb, Opta/Stats Perform), and analyst writing. Any credible literature review must engage this grey literature, but for a publishable paper you should anchor claims in citable work (arXiv, workshop proceedings, journals) wherever possible and treat blog posts as provenance for ideas rather than as evidence.

---

## 2. The problem, formally

Expected goals estimates the probability that a given goal attempt results in a goal, conditional on the observable characteristics of that attempt. In the standard formulation, each shot *i* is represented by a feature vector *xᵢ* and a binary outcome *yᵢ ∈ {0,1}* (goal or no goal), and an xG model is a calibrated estimate *p̂(yᵢ = 1 | xᵢ)*. A team's or player's cumulative xG over a set of shots is the sum of these probabilities, interpreted as the number of goals an "average" finisher would be expected to score from those same chances.

Three properties of the task drive almost every modeling decision in the literature. First, it is a **probability-estimation** problem, not a classification problem: the quantity of interest is a well-calibrated probability, so the model must be evaluated with proper scoring rules rather than accuracy (Section 6). Second, goals are **rare and high-variance** — most shots are misses, and the outcome of any single shot carries enormous stochasticity — which makes both estimation and validation statistically fragile and is the root of the finishing-skill controversy (Section 7). Third, the shots that reach the dataset are the product of **prior decisions by players** (whether to shoot, from where, under what pressure), so the training distribution is a selected, non-random sample of possible attempts — a selection-bias problem that most models ignore and that recent work has begun to confront directly (Sections 7 and 10).

It is worth being explicit that "xG" now names a *family* of related but non-identical quantities. Pre-shot xG conditions only on information available at the moment of the shot. Post-shot xG (also called expected goals on target, xGOT, or PSxG) conditions additionally on the ball's trajectory after being struck, and is used to separate chance quality from strike quality and to evaluate goalkeeping. Off-ball or possession-level quantities (OBSO, EPV) estimate scoring probability from game states that need not contain a shot at all. Conflating these is a common source of confusion and, in papers, of reviewer objections.

---

## 3. Historical development

The intellectual lineage predates the term. Charles Reep, working from the 1950s, and the statisticians Bernard Benjamin and Richard Pollard produced the first systematic quantitative studies of where goals come from — Reep and Benjamin's analyses of passing-move length and shot origin in the late 1960s, and Reep, Benjamin and Pollard's 1971 work identifying a negative-binomial structure in passing sequences, culminating in Pollard and Reep's 1997 study of the effectiveness of playing strategies, which quantified how possession in different pitch zones changed scoring chances. Reep's substantive conclusions (a pro–long-ball reading of the data) are widely regarded as overstated, but the methodological move — committing statistical structure about football to the record — is the taproot of the field. The label "expected goals" itself is often traced to Barnett and Hilditch's 1993 investigation of artificial-pitch effects, though this attribution circulates mainly in the grey literature and should be hedged in a paper.

The recognizably modern shot-level model arrives with **Ensum, Pollard and Taylor** (published in *Science and Football V*, dated 2004/2005), who fit logistic regression to shots from the 2002 World Cup and isolated the features that still anchor most models today — distance to goal and shot angle — alongside effects for shot type. In parallel, the same idea matured in ice hockey: **Alan Ryder's** 2004 "shot quality" work and **Brian Macdonald's** 2012 NHL expected-goals model at MIT Sloan established the template of situational features feeding a regression, and the explicit use of the resulting values to evaluate players. This cross-pollination from hockey is not incidental; several of football's methodological imports (rebound effects, shot-distance adjustment, skill-adjusted variants) have hockey antecedents.

The metric entered mainstream football analytics around 2012 through **Opta** (Sam Green's work on evaluating goalscorers), and then through an intense period of practitioner development roughly 2013–2015 in which analysts such as Michael Caley, Sander Ijtsma and others published shot-location "matrices" and logistic-regression variants on blogs. Van Haaren (2021) fairly characterizes this era as rich in insight but thin on rigorous experimental evaluation — a gap the academic literature has been filling since.

The most consequential recent shift is the arrival of **positional (freeze-frame and tracking) data**. Lucey et al.'s 2014 "Quality vs Quantity" paper at Sloan was an early demonstration that strategic features derived from spatiotemporal data (defender configuration, game phase) improve shot prediction beyond geometry alone. StatsBomb's introduction of a freeze frame with every shot (player positions at the moment of the shot) from 2018, and the StatsBomb 360 product from 2021, made defender- and goalkeeper-aware xG broadly reproducible on public and commercial data — a genuine inflection point that most post-2020 modeling exploits.

---

## 4. The modeling core I: features and model classes

### 4.1 Feature families

The literature's features fall into three tiers of increasing data demand. The **geometric tier** — distance to goal and the angle subtended by the goalmouth — is present in essentially every model and typically dominates feature-importance rankings; a logistic model on distance and angle alone is a strong baseline that any new model must beat. The **event-context tier** adds shot body part (foot vs. header), assist/pass type (through ball, cross, cutback, set-piece origin), play pattern (open play, fast break, corner, direct free kick), and whether the shot follows a rebound or a dribble. Mead, O'Hare and McMenemy (2023) additionally introduced team-strength context (Elo ratings, league position), reporting that competition-level quality effects can carry real predictive weight — a reminder that "chance quality" is entangled with who is taking and facing the shot.

The **positional tier** is where recent gains concentrate. Freeze-frame data support features such as the number of defenders in the triangle between the shooter and the goalposts, the fraction of the goalmouth occluded by defenders, goalkeeper position and distance, whether the shot is one-on-one, and whether the shot is taken under pressure. Anzer and Bauer (2021) and the KU Leuven work on freeze-frame features (Arslan) both show measurable improvements from encoding this information; the latter proposes a "goal projection" feature capturing how defender positioning shrinks the available target. The open modeling question at this tier is representational: positions are a variable-length set of (x, y, team, velocity) tuples, and hand-crafting scalar summaries (counts, angles) discards structure — which is precisely the motivation for the deep and graph-based approaches in Section 8.

### 4.2 Model classes and their trade-offs

**Logistic regression** remains the reference model. Its virtues are calibration (probabilities are the native output), interpretability (coefficients as log-odds effects), and data efficiency. Its limitation is that it imposes a rigid functional form; interactions and non-linearities (e.g., the joint effect of angle and defender occlusion) must be engineered by hand.

**Tree ensembles**, especially gradient-boosted trees (XGBoost), are the current workhorse for performance. Anzer and Bauer's (2021) best model is an XGBoost on hand-crafted features from 105,627 synchronized Bundesliga shots, reported at a ranked probability score of 0.197 and described as more accurate than any previously published xG model at the time; they use Shapley values for interpretability, illustrating the now-standard pairing of a black-box learner with post-hoc explanation. Comparative studies (e.g., the 2024 Opta-vs-Understat comparison) routinely find gradient boosting and random forests competitive or superior on error metrics, with SHAP used to reconcile feature importance across models.

**Deep learning** is used mainly to avoid hand-crafting positional features, learning representations directly from raw player coordinates or from the freeze frame treated as an image or point set (Section 8). The reported gains over well-tuned gradient boosting on *shot-level* xG are often modest, which is itself an important empirical fact: for the narrow pre-shot task on event+freeze-frame data, the ceiling appears close, and the marginal value of model complexity is small. This matters for framing a contribution — beating XGBoost on log-loss by a hair is not, by itself, a publishable result.

**Bayesian hierarchical models** are the most interesting recent methodological direction for this problem's specific pathologies. Scholtes and Karakuş's "Bayes-xG" (2024) uses a Bayesian hierarchical approach to apply player- and position-level corrections with principled partial pooling, and the 2025 Bayesian generalized-linear-mixed-model work reports StatsBomb-competitive discrimination (AUC ≈ 0.78 vs ≈ 0.80) from only seven interpretable variables, with transfer learning enabling small-sample team analysis. The appeal is threefold: native uncertainty quantification (a posterior over each shot's probability, not a point estimate), partial pooling that stabilizes player- and league-level effects under small samples, and a natural route to encoding structure and priors. Given that small samples and unstable player effects are the field's central statistical problem (Section 7), the Bayesian route is under-exploited relative to its fit.

---

## 5. The modeling core II: data sources and their consequences

Data availability is not a background detail in this literature — it is a first-order determinant of what a model can learn, and it has been studied as such. Robberechts and Davis (2020) show directly how the *amount* and *type* of available data affect the quality of learnable xG models, which reframes many cross-study comparisons as confounded by data provenance rather than by modeling skill.

Four data regimes recur. **Event-stream data** (Opta/Stats Perform, Wyscout, StatsBomb) provide the shot with hand-tagged context and are the basis for most public work; Pappalardo et al.'s (2019) public Wyscout release and StatsBomb's open data have been the main enablers of reproducible academic study. **Freeze-frame data** attach the positions of visible players at the shot instant. **Broadcast/optical tracking (e.g., StatsBomb 360)** extend positional coverage. **Full tracking data** (25 Hz positions of all 22 players plus ball) are the richest regime, largely proprietary, and are what the physics-based and possession-value models in Section 8 require; the recent open, integrated spatiotemporal-plus-event datasets (e.g., Bassek et al., 2025, in *Scientific Data*) are beginning to relax this constraint.

Two consequences deserve emphasis for a prospective author. First, **synchronization** of event and positional streams is a genuine, error-prone preprocessing problem with its own literature (Anzer and Bauer's synchronization work; subsequent methods including ELASTIC and others), and synchronization error propagates into positional features — a limitation reviewers increasingly expect to see acknowledged. Second, **inter-provider disagreement** is real: models built on Opta vs. Understat vs. StatsBomb data can produce systematically different values and player rankings, so results are not automatically portable across data sources, and a paper's claims are implicitly scoped to its data vendor.

---

## 6. Evaluation and calibration

This section is where ML/DS reviewers are least forgiving, so it deserves disproportionate attention in any paper.

Because xG is a probability, it must be scored with **proper scoring rules**. The standard choices are logarithmic loss (cross-entropy) and the Brier score for the binary goal/no-goal outcome, and the **ranked probability score (RPS)** when the target is a match-level or multi-outcome distribution — Anzer and Bauer report RPS, and the match-prediction literature (Section 9) uses it heavily because it rewards probabilistic sharpness on ordered outcomes. **AUC/ROC** is widely reported but is a *discrimination* measure that is insensitive to calibration; a model can rank shots well and still be systematically mis-scaled, so AUC alone is insufficient evidence that an xG model is fit for the summation that defines cumulative xG.

**Calibration** is therefore not optional. The relevant tools are reliability (calibration) curves, the expected calibration error (ECE), and post-hoc recalibration via Platt scaling, isotonic regression, or beta calibration, evaluated on held-out data and ideally on a temporally or competition-disjoint test set (e.g., training on league seasons and testing on a tournament, as some public implementations do). Calibration is what licenses the interpretation of summed probabilities as expected goal counts; a model that is well-discriminating but poorly calibrated will produce biased team and player aggregates even when its per-shot ranking is excellent.

At the metric level above single shots, Franks, D'Amour, Cervone and Bornn's "meta-analytics" framework (JQAS, 2016) is essential reading: it formalizes desirable properties of a sports metric — **discrimination** (does it separate players?), **stability** (does a player's value persist across time?), and **independence** (does it add information beyond existing metrics?) — and gives tools to estimate them. Reporting these properties for a proposed xG-derived measure, rather than only a loss number, is a mark of methodological seriousness and directly addresses the finishing-skill critique below.

A recurring, and often overlooked, evaluation pitfall is **leakage and non-independence**: shots from the same match, team, or player are correlated, so naïve random train/test splits overstate performance. Grouped or temporal cross-validation is the defensible choice, and its absence is a common, legitimate reviewer objection.

---

## 7. The finishing-skill critique and the bias problem

The most important recent development is not a better model but a sharper understanding of what xG can and cannot measure — specifically, whether the difference between a player's actual goals and cumulative xG ("goals above expectation," GAX) captures finishing skill. The practitioner intuition is that a persistent GAX surplus indicates a good finisher. The empirical problem, noted since Opta-era work (Pleuler and others), is that GAX is not stable season to season, which would be surprising if it were tracking a durable skill.

Davis and Robberechts (2024) give the most rigorous account. They examine three hypotheses for why GAX fails to capture finishing: that the deviation between actual and expected goals is an inadequate statistic given high shot-outcome variance and limited sample sizes; that summing over *all* shots (including near-hopeless ones that dominate the count) is inappropriate; and — their central contribution — that xG models contain **biases arising from interdependencies in the training data** that confound skill measurement. Borrowing from the algorithmic-fairness literature, they argue that a standard xG model does not actually represent an "average player": because better players systematically take and face different shots, the learned model absorbs those correlations and will overestimate the finishing of weak finishers and underestimate that of strong ones. This reframes an accuracy problem as an identification/confounding problem, and it is a template worth studying closely if your improvement touches player evaluation.

Two adjacent lines address the same underlying difficulty. **Post-shot models** (post-shot xG / xGOT / PSxG, and the derived "expected goals added," EGA) condition on the struck ball's trajectory to separate chance creation from strike execution, yielding a somewhat more stable execution signal — but Baron, Sandholtz, Pleuler and Chan (2023/2024, JQAS) point out that these metrics assign zero value to off-target shots, which are roughly 57–65% of all shots, discarding most of the available execution signal. Their "Miss it like Messi" proposes a player-specific generative model of shot trajectories (a mixture of truncated bivariate Gaussians) to extract skill information from where misses go, not just from goals. This is a good example of a well-scoped, publishable contribution: it identifies a specific, quantified blind spot in existing metrics and proposes a principled model targeted at it.

The through-line for an author: claims that a new xG model "measures finishing better" are now held to a high bar. You will be expected to demonstrate stability and to argue that improvements are not artifacts of the same confounding Davis and Robberechts describe.

---

## 8. Beyond the shot: possession-value models as the extension frontier

A substantial part of the "improve xG" agenda has migrated to models that value game states and actions rather than shots, on the grounds that xG's core limitation is that it says nothing about situations in which no shot was taken. These models are both the natural competitors to and the natural extensions of xG, and any improvement proposal should be positioned relative to them.

**Expected Threat (xT)**, introduced by Karun Singh (2018), discretizes the pitch into zones and uses a Markov model to assign each zone a value equal to the scoring probability attainable from it, crediting ball progression (passes and carries) that moves possession into more valuable zones. Its strengths are simplicity and transparency; its acknowledged limitations are that it typically ignores defensive actions and shots, uses coarse event data, and omits pressure and off-ball context.

**Valuing Actions by Estimating Probabilities (VAEP)**, from Decroos, Bransen, Van Haaren and Davis (KDD 2019), values any on-the-ball action by the change it induces in the probabilities of the team scoring and conceding within a short horizon, using a common action language (SPADL) across vendors. VAEP generalizes the xG idea — an xG value is essentially the scoring-probability term evaluated at a shot — to the whole action stream, and accounts for context. Comparative work (van Roy and colleagues; Decroos and Van Haaren) shows xT and VAEP make different design choices that yield different player rankings, which is a useful cautionary tale about metric-dependence.

The **spatial/physical strand** models scoring value continuously over the pitch from tracking data. Spearman et al. (2017) build a physics-based pass-probability model; Spearman's "Beyond Expected Goals" (2018) introduces **off-ball scoring opportunity (OBSO)**, decomposing scoring probability into the probability that the next on-ball event occurs at a location, that the attacking team controls the ball there (pitch control), and that a shot from there scores. Fernández, Bornn and Cervone (2019) formalize a full **expected possession value (EPV)** framework for soccer (with lineage in the basketball EPV work of Cervone, D'Amour, Bornn and Goldsberry). These models make xG one factor inside a richer spatial value surface and are the state of the art when full tracking is available.

Most recently, the field is moving to **learned relational representations**: graph neural networks that treat players as nodes and their spatial relations as edges. Examples include GNN-based defensive-value models (e.g., DEFCON, 2026, using graph attention to assign defenders credit for reducing opponent EPV; GoalNet, 2025), graph representations for multi-agent spatiotemporal analysis (Raabe et al., 2023), and reinforcement-learning formulations that learn action-value functions for players (Liu, Luo and Schulte, 2020; Luo, Schulte and Poupart's inverse-RL, 2020). These approaches directly attack the representational limitation of hand-crafted positional features noted in Section 4, and they are where deep learning is contributing the most to the broader valuation problem — a promising, if data-hungry, direction for an xG-adjacent contribution.

Two late-2025 preprints signal the current frontier and are worth reading for framing: a hierarchical-Bayesian **counterfactual** treatment ("What If They Took the Shot?", 2025), which formalizes xG for the shots that were *not* taken, and a probabilistic framework for **shot occurrence** itself ("Beyond Expected Goals: A Probabilistic Framework for Shot Occurrences," 2025), which models whether and where a shot happens — both of which chip at the selection-bias problem from Section 2.

---

## 9. Applications

Four application areas recur, and each imposes different demands on the model.

**Player and team performance evaluation** is the original use: comparing xG created and conceded to actual goals to judge over- or under-performance and to separate process from results. Brechot and Flepp (2020, *Journal of Sports Economics*) make the influential argument that raw results in European club football are noisy enough that xG-based evaluation is a better guide to underlying performance than points or goals over realistic sample sizes — a rigorous statement of the "process over outcomes" case, and a good citation for motivating xG's practical value. The finishing-skill caveats of Section 7 bound how far player-level over-performance can be trusted.

**Match and season forecasting** uses xG as an input to outcome models, typically Poisson-type goal models (with lineage in Maher, 1982, and Goddard's 2005 bivariate-Poisson/ordered-probit forecasting) or rating systems. A useful recent, tracking-based comparison is Forcher et al. (2025, Frontiers), which pits xG against EPV for match prediction and finds xG most informative in *post-match* explanation of offensive performance (RPS ≈ 0.148) while EPV is stronger for *pre-match* forecasting — a nuanced result showing the two metrics carry different predictive content depending on the task.

**Opposition analysis and tactics** use spatial xG/OBSO surfaces to identify where a team creates and concedes danger, and **recruitment/scouting** uses aggregated shot quality and possession value to find undervalued players. The high-profile club adoption stories (Midtjylland, Brentford, and the widely told Liverpool narrative) are motivational context rather than evidence, and should be cited as such.

---

## 10. Synthesis: open problems and where a novel contribution can sit

The following are, in my reading of the literature, the live problems where a well-executed paper can make a defensible novelty claim. For each I note the core gap and what a reviewer will demand.

**Uncertainty quantification.** Almost all deployed xG models emit point probabilities with no measure of confidence, yet the whole finishing-skill debate is fundamentally about sampling variability. A model that produces calibrated *predictive intervals* per shot (Bayesian posteriors, conformal prediction, or deep ensembles) and propagates them into player/team aggregates would address a real gap. Reviewers will demand that the intervals be validated for calibration, not merely produced.

**Selection bias and the counterfactual "shots not taken."** xG is trained only on realized shots, but the interesting question — how good was a chance — concerns states that may never have produced a shot. OBSO and the 2025 counterfactual/shot-occurrence preprints begin here; there is room for a principled causal or selection-corrected treatment (e.g., joint modeling of the shoot/no-shoot decision and the outcome, Heckman-style correction, or a decision-theoretic framing). Reviewers will want a clear causal estimand and an argument about identifiability, not just a new feature set.

**Finishing-skill identifiability.** Following Davis and Robberechts, the open problem is to construct an xG model and a derived skill metric that are provably (or at least demonstrably) robust to the training-data confounding they document — for instance via fairness-style reweighting, causal adjustment, or player-conditional models with honest partial pooling. The bar is a stability demonstration (season-to-season persistence) and a confounding argument, per Section 6's meta-analytics tools.

**Representation learning for the freeze frame / tracking snapshot.** Hand-crafted positional summaries are lossy. Set- or graph-based encoders (Deep Sets, GNNs, attention over players) that are permutation-invariant and geometry-aware are a natural fit and only partially explored for *shot-level* xG (as opposed to full-possession EPV). The risk to manage: showing that added complexity yields gains beyond a strong XGBoost baseline on genuinely held-out data, since the pre-shot ceiling appears near.

**Transfer, domain shift, and small-sample leagues.** Models are largely trained and validated on a handful of elite leagues and one data vendor. How xG transfers across leagues, tiers, eras, women's football, and providers is under-studied, and inter-vendor disagreement (Section 5) makes it pressing. Transfer learning and hierarchical pooling (Bayes-xG points this way) are promising, and the practical payoff — usable models where data are scarce — is high.

**Temporal/sequential context before the shot.** Most models are static snapshots at the shot instant. The pre-shot sequence (the buildup dynamics, defender recovery, momentum) plausibly carries signal that a single frame misses; sequence models (RNNs/transformers over the possession, or trajectory features) are a route, with the recent frame-clustering and situational-xT work as precedent.

**Post-shot and off-target signal.** Baron et al. show most execution signal is discarded. Richer generative models of shot outcome (including trajectory, and goalkeeper response) that unify pre-shot, post-shot, and off-target information into a coherent skill decomposition remain open.

**Evaluation methodology itself.** There is room for a *methods* contribution arguing for decision-relevant, calibration-first, leakage-aware evaluation protocols for xG, and benchmarking the field's models under them on open data — the sort of "sober second look" paper that these communities value and that is often high-impact precisely because it is rarely done.

A strategic note on positioning: given how near the pre-shot performance ceiling appears to be, the strongest contributions are unlikely to be "a more accurate p(goal|shot)." They are more likely to lie in *what the probability is used for* — uncertainty, causality/selection, skill identifiability, transfer, or evaluation — or in extending the target beyond the isolated shot toward the possession-value frameworks in Section 8. Decide which of these your improvement idea actually targets before writing the related-work section, because the closest prior work (and therefore your novelty claim) is completely different for each.

---

## 11. Practical publication landscape

**Venues.** For methods-forward ML work: KDD (and its sports-analytics offerings), the ECML-PKDD Machine Learning and Data Mining for Sports Analytics (MLSA) workshop (a natural home for exactly this topic, and where several key papers appeared), IJCAI/AAAI when the contribution is a general ML method demonstrated on soccer, and MIT Sloan SSAC for high-visibility applied work. For journal versions: the *Journal of Quantitative Analysis in Sports* (JQAS), the *Journal of Sports Analytics*, *Data Mining and Knowledge Discovery* (for methods), *Frontiers in Sports and Active Living*, and *PLOS ONE*. The StatsBomb Conference is valuable for practitioner reach and data access.

**Open data and reproducibility.** The realistic public options are StatsBomb open data (event + freeze frame, with the `statsbombpy`/`mplsoccer` tooling), the Wyscout public dataset (Pappalardo et al., 2019), and emerging open tracking+event datasets (e.g., Bassek et al., 2025). SPADL (from the VAEP work) is the de facto common action representation and the `socceraction` library implements xG/VAEP pipelines. Reviewers in these venues increasingly expect released code and a clear statement of which data vendor was used, because results do not transfer across vendors automatically.

**What reviewers reward.** A crisp, decision-relevant novelty claim; proper scoring rules *and* calibration analysis (not AUC alone); grouped/temporal validation with an honest out-of-distribution test; comparison against a genuinely strong baseline (a tuned XGBoost on freeze-frame features, not a toy logistic model); an explicit engagement with the finishing-skill/bias literature if the paper touches player evaluation; and released code and data splits.

---

## 12. Annotated bibliography (grouped by theme)

*Citations are given in enough detail to locate each work; verify exact page numbers and DOIs against the original before formal citation. Practitioner/blog sources are marked as such and should be used as provenance, not evidence.*

### Foundations and history
- **Reep, C. & Benjamin, B. (1968); Reep, Benjamin & Pollard (1971); Pollard, R. & Reep, C. (1997), "Measuring the effectiveness of playing strategies at soccer," *The Statistician* 46(4).** The origin of quantitative shot/possession-value analysis; 1997 paper is the clearest xG precursor.
- **Pollard, R. (2002), "Charles Reep (1904–2002): pioneer of notational and performance analysis in football," *Journal of Sports Sciences* 20(10).** Historical context on Reep.
- **Ensum, J., Pollard, R. & Taylor, S. (2004/2005), "Applications of logistic regression to shots at goal at association football," *Science and Football V*.** First recognizably modern logistic-regression shot model; establishes distance and angle.
- **Ryder, A. (2004), "Shot Quality," hockeyanalytics.com; Macdonald, B. (2012), "An Expected Goals Model for Evaluating NHL Teams and Players," MIT Sloan.** Hockey antecedents of xG and of using it to rate players.
- **Rathke, A. (2017), "An examination of expected goals and shot efficiency in soccer," *Journal of Human Sport and Exercise* 12(2).** Early academic soccer-xG study.

### Modern shot-level models (event + positional data)
- **Lucey, P., Bialkowski, A., Monfort, M., Carr, P. & Matthews, I. (2014), "'Quality vs Quantity': Improved Shot Prediction in Soccer Using Strategic Features from Spatiotemporal Data," MIT Sloan.** Seminal demonstration that tracking-derived strategic features beat geometry alone.
- **Anzer, G. & Bauer, P. (2021), "A Goal Scoring Probability Model for Shots Based on Synchronized Positional and Event Data in Football (Soccer)," *Frontiers in Sports and Active Living* 3:624475.** The reference modern model: XGBoost on synchronized Bundesliga data, RPS ≈ 0.197, SHAP interpretation, analyst validation. **Read first.**
- **Mead, J., O'Hare, A. & McMenemy, P. (2023), "Expected goals in football: Improving model performance and demonstrating value," *PLOS ONE* 18:e0282295.** Adds team-strength/context features; compares several learners.
- **Fairchild, A., Pelechrinis, K. & Kokkodis, M. (2018), "Spatial analysis of shots in MLS: a model for expected goals and fractal dimensionality," *Journal of Sports Analytics* 4(3).**
- **Eggels, H., van Elk, R. & Pechenizkiy, M. (2016), "Expected goals in soccer: Explaining match results using predictive analytics," MLSA workshop.**
- **StatsBomb (2021), "Freeze Frames and Defender Locations"; StatsBomb (2025), "Upgrading Expected Goals" (practitioner).** Provenance for freeze-frame/360 features and vendor-model design.

### Interpretability and Bayesian approaches
- **Van Haaren, J. (2021), "'Why Would I Trust Your Numbers?' On the Explainability of Expected Values in Soccer," arXiv:2105.13778.** Explainability of xG/xT/VAEP; also a good historiography of the blog era.
- **Cavus, M. & Biecek, P. (2022), "Explainable expected goal models for performance analysis in football analytics," arXiv:2206.07212.**
- **Scholtes, A. & Karakuş, O. (2024), "Bayes-xG: player and position correction on expected goals (xG) using Bayesian hierarchical approach," *Frontiers in Sports and Active Living* 6:1348983.** Principled player/position partial pooling.
- **(2025) "Toward interpretable expected goals modeling using Bayesian mixed models," *Frontiers in Sports and Active Living* (PMC12055760).** GLMM matching StatsBomb discrimination with seven interpretable variables; transfer learning for small samples.

### The finishing-skill / bias critique and shot-execution signal
- **Davis, J. & Robberechts, P. (2024), "Biases in Expected Goals Models Confound Finishing Ability," arXiv:2401.09940 (+ KU Leuven DTAI blog).** Central critical work; confounding/fairness framing of why GAX fails. **Read first.**
- **Robberechts, P. & Davis, J. (2020), "How data availability affects the ability to learn good xG models," MLSA (Springer CCIS 1324).** Data provenance as a confounder in model comparisons.
- **Baron, E., Sandholtz, N., Pleuler, D. & Chan, T. C. Y. (2023/2024), "Miss it like Messi: Extracting value from off-target shots in soccer," *JQAS* 20(1).** Generative shot-trajectory model recovering skill signal from misses; post-shot xG (PSxG), expected goals added (EGA).
- **Franks, A., D'Amour, A., Cervone, D. & Bornn, L. (2016), "Meta-analytics: tools for understanding the statistical properties of sports metrics," *JQAS* 12(4).** Discrimination/stability/independence framework for evaluating any metric.

### Beyond xG: possession value, spatial and relational models
- **Singh, K. (2018), "Introducing Expected Threat (xT)," karun.in (practitioner).** Zone-based Markov possession-value model.
- **Decroos, T., Bransen, L., Van Haaren, J. & Davis, J. (2019), "Actions Speak Louder than Goals: Valuing Player Actions in Soccer," KDD '19 (arXiv:1802.07127).** VAEP + SPADL; generalizes xG to all on-ball actions.
- **Spearman, W. et al. (2017), "Physics-Based Modeling of Pass Probabilities in Soccer," MIT Sloan; Spearman, W. (2018), "Beyond Expected Goals," MIT Sloan.** Pitch control and off-ball scoring opportunity (OBSO).
- **Fernández, J., Bornn, L. & Cervone, D. (2019), "Decomposing the Immeasurable Sport: A Deep Learning Expected Possession Value Framework for Soccer," MIT Sloan.** Full EPV framework (with basketball-EPV lineage: Cervone, D'Amour, Bornn & Goldsberry, 2014).
- **Power, P., Ruiz, H., Wei, X. & Lucey, P. (2017), "Not All Passes Are Created Equal," KDD.** Risk/reward of passes from tracking data.
- **Liu, G., Luo, Y. & Schulte, O. (2020), "Deep soccer analytics: learning an action-value function," *DMKD* 34(5); Luo, Y., Schulte, O. & Poupart, P. (2020), "Inverse Reinforcement Learning for Team Sports," IJCAI.**
- **Raabe, D., Nabben, R. & Memmert, D. (2023), "Graph representations for the analysis of multi-agent spatiotemporal sports data," *Applied Intelligence* 53; plus GNN valuation models — DEFCON (2026), GoalNet (2025).** Relational/graph learning frontier.
- **(2025) "What If They Took the Shot? A Hierarchical Bayesian Framework for Counterfactual Expected Goals," arXiv:2511.23072; (2025) "Beyond Expected Goals: A Probabilistic Framework for Shot Occurrences in Soccer," arXiv:2512.00203.** Current frontier on counterfactuals and shot-occurrence (selection) modeling.

### Applications: evaluation and forecasting
- **Brechot, M. & Flepp, R. (2020), "Dealing with Randomness in Match Outcomes: How to Rethink Performance Evaluation in European Club Football Using Expected Goals," *Journal of Sports Economics* 21(4).** Rigorous case for xG over results in evaluation.
- **Goddard, J. (2005), "Regression models for forecasting goals and match results in association football," *International Journal of Forecasting* 21; Maher (1982).** Foundational goal/result forecasting (Poisson, ordered probit).
- **Forcher, L. et al. (2025), "AI in Bundesliga match analysis — EPV vs. xG to predict match outcomes," *Frontiers in Sports and Active Living* (DOI 10.3389/fspor.2025.1713852).** Tracking-based comparison; xG best post-match, EPV best pre-match.

### Surveys, datasets, tooling
- **Herold, M. et al. (2019), "Machine learning in men's professional football: current applications and future directions," *IJSSC* 14(6).** Field survey.
- **"Short Survey in Machine Learning for Soccer Analytics" (2024), Preprints.org/ResearchGate.** Recent, broad, useful for orientation and its bibliography.
- **Pappalardo, L. et al. (2019), "A public data set of spatio-temporal match events in soccer competitions," *Scientific Data* 6; PlayeRank, *ACM TIST* 10(5).** Public Wyscout data + player-ranking framework.
- **Bassek, M., Rein, R., Weber, H. & Memmert, D. (2025), "An integrated dataset of spatiotemporal and event data in elite soccer," *Scientific Data* 12.** Open tracking+event data.
- **Tooling:** SPADL / `socceraction`, `statsbombpy`, `mplsoccer` (practitioner libraries; the standard reproducibility stack).

---

## 13. Candidate research directions (working notes)

*These are the author's own improvement ideas, recorded with an assessment against the literature above and the closest prior work identified as of July 2026. New references surfaced for this section are collected at the end (Section 13.4) and should be merged into the main bibliography if pursued.*

### 13.1 The author's seven ideas, as stated

1. Try vision models, VLMs, and Gemma 4 on shot databases that include images.
2. xG personalized by player — on physique or previous stats — more accurate than the "average-player" model.
3. Benchmarking — what constitutes a good xG model?
4. Biases — where is current xG falling short?
5. Vision models via the practitioner CV stack (Roboflow **Supervision**, maintained by Piotr Skalski / SkalskiP), a model-agnostic open-source library for detection, tracking, and annotation.
6. Automated, vision-based xG — explainable, and more accurate than conventional xG.
7. Additional directions (added below in 13.3).

### 13.2 Assessment against the literature

**Thematic synthesis.** Ideas 1, 2, 5, and 6 converge on a single coherent program — an *explainable, vision-derived, player-aware xG* — while ideas 3 (benchmarking) and 4 (biases) are the evaluation and motivation scaffolding around it. Treating these as one project with a clearly located novelty claim is far stronger than pursuing them separately.

**Vision → xG (ideas 1, 5, 6).** The enabling step — recovering player/ball positions and a top-view minimap from a single broadcast — is a benchmarked, actively solved task, so a pipeline that merely extracts positions and feeds a conventional xG model is engineering integration rather than a research contribution. The two pieces of prior work that most constrain novelty here are **xGHub / "Once Upon a Goal"** (an open dataset of shots enriched with pose, body orientation, and contextual features from broadcast video, with a use case feeding those into a standard xG framework — essentially ideas 1+2+5+6 already executed) and the **SoccerNet Game State Reconstruction** benchmark and codebase (localizing and identifying all players from broadcast onto a 2D minimap, with the GS-HOTA metric). Supervision (Roboflow/SkalskiP) is the correct *tooling* for such a pipeline but is not itself a claim. A first-order feasibility caveat: broadcast-derived positions are noisy (reported position RMSE roughly 1.7–16.4 m across providers, with most error occurring when a player is not detected), and that noise propagates directly into positional xG features. Where genuine novelty remains: rigorous *explainability* (idea 6 done with a faithfulness evaluation, not just narration), *new feature modalities* that vision unlocks and event data lacks (pose, body orientation, goalkeeper set/position, defender balance), and using vision to observe *off-ball* states in which no shot occurred (connecting to selection bias, Section 10).

**Gemma 4 (idea 1).** Gemma 4 is real and relevant (Google DeepMind; family released April 2, 2026 under Apache 2.0, spanning ~2.3B–31B dense and MoE variants; a 12B unified encoder-free multimodal model followed on June 3, 2026; natively multimodal with native open-vocabulary bounding-box output, a thinking mode, and long context). The honest caveat is that as a raw detector it will not match a fine-tuned YOLO on throughput or accuracy; its comparative advantage is open-vocabulary querying without retraining and reasoning. It should therefore be cast as an *open-vocabulary semantic-feature extractor* or an *explanation/reasoning layer* over a strong xG model, not as "a better player detector." VLMs are already entering soccer understanding (prompt-based LLMs for foul recognition in the SoccerNet 2025 challenges; SoccerMaster as a soccer vision foundation model), but using a VLM as an *xG explainer with an explicit faithfulness metric* appears to be open.

**Personalized xG (idea 2).** This is the idea with the most dangerous trap. Naively conditioning xG on player identity or past conversion is precisely the confounding that Davis and Robberechts (2024) show yields unstable, near-circular skill estimates — the model absorbs the player's own conversion tendency, and one then "rediscovers" it. Moreover, player- and position-level correction with partial pooling already exists (Bayes-xG, Scholtes and Karakuş 2024; the Bayesian mixed-model work). The defensible, open version has two parts: (a) personalize by *physique/biomechanics* — pose, body orientation, physical attributes — rather than by identity or historical conversion (xGHub opens this door); and (b) treat it as a research question in its own right: *can a player-conditional xG be constructed that provably is not merely memorizing conversion rate?* The latter is an essentially unsolved problem framed by the bias literature and is publishable on those terms.

**Biases (idea 4).** Less a standalone project than the lens that makes idea 2 rigorous. The concrete open problems are the training-data confounding (framed via algorithmic fairness by Davis and Robberechts), the off-target-shot blind spot quantified by Baron et al., and the selection bias of shots never taken.

**Benchmarking (idea 3).** Potentially the strongest single starting point. The field's evaluation practice is weak: AUC (a discrimination-only measure) is over-reported while calibration is under-reported; random train/test splits leak across correlated shots; there is no shared test set; and models disagree across data vendors without systematic cross-vendor comparison. A proper open xG benchmark — fixed grouped/temporal splits, proper scoring rules together with calibration (ECE), the Franks et al. stability/discrimination/independence metrics, strong baselines, and multiple data providers — would constitute a recognized benchmark/resource contribution (a real category at KDD and NeurIPS dataset tracks, and at JQAS). It is tractable, needed, and becomes the yardstick against which a later vision/personalized model is measured.

### 13.3 Additional directions proposed in review

1. **xG under measurement error.** Broadcast positions carry meters of noise, yet no published model propagates that perception uncertainty into the xG estimate. A model emitting calibrated xG *with* uncertainty reflecting both shot stochasticity and perception noise addresses the vision and uncertainty-quantification gaps simultaneously.
2. **Faithful VLM explanations for xG.** Use a strong model for the probability and a VLM for the rationale, with the contribution being a *faithfulness metric* that tests whether the generated explanation matches the model's true drivers (e.g., validated against SHAP). Explainability is demanded frequently and evaluated honestly rarely.
3. **Counterfactual / off-ball xG from vision.** Use game-state reconstruction to recover all-player positions even when no shot occurs, then apply the shot-occurrence/counterfactual framing (the 2025 preprints in Section 8) to value chances that were never taken. Vision supplies exactly the off-ball states event data lacks; this connects the author's CV strengths to the deepest open problem in the field. Strongest option, also hardest.
4. **Does pose actually help, or is it identity leakage?** A focused study quantifying how much pose/orientation adds over freeze-frame features, and whether the gain survives player-confounding controls. Small, clean, publishable, and it de-risks the larger program.

**Recommendation on scope and sequencing.** Decide which *kind* of paper is intended, since the bar and venue differ: an applied/systems paper (vision pipeline → xG; MIT Sloan, StatsBomb Conference), a methods paper (uncertainty, causality, calibration; KDD, the ECML-PKDD MLSA workshop, JQAS), or a benchmark/resource paper (idea 3; dataset tracks, JQAS). Suggested order: the benchmark first (idea 3), then a methods contribution on either uncertainty-under-vision-noise (13.3 #1) or player-conditional-xG-without-confounding (idea 2 done rigorously via 13.2). The pure vision→xG pipeline is the weakest standalone claim given that xGHub and SoccerNet GSR already exist.

### 13.4 New references surfaced for this section (vision, VLM, tooling)

- **"Once Upon a Goal: Towards orientation-based shot metrics in football" (xGHub) (2026), *Computer Vision and Image Understanding* (ScienceDirect S1077314226000998).** Open dataset of shots with pose, body orientation, and contextual features from broadcast video; automated detection/tracking pipeline with external verification; use case incorporating pose/orientation into a standard xG framework. **Closest prior work to ideas 1/2/5/6.**
- **Somers, V. et al. (2024), "SoccerNet Game State Reconstruction: End-to-End Athlete Tracking and Identification on a Minimap," CVPRW (CVSports).** Dataset (200 clips), GS-HOTA metric, open codebase (`SoccerNet/sn-gamestate`).
- **Giancola, S. et al. (2025), "SoccerNet 2025 Challenges Results," arXiv:2508.19182.** Team ball-action spotting, monocular depth, multi-view foul recognition, and game-state reconstruction; large open participation; note the emerging use of prompt-based LLMs for contextual reasoning.
- **"Concurrent validity of computer-vision AI player tracking software using broadcast footage" (2025), arXiv:2508.19477.** FIFA-involved validity study; position RMSE ≈ 1.7–16.4 m; error concentrated when players are undetected; tactical feed improves detection. **Key feasibility/error-propagation citation for any vision-xG claim.**
- **Gemma Team, Google DeepMind (2026), "Gemma 4 Technical Report," arXiv:2607.02770; model card at ai.google.dev.** Natively multimodal open-weight family (dense + MoE, ~2.3B–31B), 12B unified encoder-free variant, native open-vocabulary detection, thinking mode, 256K context, Apache 2.0.
- **"SoccerMaster: A Vision Foundation Model for Soccer Understanding" (2026), arXiv:2512.11016; FOOTPASS (2025), arXiv:2511.16183.** VLM/foundation-model and multi-modal tactical-context directions for soccer video.
- **Tooling:** Roboflow **Supervision** (`github.com/roboflow/supervision`, model-agnostic detection/tracking/annotation; maintainer Piotr Skalski / SkalskiP) as the practical pipeline glue, layered on a detector/tracker and pitch-keypoint homography.

---

### Suggested next step

This review is deliberately broad so that you can situate any improvement idea within it. The single most useful thing you can do now is tell me which open problem in Section 10 (or which direction in Section 13) your idea actually targets. With that, I can (1) deepen the corresponding subsection, (2) find and characterize the closest prior work so your novelty claim is airtight, and (3) sketch a concrete evaluation protocol reviewers in your target venue will accept.
