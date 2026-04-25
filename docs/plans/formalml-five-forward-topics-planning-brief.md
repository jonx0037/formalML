# formalml.com — Planning Brief for Five Forward-Map Topics

**Status:** planning document (Claude Chat seed brief, not Claude Code handoff)
**Scope:** five forthcoming topics named in `formalstatistics.com` Topic 29 §29.10 Rem 18–22 as "forthcoming on formalml"
**Audience:** Jonathan, for use when starting a fresh Claude Chat session to plan formalml content
**Version:** v1 (April 2026 — drafted at Topic 29 delivery)

---

## § 1 Context

### §1.1 Why this brief exists

The five topics below are named as pointers from `formalstatistics.com` Topic 29 (Track 8 opener, "Order Statistics & Quantiles") but live on `formalml.com` rather than on formalstatistics. They are **bridge topics** — the tail ends of the statistical-theory curriculum that happen to sit on the ML side of the site boundary because they either (a) target ML practitioners specifically (conformal prediction, quantile regression as used for risk forecasting) or (b) require ML infrastructure Jonathan has chosen to center on formalml (multivariate geometry, high-dim asymptotics). Each of the five has a direct predecessor section or remark on formalstatistics that forward-points to its eventual formalml home.

Topic 29 ships with placeholder prose ("forthcoming on formalml") in place of live `<ExternalLink>` components for these five, because breaking hyperlinks to non-existent slugs is worse than honest deferral. Once any of the five ships on formalml, the pointer can be converted to a proper cross-site link in a one-line `formalstatistics` PR. This brief is the road map for that ship sequence.

### §1.2 What this brief is NOT

This is not a handoff brief in the `formalstatistics-<topic>-handoff-brief.md` sense. It does **not** contain:

- Exact MDX frontmatter YAML — structure depends on formalml conventions.
- Specific file paths — depends on formalml repo layout.
- Line-by-line cross-reference edits — depends on which formalstatistics forward-pointers actually go live.
- Test pins or TypeScript signatures — depends on formalml's shared-module architecture.
- Build-order instructions — this is pre-implementation planning.

Each of the five topics, when Jonathan commits to writing it, gets its own handoff brief following the formalstatistics pattern — likely with a formalml-adapted frontmatter schema. This brief is the *upstream* document that seeds those handoff briefs.

### §1.3 What I'm assuming about formalml.com

Because I don't have the formalml.com project loaded in this session, I am assuming:

- **Tech stack and MDX conventions are broadly parallel to formalstatistics.** MDX + React + KaTeX + Tailwind, interactive components under `src/components/viz/`, shared utility modules under `src/components/viz/shared/`.
- **Curriculum has some track/section structure.** If formalml is flat (just a topic list), Jonathan adapts the "Track X" language in §3 below.
- **Citation convention is Chicago 17 Notes & Bibliography** (formalstatistics precedent).
- **Mathematical style is collaborative "we" prose, full-proof tolerant, no `\begin{aligned}`** (formalstatistics KaTeX lock propagates forward).
- **Component-interactivity bar is similar:** each topic gets 2–4 interactive components on the `QuantileAsymptoticsExplorer` aesthetic scale.

If any of these assumptions are wrong, the affected section is noted in §9 (Open Questions) — flag at session start, and I'll adapt.

### §1.4 The formalstatistics ↔ formalml boundary

Before diving in, a one-paragraph restatement of which topic lives where:

**formalstatistics owns:** the scalar-sample nonparametric machinery — ECDF, order statistics, sample quantile, Bahadur representation, DKW, Kolmogorov–Smirnov (Topic 29); kernel density estimation (Topic 30); bootstrap (Topic 31); empirical processes (Topic 32). Track 8 is "distribution-free inference for scalar or iid vector samples under no parametric model."

**formalml owns:** everything that either (a) conditions on covariates (quantile regression), (b) targets multivariate or high-dimensional geometry (statistical depth), (c) centers on prediction-set construction for ML models (conformal prediction), (d) sits far from central tendency (extreme value theory), or (e) is natively nonparametric-hypothesis-testing with permutation machinery (rank tests).

The five topics below are the specific cut points at which Topic 29 hands off to formalml.

---

## § 2 Executive Summary: The Five Topics

### §2.1 At a glance

| # | Title | Slug | Difficulty | Primary formalstatistics predecessor | Suggested write order | Approx read time |
|---|---|---|---|---|---|---|
| 1 | Conformal Prediction | `conformal-prediction` | intermediate | Topic 29 §29.7 (distribution-free quantile CIs) | **1st** | 60 min |
| 2 | Quantile Regression | `quantile-regression` | intermediate | Topic 29 §29.4 + §29.6 (sample quantile, Bahadur); Topic 15 §15.9 (check-loss M-estimator); Topic 21 (linear regression) | **2nd** | 55 min |
| 3 | Rank Tests | `rank-tests` | intermediate | Topic 29 §29.10 Rem 20; Topic 18 §18.10 Rem 25 | **3rd** | 55 min |
| 4 | Extreme Value Theory | `extreme-value-theory` | advanced | Topic 29 §29.10 Rem 18; Topic 11 (CLT); Topic 32 (empirical processes — helpful) | **4th** | 70 min |
| 5 | Statistical Depth | `statistical-depth` | advanced | Topic 29 §29.10 Rem 22; Topic 8 (multivariate distributions) | **5th** | 65 min |

### §2.2 Writing-order rationale

The order above optimizes for (a) prerequisite chain, (b) ML-practical traction per unit of effort, and (c) incremental buildup of formalml shared-module infrastructure.

- **Conformal prediction first** because it has the lowest formalstatistics prerequisite load (just Topic 29's §29.7 distribution-free quantile CI machinery plus basic CLT), the highest ML-practitioner audience value, and the most well-defined problem space. It's also the most likely to drive inbound traffic from "how do I get prediction intervals from my ML model" queries, making it a strategic opener for the Track 8 formalml satellites.
- **Quantile regression second** because it reuses the same check-loss / M-estimator machinery that Topic 29 sets up in §29.6 Cor 1, now conditioned on covariates. Once conformal prediction is live, *conformalized quantile regression* (CQR — Romano-Patterson-Candes 2019) becomes a Topic 2 ↔ Topic 1 cross-reference that lands well.
- **Rank tests third** as a standalone nonparametric hypothesis-testing piece. Requires Topic 29 forward-pointer backdrop but no deep covariate-conditional machinery.
- **EVT fourth** — harder pedagogically (new asymptotic framework — the Fisher-Tippett-Gnedenko trichotomy is not reducible to anything in Topics 17–19), but payoff is high (risk management, anomaly detection, OOD). Needs Topic 32 (empirical processes) to exist or at least to be partially sketched as a forward-pointer for the functional-CLT arguments.
- **Statistical depth fifth** — requires the most background (multivariate geometry, computational complexity concerns, a more abstract axiomatic framework). Natural capstone: it is to multivariate nonparametric inference what Topic 29 is to scalar nonparametric inference.

### §2.3 What changes if you write them in a different order

If Jonathan wants to ship the *easiest two first* for momentum: start with **rank tests + conformal prediction** (both have minimal prerequisites, high ML audience value, and can stand alone). QR, EVT, depth follow.

If Jonathan wants to ship the *most-asked-for first* for discoverability: **conformal prediction + quantile regression** (both show up constantly in ML-practitioner searches).

If Jonathan wants to match the formalstatistics-30/31/32 release cadence: **conformal prediction releases concurrent with formalstatistics 31 (Bootstrap)** since CQR + conformal are both "distribution-free wrappers"; **rank tests releases concurrent with formalstatistics 32 (Empirical Processes)**; EVT and depth release on their own clock.

---

## § 3 Topic 1 — Conformal Prediction

### §3.1 Slug & title

- **Slug:** `conformal-prediction`
- **Title:** "Conformal Prediction"
- **Subtitle (suggested):** "Distribution-free prediction intervals for any ML model."

### §3.2 One-sentence pitch

Wrap any black-box ML predictor in a prediction-set procedure that has a finite-sample coverage guarantee — $P(Y_{n+1} \in \hat C_\alpha(X_{n+1})) \geq 1 - \alpha$ — requiring only the assumption that training and test data are exchangeable.

### §3.3 Why this topic, and why first

Conformal prediction is the modern answer to the question formalstatistics Topic 29 §29.7 set up: "How do you get a distribution-free confidence interval?" Topic 29 answered this for the scalar quantile via order-statistic pairs; conformal prediction generalizes to prediction intervals for *any* function of $X$ — classification probability, regression conditional mean, quantile regression, black-box ML output. The exchangeability assumption is weaker than "iid from $F$," making it strictly more general.

Ship first because: (a) highest ML-practitioner audience; (b) lightest prerequisite load (Topic 29 §29.7 + CLT); (c) well-defined intellectual footprint (one core theorem, clean constructive proofs); (d) immediately actionable — readers can wrap their own models in 20 lines of code.

### §3.4 Scope — IN

- **Split (inductive) conformal prediction** as the primary treatment. Algorithm: split data into training + calibration; compute nonconformity scores on calibration set; take the $\lceil (1-\alpha)(n_{\text{cal}}+1) \rceil / n_{\text{cal}}$ quantile of calibration scores as threshold; construct $\hat C_\alpha(X_{n+1})$ from the set of $y$-values whose nonconformity score is below threshold.
- **The marginal coverage guarantee** as featured theorem: if $(X_i, Y_i)_{i=1}^{n+1}$ are exchangeable and nonconformity scores are a symmetric function of the training data, then $P(Y_{n+1} \in \hat C_\alpha) \geq 1 - \alpha$. Proof via rank symmetry on the calibration + test scores — an order-statistic argument of exactly the Topic 29 §29.7 flavor.
- **Full conformal prediction** — statement, computational cost noted, mostly deferred to Vovk et al. 2005 monograph.
- **Jackknife+ and CV+** — Barber-Candes-Ramdas-Tibshirani 2021 Annals. Constructs prediction intervals from leave-one-out residuals with a coverage guarantee that degrades by an explicit constant under non-exchangeable perturbations. Mid-depth treatment.
- **Adaptive Prediction Sets (APS, Romano-Sesia-Candes 2020)** for classification. Natural way to achieve approximately conditional coverage while keeping the marginal guarantee.
- **Conformalized Quantile Regression (CQR, Romano-Patterson-Candes 2019).** The cleanest cross-reference to Topic 2 (Quantile Regression) — CQR wraps a QR base predictor in the conformal envelope to get guaranteed coverage. Featured in §3.8 as a bridge example.
- **Conditional-coverage impossibility** (Barber-Candes-Ramdas-Tibshirani 2021 *Impossibility*). One sharp negative result: exact conditional coverage $P(Y_{n+1} \in \hat C | X_{n+1}) \geq 1 - \alpha$ is impossible with finite data under a non-trivial class of distributions. Defines the gap that APS and CQR narrow without closing.
- **Extensions under structured non-exchangeability:** covariate shift (Tibshirani-Barber-Candes-Ramdas 2019 NeurIPS), time-series / distribution drift (Chernozhukov-Wüthrich-Zhu 2018 + 2021). One section each, mostly pointing to papers.
- **Modern ML applications:** LLM token-level conformal, medical-diagnostic prediction sets, regression for tabular models. Brief remarks, with pointers.

### §3.5 Scope — OUT (deferred or dropped)

- **Online/sequential conformal** (Gibbs-Candes 2021) — in-depth treatment deferred to a second topic on "online and sequential distribution-free inference" if Jonathan chooses to write one; otherwise, a two-paragraph remark.
- **Bayesian-conformal hybrids** — formalml "Bayesian neural networks" adjacency; pointer only.
- **Multi-target / multi-output conformal** — brief remark.
- **Conformal for generative models** — active research, pointer to literature.
- **Deep proofs of exchangeability variants** (weak exchangeability, local exchangeability) — pointer to literature; proof sketches only.

### §3.6 Primary predecessor chain (from formalstatistics)

- **Topic 29 §29.7** — distribution-free quantile CIs via order-statistic pairs. The direct conceptual parent: conformal prediction generalizes the "invert a rank-based test" logic from scalar quantiles to general prediction sets. Topic 1 opens with an explicit back-reference: *"Topic 29 §29.7 showed how to build a distribution-free confidence interval for a scalar quantile using order-statistic pairs. Conformal prediction generalizes this idea to prediction intervals for any function of the data, replacing 'order statistic' with 'rank of the test-point nonconformity score among the calibration scores.'"*
- **Topic 29 §29.5** — ECDF + DKW. The calibration-set ECDF is the core object; DKW doesn't appear directly but provides the intuition for why $n_{\text{cal}}$-dependent coverage error shrinks at $1/\sqrt{n}$.
- **Topic 31 (Bootstrap)** — once shipped, a natural comparison: bootstrap gives asymptotic distribution-free CIs; conformal gives *finite-sample* distribution-free prediction sets. Coverage semantics differ (CI on a parameter vs prediction set for a new observation).

### §3.7 Key theorems

1. **Marginal coverage guarantee (featured, full proof).** For exchangeable $(X_i, Y_i)_{i=1}^{n+1}$ and a symmetric nonconformity score $s$, the split-conformal prediction set $\hat C_\alpha(X_{n+1}) = \{y : s(X_{n+1}, y; \mathcal{D}_{\text{train}}) \leq \hat q_\alpha\}$ satisfies $P(Y_{n+1} \in \hat C_\alpha(X_{n+1})) \geq 1 - \alpha$. Proof is a rank-symmetry argument; it can be rendered in ~15 MDX lines.
2. **Upper coverage bound** (often-cited companion to the main theorem). Under the same conditions + the technical "no ties in nonconformity scores" assumption, coverage is at most $1 - \alpha + 1/(n_{\text{cal}} + 1)$. Full proof.
3. **Conditional-coverage impossibility** (stated, cited to BCRT 2021). Any procedure with exact conditional coverage $P(Y \in \hat C | X = x) \geq 1 - \alpha$ for all $x$ must produce sets of infinite Lebesgue measure on non-trivial distributions. Stated, full proof in paper.
4. **Jackknife+ coverage bound** (Barber et al. 2021 Annals). Stated with constants.
5. **CQR coverage inheritance** (Romano et al. 2019). Conformalized QR inherits the conformal coverage guarantee regardless of whether the base QR estimator is well-calibrated — the conformal layer fixes miscalibration at the cost of wider intervals. Stated.

### §3.8 Suggested section structure

Ten sections mirroring the formalstatistics track-opener aesthetic:

| § | Title | Primary content |
|---|---|---|
| 1.1 | Motivation: the prediction-interval problem | The calibration/coverage gap in standard ML; why Bayesian and bootstrap approaches require distributional assumptions that ML models lack. |
| 1.2 | Exchangeability and the rank trick | Definition of exchangeability; rank symmetry on iid + permutation-invariant functions; the core insight that drives conformal. |
| 1.3 | Split conformal: the algorithm | Algorithm box. Discussion of nonconformity-score choices (residual, absolute residual, regression confidence, classification logit). |
| 1.4 | **Marginal coverage guarantee (featured)** | Full proof + upper-coverage bound. The Topic 29 §29.7 back-reference lives here. |
| 1.5 | Choice of nonconformity score | Regression: residual vs. absolute residual vs. normalized residual. Classification: hinge, cumulative-softmax (APS), probability of truth. |
| 1.6 | Full conformal prediction | Stated. Computational cost $O(n^2)$ or worse. Forward-pointer to efficient variants. |
| 1.7 | Jackknife+ and CV+ | Statement + coverage bound. Code sketch. |
| 1.8 | Conformalized quantile regression | CQR algorithm. The bridge to Topic 2. Worked example. |
| 1.9 | Conditional coverage: what's possible and what isn't | APS (for classification), CQR (for regression) as approximate-conditional approaches. Impossibility result. |
| 1.10 | Non-exchangeable data: drift, covariate shift, time series | Statement of the reweighted-conformal bound (Tibshirani-Barber-Candes-Ramdas 2019). Conformal for time series (Chernozhukov et al. 2021). Forward-map to future formalml topics. |

### §3.9 Featured theorem + featured component

- **Featured theorem:** §1.4 — marginal coverage guarantee (the reason this topic exists).
- **Featured component:** `ConformalPredictionExplorer`. Sliders for $\alpha$ (0.01–0.5), $n_{\text{cal}}$ (50–5000), base-model class (linear regression / random forest/kernel regression). Display: scatter of $(X, Y)$ data + fitted model + conformal prediction band. Readouts: empirical coverage on a fresh test set, average interval width, interval-width-vs-$X$ curve (showing homo- or heteroscedasticity). "Resample" button advances seed.

### §3.10 Additional interactive component ideas

- **CQRExplorer.** Heteroscedastic synthetic regression (noise variance increases with $|X|$). Side-by-side: (a) naive QR prediction interval (no coverage guarantee), (b) CQR-corrected interval (guaranteed $\geq 1 - \alpha$ coverage, wider where calibration was off). Live empirical coverage readout on both.
- **ExchangeabilityBreakdown.** Show what happens when exchangeability fails: simulate covariate shift with a slider for shift magnitude; plot empirical coverage (which drops below nominal). Demonstrates that the guarantee is assumption-dependent.
- **APSForClassification.** Three-class classification with a softmax model; APS prediction sets shown on a 2D decision-boundary grid. Color-coded by set size; highlights regions where the classifier is ambiguous.

### §3.11 Reference shortlist

**Primary textbook/monograph:**
- Vovk, Vladimir, Alexander Gammerman, and Glenn Shafer. 2005. *Algorithmic Learning in a Random World*. Springer. ISBN 978-0387001524. The definitive book-length treatment, primary reference for full-conformal machinery.

**Accessible introductions:**
- Angelopoulos, Anastasios N., and Stephen Bates. 2023. *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*. Foundations and Trends in Machine Learning 16(4): 494–591. [https://doi.org/10.1561/2200000101](https://doi.org/10.1561/2200000101). **The best modern tutorial**; align exposition with this.
- Shafer, Glenn, and Vladimir Vovk. 2008. "A Tutorial on Conformal Prediction." *Journal of Machine Learning Research* 9: 371–421. [https://jmlr.org/papers/v9/shafer08a.html](https://jmlr.org/papers/v9/shafer08a.html).

**Foundational / method papers:**
- Lei, Jing, Max G'Sell, Alessandro Rinaldo, Ryan J. Tibshirani, and Larry Wasserman. 2018. "Distribution-Free Predictive Inference for Regression." *JASA* 113(523): 1094–1111. [https://doi.org/10.1080/01621459.2017.1307116](https://doi.org/10.1080/01621459.2017.1307116). Split conformal for regression, definitive.
- Romano, Yaniv, Evan Patterson, and Emmanuel Candès. 2019. "Conformalized Quantile Regression." *NeurIPS 2019*. [https://papers.nips.cc/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html](https://papers.nips.cc/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html). CQR paper.
- Romano, Yaniv, Matteo Sesia, and Emmanuel Candès. 2020. "Classification with Valid and Adaptive Coverage." *NeurIPS 2020*. [https://papers.nips.cc/paper/2020/hash/244edd7e85dc81602b7615cd705545f5-Abstract.html](https://papers.nips.cc/paper/2020/hash/244edd7e85dc81602b7615cd705545f5-Abstract.html). APS for classification.
- Barber, Rina Foygel, Emmanuel J. Candès, Aaditya Ramdas, and Ryan J. Tibshirani. 2021. "Predictive Inference with the Jackknife+." *Annals of Statistics* 49(1): 486–507. [https://doi.org/10.1214/20-AOS1965](https://doi.org/10.1214/20-AOS1965). Jackknife+.
- Barber, Rina Foygel, Emmanuel J. Candès, Aaditya Ramdas, and Ryan J. Tibshirani. 2021. "The Limits of Distribution-Free Conditional Predictive Inference." *Information and Inference* 10(2): 455–482. [https://doi.org/10.1093/imaiai/iaaa017](https://doi.org/10.1093/imaiai/iaaa017). Conditional-coverage impossibility.
- Tibshirani, Ryan J., Rina Foygel Barber, Emmanuel J. Candès, and Aaditya Ramdas. 2019. "Conformal Prediction Under Covariate Shift." *NeurIPS 2019*. [https://papers.nips.cc/paper/2019/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html](https://papers.nips.cc/paper/2019/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html).
- Chernozhukov, Victor, Kaspar Wüthrich, and Yinchu Zhu. 2021. "Exact and Robust Conformal Inference Methods for Predictive Machine Learning with Dependent Data." *PNAS* 118(48). [https://doi.org/10.1073/pnas.2107794118](https://doi.org/10.1073/pnas.2107794118). Time-series conformal.

**Applications / recent:**
- Angelopoulos, Anastasios N., Stephen Bates, Jitendra Malik, and Michael I. Jordan. 2021. "Uncertainty Sets for Image Classifiers using Conformal Prediction." *ICLR 2021*. Worked example material.

### §3.12 Difficulty/length target

- **Difficulty:** intermediate.
- **Read time:** 60 min (~11K words).
- **Full proofs:** 2 (§1.4 marginal coverage, §1.4 upper bound — both short).
- **Stated theorems:** 3 (Jackknife+ bound, conditional-impossibility, CQR inheritance).
- **Examples:** 12–14.
- **Remarks:** 18–22.
- **Figures:** 9 (calibration-score histogram + threshold; prediction-interval scatter at 3 base models; empirical coverage vs $n_{\text{cal}}$; CQR vs naive QR; APS decision-boundary; covariate-shift coverage drop; time-series conformal trace; Topic-1 forward-map).
- **Interactive components:** 3–4 (ConformalPredictionExplorer, CQRExplorer, APSForClassification; optional ExchangeabilityBreakdown).

### §3.13 Open questions for Jonathan before implementing

1. How deep on the computational side? Pure-JS conformal-wrapper components can handle small data; for the "wrap my random forest" demo you'd need a JS random-forest library or a precomputed fixture. Lean on fixtures and skip the in-browser fitting?
2. Which formalml topics does Conformal Prediction feed into? (Knowing the downstream graph helps frame §1.10.)
3. Is there a formalml "online learning" topic in scope? If yes, online conformal goes there; if no, §1.10 stays terminal.

---

## § 4 Topic 2 — Quantile Regression

### §4.1 Slug & title

- **Slug:** `quantile-regression`
- **Title:** "Quantile Regression"
- **Subtitle (suggested):** "Koenker-Bassett check-loss regression and conditional distribution modeling."

### §4.2 One-sentence pitch

Generalize OLS-for-the-conditional-mean to a regression framework that estimates the conditional $\tau$-th quantile $Q_\tau(Y \mid X = x)$ for any $\tau \in (0, 1)$, via minimization of the check loss $\rho_\tau(u) = u(\tau - \mathbb{1}\{u < 0\})$.

### §4.3 Why this topic

Quantile regression is the natural covariate-conditional extension of Topic 29 §29.4 + §29.6: the sample quantile $\hat\xi_p$ is the intercept-only special case. Koenker–Bassett (1978) showed that $\hat\beta_\tau = \arg\min \sum_i \rho_\tau(Y_i - X_i^\top \beta)$ is a valid estimator of the conditional-quantile coefficients with closed-form asymptotic distribution. Modern applications include risk management (value-at-risk), distributional regression (modeling not just the mean but entire conditional distribution shape), robust statistics (the median-regression special case $\tau = 0.5$ is natively robust to heavy tails), and — most directly for ML — conformalized quantile regression (Topic 1 §3.8).

### §4.4 Scope — IN

- **The check loss and its motivations.** $\rho_\tau(u) = u(\tau - \mathbb{1}\{u < 0\})$. Piecewise-linear, non-differentiable at 0. Interpretation as the "pinball loss" in forecasting. Generalization of absolute error ($\tau = 0.5$).
- **The quantile-regression estimator.** $\hat\beta_\tau = \arg\min \sum \rho_\tau(Y_i - X_i^\top \beta)$. Existence (always exists; non-unique in degenerate cases). Optimality conditions.
- **Computation.** Linear-programming formulation (standard simplex / revised simplex, Koenker-d'Orey 1987). Interior-point methods (Portnoy-Koenker 1997) for large $n$. Computational cost analysis. Modern fast solvers (e.g., `quantreg::rq` in R, `statsmodels.QuantReg` in Python).
- **Asymptotic theory.** $\sqrt n (\hat\beta_\tau - \beta_\tau) \Rightarrow \mathcal{N}(0, \tau(1-\tau) \cdot (\mathbb{E}[f_{Y|X}(Q_\tau(Y|X)|X) X X^\top])^{-1} \cdot \mathbb{E}[X X^\top] \cdot (\mathbb{E}[f_{Y|X}(Q_\tau(Y|X)|X) X X^\top])^{-1})$. Sandwich form. Bahadur-representation derivation — the direct extension of Topic 29 §29.6's scalar Bahadur to the regression case.
- **Inference.** Standard errors via sandwich estimator (density estimation required — Topic 30 KDE once shipped, or Hendricks-Koenker 1992 kernel rule as interim). Rank-score tests. Quantile-specific bootstrap.
- **Multiple quantiles and the crossing problem.** Fitting $\hat\beta_\tau$ at several $\tau$ values can give estimates that violate the monotonicity $\hat Q_{\tau_1} \leq \hat Q_{\tau_2}$ for $\tau_1 < \tau_2$. Rearrangement (Chernozhukov-Fernández-Val-Galichon 2010) as the standard fix, stated with reference.
- **Regularized quantile regression.** $\ell_1$-penalized QR (Belloni-Chernozhukov 2011); connection to formalstatistics Topic 23 (regularization).
- **Nonlinear/nonparametric extensions.** Quantile regression forests (Meinshausen 2006). Neural quantile regression. Brief treatments with pointers.
- **ML applications.** Distributional forecasting (weather, demand, financial returns). Heteroscedasticity-aware prediction intervals (feeds Topic 1 §3.8 CQR). Median regression for heavy-tailed ML noise.

### §4.5 Scope — OUT

- **Censored / truncated quantile regression** (Powell 1986; Portnoy 2003) — §4.10 pointer only.
- **Bayesian quantile regression** (Yu-Moyeed 2001 asymmetric Laplace trick) — §4.10 pointer; formalstatistics Topic 25 territory if revisited.
- **Instrumental-variable quantile regression** — pointer to Chernozhukov-Hansen 2006 literature.
- **Composite quantile regression** (Zou-Yuan 2008) — pointer.
- **Infinite-dimensional quantile regression / functional-response quantile regression** — pointer.

### §4.6 Primary predecessor chain

- **Topic 29 §29.4 + §29.6 (formalstatistics)** — sample-quantile Type-7 definition + Bahadur representation. Topic 2 §4 opens with: *"Topic 29 §29.6 established that the sample quantile $\hat\xi_p$ satisfies a Bahadur representation $\hat\xi_p - \xi_p = -(F_n(\xi_p) - p)/f(\xi_p) + o_p(n^{-1/2})$. Quantile regression generalizes this: replace 'sample' with 'conditional sample given $X = x$,' parametrize $Q_\tau(Y|X) = X^\top \beta_\tau$, and the Bahadur representation survives almost unchanged — with the 'density at the quantile' $f(\xi_p)$ replaced by the conditional density $f_{Y|X}(Q_\tau(Y|X)|X)$ averaged against $X X^\top$."*
- **Topic 15 §15.9 (formalstatistics)** — check-loss as M-estimator / Z-estimator. Topic 2 picks this up: the M-estimation framework is the asymptotic-theory scaffolding for §4.4 Thm 2 (asymptotic normality of $\hat\beta_\tau$).
- **Topic 21 (formalstatistics, Linear Regression)** — the OLS analog. Every claim made in Topic 21 about $\hat\beta_{\text{OLS}}$ (asymptotic normality, sandwich estimator under misspecification, F-test for nested models) has a quantile-regression counterpart. Topic 2 structures in parallel to Topic 21.
- **Topic 29 §29.10 Rem 19 (formalstatistics)** — explicit forward-pointer: *"Topic 29 handles the scalar-intercept case; quantile regression is the covariate-conditional extension."*
- **Topic 30 (formalstatistics, KDE, forthcoming)** — once shipped, density estimation for the asymptotic-variance sandwich is a direct KDE application.
- **Topic 1 (formalml, Conformal Prediction)** — CQR cross-reference (Topic 2 is where the QR machinery lives; Topic 1 is where the coverage wrapper lives).

### §4.7 Key theorems

1. **Existence and optimality of $\hat\beta_\tau$** (Koenker-Bassett 1978 Thm 3.1). Stated.
2. **Asymptotic distribution of $\hat\beta_\tau$ (featured, full proof).** Under $(X_i, Y_i)$ iid with appropriate regularity (conditional density $f_{Y|X}$ bounded and continuous at $Q_\tau(Y|X)$, $\mathbb{E}[XX^\top]$ positive-definite), $\sqrt n(\hat\beta_\tau - \beta_\tau) \Rightarrow \mathcal{N}(0, V_\tau)$ with the sandwich form above. Proof via the M-estimator framework (Huber 1967 → vdV 2000 §5). Can be rendered in ~25 MDX lines.
3. **Bahadur representation in the regression case.** $\hat\beta_\tau - \beta_\tau = D^{-1} n^{-1} \sum_i X_i (\tau - \mathbb{1}\{Y_i < X_i^\top \beta_\tau\}) + o_p(n^{-1/2})$, where $D = \mathbb{E}[f_{Y|X}(Q_\tau(Y|X)|X) X X^\top]$. Stated — full proof requires Donsker-class machinery (formalstatistics Topic 32).
4. **Equivariance.** $\hat\beta_\tau$ is equivariant under linear transformations of $Y$ and under monotone transformations of $X$ in a specific sense (Koenker 2005 §2.2). Stated.
5. **Rank-score duality.** The QR optimization dual is the rank-score process, providing a way to compute QR confidence intervals without density estimation (Gutenbrunner-Jurečková-Koenker-Portnoy 1993). Stated.

### §4.8 Suggested section structure

| § | Title | Primary content |
|---|---|---|
| 2.1 | From conditional mean to conditional quantile | Motivation. The OLS-for-$\ mathbb {E}[Y\|X]$ framework; why conditional quantiles matter (heteroscedasticity, distributional forecasting, robust regression). |
| 2.2 | The check loss | Definition of $\rho_\tau$. Geometric intuition. Recovery of absolute error at $\tau = 0.5$. Relationship to the quantile loss in forecasting (pinball loss). |
| 2.3 | The quantile-regression estimator | $\hat\beta_\tau = \arg\min \sum \rho_\tau(Y_i - X_i^\top \beta)$. Non-differentiability. Optimality conditions via subdifferential. |
| 2.4 | Computation | LP formulation. Simplex. Interior-point methods. Modern practical advice. |
| 2.5 | **Asymptotic distribution of $\hat\beta_\tau$ (featured)** | M-estimator framework. Full proof. Sandwich form. Worked example on simulated data. |
| 2.6 | Standard errors and inference | Sandwich estimator (requires density estimation, forward-pointer to Topic 30 KDE). Rank-score tests. QR bootstrap. |
| 2.7 | Multiple quantiles and the crossing problem | Crossing example. Rearrangement. Monotone rearrangement operator (Chernozhukov, Fernández, Val, Galichon, 2010). |
| 2.8 | Regularized QR | $\ell_1$-penalized QR. Lasso-QR connection. Forward-pointer to Topic 23. |
| 2.9 | Extensions | Quantile regression forests (Meinshausen 2006). Neural QR. Brief. |
| 2.10 | Forward-map: CQR, distributional regression, and beyond | Bridge to Topic 1 (CQR). Bridge to distributional regression (Chernozhukov et al. 2013). Pointer to formalml topics. |

### §4.9 Featured theorem + featured component

- **Featured theorem:** §2.5 — asymptotic distribution of $\hat\beta_\tau$ (Koenker-Bassett 1978 Thm 4). The centerpiece.
- **Featured component:** `QuantileRegressionExplorer`. 2D scatter of $(X, Y)$ from a heteroscedastic synthetic distribution. Slider for $\tau$ with default $\{0.1, 0.25, 0.5, 0.75, 0.9\}$ shown simultaneously as nested regression lines. Toggle: "show OLS" overlay (the conditional-mean line, obviously failing to track the heteroscedastic spread). Sample-size slider; seed control; CI bands toggle (sandwich SE). Readout: current $\hat\beta_\tau$ at the selected $\tau$.

### §4.10 Additional interactive component ideas

- **CheckLossComparison.** Three-panel: squared loss, absolute loss, check loss at $\tau = 0.25$. Same data (say, residuals from a regression). Shows what each loss function penalizes, and which "optimal prediction" falls out of each (mean / median / 25th percentile).
- **CrossingQuantilesFixer.** Small data example where fitted QR lines cross. Toggle "apply monotone rearrangement" to see the fixed version.
- **RankScoreExplorer.** Shows the dual rank-score process at a given $\tau$; its zero-crossings identify the QR estimate without density estimation.

### §4.11 Reference shortlist

**Primary monograph:**
- Koenker, Roger. 2005. *Quantile Regression*. Cambridge University Press. ISBN 978-0521845731. [https://doi.org/10.1017/CBO9780511754098](https://doi.org/10.1017/CBO9780511754098). Definitive reference; align Topic 2 exposition here.

**Accessible introductions:**
- Koenker, Roger, and Kevin F. Hallock. 2001. "Quantile Regression." *Journal of Economic Perspectives* 15(4): 143–156. [https://doi.org/10.1257/jep.15.4.143](https://doi.org/10.1257/jep.15.4.143). Two-hour read; ideal opener.
- Hao, Lingxin, and Daniel Q. Naiman. 2007. *Quantile Regression*. Sage Publications. ISBN 978-1412926287. Short/approachable.

**Foundational papers:**
- Koenker, Roger, and Gilbert Bassett. 1978. "Regression Quantiles." *Econometrica* 46(1): 33–50. [https://www.jstor.org/stable/1913643](https://www.jstor.org/stable/1913643). Original paper; Thm 4 is the featured-theorem source.
- Portnoy, Stephen, and Roger Koenker. 1997. "The Gaussian Hare and the Laplacian Tortoise: Computability of Squared-Error versus Absolute-Error Estimators." *Statistical Science* 12(4): 279–300. [https://doi.org/10.1214/ss/1030037960](https://doi.org/10.1214/ss/1030037960). Interior-point computation.
- Chernozhukov, Victor, Iván Fernández-Val, and Alfred Galichon. 2010. "Quantile and Probability Curves without Crossing." *Econometrica* 78(3): 1093–1125. [https://doi.org/10.3982/ECTA7880](https://doi.org/10.3982/ECTA7880). Monotone rearrangement.
- Chernozhukov, Victor, Iván Fernández-Val, and Blaise Melly. 2013. "Inference on Counterfactual Distributions." *Econometrica* 81(6): 2205–2268. [https://doi.org/10.3982/ECTA10582](https://doi.org/10.3982/ECTA10582). Distributional regression framework.

**ML-practical / nonparametric extensions:**
- Meinshausen, Nicolai. 2006. "Quantile Regression Forests." *Journal of Machine Learning Research* 7: 983–999. [https://jmlr.org/papers/v7/meinshausen06a.html](https://jmlr.org/papers/v7/meinshausen06a.html). Standard forest-based QR reference.
- Belloni, Alexandre, and Victor Chernozhukov. 2011. "$\ell_1$-Penalized Quantile Regression in High-Dimensional Sparse Models." *Annals of Statistics* 39(1): 82–130. [https://doi.org/10.1214/10-AOS827](https://doi.org/10.1214/10-AOS827). Regularized QR.

### §4.12 Difficulty/length target

- **Difficulty:** intermediate.
- **Read time:** 55 min (~10K words).
- **Full proofs:** 1 (§2.5 asymptotic normality of $\hat\beta_\tau$).
- **Stated theorems:** 4 (existence, Bahadur representation, equivariance, rank-score duality).
- **Examples:** 10–12.
- **Remarks:** 15–20.
- **Figures:** 8 (check-loss shapes; multi-quantile regression fit on heteroscedastic data; Bahadur representation visualization; sandwich-SE comparison; crossing-quantiles + rearrangement; QR forest vs linear QR; distributional regression fan; Topic-2 forward map).
- **Interactive components:** 2–3 (QuantileRegressionExplorer, CheckLossComparison; optional CrossingQuantilesFixer).

### §4.13 Open questions for Jonathan

1. Does formalml have Topic 21 (Linear Regression) already, or does Topic 2 need to do more self-containment on the OLS backdrop? If Topic 21 exists on formalml or formalstatistics is a hard prerequisite, §2 can be more terse on framework.
2. Computational-demos depth: pure-JS LP solver is achievable but nontrivial; precompute fixtures or implement lightweight simplex in `quantile-regression.ts`?
3. Treatment of quantile-regression forests — substantial sub-topic or brief pointer? My recommendation: brief pointer + one paragraph, defer full treatment to a later "tree-based methods" formalml topic if such exists.

---

## § 5 Topic 3 — Rank Tests

### §5.1 Slug & title

- **Slug:** `rank-tests`
- **Title:** "Rank Tests"
- **Subtitle (suggested):** "Wilcoxon, Mann–Whitney, Kruskal–Wallis, and permutation inference"

### §5.2 One-sentence pitch

Nonparametric hypothesis tests that replace raw data values with ranks — distribution-free under the null, computable via the exact permutation distribution, and asymptotically efficient with only a small Pitman-ARE penalty relative to the parametric $z$ / $t$ / $F$ trio.

### §5.3 Why this topic

Rank tests are the classical nonparametric answer to "I want to test a location hypothesis but I don't want to assume Normal errors." They sit squarely in Track 8 territory (distribution-free) but were explicitly deferred from formalstatistics Topic 29 because (a) permutation-distribution machinery is orthogonal to the Bahadur-asymptotics machinery that Topic 29 centers on; (b) the natural home for exchangeability-based inference is alongside conformal prediction (Topic 1) on formalml; (c) rank tests are more "inference procedures" than "foundational objects" and belong adjacent to ML-practical formalml topics. Topic 3 covers the classical trio (Wilcoxon signed-rank, Mann-Whitney U, Kruskal-Wallis H) with permutation-distribution foundations + Pitman-ARE efficiency analysis + Hodges-Lehmann point estimation.

### §5.4 Scope — IN

- **The rank transform.** Given a sample, the vector of ranks. Properties: distribution-free under any continuous null with permutation-symmetric sampling distribution.
- **Wilcoxon signed-rank test** (one-sample). Test statistic, null distribution (exact via permutation for small $n$, Normal asymptotic for large). Power against location alternatives.
- **Mann–Whitney U / Wilcoxon rank-sum test** (two-sample). The U-statistic formulation. Relationship to the rank-sum formulation (they are linearly related). Null distribution.
- **Kruskal–Wallis H test** ($k$-sample). Generalization of rank-sum to multiple groups. Asymptotic $\chi^2_{k-1}$ distribution under the null.
- **Permutation-distribution machinery.** The general framework: any test statistic + permutation of labels → exact null distribution. Computation: exhaustive for small $n$, Monte Carlo for large. Proof of exactness: exchangeability under the null.
- **Pitman asymptotic relative efficiency (ARE).** The efficiency of a rank test relative to a parametric competitor at a local alternative. Classical result: $\mathrm{ARE}(\text{Wilcoxon rank-sum}, t\text{-test}) = 3/\pi \approx 0.955$ at Normal alternatives (tiny loss); $> 1$ at heavier-tailed alternatives (Laplace, Cauchy); $\infty$ at Cauchy (since $t$-test has undefined asymptotic size).
- **Hodges-Lehmann estimator.** The median-of-pairwise-differences estimator of the location-shift between two samples; robust, distribution-free, asymptotically equivalent to the Mann-Whitney-inverted confidence interval.
- **Spearman rho and Kendall tau** (brief). Rank-based correlation measures. Distribution-free test of independence.
- **ML connections.** Permutation tests in feature importance (Breiman 2001 random forests), A/B testing with nonparametric null hypotheses, robust outlier detection via ranks.

### §5.5 Scope — OUT

- **Aligned-rank tests** (Hodges-Lehmann 1962; for two-way layouts) — pointer only.
- **Semiparametric efficient tests** (Hájek's asymptotically best-ranking) — pointer; depth at Hájek-Šidák 1967.
- **Rank methods for censored / survival data** (log-rank test, Peto test) — formalml "survival analysis" topic if such exists; pointer.
- **Robust rank-based regression** (rank-based M-estimators of Jurečková-Sen 1996) — pointer.
- **Permutation tests for complex hypotheses** (ANOVA, linear models via permutation) — brief §3.10 remark; defer deep treatment.

### §5.6 Primary predecessor chain

- **Topic 17 (formalstatistics, Hypothesis Testing)** — hard prerequisite. Null / alternative / rejection region / size / power framework. Topic 3 inherits all this terminology.
- **Topic 18 (formalstatistics, LRT + Neyman-Pearson) §18.10 Rem 25** — explicitly names rank tests as Track 8 territory. Topic 3 discharges this.
- **Topic 29 §29.10 Rem 20 (formalstatistics)** — forward-pointer: *"Wilcoxon, Mann-Whitney, Kruskal-Wallis are nonparametric alternatives to the $z/t/\chi^2$ trio. The Pitman ARE framework (Remark 20) is the bridge — nonparametric tests pay a constant efficiency penalty in exchange for distributional robustness."* Topic 3 is where this pointer lands.
- **Topic 29 §29.2 (formalstatistics)** — rank transform relates to the order-statistic framework; the ranks are the indices of the order-statistic permutation. Light cross-reference.
- **Topic 1 (formalml, Conformal Prediction)** — permutation / exchangeability connections. Topic 3 opens with a one-paragraph remark: *"Conformal prediction (Topic 1) exploits exchangeability to produce a finite-sample coverage guarantee. Rank tests exploit exchangeability to produce a finite-sample test size guarantee. The machinery is the same — we permute labels under the null, and the rank of the test statistic gives us a $p$-value."*

### §5.7 Key theorems

1. **Distribution-freeness of ranks under continuous null.** If $X_1, \dots, X_n$ iid from any continuous distribution, the rank vector is uniformly distributed over permutations of $\{1, \dots, n\}$. Full proof via probability-integral transform argument (exactly the Topic 29 §29.8 trick).
2. **Asymptotic distribution of the Wilcoxon rank-sum statistic (featured, full proof).** Under $H_0$: $U \Rightarrow \mathcal{N}(n_1 n_2 / 2, n_1 n_2 (n_1 + n_2 + 1) / 12)$ after standardization. Proof via Hoeffding's theorem for $U$-statistics. Can be rendered in ~20 MDX lines.
3. **Pitman ARE of Wilcoxon vs $t$-test.** For location-shift alternatives $\theta = c/\sqrt n$, $\mathrm{ARE}(\text{Wilcoxon}, t) = 12 \sigma_F^2 (\int f^2)^2$, where $\sigma_F^2$ is the parent variance and $\int f^2$ is the integrated squared density. At Normal parent: $\mathrm{ARE} = 3/\pi \approx 0.955$. Stated with derivation sketch.
4. **Asymptotic $\chi^2$ distribution of Kruskal-Wallis $H$** under $H_0$. Stated.
5. **Consistency of the Hodges-Lehmann estimator.** $\hat\Delta_{\mathrm{HL}} \to \Delta$ in probability, with root-$n$ asymptotic normality. Stated.

### §5.8 Suggested section structure

| § | Title | Primary content |
|---|---|---|
| 3.1 | Motivation: distribution-free testing | When Normal-theory fails. Heavy tails, contaminated distributions. The rank-based alternative. |
| 3.2 | The rank transform | Definition. Distribution-freeness under continuous null (full proof). |
| 3.3 | Wilcoxon signed-rank test (one sample) | Statistic. Exact distribution. Normal asymptotic. Worked example. |
| 3.4 | **Mann-Whitney U / Wilcoxon rank-sum (featured, asymptotic proof)** | U-statistic formulation. Full proof of asymptotic normality under $H_0$. The rank-sum ↔ U-statistic duality. |
| 3.5 | Kruskal-Wallis H-test | Generalization to $k$ samples. $\chi^2_{k-1}$ asymptotic. |
| 3.6 | Permutation-distribution machinery | General framework. Exchangeability → exact test. Exhaustive vs Monte Carlo computation. |
| 3.7 | Pitman ARE | The efficiency comparison. Classical results ($3/\pi$ at Normal, $> 1$ at Laplace). Derivation sketch. |
| 3.8 | The Hodges-Lehmann estimator | Median-of-pairwise-differences. Inverting the Mann-Whitney test for a CI on the shift. |
| 3.9 | Rank-based correlation | Spearman rho, Kendall tau. Brief treatment. |
| 3.10 | Forward-map: rank methods in ML and beyond | Feature importance. Permutation A/B testing. Pointer to survival analysis, aligned ranks, semiparametric efficient ranks. |

### §5.9 Featured theorem + featured component

- **Featured theorem:** §3.4 — asymptotic distribution of Mann-Whitney U under $H_0$, via Hoeffding's theorem for U-statistics.
- **Featured component:** `PermutationDistributionExplorer`. Input: two small samples (e.g., $n_1 = 5, n_2 = 6$, sliders). Display: the exact permutation distribution of the rank-sum statistic (all $\binom{11}{5} = 462$ permutations enumerated and histogrammed). Overlay: observed test statistic. Readout: exact two-sided $p$-value. Toggle: "show Normal approximation" to contrast the exact distribution with $\mathcal{N}(\mu, \sigma^2)$.

### §5.10 Additional interactive component ideas

- **PitmanAREcalculator.** Sliders for parent distribution (Normal / Laplace / Cauchy / Uniform / Logistic / mixture) and effect size. Computes and displays $\mathrm{ARE}(\text{Wilcoxon rank-sum}, t\text{-test})$. Shows where rank tests *win* (heavy tails) vs where they *lose* slightly ($3/\pi$ at Normal).
- **HodgesLehmannEstimatorExplorer.** Two samples on parallel number lines. Drag-and-drop to adjust values. Live readout of $\hat\Delta_{\mathrm{HL}} = \mathrm{median}\{Y_j - X_i : i = 1, \dots, n_1; j = 1, \dots, n_2\}$. Compare with the sample-mean-difference $\bar Y - \bar X$; show robustness to outliers.

### §5.11 Reference shortlist

**Primary textbooks:**
- Hollander, Myles, Douglas A. Wolfe, and Eric Chicken. 2014. *Nonparametric Statistical Methods*. 3rd ed. Wiley. ISBN 978-0470387375. **Primary reference**; align Topic 3 exposition here.
- Hettmansperger, Thomas P., and Joseph W. McKean. 2011. *Robust Nonparametric Statistical Methods*. 2nd ed. CRC Press. ISBN 978-1439809082. Secondary; good on efficient-rank methods.
- Lehmann, Erich L. 1975. *Nonparametrics: Statistical Methods Based on Ranks*. Holden-Day (later reprints). ISBN 978-0387352121 (Springer revised ed. 2006). Classic foundational text.

**Foundational papers:**
- Wilcoxon, Frank. 1945. "Individual Comparisons by Ranking Methods." *Biometrics Bulletin* 1(6): 80–83. [https://doi.org/10.2307/3001968](https://doi.org/10.2307/3001968). Rank-sum origin paper.
- Mann, Henry B., and Donald R. Whitney. 1947. "On a Test of Whether one of Two Random Variables is Stochastically Larger than the Other." *Annals of Mathematical Statistics* 18(1): 50–60. [https://doi.org/10.1214/aoms/1177730491](https://doi.org/10.1214/aoms/1177730491). Mann-Whitney U.
- Kruskal, William H., and W. Allen Wallis. 1952. "Use of Ranks in One-Criterion Variance Analysis." *JASA* 47(260): 583–621. [https://doi.org/10.1080/01621459.1952.10483441](https://doi.org/10.1080/01621459.1952.10483441). Kruskal-Wallis.
- Hodges, Joseph L., and Erich L. Lehmann. 1963. "Estimates of Location Based on Rank Tests." *Annals of Mathematical Statistics* 34(2): 598–611. [https://doi.org/10.1214/aoms/1177704172](https://doi.org/10.1214/aoms/1177704172). Hodges-Lehmann estimator.
- Hájek, Jaroslav, and Zbyněk Šidák. 1967. *Theory of Rank Tests*. Academic Press. ISBN 978-0126423501. Deep reference for asymptotic efficiency.

**Permutation-testing foundations:**
- Good, Phillip I. 2005. *Permutation, Parametric, and Bootstrap Tests of Hypotheses*. 3rd ed. Springer. ISBN 978-0387202792. Permutation-testing framework.
- Lehmann, Erich L., and Joseph P. Romano. 2005. *Testing Statistical Hypotheses*. 3rd ed. Springer. ISBN 978-0387988641. Ch. 15 on permutation tests; reused from formalstatistics.

### §5.12 Difficulty/length target

- **Difficulty:** intermediate.
- **Read time:** 55 min (~10K words).
- **Full proofs:** 2 (§3.2 rank distribution-freeness; §3.4 asymptotic normality of U).
- **Stated theorems:** 3 (Kruskal-Wallis $\chi^2$; Pitman ARE; Hodges-Lehmann consistency).
- **Examples:** 10–12.
- **Remarks:** 15–18.
- **Figures:** 8 (permutation-distribution exact vs Normal approx; ARE curve across distributions; HL estimator on contaminated data; rank-sum vs $t$-test power curves; correlation measures comparison; feature-importance permutation example; Topic-3 forward-map).
- **Interactive components:** 2–3 (PermutationDistributionExplorer, PitmanAREcalculator; optional HodgesLehmannEstimatorExplorer).

### §5.13 Open questions

1. Does formalml have a feature-importance / explainability topic where permutation-importance would be the natural home? If yes, Topic 3 §3.10 stays as a pointer; if no, include a fuller treatment in §3.10 or a companion topic.
2. How much Pitman-ARE rigor? A full proof requires setting up the Le Cam / contiguous-alternatives framework. My recommendation: state the result, give a derivation sketch, cite Hájek-Šidák for the full proof, include the classical numerical table (Normal / Laplace / Cauchy / Logistic / Uniform) as a figure.
3. Two-sample aligned-ranks and repeated-measures (Friedman test) — include or pointer? My recommendation: one-paragraph pointer to Friedman 1937 + Hollander-Wolfe-Chicken Ch. 7.

---

## § 6 Topic 4 — Extreme Value Theory

### §6.1 Slug & title

- **Slug:** `extreme-value-theory`
- **Title:** "Extreme Value Theory."
- **Subtitle (suggested):** "The asymptotic behavior of sample maxima and tail-based inference."

### §6.2 One-sentence pitch

When the CLT describes what happens at the *center* of the distribution, EVT describes what happens at the *tail* — the Fisher-Tippett-Gnedenko trichotomy (Gumbel / Fréchet / Weibull) classifies all possible limit distributions of the properly normalized sample maximum, and the peaks-over-threshold framework extends this to general tail modeling via the generalized Pareto distribution.

### §6.3 Why this topic

EVT answers the question that formalstatistics Topic 29 §29.10 Rem 18 explicitly defers: *"What happens to the maximum $X_{(n)}$ after proper centering and scaling?"* Topic 29's Bahadur representation describes $\hat\xi_p$ at interior $p \in (0, 1)$; EVT describes $\hat\xi_p$ at $p \to 1$ and extends this to the modeling of exceedances. ML applications: tail-risk quantification for deployed models (the 99.9th percentile of prediction error), anomaly / OOD detection (modeling the tail of score distributions), heavy-tail-aware training (distribution of worst-case losses). Classical applications that formalml should motivate: financial risk (Value-at-Risk, Expected Shortfall), hydrology (return periods for floods), climate science (extreme heat events), reliability engineering.

### §6.4 Scope — IN

- **The block-maxima approach.** Given iid $X_1, \dots, X_n$ with CDF $F$, consider $M_n = \max(X_1, \dots, X_n)$. Its CDF is $F^n$, which degenerates as $n \to \infty$. Normalize: find sequences $a_n > 0, b_n \in \mathbb{R}$ such that $P((M_n - b_n)/a_n \leq x) \to G(x)$ for a non-degenerate $G$.
- **Fisher-Tippett-Gnedenko trichotomy (featured).** The only possible non-degenerate limits $G$ are (up to location/scale):
  1. **Gumbel:** $\Lambda(x) = \exp(-e^{-x})$, $x \in \mathbb{R}$. Light-tailed parents (Normal, Exponential, Gamma).
  2. **Fréchet:** $\Phi_\alpha(x) = \exp(-x^{-\alpha})$, $x > 0$. Heavy-tailed parents (Pareto, Cauchy, $t_\nu$).
  3. **Weibull:** $\Psi_\alpha(x) = \exp(-(-x)^\alpha)$, $x < 0$. Bounded-support parents (Uniform, Beta).
- **The generalized extreme value (GEV) family.** Unified parametrization: $G_\xi(x) = \exp(-(1 + \xi x)^{-1/\xi})$ for $\xi \neq 0$, $G_0(x) = \exp(-e^{-x})$. The shape parameter $\xi$ controls: $\xi > 0$ → Fréchet, $\xi = 0$ → Gumbel, $\xi < 0$ → Weibull.
- **Domain of attraction.** Which parent CDFs $F$ get mapped to which GEV family $G_\xi$? Criterion: regular-variation / von Mises conditions. Stated with examples.
- **The peaks-over-threshold (POT) approach.** Instead of block maxima, consider exceedances over a high threshold $u$: $\{X_i - u: X_i > u\}$. Pickands–Balkema–de Haan theorem: conditional on $X > u$, as $u \to u^* = \sup F$, $(X - u) \mid X > u$ converges to the **generalized Pareto distribution** $H_{\xi, \beta}$.
- **Tail-index estimation.** Hill estimator (Hill 1975), Pickands estimator (Pickands 1975), moment estimator (Dekkers-Einmahl-de Haan 1989). Properties: consistency under regularity, asymptotic normality rates.
- **Return levels/return periods.** The $T$-year return level $x_T$ is the $1 - 1/T$ quantile of the annual-maxima distribution. Estimation from fitted GEV.
- **Expected Shortfall / Tail Value-at-Risk.** $\mathrm{ES}_\alpha = \mathbb{E}[X \mid X > \mathrm{VaR}_\alpha]$. Estimation via fitted GPD in the tail.
- **Modern ML connections.** Out-of-distribution detection via extreme-value score modeling (e.g., the EVT-backed extension of the classical softmax baseline). Heavy-tail-aware gradient-clipping thresholds. Tail-risk quantification for deployed models.
- **Sketch of multivariate EVT.** Component-wise maxima; dependence structure via the *extreme-value copula* / *spectral measure*. Brief — one section pointing to Resnick 1987 for depth.

### §6.5 Scope — OUT

- **Stochastic-process extremes / extremes of dependent sequences** (Leadbetter-Lindgren-Rootzén 1983 extremal index) — §4.10 pointer.
- **Spatial-extremes / max-stable processes** (Coles 2001 Ch. 9, Davison-Padoan-Ribatet 2012) — pointer.
- **Bayesian EVT** (Coles-Tawn 1996) — pointer; formalstatistics Topic 25 territory if cross-referenced.
- **Deep learning for EVT** — brief § 4.10 remark; pointer to recent literature (Schotanus et al. 2021).
- **Extreme-value VaR / ES backtesting methodology** — brief application, full treatment in formalml risk-management topic if such exists.

### §6.6 Primary predecessor chain

- **Topic 29 §29.10 Rem 18 (formalstatistics)** — the direct forward-pointer: *"Extreme value theory. Fisher-Tippett-Gnedenko, Gumbel/Fréchet/Weibull, max-stability, generalized Pareto. Forthcoming formalml."* Topic 4 discharges this.
- **Topic 11 (formalstatistics, CLT)** — framework for "what does the limit distribution of a normalized sample statistic look like?" EVT is the analog at the extreme. Topic 4 opens with: *"Topic 11 (CLT) and Topic 4 (EVT) are companions: CLT describes the limit of the normalized sample mean, EVT describes the limit of the normalized sample maximum. Both reduce an infinite-dimensional limit-distribution problem to a 3-parameter family — in CLT's case $\mathcal{N}(\mu, \sigma^2)$ with mean and variance; in EVT's case $\mathrm{GEV}_{\xi, \mu, \sigma}$ with shape, location, scale."*
- **Topic 29 (formalstatistics)** — order-statistic framework. The sample maximum $M_n = X_{(n)}$ is the extreme order statistic; EVT continues the Topic 29 story into the tail regime.
- **Topic 32 (formalstatistics, Empirical Processes, forthcoming)** — Donsker / functional-CLT framework. Needed in spots; can be stated and cited where it appears.

### §6.7 Key theorems

1. **Fisher-Tippett-Gnedenko trichotomy (featured, partial proof).** Statement in full. Proof outline: (a) max-stability of any non-degenerate limit; (b) max-stability characterizes the GEV family; (c) the three cases $\xi > 0, \xi = 0, \xi < 0$. Full proof requires the convergence-of-types lemma and some regular-variation theory; in the topic, we sketch the key steps and cite Embrechts-Klüppelberg-Mikosch 1997 §3.2 for full detail.
2. **Domain-of-attraction characterization.** For each of the three cases, a necessary-and-sufficient condition on the parent $F$ in terms of the tail function $1 - F(x)$. Stated — the proofs are long and technical; point to Resnick 1987 §0.
3. **Pickands-Balkema-de Haan theorem.** Conditional on $X > u$, $(X - u) \mid X > u$ converges in distribution to the generalized Pareto $H_{\xi, \beta(u)}$ as $u \to u^*$ if and only if $F$ is in the domain of attraction of $G_\xi$. Stated; proof via the GEV-GPD duality, outlined.
4. **Hill estimator asymptotics.** Under $F$ in the Fréchet domain with tail-index $\alpha > 0$, $\sqrt{k_n} (\hat\alpha_{\mathrm{Hill}} - \alpha) \Rightarrow \mathcal{N}(0, \alpha^2)$ for an appropriate intermediate sequence $k_n$. Stated.
5. **MLE for GEV / GPD.** Under shape $\xi > -1/2$, the MLE satisfies the usual asymptotic normality with standard Fisher-information variance. Stated, regularity is non-trivial — at $\xi = -1$ boundary, MLE issues appear.

### §6.8 Suggested section structure

| § | Title | Primary content |
|---|---|---|
| 4.1 | Motivation: tail risk and the inadequacy of central limits | ML risk scenarios where the tail matters. Why CLT doesn't help for $X_{(n)}$. The target object: $P((M_n - b_n)/a_n \leq x)$. |
| 4.2 | Max-stability | Definition. $G$ is max-stable iff $G^n(a_n x + b_n) = G(x)$ for some sequences. Full proof that a non-degenerate limit must be max-stable. |
| 4.3 | **Fisher-Tippett-Gnedenko trichotomy (featured)** | Statement + the three families (Gumbel, Fréchet, Weibull). Unified GEV parametrization via shape $\xi$. Partial proof sketch. |
| 4.4 | Domain of attraction | Which $F$ maps to which GEV? Regular-variation criterion. Examples: Normal → Gumbel, Pareto → Fréchet, Uniform → Weibull. |
| 4.5 | Block-maxima inference | Fit GEV to annual / block maxima via MLE. Return levels, return periods. Worked example on simulated extremes. |
| 4.6 | Peaks-over-threshold and the GPD | Pickands-Balkema-de Haan theorem. Threshold selection (mean-excess plot). GPD MLE. |
| 4.7 | Tail-index estimation | Hill, Pickands, moment estimators. Bias-variance tradeoff in threshold / $k_n$ choice. Asymptotic normality. |
| 4.8 | Value-at-Risk and Expected Shortfall | Classical risk-management application. POT-based ES estimation. |
| 4.9 | ML connections: OOD detection, tail-risk quantification | OOD detection via extreme-score modeling. Worst-case performance bounds via GEV modeling of error tails. |
| 4.10 | Forward-map: dependent extremes, spatial extremes, Bayesian EVT | Pointers only. The extremal index for non-iid. Max-stable processes. Bayesian approaches. |

### §6.9 Featured theorem + featured component

- **Featured theorem:** §4.3 — Fisher-Tippett-Gnedenko trichotomy.
- **Featured component:** `GEVFamilyExplorer`. Central: PDF of GEV$_{\xi, \mu, \sigma}$ plotted. Sliders for $\xi$ (shape, $-1$ to $+1$), $\mu$ (location), $\sigma$ (scale). The shape slider has three colored zones: red for "Weibull" ($\xi < 0$), green for "Gumbel" ($\xi = 0$), and blue for "Fréchet" ($\xi > 0$). Display: density + CDF, support marked, limits indicated. Parent-distribution preset dropdown: "Normal (→ Gumbel)", "Exponential (→ Gumbel)", "Pareto($\alpha=2$) (→ Fréchet, $\xi = 1/2$)", "Uniform(0,1) (→ Weibull, $\xi = -1$)". Selecting a preset: simulate $n = 1000$ samples of block-size-$m=50$ maxima, fit GEV via MLE, show fitted density overlaid on histogram. Readout: fitted $\hat\xi, \hat\mu, \hat\sigma$ + their SEs.

### §6.10 Additional interactive component ideas

- **BlockMaximaVsPOT.** Same parent distribution, same sample. Left panel: block-maxima approach (fit GEV to block-max values). Right panel: POT approach (fit GPD to threshold exceedances). Compare fitted tail quantiles and VaR estimates. Shows why POT is usually more efficient (uses more data from the tail).
- **HillPlotExplorer.** Classic Hill plot: $\hat\alpha_{\mathrm{Hill}}(k)$ vs $k$ for $k = 1, \dots, n$. Reader slides a vertical cursor for the "selected $k$." Shows bias-variance tradeoff visually: small $k$ high variance, large $k$ high bias. Compare with Pickands estimator on the same data.
- **ReturnLevelExplorer.** Fit GEV to historical annual-maxima data (preset: synthetic flood levels, 100 years). Compute the $T = \{10, 50, 100, 500\}$-year return levels with confidence bands. Illustrates the extrapolation-to-tail-quantile problem that motivates EVT.

### §6.11 Reference shortlist

**Primary textbooks:**
- Embrechts, Paul, Claudia Klüppelberg, and Thomas Mikosch. 1997. *Modelling Extremal Events for Insurance and Finance*. Springer. ISBN 978-3540609315. [https://doi.org/10.1007/978-3-642-33483-2](https://doi.org/10.1007/978-3-642-33483-2). **Primary reference**; align Topic 4 exposition here, especially §3 (max-stability and GEV), §6 (POT and GPD), §7 (tail-index estimation).
- Coles, Stuart. 2001. *An Introduction to Statistical Modeling of Extreme Values*. Springer. ISBN 978-1852334598. [https://doi.org/10.1007/978-1-4471-3675-0](https://doi.org/10.1007/978-1-4471-3675-0). **Most accessible treatment**; good for worked examples and applications.
- de Haan, Laurens, and Ana Ferreira. 2006. *Extreme Value Theory: An Introduction*. Springer. ISBN 978-0387239460. [https://doi.org/10.1007/0-387-34471-3](https://doi.org/10.1007/0-387-34471-3). Modern rigorous treatment.
- Beirlant, Jan, Yuri Goegebeur, Johan Segers, and Jozef Teugels. 2004. *Statistics of Extremes: Theory and Applications*. Wiley. ISBN 978-0471976479. [https://doi.org/10.1002/0470012382](https://doi.org/10.1002/0470012382). Applied focus.

**Foundational papers:**
- Fisher, R. A., and L. H. C. Tippett. 1928. "Limiting Forms of the Frequency Distribution of the Largest or Smallest Member of a Sample." *Mathematical Proceedings of the Cambridge Philosophical Society* 24(2): 180–190. [https://doi.org/10.1017/S0305004100015681](https://doi.org/10.1017/S0305004100015681). Origin paper.
- Gnedenko, B. 1943. "Sur la distribution limite du terme maximum d'une série aléatoire." *Annals of Mathematics* 44(3): 423–453. [https://www.jstor.org/stable/1968974](https://www.jstor.org/stable/1968974). Full trichotomy proof.
- Pickands, James III. 1975. "Statistical Inference Using Extreme Order Statistics." *Annals of Statistics* 3(1): 119–131. [https://doi.org/10.1214/aos/1176343003](https://doi.org/10.1214/aos/1176343003). POT framework + Pickands estimator.
- Balkema, A. A., and L. de Haan. 1974. "Residual Life Time at Great Age." *Annals of Probability* 2(5): 792–804. [https://doi.org/10.1214/aop/1176996548](https://doi.org/10.1214/aop/1176996548). Companion paper for POT.
- Hill, Bruce M. 1975. "A Simple General Approach to Inference About the Tail of a Distribution." *Annals of Statistics* 3(5): 1163–1174. [https://doi.org/10.1214/aos/1176343247](https://doi.org/10.1214/aos/1176343247). Hill estimator.

**Multivariate / advanced:**
- Resnick, Sidney I. 1987. *Extreme Values, Regular Variation, and Point Processes*. Springer. ISBN 978-0387964812. Rigorous; pointer for deeper study including multivariate extremes.

### §6.12 Difficulty / length target

- **Difficulty:** advanced.
- **Read time:** 70 min (~13K words).
- **Full proofs:** 2 (§4.2 max-stability + §4.3 Fisher-Tippett-Gnedenko partial).
- **Stated theorems:** 3 (domain-of-attraction, Pickands-Balkema-de Haan, Hill asymptotics, GEV MLE).
- **Examples:** 12–14.
- **Remarks:** 18–22.
- **Figures:** 10 (block maxima motivation; three GEV shapes; domain-of-attraction matrix; POT threshold-selection plot; Hill plot; VaR / ES illustration; return-level plot; OOD-detection example; heavy-vs-light tail contrast; Topic-4 forward-map).
- **Interactive components:** 3 (GEVFamilyExplorer featured; BlockMaximaVsPOT; HillPlotExplorer; optional ReturnLevelExplorer).

### §6.13 Open questions

1. Substantial topic (70 min / 13K words) — acceptable at formalml length scale or too much? If formalml tends shorter, consider splitting into "EVT I: Block Maxima" and "EVT II: POT + Applications" as a two-topic treatment.
2. Multivariate EVT — one-section overview or dedicated follow-up topic? My recommendation: brief overview in §4.10 pointing to Resnick 1987.
3. Financial applications depth — ES and VaR backtesting could easily fill another section. Include at what level? My recommendation: compact §4.8, a half-page of applications, and a pointer to McNeil-Frey-Embrechts 2015 *Quantitative Risk Management* for the full treatment.

---

## § 7 Topic 5 — Statistical Depth

### §7.1 Slug & title

- **Slug:** `statistical-depth`
- **Title:** "Statistical Depth"
- **Subtitle (suggested):** "Multivariate quantile functions via halfspace depth and its relatives"

### §7.2 One-sentence pitch

In one dimension, the sample median and sample quantiles order points by how "central" they are within the distribution. Statistical depth generalizes this ordering to $\mathbb{R}^d$ — Tukey (halfspace) depth, Mahalanobis depth, simplicial depth, spatial depth — producing multivariate medians and nested depth regions that serve as the multivariate analog of quantile regions.

### §7.3 Why this topic

Statistical depth is the multivariate extension that formalstatistics Topic 29 §29.10 Rem 22 explicitly defers: *"Multivariate order statistics / statistical depth. Tukey depth, Mahalanobis depth, halfspace depth. Forthcoming formalml."* Topic 29 is scalar throughout; the natural follow-up question — "how do you order points in $\mathbb{R}^d$?" — has no canonical answer (there is no total order on $\mathbb{R}^d$ respecting the vector-space structure), but depth functions provide a principled family of center-out orderings. ML applications: outlier / anomaly detection in multivariate data, robust multivariate estimation, nonparametric multivariate ranking (e.g., DD-plots for comparing distributions), conformal-prediction score constructions for multi-output regression.

Topic 5 is ambitious and the most pedagogically demanding of the five — both because the mathematical framework is less familiar and because computational issues (Tukey depth is NP-hard in general dimension) force more careful treatment than the scalar case.

### §7.4 Scope — IN

- **The ordering problem in $\mathbb{R}^d$.** No total order that respects the affine structure. Center-out orderings as the resolution. Depth functions as a family of such orderings.
- **Tukey (halfspace) depth.** Definition: $\mathrm{HD}(x; P) = \inf_{H \ni x} P(H)$ over all closed halfspaces $H$ containing $x$. Properties: affine invariance, upper semi-continuity, vanishing at infinity, and maximality at the distribution center. The halfspace median as $\arg\max_x \mathrm{HD}(x; P)$.
- **Mahalanobis depth.** $\mathrm{MD}(x; \mu, \Sigma) = 1 / (1 + (x - \mu)^\top \Sigma^{-1} (x - \mu))$. Simple closed form; affine invariant; maximized at $\mu$. Useful as a parametric baseline.
- **Simplicial depth** (Liu 1990). $\mathrm{SD}(x; P) = P(x \in \mathrm{conv}(X_1, \dots, X_{d+1}))$ for $X_i \sim P$ iid. The "fraction of random simplices that contain $x$." Affine invariant; has a combinatorial structure.
- **Spatial / L1 depth** (Vardi-Zhang 2000). $\mathrm{SpD}(x; P) = 1 - \|\mathbb{E}[(x - X) / \|x - X\|]\|$. Geometric interpretation: the spatial median as the maximizer.
- **Zuo-Serfling axiomatic framework (featured).** Zuo & Serfling 2000 Annals of Statistics proposed four axioms a "statistical depth function" should satisfy: (A1) affine invariance, (A2) maximality at the center (if a natural center exists), (A3) monotonicity along rays from the deepest point, (A4) vanishing at infinity. Halfspace depth satisfies all four; Mahalanobis depth satisfies three (fails monotonicity in general); simplicial depth fails (A3).
- **Depth regions and the multivariate quantile function.** $R_\alpha(P) = \{x: \mathrm{HD}(x; P) \geq \alpha\}$ for $\alpha \in [0, 1/2]$ — the nested depth regions, analogous to univariate quantile intervals.
- **Computation.** In $d = 2$: Tukey depth computable in $O(n \log n)$ via the Rousseeuw-Ruts 1996 algorithm. In general, $d$: Tukey depth computation is NP-hard (Johnson-Preparata 1978 reduction). Approximation algorithms. Mahalanobis depth is $O(d^2)$ per point. Simplicial depth is $O(n^d)$ in general — prohibitive.
- **Applications.** Multivariate outlier detection (points with depth $< \alpha$ threshold). DD-plot (Liu-Parelius-Singh 1999): plot $\{(\mathrm{HD}(x_i; P), \mathrm{HD}(x_i; Q)) : x_i \in X \cup Y\}$ to visually compare distributions $P, Q$. Depth-based discriminant analysis. Depth-based clustering.
- **ML connections.** Anomaly scoring in high-dimensional data (robust multivariate feature-distribution detection). Score construction for multi-output conformal prediction (Topic 1 cross-reference). Nonparametric classification via the "classify to the distribution where the point has higher depth" rule.

### §7.5 Scope — OUT

- **Functional depth** (depth of functions in function spaces — López-Pintado–Romo 2009 band depth, Fraiman-Muniz 2001) — §5.10 pointer. Dedicated follow-up topic if formalml goes there.
- **Half-region depth / integrated depth** — pointer.
- **Robust-statistics classical content** (breakdown point, influence function — Huber-Ronchetti 2009 framework). Related to depth but separate track; Topic 5 cross-references, does not redevelop.
- **Depth for high-dimensional / infinite-dimensional data** — §5.10 pointer.
- **Directional data depth** (angular Mahalanobis, etc. — Agostinelli-Romanazzi 2013) — pointer.
- **Bayesian / depth-based posterior credible regions** — pointer.

### §7.6 Primary predecessor chain

- **Topic 29 §29.10 Rem 22 (formalstatistics)** — forward-pointer: *"Multivariate order statistics / statistical depth. Tukey depth, Mahalanobis depth, and halfspace depth. Forthcoming formalml."* Topic 5 discharges this.
- **Topic 8 (formalstatistics, Multivariate Distributions)** — multivariate distribution framework, ellipsoids, multivariate Normal. Topic 5 requires this as background (Mahalanobis depth is closely tied to multivariate Normal geometry).
- **Topic 29 (formalstatistics)** — Topic 5 opens explicitly: *"Topic 29 developed the sample quantile as the scalar-sample analog of the population quantile $\xi_p = F^{-1}(p)$. In $\mathbb{R}^d$, the quantile function has no direct analog — the inverse of a multivariate CDF is not a single point but a contour or region. Statistical depth reverses the direction: instead of inverting the CDF, it defines a scalar-valued 'centrality' function that orders points center-out, and its level sets serve as the multivariate analog of quantile regions."*
- **Topic 1 (formalml, Conformal Prediction)** — multi-output conformal prediction can use depth-based nonconformity scores; Topic 5 provides those scores.

### §7.7 Key theorems

1. **Zuo-Serfling axiomatic characterization (featured, stated).** Any function $D: \mathbb{R}^d \times \mathcal{P}(\mathbb{R}^d) \to [0, 1]$ satisfying (A1)–(A4) is a "statistical depth function." Halfspace depth satisfies all four; this grounds its canonical status.
2. **Uniqueness of the halfspace median.** If $P$ is absolutely continuous, the halfspace median $\arg\max_x \mathrm{HD}(x; P)$ exists and is unique. Stated.
3. **Affine invariance of halfspace depth (full proof).** $\mathrm{HD}(Ax + b; A P + b) = \mathrm{HD}(x; P)$ for invertible $A$. Short proof via change-of-variables on halfspaces.
4. **Consistency of sample halfspace depth.** $\mathrm{HD}(x; P_n) \to \mathrm{HD}(x; P)$ almost surely and uniformly on compact sets. Full proof via Glivenko-Cantelli applied to halfspace classes (VC class of dimension $d + 1$; formalstatistics Topic 10 §10.7 backbone).
5. **Computational complexity (stated).** Tukey depth computation is $O(n \log n)$ in $d = 2$ (Rousseeuw-Ruts 1996) and NP-hard in general $d$ (reduction to open-hemisphere problem; Johnson-Preparata 1978). Mahalanobis depth is $O(d^2)$ per query after $O(n d^2)$ covariance estimation.

### §7.8 Suggested section structure

| § | Title | Primary content |
|---|---|---|
| 5.1 | The ordering problem in $\mathbb{R}^d$ | No total order. The need for center-out orderings. Historical origins (Tukey 1975 in the Proceedings of the ICM). |
| 5.2 | Tukey (halfspace) depth | Definition, properties. Affine invariance proof. The halfspace median. |
| 5.3 | Mahalanobis depth | Definition. Relationship to multivariate Normal. Virtues and limits (requires mean + covariance; non-robust). |
| 5.4 | Simplicial depth and spatial depth | Liu 1990 simplicial. Vardi-Zhang 2000 spatial. Brief comparison. |
| 5.5 | **The Zuo-Serfling axiomatic framework (featured)** | Four axioms. Which depth functions satisfy which. Pedagogical payoff: shows that halfspace depth is the "canonical" choice. |
| 5.6 | Depth regions and multivariate quantile contours | $R_\alpha(P) = \{x : \mathrm{HD}(x; P) \geq \alpha\}$. Visualization. Connection to scalar quantile intervals. |
| 5.7 | Sample depth and its consistency | Empirical depth. Uniform consistency via Glivenko-Cantelli on halfspaces. Rate of convergence. |
| 5.8 | Computation | $O(n \log n)$ for $d = 2$ (Rousseeuw-Ruts 1996). NP-hardness in general $d$. Approximation algorithms (random-projection depth). |
| 5.9 | Applications: outlier detection, DD-plots, classification | Depth-based anomaly scoring. Liu-Parelius-Singh DD-plots. Depth-based discriminant rule. |
| 5.10 | Forward-map: functional depth, directional data, high-dim depth | Pointers. Band depth. Half-region depth. Depth-weighted means. |

### §7.9 Featured theorem + featured component

- **Featured theorem:** §5.5 — Zuo-Serfling axiomatic characterization. Stated with full motivation for each axiom and a comparison matrix (halfspace / Mahalanobis / simplicial / spatial × axioms A1-A4).
- **Featured component:** `DepthContours2D`. A 2D scatter-plot canvas. Reader drags points to place them. Live-computed halfspace-depth contour regions $R_\alpha$ for $\alpha \in \{0.05, 0.15, 0.25, 0.35, 0.5\}$ shown as nested convex polygons. The halfspace median marked with a cross. Preset distributions: "iid Normal", "mixture", "contaminated Normal (one outlier)", "skewed Beta × 2". Toggle: overlay Mahalanobis-depth contours (ellipsoids) for comparison — demonstrates that Mahalanobis produces ellipsoidal contours regardless of the true distribution, while Tukey depth adapts.

### §7.10 Additional interactive component ideas

- **DepthTypeComparison.** Same 2D scatter. Four panels: Tukey depth / Mahalanobis depth / simplicial depth / spatial depth contours. Reader sees how each depth function orders the same data differently.
- **DDPlotExplorer.** Two 2D distributions (presets: "same", "mean shift", "scale change", "shape change"). Computes depths for all points in both under both $P_n$ and $Q_n$. Plots $(\mathrm{HD}(x_i; P_n), \mathrm{HD}(x_i; Q_n))$. Under the null "$P = Q$" the DD-plot hugs the diagonal; under alternatives it departs in characteristic ways.
- **OutlierDetectionDemo.** 2D scatter with injected outliers. Compute sample Tukey depth for each point. Highlight points with depth below a threshold (e.g., $\alpha = 0.05$). Compare with Mahalanobis-based detection and a simple distance-from-centroid rule.

### §7.11 Reference shortlist

**Foundational:**
- Tukey, John W. 1975. "Mathematics and the Picturing of Data." In *Proceedings of the International Congress of Mathematicians*, Vancouver 1974, Vol. 2, 523–531. Introduction of halfspace depth (in the context of visualizing multivariate data).
- Liu, Regina Y. 1990. "On a Notion of Data Depth Based on Random Simplices." *Annals of Statistics* 18(1): 405–414. [https://doi.org/10.1214/aos/1176347507](https://doi.org/10.1214/aos/1176347507). Simplicial depth.

**Axiomatic framework and theory:**
- Zuo, Yijun, and Robert Serfling. 2000. "General Notions of Statistical Depth Function." *Annals of Statistics* 28(2): 461–482. [https://doi.org/10.1214/aos/1016218226](https://doi.org/10.1214/aos/1016218226). **Primary reference for Topic 5 §5.5.** The axiomatic framework.
- Zuo, Yijun, and Robert Serfling. 2000. "Structural Properties and Convergence Results for Contours of Sample Statistical Depth Functions." *Annals of Statistics* 28(2): 483–499. [https://doi.org/10.1214/aos/1016218227](https://doi.org/10.1214/aos/1016218227). Companion paper on sample depth.

**Review/textbook:**
- Mosler, Karl. 2013. "Depth Statistics." In *Robustness and Complex Data Structures*, eds. Becker, Fried, Kuhnt. Springer. [https://doi.org/10.1007/978-3-642-35494-6_2](https://doi.org/10.1007/978-3-642-35494-6_2). **Most accessible review**; align Topic 5 prose with this.
- Serfling, Robert. 2006. "Depth Functions in Nonparametric Multivariate Inference." In *DIMACS Series in Discrete Mathematics and Theoretical Computer Science*, Vol. 72, 1–16. Applied review.

**Application papers:**
- Liu, Regina Y., Jesse M. Parelius, and Kesar Singh. 1999. "Multivariate Analysis by Data Depth: Descriptive Statistics, Graphics and Inference." *Annals of Statistics* 27(3): 783–858. [https://doi.org/10.1214/aos/1018031260](https://doi.org/10.1214/aos/1018031260). DD-plots.
- Vardi, Yehuda, and Cun-Hui Zhang. 2000. "The Multivariate L1-Median and Associated Data Depth." *Proceedings of the National Academy of Sciences* 97(4): 1423–1426. [https://doi.org/10.1073/pnas.97.4.1423](https://doi.org/10.1073/pnas.97.4.1423). Spatial depth.

**Computational:**
- Rousseeuw, Peter J., and Ida Ruts. 1996. "Algorithm AS 307: Bivariate Location Depth." *Journal of the Royal Statistical Society Series C (Applied Statistics)* 45(4): 516–526. [https://doi.org/10.2307/2986073](https://doi.org/10.2307/2986073). $O(n \log n)$ algorithm for $d = 2$.
- Dyckerhoff, Rainer, and Pavlo Mozharovskyi. 2016. "Exact Computation of the Halfspace Depth." *Computational Statistics & Data Analysis* 98: 19–30. [https://doi.org/10.1016/j.csda.2016.01.005](https://doi.org/10.1016/j.csda.2016.01.005). Modern exact computation.

**Foundational pre-depth:**
- Oja, Hannu. 1983. "Descriptive Statistics for Multivariate Distributions." *Statistics & Probability Letters* 1(6): 327–332. [https://doi.org/10.1016/0167-7152(83)90054-8](https://doi.org/10.1016/0167-7152(83)90054-8). Oja median — a relative of simplicial depth.

### §7.12 Difficulty / length target

- **Difficulty:** advanced.
- **Read time:** 65 min (~12K words).
- **Full proofs:** 2 (§5.2 affine invariance; §5.7 uniform consistency via Glivenko-Cantelli on halfspaces).
- **Stated theorems:** 3 (Zuo-Serfling axioms characterization; halfspace median uniqueness; computational complexity).
- **Examples:** 10–12.
- **Remarks:** 15–18.
- **Figures:** 9 (ordering-in-$\mathbb{R}^d$ motivation; halfspace depth contours on 4 distributions; Mahalanobis ellipsoids vs Tukey contours; simplicial/spatial depth comparison; Zuo-Serfling axiom-compliance matrix; depth regions as multivariate quantile contours; DD-plot under null and 3 alternatives; outlier-detection example; Topic-5 forward-map).
- **Interactive components:** 3 (DepthContours2D featured; DepthTypeComparison; DDPlotExplorer; optional OutlierDetectionDemo).

### §7.13 Open questions

1. Treatment of simplicial + spatial depth: full derivation or one-section comparison? My recommendation: one section (§5.4) covering both, about 500 words each, emphasizing relationship to halfspace depth rather than full theory.
2. Computational-complexity depth: NP-hardness proof — statement only (cite Johnson-Preparata 1978) or half-page sketch? My recommendation: statement only; the reduction argument is graph-theoretic and doesn't advance the statistical narrative.
3. Depth for formalml's conformal-prediction topic (Topic 1): is cross-reference worth pulling into Topic 5 directly, or leave as external link? My recommendation: one §5.9 remark + cross-ref.

---

## § 8 Cross-Topic Dependency Graph

### §8.1 Internal dependencies (among the five formalml topics)

The five topics form a mostly-independent set, but three pairs have natural cross-references that improve the reader experience when both topics exist:

| From | To | Nature of dependency | Mandatory? |
|------|-----|-----------|------|
| Topic 1 (Conformal Prediction) §3.8 | Topic 2 (Quantile Regression) | Conformalized quantile regression (CQR — Romano et al. 2019). Topic 1 §3.8 defines CQR in a self-contained way; if Topic 2 exists, §3.8 can add a cross-link and be shorter. | No — Topic 1 self-contains the QR details it needs. |
| Topic 3 (Rank Tests) §3.1 | Topic 1 (Conformal Prediction) §3.2 | Exchangeability is the common assumption. Topic 3 opens with a one-paragraph cross-link to Topic 1, noting the shared framework: "Conformal prediction exploits exchangeability for coverage; rank tests exploit exchangeability for size. The mathematical machinery is the same." | No — Topic 3 self-contains. |
| Topic 5 (Statistical Depth) §5.9 | Topic 1 (Conformal Prediction) | Depth-based nonconformity scores for multi-output conformal. Topic 5 §5.9 cross-references; Topic 1 can optionally mention "for multi-output Y, see Topic 5 §5.9 on depth-based score construction." | No — both topics self-contain the essentials. |

**No hard internal dependencies.** All five are writable in any order; cross-references are an enhancement, not a prerequisite.

### §8.2 Dependencies on formalstatistics.com topics

| formalml topic | formalstatistics hard prerequisites | formalstatistics soft prerequisites |
|------|-----|-----|
| Topic 1 (Conformal Prediction) | 17 (Hypothesis Testing), 29 §29.7 (distribution-free quantile CIs) | 11 (CLT), 19 (Confidence Intervals), 29 §29.5 (ECDF + DKW), 31 (Bootstrap — once shipped) |
| Topic 2 (Quantile Regression) | 15 (Method of Moments / M-estimators), 21 (Linear Regression), 29 §29.4 + §29.6 (sample quantile + Bahadur) | 23 (Regularization), 30 (KDE — once shipped; for density in asymptotic variance) |
| Topic 3 (Rank Tests) | 17 (Hypothesis Testing), 18 (LRT + NP), 29 §29.2 (order stats) | 11 (CLT), 19 (CIs) |
| Topic 4 (EVT) | 11 (CLT), 29 (order statistics framework) | 9 (Modes of Convergence), 32 (Empirical Processes — once shipped) |
| Topic 5 (Statistical Depth) | 8 (Multivariate Distributions), 29 (scalar nonparametrics) | 10 §10.7 (Glivenko-Cantelli), 13 (Point Estimation — for consistency framework) |

**Key observation:** all five depend hardest on formalstatistics Topics 11, 17, 18, 19, 29, plus varying other topics. Topics 30 and 32 (KDE and Empirical Processes, Track 8's remaining pieces) are soft prerequisites for Topics 2 and 4 respectively. **Topic 29 is a hard prerequisite for all five.** This is not an accident — the five were chosen as the topics that Topic 29's §29.10 forward-pointer directed to formalml.

### §8.3 Suggested writing-order dependency chain

Most efficient order given §8.1 and §8.2:

1. **Conformal Prediction** (Topic 1) — no formalml prerequisite; lightest formalstatistics prerequisite load.
2. **Quantile Regression** (Topic 2) — no formalml prerequisite (can stand alone); sets up CQR cross-link to Topic 1.
3. **Rank Tests** (Topic 3) — no formalml prerequisite; §3.1 cross-link to Topic 1 is enhancement.
4. **Extreme Value Theory** (Topic 4) — benefits from Topic 32 (empirical processes) on formalstatistics if shipped; otherwise states Donsker and cites.
5. **Statistical Depth** (Topic 5) — benefits from Topic 1 for the multi-output conformal cross-reference; otherwise stands alone.

The writing order matches the "at a glance" table in §2.1. Departures from this order: see §2.3 for rationale.

---

## § 9 Shared-Module Suggestions

### §9.1 Suggested `src/components/viz/shared/` modules on formalml

Each topic introduces computational primitives that benefit from a shared module. Following the formalstatistics convention (Topic 29 introduced `nonparametric.ts`), each formalml topic introduces its own module at opening and subsequent topics extend rather than duplicate.

| Topic | Module | Suggested primary exports |
|-------|--------|-----------------------|
| 1 (Conformal) | `conformal.ts` | `splitConformalInterval(trainX, trainY, calX, calY, predictor, alpha)`, `nonconformityScore(x, y, predictor, method)` (with method options: residual / absResidual / normalizedResidual / classificationHinge / APS), `jackknifePlusInterval(...)`, `cqrInterval(x, calX, calY, loQuantilePredictor, hiQuantilePredictor)`, `empiricalCoverage(testY, intervals)` |
| 2 (QR) | `quantile-regression.ts` | `checkLoss(u, tau)`, `quantileRegressionFit(X, y, tau, options)` (simplex / IPM selector), `quantileRegressionSandwichSE(...)`, `conditionalQuantile(x, fitted)`, `rearrangementMonotonize(quantileCurves)`, `pinballLoss(yTrue, yPred, tau)` |
| 3 (Rank Tests) | `ranks.ts` | `rankTransform(sample)`, `wilcoxonSignedRank(sample)`, `mannWhitneyU(sample1, sample2)`, `kruskalWallisH(groups)`, `permutationDistribution(stat, data, nResamples)`, `hodgesLehmann(sample1, sample2)`, `spearmanRho(x, y)`, `kendallTau(x, y)`, `pitmanARE(distribution, test1, test2)` |
| 4 (EVT) | `evt.ts` | `gevDensity(x, xi, mu, sigma)`, `gevCDF(x, xi, mu, sigma)`, `gevQuantile(p, xi, mu, sigma)`, `gpdDensity(x, xi, beta)`, `gpdCDF(x, xi, beta)`, `gevFitMLE(blockMaxima)`, `gpdFitMLE(exceedances)`, `hillEstimator(sample, k)`, `pickandsEstimator(sample, k)`, `returnLevel(T, fit)`, `valueAtRisk(alpha, fit)`, `expectedShortfall(alpha, fit)` |
| 5 (Depth) | `depth.ts` | `halfspaceDepth2D(x, sample)`, `halfspaceMedian2D(sample)`, `mahalanobisDepth(x, mean, covariance)`, `simplicialDepth2D(x, sample)`, `spatialDepth(x, sample)`, `depthRegion2D(sample, alpha)`, `ddPlot(sample1, sample2, depthFunction)` |

Each module should follow the formalstatistics pattern: TypeScript with explicit types, JSDoc for every export, a matching `.test.ts` with expected values printed by a verification notebook cell.

### §9.2 Shared dependencies across modules

- **All modules** benefit from `lgamma` (log-gamma) and `binomialPmf` / `betaPdf` helpers — these already exist in `nonparametric.ts` on formalstatistics; if formalml shares the same npm/monorepo workspace, reuse; otherwise copy.
- **`conformal.ts`** and **`quantile-regression.ts`** have overlap in the "quantile of a calibration set" function — both need it. Define in `quantile-regression.ts`; `conformal.ts` imports from it.
- **`evt.ts`** and **`depth.ts`** both need multivariate Normal density primitives — suggest a separate `multivariate.ts` shared by both if formalml's scope extends beyond these five topics.

### §9.3 Palette extensions

Topic 29 defined `COLOR_ECDF`, `COLOR_DKW`, `COLOR_BAHADUR`, `COLOR_KS` for Track 8. Suggested formalml additions:

| Topic | New color token | Hex | Purpose |
|-------|-----------------|-----|-----|
| 1 (Conformal) | `COLOR_CONFORMAL` | `#0891B2` (cyan-600) | Conformal prediction intervals |
| 1 (Conformal) | `COLOR_MISCOVERAGE` | `#DC2626` (red-600) | Miscovered test points (when empirical coverage < 1 − α) |
| 2 (QR) | `COLOR_QR` | `#9333EA` (purple-600) | Quantile-regression fits |
| 3 (Ranks) | `COLOR_RANK` | `#65A30D` (lime-600) | Rank transforms and permutation dist |
| 4 (EVT) | `COLOR_GEV_GUMBEL` | `#0EA5E9` (sky-500) | Gumbel family |
| 4 (EVT) | `COLOR_GEV_FRECHET` | `#DC2626` (red-600) | Fréchet family |
| 4 (EVT) | `COLOR_GEV_WEIBULL` | `#059669` (emerald-600) | Weibull family |
| 5 (Depth) | `COLOR_DEPTH_CONTOUR` | `#7C3AED` (violet-600) | Depth-region contours |
| 5 (Depth) | `COLOR_HALFSPACE_MEDIAN` | `#DC2626` (red-600) | Halfspace median marker |

These are suggestions; Jonathan adapts to formalml's existing palette if one is established.

---

## § 10 Recommendations for Jonathan Before Starting

### §10.1 Prerequisites to resolve before writing any of the five

1. **Confirm formalml architecture.** Tech stack? Tracks-or-flat? MDX frontmatter schema? This determines the shape of each topic's handoff brief when you eventually write it. If formalml uses a different schema, a meta-template (analogous to formalstatistics' `claude-code-starter-prompt-template.md`) should be established first.
2. **Decide whether formalstatistics and formalml share a repo/workspace.** If yes, shared modules (§9) can live in a common location. If not, decide on a copy vs. import strategy.
3. **Map the formalml topic-numbering.** These five are bridge topics — where do they sit relative to any existing formalml topics? Do they get a shared track (e.g., "Distribution-Free Inference" or "Nonparametric ML")? Or are they scattered across ML-methods-oriented tracks?
4. **Decide on cross-site citation convention.** formalstatistics uses `<ExternalLink href="https://formalml.com/topics/...">` components. Reverse direction: Does formalml have an analogous component for citing back to formalstatistics? If not, establish one before Topic 1 ships so all backward citations can use it consistently.
5. **Resolve the Topic 30 / Topic 31 / Topic 32 timing question on formalstatistics.** Topic 2 (QR) benefits from Topic 30 (KDE). Topic 4 (EVT) benefits from Topic 32 (Empirical Processes). Do you ship the formalstatistics Track 8 remainder first (30, 31, 32) and *then* the formalml five? Or interleave? My recommendation: finish formalstatistics Track 8 through Topic 31 (Bootstrap), then ship formalml Topic 1 (Conformal) — by that point, Topic 1's natural Bootstrap comparison is live.

### §10.2 Pacing suggestion

- formalstatistics Topics 30–32: roughly 3–4 months at current single-developer cadence.
- formalml Topic 1 (Conformal Prediction): 4–6 weeks. Highest ROI per week.
- formalml Topic 2 (Quantile Regression): 4–6 weeks. Second-highest ROI.
- formalml Topic 3 (Rank Tests): 3–5 weeks. Classical content, straightforward exposition.
- formalml Topic 4 (EVT): 6–8 weeks. Heavier content, more technical.
- formalml Topic 5 (Statistical Depth): 5–7 weeks. Multivariate geometry is harder to visualize, but the interactive components carry a lot of the pedagogical weight.

Total for the five: 5–8 months, depending on cadence. Probably ships over the following ~12 months if interleaved with formalstatistics work and other commitments.

### §10.3 What to *not* try to optimize

- **Don't pre-commit to a rigid schedule.** The five topics are independent enough that order can shift based on reader interest / incoming citations / ML trend alignment.
- **Don't pre-write the shared-module infrastructure all at once.** Build incrementally as each topic ships — trying to design all five shared modules upfront invites over-abstraction.
- **Don't try to make every formalstatistics cross-reference live as an `<ExternalLink>` immediately.** Until each formalml topic ships, the cross-references on formalstatistics live as plain-text "(forthcoming on formalml)" markers. When each formalml topic goes live, a trivial formalstatistics PR upgrades the pointers to live links.

### §10.4 Open decision: a sixth topic?

During the Topic 29 brief-drafting session, three candidate sixth-topic directions came up but were deferred:

- **Bootstrap at scale + modern variants** (block bootstrap, wild bootstrap, stationary bootstrap for time series, bag-of-little-bootstraps for big data). Different from formalstatistics Topic 31, which will cover Efron-1979 foundations + bootstrap CIs. A formalml follow-up would cover the ML-practical scale.
- **Online learning and sequential conformal** (Vovk-Gammerman online learning, adaptive conformal — Gibbs-Candes 2021). Deferred from Topic 1 §3.10 as "could be a sibling topic."
- **Nonparametric regression beyond QR** (locally-weighted regression, kernel regression, Nadaraya-Watson). Adjacent to formalstatistics Topic 30 (KDE) but conditioning on covariates rather than estimating densities.

None are urgent; leave on the queue.

---

## § 11 Open Questions

This brief makes assumptions about formalml.com that need Jonathan's confirmation before any of the five topics enters the formal handoff-brief phase.

1. **formalml tech stack and MDX conventions.** Assumed parallel to formalstatistics. Confirm, or specify deviations (different framework? different math rendering? different component architecture?).
2. **Curriculum structure.** Assumed tracks-or-flat; each topic gets a track-relative position. Confirm structure, or indicate flat topic-list layout.
3. **Existing formalml topics.** Are there already-published topics on formalml that any of the five should cross-reference in either direction? This brief assumes no priors; list of existing topics would let me refine.
4. **Audience calibration on formalml vs formalstatistics.** Assumed the same "MS Data Science student / ML practitioner" audience; confirm. If formalml is more practitioner-oriented, I'd reduce formal-proof density and expand ML-application sections.
5. **Component-complexity budget on formalml.** Assumed similar to formalstatistics (3–4 components per topic, each ~500 lines of React). If formalml has a different complexity budget (e.g., more ambitious, or simpler), §§3.9–§7.9 component specs adjust.
6. **Cross-site navigation.** How does a reader move between formalstatistics Topic 29 and formalml Topic 1 (Conformal Prediction)? Shared header? Cross-link sidebar? This affects how prominent the back-pointers to Topic 29 should be in each of the five topics' §1 Position in Track sections.
7. **Depth of formalstatistics self-containment.** If a reader arrives at formalml Topic 1 without having read formalstatistics Topic 29, how much background should Topic 1 §3.1 re-establish? Full recap vs. "strongly recommend reading formalstatistics Topic 29 first" with terse recap?
8. **Should Topic 4 (EVT) be split?** 70 min / 13K words is the largest of the five. If formalml tends shorter, split into "EVT I: Block Maxima and the GEV Distribution" + "EVT II: POT, Tail-Index Estimation, and Risk Applications."
9. **Which of the five to write first?** My recommendation: Topic 1 (Conformal Prediction). Confirm or overrule.
10. **Hard deadline considerations?** If formalml has a release milestone (e.g., end-of-year with 10+ topics live), the writing order might prioritize "maximum topic count in 6 months" over "best pedagogical sequence." Confirm or specify.

---

## § 12 Appendix A — Complete Citation Shortlist

Citations across all five topics, alphabetized by first author. Each entry flagged with the topic(s) that cite it.

### Books / monographs

- Beirlant, Jan, Yuri Goegebeur, Johan Segers, and Jozef Teugels. 2004. *Statistics of Extremes: Theory and Applications*. Wiley. **[Topic 4]**
- Coles, Stuart. 2001. *An Introduction to Statistical Modeling of Extreme Values*. Springer. **[Topic 4]**
- de Haan, Laurens, and Ana Ferreira. 2006. *Extreme Value Theory: An Introduction*. Springer. **[Topic 4]**
- Embrechts, Paul, Claudia Klüppelberg, and Thomas Mikosch. 1997. *Modelling Extremal Events for Insurance and Finance*. Springer. **[Topic 4]**
- Good, Phillip I. 2005. *Permutation, Parametric, and Bootstrap Tests of Hypotheses*. 3rd ed. Springer. **[Topic 3]**
- Hájek, Jaroslav, and Zbyněk Šidák. 1967. *Theory of Rank Tests*. Academic Press. **[Topic 3]**
- Hao, Lingxin, and Daniel Q. Naiman. 2007. *Quantile Regression*. Sage Publications. **[Topic 2]**
- Hettmansperger, Thomas P., and Joseph W. McKean. 2011. *Robust Nonparametric Statistical Methods*. 2nd ed. CRC Press. **[Topic 3]**
- Hollander, Myles, Douglas A. Wolfe, and Eric Chicken. 2014. *Nonparametric Statistical Methods*. 3rd ed. Wiley. **[Topic 3]**
- Koenker, Roger. 2005. *Quantile Regression*. Cambridge University Press. **[Topic 2]**
- Lehmann, Erich L. 1975. *Nonparametrics: Statistical Methods Based on Ranks*. Holden-Day / Springer. **[Topic 3]**
- Lehmann, Erich L., and Joseph P. Romano. 2005. *Testing Statistical Hypotheses*. 3rd ed. Springer. **[Topic 3]**
- Resnick, Sidney I. 1987. *Extreme Values, Regular Variation, and Point Processes*. Springer. **[Topic 4]**
- Vovk, Vladimir, Alexander Gammerman, and Glenn Shafer. 2005. *Algorithmic Learning in a Random World*. Springer. **[Topic 1]**

### Review / tutorial articles

- Angelopoulos, Anastasios N., and Stephen Bates. 2023. "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification." *Foundations and Trends in Machine Learning* 16(4): 494–591. **[Topic 1]**
- Koenker, Roger, and Kevin F. Hallock. 2001. "Quantile Regression." *Journal of Economic Perspectives* 15(4): 143–156. **[Topic 2]**
- Mosler, Karl. 2013. "Depth Statistics." In *Robustness and Complex Data Structures*, eds. Becker, Fried, Kuhnt. Springer. **[Topic 5]**
- Serfling, Robert. 2006. "Depth Functions in Nonparametric Multivariate Inference." *DIMACS* 72: 1–16. **[Topic 5]**
- Shafer, Glenn, and Vladimir Vovk. 2008. "A Tutorial on Conformal Prediction." *JMLR* 9: 371–421. **[Topic 1]**

### Foundational / method papers

- Balkema, A. A., and L. de Haan. 1974. "Residual Life Time at Great Age." *Annals of Probability* 2(5): 792–804. **[Topic 4]**
- Barber, Rina Foygel, Emmanuel J. Candès, Aaditya Ramdas, and Ryan J. Tibshirani. 2021. "Predictive Inference with the Jackknife+." *Annals of Statistics* 49(1): 486–507. **[Topic 1]**
- Barber, Rina Foygel, Emmanuel J. Candès, Aaditya Ramdas, and Ryan J. Tibshirani. 2021. "The Limits of Distribution-Free Conditional Predictive Inference." *Information and Inference* 10(2): 455–482. **[Topic 1]**
- Belloni, Alexandre, and Victor Chernozhukov. 2011. "ℓ1-Penalized Quantile Regression in High-Dimensional Sparse Models." *Annals of Statistics* 39(1): 82–130. **[Topic 2]**
- Chernozhukov, Victor, Iván Fernández-Val, and Alfred Galichon. 2010. "Quantile and Probability Curves without Crossing." *Econometrica* 78(3): 1093–1125. **[Topic 2]**
- Chernozhukov, Victor, Iván Fernández-Val, and Blaise Melly. 2013. "Inference on Counterfactual Distributions." *Econometrica* 81(6): 2205–2268. **[Topic 2]**
- Chernozhukov, Victor, Kaspar Wüthrich, and Yinchu Zhu. 2021. "Exact and Robust Conformal Inference Methods for Predictive Machine Learning with Dependent Data." *PNAS* 118(48). **[Topic 1]**
- Dyckerhoff, Rainer, and Pavlo Mozharovskyi. 2016. "Exact Computation of the Halfspace Depth." *Computational Statistics & Data Analysis* 98: 19–30. **[Topic 5]**
- Fisher, R. A., and L. H. C. Tippett. 1928. "Limiting Forms of the Frequency Distribution of the Largest or Smallest Member of a Sample." *Mathematical Proceedings of the Cambridge Philosophical Society* 24(2): 180–190. **[Topic 4]**
- Gnedenko, B. 1943. "Sur la distribution limite du terme maximum d'une série aléatoire." *Annals of Mathematics* 44(3): 423–453. **[Topic 4]**
- Hill, Bruce M. 1975. "A Simple General Approach to Inference About the Tail of a Distribution." *Annals of Statistics* 3(5): 1163–1174. **[Topic 4]**
- Hodges, Joseph L., and Erich L. Lehmann. 1963. "Estimates of Location Based on Rank Tests." *Annals of Mathematical Statistics* 34(2): 598–611. **[Topic 3]**
- Koenker, Roger, and Gilbert Bassett. 1978. "Regression Quantiles." *Econometrica* 46(1): 33–50. **[Topic 2]**
- Kruskal, William H., and W. Allen Wallis. 1952. "Use of Ranks in One-Criterion Variance Analysis." *JASA* 47(260): 583–621. **[Topic 3]**
- Lei, Jing, Max G'Sell, Alessandro Rinaldo, Ryan J. Tibshirani, and Larry Wasserman. 2018. "Distribution-Free Predictive Inference for Regression." *JASA* 113(523): 1094–1111. **[Topic 1]**
- Liu, Regina Y. 1990. "On a Notion of Data Depth Based on Random Simplices." *Annals of Statistics* 18(1): 405–414. **[Topic 5]**
- Liu, Regina Y., Jesse M. Parelius, and Kesar Singh. 1999. "Multivariate Analysis by Data Depth." *Annals of Statistics* 27(3): 783–858. **[Topic 5]**
- Mann, Henry B., and Donald R. Whitney. 1947. "On a Test of Whether one of Two Random Variables is Stochastically Larger than the Other." *Annals of Mathematical Statistics* 18(1): 50–60. **[Topic 3]**
- Meinshausen, Nicolai. 2006. "Quantile Regression Forests." *JMLR* 7: 983–999. **[Topic 2]**
- Pickands, James III. 1975. "Statistical Inference Using Extreme Order Statistics." *Annals of Statistics* 3(1): 119–131. **[Topic 4]**
- Portnoy, Stephen, and Roger Koenker. 1997. "The Gaussian Hare and the Laplacian Tortoise." *Statistical Science* 12(4): 279–300. **[Topic 2]**
- Romano, Yaniv, Evan Patterson, and Emmanuel Candès. 2019. "Conformalized Quantile Regression." *NeurIPS 2019*. **[Topic 1]**
- Romano, Yaniv, Matteo Sesia, and Emmanuel Candès. 2020. "Classification with Valid and Adaptive Coverage." *NeurIPS 2020*. **[Topic 1]**
- Rousseeuw, Peter J., and Ida Ruts. 1996. "Algorithm AS 307: Bivariate Location Depth." *JRSS C* 45(4): 516–526. **[Topic 5]**
- Tibshirani, Ryan J., Rina Foygel Barber, Emmanuel J. Candès, and Aaditya Ramdas. 2019. "Conformal Prediction Under Covariate Shift." *NeurIPS 2019*. **[Topic 1]**
- Tukey, John W. 1975. "Mathematics and the Picturing of Data." *Proceedings of the International Congress of Mathematicians* 2: 523–531. **[Topic 5]**
- Vardi, Yehuda, and Cun-Hui Zhang. 2000. "The Multivariate L1-Median and Associated Data Depth." *PNAS* 97(4): 1423–1426. **[Topic 5]**
- Wilcoxon, Frank. 1945. "Individual Comparisons by Ranking Methods." *Biometrics Bulletin* 1(6): 80–83. **[Topic 3]**
- Zuo, Yijun, and Robert Serfling. 2000. "General Notions of Statistical Depth Function." *Annals of Statistics* 28(2): 461–482. **[Topic 5]**
- Zuo, Yijun, and Robert Serfling. 2000. "Structural Properties and Convergence Results for Contours of Sample Statistical Depth Functions." *Annals of Statistics* 28(2): 483–499. **[Topic 5]**

---

## § 13 Appendix B — formalstatistics.com → formalml.com Cross-Reference Map

For quick reference: which formalstatistics topic's forward-pointers land where on formalml.

| formalstatistics source | Section | Forward-pointer target | Format |
|-------|----|------|------|
| Topic 7 (Continuous Distributions) | §7.13 | (none — forward-pointer goes to formalstatistics Topic 29) | plain text |
| Topic 13 (Point Estimation) | §13.11 | Topic 29 §29.6 (formalstatistics) + (no direct formalml pointer here) | `<a>` |
| Topic 15 (Method of Moments) | §15.9 Rem + frontmatter | formalml `quantile-regression` (Topic 2) | plain text "(forthcoming)" |
| Topic 18 (LRT + NP) | §18.10 Rem 25 | formalml `rank-tests` (Topic 3) | plain text |
| Topic 19 (Confidence Intervals) | §19.10 Rem 20 + method-selection table | formalstatistics Topic 31 (Bootstrap) | `<a>` |
| Topic 22 (GLMs) | §22.10 Rem 24–25 + forward-map | formalml advanced topics (GAMs, mixed-effects, high-dim regression) | plain text |
| Topic 28 (Hierarchical Bayes) | §28.10 Rem 22–25 | formalml (BNN, VI, DP, etc.) | plain text |
| Topic 29 (Order Statistics) | §29.10 Rem 18 | formalml `extreme-value-theory` (Topic 4) | plain text "(forthcoming)" |
| Topic 29 | §29.10 Rem 19 | formalml `quantile-regression` (Topic 2) | plain text |
| Topic 29 | §29.10 Rem 20 | formalml `rank-tests` (Topic 3) | plain text |
| Topic 29 | §29.10 Rem 21 | formalml `conformal-prediction` (Topic 1) | plain text |
| Topic 29 | §29.10 Rem 22 | formalml `statistical-depth` (Topic 5) | plain text |

**When each formalml topic ships**, update the corresponding formalstatistics plain-text marker to a live `<ExternalLink href="https://formalml.com/topics/<slug>" site="formalml" topic="<title>" />` component. This is a trivial one-line edit per formalstatistics file.

---

*End of formalml.com planning brief. Five forthcoming topics mapped: `conformal-prediction`, `quantile-regression`, `rank-tests`, `extreme-value-theory`, `statistical-depth`. Writing order and dependency structure documented. Each topic has its own eventual handoff brief when Jonathan commits to writing it; this document serves as the upstream planning substrate.*
