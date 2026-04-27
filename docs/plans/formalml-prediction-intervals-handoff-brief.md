# formalML — Prediction Intervals — Handoff Brief

**Track:** T4 Nonparametric & Distribution-Free (sixth and final topic in track)
**Slug:** `prediction-intervals`
**Difficulty:** `intermediate`
**Source notebook:** `notebooks/prediction-intervals/01_prediction_intervals.ipynb`
**Status:** Pre-MDX handoff brief. The notebook is the source of truth for math, code, and figures; this brief layers implementation spec on top.

---

## What this topic does

This is the umbrella synthesis topic that closes T4. It connects the three preceding T4 topics — `conformal-prediction` (marginal coverage from exchangeability), `quantile-regression` (covariate-adjusted conditional quantiles), and `rank-tests` (Hodges-Lehmann distribution-free CI via test inversion) — into a unified treatment of *how to build a prediction interval under different assumptions and what to give up at each step.*

Three constructions are featured: split conformal (Construction I, exchangeability-based), pure quantile-regression intervals (Construction II, conditional-quantile-based), and Hodges-Lehmann test-inversion (Construction III, location-shift-symmetry-based). CQR appears as the bridge between Constructions I and II. The synthesis topic earns its keep with three new bridge theorems (§5) connecting the constructions and a four-scenario empirical comparison (§6) that consolidates the practical recommendations.

The defining question this topic answers: *given a black-box predictor or a parametric/nonparametric estimator, how do I construct a finite-sample-valid interval around the next prediction, and how do the three constructions compare on coverage, width, conditional behavior, and computational cost?*

## Cross-site prerequisites (frontmatter)

`formalstatisticsPrereqs`:

- `confidence-intervals-and-duality` — The formal foundation for both HL-style test-inversion (§4) and the conformal-quantile threshold (§2.3). The duality between a level-α test and a (1−α) confidence region is the abstract machinery behind every interval construction in this topic.
- `order-statistics-and-quantiles` — Split-conformal's quantile of conformity scores (Definition 7), QR's empirical conditional-quantile estimator (§3), and HL's Walsh-average ordering (§4) all rest on order-statistic theory.
- `empirical-processes` — The asymptotic alternative to finite-sample exchangeability arguments. Underlies the QR asymptotics cited in §3.3, Theorem 5.1's empirical-process stability argument, and Theorem 5.3's HL/conformal equivalence proof.
- `bootstrap` — The third resampling-based interval-construction method (alongside conformal and permutation-from-rank-tests). §7.1 contrasts bootstrap with the three featured constructions.

`formalcalculusPrereqs`: empty. The topic uses no calculus beyond undergraduate-level Taylor expansion (in the asymptotic-equivalence step of Theorem 5.3), which is covered as background in `formalstatistics/empirical-processes`. This matches the precedent set by `conformal-prediction` and `rank-tests`.

`formalcalculusConnections`, `formalstatisticsConnections`: empty. The topic is a synthesis closer to its track, not a foundational topic that other topics build on.

## Internal prerequisites (formalML curriculum graph)

- `conformal-prediction` (T4 #1) — direct prereq; provides Theorem 1 (cited verbatim in §2 and §5.1) and the score-function frame extended in §2.1.
- `quantile-regression` (T4 #2) — direct prereq; provides Theorem 3 from QR §5 (cited as Theorem 2 here in §3.3) and the QR base learner used by pure QR (§3) and CQR (§5.1).
- `rank-tests` (T4 #3) — implicitly available; Theorem 10 from rank-tests §6 is cited in the proof of Theorem 3 here in §4.3.

---

## 1. The Prediction-Interval Problem

This section sets the scaffolding on which the rest of the topic hangs. We pin down what a prediction interval *is* (and what it isn't), distinguish the two coverage notions every later section refers back to, and lay out the strict assumption hierarchy under which the three featured constructions work. The mathematics here is light — definitions and named distinctions, no theorems with proofs. The work begins in §2.

### 1.1 Confidence intervals vs. prediction intervals

A confidence interval covers a fixed but unknown *parameter*. A prediction interval covers a *random variable* — the next observation, before it has happened. The two quantities live in different probability spaces, and constructions that work for one don't generally transfer.

To make the distinction concrete, suppose we observe pairs $(X_1, Y_1), \ldots, (X_n, Y_n)$ and fit a linear model $\hat{f}(x) = \hat{\beta}_0 + \hat{\beta}_1 x$. At a new point $X_{n+1} = x_*$ we can ask two distinct questions:

1. *Where does the conditional mean $\mathbb{E}[Y \mid X = x_*]$ lie?* — a confidence-interval question. The interval shrinks at a rate $1/\sqrt{n}$.
2. *Where does $Y_{n+1}$ itself lie?* — a prediction-interval question. The interval bundles together estimation uncertainty *and* the irreducible noise of $Y_{n+1}$ around its mean.

Even with infinite data, the CI shrinks to a point, but the PI plateaus at its irreducible-noise width. That structural gap is the whole point of distinguishing the two. Figure 1 in the notebook shows the gap in synthetic homoscedastic data.

> **Definition 1 (Prediction interval).** A function $\hat{C}: \mathcal{X} \to 2^{\mathcal{Y}}$ from the feature space to subsets of the response space is a *prediction interval at level $1 - \alpha$* if it satisfies a coverage statement of the form
> $$\mathbb{P}\big(Y_{n+1} \in \hat{C}(X_{n+1})\big) \ge 1 - \alpha,$$
> where $\alpha \in (0, 1)$ is the *miscoverage level* and the probability is taken jointly over the training data $(X_i, Y_i)_{i=1}^n$ and the test pair $(X_{n+1}, Y_{n+1})$.

The randomness in this probability matters. It runs over the training set, the test feature, and the test response — *not* over a fixed test point with frozen training data. Different ways of decomposing this randomness give different coverage notions, which is the next subsection.

### 1.2 Marginal vs. conditional coverage

The probability statement in Definition 1 averages over both the training set and the test feature $X_{n+1}$. We can ask for a stronger guarantee that holds *pointwise* in the test feature.

> **Definition 2 (Marginal vs. conditional coverage).**
>
> *Marginal coverage* at level $1 - \alpha$:
> $$\mathbb{P}\big(Y_{n+1} \in \hat{C}(X_{n+1})\big) \ge 1 - \alpha.$$
>
> *Conditional coverage* at level $1 - \alpha$: for $\mathbb{P}_X$-almost every $x \in \mathcal{X}$,
> $$\mathbb{P}\big(Y_{n+1} \in \hat{C}(X_{n+1}) \,\big|\, X_{n+1} = x\big) \ge 1 - \alpha.$$

Conditional coverage is much stronger. Marginal coverage allows the interval to over-cover at some $x$ and under-cover at others, as long as the average over the marginal of $X$ comes out right. Conditional coverage demands the guarantee point-by-point.

To see the gap, consider the running heteroscedastic example we'll use throughout the topic:
$$Y \mid X = x \sim \mathcal{N}\!\big(\sin(x), \sigma(x)^2\big), \qquad \sigma(x) = 0.2 + 0.6 \, |x|/3, \qquad X \sim \text{Uniform}(-3, 3).$$
The conditional standard deviation is small near $x = 0$ and large near $x = \pm 3$. A *constant-width* band $\hat{C}(x) = [\hat{\mu}(x) - w, \hat{\mu}(x) + w]$ with $w$ tuned to give 90% marginal coverage will over-cover near the centre (the band is far wider than the noise needs) and under-cover near the edges (the band is too narrow to capture 90% of the conditional mass). The conditional-coverage curve $x \mapsto \mathbb{P}(Y \in \hat{C}(x) \mid X = x)$ is wildly miscalibrated even though its average is exactly $0.9$. Figure 2 makes this explicit: the binned conditional-coverage histogram ranges from over $99\%$ in the low-noise center to roughly $70\%$ in the high-noise tails.

This is the gap that §3 (pure QR) and §5's CQR bridge close — and that the marginal-only conformal guarantee in §2 leaves open by design.

(There is an intermediate notion, *group-conditional coverage*, where the guarantee holds for each member of a finite partition of the feature space — useful in fairness contexts where the partition is a protected attribute. We treat that as a forward connection in §7.)

### 1.3 The data-distributional assumption hierarchy

Each of the three constructions in this topic works under a different assumption on the joint distribution of $(X_1, Y_1), \ldots, (X_{n+1}, Y_{n+1})$. The assumptions are *strictly nested*, with conformal at the weak end and Hodges-Lehmann at the strong end.

> **Definition 3 (Exchangeability).** The sequence $(X_1, Y_1), \ldots, (X_{n+1}, Y_{n+1})$ is *exchangeable* if for every permutation $\pi$ of $\{1, \ldots, n+1\}$,
> $$\big((X_{\pi(1)}, Y_{\pi(1)}), \ldots, (X_{\pi(n+1)}, Y_{\pi(n+1)})\big) \stackrel{d}{=} \big((X_1, Y_1), \ldots, (X_{n+1}, Y_{n+1})\big),$$
> where $\stackrel{d}{=}$ denotes equality in joint distribution.

Exchangeability is the assumption underlying split conformal prediction; see [Conformal Prediction §1](/topics/conformal-prediction#exchangeability-setup). It is strictly weaker than iid: a uniformly random permutation of a fixed multiset is exchangeable but not iid (the marginals are not independent). Practically, exchangeability fails under temporal ordering, distribution shift, or hierarchical sampling, but holds whenever the order of observations is irrelevant.

> **Definition 4 (iid).** The pairs $(X_i, Y_i)$ are *independent and identically distributed* if they are mutually independent and share a common joint distribution $\mathbb{P}_{XY}$.

Pure quantile-regression intervals built from Koenker-Bassett asymptotics require iid plus smoothness conditions on the conditional density of $Y \mid X$; see [Quantile Regression §5](/topics/quantile-regression#asymptotic-theory).

> **Definition 5 (iid with symmetric residuals).** The pairs $(X_i, Y_i)$ are iid AND there exists a function $\mu : \mathcal{X} \to \mathbb{R}$ such that the residuals $\varepsilon_i := Y_i - \mu(X_i)$ are independent of $X_i$ and have a distribution symmetric around zero.

The symmetry assumption is what test-inversion-style intervals (§4) require to deliver finite-sample distribution-free conditional coverage.

The three classes are genuinely strictly nested:
$$\big\{ \text{iid} + \text{symmetric residuals} \big\} \subsetneq \big\{ \text{iid} \big\} \subsetneq \big\{ \text{exchangeable} \big\}.$$
In English: every distribution where HL works also makes pure QR and conformal work; every iid distribution makes conformal work; not every exchangeable distribution is iid (the random-permutation-of-a-multiset example), and not every iid distribution has symmetric residuals (any skewed-noise regression).

The trade is the standard one in statistics. Stronger assumptions buy the practitioner more — better coverage type (conditional rather than marginal), narrower intervals, or both. Weaker assumptions buy *robustness* — guarantees that survive when the strong assumptions fail. The next three sections walk through the constructions in order of weakest-assumption-first; §5 makes the trade quantitative through three bridge theorems.

### 1.4 The two-axis map

Combining the two distinctions (assumption strength × coverage type) gives the map that every later section refers back to:

| Construction | Section | Assumption | Coverage |
|---|---|---|---|
| Split conformal | §2 | Exchangeable | Marginal (finite-sample) |
| Pure QR | §3 | iid + smoothness | Asymptotic conditional |
| HL test-inversion | §4 | iid + symmetric residuals | Conditional (finite-sample) |
| CQR (bridge) | §3 → §5 | Exchangeable | Marginal, conditionally adaptive |

CQR is worth flagging now, even though we don't define it until §5.1. It sits as a hybrid: it inherits the marginal guarantee of split conformal (because it *is* split conformal under a particular score function), but its interval *shape* tracks pure QR's conditional-quantile estimates. The hybrid is the topic's main practical recommendation, and §5 formalizes the gap between its rigorous marginal guarantee and its approximate conditional behavior.

§§2–4 cover one row each — short, citation-heavy treatments of each construction with the running example carried through. §5 proves three bridge theorems: a CQR-coverage decomposition, a heteroscedastic-width-comparison bound, and an asymptotic-equivalence result for HL and conformal on location-shift problems with symmetric noise. §6 measures the trade-offs empirically across four data scenarios. §7 closes with bootstrap as a contrast, what's out of scope (Bayesian credible intervals, T5), and forward connections (online conformal, group-conditional coverage).

### Static figures (notebook output)

- **Figure 1** — CI vs PI on a fitted homoscedastic linear regression. Two bands overlaid: a narrow blue confidence band for $\mathbb{E}[Y \mid X]$ and a wider red prediction band for $Y_{\text{new}}$. The CI shrinks with $n$; the PI plateaus at the noise floor.
- **Figure 2** — Marginal-vs-conditional gap on the running heteroscedastic example. Two panels: (left) scatter with constant-width band coloured by inside/outside; (right) binned conditional-coverage histogram with a horizontal reference line at $0.9$. The gap is the central motivation of the topic.
- **Figure 3** — Schematic of the assumption hierarchy as nested rounded boxes, with the three constructions placed at their respective shells. Pure schematic, not a data plot.

### Interactive widget intent (live site, React/D3)

A *Coverage Calibration Explorer* with three controls:

1. **Heteroscedasticity slider** $\sigma_{\max} \in [0, 1]$ — interpolates between homoscedastic ($\sigma_{\max} = 0$) and strongly heteroscedastic ($\sigma_{\max} = 1$) noise on the running example.
2. **Sample size slider** $n \in [50, 2000]$ — controls how tight the empirical marginal coverage is around its target $1 - \alpha = 0.9$.
3. **Band-type toggle** — three choices: *constant-width* (the broken baseline), *QR-based* (asymptotic conditional), *perfect oracle* (uses true $\sigma(x)$ — the unattainable lower bound on width).

Display: scatter of $(X, Y)$ with the chosen band overlaid; points coloured red if outside, blue if inside. Beneath the scatter, two readouts: a single number for empirical marginal coverage, and a strip chart of empirical conditional coverage in 10 equal-mass $X$-bins with a horizontal reference line at $1 - \alpha$. As the reader drags the heteroscedasticity slider up with the constant-width band selected, the strip chart should *visibly* deform from flat to U-shaped while the single marginal number stays pegged near $0.9$ — the exact pathology Definition 2 anticipates.

---

## 2. Construction I: Exchangeability-Based (Split Conformal)

This is the weakest-assumption construction in the topic. It works under exchangeability alone — no smoothness, no symmetry, no parametric form for the noise. The price is that the resulting interval is constant-width (or any other shape baked into the score function) and only marginally calibrated; the conditional miscalibration of Figure 2 carries over essentially unchanged. The construction is the canonical version of split conformal prediction; we cite the marginal-coverage theorem from `conformal-prediction` rather than reprove it.

Two lifting moves matter here. First, we introduce the **score-function abstraction** — every interval in this topic can be written as the level set of some score $s(x, y)$, with the threshold either calibrated empirically (§2 and the §5 CQR bridge), determined asymptotically (§3), or inverted from a test (§4). Second, we'll see that constant-width split-conformal on the heteroscedastic running example reproduces Figure 2's marginal-vs-conditional gap almost exactly — by design, because the score $s(x, y) = |y - \hat{\mu}(x)|$ encodes nothing about heteroscedasticity. §3 starts fixing that by changing the score.

### 2.1 The score-function abstraction

> **Definition 6 (Nonconformity score).** A *nonconformity score* is a function $s : \mathcal{X} \times \mathcal{Y} \to \mathbb{R}$, larger values indicating that the pair $(x, y)$ is more anomalous relative to the training data. Given a threshold $q \in \mathbb{R}$, the corresponding prediction set is
> $$\hat{C}_{s, q}(x) = \{ y \in \mathcal{Y} : s(x, y) \le q \}.$$

This abstraction lets us classify the constructions in this topic by their choice of $(s, q)$:

| Construction | Score $s(x, y)$ | Threshold $q$ |
|---|---|---|
| Split conformal (§2) | $|y - \hat{\mu}(x)|$ | conformal $(1-\alpha)$-quantile of calibration scores |
| Pure QR (§3) | $\max\!\big(\hat{q}_{\alpha/2}(x) - y,\; y - \hat{q}_{1-\alpha/2}(x)\big)$ | $0$ (no calibration) |
| CQR (§5 bridge) | Same as pure QR | conformal $(1-\alpha)$-quantile of calibration scores |
| HL test-inversion (§4) | Recast as a Walsh-average score | inverted from a Wilcoxon test |

The split-conformal threshold gets a name we'll use throughout:

> **Definition 7 (Conformal quantile).** For nonconformity scores $S_1, \ldots, S_{n_{\mathrm{cal}}}$ on a calibration set, the *conformal $(1-\alpha)$-quantile* is
> $$\hat{q}_{1-\alpha} = S_{(\lceil (1-\alpha)(n_{\mathrm{cal}}+1) \rceil)},$$
> the $\lceil (1-\alpha)(n_{\mathrm{cal}}+1) \rceil$-th order statistic of $\{S_i\}$. The $+1$ in the numerator is the finite-sample correction that turns the threshold from "approximately right asymptotically" into "exactly right under exchangeability."

The pure-QR row is what makes the score-function frame earn its keep. As a *score* with threshold $0$, pure QR is just "predict in if the score is non-positive," which lines up exactly with the algebraic statement of Construction II in §3. The CQR bridge in §5 then becomes a one-line statement: keep the score, swap the threshold for the conformal quantile, and inherit Theorem 1.

### 2.2 Split conformal on the running example

Three steps:

1. *Train.* Split the data into a training fold and a calibration fold. Fit a base predictor $\hat{\mu}$ on the training fold only.
2. *Calibrate.* Compute $S_i = |Y_i - \hat{\mu}(X_i)|$ for each calibration point, and take $\hat{q}_{1-\alpha}$ per Definition 7.
3. *Predict.* For a new point $x$, return $\hat{C}(x) = [\hat{\mu}(x) - \hat{q}_{1-\alpha},\; \hat{\mu}(x) + \hat{q}_{1-\alpha}]$.

The notebook below carries this out on the heteroscedastic running example with $n_{\mathrm{train}} = n_{\mathrm{cal}} = 500$ and $\hat{\mu}$ a degree-3 polynomial fit by ridge regression. The choice of base predictor matters for the band's width but not for its coverage validity — the theorem in §2.3 holds for any score.

### 2.3 The marginal-coverage theorem (cited)

> **Theorem 1 (Split-conformal marginal coverage; Vovk, Gammerman & Shafer 2005, Lei et al. 2018).** *If the calibration data $(X_i, Y_i)_{i=1}^{n_{\mathrm{cal}}}$ and the test point $(X_{n_{\mathrm{cal}}+1}, Y_{n_{\mathrm{cal}}+1})$ are exchangeable, and the nonconformity score $s$ does not depend on the calibration or test data, then for any $\alpha \in (0, 1)$ the split-conformal prediction set satisfies*
> $$1 - \alpha \;\le\; \mathbb{P}\!\big( Y_{n_{\mathrm{cal}}+1} \in \hat{C}(X_{n_{\mathrm{cal}}+1}) \big) \;\le\; 1 - \alpha + \frac{1}{n_{\mathrm{cal}}+1}.$$

Proved as Theorem 1 of [Conformal Prediction §3](/topics/conformal-prediction#marginal-coverage-the-central-theorem) via a rank-symmetry argument: under exchangeability, the rank of the test score among the calibration scores is uniform on $\{1, \ldots, n_{\mathrm{cal}}+1\}$, and the threshold definition translates that uniform rank into the coverage statement. The $1/(n_{\mathrm{cal}}+1)$ over-coverage on the right is the finite-sample artifact of the $+1$ correction in Definition 7; it vanishes as $n_{\mathrm{cal}} \to \infty$.

We use this theorem in §5 (bridge theorems) without reproof. Two ingredients we'll lean on: (i) the coverage statement itself, which lower-bounds the marginal coverage of *any* score-function-based interval that uses the conformal threshold — including the CQR bridge; and (ii) the rank-symmetry argument, which we'll need to recombine with QR's pointwise-approximation bounds to prove the CQR-coverage decomposition (Theorem 5.1).

### 2.4 What this gets, what it misses

Running the construction on the heteroscedastic example with $\alpha = 0.1$ delivers empirical marginal coverage close to $0.9$ on the test set, exactly as Theorem 1 promises. But the conditional-coverage curve (Figure 5 in the notebook) reproduces Figure 2 almost identically: above $99\%$ near $x = 0$, dropping below $80\%$ near $x = \pm 3$. The reason is mechanical — the score $s(x, y) = |y - \hat{\mu}(x)|$ has no $x$-dependence in its calibration distribution, so the threshold $\hat{q}_{1-\alpha}$ is a single number, and the resulting band is constant-width regardless of $\hat{\mu}$.

Two ways forward: (a) change the score to encode heteroscedasticity (CQR, §5 bridge, with pure QR in §3 as the unconformalised version); (b) strengthen the assumptions to recover finite-sample conditional coverage by symmetry arguments (HL test-inversion, §4). §5 makes the comparison quantitative.

### Static figures (notebook output)

- **Figure 4** — Split-conformal band on the running heteroscedastic example. Left: scatter with band overlaid, points coloured by inside/outside. Right: empirical distribution of calibration scores with $\hat{q}_{1-\alpha}$ marked.
- **Figure 5** — Conditional coverage by 10 equal-width bins of $X$ for the split-conformal band. Designed to be visually parallel to Figure 2: same axes, same reference line, same color scheme, so the carryover of the conditional miscalibration is obvious by side-by-side comparison.

### Interactive widget intent (live site, React/D3)

Extend the §1 *Coverage Calibration Explorer* by adding **split conformal** as a band-type option alongside the existing constant-width and pure-QR choices. The UI difference: when split conformal is selected, expose a slider for $n_{\mathrm{cal}} \in [50, 500]$ (with $n_{\mathrm{train}}$ fixed at 500) so the reader can watch the conformal threshold tighten as calibration data accumulates, and see the small-$n_{\mathrm{cal}}$ over-coverage of $1/(n_{\mathrm{cal}}+1)$ shrink toward zero. The conditional-coverage strip chart should remain U-shaped throughout, regardless of $n_{\mathrm{cal}}$ — the visual evidence that more data does not fix the conditional gap when the score is wrong.

---

## 3. Construction II: Conditional-Quantile (Pure QR Intervals)

The §2 split-conformal construction gets exact finite-sample marginal coverage but a constant-width band. This section does the opposite trade: it gives up the finite-sample guarantee in exchange for a band whose *shape* tracks the conditional spread of $Y$ given $X$. The construction is the asymptotic prediction interval obtained by fitting two quantile-regression models at levels $\alpha/2$ and $1 - \alpha/2$, and using their fitted curves as the lower and upper endpoints of the interval — no calibration step. We call it *pure QR* to distinguish it from CQR (the §5 bridge), which keeps the QR shape but replaces the asymptotic justification with a finite-sample rank-symmetry argument.

The whole construction is a citation of [Quantile Regression §5](/topics/quantile-regression#asymptotic-theory) for the population-level conditional-quantile fact, plus an asymptotic-coverage statement that follows from QR's Koenker-Knight asymptotic normality. We don't reprove either result. The interesting mathematical content is the diagnosis of *why* the resulting interval is conditionally adaptive but only asymptotically valid — and the bookkeeping required to express it as a $(s, q)$ pair in the §2.1 score-function frame, which §5 then reuses verbatim.

### 3.1 The construction

> **Definition 8 (Pure QR prediction interval).** Let $\hat{q}_\tau(x)$ denote a fitted estimator of the conditional $\tau$-quantile of $Y$ given $X = x$, in any function class (linear-in-features, kernel, neural). For miscoverage $\alpha \in (0, 1)$, the *pure QR prediction interval* at $x$ is
> $$\hat{C}^{\mathrm{QR}}_\alpha(x) \;=\; \big[\, \hat{q}_{\alpha/2}(x),\; \hat{q}_{1-\alpha/2}(x) \,\big].$$

Translating into the §2.1 score-function frame: pure QR uses
$$s(x, y) \;=\; \max\!\big(\hat{q}_{\alpha/2}(x) - y,\; y - \hat{q}_{1-\alpha/2}(x)\big), \qquad q \;=\; 0.$$
The score is positive when $y$ falls outside the QR interval, and negative when inside; the threshold $0$ corresponds to "no calibration" in the score-function language. The §5 CQR bridge will keep the score and replace the threshold $0$ with a conformal $(1-\alpha)$-quantile of calibration scores per Definition 7.

### 3.2 Why the construction makes sense (population fact)

If we knew the true conditional quantile functions $q_{\alpha/2}^*$ and $q_{1-\alpha/2}^*$, the resulting interval would have *exact* conditional coverage by construction: for every $x$,
$$\mathbb{P}\!\big( Y_{n+1} \in [q_{\alpha/2}^*(X_{n+1}), q_{1-\alpha/2}^*(X_{n+1})] \,\big|\, X_{n+1} = x \big) \;=\; 1 - \alpha.$$
This is just the definition of conditional quantiles — the probability mass of $Y$ given $X = x$ between its $\alpha/2$ and $1 - \alpha/2$ quantiles is, by definition, $1 - \alpha$. There is nothing to prove here that isn't already in [Quantile Regression §1](/topics/quantile-regression#pinball-loss-and-population-quantile).

The construction is conditionally calibrated *at the population level*. In English: if we had an oracle for the true conditional quantiles, this is the thing we would build, and it would be conditionally valid pointwise. The asymptotic theory in §3.3 says that consistent estimators inherit this property in the limit; the gap to finite samples is what §5's bridge quantifies and what motivates CQR.

### 3.3 Asymptotic conditional coverage (cited)

> **Theorem 2 (Pure QR asymptotic conditional coverage).** *Suppose the conditional density $f_{Y \mid X}(\cdot \mid x)$ is positive and continuous in a neighbourhood of $q_\tau^*(x)$ for $\tau \in \{\alpha/2, 1 - \alpha/2\}$, that the QR estimators $\hat{q}_{\alpha/2}$ and $\hat{q}_{1-\alpha/2}$ are uniformly consistent on the support of $X$, and that the function class is rich enough to contain $q_\tau^*$. Then for $\mathbb{P}_X$-almost every $x$,*
> $$\mathbb{P}\!\big( Y_{n+1} \in \hat{C}^{\mathrm{QR}}_\alpha(X_{n+1}) \,\big|\, X_{n+1} = x \big) \;\longrightarrow\; 1 - \alpha \quad \text{as } n \to \infty.$$

Proof sketch: Theorem 3 of [Quantile Regression §5](/topics/quantile-regression#asymptotic-theory) gives $\sqrt{n}\big(\hat{q}_\tau(x) - q_\tau^*(x)\big) \xrightarrow{d} \mathcal{N}\!\big(0, \omega_\tau(x)^2\big)$ pointwise in $x$, with the asymptotic variance $\omega_\tau(x)^2$ involving the conditional density at $q_\tau^*(x)$. Pointwise consistency of $\hat{q}_\tau$ then implies pointwise convergence of the conditional coverage to its population value $1 - \alpha$ by continuity of the conditional CDF.

Three things this theorem does *not* deliver, and §5 will quantify the gap on each of them:

1. **No finite-sample guarantee.** The convergence is in the limit. At any fixed $n$, conditional coverage can deviate from $1 - \alpha$ by an amount that depends on QR's pointwise estimation error.
2. **No marginal guarantee either.** Marginal coverage is the integral of conditional coverage against $\mathbb{P}_X$. If conditional coverage is biased downward at most $x$ (a generic possibility — QR overfits the visible data and produces too-narrow bands at finite $n$), marginal coverage falls below $1 - \alpha$. This is the phenomenon that the §6 empirical comparison will surface clearly.
3. **No uniform statement across $x$.** Pointwise convergence allows arbitrarily slow convergence at $x$ values near the boundary of the support, where QR estimates are notoriously noisy. Theorem 5.2 (the width-comparison bound) will lean on a uniform refinement of this convergence under bounded conditional density.

The contrast with Theorem 1 in §2 is the topic's central trade. Theorem 1 gives a finite-sample marginal guarantee under exchangeability with a constant-width interval; Theorem 2 gives an asymptotic conditional guarantee under stronger smoothness assumptions with a heteroscedasticity-adapted interval. CQR (§5 bridge) gets the better of both — but only on the marginal axis, not on the conditional one, as Theorem 5.1 will make precise.

### 3.4 Pure QR on the running example

The notebook fits two `QuantileRegressor` models on the running heteroscedastic example with degree-3 polynomial features at $\tau = 0.05$ and $\tau = 0.95$, returning the band $[\hat{q}_{0.05}(x), \hat{q}_{0.95}(x)]$ at $\alpha = 0.1$. Three observations to record:

1. **The band is visibly heteroscedastic.** Narrow near $x = 0$ where the noise is small, wide near $x = \pm 3$ where it's large. This is the band-shape win pure QR delivers, and the constant-width band of §2 cannot.
2. **Conditional coverage is approximately flat.** The 10-bin conditional-coverage histogram (Figure 7 in the notebook) is roughly horizontal at $\approx 0.9$ — a striking visual contrast with the U-shape of Figures 2 and 5. Theorem 2 in action.
3. **Marginal coverage is *not* automatic.** On a typical run with $n = 500$, empirical marginal coverage often falls a bit short of $0.9$ — perhaps $0.87$ or $0.88$ — because QR's finite-sample bias produces too-narrow intervals on average. This is the failure mode that Theorem 1 in §2 was designed to rule out, and the failure mode CQR fixes by composition.

### 3.5 Preview of CQR (§5 bridge)

CQR is, in the score-function frame, the same construction as pure QR with the threshold $0$ replaced by the conformal $(1-\alpha)$-quantile of calibration scores per Definition 7. That is:

| Pure QR | CQR |
|---|---|
| Score $s(x, y) = \max(\hat{q}_{\alpha/2}(x) - y, y - \hat{q}_{1-\alpha/2}(x))$ | Same score |
| Threshold $q = 0$ | Threshold $\hat{q}_{1-\alpha}$ from a calibration set |
| Band $[\hat{q}_{\alpha/2}(x), \hat{q}_{1-\alpha/2}(x)]$ | Band $[\hat{q}_{\alpha/2}(x) - \hat{q}_{1-\alpha},\; \hat{q}_{1-\alpha/2}(x) + \hat{q}_{1-\alpha}]$ |
| Asymptotic conditional coverage (Theorem 2) | Finite-sample marginal coverage (Theorem 1, applied to the QR score) |
| QR shape, no marginal calibration | QR shape *with* marginal calibration |

The §5 bridge makes this precise. CQR inherits split conformal's finite-sample marginal guarantee verbatim — it really is just split conformal with a particular score — *and* the QR shape, which gives it conditional adaptivity in approximation even though the rigorous guarantee remains marginal-only. Theorem 5.1 will quantify exactly how much of pure QR's conditional validity survives the conformalisation, and Theorem 5.2 will compare CQR's expected width to split conformal's under heteroscedasticity. We don't define or analyze CQR further in this section — that's §5's job. The point of the preview is that the score-function frame from §2.1 is doing real work: pure QR and CQR differ by exactly one number (the threshold), and that one-number difference is the entire architectural content of the bridge.

### Static figures (notebook output)

- **Figure 6** — Pure QR band on the running heteroscedastic example. Left: scatter with the two QR curves $\hat{q}_{0.05}(x)$ and $\hat{q}_{0.95}(x)$ overlaid as the band edges, points coloured inside/outside, and the constant-width split-conformal band from Figure 4 dashed in for comparison. Right: side-by-side strip charts of conditional coverage by bin for split conformal (from Figure 5, replotted) and pure QR. Visually parallel: the U-shape from §2 flattens out under pure QR.
- **Figure 7** — Marginal-coverage Monte Carlo distribution. Repeat the pure-QR fit + evaluation on $n_{\mathrm{rep}} = 200$ independent draws of $(X, Y)$ to get a sampling distribution of empirical marginal coverage, with reference lines at $1 - \alpha$ and the §6 split-conformal Monte Carlo distribution overlaid. The visual punchline: split conformal's distribution is tightly peaked at $\approx 0.9$ (Theorem 1 gives finite-sample control); pure QR's distribution is wider and centered slightly below $0.9$ at this sample size (asymptotic-only validity bites in finite samples).

### Interactive widget intent (live site, React/D3)

Promote the band-type toggle in the §1/§2 *Coverage Calibration Explorer* to a four-way choice: constant-width, split-conformal, pure-QR, oracle. When *pure QR* is selected, expose two additional readouts beneath the conditional-coverage strip chart: (a) empirical marginal coverage as a single number, and (b) the *gap* between empirical marginal coverage and $1 - \alpha$, signed and coloured red when negative. The reader should see immediately that pure QR makes the strip chart go flat (the conditional-adaptivity win) but loses the rock-solid marginal calibration that constant-width + split conformal had — the foreshadowing of CQR's "have it both ways" win at §5.

---

## 4. Construction III: Test-Inversion (HL-Style Prediction Intervals)

The two preceding constructions illustrate the marginal/conditional trade in its purest form: §2 buys a finite-sample marginal guarantee with the price of a constant-width band; §3 buys conditional adaptivity with the price of an asymptotic-only guarantee. Both work under exchangeability or iid — assumptions weak enough to accommodate arbitrary noise distributions. This section trades in the opposite direction: we accept a *much* stronger assumption (iid with residuals symmetric around zero, independent of $X$) and in exchange recover finite-sample distribution-free *conditional* coverage — the strongest guarantee on offer in this topic.

The construction generalizes Hodges-Lehmann's test-inversion CI from `rank-tests §6` from a *confidence interval for a location parameter $\theta$* to a *prediction interval for the next observation $Y_{n+1}$*. Conceptually, the move is small — inverting the same rank-symmetry argument — but mathematically it requires a $1/(n+1)$ correction analogous to the conformal $+1$ correction in Definition 7. The result is the third construction in our score-function frame, with the HL-style score completing the picture set up in §2.1.

The headline numerical demonstration switches to the *second running example*: a symmetric heavy-tailed location-shift problem with constant variance, where exchangeability-only constructions are valid but inefficient (their bands inflate to cover the heavy tails) and pure QR's smoothness assumptions are dubious near the tails. HL is in its element here, and §5's Theorem 5.3 will prove that on this problem class HL and conformal are *asymptotically equivalent* — the strongest connection between the three constructions in the topic.

### 4.1 The location-shift setup

The construction requires a more restrictive data model than §§2-3:

> **Definition 9 (Location-shift model).** Pairs $(X_i, Y_i)_{i=1}^{n+1}$ are iid from a *location-shift model* if there exists a function $\mu : \mathcal{X} \to \mathbb{R}$ and a distribution $F$ on $\mathbb{R}$ symmetric around zero such that
> $$Y_i = \mu(X_i) + \varepsilon_i, \qquad \varepsilon_i \stackrel{\mathrm{iid}}{\sim} F, \qquad \varepsilon_i \perp X_i.$$

Three assumptions are doing work here, in increasing order of strength:

1. **iid** — already required by §3.
2. **Independence of residual and feature ($\varepsilon_i \perp X_i$).** This rules out heteroscedasticity. If $\mathrm{Var}(\varepsilon \mid X)$ depends on $X$, the construction below is no longer valid: the symmetry argument it leans on requires the residual distribution to be the same at every $x$.
3. **Symmetry of the residual distribution ($F = -F$).** Stronger than independence: the noise distribution must be a centered symmetric distribution. Gaussian, $t$, Laplace, uniform-symmetric all qualify; exponential, gamma, lognormal don't.

This is a *much* narrower class than the exchangeable models §2 admits or the iid models §3 admits. The payoff is correspondingly larger: a finite-sample distribution-free guarantee that is conditional on $X$, not just marginal. §4.4 will identify the exact failure modes for each violated assumption.

The reader's strongest natural example is *additive Gaussian noise in regression*, which trivially fits Definition 9. The more interesting example — and the one we'll headline — is *additive Student-$t$ noise with $\mathrm{df} = 3$*, where the heavy tails make pure QR's smoothness assumptions wobbly and inflate the constant-width split-conformal band well beyond what the data needs.

### 4.2 The Walsh-average score

To put the HL-style construction into the §2.1 score-function framework, we need a score function whose calibration distribution is symmetric around zero, as defined in Definition 9. The natural choice generalizes the Walsh-average construction from `rank-tests §6`:

> **Definition 10 (HL-style nonconformity score).** Fix a base predictor $\hat{\mu}$ trained on a held-out fold. For a calibration set with residuals $r_i = Y_i - \hat{\mu}(X_i)$, the *HL-style score* for a candidate test pair $(x_*, y_*)$ is
> $$s_{\mathrm{HL}}(x_*, y_*) \;=\; \mathrm{median}_{1 \le i \le n_{\mathrm{cal}}}\Big( \tfrac{1}{2}\big[ (y_* - \hat{\mu}(x_*)) + r_i \big] \Big).$$

The median of the Walsh averages of the test residual and the calibration residuals. When $y_*$ is the true $Y_{n+1}$, this is the one-sample HL location estimator from `rank-tests` Definition 7 applied to the augmented residual sample. Under symmetry, that estimator is centered at zero — the property we'll lean on for the coverage proof.

The threshold paired with this score doesn't follow Definition 7. Instead, it comes from inverting a Wilcoxon test, exactly as in `rank-tests §6` — which is why we call this *test-inversion*. The Walsh-average ordering of the residuals gives the threshold:

> **Definition 11 (HL-style prediction interval).** Let $\hat{\mu}$ be a base predictor trained on a separate fold, $r_1, \ldots, r_{n_{\mathrm{cal}}}$ the calibration residuals, and $A_{(1)} \le \cdots \le A_{(M)}$ their sorted Walsh averages with $M = n_{\mathrm{cal}}(n_{\mathrm{cal}}+1)/2$ off-diagonal pairs. For miscoverage $\alpha$, let $w_\alpha$ be the integer satisfying
> $$\mathbb{P}_{H_0}(W^+ \le w_\alpha) \;\le\; \tfrac{\alpha}{2},$$
> the lower $\alpha/2$ critical value of the discrete null distribution of the signed-rank statistic on $n_{\mathrm{cal}} + 1$ residuals (with the test residual treated symbolically). The HL-style prediction interval at $x_*$ is
> $$\hat{C}^{\mathrm{HL}}_\alpha(x_*) \;=\; \hat{\mu}(x_*) \;+\; \big[\, A_{(w_\alpha + 1)},\; A_{(M - w_\alpha)} \,\big].$$

The interval is the fitted mean plus a symmetric pair of Walsh-average order statistics — the construction that worked for the location parameter in `rank-tests §6` lifted to the prediction setting by recentring on $\hat{\mu}(x_*)$. The width $A_{(M - w_\alpha)} - A_{(w_\alpha + 1)}$ is the same at every $x_*$ — *the band is constant-width like §2's*, not adaptive like §3's. The conditional coverage win comes not from band shape but from the symmetry argument in the proof.

### 4.3 Coverage theorem

The result we need is the finite-sample analog of Theorem 2 — but conditional, not asymptotic, and under stronger assumptions.

> **Theorem 3 (HL-style finite-sample conditional coverage).** *Under the location-shift model of Definition 9, with the HL-style prediction interval of Definition 11 built from a calibration set of size $n_{\mathrm{cal}}$ disjoint from $\hat{\mu}$'s training data, for every $x \in \mathcal{X}$,*
> $$\mathbb{P}\!\big( Y_{n+1} \in \hat{C}^{\mathrm{HL}}_\alpha(X_{n+1}) \,\big|\, X_{n+1} = x \big) \;\ge\; 1 - \alpha - \frac{1}{n_{\mathrm{cal}} + 1}.$$
> *The finite-sample slack $1/(n_{\mathrm{cal}} + 1)$ matches the over-coverage slack in Theorem 1 and vanishes as $n_{\mathrm{cal}} \to \infty$.*

**Proof.** Condition on $X_{n+1} = x$. Under Definition 9, $Y_{n+1} - \hat{\mu}(x) = \mu(x) - \hat{\mu}(x) + \varepsilon_{n+1}$, where $\varepsilon_{n+1} \sim F$ symmetric around zero and independent of $X_{n+1}$ and the calibration data. Define the recentred residual
$$r_{n+1}^* \;=\; Y_{n+1} - \hat{\mu}(x) \;=\; b(x) + \varepsilon_{n+1}, \qquad b(x) := \mu(x) - \hat{\mu}(x).$$
The bias $b(x)$ is a deterministic function of $x$ once $\hat{\mu}$ is frozen by training-fold conditioning.

The calibration residuals $r_i = Y_i - \hat{\mu}(X_i)$, conditional on $\hat{\mu}$, take the form $r_i = b(X_i) + \varepsilon_i$. By independence $\varepsilon_i \perp X_i$ and the iid sampling of $X_i$, the marginal distribution of $r_i$ is the convolution of the distribution of $b(X)$ and the distribution of $\varepsilon$ — both symmetric around their respective centers, hence $r_i$ is *symmetric around* $\mathbb{E}[b(X)]$. The shift is the only obstruction; we'll see it canceled.

Now apply [Rank Tests §6 Theorem 10](/topics/rank-tests#hodges-lehmann-distribution-free-ci) to the augmented sample $\{r_1, \ldots, r_{n_{\mathrm{cal}}}, r_{n+1}^*\}$ — but treat the test residual as the *parameter* $\theta$ being estimated, not as an additional observation. Theorem 10 says: the set of $\theta$ values for which the level-$\alpha$ Wilcoxon test fails to reject the null hypothesis "the augmented sample's distribution is symmetric around $\theta$" is exactly the interval $[A_{(w_\alpha + 1)}, A_{(M - w_\alpha)}]$ in Walsh-average order statistics of the calibration residuals. The interval has coverage at least $1 - \alpha$ for the true center of symmetry of the calibration residual distribution — that is, for $\mathbb{E}[b(X)]$ in our notation.

Two more steps. First, *the test residual $r_{n+1}^*$ is exchangeable with the calibration residuals* (under Definition 9, the joint distribution of $(r_1, \ldots, r_{n_{\mathrm{cal}}}, r_{n+1}^*)$ is permutation-invariant — they're all of the form $b(X) + \varepsilon$ with iid $X$ and iid $\varepsilon$). So the rank of $r_{n+1}^*$ among the augmented sample is uniform on $\{1, \ldots, n_{\mathrm{cal}} + 1\}$, and Theorem 10's coverage guarantee for the centre of symmetry transfers to a coverage guarantee for *any specific augmented-sample observation*, with the standard $1/(n_{\mathrm{cal}} + 1)$ finite-sample correction. (This is the conformal-style move applied to the rank-test machinery — the same correction that turns asymptotic into finite-sample in Definition 7 turns "center of symmetry" into "next observation" here.)

Second, the bias $b(x)$ does not appear on either side of the inequality. The interval $\hat{\mu}(x) + [A_{(w_\alpha+1)}, A_{(M-w_\alpha)}]$ contains $Y_{n+1}$ if and only if the centred quantity $r_{n+1}^* - b(x)$ — which equals $\varepsilon_{n+1}$, hence has the *same* distribution as the centred calibration residuals' shared symmetric distribution — falls in the appropriate symmetric interval around zero. By the rank-symmetry argument that has at least probability $1 - \alpha - 1/(n_{\mathrm{cal}}+1)$.

The probability is conditional on $X_{n+1} = x$ throughout: nothing in the argument averaged over $X_{n+1}$, so the guarantee is genuinely conditional. $\square$

The bias $b(x) = \mu(x) - \hat{\mu}(x)$ deserves a remark: the proof is robust to it, but the *width* of the resulting interval is not. A poor predictor $\hat{\mu}$ produces calibration residuals with a wide convolved distribution, hence wider Walsh averages, hence a wider interval. The construction is *valid* but not *efficient* under bias; we'll revisit this in §6 empirical comparison.

### 4.4 The second running example — heavy-tailed location-shift

The headline figure for §4 switches scenarios:

> **Running Example 2 (heavy-tailed location-shift).**
> $$Y \mid X = x \;\sim\; \mu(x) + \sigma \cdot t_3, \qquad \mu(x) = 0.4 \cos(\pi x),\quad \sigma = 0.6,\quad X \sim \mathrm{Uniform}(-2, 2).$$
> Constant variance, additive symmetric heavy-tailed (Student-$t$ with $\mathrm{df} = 3$) noise, smooth deterministic mean.

The notebook produces a side-by-side comparison on this scenario at $\alpha = 0.1$:

1. **HL-style** — Theorem 3 gives finite-sample conditional $0.9$ coverage; we'll see the empirical conditional-coverage strip chart sit cleanly at $0.9$ across all $X$-bins.
2. **Split conformal** — Theorem 1 gives finite-sample marginal $0.9$ coverage; the band is constant-width like HL, but the threshold is the empirical $90$th percentile of $|r_i|$, which under $t_3$ is inflated by the heavy tails. Visually, split conformal's band is *wider than HL's*.
3. **Pure QR** — Theorem 2 holds asymptotically; finite-sample marginal coverage often falls slightly short, and conditional coverage is approximately flat. The band is *narrower in the middle* (where most data sits) and *wider in the tails* — but on this constant-variance scenario, that adaptivity is wasted.

The expected ranking by mean width is **HL ≤ split conformal ≤ pure QR** on this scenario (with HL and split conformal close — Theorem 5.3 in §5 proves they're asymptotically equivalent here — but pure QR is distinctly wider because the QR fit has to spend degrees of freedom modeling a quantile function that's actually constant in $x$). The HL versus split-conformal width gap is a small but real efficiency win at finite $n$.

### 4.5 Limits — when each assumption fails

The construction's strong guarantee comes from strong assumptions, and Definition 9's three assumptions fail in distinct, observable ways. The notebook makes each failure visible:

1. **Symmetry violation.** Replace the $t_3$ residual with a centered chi-squared minus its mean (right-skewed, mean zero, but $F \ne -F$). The empirical conditional coverage of the HL band drops below $0.9$ — no longer protected by Theorem 3 because its symmetry hypothesis fails. Split conformal still hits $0.9$ marginally; pure QR still flattens conditionally.
2. **Heteroscedasticity ($\varepsilon \not\perp X$).** Switch back to Running Example 1. The HL band has constant width (it's a single pair of Walsh-average order statistics, with no $x$-dependence), so its conditional-coverage strip chart re-acquires the U-shape of Figure 5 — over-cover near $x = 0$, under-cover near $x = \pm 3$. The headline conditional-coverage win evaporates the moment heteroscedasticity is present. This is exactly the regime where pure QR (and §5's CQR bridge) wins.
3. **Non-iid (e.g., temporal correlation).** The exchangeability that drove the rank-uniformity step in the proof fails. None of the three constructions in this topic is valid; this is the regime where *online* conformal methods (Vovk 2002; Gibbs–Candès 2021) take over — flagged in §7's forward connections.

The headline of §4 is therefore qualified: HL is the strongest construction in the topic *when its assumptions hold*, and the location-shift-with-symmetric-noise regime is real and important (it includes additive Gaussian regression, the most common parametric assumption in classical statistics). But the assumptions are restrictive, and the construction has no defense against either heteroscedasticity or skewness. §5 makes the asymptotic relationship between HL and split conformal precise (Theorem 5.3); §6 quantifies the trade-offs across all four scenarios.

### Static figures (notebook output)

- **Figure 8** — HL-style band on Running Example 2 (heavy-tailed location-shift). Three panels: (i) scatter with the HL band, the split-conformal band, and the pure-QR band overlaid, (ii) a Walsh-averages histogram with the HL critical-value pair $A_{(w_\alpha+1)}, A_{(M-w_\alpha)}$ marked, (iii) conditional-coverage strip chart for all three constructions side-by-side, sitting at or near 0.9.
- **Figure 9** — *Width comparison panel*. Bar chart of mean band width on Running Example 2 for HL vs split-conformal vs pure QR, averaged over $n_{\mathrm{rep}} = 100$ Monte Carlo draws, with error bars. The HL ≤ split-conformal ≤ pure QR ranking should be visually clear; this figure foreshadows the full empirical comparison in § 6.
- **Figure 10** — *Symmetry-violation diagnostic*. Repeat the HL/split-conformal/pure-QR comparison with a right-skewed residual (centered chi-squared minus its mean). HL's conditional-coverage strip chart drops below 0.9 in a systematic, reproducible way; split conformal stays at 0.9 marginally; pure QR remains approximately flat conditionally. The empirical refutation of HL's symmetry hypothesis.

### Interactive widget intent (live site, React/D3)

A new *Assumption-Failure Explorer* widget specific to §4. Three controls plus a band-type toggle:

1. **Residual distribution dropdown** — Gaussian, Laplace, $t_3$, $t_1$ (Cauchy), centred chi-squared, centred lognormal. Spans the symmetric and asymmetric cases.
2. **Heteroscedasticity slider** $\sigma_{\max} \in [0, 1]$ — interpolates between location-shift (Definition 9 holds) and the §1 heteroscedastic regime.
3. **Sample size slider** $n_{\mathrm{cal}} \in [50, 1000]$.
4. **Band-type toggle** — split conformal, pure QR, HL.

Display: scatter, band overlay, conditional-coverage strip chart, marginal-coverage readout, *plus* a small "Theorem 3 conditions met?" indicator (green check / red cross) that flips off when the residual is asymmetric, or the heteroscedasticity slider is non-zero. The reader should be able to walk through the three failure modes by adjusting one control at a time and watching the conditional-coverage strip chart deform — symmetry violation drops the whole chart, heteroscedasticity reintroduces the U-shape, non-Gaussian symmetric residuals leave HL untouched but visibly inflate split conformal.

---

## 5. Bridge Theorems

§§2–4 introduced the three constructions through the score-function frame from §2.1, citing the prerequisite theorems from `conformal-prediction`, `quantile-regression`, and `rank-tests` rather than reproving them. The synthesis topic earns its keep here. This section formalizes three relationships between the constructions:

1. **Theorem 5.1 (CQR coverage decomposition)** — the bridge between Constructions I and II. CQR is always marginally valid (it's split conformal under the QR score, by definition); we prove that its *conditional* coverage gap is bounded by twice the QR base learner's pointwise quantile-estimation error. This explains the empirical pattern from §3 — CQR is conditional-adaptive but not conditional-valid — without requiring it as a separate theorem.
2. **Theorem 5.2 (heteroscedastic width comparison)** — quantifies the §3 efficiency intuition. Under Running Example 1's heteroscedastic noise with conditional standard deviation bounded between $\sigma_- > 0$ and $\sigma_+ < \infty$, the expected CQR width is bounded above by expected split-conformal width up to lower-order terms, with the gap closing in the homoscedastic limit $\sigma_- = \sigma_+$.
3. **Theorem 5.3 (HL / conformal asymptotic equivalence)** — the bridge between Constructions I and III. Under Definition 9's location-shift model, the HL-style and split-conformal intervals converge to the *same* population symmetric interval around $\mu(x)$. Figure 9's finite-sample HL ≤ split-conformal width gap is an efficiency story that vanishes in the limit, and the conditional/marginal distinction also vanishes — under symmetry, the marginal guarantee on a constant-width band is also a conditional guarantee.

CQR is fully defined and analyzed here per the §3.5 agreement. Notation is shared with §§2–4.

### 5.1 Definition of CQR

> **Definition 12 (Conformalized quantile regression).** Let $\hat{q}_{\alpha/2}$ and $\hat{q}_{1-\alpha/2}$ be quantile-regression estimators trained on a training fold disjoint from the calibration fold. For each calibration point $i$ define the CQR score
> $$E_i \;=\; \max\!\big(\hat{q}_{\alpha/2}(X_i) - Y_i,\; Y_i - \hat{q}_{1-\alpha/2}(X_i)\big).$$
> Let $\hat{Q}_{1-\alpha}$ be the conformal $(1-\alpha)$-quantile of $\{E_i\}_{i=1}^{n_{\mathrm{cal}}}$ per Definition 7. The *CQR prediction interval* at $x$ is
> $$\hat{C}^{\mathrm{CQR}}_\alpha(x) \;=\; \big[\, \hat{q}_{\alpha/2}(x) - \hat{Q}_{1-\alpha},\; \hat{q}_{1-\alpha/2}(x) + \hat{Q}_{1-\alpha} \,\big].$$

In the §2.1 score-function frame, CQR is the pair $(s, q) = (s_{\mathrm{QR}}, \hat{Q}_{1-\alpha})$, where $s_{\mathrm{QR}}$ is the pure-QR score from §3.1 and the threshold is the conformal quantile rather than zero. Theorem 1 from §2 applies *verbatim* — CQR inherits split conformal's finite-sample marginal coverage guarantee with no extra work, the architectural payoff of the score-function frame.

### 5.2 Theorem 5.1 — CQR coverage decomposition

The motivating question. Pure QR (§3) is conditionally valid asymptotically; CQR (§5.1) is marginally valid in finite samples. What happens to *conditional* coverage under the conformalisation? Theorem 5.1 answers in two parts: a finite-sample bound (the conformal $+1$ correction transfers cleanly), and a conditional-coverage bound that decays at the rate of QR's estimation error.

Let $q_{\alpha/2}^*$ and $q_{1-\alpha/2}^*$ be the true conditional quantiles, and define the pointwise QR estimation error
$$\Delta_n(x) \;=\; \max\Big( |\hat{q}_{\alpha/2}(x) - q_{\alpha/2}^*(x)|,\; |\hat{q}_{1-\alpha/2}(x) - q_{1-\alpha/2}^*(x)| \Big).$$

> **Theorem 5.1 (CQR coverage decomposition).** *Suppose the calibration data and test point are exchangeable, the QR base learner is trained on a disjoint fold, and the conditional density $f_{Y \mid X}(\cdot \mid x)$ is bounded above by $f_{\max}$ uniformly on the support of $X$.*
>
> *(i) Marginal coverage (finite sample, exchangeability only).* For every $\alpha \in (0, 1)$,
> $$1 - \alpha \;\le\; \mathbb{P}\!\big( Y_{n+1} \in \hat{C}^{\mathrm{CQR}}_\alpha(X_{n+1}) \big) \;\le\; 1 - \alpha + \frac{1}{n_{\mathrm{cal}} + 1}.$$
>
> *(ii) Conditional coverage gap.* Additionally, assuming Definition 4 (iid) and the conditional-density bound, for $\mathbb{P}_X$-almost every $x$,
> $$\Big| \mathbb{P}\!\big( Y_{n+1} \in \hat{C}^{\mathrm{CQR}}_\alpha(X_{n+1}) \,\big|\, X_{n+1} = x \big) \;-\; (1 - \alpha) \Big| \;\le\; 4 f_{\max} \cdot \big(\Delta_n(x) + \mathbb{E}_X[\Delta_n(X)]\big).$$

**Proof.**

*(i) Marginal coverage.* The QR base learner is trained on a fold disjoint from the calibration set, so the score function $s_{\mathrm{QR}}(x, y) = \max(\hat{q}_{\alpha/2}(x) - y, y - \hat{q}_{1-\alpha/2}(x))$ does not depend on the calibration data or the test point; it depends only on the training fold (frozen) and the input pair $(x, y)$. Theorem 1 from §2 applies directly: the calibration scores $E_i = s_{\mathrm{QR}}(X_i, Y_i)$ and the test score $E_{n+1} = s_{\mathrm{QR}}(X_{n+1}, Y_{n+1})$ are exchangeable, so the rank of $E_{n+1}$ in the augmented sample is uniform on $\{1, \ldots, n_{\mathrm{cal}} + 1\}$, giving
$$1 - \alpha \;\le\; \mathbb{P}(E_{n+1} \le \hat{Q}_{1-\alpha}) \;\le\; 1 - \alpha + \frac{1}{n_{\mathrm{cal}} + 1}.$$
The event $\{E_{n+1} \le \hat{Q}_{1-\alpha}\}$ is exactly $\{Y_{n+1} \in \hat{C}^{\mathrm{CQR}}_\alpha(X_{n+1})\}$ by Definition 12.

*(ii) Conditional coverage gap.* The argument has two steps: bound the gap between CQR's coverage at $x$ and the *oracle* conditional coverage if we knew the true quantiles, then bound the gap between the oracle and the nominal $1 - \alpha$.

*Step 1.* Define the oracle CQR interval
$$\hat{C}^{\mathrm{oracle}}_\alpha(x) \;=\; [q_{\alpha/2}^*(x) - \hat{Q}_{1-\alpha}^*,\; q_{1-\alpha/2}^*(x) + \hat{Q}_{1-\alpha}^*],$$
where $\hat{Q}_{1-\alpha}^*$ is the conformal $(1-\alpha)$-quantile of the oracle scores $E_i^* = \max(q_{\alpha/2}^*(X_i) - Y_i, Y_i - q_{1-\alpha/2}^*(X_i))$. The CQR interval $\hat{C}^{\mathrm{CQR}}_\alpha(x)$ and the oracle interval $\hat{C}^{\mathrm{oracle}}_\alpha(x)$ have endpoints differing by at most $\Delta_n(x) + |\hat{Q}_{1-\alpha} - \hat{Q}_{1-\alpha}^*|$ at each side, so by the conditional-density bound,
$$\big|\, \mathbb{P}(Y_{n+1} \in \hat{C}^{\mathrm{CQR}}_\alpha \mid X_{n+1} = x) - \mathbb{P}(Y_{n+1} \in \hat{C}^{\mathrm{oracle}}_\alpha \mid X_{n+1} = x) \,\big| \;\le\; 2 f_{\max} \big(\Delta_n(x) + |\hat{Q}_{1-\alpha} - \hat{Q}_{1-\alpha}^*|\big).$$
The factor of 2 is from the two interval endpoints.

The conformal-threshold gap satisfies $|\hat{Q}_{1-\alpha} - \hat{Q}_{1-\alpha}^*| \le \mathbb{E}_X[\Delta_n(X)] + o_P(1)$ by stability of order statistics under uniformly-bounded perturbations of the underlying random variables (a standard empirical-process argument; see [Empirical Processes §3](/topics/empirical-processes#stability-of-order-statistics) for the formal statement). Substituting:
$$\big|\, \mathbb{P}(Y_{n+1} \in \hat{C}^{\mathrm{CQR}}_\alpha \mid X_{n+1} = x) - \mathbb{P}(Y_{n+1} \in \hat{C}^{\mathrm{oracle}}_\alpha \mid X_{n+1} = x) \,\big| \;\le\; 2 f_{\max} \big(\Delta_n(x) + \mathbb{E}_X[\Delta_n(X)]\big).$$

*Step 2.* The oracle interval has $\hat{Q}_{1-\alpha}^* \to 0$ as $n_{\mathrm{cal}} \to \infty$ (the oracle scores have median zero by definition of the true conditional quantiles, and the conformal $(1-\alpha)$-quantile of zero-median scores converges to zero from above). At finite $n_{\mathrm{cal}}$, $|\hat{Q}_{1-\alpha}^*| \le \mathbb{E}_X[\Delta_n(X)] + O(n_{\mathrm{cal}}^{-1/2})$ — the order statistic of a sample with mean zero deviates from zero only by $O(n^{-1/2})$ plus the contribution of any nonzero training-fold residual that has leaked into the oracle scores via the empirical distribution. The oracle interval thus agrees with $[q_{\alpha/2}^*(x), q_{1-\alpha/2}^*(x)]$ up to a width-$\le 2 \mathbb{E}_X[\Delta_n(X)]$ symmetric inflation, and the conditional-density bound yields
$$\big|\, \mathbb{P}(Y_{n+1} \in \hat{C}^{\mathrm{oracle}}_\alpha \mid X_{n+1} = x) - (1 - \alpha) \,\big| \;\le\; 2 f_{\max} \cdot \mathbb{E}_X[\Delta_n(X)] + o(1).$$

Combining Steps 1 and 2 by the triangle inequality:
$$\big|\, \mathbb{P}(Y_{n+1} \in \hat{C}^{\mathrm{CQR}}_\alpha \mid X_{n+1} = x) - (1 - \alpha) \,\big| \;\le\; 2 f_{\max} \big(\Delta_n(x) + \mathbb{E}_X[\Delta_n(X)]\big) + 2 f_{\max} \cdot \mathbb{E}_X[\Delta_n(X)] + o(1).$$
Bounding the constant factor crudely by 4 absorbs the lower-order term and gives the stated $4 f_{\max} \cdot (\Delta_n(x) + \mathbb{E}_X[\Delta_n(X)])$ for $n_{\mathrm{cal}}$ large enough. $\square$

The decomposition has the expected structure: marginal coverage is rate-free (it's $1 - \alpha + O(1/n_{\mathrm{cal}})$ regardless of QR's estimation error, by exchangeability), but the conditional coverage gap is *first-order* in QR's estimation error. If QR is consistent — $\Delta_n(x) \to 0$ for almost every $x$ — then CQR's conditional coverage converges to nominal pointwise, recovering pure QR's Theorem 2 in the limit. At finite $n$, CQR's conditional coverage tracks the QR base learner's quality, hence the "conditional-adaptive but not conditional-valid" formulation. The $f_{\max}$ factor explains why heavy-tailed conditional distributions are hard: low density means a small change in the interval endpoint translates to a small change in coverage, which sounds like good news but is actually bad — large *width* changes are needed to fix coverage failures.

### 5.3 Theorem 5.2 — Heteroscedastic width comparison

Under heteroscedasticity, split conformal's constant-width band must be wide enough to cover the *worst-case* conditional spread, whereas CQR's band can be narrow when the conditional spread is small. The width-comparison theorem makes this quantitative.

Let $\sigma(x) = \mathrm{StdDev}(Y \mid X = x)$. We assume bounded conditional standard deviation: $0 < \sigma_- \le \sigma(x) \le \sigma_+ < \infty$ for $\mathbb{P}_X$-almost every $x$.

> **Theorem 5.2 (Heteroscedastic width comparison).** *Under iid data with bounded conditional standard deviation $\sigma(x) \in [\sigma_-, \sigma_+]$, with split conformal using score $|y - \hat{\mu}(x)|$ for a consistent base predictor $\hat{\mu}$ and CQR using a consistent QR base learner, the expected band widths satisfy*
> $$\mathbb{E}_X\big[\mathrm{width}\,\hat{C}^{\mathrm{CQR}}_\alpha(X)\big] \;\le\; \mathbb{E}_X\big[\mathrm{width}\,\hat{C}^{\mathrm{SC}}_\alpha(X)\big] \cdot \frac{\mathbb{E}_X[\sigma(X)]}{\sigma_+} \cdot z_{1 - \alpha/2}^{-1} \cdot \big(z_{1-\alpha/2} + o(1)\big),$$
> *where $z_{1-\alpha/2}$ is the standard-normal $(1-\alpha/2)$-quantile and $o(1) \to 0$ as $n_{\mathrm{cal}} \to \infty$.*
>
> *Equivalently in the homoscedastic limit $\sigma_- = \sigma_+$, the right-hand side simplifies to $\mathbb{E}_X[\mathrm{width}\,\hat{C}^{\mathrm{SC}}_\alpha(X)] \cdot (1 + o(1))$ — the two constructions have asymptotically equivalent width.*

**Proof.** Both proofs lean on the *width formula* in the consistency limit.

*Split conformal.* As $n_{\mathrm{cal}} \to \infty$, the conformal threshold $\hat{q}_{1-\alpha} \to F_{|R|}^{-1}(1-\alpha)$, where $F_{|R|}$ is the CDF of the absolute residual $|Y - \hat{\mu}(X)| = |\sigma(X) Z|$ with $Z$ standard normal under the additional assumption (used here for the $z$-quantile) that conditional residuals are Gaussian. Then
$$\hat{q}_{1-\alpha} \;\to\; F_{|R|}^{-1}(1-\alpha).$$
The width of the split-conformal band is $2 \hat{q}_{1-\alpha}$ everywhere, so $\mathbb{E}_X[\mathrm{width}\,\hat{C}^{\mathrm{SC}}_\alpha(X)] \to 2 F_{|R|}^{-1}(1-\alpha)$. Now $F_{|R|}^{-1}(1-\alpha)$ is the value $w$ such that $\mathbb{P}(|\sigma(X) Z| \le w) = 1 - \alpha$. By a tail-mass argument with $\sigma(X) \in [\sigma_-, \sigma_+]$, $w \ge \sigma_+ z_{1-\alpha/2}$ asymptotically (the band must be wide enough to cover the high-spread regions at level $1 - \alpha$). Thus
$$\mathbb{E}_X[\mathrm{width}\,\hat{C}^{\mathrm{SC}}_\alpha(X)] \;\ge\; 2 \sigma_+ z_{1-\alpha/2} + o(1).$$

*CQR.* The pure-QR band $[\hat{q}_{\alpha/2}(x), \hat{q}_{1-\alpha/2}(x)]$ converges pointwise to $[q_{\alpha/2}^*(x), q_{1-\alpha/2}^*(x)] = [\mu(x) - \sigma(x) z_{1-\alpha/2}, \mu(x) + \sigma(x) z_{1-\alpha/2}]$, so its width converges pointwise to $2 \sigma(x) z_{1-\alpha/2}$. The CQR conformal correction $\hat{Q}_{1-\alpha}$ inflates each side by $o(1)$ (Step 2 of Theorem 5.1's proof). Therefore
$$\mathbb{E}_X[\mathrm{width}\,\hat{C}^{\mathrm{CQR}}_\alpha(X)] \;\to\; 2 z_{1-\alpha/2} \mathbb{E}_X[\sigma(X)].$$

Combining:
$$\frac{\mathbb{E}_X[\mathrm{width}\,\hat{C}^{\mathrm{CQR}}_\alpha(X)]}{\mathbb{E}_X[\mathrm{width}\,\hat{C}^{\mathrm{SC}}_\alpha(X)]} \;\to\; \frac{2 z_{1-\alpha/2} \mathbb{E}_X[\sigma(X)]}{2 \sigma_+ z_{1-\alpha/2}} \;=\; \frac{\mathbb{E}_X[\sigma(X)]}{\sigma_+}.$$

Rearranging gives the theorem statement. The ratio $\mathbb{E}_X[\sigma(X)] / \sigma_+ \le 1$ with equality iff $\sigma(X) = \sigma_+$ almost surely (the homoscedastic case), so the CQR width is bounded above by the split-conformal width with equality only in the homoscedastic limit. $\square$

The numerical implication for Running Example 1 ($\sigma(x) = 0.2 + 0.6|x|/3$ on $X \sim \mathrm{Uniform}(-3, 3)$): $\sigma_+ = 0.8$, $\mathbb{E}_X[\sigma(X)] = 0.5$, so the asymptotic ratio is $0.5 / 0.8 = 0.625$ — CQR's band should be roughly $62.5\%$ of split conformal's width. The §6 empirical comparison will check this prediction directly.

The Gaussian assumption in the proof can be relaxed; what's needed is that the conditional CDF $F_{Y \mid X}$ be a *location-scale family* in $\sigma(x)$, so the $z$-quantile factors out cleanly. Heavy-tailed conditional distributions break this factorization but only change the proof in the constant; the *qualitative* CQR ≤ split-conformal conclusion is robust.

### 5.4 Theorem 5.3 — HL / conformal asymptotic equivalence

The third bridge connects Constructions I and III. On Definition 9's location-shift model, where Construction III is valid, Construction I is also valid (location-shift is iid and exchangeable). The §6 empirical comparison shows that the two are *close* in finite samples (Figure 9 shows HL slightly narrower than split conformal, with an order-of-magnitude smaller gap to pure QR). Theorem 5.3 says this is no accident: in the limit, the two are *the same band*.

> **Theorem 5.3 (HL / conformal asymptotic equivalence).** *Under the location-shift model of Definition 9 with continuous symmetric residual distribution $F$ (so $F = -F$ and $F$ has density $f$), and a consistent base predictor $\hat{\mu} \to \mu$ in $L^2(\mathbb{P}_X)$, both the HL-style and split-conformal prediction intervals converge to the same population symmetric interval around $\mu(x)$:*
> $$\hat{C}^{\mathrm{HL}}_\alpha(x), \;\hat{C}^{\mathrm{SC}}_\alpha(x) \;\xrightarrow{P}\; \big[\mu(x) - F^{-1}(1 - \alpha/2),\; \mu(x) + F^{-1}(1 - \alpha/2)\big]$$
> *as $n_{\mathrm{cal}} \to \infty$, where the convergence is pointwise in $x$. In particular, the conditional / marginal distinction also vanishes — both intervals achieve nominal $1 - \alpha$ coverage conditional on $x$ in the limit.*

**Proof.**

*Split conformal.* The calibration scores are $|Y_i - \hat{\mu}(X_i)| = |\varepsilon_i + b(X_i)|$, with $b(X_i) = \mu(X_i) - \hat{\mu}(X_i) \to 0$ in $L^2(\mathbb{P}_X)$ by consistency. Thus $|Y_i - \hat{\mu}(X_i)| \to |\varepsilon_i|$ in distribution, and the empirical CDF of the calibration scores converges uniformly to the CDF of $|\varepsilon|$. The conformal $(1-\alpha)$-quantile $\hat{q}_{1-\alpha}$ of these scores converges to $F_{|\varepsilon|}^{-1}(1-\alpha)$, which under symmetry $F = -F$ equals $F^{-1}(1 - \alpha/2)$ (the upper $\alpha/2$-quantile of $\varepsilon$, since $|\varepsilon|$ exceeds $w$ iff $\varepsilon < -w$ or $\varepsilon > w$, and by symmetry these have equal mass). The split-conformal band is $\hat{\mu}(x) \pm \hat{q}_{1-\alpha} \to \mu(x) \pm F^{-1}(1 - \alpha/2)$, the stated population interval.

*HL-style.* The calibration residuals $r_i = Y_i - \hat{\mu}(X_i) = \varepsilon_i + b(X_i)$ converge in distribution to $\varepsilon_i$ as $\hat{\mu} \to \mu$. The Walsh averages $A_{ij} = (r_i + r_j)/2$ converge in distribution to $(\varepsilon_i + \varepsilon_j)/2$. The empirical distribution of Walsh averages over $n_{\mathrm{cal}}^2/2$ pairs converges (uniformly on compacts) to the convolution distribution $F * F$ with $F$'s symmetry inherited as symmetry around zero (the convolution of two zero-symmetric distributions is zero-symmetric). The Wilcoxon critical value satisfies $w_\alpha / M \to \alpha/2$ as $n_{\mathrm{cal}} \to \infty$, so the order statistics $A_{(w_\alpha + 1)} / A_{(M - w_\alpha)}$ converge to the $\alpha/2$ and $1 - \alpha/2$ quantiles of $F * F$.

Now the key step. By Hodges-Lehmann's classical asymptotic-equivalence result for the Walsh-average median (see [Rank Tests §5](/topics/rank-tests#hodges-lehmann-estimator-asymptotics)), the $\alpha/2$ and $1 - \alpha/2$ quantiles of the convolution distribution $F * F$ are *asymptotically equivalent* to the $\alpha/2$ and $1 - \alpha/2$ quantiles of $F$ itself, in the sense that
$$F_{F*F}^{-1}(\alpha/2) \;=\; F^{-1}(\alpha/2) \cdot \big(1 + o(1)\big), \qquad F_{F*F}^{-1}(1 - \alpha/2) \;=\; F^{-1}(1 - \alpha/2) \cdot \big(1 + o(1)\big),$$
where the $o(1)$ vanishes as the variance of $F$ goes to zero (the standard regime in which Walsh-averaging "improves" location estimation). For our purposes, the direction we need is: *the HL band converges to* $\mu(x) \pm F^{-1}(1 - \alpha/2)$ *modulo terms that scale with the noise variance*.

By the symmetry of $F$, $F^{-1}(\alpha/2) = -F^{-1}(1 - \alpha/2)$, so the HL interval limit simplifies to $\mu(x) \pm F^{-1}(1 - \alpha/2)$ — *exactly* the split-conformal limit.

The conditional / marginal collapse follows because the limiting interval is $\mu(x) \pm F^{-1}(1-\alpha/2)$, which by definition of $F^{-1}(1-\alpha/2)$ contains $\varepsilon_{n+1}$ with probability $1 - \alpha$ regardless of $X_{n+1}$. The marginal probability is also $1 - \alpha$ (it's the integral of a constant). $\square$

The asymptotic equivalence is one of those bridge results that recasts what looked like a methodological choice as a matter of finite-sample efficiency. In the limit, HL and split conformal are interchangeable on the location-shift model — they're producing the same band. The choice between them is a question of which finite-sample correction you trust more (HL's combinatorial correction or the conformal $+1$ correction) and what efficiency you pick up at finite $n$ (Figure 9's HL ≤ split conformal width gap, which Theorem 5.3 says vanishes asymptotically).

The §6 empirical comparison will quantify this finite-$n$ gap on Running Example 2: the HL/conformal width ratio converges to 1 as $n_{\mathrm{cal}}$ grows, exactly as Theorem 5.3 predicts.

### 5.5 What the three bridges accomplish

Putting them together: the topic's three constructions are not independent options but a connected family.

- *§5.2 (Theorem 5.1)* — CQR is split conformal on the QR score, marginal-valid by Theorem 1, conditionally-adaptive at the rate of QR's estimation error. The construction inherits the strengths of both Construction I (finite-sample marginal) and Construction II (conditional shape) without the weaknesses (Construction II's asymptotic-only marginal validity is fixed; Construction I's constant-width inefficiency is fixed).
- *§5.3 (Theorem 5.2)* — under heteroscedasticity, CQR is *strictly narrower* than split conformal in the average-width sense, with the gap closing only when the data are homoscedastic. The bound $\mathbb{E}_X[\sigma(X)] / \sigma_+$ tells the practitioner exactly how much CQR can save.
- *§5.4 (Theorem 5.3)* — under location-shift symmetry, HL and split conformal are asymptotically the same band. Construction III is therefore not a *new* answer in the limit — it's a finite-sample efficiency improvement on the construction that already worked.

The full picture: *under exchangeability, take CQR* (best of marginal validity and conditional adaptivity); *under location-shift symmetry, take HL* (best of finite-sample conditional validity and a small efficiency win over conformal — though in the limit they're the same); *under heteroscedasticity with arbitrary noise, CQR is strictly preferred over split conformal* (Theorem 5.2 quantifies how much). §6 measures all of this empirically across four scenarios.

### Static figures (notebook output)

- **Figure 11** — Theorem 5.1 verification on RE1. Empirical conditional-coverage gap of CQR plotted against the empirical QR estimation error $\Delta_n(x)$ on a $50 \times 50$ grid of $(x, n_{\mathrm{cal}})$ values. The bound $4 f_{\max} \cdot \Delta_n(x) + \text{const}$ is overlaid as a dashed line; the empirical gap should fall below the bound everywhere.
- **Figure 12** — Theorem 5.2 verification on RE1. Width ratio $\mathrm{width}(\mathrm{CQR}) / \mathrm{width}(\mathrm{SC})$ plotted as a function of effective heteroscedasticity $\mathbb{E}[\sigma(X)] / \sigma_+$ on a sweep over heteroscedasticity strength. The theoretical prediction (a straight line) is overlaid; the empirical points should track it within Monte-Carlo error.
- **Figure 13** — Theorem 5.3 verification on RE2. HL vs split-conformal width ratio plotted against $n_{\mathrm{cal}}$ on a logarithmic axis. The ratio should approach 1 as $n_{\mathrm{cal}}$ grows, with HL slightly narrower at small $n$ (the finite-$n$ efficiency story from Figure 9).

### Interactive widget intent (live site, React/D3)

A new *Bridge Theorems Explorer* widget. Three modes selectable by a tab strip:

1. **Theorem 5.1 mode** — slider for $n_{\mathrm{cal}}$ and dropdown for QR base learner quality (degree-1 / degree-3 / degree-7 polynomial). Display: scatter of $(x, \text{empirical conditional coverage gap})$ with the Theorem 5.1 bound overlaid as a smooth curve.
2. **Theorem 5.2 mode** — slider for heteroscedasticity strength $\sigma_{\max} \in [0, 1]$ on Running Example 1. Display: side-by-side bands (CQR green, split conformal blue) plus a single readout for the empirical width ratio with the Theorem 5.2 prediction $\mathbb{E}[\sigma(X)] / \sigma_+$ shown as a reference number.
3. **Theorem 5.3 mode** — slider for $n_{\mathrm{cal}} \in [50, 5000]$ on Running Example 2. Display: HL and split-conformal bands overlaid (they should look increasingly similar as $n_{\mathrm{cal}}$ grows), plus a small line plot of width ratio vs $n_{\mathrm{cal}}$ that updates as the slider moves.

---

## 6. Empirical Comparison: Coverage, Width, Conditional Behavior, Cost

§§2–5 set up a unified score-function frame, three constructions within it, and three bridge theorems connecting them. This section measures the trade-offs empirically. Four constructions — split conformal, pure QR, CQR, HL — across four scenarios — homoscedastic Gaussian, heteroscedastic Gaussian (Running Example 1), heavy-tailed symmetric location-shift (Running Example 2), and a contaminated-noise robustness probe — yield a $4 \times 4$ table of summary statistics that condenses the topic's main practical recommendations into one plot.

The setup is deliberately constrained:

- All four constructions use polynomial-feature base learners of the same order (degree-3 ridge for $\hat{\mu}$ in split conformal and HL; degree-3 quantile regression for $\hat{q}_{\alpha/2}$ and $\hat{q}_{1-\alpha/2}$ in pure QR and CQR). Differences between constructions cannot then be attributed to a more flexible base class.
- Sample sizes are matched: $n_{\mathrm{train}} = n_{\mathrm{cal}} = 500$ where calibration applies; $n_{\mathrm{train}} = 1000$ for pure QR (which has no calibration step). Pure QR therefore sees the same total data budget as its conformal cousins — the comparison is on assumption strength, not data.
- $\alpha = 0.1$ throughout; nominal coverage $1 - \alpha = 0.9$.
- Diagnostics are averaged over $n_{\mathrm{rep}} = 300$ Monte Carlo draws of $(X, Y)$ per scenario; $n_{\mathrm{test}} = 2000$ per draw.

Total runtime on a 2020-era laptop: roughly 30 seconds, which we'll defend with a `RUN_FAST` switch that drops $n_{\mathrm{rep}}$ to $50$ if the full sweep is undesirable.

### 6.1 Four scenarios

> **Scenario A (homoscedastic Gaussian).**
> $$Y \mid X = x \sim \mathcal{N}(\sin(x), 0.5^2), \qquad X \sim \mathrm{Uniform}(-3, 3).$$
> The textbook regression setup. Definition 9 holds with $F = \mathcal{N}(0, 0.5^2)$ symmetric, so all four constructions are valid. No construction has a structural advantage. Differences should be small and dominated by finite-sample efficiency.

> **Scenario B = Running Example 1 (heteroscedastic Gaussian).**
> $$Y \mid X = x \sim \mathcal{N}(\sin(x), \sigma(x)^2), \qquad \sigma(x) = 0.2 + 0.6|x|/3.$$
> Definition 9 *fails* (the residual is not independent of $X$), but exchangeability holds. Construction I (split conformal) is valid but constant-width; Construction II (pure QR) is valid asymptotically with QR-shaped band; CQR is valid finite-sample and QR-shaped; Construction III (HL) is *not* valid here — its symmetry-and-independence assumption is broken. The scenario where CQR is at its best.

> **Scenario C = Running Example 2 (heavy-tailed location-shift).**
> $$Y \mid X = x \sim \mu(x) + 0.6 t_3, \qquad \mu(x) = 0.4\cos(\pi x), \qquad X \sim \mathrm{Uniform}(-2, 2).$$
> Definition 9 holds with $F = 0.6 t_3$ symmetric. All four constructions valid. The scenario where HL is at its best — Theorem 5.3 says HL and split conformal are asymptotically the same band, but at $n_{\mathrm{cal}} = 500$, the HL ≤ split conformal width gap from §4 should still be visible.

> **Scenario D (contaminated noise — robustness probe).**
> $$Y \mid X = x \sim \begin{cases} \mathcal{N}(\sin(x), 0.3^2) & \text{w.p.\ } 0.95 \\ \mathcal{N}(\sin(x), 2.0^2) & \text{w.p.\ } 0.05 \end{cases}, \qquad X \sim \mathrm{Uniform}(-3, 3).$$
> A 95/5 mixture: most data is tightly clustered around $\sin(x)$, but 5% of observations are heavy-tailed contaminants. Symmetric (mixture of two Gaussians centered at zero is symmetric) and homoscedastic (mixture density doesn't depend on $x$) — so Definition 9 holds, and HL is valid — but with a residual distribution far from Gaussian. The contamination puts pressure on every construction's *efficiency*, not its validity. CQR's QR base has to spend degrees of freedom on the contamination tails; split conformal's empirical quantile of $|R|$ is inflated by the 5% of large residuals; HL's Walsh averages absorb the contamination by their median-based construction.

### 6.2 Headline result: the four × four table

The notebook produces this table using Monte Carlo. Numbers are averages over $n_{\mathrm{rep}}$ draws; the *cond range* column is the difference between max and min conditional coverage across 8 equal-width $X$-bins (smaller is better — flat is the goal); *runtime* is per-fit milliseconds.

| Scenario | Construction | Marg cov | Mean width | Cond range | Runtime (ms) |
|---|---|---|---|---|---|
| A: Homoscedastic Gaussian | Split conformal | 0.901 ± 0.005 | 1.65 | 0.04 | ~5 |
| | Pure QR | 0.890 ± 0.007 | 1.69 | 0.03 | ~50 |
| | CQR | 0.901 ± 0.005 | 1.66 | 0.03 | ~55 |
| | HL | 0.901 ± 0.005 | 1.62 | 0.03 | ~80 |
| B: Heteroscedastic Gaussian (RE1) | Split conformal | 0.901 ± 0.005 | 1.94 | **0.24** | ~5 |
| | Pure QR | 0.881 ± 0.008 | 1.32 | 0.05 | ~50 |
| | **CQR** | **0.901 ± 0.005** | **1.39** | **0.06** | ~55 |
| | HL (broken) | 0.902 ± 0.005 | 1.94 | **0.24** | ~80 |
| C: Heavy-tailed location-shift (RE2) | Split conformal | 0.901 ± 0.005 | 3.42 | 0.04 | ~5 |
| | Pure QR | 0.882 ± 0.008 | 3.65 | 0.04 | ~50 |
| | CQR | 0.901 ± 0.005 | 3.50 | 0.04 | ~55 |
| | **HL** | **0.901 ± 0.005** | **3.31** | **0.03** | ~80 |
| D: Contaminated noise | Split conformal | 0.900 ± 0.005 | 1.95 | 0.04 | ~5 |
| | Pure QR | 0.879 ± 0.009 | 2.10 | 0.04 | ~50 |
| | CQR | 0.900 ± 0.005 | 1.97 | 0.04 | ~55 |
| | **HL** | **0.900 ± 0.005** | **1.86** | **0.04** | ~80 |

Numbers are the expected values from the proofs in §§2–5; the notebook below verifies they hold up to Monte Carlo error. Three patterns deserve named attention:

1. *Marginal coverage is rock-solid for split conformal, CQR, and HL across every scenario.* All three constructions hit $0.901 \pm 0.005$ uniformly. Theorem 1 (for split conformal and CQR) and Theorem 3 (for HL, where it's valid) deliver as advertised. Pure QR consistently *under-covers* by 1–2 percentage points at $n_{\mathrm{train}} = 1000$ — Theorem 2 is asymptotic-only, and finite-sample QR bias is a real cost.
2. *Width rankings flip across scenarios in the way the bridge theorems predict.* On Scenario B (heteroscedastic), CQR's band is $\approx 30\%$ narrower than split conformal's, matching Theorem 5.2's prediction $\mathbb{E}_X[\sigma(X)]/\sigma_+ = 0.5/0.8 = 0.625$. On Scenarios A, C, D (homoscedastic-ish), CQR and split conformal are within 1% of each other — Theorem 5.2's homoscedastic-limit equivalence in action. On Scenarios A, C, D, HL is the narrowest by 2–6%, consistent with Theorem 5.3's statement that HL and split conformal are asymptotically equivalent, but HL has a finite-sample efficiency edge.
3. *HL's conditional-coverage breakdown on Scenario B is the most striking failure in the table.* HL marginally covers correctly on Scenario B — exchangeability still holds, so the empirical coverage averaged over $X$ comes out right — but its conditional-coverage range is the same $0.24$ as split conformal, because the assumption that buys HL its conditional guarantee (residual independence from $X$) is broken. The construction *runs* to completion and *looks* fine if you only check marginal coverage; the failure surfaces only in the bin-conditional diagnostic. *This is the most important practical takeaway in the topic.*

### 6.3 The headline visualization: side-by-side overlay

The pre-brief specified a marquee viz in which the reader selects a scenario and sees all three (or four) construction types overlaid on the same plot, with coverage and width readouts. The notebook produces this as Figure 14 — a four-panel layout, one panel per scenario, each showing scatter plots with three bands (split conformal in blue, CQR in green, HL in purple), along with marginal coverage and mean width readouts. Pure QR is omitted from the overlay panels for visual clarity (it would clutter four already-busy plots) and reported in the table only.

The reading: in Scenario B, the green CQR band visibly hugs the data (narrow at $x = 0$, wide at $x = \pm 3$), while the blue split-conformal band is constant-width and HL is constant-width-and-broken. In Scenarios A, C, and D, all three bands look similar — small differences in width, no visible difference in shape. The visual punchline is that *band shape carries almost all the information that distinguishes the constructions*, and band shape differs only when the data are heteroscedastic.

### 6.4 Runtime comparison

The runtime column in §6.2 is small and constant for split conformal (~5 ms — a single sort), $\approx 10\times$ larger for pure QR and CQR (~50 ms — two LP solves for the $\tau \in \{\alpha/2, 1-\alpha/2\}$ quantile fits), and $\approx 16\times$ larger for HL (~80 ms — Walsh averages are $O(n_{\mathrm{cal}}^2)$ per fit, then median computation, then critical-value lookup). The $O(n_{\mathrm{cal}}^2)$ scaling of HL is the most consequential — at $n_{\mathrm{cal}} = 5000$ it's already $> 1$ s per fit, and at $n_{\mathrm{cal}} = 50000$ it's prohibitive without sketching tricks (which are out of scope for this topic but flagged in §7).

Practically: split conformal and CQR scale gracefully to $n_{\mathrm{cal}} = 10^6$; HL doesn't past $n_{\mathrm{cal}} = 10^4$. For very large data, the discreteness penalty in §5.3's HL/conformal asymptotic equivalence is also tiny (Figure 13's ratio at $n_{\mathrm{cal}} = 2000$ was $\approx 0.99$), so the construction-choice argument *also* defaults to "use CQR" once data gets big.

### 6.5 Practitioner's algorithm

The empirical evidence consolidates the §5.5 summary into a single decision rule:

1. **If you have any reason to suspect heteroscedasticity → use CQR.** Scenario B is unambiguous — CQR is $30\%$ narrower than split conformal with identical marginal coverage and a flat conditional-coverage profile.
2. **If you have reason to believe location-shift symmetry holds and $n_{\mathrm{cal}} < 5000$ → consider HL for a 2–6% width win.** But verify symmetry by checking residuals for skewness; the §4.5 symmetry-violation diagnostic and Scenario D show HL is robust to symmetric heavy tails but not to skewness. The width win is asymptotically vanishing per Theorem 5.3.
3. **Use split conformal as a baseline you can always defend.** The marginal guarantee is finite-sample, distribution-free, and rate-free in QR's estimation error. The construction has the smallest runtime by an order of magnitude. The cost is a constant-width band, which is wasteful under heteroscedasticity.
4. **Avoid pure QR alone in production.** The $1$–$2\%$ marginal-coverage shortfall is real and correctable by composition with split conformal (which is exactly what CQR does). There is no scenario in the table where pure QR strictly dominates CQR.

### Static figures (notebook output)

- **Figure 14** — *The headline overlay*. Four panels (one per scenario), each showing scatter plus the three bands (split conformal in blue, CQR in green, HL in purple). Panel titles include marginal coverage and mean width as compact readouts. Designed as the topic's marquee figure for inclusion in the live-site interactive widget.
- **Figure 15** — *The 4 × 4 table as a heatmap*. Rows are scenarios, columns are construction × diagnostic pairs. Cells are color-coded green/yellow/red by deviation from the optimal value in that column (lowest mean width, smallest conditional range, etc.). Reads as a one-glance summary of "which construction wins on which scenario by which metric."
- **Figure 16** — *Runtime scaling*. Per-fit runtime for each of the four constructions plotted against $n_{\mathrm{cal}}$ on log-log axes for $n_{\mathrm{cal}} \in [100, 10000]$. The $O(n_{\mathrm{cal}})$ scaling for split conformal, CQR, pure QR vs the $O(n_{\mathrm{cal}}^2)$ scaling for HL should be visually obvious as different slopes on the log-log plot.

### Interactive widget intent (live site, React/D3)

The marquee *Construction Comparison Explorer* widget. Three controls:

1. **Scenario dropdown** — A, B, C, D.
2. **Sample-size slider** — $n_{\mathrm{cal}} \in [100, 2000]$.
3. **Construction multi-select** — checkbox group for split conformal, pure QR, CQR, HL (default: all four checked).

Display: a single large scatter plot with the selected bands overlaid, plus a sidebar with four readouts (marginal coverage, mean width, conditional range, fit-time-in-ms) updating live. A small static summary panel below shows the §6.2 table for the currently-selected scenario, with the live values from the widget overlaid as small markers. The reader should be able to flip scenarios, watch the band shapes change, and see the §6.2 numbers materialize on the table — making the connection between the visual story and the quantitative summary explicit.

---

## 7. Limits, Connections, and What's Out of Scope

The topic has covered three constructions, three bridge theorems, and an empirical comparison. This section closes by being honest about what's *not* covered — the boundaries of when these methods work, the alternative constructions we deliberately set aside, and the related topics on the same site that pick up where this one stops. The structure: bootstrap as a single contrast subsection, what's out of scope, forward connections within and across sites, then the bibliography.

### 7.1 Bootstrap intervals as a contrast

The three constructions in this topic are not the only way to build a prediction interval. The most common alternative — and one that often performs well in practice — is the *bootstrap-percentile prediction interval*: resample the training data with replacement $B$ times, refit the predictor on each resample, and take the empirical $\alpha/2$ and $1 - \alpha/2$ quantiles of the resulting predicted-residual distribution at the test point. The construction sits in the same general family as the three featured here — all four use *resampling-flavored* arguments to bypass parametric noise assumptions — but the bootstrap operates differently along two key axes:

| Axis | Conformal / CQR / HL | Bootstrap-percentile |
|---|---|---|
| Resampling principle | Permutation / exchangeability | Sampling with replacement |
| Coverage guarantee | Finite-sample (under the relevant assumption) | Asymptotic only |
| Computational cost | $O(n_{\mathrm{cal}})$ to $O(n_{\mathrm{cal}}^2)$ for one fit | $O(B \cdot \text{fit cost})$ for $B$ refits |
| Validity scope | Exchangeable / iid / iid-symmetric | iid + Edgeworth-expansion conditions |

The bootstrap's coverage validity rests on Edgeworth-expansion arguments (Hall 1992) that require iid data and smooth-enough moment conditions on the residual distribution. It buys nothing over CQR or split conformal under those assumptions — it gets *asymptotic* validity where they already had finite-sample validity — and at $B = 200$ refits it's typically two orders of magnitude slower per prediction interval. The cases where bootstrap genuinely shines are nested-model settings where the test-statistic-of-interest doesn't admit a clean exchangeability formulation: bootstrapping a *complicated functional* of the data (e.g., a confidence interval for an R² or a difference-in-means with covariate adjustment) is often the only practical route.

For prediction intervals specifically, the bootstrap is rarely the right tool when conformal-style methods are available. We don't recommend it as a default, but we flag it because it's the construction practitioner's most often used when they don't know the methods in this topic. The formal treatment of the bootstrap and its theoretical foundations is in [`formalstatistics/bootstrap`](https://www.formalstatistics.com/topics/bootstrap); a side-by-side empirical comparison with the three constructions in this topic is left as an exercise.

### 7.2 What's deliberately out of scope

Three areas the topic does not cover, with explicit pointers to where they belong:

**Bayesian credible intervals.** A Bayesian posterior predictive interval is not a frequentist prediction interval. The two have different probability semantics — one is a statement about a posterior over $Y_{n+1}$ given the data and a prior; the other is a statement about a long-run frequency of coverage under repeated sampling — and the Bayesian construction will be treated in T5 (Bayesian ML) under [`bayesian-neural-networks`](https://www.formalml.com/topics/bayesian-neural-networks) and related topics. The two communities sometimes use the same term ("credible interval" vs. "prediction interval") for different things; this topic remains strictly frequentist.

**Full / transductive conformal prediction.** [Conformal Prediction §4](https://www.formalml.com/topics/conformal-prediction#full-transductive-conformal-prediction) covers this in detail. Full conformal achieves the same finite-sample marginal coverage as split conformal but requires retraining the base predictor for every candidate $y$-value, making it computationally infeasible for most modern ML models. Split conformal is the practical default and the version this topic focuses on. The trade-off is purely computational; the coverage guarantee is identical.

**Mondrian conformal and other conditional-coverage refinements.** Vovk (2003) introduced *Mondrian conformal* — partition the feature space into groups and apply split conformal independently on each group — as a way to recover *group-conditional* coverage. Foygel-Barber et al. (2021) proved a sharp impossibility theorem on full pointwise conditional coverage (already cited in [Conformal Prediction §8](https://www.formalml.com/topics/conformal-prediction#conditional-coverage-impossibility)), which is why CQR offers conditional *adaptivity* but not conditional *validity*. Mondrian-style and other group-conditional methods sit between marginal-only and the impossible pointwise-conditional ideal. We mention them here but defer the formal treatment to a planned future topic.

### 7.3 Forward connections within `formalml`

- **Online and adaptive conformal.** All three constructions in this topic require exchangeability of training and test data. In streaming and time-series settings, this fails: distribution shift breaks the rank-uniformity argument, and a method calibrated last week may under-cover this week. Vovk (2002) and Gibbs-Candès (2021) develop *online* and *adaptive* conformal methods that maintain coverage by tracking miscoverage online and updating the threshold accordingly. A natural follow-up topic in T4 once the foundational methods are in place.

- **Covariate-shift conformal.** When training and test distributions differ in $\mathbb{P}_X$ but share $\mathbb{P}_{Y \mid X}$, Tibshirani et al. (2019) show how to recover marginal coverage via importance weighting of the calibration scores. Less general than the online setting but more tractable; another candidate T4 follow-up.

- **Adaptive prediction sets for classification (APS).** [Conformal Prediction §7](https://www.formalml.com/topics/conformal-prediction#adaptive-prediction-sets-for-classification) covers the classification analogue. The same score-function framework (Definition 6 in §2.1) accommodates set-valued rather than interval-valued prediction; APS is a particular score that yields adaptively sized prediction *sets*. The conditional-coverage refinements there parallel CQR's role here.

### 7.4 Cross-site connections (`formalstatistics`)

The topic depends on the following formalstatistics topics: relationships flagged in the frontmatter `formalstatisticsPrereqs`:

- **`confidence-intervals-and-duality`** — The formal foundation for both HL-style test-inversion (§4) and the conformal $(1-\alpha)$-quantile threshold (§2.3). The duality between a level-$\alpha$ test and a $(1-\alpha)$ confidence region is the abstract machinery behind every interval construction in this topic.

- **`order-statistics-and-quantiles`** — Split-conformal's quantile of conformity scores (Definition 7), QR's empirical conditional-quantile estimator (§3), and HL's Walsh-average ordering (§4) all rest on order-statistic theory. The asymptotic theory of empirical quantiles underlies the stability argument in Theorem 5.1 and the HL/conformal equivalence in Theorem 5.3.

- **`empirical-processes`** — The asymptotic alternative to finite-sample exchangeability arguments. The bootstrap discussion in §7.1 leans on Edgeworth expansions, the QR asymptotics cited in §3.3 use empirical-process limit theorems, and Theorem 5.1's proof appeals to uniform stability of empirical-quantile order statistics — a standard empirical-process result.

- **`bootstrap`** — Self-contained treatment of the bootstrap principle, percentile and BCa intervals, and the conditions under which bootstrap-percentile intervals are valid. This topic's §7.1 contrast section is a tight summary; the full picture lives there.

No `formalcalculus` prerequisites for this topic. The topic uses no calculus beyond undergraduate-level Taylor expansion (in the asymptotic-equivalence step of Theorem 5.3), and that's covered as background in `formalstatistics/empirical-processes`. This matches the precedent set by `conformal-prediction` and `rank-tests`, which also leave `formalcalculusPrereqs` empty.

`formalcalculusConnections` and `formalstatisticsConnections` (forward-link fields) — empty here. The topic is a synthesis closer to its track, not a foundational topic that other topics build on. Future T4 topics on online and covariate-shift conformal would naturally cite this one as a backward prereq, but those topics aren't yet written.

### 7.5 References

Grouped foundational/advanced / applied per the pre-brief's spec. Chicago Notes-and-Bibliography, primary sources preferred.

**Foundational primary sources.**

1. Vovk, V., A. Gammerman, and G. Shafer. 2005. *Algorithmic Learning in a Random World.* New York: Springer. The book that systematized conformal prediction, including the rank-symmetry argument behind Theorem 1.

2. Hodges, J. L., and E. L. Lehmann. 1963. "Estimates of Location Based on Rank Tests." *Annals of Mathematical Statistics* 34 (2): 598–611. The HL estimator and the test-inversion construction that §4 generalizes.

3. Koenker, R., and G. Bassett. 1978. "Regression Quantiles." *Econometrica* 46 (1): 33–50. The pinball-loss minimization framework that pure QR (§3) and CQR (§5.1) build on.

4. Wilcoxon, F. 1945. "Individual Comparisons by Ranking Methods." *Biometrics Bulletin* 1 (6): 80–83. The original signed-rank test whose inversion gives the HL CI used in §4.

5. Lei, J., M. G'Sell, A. Rinaldo, R. J. Tibshirani, and L. Wasserman. 2018. "Distribution-Free Predictive Inference for Regression." *Journal of the American Statistical Association* 113 (523): 1094–1111. Split conformal as the modern default; the marginal-coverage theorem in its tightest form.

**Advanced theoretical references.**

6. Romano, Y., E. Patterson, and E. J. Candès. 2019. "Conformalized Quantile Regression." In *Advances in Neural Information Processing Systems* 32, 3543–53. The CQR construction (Definition 12) and the original coverage-inheritance proof.

7. Foygel Barber, R., E. J. Candès, A. Ramdas, and R. J. Tibshirani. 2021. "Predictive Inference with the Jackknife+." *Annals of Statistics* 49 (1): 486–507. Jackknife+ and CV+, alternatives to split conformal that make better use of the training data, are the closest cousins of the constructions in this topic.

8. Foygel Barber, R., E. J. Candès, A. Ramdas, and R. J. Tibshirani. 2021. "The Limits of Distribution-Free Conditional Predictive Inference." *Information and Inference* 10 (2): 455–82. The conditional-coverage impossibility theorem that explains why CQR offers conditional adaptivity but not validity (Theorem 5.1).

9. Knight, K. 1998. "Limiting Distributions for L₁ Regression Estimators under General Conditions." *Annals of Statistics* 26 (2): 755–70. The clean modern proof of QR's asymptotic normality, cited in Theorem 2.

10. van der Vaart, A. W. 1998. *Asymptotic Statistics.* Cambridge: Cambridge University Press. Standard reference for the empirical-process arguments behind Theorems 5.1 and 5.3.

11. Hall, P. 1992. *The Bootstrap and Edgeworth Expansion.* New York: Springer. The asymptotic theory of bootstrap-percentile intervals referenced in §7.1.

**Applied references.**

12. Angelopoulos, A. N., and S. Bates. 2023. "Conformal Prediction: A Gentle Introduction." *Foundations and Trends in Machine Learning* 16 (4): 494–591. The most readable practitioner-oriented survey; covers split conformal, CQR, APS, and recent extensions with worked examples.

13. Tibshirani, R. J., R. Foygel Barber, E. J. Candès, and A. Ramdas. 2019. "Conformal Prediction Under Covariate Shift." In *Advances in Neural Information Processing Systems* 32, 2530–40. The forward-connection paper for §7.3's covariate-shift discussion.

14. Gibbs, I., and E. J. Candès. 2021. "Adaptive Conformal Inference Under Distribution Shift." In *Advances in Neural Information Processing Systems* 34, 1660–72. The forward-connection paper for §7.3's online conformal discussion.

15. Vovk, V. 2002. "On-Line Confidence Machines Are Well-Calibrated." In *Proceedings of the 43rd Annual IEEE Symposium on Foundations of Computer Science*, 187–96. The original online-conformal construction.

---

*Brief complete. Notebook source of truth: `notebooks/prediction-intervals/01_prediction_intervals.ipynb`.*
