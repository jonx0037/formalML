# Planning Brief — formalml.com Satellite Topics

**Five topics proposed as forward-pointers from formalStatistics Topic 30 (Kernel Density Estimation):**

1. **`kernel-regression`** — Nadaraya–Watson estimator (Nadaraya 1964; Watson 1964).
2. **`local-regression`** — Local polynomial modeling (Fan–Gijbels 1996).
3. **`clustering`** — Mean-shift clustering (Fukunaga–Hostetler 1975; Comaniciu–Meer 2002), focused scope.
4. **`density-ratio-estimation`** — Direct ratio estimators (Sugiyama–Suzuki–Kanamori 2012).
5. **`normalizing-flows`** — Invertible neural density estimators (Rezende–Mohamed 2015; Papamakarios et al. 2021).

**Depth.** Planning-brief depth, not production-brief. Each topic below covers the scope boundary, prerequisite chain across sites, section narrative arc (outlines, not rendered proofs), figure and component sketches, cross-site pointer design, and open design questions. Production briefs (the Topic-30-scale artifacts) are deferred to per-topic sessions after Jon has picked the first target and confirmed the formalml site scaffold.

**Document structure.** Part I is the ~1,200-word master plan (site context, common-thread narrative, prerequisite graph, sequencing recommendation, convention inheritance, open questions). Part II is the five per-topic planning briefs (~1,800-2,500 words each). The appendix collects all open questions in one place for a single reading pass.

**Assumption that I'm running with, and that Jon should flag if wrong.** formalml.com inherits the formalStatistics tech stack (Astro 6 + React 19 + MDX + KaTeX + Tailwind + Vercel) and conventions (separate `$$...$$` blocks — no `\begin{aligned}` or `\begin{array}`; `\frac` not `\tfrac` for matplotlib compatibility; `client:visible` directive; hand-rolled References blocks at bottom of MDX; Chicago 17th edition citations with verified URLs; `\perp\!\!\!\perp` for independence; collaborative "we" voice; geometric-first framing; `np.random.seed(42)` in notebooks; notebook filename pattern `NN_slug.ipynb`; shared-module "extend, don't create" rule once a module exists). Cross-site links use full origins (`https://formalstatistics.com/topics/...`).

---

## Part I — Master Plan

### I.1 formalml.com context

formalml is the third site in the triad (formalCalculus → formalStatistics → formalml). formalStatistics covers the classical statistics curriculum through empirical processes (Track 8); formalml extends into machine-learning-specific methodology — methods where the goal is prediction, representation, or generation, rather than inference about a parametric model.

**Open question (O1) for Jon:** *What's the current state of formalml.com?* Three scenarios materially affect this brief:

- **(a) Greenfield** — no topics published yet, no CLAUDE.md, no curriculum graph. In this case, the first deliverable is a site scaffold + track structure + a CLAUDE.md, and the five topics below slot into whichever tracks get created.
- **(b) Early-stage** — some topics published, but none of the five proposed here. The five get added to the existing track structure; no site-level decisions needed.
- **(c) Advanced-stage** — formalml has a mature curriculum; some of the five topics may overlap with existing content, others may have prerequisites that already exist. The briefs below need to be cross-referenced against the published formalml topic list before implementation.

Scenario (a) or (b) is what this brief assumes. For scenario (c), flag overlaps, and I'll adapt.

**Track structure recommendation** (for scenario (a) — greenfield). Four tracks, roughly the common ML curriculum structure:

1. **Foundations** — concentration, MLE, empirical risk minimization, information theory, linear algebra for ML. (Mostly new material, with formalStatistics prerequisites.)
2. **Supervised Learning** — linear models, kernel methods, trees/boosting, neural networks. (Includes **kernel-regression**, **local-regression**.)
3. **Unsupervised and Generative** — dimensionality reduction, clustering, mixture models, generative modeling. (Includes **clustering**, **normalizing-flows**, arguably **density-ratio-estimation**.)
4. **Learning Theory and Methodology** — generalization bounds, bias-variance in deep learning, transfer learning, covariate shift. (Includes **density-ratio-estimation** as a methodology tool.)

**Placement of the five topics within this structure:**

- **kernel-regression** → Supervised Learning (track opener or near-opener; direct extension of formalStatistics Topic 24 and Topic 30)
- **local-regression** → Supervised Learning (immediately after kernel-regression)
- **clustering** → Unsupervised and Generative (standalone introductory topic)
- **density-ratio-estimation** → Learning Theory and Methodology (or cross-listed in Unsupervised)
- **normalizing-flows** → Unsupervised and Generative (requires neural-network prerequisites that may or may not exist)

### I.2 The common-thread narrative

All five topics share a kernel-methods/nonparametric-density inheritance from Topic 30. This is not an accident — formalStatistics deliberately planted the five forward-pointers in Topic 30's §30.10 because they represent the natural extensions of KDE in five different directions:

| Direction | Topic | Core extension |
|-----------|-------|----|
| From unconditional to conditional density | **kernel-regression** | $\hat p(x) \to \hat p(y \mid x)$ via Nadaraya–Watson |
| From kernel-constant to kernel-polynomial | **local-regression** | Boundary-bias fix + regression-equation generalization |
| From density estimation to mode detection | **clustering** | Gradient ascent on $\hat f_h$ gives cluster centres |
| From one-sample to two-sample | **density-ratio-estimation** | $\hat p(x) / \hat q(x)$ without estimating the numerator and denominator separately |
| From "average of bumps" to learned transformations | **normalizing-flows** | Replace the kernel-average representation with an invertible neural net |

This thread motivates the recommended sequencing in §I.5 and structures the cross-references in each topic below.

### I.3 Cross-site architecture

Cross-site pointers appear in three directions:

- **Back to formalStatistics** — every topic has multiple formalStatistics prerequisites (Topic 30 universally; others vary by topic). These are full-URL hyperlinks: `[Topic 30 (Kernel Density Estimation)](https://formalstatistics.com/topics/kernel-density-estimation)`.
- **Back to formalCalculus** — Jacobians, gradients, convex optimization, variational calculus all come up in these topics. `[Jacobians](https://formalcalculus.com/topics/jacobians)`-style.
- **Forward within formalml** — the five topics cross-reference each other. Many appear together (mean-shift references NW kernel regression; DRE references kernel methods generally; flows reference DRE as a motivation for GANs). Internal formalml cross-references use the same `/topics/slug` pattern as formalStatistics.

**Convention for forthcoming topics.** When a topic references another not-yet-published formalml topic, use the `(forthcoming)` marker pattern from formalStatistics: link with an empty URL that renders as plain text, e.g., `[kernel methods topic (forthcoming)](empty)`. This is the Topic 30 §30.10 pattern; it makes forward-intention visible without breaking links.

**Citations.** Chicago 17th edition, verified URLs, manually rendered `### References` block at the bottom of each MDX (because Astro schema strips `url` and `isbn`; this is the formalStatistics lock). The citations spreadsheet convention carries over: `formalml-citations.xlsx` tracks author/year/title/venue/URL/type/topics for every reference across the site.

### I.4 Prerequisite graph across the five topics

Reading left-to-right is "depends on":

```
formalStatistics                  formalml                                    formalml (forthcoming)
─────────────────                 ────────                                    ───────────────────────
Topic 24 (linear reg)         ─→  kernel-regression          ─┬─→   local-regression
Topic 30 (KDE)                ─→  kernel-regression          ─┤    
Topic 13 (point est)          ─→  kernel-regression          ─┘
                                                               └─→   clustering (mean-shift)

Topic 30 (KDE)                ─→  density-ratio-estimation   ─→  (applied topics: covariate-shift, MMD, GAN)
Topic 17 (hypothesis testing) ─→  density-ratio-estimation
Topic 27 (Bayesian)           ─→  density-ratio-estimation

Topic 30 (KDE)                ─→  normalizing-flows
Topic 14 (MLE)                ─→  normalizing-flows
Topic 27 (Bayesian)           ─→  normalizing-flows
NEURAL-NETS-FOUNDATION (?)    ─→  normalizing-flows
```

The "NEURAL-NETS-FOUNDATION" row is a flag: normalizing flows is the only topic in this set that cannot stand alone without a prerequisite covering neural network fundamentals (universal approximation, backprop, invertible architectures). If such a topic doesn't exist on formalml, it needs to be written first — or the flows topic needs to include a heavier foundations section at the cost of narrative focus.

### I.5 Recommended sequencing

Suggested implementation order, with rationale:

1. **kernel-regression first.** Direct extension of Topic 30 and Topic 24. Smallest prerequisite footprint — nothing from formalml itself is required. Establishes the "how we link back to formalStatistics" pattern for every subsequent topic. ~3 weeks of authoring at the Topic-30 cadence.

2. **local-regression second.** Immediate follow-on to kernel-regression; the "fix the boundary problem" pedagogical arc carries directly from NW. Small topic (~1,800 words target); natural sequel. ~2 weeks.

3. **clustering third (or fourth).** Mean-shift is Nadaraya–Watson in disguise — the fixed-point iteration of the NW estimator is exactly the mean-shift update. Natural payoff from topics 1-2. Narrow scope (mean-shift focus; broader clustering deferred to an umbrella "unsupervised-methods" topic). ~2 weeks.

4. **density-ratio-estimation fourth (or third — parallel-izable with clustering).** Independent thread; doesn't depend on kernel-regression. Could be tackled in parallel with clustering by a second author; single-author sequencing would put this right after local-regression or after clustering, reader preference. ~3 weeks.

5. **normalizing-flows last.** The heaviest topic requires neural network prerequisites that may not yet exist in formalml. Best tackled after the first four are shipped and the neural-foundations topic is in place. ~4 weeks, including the neural-prerequisite checking.

**Total timeline estimate for all five:** 14-17 weeks at single-author pace, assuming formalml scaffold is in place and the formalStatistics-style three-phase deliverable workflow (notebook → brief → Claude Code implementation) is kept.

### I.6 Convention inheritance from formalStatistics

Recommended locks, carried verbatim:

- **KaTeX composition.** No `\begin{aligned}`; no `\begin{array}`. Multi-line derivations as separate `$$...$$` blocks with prose connectors. `\frac` not `\tfrac` (matplotlib compatibility in notebooks). `\perp\!\!\!\perp` for independence.
- **MDX technique.** `client:visible` on all components (never `client:load`). Section-anchor pattern `<a id="section-N-X"></a>` before `## N.X` headers for X ≥ 2. Hand-rolled `### References` at the bottom (Astro schema strips url/isbn).
- **Proof formatting.** End-of-proof marker: `$\blacksquare$ — using [citations]`. Never `QED` or `\Box` or `□`.
- **Narrative voice.** Collaborative "we"; no "simply" / "obviously" / "it can be shown"; geometric-first framing.
- **Citations.** Chicago 17th edition, verified URLs, `formalml-citations.xlsx` tracking.
- **Code style.** TypeScript for components; shared-module "extend, don't create" once a module exists; consistent type annotations; explicit return types on exported functions.
- **Notebook style.** Native `.ipynb` (never `.py` percent format); outputs cleared for final delivery; `np.random.seed(42)`; `dpi=150, bbox_inches='tight'` for all savefigs; file naming `NN_slug.ipynb` where `NN` is track/topic number.

**Two conventions that may need formalml-specific extension:**

- **Math preferences.** ML topics use more matrix/vector notation than classical statistics. The formalStatistics lock is `\boldsymbol\beta` for coefficient vectors; formalml may want `\mathbf{X}` / `\boldsymbol\theta` / `\boldsymbol\phi` — confirm in an early formalml topic.
- **Code examples.** formalStatistics uses Python + NumPy + SciPy for all notebook code. formalml topics may want to include PyTorch or JAX examples (flows and DRE especially). Decision: Python + NumPy/SciPy for theoretical illustrations, with a "see [topic on implementation] for production code" pointer for PyTorch/JAX-heavy aspects. Alternatively, introduce a `NeuralNotebook` pattern that includes PyTorch.

### I.7 Open questions for Jon

- **O1. Current state of formalml** (scenario a/b/c). Determines whether to scaffold first or jump to topics.
- **O2. Track structure.** Accept the four-track proposal (Foundations / Supervised / Unsupervised-Generative / Learning-Theory-Methodology), or a different structure?
- **O3. Neural-network prerequisites.** Does a topic covering neural-network fundamentals (universal approximation, backprop, basic architectures) exist, is planned, or needs to be slotted in before normalizing-flows?
- **O4. Code-example language policy.** Python + NumPy only, or admit PyTorch/JAX for deep-learning-adjacent topics?
- **O5. First-topic decision.** Greenlight kernel-regression as the first target, or a different starting point?

**One immediate next step regardless of these:** scaffold (or confirm the existence of) `formalml/CLAUDE.md` as the formalml equivalent of the formalStatistics instructions Jon has in his `Developer/Sites/formalStatistics/CLAUDE.md`. This is a one-session task and unblocks all five topics.

---
## Part II — Per-Topic Planning Briefs

### Topic A: `kernel-regression` (Nadaraya–Watson)

**Positioning.** The covariate-conditional extension of Topic 30's KDE. Where Topic 30 estimated the unconditional density $f(x)$, kernel regression estimates the conditional mean $m(x) = \mathbb{E}[Y \mid X = x]$ without parametric assumptions on $m$. The Nadaraya–Watson (NW) estimator is the locally weighted average of $Y$ values, with weights drawn from a kernel-smoothed neighborhood of $x$. It is the natural next topic after KDE and the direct prerequisite for local-regression, clustering (mean-shift), and much of the kernel-methods machinery that follows.

**Scope boundary.**

*In scope.* Nadaraya–Watson estimator definition; derivation as the minimizer of locally-weighted squared error with constant $m$ (the "local constant regression" interpretation); leading-order bias and variance with the design-density correction term ($m''/2 + m'f'/f$); MSE decomposition and bandwidth selection (analogous to but distinct from Topic 30's AMISE); boundary behavior; design-density-adjusted bias; curse of dimensionality and multivariate NW with tensor-product or bandwidth-matrix kernels; connections to k-nearest-neighbors regression; asymptotic normality (pointwise); bootstrap CIs for $m(x)$; ML bridges (locally-weighted regression in robotics; kernel smoothers as feature engineering).

*Out of scope.* Local-polynomial generalization (→ local-regression topic). Covariate-conditional density estimation $\hat p(y \mid x)$ beyond the mean (→ quantile regression / conditional density estimation, future). Nadaraya–Watson with kernels defined on non-Euclidean spaces (manifolds, strings, graphs) — pointer only. Gaussian-process regression (→ GP topic, future). Reproducing-kernel Hilbert-space treatment — one-paragraph name-check.

**Prerequisites.**

- formalStatistics Topic 30 — KDE, kernel properties, bandwidth selection, AMISE rate.
- formalStatistics Topic 24 — linear regression, least squares, residual analysis (as the parametric baseline).
- formalStatistics Topic 13 — MSE / bias-variance decomposition.
- formalStatistics Topic 19 — pointwise CI construction.
- formalCalculus Topic on Taylor series and partial derivatives.
- No formalml prerequisites.

**Section-by-section narrative (~9 sections).**

1. **Motivation: from density to conditional mean.** Open with the scatter-plot question: "Given $(X_i, Y_i)$ pairs, how do I estimate $\mathbb{E}[Y \mid X = x]$ without assuming linearity?" Narrate the move from Topic 30's "smooth a bag of $X_i$'s into a density" to "smooth a bag of $(X_i, Y_i)$'s into a conditional mean". Figure 1: scatter plot of noisy data from a non-linear $m$ with three smoothers overlaid (kernel too narrow, kernel too wide, well-tuned). 2–3 remarks including the KDE-as-special-case observation (NW with $Y_i \equiv 1$ recovers KDE).

2. **The Nadaraya–Watson estimator.** Formal definition: $\hat m_h(x) = \sum_i Y_i K_h(x - X_i) / \sum_i K_h(x - X_i)$. Geometric intuition: weighted average where kernel weights drop off smoothly with distance from $x$. Normalization properties (the denominator ensures the weights sum to 1 at every $x$). Example: NW on toy data with 3 choices of kernel (Gaussian, Epanechnikov, uniform).

3. **Local constant regression.** Derivation: NW is the minimizer of $\sum_i (Y_i - m)^2 K_h(x - X_i)$ over constant $m$. The first-order condition gives the NW formula directly. Remark on "local constant" framing as the gateway to local-polynomial generalization. Forward-pointer to local-regression.

4. **Bias and variance (featured section).** Leading-order analysis analogous to Topic 30 §30.4 but with the design-density correction:
   $$\mathrm{Bias}(\hat m_h(x)) = \tfrac{h^2}{2} \mu_2(K) \left(m''(x) + 2\,\tfrac{m'(x) f'(x)}{f(x)}\right) + o(h^2)$$
   $$\mathrm{Var}(\hat m_h(x)) = \tfrac{\sigma^2(x) R(K)}{n h f(x)} + o((nh)^{-1})$$
   Full proof as a short-ish derivation (~20 MDX lines). Key observation: bias depends on the design density $f$ through its log-derivative — NW does worse where data is sparse. This explicit $f'/f$ term motivates local regression.

5. **AMISE and bandwidth selection.** Analogous to Topic 30 §30.6 Thm 4 — derivation of the optimal $h^*$ for NW. Same $n^{-1/5}$ rate with different constants. Data-driven selection: cross-validation, plug-in methods. Figure: bandwidth-vs-MISE curve, oracle marker.

6. **Boundary behavior.** NW has the same "mass-leakage" issue as KDE at the boundary, plus an extra problem: the weighting inside the boundary is asymmetric, so the estimate at $x$ close to the boundary is biased toward the interior mean. Figure: NW on the boundary of a domain showing the pull-toward-interior. Motivates local regression (where local-linear fits automatically correct for this — see §B.5).

7. **Multivariate Nadaraya–Watson.** $\hat m_{\mathbf{H}}(\mathbf{x}) = \sum_i Y_i K_{\mathbf{H}}(\mathbf{x} - \mathbf{X}_i) / \sum_i K_{\mathbf{H}}(\mathbf{x} - \mathbf{X}_i)$ with tensor-product or full-matrix bandwidth. Curse of dimensionality: AMISE rate $O(n^{-4/(4+d)})$ — very slow for $d > 3$. Figure: rate-vs-dimension plot showing effective sample size for $d \in \{1, 2, 4, 10\}$.

8. **Connection to k-NN regression.** NW with uniform kernel and $h$ set by the $k$-th-nearest-neighbor distance = k-NN regression. Illustrates that NW is a unifying framework for nonparametric regressors. One example, one remark.

9. **ML bridges and forward-map.** Locally-weighted regression for robotics (LWR); kernel smoothers for feature engineering; connection to Gaussian-process regression (forward pointer). Forward-map to local-regression, clustering (mean-shift), density-ratio-estimation (kernel-density-ratio estimators).

**Figure sketches (~6-8 figures).**

- Fig 1: Scatter + NW smoothers at 3 bandwidths (motivation).
- Fig 2: Three kernel choices overlaid on the same sample (Gaussian vs Epanechnikov vs uniform).
- Fig 3: Bias-variance decomposition vs $h$ (analogous to Topic 30 Fig 3).
- Fig 4: Boundary-bias demonstration.
- Fig 5: Multivariate NW curse-of-dimensionality rate plot.
- Fig 6: NW vs k-NN comparison on the same data.
- Fig 7: AMISE-optimal bandwidth + data-driven selector comparison.
- Fig 8: Forward-map to local-regression / clustering / density-ratio.

**Interactive component sketches (~2-3 required).**

- **NWExplorer (featured).** Three-panel analog of Topic 30's BandwidthExplorer. Scatter + NW smoother + true $m$ overlay; bandwidth and kernel selectors; preset toggles (linear $m$ / sinusoidal $m$ / step function $m$); noise-level slider; residual panel.
- **BoundaryBiasInRegression.** Demonstrates NW's boundary issue; toggles between "no correction", "reflection", and (preview) "local linear". Forward-links to local-regression.
- **CurseOfDimensionalityDemo** (optional). 1-D / 2-D / 3-D NW on same underlying $m$; user watches the MISE explode with dimension.

**Cross-site pointers.**

- **Back to formalStatistics:** Topic 30 (kernel/KDE machinery); Topic 24 (linear regression baseline); Topic 13 (bias-variance); Topic 19 (CI construction); Topic 29 forward-pointer to quantile-regression (mentioned in this topic's §9 as a conditional-distribution cousin of NW).
- **Back to formalCalculus:** Taylor series; partial derivatives; multivariable integration.
- **Forward within formalml:** local-regression (§5, §6 pointers); clustering/mean-shift (§9 pointer); density-ratio-estimation (§9, the kernel-density-ratio family uses NW-like smoothers); future GP-regression topic; future kernel-methods / RKHS topic.

**Open design questions.**

- **K1. Multivariate scope.** Is multivariate NW (§7) a full section or a one-paragraph sidebar? A full section makes the topic ~2,500 words and requires a bandwidth-matrix-selection subsection; a sidebar makes it ~2,100 words and defers $d > 1$ entirely. **Recommendation:** one-section sidebar in this topic; a dedicated "high-dim nonparametrics" topic elsewhere in the Supervised Learning track covers the full picture.
- **K2. Asymptotic normality proof depth.** Topic 30 §30.7 Thm 6 stated-with-sketch Parzen's result. The NW analog is Nadaraya–Watson's own asymptotic normality — same Lindeberg-Feller machinery. **Recommendation:** follow Topic 30 exactly — stated with a one-paragraph sketch, not full proof.
- **K3. Bandwidth-selection section depth.** Topic 30 §30.8 had a dedicated section with four selectors. Kernel regression has analogous selectors (plus NW-specific methods like "leave-one-out CV of the regression residual"). **Recommendation:** one-section treatment with LOOCV as the featured selector (most useful for regression) and 2-3 alternatives name-checked.

**Estimated scope.** ~2,500 words, 8 figures, 3 interactive components, 7-8 new shared-module exports (NW evaluator; NW bandwidth selectors; multivariate-NW kernel; curvature-integral helper for the regression variant; boundary-reflection fallback). Expected authoring time at the Topic-30 cadence: three weeks (notebook + brief + Claude Code).

---
### Topic B: `local-regression` (Local polynomial modeling)

**Positioning.** The Fan–Gijbels (1996) generalization of kernel regression: instead of fitting a locally-weighted constant (NW), fit a locally-weighted polynomial of degree $p$. The two big wins: (i) *automatic boundary-bias correction* when $p$ is odd (no reflection or ad-hoc fix needed — the local linear fit at the boundary has the same $O(h^2)$ bias as the interior); (ii) *bias reduction to $O(h^{p+1})$ for odd $p$*. Local linear ($p = 1$) is the standard workhorse; local quadratic ($p = 2$) is used for derivative estimation. This topic is a tight sequel to kernel regression and a prerequisite for any "bandwidth selection with bias correction" treatment later in the track.

**Scope boundary.**

*In scope.* Local polynomial regression setup: for each evaluation point $x$, fit $\boldsymbol{\beta}(x) = \arg\min \sum_i (Y_i - \beta_0 - \beta_1(X_i - x) - \cdots - \beta_p(X_i - x)^p)^2 K_h(x - X_i)$; matrix form $\hat{\boldsymbol{\beta}}(x) = (\mathbf{X}_x^\top \mathbf{W}_x \mathbf{X}_x)^{-1} \mathbf{X}_x^\top \mathbf{W}_x \mathbf{Y}$; local-constant ($p=0$, recovers NW), local-linear ($p=1$, the workhorse), local-quadratic ($p=2$, for $m'$ estimation). Asymptotic bias and variance analysis (featured theorem: local-linear bias is $\tfrac{h^2}{2} \mu_2(K) m''(x) + o(h^2)$ — no $m'f'/f$ correction term, unlike NW). Boundary behavior (the Fan–Gijbels wins the same $O(h^2)$ bias at boundaries as in the interior for odd $p$). Equivalent-kernel formulation. Bandwidth selection with bias correction. Connection to smoothing splines (name-check). Variable-bandwidth / adaptive schemes (name-check).

*Out of scope.* Full smoothing-splines development (→ future topic). Full varying-coefficient model machinery (Hastie–Tibshirani 1993) — one-paragraph pointer. Robust local polynomial (local M-estimators) — one-paragraph name-check citing Fan–Gijbels §3. Non-parametric hypothesis testing via local polynomials (→ future topic). Semi-parametric models (partially linear, single-index) — pointer only.

**Prerequisites.**

- **kernel-regression** (this track, Topic A) — NW as the local-constant case is the direct predecessor. §1 opens with "NW is local constant; what happens if we go local linear?"
- formalStatistics Topic 24 — linear regression in design-matrix form; WLS; $(\mathbf{X}^\top \mathbf{W} \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{W} \mathbf{Y}$ formula.
- formalStatistics Topic 30 — kernels, bandwidth selection.
- formalStatistics Topic 13 — bias-variance decomposition.
- formalCalculus: linear algebra (matrix inverse, normal equations), Taylor series to higher orders.
- No additional formalml prerequisites beyond kernel-regression.

**Section-by-section narrative (~8 sections).**

1. **Motivation: the two problems NW leaves open.** Open with two side-by-side panels from the previous topic: (a) NW on a sample where the true $m$ has a sharp boundary-region behavior, and NW visibly pulls toward the interior mean; (b) NW on a sample with rapidly-varying $m'$ in a sparse region of $X$ where NW is badly biased because of the $m' f'/f$ term. Claim: local-linear regression fixes both.

2. **Local polynomial setup.** Definition and notation. Design matrix $\mathbf{X}_x$ (rows $(1, X_i - x, (X_i - x)^2, \ldots, (X_i - x)^p)$), kernel weight matrix $\mathbf{W}_x$ (diagonal with entries $K_h(x - X_i)$), estimate $\hat{\boldsymbol{\beta}}(x) = (\mathbf{X}_x^\top \mathbf{W}_x \mathbf{X}_x)^{-1} \mathbf{X}_x^\top \mathbf{W}_x \mathbf{Y}$. The estimate of $m(x)$ is $\hat m(x) = \mathbf{e}_1^\top \hat{\boldsymbol{\beta}}(x) = \hat\beta_0(x)$. Example: local-linear fit on toy data, visualized as the intercept of each local-regression line.

3. **Local-constant = NW (unification).** Plug $p = 0$ and recover $\hat m_h(x) = \sum_i Y_i K_h / \sum_i K_h$ exactly. This binds local regression to the previous topic and sets up the "what changes at $p = 1$" question.

4. **Bias and variance at degree $p$ (featured theorem).** For odd $p$, the local polynomial estimator has bias $\tfrac{h^{p+1}}{(p+1)!} \mu_{p+1}(K) m^{(p+1)}(x) + o(h^{p+1})$ and variance $\tfrac{\sigma^2(x) R^*(K, p)}{n h f(x)} + o((nh)^{-1})$, where $R^*(K, p)$ is the equivalent-kernel roughness (computed from the matrix-inverse formula). **Key observation:** the bias does *not* have the $m' f'/f$ term for odd $p$ — the local fit absorbs the design-density asymmetry. This is the Fan–Gijbels victory over NW. Full proof at the local-linear case only; higher-$p$ stated without derivation.

5. **Boundary behavior: the Fan–Gijbels win.** Local-linear regression has *the same* $O(h^2)$ bias at a boundary as it does in the interior — a result that NW does not enjoy. Proof: the matrix-inverse $(\mathbf{X}_x^\top \mathbf{W}_x \mathbf{X}_x)^{-1}$ self-corrects the asymmetry in the design distribution near the boundary. Figure: side-by-side panels showing NW vs local-linear on the same boundary problem from Topic A §6.

6. **Equivalent kernel formulation.** The local-polynomial estimator can be rewritten as $\hat m(x) = \sum_i w_{n,i}^*(x) Y_i$ where the *equivalent kernel* $w_{n,i}^*$ is derived from the matrix-inverse formula. Figure: equivalent kernels for $p = 0, 1, 2, 3$ at an interior point and at a boundary point. Illustrates how the Fan–Gijbels machinery "reshapes" the kernel near the boundary.

7. **Bandwidth selection (revisited).** CV methods from kernel regression carry over. Plug-in methods now use the Fan–Gijbels bias formula. Connection to Ruppert–Sheather–Wand (1995) plug-in. One or two figures.

8. **Extensions and forward-map.** Varying-coefficient models (pointer); local generalized linear models (pointer); local maximum-likelihood; derivative estimation (local-quadratic for $\hat m'$, local-cubic for $\hat m''$). Forward-pointers to: future formalml topics on additive models, partially linear models, smoothing splines.

**Figure sketches (~5-6 figures).**

- Fig 1: NW's two failure modes (boundary bias + design-density bias), setting up the motivation.
- Fig 2: Side-by-side NW vs local-linear fit on a boundary problem (the headline result).
- Fig 3: Bias-variance vs bandwidth for local-linear (analog of Topic A Fig 3 but with the $f'/f$ term gone).
- Fig 4: Equivalent kernels at different $p$ and at interior vs boundary points.
- Fig. 5: Local-polynomial derivative estimation: $\hat{m}$, $\hat{m}'$, and $\hat{m}''$ on sinusoidal data.
- Fig 6: Forward-map figure (local-regression as the bridge to additive models, varying-coefficient, etc.).

**Interactive component sketches (~2 required).**

- **LocalPolyExplorer (featured).** Scatter + regression smoother at user-controlled polynomial degree $p \in \{0, 1, 2, 3\}$. Same preset toggles as NWExplorer but with the extra $p$ control. Bandwidth slider. Overlay: true $m$, NW fit (for comparison), local-polynomial fit. Boundary-zone highlight to make the Fan–Gijbels advantage visible.
- **EquivalentKernelViewer.** Shows the equivalent kernel weights at an interior point vs a boundary point for $p = 0, 1, 2$. Illustrates how the matrix-inverse machinery "reweights" the data near boundaries.

**Cross-site pointers.**

- **Back to formalStatistics:** Topic 30 (kernels); Topic 24 (WLS design-matrix form).
- **Back to formalCalculus:** Taylor series to order $p$; linear algebra (matrix inversion, normal equations).
- **Within formalml:** kernel-regression (immediate predecessor); future additive-models / varying-coefficient / smoothing-splines topics; future semi-parametric regression topic.

**Open design questions.**

- **L1. Proof of completeness.** The local-linear bias/variance proof is ~15-20 MDX lines (similar complexity to Topic 30 Thm 4). Local-$p$ general case is another ~15 lines. **Recommendation:** full proof of local-linear; local-$p$ general stated-with-sketch pointing to Fan–Gijbels §3.
- **L2. Smoothing-splines treatment.** Smoothing splines share the "local smooth + penalty" framing. Full treatment is a separate topic; this topic needs a §8 paragraph naming them and the forward-pointer. **Recommendation:** one paragraph, no figure.
- **L3. Derivative estimation depth.** Local-quadratic for $\hat m'$ is a natural extension but expands the topic substantially. **Recommendation:** one-section treatment (a ~500-word §7b or part of §8), single figure, no dedicated component. This is enough for the reader to know it's available and to find the Fan–Gijbels reference; full development stays out of scope.

**Estimated scope.** ~2,000 words (tighter than Topic A because of the tight scope and NW-already-built foundation), 6 figures, 2 interactive components, 3-4 new shared-module exports (local-polynomial evaluator via design-matrix form; equivalent-kernel computation helper; local-derivative estimator). Expected authoring time: two weeks.

---
### Topic C: `clustering` (Mean-shift clustering)

**Positioning.** The KDE-gradient-ascent algorithm that discovers clusters as basins of attraction of the density modes. Mean-shift is the direct "apply calculus to a KDE" payoff — take the gradient of $\hat f_h$, iterate, and each starting point converges to a local mode; cluster = set of points converging to the same mode. The fixed-point iteration turns out to be exactly the Nadaraya–Watson formula with the data itself as both $X$ and $Y$, tying this topic closely to Topic A. Scope is deliberately narrow: mean-shift as the focus, with k-means/DBSCAN/Gaussian-mixture clustering introduced only for comparison. An umbrella "unsupervised-methods" topic (not in this batch) handles the broader clustering taxonomy.

**Scope boundary.**

*In scope.* Mean-shift as gradient ascent on $\hat f_h$. The mean-shift update formula and its derivation from $\nabla \hat f_h$. Fixed-point interpretation as the Nadaraya–Watson equation with $Y_i \equiv X_i$. Convergence properties (monotone density increase, convergence to local modes under mild conditions; Comaniciu–Meer 2002). Clustering as a basin-of-attraction assignment. Bandwidth sensitivity and selection in the clustering context (fundamentally different from density estimation — we care about mode stability, not density accuracy). Comparison to k-means (parametric, non-adaptive cluster count) and DBSCAN (density-based, non-gradient). Image segmentation application (the original computer vision use case). Extensions: adaptive-bandwidth mean-shift (Comaniciu–Ramesh–Meer 2001), tracking (CamShift / meanshift-tracking).

*Out of scope.* k-means algorithm details and convergence (→ umbrella unsupervised topic or a dedicated k-means topic). Hierarchical clustering, agglomerative methods — pointer only. Spectral clustering — pointer only. Gaussian mixture models, EM algorithm — pointer only; forward-link to a GMM topic. DBSCAN full treatment — one comparative example. High-dimensional mean-shift performance issues — brief mention. Theoretical clustering (clustering with guarantees, k-means++ analyses, optimal transport formulations) — out of scope for this topic.

**Prerequisites.**

- formalStatistics Topic 30 — KDE as the underlying density. Mean-shift modes are modes of $\hat {f_h} $.
- formalStatistics Topic 13 — MSE / point estimation (for interpreting cluster centers as mode estimates).
- **kernel-regression** (Topic A) — the NW structure is reused directly; mean-shift update = NW with $Y_i = X_i$. This is the central pedagogical connection.
- formalCalculus: gradient, partial derivatives, fixed-point iteration.
- No additional formalml prerequisites.

**Section-by-section narrative (~8 sections).**

1. **Motivation: modes of a KDE.** Plot a 1-D KDE of a bimodal sample from Topic 30. Visible modes — but how do we find them algorithmically? One answer: gradient ascent on $\hat f_h$, starting from every data point. Each data point traces a path to a local mode; group points by the mode they reach. Figure 1: KDE with two starting points, trajectories traced, convergence to two different modes.

2. **The mean-shift update.** Derivation from $\nabla \hat f_h(x) = 0$. With a Gaussian kernel: $\nabla \hat f_h(x) \propto \sum_i (X_i - x) K_h(x - X_i)$, so the zero-gradient condition is $x = \sum_i X_i K_h(x - X_i) / \sum_i K_h(x - X_i)$. This is the NW formula with $Y_i = X_i$ — the mean-shift fixed-point equation. Iteration: $x_{t+1} = \sum_i X_i K_h(x_t - X_i) / \sum_i K_h(x_t - X_i)$. Example: one iteration worked through on a small toy sample.

3. **Fixed-point interpretation (featured connection).** The mean-shift iteration is the NW operator applied to $x$: $x_{t+1} = \mathrm{NW}(x_t)$. Consequence: each iteration moves $x$ toward the weighted local mean, which is structurally the "pull-toward-density-peak" operation. Sitting at a fixed point means $x$ equals its own NW-weighted average — $x$ is at a local mode. Two short remarks: (i) the non-Gaussian-kernel case requires care (shadow kernel argument, Comaniciu–Meer 2002); (ii) the gradient-ascent interpretation connects to the broader EM-algorithm family.

4. **Convergence.** Under the Gaussian kernel, the mean-shift iteration is monotone in density: $\hat f_h(x_{t+1}) \geq \hat f_h(x_t)$. Combined with boundedness and continuity, this gives convergence to a local mode. Full proof of the Gaussian-kernel monotonicity (~10 MDX lines). Comaniciu–Meer's general-kernel result is stated without proof.

5. **Basin-of-attraction clustering.** Run mean-shift from every $X_i$ in the dataset; each $X_i$ converges to a mode $\mu_k$; define cluster $C_k = \{i : X_i \to \mu_k\}$. The cluster count is *not* specified in advance — it emerges from the bandwidth choice and the density structure. Figure: 2-D point cloud with mean-shift trajectories color-coded by final cluster.

6. **Bandwidth selection for clustering.** Fundamentally different from density-estimation bandwidth selection. We want the bandwidth that reveals the correct number of modes. Too small: every data point is its own mode (overclustering). Too large: one big mode (underclustering). Practical strategies: cross-validated cluster stability, Silverman-type rules as a starting point with sensitivity analysis, adaptive-bandwidth schemes (Comaniciu–Ramesh–Meer 2001). Figure: same data at 3 bandwidths showing under-/well- / over-clustering.

7. **Comparison to k-means and DBSCAN.** All three applied to the same 2-D dataset. k-means: fixed cluster count, spherical Voronoi regions. DBSCAN: density-threshold based, can produce non-convex clusters, but needs a density threshold. Mean-shift: automatic cluster-count detection, kernel-smooth density basis. Trade-off table: which method wins for which data structure? One figure.

8. **Applications and extensions.** Image segmentation (Comaniciu–Meer 2002's original application — mean-shift in color+spatial space). Object tracking (CamShift, mean-shift tracker). Adaptive-bandwidth extensions. Forward-pointer to neural clustering (SOM, deep-clustering methods — defer).

**Figure sketches (~5-6 figures).**

- Fig 1: KDE + mean-shift trajectory illustration.
- Fig 2: Mean-shift update visualized: one iteration as an NW-weighted average.
- Fig 3: 2-D point cloud with trajectories color-coded by basin of attraction.
- Fig 4: Bandwidth sensitivity: 3 bandwidths on the same data, showing under/well/over-clustering.
- Fig 5: Comparison plot — k-means vs DBSCAN vs mean-shift on three test datasets (blobs, moons, noise).
- Fig 6: Image segmentation example using mean-shift in color+spatial feature space.

**Interactive component sketches (~2 required).**

- **MeanShiftStepper (featured).** Scatter plot; user clicks "step" to advance mean-shift iteration by one; user sees each trajectory evolve point-by-point. Bandwidth slider; kernel selector; reset button; "run to convergence" shortcut. The side panel shows the per-point iteration count and the current position. Two preset datasets: blobs (easy), moons (medium).
- **ClusteringMethodComparator.** Three algorithms (k-means, DBSCAN, mean-shift) on the same user-selectable 2-D preset dataset (blobs, moons, concentric circles, varying-density). User adjusts algorithm parameters (k, ε/minPts, bandwidth) and compares side-by-side. Annotations highlight when each algorithm succeeds/fails.

**Cross-site pointers.**

- **Back to formalStatistics:** Topic 30 (KDE); Topic 13 (point estimation framework for cluster centroids).
- **Back to formalCalculus:** gradient; fixed-point iteration; convergence criteria.
- **Within formalml:** kernel-regression (direct structural connection — mean-shift IS NW); future umbrella unsupervised-methods topic; future GMM + EM topic; future deep-clustering / representation-learning topics.

**Open design questions.**

- **C1. Scope narrowness.** Mean-shift-only or mean-shift-plus-overview-of-clustering-taxonomy? **Recommendation:** narrow scope as written. Mean-shift has a clean pedagogical arc from KDE that k-means and others don't share; mixing them dilutes the "KDE gradient" story. The comparison section (§7) gives enough breadth.
- **C2. Image-segmentation example depth.** §8 could be a paragraph (current recommendation) or a full section with a worked image and a figure. **Recommendation:** full section with one figure (the classic Comaniciu–Meer cell-image). Image segmentation is the canonical application of mean shift and makes the topic memorable; an extra 200 words are worth it.
- **C3. Adaptive-bandwidth treatment.** Adaptive-bandwidth mean-shift (Comaniciu–Ramesh–Meer 2001) is a practically important extension, but adds complexity. **Recommendation:** one-paragraph name-check in §8, no separate section. Full treatment deferred.
- **C4. Shadow-kernel argument.** The non-Gaussian-kernel case requires the "shadow kernel" derivation (Comaniciu–Meer 2002) to prove convergence. **Recommendation:** one-remark pointer in §4, no proof. The reader gets the Gaussian-kernel proof, which is sufficient.

**Estimated scope.** ~1,800 words (narrow-scope topic), 6 figures, 2 interactive components, 4-5 new shared-module exports (mean-shift iterator; convergence detector; basin-of-attraction labeler; k-means baseline for comparison; DBSCAN baseline for comparison). Expected authoring time: two weeks.

---
### Topic D: `density-ratio-estimation` (DRE)

**Positioning.** Given two samples drawn from $p$ and $q$, estimate $r(x) = p(x)/q(x)$ *without* separately estimating $p$ and $q$. The key insight: directly estimating the ratio avoids two-stage error accumulation and is often statistically and computationally superior. Density-ratio estimation is an independently motivated branch of non-parametric inference that sits atop KDE as a comparator and connects to several high-impact ML methodologies: covariate-shift correction in transfer learning, two-sample testing (MMD), mutual information estimation, change-point detection, outlier detection, and GAN training (which is implicit DRE). This topic serves as a methodological bridge from nonparametric density estimation to modern ML applications.

**Scope boundary.**

*In scope.* Motivation: why ratios matter (covariate shift, divergences, hypothesis testing). Naive plug-in: $\hat r(x) = \hat p(x) / \hat q(x)$ via two KDEs, and its failure modes (divide-by-zero; boundary amplification; error accumulation). Direct estimation framework: model $r_\theta$ in some function class; fit $\theta$ by minimizing a divergence functional. KLIEP (Sugiyama–Nakajima–Kashima–Bünau–Kawanabe 2008): maximize $\mathbb{E}_p[\log r_\theta]$ subject to $\mathbb{E}_q[r_\theta] = 1$. LSIF / uLSIF (Kanamori–Hido–Sugiyama 2009): squared-error objective, closed-form solution for the unconstrained variant. Relative density-ratio: $r_\alpha = p / (\alpha p + (1-\alpha) q)$, the uLSIF-with-relative-denominator bias/variance trade-off. Model selection via cross-validation on the divergence objective. Applications: covariate-shift correction in supervised learning (weighted ERM with $\hat r$ weights); two-sample testing (MMD statistic is a special case of a kernel ratio estimator); mutual information $I(X;Y) = \mathbb{E}[\log (p_{XY}/p_X p_Y)]$ as a DRE problem; f-GAN / GAN training as implicit DRE.

*Out of scope.* Full generative-adversarial theory (→ future GAN topic). Wasserstein / optimal-transport divergences — pointer only. Full PAC-Bayes / generalization bound development — pointer only. Concentration inequalities specific to DRE (Rademacher complexity bounds on LSIF) — one-paragraph sketch, full treatment in a future statistical-learning topic. Non-kernel DRE methods (neural-network-based DRE, logistic regression DRE) — named, with one-example demonstration, but full development deferred to a post-neural-networks-topic DRE chapter.

**Prerequisites.**

- formalStatistics Topic 30 — KDE as the naive plug-in baseline; also provides kernel machinery.
- formalStatistics Topic 17 — hypothesis testing framework (for the two-sample testing application).
- formalStatistics Topic 13 — MSE / bias-variance for the ratio estimator.
- formalStatistics Topic 14 — MLE (KLIEP is a constrained-MLE-style estimator).
- formalStatistics Topic 27 — Bayesian methods (for the variational-DRE view).
- formalCalculus: gradient, Lagrangian optimization, convex optimization basics.
- Within formalml: no hard prerequisites (can be read in parallel with kernel-regression / local-regression).

**Section-by-section narrative (~9 sections).**

1. **Motivation: why ratios are fundamental.** Open with three motivating problems: (i) covariate shift — training on $p_{\mathrm{train}}$ but deploying on $p_{\mathrm{test}} \neq p_{\mathrm{train}}$ requires weighted ERM with weights $r(x) = p_{\mathrm{test}}(x) / p_{\mathrm{train}}(x)$; (ii) two-sample testing — "do samples $\{X_i\}$ and $\{Y_j\}$ come from the same distribution?" is a question about the ratio $r \equiv 1$ or not; (iii) mutual information — $I(X;Y)$ is an expectation of $\log(p_{XY}/p_X p_Y)$. All three are ratio problems. Figure 1: The three motivating problems schematically.

2. **Naive plug-in and its problems.** Estimate $\hat r(x) = \hat p_h(x) / \hat q_{h'}(x)$ using two separate KDEs. Problems: (a) zero-denominator when $\hat q$ is zero in a region where $\hat p$ has mass; (b) boundary amplification — small errors in each density multiply; (c) cumulative MISE rate $O(n^{-4/(4+d)})^2 = O(n^{-8/(4+d)})$ is slower than direct estimation can achieve. Figure 2: side-by-side illustration of plug-in failure on a preset with disjoint support.

3. **The direct-estimation framework.** General setup: model $r$ in a parametric family $r_\theta$; choose a divergence functional $D(\hat r, r)$; fit $\theta$ by minimizing an empirical approximation to $D$ with the constraint that $\hat r$ integrates correctly. Parametric family choices: linear combinations of kernel functions, $r_\theta(x) = \sum_j \theta_j K(x - c_j)$ — this sets up both KLIEP and LSIF.

4. **KLIEP.** Objective: $\max_\theta \mathbb{E}_p[\log r_\theta]$ s.t. $\mathbb{E}_q[r_\theta] = 1, r_\theta \geq 0$. The constraint ensures $r_\theta$ is a valid ratio. Convex problem; solve via projected gradient or interior-point. Intuition: log-likelihood under $p$ of $r_\theta$ as a density ratio. One short worked example on 1-D toy data.

5. **LSIF / uLSIF (featured).** Objective: $\min_\theta \mathbb{E}_q[(r_\theta - r)^2]$ = $\min_\theta [\tfrac{1}{2}\mathbb{E}_q[r_\theta^2] - \mathbb{E}_p[r_\theta]] + \text{const}$ (using the ratio-identity trick). The unconstrained variant (uLSIF) has a closed-form solution: $\hat\theta = \hat{\mathbf{H}}^{-1} \hat{\mathbf{h}}$ where $\hat{\mathbf{H}} = n_q^{-1} \sum_j K(X_j^q) K(X_j^q)^\top$ and $\hat{\mathbf{h}} = n_p^{-1} \sum_i K(X_i^p)$. **Full derivation as a featured theorem** (~15 MDX lines). Contrast with KLIEP: uLSIF is computationally much simpler (closed form!) at a slight statistical cost (can produce negative ratio estimates).

6. **Relative density-ratio.** Replace $r(x) = p(x)/q(x)$ with $r_\alpha(x) = p(x) / (\alpha p(x) + (1-\alpha) q(x))$. Advantage: bounded in $[0, 1/\alpha]$; well-defined even when $q$ has zero-support regions where $p$ has mass. Reduction to standard DRE via the transformation. One remark: one figure showing stability of $r_\alpha$ vs $r$ on a near-singular preset.

7. **Application 1: Covariate-shift correction.** Training loss becomes $\sum_i r(X_i^{\mathrm{train}}) \ell(Y_i^{\mathrm{train}}, f(X_i^{\mathrm{train}}))$ where $r = p_{\mathrm{test}} / p_{\mathrm{train}}$. Demonstrate on a regression preset where $p_{\mathrm{train}}$ and $p_{\mathrm{test}}$ have different support. Figure: predictions with and without reweighting.

8. **Application 2: Two-sample testing, MMD, and mutual information.** Brief tour: MMD (maximum mean discrepancy) as a kernel ratio; permutation tests on MMD; f-divergence estimation as DRE; mutual information $\hat I(X;Y)$ via $\hat r = p_{XY}/p_Xp_Y$ estimation. One paragraph per application.

9. **Application 3: GANs as implicit DRE + forward-map.** The discriminator in a GAN learns an approximate density ratio $\hat r = p_{\mathrm{data}} / p_{\mathrm{generator}}$. Training the generator to minimize this ratio (or its logarithm) is implicit DRE. This connects to **normalizing flows** (explicit ratio/density) and to future GAN topics. Forward-pointer to a dedicated GAN topic.

**Figure sketches (~6-7 figures).**

- Fig 1: Motivation schematic — 3 panels (covariate shift, two-sample, mutual info).
- Fig 2: Plug-in DRE failure modes (divide-by-zero, error amplification).
- Fig 3: KLIEP vs uLSIF on 1-D toy — estimates vs true ratio.
- Fig 4: Relative density ratio $r_\alpha$ vs standard $r$ stability comparison.
- Fig 5: Covariate-shift regression: predictions with/without ratio reweighting.
- Fig 6: MMD two-sample testing: statistic distribution under null vs alternative.
- Fig 7: GAN discriminator as DRE (schematic).

**Interactive component sketches (~2 required).**

- **RatioEstimatorComparator (featured).** Two-sample 1-D preset with adjustable distributions ($p$ and $q$). Three ratio estimators overlaid: plug-in KDE ratio, KLIEP, and uLSIF. True ratio in black. User adjusts the sample sizes, the relative density-ratio parameter $\alpha$, and the kernel bandwidth. Side panel: MSE of each estimator.
- **CovariateShiftReweighter.** Supervised regression preset with train/test distribution mismatch. User toggles "no reweighting" vs "uLSIF reweighting" and watches test-MSE change. Visualizes the learned ratio over the input space.

**Cross-site pointers.**

- **Back to formalStatistics:** Topic 30 (KDE plug-in baseline and kernel machinery); Topic 17 (hypothesis testing framework — MMD application); Topic 14 (MLE — KLIEP connection); Topic 13 (MSE).
- **Back to formalCalculus:** Lagrangian optimization; convex optimization basics; matrix calculus for the uLSIF closed-form derivation.
- **Within formalml:** kernel-regression (kernel-method baseline); normalizing-flows (explicit-ratio relationship; flows as an alternative to DRE for density estimation); future GAN topic; future distribution-shift / transfer-learning topic.

**Open design questions.**

- **D1. Neural DRE.** Neural-network-based DRE (training a classifier to distinguish $p$ and $q$ samples, then taking the logit difference — logistic-regression DRE) is methodologically important. It's also outside the kernel-methods family and requires prerequisites in neural networks. **Recommendation:** one-paragraph section at the end of §9 naming neural DRE with a forward-pointer; full treatment deferred.
- **D2. MMD treatment depth.** MMD (Gretton et al. 2012) is close to DRE in spirit — it uses a kernel to measure distance between distributions, with a closed-form estimator. One could argue MMD deserves its own topic. **Recommendation:** §8 treats MMD as a DRE application (~200 words), not a separate topic. The full MMD /kernel-two-sample treatment could come later as a dedicated topic.
- **D3. GAN application depth.** §9 could be a paragraph or a full section with a figure. Given GAN is a major ML topic and this is only a forward-pointer, **recommendation:** a tight ~300-word paragraph with one schematic figure, not a full worked example.
- **D4. Mutual-information estimation depth.** A worked example on MI estimation would strengthen the topic. **Recommendation:** one worked example in §8 on a bivariate Gaussian preset (closed-form $I$ is known, so error can be measured); no separate section.

**Estimated scope.** ~2,000 words, 7 figures, 2 interactive components, 5-6 new shared-module exports (plug-in ratio estimator, KLIEP solver, uLSIF closed-form solver, relative-density-ratio variant, MMD statistic, MI estimator). Expected authoring time: three weeks.

---
### Topic E: `normalizing-flows`

**Positioning.** The neural-network-based approach to density estimation: parametrize an invertible transformation $f_\theta: \mathbb{R}^d \to \mathbb{R}^d$ that maps a simple base density (Gaussian) to the data density. The change-of-variables formula gives the learned density $p_\theta(x) = p_Z(f_\theta^{-1}(x)) \cdot |\det J_{f_\theta^{-1}}(x)|$ in closed form, enabling maximum-likelihood training. Normalizing flows are the "modern" (post-2014) approach that sidesteps the curse-of-dimensionality bottleneck of kernel methods (Topic 30 AMISE at $n^{-4/(4+d)}$) by imposing a parametric structure. The topic sits at the intersection of variational methods, deep learning, and density estimation. It is the heaviest topic in this batch and comes last in the recommended sequencing because it requires neural-network prerequisites that may not yet exist on formalml (see open question NF1).

**Scope boundary.**

*In scope.* Motivation: the curse-of-dimensionality escape via parametric structure. The change-of-variables formula and its role in likelihood computation. Flow architectures: coupling flows (NICE 2014, Real NVP 2017) — the simplest explicit family with tractable Jacobian; autoregressive flows (MADE, MAF 2017, IAF 2016) — density vs sampling asymmetry; residual flows and invertible residual networks; continuous-time flows (Neural ODEs 2018, FFJORD 2019) — the differential-equation formulation. Training: maximum-likelihood via reverse-KL between $p_{\mathrm{data}}$ (empirical) and $p_\theta$, which reduces to the log-det-Jacobian sum plus base-density log-likelihood at the preimages. Practical considerations: initialization, stability, and permutations between coupling layers. Universal-approximation properties. Applications: generative modeling, density estimation for high-dimensional structured data, variational inference (IAF for variational posteriors), density-ratio estimation (explicit ratio from explicit densities — connection to Topic D). Comparison to KDE: parametric structure vs nonparametric; finite-sample bias-variance vs universal approximation asymptotically. Recent developments (score-based models, diffusion models) — pointer only.

*Out of scope.* Full neural-network training machinery (backprop, Adam, weight initialization) — prerequisite, not developed here. Full VAE theory — pointer. Full GAN theory — pointer. Score-based diffusion models in full — one-section pointer with a forward-map figure. Normalizing flows on manifolds (Riemannian flows) — pointer only. Continuous-time flows beyond the FFJORD name-check — pointer. PyTorch / JAX implementation details — brief code snippets illustrating the core idea, but no full training-loop implementations in the main narrative (save those for a companion "normalizing-flows in practice" topic).

**Prerequisites.**

- formalStatistics Topic 30 — KDE as the parametric vs. nonparametric comparator.
- formalStatistics Topic 14 — MLE (flows are trained by MLE).
- formalStatistics Topic 27 — Bayesian/variational inference (for IAF-as-posterior use case).
- formalStatistics Topic 9 — modes of convergence (for universal-approximation discussions).
- formalCalculus: Jacobians, change of variables, determinant properties, chain rule, ordinary differential equations (for Neural-ODE section).
- **Within formalml: neural-network fundamentals topic** — this is the critical missing prerequisite (see NF1). Need: universal approximation theorem statement; backprop one-paragraph sketch; architecture basics (layers, activations, loss functions). If not yet on formalml, a dedicated prerequisite topic needs to be written first, OR this topic includes a §2 expansion covering the essentials at the cost of narrative focus.

**Section-by-section narrative (~10 sections).**

1. **Motivation: beyond the curse.** Topic 30's MISE rate is $O(n^{-4/(4+d)})$ — useless for $d \gg 3$. Parametric estimators achieve $O(n^{-1})$ at the cost of assuming the parametric form is correct. Normalizing flows offer a middle path: *learned* parametric structure with enough capacity to approximate any smooth density (universality), enabling tractable maximum-likelihood training. Figure 1: the sample-complexity landscape — KDE at $n^{-4/(4+d)}$ vs parametric Gaussian at $n^{-1}$ vs flows (empirical curve).

2. **Change of variables: the workhorse formula.** If $Z = f^{-1}(X)$ with $f$ a diffeomorphism $\mathbb{R}^d \to \mathbb{R}^d$, then $p_X(x) = p_Z(f^{-1}(x)) \cdot |\det J_{f^{-1}}(x)|$. Full derivation from first principles (the Jacobian-of-the-inverse-is-inverse-of-the-Jacobian identity). Three requirements for a practical flow family: (i) $f$ is expressive enough to transform simple densities to complex ones; (ii) $f$ is invertible; (iii) $\det J_f$ is tractable to compute. Figure 2: A simple bivariate flow schematic showing a Gaussian base being transformed into a crescent density.

3. **Coupling flows: NICE and Real NVP.** The coupling-layer trick: split $x = (x_1, x_2)$, transform $x_2 \leftarrow s(x_1) \cdot x_2 + t(x_1)$, leave $x_1$ unchanged. Jacobian is triangular, determinant is the product of $s(x_1)$ entries — tractable. NICE (Dinh et al. 2014) uses $s \equiv 1$ (volume-preserving); Real NVP (Dinh et al. 2017) uses learnable $s$ (not volume-preserving, more expressive). Stack layers with alternating splits. Figure 3: one coupling-layer architecture diagram.

4. **Autoregressive flows: MAF and IAF.** $x_i = \mu_i(x_{1:i-1}) + \sigma_i(x_{1:i-1}) \cdot z_i$. Jacobian is triangular by construction. MAF (Papamakarios et al. 2017) has fast density computation but slow sampling; IAF (Kingma et al. 2016) is the reverse. Asymmetry between density computation and sampling speed is an architectural trade-off. Figure 4: density-vs-sampling speed trade-off table across architectures.

5. **Training via maximum likelihood.** Negative log-likelihood $-\mathbb{E}_{p_{\mathrm{data}}}[\log p_\theta(x)]$ evaluated on a training batch, backprop through the change-of-variables formula, gradient step. Practical stability: layer normalization between flow layers; initialization so initial flow is close to identity; permutations between coupling layers to mix dimensions. Featured subsection: the connection to KL-divergence minimization between $p_{\mathrm{data}}$ and $p_\theta$.

6. **Universality.** Statement-level claim: coupling flows with enough capacity and depth can approximate any smooth invertible map on $\mathbb{R}^d$ (Huang et al. 2018; Koehler et al. 2021). One-paragraph sketch; full proofs deferred. Caveat: the required depth can be very high; in practice, architectures with bounded depth may not converge to the target density at small sample sizes.

7. **Continuous-time flows: Neural ODEs and FFJORD.** Generalize discrete flow layers to a continuous-time ODE $\dot{x}(t) = f_\theta(x(t), t)$; the final state $x(T) = f_\theta \circ \cdots \circ f_\theta(x_0)$ is the "flow-transformed" data. Change-of-variables becomes integration over time: $\log p_X(x(T)) = \log p_Z(x(0)) - \int_0^T \mathrm{tr}(J_{f_\theta}(x(t), t)) dt$. Trace vs determinant — FFJORD's central trick. Figure 5: Neural-ODE trajectory visualization in 2-D.

8. **Comparison to KDE.** Head-to-head on a controlled task: toy 2-D density (e.g., two moons). KDE with Silverman bandwidth vs. a Real NVP flow with matched training budget. Both estimates overlaid; likelihood on held-out data computed. Figure 6: likelihood-vs-sample-size for KDE and Real NVP on the two-moons preset. Pedagogical payoff: parametric structure wins for $d > 3$ at moderate $n$.

9. **Applications.** Generative modeling (sample from $p_\theta$ by drawing $z$ from Gaussian and applying $f$). Variational inference (IAF as posterior family). Density-ratio estimation via explicit densities (connection to Topic D). Anomaly detection via low $p_\theta(x)$ score. One paragraph per application.

10. **Forward-map: score-based and diffusion models.** Score-based models (Song–Ermon 2019) and diffusion models (Ho et al. 2020) generalize the flow idea by making the transformation *stochastic*. No longer invertible, but density-tractable via reverse-time SDE. Forward-pointer to a dedicated diffusion-models topic.

**Figure sketches (~7-8 figures).**

- Fig 1: Sample-complexity comparison (KDE rate vs parametric rate vs flow empirical).
- Fig 2: Change-of-variables visualized (Gaussian base → crescent target).
- Fig 3: Coupling-layer architecture diagram.
- Fig 4: Density vs. sampling speed trade-off table across flow families.
- Fig 5: Neural-ODE / FFJORD trajectory in 2-D.
- Fig 6: KDE vs Real NVP head-to-head on two moons.
- Fig 7: Samples from trained flow (show diversity).
- Fig 8: Forward-map to score-based and diffusion models.

**Interactive component sketches (~2 required).**

- **FlowVisualizer2D (featured).** Pre-trained small Real NVP on a 2-D preset (moons/pinwheel/grid). User watches individual points being transformed by each coupling layer and sees the base Gaussian gradually shaped into the target density. "Step through layers" control; "show inverse" toggle.
- **CouplingLayerDemo.** Interactive coupling layer with user-adjustable $s$ and $t$ functions (simple MLPs with 2-3 tunable parameters). User sees how each parameter affects the transformed output. Builds intuition for why coupling layers can be expressive.

**Cross-site pointers.**

- **Back to formalStatistics:** Topic 30 (KDE comparator); Topic 14 (MLE training); Topic 27 (Bayesian / variational inference, for IAF-as-posterior); Topic 9 (modes of convergence).
- **Back to formalCalculus:** Jacobians, change of variables, determinant rules, chain rule, ODEs.
- **Within formalml:** prerequisite neural-network-foundations topic (critical — see NF1); density-ratio-estimation (flows give explicit ratios); future VAE topic; future GAN topic; future score-based / diffusion-models topic; future variational-inference topic.

**Open design questions.**

- **NF1 (CRITICAL).** Is there a formalml topic covering neural-network fundamentals (universal approximation, backprop, basic architectures)? If yes, this topic's §1-§2 can be tight. If no, either (a) write that prerequisite topic first, or (b) include an extended §2 in this topic covering essentials (~500 extra words, at the cost of narrative focus). **Recommendation:** option (a) — write a short neural-foundations topic first. This is the single largest open decision for the entire five-topic batch.
- **NF2. Continuous-time flow depth.** Neural ODEs and FFJORD are a substantial body of material. Can they fit in one section? **Recommendation:** one-section treatment (§7) with one figure; full Neural-ODE treatment reserved for a dedicated topic.
- **NF3. Training-implementation detail.** Should the topic include actual PyTorch / JAX training code? **Recommendation:** schematic code snippets (20-line minimum Real NVP definition in PyTorch) as an Appendix; full training loop in a companion "normalizing-flows in practice" follow-on topic.
- **NF4. Score-based / diffusion-model pointer length.** §10 could be a paragraph or a full section. **Recommendation:** a tight ~400-word section with one schematic figure. These models are methodologically important and closely related enough that a forward-pointer-only treatment feels incomplete.
- **NF5. Comparison to KDE — quantitative or qualitative?** §8 could be a quantitative experiment (full figure with empirical sample-complexity curves) or a qualitative discussion. **Recommendation:** quantitative — one figure with actual held-out likelihood on the two-moons preset. Reinforces the curse-of-dimensionality framing from §1.

**Estimated scope.** ~2,500 words (the heaviest topic), 8 figures, 2 interactive components. 6-8 new exports — but these are less "shared-module" style because flow machinery is fundamentally neural-network-based and lives in a separate module (`flows.ts` or similar) rather than extending a kernel-methods shared module. Expected authoring time: four weeks (including whatever prerequisite-topic work NF1 entails).

---

## Appendix — Open Questions Consolidated

Gathered in one place for a single-reading pass. Roman numeral for site-level, letter-numeric for topic-specific.

### Site-level (for the Master Plan, Part I)

- **O1.** What is the current state of formalml.com? Greenfield / early-stage / advanced?
- **O2.** Accept the four-track structure (Foundations / Supervised / Unsupervised-Generative / Learning-Theory-Methodology), or propose a different structure?
- **O3.Does a formalml topic covering neural network fundamentals exist, is planned, or needs to be slotted in as a dedicated prerequisite before normalizing flows?
- **O4.** Code-example language policy: Python + NumPy only, or admit PyTorch / JAX for deep-learning-adjacent topics?
- **O5.** First-topic decision: greenlight kernel-regression as the first target, or a different starting point?

### Topic A (kernel-regression)

- **K1.** Multivariate NW scope — full section or sidebar?
- **K2.** Asymptotic-normality proof depth — stated-with-sketch (Recommended) or full proof?
- **K3.** Bandwidth-selection section — one-selector-featured-plus-name-checks (Recommended) or full multi-selector treatment?

### Topic B (local-regression)

- **L1.** Local-linear vs local-$p$ proof completeness — full for local-linear, sketch for general (Recommended)?
- **L2.** Smoothing-splines treatment — paragraph only (Recommended) or full section?
- **L3.** Derivative-estimation depth — one-section treatment (Recommended) or paragraph only?

### Topic C (clustering)

- **C1.** Mean-shift narrow scope (Recommended) or clustering-broad treatment?
- **C2.** Image-segmentation depth — full section (Recommended) or paragraph only?
- **C3.** Adaptive-bandwidth depth — paragraph only (Recommended)?
- **C4.** Shadow-kernel argument — remark pointer (Recommended) or full treatment?

### Topic D (density-ratio-estimation)

- **D1.** Neural DRE — paragraph pointer (Recommended) or full section?
- **D2.** MMD treatment depth — application-only (Recommended) or separate topic?
- **D3.** GAN application depth — paragraph with schematic (Recommended) or full section?
- **D4.** MI-estimation depth — worked example (Recommended) or no example?

### Topic E (normalizing-flows)

- **NF1 (CRITICAL).** Neural-network fundamentals prerequisite — write prerequisite topic first (Recommended), or extend §2 of this topic?
- **NF2.** Continuous-time flow depth — one section (Recommended) or full treatment?
- **NF3.** Training-implementation detail — snippet-level (Recommended) or full training-loop?
- **NF4.** Score-based / diffusion-model pointer length — tight section (Recommended) or paragraph only?
- **NF5.** KDE-comparison empirical — quantitative experiment (Recommended) or qualitative discussion?

---

## Summary and Next Steps

**Five topic briefs at planning depth.** Production briefs (Topic-30-scale, ~1,900 lines each) are a separate per-topic session after greenlight.

**Immediate decisions needed** (in priority order):

1. **O1** — current state of formalml.com. Blocks everything.
2. **NF1** — neural-network foundations prerequisite status. Blocks normalizing-flows (the heaviest topic); also a useful topic in its own right.
3. **O5** — first-topic target. Recommendation: **kernel-regression** (smallest prerequisite footprint, establishes patterns, direct extension of Topic 30).
4. **O2, O3, O4** — track structure, prerequisite planning, code policy.

**Session-workflow recommendation** (mirroring formalStatistics):

1. *Now:* this brief (done).
2. *Session 2:* confirm open questions above; scaffold formalml CLAUDE.md if not yet present; produce neural-foundations mini-brief if NF1 indicates writing one.
3. *Session 3:* production brief for kernel-regression (Topic A), notebook + handoff brief in the Topic-30 style.
4. *Session 4:* Claude Code implementation of kernel-regression using the session-3 artifacts.
5. *Sessions 5+:* repeat for local-regression, clustering, density-ratio-estimation, normalizing-flows in the sequencing from §I.5.

**— end of formalml satellite-topics planning brief.**
