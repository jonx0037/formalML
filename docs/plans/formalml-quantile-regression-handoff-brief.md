# formalML — Quantile Regression: Handoff Brief

> **Implementation brief for `formalml/quantile-regression`.** The notebook is the source of truth — `notebooks/quantile-regression/01_quantile_regression.ipynb` (already verified, end-to-end runtime 50.1s). This brief layers MDX-implementation specifics on top of that notebook. All math, code, and figures should come from the notebook directly; this document specifies how to render them on the site.

> **Scope of this brief.** Sections 1–14 below mirror the conformal-prediction handoff brief structure (PR #54). Per strategic plan §6.1 reuse policy, this brief reuses the `<ExternalLink>` and `<NamedSection>` infrastructure already shipped with conformal-prediction (PR #55, #56) — so §11 (component port) is **omitted**, with all subsequent section numbers preserved.

---

## 1. Strategic Decisions (Already Locked)

These decisions were finalized in the planning conversation prior to notebook authoring. They are not up for revision in implementation; flag any deviation as a design change.

| # | Decision | Rationale |
|---|---|---|
| 1 | **Sub-topic scope.** Linear QR / KB78 + asymptotic theory (Knight 1998) + multiple-quantile + crossing/rearrangement (CFV-G 2010) + multi-τ joint estimation + penalized QR (L1/L2) + CQR base-learner callback. **Defer:** nonparametric QR, Bayesian QR, censored/extreme-tail QR. | Gives a coherent "linear QR + practical extensions" arc without spilling into framework-level detours. Extreme tails are T4 Topic 4's territory (extreme-value-theory). |
| 2 | **Theorem density.** 4 Definitions + 4 Theorems (2 full proofs, 2 sketches) + 13 Remarks. | Thm 1 (pinball-min ⇒ quantile, FULL), Thm 2 (equivariance, FULL), Thm 3 (Knight 1998 asymptotic normality, SKETCH), Thm 4 (Belloni-Chernozhukov 2011 oracle rate, SKETCH). Sketches are detailed enough to be self-contained but defer the empirical-process scaffolding to formalstatistics Topic 32. |
| 3 | **Interactive viz.** 3 React/D3 components: `PinballLossExplorer`, `QRFitExplorer`, `BootstrapQuantileCI`. The penalized-path content in §7 ships as a static figure rather than a live component (computational cost of rerunning the full LP path on slider movement is prohibitive in-browser). | Same Approach-B (`useEffect` + manual refs) precedent as conformal-prediction's four components. Static fallback for §7 keeps the topic shippable without browser-side LP solving. |
| 4 | **Shared-module additions to `nonparametric-ml.ts`.** Three new exports: `bootstrapQuantileCI`, `fitPredictMultipleQuantiles`, `rearrangedQuantilePredictions`. `fitPredictQuantile` and `cqrInterval` already exist (shipped with conformal-prediction). **Defer** `fitPredictQuantileLasso` to a later T4 topic that needs it (likely prediction-intervals or a high-dim extension). | The L1-QR LP is computationally heavy and only used for the static §7 figure; the in-browser viz components don't need it. Kept out of the shared module to avoid bloating the cold-start payload. |
| 5 | **Code-library policy.** sklearn `QuantileRegressor(solver='highs')` for most QR fits in the notebook; `scipy.optimize.linprog(method='highs')` used in **one cell** (§3) to demonstrate the LP reformulation directly and verify equivalence to ~1e-14. | Per strategic plan §8.2: T4 stays in NumPy/SciPy/sklearn, no PyTorch/JAX. sklearn does the daily work; one explicit linprog cell makes Definition 3 concrete. |
| 6 | **Cross-site prerequisites.** `formalstatisticsPrereqs`: `linear-regression`, `order-statistics-and-quantiles`, `empirical-processes` (3 entries). `formalcalculusPrereqs`: `[]` (explicit). Internal `prerequisites`: `conformal-prediction`, `convex-analysis`, `gradient-descent`. | Linear regression is the OLS analog (Theorem 3 mirrors OLS asymptotic normality). Order statistics is the no-covariate special case of Theorem 1. Empirical processes provides the scaffolding for Theorem 3's full proof and Theorem 4's restricted-eigenvalue argument. The internal prereqs cover the LP/convexity machinery (convex-analysis), the smoothed-check-loss solver story (gradient-descent), and the CQR callback in §8 (conformal-prediction). |

> **Note on a fourth potential cross-site prereq.** `deferred-reciprocals.md` lists `formalStatistics/method-of-moments → quantile-regression` (M-estimation framing) under "When `formalml/quantile-regression` ships". Per Decision 6 above, method-of-moments is **not** included in this brief's `formalstatisticsPrereqs`. See §10 for the proposed resolution path (treat as a connection rather than a prereq, or add `formalstatisticsConnections` schema support).

---

## 2. Frontmatter

```yaml
---
title: "Quantile Regression"
description: "The Koenker–Bassett 1978 estimator: pinball-loss minimization, the LP reformulation, asymptotic normality, quantile crossing and rearrangement, and quantile regression as the base learner inside Conformalized Quantile Regression."
slug: "quantile-regression"
domain: "nonparametric-ml"
difficulty: "intermediate"
publishedDate: 2026-MM-DD
readingTime: 45 # estimate; refine after authoring
notebookPath: "notebooks/quantile-regression/01_quantile_regression.ipynb"
prerequisites:
  - topic: "conformal-prediction"
    relationship: "T4 Topic 1. The CQR callback in §8 builds directly on conformal-prediction §6's Definition 4. Reading conformal-prediction first frames QR as the base learner that supplies the band SHAPE; conformal calibration supplies the band WIDTH."
  - topic: "convex-analysis"
    relationship: "The pinball loss is a piecewise-linear convex function. The LP reformulation in §3 is exactly the standard reduction of piecewise-linear convex optimization to a linear program; the slack-variable trick (u = u⁺ − u⁻ with u⁺, u⁻ ≥ 0) is the canonical convex-analysis device."
  - topic: "gradient-descent"
    relationship: "The smoothed-check-loss accelerated-gradient solver underlying the in-browser visualization components is exactly the smoothed-objective + Nesterov-acceleration construction from gradient-descent §X — adapted to a non-smooth piecewise-linear loss via Moreau envelope smoothing."
formalstatisticsPrereqs:
  - topic: "linear-regression"
    site: "formalstatistics"
    relationship: "The OLS analog. Theorem 3's asymptotic normality of √n(β̂(τ) − β(τ)) ⟶ N(0, τ(1−τ) D⁻¹ Ω D⁻¹) is the QR replacement of OLS's √n(β̂ − β) ⟶ N(0, σ² (X′X)⁻¹). The Bernoulli factor τ(1−τ) plays the role of σ²; the density-weighted Gram matrix D(τ) sandwich-replaces (X′X)⁻¹. Reading linear-regression first makes the structural parallel transparent."
  - topic: "order-statistics-and-quantiles"
    site: "formalstatistics"
    relationship: "The no-covariate special case. With a single intercept feature, the KB78 estimator reduces to the empirical τ-quantile of {Yᵢ}; Theorem 1 reduces to the classical pinball-loss minimization characterization (Topic 29 §29.4); Theorem 3 reduces to the standard Bahadur-Ghosh asymptotics (Topic 29 §29.6). Topic 29 §29.10 Rem 19 explicitly motivates QR as the covariate-conditional extension."
  - topic: "empirical-processes"
    site: "formalstatistics"
    relationship: "Theorem 3's full proof runs through Knight 1998's identity plus an empirical-process limit of the rescaled QR objective; the argmin lemma is the continuous-mapping step. Theorem 4's restricted-eigenvalue argument and oracle inequality both use empirical-process tools (covering numbers, self-normalized concentration). Topic 32 §32.5 covers the full scaffold."
formalcalculusPrereqs: [] # no formalcalculus prerequisites — explicit acknowledgment per strategic plan §11.1
references:
  - type: "paper"
    title: "Regression Quantiles"
    authors: "Koenker & Bassett"
    year: 1978
    journal: "Econometrica"
    note: "The foundational paper. Definition 2 (linear QR estimator) and Theorem 3 (asymptotic normality) originate here."
  - type: "paper"
    title: "Limiting distributions for L1 regression estimators under general conditions"
    authors: "Knight"
    year: 1998
    journal: "Annals of Statistics"
    note: "The clean modern proof of Theorem 3 via the Knight-identity decomposition used in §5's proof sketch."
  - type: "book"
    title: "Quantile Regression"
    authors: "Koenker"
    year: 2005
    note: "The canonical book-length reference. Comprehensive coverage of LP reformulation, asymptotic theory, applications, and the rearrangement/crossing literature."
  - type: "paper"
    title: "Quantile and probability curves without crossing"
    authors: "Chernozhukov, Fernández-Val & Galichon"
    year: 2010
    journal: "Econometrica"
    note: "The rearrangement procedure (Definition 4 and Remark 11). Establishes that rearrangement weakly improves Lp approximation."
  - type: "paper"
    title: "L1-penalized quantile regression in high-dimensional sparse models"
    authors: "Belloni & Chernozhukov"
    year: 2011
    journal: "Annals of Statistics"
    note: "Theorem 4's oracle-rate result and the BC plug-in choice of λ (Remark 13). Restricted-eigenvalue conditions for QR-lasso."
  - type: "paper"
    title: "Noncrossing quantile regression curve estimation"
    authors: "Bondell, Reich & Wang"
    year: 2010
    journal: "Biometrika"
    note: "The constrained-LP alternative to rearrangement (Remark 10). Joint estimation of multiple quantile levels with monotonicity constraints."
  - type: "paper"
    title: "A note on quantiles in large samples"
    authors: "Bahadur"
    year: 1966
    journal: "Annals of Mathematical Statistics"
    note: "The original Bahadur representation in the no-covariate case. Remark 9 references the QR generalization."
  - type: "paper"
    title: "Conformalized quantile regression"
    authors: "Romano, Patterson & Candès"
    year: 2019
    journal: "NeurIPS"
    note: "The CQR procedure used in §8 (callback to T4 Topic 1). Same reference appears in conformal-prediction.mdx; intentional duplication for self-containment."
---
```

> **Notes on the frontmatter.**
> - Three internal prereqs (`prerequisites`) and three cross-site prereqs (`formalstatisticsPrereqs`) is the densest cross-site graph in the T4 track to date — reflects QR's position as the methodological hub between CQR (T4 #1), prediction intervals (T4 #6), and the formalstatistics regression / quantile-CI / empirical-process triangle.
> - `formalcalculusPrereqs: []` is explicit. QR's calculus content (the differentiation in Theorem 1's proof, the Knight-identity manipulation in Theorem 3's sketch) is light enough to live entirely within the topic.
> - All three `formalstatisticsPrereqs` entries discharge the corresponding "When `formalml/quantile-regression` ships" reciprocity-deferred items per `deferred-reciprocals.md`. See §10 for the audit-delta calculation and the open `method-of-moments` question.

---

## 3. Content Sections

The MDX body should mirror the notebook's section order. Pull all mathematical content (definitions, theorems, proofs, code snippets, figure captions) from the notebook — it has been verified end-to-end and the figures are reproducible bit-for-bit on a fresh run.

### Section outline

| § | MDX section title | Notebook cell | TheoremBlocks | Viz embedded |
|---|---|---|---|---|
| §1 | The Pinball Loss & The Population Quantile | Cell 4 | Definition 1 (Pinball loss), Theorem 1 (Pinball-minimization recovers the population quantile, with full proof), Remarks 1–2 | `PinballLossExplorer` (after Theorem 1) |
| §2 | Linear Quantile Regression | Cell 6 | Definition 2 (Linear QR estimator), Remarks 3–4 | `QRFitExplorer` (after Definition 2) |
| §3 | The LP Reformulation | Cell 8 | Definition 3 (LP reformulation of QR), Remark 5 | — (static figure only) |
| §4 | Equivariance Under Monotone Transformations | Cell 10 | Theorem 2 (Equivariance of conditional quantiles, with full proof), Remarks 6–7 | — |
| §5 | Asymptotic Theory: Koenker–Bassett 1978, Knight 1998 | Cell 12 | Theorem 3 (Asymptotic normality of QR, sketch), Remarks 8–9 | `BootstrapQuantileCI` (after Theorem 3) |
| §6 | Multi-Quantile Estimation, Crossing, and Rearrangement | Cell 14 | Definition 4 (Rearrangement), Remarks 10–11 | (re-uses `QRFitExplorer` with multi-τ + rearrange toggle) |
| §7 | Penalized Quantile Regression | Cell 16 | Theorem 4 (Belloni–Chernozhukov 2011 oracle rate, sketch), Remarks 12–13 | — (static figure only) |
| §8 | Quantile Regression as the Base Learner of Conformalized QR | Cell 18 | — (callback to conformal-prediction Definition 4) | — (static figure only) |
| §9 | Connections & Further Reading | Cell 20 (markdown) | — | — |

### TheoremBlock usage

Use the existing `TheoremBlock.astro` component for all numbered formal elements. The notebook contains:

| Type | # | Name | TheoremBlock `type` |
|---|---|---|---|
| Definition | 1 | Pinball loss | `definition` |
| Definition | 2 | Linear QR estimator | `definition` |
| Definition | 3 | LP reformulation of QR | `definition` |
| Definition | 4 | Rearrangement of conditional-quantile estimates | `definition` |
| Theorem | 1 | Pinball-minimization recovers the population quantile | `theorem` (with full `proof`) |
| Theorem | 2 | Equivariance of conditional quantiles | `theorem` (with full `proof`) |
| Theorem | 3 | Asymptotic normality of QR (Koenker–Bassett 1978, Knight 1998) | `theorem` (with `proof` — sketch) |
| Theorem | 4 | Belloni–Chernozhukov 2011 oracle rate for L1-QR | `theorem` (with `proof` — sketch) |
| Remark | 1 | Pinball-loss derivative requires no smoothness on Y | `remark` |
| Remark | 2 | Conditional version: argmin over functions = conditional τ-quantile | `remark` |
| Remark | 3 | When linear QR targets the right thing (well-specified vs heteroscedastic) | `remark` |
| Remark | 4 | Why an intercept matters | `remark` |
| Remark | 5 | Combinatorial structure of the QR LP optimum | `remark` |
| Remark | 6 | Equivariance is about the conditional quantile, not the linear estimator | `remark` |
| Remark | 7 | Contrast with the conditional mean (Jensen) | `remark` |
| Remark | 8 | The Bernoulli variance factor τ(1−τ) and tail inflation | `remark` |
| Remark | 9 | Bahadur representation at covariate values | `remark` |
| Remark | 10 | Quantile crossing as a coherence violation; constrained-LP fix | `remark` |
| Remark | 11 | Rearrangement weakly improves Lp approximation | `remark` |
| Remark | 12 | Why L1-QR is the natural high-dim QR | `remark` |
| Remark | 13 | Choice of λ (CV vs BC plug-in) | `remark` |

### LaTeX symbols to verify render correctly

The notation introduction and reading flow assume correct rendering of:

- `\rho_\tau(u) = u(\tau - \mathbb{1}\{u < 0\})` — Pinball loss
- `\xi_\tau(Y \mid X = x)` — Conditional τ-quantile
- `\hat\beta(\tau)` — QR estimator at level τ
- `\hat q_\tau(x) = x^\top \hat\beta(\tau)` — Fitted conditional quantile
- `\Phi^{-1}(\tau)` — Standard normal inverse CDF (used in the closed-form true conditional quantile under the heteroscedastic DGP)
- `\sigma(x) = 0.3 + 0.6 |x|` — Heteroscedastic noise scale (DGP)
- `u^+, u^-` — Slack variables in the LP reformulation
- `A_{eq} = [X, -I, I]`, `b_{eq} = y` — LP equality constraint matrix
- `\mathbf{1}^\top u^+`, `\mathbf{1}^\top u^-` — Sums of slacks in the LP cost
- `D(\tau) = \mathbb{E}[f_{Y \mid X}(\xi_\tau(Y \mid X) \mid X) \, X X^\top]` — Density-weighted Gram matrix (Theorem 3)
- `\Omega = \mathbb{E}[X X^\top]` — Standard Gram matrix (Theorem 3)
- `\tau(1-\tau)` — Bernoulli variance factor (Remark 8)
- `\tilde q_\tau(x) = \mathrm{sort}_\tau \{\hat q_{\tau_k}(x)\}` — Rearranged estimate (Definition 4)
- `\|\beta\|_1 = \mathbf{1}^\top \beta^+ + \mathbf{1}^\top \beta^-` — L1-norm split for the lasso-QR LP
- `\lambda \asymp \sqrt{\tau(1-\tau)} \sqrt{\log p / n}` — BC2011 λ-rate
- `\|\hat\beta(\tau) - \beta_0(\tau)\|_2 = O_P(\sqrt{s \log p / n})` — Oracle rate (Theorem 4)
- `E_i = \max\{\hat q_{\alpha/2}(X_i) - Y_i,\ Y_i - \hat q_{1-\alpha/2}(X_i)\}` — CQR nonconformity score (§8 callback)
- `\hat C_\alpha(x) = [\hat q_{\alpha/2}(x) - \hat Q,\ \hat q_{1-\alpha/2}(x) + \hat Q]` — CQR prediction set (§8 callback)

---
## 4. Interactive Visualizations

Three new React components in `src/components/viz/`. All use `client:visible` hydration. Follow Approach B (`useEffect` + manual refs for multi-panel layouts) per the conformal-prediction precedent (PR #54).

### 4a. `PinballLossExplorer.tsx` — Embedded after Theorem 1 in §1

**Mathematical content.** Live demonstration of Theorem 1: as τ varies, the pinball-loss V rotates asymmetrically; the population minimizer of E[ρ_τ(Y − q)] tracks Φ⁻¹(τ) for Y ∼ N(0,1) (or the user's chosen distribution); the gradient F(q) − τ vanishes precisely at the τ-quantile.

**Layout (3 panels, vertical stack on mobile, 3-column grid on desktop ≥ 768px).**
- **Top:** The pinball-loss V at the current τ-slider value. Show ρ_τ(u) for u ∈ [−2, 2] with the slope-(τ) right arm and slope-(τ−1) left arm visible. As τ moves, the V tilts.
- **Middle:** Expected pinball loss L(q) = E[ρ_τ(Y − q)] as a function of q for Y ∼ N(0,1) (via 5000 Monte Carlo samples, recomputed on slider change). Vertical dashed line at the empirical argmin; vertical solid line at Φ⁻¹(τ); they should agree to ~0.02 at this n.
- **Bottom:** The gradient L′(q) = F(q) − τ. Crosses zero exactly at q = Φ⁻¹(τ). Useful for the "first-order condition" framing in Remark 1.

**Controls.**
- Slider 1: τ ∈ [0.05, 0.95], default 0.50, step 0.01.
- Toggle: distribution of Y — Normal (default) | Exponential | Uniform[0, 1]. Each option uses a different closed-form true quantile so the "tracks Φ⁻¹(τ)" message generalizes correctly.

**Verification.** At τ = 0.5, all three panels should be symmetric about q = 0 for the Normal distribution. At τ = 0.95, the V tilts steeply right; the population minimizer is ≈ 1.645; L′(q) crosses zero at the same point.

**Performance.** 5000-sample MC on slider change is ~5ms in modern browsers. No throttling needed. Pre-compute the 5000 Y samples on mount to avoid recomputing them on each τ change.

### 4b. `QRFitExplorer.tsx` — Embedded after Definition 2 in §2 (and re-rendered in §6 with multi-τ + rearrange toggle)

**Mathematical content.** Live linear-QR fit on the synthetic heteroscedastic dataset from `synthHeteroscedastic`. User varies τ (single fit) or activates a multi-τ mode showing K = 11 simultaneous fits with optional rearrangement toggle (§6 anchor).

**Layout (single panel, full width).**
- Scatter of (X_i, Y_i) for the current dataset (seeded; resample button regenerates).
- One QR fit (single-τ mode) or 11 fits (multi-τ mode) overlaid.
- True conditional quantile (closed form under the DGP) shown as a dashed reference line.
- Crossings highlighted with red `×` markers when the multi-τ mode is active and rearrangement is OFF.

**Controls.**
- Slider 1: τ ∈ [0.05, 0.95], default 0.50, step 0.01. Active in single-τ mode only.
- Slider 2: n ∈ {50, 100, 200, 500, 1000}, default 200. Affects fit variance.
- Slider 3: polynomial degree ∈ {1, 2, 3, 5}, default 3.
- Toggle: single-τ vs multi-τ (K = 11: τ ∈ {0.05, 0.10, 0.20, …, 0.90, 0.95}).
- Toggle: rearrangement ON/OFF (active in multi-τ mode only).
- Button: "Resample data" — re-seed and refit.

**Verification.** In single-τ mode at τ = 0.5, n = 500, degree 3: the fit should track the true conditional median almost exactly. In multi-τ mode at n = 100 with rearrangement OFF: visible crossings near the boundaries x ≈ ±2; rearrangement ON: crossings disappear.

**Performance.** Each QR fit at n = 200, degree 3 is ~20ms via the smoothed-check-loss + Nesterov-AGD solver in `nonparametric-ml.ts` (`fitPredictQuantile`). Multi-τ mode (11 fits) runs in ~220ms — debounce slider input by 200ms to avoid stutter.

### 4c. `BootstrapQuantileCI.tsx` — Embedded after Theorem 3 in §5

**Mathematical content.** Live demonstration of Theorem 3: as n grows, the bootstrap distribution of β̂₁(τ) (the slope coefficient on x) concentrates around its mean and approaches a Gaussian. The std scales as 1/√n at fixed τ (Remark 8); cross-τ comparison shows tail inflation.

**Layout (2 panels side-by-side on desktop, stacked on mobile).**
- **Top/left:** Histogram of B = 200 bootstrap β̂₁(τ) values at the current (n, τ). Overlay a fitted Gaussian density. Annotate empirical mean and std.
- **Bottom/right:** Convergence plot — empirical std × √n on the y-axis, n on the x-axis (log-scale). At fixed τ, this should stabilize as n grows (Theorem 3's 1/√n rate). Plot two lines (one per τ) so users can see the τ → 0.9 inflation directly.

**Controls.**
- Slider 1: n ∈ {50, 100, 200, 500, 1000}, default 200, step on log scale.
- Slider 2: τ ∈ [0.10, 0.90], default 0.5, step 0.05. (Restricted to [0.10, 0.90] because the in-browser solver becomes unstable at extreme τ — the topic mentions this in Remark 8 and the EVT forward-pointer.)
- Button: "Run B = 200 bootstraps" (debounced; do not run on every slider tick).

**Verification.** At n = 200, τ = 0.5: empirical std × √n ≈ 2.0–2.5. At n = 200, τ = 0.9: same product is ≈ 3.0–3.5 (tail inflation). Histograms should look approximately bell-shaped, sharper at τ = 0.5 than at τ = 0.9.

**Performance.** B = 200 bootstrap fits at n = 200 takes ~5s in-browser (each fit ~25ms). This is at the edge of acceptable interactive latency. Use a "Run" button rather than auto-run-on-slider to keep the page responsive.

### 4d. Static figure for §7 (Penalized QR)

The L1-QR coefficient path with cross-validated λ is computed via a custom LP (the lasso-QR LP from the notebook's `solve_lasso_qr_linprog`). Browser-side LP solving for a 30 × 200 design over 30 λ values plus 5-fold CV would take ~30s on the user's machine and is not feasible for an interactive viz. Ship the notebook's `qr_penalized_path.png` as a static figure with the Theorem 4 callout adjacent.

> ⚠️ **Future work.** A simplified "lasso-QR with fixed CV-selected λ" interactive component is on the table for a later T4 topic (likely prediction-intervals's high-dim section). Out of scope for this brief.

### Component-shared concerns

- **Static fallback PNGs** for `client:visible` failure / no-JS environments: use `qr_pinball_loss.png`, `qr_linear_fit.png`, and `qr_asymptotic_normality.png` from the notebook's `figures/` output as `<noscript>`-wrapped fallbacks at the three viz anchor positions.
- **Color palette:** All three components use the same palette as the notebook (BLUE = `#2563EB`, RED = `#DC2626`, GREEN = `#059669`, AMBER = `#D97706`, PURPLE = `#7C3AED`, SLATE = `#475569`, TEAL = `#0F6E56`) — drawn from the formalML site's design tokens.
- **Mobile responsiveness:** All three must render correctly at 375px viewport. Multi-panel layouts collapse to vertical stacks below `md` breakpoint (768px).

---

## 5. Shared Module: `nonparametric-ml.ts`

The shared module exists from PR #54 (conformal-prediction). This brief **extends** it with three new exports.

### File location

`src/components/viz/shared/nonparametric-ml.ts`

### New exports required for this topic

```typescript
// === New for quantile-regression ===

/**
 * Bootstrap CI for a single coefficient of the linear-QR estimator.
 *
 * Internally fits B QR estimators on bootstrap resamples and returns the
 * empirical (alpha/2, 1 - alpha/2) quantiles of the chosen coefficient
 * (default index 1, i.e., the linear slope on the first non-intercept feature).
 *
 * Used by the BootstrapQuantileCI viz component.
 */
export function bootstrapQuantileCI(
  x: Float64Array,
  y: Float64Array,
  tau: number,
  B: number,
  alpha: number,
  rng: () => number,
  coefIndex?: number, // default 1
  degree?: number, // default 3
  lambda?: number, // L2 ridge for solver stability; default 0.01
): {
  coefDraws: Float64Array; // shape (B,) — values of the chosen coefficient across bootstraps
  ciLower: number;
  ciUpper: number;
  empiricalMean: number;
  empiricalStd: number;
};

/**
 * Fit linear QR at K levels simultaneously on the same training data.
 *
 * Returns a (K, nEval) row-major array Q where Q[k, j] = q-hat_{taus[k]}(xEval[j]).
 *
 * Used by the QRFitExplorer viz component (multi-tau mode) and by §6 of the
 * topic page.
 */
export function fitPredictMultipleQuantiles(
  xTrain: Float64Array,
  yTrain: Float64Array,
  xEval: Float64Array,
  taus: Float64Array,
  degree?: number, // default 3
  lambda?: number, // default 0.01
): Float64Array; // shape (taus.length * xEval.length), row-major

/**
 * Apply CFV-G 2010 rearrangement: sort along the tau axis at each evaluation point.
 *
 * Input Q has shape (K, nEval) row-major (as returned by fitPredictMultipleQuantiles).
 * Output Qtilde has the same shape, with each "column" Qtilde[:, j] sorted ascending.
 *
 * The function is a pure post-processing step on already-fitted quantile predictions;
 * it has no statistical side effects beyond enforcing monotonicity in tau.
 *
 * Used by the QRFitExplorer viz component (rearrange toggle in multi-tau mode).
 */
export function rearrangedQuantilePredictions(
  Q: Float64Array, // shape (K * nEval), row-major
  K: number,
  nEval: number,
): Float64Array; // shape (K * nEval), row-major
```

### Verification tests

Before the viz components reference these helpers, run a verification scratch script against the notebook's printed numerical outputs:

1. **`bootstrapQuantileCI` at (n = 100, τ = 0.5, B = 200).** Empirical std should be in [0.27, 0.34] (notebook reports 0.3075 at the same setting, ±15% bootstrap-of-bootstrap noise).
2. **`bootstrapQuantileCI` at (n = 500, τ = 0.5, B = 200).** Empirical std should be in [0.07, 0.11] (notebook reports 0.0903).
3. **`fitPredictMultipleQuantiles` at K = 11, n = 120 on `synthHeteroscedastic`.** Crossing count (i.e., evaluation points where some Q[k, j] > Q[k+1, j]) should be in [40, 80] (notebook reports 58/200 at this exact setting; bootstrap noise from the seed gives the range).
4. **`rearrangedQuantilePredictions` after step 3.** Crossing count should be exactly 0. L² error to the closed-form true conditional quantile should be ≤ the marginal-fit L² error (Remark 11).
5. **Sanity:** `rearrangedQuantilePredictions` applied to an already-monotone input is the identity (up to floating-point).

These tests can live in `src/components/viz/shared/__tests__/nonparametric-ml.test.ts` (added in PR #54) or be verified in a scratch component during development.

---

## 6. Curriculum Graph Updates

The `nonparametric-ml` track exists from the conformal-prediction ship (PR #54). The `quantile-regression` node should already be registered as `planned` per the curriculum-graph expansion brief.

### `src/data/curriculum-graph.json`

**Update existing node:** Set `quantile-regression` node's `status` from `"planned"` to `"published"`. Confirm `url` field is `"/topics/quantile-regression"`.

**Add edges:** The internal prereq edge `conformal-prediction → quantile-regression` already exists from PR #54. Add the two remaining internal-prereq edges:

```json
{ "source": "convex-analysis", "target": "quantile-regression" }
{ "source": "gradient-descent", "target": "quantile-regression" }
```

Add the downstream forward edge to the next-T4 topic (which exists as a `planned` node):

```json
{ "source": "quantile-regression", "target": "prediction-intervals" }
```

(The edge `conformal-prediction → prediction-intervals` is already in place from PR #54; this brief adds a second forward edge `quantile-regression → prediction-intervals` representing the second pillar of the prediction-intervals umbrella topic.)

**Cross-site nodes:** Three cross-site prereq nodes need to be present:

- `formalstatistics/linear-regression` — **new**, add with `domain: "external"`, `status: "external"`, `url: "https://formalstatistics.com/topics/linear-regression"`.
- `formalstatistics/order-statistics-and-quantiles` — already present from PR #54 (shared with conformal-prediction).
- `formalstatistics/empirical-processes` — already present from PR #54 (shared with conformal-prediction).

Add cross-site edges:

```json
{ "source": "formalstatistics/linear-regression", "target": "quantile-regression", "type": "cross-site-prereq" }
{ "source": "formalstatistics/order-statistics-and-quantiles", "target": "quantile-regression", "type": "cross-site-prereq" }
{ "source": "formalstatistics/empirical-processes", "target": "quantile-regression", "type": "cross-site-prereq" }
```

### `src/data/curriculum.ts`

Move `"quantile-regression"` from the `planned` array to the `published` array in the `nonparametric-ml` track.

> ⚠️ **If the curriculum-graph expansion PR is still pending.** The same minimal-registration fallback as conformal-prediction's brief applies: add the `quantile-regression` node with `domain: "nonparametric-ml"`, `status: "published"`, `url: "/topics/quantile-regression"`. Do NOT add the forward edge to `prediction-intervals` (would point at an unregistered node).

---

## 7. Cross-References (in-page)

### Outbound (from `quantile-regression.mdx`)

**Backward references — these pages exist, use live links:**

- `[Conformal Prediction](/topics/conformal-prediction)` — referenced in §8 (the entire CQR callback hangs on this), §9 Connections (forward direction inside the T4 track).
- `[Convex Analysis](/topics/convex-analysis)` — referenced in §3 (the LP reformulation is a direct application of piecewise-linear → LP reduction; the slack-split is the canonical convex-analysis device).
- `[Gradient Descent](/topics/gradient-descent)` — referenced in §7 Remark 12 (the smoothed-check-loss accelerated-gradient solver behind the in-browser viz components is a direct adaptation of the Nesterov-acceleration construction).

**Cross-site references to formalstatistics — body-prose `<ExternalLink>` per CLAUDE.md:**

- `<ExternalLink site="formalstatistics" topic="linear-regression">linear regression</ExternalLink>` — referenced in §1 (mean-vs-quantile contrast), §5 (the OLS asymptotic-normality analog of Theorem 3).
- `<ExternalLink site="formalstatistics" topic="order-statistics-and-quantiles">order statistics & quantiles</ExternalLink>` — referenced in §1 (no-covariate special case of Theorem 1), §5 Remark 9 (Bahadur 1966 baseline), §9 Connections.
- `<ExternalLink site="formalstatistics" topic="empirical-processes">empirical processes</ExternalLink>` — referenced in §5 (Theorem 3's full proof requires Topic 32 §32.5's scaffold), §7 (Theorem 4's restricted-eigenvalue argument).

> **Note:** Body-prose external links use the `<ExternalLink>` component shipped in PR #55. Frontmatter `formalstatisticsPrereqs` declares the same edges for the audit.

**Forward references to planned formalML topics — plain text per CLAUDE.md (do NOT hyperlink to unbuilt topics):**

- **Prediction Intervals** *(coming soon)* — referenced in §8 ("the umbrella treatment, comparing CQR with locally adaptive variants and alternative base learners") and §9 Connections.
- **Extreme Value Theory** *(coming soon)* — referenced in §5 Remark 8 ("the τ → 0 and τ → 1 regimes where Theorem 3's asymptotics break down") and §9 Connections.

⚠️ **Do NOT hyperlink to unbuilt topics.** Use the plain-text pattern `**Topic Name** *(coming soon)*`.

### Inbound (update existing pages)

- **`conformal-prediction.mdx`:** §6 (the CQR section) currently has a forward plain-text reference to "Quantile Regression *(coming soon)*". Replace with a live link `[Quantile Regression](/topics/quantile-regression)`. Specifically the line that reads: "_The base learner inside CQR. The full topic covers QR estimation theory, asymptotic distribution of $\hat\beta(\tau)$, and the broader use of quantile regression beyond conformal._" — keep the prose, swap the marker to a live link.
- **`convex-analysis.mdx`:** If the page has a §X Connections section listing applications of LP reductions, add a forward link to `[Quantile Regression](/topics/quantile-regression)` — the QR LP is one of the canonical examples of "piecewise-linear convex objective ⇒ LP". Only add if there's a natural insertion point.
- **`gradient-descent.mdx`:** If §X Connections mentions smoothed-objective methods or Moreau envelopes, add a forward link to `[Quantile Regression](/topics/quantile-regression)` — the smoothed-check-loss + Nesterov-AGD solver is a natural application. Only add if there's a natural insertion point.

---
## 8. `/paths` Page Update

The **Nonparametric & Distribution-Free** track exists from the conformal-prediction ship (PR #54). Update the existing track section.

### Update existing track section

**Track row to update:**

| Topic | Status | Difficulty | Badge |
|---|---|---|---|
| Conformal Prediction | **Published** (already linked) | Intermediate | "Start here" |
| **Quantile Regression** (this brief) | **Published** (linked to `/topics/quantile-regression`) | **Intermediate** | — |
| Rank Tests & Permutation Inference | Planned (gray, unlinked) | Intermediate | — |
| Extreme Value Theory | Planned (gray, unlinked) | Advanced | — |
| Statistical Depth | Planned (gray, unlinked) | Intermediate | — |
| Prediction Intervals | Planned (gray, unlinked) | Intermediate | — |

### Cross-site prereq rendering on `/paths` (per Decision 4, Option C — established in PR #54)

The cross-site rendering logic (reduced opacity 0.6 + dashed border) is already in place from PR #54. New cross-site node to add: `formalstatistics/linear-regression` — should render with the same Option-C styling. No new infrastructure required; just confirm the node renders correctly when it appears in the `quantile-regression` prereq subgraph.

---

## 9. Cross-Site Prerequisites

Per CLAUDE.md and strategic plan §11.1.

### formalstatistics

- <ExternalLink site="formalstatistics" topic="linear-regression">Linear Regression</ExternalLink> — The OLS analog. Theorem 3's asymptotic normality of √n(β̂(τ) − β(τ)) ⟶ N(0, τ(1−τ) D⁻¹ Ω D⁻¹) is the QR replacement of OLS's √n(β̂_OLS − β) ⟶ N(0, σ² (X′X)⁻¹). The Bernoulli factor τ(1−τ) plays the role of σ²; the density-weighted Gram matrix D(τ) sandwich-replaces (X′X)⁻¹. The structural parallel is the cleanest entry point. Referenced in §1 (mean-vs-quantile motivation), §5 (asymptotic-normality contrast).
- <ExternalLink site="formalstatistics" topic="order-statistics-and-quantiles">Order Statistics and Quantiles</ExternalLink> — The no-covariate special case. Topic 29 §29.4's pinball-loss minimization characterization of the empirical quantile is the X = constant case of Theorem 1; Topic 29 §29.6's Bahadur-Ghosh asymptotics is the X = constant case of Theorem 3. Topic 29 §29.10 Rem 19 explicitly motivates QR as the covariate-conditional extension. Referenced in §1 (special case), §5 Remark 9 (Bahadur 1966 baseline), §9 (connections).
- <ExternalLink site="formalstatistics" topic="empirical-processes">Empirical Processes</ExternalLink> — Theorem 3's full proof runs through Knight 1998's algebraic identity plus an empirical-process limit of the rescaled QR objective; the argmin lemma (continuous mapping) is the convergence-transfer step. Theorem 4's restricted-eigenvalue argument and oracle inequality both depend on covering-number / self-normalized-concentration tools. Topic 32 §32.5 covers the full scaffold. Referenced in §5 (proof sketch deferral), §7 (Theorem 4 sketch deferral).

### formalcalculus

None. This topic has no formalcalculus prerequisites — explicit acknowledgment per strategic plan §11.1. The calculus content (the differentiation in Theorem 1's proof, the Knight-identity manipulation in Theorem 3's sketch, the L′(q) = F(q) − τ identity) is light and self-contained within this topic; no formalcalculus dependencies.

---

## 10. Reciprocity Sweep at Ship Time

When this topic ships, the following sister-site MDX files need to have their `formalmlConnections` entries discharged. Per `docs/plans/deferred-reciprocals.md` heading "When `formalml/quantile-regression` ships":

1. **`formalstatistics/src/content/topics/method-of-moments.mdx`** — currently declares `formalmlConnections → quantile-regression` (M-estimation framing). **OPEN ITEM:** This brief's frontmatter does NOT include `method-of-moments` in `formalstatisticsPrereqs` per Decision 6 (method-of-moments is not a strict prereq for understanding QR). Three resolution paths:
   - **(a)** Add `method-of-moments` as a 4th `formalstatisticsPrereqs` entry (overrides Decision 6).
   - **(b)** Add a `formalstatisticsConnections` field to formalML's frontmatter schema, populate it with `method-of-moments`. The audit logic would need a parallel update to recognize `*Connections` as discharging deferred reciprocals where the source side is also a `formalmlConnections` (rather than `formalmlPrereqs`).
   - **(c)** Remove the `formalmlConnections → quantile-regression` declaration from `method-of-moments.mdx` on the formalstatistics side (treat the M-estimation framing as out-of-scope for QR's first ship; revisit when QR is later expanded to cover M-estimation explicitly).
   - **Recommended:** (b). It correctly captures that method-of-moments is a *connection* (mutual relevance) rather than a *prereq* (strict reading order), and establishes a pattern other T4–T6 topics will likely need. Coordinate with the audit-tooling owner before merging.
2. **`formalstatistics/src/content/topics/order-statistics-and-quantiles.mdx`** — currently declares `formalmlConnections → quantile-regression` (Koenker-Bassett extension framing). **DISCHARGED** automatically: this brief's `formalstatisticsPrereqs[order-statistics-and-quantiles]` entry pairs with the existing `formalmlConnections` entry, and the audit's "Reciprocated" count will increment on ship.

**Verification command (run from any of the three repos with sibling paths configured):**

```bash
pnpm audit:cross-site
```

Expected delta after ship:
- `Reciprocated` count: +1 (the `order-statistics-and-quantiles` pair).
- `Reciprocated` count via the (b) resolution: +2 (also `method-of-moments` if `formalstatisticsConnections` lands).
- Two existing entries under "When `formalml/quantile-regression` ships" in `deferred-reciprocals.md` should disappear (or one, if (c) is chosen).

### Sister-site cleanup (separate PR, out of scope here)

- The audit tooling does NOT yet recognize `formalstatisticsConnections` as a frontmatter field if path (b) is chosen. Adding parser support is a small audit-side change (~1 hour) and ships in a separate PR before this topic is merged.

---

## 12. Notebook Reference

> §11 (`<ExternalLink>` component port) is intentionally **omitted** — the component shipped with PR #55 (conformal-prediction). Section numbers below preserve the conformal-prediction brief's numbering for cross-brief consistency.

**Path:** `notebooks/quantile-regression/01_quantile_regression.ipynb`

**Structure:** 20 cells = 1 title md + 1 setup-header md + 1 setup code + (8 × [separator md + section code]) + 1 connections md.

**Status:** ✅ Verified — runs end-to-end in 50.1s on the development machine (well under the 60s CPU-only constraint per strategic plan §8.3). All numerical outputs match theoretical predictions:
- §1 empirical quantile recovery: |q̂ − Φ⁻¹(τ)| < 0.012 at n = 20,000.
- §2 KKT condition: P(Y < q̂_τ(X)) matches τ to ±0.005 at n = 500.
- §3 LP-vs-sklearn coefficient agreement: 8.4e-15 (machine precision).
- §3 interpolation count: exactly p = 4 (Remark 5).
- §4 equivariance population check: 0.004 max error.
- §5 bootstrap stds at fixed τ scale ~ 1/√n; cross-τ inflation visible (τ = 0.5 vs τ = 0.9).
- §6 crossings: 58/200 marginal → 0/200 rearranged. L²: 0.480 → 0.471 (Remark 11 verified strict-improvement case).
- §7 BC2011 active-set recovery: CV-selected λ = 8.20 captures all 5 true active features (4 false positives — Theorem 4 governs rate, not exact recovery).
- §8 CQR coverage: 0.872 (uncalibrated) → 0.918 (CQR-corrected) at α = 0.10 target.

All 8 figures generated cleanly with no matplotlib warnings.

**Figures produced (saved to notebook output cells; static PNGs to be exported to `public/images/topics/quantile-regression/`):**

| File | Source cell | Use in MDX |
|---|---|---|
| `qr_pinball_loss.png` | Cell 4 (§1) | §1 inline figure (3-panel: loss family + expected-loss minimization + n-convergence) — also `<noscript>` fallback for `PinballLossExplorer` |
| `qr_linear_fit.png` | Cell 6 (§2) | §2 inline figure (heteroscedastic data with 3-τ fits + residual histograms) — also `<noscript>` fallback for `QRFitExplorer` |
| `qr_lp_reformulation.png` | Cell 8 (§3) | §3 inline figure (residual structure + linprog-vs-sklearn agreement) |
| `qr_equivariance.png` | Cell 10 (§4) | §4 inline figure (4-panel equivariance demonstration) |
| `qr_asymptotic_normality.png` | Cell 12 (§5) | §5 inline figure (4-panel bootstrap distributions) — also `<noscript>` fallback for `BootstrapQuantileCI` |
| `qr_crossing_rearrangement.png` | Cell 14 (§6) | §6 inline figure (marginal vs rearranged 11-quantile fit) |
| `qr_penalized_path.png` | Cell 16 (§7) | §7 inline figure (L1-QR coefficient path + CV curve) — primary content; no live viz |
| `qr_cqr_callback.png` | Cell 18 (§8) | §8 inline figure (uncalibrated QR vs CQR-corrected band) |

The three live viz components have static PNG fallbacks per the table above. The `PinballLossExplorer`'s fallback is the full 3-panel `qr_pinball_loss.png`; the `QRFitExplorer`'s fallback shows just the single-τ mode at default settings; the `BootstrapQuantileCI`'s fallback shows the (n = 500, τ = 0.5) panel as a single-panel crop.

---

## 13. Build Order

1. **Pre-flight: verify the curriculum-graph expansion PR has shipped.** If yes, proceed. If no, follow the §6 minimal-registration fallback path.
2. **Confirm `<ExternalLink>` and `<NamedSection>` are available** in `src/components/ui/` (shipped in PR #55, #56). Smoke-test with a one-line MDX import.
3. **Decide path (a), (b), or (c) for the `method-of-moments` reciprocity item** (§10). If (b), coordinate with audit-tooling owner and schedule the schema-update PR before this topic merges.
4. **Extend `src/components/viz/shared/nonparametric-ml.ts`** (§5) with the three new exports. Run the §5 verification tests against the notebook's printed numerical outputs before any viz components reference them.
5. **Add the notebook** at `notebooks/quantile-regression/01_quantile_regression.ipynb` (Jonathan supplies; do not regenerate).
6. **Export static figures** from the notebook to `public/images/topics/quantile-regression/`. Names per §12.
7. **Create `src/content/topics/quantile-regression.mdx`** with full frontmatter (§2) and all markdown/LaTeX content (§3). Use `TheoremBlock` for all 8 numbered formal elements (4 Defs + 4 Theorems). Use `<ExternalLink>` for inline cross-site mentions of `linear-regression`, `order-statistics-and-quantiles`, `empirical-processes`. No interactive components yet — embed static figure imports as placeholders at the three viz-anchor positions (§1 after Theorem 1, §2 after Definition 2, §5 after Theorem 3).
8. **Build `PinballLossExplorer.tsx`** (§4a). Anchor in §1 after Theorem 1's proof. Verify against the pinball-loss minimization at the Normal default — empirical argmin should land on Φ⁻¹(τ) within Monte Carlo error.
9. **Build `QRFitExplorer.tsx`** (§4b). Anchor in §2 after Definition 2; component is re-rendered (with multi-τ + rearrange toggle activated) in §6 after Definition 4. Verify against notebook Cell 14's crossing counts at n = 120, K = 11.
10. **Build `BootstrapQuantileCI.tsx`** (§4c). Anchor in §5 after Theorem 3. Verify std × √n stabilization against notebook Cell 12's printed numerical outputs.
11. **Replace placeholders.** Swap each viz-anchor's static-figure import for the live React component import. Keep the static figure as `<noscript>` fallback.
12. **Verify §8's CQR callback** has a live link `[Conformal Prediction](/topics/conformal-prediction)` and reads correctly without re-deriving CQR — the callback should refer back to conformal-prediction §6 Definition 4 explicitly.
13. **Update curriculum graph data** (§6) — node status, internal-prereq edges, downstream forward edge to `prediction-intervals`, cross-site nodes/edges.
14. **Update `/paths` page** (§8) — flip `quantile-regression` from Planned → Published in the existing nonparametric-ml track section.
15. **Update inbound references** (§7 inbound list) — `conformal-prediction.mdx` (replace forward plain-text reference with live link), `convex-analysis.mdx`, `gradient-descent.mdx`. Only natural insertion points.
16. **Run testing checklist** (§14).
17. **Run `pnpm audit:cross-site`** from any sibling-configured repo. Verify the `order-statistics-and-quantiles` deferred-reciprocal discharges; verify the `method-of-moments` item's resolution path is reflected per §10.
18. **Commit and deploy.**

---

## 14. Testing Checklist

### Build / lint

- [ ] `pnpm build` completes with no errors.
- [ ] `pnpm typecheck` passes for the three new viz components.
- [ ] No console warnings in browser dev tools when viewing the topic.

### Math / proof rendering

- [ ] All 4 Definitions, 4 Theorems, and 13 Remarks render via `TheoremBlock` with correct numbering.
- [ ] Theorem 1's full proof renders correctly: the L(q) decomposition into right-side and left-side integrals, the L′(q) = F(q) − τ derivative, the F(q) = τ first-order condition, the F-flat-at-ξ_τ argmin convention.
- [ ] Theorem 2's full proof renders correctly: the {h(Y) ≤ v} = {Y ≤ h⁻¹₊(v)} step, the right-continuous inverse handling of non-strict monotonicity.
- [ ] Theorem 3's sketch renders the Knight-identity decomposition and the empirical-process scaffold deferral cleanly.
- [ ] Theorem 4's sketch renders the three-piece argument (gradient-domination, restricted-eigenvalue, oracle inequality) with the BC2011 reference intact.
- [ ] All LaTeX symbols from the §3 list render correctly (spot-check on dev server).
- [ ] No naked symbols introduced before their plain-English gloss.

### Interactive components

- [ ] `PinballLossExplorer`: τ slider responsive; the three panels (V, expected-loss, gradient) update together; the empirical argmin tracks Φ⁻¹(τ) within ±0.05 across the slider range.
- [ ] `PinballLossExplorer`: distribution toggle (Normal/Exponential/Uniform) re-runs the MC and updates the closed-form true-quantile reference line correctly.
- [ ] `QRFitExplorer`: single-τ mode at default settings shows a clean fit on heteroscedastic data; multi-τ mode shows ~K = 11 colored curves; rearrange toggle visibly removes crossings near x ≈ ±2 at n = 100.
- [ ] `QRFitExplorer`: degree slider (1, 2, 3, 5) shows clear under/over-fitting at the extremes — degree 1 misses heteroscedastic curvature; degree 5 overfits.
- [ ] `QRFitExplorer`: "Resample data" button regenerates the dataset and re-fits without page reload.
- [ ] `BootstrapQuantileCI`: histogram is approximately bell-shaped at n ≥ 200; sharper at τ = 0.5 than at τ = 0.9 (Remark 8 visible).
- [ ] `BootstrapQuantileCI`: convergence plot's std × √n line stabilizes as n grows; cross-τ separation (τ = 0.5 vs τ = 0.9) is visible at n = 500.
- [ ] All three components render fallback static PNGs when JavaScript is disabled (`<noscript>`).
- [ ] All three components are responsive on mobile (≤ 375px viewport): multi-panel layouts collapse to vertical stacks below `md` breakpoint.

### Cross-references

- [ ] All three backward-reference live links resolve (`conformal-prediction`, `convex-analysis`, `gradient-descent`).
- [ ] Three `<ExternalLink>`-rendered cross-site mentions (`linear-regression`, `order-statistics-and-quantiles`, `empirical-processes`) render with the formalstatistics glyph and `target="_blank" rel="noopener"`.
- [ ] Two forward-reference plain-text mentions (`prediction-intervals`, `extreme-value-theory`) render as plain text (not links).
- [ ] §8's callback to conformal-prediction §6 Definition 4 has a live link to `/topics/conformal-prediction`.

### Curriculum graph + `/paths`

- [ ] `/paths` shows `quantile-regression` as Published in the existing Nonparametric & Distribution-Free track.
- [ ] Quantile-regression's prereq subgraph shows three internal nodes (`conformal-prediction`, `convex-analysis`, `gradient-descent`) plus three cross-site nodes (`formalstatistics/linear-regression`, `formalstatistics/order-statistics-and-quantiles`, `formalstatistics/empirical-processes`).
- [ ] Cross-site nodes render with reduced opacity 0.6 + dashed border (per Decision 4 / Option C from PR #54).
- [ ] `formalstatistics`-suffix tooltip appears on hover over cross-site nodes.
- [ ] Forward edge `quantile-regression → prediction-intervals` renders correctly (target node still in `planned` state).

### Audit

- [ ] `pnpm audit:cross-site` reports 0 errors specific to this topic.
- [ ] The deferred-reciprocal entry for `formalstatistics/order-statistics-and-quantiles → quantile-regression` retires.
- [ ] The `method-of-moments` deferred-reciprocal item's resolution (path a/b/c per §10) is reflected correctly: either retires (if a or b), or removed from `deferred-reciprocals.md` (if c).
- [ ] No new "Missing reciprocals" appear.

### Performance

- [ ] Topic page first-paint under 2s on a 4G connection.
- [ ] `PinballLossExplorer` and `QRFitExplorer` (single-τ) hydrate within 500ms after `client:visible` triggers.
- [ ] `BootstrapQuantileCI` shows the "Run B = 200 bootstraps" button immediately on hydration; the ~5s bootstrap run only happens on user click (not on slider movement).
- [ ] No layout shift after hydration.

---

## Open Questions for Implementation

1. **`method-of-moments` reciprocity resolution.** Per §10, three paths (a/b/c). Recommended: (b), but coordinate with audit-tooling owner before merging. Pick before §10's testing checklist runs.
2. **`<ExternalLink>` schema for `formalstatistics/linear-regression`.** This is a new cross-site node not present in the conformal-prediction prereq set. Confirm the formalstatistics-side `linear-regression` slug is exactly `linear-regression` (some sites have it under `linear-models` or `regression-fundamentals`) before populating the frontmatter and `<ExternalLink>` calls.
3. **`BootstrapQuantileCI` initial-render strategy.** Two options: (i) run B = 50 quick bootstraps on hydration to populate the histogram with a placeholder, then let the user click "Run B = 200" for the precise version; (ii) show an empty-histogram skeleton on hydration and require the click before any data renders. Pick (i) if the ~1.2s initial-bootstrap cost is acceptable; (ii) if hydration must be instant. Decide during implementation.
4. **Polynomial-degree slider in `QRFitExplorer`.** §4b lists {1, 2, 3, 5}. Should degree 4 be included for completeness? Decision: skip — the marginal pedagogical value is low and it adds slider clutter. Keep {1, 2, 3, 5}.

---

## Appendix A: Key Differences from Prior Briefs

1. **Reuses infrastructure shipped in PR #54–#56.** No new track creation (Nonparametric & Distribution-Free track exists); no new shared module (extends existing `nonparametric-ml.ts`); no new component port (`<ExternalLink>` and `<NamedSection>` already in place). This brief is a "fill in the next slot" brief, not a "new infrastructure" brief.
2. **First topic with `linear-regression` as a cross-site prereq.** Adds a new cross-site node to the curriculum graph; tests the rendering of the `formalstatistics/linear-regression` external node in the prereq subgraph.
3. **Theorem density: 2 full proofs + 2 sketches.** Slightly less rigorous than conformal-prediction's 4 full proofs, by design — Knight 1998 and BC2011 require empirical-process machinery that lives in formalstatistics Topic 32; full proofs would create circular cross-site coupling.
4. **One section without an interactive viz.** §7 (penalized QR) ships as a static figure rather than a live component because browser-side LP solving for the L1-QR path would be prohibitive. This is the first "static-only" section in the T4 track; precedents for similar trade-offs exist in Topology & TDA (the persistent-homology computation is also static-only for performance reasons).
5. **CQR callback section (§8).** First topic to call back into a previously-shipped topic at the section level. The callback is structural — §8 is largely a re-run of conformal-prediction §6 from the QR side — and tests whether the live-link-back-to-prereq pattern reads cleanly.
6. **Three live viz components vs conformal's four.** Component density per topic is a lever, not a target. Three is appropriate for QR's structure: §1 (loss family), §2/§6 (fit explorer reused with toggle), §5 (bootstrap). §3, §4, §7, §8 use static figures.

---

## Appendix B: T4 Track Status After This Brief

| Order | Topic | Difficulty | Prerequisites | Status |
|---|---|---|---|---|
| 1 | **Conformal Prediction** | Intermediate | concentration-inequalities, pca-low-rank, formalstatistics/order-statistics-and-quantiles, formalstatistics/empirical-processes | ✅ Published |
| 2 | **Quantile Regression** (this brief) | Intermediate | conformal-prediction, convex-analysis, gradient-descent, formalstatistics/linear-regression, formalstatistics/order-statistics-and-quantiles, formalstatistics/empirical-processes | 🚧 Ready for implementation |
| 3 | Rank Tests & Permutation Inference | Intermediate | conformal-prediction, formalstatistics/hypothesis-testing | Planned |
| 4 | Statistical Depth | Intermediate | conformal-prediction, formalstatistics/order-statistics-and-quantiles | Planned |
| 5 | Extreme Value Theory | Advanced | concentration-inequalities, formalstatistics/order-statistics-and-quantiles | Planned |
| 6 | Prediction Intervals | Intermediate | conformal-prediction, **quantile-regression** | Planned (now has both pillars) |

DAG edges after this brief implements:
- `conformal-prediction → quantile-regression` (already in place from PR #54)
- `convex-analysis → quantile-regression` (new)
- `gradient-descent → quantile-regression` (new)
- `quantile-regression → prediction-intervals` (new — the second pillar after `conformal-prediction → prediction-intervals`)

Cross-site edges after this brief implements:
- `formalstatistics/linear-regression → quantile-regression` (new — new cross-site node)
- `formalstatistics/order-statistics-and-quantiles → quantile-regression` (new edge to existing cross-site node)
- `formalstatistics/empirical-processes → quantile-regression` (new edge to existing cross-site node)

---

*Brief version: v1 | Last updated: 2026-04-25 | Author: Jonathan Rocha*
*Reference notebook: `notebooks/quantile-regression/01_quantile_regression.ipynb`*
*Reference doc: `docs/plans/formalml-handoff-reference.md`*
*Strategic plan: `docs/plans/formalml-consolidated-strategic-planning-document.md`*
*Predecessor brief: `docs/plans/formalml-conformal-prediction-handoff-brief.md` (PR #54)*
