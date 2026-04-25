# Handoff Brief — Conformal Prediction (Topic 1, T4)

**Status:** 🚧 Ready for implementation
**Track:** T4 Nonparametric & Distribution-Free (this is the first topic in a new track)
**Slug:** `conformal-prediction`
**Difficulty:** intermediate
**Author:** Jonathan Rocha — Last updated 2026-04-25

This brief layers MDX-structure, viz-component, and integration spec on top of the verified notebook at `notebooks/conformal-prediction/01_conformal_prediction.ipynb`. The notebook is the source of truth for math, code, and figures. Where this brief and the notebook disagree, the notebook wins.

---

## 1. Strategic Decisions (Already Locked)

These decisions come from prior planning sessions; do not relitigate.

| Decision | Choice | Reference |
|---|---|---|
| Track placement | T4 Nonparametric & Distribution-Free | Strategic plan §3.4 |
| Slug | `conformal-prediction` | Strategic plan §2.2 |
| Difficulty | intermediate | Strategic plan §3.4 |
| Code-example language | Python + NumPy / SciPy / scikit-learn | Strategic plan §8.2 (T4 stays in default stack — no PyTorch) |
| Shared module | Create `nonparametric-ml.ts` (this is the track-introducer) | Strategic plan §6.2 |
| Theorem depth | Max rigor — full proofs of Marginal Coverage, Jackknife+ (BCRT 2021), Conditional Impossibility (Foygel-Barber 2021) | Topic-planning Decision 1 |
| Viz components | Four: `ConformalPredictionExplorer`, `CQRExplorer`, `APSForClassification`, `ExchangeabilityBreakdown` | Topic-planning Decision 2 |
| CV+ scope | Named section §3.8 inside `conformal-prediction`; `cross-validation` slug never minted | Strategic plan §7.3, Topic-planning Decision 3 |
| `/paths` cross-site rendering | Option C — reduced opacity + dashed border for cross-site prereq nodes; defer site-icon badges to curriculum-graph-expansion PR | Topic-planning Decision 4 |
| `ConformalPredictionExplorer` design | Option A — empirical coverage across many trials, sliders for both α and n_cal | Topic-planning Decision 5 |
| Notebook scope | Hybrid — pure NumPy/SciPy/sklearn-QR in main body; sklearn MLP + RF in dedicated final cell | Topic-planning Decision 6 |

> ⚠️ **Curriculum-graph note.** This brief assumes the planned-but-not-yet-registered T4–T6 nodes already exist in `curriculum-graph.json`. If the curriculum-graph expansion sweep PR has *not* shipped before this topic begins, see §6 below for the minimal node-registration that this brief alone requires.

---

## 2. Frontmatter

```yaml
---
title: "Conformal Prediction"
description: "Distribution-free prediction sets with finite-sample coverage guarantees — wrap any ML model in a procedure that produces valid intervals from exchangeability alone."
slug: "conformal-prediction"
domain: "nonparametric-ml"
difficulty: "intermediate"
publishedDate: 2026-MM-DD
readingTime: 50 # estimate; refine after authoring
notebookPath: "notebooks/conformal-prediction/01_conformal_prediction.ipynb"
prerequisites:
  - topic: "concentration-inequalities"
    relationship: "Concentration is the standard tool for bounding empirical-quantile fluctuation; this topic uses the rank-symmetry alternative for a tighter, finite-sample statement. The contrast is instructive."
  - topic: "pca-low-rank"
    relationship: "Calibration scores function as low-dimensional summaries of prediction error; the leave-one-out updates underlying jackknife+ are analogous to LOO covariance updates in low-rank settings."
formalstatisticsPrereqs:
  - topic: "order-statistics-and-quantiles"
    site: "formalstatistics"
    relationship: "Topic 29's distribution-free quantile CI via paired order statistics (§29.7) uses the same exchangeability + rank-symmetry argument that drives Theorem 1 here. Topic 29 §29.10 Rem 21 explicitly motivates Vovk–Gammerman–Shafer's distribution-free prediction as a generalization from a fixed parameter to a random variable."
  - topic: "empirical-processes"
    site: "formalstatistics"
    relationship: "Topic 32's uniform-convergence bounds (DKW inequality, VC dimension) are the asymptotic alternative to conformal's exact finite-sample coverage. The two routes meet at distribution-free prediction: empirical-processes pays a uniform-convergence rate; conformal pays only marginal validity. §1 of this topic frames the trade-off."
formalcalculusPrereqs: [] # this topic has no formalcalculus prerequisites — explicit acknowledgment
references:
  - type: "book"
    title: "Algorithmic Learning in a Random World"
    authors: "Vovk, Gammerman & Shafer"
    year: 2005
    note: "The book that systematized conformal prediction. Definitive reference for the full / transductive variant."
  - type: "paper"
    title: "Distribution-free predictive inference for regression"
    authors: "Lei, G'Sell, Rinaldo, Tibshirani & Wasserman"
    year: 2018
    journal: "Journal of the American Statistical Association"
    url: "https://doi.org/10.1080/01621459.2017.1307116"
    note: "The marginal-coverage theorem (Theorem 1) and split conformal as the modern default."
  - type: "paper"
    title: "Conformalized quantile regression"
    authors: "Romano, Patterson & Candès"
    year: 2019
    journal: "NeurIPS"
    note: "CQR — locally adaptive prediction intervals via quantile-regression base learner."
  - type: "paper"
    title: "Classification with valid and adaptive coverage"
    authors: "Romano, Sesia & Candès"
    year: 2020
    journal: "NeurIPS"
    note: "APS for classification with approximate conditional coverage."
  - type: "paper"
    title: "Predictive inference with the jackknife+"
    authors: "Barber, Candès, Ramdas & Tibshirani"
    year: 2021
    journal: "The Annals of Statistics"
    url: "https://doi.org/10.1214/20-AOS1965"
    note: "Jackknife+ and CV+ — leave-one-out conformal with the 1−2α coverage bound (Theorem 2)."
  - type: "paper"
    title: "The limits of distribution-free conditional predictive inference"
    authors: "Foygel Barber, Candès, Ramdas & Tibshirani"
    year: 2021
    journal: "Information and Inference"
    url: "https://doi.org/10.1093/imaiai/iaaa017"
    note: "Conditional coverage impossibility (Theorem 4)."
  - type: "paper"
    title: "Conformal prediction under covariate shift"
    authors: "Tibshirani, Foygel Barber, Candès & Ramdas"
    year: 2019
    journal: "NeurIPS"
    note: "Importance-weighted conformal for known covariate shift — the key extension powering ExchangeabilityBreakdown viz."
  - type: "paper"
    title: "Conditional validity of inductive conformal predictors"
    authors: "Vovk"
    year: 2012
    journal: "Machine Learning"
    note: "Original conditional-coverage impossibility statement (the special case Theorem 4 generalizes)."
  - type: "paper"
    title: "Distributional conformal prediction"
    authors: "Chernozhukov, Wüthrich & Zhu"
    year: 2021
    journal: "PNAS"
    note: "Conformal prediction extended to full conditional CDF inference."
---
```

> **Notes on the frontmatter.**
> - `prerequisites` lists internal formalML prereqs only. Cross-site prereqs go in the dedicated `formalstatisticsPrereqs` / `formalcalculusPrereqs` fields per CLAUDE.md.
> - `formalcalculusPrereqs: []` is an explicit empty array, not omitted. Per strategic plan §11.1, empty = explicit acknowledgment (the audit accepts this as valid).
> - The two `formalstatisticsPrereqs` entries discharge both `When formalml/conformal-prediction ships` reciprocity requirements (see §10).

---

## 3. Content Sections

The MDX body should mirror the notebook's section order. Pull all mathematical content from the notebook — definitions, theorems, proofs, code snippets, and figure captions are already verified there.

### Section outline

| § | MDX section title | Notebook cell | TheoremBlocks | Viz embedded |
|---|---|---|---|---|
| §1 | Distribution-Free Prediction & Exchangeability | Cell 1 | Definition 1 (Exchangeability) | — |
| §2 | Split Conformal Prediction | Cell 2 | Definition 2 (Split Conformal Prediction Set) | — |
| §3 | Marginal Coverage: The Central Theorem | Cell 3 | Theorem 1 (Marginal Coverage, with full proof) | `ConformalPredictionExplorer` (after Theorem 1) |
| §3.8 | Cross-Validation in Conformal: CV+ | Cell 5 (CV+ subsection) | — | — |
| §4 | Full (Transductive) Conformal Prediction | Cell 4 | — | — |
| §5 | Jackknife+ and CV+ | Cell 5 | Definition 3 (Jackknife+ Prediction Interval), Theorem 2 (Jackknife+ Coverage Bound, with full proof) | — |
| §6 | Conformalized Quantile Regression (CQR) | Cell 6 | Definition 4 (CQR Prediction Set), Theorem 3 (CQR Coverage Inheritance) | `CQRExplorer` (after Definition 4) |
| §7 | Adaptive Prediction Sets (APS) for Classification | Cell 7 | Definition 5 (APS Score and Prediction Set, deterministic) | `APSForClassification` (after Definition 5) |
| §8 | Conditional Coverage: Impossibility and Approximations | Cell 8 | Theorem 4 (Conditional Coverage Impossibility, with full constructive proof) | `ExchangeabilityBreakdown` (after Remark 3 on covariate shift) |
| §9 | Wrapping Black-Box ML Models | Cell 9 | — | — |
| §10 | Connections & Further Reading | Cell 10 (markdown) | — | — |

### Cross-validation as named section (§3.8)

Per strategic plan §7.3 and topic-planning Decision 3, the slug `cross-validation` never gets minted as its own topic. Instead, §3.8 within `conformal-prediction.mdx` carries the CV+ content under an `id` that sister sites can deep-link to:

```mdx
<NamedSection id="cross-validation" title="Cross-Validation in Conformal: CV+">
  <!-- §3.8 content: K-fold CV+ exposition, computational tradeoff vs jackknife+,
       reference back to Theorem 2 -->
</NamedSection>
```

The `NamedSection` component (port from formalstatistics if not yet in formalML's `src/components/ui/` — it's a thin wrapper around `<section id={id}><h2>{title}</h2>{children}</section>`) gives `/topics/conformal-prediction#cross-validation` as a stable anchor URL. Sister-site links pointing at `formalml/cross-validation` should be updated to point at this anchor in a sister-site cleanup PR (out of scope for this brief).

> **Note on §3.8 placement.** Numbering it §3.8 (rather than §5.5 inside the jackknife+ section) follows the strategic plan §7.3 precedent and gives it visual prominence as a separate H2. The CV+ math itself comes from the notebook's Cell 5; §3.8 is structurally part of the same conceptual unit but anchored separately for sister-site routability.

### TheoremBlock usage

Use the existing `TheoremBlock.astro` component for all numbered formal elements. The notebook contains:

| Type | # | Name | TheoremBlock `type` |
|---|---|---|---|
| Definition | 1 | Exchangeability | `definition` |
| Definition | 2 | Split Conformal Prediction Set | `definition` |
| Definition | 3 | Jackknife+ Prediction Interval | `definition` |
| Definition | 4 | CQR Prediction Set | `definition` |
| Definition | 5 | APS Score and Prediction Set (deterministic) | `definition` |
| Theorem | 1 | Marginal Coverage (Lei et al. 2018) | `theorem` (with full `proof`) |
| Theorem | 2 | Jackknife+ Coverage Bound (BCRT 2021) | `theorem` (with full `proof`) |
| Theorem | 3 | CQR Coverage Inheritance | `theorem` (with `proof`) |
| Theorem | 4 | Conditional Coverage Impossibility (Foygel-Barber 2021) | `theorem` (with full constructive `proof`) |
| Remark | — | Marginal Coverage Proof Independence (after Thm 1 proof) | `remark` |
| Remark | — | Jackknife+ Worst-Case Tightness (after Thm 2) | `remark` |
| Remark | — | CQR's Novelty Is the Score, Not the Theorem (after Thm 3) | `remark` |
| Remark | — | Impossibility Holds with Only Measurability (after Thm 4 proof) | `remark` |
| Remark | — | Approximate Conditional Coverage of CQR/APS | `remark` |
| Remark | — | Covariate Shift Connection (Tibshirani-Barber-Candès-Ramdas 2019) | `remark` (also serves as anchor for `ExchangeabilityBreakdown` viz) |

### LaTeX symbols to verify render correctly

Notation introduction and reading flow assume correct rendering of:

- `\hat C_\alpha(x)` — Prediction set
- `s(x, y)`, `S_i = s(X_i, Y_i)` — Nonconformity score
- `\hat q_{1-\alpha}` — Threshold (the $\lceil (1-\alpha)(n_{\text{cal}}+1)\rceil$-th smallest calibration score)
- `\lceil (1 - \alpha)(n_{\text{cal}} + 1) \rceil` — Ceiling appears throughout proofs; verify spacing
- `\hat\mu_{-i}(x)` — Leave-one-out predictor (jackknife+)
- `R_i = |Y_i - \hat\mu_{-i}(X_i)|` — Leave-one-out residual
- `\hat Q_\alpha^-`, `\hat Q_{1-\alpha}^+` — Empirical quantiles with floor/ceiling conventions
- `\hat q_{\alpha/2}(x)`, `\hat q_{1-\alpha/2}(x)` — Conditional quantile estimates (CQR)
- `E_i = \max\{\hat q_{\alpha/2}(X_i) - Y_i,\ Y_i - \hat q_{1-\alpha/2}(X_i)\}` — CQR score
- `\hat\pi(c \mid x)` — Predicted softmax probability
- `c_{(j)}(x)` — $j$-th most probable class (descending order)
- `\rho(y; x)` — Class rank
- `\sigma^2(x; \varepsilon) = \sigma_0^2 + M^2 \cdot \mathbb{1}\{|x - x_0| < \varepsilon/2\}` — Spiked-variance adversarial family
- `\Phi^{-1}(1 - \alpha/2)` — Standard normal inverse CDF (in Theorem 4 proof's lower bound)

---

## 4. Interactive Visualizations

Four new React components in `src/components/viz/`. All use `client:visible` hydration. Follow Approach B (`useEffect` + manual refs for multi-panel layouts) per the random-walks / graph-laplacians precedent.

### 4a. `ConformalPredictionExplorer.tsx` — Flagship component, embedded after Theorem 1

**Mathematical content.** Empirical demonstration of marginal coverage. Real-time Monte Carlo at user-controlled $(\alpha, n_{\text{cal}})$ shows empirical coverage concentrating in the proven $[1-\alpha,\ 1-\alpha+1/(n_{\text{cal}}+1)]$ band.

**Layout (3 panels, vertical stack on mobile, horizontal on desktop).**
- **Top:** Heteroscedastic 1D regression scatter + ridge predictor + prediction band at current settings. Live update on slider change.
- **Middle:** Coverage histogram across $T = 500$ trials at current $(\alpha, n_{\text{cal}})$. Two reference lines: $1 - \alpha$ (red dashed) and $1 - \alpha + 1/(n_{\text{cal}}+1)$ (amber dotted). Mean of histogram annotated.
- **Bottom:** Running mean coverage over the $T = 500$ trials (log-x), showing convergence to the band.

**Controls.**
- Slider 1: $\alpha \in [0.01, 0.30]$, default 0.10, step 0.01.
- Slider 2: $n_{\text{cal}}$ on log scale: $\{20, 50, 100, 200, 500, 1000\}$, default 200.
- "Resample" button to re-draw all $T$ trials at current settings (otherwise debounced re-run on slider change).

**Implementation notes.**
- $T = 500$ trials at $n_{\text{cal}} = 1000$ should complete in ≤ 200 ms in-browser. Pre-compute the synthetic dataset on mount; only redo the train/cal/test splits and ridge fits per trial.
- Closed-form polynomial-degree-3 ridge is the only base predictor — keeps it fast. Function `fitPredictRidge` in `nonparametric-ml.ts` (see §5).
- Histogram uses a fixed bin count of 25 across $[0.7, 1.0]$ regardless of $\alpha$ to keep visual scale stable.
- The "running mean" panel is a line chart, not a separate computation — it's the cumulative-mean of the same $T$ trial indicators.

**Why Option A.** Per topic-planning Decision 5: showing empirical coverage across many trials emphasizes that the marginal-coverage guarantee is a statement about the *distribution* of coverage indicators, not a single-trial property. Single-dataset visualization (the alternative Option B) understates the role of resampling.

**Anchor.** Embed after Theorem 1's proof, before the upper-bound complement statement.

### 4b. `CQRExplorer.tsx` — Embedded after Definition 4 in §6

**Mathematical content.** Side-by-side comparison of naive split conformal (mean predictor + absolute residual) vs. CQR on heteroscedastic data. Demonstrates the locally adaptive widths CQR produces while preserving marginal coverage.

**Layout (2 panels).**
- **Left:** Heteroscedastic regression scatter + two prediction bands overlaid (naive in blue, CQR in green). Both at current $\alpha$.
- **Right:** Conditional coverage as a function of $x$ (binned, ~12 bins), plotted for both methods. Reference line at $1 - \alpha$. Visual story: CQR's curve is flatter (more uniform conditional coverage); naive's curve dips in high-noise regions.

**Controls.**
- Slider: $\alpha \in [0.05, 0.25]$, default 0.10.
- Toggle: "Show naive" / "Show CQR" / "Show both" (default both).
- Slider: heteroscedasticity strength (parameter $h$ in $\sigma(x) = 0.3 + h \cdot |x|$), $h \in [0, 1.5]$, default 0.6.

**Implementation notes.**
- Quantile regression must run client-side. Two options:
  1. Port a small QR solver via SciPy-style LP. The check loss minimization $\min_\beta \sum \rho_\tau(y_i - x_i^\top \beta) + \lambda \|\beta\|_2^2$ admits an LP reformulation. Use `glpk.js` or roll a simple smoothed-Huber surrogate solved via gradient descent (works for $\tau \in (0, 1)$ with degree-3 polynomial features and $n \le 500$ in well under 100 ms).
  2. Pre-compute the QR fit at multiple heteroscedasticity values on a discrete grid and interpolate. Simpler but limits the slider granularity.
  
  **Recommendation:** option 1 with a small smoothed-check-loss + L-BFGS solver. Add `fitPredictQuantile(x, y, x_eval, tau)` to `nonparametric-ml.ts`.
- Pre-compute the synthetic data once per heteroscedasticity-slider value (debounced); recompute only the calibration step on $\alpha$ change.

**Anchor.** Embed after Definition 4, before Theorem 3's coverage inheritance.

### 4c. `APSForClassification.tsx` — Embedded after Definition 5 in §7

**Mathematical content.** APS prediction sets visualized on a 2D 3-class classification problem. Interactive controls show how set sizes adapt to local difficulty.

**Layout (3 panels).**
- **Left:** Classifier softmax — argmax color coded per class, alpha-modulated by max probability. Training points overlaid.
- **Middle:** APS region map. Each grid cell colored by predicted set size: green = `|set| = 1`, amber = `|set| = 2`, red = `|set| = 3`. Training points overlaid as small dots.
- **Right:** Bar chart of empirical set-size distribution on a held-out test set, plus per-set-size empirical coverage as a secondary axis. Reference line at $1 - \alpha$.

**Controls.**
- Slider: $\alpha \in [0.05, 0.30]$, default 0.10.
- Slider: class-overlap parameter $\sigma \in [0.5, 1.5]$ (Gaussian blob standard deviation; smaller = more separable, larger = more overlap), default 1.0.
- "Resample" button to redraw the synthetic data with a new seed at current $\sigma$.

**Implementation notes.**
- Logistic regression with 2D features (linear decision boundary). Implement client-side via L-BFGS or use a lightweight library. ~50 ms fit at $n = 600$.
- For the region map, precompute predictions on a $200 \times 200$ grid; the APS set-size assignment per cell is O(K) per cell and trivially fast.
- Use the deterministic APS variant (matches the notebook's Definition 5 — always non-empty).
- Add `apsScore`, `apsPredictionSet` to `nonparametric-ml.ts`.

**Anchor.** Embed after Definition 5 ("APS Score and Prediction Set, deterministic"), before the "Why APS, not top-k?" remark.

### 4d. `ExchangeabilityBreakdown.tsx` — Flavor 1 (Covariate Shift), embedded in §8

**Mathematical content.** When the test distribution $P_X^{\text{test}}$ differs from the training distribution $P_X^{\text{train}}$, exchangeability fails and split conformal can lose its marginal coverage. The Tibshirani-Barber-Candès-Ramdas 2019 importance-weighted variant restores it under known shift. This component shows the failure (naive) and the fix (weighted) side by side.

**Layout (2 panels).**
- **Top:** 1D regression on heteroscedastic data with adjustable covariate shift. Training points sampled from $X^{\text{train}} \sim N(\mu_{\text{tr}}, 1)$; test points sampled from $X^{\text{test}} \sim N(\mu_{\text{te}}, 1)$. The shift $\mu_{\text{te}} - \mu_{\text{tr}}$ is the slider parameter. Overlay both distributions as histograms above the scatter; show prediction bands from naive and weighted procedures.
- **Bottom:** Marginal coverage as a function of shift magnitude, computed across $T = 200$ trials per shift level, swept on a discrete grid of shift values. Two curves: naive (blue, degrades) and weighted (green, holds). Reference line at $1 - \alpha$.

**Controls.**
- Slider: covariate shift $\Delta = \mu_{\text{te}} - \mu_{\text{tr}} \in [0, 2]$, default 0.5.
- Slider: $\alpha \in [0.05, 0.20]$, default 0.10.
- Toggle: "Show naive" / "Show weighted" / "Show both" (default both).

**Implementation notes.**
- The bottom panel is precomputed at a discrete grid of shift values $\{0, 0.2, 0.4, \ldots, 2.0\}$ on mount; the slider position is a vertical marker on this fixed curve. Top panel is live on slider change.
- Importance weights $w(x) = p_{\text{te}}(x) / p_{\text{tr}}(x)$ are computed in closed form for the Gaussian-shift setting (no kernel density estimation needed).
- Add `weightedSplitConformal(x_train, y_train, x_cal, y_cal, x_test, alpha, weight_fn)` to `nonparametric-ml.ts`. The weighted variant takes a per-sample weight on each calibration point and on the test point and uses the weighted empirical quantile.

**Why Flavor 1 (covariate shift) over Flavor 2 (label shift) or Flavor 3 (concept drift).** Covariate shift has the cleanest mathematical statement (TBCR 2019 exact coverage under known $w$), and known importance weights make the visualization deterministic — no estimation noise to debug. Decision 2 locks this in.

**Anchor.** Embed in §8 immediately after Remark 3 ("Covariate Shift Connection"), serving as the bridge from the impossibility result back to actionable extensions.

### Component-shared concerns

- All four components hydrate via `client:visible`. None require WebGL or large-payload datasets.
- Use `useResizeObserver` for responsive sizing per the existing pattern.
- D3 scales scoped to component refs.
- Color palette: import from `viz/shared/colorScales.ts` — primary blue, accent green, secondary red/amber. No new color definitions.
- All four reference the same `nonparametric-ml.ts` shared module (§5).
- Static fallback PNGs (rendered from the notebook's figures) live in `public/images/topics/conformal-prediction/`. Use as `<noscript>` fallbacks and as the "preview" for the social-card meta tag.

---

## 5. Shared Module: `nonparametric-ml.ts`

This topic **introduces** the T4 shared module per strategic plan §6.2. Future T4 topics (`quantile-regression`, `rank-tests`, `extreme-value-theory`, `statistical-depth`, `prediction-intervals`) will extend it.

### File location

`src/components/viz/shared/nonparametric-ml.ts`

### Required exports

```typescript
// === Synthetic data generators ===
export function synthHeteroscedastic(n: number, rng: () => number): {
  x: Float64Array;
  y: Float64Array;
};

export function synth3Class(n: number, rng: () => number, sigma?: number): {
  X: Float64Array; // shape (n, 2)
  y: Int32Array;
};

export function synthSpike(
  n: number,
  rng: () => number,
  eps: number,
  sigmaOutside?: number,
  M?: number,
  xCenter?: number
): {
  x: Float64Array;
  y: Float64Array;
  inSpike: Uint8Array;
};

// === Base predictors ===
export function fitPredictRidge(
  xTrain: Float64Array,
  yTrain: Float64Array,
  xEval: Float64Array,
  lambda?: number
): Float64Array;

export function fitPredictQuantile(
  xTrain: Float64Array,
  yTrain: Float64Array,
  xEval: Float64Array,
  tau: number,
  lambda?: number
): Float64Array;

// === Conformal procedures ===
export interface ConformalInterval {
  lower: Float64Array;
  upper: Float64Array;
  qHat: number;
  calScores: Float64Array;
}

export function splitConformalInterval(
  xTrain: Float64Array,
  yTrain: Float64Array,
  xCal: Float64Array,
  yCal: Float64Array,
  xTest: Float64Array,
  alpha: number,
  fitPredictFn: (xt: Float64Array, yt: Float64Array, xe: Float64Array) => Float64Array
): ConformalInterval;

export function jackknifePlusInterval(
  x: Float64Array,
  y: Float64Array,
  xTest: Float64Array,
  alpha: number,
  fitPredictFn: (xt: Float64Array, yt: Float64Array, xe: Float64Array) => Float64Array
): { lower: Float64Array; upper: Float64Array };

export function cvPlusInterval(
  x: Float64Array,
  y: Float64Array,
  xTest: Float64Array,
  alpha: number,
  fitPredictFn: (xt: Float64Array, yt: Float64Array, xe: Float64Array) => Float64Array,
  K: number,
  rng: () => number
): { lower: Float64Array; upper: Float64Array };

export function cqrInterval(
  xTrain: Float64Array,
  yTrain: Float64Array,
  xCal: Float64Array,
  yCal: Float64Array,
  xTest: Float64Array,
  alpha: number
): { lower: Float64Array; upper: Float64Array; qHat: number };

export function weightedSplitConformal(
  xTrain: Float64Array,
  yTrain: Float64Array,
  xCal: Float64Array,
  yCal: Float64Array,
  xTest: Float64Array,
  alpha: number,
  weightFn: (x: number) => number,
  fitPredictFn: (xt: Float64Array, yt: Float64Array, xe: Float64Array) => Float64Array
): ConformalInterval;

// === APS for classification ===
export function apsScoreDeterministic(
  probs: Float64Array, // shape (n, K) row-major
  y: Int32Array,
  n: number,
  K: number
): Float64Array;

export function apsPredictionSetDeterministic(
  probs: Float64Array,
  threshold: number,
  n: number,
  K: number
): Uint8Array; // shape (n, K) row-major bool mask
```

### Verification tests

Before any viz components are built, verify the module against the notebook's printed numerical outputs:

1. **`splitConformalInterval` empirical coverage.** Generate $T = 4000$ trials on `synthHeteroscedastic` at $(\alpha, n_{\text{cal}}) = (0.10, 200)$. Mean coverage should fall in $[0.890, 0.915]$ (matches notebook's Cell 3 output).
2. **`jackknifePlusInterval` coverage.** $T = 300$ trials at $n = 60, \alpha = 0.10$. Mean coverage in $[0.85, 0.92]$ (notebook reports 0.883).
3. **`cqrInterval` width.** Single seeded run on `synthHeteroscedastic(800)` should produce mean width 3.26 ± 0.1 (notebook reports 3.263).
4. **`apsScoreDeterministic` and `apsPredictionSetDeterministic`.** Single seeded run with 3 classes, $n = 600$, $\alpha = 0.10$ should yield set-size distribution roughly $(530, 70, 0)$ — i.e., predominantly singletons with occasional doubletons (notebook reports `(532, 68, 0)`).
5. **`weightedSplitConformal` recovers naive when weights are uniform.** Sanity check.

These tests can live in `src/components/viz/shared/__tests__/nonparametric-ml.test.ts` if a test harness is set up; otherwise verify in a scratch component during development.

---

## 6. Curriculum Graph Updates

This brief assumes the curriculum-graph expansion sweep (a separate brief / PR) has registered all T4–T6 topics. If that PR has shipped, the only change required by *this* brief is:

### `src/data/curriculum-graph.json`

**Update existing node:** Set `conformal-prediction` node's `status` from `"planned"` to `"published"` and ensure its `url` field is `"/topics/conformal-prediction"`.

**Add edges:** Internal prereqs are already in place (`concentration-inequalities → conformal-prediction`, `pca-low-rank → conformal-prediction`). Add downstream forward edges to next-T4 topics (which exist as `planned` nodes per the assumption above):

```json
{ "source": "conformal-prediction", "target": "quantile-regression" }
{ "source": "conformal-prediction", "target": "prediction-intervals" }
{ "source": "conformal-prediction", "target": "rank-tests" }
```

### `src/data/curriculum.ts`

The `nonparametric-ml` track exists (per the assumption). Move `"conformal-prediction"` from the `planned` array to the `published` array in the `nonparametric-ml` track.

> ⚠️ **If the curriculum-graph expansion PR has NOT shipped first.** This brief alone needs the following minimal additions to keep the graph valid:
> - Add `nonparametric-ml` as a domain key in the legend.
> - Add the `conformal-prediction` node with `domain: "nonparametric-ml"`, `status: "published"`, `url: "/topics/conformal-prediction"`.
> - Do NOT add the three downstream forward edges above; they would point at unregistered nodes. Leave forward registration to the expansion PR.
> - Add the cross-site rendering opacity/dashed-border logic per Decision 4 to the curriculum-graph component (the cross-site nodes for `formalstatistics/order-statistics-and-quantiles` and `formalstatistics/empirical-processes` need to render distinguishably from internal nodes when shown on this topic's prereq view).

---

## 7. Cross-References (in-page)

### Outbound (from `conformal-prediction.mdx`)

**Backward references — these pages exist, use live links:**

- `[Concentration Inequalities](/topics/concentration-inequalities)` — referenced in §1 (alternative to rank-symmetry for asymptotic coverage), §3 (the "concentration is the asymptotic alternative" frame).
- `[PCA & Low-Rank Approximation](/topics/pca-low-rank)` — referenced in §10 Connections (calibration scores as low-dimensional summaries; LOO updates).
- `[Measure-Theoretic Probability](/topics/measure-theoretic-probability)` — referenced in §1 (exchangeability as a measure-theoretic invariance).
- `[PAC Learning](/topics/pac-learning)` — referenced in §10 Connections (PAC = predictability guarantees; conformal = coverage guarantees; orthogonal axes).

**Cross-site references to formalstatistics — full-URL hyperlinks per CLAUDE.md:**

- `[Topic 29 — Order Statistics & Quantiles](https://formalstatistics.com/topics/order-statistics-and-quantiles)` — referenced in §1 (the rank-symmetry argument is identical), §3 proof (paired-order-statistic CIs as the special-parameter case).
- `[Topic 32 — Empirical Processes](https://formalstatistics.com/topics/empirical-processes)` — referenced in §3 Remark and §10 Connections (uniform-convergence rates as the asymptotic alternative).

> **Note:** Body-prose external links use the new `<ExternalLink>` component (see §11). Frontmatter `formalstatisticsPrereqs` declares the same edges for the audit.

**Forward references to planned formalML topics — plain text per CLAUDE.md (do NOT hyperlink to unbuilt topics):**

- **Quantile Regression** *(coming soon)* — "The base learner inside CQR. The full topic covers QR estimation theory, asymptotic distribution of $\hat\beta(\tau)$, and the broader use of quantile regression beyond conformal."
- **Statistical Depth** *(coming soon)* — "Depth-based prediction regions are the geometric alternative to score-based conformal. For symmetric distributions, the two reproduce each other; for asymmetric ones, they diverge in instructive ways."
- **Prediction Intervals** *(coming soon)* — "The umbrella topic covering frequentist + Bayesian + conformal + QR-based PI constructions side by side. This topic positions conformal within the broader PI ecosystem."

⚠️ **Do NOT hyperlink to unbuilt topics.** Use the plain-text pattern `**Topic Name** *(coming soon)*`.

### Inbound (update existing pages)

- **`concentration-inequalities.mdx`:** In §10 Connections (or wherever distribution-free methods are mentioned), add a forward link to `[Conformal Prediction](/topics/conformal-prediction)` describing the contrast: "Where concentration inequalities give asymptotic distribution-free bounds (e.g., DKW, Hoeffding), conformal prediction achieves *exact* finite-sample coverage via rank symmetry under exchangeability — a different mathematical machinery for the same goal." Only add if there's a natural insertion point. Do not force.
- **`pca-low-rank.mdx`:** Likely already has a Connections section with forward references to `gradient-descent` and others. If a sentence about "low-rank summaries of prediction error" or "leave-one-out updates" exists or fits, add the link. Otherwise leave alone.
- **`measure-theoretic-probability.mdx`:** If it discusses exchangeability (which it should), add a forward link to `[Conformal Prediction](/topics/conformal-prediction)` as the canonical ML application of exchangeability beyond de Finetti's theorem.
- **`pac-learning.mdx`:** In §10 Connections, add a forward link to `[Conformal Prediction](/topics/conformal-prediction)` framing the orthogonality: "PAC bounds give learning guarantees; conformal gives coverage guarantees. The two operate on orthogonal axes." Only add if the existing connections structure accommodates it.

---

## 8. `/paths` Page Update

### Create new track section

The **Nonparametric & Distribution-Free** track does NOT yet exist on `/paths`. Create a new track section.

**Track header:**
- Anchor: `#nonparametric-ml`
- Title: "Nonparametric & Distribution-Free"
- Description: "Distribution-free prediction sets, rank-based testing, quantile regression, and statistical depth — methods that work under minimal distributional assumptions."

**Track topics (per strategic plan §3.4):**

| Topic | Status | Difficulty | Badge |
|---|---|---|---|
| **Conformal Prediction** | **Published** (linked to `/topics/conformal-prediction`) | **Intermediate** | **Start here** |
| Quantile Regression | Planned (gray, unlinked) | Intermediate | — |
| Rank Tests & Permutation Inference | Planned (gray, unlinked) | Intermediate | — |
| Extreme Value Theory | Planned (gray, unlinked) | Advanced | — |
| Statistical Depth | Planned (gray, unlinked) | Intermediate | — |
| Prediction Intervals | Planned (gray, unlinked) | Intermediate | — |

> ⚠️ **First topic in a new track.** The `/paths` page infrastructure for this track must be created from scratch. Follow the exact pattern of existing track sections (Topology & TDA, Linear Algebra, Graph Theory, etc.) for layout, styling, and status badges.

### Cross-site prereq rendering on `/paths` (per Decision 4, Option C)

Cross-site prereq nodes (`formalstatistics/order-statistics-and-quantiles`, `formalstatistics/empirical-processes`) need to render visually distinct from internal nodes:

- **Reduced opacity:** 0.6 (vs. 1.0 for internal nodes).
- **Dashed border:** 1px dashed (vs. 1px solid for internal nodes).
- **Tooltip text suffix:** "(formalstatistics)" appended to the node label on hover.

Site-icon badges (a small formalstatistics or formalcalculus glyph in the corner of each cross-site node) are **deferred** to the curriculum-graph-expansion PR per Decision 4. This brief intentionally does not specify them; they're a follow-up enhancement.

---

## 9. Cross-Site Prerequisites

Per CLAUDE.md and strategic plan §11.1.

### formalstatistics

- [Topic 29 — Order Statistics & Quantiles](https://formalstatistics.com/topics/order-statistics-and-quantiles) — Section §29.7's distribution-free CI for the quantile $\xi_p$ via paired order statistics uses the same exchangeability + rank-symmetry argument that drives Theorem 1 here. §29.10 Rem 21 explicitly motivates Vovk-Gammerman-Shafer's distribution-free prediction as the generalization from a fixed parameter to a random variable. Referenced in §1 (motivation) and §3 (proof structure).
- [Topic 32 — Empirical Processes](https://formalstatistics.com/topics/empirical-processes) — Topic 32's uniform-convergence theory (DKW, VC dimension, Glivenko-Cantelli) is the asymptotic alternative to conformal's exact finite-sample coverage. Conformal achieves coverage without a uniform-convergence rate — at the price of marginal-only validity. Referenced in §1 (frame the trade-off) and §10 (connections).

### formalcalculus

None. This topic has no formalcalculus prerequisites — explicit acknowledgment per strategic plan §11.1. The exchangeability + rank-symmetry argument is combinatorial; calculus prereqs from formalcalculus are not required.

---

## 10. Reciprocity Sweep at Ship Time

When this topic ships, the following sister-site MDX files need a reciprocal `formalstatisticsConnections` entry that points back at `formalml/conformal-prediction`. Per `docs/plans/deferred-reciprocals.md` heading "When `formalml/conformal-prediction` ships":

1. **`formalstatistics/src/content/topics/order-statistics-and-quantiles.mdx`** — currently declares `formalmlConnections → conformal-prediction` with prose "_Vovk–Gammerman–Shafer distribution-free prediction. §29.10 Rem 21 motivates via §29.7's quantile-CI perspective; forthcoming on formalml._" After this topic ships, no new entry is needed on the formalstatistics side — the existing `formalmlConnections` entry remains; the audit's "Reciprocated" count will increment automatically once the formalML side adds its `formalstatisticsPrereqs` entry (which is in the frontmatter §2 above).

2. **`formalstatistics/src/content/topics/empirical-processes.mdx`** — currently declares `formalmlConnections → conformal-prediction` with prose "_Conformal prediction constructs distribution-free CIs via exchangeability rather than empirical-process asymptotics; the two meet at Topic 29 §29.5's DKW inequality and Topic 32's uniform bands._" Same workflow: existing entry remains, audit increments on ship.

**Verification command (run from any of the three repos with sibling paths configured):**

```bash
pnpm audit:cross-site
```

Expected delta after ship:
- `Reciprocated` count: +2 (the two formalstatistics topics above).
- `Missing reciprocals` count: 0 new entries.
- Two existing entries under "When `formalml/conformal-prediction` ships" in `deferred-reciprocals.md` should disappear.

### Sister-site cleanup (separate PR, out of scope here)

- **`cross-validation` slug references on formalstatistics.** Any sister-site MDX pointing at `formalml/cross-validation` (e.g., `formalstatistics/regularization-and-penalized-estimation` if it does — check the audit) should be updated to point at `formalml/conformal-prediction#cross-validation` per the §3.8 named-section policy. This is a sister-site PR that can ship anytime after this topic, not blocked by formalML.

---

## 11. `<ExternalLink>` Component Port

This is the first formalML topic with cross-site `*Prereqs` populated, so the `<ExternalLink>` component must be ported from formalstatistics. Per strategic plan §5.5 (cross-site reference workflow):

- **Source:** `formalstatistics/src/components/ui/ExternalLink.astro` — already exists in the formalstatistics repo.
- **Target:** `formalml/src/components/ui/ExternalLink.astro` — port verbatim.
- **Interface:** `{ href: string; site: "formalstatistics" | "formalcalculus"; topic: string }`.
- **Behavior:** Renders an external link with a small site-attribution glyph (e.g., a stats icon for formalstatistics) and a target=`_blank` rel=`noopener` for cross-site navigation.
- **Usage in conformal-prediction.mdx:** Use for inline-prose mentions of Topic 29 and Topic 32 throughout §1, §3, and §10. The frontmatter `formalstatisticsPrereqs` declares the edges for the audit; `<ExternalLink>` is the human-facing rendering in body prose.

> ⚠️ **One-time port; reuse forever.** Subsequent T4–T6 topics will reuse this component without modification. Verify the port works end-to-end on this topic before merging.

---

## 12. Notebook Reference

**Path:** `notebooks/conformal-prediction/01_conformal_prediction.ipynb`

**Structure:** 11 cells (1 setup + 9 mathematical body cells + 1 connections markdown).

**Status:** ✅ Verified — runs end-to-end in 21.5s on the development machine (well under the 60s CPU-only constraint per strategic plan §8.3). All numerical outputs match theoretical bounds within Monte Carlo error. All 9 figures generated cleanly with no matplotlib warnings.

**Figures produced (saved to notebook output cells; static PNGs to be exported to `public/images/topics/conformal-prediction/`):**

| File | Source cell | Use in MDX |
|---|---|---|
| `conformal_ci_vs_prediction_set.png` | Cell 1 | §1 inline figure (CI vs prediction set) |
| `conformal_split_construction.png` | Cell 2 | §2 inline figure (split conformal one-run demonstration) |
| `conformal_marginal_coverage.png` | Cell 3 | §3 inline figure (4-panel verification) — also social-card preview |
| `conformal_split_vs_full.png` | Cell 4 | §4 inline figure (efficiency comparison) |
| `conformal_jackknife_cv_plus.png` | Cell 5 | §5 inline figure (LOO + coverage + width) |
| `conformal_cqr.png` | Cell 6 | §6 inline figure (CQR vs naive) — fallback for `CQRExplorer` |
| `conformal_aps.png` | Cell 7 | §7 inline figure (APS region map) — fallback for `APSForClassification` |
| `conformal_conditional_impossibility.png` | Cell 8 | §8 inline figure (3-panel ε sweep) |
| `conformal_blackbox_applications.png` | Cell 9 | §9 inline figure (3 base predictors) |

The `ConformalPredictionExplorer` and `ExchangeabilityBreakdown` components have no direct notebook-figure analog — they're computed live in-browser. Their static fallbacks should be screenshots of the rendered components at default settings, captured during development.

---

## 13. Build Order

1. **Pre-flight: verify the curriculum-graph expansion PR has shipped.** If yes, proceed. If no, follow the §6 minimal-registration path and flag for reviewer that the expansion PR should ship next.
2. **Port `<ExternalLink>` component** from formalstatistics to `src/components/ui/ExternalLink.astro` (§11). Verify with a one-line smoke test.
3. **Create `src/components/viz/shared/nonparametric-ml.ts`** (§5) with all required exports. Run the verification tests against the notebook's printed numerical outputs before any viz components reference it.
4. **Add the notebook** to `notebooks/conformal-prediction/01_conformal_prediction.ipynb` (Jonathan supplies; do not regenerate).
5. **Export static figures** from the notebook to `public/images/topics/conformal-prediction/`. Names per §12.
6. **Create `src/content/topics/conformal-prediction.mdx`** with full frontmatter (§2) and all markdown/LaTeX content (§3). Use `TheoremBlock` for all 14 numbered formal elements. Use `<ExternalLink>` for inline cross-site mentions. No interactive components yet — embed `<noscript>`-wrapped static figure imports as placeholders at the four viz-anchor positions.
7. **Build `ConformalPredictionExplorer.tsx`** (§4a). Anchor in §3 after Theorem 1's proof. Verify against notebook Cell 3's coverage table.
8. **Build `CQRExplorer.tsx`** (§4b). Anchor in §6 after Definition 4. Verify against notebook Cell 6's printed widths.
9. **Build `APSForClassification.tsx`** (§4c). Anchor in §7 after Definition 5. Verify against notebook Cell 7's set-size distribution.
10. **Build `ExchangeabilityBreakdown.tsx`** (§4d). Anchor in §8 after Remark 3 (Covariate Shift Connection). Verify the no-shift baseline matches naive split conformal coverage.
11. **Replace placeholders.** Swap each viz-anchor's `<noscript>` static-figure import for the live React component import.
12. **Add `<NamedSection id="cross-validation">`** wrapping §3.8. Confirm `/topics/conformal-prediction#cross-validation` resolves correctly.
13. **Update curriculum graph data** (§6) — node status, downstream edges (only if expansion PR shipped), domain key.
14. **Update `/paths` page** (§8) — create new track section, add cross-site-prereq rendering logic.
15. **Update inbound references** (§7 inbound list) — `concentration-inequalities`, `pca-low-rank`, `measure-theoretic-probability`, `pac-learning`. Only natural insertion points.
16. **Run testing checklist** (§14).
17. **Run `pnpm audit:cross-site`** from any sibling-configured repo. Verify both deferred-reciprocals entries discharge and no new "Missing reciprocals" appear.
18. **Commit and deploy.**

---

## 14. Testing Checklist

### Build / lint

- [ ] `pnpm build` — no errors, no MDX parse failures.
- [ ] `pnpm dev` — topic renders at `localhost:4321/topics/conformal-prediction`.
- [ ] No TypeScript errors in any component or shared module.
- [ ] No console warnings in the browser dev tools when viewing the topic.

### Math / proof rendering

- [ ] All 5 Definitions, 4 Theorems, 6 Remarks render via `TheoremBlock` with correct numbering.
- [ ] Theorem 1's proof renders all rank-symmetry steps (~12 paragraph proof).
- [ ] Theorem 2's proof renders the BCRT 2021 tournament/comparison-graph argument.
- [ ] Theorem 4's proof renders the constructive Foygel-Barber spike-construction argument with the $\Phi^{-1}(1-\alpha/2) \cdot \sqrt{\sigma_0^2 + M^2}$ lower-bound expression intact.
- [ ] All LaTeX symbols from the §3 list render correctly (spot-check on dev server).
- [ ] No naked symbols introduced before their plain-English gloss.

### Interactive components

- [ ] `ConformalPredictionExplorer`: sliders responsive; coverage histogram concentrates around the band as $T$ grows; works at $n_{\text{cal}} = 20$ and $n_{\text{cal}} = 1000$.
- [ ] `CQRExplorer`: naive vs CQR overlay renders; conditional coverage curve is visibly flatter for CQR.
- [ ] `APSForClassification`: set-size region map shows clear amber/red zones at decision boundaries; bar-chart coverage matches notebook expectations.
- [ ] `ExchangeabilityBreakdown`: at zero shift, naive and weighted curves coincide near $1-\alpha$; at large shift, naive degrades visibly while weighted holds.
- [ ] All four components render fallback static PNGs when JavaScript is disabled (`<noscript>`).
- [ ] All four components are responsive on mobile (≤ 375px viewport).

### Cross-references

- [ ] All four backward-reference live links resolve.
- [ ] Two `<ExternalLink>`-rendered cross-site mentions render with the formalstatistics glyph and `target="_blank"`.
- [ ] All three forward-reference plain-text mentions render as plain text (not links).
- [ ] `/topics/conformal-prediction#cross-validation` scrolls to the §3.8 anchor.

### Curriculum graph + `/paths`

- [ ] `/paths` shows new "Nonparametric & Distribution-Free" track.
- [ ] Conformal Prediction node is listed as Published with "Start here" badge.
- [ ] Five other T4 topics listed as Planned (gray, unlinked).
- [ ] Cross-site prereq nodes (`order-statistics-and-quantiles`, `empirical-processes`) render with reduced opacity 0.6 + dashed border.
- [ ] `formalstatistics`-suffix tooltip appears on hover over cross-site nodes.

### Audit

- [ ] `pnpm audit:cross-site` reports 0 errors specific to this topic.
- [ ] Both deferred-reciprocals entries for `conformal-prediction` retire.
- [ ] No new "Missing reciprocals" appear.

### Performance

- [ ] Topic page first-paint under 2s on a 4G connection.
- [ ] Each interactive component hydrates within 500ms after `client:visible` triggers.
- [ ] No layout shift after hydration.

---

## Open Questions for Implementation

1. **`<NamedSection>` component existence.** Is `<NamedSection>` already in `src/components/ui/`? If not, port from formalstatistics or write inline (~10 lines of Astro). Confirm before §3.8 is built.
2. **Quantile regression solver choice.** §4b recommends a smoothed-check-loss + L-BFGS in-browser solver. If implementation effort is significant, the fallback (precompute on a discrete heteroscedasticity grid + interpolate) is acceptable but limits slider granularity. Pick during implementation.
3. **`ExchangeabilityBreakdown` sweep granularity.** §4d specifies a discrete grid `{0, 0.2, ..., 2.0}` for the bottom-panel pre-computed coverage curve. If 11 grid points is too coarse, refine to 21 (`{0, 0.1, ..., 2.0}`) — costs ~2× compute on mount.

---

## Appendix A: Key Differences from Prior Briefs

1. **First topic in a new track.** Like `graph-laplacians` (T-Graph-Theory opener), this brief creates new infrastructure: track section on `/paths`, new shared module (`nonparametric-ml.ts`), new domain key in the curriculum graph.
2. **First topic with cross-site prereqs.** Triggers the one-time `<ExternalLink>` component port from formalstatistics. All future cross-site topics reuse it.
3. **Named section as cross-site routability target.** §3.8 (CV+) carries the `<NamedSection id="cross-validation">` anchor that any sister-site `formalml/cross-validation` pointer should redirect to. New pattern in formalML, established in formalstatistics precedent.
4. **Theorem density: 4 full proofs.** This is the highest-rigor topic shipped to date — Theorem 1 (rank symmetry), Theorem 2 (BCRT 2021 tournament argument), Theorem 3 (CQR coverage inheritance, short), and Theorem 4 (Foygel-Barber constructive impossibility). Combined word count for proofs alone exceeds 4500 words. Expect ~14-15K words total.
5. **Four viz components.** Matching `graph-laplacians` (which also had four). The fourth (`ExchangeabilityBreakdown`) is unusual — it visualizes assumption *failure* and the importance-weighted fix, not a positive procedure. Treat as a "boundary-of-the-theory" visualization rather than a pedagogical demo of a single algorithm.
6. **`/paths` cross-site rendering Decision 4.** First topic to surface cross-site-prereq nodes on the curriculum graph. The Option C reduced-opacity + dashed-border treatment is intentionally minimal; site-icon badges are deferred.

---

## Appendix B: T4 Track Status After This Brief

| Order | Topic | Difficulty | Prerequisites | Status |
|---|---|---|---|---|
| 1 | **Conformal Prediction** (this brief) | Intermediate | concentration-inequalities, pca-low-rank, formalstatistics/order-statistics-and-quantiles, formalstatistics/empirical-processes | 🚧 Ready for implementation |
| 2 | Quantile Regression | Intermediate | conformal-prediction, formalstatistics/linear-regression | Planned |
| 3 | Rank Tests & Permutation Inference | Intermediate | conformal-prediction, formalstatistics/hypothesis-testing | Planned |
| 4 | Statistical Depth | Intermediate | conformal-prediction, formalstatistics/order-statistics-and-quantiles | Planned |
| 5 | Prediction Intervals | Intermediate | conformal-prediction, quantile-regression | Planned |
| 6 | Extreme Value Theory | Advanced | concentration-inequalities, formalstatistics/order-statistics-and-quantiles | Planned |

DAG edges after implementation:
- `concentration-inequalities → conformal-prediction`
- `pca-low-rank → conformal-prediction`
- `conformal-prediction → quantile-regression`
- `conformal-prediction → rank-tests`
- `conformal-prediction → statistical-depth`
- `conformal-prediction → prediction-intervals`
- `quantile-regression → prediction-intervals`

Cross-site edges:
- `formalstatistics/order-statistics-and-quantiles → conformal-prediction` (formalmlConnections, reciprocated)
- `formalstatistics/empirical-processes → conformal-prediction` (formalmlConnections, reciprocated)

---

*Brief version: v1 | Last updated: 2026-04-25 | Author: Jonathan Rocha*
*Reference notebook: `notebooks/conformal-prediction/01_conformal_prediction.ipynb`*
*Reference doc: `docs/plans/formalml-handoff-reference.md`*
*Strategic plan: `docs/plans/formalml-consolidated-strategic-planning-document.md`*
