# Claude Code Handoff Brief: Rate-Distortion Theory

**Project:** formalML — [formalml.com](https://www.formalml.com)  
**Repo:** `github.com/jonx0037/formalML`  
**Stack:** Astro 6 · React 19 · MDX · Tailwind CSS 4 · D3.js 7 · KaTeX · Vercel  
**Package Manager:** pnpm  
**Status:** Ready for implementation  
**Reference Notebook:** `notebooks/rate-distortion/03_rate_distortion.ipynb`  
**Reference Doc:** `docs/plans/formalml-handoff-reference.md`  
**Sibling Briefs:** `formalml-shannon-entropy-handoff-brief.md` (Topic 1), `formalml-kl-divergence-handoff-brief.md` (Topic 2). Follow their implementation patterns for component structure, shared utility usage, and curriculum graph updates.

---

## 1. Objective

Add a new topic page **"Rate-Distortion Theory"** to the **Information Theory** track on formalml.com. This is the **third topic** in the track:

1. The Information Theory track already exists (created by the Shannon Entropy topic, expanded by KL Divergence). No new track registration needed.
2. This topic depends on **`shannon-entropy`** (within-track prerequisite) and **`convex-analysis`** (cross-track prerequisite).
3. Downstream dependencies: none currently planned. (MDL depends on `kl-divergence`, not on this topic — see track DAG.)
4. The topic has **additional cross-track backward connections** to `lagrangian-duality` and `kl-divergence` — these are references, not prerequisites.
5. Difficulty: **Intermediate** — algebraic derivations follow geometric intuition. Match the editorial voice of SVD, PCA, Persistent Homology, Gradient Descent, and KL Divergence topics.

The companion Jupyter Notebook (`03_rate_distortion.ipynb`) contains the full mathematical content, proofs, code, and figures — use it as the authoritative content source for all definitions, theorems, examples, and code blocks.

**Content scope (comprehensive):**

- Distortion measures: distortion functions $d(x, \hat{x})$, expected distortion, Hamming distortion (discrete), squared error distortion (continuous), properties
- Rate-distortion function $R(D)$: definition as mutual information minimization over test channels, properties (non-negativity, convexity, monotonicity, boundary values), achievable vs unachievable regions
- Rate-distortion theorem: Shannon's lossy source coding theorem — achievability via random coding / covering lemma, converse via data processing inequality, operational interpretation
- Closed-form solutions: binary source with Hamming distortion ($R(D) = H_b(p) - H_b(D)$), Gaussian source with squared error ($R(D) = \frac{1}{2}\log_2(\sigma^2/D)$), parametric form via slope parameter, Shannon lower bound
- Blahut–Arimoto algorithm: alternating minimization for numerical computation of $R(D)$, convergence, connection to EM and alternating projections
- The information bottleneck: IB Lagrangian ($\min I(X;T) - \beta I(T;Y)$), equivalence to rate-distortion with KL distortion, IB curve properties, connections to deep learning (VIB, $\beta$-VAE)
- Computational notes: VAE loss as rate-distortion Lagrangian, $\beta$-VAE operating points, neural compression, successive refinement
- Connections to ML: autoencoders, variational inference, quantization, learned image/video compression

---

## 2. MDX File

### Location

```
src/content/topics/rate-distortion.mdx
```

The entry `id` will be `rate-distortion` (derived from filename by the `glob()` loader in `src/content.config.ts`). The dynamic route at `src/pages/topics/[...slug].astro` resolves this to `/topics/rate-distortion`.

### Frontmatter

Must conform to the schema in `src/content.config.ts`. Here is the complete frontmatter:

```yaml
---
title: "Rate-Distortion Theory"
subtitle: "The fundamental limits of lossy compression — how many bits per symbol when we tolerate distortion?"
status: "published"
difficulty: "intermediate"
prerequisites:
  - "shannon-entropy"
  - "convex-analysis"
tags:
  - "information-theory"
  - "rate-distortion"
  - "lossy-compression"
  - "information-bottleneck"
  - "blahut-arimoto"
  - "variational-autoencoder"
domain: "information-theory"
videoId: null
notebookPath: "notebooks/rate-distortion/03_rate_distortion.ipynb"
githubUrl: "https://github.com/jonx0037/formalML/blob/main/src/content/topics/rate-distortion.mdx"
datePublished: 2026-03-29
estimatedReadTime: 45
abstract: "Rate-distortion theory answers the fundamental question of lossy compression: how many bits per source symbol are necessary and sufficient when we tolerate an average distortion of at most D? The rate-distortion function R(D) — defined as the minimum mutual information I(X; X̂) over all test channels satisfying the distortion constraint — is convex, non-increasing, and equals H(X) at D = 0, recovering Shannon's lossless source coding limit. We derive closed-form solutions for the binary source with Hamming distortion (R(D) = H_b(p) - H_b(D)) and the Gaussian source with squared error (R(D) = (1/2) log(σ²/D)), prove Shannon's rate-distortion theorem establishing R(D) as the exact achievability boundary, and develop the Blahut–Arimoto algorithm for numerical computation via alternating minimization. The information bottleneck method extends rate-distortion to compression with relevance: minimizing I(X; T) while preserving I(T; Y), unifying lossy compression with representation learning. Applications to machine learning include the VAE loss as a rate-distortion Lagrangian, β-VAE as rate-distortion trade-off, and neural image compression as learned R(D)-optimal coding."
connections:
  - topic: "shannon-entropy"
    relationship: "R(D) = min I(X; X̂) — the rate-distortion function minimizes mutual information. At D = 0, R(0) = H(X) recovers the lossless source coding limit established by the source coding theorem."
  - topic: "kl-divergence"
    relationship: "The information bottleneck distortion measure is d_IB(x, t) = D_KL(p(y|x) || p(y|t)), and the VAE rate term is D_KL(q(z|x) || p(z)). Both bridge rate-distortion theory to modern ML."
  - topic: "convex-analysis"
    relationship: "R(D) is convex in D — the rate-distortion optimization is a convex program. The Blahut–Arimoto algorithm exploits convexity via alternating minimization, and Lagrangian duality gives the parametric form."
  - topic: "lagrangian-duality"
    relationship: "The slope parameter s in the parametric form of R(D) is the Lagrange multiplier for the distortion constraint. Strong duality holds because the optimization is convex. The KKT conditions yield the optimal test channel structure."
references:
  - type: "book"
    title: "Elements of Information Theory"
    authors: "Cover & Thomas"
    year: 2006
    note: "Chapters 10–13 — rate-distortion theory, closed-form solutions, Blahut–Arimoto algorithm"
  - type: "book"
    title: "Rate Distortion Theory: A Mathematical Basis for Data Compression"
    authors: "Berger"
    year: 1971
    note: "The classical monograph on rate-distortion theory"
  - type: "paper"
    title: "Computation of Channel Capacity and Rate-Distortion Functions"
    authors: "Blahut"
    year: 1972
    url: "https://doi.org/10.1109/TIT.1972.1054855"
    note: "The original Blahut–Arimoto algorithm for computing R(D) via alternating minimization"
  - type: "paper"
    title: "The Information Bottleneck Method"
    authors: "Tishby, Pereira & Bialek"
    year: 1999
    url: "https://arxiv.org/abs/physics/0004057"
    note: "Introduction of the information bottleneck — rate-distortion with relevance"
  - type: "paper"
    title: "Deep Variational Information Bottleneck"
    authors: "Alemi, Fischer, Dillon & Murphy"
    year: 2017
    url: "https://arxiv.org/abs/1612.00410"
    note: "Connects the information bottleneck to deep learning via variational bounds"
  - type: "paper"
    title: "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"
    authors: "Higgins, Matthey, Pal, Burgess, Glorot, Botvinick, Mohamed & Lerchner"
    year: 2017
    note: "β-VAE as rate-distortion trade-off — tuning β traces the R(D) curve"
---
```

### Content Sections

The MDX body should have the following sections. Pull all mathematical content from the notebook. Each section header maps directly to a notebook section.

| # | MDX Section | Notebook § | Key Content |
|---|---|---|---|
| 1 | Overview & Motivation | §0 (Overview) | Why lossy compression for ML; lossless vs lossy; MP3/JPEG as motivation; what the topic covers |
| 2 | Distortion Measures | §1 | Definition of distortion function $d(x, \hat{x})$; expected distortion; Hamming distortion (discrete); squared error distortion (continuous); distortion matrix visualization |
| 3 | The Rate-Distortion Function | §2 | Definition of $R(D)$ as $\min I(X; \hat{X})$ over test channels; properties (non-negativity, convexity, monotonicity, boundary values $R(0) = H(X)$ and $R(D_{\max}) = 0$); achievable vs unachievable regions |
| 4 | The Rate-Distortion Theorem | §3 | Shannon's lossy source coding theorem (achievability via covering lemma, converse via DPI); operational interpretation; block diagram of lossy compression system |
| 5 | Closed-Form Solutions | §4 | Binary source with Hamming distortion (Theorem 1); Gaussian source with squared error (Theorem 2); parametric form via slope parameter; Shannon lower bound |
| 6 | The Blahut–Arimoto Algorithm | §5 | Alternating minimization (Theorem 5); convergence; step-by-step visualization; connection to EM |
| 7 | The Information Bottleneck | §6 | IB Lagrangian; equivalence to rate-distortion with KL distortion (Proposition 3); IB curve properties (Proposition 4); connection to deep learning (VIB, β-VAE) |
| 8 | Computational Notes | §7 | VAE loss as rate-distortion; β-VAE trade-off; successive refinement; Python implementations; neural compression libraries |
| 9 | Connections & Further Reading | §8 | Connection table (Shannon entropy, KL divergence, convex analysis, Lagrangian duality, MDL); references; notation table |

---

## 3. TheoremBlock Usage

Use the existing `TheoremBlock.astro` component for all numbered formal elements. The notebook contains:

| Type | # | Name | TheoremBlock `type` |
|---|---|---|---|
| Definition | 1 | Distortion function | `definition` |
| Definition | 2 | Hamming distortion | `definition` |
| Definition | 3 | Squared error distortion | `definition` |
| Definition | 4 | Rate-distortion function | `definition` |
| Definition | 5 | Information bottleneck | `definition` |
| Proposition | 1 | Properties of $R(D)$ | `proposition` (with `proof` block — five properties, proofs of convexity and monotonicity) |
| Proposition | 2 | Shannon lower bound | `proposition` (with `proof` block — $R(D) \geq h(X) - \frac{1}{2}\log_2(2\pi e D)$) |
| Proposition | 3 | IB as rate-distortion | `proposition` (with `proof` block — KL distortion equivalence) |
| Proposition | 4 | IB curve properties | `proposition` (four properties, proof sketches) |
| Proposition | 5 | DPI and rate-distortion | `proposition` (with `proof` block — post-processing increases distortion) |
| Proposition | 6 | Successive refinement | `proposition` (stated — Equitz & Cover condition) |
| Theorem | 1 | Rate-distortion for binary source | `theorem` (with `proof` block — $R(D) = H_b(p) - H_b(D)$) |
| Theorem | 2 | Rate-distortion for Gaussian source | `theorem` (with `proof` block — $R(D) = \frac{1}{2}\log_2(\sigma^2/D)$) |
| Theorem | 3 | Shannon's rate-distortion theorem | `theorem` (with `proof` block — achievability sketch via covering lemma, converse sketch via DPI) |
| Theorem | 4 | Parametric form of $R(D)$ | `theorem` (stated — parametric form via slope $s$) |
| Theorem | 5 | Blahut–Arimoto algorithm | `theorem` (with `proof` block sketch — alternating minimization converges to global optimum) |
| Remark | — | VAE loss as rate-distortion | `remark` (ELBO = reconstruction + rate) |
| Remark | — | β-VAE and trade-off | `remark` (β traces the R(D) curve) |

**LaTeX symbols to verify render correctly:**

- `R(D) = \min_{p(\hat{x}|x):\, \mathbb{E}[d(X,\hat{X})] \leq D} I(X; \hat{X})` — Rate-distortion function
- `d(x, \hat{x})` — Distortion function
- `d_H(x, \hat{x}) = \mathbb{1}[x \neq \hat{x}]` — Hamming distortion
- `d_{SE}(x, \hat{x}) = (x - \hat{x})^2` — Squared error
- `R(D) = H_b(p) - H_b(D)` — Binary source R(D)
- `R(D) = \frac{1}{2} \log_2 \frac{\sigma^2}{D}` — Gaussian source R(D)
- `R(0) = H(X)`, `R(D_{\max}) = 0` — Boundary values
- `p_s(\hat{x}|x) \propto q(\hat{x}) \exp(s \, d(x, \hat{x}))` — Blahut–Arimoto test channel
- `\mathcal{L}_{IB} = I(X; T) - \beta \, I(T; Y)` — IB Lagrangian
- `D_{KL}(q(z|x) \| p(z))` — VAE rate term
- `\mathbb{E}_{q(z|x)}[-\log p(x|z)]` — VAE distortion term

---

## 4. Interactive Visualizations

New React components in `src/components/viz/`. All use `client:visible` hydration. Follow patterns from the Shannon Entropy and KL Divergence topic components.

### 4a. RateDistortionExplorer (Flagship)

**File:** `src/components/viz/RateDistortionExplorer.tsx`

An interactive explorer of the rate-distortion function $R(D)$, showing how source distribution and distortion measure determine the achievable compression limit:

- **Left panel:** The $R(D)$ curve for the selected source. The $x$-axis is distortion $D$, the $y$-axis is rate $R$ (bits/symbol). The achievable region (above the curve) is lightly shaded green; the unachievable region (below) is lightly shaded red. Boundary annotations: $R(0) = H(X)$ at the left endpoint, $R(D_{\max}) = 0$ at the right.

- **Right panel:** A tangent line at a user-selected point on the curve, showing the slope $s$ (the Lagrange multiplier). The operating point is a draggable dot on the curve. Numerical readouts: current $(D, R)$, slope $s$, and the Lagrangian value $L = R + s D$.

- **Controls:**
  - **Source selector** (dropdown): "Binary uniform ($p=0.5$)", "Binary ($p=0.3$)", "Binary ($p=0.1$)", "Gaussian ($\sigma^2=1$)", "Gaussian ($\sigma^2=2$)".
  - **Draggable operating point** on the curve (click-and-drag along the curve).

**Interaction:** Changing the source type redraws the $R(D)$ curve. Dragging the operating point updates the tangent line, slope readout, and Lagrangian value in real time.

**Implementation notes:**
- Binary R(D): use `H_b(p) - H_b(D)` from `informationTheory.ts` (the `binaryEntropy` function already exists).
- Gaussian R(D): use `max(0, 0.5 * Math.log2(sigma2 / D))`, computed inline.
- Tangent line slope: $dR/dD$ computed analytically. Binary: $s = \log_2(D/(1-D))$. Gaussian: $s = -1/(2D \ln 2)$.
- Achievable/unachievable shading: D3 `area` with `fill-opacity`.
- Draggable point: D3 drag behavior constrained to the curve (snap to nearest curve point).
- Use Approach B (multi-panel with `useEffect` + manual refs).
- Data generation is inline.

**Reference:** Notebook §2 and §4.

### 4b. BlahutArimotoExplorer

**File:** `src/components/viz/BlahutArimotoExplorer.tsx`

An interactive step-through of the Blahut–Arimoto algorithm, showing how alternating minimization converges to $R(D)$:

- **Left panel:** The $R(D)$ curve (computed by sweeping the slope parameter). As the algorithm runs, a dot traces the current $(D, R)$ estimate onto the curve, converging to the true optimum.

- **Right panel:** Convergence plot. $x$-axis is iteration number, $y$-axis is the current rate estimate. The curve converges to the horizontal dashed line (the true $R$ for the given slope $s$).

- **Bottom panel:** The output distribution $q(\hat{x})$ is displayed as a bar chart, updating at each iteration. Bars animate smoothly as the distribution converges.

- **Controls:**
  - **Source distribution** selector: "Uniform ternary", "Non-uniform ternary $[0.5, 0.3, 0.2]$", "Quaternary $[0.4, 0.3, 0.2, 0.1]$".
  - **Slope $s$ slider** (range: $-20$ to $-0.5$): controls the target point on the R(D) curve.
  - **"Step" button:** advances one BA iteration.
  - **"Run" button:** runs to convergence (with 100ms per step animation).
  - **"Reset" button:** clears to initial state.

**Interaction:** Clicking "Step" advances one iteration, updating the convergence plot and distribution bars. "Run" animates the full convergence. Changing the slope or source resets and shows the new target.

**Implementation notes:**
- Implement the BA algorithm in TypeScript (same logic as the Python version in the notebook).
- Use D3 transitions (200ms) for smooth bar chart updates.
- The full $R(D)$ curve in the left panel is pre-computed by running BA at 50 slope values on the component mount (fast for small alphabets).
- Hamming distortion matrix: $d(x, \hat{x}) = 1 - \delta_{x\hat{x}}$.
- Import `entropy`, `mutualInformation` from `informationTheory.ts`.
- Add `blahutArimotoStep` to `informationTheory.ts` (see §5).
- Use `requestAnimationFrame` with `setTimeout` for the step-by-step animation (same pattern as `SourceCodingExplorer` in Shannon Entropy).
- Data generation is inline.

**Reference:** Notebook §5.

### 4c. InformationBottleneckExplorer

**File:** `src/components/viz/InformationBottleneckExplorer.tsx`

An interactive visualization of the information bottleneck trade-off between compression and relevance:

- **Left panel:** The IB curve: $I(T; Y)$ (relevance) on the $y$-axis vs $I(X; T)$ (complexity) on the $x$-axis. Points are colored by $\beta$. A reference line shows $I(X; Y)$ (the maximum achievable relevance). The curve is concave and interpolates from the origin to $(H(X), I(X;Y))$.

- **Right panel:** Two side-by-side horizontal bars for the current $\beta$ showing: $I(X; T)$ (complexity, light shade) and $I(T; Y)$ (relevance, dark shade). A $\beta$ slider controls the operating point on the IB curve.

- **Controls:**
  - **$\beta$ slider** (range: 0.1 to 50, log scale): controls the compression-relevance trade-off.
  - **Joint distribution selector** (dropdown): "Noisy identity channel", "Symmetric channel", "Asymmetric channel". Each preset defines a joint distribution $p(x, y)$.
  - **Bottleneck size $|T|$** selector (dropdown: 2, 3, 4).

**Interaction:** Adjusting $\beta$ highlights the current operating point on the IB curve and updates the complexity/relevance bars. Changing the joint distribution recomputes the entire IB curve (may take ~500ms; show a brief loading state).

**Implementation notes:**
- The IB computation is iterative (alternating minimization, similar to BA). For small alphabets ($|X|, |Y|, |T| \leq 4$), this is fast enough for real-time updates.
- Pre-compute the IB curve by sweeping $\beta$ across 40–60 values on the mount, then interpolate for slider interaction.
- Import `klDivergence`, `mutualInformation`, `entropy` from `informationTheory.ts`.
- Use D3 for the scatter plot, curves, and bar charts. CSS custom properties for theming.
- Data generation is inline.

**Reference:** Notebook §6.

### 4d. VAERateDistortionExplorer

**File:** `src/components/viz/VAERateDistortionExplorer.tsx`

An interactive visualization connecting the VAE loss to rate-distortion theory:

- **Left panel:** The Gaussian $R(D)$ curve with VAE operating points marked. Each operating point corresponds to a different $\beta$ value. The current $\beta$ is highlighted with a larger marker and labeled $(D_\beta, R_\beta)$.

- **Right panel:** Two curves as a function of $\beta$: Rate ($D_{KL}$ term) and Distortion (reconstruction error). The standard VAE ($\beta = 1$) is marked with a vertical dashed line. The left region ($\beta < 1$) is labeled "Reconstruction priority"; the right region ($\beta > 1$) is labeled "Compression priority / disentangled".

- **Controls:**
  - **$\beta$ slider** (range: 0.1 to 10): controls the rate-distortion operating point.
  - **Source variance $\sigma^2$ selector** (dropdown: 0.5, 1, 2, 4): changes the R(D) curve.

**Interaction:** Adjusting $\beta$ moves the operating point on the $R(D)$ curve and highlights the corresponding position on the rate/distortion curves. Changing $\sigma^2$ redraws the $R(D)$ curve.

**Implementation notes:**
- For the Gaussian source: operating point at slope $s = -1/\beta$ (in nats), giving $D = \beta/(2 \ln 2)$ and $R = \frac{1}{2}\log_2(\sigma^2 / D)$.
- Use D3 for line plots and scatter points.
- Simple inline computation — no external data module.
- Use Approach A (`useD3` hook — single SVG with sub-groups for each panel).

**Reference:** Notebook §7.

---

## 5. Data Modules

### Shared utility additions: `src/components/viz/shared/informationTheory.ts`

This module was created in the Shannon Entropy topic and extended in the KL Divergence topic. **Add** the following functions (do not duplicate or modify existing functions):

```typescript
/** Rate-distortion function for binary source with Hamming distortion.
 *  R(D) = H_b(p) - H_b(D) for 0 ≤ D ≤ min(p, 1-p), else 0. */
export function rateDistortionBinary(p: number, D: number): number { ... }

/** Rate-distortion function for Gaussian source with squared error.
 *  R(D) = max(0, 0.5 * log2(sigma2 / D)). */
export function rateDistortionGaussian(sigma2: number, D: number): number { ... }

/** One step of the Blahut–Arimoto algorithm for rate-distortion.
 *  Takes current q(x̂), returns updated q(x̂) and test channel p(x̂|x).
 *  @param px - source distribution, shape [|X|]
 *  @param distortionMatrix - d(x, x̂), shape [|X|][|X̂|]
 *  @param qXhat - current output distribution, shape [|X̂|]
 *  @param slope - Lagrange multiplier s < 0
 *  @returns { qXhat: number[], pXhatGivenX: number[][], rate: number, distortion: number } */
export function blahutArimotoStep(
  px: number[],
  distortionMatrix: number[][],
  qXhat: number[],
  slope: number
): { qXhat: number[]; pXhatGivenX: number[][]; rate: number; distortion: number } { ... }

/** Run Blahut–Arimoto to convergence and return the full R(D) point.
 *  @param px - source distribution
 *  @param distortionMatrix - distortion matrix
 *  @param slope - Lagrange multiplier s < 0
 *  @param maxIter - maximum iterations (default: 200)
 *  @param tol - convergence tolerance (default: 1e-10)
 *  @returns { rate: number, distortion: number, qXhat: number[], pXhatGivenX: number[][] } */
export function blahutArimoto(
  px: number[],
  distortionMatrix: number[][],
  slope: number,
  maxIter?: number,
  tol?: number
): { rate: number; distortion: number; qXhat: number[]; pXhatGivenX: number[][] } { ... }

/** Hamming distortion matrix for alphabet size k.
 *  Returns k×k matrix where d[i][j] = (i === j) ? 0 : 1. */
export function hammingDistortionMatrix(k: number): number[][] { ... }
```

Use the **same conventions** as existing functions in `informationTheory.ts`:
- All divergences and rates use base-2 log (bits).
- Guard against numerical issues: `0 log 0 = 0`, clamp probabilities to `[1e-15, 1]`.
- `blahutArimotoStep` returns a single iteration result so the UI can animate step-by-step.
- `blahutArimoto` runs to convergence for pre-computing full R(D) curves.
- Direct evaluation is fine (no caching needed — small alphabets make BA fast).

### Shared types addition: `src/components/viz/shared/types.ts`

Check if the following types already exist (some may have been added in prior topics). If not, add them:

```typescript
export interface BlahutArimotoState {
  iteration: number;
  qXhat: number[];
  pXhatGivenX: number[][];
  rate: number;
  distortion: number;
}

export interface RateDistortionPoint {
  rate: number;
  distortion: number;
}
```

---

## 6. Curriculum Graph Updates

### `src/data/curriculum-graph.json`

**Add node:**

```json
{
  "id": "rate-distortion",
  "label": "Rate-Distortion Theory",
  "domain": "information-theory",
  "status": "published",
  "url": "/topics/rate-distortion"
}
```

**Add edges (two prerequisites):**

```json
{ "source": "shannon-entropy", "target": "rate-distortion" }
```
```json
{ "source": "convex-analysis", "target": "rate-distortion" }
```

**No downstream edges.** `minimum-description-length` depends on `kl-divergence`, not on `rate-distortion`. Do not add edges to planned topics.

### `src/data/curriculum.ts`

1. **Remove `"Rate-Distortion Theory"` from the `planned` array** of the `information-theory` domain track.
2. **Add it to the `published` array** (or equivalent — follow the pattern established by the Shannon Entropy and KL Divergence topics).
3. Do **not** modify other tracks or domains.

> **Implementation note:** The `/paths` page should now show "3 published" for the Information Theory track. Verify the topic appears with the "Intermediate" difficulty badge and a link to `/topics/rate-distortion`.

---

## 7. Cross-References

### Outbound (from this page)

**Backward references (these pages exist — use live links):**

- `[Shannon Entropy & Mutual Information](/topics/shannon-entropy)` — referenced in: §1 (motivation: R(D) minimizes mutual information), §2 (R(0) = H(X) recovers source coding limit), §3 (converse uses DPI from Shannon Entropy), §8 (computational notes reference entropy functions)
- `[KL Divergence & f-Divergences](/topics/kl-divergence)` — referenced in: §6 (IB distortion $d_{IB}(x,t) = D_{KL}(p(y|x) \| p(y|t))$), §7 (VAE rate term is KL divergence), §8 (connections)
- `[Convex Analysis](/topics/convex-analysis)` — referenced in: §2 (R(D) is convex — the optimization is a convex program), §5 (BA alternating minimization exploits convexity), §8 (connections)
- `[Lagrangian Duality & KKT Conditions](/topics/lagrangian-duality)` — referenced in: §4 (slope $s$ is the Lagrange multiplier), §5 (KKT conditions yield the optimal test channel structure), §8 (connections)

**Forward references (these pages do NOT exist — use plain text):**

- **Minimum Description Length** *(coming soon)* — referenced in: §8 (connections: MDL connects coding to model selection)

⚠️ **Do NOT create hyperlinks to unbuilt topics.** Use the plain-text pattern: `**Topic Name** *(coming soon)*`.

### Inbound (update existing pages)

- **`shannon-entropy.mdx`:** If there is a forward reference to Rate-Distortion Theory marked as "coming soon", convert it to a live link: *"[Rate-Distortion Theory](/topics/rate-distortion) extends the source coding theorem to lossy compression: $R(D) = \min I(X; \hat{X})$ is the minimum rate at distortion $\leq D$."*

- **`kl-divergence.mdx`:** If there is a forward reference to Rate-Distortion marked as "coming soon", convert it to a live link: *"[Rate-Distortion Theory](/topics/rate-distortion) uses KL divergence as the distortion measure in the information bottleneck, connecting lossy compression to representation learning."*

> **Do not update other existing MDX files.** The Convex Analysis and Lagrangian Duality connections are backward references only (this topic cites them, they do not cite this topic).

---

## 8. Paths Page Update

### Existing Track Section Update

The **Information Theory** track section already exists (created by Shannon Entropy, expanded by KL Divergence). Update it:

**Track topics (updated state):**

| Topic | Status | Difficulty |
|---|---|---|
| **Shannon Entropy & Mutual Information** | **Published** (linked to `/topics/shannon-entropy`) | **Foundational** |
| **KL Divergence & f-Divergences** | **Published** (linked to `/topics/kl-divergence`) | **Intermediate** |
| **Rate-Distortion Theory** | **Published** (linked to `/topics/rate-distortion`) | **Intermediate** |
| Minimum Description Length | Planned (gray, unlinked) | — |

The track counter should update from "2 published" to "3 published."

---

## 9. Images

Export from the notebook and place in:

```
public/images/topics/rate-distortion/
```

| Figure | Notebook Source | Filename |
|---|---|---|
| Distortion measures (Hamming matrix, squared error curves, expected distortion) | §1 | `distortion-measures.png` |
| Rate-distortion function (binary, Gaussian, achievable region) | §2 | `rate-distortion-function.png` |
| Rate-distortion theorem (block diagram, achievable vs unachievable) | §3 | `rate-distortion-theorem.png` |
| Closed-form solutions (binary test channel, Gaussian, Shannon lower bound) | §4 | `closed-form-solutions.png` |
| Blahut–Arimoto algorithm (R(D) curve, convergence, distribution evolution) | §5 | `blahut-arimoto.png` |
| Information bottleneck (IB curve, rate-distortion interpretation, β trade-off) | §6 | `information-bottleneck.png` |
| Successive refinement (layered coding, Gaussian refinability) | §6b | `successive-refinement.png` |
| Computational notes (VAE operating points, β-VAE trade-off, framework table) | §7 | `computational-notes.png` |

---

## 10. Code Blocks

Include selected Python snippets from the notebook as syntax-highlighted code blocks in the MDX:

| Block | Notebook Source | Purpose |
|---|---|---|
| `rate_distortion_binary()` computation | §4 | Compute $R(D) = H_b(p) - H_b(D)$ for binary source |
| `rate_distortion_gaussian()` computation | §4 | Compute $R(D) = \frac{1}{2}\log_2(\sigma^2/D)$ for Gaussian source |
| `blahut_arimoto_rd()` implementation | §5 | Full BA algorithm with convergence loop |
| `information_bottleneck()` implementation | §6 | IB via alternating optimization |
| VAE loss decomposition | §7 | Show $\mathcal{L}_{\text{VAE}} = \underbrace{E[-\log p(x|z)]}_{\text{distortion}} + \underbrace{D_{KL}(q(z|x) \| p(z))}_{\text{rate}}$ |

---

## 11. Notebook File

Place the companion notebook in the repo:

```
notebooks/rate-distortion/03_rate_distortion.ipynb
```

This follows the corrected `notebookPath` convention (subdirectory + file). The `03_` prefix indicates this is the third notebook in the Information Theory track.

---

## 12. SEO & OG

If the OG image generation script is active, it should auto-generate an OG image. If not, create a manual OG image and place it in `public/og/rate-distortion.png`.

Meta tags from `TopicLayout.astro` frontmatter. Verify:
- `<title>`: "Rate-Distortion Theory | formalML"
- `<meta name="description">`: Uses the `abstract` field
- OG tags: title, description, image

---

## 13. Testing Checklist

- [ ] All KaTeX renders correctly (especially `R(D) = \min_{p(\hat{x}|x)} I(X; \hat{X})`, `d(x, \hat{x})`, `d_H(x, \hat{x}) = \mathbb{1}[x \neq \hat{x}]`, `R(D) = H_b(p) - H_b(D)`, `R(D) = \frac{1}{2}\log_2(\sigma^2/D)`, `p_s(\hat{x}|x) \propto q(\hat{x}) \exp(s\,d(x,\hat{x}))`, `\mathcal{L}_{IB} = I(X;T) - \beta I(T;Y)`, `D_{KL}(q(z|x) \| p(z))`, `R(0) = H(X)`, `D_{\max}`)
- [ ] Prerequisite badges show: Shannon Entropy & Mutual Information (within-track) and Convex Analysis (cross-track) as linked chips
- [ ] Topic card appears on `/topics` index with "Intermediate" difficulty badge and `information-theory` domain tag
- [ ] Information Theory track on `/paths` shows "3 published"
- [ ] Rate-Distortion Theory appears on `/paths` as Published (Intermediate, linked)
- [ ] Minimum Description Length remains Planned (gray, unlinked)
- [ ] `shannon-entropy → rate-distortion` edge is active in the curriculum graph
- [ ] `convex-analysis → rate-distortion` edge is active in the curriculum graph (cross-track)
- [ ] `RateDistortionExplorer` source selector switches between all five options
- [ ] `RateDistortionExplorer` draggable operating point moves along the curve
- [ ] `RateDistortionExplorer` tangent line and slope readout update in real time
- [ ] `RateDistortionExplorer` achievable/unachievable shading is correct
- [ ] `RateDistortionExplorer` boundary annotations ($R(0) = H(X)$, $R(D_{\max}) = 0$) display correctly
- [ ] `BlahutArimotoExplorer` "Step" button advances one BA iteration
- [ ] `BlahutArimotoExplorer` "Run" button animates convergence
- [ ] `BlahutArimotoExplorer` convergence plot converges to a horizontal dashed line
- [ ] `BlahutArimotoExplorer` distribution bars animate smoothly
- [ ] `BlahutArimotoExplorer` R(D) curve dot converges to curve
- [ ] `BlahutArimotoExplorer` slope slider resets and shows new target
- [ ] `InformationBottleneckExplorer` IB curve is concave
- [ ] `InformationBottleneckExplorer` $\beta$ slider moves operating point along curve
- [ ] `InformationBottleneckExplorer` complexity/relevance bars update
- [ ] `InformationBottleneckExplorer` reference line shows $I(X;Y)$
- [ ] `InformationBottleneckExplorer` joint distribution presets produce correct curves
- [ ] `VAERateDistortionExplorer` operating points lie on the $R(D)$ curve
- [ ] `VAERateDistortionExplorer` $\beta$ slider moves operating point
- [ ] `VAERateDistortionExplorer` rate/distortion curves cross at $\beta = 1$
- [ ] `VAERateDistortionExplorer` $\sigma^2$ selector redraws the $R(D)$ curve
- [ ] Shannon Entropy MDX updated: forward references to Rate-Distortion converted to live links
- [ ] KL Divergence MDX updated: forward references to Rate-Distortion converted to live links
- [ ] Forward references to MDL use plain text + "(coming soon)" — no dead links
- [ ] All static figures load from `public/images/topics/rate-distortion/`
- [ ] New functions added to `informationTheory.ts` without breaking existing Shannon Entropy or KL Divergence components
- [ ] Page is responsive (viz components stack vertically on mobile)
- [ ] "Intermediate" difficulty badge is styled correctly
- [ ] Pagefind indexes the new topic on rebuild
- [ ] Build succeeds with zero errors: `pnpm build`

---

## 14. Build Order

1. **Add rate-distortion functions to `informationTheory.ts`** — `rateDistortionBinary`, `rateDistortionGaussian`, `blahutArimotoStep`, `blahutArimoto`, `hammingDistortionMatrix`. Run existing Shannon Entropy and KL Divergence component tests to confirm no regressions.
2. **Create `rate-distortion.mdx`** with full frontmatter and all markdown/LaTeX content. Use `TheoremBlock` for all formal elements (5 definitions, 6 propositions, 5 theorems, 2 remarks). No interactive components yet.
3. Add notebook to `notebooks/rate-distortion/` directory.
4. Export and add static figures to `public/images/topics/rate-distortion/`.
5. Build `RateDistortionExplorer.tsx` — flagship component, build first. R(D) curve + draggable operating point + tangent line.
6. Build `BlahutArimotoExplorer.tsx` — step-through animation + convergence plot + distribution evolution. Three-panel layout with animation.
7. Build `InformationBottleneckExplorer.tsx` — IB curve + β slider + complexity/relevance bars. Two-panel layout.
8. Build `VAERateDistortionExplorer.tsx` — Gaussian R(D) with VAE operating points + β trade-off curves. Two-panel layout.
9. Embed all components in the MDX at their appropriate section positions.
10. Update `curriculum-graph.json`: add a node and two edges.
11. Update `curriculum.ts`: move Rate-Distortion Theory from `planned` to published.
12. Update `shannon-entropy.mdx`: convert forward references to Rate-Distortion to live links.
13. Update `kl-divergence.mdx`: convert forward references to Rate-Distortion to live links.
14. Run testing checklist (§13).
15. Commit and deploy.

---

## Appendix A: Key Differences from the KL Divergence Brief

1. **Third topic in an existing track.** No new track registration, no new domain. The Information Theory track, `information-theory` domain, and shared `informationTheory.ts` module all exist.
2. **Two prerequisites (cross-track + within-track).** `shannon-entropy` (within-track) and `convex-analysis` (cross-track). This is the first topic in the Information Theory track with a cross-track prerequisite — creates a `convex-analysis → rate-distortion` edge. Follow the same pattern as `measure-theoretic-probability → shannon-entropy`.
3. **No downstream dependencies in the current roadmap.** MDL depends on `kl-divergence`, not on `rate-distortion`. The two branches from `shannon-entropy` (KL Divergence and Rate-Distortion) are independent in the DAG.
4. **Animation in one component.** The `BlahutArimotoExplorer` uses step-by-step animation for the alternating minimization, similar to `SourceCodingExplorer`'s animated Huffman tree construction. Ensure build/step/run/reset controls.
5. **Extends shared utility module (third extension).** Adds rate-distortion, and BA functions to `informationTheory.ts`. Must not break existing Shannon Entropy or KL Divergence components. Run both topic pages after updating.
6. **Strong ML connections (different from prior topics).** Shannon Entropy is connected to cross-entropy loss. KL Divergence and ELBO in GANs. This topic connects to VAEs ($\beta$-VAE), neural compression, and the information bottleneck — a representation-learning audience.
7. **Two inbound updates required.** Both `shannon-entropy.mdx` and `kl-divergence.mdx` have forward references to Rate-Distortion marked as "coming soon" — both must be converted to live links.
8. **Intermediate difficulty with convex optimization flavor.** The Blahut–Arimoto algorithm and parametric form connect directly to convex analysis and Lagrangian duality, which are cross-track backward references. The editorial voice should explicitly highlight these connections.

---

## Appendix B: Information Theory Track Roadmap (updated)

| Order | Topic | Difficulty | Prerequisites | Status |
|---|---|---|---|---|
| 1 | Shannon Entropy & Mutual Information | Foundational | Measure-Theoretic Probability (cross-track) | ✅ Published |
| 2 | KL Divergence & f-Divergences | Intermediate | Shannon Entropy | ✅ Published |
| 3 | **Rate-Distortion Theory** (this brief) | Intermediate | Shannon Entropy, Convex Analysis (cross-track) | 🚧 Ready for implementation |
| 4 | Minimum Description Length | Advanced | KL Divergence, PAC Learning (cross-track) | Planned |

DAG structure: `shannon-entropy → kl-divergence → minimum-description-length` and `shannon-entropy → rate-distortion` (two branches from the root, with MDL also depending on KL Divergence).

Cross-track edges:
- `measure-theoretic-probability → shannon-entropy` ✅ (exists)
- `convex-analysis → rate-distortion` (new — this brief)
- `pac-learning → minimum-description-length` (future — when MDL is implemented)

---

*Brief version: v1 | Last updated: 2026-03-26 | Author: Jonathan Rocha*  
*Reference notebook: `notebooks/rate-distortion/03_rate_distortion.ipynb`*  
*Reference doc: `docs/plans/formalml-handoff-reference.md`*  
*Sibling briefs: `formalml-shannon-entropy-handoff-brief.md` (Topic 1), `formalml-kl-divergence-handoff-brief.md` (Topic 2)*
