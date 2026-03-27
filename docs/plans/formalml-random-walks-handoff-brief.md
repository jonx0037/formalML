# Claude Code Handoff Brief: Random Walks & Mixing

**Project:** formalML — [formalml.com](https://www.formalml.com)  
**Repo:** `github.com/jonx0037/formalML`  
**Stack:** Astro 6 · React 19 · MDX · Tailwind CSS 4 · D3.js 7 · KaTeX · Vercel  
**Package Manager:** pnpm  
**Status:** Ready for implementation  
**Reference Notebook:** `notebooks/random-walks/02_random_walks.ipynb`  
**Reference Doc:** `docs/plans/formalml-handoff-reference.md`

---

## 1. Objective

Add a new topic page **"Random Walks & Mixing"** to the **Graph Theory** track on formalml.com. This is the **second topic in the track** — it builds directly on Graph Laplacians & Spectrum and extends the shared `graphTheory.ts` utility module.

1. **Position in DAG.** Second topic in the Graph Theory track. Prerequisite: `graph-laplacians`. Downstream: `expander-graphs` and `message-passing` both depend on this topic.
2. **Cross-track connections.** `measure-theoretic-probability` (Markov chain convergence relies on σ-algebra machinery), `concentration-inequalities` (Chernoff-type bounds appear in mixing time analysis), and `shannon-entropy` (entropy and KL divergence measure distance to stationarity).
3. **Difficulty:** Intermediate — algebraic derivations follow geometric intuition established in the foundational Graph Laplacians topic. Assumes familiarity with the transition matrix $P = D^{-1}A$, the normalized Laplacian spectrum, and Cheeger's inequality from the prerequisites.

**What the topic covers:**

- Markov chains on graphs: transition matrix $P = D^{-1}A$, irreducibility, aperiodicity, ergodicity
- The stationary distribution $\boldsymbol{\pi}$ with $\pi_i = d_i / 2m$ and detailed balance
- Convergence theorem: irreducible aperiodic chains converge to the unique stationary distribution
- Mixing time: total variation distance $\|P^t(x, \cdot) - \boldsymbol{\pi}\|_{\mathrm{TV}}$ and spectral gap bounds $t_{\mathrm{mix}} = \Theta(1/\gamma)$ where $\gamma = 1 - \lambda_2(P)$
- Hitting times, commute times, and effective resistance — the probabilistic interpretation of graph connectivity
- Lazy random walks: $P_{\text{lazy}} = \frac{1}{2}(I + P)$ for handling bipartite graphs
- Applications: PageRank as a random walk with teleportation, label propagation, DeepWalk/node2vec embeddings

---

## 2. MDX File

### Location

```
src/content/topics/random-walks.mdx
```

The entry `id` will be `random-walks` (derived from filename by the `glob()` loader in `src/content.config.ts`). The dynamic route at `src/pages/topics/[...slug].astro` resolves this to `/topics/random-walks`.

### Frontmatter

```yaml
---
title: "Random Walks & Mixing"
subtitle: "Markov chains on graphs — from the transition matrix to mixing times, hitting times, and the spectral gap"
status: "published"
difficulty: "intermediate"
prerequisites:
  - "graph-laplacians"
tags:
  - "graph-theory"
  - "random-walks"
  - "markov-chains"
  - "mixing-time"
  - "spectral-gap"
  - "hitting-time"
domain: "graph-theory"
videoId: null
notebookPath: "notebooks/random-walks/02_random_walks.ipynb"
githubUrl: "https://github.com/jonx0037/formalML/blob/main/src/content/topics/random-walks.mdx"
datePublished: 2026-03-28
estimatedReadTime: 45
abstract: "A random walk on a graph is a Markov chain whose transition matrix P = D⁻¹A encodes local structure: at each step, the walker moves to a uniformly random neighbor. The stationary distribution π assigns probability proportional to degree, and the Perron–Frobenius theorem guarantees convergence for connected, non-bipartite graphs. The mixing time — how many steps until the walk's distribution is close to π in total variation — is governed by the spectral gap γ = 1 − λ₂(P), tying random walk convergence directly to the Laplacian eigenvalues from spectral graph theory. Hitting times and commute times provide complementary measures of graph distance with deep connections to electrical networks: the commute time between two vertices equals the effective resistance times the total edge weight. These ideas power modern graph algorithms from PageRank to DeepWalk, and the mixing time perspective motivates expander graphs — sparse graphs that mix as fast as dense ones."
connections:
  - topic: "graph-laplacians"
    relationship: "The transition matrix P = D⁻¹A is the complement of the random walk Laplacian L_rw = I − P. The eigenvalues of P are μᵢ = 1 − λᵢ(L_rw), so the spectral gap of the walk equals the algebraic connectivity λ₂ of the normalized Laplacian."
  - topic: "spectral-theorem"
    relationship: "P is similar to D^{1/2} P D^{-1/2} = D^{-1/2} A D^{-1/2}, a real symmetric matrix. The Spectral Theorem guarantees real eigenvalues in [−1, 1] and an orthonormal eigenbasis, which underlies the spectral decomposition of P^t."
  - topic: "measure-theoretic-probability"
    relationship: "A random walk defines a sequence of random variables on the vertex set. Convergence in total variation is convergence in the L¹ norm on the probability simplex — a measure-theoretic statement about the walk's law."
  - topic: "shannon-entropy"
    relationship: "The KL divergence D_KL(P^t(x,·) ∥ π) decreases monotonically along the walk and reaches zero at stationarity. The stationary entropy H(π) = log n for regular graphs — maximum randomness. The spectral gap controls the rate of entropy production."
  - topic: "concentration-inequalities"
    relationship: "Chernoff-type bounds for sums of random variables along a Markov chain use the spectral gap to account for dependence — the walk mixes fast enough that consecutive samples are nearly independent."
  - topic: "pca-low-rank"
    relationship: "DeepWalk and node2vec learn vertex embeddings by running random walks and applying skip-gram (implicit low-rank matrix factorization). The embedding captures the random walk transition probabilities — a spectral approximation related to the Laplacian eigenvectors."
references:
  - type: "book"
    title: "Markov Chains and Mixing Times"
    authors: "Levin, Peres & Wilmer"
    year: 2009
    note: "The standard textbook — covers mixing times, spectral methods, coupling, and hitting times with full proofs"
  - type: "book"
    title: "Spectral Graph Theory"
    authors: "Chung"
    year: 1997
    note: "Chapter 2 covers random walks on graphs and the connection between mixing and the normalized Laplacian spectrum"
  - type: "paper"
    title: "The PageRank Citation Ranking: Bringing Order to the Web"
    authors: "Page, Brin, Motwani & Winograd"
    year: 1999
    url: "https://doi.org/10.2139/ssrn.1712064"
    note: "PageRank = stationary distribution of a random walk with teleportation — the foundational application"
  - type: "paper"
    title: "DeepWalk: Online Learning of Social Representations"
    authors: "Perozzi, Al-Rfou & Skiena"
    year: 2014
    url: "https://doi.org/10.1145/2623330.2623732"
    note: "Random walks for vertex embedding — the bridge from Markov chains to representation learning"
  - type: "paper"
    title: "Faster Mixing via Average Conductance"
    authors: "Lovász & Kannan"
    year: 1999
    url: "https://doi.org/10.1145/301250.301291"
    note: "Sharper mixing time bounds via average conductance — extends Cheeger's inequality to the random walk setting"
  - type: "book"
    title: "Random Walks and Electric Networks"
    authors: "Doyle & Snell"
    year: 1984
    note: "The classic treatment of the equivalence between random walk quantities and electrical network quantities"
---
```

### Content Sections

The MDX body should have the following sections. Pull all mathematical content from the notebook. Each section header maps directly to a notebook section.

| # | MDX Section | Notebook § | Key Content |
|---|---|---|---|
| 1 | Overview & Motivation | §0 | Why random walks; the drunkard on a graph; what mixing time tells us that static analysis doesn't; roadmap |
| 2 | Markov Chains on Graphs | §1 | Transition matrix $P = D^{-1}A$; random walk definition; irreducibility (connected graph), aperiodicity (non-bipartite or lazy); stochastic matrices; state space as vertex set |
| 3 | Stationary Distribution & Convergence | §2 | Stationary distribution $\pi_i = d_i / 2m$; detailed balance $\pi_i P_{ij} = \pi_j P_{ji}$ (reversibility); existence and uniqueness (Perron–Frobenius); convergence theorem for ergodic chains |
| 4 | Total Variation & Mixing Time | §3 | Total variation distance $\|P^t(x,\cdot) - \boldsymbol{\pi}\|_{\mathrm{TV}}$; mixing time $t_{\mathrm{mix}}(\varepsilon)$; worst-case vs average-case; $\varepsilon$-mixing time definition |
| 5 | Spectral Analysis of Mixing | §4 | Spectral decomposition of $P^t$; spectral gap $\gamma = 1 - \lambda_2(P)$; upper bound $\|P^t(x,\cdot) - \boldsymbol{\pi}\|_{\mathrm{TV}} \leq \frac{1}{2}\sqrt{1/\pi_{\min}} \cdot (1 - \gamma)^t$; lower bound $t_{\mathrm{mix}} \geq \frac{1}{\gamma}\ln\frac{1}{2\varepsilon}$; connection to $\lambda_2(\mathcal{L})$: $\gamma = \lambda_2(\mathcal{L})$ |
| 6 | Lazy Random Walks | §5 | Definition $P_{\text{lazy}} = \frac{1}{2}(I + P)$; why laziness fixes periodicity; spectrum of $P_{\text{lazy}}$ vs $P$; all eigenvalues in $[0, 1]$ |
| 7 | Hitting Times & Commute Times | §6 | Expected hitting time $h(i, j)$; commute time $\kappa(i, j) = h(i, j) + h(j, i)$; fundamental matrix $Z = (I - P + \Pi)^{-1}$; hitting time formula; commute time and effective resistance $\kappa(i, j) = 2m \cdot R_{\text{eff}}(i, j)$ |
| 8 | Effective Resistance & Electrical Networks | §7 | Graph as electrical network; Kirchhoff's laws; effective resistance from the Laplacian pseudoinverse $R_{\text{eff}}(i,j) = (e_i - e_j)^T L^\dagger (e_i - e_j)$; random target lemma: $\mathbb{E}_j[\kappa(i,j)] = 2m \cdot \mathrm{tr}(L^\dagger)$ |
| 9 | Applications: PageRank, DeepWalk, Label Propagation | §8 | PageRank as random walk with restart; personalized PageRank; DeepWalk = random walk + skip-gram; label propagation as iterating the transition matrix on label vectors |
| 10 | Computational Notes | §9 | Power iteration for stationary distribution; `scipy.sparse.linalg.eigs` for sparse $P$; `networkx` random walk utilities; NumPy implementation of mixing time estimation |
| 11 | Connections & Further Reading | §10 | Connection table, notation summary, references |

---

## 3. TheoremBlock Usage

Use the existing `TheoremBlock.astro` component for all numbered formal elements. The notebook contains:

| Type | # | Name | TheoremBlock `type` |
|---|---|---|---|
| Definition | 1 | Random Walk on a Graph | `definition` |
| Definition | 2 | Transition Matrix | `definition` |
| Definition | 3 | Stationary Distribution | `definition` |
| Definition | 4 | Total Variation Distance | `definition` |
| Definition | 5 | Mixing Time | `definition` |
| Definition | 6 | Spectral Gap | `definition` |
| Definition | 7 | Lazy Random Walk | `definition` |
| Definition | 8 | Hitting Time | `definition` |
| Definition | 9 | Commute Time | `definition` |
| Definition | 10 | Effective Resistance | `definition` |
| Theorem | 1 | Stationarity of $\boldsymbol{\pi}$ | `theorem` (with `proof`) |
| Theorem | 2 | Detailed Balance (Reversibility) | `theorem` (with `proof`) |
| Theorem | 3 | Convergence Theorem | `theorem` (with `proof` sketch — Perron–Frobenius) |
| Theorem | 4 | Spectral Decomposition of $P^t$ | `theorem` (with `proof`) |
| Theorem | 5 | Upper Mixing Time Bound | `theorem` (with `proof`) |
| Theorem | 6 | Lower Mixing Time Bound | `theorem` (with `proof`) |
| Theorem | 7 | Commute Time–Effective Resistance Identity | `theorem` (with `proof` sketch) |
| Proposition | 1 | Lazy Walk Spectrum | `proposition` (with `proof`) |
| Proposition | 2 | Hitting Time via Fundamental Matrix | `proposition` (stated, verified computationally) |
| Corollary | 1 | Mixing of Regular Graphs | `corollary` |
| Corollary | 2 | Spectral Gap Equals $\lambda_2(\mathcal{L})$ | `corollary` |
| Remark | — | Reversibility and the Inner Product | `remark` (connection to $\ell^2(\pi)$ inner product) |
| Remark | — | PageRank as Perturbed Random Walk | `remark` (teleportation = rank-one perturbation of $P$) |
| Remark | — | Over-Smoothing in GNNs | `remark` (repeated message passing = random walk converging to $\boldsymbol{\pi}$ — bridge to `message-passing`) |

**LaTeX symbols to verify render correctly:**

- `P = D^{-1}A` — Transition matrix
- `\pi_i = d_i / (2m)` — Stationary distribution
- `\|P^t(x,\cdot) - \boldsymbol{\pi}\|_{\mathrm{TV}}` — Total variation distance
- `\gamma = 1 - \lambda_2(P) = \lambda_2(\mathcal{L})` — Spectral gap
- `P_{\text{lazy}} = \frac{1}{2}(I + P)` — Lazy walk
- `h(i, j) = \mathbb{E}_i[\min\{t \geq 0 : X_t = j\}]` — Hitting time
- `\kappa(i, j) = h(i, j) + h(j, i)` — Commute time
- `R_{\text{eff}}(i, j) = (\mathbf{e}_i - \mathbf{e}_j)^T L^\dagger (\mathbf{e}_i - \mathbf{e}_j)` — Effective resistance
- `\frac{1}{2}\sqrt{1/\pi_{\min}}(1 - \gamma)^t` — Upper bound decay

---

## 4. Interactive Visualizations

New React components in `src/components/viz/`. All use `client:visible` hydration. Follow patterns from existing viz components (Approach B — `useEffect` + manual refs for multi-panel layouts).

### 4a. RandomWalkSimulator (Flagship)

**File:** `src/components/viz/RandomWalkSimulator.tsx`

An interactive random walk simulator that animates a walker on a graph and shows the empirical distribution converging to the stationary distribution.

- **Left panel:** Force-directed graph visualization using D3 force simulation. Nodes are circles, edges are lines. A highlighted "walker" node (large filled circle with a glow effect) moves along edges one step at a time. Visited nodes accumulate a heat map — color intensity proportional to visit frequency. Edge traversals are animated with a smooth transition.

- **Right panel:** Two vertically stacked plots:
  1. **Distribution comparison** — a bar chart with one bar per vertex. Each bar shows two values: (a) the current empirical visit frequency (filled) and (b) the stationary distribution $\pi_i = d_i/2m$ (outline). As the walk progresses, the filled bars converge to the outlines.
  2. **Total variation trace** — a line plot of $\|p_t - \boldsymbol{\pi}\|_{\mathrm{TV}}$ over time, showing the walk's convergence. A horizontal dashed line marks the $\varepsilon = 0.25$ mixing threshold. An annotation marks when the walk first drops below the threshold.

- **Controls:**
  - **Preset graph** dropdown: Path(8), Cycle(8), Complete(6), Barbell(4-4), Grid(3×3), Star(7).
  - **Speed** slider: Steps per second (1–20, default 5).
  - **Play / Pause / Reset** buttons.
  - **Walk type** toggle: "Standard" (default) vs "Lazy" (at each step, stay with probability 1/2).
  - **Starting vertex** selector: click a node to set the start.

**Interaction:** The walker moves one step per tick. Visit counts accumulate. The distribution comparison and TV trace update in real time. Reset clears all history and restarts from the selected vertex.

**Implementation notes:**
- Use `setInterval` (or `requestAnimationFrame` with a frame counter) for the walk animation. The interval is controlled by the speed slider.
- Walk logic: at each step, select a neighbor proportional to edge weights (uniform for unweighted). For lazy walks, stay with probability 1/2.
- Visit frequency = visitCount[v] / totalSteps.
- Total variation = (1/2) Σ |freq[v] - π[v]|.
- Walker animation: use D3 transitions to smoothly move the walker circle along the edge to the next node.
- Use graph construction helpers from `graphTheory.ts`.
- Cap graph size at 15 nodes.

**Reference:** Notebook §1–§3.

### 4b. MixingTimeExplorer

**File:** `src/components/viz/MixingTimeExplorer.tsx`

An interactive visualization comparing mixing times across graph families, demonstrating how graph topology and the spectral gap control convergence speed.

- **Left panel:** The current graph drawn with D3 force simulation. Node color intensity encodes the current walk distribution at a selected time step $t$ (dark = high probability, light = low).

- **Right panel:** Two vertically stacked plots:
  1. **TV distance curves** — multiple curves on the same axes, one per graph family. Each curve plots $\|P^t(x_0, \cdot) - \boldsymbol{\pi}\|_{\mathrm{TV}}$ vs $t$ for the worst-case starting vertex. The $\varepsilon = 0.25$ line is dashed. The point where each curve crosses the line is labeled $t_{\mathrm{mix}}$.
  2. **Spectral gap comparison** — a horizontal bar chart of $\gamma = 1 - \lambda_2(P)$ for each graph family, sorted ascending. Bars are color-coded to match the TV curves above.

- **Controls:**
  - **Graph families** checkboxes (select 2–5 to compare): Path($n$), Cycle($n$), Complete($n$), Barbell($k$-$k$), Grid($m \times m$), Hypercube($d$).
  - **Size parameter** $n$ slider (6–16): applies to all selected families.
  - **Time step** $t$ slider (0–200): scrub through time. The left panel shows the distribution at step $t$ for the currently highlighted graph.

**Interaction:** Checking/unchecking graph families adds/removes TV curves. Changing $n$ recomputes all spectra and redraws curves. Scrubbing $t$ updates the left panel distribution heatmap.

**Implementation notes:**
- Compute TV distance analytically via spectral decomposition: $P^t = \sum_i \mu_i^t \mathbf{v}_i \mathbf{v}_i^T$ (in the $\ell^2(\pi)$ basis). This avoids matrix exponentiation.
- For each graph family at size $n$: build graph → compute $P$ → eigendecompose → compute $\|P^t(x_0,\cdot) - \pi\|_{\mathrm{TV}}$ for $t = 0, \ldots, 200$ for worst-case $x_0$.
- Use Jacobi eigenvalue solver from `graphTheory.ts`.
- Hypercube: $d$-dimensional hypercube has $n = 2^d$ vertices, degree $d$. Use the standard binary-representation construction.
- Color-code graph families consistently using `dimensionColors` extended with additional colors from `d3.schemeTableau10`.

**Reference:** Notebook §3–§5.

### 4c. HittingTimeExplorer

**File:** `src/components/viz/HittingTimeExplorer.tsx`

An interactive visualization of hitting times, commute times, and effective resistance, showing the probabilistic and electrical interpretations of graph distance.

- **Left panel:** A graph with edge widths proportional to effective resistance (thick = high resistance = far apart in random walk terms). Node sizes are proportional to expected return time $h(i, i) = 1/\pi_i = 2m/d_i$. A selected **source** node is highlighted (blue). A selected **target** node is highlighted (red). The shortest path (fewest hops) is shown as a dotted line; the "random walk distance" (commute time) is displayed as a label.

- **Right panel:** Two views (toggled):
  1. **Hitting time matrix** — a heatmap of $h(i, j)$ for all pairs, with the selected source row and target column highlighted. Color scale: low (teal) → high (amber).
  2. **Effective resistance matrix** — same layout but showing $R_{\text{eff}}(i, j) = \kappa(i,j) / 2m$.

- **Bottom panel:** A comparison bar chart showing, for the selected source-target pair: (a) shortest path length, (b) hitting time $h(s, t)$, (c) hitting time $h(t, s)$, (d) commute time $\kappa(s, t)$, (e) effective resistance $R_{\text{eff}}(s, t)$. Demonstrates that hitting times are asymmetric while commute times are symmetric.

- **Controls:**
  - **Preset graph** dropdown: Path(8), Cycle(8), Star(7), Barbell(4-4), Complete(6), Grid(3×3).
  - **Source / Target** selection: click nodes to set source (first click) and target (second click).
  - **Right panel view** toggle: "Hitting Times" vs "Effective Resistance."

**Interaction:** Selecting a preset graph rebuilds everything. Clicking nodes sets the source/target and updates the bottom comparison panel. Toggling the right panel view switches the heatmap.

**Implementation notes:**
- **Hitting time computation:** Solve the system $h(i, j) = 1 + \sum_{k \neq j} P_{ik} h(k, j)$ by forming the reduced system $(I - P_{\bar{j}}) \mathbf{h}_j = \mathbf{1}$ where $P_{\bar{j}}$ is $P$ with row/column $j$ removed. Solve via Gaussian elimination (small matrices, ≤ 15×15).
- **Effective resistance:** $R_{\text{eff}}(i,j) = (e_i - e_j)^T L^\dagger (e_i - e_j)$. Compute $L^\dagger$ from the eigendecomposition of $L$ (zero out the zero eigenvalue, invert the rest).
- Use graph helpers and Jacobi solver from `graphTheory.ts`.
- Cap graph size at 15 nodes.

**Reference:** Notebook §6–§7.

---

## 5. Data Modules

### Extend `src/components/viz/shared/graphTheory.ts`

The Random Walks topic extends the shared `graphTheory.ts` module created in Graph Laplacians. **Do not create a new file** — add the following exports to the existing module.

```typescript
// === Random Walk Types ===

export interface TransitionResult {
  P: number[][];                 // Transition matrix P = D^{-1}A
  stationary: number[];          // Stationary distribution π
  spectralGap: number;           // γ = 1 - λ₂(P)
  eigenvaluesP: number[];        // Eigenvalues of P, sorted descending
}

export interface HittingTimeResult {
  hittingTimes: number[][];      // h(i,j) for all pairs
  commuteTimes: number[][];      // κ(i,j) = h(i,j) + h(j,i)
  effectiveResistance: number[][]; // R_eff(i,j)
}

export interface MixingProfile {
  tvDistances: number[];         // TV distance at t = 0, 1, ..., T
  mixingTime: number;            // First t where TV < ε
  spectralGap: number;           // γ
  worstStartVertex: number;      // Vertex achieving worst-case mixing
}

// === Transition Matrix ===

/** Transition matrix P = D^{-1}A for a random walk */
export function transitionMatrix(A: number[][]): number[][] { ... }

/** Lazy transition matrix P_lazy = (1/2)(I + P) */
export function lazyTransitionMatrix(A: number[][]): number[][] { ... }

/** Stationary distribution π_i = d_i / (2m) */
export function stationaryDistribution(A: number[][]): number[] { ... }

/** Full spectral analysis of the transition matrix */
export function analyzeTransitionMatrix(graph: Graph): TransitionResult { ... }

// === Mixing Time ===

/**
 * Total variation distance between distributions p and q.
 * TV(p, q) = (1/2) Σ |p_i - q_i|
 */
export function totalVariationDistance(p: number[], q: number[]): number { ... }

/**
 * Compute the mixing profile: TV distance vs time for worst-case start.
 * Uses spectral decomposition P^t = Σ μᵢ^t vᵢvᵢᵀ (in ℓ²(π) basis).
 * Returns TV distance at each step t = 0, ..., maxT.
 */
export function mixingProfile(
  graph: Graph,
  maxT: number,
  epsilon: number,
  lazy?: boolean
): MixingProfile { ... }

// === Hitting & Commute Times ===

/**
 * Compute all-pairs hitting times h(i,j).
 * Solves (I - P_bar_j) h_j = 1 for each target j.
 */
export function allPairsHittingTimes(graph: Graph): number[][] { ... }

/**
 * Compute all-pairs commute times κ(i,j) = h(i,j) + h(j,i).
 */
export function allPairsCommuteTimes(hittingTimes: number[][]): number[][] { ... }

/**
 * Compute all-pairs effective resistance from the Laplacian pseudoinverse.
 * R_eff(i,j) = (eᵢ - eⱼ)ᵀ L† (eᵢ - eⱼ)
 */
export function allPairsEffectiveResistance(graph: Graph): number[][] { ... }

/**
 * Full hitting time analysis for a graph.
 */
export function analyzeHittingTimes(graph: Graph): HittingTimeResult { ... }

// === Graph Construction (additions) ===

/** Hypercube graph Q_d (2^d vertices, degree d) */
export function hypercubeGraph(d: number): Graph { ... }
```

**Design decisions:**
- **All computations are direct (not lazy).** Random walk quantities are fast for small matrices (n ≤ 16).
- **Spectral decomposition for mixing profiles.** Computing $P^t$ via eigendecomposition avoids matrix exponentiation — store eigenvalues/eigenvectors and compute $\mu_i^t$ as scalars.
- **Hitting times via linear system.** For each target $j$, solve an $(n-1) \times (n-1)$ linear system. Use Gaussian elimination (the matrices are small).
- **Effective resistance via pseudoinverse.** Compute $L^\dagger$ from the eigendecomposition: $L^\dagger = \sum_{i: \lambda_i > 0} \lambda_i^{-1} v_i v_i^T$.

### Shared types addition: `src/components/viz/shared/types.ts`

Check for conflicts with existing types, then add:

```typescript
export interface WalkState {
  currentVertex: number;
  visitCounts: number[];
  totalSteps: number;
  trajectory: number[];
}
```

---

## 6. Curriculum Graph Updates

### `src/data/curriculum-graph.json`

**Update node status (from planned to published):**

```json
{
  "id": "random-walks",
  "label": "Random Walks & Mixing",
  "domain": "graph-theory",
  "status": "published",
  "url": "/topics/random-walks"
}
```

> ⚠️ This node may already exist as `"status": "planned"` from the Graph Laplacians implementation. If so, update the status to `"published"` and ensure the `url` field is present. If it does not exist, add the full node.

**Add within-track downstream edges:**

```json
{ "source": "random-walks", "target": "expander-graphs" }
{ "source": "random-walks", "target": "message-passing" }
```

> ⚠️ Check if `expander-graphs` and `message-passing` exist as planned nodes. If not, add them:
> ```json
> { "id": "expander-graphs", "label": "Expander Graphs", "domain": "graph-theory", "status": "planned", "url": "/topics/expander-graphs" }
> { "id": "message-passing", "label": "Message Passing & GNNs", "domain": "graph-theory", "status": "planned", "url": "/topics/message-passing" }
> ```

The existing edge `{ "source": "graph-laplacians", "target": "random-walks" }` should already be present.

### `src/data/curriculum.ts`

Move `"random-walks"` from the `planned` array to the `published` array in the `graph-theory` track.

The track should now show:
- Published: `["graph-laplacians", "random-walks"]`
- Planned: `["expander-graphs", "message-passing"]`

---

## 7. Cross-References

### Outbound (from this page)

**Backward references (these pages exist — use live links):**

- `[Graph Laplacians & Spectrum](/topics/graph-laplacians)` — referenced in: §1 (transition matrix $P$ introduced as $D^{-1}A$ in the normalized Laplacian section), §4 (spectral gap $\gamma = \lambda_2(\mathcal{L})$), §5 (Cheeger's inequality bounds mixing time), §7 (Laplacian pseudoinverse for effective resistance)
- `[The Spectral Theorem](/topics/spectral-theorem)` — referenced in: §2 (Perron–Frobenius as the stochastic-matrix analog), §4 (spectral decomposition of $P^t$, Courant-Fischer for the spectral gap)
- `[Shannon Entropy & Mutual Information](/topics/shannon-entropy)` — referenced in: §2 (entropy of stationary distribution), §4 (KL divergence as a mixing metric), §8 (information-theoretic interpretation of mixing)
- `[Concentration Inequalities](/topics/concentration-inequalities)` — referenced in: §4 (Chernoff bounds for Markov chains), §8 (MCMC variance reduction)
- `[PCA & Low-Rank Approximation](/topics/pca-low-rank)` — referenced in: §8 (DeepWalk as implicit matrix factorization)

**Forward references to planned topics (use plain text + "(coming soon)"):**

- **Expander Graphs** *(coming soon)* — "Expanders are the sparse graphs that mix the fastest — their spectral gap $\gamma$ is bounded away from zero even as $n \to \infty$. The mixing time of a walk on an expander is $O(\log n)$."
- **Message Passing & GNNs** *(coming soon)* — "Over-smoothing in GNNs is the random walk convergence theorem in disguise: stacking $t$ GCN layers applies $P^t$ to the feature matrix, driving all features toward the stationary distribution $\boldsymbol{\pi}$."

⚠️ **Do NOT create hyperlinks to unbuilt topics.** Use the plain-text pattern: `**Topic Name** *(coming soon)*`.

### Inbound (update existing pages)

- **`graph-laplacians.mdx`:** In the Connections & Further Reading section, the forward reference to **Random Walks & Mixing** should be converted from plain text *(coming soon)* to a live link: `[Random Walks & Mixing](/topics/random-walks)`. Update the connection text: *"[Random Walks & Mixing](/topics/random-walks) develops the Markov chain perspective: the transition matrix $P = D^{-1}A$ has eigenvalues $\mu_i = 1 - \lambda_i(\mathcal{L})$, and the spectral gap $\gamma = \lambda_2(\mathcal{L})$ controls mixing time — how quickly the walk's distribution converges to the stationary distribution."*

- **`graph-laplacians.mdx`:** Also check §4 (Normalized Laplacian) and §7 (Connections to ML) for any plain-text forward references to Random Walks & Mixing. Convert these to live links.

> **Do not update other existing MDX files.** The cross-track connections (Shannon Entropy, Concentration Inequalities, etc.) are backward references only.

---

## 8. Paths Page Update

### Update Existing Track Section

The Graph Theory track section already exists on `/paths` (created by the Graph Laplacians topic). Update:

| Topic | Status | Difficulty | Badge |
|---|---|---|---|
| Graph Laplacians & Spectrum | Published (linked) | Foundational | — |
| **Random Walks & Mixing** | **Published** (linked to `/topics/random-walks`) | **Intermediate** | — |
| Expander Graphs | Planned (gray, unlinked) | Intermediate | — |
| Message Passing & GNNs | Planned (gray, unlinked) | Advanced | — |

---

## 9. Images

Export from the notebook and place in:

```
public/images/topics/random-walks/
```

| Figure | Notebook Source | Filename |
|---|---|---|
| Random walk trajectory on several named graphs | §1 | `walk-trajectories.png` |
| Empirical distribution convergence on path vs complete vs barbell | §2 | `distribution-convergence.png` |
| Mixing time comparison: TV distance curves for graph families | §3 | `mixing-time-comparison.png` |
| Spectral gap vs mixing time scatter plot | §4 | `spectral-gap-mixing-scatter.png` |
| Lazy vs standard walk on bipartite graph | §5 | `lazy-vs-standard-bipartite.png` |
| Hitting time heatmaps for named graphs | §6 | `hitting-time-heatmaps.png` |
| Effective resistance network visualization | §7 | `effective-resistance-network.png` |
| PageRank on a small web graph | §8 | `pagerank-web-graph.png` |

---

## 10. Code Blocks

Include selected Python snippets from the notebook as syntax-highlighted code blocks in the MDX:

| Block | Notebook Source | Purpose |
|---|---|---|
| Transition matrix construction | §1 | Build $P = D^{-1}A$ from adjacency matrix using NumPy |
| Random walk simulation | §2 | Simulate a walk and compute empirical visit frequencies |
| Mixing time estimation | §3 | Compute TV distance at each step via spectral decomposition |
| Hitting time solver | §6 | Solve $(I - P_{\bar{j}})\mathbf{h} = \mathbf{1}$ for all-pairs hitting times |
| Effective resistance | §7 | Compute $R_{\text{eff}}$ via Laplacian pseudoinverse |
| PageRank | §8 | `networkx.pagerank` and manual power iteration |

---

## 11. Notebook File

Place the companion notebook in the repo:

```
notebooks/random-walks/02_random_walks.ipynb
```

This follows the corrected `notebookPath` convention (subdirectory + file). The `02_` prefix indicates this is the second notebook in the Graph Theory track.

---

## 12. SEO & OG

If the OG image generation script is active, it should auto-generate an OG image. If not, create a manual OG image and place it in `public/og/random-walks.png`.

Meta tags from `TopicLayout.astro` frontmatter. Verify:
- `<title>`: "Random Walks & Mixing | formalML"
- `<meta name="description">`: Uses the `abstract` field
- OG tags: title, description, image

---

## 13. Testing Checklist

- [ ] All KaTeX renders correctly (especially `P = D^{-1}A`, `\pi_i = d_i/(2m)`, `\|P^t(x,\cdot) - \boldsymbol{\pi}\|_{\mathrm{TV}}`, `\gamma = 1 - \lambda_2(P)`, `P_{\text{lazy}} = \frac{1}{2}(I+P)`, `R_{\text{eff}}(i,j) = (\mathbf{e}_i - \mathbf{e}_j)^T L^\dagger (\mathbf{e}_i - \mathbf{e}_j)`)
- [ ] Prerequisite chip shows: "Graph Laplacians & Spectrum" as a linked chip
- [ ] Topic card appears on `/topics` index with "Intermediate" difficulty badge and `graph-theory` domain tag
- [ ] Graph Theory track on `/paths` shows Random Walks & Mixing as Published (Intermediate, linked)
- [ ] Expander Graphs and Message Passing & GNNs still show as Planned (gray, unlinked)
- [ ] `graph-laplacians → random-walks` edge renders in the curriculum graph
- [ ] `random-walks → expander-graphs` edge renders in the curriculum graph
- [ ] `random-walks → message-passing` edge renders in the curriculum graph
- [ ] `RandomWalkSimulator` preset dropdown loads all named graphs correctly
- [ ] `RandomWalkSimulator` walker animates smoothly along edges
- [ ] `RandomWalkSimulator` visit frequency bars converge to stationary distribution
- [ ] `RandomWalkSimulator` TV trace decreases monotonically (on average) and crosses the ε threshold
- [ ] `RandomWalkSimulator` lazy walk toggle halves the mixing speed (approximately)
- [ ] `RandomWalkSimulator` play/pause/reset controls work correctly
- [ ] `RandomWalkSimulator` clicking a node sets the starting vertex
- [ ] `MixingTimeExplorer` graph family checkboxes add/remove TV curves
- [ ] `MixingTimeExplorer` TV curves match expected behavior (complete mixes instantly, path mixes slowly)
- [ ] `MixingTimeExplorer` spectral gap bar chart is consistent with TV curve ordering
- [ ] `MixingTimeExplorer` time slider updates left panel distribution heatmap
- [ ] `MixingTimeExplorer` size slider recomputes all spectra correctly
- [ ] `HittingTimeExplorer` click-to-select source/target works
- [ ] `HittingTimeExplorer` hitting time matrix displays correctly (asymmetric for non-regular graphs)
- [ ] `HittingTimeExplorer` effective resistance matrix is symmetric
- [ ] `HittingTimeExplorer` commute time = 2m × effective resistance (verify numerically)
- [ ] `HittingTimeExplorer` bottom comparison chart shows all five metrics
- [ ] `HittingTimeExplorer` edge widths proportional to effective resistance render correctly
- [ ] Forward reference to **Random Walks & Mixing** in `graph-laplacians.mdx` converted to live link
- [ ] Forward references to planned Graph Theory topics use plain text + "(coming soon)" — no dead links
- [ ] All static figures load from `public/images/topics/random-walks/`
- [ ] Extended `graphTheory.ts` module passes TypeScript compilation with no errors
- [ ] Page is responsive (viz components stack vertically on mobile)
- [ ] "Intermediate" difficulty badge is styled correctly
- [ ] Pagefind indexes the new topic on rebuild
- [ ] Build succeeds with zero errors: `pnpm build`

---

## 14. Build Order

1. **Extend `src/components/viz/shared/graphTheory.ts`** — add transition matrix construction, stationary distribution, mixing profile (spectral decomposition approach), hitting/commute time solvers, effective resistance via pseudoinverse, and the hypercube graph constructor. Write verification tests: e.g., stationary distribution of $K_n$ should be uniform $(1/n, \ldots, 1/n)$; spectral gap of $C_n$ should be $1 - \cos(2\pi/n)$; commute time of $K_n$ between any pair should be $2(n-1)/(n) \cdot n = 2(n-1)$.
2. **Create `random-walks.mdx`** with full frontmatter and all markdown/LaTeX content. Use `TheoremBlock` for all formal elements (10 definitions, 7 theorems, 2 propositions, 2 corollaries, 3 remarks). No interactive components yet.
3. Add notebook to `notebooks/random-walks/` directory.
4. Export and add static figures to `public/images/topics/random-walks/`.
5. Build `RandomWalkSimulator.tsx` — flagship component. Start with preset graphs + walker animation, then add distribution tracking, then TV trace.
6. Build `MixingTimeExplorer.tsx` — TV curves + spectral gap comparison. Start with analytical computation of mixing profiles, then add the graph family selector and time slider.
7. Build `HittingTimeExplorer.tsx` — hitting time matrix + effective resistance + comparison chart. Start with matrix computation and heatmap, then add click-to-select and the bottom panel.
8. Embed all components in the MDX at their appropriate section positions.
9. Update `graph-laplacians.mdx` — convert forward references to live links.
10. Update curriculum graph data — update node status and add edges.
11. Update `/paths` page — change Random Walks & Mixing from Planned to Published.
12. Run testing checklist (§13).
13. Commit and deploy.

---

## Appendix A: Key Differences from the Graph Laplacians Brief

1. **Second topic in an existing track.** Unlike Graph Laplacians (first topic, created track infrastructure), this topic builds on existing infrastructure. The `/paths` section, domain registration, and curriculum DAG subgraph already exist.
2. **Extends the shared utility module.** New exports are added to the existing `graphTheory.ts` rather than creating a new module. Ensure no naming conflicts with existing exports.
3. **Intermediate difficulty.** Algebraic derivations (spectral decomposition of $P^t$, hitting time linear systems) come after geometric intuition (watching a walk converge, seeing heat accumulate on high-degree nodes).
4. **Three viz components (not four).** The three components cover the three main conceptual areas: the walk itself (RandomWalkSimulator), the global-mixing perspective (MixingTimeExplorer), and the pairwise-distance perspective (HittingTimeExplorer).
5. **Animation.** The RandomWalkSimulator is the first formalML viz component with a continuous animation loop (step-by-step walker). Previous components were static or user-driven. Use `setInterval` with proper cleanup in `useEffect` return.
6. **Hitting time computation is $O(n^3)$ per target.** For 15-node graphs, this is 15 Gaussian eliminations on 14×14 systems — fast but not trivial. Pre-compute all pairs on graph change, not on user interaction.

---

## Appendix B: Graph Theory Track Status After This Brief

| Order | Topic | Difficulty | Prerequisites | Status |
|---|---|---|---|---|
| 1 | Graph Laplacians & Spectrum | Foundational | Spectral Theorem + Shannon Entropy | ✅ Published |
| 2 | **Random Walks & Mixing** (this brief) | Intermediate | Graph Laplacians | 🚧 Ready for implementation |
| 3 | Expander Graphs | Intermediate | Graph Laplacians, Random Walks | Planned |
| 4 | Message Passing & GNNs | Advanced | Graph Laplacians, Random Walks | Planned |

DAG edges after implementation:
- `spectral-theorem → graph-laplacians → random-walks → expander-graphs`
- `shannon-entropy → graph-laplacians → message-passing`
- `random-walks → message-passing`

Cross-track edges for future topics:
- `concentration-inequalities → expander-graphs` (Chernoff bounds in expander mixing lemma)
- `gradient-descent → message-passing` (GNN training via backpropagation)
- `random-walks → message-passing` (over-smoothing = walk convergence)

---

*Brief version: v1 | Last updated: 2026-03-27 | Author: Jonathan Rocha*  
*Reference notebook: `notebooks/random-walks/02_random_walks.ipynb`*  
*Reference doc: `docs/plans/formalml-handoff-reference.md`*
