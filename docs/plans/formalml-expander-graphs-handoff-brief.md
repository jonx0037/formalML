# Claude Code Handoff Brief: Expander Graphs

**Project:** formalML — [formalml.com](https://www.formalml.com)  
**Repo:** `github.com/jonx0037/formalML`  
**Stack:** Astro 6 · React 19 · MDX · Tailwind CSS 4 · D3.js 7 · KaTeX · Vercel  
**Package Manager:** pnpm  
**Status:** Ready for implementation  
**Reference Notebook:** `notebooks/expander-graphs/03_expander_graphs.ipynb`  
**Reference Doc:** `docs/plans/formalml-handoff-reference.md`

---

## 1. Objective

Add a new topic page **"Expander Graphs"** to the **Graph Theory** track on formalml.com. This is the **third topic in the track** — it builds on both Graph Laplacians & Spectrum and Random Walks & Mixing, and extends the shared `graphTheory.ts` utility module.

1. **Position in DAG.** Third topic in the Graph Theory track. Prerequisites: `graph-laplacians` and `random-walks`. Downstream: `message-passing` (the final topic in the track) depends on this topic.
2. **Cross-track connections.** `concentration-inequalities` (Chernoff-type bounds in the Expander Mixing Lemma and the expander walk sampling theorem), `spectral-theorem` (eigendecomposition of the adjacency matrix underlies the EML proof), `shannon-entropy` (entropy rate of walks on expanders).
3. **Difficulty:** Intermediate — assumes familiarity with the spectral gap (from `graph-laplacians`), mixing times and the spectral decomposition of $P^t$ (from `random-walks`), and Cheeger's inequality (from `graph-laplacians`). Algebraic derivations follow geometric intuition: show expansion visually before proving bounds.

The companion Jupyter Notebook (`03_expander_graphs.ipynb`) contains the full mathematical content, proofs, code, and figures — use it as the authoritative content source for all definitions, theorems, examples, and code blocks.

**What the topic covers:**

- Three equivalent notions of expansion: vertex expansion $h_V(G)$, edge expansion (Cheeger constant) $h(G)$, and spectral expansion $\lambda(G)$
- The $(n, d, \lambda)$-expander formalism for $d$-regular graphs
- The Expander Mixing Lemma: $|E(S,T) - d|S||T|/n| \leq \lambda\sqrt{|S||T|}$ — edges are quasi-random
- The Alon–Boppana bound: $\lambda_2 \geq 2\sqrt{d-1} - o(1)$ — a fundamental limit on expansion
- Ramanujan graphs: optimal expanders achieving $\lambda \leq 2\sqrt{d-1}$
- Explicit constructions: Cayley graphs, Margulis–Gabber–Galil, Lubotzky–Phillips–Sarnak
- Mixing time on expanders: $t_{\mathrm{mix}} = O(\log n)$ — the defining application
- CS applications: derandomization (expander walk sampling), error-correcting codes, network design, GNN architecture

---

## 2. MDX File

### Location

```
src/content/topics/expander-graphs.mdx
```

The entry `id` will be `expander-graphs` (derived from filename by the `glob()` loader in `src/content.config.ts`). The dynamic route at `src/pages/topics/[...slug].astro` resolves this to `/topics/expander-graphs`.

### Frontmatter

```yaml
---
title: "Expander Graphs"
subtitle: "Sparse graphs with paradoxically strong connectivity — from the Expander Mixing Lemma to Ramanujan optimality and O(log n) mixing"
status: "published"
difficulty: "intermediate"
prerequisites:
  - "graph-laplacians"
  - "random-walks"
tags:
  - "graph-theory"
  - "expander-graphs"
  - "spectral-gap"
  - "ramanujan-graphs"
  - "expander-mixing-lemma"
  - "derandomization"
domain: "graph-theory"
videoId: null
notebookPath: "notebooks/expander-graphs/03_expander_graphs.ipynb"
githubUrl: "https://github.com/jonx0037/formalML/blob/main/src/content/topics/expander-graphs.mdx"
datePublished: 2026-03-28
estimatedReadTime: 45
abstract: "Expander graphs are sparse graphs with paradoxically strong connectivity: every vertex subset has a large boundary relative to its size. Three equivalent perspectives — vertex expansion, edge expansion (the Cheeger constant), and spectral gap — capture this idea, linked by Cheeger's inequality. The Expander Mixing Lemma makes expansion quantitative: in an (n, d, λ)-expander, the number of edges between any two vertex subsets S and T deviates from the expected value d|S||T|/n by at most λ√(|S||T|), where λ is the second-largest eigenvalue in absolute value. The Alon–Boppana bound establishes 2√(d-1) as the theoretical floor for λ in d-regular graphs, and Ramanujan graphs — achieving this bound — represent optimal expanders. Explicit constructions from Cayley graphs and algebraic number theory (Lubotzky–Phillips–Sarnak) demonstrate that optimal expanders can be built deterministically. Random walks on expanders mix in O(log n) steps, enabling the expander walk sampling theorem: t steps on an expander yield nearly independent samples using only O(log n + t log d) random bits. These properties power applications from error-correcting codes and derandomization to network design and graph neural network architectures."
connections:
  - topic: "graph-laplacians"
    relationship: "Cheeger's inequality links the Laplacian spectral gap to edge expansion — the same quantity that defines expanders. The Fiedler vector provides a spectral approximation to the minimum cut, and expanders are precisely the graphs where this cut is large for every subset."
  - topic: "random-walks"
    relationship: "The mixing time of a random walk on an expander is O(log n), because the spectral gap γ = 1 − λ/d is bounded away from zero. Expander walk sampling extends this: consecutive walk vertices are nearly independent samples, enabling derandomization."
  - topic: "spectral-theorem"
    relationship: "The Expander Mixing Lemma proof decomposes indicator vectors in the eigenbasis of the adjacency matrix. The Spectral Theorem guarantees orthonormality of this basis — the Cauchy-Schwarz step in the EML relies on this structure."
  - topic: "concentration-inequalities"
    relationship: "The expander walk sampling theorem is a Chernoff bound for dependent samples on a Markov chain. The spectral gap replaces independence: a walk on an expander produces samples whose correlations decay exponentially."
  - topic: "shannon-entropy"
    relationship: "The entropy rate of a random walk on a d-regular expander approaches log d — maximum entropy. Expansion ensures the walk explores the graph uniformly, maximizing the information content of the trajectory."
references:
  - type: "paper"
    title: "Expander Graphs and their Applications"
    authors: "Hoory, Linial & Wigderson"
    year: 2006
    url: "https://doi.org/10.1090/S0273-0979-06-01126-8"
    note: "The definitive survey — covers all three expansion notions, the EML, Ramanujan graphs, and applications to CS"
  - type: "paper"
    title: "Ramanujan Graphs"
    authors: "Lubotzky, Phillips & Sarnak"
    year: 1988
    url: "https://doi.org/10.1007/BF02126799"
    note: "The original explicit construction of (p+1)-regular Ramanujan graphs via quaternion algebras"
  - type: "paper"
    title: "A Proof of Alon's Second Eigenvalue Conjecture and Related Problems"
    authors: "Friedman"
    year: 2008
    url: "https://doi.org/10.1090/memo/0910"
    note: "Random d-regular graphs are nearly Ramanujan — λ₂ ≤ 2√(d-1) + ε with high probability"
  - type: "paper"
    title: "Interlacing Families II: Mixed Characteristic Polynomials and the Kadison–Singer Problem"
    authors: "Marcus, Spielman & Srivastava"
    year: 2015
    url: "https://doi.org/10.4007/annals.2015.182.1.1"
    note: "Existence of bipartite Ramanujan graphs of every degree — resolved a major open problem"
  - type: "book"
    title: "Spectral Graph Theory"
    authors: "Chung"
    year: 1997
    note: "Standard reference for the normalized Laplacian perspective on expansion and mixing"
  - type: "paper"
    title: "Expander Codes"
    authors: "Sipser & Spielman"
    year: 1996
    url: "https://doi.org/10.1109/18.556667"
    note: "Linear-time decodable codes from bipartite expanders — expansion ensures error correction"
---
```

### Content Sections

The MDX body should have the following sections. Pull all mathematical content from the notebook. Each section header maps directly to a notebook section.

| # | MDX Section | Notebook § | Key Content |
|---|---|---|---|
| 1 | Overview & Motivation | §0 (Overview) | Why sparsity + connectivity is paradoxical; what expanders achieve that dense graphs achieve trivially; roadmap from three definitions to applications |
| 2 | Three Notions of Expansion | §1 | Vertex expansion $h_V(G)$; edge expansion (Cheeger constant) $h(G)$; spectral expansion $\lambda(G) = \max_{i \geq 2} |\lambda_i|$; the $(n, d, \lambda)$-expander formalism; concrete examples (complete, Petersen, hypercube, cycle, barbell) |
| 3 | Equivalence of Expansion Notions | §2 | Cheeger's inequality (discrete version for $d$-regular graphs); vertex vs edge expansion inequality; equivalence corollary; visualization of Cheeger bounds |
| 4 | The Expander Mixing Lemma | §3 | Statement and full proof of the EML; decomposition in eigenbasis; Cauchy–Schwarz step; interpretation: edges quasi-random in good expanders; tightness remark |
| 5 | Ramanujan Graphs & the Alon–Boppana Bound | §4 | Alon–Boppana theorem (proof sketch via infinite tree); Ramanujan graph definition; examples (Petersen, complete, cycle); existence theorem (LPS); Friedman's theorem (random regular graphs are nearly Ramanujan) |
| 6 | Explicit Constructions | §5 | Cayley graphs and their spectra; Margulis–Gabber–Galil 8-regular construction; LPS Ramanujan graphs via quaternions; comparison of spectral properties |
| 7 | Random Walks on Expanders | §6 | Mixing time $O(\log n)$ for expanders (proof via spectral bound from Random Walks); comparison table (path, cycle, hypercube, expander); corollary: rapid mixing ↔ expansion |
| 8 | Applications to CS & ML | §7 | Derandomization (expander walk sampling theorem); error-correcting codes (Sipser–Spielman); network robustness; GNN receptive field and over-smoothing |
| 9 | Computational Notes | §8 | NumPy/SciPy eigendecomposition for expansion checking; NetworkX random regular graphs; Monte Carlo expansion estimation; Ramanujan checking |
| 10 | Connections & Further Reading | §9 | Connection table, notation summary, references |

---

## 3. TheoremBlock Usage

Use the existing `TheoremBlock.astro` component for all numbered formal elements. The notebook contains:

| Type | # | Name | TheoremBlock `type` |
|---|---|---|---|
| Definition | 1 | Vertex Expansion | `definition` |
| Definition | 2 | Edge Expansion (Cheeger Constant) | `definition` |
| Definition | 3 | Spectral Expansion | `definition` |
| Definition | 4 | Ramanujan Graph | `definition` |
| Definition | 5 | Cayley Graph | `definition` |
| Theorem | 1 | Cheeger's Inequality (Discrete, $d$-Regular) | `theorem` (with `proof` — both directions) |
| Theorem | 2 | Expander Mixing Lemma | `theorem` (with `proof` — full, using Cauchy–Schwarz) |
| Theorem | 3 | Alon–Boppana Bound | `theorem` (with `proof` sketch — infinite tree argument) |
| Theorem | 4 | Existence of Ramanujan Graphs (LPS) | `theorem` (stated without proof) |
| Theorem | 5 | Mixing on Expanders | `theorem` (with `proof` — via spectral bound from Random Walks) |
| Theorem | 6 | Expander Walk Sampling | `theorem` (stated without proof — Chernoff bound for dependent samples) |
| Proposition | 1 | Vertex vs Edge Expansion | `proposition` (with `proof`) |
| Proposition | 2 | Cayley Graph Spectrum | `proposition` (stated for abelian groups) |
| Corollary | 1 | Equivalence of Expansion Notions | `corollary` |
| Corollary | 2 | Rapid Mixing Implies Expansion | `corollary` |
| Remark | — | Tightness of the EML | `remark` (achieved by bipartite Ramanujan graphs) |
| Remark | — | Friedman's Theorem | `remark` (random $d$-regular graphs are nearly Ramanujan) |
| Remark | — | GNN Over-Smoothing on Expanders | `remark` (over-smoothing happens in $O(\log n)$ layers — bridge to `message-passing`) |

**LaTeX symbols to verify render correctly:**

- `\lambda(G) = \max_{i \geq 2} |\lambda_i|` — Spectral expansion parameter
- `h_V(G) = \min_{|S| \leq n/2} |N(S) \setminus S| / |S|` — Vertex expansion
- `|E(S, T)| - d|S||T|/n| \leq \lambda\sqrt{|S||T|}` — EML
- `2\sqrt{d - 1}` — Alon–Boppana / Ramanujan bound
- `t_{\mathrm{mix}}(\varepsilon) \leq \frac{d}{d - \lambda}(\ln n + \ln(1/\varepsilon))` — Mixing on expanders
- `\mathrm{Cay}(\Gamma, S)` — Cayley graph notation
- `\tilde{P} = \alpha P + (1-\alpha)\frac{1}{n}\mathbf{1}\mathbf{1}^T` — PageRank on expanders

---

## 4. Interactive Visualizations

New React components in `src/components/viz/`. All use `client:visible` hydration. Follow patterns from existing viz components (Approach B — `useEffect` + manual refs for multi-panel layouts).

### 4a. ExpanderExplorer (Flagship)

**File:** `src/components/viz/ExpanderExplorer.tsx`

An interactive visualization comparing expansion properties across graph families, showing how vertex expansion, edge expansion, and spectral gap correlate.

- **Left panel:** A graph drawn with D3 force simulation. Node color encodes a partition: the minimum cut (the subset $S^*$ achieving $h(G)$) is highlighted in two colors (teal for $S^*$, amber for $\bar{S}^*$). Cut edges are drawn as thick dashed lines. Annotations show $|S^*|$, $|E(S^*, \bar{S}^*)|$, and $h(G)$.

- **Right panel:** A three-metric comparison display:
  1. **Vertex expansion $h_V(G)$** — bar with value
  2. **Edge expansion $h(G)$** — bar with value  
  3. **Spectral parameter $\lambda(G)$** — bar with value and the Ramanujan threshold $2\sqrt{d-1}$ marked as a vertical line

  A visual indicator (✓ or ✗) shows whether the graph satisfies $\lambda \leq 2\sqrt{d-1}$ (Ramanujan).

- **Bottom panel:** Eigenvalue spectrum of the adjacency matrix — all eigenvalues plotted on a number line. $\lambda_1 = d$ is marked separately. The Ramanujan window $[-2\sqrt{d-1}, 2\sqrt{d-1}]$ is shaded. Eigenvalues outside this window are highlighted in red.

- **Controls:**
  - **Graph preset** dropdown: Petersen, Hypercube $Q_3$, Hypercube $Q_4$, Cycle(12), Path(12), Barbell(5-5), Random 3-regular(12), Random 4-regular(12), Complete(8), Cayley($\mathbb{Z}_{15}$, $\{\pm 1, \pm 3\}$).
  - **Size parameter** slider (6–16): adjusts $n$ for families that support it.
  - **"Add/Remove random edge"** buttons: modify the graph and show how expansion metrics change.

**Interaction:** Selecting a preset graph recomputes all three expansion metrics, the eigenvalue spectrum, and the minimum cut visualization. The Ramanujan check updates in real time. Adding/removing edges shows how the expansion metrics respond.

**Implementation notes:**
- **Vertex and edge expansion:** Brute-force enumeration for $n \leq 14$ (as in `CheegerExplorer`). For larger graphs, fall back to spectral approximation.
- **Ramanujan check:** Compare $\lambda(G)$ against $2\sqrt{d-1}$. Only valid for $d$-regular graphs — show "N/A (irregular)" for non-regular.
- Use D3 force simulation for graph layout.
- Eigenvalue spectrum: `d3.scaleLinear` for the number line. Shade the Ramanujan window with a rectangle.
- Cap graph size at 16 nodes for interactive performance.
- Data generation is inline. Graph construction helpers from `graphTheory.ts`.

**Reference:** Notebook §1–§2.

### 4b. MixingLemmaExplorer

**File:** `src/components/viz/MixingLemmaExplorer.tsx`

An interactive demonstration of the Expander Mixing Lemma, showing how edge distribution deviates from expectation for different graph families and subset choices.

- **Left panel:** A graph with two highlighted subsets $S$ (teal) and $T$ (amber). Edges between $S$ and $T$ are drawn in bold purple. The rest of the graph is dimmed. Annotations show $|S|$, $|T|$, $|E(S,T)|$, expected $= d|S||T|/n$, and EML bound $= \lambda\sqrt{|S||T|}$.

- **Right panel:** Two vertically stacked plots:
  1. **Deviation bar chart** — for the current $(S, T)$: three bars showing (a) the actual $|E(S,T)|$, (b) the expected value $d|S||T|/n$, (c) the EML bound above/below the expected. A visual check shows whether the deviation is within the bounds.
  2. **Distribution of deviations** — a histogram of $|E(S,T) - \text{expected}|$ for all pairs of subsets of a fixed size (e.g., $|S| = |T| = 3$), with the EML bound marked as a vertical dashed line. Shows that *all* deviations fall within the bounds.

- **Controls:**
  - **Graph preset** dropdown: Petersen (good expander), Cycle $C_{10}$ (poor expander), Random 3-regular(10), Random 4-regular(10).
  - **Subset selection mode:** "Click to select S/T" — click nodes to assign them to $S$ (first clicks) or $T$ (subsequent clicks). A "Random subsets" button generates random $S, T$ of a specified size.
  - **Subset size** slider (2–5): for the random subset generator and the histogram panel.

**Interaction:** Clicking nodes assigns them to $S$ or $T$. The edge count, expected value, and EML bound update in real time. The histogram recomputes on graph change. Toggling between Petersen (tight bound) and Cycle (loose bound) demonstrates the power of expansion.

**Implementation notes:**
- For the histogram panel, enumerate all $\binom{n}{k}$ pairs of subsets of the specified size. For $n = 10, k = 3$, this is $\binom{10}{3}^2 = 14400$ pairs — compute on mount in a web worker or chunked `setTimeout` to avoid blocking.
- Edge counting: iterate over $S \times T$ and sum adjacency entries.
- EML bound uses $\lambda = \max(|\lambda_2|, |\lambda_n|)$, the spectral expansion parameter.
- Use D3 for all panels.

**Reference:** Notebook §3.

### 4c. RamanujanBoundExplorer

**File:** `src/components/viz/RamanujanBoundExplorer.tsx`

An interactive visualization of the Alon–Boppana bound, showing how the second eigenvalue of random regular graphs concentrates around the Ramanujan threshold as $n$ grows.

- **Left panel:** A histogram of $\lambda_2$ values for an ensemble of random $d$-regular graphs at the selected size $n$. The Ramanujan threshold $2\sqrt{d-1}$ is indicated by a vertical, dashed, red line. Graphs with $\lambda_2 \leq 2\sqrt{d-1}$ are colored teal (Ramanujan); those exceeding it are colored amber.

- **Right panel:** Two vertically stacked plots:
  1. **$\lambda_2$ vs $n$** — a box plot (or violin plot) of $\lambda_2$ distributions at $n = 10, 20, 30, 40, 50$ for the selected degree $d$. The Alon–Boppana bound is drawn as a horizontal dashed line. Shows concentration approaching the bound as $n$ grows.
  2. **Ramanujan fraction** — a line plot showing the fraction of random $d$-regular graphs that are Ramanujan at each $n$. Demonstrates that most random regular graphs are nearly Ramanujan (Friedman's theorem).

- **Controls:**
  - **Degree $d$** selector: 3, 4, 5, 6.
  - **Size $n$** slider (8–50): for the histogram in the left panel.
  - **Ensemble size** selector: 50, 100, 200 (number of random graphs to sample).

**Interaction:** Changing $d$ or $n$ triggers re-sampling of the random graph ensemble and updates all panels. The Alon–Boppana line moves with $d$. The concentration phenomenon is visible as $n$ increases.

**Implementation notes:**
- **Random graph generation:** Use a seeded PRNG to generate random $d$-regular graphs. Implement a simple configuration-model-based generator in `graphTheory.ts`, or use the existing `networkx`-style approach already in the module (if the Jacobi solver can handle $n = 50$ matrices — 50×50 eigendecompositions are fast).
- **Pre-computation strategy:** For the right panel, pre-compute ensembles at $n = 10, 20, 30, 40, 50$ on first mount. Store results in a ref. Update only the left panel histogram on $n$ slider change.
- **Violin/box plot:** Use D3 to draw either box plots (median, quartiles, whiskers) or simple dot-strips. Box plots are recommended for clarity.
- Eigendecomposition: Jacobi solver from `graphTheory.ts` (50×50 is within performance budget).

**Reference:** Notebook §4.

---

## 5. Data Modules

### Extend `src/components/viz/shared/graphTheory.ts`

The Expander Graphs topic extends the shared `graphTheory.ts` module created in Graph Laplacians and extended in Random Walks. **Do not create a new file** — add the following exports to the existing module.

```typescript
// === Expander Types ===

export interface ExpanderMetrics {
  vertexExpansion: number;         // h_V(G)
  edgeExpansion: number;           // h(G) = Cheeger constant
  spectralParameter: number;       // λ(G) = max(|λ₂|, |λₙ|) of adjacency matrix
  spectralGapAdj: number;         // d - λ₂ (adjacency spectral gap)
  degree: number;                  // d (for d-regular graphs)
  isRegular: boolean;              // whether the graph is d-regular
  isRamanujan: boolean;            // λ(G) ≤ 2√(d-1)
  ramanujanBound: number;          // 2√(d-1) for d-regular, NaN for irregular
}

export interface EMLResult {
  actualEdges: number;             // |E(S,T)|
  expectedEdges: number;           // d|S||T|/n
  emlBound: number;                // λ√(|S||T|)
  deviation: number;               // |actual - expected|
  withinBound: boolean;            // deviation ≤ emlBound
}

// === Expansion Metrics ===

/**
 * Compute vertex expansion h_V(G) = min_{|S| ≤ n/2} |N(S)\S| / |S|.
 * Brute-force for n ≤ 14, returns Infinity for larger graphs.
 * Also returns the optimal subset S*.
 */
export function vertexExpansion(graph: Graph): { expansion: number; optimalSet: number[] } { ... }

/**
 * Compute edge expansion h(G) = min_{|S| ≤ n/2} |E(S,S^c)| / |S|.
 * Brute-force for n ≤ 14, spectral approximation for larger graphs.
 * Also returns the optimal subset S* and the cut edges.
 */
export function edgeExpansionFull(graph: Graph): { expansion: number; optimalSet: number[]; cutEdges: [number, number][] } { ... }

/**
 * Spectral expansion parameter λ(G) = max(|λ₂|, |λₙ|) of the adjacency matrix.
 */
export function spectralParameter(graph: Graph): number { ... }

/**
 * Full expansion analysis for a d-regular graph.
 */
export function analyzeExpansion(graph: Graph): ExpanderMetrics { ... }

/**
 * Alon–Boppana bound: 2√(d-1).
 */
export function alonBoppanaBound(d: number): number { ... }

// === Expander Mixing Lemma ===

/**
 * Compute the EML quantities for subsets S, T in a d-regular graph.
 * Returns actual edges, expected edges, EML bound, and whether the bound holds.
 */
export function expanderMixingLemma(graph: Graph, S: number[], T: number[]): EMLResult { ... }

/**
 * Compute EML deviations for all pairs of subsets of size k.
 * Used for the histogram in MixingLemmaExplorer.
 * Returns array of { deviation, bound } for each pair.
 */
export function emlAllSubsetPairs(graph: Graph, k: number): { deviation: number; bound: number }[] { ... }

// === Graph Construction (additions) ===

/**
 * Cayley circulant graph: Cay(Z_n, generators).
 * Generators should be symmetric (include both g and -g mod n).
 */
export function cayleyCirulantGraph(n: number, generators: number[]): Graph { ... }

/**
 * Random d-regular graph using the configuration model with retry.
 * Seeded PRNG for reproducibility.
 */
export function randomRegularGraph(n: number, d: number, seed: number): Graph { ... }

/**
 * Adjacency eigenvalues sorted descending (not Laplacian).
 * For d-regular graphs: λ₁ = d, λₙ ∈ [-d, d].
 */
export function adjacencySpectrum(graph: Graph): number[] { ... }
```

**Design decisions:**
- **Vertex expansion is brute-force only for $n \leq 14$.** Beyond that, the $2^n$ enumeration is too slow. Return `Infinity` with a flag indicating the result is a placeholder. The UI should show "N/A (graph too large)" in that case.
- **Edge expansion reuses `CheegerExplorer` logic.** The `edgeExpansionFull` function is equivalent to the Cheeger constant computation from the Graph Laplacians topic, but returns cut edges for visualization.
- **Adjacency eigenvalues sorted descending.** Unlike the Laplacian spectrum (sorted ascending), the adjacency spectrum convention sorts by magnitude: $\lambda_1 = d \geq \lambda_2 \geq \cdots \geq \lambda_n$.
- **Random regular graph generator.** Implement via the configuration model: pair $nd$ half-edges uniformly at random, check for multi-edges and self-loops, retry if invalid. Alternatively, use a stub that calls into a pre-generated array of known-good, random, regular graphs (for deterministic testing). Seed the PRNG explicitly.
- **All computations are direct (not lazy).** Expansion metrics are fast for small graphs ($n \leq 16$).

### Shared types addition: `src/components/viz/shared/types.ts`

Check for conflicts with existing types, then add:

```typescript
export interface ExpanderSubsetPair {
  S: number[];
  T: number[];
  actualEdges: number;
  expectedEdges: number;
  deviation: number;
  bound: number;
}
```

---

## 6. Curriculum Graph Updates

### `src/data/curriculum-graph.json`

**Update node status (from planned to published):**

```json
{
  "id": "expander-graphs",
  "label": "Expander Graphs",
  "domain": "graph-theory",
  "status": "published",
  "url": "/topics/expander-graphs"
}
```

> ⚠️ This node should already exist as `"status": "planned"` from the Graph Laplacians or Random Walks implementation. If so, update the status to `"published"` and ensure the `url` field is present. If it does not exist, add the full node.

**Add prerequisite edges:**

```json
{ "source": "graph-laplacians", "target": "expander-graphs" }
{ "source": "random-walks", "target": "expander-graphs" }
```

> ⚠️ The edge `{ "source": "random-walks", "target": "expander-graphs" }` should already exist from the Random Walks implementation. Verify it is present; add it if not.

**Add cross-track edge:**

```json
{ "source": "concentration-inequalities", "target": "expander-graphs" }
```

**Verify downstream edge exists:**

```json
{ "source": "expander-graphs", "target": "message-passing" }
```

> ⚠️ If this edge does not exist, add it. The `message-passing` node should already exist as planned.

### `src/data/curriculum.ts`

Move `"expander-graphs"` from the `planned` array to the `published` array in the `graph-theory` track.

The track should now show:
- Published: `["graph-laplacians", "random-walks", "expander-graphs"]`
- Planned: `["message-passing"]`

---

## 7. Cross-References

### Outbound (from this page)

**Backward references (these pages exist — use live links):**

- `[Graph Laplacians & Spectrum](/topics/graph-laplacians)` — referenced in: §2 (Cheeger constant $h(G)$ defined and bounded by Cheeger's inequality), §3 (Laplacian spectral gap vs adjacency spectral gap), §6 (spectral clustering contrast — expanders have no bottleneck)
- `[Random Walks & Mixing](/topics/random-walks)` — referenced in: §6 (mixing time $O(\log n)$ on expanders, spectral gap of transition matrix), §7 (expander walk sampling — extending Chernoff bounds to Markov chains)
- `[The Spectral Theorem](/topics/spectral-theorem)` — referenced in: §3 (eigenbasis decomposition in the EML proof, Cauchy-Schwarz step relies on orthonormality)
- `[Concentration Inequalities](/topics/concentration-inequalities)` — referenced in: §7 (Chernoff bounds for independent samples; expander walk sampling extends these to dependent samples with spectral gap correction)
- `[Shannon Entropy & Mutual Information](/topics/shannon-entropy)` — referenced in: §6 (entropy rate of walks on expanders approaches $\log d$)

**Forward references to planned topics (use plain text + "(coming soon)"):**

- **Message Passing & GNNs** *(coming soon)* — "On expanders, $O(\log n)$ GNN layers suffice for each node's representation to aggregate information from all other nodes — but the same property causes over-smoothing at the same logarithmic rate. Expander-based graph rewiring is an active area addressing this tradeoff."

⚠️ **Do NOT create hyperlinks to unbuilt topics.** Use the plain-text pattern: `**Topic Name** *(coming soon)*`.

### Inbound (update existing pages)

- **`graph-laplacians.mdx`:** Check for any plain-text forward reference to **Expander Graphs**. If present (as `**Expander Graphs** *(coming soon)*`), convert to a live link: `[Expander Graphs](/topics/expander-graphs)`. Update the connection text: *"[Expander Graphs](/topics/expander-graphs) studies the graphs that maximize connectivity at fixed sparsity. The spectral gap — bounded below by Cheeger's inequality — is the defining quantity: expanders have spectral gap bounded away from zero."*

- **`random-walks.mdx`:** Check for any plain-text forward reference to **Expander Graphs**. If present (as `**Expander Graphs** *(coming soon)*`), convert to a live link: `[Expander Graphs](/topics/expander-graphs)`. Update the connection text: *"[Expander Graphs](/topics/expander-graphs) are the sparse graphs that mix the fastest — their spectral gap $\gamma$ is bounded away from zero even as $n \to \infty$, giving $O(\log n)$ mixing time. The expander walk sampling theorem extends the mixing perspective to derandomization."*

> **Do not update other existing MDX files.** The cross-track connections (Spectral Theorem, Concentration Inequalities, Shannon Entropy) are backward references only.

---

## 8. Paths Page Update

### Update Existing Track Section

The Graph Theory track section already exists on `/paths` (created by Graph Laplacians, updated by Random Walks). Update:

| Topic | Status | Difficulty | Badge |
|---|---|---|---|
| Graph Laplacians & Spectrum | Published (linked) | Foundational | — |
| Random Walks & Mixing | Published (linked) | Intermediate | — |
| **Expander Graphs** | **Published** (linked to `/topics/expander-graphs`) | **Intermediate** | — |
| Message Passing & GNNs | Planned (gray, unlinked) | Advanced | — |

---

## 9. Images

Export from the notebook and place in:

```
public/images/topics/expander-graphs/
```

| Figure | Notebook Source | Filename |
|---|---|---|
| Three expansion notions compared across graph families | §1 | `expansion-comparison.png` |
| Cheeger inequality scatter for random regular graphs | §2 | `cheeger-expander-scatter.png` |
| Expander Mixing Lemma — edge distribution histograms | §3 | `expander-mixing-lemma.png` |
| Alon–Boppana bound and Ramanujan threshold | §4 | `ramanujan-bound.png` |
| Explicit constructions — Cayley, Margulis, spectra comparison | §5 | `explicit-constructions.png` |
| Mixing time comparison: expanders vs non-expanders | §6 | `mixing-expanders.png` |
| Applications: walk sampling, robustness, GNN receptive field | §7 | `expander-applications.png` |

---

## 10. Code Blocks

Include selected Python snippets from the notebook as syntax-highlighted code blocks in the MDX:

| Block | Notebook Source | Purpose |
|---|---|---|
| Expansion metrics computation | §1 | Brute-force vertex and edge expansion |
| Cheeger inequality verification | §2 | Scatter plot of $(d - \lambda_2, h(G))$ with bounds |
| EML verification | §3 | Compute $|E(S,T)|$ vs expected vs bound for random subsets |
| Ramanujan checking | §4 | Test $\lambda(G) \leq 2\sqrt{d-1}$ for random regular graphs |
| Cayley graph construction | §5 | Build Cayley circulant $\mathrm{Cay}(\mathbb{Z}_n, S)$ |
| Mixing profile on expanders | §6 | TV distance curves for expanders vs paths |

---

## 11. Notebook File

Place the companion notebook in the repo:

```
notebooks/expander-graphs/03_expander_graphs.ipynb
```

This follows the corrected `notebookPath` convention (subdirectory + file). The `03_` prefix indicates this is the third notebook in the Graph Theory track.

---

## 12. SEO & OG

If the OG image generation script is active, it should auto-generate an OG image. If not, create a manual OG image and place it in `public/og/expander-graphs.png`.

Meta tags from `TopicLayout.astro` frontmatter. Verify:
- `<title>`: "Expander Graphs | formalML"
- `<meta name="description">`: Uses the `abstract` field
- OG tags: title, description, image

---

## 13. Testing Checklist

- [ ] All KaTeX renders correctly (especially `\lambda(G) = \max_{i \geq 2} |\lambda_i|`, `h_V(G)`, `|E(S,T) - d|S||T|/n| \leq \lambda\sqrt{|S||T|}`, `2\sqrt{d-1}`, `t_{\mathrm{mix}} \leq \frac{d}{d-\lambda}(\ln n + \ln(1/\varepsilon))`, `\mathrm{Cay}(\Gamma, S)`)
- [ ] Prerequisite chips show: "Graph Laplacians & Spectrum" and "Random Walks & Mixing" as linked chips
- [ ] Topic card appears on `/topics` index with "Intermediate" difficulty badge and `graph-theory` domain tag
- [ ] Graph Theory track on `/paths` shows Expander Graphs as Published (Intermediate, linked)
- [ ] Message Passing & GNNs still shows as Planned (gray, unlinked)
- [ ] `graph-laplacians → expander-graphs` edge renders in the curriculum graph
- [ ] `random-walks → expander-graphs` edge renders in the curriculum graph
- [ ] `concentration-inequalities → expander-graphs` cross-track edge renders in the curriculum graph
- [ ] `expander-graphs → message-passing` edge renders in the curriculum graph
- [ ] `ExpanderExplorer` preset dropdown loads all named graphs correctly
- [ ] `ExpanderExplorer` three expansion metrics update correctly for each preset
- [ ] `ExpanderExplorer` Ramanujan check shows ✓ for Petersen, ✗ for cycle
- [ ] `ExpanderExplorer` eigenvalue spectrum shows Ramanujan window shading
- [ ] `ExpanderExplorer` add/remove edge buttons, update all metrics in real time
- [ ] `ExpanderExplorer` minimum cut partition is visualized correctly (two colors, dashed cut edges)
- [ ] `ExpanderExplorer` shows "N/A (irregular)" for non-regular graphs
- [ ] `MixingLemmaExplorer` click-to-select S/T works correctly
- [ ] `MixingLemmaExplorer` edge count, expected, and EML bound update on selection
- [ ] `MixingLemmaExplorer` histogram shows all deviations within EML bound
- [ ] `MixingLemmaExplorer` Petersen (tight bound) vs Cycle (loose bound) is visually clear
- [ ] `MixingLemmaExplorer` random subset generator works correctly
- [ ] `RamanujanBoundExplorer` histogram updates on degree/size change
- [ ] `RamanujanBoundExplorer` Ramanujan threshold line moves with degree selection
- [ ] `RamanujanBoundExplorer` box plot shows concentration around Alon–Boppana bound
- [ ] `RamanujanBoundExplorer` Ramanujan fraction plot shows high fraction for large $n$
- [ ] Forward reference to **Expander Graphs** in `graph-laplacians.mdx` converted to live link
- [ ] Forward reference to **Expander Graphs** in `random-walks.mdx` converted to live link
- [ ] Forward references to **Message Passing & GNNs** use plain text + "(coming soon)" — no dead links
- [ ] All static figures load from `public/images/topics/expander-graphs/`
- [ ] Extended `graphTheory.ts` module passes TypeScript compilation with no errors
- [ ] Page is responsive (viz components stack vertically on mobile)
- [ ] "Intermediate" difficulty badge is styled correctly
- [ ] Pagefind indexes the new topic on rebuild
- [ ] Build succeeds with zero errors: `pnpm build`

---

## 14. Build Order

1. **Extend `src/components/viz/shared/graphTheory.ts`** — add the expander-specific exports: `vertexExpansion`, `edgeExpansionFull`, `spectralParameter`, `analyzeExpansion`, `alonBoppanaBound`, `expanderMixingLemma`, `emlAllSubsetPairs`, `cayleyCirulantGraph`, `randomRegularGraph`, `adjacencySpectrum`. Write verification tests: e.g., Petersen graph should have $\lambda = 1$, $2\sqrt{2} \approx 2.83$, isRamanujan = true; complete graph $K_n$ should have $\lambda = 1$, but $2\sqrt{n-2}$ exceeds 1 for $n > 5$; EML bound should hold for all subset pairs on the Petersen graph.
2. **Create `expander-graphs.mdx`** with full frontmatter and all markdown/LaTeX content. Use `TheoremBlock` for all formal elements (5 definitions, 6 theorems, 2 propositions, 2 corollaries, 3 remarks). No interactive components yet.
3. Add notebook to `notebooks/expander-graphs/` directory.
4. Export and add static figures to `public/images/topics/expander-graphs/`.
5. Build `ExpanderExplorer.tsx` — flagship component. Start with preset graphs, a three-metric display, and an eigenvalue spectrum, then add interactive edge modification.
6. Build `MixingLemmaExplorer.tsx` — subset selection + EML verification + histogram. Start with preset graphs and random subsets, then add click-to-select interaction.
7. Build `RamanujanBoundExplorer.tsx` — random ensemble generation + histogram + box plot + Ramanujan fraction. Start with a single degree, then add a multi-degree comparison.
8. Embed all components in the MDX at their appropriate section positions.
9. Update `graph-laplacians.mdx` — convert forward references to live links.
10. Update `random-walks.mdx` — convert forward references to live links.
11. Update curriculum graph data — update node status and add edges.
12. Update `/paths` page — change Expander Graphs from Planned to Published.
13. Run testing checklist (§13).
14. Commit and deploy.

---

## Appendix A: Key Differences from the Random Walks Brief

1. **Two prerequisites.** Unlike Random Walks (one prerequisite: `graph-laplacians`), this topic has two prerequisites: `graph-laplacians` and `random-walks`. Both contribute essential machinery (Cheeger's inequality from the former, spectral mixing bounds from the latter).
2. **Cross-track prerequisite edge.** `concentration-inequalities → expander-graphs` is a cross-track edge. This is the first cross-track prerequisite in the Graph Theory track beyond the track-founding edges in Graph Laplacians.
3. **Adjacency spectrum (not Laplacian).** This topic works primarily with the adjacency matrix spectrum (eigenvalues of $A$, sorted descending), rather than the Laplacian spectrum. The Expander Mixing Lemma, the Ramanujan bound, and the $(n, d, \lambda)$-expander formalism all use adjacency eigenvalues. The `adjacencySpectrum` function must sort descending, not ascending.
4. **Random regular graph generation in the browser.** The `RamanujanBoundExplorer` needs to generate dozens of random $d$-regular graphs at sizes up to $n = 50$. The configuration model must handle rejection (multi-edges, self-loops) efficiently. Pre-generating on mount with progress indication is recommended.
5. **Brute-force expansion for $n \leq 14$ only.** Both vertex and edge expansion require enumerating all $2^n$ subsets. Cap at $n = 14$ ($2^{14} = 16384$ subsets). For $n > 14$, show spectral approximation only with a clear UI note.
6. **Two existing pages to update.** Both `graph-laplacians.mdx` and `random-walks.mdx` have forward references to Expander Graphs that must be converted to live links.
7. **Three viz components.** Matching the Random Walks topic in component count. Each component addresses a distinct conceptual area: the definition (ExpanderExplorer), the central theorem (MixingLemmaExplorer), and the optimality bound (RamanujanBoundExplorer).

---

## Appendix B: Graph Theory Track Status After This Brief

| Order | Topic | Difficulty | Prerequisites | Status |
|---|---|---|---|---|
| 1 | Graph Laplacians & Spectrum | Foundational | Spectral Theorem + Shannon Entropy | ✅ Published |
| 2 | Random Walks & Mixing | Intermediate | Graph Laplacians | ✅ Published |
| 3 | **Expander Graphs** (this brief) | Intermediate | Graph Laplacians, Random Walks | 🚧 Ready for implementation |
| 4 | Message Passing & GNNs | Advanced | Graph Laplacians, Random Walks, Expander Graphs | Planned |

DAG edges after implementation:
- `spectral-theorem → graph-laplacians → random-walks → expander-graphs`
- `shannon-entropy → graph-laplacians`
- `random-walks → message-passing`
- `expander-graphs → message-passing`
- `concentration-inequalities → expander-graphs` (cross-track)

Cross-track edges for the final topic:
- `gradient-descent → message-passing` (GNN training via backpropagation)
- `random-walks → message-passing` (over-smoothing = walk convergence)
- `expander-graphs → message-passing` (expander-based graph rewiring)

---

*Brief version: v1 | Last updated: 2026-03-27 | Author: Jonathan Rocha*  
*Reference notebook: `notebooks/expander-graphs/03_expander_graphs.ipynb`*  
*Reference doc: `docs/plans/formalml-handoff-reference.md`*
