# Claude Code Handoff Brief: Graph Laplacians & Spectrum

**Project:** formalML — [formalml.com](https://www.formalml.com)  
**Repo:** `github.com/jonx0037/formalML`  
**Stack:** Astro 6 · React 19 · MDX · Tailwind CSS 4 · D3.js 7 · KaTeX · Vercel  
**Package Manager:** pnpm  
**Status:** Ready for implementation  
**Reference Notebook:** `notebooks/graph-laplacians/01_graph_laplacians.ipynb`  
**Reference Doc:** `docs/plans/formalml-handoff-reference.md`

---

## 1. Objective

Add a new topic page **"Graph Laplacians & Spectrum"** to the **Graph Theory** track on formalml.com. This is the **first topic in the track** — it establishes the entire Graph Theory track infrastructure:

1. **First topic in a new track.** The Graph Theory track does not yet exist on the `/paths` page. This topic creates the track section, registers the domain, and establishes the track's design presence.
2. **Cross-track prerequisites:** `spectral-theorem` (eigendecomposition of symmetric matrices — the Laplacian is real symmetric) and `shannon-entropy` (entropy of stationary distributions, information-theoretic interpretation of the spectral gap).
3. **Downstream:** `random-walks` depends on this topic (the transition matrix $D^{-1}A$ and its spectrum govern mixing, and the normalized Laplacian eigenvectors are the eigenvectors of the random walk matrix). `expander-graphs` depends on both this topic and `random-walks`.
4. **Difficulty:** Foundational — this is the entry point to the Graph Theory track. Visual and concrete examples first (draw graphs, color vertices, show matrices). Algebraic derivations only where essential (e.g., proving $L$ is positive semidefinite). Prerequisite knowledge from `spectral-theorem` (eigenvalues, eigenvectors, diagonalization) and `shannon-entropy` (entropy, divergence) is assumed.

The companion Jupyter Notebook (`01_graph_laplacians.ipynb`) contains the full mathematical content, proofs, code, and figures — use it as the authoritative content source for all definitions, theorems, examples, and code blocks.

**What the topic covers:**

- Graph adjacency matrices, degree matrices, and the unnormalized Laplacian $L = D - A$
- The normalized Laplacian $\mathcal{L} = D^{-1/2} L D^{-1/2}$ and the random walk Laplacian $L_{\text{rw}} = D^{-1}L$
- Spectral properties: $L$ is positive semidefinite, multiplicity of $\lambda = 0$ equals number of connected components, the Fiedler value $\lambda_2$ as algebraic connectivity
- The Fiedler vector and spectral graph bipartitioning — using the second eigenvector to find natural cuts
- Cheeger's inequality: the spectral gap $\lambda_2$ is sandwiched by the Cheeger constant $h(G)$ via $h^2/(2d_{\max}) \leq \lambda_2 \leq 2h$
- Spectral clustering: from similarity graphs to Laplacian eigenmaps to k-means in the spectral embedding
- Connections to ML: GNNs as Laplacian smoothing, graph signal processing, and the bridge to message passing

---

## 2. MDX File

### Location

```
src/content/topics/graph-laplacians.mdx
```

The entry `id` will be `graph-laplacians` (derived from filename by the `glob()` loader in `src/content.config.ts`). The dynamic route at `src/pages/topics/[...slug].astro` resolves this to `/topics/graph-laplacians`.

### Frontmatter

```yaml
---
title: "Graph Laplacians & Spectrum"
subtitle: "The eigenvalues and eigenvectors of graph Laplacians — from connectivity to clustering to graph neural networks"
status: "published"
difficulty: "foundational"
prerequisites:
  - "spectral-theorem"
  - "shannon-entropy"
tags:
  - "graph-theory"
  - "spectral-graph-theory"
  - "graph-laplacian"
  - "cheeger-inequality"
  - "spectral-clustering"
  - "graph-neural-networks"
domain: "graph-theory"
videoId: null
notebookPath: "notebooks/graph-laplacians/01_graph_laplacians.ipynb"
githubUrl: "https://github.com/jonx0037/formalML/blob/main/src/content/topics/graph-laplacians.mdx"
datePublished: 2026-03-28
estimatedReadTime: 40
abstract: "The graph Laplacian L = D - A encodes a graph's connectivity structure in a real symmetric matrix whose spectrum reveals global properties invisible to local inspection. The smallest eigenvalue is always zero, with multiplicity equal to the number of connected components. The second-smallest eigenvalue — the algebraic connectivity or Fiedler value — quantifies how well-connected the graph is, and its eigenvector provides a principled way to bipartition the graph. Cheeger's inequality makes this precise: the spectral gap is bounded above and below by the Cheeger constant, linking linear algebra to combinatorial optimization. The normalized Laplacian connects these ideas to random walks and entropy: its eigenvalues govern mixing times, and the stationary distribution's entropy reflects the graph's regularity. Spectral clustering exploits this theory by embedding vertices into the eigenspace of the Laplacian and clustering in the embedding — the mathematical foundation of modern graph-based learning methods including graph neural networks, where message passing is Laplacian smoothing in disguise."
connections:
  - topic: "spectral-theorem"
    relationship: "The graph Laplacian is a real symmetric matrix. The Spectral Theorem guarantees it has a complete orthonormal eigenbasis with real eigenvalues — the foundation of every result in spectral graph theory."
  - topic: "shannon-entropy"
    relationship: "The entropy of a random walk's stationary distribution measures graph regularity. On a d-regular graph, the stationary distribution is uniform with maximum entropy log(n). The spectral gap governs how quickly the walk's distribution converges to stationarity — how fast entropy increases."
  - topic: "convex-analysis"
    relationship: "The Laplacian quadratic form x^T L x = sum_{(i,j)} w_{ij}(x_i - x_j)^2 is a convex function whose minimization (subject to constraints) yields the Fiedler vector. Graph cuts are combinatorial optimization problems relaxed to spectral problems via convex relaxation."
  - topic: "pca-low-rank"
    relationship: "Spectral clustering uses the bottom eigenvectors of the Laplacian as a low-dimensional embedding — analogous to PCA using the top eigenvectors of the covariance matrix. Both are instances of eigenmap embeddings that preserve specific notions of structure."
references:
  - type: "book"
    title: "Spectral Graph Theory"
    authors: "Chung"
    year: 1997
    note: "The foundational text — covers the normalized Laplacian, Cheeger's inequality, and connections to random walks"
  - type: "book"
    title: "Algebraic Graph Theory"
    authors: "Godsil & Royle"
    year: 2001
    note: "Chapters 8-9 cover the adjacency and Laplacian spectra with algebraic rigor"
  - type: "paper"
    title: "A Tutorial on Spectral Clustering"
    authors: "von Luxburg"
    year: 2007
    url: "https://doi.org/10.1007/s11222-007-9033-z"
    note: "The standard reference for spectral clustering — bridges graph Laplacian theory to machine learning practice"
  - type: "paper"
    title: "Semi-Supervised Classification with Graph Convolutional Networks"
    authors: "Kipf & Welling"
    year: 2017
    url: "https://arxiv.org/abs/1609.02907"
    note: "GCN = renormalized Laplacian smoothing — the paper that made graph Laplacians central to deep learning"
  - type: "paper"
    title: "The Laplacian Spectrum of Graphs"
    authors: "Mohar"
    year: 1991
    url: "https://doi.org/10.1007/978-3-642-58219-8_12"
    note: "Survey of Laplacian eigenvalue bounds and their combinatorial interpretations"
  - type: "book"
    title: "Graph Signal Processing: Overview, Challenges, and Applications"
    authors: "Ortega, Frossard, Kovačević, Moura & Vandergheynst"
    year: 2018
    note: "Graph Fourier transform = eigenvector expansion of the Laplacian"
---
```

### Content Sections

The MDX body should have the following sections. Pull all mathematical content from the notebook. Each section header maps directly to a notebook section.

| # | MDX Section | Notebook § | Key Content |
|---|---|---|---|
| 1 | Overview & Motivation | §0 (Overview) | Why graphs need linear algebra; the adjacency matrix is not enough; what eigenvalues reveal that degree sequences don't; roadmap |
| 2 | Graphs, Adjacency Matrices, and the Degree Matrix | §1 | Graph notation $(V, E, W)$; adjacency matrix $A$; degree matrix $D = \text{diag}(d_1, \ldots, d_n)$; weighted vs unweighted; directed vs undirected (we restrict to undirected); concrete examples (path, cycle, complete, star, barbell) |
| 3 | The Graph Laplacian | §2 | Unnormalized Laplacian $L = D - A$; the quadratic form $\mathbf{x}^T L \mathbf{x} = \sum_{(i,j) \in E} w_{ij}(x_i - x_j)^2$ (proof); $L$ is positive semidefinite (proof); $L\mathbf{1} = \mathbf{0}$ (proof); $\lambda_1 = 0$ always |
| 4 | Spectral Properties of the Laplacian | §3 | Eigenvalue ordering $0 = \lambda_1 \leq \lambda_2 \leq \cdots \leq \lambda_n$; multiplicity of zero = number of connected components (proof); Fiedler value $\lambda_2$ (algebraic connectivity); Fiedler vector; upper bounds on $\lambda_n$; spectra of named graphs (path, cycle, complete, star) |
| 5 | The Normalized Laplacian | §4 | Definition $\mathcal{L} = D^{-1/2}LD^{-1/2} = I - D^{-1/2}AD^{-1/2}$; random walk Laplacian $L_{\text{rw}} = D^{-1}L = I - D^{-1}A$; eigenvalue range $[0, 2]$; when $\lambda_n = 2$ (bipartite graphs); connection to transition matrix $P = D^{-1}A$ |
| 6 | Cheeger's Inequality | §5 | Cheeger constant (isoperimetric number) $h(G) = \min_{S} |E(S, \bar{S})| / \min(\text{vol}(S), \text{vol}(\bar{S}))$; Cheeger's inequality $h^2/(2) \leq \lambda_2 \leq 2h$ for the normalized Laplacian (proof of easy direction $\lambda_2 \leq 2h$; proof sketch of hard direction); interpretation: spectral gap ↔ bottleneck |
| 7 | Spectral Clustering | §6 | From similarity graph to Laplacian; unnormalized spectral clustering algorithm (Laplacian eigenmaps → k-means); normalized variants (Shi–Malik, Ng–Jordan–Weiss); why spectral clustering works (block diagonal Laplacian for ideal clusters); practical considerations |
| 8 | Connections to Machine Learning | §7 | Graph signal processing (graph Fourier transform = eigenvector expansion); GCN as Laplacian smoothing ($\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}X$ is one step of smoothing on the renormalized graph); bridge to message passing |
| 9 | Computational Notes | §8 | NumPy/SciPy eigendecomposition of $L$; sparse Laplacian construction; `scipy.sparse.csgraph.laplacian`; `networkx` spectral methods; scikit-learn `SpectralClustering`; numerical considerations for near-zero eigenvalues |
| 10 | Connections & Further Reading | §9 | Connection table, notation summary, references |

---

## 3. TheoremBlock Usage

Use the existing `TheoremBlock.astro` component for all numbered formal elements. The notebook contains:

| Type | # | Name | TheoremBlock `type` |
|---|---|---|---|
| Definition | 1 | Graph (undirected, weighted) | `definition` |
| Definition | 2 | Adjacency Matrix | `definition` |
| Definition | 3 | Degree Matrix | `definition` |
| Definition | 4 | Graph Laplacian (unnormalized) | `definition` |
| Definition | 5 | Normalized Laplacian | `definition` |
| Definition | 6 | Random Walk Laplacian | `definition` |
| Definition | 7 | Algebraic Connectivity (Fiedler Value) | `definition` |
| Definition | 8 | Cheeger Constant (Isoperimetric Number) | `definition` |
| Theorem | 1 | Laplacian Quadratic Form | `theorem` (with `proof`) |
| Theorem | 2 | Positive Semidefiniteness of $L$ | `theorem` (with `proof`) |
| Theorem | 3 | Zero Eigenvalue Multiplicity | `theorem` (with `proof`) |
| Theorem | 4 | Fiedler's Theorem (Spectral Bipartitioning) | `theorem` (with `proof` sketch) |
| Theorem | 5 | Normalized Laplacian Eigenvalue Range | `theorem` (with `proof`) |
| Theorem | 6 | Cheeger's Inequality | `theorem` (with `proof` — easy direction full, hard direction sketch) |
| Theorem | 7 | Bipartiteness and $\lambda_n = 2$ | `theorem` (with `proof`) |
| Corollary | 1 | Disconnected Graphs Have $\lambda_2 = 0$ | `corollary` |
| Corollary | 2 | Complete Graph Spectrum | `corollary` |
| Proposition | 1 | Spectra of Named Graphs | `proposition` (path, cycle, star — stated without proof, verified computationally) |
| Proposition | 2 | Laplacian of $d$-regular Graphs | `proposition` (with `proof` — $L = dI - A$, spectrum shifted) |
| Remark | — | Graph Laplacian as Discrete Laplace Operator | `remark` (connection to $\nabla^2$ on manifolds via Differential Geometry track) |
| Remark | — | Why "Spectral" in Spectral Clustering | `remark` (eigenvalues = spectrum in functional analysis) |
| Remark | — | GCN as Laplacian Smoothing | `remark` (Kipf & Welling's renormalization trick) |

**LaTeX symbols to verify render correctly:**

- `L = D - A` — Laplacian definition
- `\mathcal{L} = D^{-1/2} L D^{-1/2}` — Normalized Laplacian
- `L_{\text{rw}} = D^{-1}L = I - D^{-1}A` — Random walk Laplacian
- `\mathbf{x}^T L \mathbf{x} = \sum_{(i,j) \in E} w_{ij}(x_i - x_j)^2` — Quadratic form
- `0 = \lambda_1 \leq \lambda_2 \leq \cdots \leq \lambda_n` — Eigenvalue ordering
- `h(G) = \min_{S \subset V} \frac{|E(S, \bar{S})|}{\min(\mathrm{vol}(S), \mathrm{vol}(\bar{S}))}` — Cheeger constant
- `\frac{h(G)^2}{2} \leq \lambda_2(\mathcal{L}) \leq 2h(G)` — Cheeger's inequality
- `\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}` — Renormalized adjacency (GCN)

---

## 4. Interactive Visualizations

New React components in `src/components/viz/`. All use `client:visible` hydration. Follow patterns from existing viz components (Approach B — `useEffect` + manual refs for multi-panel layouts).

### 4a. GraphLaplacianExplorer (Flagship)

**File:** `src/components/viz/GraphLaplacianExplorer.tsx`

An interactive graph editor that shows the adjacency matrix, degree matrix, and Laplacian in real time, with the Fiedler vector coloring the nodes to reveal graph structure.

- **Left panel:** Force-directed graph visualization using D3 force simulation. Nodes are circles, edges are lines. Node color encodes the Fiedler vector value (diverging color scale: blue → white → red). Users can:
  - **Click empty space** to add a node.
  - **Click two nodes** to toggle an edge between them.
  - **Drag nodes** to rearrange the layout.
  - **Delete** a node by double-clicking.

- **Right panel:** Matrix display — shows $A$, $D$, and $L = D - A$ side by side (or stacked on mobile). Cells are color-coded by value. Updates in real time as the graph is edited.

- **Bottom panel:** Eigenvalue bar chart — horizontal bars for $\lambda_1, \lambda_2, \ldots, \lambda_n$. The $\lambda_2$ bar is highlighted (the Fiedler value). A text annotation shows the current algebraic connectivity and connectivity status.

- **Controls:**
  - **Preset graph** dropdown: Path(6), Cycle(6), Complete(6), Star(6), Barbell(3-3), Petersen, Grid(3×3).
  - **Coloring mode** toggle: "Fiedler vector" (default), "Degree", "None".
  - **Matrix view** toggle: Show/hide the matrix panel (for cleaner view on mobile).

**Interaction:** Every graph edit (add/remove node or edge) triggers a recomputation of $A$, $D$, $L$, and the eigendecomposition. The node colors and eigenvalue chart update immediately. Preset selector replaces the current graph entirely.

**Implementation notes:**
- Use D3 force simulation (`d3.forceSimulation`) for the graph layout — same physics-based approach as the curriculum DAG, but simpler.
- Eigendecomposition: use the Jacobi eigenvalue algorithm for symmetric matrices, implemented in `graphTheory.ts` (see §5). Matrices are at most ~15×15 for interactive performance.
- Fiedler vector coloring: `d3.scaleDiverging(d3.interpolateRdBu)` centered at 0.
- Matrix display: render as a grid of `<rect>` elements in SVG with `d3.scaleSequential` color mapping.
- Cap graph size at 15 nodes — show a warning if the user tries to add more.

**Reference:** Notebook §1–§3.

### 4b. SpectralGapExplorer

**File:** `src/components/viz/SpectralGapExplorer.tsx`

An interactive visualization of how graph topology affects the spectral gap $\lambda_2$, demonstrating the connection between algebraic connectivity and graph structure.

- **Left panel:** A graph drawn with D3 force simulation. Edge thickness encodes weight (for weighted graphs). A "bridge edge" is highlighted in red when the graph has a clear bottleneck.

- **Right panel:** Two vertically stacked plots:
  1. **Eigenvalue spectrum** — all eigenvalues as dots on a number line, with $\lambda_2$ highlighted and labeled.
  2. **$\lambda_2$ evolution** — as the user adjusts a parameter (e.g., adds edges to a barbell), $\lambda_2$ is plotted over time, showing how connectivity improves.

- **Controls:**
  - **Graph family** selector: Path($n$), Cycle($n$), Complete($n$), Barbell($k$-bridge-$k$), Grid($m \times m$), Random Erdős–Rényi($n$, $p$).
  - **Size parameter** slider ($n$, $k$, $m$, or $p$ depending on family): adjusts graph size/density.
  - **"Add random edge"** button: adds an edge between two non-adjacent vertices. Shows $\lambda_2$ increasing.
  - **"Remove random edge"** button: removes a random non-bridge edge. Shows $\lambda_2$ decreasing.

**Interaction:** Changing the graph family or size parameter rebuilds the graph and recomputes the spectrum. The "add/remove edge" buttons modify the current graph incrementally, and the $\lambda_2$ evolution plot traces the history.

**Implementation notes:**
- Erdős–Rényi: use seeded PRNG to generate random edges with probability $p$.
- The evolution plot stores a history array of $\lambda_2$ values (up to 50 data points).
- Named graph eigenvalues can be computed analytically for the eigenvalue spectrum display (as a verification), but the Jacobi solver handles arbitrary graphs.
- Data generation is inline. Graph construction helpers from `graphTheory.ts`.

**Reference:** Notebook §3–§4.

### 4c. CheegerExplorer

**File:** `src/components/viz/CheegerExplorer.tsx`

An interactive visualization of Cheeger's inequality, showing the relationship between the spectral gap and the combinatorial bottleneck.

- **Left panel:** A graph with the **minimum cut** highlighted. The two sides of the cut are colored differently (blue/red). The cut edges are drawn as thick dashed lines. The Cheeger constant $h(G)$ is displayed prominently.

- **Right panel:** A number line or bar chart showing the Cheeger inequality bounds:
  - Left bound: $h(G)^2 / 2$
  - The actual $\lambda_2$
  - Right bound: $2h(G)$
  
  All three values are plotted, with the inequality $h^2/2 \leq \lambda_2 \leq 2h$ shown visually as a range.

- **Bottom panel:** A scatter plot of (Cheeger constant, $\lambda_2$) for a family of random graphs, with the Cheeger inequality bounds drawn as curves. This shows that the inequality is tight for some graphs and loose for others.

- **Controls:**
  - **Graph preset** dropdown: Barbell(5-1-5), Path(10), Cycle(10), Grid(4×4), Two-cliques-bridge(5-5), Expander-like (random 3-regular).
  - **Cut viewer** toggle: highlight the minimum cut partition vs. the Fiedler vector partition (to show they often agree).

**Interaction:** Selecting a preset graph recomputes everything. Toggling between the minimum cut and Fiedler partition shows how close the spectral approximation is to the combinatorial optimum.

**Implementation notes:**
- **Cheeger constant computation:** For small graphs (≤ 15 nodes), enumerate all cuts by iterating over subsets $S$ with $|S| \leq n/2$ (at most $2^{14}$ subsets — feasible in the browser for $n \leq 14$). For larger presets, use a greedy approximation.
- **Minimum cut visualization:** Color nodes by partition, thicken cut edges.
- **Scatter plot:** Pre-generate 100 random graphs (Erdős–Rényi, $n = 10$, $p$ varying from 0.15 to 0.8) on mount. Compute $(h(G), \lambda_2)$ for each. Store in a ref.
- Use D3 for all panels.

**Reference:** Notebook §5.

### 4d. SpectralClusteringDemo

**File:** `src/components/viz/SpectralClusteringDemo.tsx`

An interactive demonstration of the spectral clustering pipeline, showing each step from data to graph to embedding to clusters.

- **Panel 1 (Input):** A 2D scatter plot of points from a dataset with non-convex cluster structure (e.g., two concentric rings, two moons, or a spiral). The user can select the dataset.

- **Panel 2 (Similarity Graph):** The $k$-nearest-neighbor or $\varepsilon$-ball similarity graph constructed from the points. Edges drawn between similar points. Edge opacity encodes similarity weight.

- **Panel 3 (Spectral Embedding):** The bottom $k$ eigenvectors of the normalized Laplacian, plotted as a 2D scatter (using eigenvectors 2 and 3 as coordinates). Points that were non-linearly separable in the original space become linearly separable in the spectral embedding.

- **Panel 4 (Result):** The original data colored by cluster assignment from k-means on the spectral embedding. Correctly separates the non-convex clusters.

- **Controls:**
  - **Dataset** dropdown: Two Moons, Two Circles, Three Blobs, Spiral.
  - **Number of clusters $k$** selector: 2, 3, 4.
  - **Graph construction** toggle: "$k$-NN" (default, $k = 7$) vs "$\varepsilon$-ball."
  - **Noise level** slider: adds Gaussian noise to the dataset.

**Interaction:** Changing the dataset or parameters recomputes the entire pipeline. The four panels update together, showing how each step transforms the data.

**Implementation notes:**
- Dataset generation: seeded PRNG, 150 points per dataset. `twoMoons`, `twoCircles`, `threeBlobs`, `spiral` generator functions in `graphTheory.ts`.
- $k$-NN graph: compute pairwise Euclidean distances, connect each point to its $k$ nearest neighbors (symmetrize).
- Gaussian similarity weights: $w_{ij} = \exp(-\|x_i - x_j\|^2 / (2\sigma^2))$ with $\sigma$ = median distance.
- Eigendecomposition of the normalized Laplacian: Jacobi solver from `graphTheory.ts`. For 150 nodes, this is a 150×150 matrix — Jacobi may be slow. **Optimization:** use the Lanczos algorithm (compute only the bottom $k$ eigenvalues/vectors) via a simple TypeScript implementation, or pre-compute eigendecompositions for preset datasets and cache them.
- k-means: simple Lloyd's algorithm with $k$-means++ initialization, 20 iterations.
- Use D3 for all four panels.

**Reference:** Notebook §6.

---

## 5. Data Modules

### New shared utility module: `src/components/viz/shared/graphTheory.ts`

This is a **new** shared utility module for the Graph Theory track (analogous to `informationTheory.ts` for the Information Theory track). It will be extended by subsequent Graph Theory topics (`random-walks`, `expander-graphs`, `message-passing`).

```typescript
// === Types ===

export interface Graph {
  n: number;                    // Number of vertices
  adjacency: number[][];        // n×n adjacency matrix (symmetric, zero diagonal)
  labels?: string[];            // Optional vertex labels
}

export interface EigenResult {
  eigenvalues: number[];        // Sorted ascending
  eigenvectors: number[][];     // eigenvectors[i] = i-th eigenvector (column)
}

export interface GraphSpectrum {
  laplacian: number[][];
  normalizedLaplacian: number[][];
  eigen: EigenResult;
  normalizedEigen: EigenResult;
  fiedlerValue: number;
  fiedlerVector: number[];
  numComponents: number;
}

export interface CheegerResult {
  cheegerConstant: number;
  optimalPartition: [number[], number[]];
  cutEdges: [number, number][];
}

// === Graph Construction ===

/** Create a path graph P_n */
export function pathGraph(n: number): Graph { ... }

/** Create a cycle graph C_n */
export function cycleGraph(n: number): Graph { ... }

/** Create a complete graph K_n */
export function completeGraph(n: number): Graph { ... }

/** Create a star graph S_n (one hub + n-1 leaves) */
export function starGraph(n: number): Graph { ... }

/** Create a barbell graph: two complete graphs K_k connected by a single bridge edge */
export function barbellGraph(k: number): Graph { ... }

/** Create a grid graph m × m */
export function gridGraph(m: number): Graph { ... }

/** Create an Erdős–Rényi random graph G(n, p) with seeded PRNG */
export function erdosRenyiGraph(n: number, p: number, seed: number): Graph { ... }

/** Create a k-NN similarity graph from 2D points */
export function knnGraph(points: [number, number][], k: number): Graph { ... }

/** Create an ε-ball similarity graph from 2D points */
export function epsilonBallGraph(points: [number, number][], epsilon: number): Graph { ... }

// === Matrix Operations ===

/** Degree matrix D = diag(row sums of A) */
export function degreeMatrix(A: number[][]): number[][] { ... }

/** Unnormalized Laplacian L = D - A */
export function laplacian(A: number[][]): number[][] { ... }

/** Normalized Laplacian L_sym = D^{-1/2} L D^{-1/2} = I - D^{-1/2} A D^{-1/2} */
export function normalizedLaplacian(A: number[][]): number[][] { ... }

/** Random walk Laplacian L_rw = D^{-1} L = I - D^{-1} A */
export function randomWalkLaplacian(A: number[][]): number[][] { ... }

/** Laplacian quadratic form x^T L x */
export function quadraticForm(L: number[][], x: number[]): number { ... }

// === Eigendecomposition ===

/**
 * Jacobi eigenvalue algorithm for real symmetric matrices.
 * Returns eigenvalues sorted ascending and corresponding eigenvectors.
 * Suitable for matrices up to ~20×20 in real time.
 */
export function jacobiEigen(M: number[][], maxIter?: number, tol?: number): EigenResult { ... }

// === Spectral Analysis ===

/** Full spectral analysis of a graph */
export function analyzeSpectrum(graph: Graph): GraphSpectrum { ... }

/** Number of connected components (count eigenvalues ≈ 0) */
export function countComponents(eigenvalues: number[], tol?: number): number { ... }

// === Cheeger Constant ===

/**
 * Compute the Cheeger constant h(G) by enumeration for small graphs (n ≤ 14).
 * Returns the constant, optimal partition, and cut edges.
 */
export function cheegerConstant(graph: Graph): CheegerResult { ... }

// === Dataset Generators ===

/** Two moons dataset with adjustable noise */
export function twoMoons(n: number, noise: number, seed: number): [number, number][] { ... }

/** Two concentric circles with adjustable noise */
export function twoCircles(n: number, noise: number, seed: number): [number, number][] { ... }

/** Three Gaussian blobs */
export function threeBlobs(n: number, noise: number, seed: number): [number, number][] { ... }

/** Spiral dataset */
export function spiral(n: number, noise: number, seed: number): [number, number][] { ... }

// === Clustering ===

/** k-means clustering with k-means++ initialization */
export function kMeans(points: number[][], k: number, maxIter?: number, seed?: number): number[] { ... }
```

**Design decisions:**
- **All computations are direct (not lazy).** Graph operations are fast for small matrices (n ≤ 150). No caching needed — re-compute on every parameter change.
- **Seeded PRNG everywhere.** Use a simple LCG for `erdosRenyiGraph`, dataset generators, and k-means initialization. Never `Math.random()`.
- **Jacobi eigenvalue algorithm.** Chosen over QR because it's simpler to implement correctly in TypeScript, naturally handles symmetric matrices, and is fast enough for n ≤ 150.
- **Cheeger constant by enumeration.** Only for n ≤ 14 (at most 2^13 subsets to check). For larger graphs, fall back to the Fiedler vector partition as an approximation.

### Shared types addition: `src/components/viz/shared/types.ts`

Add the following types (check for conflicts with existing types first):

```typescript
export interface GraphEdge {
  source: number;
  target: number;
  weight: number;
}

export interface GraphPartition {
  sides: [number[], number[]];
  cutSize: number;
  cheegerRatio: number;
}
```

---

## 6. Curriculum Graph Updates

### `src/data/curriculum-graph.json`

**Add node (new topic — first in a new track):**

```json
{
  "id": "graph-laplacians",
  "label": "Graph Laplacians & Spectrum",
  "domain": "graph-theory",
  "status": "published",
  "url": "/topics/graph-laplacians"
}
```

**Add cross-track prerequisite edges:**

```json
{ "source": "spectral-theorem", "target": "graph-laplacians" }
{ "source": "shannon-entropy", "target": "graph-laplacians" }
```

**Add within-track downstream edge (to planned topic):**

```json
{ "source": "graph-laplacians", "target": "random-walks" }
```

> ⚠️ Check if `random-walks` exists as a node in the curriculum graph (it may have been added as a planned node when the `/paths` page was created). If not, add it:
> ```json
> { "id": "random-walks", "label": "Random Walks & Mixing", "domain": "graph-theory", "status": "planned", "url": "/topics/random-walks" }
> ```

### `src/data/curriculum.ts`

1. **Register the `graph-theory` domain track** if it doesn't exist yet. Follow the pattern of existing tracks. The track should include:
   - Track name: "Graph Theory"
   - Track description: "Spectral graph theory and random walks — the mathematical foundation of graph neural networks."
   - Published topics: `["graph-laplacians"]`
   - Planned topics: `["random-walks", "expander-graphs", "message-passing"]`

2. If the track already exists as a planned track (with all four topics in the `planned` array), move `"graph-laplacians"` from `planned` to `published`.

---

## 7. Cross-References

### Outbound (from this page)

**Backward references (these pages exist — use live links):**

- `[The Spectral Theorem](/topics/spectral-theorem)` — referenced in: §2 (the Laplacian is real symmetric → has a complete eigenbasis), §3 (eigenvalue ordering, positive semidefiniteness), §4 (Fiedler's theorem uses the variational characterization of eigenvalues)
- `[Shannon Entropy & Mutual Information](/topics/shannon-entropy)` — referenced in: §5 (entropy of the random walk stationary distribution; uniform stationary distribution on regular graphs has maximum entropy), §8 (graph entropy)
- `[PCA & Low-Rank Approximation](/topics/pca-low-rank)` — referenced in: §6 (spectral clustering as "PCA on the Laplacian" — bottom eigenvectors as embedding, analogous to top eigenvectors of the covariance matrix)
- `[Convex Analysis](/topics/convex-analysis)` — referenced in: §5 (Cheeger's inequality proof uses convex relaxation of the discrete optimization problem), §6 (spectral clustering as a relaxation of the NP-hard graph cut problem)

**Forward references to planned topics (use plain text + "(coming soon)"):**

- **Random Walks & Mixing** *(coming soon)* — "The transition matrix $P = D^{-1}A$ and its spectrum (related to the random walk Laplacian eigenvalues) govern mixing times. The spectral gap $1 - \lambda_2(P)$ determines how quickly the walk forgets its starting position."
- **Expander Graphs** *(coming soon)* — "Graphs with large spectral gap (relative to degree) are expanders — sparse graphs that are as well-connected as dense ones. Cheeger's inequality provides the bridge."
- **Message Passing & GNNs** *(coming soon)* — "GCN message passing is equivalent to one step of Laplacian smoothing with the renormalized adjacency matrix."

⚠️ **Do NOT create hyperlinks to unbuilt topics.** Use the plain-text pattern: `**Topic Name** *(coming soon)*`.

### Inbound (update existing pages)

- **`spectral-theorem.mdx`:** In the Connections & Further Reading section, add a reference: *"The [Graph Laplacians & Spectrum](/topics/graph-laplacians) topic applies eigendecomposition to the graph Laplacian $L = D - A$, where the spectral properties of this real symmetric matrix reveal graph connectivity, clustering structure, and the foundations of graph neural networks."* Only add if there is a natural insertion point — the Spectral Theorem topic should already mention "spectral graph theory" as a forward reference (possibly as plain text). Convert to a live link.

- **`shannon-entropy.mdx`:** If there is a reference to graph entropy, random walks, or information-theoretic graph analysis, add a link: *"[Graph Laplacians & Spectrum](/topics/graph-laplacians) connects entropy to graph structure through the stationary distribution of random walks on graphs."* Only add if there is a natural insertion point — do not force.

> **Do not update other existing MDX files.** The PCA and Convex Analysis connections are backward references only — those topics already exist and don't need to know about this new topic unless they explicitly forward-reference it.

---

## 8. Paths Page Update

### Create New Track Section

The **Graph Theory** track does NOT yet exist on `/paths`. Create a new track section:

**Track header:**
- Anchor: `#graph-theory`
- Title: "Graph Theory"
- Description: "Spectral graph theory and random walks — the mathematical foundation of graph neural networks."

**Track topics:**

| Topic | Status | Difficulty | Badge |
|---|---|---|---|
| **Graph Laplacians & Spectrum** | **Published** (linked to `/topics/graph-laplacians`) | **Foundational** | **Start here** |
| Random Walks & Mixing | Planned (gray, unlinked) | Intermediate | — |
| Expander Graphs | Planned (gray, unlinked) | Intermediate | — |
| Message Passing & GNNs | Planned (gray, unlinked) | Advanced | — |

> ⚠️ This is the first topic in a new track. The `/paths` page infrastructure for this track needs to be created from scratch. Follow the exact pattern of existing track sections (Topology & TDA, Linear Algebra, etc.) for layout, styling, and status badges.

---

## 9. Images

Export from the notebook and place in:

```
public/images/topics/graph-laplacians/
```

| Figure | Notebook Source | Filename |
|---|---|---|
| Named graphs gallery (path, cycle, complete, star, barbell, grid) | §1 | `named-graphs-gallery.png` |
| Adjacency, degree, and Laplacian matrices for a small graph | §2 | `laplacian-matrices.png` |
| Laplacian spectrum of named graphs | §3 | `laplacian-spectra.png` |
| Fiedler vector coloring on several graph topologies | §3 | `fiedler-vector-coloring.png` |
| Normalized vs unnormalized Laplacian eigenvalues | §4 | `normalized-vs-unnormalized.png` |
| Cheeger's inequality (scatter plot + bounds) | §5 | `cheeger-inequality.png` |
| Spectral clustering pipeline (four-panel: data → graph → embedding → clusters) | §6 | `spectral-clustering-pipeline.png` |
| GCN as Laplacian smoothing (before/after feature propagation) | §7 | `gcn-laplacian-smoothing.png` |

---

## 10. Code Blocks

Include selected Python snippets from the notebook as syntax-highlighted code blocks in the MDX:

| Block | Notebook Source | Purpose |
|---|---|---|
| Laplacian construction | §2 | Build $L = D - A$ from an adjacency matrix using NumPy |
| Eigendecomposition | §3 | `np.linalg.eigh` on $L$, sort eigenvalues, identify Fiedler value |
| Normalized Laplacian | §4 | Build $\mathcal{L} = D^{-1/2}LD^{-1/2}$ with sparse matrices |
| Spectral clustering pipeline | §6 | Full `sklearn.cluster.SpectralClustering` on two-moons data |
| GCN propagation step | §7 | Single GCN layer $\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}XW$ in PyTorch |

---

## 11. Notebook File

Place the companion notebook in the repo:

```
notebooks/graph-laplacians/01_graph_laplacians.ipynb
```

This follows the corrected `notebookPath` convention (subdirectory + file). The `01_` prefix indicates this is the first notebook in the Graph Theory track.

---

## 12. SEO & OG

If the OG image generation script is active, it should auto-generate an OG image. If not, create a manual OG image and place it in `public/og/graph-laplacians.png`.

Meta tags from `TopicLayout.astro` frontmatter. Verify:
- `<title>`: "Graph Laplacians & Spectrum | formalML"
- `<meta name="description">`: Uses the `abstract` field
- OG tags: title, description, image

---

## 13. Testing Checklist

- [ ] All KaTeX renders correctly (especially `L = D - A`, `\mathcal{L} = D^{-1/2}LD^{-1/2}`, `L_{\text{rw}} = D^{-1}L`, `\mathbf{x}^T L \mathbf{x}`, `\lambda_2(\mathcal{L})`, `h(G)`, `\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}`)
- [ ] Prerequisite chips show: "The Spectral Theorem" and "Shannon Entropy & Mutual Information" as linked chips (cross-track prerequisites)
- [ ] Topic card appears on `/topics` index with "Foundational" difficulty badge and `graph-theory` domain tag
- [ ] **New Graph Theory track section appears on `/paths`** with correct header, description, and anchor `#graph-theory`
- [ ] Graph Laplacians & Spectrum appears as Published (Foundational, linked) with "Start here" badge
- [ ] Random Walks & Mixing, Expander Graphs, and Message Passing & GNNs appear as Planned (gray, unlinked)
- [ ] `spectral-theorem → graph-laplacians` edge renders in the curriculum graph
- [ ] `shannon-entropy → graph-laplacians` edge renders in the curriculum graph
- [ ] `GraphLaplacianExplorer` preset dropdown loads all named graphs correctly
- [ ] `GraphLaplacianExplorer` node add/remove/edge toggle update matrices and spectrum in real time
- [ ] `GraphLaplacianExplorer` Fiedler vector coloring correctly identifies graph bipartition
- [ ] `GraphLaplacianExplorer` shows λ₂ = 0 for disconnected graphs (e.g., remove bridge from barbell)
- [ ] `GraphLaplacianExplorer` matrix panel renders correctly (colors, labels, values)
- [ ] `SpectralGapExplorer` graph family selector rebuilds graph correctly
- [ ] `SpectralGapExplorer` size slider updates graph and spectrum
- [ ] `SpectralGapExplorer` "add/remove edge" buttons correctly track λ₂ history
- [ ] `SpectralGapExplorer` Erdős–Rényi graphs are connected (regenerate if disconnected)
- [ ] `CheegerExplorer` minimum cut is correctly highlighted for all presets
- [ ] `CheegerExplorer` Cheeger inequality bounds are correctly computed (h²/2 ≤ λ₂ ≤ 2h)
- [ ] `CheegerExplorer` Fiedler partition vs minimum cut toggle works
- [ ] `CheegerExplorer` scatter plot shows all points between the Cheeger bounds
- [ ] `SpectralClusteringDemo` four panels update together
- [ ] `SpectralClusteringDemo` correctly separates non-convex clusters (two moons, circles)
- [ ] `SpectralClusteringDemo` k-NN vs ε-ball toggle works
- [ ] `SpectralClusteringDemo` noise slider degrades clustering gracefully
- [ ] Cross-reference updated in `spectral-theorem.mdx` (forward ref → live link)
- [ ] Forward references to planned Graph Theory topics use plain text + "(coming soon)" — no dead links
- [ ] All static figures load from `public/images/topics/graph-laplacians/`
- [ ] New `graphTheory.ts` module passes TypeScript compilation with no errors
- [ ] Page is responsive (viz components stack vertically on mobile)
- [ ] "Foundational" difficulty badge is styled correctly
- [ ] Pagefind indexes the new topic on rebuild
- [ ] Build succeeds with zero errors: `pnpm build`

---

## 14. Build Order

1. **Create `src/components/viz/shared/graphTheory.ts`** — the shared utility module. Implement graph construction functions (path, cycle, complete, star, barbell, grid, Erdős–Rényi), matrix operations (degree, Laplacian, normalized Laplacian), the Jacobi eigenvalue algorithm, Cheeger constant computation, dataset generators (two moons, circles, blobs, spiral), and k-means. Write unit tests or at least console-log verification for the Jacobi solver against known spectra (e.g., $K_n$ has eigenvalues $0$ with multiplicity 1 and $n$ with multiplicity $n-1$).
2. **Create `graph-laplacians.mdx`** with full frontmatter and all markdown/LaTeX content. Use `TheoremBlock` for all formal elements (8 definitions, 7 theorems, 2 corollaries, 2 propositions, 3 remarks). No interactive components yet.
3. Add notebook to `notebooks/graph-laplacians/` directory.
4. Export and add static figures to `public/images/topics/graph-laplacians/`.
5. Build `GraphLaplacianExplorer.tsx` — flagship component. Start with preset graphs + matrix display + eigenvalue chart, then add interactive graph editing.
6. Build `SpectralGapExplorer.tsx` — graph family selector + spectrum + λ₂ evolution tracker.
7. Build `CheegerExplorer.tsx` — minimum cut visualization + inequality bounds + scatter plot.
8. Build `SpectralClusteringDemo.tsx` — four-panel pipeline. Start with preset datasets, then add interactivity.
9. Embed all components in the MDX at their appropriate section positions.
10. Create the Graph Theory track on `/paths` page.
11. Update curriculum graph data — add node, edges, and track registration.
12. Update cross-references in `spectral-theorem.mdx` and optionally `shannon-entropy.mdx`.
13. Run testing checklist (§13).
14. Commit and deploy.

---

## Appendix A: Key Differences from Previous Briefs

1. **First topic in a NEW track.** Unlike Concentration Inequalities (second in existing track) or KL Divergence (second in existing track), this topic creates the entire Graph Theory track infrastructure: the `/paths` section, the domain registration in `curriculum.ts`, and the first node in a new subgraph of the curriculum DAG.
2. **Cross-track prerequisites (two of them).** Both `spectral-theorem` (Linear Algebra) and `shannon-entropy` (Information Theory) are prerequisites. This creates cross-track edges in the curriculum DAG — the first time a Graph Theory topic is downstream of two different completed tracks.
3. **Foundational difficulty.** Entry point to the track. Visual-first, concrete examples before algebra. No heavy algebraic derivations without geometric motivation.
4. **New shared utility module.** Creates `graphTheory.ts` (analogous to `informationTheory.ts`). This module will be extended by all subsequent Graph Theory topics.
5. **Eigendecomposition in the browser.** The Jacobi solver is a non-trivial implementation — it must handle up to 150×150 matrices for the spectral clustering demo. Correctness should be verified against known spectra before building components.
6. **Four viz components.** One more than the typical three. The spectral clustering demo is essential for the ML connection and justifies the extra complexity.
7. **Cheeger constant by enumeration.** Exponential-time algorithm capped at $n \leq 14$. Must degrade gracefully for larger graphs (fall back to Fiedler partition).

---

## Appendix B: Graph Theory Track Roadmap

| Order | Topic | Difficulty | Prerequisites | Status |
|---|---|---|---|---|
| 1 | **Graph Laplacians & Spectrum** (this brief) | Foundational | Spectral Theorem + Shannon Entropy (cross-track) | 🚧 Ready for implementation |
| 2 | Random Walks & Mixing | Intermediate | Graph Laplacians | Planned |
| 3 | Expander Graphs | Intermediate | Graph Laplacians, Random Walks | Planned |
| 4 | Message Passing & GNNs | Advanced | Graph Laplacians, Random Walks | Planned |

DAG structure: `graph-laplacians → random-walks → expander-graphs` and `graph-laplacians → message-passing` (with `random-walks → message-passing` also). Two entry edges from completed tracks: `spectral-theorem → graph-laplacians` and `shannon-entropy → graph-laplacians`.

Cross-track edges for future topics:
- `gradient-descent → message-passing` (GNN training via backpropagation through message-passing layers)
- `concentration-inequalities → expander-graphs` (Chernoff-type bounds in expander mixing lemma)

---

*Brief version: v1 | Last updated: 2026-03-26 | Author: Jonathan Rocha*  
*Reference notebook: `notebooks/graph-laplacians/01_graph_laplacians.ipynb`*  
*Reference doc: `docs/plans/formalml-handoff-reference.md`*
