# Claude Code Handoff Brief: Message Passing & GNNs

**Project:** formalML — [formalml.com](https://www.formalml.com)  
**Repo:** `github.com/jonx0037/formalML`  
**Stack:** Astro 6 · React 19 · MDX · Tailwind CSS 4 · D3.js 7 · KaTeX · Vercel  
**Package Manager:** pnpm  
**Status:** Ready for implementation  
**Reference Notebook:** `notebooks/message-passing/04_message_passing.ipynb`  
**Reference Doc:** `docs/plans/formalml-handoff-reference.md`

---

## 1. Objective

Add a new topic page **"Message Passing & GNNs"** to the **Graph Theory** track on formalml.com. This is the **fourth and final topic (capstone)** in the track — it synthesizes all three preceding topics into a unified framework for graph neural networks.

1. **Position in DAG.** Capstone of the Graph Theory track. Prerequisites: `graph-laplacians`, `random-walks`, `expander-graphs`. No downstream topics within the track.
2. **Cross-track connections.** `gradient-descent` (GNN training via backpropagation through message-passing layers — vanishing/exploding gradients governed by spectral radius), `spectral-theorem` (eigendecomposition underlies both GCN smoothing analysis and WL expressiveness proof), `pca-low-rank` (node embeddings as low-rank factorization, DeepWalk = implicit matrix factorization), `concentration-inequalities` (PAC-style generalization bounds for GNNs).
3. **Difficulty:** Advanced — the capstone assumes mastery of the Laplacian spectrum (eigenvalues, eigenvectors, Cheeger's inequality), random walk convergence (spectral gap, mixing time, total variation), and expander properties (EML, Ramanujan bounds, rapid mixing). Algebraic derivations are central — e.g., the spectral derivation of GCN, the WL expressiveness proof, and the over-smoothing convergence rate.

The companion Jupyter Notebook (`04_message_passing.ipynb`) contains the full mathematical content, proofs, code, and figures — use it as the authoritative content source for all definitions, theorems, examples, and code blocks.

**What the topic covers:**

- The message passing framework: aggregate, update, readout as the universal GNN template
- Spectral graph convolutions: graph Fourier transform → ChebNet → GCN as first-order spectral filter
- Spatial GNNs: GCN, GraphSAGE, GIN, and the neighborhood aggregation perspective
- The Weisfeiler-Leman test: 1-WL color refinement, equivalence to MPNNs, GIN matching 1-WL
- Attention-based message passing: GAT, multi-head attention, learned edge weights
- Over-smoothing: Dirichlet energy decay, random walk convergence connection, spectral gap control
- Expander-based graph rewiring: FoSR, SDRF, spectral gap optimization for GNN architecture design

---

## 2. MDX File

### Location

```
src/content/topics/message-passing.mdx
```

The entry `id` will be `message-passing` (derived from filename by the `glob()` loader in `src/content.config.ts`). The dynamic route at `src/pages/topics/[...slug].astro` resolves this to `/topics/message-passing`.

### Frontmatter

```yaml
---
title: "Message Passing & GNNs"
subtitle: "From spectral graph convolutions to neighborhood aggregation — the mathematical foundations of graph neural networks"
status: "published"
difficulty: "advanced"
prerequisites:
  - "graph-laplacians"
  - "random-walks"
  - "expander-graphs"
tags:
  - "graph-theory"
  - "graph-neural-networks"
  - "message-passing"
  - "weisfeiler-leman"
  - "over-smoothing"
  - "graph-attention"
  - "graph-rewiring"
domain: "graph-theory"
videoId: null
notebookPath: "notebooks/message-passing/04_message_passing.ipynb"
githubUrl: "https://github.com/jonx0037/formalML/blob/main/src/content/topics/message-passing.mdx"
datePublished: 2026-03-28
estimatedReadTime: 50
abstract: "Message passing neural networks learn graph representations by iteratively aggregating information along edges: at each layer, every node collects features from its neighbors, transforms the result, and produces an updated embedding. This framework unifies the Graph Theory track. The GCN update rule is a first-order polynomial of the normalized Laplacian — Laplacian smoothing with learnable weights. Repeated message passing drives node features toward the stationary distribution of the random walk on the graph, with the spectral gap controlling the convergence rate — the over-smoothing phenomenon. The Weisfeiler-Leman isomorphism test provides the expressiveness ceiling: no message passing GNN can distinguish graphs that 1-WL cannot, and GIN (sum aggregation + MLP) achieves this ceiling. Graph Attention Networks generalize GCN by learning data-dependent edge weights, effectively performing soft graph rewiring. Expander-based rewiring methods (FoSR, SDRF) explicitly increase the spectral gap by adding shortcut edges, trading the O(log n) mixing time of expanders for faster information propagation — but inheriting the O(log n) over-smoothing depth as a fundamental tradeoff."
connections:
  - topic: "graph-laplacians"
    relationship: "The GCN update rule is a first-order polynomial of the normalized Laplacian: spectral graph convolutions filter signals via the Laplacian eigendecomposition. The Fiedler vector drives spectral rewiring (FoSR), and Cheeger's inequality bounds the information bottleneck that limits message passing depth."
  - topic: "random-walks"
    relationship: "Over-smoothing is random walk convergence: repeated application of the normalized adjacency drives features to the stationary distribution π. The spectral gap γ = 1 − λ₂(P) controls the over-smoothing rate, and the mixing time bounds from random walk theory directly predict GNN depth limits."
  - topic: "expander-graphs"
    relationship: "Expanders give O(log n) receptive fields but also O(log n) over-smoothing depth — the fundamental tradeoff. Expander-based rewiring (FoSR, SDRF) increases λ₂ to improve information flow. The Expander Mixing Lemma implies quasi-random message passing on expanders."
  - topic: "spectral-theorem"
    relationship: "The spectral decomposition of the normalized adjacency underlies both the GCN smoothing analysis and the over-smoothing convergence proof. The eigenvectors form the graph Fourier basis, and eigenvalue bounds govern the convergence rate of iterated message passing."
  - topic: "gradient-descent"
    relationship: "GNNs are trained via backpropagation through L layers of message passing. The gradient flows through Â^L, where vanishing/exploding gradient behavior is governed by the spectral radius — the same convergence analysis from gradient descent theory."
  - topic: "pca-low-rank"
    relationship: "Node embeddings from GNNs approximate a low-rank factorization of the graph. DeepWalk is implicit matrix factorization (Qiu et al. 2018). Spectral clustering uses the bottom Laplacian eigenvectors as a low-rank embedding — the non-parametric precursor to learned graph embeddings."
  - topic: "concentration-inequalities"
    relationship: "PAC-style generalization bounds for GNNs use Rademacher complexity and VC dimension. The number of message-passing layers affects the model's effective capacity, and concentration inequalities bound the gap between training and test performance."
references:
  - type: "paper"
    title: "Semi-Supervised Classification with Graph Convolutional Networks"
    authors: "Kipf & Welling"
    year: 2017
    url: "https://arxiv.org/abs/1609.02907"
    note: "The GCN paper — spectral derivation, renormalization trick, and the architecture that launched graph deep learning"
  - type: "paper"
    title: "Neural Message Passing for Quantum Chemistry"
    authors: "Gilmer, Schoenholz, Riley, Vinyals & Dahl"
    year: 2017
    url: "https://arxiv.org/abs/1704.01212"
    note: "The MPNN framework that unifies GNN architectures into aggregate-update-readout"
  - type: "paper"
    title: "How Powerful are Graph Neural Networks?"
    authors: "Xu, Hu, Leskovec & Jegelka"
    year: 2019
    url: "https://arxiv.org/abs/1810.00826"
    note: "GIN paper — proves MPNN ≤ 1-WL expressiveness and shows GIN matches 1-WL"
  - type: "paper"
    title: "Graph Attention Networks"
    authors: "Veličković, Cucurull, Casanova, Romero, Liò & Bengio"
    year: 2018
    url: "https://arxiv.org/abs/1710.10903"
    note: "GAT — learned attention-based aggregation replacing uniform neighborhood weights"
  - type: "paper"
    title: "Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning"
    authors: "Li, Han & Wu"
    year: 2018
    url: "https://arxiv.org/abs/1801.07606"
    note: "Over-smoothing analysis — proves GCN is Laplacian smoothing and characterizes depth limits"
  - type: "paper"
    title: "Understanding Over-Squashing and Bottlenecks on Graphs via Curvature"
    authors: "Topping, Di Giovanni, Chamberlain, Dong & Bronstein"
    year: 2022
    url: "https://arxiv.org/abs/2111.14522"
    note: "SDRF rewiring — discrete Ricci curvature identifies bottlenecks and guides edge addition"
  - type: "paper"
    title: "Expander Graph Propagation"
    authors: "Deac, Lackenby & Veličković"
    year: 2022
    url: "https://arxiv.org/abs/2210.02997"
    note: "FoSR — first-order spectral rewiring using the Fiedler vector to maximize λ₂"
  - type: "paper"
    title: "Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering"
    authors: "Defferrard, Bresson & Vandergheynst"
    year: 2016
    url: "https://arxiv.org/abs/1606.09375"
    note: "ChebNet — Chebyshev polynomial spectral filters, the precursor to GCN"
  - type: "paper"
    title: "Weisfeiler and Leman Go Neural: Higher-Order Graph Neural Networks"
    authors: "Morris, Ritzert, Fey, Hamilton, Lenssen, Rattan & Grohe"
    year: 2019
    url: "https://arxiv.org/abs/1810.02244"
    note: "k-WL GNNs — higher-order expressiveness beyond standard message passing"
---
```

### Content Sections

The MDX body should have the following sections. Pull all mathematical content from the notebook. Each section header maps directly to a notebook section.

| # | MDX Section | Notebook § | Key Content |
|---|---|---|---|
| 1 | Overview & Motivation | §0 (Overview) | How message passing unifies the track; GCN as Laplacian smoothing; over-smoothing as random walk convergence; expander rewiring; roadmap |
| 2 | The Message Passing Framework | §1 | MPNN template (aggregate, update, readout); matrix form $H^{(\ell+1)} = \sigma(\hat{A}H^{(\ell)}W^{(\ell)})$; architecture comparison table ($\hat{A}$ choices); receptive field growth and diameter |
| 3 | Spectral Graph Convolutions | §2 | Graph Fourier transform $\hat{\mathbf{x}} = U^T\mathbf{x}$; spectral convolution $g_\theta \star \mathbf{x} = U g_\theta(\Lambda) U^T \mathbf{x}$; ChebNet polynomial filters; GCN as first-order spectral filter (full derivation: ChebNet → K=1 → renormalization trick) |
| 4 | Spatial Graph Convolutions | §3 | GCN spatial view; GraphSAGE (concat + AGG); GIN (sum + MLP); comparison of aggregation functions |
| 5 | The Weisfeiler-Leman Test & GNN Expressiveness | §4 | 1-WL color refinement algorithm; MPNN ≤ 1-WL theorem (proof); GIN = 1-WL theorem (proof sketch); 1-WL failure cases (regular graphs); higher-order k-WL |
| 6 | Attention-Based Message Passing | §5 | GAT definition (attention coefficients, softmax); GAT generalizes GCN (proof); multi-head attention; attention matrix perspective |
| 7 | Over-Smoothing & the Random Walk Connection | §6 | Dirichlet energy definition; over-smoothing rate theorem (proof via spectral decomposition); corollary on expanders; MAD metric; mitigation strategies (residual, JK, DropEdge, PairNorm, rewiring) |
| 8 | Expander-Based Graph Rewiring | §7 | Bottleneck problem and Cheeger; spectral gap rewiring definition; SDRF (Ricci curvature); FoSR (Fiedler-based); expander augmentation theorem; beyond message passing (transformers, k-GNN, subgraph GNNs) |
| 9 | Computational Notes | §8 | PyTorch Geometric and DGL code examples; spectral vs spatial complexity table; monitoring over-smoothing in practice |
| 10 | Connections & Further Reading | §9 | Connection table, notation summary, references |

---

## 3. TheoremBlock Usage

Use the existing `TheoremBlock.astro` component for all numbered formal elements. The notebook contains:

| Type | # | Name | TheoremBlock `type` |
|---|---|---|---|
| Definition | 1 | Message Passing Neural Network (MPNN) | `definition` |
| Definition | 2 | Graph Fourier Transform | `definition` |
| Definition | 3 | Spectral Graph Convolution | `definition` |
| Definition | 4 | Graph Isomorphism Network (GIN) | `definition` |
| Definition | 5 | 1-WL Color Refinement | `definition` |
| Definition | 6 | Graph Attention Layer (GAT) | `definition` |
| Definition | 7 | Over-Smoothing (Dirichlet Energy) | `definition` |
| Definition | 8 | Mean Average Distance (MAD) | `definition` |
| Definition | 9 | Spectral Gap Rewiring | `definition` |
| Theorem | 1 | Chebyshev Approximation (ChebNet) | `theorem` (stated — Defferrard et al.) |
| Theorem | 2 | GCN is First-Order Spectral | `theorem` (with `proof`) |
| Theorem | 3 | MPNN ≤ 1-WL Expressiveness | `theorem` (with `proof` sketch) |
| Theorem | 4 | GIN Matches 1-WL | `theorem` (with `proof` sketch) |
| Theorem | 5 | GAT Generalizes GCN | `theorem` (with `proof`) |
| Theorem | 6 | Over-Smoothing Rate | `theorem` (with `proof`) |
| Theorem | 7 | FoSR Spectral Gap Improvement | `theorem` (stated — Deac et al.) |
| Proposition | 1 | Receptive Field and Diameter | `proposition` |
| Proposition | 2 | SDRF Spectral Gap Increase | `proposition` (stated — Topping et al.) |
| Corollary | 1 | Over-Smoothing on Expanders | `corollary` |
| Remark | — | GCN is Laplacian Smoothing | `remark` |
| Remark | — | Regular Graphs and 1-WL Failure | `remark` |
| Remark | — | Connection to Expanders | `remark` (rewiring → Ramanujan bound) |

**LaTeX symbols to verify render correctly:**

- `H^{(\ell+1)} = \sigma(\hat{A} H^{(\ell)} W^{(\ell)})` — MPNN matrix form
- `\hat{\mathbf{x}} = U^T \mathbf{x}` — Graph Fourier transform
- `g_\theta \star \mathbf{x} = U g_\theta(\Lambda) U^T \mathbf{x}` — Spectral convolution
- `\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}` — GCN renormalized adjacency
- `T_k(\tilde{\mathcal{L}})` — Chebyshev polynomial of Laplacian
- `c_v^{(\ell+1)} = \text{HASH}(c_v^{(\ell)}, \{\!\{c_u^{(\ell)}\}\!\})` — 1-WL update
- `\alpha_{vu} = \text{softmax}(\text{LeakyReLU}(\mathbf{a}^T [W\mathbf{h}_v \| W\mathbf{h}_u]))` — GAT attention
- `E(H) = \text{tr}(H^T L H)` — Dirichlet energy
- `||\mu_2||^{2\ell} \cdot E(H^{(0)})` — Over-smoothing bound

---

## 4. Interactive Visualizations

New React components in `src/components/viz/`. All use `client:visible` hydration. Follow patterns from existing viz components (Approach B — `useEffect` + manual refs for multi-panel layouts).

### 4a. MessagePassingExplorer (Flagship)

**File:** `src/components/viz/MessagePassingExplorer.tsx`

An interactive visualization of message passing layers on a graph, showing feature propagation, receptive field growth, and architecture comparison.

- **Left panel:** A graph drawn with D3 force simulation. Node color encodes the current feature vector (e.g., hue of the dominant component, or a 1D projection). A "source node" is highlighted. At each layer, the colored features spread outward — the receptive field expands visibly.

- **Right panel:** Two vertically stacked plots:
  1. **Feature heatmap** — a heatmap of $H^{(\ell)} \in \mathbb{R}^{n \times d}$ showing how node features evolve across layers. Rows = nodes (sorted by community), columns = feature dimensions. As layers increase, rows become more similar (over-smoothing).
  2. **Dirichlet energy trace** — a line plot of $E(H^{(\ell)})$ vs layer $\ell$. Shows exponential decay, with the spectral gap governing the rate. Annotate where energy drops below 1% of the initial (the "over-smoothing depth").

- **Controls:**
  - **Preset graph** dropdown: Karate Club, Barbell(5), Grid(4×4), Petersen, Cycle(10), Path(10).
  - **Architecture** selector: GCN (symmetric), GraphSAGE (mean), GIN (sum).
  - **Layer depth** slider (0–20): step through layers. The graph, heatmap, and energy trace update together.
  - **"Animate layers"** play/pause button: auto-advance layer by layer.

**Interaction:** Changing the architecture selector recomputes the $\hat{A}$ matrix and resets the layer progression. Changing the preset graph resets everything. The layer slider is the primary interaction — scrubbing through layers shows receptive field growth and over-smoothing in real time.

**Implementation notes:**
- Compute $\hat{A}$ for each architecture (GCN: `renormalizedAdjacency`; GraphSAGE: `transitionMatrix`; GIN: $(1+\varepsilon)I + A$ with $\varepsilon = 0$).
- Feature propagation: $H^{(\ell)} = \hat{A}^{\ell} H^{(0)}$ (no weights or nonlinearity — pure propagation for clarity).
- Initial features $H^{(0)}$: seeded random $n \times 4$ matrix (same seed for reproducibility).
- Dirichlet energy: `trace(H^T L H)` at each layer, precomputed up to layer 20.
- Node coloring: project the 4D feature to 1D via first PC, then use `d3.scaleDiverging(d3.interpolateRdBu)`.
- Use graph construction and Laplacian helpers from `graphTheory.ts`.
- Cap graph size at 34 nodes (Karate Club is 34). For larger presets, subsample.
- D3 force simulation for graph layout.

**Reference:** Notebook §1–§2.

### 4b. WLExpressivenessExplorer

**File:** `src/components/viz/WLExpressivenessExplorer.tsx`

An interactive visualization of the 1-WL color refinement algorithm and its connection to GNN expressiveness.

- **Left panel:** Two graphs displayed side by side. Nodes are colored by their current WL color. At each refinement step, colors update. If the color histograms diverge, the text reads "1-WL distinguishes ✓" in green. If they stabilize identically, "1-WL cannot distinguish ✗" in red.

- **Right panel:** Two vertically stacked:
  1. **Color histogram** — bar chart comparing the color distribution of Graph 1 vs Graph 2 at the current step. Bars are side-by-side for each color.
  2. **Refinement trace** — a timeline showing the number of distinct colors at each step for both graphs. Lines converge (stabilize) when no further refinement is possible.

- **Controls:**
  - **Graph pair** dropdown: "C₆ vs Two K₃" (1-WL succeeds), "K₃,₃ vs Prism" (1-WL fails), "Petersen vs Custom 3-reg" (1-WL fails), "Custom pair" (user-editable).
  - **Step** slider (0 to convergence): manually step through WL iterations.
  - **"Auto-refine"** play button: animate the refinement steps.

**Interaction:** Selecting a graph pair loads two preset graphs and resets the WL iteration. The step slider advances the color refinement. The histogram and trace update in real time.

**Implementation notes:**
- WL color refinement: implement in `graphTheory.ts` as a new export `wlColorRefinement(graph: Graph, maxIters: number): number[][]` returning the color array at each step.
- Color mapping: use `d3.schemeSet3` for up to 12 distinct colors, then cycle.
- Graph pair presets: hardcode adjacency matrices for each pair.
- For "Custom pair": allow editing of two small graphs (≤ 10 nodes) via the same click-to-add-edge interface from `GraphLaplacianExplorer`.
- D3 force simulation for both graphs, arranged side by side.

**Reference:** Notebook §4.

### 4c. OverSmoothingAnalyzer

**File:** `src/components/viz/OverSmoothingAnalyzer.tsx`

An interactive analysis tool that shows over-smoothing across graph families and architectures, with connections to the spectral gap and a rewiring comparison.

- **Left panel:** The current graph. Node color shows features at the selected layer. As the layer slider increases, colors become homogenized (oversmoothed). The graph animates smoothly between layers.

- **Right panel:** Three vertically stacked plots:
  1. **Dirichlet energy decay** — log-scale plot of $E(H^{(\ell)}) / E(H^{(0)})$ vs $\ell$ for multiple graph families (selectable). Shows exponential decay rates.
  2. **Spectral gap vs smoothing depth** — scatter plot of (spectral gap $\gamma$, over-smoothing depth) for all graph presets. Higher $\gamma$ → faster smoothing.
  3. **Rewiring comparison** — for the selected graph, compare Dirichlet energy decay on the original graph vs the FoSR-rewired graph. Rewiring accelerates both information spread and over-smoothing.

- **Controls:**
  - **Graph family** checkboxes (select 2–5): Path(10), Cycle(10), Grid(3×3), Barbell(5), Petersen, Complete(8), Karate Club.
  - **Layer depth** slider (0–30): controls the left panel and first right-panel plot.
  - **"Show rewiring"** toggle: adds the FoSR-rewired variant to the comparison.
  - **Rewiring edges** slider (2–10): how many FoSR edges to add when rewiring is enabled.

**Interaction:** Checking graph families adds energy decay curves. The layer slider updates the left panel node colors and a vertical marker on the energy plot. The rewiring toggle adds dashed curves for the rewired variant.

**Implementation notes:**
- FoSR rewiring: implement in `graphTheory.ts` as `fosrRewire(graph: Graph, numEdges: number): Graph`. At each step, compute the Fiedler vector, find the non-edge $(u,v)$ maximizing $(f_u - f_v)^2$, and add it.
- Pre-compute energy traces for all preset graphs up to 30 layers on the mount. Store in a ref.
- Spectral gap: eigenvalue of the renormalized adjacency $\hat{A}$ — gap = $1 - |\mu_2|$.
- Over-smoothing depth: first $\ell$ where $E(H^{(\ell)}) < 0.01 \cdot E(H^{(0)})$.
- Use Jacobi eigenvalue solver from `graphTheory.ts`.

**Reference:** Notebook §6–§7.

---

## 5. Data Modules

### Extend `src/components/viz/shared/graphTheory.ts`

The Message Passing topic extends the shared `graphTheory.ts` module. **Do not create a new file** — add the following exports to the existing module.

```typescript
// === Message Passing Types ===

export interface MPNNConfig {
  architecture: 'gcn' | 'graphsage' | 'gin';
  layers: number;
  epsilon?: number;  // GIN self-loop weight (default 0)
}

export interface PropagationResult {
  features: number[][][];         // features[ell] = H^(ell), shape n × d
  dirichletEnergy: number[];      // E(H^(ell)) at each layer
  mad: number[];                  // MAD(H^(ell)) at each layer
  spectralGap: number;            // γ = 1 - |μ₂| of Â
  overSmoothingDepth: number;     // First ℓ where E < 0.01 * E₀
}

export interface WLResult {
  colorHistory: number[][];       // colors at each refinement step
  numColorsHistory: number[];     // number of distinct colors at each step
  convergedAt: number;            // step at which colors stabilized
}

export interface RewireResult {
  graph: Graph;                   // rewired graph
  gapHistory: number[];           // λ₂ after each rewiring step
  addedEdges: [number, number][]; // edges added by rewiring
}

// === Normalized Adjacency Operators ===

/**
 * GCN renormalized adjacency: D_tilde^{-1/2} A_tilde D_tilde^{-1/2}
 * where A_tilde = A + I, D_tilde = diag(A_tilde * 1).
 */
export function renormalizedAdjacency(graph: Graph): number[][] { ... }

/**
 * Compute the normalized adjacency for a given architecture.
 * 'gcn' → D_tilde^{-1/2} A_tilde D_tilde^{-1/2}
 * 'graphsage' → D^{-1} A (transition matrix)
 * 'gin' → (1 + ε)I + A
 */
export function normalizedAdjacencyForArch(graph: Graph, config: MPNNConfig): number[][] { ... }

// === Feature Propagation ===

/**
 * Propagate features through L layers of message passing (no weights/nonlinearity).
 * H^(ℓ) = Â^ℓ H^(0), computing Dirichlet energy and MAD at each layer.
 * Returns the full propagation trace.
 */
export function propagateFeatures(
  graph: Graph,
  H0: number[][],
  config: MPNNConfig
): PropagationResult { ... }

/**
 * Dirichlet energy: trace(H^T L H).
 */
export function dirichletEnergy(H: number[][], L: number[][]): number { ... }

/**
 * Mean Average Distance of node features.
 */
export function meanAverageDistance(H: number[][]): number { ... }

// === Weisfeiler-Leman ===

/**
 * Run 1-WL color refinement on a graph.
 * Returns color arrays at each iteration until convergence.
 */
export function wlColorRefinement(graph: Graph, maxIters?: number): WLResult { ... }

/**
 * Check if 1-WL distinguishes two graphs.
 * Returns { distinguishes: boolean, step: number, histograms: [Map, Map] }.
 */
export function wlDistinguishes(
  g1: Graph, g2: Graph
): { distinguishes: boolean; step: number } { ... }

// === Graph Rewiring ===

/**
 * First-Order Spectral Rewiring (FoSR).
 * At each step, add the non-edge (u,v) maximizing (f_u - f_v)^2
 * where f is the Fiedler vector.
 */
export function fosrRewire(graph: Graph, numEdges: number): RewireResult { ... }

// === GAT Utilities (for visualization only) ===

/**
 * Compute GAT-style attention weights given node features and random parameters.
 * For visualization purposes — not a full GAT implementation.
 */
export function computeGATAttention(
  graph: Graph,
  H: number[][],
  seed?: number
): number[][] { ... }
```

**Computation notes:**
- `renormalizedAdjacency` must handle isolated nodes (degree 0 after adding self-loop → degree 1).
- `propagateFeatures` computes `Â^ℓ H₀` iteratively (multiply by `Â` each step), not by matrix power.
- `wlColorRefinement` uses a hash map (string key of `[ownColor, sortedNeighborColors]`) for color assignment. New colors are assigned incrementing integers.
- `fosrRewire` calls `jacobiEigendecomposition` at each step (small matrices, ≤ 34×34 for Karate Club).
- All computation should be **lazy** — wrap in `getX()` functions if expensive.

---

## 6. Curriculum Graph Updates

### `src/data/curriculum-graph.json`

**Add node:**
```json
{ "id": "message-passing", "label": "Message Passing & GNNs", "domain": "graph-theory", "status": "published", "url": "/topics/message-passing" }
```

**Add edges:**
```json
{ "source": "graph-laplacians", "target": "message-passing" },
{ "source": "random-walks", "target": "message-passing" },
{ "source": "expander-graphs", "target": "message-passing" },
{ "source": "gradient-descent", "target": "message-passing" }
```

> Note: The `gradient-descent → message-passing` edge is a cross-track edge (Optimization → Graph Theory).

### `src/data/curriculum.ts`

Remove `"Message Passing & GNNs"` from the `planned` array of the `graph-theory` domain track. After this, the `planned` array for `graph-theory` should be **empty** — all four topics are published, completing the track.

Update the Graph Theory track status from `🚧 Next` to `✅ Complete`.

---

## 7. Cross-References

### Existing topics to update (convert forward references to live links):

1. **`graph-laplacians.mdx`** — §7.3 "Bridge to Message Passing" and the Connections table should link to `/topics/message-passing` (currently plain text + "(coming soon)").
2. **`random-walks.mdx`** — The "Over-Smoothing in GNNs" remark and the Connections table should link to `/topics/message-passing` (currently plain text + "(coming soon)").
3. **`expander-graphs.mdx`** — §7.4 "Graph Neural Networks" and the Connections table should link to `/topics/message-passing` (currently plain text + "(coming soon)").

### This topic links FROM:

- `/topics/graph-laplacians` — Laplacian eigendecomposition, Fiedler vector, Cheeger inequality
- `/topics/random-walks` — spectral gap, mixing time, stationary distribution
- `/topics/expander-graphs` — expansion, Ramanujan bound, O(log n) mixing
- `/topics/spectral-theorem` — eigendecomposition guarantees
- `/topics/gradient-descent` — backpropagation, convergence analysis
- `/topics/pca-low-rank` — low-rank factorization, embeddings

### Forward references:

No forward references needed — this is the final topic in the Graph Theory track. The Category Theory track is planned but has no direct dependency on this topic.

---

## 8. Images

Copy the following images from the notebook to `public/images/topics/message-passing/`:

| Filename | Description |
|---|---|
| `receptive-field-growth.png` | 4-panel showing receptive field expansion on barbell graph |
| `spectral-filters.png` | Spectral filter design (low/band/high-pass) and filtered signals |
| `gcn-smoothing.png` | GCN smoothing: repeated application of renormalized adjacency |
| `wl-color-refinement.png` | 1-WL color refinement on C₆ vs two triangles |
| `gat-attention.png` | GAT attention weights on Karate Club |
| `over-smoothing-analysis.png` | Dirichlet energy decay, MAD, spectral gap vs smoothing depth |
| `graph-rewiring.png` | FoSR rewiring on barbell: original, rewired, spectral gap evolution |
| `rewiring-vs-oversmoothing.png` | Energy decay comparison: original vs rewired barbell |

---

## 9. Static Figures in MDX

Reference notebook images in MDX with:

```mdx
![Receptive field growth on a barbell graph](/images/topics/message-passing/receptive-field-growth.png)
```

---

## 10. Notebook File

Place the companion notebook in the repo:

```
notebooks/message-passing/04_message_passing.ipynb
```

This follows the corrected `notebookPath` convention (subdirectory + file). The `04_` prefix indicates this is the fourth and final notebook in the Graph Theory track.

---

## 11. SEO & OG

If the OG image generation script is active, it should auto-generate an OG image. If not, create a manual OG image and place it in `public/og/message-passing.png`.

Meta tags from `TopicLayout.astro` frontmatter. Verify:
- `<title>`: "Message Passing & GNNs | formalML"
- `<meta name="description">`: Uses the `abstract` field
- OG tags: title, description, image

---

## 12. Testing Checklist

- [ ] All KaTeX renders correctly (especially `H^{(\ell+1)} = \sigma(\hat{A}H^{(\ell)}W^{(\ell)})`, `\hat{\mathbf{x}} = U^T\mathbf{x}`, `g_\theta \star \mathbf{x} = U g_\theta(\Lambda) U^T \mathbf{x}`, `\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}`, `T_k(\tilde{\mathcal{L}})`, `\alpha_{vu}`, `E(H) = \text{tr}(H^TLH)`, `c_v^{(\ell+1)} = \text{HASH}(c_v^{(\ell)}, \{\!\{c_u^{(\ell)}\}\!\})`)
- [ ] Prerequisite chips show: "Graph Laplacians & Spectrum," "Random Walks & Mixing," and "Expander Graphs" as linked chips
- [ ] Topic card appears on `/topics` index with "Advanced" difficulty badge and `graph-theory` domain tag
- [ ] Graph Theory track on `/paths` shows Message Passing & GNNs as Published (Advanced, linked)
- [ ] **Graph Theory track shows as Complete (4/4 published)** — no more Planned topics
- [ ] `graph-laplacians → message-passing` edge renders in the curriculum graph
- [ ] `random-walks → message-passing` edge renders in the curriculum graph
- [ ] `expander-graphs → message-passing` edge renders in the curriculum graph
- [ ] `gradient-descent → message-passing` cross-track edge renders in the curriculum graph
- [ ] `MessagePassingExplorer` preset dropdown loads all graphs correctly (including Karate Club at 34 nodes)
- [ ] `MessagePassingExplorer` architecture selector switches between GCN/GraphSAGE/GIN and recomputes propagation
- [ ] `MessagePassingExplorer` layer slider updates node colors, heatmap, and energy trace together
- [ ] `MessagePassingExplorer` animate button auto-advances layers smoothly
- [ ] `MessagePassingExplorer` Dirichlet energy trace shows exponential decay on log scale
- [ ] `MessagePassingExplorer` shows different decay rates for different architectures (GCN vs GIN)
- [ ] `WLExpressivenessExplorer` "C₆ vs Two K₃" correctly shows 1-WL distinguishing (different histograms)
- [ ] `WLExpressivenessExplorer` "K₃,₃ vs Prism" correctly shows 1-WL failure (identical histograms, non-isomorphic)
- [ ] `WLExpressivenessExplorer` step slider advances color refinement with visible color changes
- [ ] `WLExpressivenessExplorer` color histograms update correctly at each step
- [ ] `WLExpressivenessExplorer` auto-refine button animates through steps
- [ ] `OverSmoothingAnalyzer` graph family checkboxes add/remove energy decay curves
- [ ] `OverSmoothingAnalyzer` energy curves match expected ordering (complete fastest, path slowest)
- [ ] `OverSmoothingAnalyzer` spectral gap scatter plot correctly positions all graph families
- [ ] `OverSmoothingAnalyzer` rewiring toggle adds FoSR-rewired curves (dashed)
- [ ] `OverSmoothingAnalyzer` rewired curves show faster energy decay than originals
- [ ] `OverSmoothingAnalyzer` rewiring slider adjusts the number of FoSR edges and updates curves
- [ ] Forward reference to **Message Passing & GNNs** in `graph-laplacians.mdx` converted to live link
- [ ] Forward reference to **Message Passing & GNNs** in `random-walks.mdx` converted to live link
- [ ] Forward reference to **Message Passing & GNNs** in `expander-graphs.mdx` converted to live link
- [ ] No forward references to non-existent topics (no dead links)
- [ ] All static figures load from `public/images/topics/message-passing/`
- [ ] Extended `graphTheory.ts` module passes TypeScript compilation with no errors
- [ ] Page is responsive (viz components stack vertically on mobile)
- [ ] "Advanced" difficulty badge is styled correctly
- [ ] Pagefind indexes the new topic on rebuild
- [ ] Build succeeds with zero errors: `pnpm build`

---

## 13. Build Order

1. **Extend `src/components/viz/shared/graphTheory.ts`** — add all message-passing exports: `renormalizedAdjacency`, `normalizedAdjacencyForArch`, `propagateFeatures`, `dirichletEnergy`, `meanAverageDistance`, `wlColorRefinement`, `wlDistinguishes`, `fosrRewire`, `computeGATAttention`. Write verification tests: e.g., Dirichlet energy of $\hat{A}^0 H_0$ should equal $\text{tr}(H_0^T L H_0)$; WL on $C_6$ vs two $K_3$ should distinguish; WL on $K_{3,3}$ vs Prism should fail; FoSR on Barbell(5) should increase $\lambda_2$ monotonically.
2. **Create `message-passing.mdx`** with full frontmatter and all markdown/LaTeX content. Use `TheoremBlock` for all formal elements (9 definitions, 7 theorems, 2 propositions, 1 corollary, 3 remarks). No interactive components yet.
3. Add notebook to `notebooks/message-passing/` directory.
4. Export and add static figures to `public/images/topics/message-passing/`.
5. Build `MessagePassingExplorer.tsx` — flagship component. Start with preset graphs + feature propagation + energy trace, then add architecture selector and layer animation.
6. Build `WLExpressivenessExplorer.tsx` — graph pair selector + WL refinement + histogram + trace. Start with preset pairs and the step slider, then add auto-refine animation.
7. Build `OverSmoothingAnalyzer.tsx` — graph family selector + energy decay curves + scatter plot + rewiring comparison. Start with energy curves, then add the rewiring toggle and slider.
8. Embed all components in the MDX at their appropriate section positions.
9. Update `graph-laplacians.mdx` — convert "(coming soon)" forward references to live `/topics/message-passing` links.
10. Update `random-walks.mdx` — convert "(coming soon)" forward references to live `/topics/message-passing` links.
11. Update `expander-graphs.mdx` — convert "(coming soon)" forward references to live `/topics/message-passing` links.
12. Update curriculum graph data — add node, edges, and mark Graph Theory track as complete.
13. Update `/paths` page — change Message Passing & GNNs from Planned to Published. Mark Graph Theory track as Complete.
14. Run testing checklist (§12).
15. Commit and deploy.

---

## Appendix A: Key Differences from the Expander Graphs Brief

1. **Capstone / final topic.** Unlike Expander Graphs (third topic, intermediate), this is the advanced capstone that synthesizes the entire track. It must demonstrate mastery of all three preceding topics by showing how GNN theory emerges from their union.
2. **Three prerequisites (maximum in track).** All three preceding Graph Theory topics are prerequisites: `graph-laplacians`, `random-walks`, `expander-graphs`. Plus cross-track: `gradient-descent`.
3. **Three pages to update.** All three preceding topics have "(coming soon)" forward references that must be converted to live links. This is the most cross-referenced topic in the track.
4. **Advanced difficulty.** Full algebraic proofs: spectral derivation of GCN from ChebNet, WL expressiveness proof, and over-smoothing convergence rate proof. No hand-waving.
5. **Track completion.** After this topic, the Graph Theory track is fully complete (4/4). The `curriculum.ts` planned array for `graph-theory` should be empty. The `/paths` page should show the track as Complete.
6. **WL color refinement.** The `wlColorRefinement` function uses hash-based color assignment — this is a new algorithmic pattern not used in previous Graph Theory modules. Test carefully on known pairs.
7. **FoSR rewiring.** The `fosrRewire` function calls the Jacobi solver at each step (up to 10 steps for graphs up to 34 nodes). Total: 10 × Jacobi(34×34) — verify this is within performance budget.
8. **Karate Club graph.** The largest graph preset (34 nodes). Verify that the Jacobi solver, WL refinement, and feature propagation all handle 34×34 matrices within the browser performance budget.

---

## Appendix B: Graph Theory Track Final Status

| Order | Topic | Difficulty | Prerequisites | Status |
|---|---|---|---|---|
| 1 | Graph Laplacians & Spectrum | Foundational | Spectral Theorem + Shannon Entropy | ✅ Published |
| 2 | Random Walks & Mixing | Intermediate | Graph Laplacians | ✅ Published |
| 3 | Expander Graphs | Intermediate | Graph Laplacians, Random Walks | ✅ Published |
| 4 | **Message Passing & GNNs** (this brief) | Advanced | Graph Laplacians, Random Walks, Expander Graphs | 🚧 Ready for implementation |

DAG edges after implementation:
- `spectral-theorem → graph-laplacians → random-walks → expander-graphs → message-passing`
- `shannon-entropy → graph-laplacians`
- `graph-laplacians → message-passing` (direct)
- `random-walks → message-passing` (direct)
- `concentration-inequalities → expander-graphs` (cross-track)
- `gradient-descent → message-passing` (cross-track)

**After implementation, the Graph Theory track is complete (4/4 topics published).**

---

*Brief version: v1 | Last updated: 2026-03-27 | Author: Jonathan Rocha*  
*Reference notebook: `notebooks/message-passing/04_message_passing.ipynb`*  
*Reference doc: `docs/plans/formalml-handoff-reference.md`*
