# Claude Code Handoff Brief: The Mapper Algorithm

**Project:** formalML — [formalml.com](https://www.formalml.com)  
**Repo:** `github.com/jonx0037/formalML`  
**Stack:** Astro 5 · React 18 · MDX · Tailwind CSS 4 · D3.js v7 · KaTeX · Vercel  
**Package Manager:** pnpm  
**Status:** Ready for implementation  
**Reference Notebook:** `mapper-algorithm-notebook.ipynb` (included in this handoff — content reference only, not published)  
**Parent Brief:** `formalml-handoff-brief-v1.md` — this brief adds a new topic to the existing site. Refer to v1 Sections 4 (schema), 6 (viz components), 7 (TheoremBlock), and 8 (design system) for architectural context.  
**Sibling Brief:** `formalml-cech-complexes-handoff-brief.md` — the Čech Complexes topic was the most recent addition. Follow the same implementation patterns established there.

---

## 1. Objective

Add a new topic page **"The Mapper Algorithm"** to the Topology & TDA track on formalml.com. This is the fourth published topic, following Simplicial Complexes (foundational), Persistent Homology (intermediate), and Čech Complexes & Nerve Theorem (intermediate). It is currently listed as "Planned" on the `/paths` page and should become published upon deployment.

This is the first **advanced** topic on the site. The content builds directly on the Nerve Theorem from the Čech Complexes page — Mapper's output is literally a nerve of a refined cover, and the Nerve Theorem is the theoretical justification for why the algorithm works.

The companion Jupyter Notebook (`mapper-algorithm-notebook.ipynb`) contains the full mathematical content, proofs, code, and figures — use it as the authoritative content source for all definitions, theorems, examples, and code blocks.

---

## 2. MDX File

### Location

```
src/content/topics/mapper-algorithm.mdx
```

The entry `id` will be `mapper-algorithm` (derived from filename by the `glob()` loader in `src/content.config.ts`). The dynamic route at `src/pages/topics/[...slug].astro` resolves this to `/topics/mapper-algorithm`.

### Frontmatter

Must conform to the schema in `src/content.config.ts` (see v1 brief §4). Here is the complete frontmatter:

```yaml
---
title: "The Mapper Algorithm"
subtitle: "A topological lens for high-dimensional data — building interpretable graphs from point clouds via the Nerve Theorem"
status: "published"
difficulty: "advanced"
prerequisites:
  - "simplicial-complexes"
  - "persistent-homology"
  - "cech-complexes"
tags:
  - "topology"
  - "tda"
  - "data-analysis"
  - "visualization"
  - "algebraic-topology"
domain: "topology"
videoId: null
notebookPath: "notebooks/mapper-algorithm-notebook.ipynb"
githubUrl: "https://github.com/jonx0037/formalML/blob/main/src/content/topics/mapper-algorithm.mdx"
datePublished: 2026-03-16
estimatedReadTime: 40
abstract: "The Mapper algorithm constructs a compressed graph summary of high-dimensional data by pulling back an interval cover through a filter function, clustering within each pullback, and taking the nerve of the resulting cover. The Nerve Theorem guarantees that this graph faithfully captures the data's topology. We develop the full construction from first principles, build Mapper from scratch, compare with the KeplerMapper library, analyze parameter sensitivity, and apply the algorithm to financial market regime detection."
connections:
  - topic: "simplicial-complexes"
    relationship: "the Mapper graph is a 1-dimensional simplicial complex (graph)"
  - topic: "persistent-homology"
    relationship: "PH on the Mapper graph quantifies topological features; multiscale Mapper uses PH"
  - topic: "cech-complexes"
    relationship: "the Nerve Theorem (proved in Čech Complexes) is the theoretical foundation of Mapper"
  - topic: "barcodes-bottleneck-distance"
    relationship: "stability of Mapper under perturbation is measured via bottleneck distance"
references:
  - type: "paper"
    title: "Topological Methods for the Analysis of High Dimensional Data Sets and 3D Object Recognition"
    authors: "Singh, Mémoli & Carlsson"
    year: 2007
    note: "The original Mapper paper"
  - type: "paper"
    title: "Extracting insights from the shape of complex data using topology"
    authors: "Lum, Singh, Lehman, Ishkanov, Vejdemo-Johansson, Alagappan, Carlsson & Carlsson"
    year: 2013
    url: "https://doi.org/10.1038/srep01236"
    note: "Landmark applications — breast cancer, NBA, politics"
  - type: "paper"
    title: "Structure and Stability of the One-Dimensional Mapper"
    authors: "Carrière & Oudot"
    year: 2018
    url: "https://doi.org/10.1007/s10208-017-9370-z"
    note: "Convergence and stability theory"
  - type: "paper"
    title: "Towards persistence-based reconstruction in Euclidean spaces"
    authors: "Chazal & Oudot"
    year: 2008
    note: "Persistent Nerve Theorem"
  - type: "paper"
    title: "Kepler Mapper: A flexible Python implementation of the Mapper algorithm"
    authors: "van Veen, Saul, Eargle & Mangham"
    year: 2019
    url: "https://doi.org/10.21105/joss.01315"
    note: "KeplerMapper library"
  - type: "paper"
    title: "Multiscale Mapper: Topological Summarization via Codomain Covers"
    authors: "Dey, Mémoli & Wang"
    year: 2016
    note: "Multiscale Mapper construction"
  - type: "paper"
    title: "Ball Mapper: A Shape Summary for Topological Data Analysis"
    authors: "Dłotko"
    year: 2019
    note: "Ball Mapper variant — removes the filter function"
  - type: "book"
    title: "Computational Topology: An Introduction"
    authors: "Edelsbrunner & Harer"
    year: 2010
    note: "General reference for TDA constructions"
---
```

### Content Sections

The MDX body should have the following sections. Pull all mathematical content from the notebook. Each section header maps directly to a notebook section.

| # | MDX Section | Notebook § | Key Content |
|---|---|---|---|
| 1 | Overview & Motivation | §1 | Why Mapper — compressed topological summary; comparison with PH and dim reduction; Nerve Theorem connection; Lum et al. landmark result |
| 2 | The Mapper Pipeline | §2 | Def 1 (filter function), Def 2 (interval cover), Def 3 (pullback cover), Def 4 (refined cover), Def 5 (Mapper graph); 5-step pseudocode |
| 3 | Filter Functions | §3 | Eccentricity, density, PCA; "choosing a filter is choosing a perspective"; figure-eight three-filter comparison |
| 4 | Covering & Pullback | §4 | Interval cover parameterization (n, p); Proposition 1 (overlap and connectivity); pullback visualization |
| 5 | From-Scratch Implementation | §5 | Full `mapper_from_scratch()` implementation; circle example; verification against known topology |
| 6 | Classic Examples | §6 | Figure-eight (H₁ = 2), torus (3D); filter choice determines visible features |
| 7 | KeplerMapper | §7 | Production workflow; same examples; HTML visualization note |
| 8 | Parameter Sensitivity | §8 | n_intervals sweep, overlap sweep; "stable range" principle; Goldilocks zone |
| 9 | The Nerve Theorem Connection | §9 | Theorem 1 (Mapper–Nerve correspondence), when is the cover "good", Persistent Nerve Theorem (Theorem 2) |
| 10 | Financial Market Regimes | §10 | Synthetic regime data; rolling features; Mapper reveals regime clusters + transition edges |
| 11 | Stability & Convergence | §11 | Theorem 3 (Carrière & Oudot); bootstrap Mapper; stable feature heuristic |
| 12 | Connections & Further Reading | §12 | Reeb graph, multiscale Mapper, Ball Mapper; connection table to other formalML topics; full references |

---

## 3. TheoremBlock Usage

Use the existing `TheoremBlock.astro` component (spec: v1 brief §7) for all numbered formal elements. The notebook contains:

| Type | # | Name | TheoremBlock `type` |
|---|---|---|---|
| Definition | 1 | Filter function (lens) | `definition` |
| Definition | 2 | Interval cover | `definition` |
| Definition | 3 | Pullback cover | `definition` |
| Definition | 4 | Refined cover | `definition` |
| Definition | 5 | Mapper graph | `definition` |
| Proposition | 1 | Overlap and connectivity | `proposition` (with `proof` block) |
| Theorem | 1 | Mapper–Nerve correspondence | `theorem` (stated with justification, not full proof) |
| Theorem | 2 | Persistent Nerve Theorem (Chazal & Oudot 2008) | `theorem` (stated without proof) |
| Theorem | 3 | Statistical convergence (Carrière & Oudot 2018) | `theorem` (stated without proof) |

Remarks → `type="remark"`. Examples → `type="example"`.

**LaTeX symbols to verify render correctly:**
- `f: X \to \mathbb{R}^m` — Filter function signature
- `f^{-1}(\mathcal{V})` — Pullback notation
- `\text{Nrv}(\mathcal{W})` — Nerve of refined cover (same symbol as Čech page)
- `C_{i,j}` — Cluster indexing (interval i, cluster j)
- `\mathcal{V} = \{V_1, \ldots, V_n\}` — Interval cover
- `\mathcal{W} = \{C_{i,j}\}` — Refined cover
- `\simeq` — Homotopy equivalence (same as Čech page)
- `H_0, H_1` — Homology groups
- `\hat{p}(x)` — Density estimator
- `\sqcup` — Disjoint union (in clustering decomposition)

---

## 4. Interactive Visualizations

New React components in `src/components/viz/`. All use `client:visible` hydration. Follow patterns from existing viz components (`SimplicialComplex.tsx`, `PersistenceDiagram.tsx`, `FiltrationSlider.tsx`, `PhantomCycleExplorer.tsx`, `CoverNerveVisualizer.tsx`). Use shared utilities from `src/components/viz/shared/`.

### 4a. MapperPipelineExplorer (Flagship)

**File:** `src/components/viz/MapperPipelineExplorer.tsx`

An interactive, step-by-step walkthrough of the Mapper pipeline on a small circle dataset (~80 points):

- **Step 1 (Filter):** Point cloud colored by filter value (x-coordinate). Slider to choose between eccentricity / PCA / x-coordinate.
- **Step 2 (Cover):** 1D bar below the scatter plot showing overlapping intervals. Slider for `n_intervals` (5–20) and `overlap` (0.1–0.7).
- **Step 3 (Pullback):** Points colored by their pullback assignment; points in overlap regions highlighted.
- **Step 4 (Cluster):** Convex hulls drawn around clusters within each pullback.
- **Step 5 (Nerve/Graph):** The Mapper graph appears beside the point cloud, with edges animated in.

User can step through stages 1→5 with a "Next Step" button, or freely adjust parameters and see the graph update in real time.

**Implementation notes:**
- Pre-compute a small circle dataset (80 points) and embed as JSON.
- Implement a simplified Mapper in TypeScript (the from-scratch algorithm is simple enough to run client-side on ~80 points).
- Use D3 for SVG rendering (consistent with existing viz components).
- The nerve/graph layout can use D3-force or a simple spring layout.

**Reference:** Notebook §2 (pipeline), §5 (from-scratch), §9 (cover → nerve → graph pipeline figure).

### 4b. ParameterSensitivityGrid

**File:** `src/components/viz/ParameterSensitivityGrid.tsx`

A 2×4 grid of small Mapper graphs (same circle data) showing the effect of varying parameters:
- **Row 1:** n_intervals ∈ {5, 10, 15, 25} with fixed overlap = 0.35
- **Row 2:** overlap ∈ {0.1, 0.25, 0.45, 0.65} with fixed n_intervals = 12

Each cell shows a small graph with annotations for node count and cycle count. A "Goldilocks zone" highlights the parameter range where the topology is stable.

Can use **pre-computed graphs** (JSON) from the notebook's parameter sweep (§8) to avoid running Mapper in the browser for 8 configurations.

**Reference:** Notebook §8 (parameter sweep figure).

### 4c. MarketRegimeMapper

**File:** `src/components/viz/MarketRegimeMapper.tsx`

A three-panel visualization showing the financial Mapper application:
- **Left:** Synthetic returns time series colored by regime (blue = bull, red = bear, green = recovery).
- **Center:** PCA scatter of the feature space, colored by regime.
- **Right:** Mapper graph with nodes colored by dominant regime, edges representing transitions.

Use **pre-computed data** from the notebook's financial section (§10). Hovering over a node in the Mapper graph highlights the corresponding time points in the left panel and feature-space points in the center panel.

**Reference:** Notebook §10 (financial application figure).

### 4d. FilterFunctionComparison (Optional — lower priority)

**File:** `src/components/viz/FilterFunctionComparison.tsx`

Side-by-side view of the same figure-eight dataset colored by three different filters (eccentricity, density, PCA). Clicking a filter tab shows the resulting Mapper graph below. Pre-computed data from notebook §3.

**Reference:** Notebook §3 (filter function comparison figure).

---

## 5. Static Figures

Run the notebook, export figures as PNG (or SVG where practical), place in:

```
public/images/topics/mapper-algorithm/
```

| Figure | Notebook Source | Filename |
|---|---|---|
| Mapper motivation triptych (cloud → filter → graph) | §1, cell 3 | `mapper-motivation.png` |
| Filter function comparison (eccentricity, density, PCA) | §3, cell 8 | `filter-functions.png` |
| Covering & pullback visualization | §4, cell 11 | `covering-pullback.png` |
| From-scratch Mapper on circle | §5, cell 15 | `mapper-from-scratch-circle.png` |
| Classic examples (figure-eight, torus) | §6, cell 18 | `mapper-classic-examples.png` |
| KeplerMapper comparison | §7, cell 21 | `keplermapper-examples.png` |
| Parameter sensitivity grid | §8, cell 24 | `parameter-sensitivity.png` |
| Cover → Nerve → Mapper pipeline | §9, cell 27 | `mapper-nerve-pipeline.png` |
| Financial market regimes | §10, cell 29 | `mapper-financial-regimes.png` |
| Bootstrap stability | §11, cell 32 | `mapper-stability.png` |

Reference in MDX via standard image tags. These serve as fallbacks and context for the interactive versions.

---

## 6. Code Blocks

Include selected Python snippets from the notebook as syntax-highlighted code blocks (Shiki, not runnable). Include these five:

| Block | Notebook Source | Purpose |
|---|---|---|
| `mapper_from_scratch()` | §5, cell 14 | The complete from-scratch implementation — pedagogical centerpiece |
| Filter function computation | §3, cell 8 | Shows eccentricity, density (KDE), and PCA filter computation |
| KeplerMapper workflow | §7, cell 21 | The 3-line production API pattern: `fit_transform → map → visualize` |
| Financial feature engineering | §10, cell 29 | Rolling mean, volatility, skewness computation from returns |
| Bootstrap stability check | §11, cell 32 | The bootstrap Mapper stability pattern |

---

## 7. Cross-Links

### Outbound (from this page)

- `[Simplicial Complexes](/topics/simplicial-complexes)` — reference VR construction and simplicial complex definition
- `[Persistent Homology](/topics/persistent-homology)` — reference filtrations, persistence diagrams, Betti numbers
- `[Čech Complexes & Nerve Theorem](/topics/cech-complexes)` — reference the Nerve Theorem (Theorem 2 on that page), nerve construction (§3), good covers (Definition 5)

### Inbound (update existing pages)

- **`cech-complexes.mdx`:** In the "Applications & Connections" section (§8), where Mapper is currently mentioned, update to include a live link: *"The [Mapper Algorithm](/topics/mapper-algorithm) provides the full construction and implementation."*
- **`persistent-homology.mdx`:** In the applications/connections section, add: *"The [Mapper Algorithm](/topics/mapper-algorithm) uses persistent homology on its output graph to identify stable topological features across parameter choices."*
- **`simplicial-complexes.mdx`:** In the section discussing different complex constructions, add a forward reference: *"The [Mapper Algorithm](/topics/mapper-algorithm) constructs a 1-dimensional simplicial complex (a graph) that captures the data's topological skeleton."*

---

## 8. Paths Page Update

On `/paths` (the Topology & TDA section), change the Mapper Algorithm entry from "Planned" status to a published link matching the format of the other published entries.

**Before:**
```
Mapper Algorithm  Planned
```

**After:**
```
Mapper Algorithm  Advanced  → linked to /topics/mapper-algorithm
```

**Note:** This will be the first entry with the "Advanced" difficulty badge. Verify that the badge styling handles this level correctly (the existing CSS should cover it via the difficulty field in frontmatter, but confirm the color/style for "advanced" is defined in the design system).

---

## 9. Prerequisite Graph Update

If the DAGGraph component and/or `curriculum-graph.json` (or equivalent data source) are implemented, add:

**Node:**
```json
{
  "id": "mapper-algorithm",
  "label": "The Mapper Algorithm",
  "track": "topology",
  "difficulty": "advanced",
  "status": "published"
}
```

**Edges:**
- `simplicial-complexes → mapper-algorithm`
- `persistent-homology → mapper-algorithm`
- `cech-complexes → mapper-algorithm` (this is the primary dependency — Nerve Theorem)
- `mapper-algorithm → barcodes-bottleneck-distance` (forward edge — Mapper stability uses bottleneck distance)

If the graph is auto-generated from frontmatter `prerequisites` fields, the first three edges are handled automatically. The forward edge to Barcodes & Bottleneck Distance should be added when that topic is created.

---

## 10. Notebook File

Place the companion notebook in the repo:

```
notebooks/mapper-algorithm-notebook.ipynb
```

This matches the pattern from the existing notebooks directory and the `notebookPath` frontmatter field. The notebook is a content reference, not served to users — it lives in the repo for developer/author reference.

---

## 11. SEO & OG

If the OG image generation script (`scripts/generate-og-images.ts`) from v1 brief §12 is active, it should auto-generate an OG image for this topic. If not, create a manual OG image following the existing pattern and place it in `public/og/mapper-algorithm.png`.

Meta tags should be generated by the existing `TopicLayout.astro` from frontmatter fields. Verify:
- `<title>`: "The Mapper Algorithm | formalML"
- `<meta name="description">`: Uses the `abstract` field
- OG tags: title, description, image

---

## 12. Testing Checklist

- [ ] All KaTeX renders correctly (especially `f^{-1}(\mathcal{V})`, `\text{Nrv}(\mathcal{W})`, `\sqcup`, `\hat{p}(x)`)
- [ ] Prerequisite chips link to `/topics/simplicial-complexes`, `/topics/persistent-homology`, and `/topics/cech-complexes`
- [ ] Topic card appears on `/topics` index with correct "Advanced" difficulty badge and domain tag
- [ ] Topic appears on `/paths` as published (Advanced, linked)
- [ ] Forward references added to Simplicial Complexes, Persistent Homology, and Čech Complexes pages
- [ ] `MapperPipelineExplorer` step-through works across all 5 stages
- [ ] `MapperPipelineExplorer` parameter sliders update the graph in real time
- [ ] `ParameterSensitivityGrid` renders all 8 cells with correct annotations
- [ ] `MarketRegimeMapper` cross-highlighting works (hover node → highlights time series + PCA)
- [ ] `FilterFunctionComparison` tab switching works (if implemented)
- [ ] All static figures load from `public/images/topics/mapper-algorithm/`
- [ ] Page is responsive (viz components stack vertically on mobile)
- [ ] "Advanced" difficulty badge is styled correctly (check design system has this level)
- [ ] Pagefind indexes the new topic on rebuild
- [ ] Build succeeds with zero errors: `pnpm build`

---

## 13. Build Order

1. Create `mapper-algorithm.mdx` with full frontmatter and all markdown/LaTeX content. Use `TheoremBlock` for all formal elements. No interactive components yet.
2. Add notebook to `notebooks/` directory.
3. Export and add static figures to `public/images/topics/mapper-algorithm/`.
4. Build `MapperPipelineExplorer.tsx` — this is the flagship component, build first and test in isolation with `client:visible`. Implement the TypeScript Mapper algorithm (simple enough for ~80 points on the client side).
5. Build `ParameterSensitivityGrid.tsx` — can use pre-computed data from notebook, simplest of the group.
6. Build `MarketRegimeMapper.tsx` — pre-computed data, focus on the cross-highlighting interaction.
7. (Optional) Build `FilterFunctionComparison.tsx` — lower priority, skip if time is tight.
8. Embed all components in the MDX at their appropriate section positions.
9. Add cross-link forward references to `simplicial-complexes.mdx`, `persistent-homology.mdx`, and `cech-complexes.mdx`.
10. Update `/paths` page — change Mapper entry from Planned to Published (Advanced).
11. Update prerequisite graph data (if applicable).
12. Verify "Advanced" difficulty badge styling exists in the design system.
13. Run testing checklist (§12).
14. Commit and deploy.

---

## Appendix: Key Differences from the Čech Complexes Brief

This topic has several characteristics that distinguish it from the Čech Complexes implementation:

1. **First "Advanced" topic.** Verify that the difficulty badge, card styling, and paths page all handle the "advanced" level correctly.
2. **Three prerequisites** (vs. two for Čech). The prerequisite chips row will be wider — verify layout.
3. **More interactive components** (3–4 vs. 3). The flagship `MapperPipelineExplorer` requires a client-side Mapper implementation in TypeScript, which is the most complex viz component on the site so far.
4. **Financial application.** The `MarketRegimeMapper` component introduces domain-specific coloring (regime labels) and cross-panel highlighting — a new interaction pattern.
5. **Inbound cross-links to three pages** (vs. two). All three existing Topology & TDA pages need forward references added.
6. **Longer estimated read time** (40 min vs. 30 min). This is the most substantial topic page to date.

---

*Brief version: v1 | Last updated: 2026-03-15 | Author: Jonathan Rocha*  
*Reference notebook: `mapper-algorithm-notebook.ipynb`*  
*Parent brief: `formalml-handoff-brief-v1.md`*  
*Sibling brief: `formalml-cech-complexes-handoff-brief.md`*
