# Barcodes & Bottleneck Distance Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the "Barcodes & Bottleneck Distance" topic article to formalml.com as the 5th article in the Topology & TDA track, with interactive visualizations, working code examples, and curriculum data updates.

**Architecture:** MDX article following the exact pattern of `persistent-homology.mdx` — same frontmatter schema, section ordering, TheoremBlock usage, and code block style. Two new React/D3 visualization components (`BottleneckMatchingViz` and `StabilityDemo`) reuse existing shared utilities (`useD3`, `useResizeObserver`, `dimensionColors`, `PersistenceInterval` type). Curriculum graph and track data updated to mark the topic as published and add "Statistical TDA" as next planned entry.

**Tech Stack:** Astro 6 + MDX, React 19, D3 7, TypeScript, Tailwind CSS 4, KaTeX (via remark-math/rehype-katex)

**Reference spec:** `docs/plans/formalml-barcodes-bottleneck-handoff.md`

**Reference implementation:** `src/content/topics/persistent-homology.mdx` (match exactly)

**Companion notebook:** Already exists at `drafts/notebooks/barcodes-bottleneck/barcodes-bottleneck.ipynb`

---

## File Structure

### Files to Create

| File | Responsibility |
|------|---------------|
| `src/content/topics/barcodes-bottleneck.mdx` | Full article content — frontmatter, imports, 6 formal framework sections, visual intuition, working code, connections, references |
| `src/components/viz/BottleneckMatchingViz.tsx` | Interactive visualization showing two pre-defined persistence diagram pairs with optimal bottleneck matching lines drawn between matched points, selectable via tabs/buttons, and displaying the bottleneck distance. |
| `src/components/viz/StabilityDemo.tsx` | Noise slider demo — generates a point cloud, adds Gaussian noise at user-controlled σ, shows persistence diagram shifting and displays d_B and d_H values, confirming the stability bound empirically. |

### Files to Modify

| File | Change |
|------|--------|
| `src/data/curriculum-graph.json` | Change `barcodes-bottleneck` node status from `"planned"` to `"published"`, set `url` to `"/topics/barcodes-bottleneck"`. Add `statistical-tda` node as `"planned"`. Add edge `barcodes-bottleneck → statistical-tda`. |
| `src/data/curriculum.ts` | Remove `'Barcodes & Bottleneck Distance'` from topology `planned` array. Add `'Statistical TDA'` to topology `planned` array. |

---

## Chunk 1: Interactive Visualization Components

### Task 1: Create BottleneckMatchingViz Component

**Files:**
- Create: `src/components/viz/BottleneckMatchingViz.tsx`
- Uses: `src/components/viz/shared/useD3.ts`, `src/components/viz/shared/useResizeObserver.ts`, `src/components/viz/shared/colorScales.ts`, `src/components/viz/shared/types.ts`

This component renders two small persistence diagrams side by side (or stacked on mobile) with the optimal bottleneck matching drawn as lines between matched points. It computes a greedy approximation of the bottleneck matching in the browser.

**Key design decisions:**
- Two pre-defined diagram pairs (selectable via tabs/buttons): "Circle vs Cluster" and "Circle vs Figure-Eight"
- Matching lines drawn from points in diagram A to matched points in diagram B (or to diagonal for unmatched)
- Display the bottleneck distance value prominently
- Use `dimensionColors` for point coloring (H₀ teal, H₁ purple)
- Responsive via `useResizeObserver`, D3 rendering via `useD3`

- [ ] **Step 1: Create the component file with basic structure**

Create `src/components/viz/BottleneckMatchingViz.tsx` with:
- Imports from shared utilities
- Pre-defined persistence diagram data (hardcoded from notebook results for Circle, Cluster, Figure-Eight H₁ diagrams)
- A greedy bottleneck matching function that computes approximate optimal matching
- React component with state for selected diagram pair
- D3 rendering: left diagram, right diagram, matching lines between them
- Display of d_B value

```tsx
// Key interfaces and data structures:

interface DiagramPoint {
  birth: number;
  death: number;
  dimension: number;
}

interface MatchingPair {
  from: DiagramPoint | { birth: number; death: number }; // diagonal point
  to: DiagramPoint | { birth: number; death: number };
  cost: number;
}

// Pre-defined diagrams derived from notebook outputs:
// Circle H₁: one dominant bar ~[0.39, 1.80]
// Cluster H₁: several short bars near diagonal
// Figure-Eight H₁: two dominant bars
```

Component layout:
```
┌─────────────────────────────────────────────┐
│  [Circle vs Cluster]  [Circle vs Figure-Eight]  │  ← tab buttons
├─────────────────────────────────────────────┤
│     Diagram A          Diagram B            │
│     ●                  ●                    │
│      ● ─────────────── ●                   │
│     ●  ─────────── ✕ (diagonal)            │
│                                             │
│     d_B = 0.7029                            │
└─────────────────────────────────────────────┘
```

- [ ] **Step 2: Verify component renders in isolation**

Temporarily add to an existing MDX file or test page to verify D3 rendering works.

Run: `pnpm dev` and check browser.

- [ ] **Step 3: Commit**

```bash
git add src/components/viz/BottleneckMatchingViz.tsx
git commit -m "feat: add BottleneckMatchingViz component for barcodes article"
```

---

### Task 2: Create StabilityDemo Component

**Files:**
- Create: `src/components/viz/StabilityDemo.tsx`
- Uses: `src/components/viz/shared/useD3.ts`, `src/components/viz/shared/useResizeObserver.ts`, `src/components/viz/shared/colorScales.ts`, `src/components/viz/shared/types.ts`

This component generates a circle point cloud, lets the user control noise via a slider, computes persistence in the browser (H₀ via union-find, reuse pattern from `LinkedVizDemo.tsx`), and shows the persistence diagram shifting. Displays d_B and 2·d_H values to confirm stability.

**Key design decisions:**
- Reuse the H₀ union-find computation pattern from `LinkedVizDemo.tsx`
- H₁ is shown via a heuristic (same approach as LinkedVizDemo) — for educational purposes this is sufficient
- Left panel: point cloud scatter plot with noise overlay
- Right panel: persistence diagram (reuse `PersistenceDiagram` component with `mode="diagram"`)
- Bottom: slider for σ (0 to 0.5), display d_B and 2·d_H values
- Deterministic "noise" via seeded positions to avoid jarring re-renders

- [ ] **Step 1: Create the component file**

Create `src/components/viz/StabilityDemo.tsx` with:
- Circle point cloud generator (deterministic base + noise scaled by σ)
- H₀ persistence computation (union-find, from LinkedVizDemo pattern)
- Noise slider (range input)
- Left panel: SVG scatter of noisy point cloud
- Right panel: embedded `PersistenceDiagram` component showing current diagram
- Metric display: d_B ≈ ..., 2·d_H ≈ ..., bound holds: ✓

```tsx
// Layout:
// ┌──────────────┬──────────────────────┐
// │  Point Cloud  │  Persistence Diagram │
// │   (noisy)     │    (shifts w/ noise) │
// ├──────────────┴──────────────────────┤
// │  σ: ═══════●═══════  0.15           │
// │  d_B ≈ 0.043  |  2·d_H ≈ 0.122     │
// │  Stability bound holds ✓            │
// └─────────────────────────────────────┘
```

- [ ] **Step 2: Verify component renders**

Run: `pnpm dev` and check browser.

- [ ] **Step 3: Commit**

```bash
git add src/components/viz/StabilityDemo.tsx
git commit -m "feat: add StabilityDemo component for barcodes article"
```

---

## Chunk 2: MDX Article

### Task 3: Create the MDX Article

**Files:**
- Create: `src/content/topics/barcodes-bottleneck.mdx`

This is the main deliverable. Follow the exact structure of `persistent-homology.mdx`:

**Frontmatter** (must match `content.config.ts` schema exactly):
```yaml
---
title: "Barcodes & Bottleneck Distance"
subtitle: "Comparing persistence diagrams — the metrics that make TDA a rigorous statistical tool"
status: "published"
difficulty: "intermediate"
prerequisites:
  - "persistent-homology"
tags:
  - "topology"
  - "tda"
  - "persistent-homology"
  - "metric-spaces"
domain: "topology"
videoId: null
notebookPath: "notebooks/barcodes-bottleneck.ipynb"
githubUrl: null
datePublished: 2026-03-16
estimatedReadTime: 30
abstract: "Persistence barcodes decompose a persistence module into interval summands, giving a complete invariant of the multiscale topology captured by a filtration. The bottleneck and Wasserstein distances turn the space of persistence diagrams into a metric space, and the Stability Theorem guarantees that small perturbations of the input produce small changes in the diagram — the theoretical foundation that makes topological data analysis a rigorous tool for real-world data."
connections:
  - topic: "persistent-homology"
    relationship: "builds on"
  - topic: "metric-spaces"
    relationship: "uses concepts from"
references:
  - type: "paper"
    title: "Stability of Persistence Diagrams"
    authors: "David Cohen-Steiner, Herbert Edelsbrunner & John Harer"
    year: 2007
    url: "https://doi.org/10.1007/s00454-006-1276-5"
    note: "The foundational stability result — Theorem 4 in this topic"
  - type: "paper"
    title: "Lipschitz Functions Have Lp-Stable Persistence"
    authors: "David Cohen-Steiner, Herbert Edelsbrunner, John Harer & Yuriy Mileyko"
    year: 2010
    url: "https://doi.org/10.1007/s10208-010-9060-6"
    note: "Wasserstein stability for persistence diagrams"
  - type: "paper"
    title: "Computing Persistent Homology"
    authors: "Afra Zomorodian & Gunnar Carlsson"
    year: 2005
    url: "https://doi.org/10.1007/s00454-004-1146-y"
    note: "The Structure Theorem for persistence modules"
  - type: "paper"
    title: "Induced Matchings and the Algebraic Stability of Persistence Barcodes"
    authors: "Ulrich Bauer & Michael Lesnick"
    year: 2015
    url: "https://doi.org/10.1007/s41468-015-0004-0"
    note: "The Isometry Theorem — bottleneck distance equals interleaving distance"
  - type: "paper"
    title: "Confidence Sets for Persistence Diagrams"
    authors: "Brittany Terese Fasy, Fabrizio Lecci, Alessandro Rinaldo, Larry Wasserman, Sivaraman Balakrishnan & Aarti Singh"
    year: 2014
    url: "https://doi.org/10.1214/14-AOS1252"
    note: "Bootstrap methods for statistical inference on persistence diagrams"
  - type: "paper"
    title: "Statistical Topological Data Analysis using Persistence Landscapes"
    authors: "Peter Bubenik"
    year: 2015
    url: "https://jmlr.org/papers/v16/bubenik15a.html"
    note: "Persistence landscapes as Banach space elements for statistical analysis"
  - type: "book"
    title: "Computational Topology: An Introduction"
    authors: "Herbert Edelsbrunner & John Harer"
    year: 2010
    note: "Chapters 8-9 on stability and distances between diagrams"
  - type: "paper"
    title: "Geometry Helps to Compare Persistence Diagrams"
    authors: "Michael Kerber, Dmitriy Morozov & Arnur Nigmetov"
    year: 2017
    url: "https://doi.org/10.1145/3064175"
    note: "Efficient algorithms for Wasserstein distance computation"
---
```

**Imports** (after frontmatter):
```tsx
import TheoremBlock from '../../components/ui/TheoremBlock.astro';
import BottleneckMatchingViz from '../../components/viz/BottleneckMatchingViz.tsx';
import StabilityDemo from '../../components/viz/StabilityDemo.tsx';
```

**Article sections** (follow handoff brief sections 3.1-3.6 exactly):

1. **Overview & Motivation** (~400 words) — Hook about comparing diagrams, motivating questions, frame the answer, connect to practice
2. **Formal Framework** with 6 subsections:
   - 3.1: Barcodes as Interval Decompositions (Definition + Structure Theorem + equivalence remark + example)
   - 3.2: The Space of Persistence Diagrams (Definition of diagram, diagonal, partial matching)
   - 3.3: Bottleneck Distance (Definition, geometric interpretation, metric theorem, worked example)
   - 3.4: Wasserstein Distance (Definition, contrast with bottleneck, when to use which, limit remark)
   - 3.5: The Stability Theorem (Full treatment — Cohen-Steiner et al. 2007, proof sketch, corollary, Wasserstein stability)
   - 3.6: Isometry Theorem (Bauer & Lesnick 2015, brief statement, significance)
3. **Visual Intuition** — Embed `BottleneckMatchingViz` and `StabilityDemo` with explanatory text
4. **Working Code** — 4 Python code blocks from the notebook (computing barcodes, bottleneck distance, Wasserstein, stability verification)
5. **Connections & Applications** (~200 words) — Statistical TDA, persistence landscapes, optimal transport, sheaf theory

**Style requirements:**
- All math in KaTeX: `$inline$` and `$$display$$`
- Use `<TheoremBlock type="definition|theorem|proof|remark|example" number={N}>` for formal content
- Number definitions, theorems, examples sequentially
- Python code blocks with `ripser`, `persim`, `gudhi`, `numpy`, `scipy`
- Tone: rigorous but motivated, practitioner-aware (match persistent-homology.mdx)

- [ ] **Step 1: Write the frontmatter and imports section**

- [ ] **Step 2: Write Overview & Motivation section**

- [ ] **Step 3: Write Formal Framework sections 3.1-3.2 (Barcodes, Space of Diagrams)**

- [ ] **Step 4: Write Formal Framework sections 3.3-3.4 (Bottleneck, Wasserstein)**

- [ ] **Step 5: Write Formal Framework sections 3.5-3.6 (Stability, Isometry)**

- [ ] **Step 6: Write Visual Intuition section with component embeds**

- [ ] **Step 7: Write Working Code section (4 Python code blocks)**

- [ ] **Step 8: Write Connections & Applications section**

- [ ] **Step 9: Commit**

```bash
git add src/content/topics/barcodes-bottleneck.mdx
git commit -m "feat: add Barcodes & Bottleneck Distance topic article"
```

---

## Chunk 3: Curriculum Data Updates

### Task 4: Update curriculum-graph.json

**Files:**
- Modify: `src/data/curriculum-graph.json`

Changes:
1. Change the `barcodes-bottleneck` node:
   - `"status": "planned"` → `"status": "published"`
   - `"url": null` → `"url": "/topics/barcodes-bottleneck"`
2. Add new node after `barcodes-bottleneck`:
   ```json
   { "id": "statistical-tda", "label": "Statistical TDA", "domain": "topology", "status": "planned", "url": null }
   ```
3. Add new edge:
   ```json
   { "source": "barcodes-bottleneck", "target": "statistical-tda" }
   ```

- [ ] **Step 1: Apply the three changes to curriculum-graph.json**

- [ ] **Step 2: Verify JSON is valid**

Run: `node -e "JSON.parse(require('fs').readFileSync('src/data/curriculum-graph.json', 'utf8')); console.log('Valid JSON')"`

### Task 5: Update curriculum.ts

**Files:**
- Modify: `src/data/curriculum.ts`

Changes:
1. In the topology track `planned` array, remove `'Barcodes & Bottleneck Distance'`
2. Add `'Statistical TDA'` to the topology track `planned` array (before 'Sheaf Theory')

- [ ] **Step 1: Apply changes to curriculum.ts**

- [ ] **Step 2: Commit both data file changes**

```bash
git add src/data/curriculum-graph.json src/data/curriculum.ts
git commit -m "feat: mark Barcodes & Bottleneck Distance as published, add Statistical TDA placeholder"
```

---

## Chunk 4: Build Verification

### Task 6: Build and Verify

- [ ] **Step 1: Run pnpm dev and check for build errors**

Run: `pnpm build` (in the worktree directory)
Expected: Clean build with no errors

- [ ] **Step 2: Verify article renders at /topics/barcodes-bottleneck**

Check that:
- Article loads without errors
- Math renders correctly (especially `$\inf$`, `$\sup$`, `$\|\cdot\|_\infty$`)
- TheoremBlock components render with proper styling
- Interactive visualizations load and respond to user input
- Code blocks render with syntax highlighting

- [ ] **Step 3: Verify paths page shows article as published**

Check that:
- Barcodes & Bottleneck Distance appears as a published node in the curriculum graph
- Statistical TDA appears as a planned node
- The prerequisite edge from persistent-homology is visible

- [ ] **Step 4: Fix any issues found**

- [ ] **Step 5: Final commit if fixes were needed**
