# Claude Code Handoff Brief: formalML — README Rewrite & /paths Page Enhancements

**Project:** formalml.com  
**Repo:** github.com/jonx0037/formalML  
**Stack:** Astro (SSG), MDX, TypeScript, Tailwind CSS, pnpm, Vercel  
**Author:** Jonathan Rocha  
**Date:** March 2026  
**Status:** Ready for implementation

---

## Scope

Two discrete tasks:

1. **README.md rewrite** — Replace the default Astro starter kit boilerplate with a proper project README
2. **/paths page enhancements** — Add an interactive prerequisite graph, cross-navigation improvements, and structural refinements

These are independent and can be committed separately.

---

## Task 1: README.md Rewrite

### Problem

The current `README.md` is the unmodified Astro starter template ("Astro Starter Kit: Minimal"). The repo is linked from the footer of every page on formalml.com. Anyone who clicks through — a PhD committee member, a potential collaborator, a hiring manager — sees generic boilerplate instead of a description of the project.

### Deliverable

Replace `README.md` at the repo root with a project-specific README. Use the content and structure below as the source of truth.

### README Content

```markdown
# formalML

**The mathematical machinery behind modern machine learning.**

Deep-dive explainers combining rigorous mathematics, interactive visualizations, and working code. Built for practitioners, graduate students, and researchers.

🌐 **[www.formalml.com](https://www.formalml.com)**

---

## What This Is

formalML is a curated collection of long-form explainers on the mathematical foundations of modern ML. Every topic receives a three-pillar treatment:

1. **Rigorous exposition** — Formal definitions, theorems, and proofs presented with full mathematical detail
2. **Interactive visualization** — Embedded widgets that let you manipulate parameters and watch the math come alive (e.g., drag an ε slider to watch a Vietoris-Rips complex form around a point cloud)
3. **Working code** — Production-oriented Python implementations you can run immediately, with bridges to standard libraries like `ripser`, `gudhi`, and `scikit-tda`

The site exists because the gap between textbook formalism and practical ML understanding is wider than it needs to be.

## Published Topics

| Topic | Track | Level | Description |
|-------|-------|-------|-------------|
| [Simplicial Complexes](https://www.formalml.com/topics/simplicial-complexes) | Topology & TDA | Foundational | The combinatorial scaffolding that turns point clouds into topology |
| [Persistent Homology](https://www.formalml.com/topics/persistent-homology) | Topology & TDA | Intermediate | Tracking topological features across scales — the workhorse of TDA |

## Curriculum Roadmap

The full roadmap spans 8 mathematical tracks with 34+ planned topics:

- **Topology & TDA** — Simplicial complexes → persistent homology → Čech complexes → Mapper → barcodes → sheaf theory
- **Linear Algebra** — Spectral theorem, SVD, PCA, tensor decompositions
- **Probability & Statistics** — Measure-theoretic foundations, concentration inequalities, PAC learning
- **Optimization** — Convex analysis, gradient methods, proximal methods, KKT
- **Differential Geometry** — Smooth manifolds, Riemannian metrics, information geometry
- **Information Theory** — Shannon entropy, divergences, rate-distortion, MDL
- **Graph Theory** — Graph Laplacians, random walks, expanders, GNNs
- **Category Theory** — Functors, natural transformations, adjunctions, monads

See the full interactive roadmap at **[formalml.com/paths](https://www.formalml.com/paths)**.

## Tech Stack

| Layer | Tool |
|-------|------|
| Framework | [Astro](https://astro.build) (static site generation) |
| Content | MDX with KaTeX for math rendering |
| Styling | Tailwind CSS |
| Visualizations | Custom interactive components (TypeScript) |
| Package manager | pnpm |
| Hosting | Vercel |

## Project Structure

```
├── src/
│   ├── pages/          # Astro page routes
│   ├── content/        # MDX topic files
│   ├── components/     # Astro/TS components (visualizations, layout)
│   └── layouts/        # Page layout templates
├── public/             # Static assets
├── drafts/             # Work-in-progress topic drafts
├── docs/plans/         # Planning documents
├── astro.config.mjs    # Astro configuration
├── package.json
└── tsconfig.json
```

## Local Development

```bash
# Install dependencies
pnpm install

# Start dev server (localhost:4321)
pnpm dev

# Build for production
pnpm build

# Preview production build
pnpm preview
```

## Author

**Jonathan Rocha** — Data scientist and researcher. MS Data Science (SMU), MA English (Texas A&M University–Central Texas), BA History (Texas A&M University). Research interests: time-series data mining, topology-aware deep learning.

- GitHub: [@jonx0037](https://github.com/jonx0037)
- Consultancy: [DataSalt LLC](https://datasalt.ai)

## License

© 2026 formalML. All rights reserved.
```

### Implementation Notes

- The nested code fence in "Project Structure" needs to render correctly — use 4-space indentation or a different fence style if the triple-backtick nesting causes issues in the markdown
- Verify the `src/` directory structure matches reality before committing; adjust folder names if Claude Code scaffolded differently than shown above (e.g., `content/` might be `content/topics/` or similar)
- The project structure section should reflect the *actual* repo tree — run `find src -type d -maxdepth 2` and adjust

---

## Task 2: /paths Page Enhancements

### 2a. Interactive Prerequisite Graph

#### Problem

The curriculum roadmap is currently a flat list of tracks with planned/published badges. There's no visual representation of how topics relate across tracks. The Simplicial Complexes topic already declares `builds on: metric-spaces` and `is prerequisite for: persistent-homology`, but these dependency relationships aren't surfaced on the /paths page.

#### Deliverable

Add an **interactive prerequisite graph** above or below the existing track listing. This is a node-link diagram where:

- **Nodes** = topics (both published and planned)
- **Edges** = prerequisite relationships (directed: A → B means "A is prerequisite for B")
- **Node styling** distinguishes published (solid, clickable, links to topic page) from planned (dashed outline, non-clickable, muted color)
- **Track membership** is encoded via node color (each of the 8 tracks gets a distinct color from the site's existing palette)

#### Technical Approach

Use **D3.js** (available via CDN or npm) for the graph rendering. Astro supports client-side JS via `<script>` tags in `.astro` pages or via framework components with `client:load` directives.

**Data source:** Create a `src/data/curriculum-graph.json` (or `.ts`) file that defines the graph:

```jsonc
{
  "nodes": [
    {
      "id": "simplicial-complexes",
      "label": "Simplicial Complexes",
      "track": "topology-tda",
      "level": "foundational",
      "status": "published",   // "published" | "planned"
      "url": "/topics/simplicial-complexes"
    },
    {
      "id": "persistent-homology",
      "label": "Persistent Homology",
      "track": "topology-tda",
      "level": "intermediate",
      "status": "published",
      "url": "/topics/persistent-homology"
    },
    {
      "id": "cech-complexes",
      "label": "Čech Complexes & Nerve Theorem",
      "track": "topology-tda",
      "level": "intermediate",
      "status": "planned",
      "url": null
    }
    // ... all 34+ topics
  ],
  "edges": [
    { "source": "simplicial-complexes", "target": "persistent-homology" },
    { "source": "simplicial-complexes", "target": "cech-complexes" },
    { "source": "persistent-homology", "target": "barcodes-bottleneck" },
    { "source": "cech-complexes", "target": "mapper-algorithm" },
    { "source": "spectral-theorem", "target": "svd" },
    { "source": "svd", "target": "pca-low-rank" },
    { "source": "measure-theoretic-probability", "target": "concentration-inequalities" },
    { "source": "concentration-inequalities", "target": "pac-learning" },
    { "source": "convex-analysis", "target": "gradient-descent" },
    { "source": "gradient-descent", "target": "proximal-methods" },
    { "source": "convex-analysis", "target": "lagrangian-kkt" },
    { "source": "smooth-manifolds", "target": "riemannian-geometry" },
    { "source": "riemannian-geometry", "target": "geodesics-curvature" },
    { "source": "geodesics-curvature", "target": "information-geometry" },
    { "source": "shannon-entropy", "target": "kl-divergence" },
    { "source": "kl-divergence", "target": "rate-distortion" },
    { "source": "shannon-entropy", "target": "mdl" },
    { "source": "graph-laplacians", "target": "random-walks-mixing" },
    { "source": "random-walks-mixing", "target": "expander-graphs" },
    { "source": "graph-laplacians", "target": "message-passing-gnns" },
    { "source": "categories-functors", "target": "natural-transformations" },
    { "source": "natural-transformations", "target": "adjunctions" },
    { "source": "adjunctions", "target": "monads-comonads" }
    // Cross-track edges (these are the interesting ones):
    // { "source": "spectral-theorem", "target": "graph-laplacians" },
    // { "source": "measure-theoretic-probability", "target": "bayesian-nonparametrics" },
    // { "source": "information-geometry", "target": "kl-divergence" }
  ]
}
```

**Important cross-track edges to include** (these show why a graph beats a flat list):
- `spectral-theorem` → `graph-laplacians` (Linear Algebra feeds Graph Theory)
- `measure-theoretic-probability` → `bayesian-nonparametrics` (Probability self-loop)
- `riemannian-geometry` → `information-geometry` (DiffGeo → Info Theory bridge)
- `kl-divergence` → `information-geometry` (Info Theory → DiffGeo bridge)
- `pca-low-rank` → `tensor-decompositions` (Linear Algebra self-loop)

These cross-track edges are what make the graph valuable — they reveal structure the flat track listing can't show.

#### Graph Interaction

- **Hover** on a node: highlight its immediate prerequisites and dependents
- **Click** on a published node: navigate to the topic page
- **Click** on a planned node: no-op (or show a tooltip saying "Coming soon")
- **Zoom/pan**: if the graph is dense, allow basic zoom and pan (D3 zoom behavior)
- **Legend**: show track colors and published/planned distinction

#### Layout

- Use a **force-directed layout** (D3 force simulation) with a left-to-right bias so foundational topics cluster left and advanced topics drift right
- Alternatively, a **layered/hierarchical layout** (Dagre or D3-dag) would emphasize the prerequisite chain more clearly — implementer's judgment call based on what looks better with this specific graph

#### Placement

Add the graph as a new section on `/paths` — either:
- **Above** the existing track listing, as a visual overview ("Explore the full dependency graph, or browse by track below")
- **As a toggle** — "View as: Graph | Tracks" — letting the user switch between the visual graph and the existing list view

The implementer should try both and commit whichever reads better.

#### Responsive Behavior

On mobile (< 768px), the graph may be too dense to be useful. Options:
- Hide the graph entirely and show only the track listing
- Show a simplified version (track-level nodes only, not individual topics)
- Show the graph in a horizontally scrollable container

Pick whichever gives the best mobile experience.

### 2b. Cross-Navigation: Back-links from Topic Pages to /paths

#### Problem

Someone who lands on a topic page (e.g., Simplicial Complexes) from a search engine has no way to discover the /paths curriculum roadmap unless they manually navigate there.

#### Deliverable

Add a **breadcrumb or contextual link** on each topic page that links back to the /paths page, scoped to the relevant track. Two options (pick one):

**Option A — Breadcrumb-style:**
```
Paths / Topology & TDA / Simplicial Complexes
```
Where "Paths" links to `/paths` and "Topology & TDA" links to `/paths#topology-tda` (anchor to the relevant track section).

**Option B — Contextual banner:**
A small banner or callout below the topic title:
```
Part of the Topology & TDA track · View full curriculum →
```
Where "View full curriculum →" links to `/paths#topology-tda`.

#### Implementation

- Each topic's MDX frontmatter already has a `track` field (or equivalent). Use that to generate the backlink.
- Add anchor IDs to each track section on `/paths` (e.g., `id="topology-tda"`, `id="linear-algebra"`, etc.) so deep-linking works.

### 2c. Track Section Anchor IDs

#### Problem

There's no way to link directly to a specific track on the /paths page.

#### Deliverable

Add `id` attributes to each track heading on `/paths`:

- `#topology-tda`
- `#linear-algebra`
- `#probability-statistics`
- `#optimization`
- `#differential-geometry`
- `#information-theory`
- `#graph-theory`
- `#category-theory`

This is a prerequisite for 2b (back-links from topic pages) and also useful for sharing direct links to specific tracks.

### 2d. "Start Here" Indicator

#### Problem

New visitors to /paths see 8 tracks and 34+ topics with no guidance on where to begin.

#### Deliverable

Add a visual callout or badge to the recommended starting point(s). Currently, Simplicial Complexes is the only foundational published topic and the natural entry point. Mark it with a "Start here" badge, pin, or visual indicator that's distinct from the "Published" / "Planned" badges.

If multiple entry points are added in the future (e.g., one per track), this should scale—but for now, just mark the single starting topic.

---

## Implementation Priority

1. **README.md rewrite** — quickest win, zero risk, immediate portfolio improvement
2. **Track section anchor IDs (2c)** — trivial, enables everything else
3. **Cross-navigation back-links (2b)** — small change, big UX improvement
4. **"Start Here" indicator (2d)** — small visual addition
5. **Interactive prerequisite graph (2a)** — biggest lift, most impressive result

Items 1–4 can be shipped in a single commit. Item 5 (the graph) is a separate feature branch.

---

## Design Constraints

- Match the existing site's visual language: clean, minimal, dark-mode-first, mathematical
- The /paths page currently uses a card-based layout per track with badge indicators — enhancements should extend this design, not replace it
- All new components should be Astro-native or use Astro's `client:load` / `client:visible` directive for any client-side JS (like the D3 graph)
- Keep bundle size minimal — D3 can be imported modularly (`d3-force`, `d3-selection`, `d3-zoom`) rather than the full library
- Typography and color should inherit from the existing Tailwind config

---

## Files Likely Touched

| File | Change |
|------|--------|
| `README.md` | Full replacement |
| `src/pages/paths.astro` (or equivalent) | Add anchor IDs, graph section, "Start here" badge |
| `src/data/curriculum-graph.json` (new) | Graph node/edge data |
| `src/components/PrerequisiteGraph.astro` (new) | D3-powered graph component |
| Topic layout template | Add breadcrumb/back-link to /paths |
| `package.json` | Add D3 dependencies if not already present |
