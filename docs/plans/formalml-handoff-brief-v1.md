# formalml.com — Claude Code Handoff Brief v1

**Project:** Advanced Foundations in Machine Learning Mathematics
**Author:** Jonathan Rocha (@jonx0037)
**Date:** 2026-03-15
**Status:** Scaffolding + Day-One Content

---

## 1. Project Overview

formalml.com is a curated collection of deep-dive explainers on the mathematical machinery behind modern ML/DS. Each topic receives the same three-pillar treatment: **rigorous math**, **visual intuition** (interactive D3/Three.js diagrams), and **working code** (Python notebooks + runnable snippets).

The site serves three audiences:
- **ML practitioners** who want to understand the math beneath the tools they use
- **Graduate students** bridging coursework and research-level material
- **Researchers** who want accessible visual intuition for adjacent mathematical domains

**Brand positioning:** Standalone academic resource. Subtle affiliation with DataSalt LLC (datasalt.ai) via footer credit only. Tone is rigorous but not exclusionary — think graduate textbook meets 3Blue1Brown, not Wikipedia.

---

## 2. Tech Stack (Locked Decisions)

| Layer | Choice | Rationale |
|---|---|---|
| **Framework** | Astro 5.x (App Router) | Content-first, zero-JS default, island hydration for interactive components |
| **Interactive components** | React 18 via `@astrojs/react` | Author's primary framework; hydrates as Astro islands with `client:visible` |
| **Content** | MDX files in repo | Single-author, math-heavy content with embedded React components; no CMS |
| **Math rendering** | KaTeX via `remark-math` + `rehype-katex` | Server-side rendering, fast, covers all needed notation |
| **Code highlighting** | Shiki (Astro built-in) | Server-rendered, theme-aware, supports Python/R/Julia |
| **2D visualization** | D3.js v7 | Persistence diagrams, filtrations, graph layouts, matrix views |
| **3D visualization** | Three.js r170+ | Point clouds, manifold surfaces, embeddings |
| **Styling** | Tailwind CSS 4.x | Utility-first, pairs well with Astro |
| **Search** | Pagefind | Static build-time index, zero cost, sufficient for 10-30 topics |
| **Prerequisite graph** | D3 force-directed DAG | Auto-generated from frontmatter `prerequisites` field |
| **Deployment** | Vercel | Zero-config Astro support, already in author's workflow |
| **Package manager** | pnpm | Fast, strict, workspace-friendly |

---

## 3. Repo Structure

```
formalml/
├── astro.config.mjs
├── tailwind.config.mjs
├── tsconfig.json
├── package.json
├── public/
│   ├── fonts/                    # Self-hosted if needed
│   ├── og/                       # Generated OG images per topic
│   └── favicon.svg
├── src/
│   ├── layouts/
│   │   ├── BaseLayout.astro      # HTML shell, meta, KaTeX CSS, analytics
│   │   ├── TopicLayout.astro     # Three-pillar topic page template
│   │   └── PageLayout.astro      # Generic static pages (about, etc.)
│   ├── pages/
│   │   ├── index.astro           # Landing page
│   │   ├── about.astro           # Mission + author bio
│   │   ├── topics/
│   │   │   └── index.astro       # Topic browse/grid with filters
│   │   ├── paths/
│   │   │   └── index.astro       # Learning paths index
│   │   └── videos/
│   │       └── index.astro       # YouTube video index (future)
│   ├── content/
│   │   ├── config.ts             # Astro content collection schema
│   │   └── topics/
│   │       ├── persistent-homology.mdx
│   │       ├── simplicial-complexes.mdx
│   │       └── metric-spaces.mdx
│   ├── components/
│   │   ├── ui/                   # Site-level UI components
│   │   │   ├── TopicCard.astro
│   │   │   ├── PrerequisiteChip.astro
│   │   │   ├── DifficultyBadge.astro
│   │   │   ├── CodeBlock.astro   # Enhanced Shiki block + copy + Colab link
│   │   │   ├── TheoremBlock.astro # Styled definition/theorem/proof blocks
│   │   │   ├── VideoEmbed.astro  # Conditional YouTube embed
│   │   │   ├── SearchDialog.tsx  # Pagefind search (React, client:load)
│   │   │   ├── Nav.astro
│   │   │   └── Footer.astro
│   │   └── viz/                  # Interactive visualization library (React)
│   │       ├── PointCloud2D.tsx
│   │       ├── PointCloud3D.tsx
│   │       ├── SimplicialComplex.tsx
│   │       ├── FiltrationSlider.tsx
│   │       ├── PersistenceDiagram.tsx
│   │       ├── PersistenceBarcode.tsx
│   │       ├── MatrixHeatmap.tsx
│   │       ├── DAGGraph.tsx       # Prerequisite graph + general DAG
│   │       ├── ParametricSurface.tsx
│   │       └── shared/
│   │           ├── useResizeObserver.ts
│   │           ├── useD3.ts       # D3 bindings for React refs
│   │           ├── colorScales.ts # Shared D3 color scales
│   │           └── types.ts       # Shared viz type definitions
│   ├── data/
│   │   ├── learningPaths.ts      # Curated topic sequences
│   │   └── sampleData/           # Bundled example datasets for viz
│   │       ├── torus-point-cloud.json
│   │       ├── circle-filtration.json
│   │       └── two-moons.json
│   ├── lib/
│   │   ├── prerequisiteGraph.ts  # Builds DAG from content collection
│   │   ├── topicUtils.ts         # Slug helpers, sorting, filtering
│   │   └── mathMacros.ts         # Shared KaTeX macro definitions
│   └── styles/
│       └── global.css            # Tailwind directives + KaTeX overrides + prose styling
├── scripts/
│   ├── validate-frontmatter.ts   # Build-time schema check
│   └── generate-og-images.ts     # OG image generator per topic
└── notebooks/                    # Companion Jupyter notebooks (mirrored to GitHub)
    ├── persistent-homology.ipynb
    ├── simplicial-complexes.ipynb
    └── metric-spaces.ipynb
```

---

## 4. Content Collection Schema

File: `src/content/config.ts`

```typescript
import { defineCollection, z, reference } from 'astro:content';

const topics = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    slug: z.string(),
    subtitle: z.string().optional(),        // One-line hook
    status: z.enum(['draft', 'review', 'published']),
    difficulty: z.enum(['foundational', 'intermediate', 'advanced']),
    prerequisites: z.array(z.string()).default([]),  // slugs of other topics
    tags: z.array(z.string()),               // e.g. ['topology', 'tda', 'geometry']
    domain: z.enum([
      'topology',
      'geometry',
      'probability',
      'optimization',
      'linear-algebra',
      'information-theory',
      'graph-theory',
      'category-theory'
    ]),
    videoId: z.string().nullable().default(null),     // YouTube video ID
    notebookPath: z.string().nullable().default(null), // relative path to .ipynb
    githubUrl: z.string().url().nullable().default(null),
    datePublished: z.date().optional(),
    dateUpdated: z.date().optional(),
    estimatedReadTime: z.number().optional(), // minutes
    abstract: z.string(),                    // 2-3 sentence summary for cards/SEO
    connections: z.array(z.object({
      topic: z.string(),                     // slug or free text
      relationship: z.string(),              // e.g. "generalizes", "is dual to", "applied in"
    })).default([]),
    references: z.array(z.object({
      type: z.enum(['paper', 'book', 'course', 'blog', 'video']),
      title: z.string(),
      authors: z.string().optional(),
      year: z.number().optional(),
      url: z.string().url().optional(),
      note: z.string().optional(),           // e.g. "Chapter 4 covers this in detail"
    })).default([]),
  }),
});

export const collections = { topics };
```

---

## 5. Page Specifications

### 5.1 Landing Page (`/`)

**Purpose:** Communicate the site's identity in under 5 seconds.

**Structure:**
- Hero section: Site name, one-sentence tagline ("The mathematical machinery behind modern machine learning"), and a brief 2-3 sentence elaboration
- Animated prerequisite graph as visual centerpiece (the DAGGraph component, auto-generated from all topic frontmatter, interactive — click a node to navigate to that topic)
- "Latest topics" — 3 most recently published topic cards
- "Learning paths" — 2-3 featured path cards
- Footer: "A project by Jonathan Rocha" with subtle DataSalt link, GitHub link, newsletter signup

**Design direction:** Clean, typographically driven. Think academic monograph meets modern web. Generous whitespace. No hero images, no gradients, no stock art. The math and the interactive graph ARE the visual interest. Color palette: near-black text on warm white, with a single accent color (muted teal or deep blue) for interactive elements and links.

### 5.2 Topic Index (`/topics`)

**Purpose:** Browse and filter all published topics.

**Structure:**
- Grid of TopicCard components
- Filter bar: by domain (topology, probability, etc.), difficulty, tags
- Sort: alphabetical, newest, difficulty
- Each card shows: title, subtitle, difficulty badge, domain tag, abstract (truncated), prerequisite count

### 5.3 Topic Page (`/topics/[slug]`)

**Purpose:** The core content experience. This is the product.

**Structure (top to bottom):**

1. **Header**: Title, subtitle, difficulty badge, domain tag, estimated read time
2. **Prerequisites bar**: Horizontal list of prerequisite chips (linked to their topic pages). If any prerequisite is `status: draft`, show it grayed out with "Coming soon"
3. **Video embed** (conditional): If `videoId` is present, render YouTube embed. If null, omit entirely — no placeholder
4. **Overview + Motivation**: Prose section explaining why this math matters for ML practitioners. No formulas here — just narrative
5. **Formal Framework** (Pillar 1 — Rigorous Math): Definitions, theorems, proof sketches. Uses `TheoremBlock` component for styled callouts:
   - `Definition` blocks: numbered, with term in bold
   - `Theorem` blocks: numbered, statement + proof sketch (collapsible)
   - `Lemma` / `Proposition` / `Corollary`: same styling with different labels
   - All math rendered via KaTeX in display and inline modes
6. **Visual Intuition** (Pillar 2 — Interactive Diagrams): One or more `client:visible` React visualization components embedded in MDX. Each diagram has:
   - A brief prose caption above it
   - Interactive controls (sliders, toggles, hover states)
   - A "What to notice" callout below highlighting the key insight
7. **Working Code** (Pillar 3 — Implementation): Python code blocks with Shiki highlighting. Each block includes:
   - Copy button
   - "Open in Colab" link (points to companion notebook)
   - Brief commentary above/below explaining what the code demonstrates
8. **Connections + Applications**: Prose section linking this topic to other areas of ML/DS. Uses the `connections` frontmatter field to auto-generate "Related topics" links
9. **References + Further Reading**: Structured list from frontmatter `references` field, grouped by type (papers, books, courses)
10. **Navigation footer**: "Prerequisites" (links back) + "Where to go next" (topics that list this one as a prerequisite — computed at build time)

### 5.4 Learning Paths (`/paths`)

**Purpose:** Curated sequences through the topic graph.

**Structure:**
- Each path has: title, description, an ordered list of topic slugs, and an estimated total time
- Path page renders a linear topic list with completion state (future: localStorage or account)
- Day-one paths:
  - "The geometry of deep learning" — metric spaces → manifolds → Riemannian optimization → information geometry
  - "Topological data analysis" — simplicial complexes → persistent homology → stability theorems → applications in ML

### 5.5 About (`/about`)

**Purpose:** Establish credibility and the site's editorial voice.

**Structure:**
- Author bio: Jonathan Rocha, academic lineage (SMU MSDS, TAMU English MA, TAMU History BA), PhD trajectory at UTRGV, research interests (time-series data mining, topology-aware deep learning)
- Site mission: 2-3 paragraphs on why this exists and who it's for
- DataSalt mention: "formalml.com is an independent educational project by the founder of DataSalt LLC (datasalt.ai)"
- Links: GitHub (@jonx0037), DataSalt, LinkedIn (if desired)

---

## 6. Visualization Component Specifications

All viz components are React, live in `src/components/viz/`, and hydrate via `client:visible` in MDX.

### 6.1 PointCloud2D (`PointCloud2D.tsx`)

**Props:**
```typescript
interface PointCloud2DProps {
  data: { x: number; y: number; label?: string; group?: number }[];
  width?: number;          // default: 100% of container
  height?: number;         // default: 400
  colorBy?: 'group' | 'density' | 'none';
  showAxes?: boolean;      // default: true
  interactive?: boolean;   // default: true (pan/zoom/hover)
  highlightRadius?: number; // for epsilon-ball visualization
  onPointHover?: (point: DataPoint | null) => void;
}
```

**Behavior:** D3 SVG scatter plot. Pan and zoom via `d3-zoom`. Hover shows coordinates. When `highlightRadius` is set, hovering a point draws a translucent circle of that radius — key for demonstrating epsilon-neighborhoods in metric spaces and Vietoris-Rips complex construction.

### 6.2 SimplicialComplex (`SimplicialComplex.tsx`)

**Props:**
```typescript
interface SimplicialComplexProps {
  points: { x: number; y: number; id: string }[];
  epsilon: number;                    // current filtration parameter
  maxEpsilon?: number;                // slider max
  showSlider?: boolean;               // default: true
  highlightSimplices?: number[];      // dimension to highlight (0=vertices, 1=edges, 2=triangles)
  colorScheme?: 'dimension' | 'birth-time';
  onEpsilonChange?: (eps: number) => void;
}
```

**Behavior:** D3 SVG. Computes Vietoris-Rips complex at given epsilon. Renders vertices as circles, edges as lines, 2-simplices as filled triangles (low opacity). Slider controls epsilon — as it increases, edges and faces appear. This is the flagship component: a reader should be able to watch topology emerge from a point cloud.

**Linked usage:** Pairs with `PersistenceDiagram` — as epsilon changes, corresponding points on the persistence diagram highlight.

### 6.3 PersistenceDiagram (`PersistenceDiagram.tsx`)

**Props:**
```typescript
interface PersistenceDiagramProps {
  intervals: { birth: number; death: number; dimension: number }[];
  currentEpsilon?: number;            // draws vertical line at current filtration value
  showDiagonal?: boolean;             // default: true
  highlightDimension?: number | null;
  mode?: 'diagram' | 'barcode';      // toggle between scatter and barcode view
}
```

**Behavior:** D3 SVG. Standard persistence diagram (birth vs death scatter above diagonal) or barcode (horizontal bars). Points colored by homological dimension (H0, H1, H2). When `currentEpsilon` is provided, a vertical sweep line shows which features exist at that filtration value.

### 6.4 DAGGraph (`DAGGraph.tsx`)

**Props:**
```typescript
interface DAGGraphProps {
  nodes: { id: string; label: string; status: string; domain: string }[];
  edges: { source: string; target: string }[];
  highlightNode?: string;             // current topic slug
  onNodeClick?: (id: string) => void;
  layout?: 'force' | 'layered';      // default: layered (top-down DAG)
}
```

**Behavior:** D3 force-directed (or dagre-based layered) graph. Nodes colored by domain. Current topic highlighted. Click navigates to the topic page. Used on the landing page (full graph) and on individual topic pages (subgraph of neighbors).

### 6.5 PointCloud3D (`PointCloud3D.tsx`)

**Props:**
```typescript
interface PointCloud3DProps {
  data: { x: number; y: number; z: number; group?: number }[];
  colorBy?: 'group' | 'height' | 'none';
  showAxes?: boolean;
  orbitControls?: boolean;            // default: true
  pointSize?: number;                 // default: 0.05
}
```

**Behavior:** Three.js scene with orbit controls. Renders point cloud as instanced spheres or buffer geometry points. Used for 3D datasets (torus, sphere, Swiss roll) that motivate dimensionality reduction and TDA.

### 6.6 MatrixHeatmap (`MatrixHeatmap.tsx`)

**Props:**
```typescript
interface MatrixHeatmapProps {
  matrix: number[][];
  rowLabels?: string[];
  colLabels?: string[];
  colorScale?: 'sequential' | 'diverging';
  showValues?: boolean;               // default: false for large matrices
  onCellHover?: (row: number, col: number, value: number) => void;
}
```

**Behavior:** D3 SVG heatmap. Hover shows value. Useful for distance matrices, kernel matrices, adjacency matrices, attention weights.

### 6.7 ParametricSurface (`ParametricSurface.tsx`)

**Props:**
```typescript
interface ParametricSurfaceProps {
  fn: (u: number, v: number) => [number, number, number];
  uRange: [number, number];
  vRange: [number, number];
  resolution?: number;                // grid resolution, default: 64
  colorFn?: (x: number, y: number, z: number) => string;
  wireframe?: boolean;
  orbitControls?: boolean;
}
```

**Behavior:** Three.js parametric geometry surface with orbit controls. Used for loss landscapes, manifold visualizations, and information geometry surfaces.

---

## 7. TheoremBlock Component Spec

File: `src/components/ui/TheoremBlock.astro`

Renders styled callout blocks for mathematical content. Accepts a `type` prop.

**Types and styling:**
- `definition` — Left border accent (teal), "Definition {n}." label in bold
- `theorem` — Left border accent (purple), "Theorem {n}." label in bold, optional collapsible proof
- `lemma` — Same as theorem with "Lemma {n}." label
- `proposition` — Same as theorem with "Proposition {n}." label
- `corollary` — Same as theorem with "Corollary {n}." label
- `proof` — Indented block, starts with "Proof.", ends with QED square (∎)
- `remark` — Light gray background, "Remark." label in italic
- `example` — Light background, "Example {n}." label

**Usage in MDX:**
```mdx
<TheoremBlock type="definition" number={1}>
A **metric space** is a pair $(X, d)$ where $X$ is a set and $d: X \times X \to \mathbb{R}_{\geq 0}$ is a function satisfying...
</TheoremBlock>
```

Numbering is per-page, sequential by type (Definition 1, Definition 2, Theorem 1, etc.).

---

## 8. Design System

### Typography
- **Headings:** System sans-serif stack (Inter or similar), weight 600
- **Body prose:** Serif stack for long-form reading (Literata, Source Serif 4, or Charter). 18px / 1.7 line-height
- **Math:** KaTeX default (Computer Modern derivative)
- **Code:** JetBrains Mono or Fira Code, 14px

### Color Palette
- **Background:** `#FAFAF8` (warm white)
- **Text primary:** `#1A1A1A`
- **Text secondary:** `#6B6B6B`
- **Accent:** `#0F6E56` (muted teal — same family as DataSalt's palette, the subtle nod)
- **Accent secondary:** `#534AB7` (muted purple — for topology/geometry topics)
- **Code background:** `#F5F5F0`
- **Definition blocks:** Teal left border on `#E1F5EE` background
- **Theorem blocks:** Purple left border on `#EEEDFE` background
- **Dark mode:** Yes, mandatory. Invert appropriately. Background `#1A1A18`, text `#E8E8E4`

### Spacing
- Max content width: 720px (prose), 960px (pages with sidebar/viz)
- Section spacing: 3rem between major sections
- Generous vertical rhythm throughout — this is a reading experience

---

## 9. Day-One Scope

### Must ship:
1. **Site shell**: Landing page, about page, topic index, nav, footer, search
2. **Prerequisite DAG**: Interactive graph on landing page (even with only 2-3 nodes)
3. **Topic: Persistent Homology** (complete three-pillar treatment):
   - Overview + motivation (why TDA matters for ML)
   - Formal framework: filtrations, simplicial homology, persistence modules, stability theorem
   - Visual intuition: `SimplicialComplex` with filtration slider + linked `PersistenceDiagram`
   - Working code: Python with `ripser`, `persim`, `scikit-tda`
   - Companion Jupyter notebook
4. **Topic: Simplicial Complexes** (prerequisite entry):
   - Lighter treatment — overview, formal definitions, basic visualization
   - `SimplicialComplex` component reused with static examples
5. **Responsive design**: Must work on mobile (viz components scale down or show fallback)
6. **SEO basics**: Meta tags, OG images, structured data, sitemap

### Nice to have (v1.1):
- Topic: Metric Spaces
- Learning paths page with at least one complete path
- Newsletter signup integration (Buttondown or Resend)
- YouTube video index page
- RSS feed
- "Edit on GitHub" links on topic pages

### Explicitly deferred:
- User accounts / progress tracking
- Comments or discussion
- Full 3D visualizations (PointCloud3D, ParametricSurface) — ship 2D first
- Algolia search
- CMS migration

---

## 10. Deployment Configuration

### Vercel Settings
- Framework preset: Astro
- Build command: `pnpm build`
- Output directory: `dist`
- Node.js version: 20.x
- Domain: `formalml.com` (to be purchased on Namecheap, DNS pointed to Vercel)

### Environment Variables
- None required for initial build (static site, no API keys)
- Future: `YOUTUBE_API_KEY` if we add live video metadata fetching

### Build Validation
- `scripts/validate-frontmatter.ts` runs as part of build
- Fails build if any published topic has missing required fields
- Warns on draft topics with incomplete prerequisites

---

## 11. GitHub Repository Setup

- **Repo name:** `formalml` under `jonx0037`
- **Visibility:** Public (the content IS the product)
- **Branch strategy:** `main` (production) + `draft/*` branches for WIP topics
- **CI:** GitHub Actions — lint, type-check, build, validate frontmatter on PR
- **Notebooks:** Stored in `/notebooks`, mirrored to Colab via GitHub URLs

---

## 12. Claude Code Execution Notes

This brief follows the author's established workflow: draft in Claude → handoff to Claude Code → implementation.

**Priority order for Claude Code:**
1. Initialize Astro project with React, Tailwind, and MDX integrations
2. Set up the content collection schema and validate that it compiles
3. Build `BaseLayout.astro` and `TopicLayout.astro`
4. Build `TheoremBlock.astro` component
5. Build `SimplicialComplex.tsx` and `PersistenceDiagram.tsx` (the flagship viz pair)
6. Build `DAGGraph.tsx` for the prerequisite graph
7. Scaffold landing page, about page, topic index
8. Create `persistent-homology.mdx` with embedded viz components
9. Create `simplicial-complexes.mdx` as the prerequisite entry
10. Pagefind integration
11. OG image generation script
12. Vercel deployment config

**Key constraint:** All interactive viz components must work with `client:visible` hydration. Test that they render correctly when scrolled into view, not just on page load.

---

*Brief version: v1 | Last updated: 2026-03-15 | Author: Jonathan Rocha*
