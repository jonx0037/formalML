# formalML — Handoff Brief Reference

> **Purpose:** Paste this document into Claude Chat when composing a new topic handoff brief.
> It captures the ground-truth codebase conventions so the brief aligns with what Claude Code
> actually sees during implementation. Last updated after the PCA & Low-Rank Approximation topic
> (2026-03-22, 10 topics published).

---

## 1. Tech Stack (exact versions)

| Layer | Technology | Version |
|-------|-----------|---------|
| Framework | Astro | 6.x |
| UI | React | 19.x |
| Content | MDX (remark-math + rehype-katex) | — |
| Styling | Tailwind CSS | 4.x |
| Visualizations | D3.js | 7.x |
| Search | Pagefind | 1.4.x |
| Package manager | **pnpm** (never npm) | — |
| Deploy | Vercel | — |
| TypeScript | 5.9.x | — |

> ⚠️ The PCA handoff brief said "Astro 5 · React 18" — both were wrong. Always use **Astro 6 · React 19**.

---

## 2. MDX Frontmatter Schema

Every topic file lives at `src/content/topics/{slug}.mdx`. The slug is the filename without extension.

```yaml
---
title: "Topic Title"                        # Required. String.
subtitle: "One-line scope description"      # Required. String.
status: "published"                         # Required. "published" | "draft"
difficulty: "intermediate"                  # Required. "foundational" | "intermediate" | "advanced"
prerequisites:                              # Required. Array of topic slugs (filenames, not labels).
  - "svd"                                   #   ← NOT "singular-value-decomposition"
tags:                                       # Required. Lowercase, hyphen-separated.
  - "linear-algebra"
  - "dimensionality-reduction"
domain: "linear-algebra"                    # Required. One of 8 domain keys (see §6).
videoId: null                               # Optional. YouTube video ID or null.
notebookPath: "notebooks/{folder}/{file}"   # Required. Path from repo root.
githubUrl: "https://github.com/..."         # Optional. Full URL to the MDX on GitHub, or null.
datePublished: 2026-03-22                   # Required. ISO date (YYYY-MM-DD).
estimatedReadTime: 40                       # Required. Integer, in minutes.
abstract: "Multi-sentence summary..."       # Required. One paragraph, plain text (no LaTeX).
connections:                                # Required. Array of related-topic objects.
  - topic: "svd"                            #   ← Uses the slug, not the display title.
    relationship: "Describes the connection in 1-2 sentences."
references:                                 # Required. Array of citation objects.
  - type: "book"                            #   type: "book" | "paper"
    title: "Book Title"
    authors: "Surname"                      #   or "Surname1, Surname2 & Surname3"
    year: 2016
    url: "https://doi.org/..."              #   REQUIRED. DOI, proceedings URL, or arXiv.
    note: "Chapter or section relevance"
  - type: "paper"
    title: "Paper Title"
    authors: "Surname1, Surname2 & Surname3"
    year: 2011
    url: "https://doi.org/..."              #   REQUIRED. DOI, proceedings URL, or arXiv.
    note: "Brief relevance note"
formalcalculusPrereqs:                      # Optional. Cross-site backward edges to formalcalculus.com.
  - topic: "jacobian"                       #   `topic` is the slug on formalcalculus (no extension).
    site: "formalcalculus"
    relationship: "≥40 chars of explanatory prose tying this topic's section/theorem to the calculus prereq."
formalstatisticsPrereqs:                    # Optional. Cross-site backward edges to formalstatistics.com.
  - topic: "kernel-density-estimation"      #   `topic` is the slug on formalstatistics.
    site: "formalstatistics"
    relationship: "≥40 chars of explanatory prose."
---
```

**Cross-site fields** are validated end-to-end by `pnpm audit:cross-site` (`scripts/audit-cross-site-links.mjs`). See §10 for the full schema, the deferred-reciprocals workflow, and the `<ExternalLink>` body-component port.

### Common slug mistakes to avoid

| Brief wrote | Codebase actually uses |
|------------|----------------------|
| `"singular-value-decomposition"` | `"svd"` |
| `"spectral-theorem-for-symmetric-matrices"` | `"spectral-theorem"` |
| `"simplicial-complexes-and-topology"` | `"simplicial-complexes"` |
| `"cech-complexes-and-nerve-theorem"` | `"cech-complexes"` |

**Rule:** The slug is always the MDX filename without `.mdx`. Check the published topics table in §5.

### `notebookPath` format

The brief should specify the path from the repo root. Notebooks live in `notebooks/{topic-folder}/`.

| Brief wrote | Correct |
|------------|---------|
| `"notebooks/03_pca_low_rank.ipynb"` | `"notebooks/pca-low-rank/03_pca_low_rank.ipynb"` |

---

## 3. MDX Body Conventions

### Imports (after frontmatter `---`)

```mdx
import TheoremBlock from '../../components/ui/TheoremBlock.astro';
import MyVizComponent from '../../components/viz/MyVizComponent.tsx';
```

- TheoremBlock is the only UI component typically imported.
- Viz components use relative paths from `src/content/topics/`.

### Viz component embedding

```mdx
<MyVizComponent client:visible />
```

> ⚠️ **Always use `client:visible`** — this is Astro's hydration directive that loads the React
> component only when it scrolls into view. Never use `client:load` (hydrates immediately) for
> viz components — they are heavy D3 renderers that should lazy-load.

### TheoremBlock usage

```mdx
<TheoremBlock type="definition" number={1} title="Principal Components">
The principal components of $X$ are the eigenvectors of...
</TheoremBlock>

<TheoremBlock type="theorem" number={1} title="Eckart–Young–Mirsky Theorem">
For any matrix $A$ of rank $r$...
</TheoremBlock>

<TheoremBlock type="proof">
We proceed by showing...
</TheoremBlock>
```

**Valid `type` values:** `definition`, `theorem`, `lemma`, `proposition`, `corollary`, `proof`, `remark`, `example`

**Numbering:** `number` is per-type within the topic. Definitions are numbered separately from theorems. Proofs and remarks typically omit `number`.

### Internal cross-references

Link to other topics with Markdown links using absolute paths:

```mdx
The [SVD](/topics/svd) established that...
By the [Spectral Theorem](/topics/spectral-theorem), this matrix has...
```

For planned (unwritten) topics, use plain text with an annotation:

```mdx
**Tensor Decompositions** *(coming soon)* generalize PCA to multi-way arrays.
```

### Section structure pattern

Every topic follows this skeleton:

```
## Overview & Motivation          ← What and why. Concrete example before formalism.
## [Core Theory Sections]         ← 3-6 sections of math + viz + proofs.
## [Extensions / Variants]        ← Modern developments, algorithms.
## Computational Notes            ← NumPy/SciPy code, numerical pitfalls.
## Connections & Further Reading  ← Cross-reference table + DAG diagram.
```

---

## 4. Visualization Component Patterns

### File location and naming

- All viz components: `src/components/viz/{ComponentName}.tsx`
- PascalCase filenames matching the component name.
- Data modules: `src/data/{topic-slug}-data.ts` (or `{descriptor}-data.ts`)

### Standard imports

```typescript
import { useState, useMemo, useCallback, useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
// Optional, depending on component needs:
import { useD3 } from './shared/useD3';
import { dimensionColors, domainColorScale } from './shared/colorScales';
import type { Point2D } from './shared/types';
```

### Two rendering approaches

**Approach A — `useD3` hook** (simpler, for single-SVG components):

```typescript
const svgRef = useD3<SVGSVGElement>(
  (svg) => {
    svg.selectAll('*').remove();
    // D3 rendering...
  },
  [data, width, height],
);
return <svg ref={svgRef} width={width} height={height} />;
```

**Approach B — `useEffect` + manual refs** (for multi-panel components):

```typescript
const leftSvgRef = useRef<SVGSVGElement>(null);
const rightSvgRef = useRef<SVGSVGElement>(null);

useEffect(() => {
  const svg = d3.select(leftSvgRef.current);
  if (!leftSvgRef.current || innerW <= 0) return;
  svg.selectAll('*').remove();
  // D3 rendering...
}, [data, innerW, innerH]);
```

Recent components (SVD, PCA) use Approach B for multi-panel layouts. Either is fine.

### CSS custom properties (mandatory for theme support)

```typescript
// ✅ Correct — uses CSS custom properties
svg.append('text').style('fill', 'var(--color-text)');
svg.append('rect').style('fill', 'var(--color-surface)');
svg.append('line').style('stroke', 'var(--color-border)');

// ❌ Wrong — hardcoded colors break dark mode
svg.append('text').style('fill', '#1A1A1A');
```

**Available color tokens:**

| Token | Light | Dark | Use for |
|-------|-------|------|---------|
| `--color-bg` | #FAFAF8 | #0D0D0D | Page background |
| `--color-text` | #1A1A1A | #E5E5E5 | Primary text, axis labels |
| `--color-text-secondary` | #6B6B6B | #A3A3A3 | Secondary text |
| `--color-accent` | #0F6E56 | #34D399 | Accent/highlight color |
| `--color-surface` | #FFFFFF | #1A1A1A | Card/panel backgrounds |
| `--color-border` | #E5E5E0 | #2D2D2D | Borders, grid lines |

**Font tokens:** `--font-sans` (Inter), `--font-serif` (Source Serif 4), `--font-mono` (JetBrains Mono)

### Responsive layout

```typescript
const SM_BREAKPOINT = 640;
const isMobile = (containerWidth || 800) < SM_BREAKPOINT;

// Stack panels vertically on mobile, side-by-side on desktop
<div style={{ flexDirection: isMobile ? 'column' : 'row' }}>
```

### Export style

```typescript
export default function ComponentName() { ... }
```

Default export, named function. No class components.

### Data module conventions

- Export TypeScript interfaces for all data shapes.
- Use deterministic pseudo-random generators (seeded LCG/hash) — never `Math.random()`.
- For expensive computations, use lazy initialization (not eager module-level execution):

```typescript
// ✅ Correct — lazy
let cache: Result[] | null = null;
export function getResults(): Result[] {
  if (cache === null) cache = computeExpensiveData();
  return cache;
}

// ❌ Wrong — runs on import, blocks page load
export const results = computeExpensiveData();
```

---

## 5. Published Topics (current state)

The Foundations layer (8 tracks, 35 topics) is feature-complete. The ML Methodology layer (5 tracks, 32 topics) is planned per the strategic planning document — track sub-tables below are seeded empty and grow as topics ship.

### Foundations layer

| Slug | Title | Domain | Difficulty |
|------|-------|--------|------------|
| `simplicial-complexes` | Simplicial Complexes | topology | foundational |
| `persistent-homology` | Persistent Homology | topology | intermediate |
| `cech-complexes` | Čech Complexes & Nerve Theorem | topology | intermediate |
| `mapper-algorithm` | The Mapper Algorithm | topology | intermediate |
| `barcodes-bottleneck` | Barcodes & Bottleneck Distance | topology | intermediate |
| `statistical-tda` | Statistical TDA | topology | intermediate |
| `sheaf-theory` | Sheaf Theory | topology | advanced |
| `spectral-theorem` | The Spectral Theorem | linear-algebra | foundational |
| `svd` | Singular Value Decomposition | linear-algebra | intermediate |
| `pca-low-rank` | PCA & Low-Rank Approximation | linear-algebra | intermediate |

(Plus 25 more topics across probability, optimization, geometry, information-theory, graph-theory, category-theory — all 35 are live and inspectable in `src/content/topics/`.)

### ML Methodology layer (planned, no rows yet)

#### Supervised Learning (3 topics planned — strategic doc §3.2)

| Slug | Title | Difficulty |
|------|-------|------------|
| `kernel-regression` | _planned_ | intermediate |
| `local-regression` | _planned_ | intermediate |
| `high-dimensional-regression` | _planned_ | advanced |

#### Unsupervised & Generative (3 topics planned — strategic doc §3.3)

| Slug | Title | Difficulty |
|------|-------|------------|
| `clustering` | _planned_ | intermediate |
| `density-ratio-estimation` | _planned_ | advanced |
| `normalizing-flows` | _planned_ | advanced |

#### Nonparametric & Distribution-Free (6 topics planned — strategic doc §3.4)

| Slug | Title | Difficulty |
|------|-------|------------|
| `conformal-prediction` | _planned_ | intermediate |
| `quantile-regression` | _planned_ | intermediate |
| `rank-tests` | _planned_ | intermediate |
| `extreme-value-theory` | _planned_ | advanced |
| `statistical-depth` | _planned_ | advanced |
| `prediction-intervals` | _planned_ | intermediate |

#### Bayesian & Probabilistic ML (13 topics planned — strategic doc §3.5)

| Slug | Title | Difficulty |
|------|-------|------------|
| `variational-inference` | _planned_ | intermediate |
| `gaussian-processes` | _planned_ | intermediate |
| `probabilistic-programming` | _planned_ | intermediate |
| `mixed-effects` | _planned_ | intermediate |
| `stacking-and-predictive-ensembles` | _planned_ | intermediate |
| `bayesian-neural-networks` | _planned_ | advanced |
| `variational-bayes-for-model-selection` | _planned_ | advanced |
| `sparse-bayesian-priors` | _planned_ | advanced |
| `meta-learning` | _planned_ | advanced |
| `stochastic-gradient-mcmc` | _planned_ | advanced |
| `sequential-monte-carlo` | _planned_ | advanced |
| `reversible-jump-mcmc` | _planned_ | advanced |
| `riemann-manifold-hmc` | _planned_ | advanced |

#### Learning Theory & Methodology (7 topics planned — strategic doc §3.7)

| Slug | Title | Difficulty |
|------|-------|------------|
| `generalization-bounds` | _planned_ | intermediate |
| `vc-dimension` | _planned_ | intermediate |
| `uncertainty-quantification` | _planned_ | intermediate |
| `pac-bayes-bounds` | _planned_ | advanced |
| `semiparametric-inference` | _planned_ | advanced |
| `causal-inference-methods` | _planned_ | advanced |
| `double-descent` | _planned_ | advanced |

When writing a new brief, check the appropriate table for the correct slug. As topics ship, replace `_planned_` with the actual title.

---

## 6. Curriculum Graph Updates

When adding a new topic, the brief should specify changes to two files:

### `src/data/curriculum-graph.json`

**Add a node:**
```json
{ "id": "new-topic-slug", "label": "Display Title", "domain": "domain-key", "status": "published", "url": "/topics/new-topic-slug" }
```

**Add edges** (prerequisite → this topic, this topic → downstream):
```json
{ "source": "prerequisite-slug", "target": "new-topic-slug" }
```

### `src/data/curriculum.ts`

Remove the topic title from the `planned` array of its domain track. Do **not** remove other topics.

### Domain keys (exhaustive list)

**Foundations layer (8 keys):** `topology`, `linear-algebra`, `probability`, `optimization`, `geometry`, `information-theory`, `graph-theory`, `category-theory`

**ML Methodology layer (5 new keys, added as tracks open):** `supervised-learning`, `unsupervised`, `nonparametric-ml`, `bayesian-ml`, `learning-theory`

These match the proposed shared-module names from strategic planning doc §6.2 and §7.4.

---

## 7. Image Conventions

- Directory: `public/images/topics/{topic-slug}/`
- Flat structure (no subdirectories within).
- Formats: PNG preferred for notebook exports; SVG for diagrams.
- Naming: Kebab-case, descriptive (e.g., `pca-variance-maximization.png`).
- Referenced in MDX with: `![Alt text](/images/topics/{slug}/filename.png)`

---

## 8. Lessons from Past Briefs (error log)

These are real discrepancies found during implementation. Each one wasted time or caused bugs.

| Brief Section | What the brief said | What the codebase needed | Impact |
|--------------|-------------------|------------------------|--------|
| Stack versions | "Astro 5 · React 18" | Astro 6 · React 19 | Minor — Claude Code used actual `package.json` |
| Prerequisite slug | `"singular-value-decomposition"` | `"svd"` | Broken prerequisite resolution at build time |
| `notebookPath` | `"notebooks/03_pca_low_rank.ipynb"` | `"notebooks/pca-low-rank/03_pca_low_rank.ipynb"` | Wrong path in frontmatter |
| Eager data export | "export const results = generate()" | Lazy `getResults()` function | Page load performance — blocks hydration |
| Forward links | Linked to `/topics/tensor-decompositions` | Topic doesn't exist yet → 404 | Needed plain text + "(coming soon)" |
| Duplicate constants | Defined `GAMMAS` in both data module and component | Export from single source | Maintenance hazard — values could diverge |
| Dead code | Computed `nnzTarget` but never used it | Remove or use | TypeScript `noUnusedLocals` failure |
| Click handler | Empty `.on('click')` then re-bound via `.each()` | Single `.each()` handler | Dead code, confusing to read |
| Disabled UI option | `polynomial` kernel type shown but disabled | Remove until implemented | Type mismatch risk, confusing UX |
| Cross-site frontmatter | Missing `formalcalculusPrereqs` / `formalstatisticsPrereqs` | Required for all ML Methodology topics; auto-validated by `pnpm audit:cross-site` | Missing reciprocals on sister sites — breaks the audit |
| Track placement | Brief proposes `gaussian-processes` in T2 Supervised | Strategic doc §3.5 places it in T5 Bayesian ML | Wrong navigation grouping |
| Standalone slug for cross-cutting concept | Brief proposes `cross-validation` as own topic | Strategic doc §7.3 names-section policy: lives inside `conformal-prediction` / `high-dimensional-regression` | Duplicate content, slug never minted |
| References missing URLs | Brief listed references without `url` fields | Every reference needs a DOI, proceedings URL, or arXiv link | 46 refs across 5 T4 topics shipped without links — batch fix required |

### How to prevent these in future briefs

1. **Always reference this document** for slug names, stack versions, and path formats.
2. **Only link to published or in-PR topics** — never link to planned topics as if they exist.
3. **Specify data module patterns explicitly** — state whether exports should be lazy or eager.
4. **Don't include UI for unimplemented features** — if polynomial kernel isn't ready, don't spec the radio button.
5. **Include the exact `notebookPath`** with the subdirectory, not just the filename.
6. **Every reference must have a `url` field** — DOI for journal articles/books, proceedings URL for conferences, arXiv as last resort. Verify before handing off to Claude Code.
7. **Update `docs/formalml-content-metrics.xlsx`** when shipping — add the topic row, update domain summary, flag gaps.

---

## 9. Editorial Voice (summary for Claude Chat)

These rules come from `CLAUDE.md` and should be reflected in the brief's content outline:

- **Geometric-first:** Visuals and concrete examples before algebraic machinery.
- **Foundational = no algebra.** Only vectors and basic operations.
- **Intermediate = algebra after geometry.** Algebraic derivations follow geometric intuition.
- **Proofs:** Fully expanded with every combinatorial step. Never "it can be shown."
- **Notation:** Introduced explicitly on first use, with plain-English gloss.
- **Tone:** Informed peer at a whiteboard. Contractions OK, hand-waving not OK.
- **Pronouns:** Mathematical "we" by default. "You" only for direct reader instructions.
- **Forbidden phrases:** "simply," "obviously," "it's easy to see," "trivially."

---

## 10. Cross-Site References

formalML is the third site in the triad: **formalcalculus → formalstatistics → formalML**. Every ML Methodology topic should declare its sister-site prerequisites explicitly.

### Frontmatter schema (six fields, auto-audited)

| Field | Direction | Use on formalML topics |
|---|---|---|
| `formalcalculusPrereqs` | backward | Calculus topics this ML topic requires |
| `formalstatisticsPrereqs` | backward | Statistics topics this ML topic requires |
| `formalcalculusConnections` | forward | Calculus topics this ML topic informs (rare) |
| `formalstatisticsConnections` | forward | Statistics topics this ML topic informs (rare) |
| `formalmlPrereqs` / `formalmlConnections` | self | Do **not** use — flagged as `self-site` by the audit |

Per-entry shape:

```yaml
formalstatisticsPrereqs:
  - topic: "kernel-density-estimation"      # slug, no extension
    site: "formalstatistics"                # explicit, even though field implies it
    relationship: "≥40 chars of explanatory prose tying this topic's
                   section/theorem to the prereq. Audit warns if <40 chars."
```

### Audit and reciprocity

`pnpm audit:cross-site` runs `scripts/audit-cross-site-links.mjs` against all three sibling repos (paths configurable via `FORMAL_CALCULUS_PATH` / `FORMAL_STATISTICS_PATH`). It walks every `.mdx`, extracts the six fields, and verifies that every `A.x → B.y` edge has a reciprocal `B.y → A.x` with the opposite direction (`Prereqs ↔ Connections`).

Outputs:

- [docs/plans/cross-site-audit-report.md](cross-site-audit-report.md) — consolidated reciprocity report (slug drift, missing reciprocals, deferred targets)
- [docs/plans/deferred-reciprocals.md](deferred-reciprocals.md) — log of edges pointing at slugs that don't yet exist on the target repo, organized by "When `<repo>/<slug>` ships" sections with the source-side prose stub
- `docs/plans/audit-output/<site>-references.json` — per-site edge dumps (machine-readable)

### Workflow at ship time

When a new ML Methodology topic ships on formalML:

1. Confirm the topic's frontmatter includes `formalcalculusPrereqs` / `formalstatisticsPrereqs` for every cross-site prereq, with `relationship` ≥ 40 chars.
2. Search [deferred-reciprocals.md](deferred-reciprocals.md) for the new topic's slug under "When `formalml/<slug>` ships" headings — those entries are the canonical PR sweep targets on formalstatistics and/or formalcalculus.
3. On the sister-site repo(s), add the reciprocal field (`formalstatisticsConnections` / `formalcalculusConnections`) to each source topic, copying the source-side prose stub from `deferred-reciprocals.md` and rewriting from the new topic's vantage.
4. Re-run `pnpm audit:cross-site` (from any of the three repos with the sibling paths configured) to confirm the deferred entry disappears and reciprocity holds.

### Inline body links to sister-site topics

For specific in-prose mentions of sister-site topics, port the formalstatistics `<ExternalLink>` component (interface `{ href, site, topic }`) to `src/components/ui/ExternalLink.astro`. Build it during the first-wave topic that introduces a cross-site body link — typically the same PR that introduces the first `formalcalculusPrereqs` / `formalstatisticsPrereqs` frontmatter.

For planned-but-not-yet-published *internal* formalML topics, use plain text:

```mdx
**Tensor Decompositions** *(coming soon)* generalize PCA to multi-way arrays.
```

No link, no 404 risk. Subsequent authors convert plain-text references to Markdown links when the referenced topic ships.

---

## 11. Brief Template Skeleton

When composing a new handoff brief, use this structure:

```markdown
# Claude Code Handoff Brief: {Topic Title}

**Project:** formalML — formalml.com
**Repo:** github.com/jonx0037/formalML
**Stack:** Astro 6 · React 19 · MDX · Tailwind CSS 4 · D3.js 7 · KaTeX · Vercel
**Package Manager:** pnpm
**Status:** Ready for implementation
**Reference Notebook:** `notebooks/{folder}/{file}.ipynb`
**Reference Doc:** `docs/plans/formalml-handoff-reference.md`

---

## 1. Objective
- Which track, position in DAG, prerequisites, difficulty level.
- What the topic covers (3-5 bullet points).

## 2. MDX File
- Location: `src/content/topics/{slug}.mdx`
- Complete frontmatter (copy from §2 of the reference doc and fill in).
- Use correct slugs from the published topics table.

## 3. Content Outline
- Section-by-section outline with section titles.
- Which sections get TheoremBlocks (with type, number, title).
- Which sections get interactive visualizations (with component names).
- Which sections get static images from the notebook.

## 4. Visualizations
- For each viz component:
  - Component name and filename.
  - What it visualizes (1-2 sentences).
  - User interactions (sliders, dropdowns, click targets).
  - Data source (inline generation vs. data module).
  - Panel layout (single panel, side-by-side, three-panel).

## 5. Data Modules
- For each data module:
  - Filename and location.
  - Exported interfaces.
  - Exported functions/constants.
  - Whether computation should be lazy or eager.

## 6. Curriculum Graph Updates
- Node to add (with exact slug, label, domain, status, url).
- Edges to add (with exact source → target).
- Changes to `curriculum.ts` planned arrays.

## 7. Cross-References
- Which existing topics should link TO this topic (update their MDX).
- Which existing topics this topic links FROM.
- Any forward references to planned topics (mark as plain text).

## 8. Images
- List of images from the notebook to copy to `public/images/topics/{slug}/`.
- Filenames and what each depicts.

## 9. Content Metrics
- Update `docs/formalml-content-metrics.xlsx` with this topic's row (see §13 of the reference doc).
- Confirm all references have `url` fields (Refs Missing URL = 0).
```

---

## 12. Shared Infrastructure API Reference

### `useD3<T>(renderFn, deps)` — `src/components/viz/shared/useD3.ts`

Returns a `ref` to attach to an SVG element. Calls `renderFn(d3Selection)` whenever `deps` change.

### `useResizeObserver<T>()` — `src/components/viz/shared/useResizeObserver.ts`

Returns `{ ref, width, height }`. Attach `ref` to a container `<div>` to get its pixel dimensions.

### Color scales — `src/components/viz/shared/colorScales.ts`

- `dimensionColors`: `['#0F6E56', '#534AB7', '#D97706']` (H0 teal, H1 purple, H2 amber)
- `dimensionColorScale`: Ordinal scale mapping 0/1/2 → dimension colors
- `consistencyColorScale`: Linear scale 0→green, 1→red
- `domainColorScale`: Ordinal scale mapping domain strings → `d3.schemeTableau10`

### Shared types — `src/components/viz/shared/types.ts`

`Point2D`, `PersistenceInterval`, `Simplex`, `DAGNode`, `DAGEdge`, `MapperParams`, `MapperResult`, etc.

---

## 13. Content Metrics Spreadsheet

`docs/formalml-content-metrics.xlsx` tracks per-topic quality metrics. It has three sheets:

### Topic Detail (18 columns)

| Column | Description |
|--------|-------------|
| Domain | Domain key (e.g., `nonparametric-ml`) |
| Topic | Display title |
| Difficulty | `foundational` / `intermediate` / `advanced` |
| Words | Body word count (excluding imports, HTML, math) |
| Sections | Count of `##` headings |
| Overview? | Has an Overview section (`Yes`/`No`) |
| Formal Framework? | Has a Formal Framework section (`Yes`/`No`) |
| Code Section? | Has a Code/Computational section (`Yes`/`No`) |
| Viz Imported? | Has `client:` directives (`Yes`/`No`) |
| Connections? | Has `connections` frontmatter entries (`Yes`/`No`) |
| Python Blocks | Count of `` ```python `` code blocks |
| TheoremBlocks | Count of `<TheoremBlock` usages |
| Display Eqs | Count of `$$...$$` display equations |
| Inline Math | Count of `$...$` inline math expressions |
| Refs Total | Total number of reference entries |
| Refs w/ URL | References that have a `url` field |
| Refs Missing URL | References without a `url` field (should be 0) |
| Section Names | Comma-separated list of `##` heading titles |

### Domain Summary

Aggregates per domain: topic count, average/min/max words, average sections, count of topics below quality thresholds, total refs missing URL.

### Gap Analysis

Flags topics with: content too short (<3k words), too few sections (<7), missing code section. Each row includes the topic name, domain, issue description, priority, and notes.

### When to update

Update the spreadsheet whenever a new topic ships. The Claude Code implementation template includes this as a required step before the final PR.

### Reference URL requirement

Every entry in the `references` frontmatter array must have a `url` field. Use DOI links (`https://doi.org/...`) for journal articles and books with DOIs, proceedings URLs for conference papers, or arXiv links as a last resort. The "Refs Missing URL" column in the spreadsheet should always be 0 for every topic.
