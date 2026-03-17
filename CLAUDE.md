# CLAUDE.md — formalML

## Project Overview

formalML is a static site of long-form mathematical explainers for ML practitioners, grad students, and researchers. Every topic gets three pillars: rigorous math, interactive visualization, and working code.

Live site: https://formalml.com

## Tech Stack

- **Framework:** Astro 6 (static site generation)
- **UI:** React 19 (interactive components only — Astro handles static markup)
- **Content:** MDX with remark-math + rehype-katex for LaTeX rendering
- **Styling:** Tailwind CSS 4
- **Visualizations:** D3.js 7 (via React components in `src/components/viz/`)
- **Search:** Pagefind (runs post-build)
- **Package manager:** pnpm (not npm — no package-lock.json)
- **Deploy:** Vercel

## Commands

```bash
pnpm dev        # Dev server at localhost:4321
pnpm build      # Production build (runs pagefind post-build)
pnpm preview    # Preview production build
```

## Project Structure

```
src/
├── pages/              # Astro routes (topics use [...slug].astro)
├── content/topics/     # MDX topic files (the content)
├── components/
│   ├── ui/             # Astro structural components (Nav, TopicCard, TheoremBlock, etc.)
│   └── viz/            # React + D3 interactive visualizations
│       └── shared/     # Shared hooks, types, color scales
├── data/               # Curriculum graph, sample datasets
├── layouts/            # Page layout templates
├── lib/                # Utility modules
└── styles/             # Global CSS, design tokens

docs/plans/             # Planning & handoff documents
notebooks/              # Research notebooks (Jupyter, not tracked in git)
public/images/          # Static images organized by topic
```

## Content Conventions

### Mathematical exposition style

- **Geometric-first:** Introduce concepts visually and concretely before algebraic machinery
- **Foundational topics:** Zero algebra — stop at geometric intuition
- **Intermediate topics:** Algebra only after geometric setup is established
- **Proofs:** Expand fully with combinatorial detail — never "it can be shown"
- **Examples:** Concrete, motivating examples before every definition

### MDX topic file structure

Each topic in `src/content/topics/` is an MDX file with YAML frontmatter defining:
- title, description, domain, difficulty, prerequisites, references
- Interactive viz components are imported and embedded inline

### Visualization components

- All viz components live in `src/components/viz/`
- Use D3.js via the `useD3` hook in `viz/shared/useD3.ts`
- Use `useResizeObserver` for responsive sizing
- Shared color scales in `viz/shared/colorScales.ts`
- Shared types in `viz/shared/types.ts`
- Use `.style()` for CSS custom properties in D3 SVG elements (not `.attr("style", ...)`)

### Curriculum graph

- Topic metadata and prerequisite DAG defined in `src/data/curriculum-graph.json`
- Track definitions in `src/data/curriculum.ts`
- When adding a new topic, update both files and add cross-links in related topics

## Code Style

- TypeScript throughout (Astro + React)
- Functional React components with hooks
- No class components
- Prefer named exports
- D3 selections scoped to component refs — no global DOM manipulation

## Do NOT

- Use npm or generate package-lock.json
- Commit .vscode/, .DS_Store, or firebase-debug.log
- Create draft files outside src/content/topics/ — drafts live as unpublished MDX
- Add algebra to foundational-level topics
- Write one-line proof sketches — expand or omit

## Editorial Voice

<!-- TODO: Jonathan — fill in 3-5 bullet points describing the writing voice
     you want for formalML articles. Examples of what to capture:
     - Tone (conversational? academic? somewhere between?)
     - Use of "we" vs "you" vs passive voice
     - How to handle assumed reader knowledge
     - Attitude toward jargon and notation
-->
