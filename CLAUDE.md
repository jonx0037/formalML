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

### Images & figures

Use the `<Figure>` component from `src/components/ui/Figure.astro` for any image that deserves a caption or is worth optimizing. Two patterns are supported:

Topic MDX files use YAML frontmatter between `---` lines for content-collection metadata; JS imports go **after** the closing `---`, at the top of the MDX body. `Figure` must be imported explicitly — no global MDX components mapping is configured.

**Optimized path (preferred for new images):** place the file in `src/assets/topics/<topic>/` and import it. Astro generates srcset, WebP, and width/height automatically.

```mdx
---
title: "Adjunctions"
# ... other YAML frontmatter ...
---

import Figure from '../../components/ui/Figure.astro';
import galoisConnections from '../../assets/topics/adjunctions/galois-connections.png';

<Figure
  src={galoisConnections}
  alt="Galois connection between ordered sets, showing f ⊣ g"
  caption="A Galois connection is an adjunction in the 2-category of posets."
/>
```

**Legacy path (for images still in `public/`):** pass the string URL. Falls back to a plain `<img>` styled by global `.prose img` rules. No srcset, but responsive via CSS.

```mdx
---
title: "..."
---

import Figure from '../../components/ui/Figure.astro';

<Figure
  src="/images/topics/adjunctions/galois-connections.png"
  alt="..."
  caption="..."
/>
```

Bare markdown `![alt](path)` images still render correctly (mobile-safe via global CSS), but prefer `<Figure>` when a caption adds value. Migrate legacy images into `src/assets/topics/` opportunistically as topics are revisited.

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

- **Tone:** Informed peer, not lecturer. Think "a sharp colleague explaining something at a whiteboard" — conversational enough to use contractions and the occasional aside, but precise enough that no claim is hand-wavy. The prose should read well *as prose*, not just as a vehicle for equations.
- **Pronouns:** Default to "we" as the collaborative mathematical "we" (we define, we observe, we can now see that…). Use "you" sparingly and only for direct reader instructions — "you can verify this by…" or "try dragging the slider to see…". Avoid passive voice for derivations; if someone is doing the math, say who.
- **Assumed reader knowledge:** The reader has taken linear algebra and multivariable calculus and has seen some probability. They may or may not have a CS degree. Don't explain what a matrix is; do explain why a specific decomposition matters here. When a topic's difficulty level is foundational, assume even less — just vectors and basic operations.
- **Jargon and notation:** Introduce notation explicitly on first use in every topic — even standard stuff like $\|\mathbf{x}\|$. Never let a symbol appear without a plain-English gloss nearby. Jargon is fine once defined, but prefer the concrete name over the abstract one when both exist (say "the gap between the closest points" before saying "the margin").
- **Attitude toward the reader:** Respect without flattery. Don't say "simply," "obviously," or "it's easy to see." If something is genuinely straightforward, the exposition will make that self-evident. If something is hard, say so — "this step is where the real work happens" is more useful than pretending it's trivial.
