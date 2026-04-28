# CLAUDE.md — formalML

## Project Overview

formalML is a static site of long-form mathematical explainers for ML practitioners, grad students, and researchers. Every topic gets three pillars: rigorous math, interactive visualization, and working code.

Live site: https://formalml.com
Predecessor sites: https://formalcalculus.com, https://formalstatistics.com

## Curriculum Architecture

formalML has two layers:

**Foundations layer (8 tracks, 35 topics — feature-complete):**
- Topology & TDA, Linear Algebra, Probability & Statistics, Optimization,
  Differential Geometry, Information Theory, Graph Theory, Category Theory

**ML Methodology layer (5 tracks, 32 topics planned):**
- T2 Supervised Learning (3 topics)
- T3 Unsupervised & Generative (3 topics)
- T4 Nonparametric & Distribution-Free (6 topics)
- T5 Bayesian & Probabilistic ML (13 topics)
- T6 Learning Theory & Methodology (7 topics)

The 32 ML Methodology topics discharge forward-pointers from formalstatistics.com Topics 22–32. See [docs/plans/formalml-consolidated-strategic-planning-document.md](docs/plans/formalml-consolidated-strategic-planning-document.md) for the full inventory, track rationale, and first-wave sequencing recommendation.

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
pnpm dev                                            # Dev server at localhost:4321
pnpm build                                          # Production build (runs pagefind). Needs NODE_OPTIONS=--max-old-space-size=8192 (default 4GB OOMs).
pnpm preview                                        # Preview production build
pnpm verify:nonparametric-ml                        # Numerical regression tests for src/components/viz/shared/nonparametric-ml.ts vs notebook printed outputs
pnpm audit:cross-site                               # Cross-site reciprocity validator. AUTO-REWRITES docs/plans/deferred-reciprocals.md — don't hand-edit that file.
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
- **Proofs:** Expand fully with combinatorial detail — never "it can be shown."
- **Examples:** Concrete, motivating examples before every definition
- **Language:** Always use American English spelling and conventions (e.g., "organize," "color," "analyze," "defense," "catalog") — never British English variants.

### MDX topic file structure

Each topic in `src/content/topics/` is an MDX file with YAML frontmatter defining:
- title, description, domain, difficulty, prerequisites, references
- `formalcalculusPrereqs` — array of objects (each with `topic`, `site`, `relationship`) declaring formalcalculus.com prerequisites; full schema in the "Cross-site references" subsection below
- `formalstatisticsPrereqs` — array of objects (same shape) declaring formalstatistics.com prerequisites
- Interactive viz components are imported and embedded inline

**MDX top-level rule:** between the closing frontmatter `---` and the first heading, only `import`/`export` statements are allowed. JSX comments (`{/* … */}`) at top level cause `MDXError: Unexpected BlockStatement`; put comments inside the body.

**Proofs:** use `<TheoremBlock type="proof">…$\square$</TheoremBlock>` with no `number` and no `title`. There is no `<ProofExpand>` component — the standalone `proof` block is the codebase pattern (rank-tests, conformal-prediction, quantile-regression).

### Cross-site references

formalML is the third site in the triad: **formalcalculus → formalstatistics → formalML**. Cross-site relationships are declared in MDX frontmatter using six fields, all auto-validated by `pnpm audit:cross-site` for reciprocity:

| Field | Direction | Use on formalML topics |
|---|---|---|
| `formalcalculusPrereqs` | backward | Calculus topics this ML topic requires |
| `formalstatisticsPrereqs` | backward | Statistics topics this ML topic requires |
| `formalcalculusConnections` | forward | Calculus topics this ML topic informs (rare) |
| `formalstatisticsConnections` | forward | Statistics topics this ML topic informs (rare) |
| `formalmlPrereqs` / `formalmlConnections` | self | Do **not** use — flagged as `self-site` by the audit |

Each entry is an object with `topic` (slug, no extension), `site` (`formalcalculus` \| `formalstatistics`), and `relationship` (≥40 chars of explanatory prose). Reciprocal entries on the target side are required — when target topic doesn't exist yet, the audit logs it in [docs/plans/deferred-reciprocals.md](docs/plans/deferred-reciprocals.md) for retrieval at ship time.

For inline body references to sister-site topics, port formalstatistics's `<ExternalLink>` component (interface `{ href, site, topic }`) to `src/components/ui/ExternalLink.astro` when the first cross-site-prereq topic ships. For planned-but-not-yet-published *internal* formalML topics, use plain text: `**Variational Inference** *(coming soon)*`.

The cross-site infrastructure is documented in detail in [docs/plans/cross-site-audit-report.md](docs/plans/cross-site-audit-report.md) and the strategic planning doc §5.

### Visualization components

- All viz components live in `src/components/viz/`
- Use D3.js via the `useD3` hook in `viz/shared/useD3.ts`
- Use `useResizeObserver` for responsive sizing
- Shared color scales in `viz/shared/colorScales.ts`
- Shared types in `viz/shared/types.ts`
- Use `.style()` for CSS custom properties in D3 SVG elements (not `.attr("style", ...)`)
- Hydration: `client:visible` defers React mount until the component scrolls into view. Until then `useResizeObserver` returns `width = 0` and `useD3` paints nothing. When testing via `preview_eval`, scroll the viz into view (`el.scrollIntoView({block: 'center'})`) before inspecting children.

### Images & figures

Use the `<Figure>` component from `src/components/ui/Figure.astro` for any image that deserves a caption or is worth optimizing. Two patterns are supported:

Topic MDX files use YAML frontmatter between `---` lines for content-collection metadata; JS imports go **after** the closing `---`, at the top of the MDX body. `Figure` must be imported explicitly — no global MDX components mapping is configured.

The `caption` prop is a plain string — `$x$` math syntax inside it does NOT get KaTeX-processed; it renders as literal text. Match the track-mate precedent (rank-tests, conformal-prediction) rather than working around it. If you need rendered math in a caption, that's a `Figure.astro` architecture change.

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

### References

Every entry in the `references` frontmatter array **must** have a `url` field — a DOI link (`https://doi.org/...`) for journal articles and books with DOIs, a proceedings URL for conference papers (NeurIPS, ICML, etc.), or an arXiv link as a last resort. The layout renders references with URLs as clickable links and those without as plain text; missing URLs are a content gap, not a style choice.

### Content metrics spreadsheet

`docs/formalml-content-metrics.xlsx` tracks per-topic metrics (word count, section count, theorem blocks, reference counts with/without URLs, etc.) across three sheets: **Topic Detail**, **Domain Summary**, and **Gap Analysis**. When a new topic ships, add its row to Topic Detail, update the Domain Summary for its domain, and add any gap-analysis entries (e.g., `<3k words`, `<7 sections`, missing code section). The spreadsheet is the single source of truth for content-quality tracking — do not let it drift from the published topics.

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

### Code-example language policy

Default for all topics: Python + NumPy / SciPy / scikit-learn / pandas / matplotlib.

PyTorch / JAX allowed only for these seven planned topics (per strategic planning doc §8.2):
`normalizing-flows`, `bayesian-neural-networks`, `meta-learning`, `stochastic-gradient-mcmc`, `variational-inference`, `density-ratio-estimation` (neural DRE section only), `probabilistic-programming` (Stan / PyMC / NumPyro is the subject matter).

All other topics stay in the NumPy/SciPy default. Notebook cells must run CPU-only in under 60 seconds on a 2020-era laptop — no GPU requirements.

## Do NOT

- Run `git stash -u` / `--include-untracked`. Untracked files in this repo are user-owned working state (notebooks, briefs, drafts, `.venv/`) — sweeping them into stashes loses them invisibly. To branch off main, `git checkout main && git checkout -b <new-branch>` directly; the working tree follows.
- Touch any untracked file without per-instance authorization — no `rm`/`mv`/overwrite/`git add`. Reading is fine.
- Use npm or generate package-lock.json
- Commit .vscode/, .DS_Store, or firebase-debug.log
- Create draft files outside src/content/topics/ — drafts live as unpublished MDX
- Add algebra to foundational-level topics
- Write one-line proof sketches — expand or omit
- Create formalML topics for engineering duplicates of formalstatistics coverage (`logistic-regression`, `generalized-linear-models` engineering flavors) — redirect per strategic planning doc §5.4
- Create standalone topics for `cross-validation`, `ab-testing`, `weight-decay` — these are named sections within host topics per strategic planning doc §7.3
- Place Gaussian processes in T2 Supervised Learning — they belong in T5 Bayesian ML per strategic planning doc §3.5
- Use `formalmlPrereqs` or `formalmlConnections` on a formalML topic — those are reserved for sister sites pointing inward; the audit flags self-pointing as `self-site`

## Editorial Voice

- **Tone:** Informed peer, not lecturer. Think "a sharp colleague explaining something at a whiteboard" — conversational enough to use contractions and the occasional aside, but precise enough that no claim is hand-wavy. The prose should read well *as prose*, not just as a vehicle for equations.
- **Pronouns:** Default to "we" as the collaborative mathematical "we" (we define, we observe, we can now see that…). Use "you" sparingly and only for direct reader instructions — "you can verify this by…" or "try dragging the slider to see…". Avoid passive voice for derivations; if someone is doing the math, say who.
- **Assumed reader knowledge:** The reader has taken linear algebra and multivariable calculus and has seen some probability. They may or may not have a CS degree. Don't explain what a matrix is; do explain why a specific decomposition matters here. When a topic's difficulty level is foundational, assume even less — just vectors and basic operations.
- **Jargon and notation:** Introduce notation explicitly on first use in every topic — even standard stuff like $\|\mathbf{x}\|$. Never let a symbol appear without a plain-English gloss nearby. Jargon is fine once defined, but prefer the concrete name over the abstract one when both exist (say "the gap between the closest points" before saying "the margin").
- **Attitude toward the reader:** Respect without flattery. Don't say "simply," "obviously," or "it's easy to see." If something is genuinely straightforward, the exposition will make that self-evident. If something is hard, say so — "this step is where the real work happens" is more useful than pretending it's trivial.
- **No web-dev metadata in reader-facing prose.** Frontmatter field names (`formalstatisticsPrereqs`, `connections`, etc.), schema details, repo conventions, file paths, build commands, and similar internal concepts belong in `docs/` plans, code comments, and CLAUDE.md — not in topic body content. The reader is an ML practitioner, not a webdev; internal concepts read as noise. E.g., don't write "relationships are flagged in `formalstatisticsPrereqs`" — just present the cross-site prereq with its substance.
