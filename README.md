# formalML

**The mathematical machinery behind modern machine learning.**

Deep-dive explainers combining rigorous mathematics, interactive visualizations, and working code. Built for practitioners, graduate students, and researchers.

[www.formalml.com](https://www.formalml.com)

---

## What This Is

formalML is a curated collection of long-form explainers on the mathematical foundations of modern ML. Every topic receives a three-pillar treatment:

1. **Rigorous exposition** — Formal definitions, theorems, and proofs presented with full mathematical detail
2. **Interactive visualization** — Embedded widgets that let you manipulate parameters and watch the math come alive (e.g., drag an epsilon slider to watch a Vietoris-Rips complex form around a point cloud)
3. **Working code** — Production-oriented Python implementations you can run immediately, with bridges to standard libraries like `ripser`, `gudhi`, and `scikit-tda`

The site exists because the gap between textbook formalism and practical ML understanding is wider than it needs to be.

## Published Topics

| Topic | Track | Level | Description |
|-------|-------|-------|-------------|
| [Simplicial Complexes](https://www.formalml.com/topics/simplicial-complexes) | Topology & TDA | Foundational | The combinatorial scaffolding that turns point clouds into topology |
| [Persistent Homology](https://www.formalml.com/topics/persistent-homology) | Topology & TDA | Intermediate | Tracking topological features across scales — the workhorse of TDA |

## Curriculum Roadmap

The full roadmap spans 8 mathematical tracks with 34+ planned topics:

- **Topology & TDA** — Simplicial complexes, persistent homology, Cech complexes, Mapper, barcodes, sheaf theory
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
| Visualizations | React 19 + D3.js (interactive components) |
| Search | [Pagefind](https://pagefind.app) (static search) |
| Package manager | pnpm |
| Hosting | Vercel |

## Project Structure

```
├── src/
│   ├── pages/              # Astro page routes
│   ├── content/
│   │   └── topics/         # MDX topic files
│   ├── components/
│   │   ├── ui/             # Astro UI components (Nav, TopicCard, etc.)
│   │   └── viz/            # React + D3 visualization components
│   │       └── shared/     # Shared types, color scales, hooks
│   ├── layouts/            # Page layout templates
│   ├── data/               # Curriculum graph data, sample datasets
│   ├── lib/                # Utility modules
│   └── styles/             # Global CSS, design tokens
├── public/                 # Static assets
├── drafts/                 # Work-in-progress topic drafts
├── docs/plans/             # Planning documents
├── astro.config.mjs        # Astro configuration
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

**Jonathan Rocha** — Data scientist and researcher. MS Data Science (SMU), MA English (Texas A&M University-Central Texas), BA History (Texas A&M University). Research interests: time-series data mining, topology-aware deep learning.

- GitHub: [@jonx0037](https://github.com/jonx0037)
- Consultancy: [DataSalt LLC](https://datasalt.ai)

## License

All rights reserved.
