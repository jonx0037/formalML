export const tracks = [
  {
    domain: 'topology',
    label: 'Topology & TDA',
    description:
      'Topological Data Analysis \u2014 from simplices to persistence diagrams. The geometric heart of the site.',
    planned: [
      '\u010cech Complexes & Nerve Theorem',
      'Mapper Algorithm',
      'Barcodes & Bottleneck Distance',
      'Sheaf Theory',
    ],
  },
  {
    domain: 'linear-algebra',
    label: 'Linear Algebra',
    description:
      'Spectral theory, matrix decompositions, and the algebraic backbone of nearly every ML method.',
    planned: [
      'Spectral Theorem',
      'Singular Value Decomposition',
      'PCA & Low-Rank Approximation',
      'Tensor Decompositions',
    ],
  },
  {
    domain: 'probability',
    label: 'Probability & Statistics',
    description:
      'Measure-theoretic foundations through PAC learning and concentration inequalities.',
    planned: [
      'Measure-Theoretic Probability',
      'Concentration Inequalities',
      'PAC Learning Framework',
      'Bayesian Nonparametrics',
    ],
  },
  {
    domain: 'optimization',
    label: 'Optimization',
    description:
      'Convex analysis, gradient methods, and the theoretical tools behind modern training algorithms.',
    planned: [
      'Convex Analysis',
      'Gradient Descent & Convergence',
      'Proximal Methods',
      'Lagrangian Duality & KKT',
    ],
  },
  {
    domain: 'geometry',
    label: 'Differential Geometry',
    description:
      'Smooth manifolds, Riemannian metrics, and information geometry for probabilistic models.',
    planned: [
      'Smooth Manifolds',
      'Riemannian Geometry',
      'Geodesics & Curvature',
      'Information Geometry & Fisher Metric',
    ],
  },
  {
    domain: 'information-theory',
    label: 'Information Theory',
    description:
      'Entropy, divergences, and the theoretical limits that underpin compression, coding, and learning.',
    planned: [
      'Shannon Entropy & Mutual Information',
      'KL Divergence & f-Divergences',
      'Rate-Distortion Theory',
      'Minimum Description Length',
    ],
  },
  {
    domain: 'graph-theory',
    label: 'Graph Theory',
    description:
      'Spectral graph theory and random walks \u2014 the mathematical foundation of graph neural networks.',
    planned: [
      'Graph Laplacians & Spectrum',
      'Random Walks & Mixing',
      'Expander Graphs',
      'Message Passing & GNNs',
    ],
  },
  {
    domain: 'category-theory',
    label: 'Category Theory',
    description:
      'Functors, adjunctions, and monads \u2014 the abstract language that unifies disparate ML structures.',
    planned: [
      'Categories & Functors',
      'Natural Transformations',
      'Adjunctions',
      'Monads & Comonads',
    ],
  },
] as const;

export type Domain = (typeof tracks)[number]['domain'];

export const domainLabelMap: Record<string, string> = Object.fromEntries(
  tracks.map((t) => [t.domain, t.label])
);
