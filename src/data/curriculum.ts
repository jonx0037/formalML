// Each track declares which curriculum *layer* it belongs to. The /paths page
// reads this to inject layer-section headers above the first track of each
// layer \u2014 so adding/removing/reordering tracks here is the single source of
// truth, and the presentation layer derives layer transitions from data.
export const tracks = [
  {
    domain: 'topology',
    label: 'Topology & TDA',
    layer: 'Foundations',
    description:
      'Topological Data Analysis \u2014 from simplices to persistence diagrams. The geometric heart of the site.',
    planned: [],
    calculusDependent: false,
  },
  {
    domain: 'linear-algebra',
    label: 'Linear Algebra',
    layer: 'Foundations',
    description:
      'Spectral theory, matrix decompositions, and the algebraic backbone of nearly every ML method.',
    planned: [],
    calculusDependent: false,
  },
  {
    domain: 'probability',
    label: 'Probability & Statistics',
    layer: 'Foundations',
    description:
      'Measure-theoretic foundations through PAC learning and concentration inequalities.',
    planned: [],
    calculusDependent: true,
  },
  {
    domain: 'optimization',
    label: 'Optimization',
    layer: 'Foundations',
    description:
      'Convex analysis, gradient methods, and the theoretical tools behind modern training algorithms.',
    planned: [],
    calculusDependent: true,
  },
  {
    domain: 'geometry',
    label: 'Differential Geometry',
    layer: 'Foundations',
    description:
      'Smooth manifolds, Riemannian metrics, and information geometry for probabilistic models.',
    planned: [],
    calculusDependent: true,
  },
  {
    domain: 'information-theory',
    label: 'Information Theory',
    layer: 'Foundations',
    description:
      'Entropy, divergences, and the theoretical limits that underpin compression, coding, and learning.',
    planned: [],
    calculusDependent: false,
  },
  {
    domain: 'graph-theory',
    label: 'Graph Theory',
    layer: 'Foundations',
    description:
      'Spectral graph theory and random walks \u2014 the mathematical foundation of graph neural networks.',
    planned: [],
    calculusDependent: false,
  },
  {
    domain: 'category-theory',
    label: 'Category Theory',
    layer: 'Foundations',
    description:
      'Functors, adjunctions, and monads \u2014 the abstract language that unifies disparate ML structures.',
    planned: [],
    calculusDependent: false,
  },
  {
    domain: 'supervised-learning',
    label: 'Supervised Learning',
    layer: 'ML Methodology',
    description:
      'Nonparametric and high-dimensional supervised learning \u2014 kernel and local-polynomial regression, lasso and debiased lasso, p \u226b n asymptotics.',
    planned: ['kernel-regression', 'local-regression', 'high-dimensional-regression'],
    calculusDependent: false,
  },
  {
    domain: 'unsupervised',
    label: 'Unsupervised & Generative',
    layer: 'ML Methodology',
    description:
      'Density estimation, dimensionality reduction, clustering, and generative modeling from a statistical perspective \u2014 mean-shift, density-ratio estimation, normalizing flows.',
    planned: ['clustering', 'density-ratio-estimation', 'normalizing-flows'],
    calculusDependent: false,
  },
  {
    domain: 'nonparametric-ml',
    label: 'Nonparametric & Distribution-Free',
    layer: 'ML Methodology',
    description:
      'Distribution-free prediction sets, rank-based testing, quantile regression, and statistical depth \u2014 methods that work under minimal distributional assumptions.',
    planned: [
      'conformal-prediction',
      'quantile-regression',
      'rank-tests',
      'prediction-intervals',
      'extreme-value-theory',
      'statistical-depth',
    ],
    calculusDependent: false,
  },
  {
    domain: 'bayesian-ml',
    label: 'Bayesian & Probabilistic ML',
    layer: 'ML Methodology',
    description:
      'The Bayesian toolkit at modern ML scale \u2014 variational methods, neural-network posteriors, Gaussian processes, probabilistic programming, specialized MCMC.',
    planned: [
      'variational-inference',
      'gaussian-processes',
      'probabilistic-programming',
      'mixed-effects',
      'stacking-and-predictive-ensembles',
      'bayesian-neural-networks',
      'variational-bayes-for-model-selection',
      'sparse-bayesian-priors',
      'meta-learning',
      'stochastic-gradient-mcmc',
      'sequential-monte-carlo',
      'reversible-jump-mcmc',
      'riemann-manifold-hmc',
    ],
    calculusDependent: false,
  },
  {
    domain: 'learning-theory',
    label: 'Learning Theory & Methodology',
    layer: 'ML Methodology',
    description:
      'Generalization theory, semiparametric efficiency, causal inference, and uncertainty quantification \u2014 methodology that cuts across model families.',
    planned: [
      'generalization-bounds',
      'vc-dimension',
      'uncertainty-quantification',
      'pac-bayes-bounds',
      'semiparametric-inference',
      'causal-inference-methods',
      'double-descent',
    ],
    calculusDependent: false,
  },
] as const;

export type Layer = (typeof tracks)[number]['layer'];

export type Domain = (typeof tracks)[number]['domain'];

export const domainLabelMap: Record<string, string> = Object.fromEntries(
  tracks.map((t) => [t.domain, t.label])
);
