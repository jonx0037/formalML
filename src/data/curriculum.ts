export const tracks = [
  {
    domain: 'topology',
    label: 'Topology & TDA',
    description:
      'Topological Data Analysis \u2014 from simplices to persistence diagrams. The geometric heart of the site.',
    planned: [],
    calculusDependent: false,
  },
  {
    domain: 'linear-algebra',
    label: 'Linear Algebra',
    description:
      'Spectral theory, matrix decompositions, and the algebraic backbone of nearly every ML method.',
    planned: [],
    calculusDependent: false,
  },
  {
    domain: 'probability',
    label: 'Probability & Statistics',
    description:
      'Measure-theoretic foundations through PAC learning and concentration inequalities.',
    planned: [],
    calculusDependent: true,
  },
  {
    domain: 'optimization',
    label: 'Optimization',
    description:
      'Convex analysis, gradient methods, and the theoretical tools behind modern training algorithms.',
    planned: [],
    calculusDependent: true,
  },
  {
    domain: 'geometry',
    label: 'Differential Geometry',
    description:
      'Smooth manifolds, Riemannian metrics, and information geometry for probabilistic models.',
    planned: [],
    calculusDependent: true,
  },
  {
    domain: 'information-theory',
    label: 'Information Theory',
    description:
      'Entropy, divergences, and the theoretical limits that underpin compression, coding, and learning.',
    planned: [],
    calculusDependent: false,
  },
  {
    domain: 'graph-theory',
    label: 'Graph Theory',
    description:
      'Spectral graph theory and random walks \u2014 the mathematical foundation of graph neural networks.',
    planned: [],
    calculusDependent: false,
  },
  {
    domain: 'category-theory',
    label: 'Category Theory',
    description:
      'Functors, adjunctions, and monads \u2014 the abstract language that unifies disparate ML structures.',
    planned: [],
    calculusDependent: false,
  },
] as const;

export type Domain = (typeof tracks)[number]['domain'];

export const domainLabelMap: Record<string, string> = Object.fromEntries(
  tracks.map((t) => [t.domain, t.label])
);
