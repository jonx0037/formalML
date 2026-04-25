import * as d3 from 'd3';

// Homological dimension colors: H0 = teal, H1 = purple, H2 = amber
export const dimensionColors = ['#0F6E56', '#534AB7', '#D97706'];

export const dimensionColorScale = d3
  .scaleOrdinal<number, string>()
  .domain([0, 1, 2])
  .range(dimensionColors);

// Sheaf consistency: 0 = consistent (green), 1 = inconsistent (red)
export const consistencyColorScale = d3
  .scaleLinear<string>()
  .domain([0, 1])
  .range(['#22c55e', '#ef4444']);

// Tableau10 (8 of 10 used) for Foundations + Set2-derived hues for ML Methodology
// + neutral grays for the two sister sites. Explicit range required because
// the scale now exceeds 10 entries and ordinal scales otherwise cycle.
export const domainColorScale = d3
  .scaleOrdinal<string, string>()
  .domain([
    'topology',
    'geometry',
    'probability',
    'optimization',
    'linear-algebra',
    'information-theory',
    'graph-theory',
    'category-theory',
    'supervised-learning',
    'unsupervised',
    'nonparametric-ml',
    'bayesian-ml',
    'learning-theory',
    'formalcalculus',
    'formalstatistics',
  ])
  .range([
    ...d3.schemeTableau10.slice(0, 8),
    '#66c2a5',
    '#fc8d62',
    '#8da0cb',
    '#e78ac3',
    '#a6d854',
    '#9ca3af',
    '#6b7280',
  ]);
