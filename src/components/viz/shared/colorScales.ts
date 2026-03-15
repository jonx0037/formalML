import * as d3 from 'd3';

// Homological dimension colors: H0 = teal, H1 = purple, H2 = amber
export const dimensionColors = ['#0F6E56', '#534AB7', '#D97706'];

export const dimensionColorScale = d3
  .scaleOrdinal<number, string>()
  .domain([0, 1, 2])
  .range(dimensionColors);

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
  ])
  .range(d3.schemeTableau10);
