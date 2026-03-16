/**
 * Synthetic financial market data for the MarketRegimeMapper visualization.
 *
 * 150 trading days across 3 regimes: bull → transition → bear → transition → bull.
 * The Mapper algorithm clusters these into ~9 nodes connected in a path graph,
 * revealing how market regimes transition smoothly — a key insight for risk management.
 *
 * All values are deterministic via a hash-based noise function.
 */

export type Regime = 'bull' | 'bear' | 'transition';

export interface FinancialDataPoint {
  /** Time index 0..149. */
  t: number;
  /** Simulated daily return (typically -0.05 to 0.05). */
  returns: number;
  /** Rolling volatility (typically 0.005 to 0.04). */
  volatility: number;
  /** Market regime label. */
  regime: Regime;
  /** PCA projection x-coordinate. */
  pcaX: number;
  /** PCA projection y-coordinate. */
  pcaY: number;
}

export interface FinancialMapperNode {
  /** Node identifier. */
  id: number;
  /** Number of data points in this cluster. */
  size: number;
  /** Indices into financialData belonging to this node. */
  members: number[];
  /** Dominant market regime in this cluster. */
  dominantRegime: Regime;
  /** Pre-computed layout x-position. */
  x: number;
  /** Pre-computed layout y-position. */
  y: number;
}

/** Deterministic noise in [-1, 1]. */
function noise(i: number, seed: number): number {
  return ((i * 7919 + seed * 104729) % 100000) / 100000 * 2 - 1;
}

/** Round to 5 decimal places. */
function r5(v: number): number {
  return Math.round(v * 100000) / 100000;
}

function getRegime(t: number): Regime {
  if (t < 40) return 'bull';
  if (t < 55) return 'transition';
  if (t < 100) return 'bear';
  if (t < 120) return 'transition';
  return 'bull';
}

function generateFinancialData(): FinancialDataPoint[] {
  const data: FinancialDataPoint[] = [];

  for (let t = 0; t < 150; t++) {
    const regime = getRegime(t);
    const n1 = noise(t, 1);
    const n2 = noise(t, 2);
    const n3 = noise(t, 3);
    const n4 = noise(t, 4);

    let returns: number;
    let volatility: number;

    switch (regime) {
      case 'bull':
        if (t < 40) {
          // First bull phase
          returns = 0.01 + n1 * 0.01;
          volatility = 0.01 + Math.abs(n2) * 0.003;
        } else {
          // Second bull phase (t >= 120)
          returns = 0.012 + n1 * 0.008;
          volatility = 0.012 + Math.abs(n2) * 0.003;
        }
        break;

      case 'transition':
        if (t < 55) {
          // Bull-to-bear transition: declining returns, rising vol
          const frac = (t - 40) / 15; // 0→1
          returns = 0.01 * (1 - frac) + (-0.015) * frac + n1 * 0.015;
          volatility = 0.01 * (1 - frac) + 0.03 * frac + Math.abs(n2) * 0.005;
        } else {
          // Bear-to-bull transition: recovering returns, declining vol
          const frac = (t - 100) / 20; // 0→1
          returns = -0.015 * (1 - frac) + 0.012 * frac + n1 * 0.015;
          volatility = 0.03 * (1 - frac) + 0.012 * frac + Math.abs(n2) * 0.005;
        }
        break;

      case 'bear':
        returns = -0.015 + n1 * 0.012;
        volatility = 0.03 + Math.abs(n2) * 0.005;
        break;
    }

    // PCA projection: roughly returns × 10 for x, volatility × 20 for y, with noise
    const pcaX = returns! * 10 + n3 * 0.02;
    const pcaY = volatility! * 20 + n4 * 0.02;

    data.push({
      t,
      returns: r5(returns!),
      volatility: r5(volatility!),
      regime,
      pcaX: r5(pcaX),
      pcaY: r5(pcaY),
    });
  }

  return data;
}

/** 150 synthetic trading days across bull/transition/bear regimes. */
export const financialData: FinancialDataPoint[] = generateFinancialData();

/**
 * 9 Mapper nodes clustering the financial data by regime.
 * Arranged as a path graph: bull → transition → bear → transition → bull.
 */
export const financialMapperNodes: FinancialMapperNode[] = [
  {
    id: 0,
    size: 20,
    members: Array.from({ length: 20 }, (_, i) => i),
    dominantRegime: 'bull',
    x: 0.1,
    y: 0.2,
  },
  {
    id: 1,
    size: 20,
    members: Array.from({ length: 20 }, (_, i) => i + 17),
    dominantRegime: 'bull',
    x: 0.2,
    y: 0.22,
  },
  {
    id: 2,
    size: 18,
    members: Array.from({ length: 18 }, (_, i) => i + 35),
    dominantRegime: 'transition',
    x: 0.35,
    y: 0.4,
  },
  {
    id: 3,
    size: 20,
    members: Array.from({ length: 20 }, (_, i) => i + 50),
    dominantRegime: 'bear',
    x: 0.5,
    y: 0.75,
  },
  {
    id: 4,
    size: 20,
    members: Array.from({ length: 20 }, (_, i) => i + 67),
    dominantRegime: 'bear',
    x: 0.6,
    y: 0.8,
  },
  {
    id: 5,
    size: 18,
    members: Array.from({ length: 18 }, (_, i) => i + 84),
    dominantRegime: 'bear',
    x: 0.7,
    y: 0.72,
  },
  {
    id: 6,
    size: 18,
    members: Array.from({ length: 18 }, (_, i) => i + 98),
    dominantRegime: 'transition',
    x: 0.78,
    y: 0.5,
  },
  {
    id: 7,
    size: 18,
    members: Array.from({ length: 18 }, (_, i) => i + 113),
    dominantRegime: 'transition',
    x: 0.85,
    y: 0.35,
  },
  {
    id: 8,
    size: 20,
    members: Array.from({ length: 20 }, (_, i) => i + 128),
    dominantRegime: 'bull',
    x: 0.92,
    y: 0.2,
  },
];

/**
 * Edges connecting adjacent Mapper nodes in the regime path graph.
 * The path structure reveals smooth transitions between market regimes.
 */
export const financialMapperEdges: [number, number][] = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [4, 5],
  [5, 6],
  [6, 7],
  [7, 8],
];
