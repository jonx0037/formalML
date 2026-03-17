// Pre-computed data for Statistical TDA interactive components.
// All randomness is deterministic via integer hash functions.

// ─── Deterministic noise function ───
function noise(i: number): number {
  return ((i * 7919 + 104729) % 100000) / 100000 * 0.1 - 0.05;
}

// ─── 1. Circle points: 100 points on unit circle with light noise ───
export interface CirclePoint {
  x: number;
  y: number;
}

export const circlePoints: CirclePoint[] = Array.from({ length: 100 }, (_, i) => {
  const theta = (2 * Math.PI * i) / 100;
  return {
    x: Math.cos(theta) + noise(i),
    y: Math.sin(theta) + noise(i + 100),
  };
});

// ─── 2. Stability data ───
// Perturbation levels and pre-computed bottleneck distances.
// For a noisy circle, d_B grows roughly linearly with sigma.
export const stabilitySigmas = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5];

export const stabilityBottleneckDistances = [
  0.0, 0.032, 0.068, 0.105, 0.145, 0.183, 0.218, 0.252, 0.290, 0.322, 0.350,
];

// Per-point deterministic perturbation offsets (used by StabilityExplorer)
export const perturbationOffsetsX: number[] = Array.from({ length: 100 }, (_, i) =>
  ((i * 7919 + 200 * 7919 + 104729) % 100000) / 100000 * 2 - 1,
);
export const perturbationOffsetsY: number[] = Array.from({ length: 100 }, (_, i) =>
  ((i * 7919 + 300 * 7919 + 104729) % 100000) / 100000 * 2 - 1,
);

// ─── 3. Bootstrap data ───
// H1 persistence diagram for the noisy circle.
export interface DiagramPoint {
  birth: number;
  death: number;
}

export const diagramPoints: DiagramPoint[] = [
  // The significant loop
  { birth: 0.15, death: 0.85 },
  // Noise points — short-lived features
  { birth: 0.08, death: 0.14 },
  { birth: 0.12, death: 0.18 },
  { birth: 0.22, death: 0.27 },
  { birth: 0.30, death: 0.35 },
  { birth: 0.38, death: 0.41 },
  { birth: 0.45, death: 0.52 },
  { birth: 0.55, death: 0.60 },
  { birth: 0.62, death: 0.68 },
];

// Confidence thresholds c_alpha for alpha = 0.01..0.20
// These are pre-computed so that the significant loop (persistence=0.70) is always
// significant, while noise points (persistence ≤ 0.07) flip between significant and
// noise as alpha increases.
export const confidenceThresholds: Record<string, number> = {
  '0.01': 0.025,
  '0.02': 0.030,
  '0.03': 0.035,
  '0.04': 0.040,
  '0.05': 0.060,
  '0.06': 0.065,
  '0.07': 0.070,
  '0.08': 0.072,
  '0.09': 0.075,
  '0.10': 0.078,
  '0.11': 0.080,
  '0.12': 0.082,
  '0.13': 0.085,
  '0.14': 0.088,
  '0.15': 0.090,
  '0.16': 0.092,
  '0.17': 0.095,
  '0.18': 0.098,
  '0.19': 0.100,
  '0.20': 0.105,
};

// ─── 4. Landscape data ───
// Persistence landscape for H1 of the noisy circle.
// t ranges from 0 to 1.0 with 200 evaluation points.

const NUM_LANDSCAPE_PTS = 200;

export const landscapeT: number[] = Array.from(
  { length: NUM_LANDSCAPE_PTS },
  (_, i) => (i / (NUM_LANDSCAPE_PTS - 1)) * 1.0,
);

// Compute landscape function for a single persistence interval at parameter t.
// Lambda_k(t) for interval (b,d) is max(0, min(t-b, d-t)).
function singleLandscape(t: number, birth: number, death: number): number {
  return Math.max(0, Math.min(t - birth, death - t));
}

// For each t, compute all interval contributions, sort descending, and pick layers.
function computeLandscapeLayers(
  points: DiagramPoint[],
  tValues: number[],
  numLayers: number,
): number[][] {
  const layers: number[][] = Array.from({ length: numLayers }, () =>
    new Array(tValues.length).fill(0),
  );

  for (let ti = 0; ti < tValues.length; ti++) {
    const t = tValues[ti];
    const values = points
      .map((p) => singleLandscape(t, p.birth, p.death))
      .sort((a, b) => b - a);

    for (let k = 0; k < numLayers; k++) {
      layers[k][ti] = k < values.length ? values[k] : 0;
    }
  }

  return layers;
}

export const landscapeLayers: number[][] = computeLandscapeLayers(
  diagramPoints,
  landscapeT,
  5,
);
