// Pre-computed datasets and kernel PCA embeddings for interactive visualization.
// Points are generated deterministically; embeddings are synthetic but capture
// the qualitative behavior of RBF kernel PCA at each gamma value.

// ─── Deterministic noise ───

function noise(i: number): number {
  return ((i * 7919 + 104729) % 100000) / 100000 * 0.1 - 0.05;
}

/** Wider-range deterministic pseudo-random in [0, 1). */
function rand(seed: number): number {
  return ((seed * 7919 + 104729) % 100000) / 100000;
}

/** Seeded pseudo-random in [-1, 1). */
function srand(seed: number): number {
  return rand(seed) * 2 - 1;
}

// ─── Types ───

export interface KernelPCAPoint {
  x: number;
  y: number;
  label: number;
}

export interface KernelPCAEmbedding {
  gamma: number;
  points: { x: number; y: number }[];
}

export interface KernelPCADataset {
  name: string;
  points: KernelPCAPoint[];
  embeddings: KernelPCAEmbedding[];
}

// ─── Gamma values for pre-computed embeddings ───

export const GAMMAS = [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100];

// ─── Round helper ───

function r4(v: number): number {
  return Math.round(v * 10000) / 10000;
}

// ─── Dataset 1: Concentric circles ───

function generateCircles(): KernelPCAPoint[] {
  const points: KernelPCAPoint[] = [];
  // Inner ring: 100 points, r = 0.3
  for (let i = 0; i < 100; i++) {
    const theta = (2 * Math.PI * i) / 100;
    points.push({
      x: r4(0.3 * Math.cos(theta) + noise(i)),
      y: r4(0.3 * Math.sin(theta) + noise(i + 200)),
      label: 0,
    });
  }
  // Outer ring: 100 points, r = 1.0
  for (let i = 0; i < 100; i++) {
    const theta = (2 * Math.PI * i) / 100;
    points.push({
      x: r4(1.0 * Math.cos(theta) + noise(i + 400)),
      y: r4(1.0 * Math.sin(theta) + noise(i + 600)),
      label: 1,
    });
  }
  return points;
}

function circleEmbeddings(points: KernelPCAPoint[]): KernelPCAEmbedding[] {
  // At low gamma: classes are mixed (RBF acts like linear PCA — circles overlap on PC1).
  // At high gamma: classes separate cleanly on PC1.
  return GAMMAS.map((gamma) => {
    const separation = Math.min(1, gamma / 5); // 0→mixed, 1→fully separated
    const embPts = points.map((p, i) => {
      const isOuter = p.label === 1;
      const angle = Math.atan2(p.y, p.x);
      // PC1: at high gamma, inner→negative, outer→positive
      const classSign = isOuter ? 1 : -1;
      const pc1 =
        separation * classSign * (0.8 + 0.2 * Math.cos(angle)) +
        (1 - separation) * (p.x * 0.5 + noise(i + 800) * 3);
      // PC2: angular structure
      const pc2 =
        separation * 0.4 * Math.sin(angle * 2) +
        (1 - separation) * (p.y * 0.5 + noise(i + 1000) * 3);
      return { x: r4(pc1), y: r4(pc2) };
    });
    return { gamma, points: embPts };
  });
}

// ─── Dataset 2: Two moons ───

function generateMoons(): KernelPCAPoint[] {
  const points: KernelPCAPoint[] = [];
  // Upper crescent: label 0
  for (let i = 0; i < 100; i++) {
    const theta = (Math.PI * i) / 99;
    points.push({
      x: r4(Math.cos(theta) + noise(i + 1200)),
      y: r4(Math.sin(theta) + noise(i + 1400)),
      label: 0,
    });
  }
  // Lower crescent: label 1, offset and flipped
  for (let i = 0; i < 100; i++) {
    const theta = (Math.PI * i) / 99;
    points.push({
      x: r4(1 - Math.cos(theta) + noise(i + 1600)),
      y: r4(-Math.sin(theta) + 0.5 + noise(i + 1800)),
      label: 1,
    });
  }
  return points;
}

function moonEmbeddings(points: KernelPCAPoint[]): KernelPCAEmbedding[] {
  // At low gamma: classes overlap. At moderate gamma (1–5): good separation.
  // At very high gamma: overfit / noisy structure.
  return GAMMAS.map((gamma) => {
    const sep = Math.min(1, gamma / 2);
    const overfit = Math.max(0, (gamma - 10) / 90);
    const embPts = points.map((p, i) => {
      const isLower = p.label === 1;
      const classSign = isLower ? 1 : -1;
      const t = (i % 100) / 99; // parameter along crescent
      const pc1 =
        sep * classSign * (0.7 + 0.3 * Math.cos(Math.PI * t)) +
        (1 - sep) * (p.x - 0.5) * 0.8 +
        overfit * srand(i * 31 + 7) * 0.3;
      const pc2 =
        sep * 0.5 * Math.sin(Math.PI * t) * (isLower ? -1 : 1) +
        (1 - sep) * p.y * 0.6 +
        overfit * srand(i * 37 + 13) * 0.3;
      return { x: r4(pc1), y: r4(pc2) };
    });
    return { gamma, points: embPts };
  });
}

// ─── Dataset 3: Swiss roll (top-down view — spiral) ───

function generateSwissRoll(): KernelPCAPoint[] {
  const points: KernelPCAPoint[] = [];
  for (let i = 0; i < 200; i++) {
    // Angle parameter from 1.5π to 4.5π (1.5 turns)
    const t = 1.5 * Math.PI + (3 * Math.PI * i) / 199;
    const r = t / (4.5 * Math.PI); // normalized radius
    points.push({
      x: r4(r * Math.cos(t) + noise(i + 2000)),
      y: r4(r * Math.sin(t) + noise(i + 2200)),
      label: i, // color by index (angle parameter)
    });
  }
  return points;
}

function swissRollEmbeddings(points: KernelPCAPoint[]): KernelPCAEmbedding[] {
  // At low gamma: linear PCA — no unfolding, spiral structure preserved.
  // At moderate–high gamma: gradual unfolding into a line on PC1.
  return GAMMAS.map((gamma) => {
    const unfold = Math.min(1, gamma / 10);
    const embPts = points.map((p, i) => {
      const t = i / 199; // 0 to 1 parameter
      // Unfolded: PC1 spreads points along the arc-length parameter
      // Folded: PC1 ~ x projection of original spiral
      const pc1 =
        unfold * (t * 2 - 1) + (1 - unfold) * (p.x * 1.5 + noise(i + 2400) * 2);
      // PC2: at low gamma shows spiral curvature; at high gamma, small residual
      const pc2 =
        (1 - unfold) * (p.y * 1.5 + noise(i + 2600) * 2) +
        unfold * (0.15 * Math.sin(Math.PI * t * 4) + noise(i + 2800) * 0.5);
      return { x: r4(pc1), y: r4(pc2) };
    });
    return { gamma, points: embPts };
  });
}

// ─── Export ───

const circlePoints = generateCircles();
const moonPoints = generateMoons();
const swissRollPoints = generateSwissRoll();

export const kernelPCADatasets: KernelPCADataset[] = [
  {
    name: 'Concentric circles',
    points: circlePoints,
    embeddings: circleEmbeddings(circlePoints),
  },
  {
    name: 'Two moons',
    points: moonPoints,
    embeddings: moonEmbeddings(moonPoints),
  },
  {
    name: 'Swiss roll (top view)',
    points: swissRollPoints,
    embeddings: swissRollEmbeddings(swissRollPoints),
  },
];
