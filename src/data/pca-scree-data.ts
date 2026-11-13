// Pre-computed eigenvalue data for PCA scree plot and selection-criteria visualizations.
// All values are either from real datasets (Wine) or deterministic synthetic constructions.

// ─── Types ───

export interface ScreeDataset {
  name: string;
  nFeatures: number;
  eigenvalues: number[];
  cumulativeEVR: number[];
  parallelAnalysis95th: number[];
  kaiserK: number;
  threshold95K: number;
}

// ─── Helper: compute cumulative explained variance ratio ───

function cumulativeEVR(eigenvalues: number[]): number[] {
  const total = eigenvalues.reduce((s, v) => s + v, 0);
  const result: number[] = [];
  let running = 0;
  for (const ev of eigenvalues) {
    running += ev;
    result.push(Math.round((running / total) * 10000) / 10000);
  }
  return result;
}

// ─── Helper: count eigenvalues > threshold ───

function countAbove(eigenvalues: number[], threshold: number): number {
  return eigenvalues.filter((v) => v > threshold).length;
}

// ─── Helper: smallest k where cumulative EVR >= target ───

function thresholdK(cumEVR: number[], target: number): number {
  for (let i = 0; i < cumEVR.length; i++) {
    if (cumEVR[i] >= target) return i + 1;
  }
  return cumEVR.length;
}

// ─── Dataset 1: Wine (13 features, standardized) ───

// Eigenvalues of the sample covariance of sklearn `load_wine` after
// `StandardScaler` (population sd): sigma**2 / (n - 1) from the SVD of the
// standardized 178×13 matrix, rounded to 4 decimals (matches the MDX code).
const wineEigenvalues = [
  4.7324, 2.5111, 1.4542, 0.9242, 0.8580, 0.6453, 0.5541, 0.3505, 0.2905,
  0.2523, 0.2271, 0.1697, 0.1040,
];

// Parallel analysis (Horn, 1965), same recipe as the MDX code: 50,000 draws of
// a 178×13 i.i.d. N(0, 1) matrix, eigenvalues sigma**2 / (n - 1) from its SVD,
// 95th percentile of each ordered eigenvalue, rounded to 2 decimals
// (numpy default_rng(20261112); Monte Carlo error < 0.005). Wine's
// λ3 = 1.4542 > 1.38 and λ4 = 0.9242 < 1.28, so parallel analysis keeps k = 3.
const wineParallel95 = [
  1.67, 1.50, 1.38, 1.28, 1.20, 1.12, 1.05, 0.98, 0.92, 0.85, 0.79, 0.73,
  0.65,
];

const wineCumEVR = cumulativeEVR(wineEigenvalues);

const wineDataset: ScreeDataset = {
  name: 'Wine (13 features)',
  nFeatures: 13,
  eigenvalues: wineEigenvalues,
  cumulativeEVR: wineCumEVR,
  parallelAnalysis95th: wineParallel95,
  kaiserK: countAbove(wineEigenvalues, 1.0),
  threshold95K: thresholdK(wineCumEVR, 0.95),
};

// ─── Dataset 2: Synthetic low-rank (20 features, true rank 3) ───

const lowRankEigenvalues = [
  15.2, 8.7, 5.1, 0.52, 0.48, 0.45, 0.42, 0.40, 0.38, 0.36, 0.34, 0.32,
  0.30, 0.28, 0.26, 0.24, 0.22, 0.20, 0.18, 0.16,
];

// Parallel analysis 95th percentile for 20 features — plausible values
// from random correlation matrices of size 20×200 (n=200 observations).
const lowRankParallel95 = [
  1.92, 1.74, 1.61, 1.51, 1.42, 1.34, 1.27, 1.20, 1.14, 1.08, 1.02, 0.97,
  0.91, 0.86, 0.81, 0.76, 0.70, 0.64, 0.58, 0.50,
];

const lowRankCumEVR = cumulativeEVR(lowRankEigenvalues);

const lowRankDataset: ScreeDataset = {
  name: 'Synthetic low-rank (20 features, true rank 3)',
  nFeatures: 20,
  eigenvalues: lowRankEigenvalues,
  cumulativeEVR: lowRankCumEVR,
  parallelAnalysis95th: lowRankParallel95,
  kaiserK: countAbove(lowRankEigenvalues, 1.0),
  threshold95K: thresholdK(lowRankCumEVR, 0.95),
};

// ─── Dataset 3: Synthetic high-rank (20 features, no clear elbow) ───

const highRankEigenvalues = [
  3.2, 2.8, 2.5, 2.2, 2.0, 1.8, 1.6, 1.5, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8,
  0.7, 0.6, 0.5, 0.4, 0.3, 0.2,
];

// Parallel analysis 95th percentile for this configuration.
const highRankParallel95 = [
  1.92, 1.74, 1.61, 1.51, 1.42, 1.34, 1.27, 1.20, 1.14, 1.08, 1.02, 0.97,
  0.91, 0.86, 0.81, 0.76, 0.70, 0.64, 0.58, 0.50,
];

const highRankCumEVR = cumulativeEVR(highRankEigenvalues);

const highRankDataset: ScreeDataset = {
  name: 'Synthetic high-rank (20 features, no clear elbow)',
  nFeatures: 20,
  eigenvalues: highRankEigenvalues,
  cumulativeEVR: highRankCumEVR,
  parallelAnalysis95th: highRankParallel95,
  kaiserK: countAbove(highRankEigenvalues, 1.0),
  threshold95K: thresholdK(highRankCumEVR, 0.95),
};

// ─── Export ───

export const screeDatasets: ScreeDataset[] = [
  wineDataset,
  lowRankDataset,
  highRankDataset,
];
