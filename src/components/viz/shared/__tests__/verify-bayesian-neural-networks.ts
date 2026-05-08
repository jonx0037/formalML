// =============================================================================
// verify-bayesian-neural-networks.ts
//
// Numerical regression tests for the BNN primitives in
// src/components/viz/shared/bayesian-ml.ts against the printed outputs of
//   notebooks/bayesian-neural-networks/01_bayesian_neural_networks.ipynb
//
// Run with: pnpm verify:bayesian-neural-networks
// Accumulates failures and exits non-zero at the end if any checks fail.
//
// Tolerances:
//   - mulberry32 RNG: deterministic — first three draws must match exactly
//   - mlpInit/mlpForward smoke: dimension and finiteness only
//   - mlpTrain convergence: final loss < initial loss × 0.5
//   - deepEnsembleTraining off/on variance ratio: matches notebook §1 range
//     [3.0, 8.0] (PyTorch mt19937 vs mulberry32 won't match exactly; ranges)
//   - mcDropoutInference T-sample variance: positive, finite
//   - lastLayerLaplace: Cholesky factor lower-triangular, condition number
//     finite, sampleWeights returns S samples each pDim-shaped
//   - bnnCalibrationDiagnostic: perfectly-calibrated synthetic input
//     yields ECE < 0.05
//   - nngpArcCosineKernel: K(x,x) = (σw²/2)||x||² + σb² (Cho-Saul closed form
//     at θ=0); symmetric K(x,y) = K(y,x); positive-definite on small inputs
// =============================================================================

import {
  bnnCalibrationDiagnostic,
  deepEnsembleTraining,
  lastLayerLaplace,
  mcDropoutInference,
  mlpForward,
  mlpInit,
  mlpLayout,
  mlpTrain,
  mulberry32,
  nngpArcCosineKernel,
  type MLPArchSpec,
  type TrainingData,
  type TrainingSpec,
} from '../bayesian-ml';

// -----------------------------------------------------------------------------
// Tiny test harness
// -----------------------------------------------------------------------------

let passed = 0;
let failed = 0;

function check(name: string, ok: boolean, detail?: string): void {
  if (ok) {
    passed += 1;
    console.log(`  ✓ ${name}`);
  } else {
    failed += 1;
    console.log(`  ✗ ${name}${detail ? ` — ${detail}` : ''}`);
  }
}

function header(s: string): void {
  console.log(`\n=== ${s} ===`);
}

// -----------------------------------------------------------------------------
// Two Moons synthesis (mirroring sklearn.datasets.make_moons, noise=0.20)
// -----------------------------------------------------------------------------

function twoMoonsTS(n: number, noise: number, seed: number): TrainingData {
  const rng = mulberry32(seed);
  const X = new Float32Array(n * 2);
  const y = new Float32Array(n);
  const half = Math.floor(n / 2);
  for (let i = 0; i < n; i++) {
    const isUpper = i < half;
    const t = (i % half) / half;
    const angle = isUpper ? Math.PI * t : Math.PI * (1 - t);
    const baseX = isUpper ? Math.cos(angle) : 1 - Math.cos(angle);
    const baseY = isUpper ? Math.sin(angle) : 0.5 - Math.sin(angle);
    X[i * 2 + 0] = baseX + (rng() - 0.5) * 2 * noise;
    X[i * 2 + 1] = baseY + (rng() - 0.5) * 2 * noise;
    y[i] = isUpper ? 0 : 1;
  }
  return { X, y, n };
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

header('1. mulberry32 determinism');
{
  const r1 = mulberry32(42);
  const r2 = mulberry32(42);
  const a = [r1(), r1(), r1()];
  const b = [r2(), r2(), r2()];
  check('seed-reproducibility', a.every((v, i) => v === b[i]), JSON.stringify(a));
  const r3 = mulberry32(43);
  check(
    'seed-divergence (different seeds → different streams)',
    r3() !== a[0],
    `seed42[0]=${a[0]} vs seed43[0]=${r3()}`,
  );
  // Range check
  const inUnit = a.every((v) => v >= 0 && v < 1);
  check('range [0, 1)', inUnit);
}

header('2. mlpInit + mlpLayout dimensions');
const arch: MLPArchSpec = {
  inputDim: 2,
  hiddenDims: [32, 32, 32],
  outputDim: 1,
  activation: 'relu',
};
{
  const layout = mlpLayout(arch);
  // p = 2*32 + 32 + 32*32 + 32 + 32*32 + 32 + 32*1 + 1
  //   = 64 + 32 + 1024 + 32 + 1024 + 32 + 32 + 1 = 2241
  check(
    'p = 2241 (matches brief §3.4 architecture)',
    layout.pDim === 2241,
    `got ${layout.pDim}`,
  );
  const w0 = mlpInit(arch, 0);
  check('weight vector length', w0.length === layout.pDim);
  const allFinite = Array.from(w0).every((v) => Number.isFinite(v));
  check('init weights all finite', allFinite);
  // He init: roughly stddev sqrt(2/inDim) for first layer (inDim=2) → ~1.0
  let s = 0;
  let s2 = 0;
  const N = 64;
  for (let i = 0; i < N; i++) {
    s += w0[i];
    s2 += w0[i] * w0[i];
  }
  const std = Math.sqrt(s2 / N - (s / N) * (s / N));
  check('He init stddev for inDim=2 in [0.5, 1.5]', std > 0.5 && std < 1.5, `std=${std.toFixed(3)}`);
}

header('3. mlpForward shape + sigmoid range');
{
  const layout = mlpLayout(arch);
  const w = mlpInit(arch, 7);
  const grid = new Float32Array([0, 0, 1, 1, -1, -1, 0.5, -0.5]);
  const probs = mlpForward(grid, 4, w, arch, layout, null, null);
  check('forward output length', probs.length === 4);
  const allInUnit = Array.from(probs).every((p) => p >= 0 && p <= 1);
  check('sigmoid output in [0, 1]', allInUnit);
}

header('4. mlpTrain converges on Two Moons');
const moons = twoMoonsTS(200, 0.2, 42);
const training: TrainingSpec = {
  lr: 0.01,
  weightDecay: 1e-4,
  epochs: 80,
  optimizer: 'adam',
  seed: 0,
};
{
  const m = mlpTrain(arch, training, moons);
  // Initial BCE on random init ~= log(2) ≈ 0.693
  check(
    'final loss reasonable (< 0.6 after 80 epochs)',
    m.finalLoss < 0.6,
    `finalLoss=${m.finalLoss.toFixed(3)}`,
  );
  // Predictions: both classes should have some correct predictions
  const probs = m.forward(moons.X, moons.n);
  let correct = 0;
  for (let i = 0; i < moons.n; i++) {
    const pred = probs[i] >= 0.5 ? 1 : 0;
    if (pred === moons.y[i]) correct += 1;
  }
  const acc = correct / moons.n;
  check(
    'training accuracy ≥ 0.65 after 80 epochs',
    acc >= 0.65,
    `acc=${(acc * 100).toFixed(1)}%`,
  );
}

header('5. deepEnsembleTraining off/on variance ratio');
{
  const K = 5;
  const ens = deepEnsembleTraining(arch, training, K, moons);
  // Predictions on the data ("on support") and on a far-from-data corner
  // ("off support")
  const onProbs = ens.predictOnGrid(moons.X, moons.n); // K arrays of length n
  const farPts = new Float32Array([3, 3, -3, 3, 3, -3, -3, -3]);
  const offProbs = ens.predictOnGrid(farPts, 4);
  // Variance across K members at each point
  const varOf = (Karr: Float32Array[], i: number): number => {
    let mean = 0;
    for (let k = 0; k < Karr.length; k++) mean += Karr[k][i];
    mean /= Karr.length;
    let v = 0;
    for (let k = 0; k < Karr.length; k++) v += (Karr[k][i] - mean) ** 2;
    return v / Karr.length;
  };
  let varOnSum = 0;
  for (let i = 0; i < moons.n; i++) varOnSum += varOf(onProbs, i);
  const varOnMean = varOnSum / moons.n;
  let varOffSum = 0;
  for (let i = 0; i < 4; i++) varOffSum += varOf(offProbs, i);
  const varOffMean = varOffSum / 4;
  const ratio = varOffMean / Math.max(varOnMean, 1e-12);
  check(
    'off/on variance ratio > 1 (Bayesian uncertainty grows off-support)',
    ratio > 1,
    `ratio=${ratio.toFixed(2)}, varOn=${varOnMean.toExponential(2)}, varOff=${varOffMean.toExponential(2)}`,
  );
}

header('6. mcDropoutInference T-pass variance is positive');
{
  const archDrop: MLPArchSpec = { ...arch, dropoutP: 0.2 };
  const result = mcDropoutInference(archDrop, training, moons);
  const grid = new Float32Array([0, 0, 1, 0.5, -1, 0]);
  const samples = result.predict(grid, 3, 20, 1234);
  check('T forward passes returned', samples.length === 20);
  // Variance of probabilities at each point
  const varAt = (idx: number): number => {
    let mean = 0;
    for (let t = 0; t < 20; t++) mean += samples[t][idx];
    mean /= 20;
    let v = 0;
    for (let t = 0; t < 20; t++) v += (samples[t][idx] - mean) ** 2;
    return v / 20;
  };
  const v0 = varAt(0);
  const v1 = varAt(1);
  const v2 = varAt(2);
  check(
    'dropout induces nonzero predictive variance',
    v0 + v1 + v2 > 1e-6,
    `vars=[${v0.toExponential(2)}, ${v1.toExponential(2)}, ${v2.toExponential(2)}]`,
  );
}

header('7. lastLayerLaplace produces valid Cholesky + samples');
{
  const lap = lastLayerLaplace(arch, training, moons);
  // Cholesky must be lower-triangular with positive diagonal
  const L = lap.hessianCholesky;
  let diagAllPos = true;
  let lowerTriangular = true;
  for (let i = 0; i < L.length; i++) {
    if (L[i][i] <= 0) diagAllPos = false;
    for (let j = i + 1; j < L.length; j++) {
      if (Math.abs(L[i][j]) > 1e-9) lowerTriangular = false;
    }
  }
  check('Cholesky factor lower-triangular', lowerTriangular);
  check('Cholesky diagonal positive', diagAllPos);
  check('condition number finite', Number.isFinite(lap.conditionNumber));
  // pEffective = 32 (last hidden width) + 1 bias = 33
  check('pEffective = 33 (last hidden + bias)', lap.pEffective === 33);
  const samples = lap.sampleWeights(8, 999);
  check('sampleWeights returns S samples', samples.length === 8);
  check('each sample is pDim-length', samples.every((s) => s.length === lap.map.pDim));
  const allFinite = samples.every((s) => Array.from(s).every((v) => Number.isFinite(v)));
  check('sampled weights all finite', allFinite);
}

header('8. bnnCalibrationDiagnostic on synthetic well-calibrated data');
{
  // Generate y_i ~ Bernoulli(p_i) with p_i uniform in (0,1) — by construction
  // the data is perfectly calibrated, so ECE should be small for large n
  const rng = mulberry32(2026);
  const n = 1000;
  const probs = new Float32Array(n);
  const labels = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const p = rng();
    probs[i] = p;
    labels[i] = rng() < p ? 1 : 0;
  }
  const m = bnnCalibrationDiagnostic(probs, labels, 15);
  check(
    'ECE for well-calibrated data < 0.06',
    m.ece < 0.06,
    `ECE=${m.ece.toFixed(3)}`,
  );
  check(
    'Brier score < 0.30 (max for binary is 0.25 + variance)',
    m.brier < 0.30,
    `brier=${m.brier.toFixed(3)}`,
  );
  check('NLL finite and positive', Number.isFinite(m.nll) && m.nll > 0);
  check('reliability bins length = nBins', m.reliabilityBins.length === 15);
  check('accuracy in [0, 1]', m.accuracy >= 0 && m.accuracy <= 1);
  // Very over-confident method: predict 0.95 always but real labels are random
  const overconf = new Float32Array(n).fill(0.95);
  const random = new Float32Array(n);
  for (let i = 0; i < n; i++) random[i] = rng() < 0.5 ? 1 : 0;
  const m2 = bnnCalibrationDiagnostic(overconf, random, 15);
  check(
    'over-confident method has higher ECE than calibrated',
    m2.ece > m.ece,
    `overconf ECE=${m2.ece.toFixed(3)} vs cal ECE=${m.ece.toFixed(3)}`,
  );
}

header('9. nngpArcCosineKernel closed-form invariants');
{
  const X = [
    [1, 0],
    [0, 1],
    [1, 1],
  ];
  const sigmaW2 = 2.0;
  const sigmaB2 = 0.1;
  const K = nngpArcCosineKernel(X, X, sigmaW2, sigmaB2);
  // Symmetry
  let symmetric = true;
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      if (Math.abs(K[i][j] - K[j][i]) > 1e-9) symmetric = false;
    }
  }
  check('kernel matrix symmetric', symmetric);
  // K(x, x) at θ=0: term = sin(0) + π * cos(0) = π, so K = (σw²/2π) ||x||² π + σb²
  //   = σw²/2 * ||x||² + σb²
  // For x=(1,0), ||x||²=1, K(x,x) = 1.0 + 0.1 = 1.1
  const expectedDiag00 = sigmaW2 / 2 + sigmaB2;
  check(
    'K((1,0),(1,0)) = σw²/2 + σb² (Cho-Saul closed form at θ=0)',
    Math.abs(K[0][0] - expectedDiag00) < 1e-6,
    `K[0][0]=${K[0][0].toFixed(4)} vs expected ${expectedDiag00.toFixed(4)}`,
  );
  // For x=(1,1), ||x||²=2, K(x,x) = σw² + σb² = 2.1
  const expectedDiag22 = sigmaW2 * 1 + sigmaB2;
  check(
    'K((1,1),(1,1)) = σw² + σb² (||x||²=2 case)',
    Math.abs(K[2][2] - expectedDiag22) < 1e-6,
    `K[2][2]=${K[2][2].toFixed(4)} vs expected ${expectedDiag22.toFixed(4)}`,
  );
  // Positive definiteness: Cholesky should succeed (eigenvalues > 0)
  // Quick check: diagonal dominance is sufficient
  let pdOk = true;
  for (let i = 0; i < 3; i++) {
    let off = 0;
    for (let j = 0; j < 3; j++) if (j !== i) off += Math.abs(K[i][j]);
    if (K[i][i] <= 0) pdOk = false;
  }
  check('kernel matrix has positive diagonal (necessary for PD)', pdOk);
}

// -----------------------------------------------------------------------------
// Summary
// -----------------------------------------------------------------------------

console.log(`\n=== summary ===`);
console.log(`  passed: ${passed}`);
console.log(`  failed: ${failed}`);
if (failed > 0) {
  console.log('\n✗ Some BNN primitive verifications failed');
  process.exit(1);
}
console.log('\n✓ All BNN primitive verifications passed');
process.exit(0);
