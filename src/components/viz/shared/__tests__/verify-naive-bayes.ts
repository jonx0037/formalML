// =============================================================================
// verify-naive-bayes.ts
//
// Numerical regression tests for src/components/viz/shared/naive-bayes.ts.
// Each test reproduces a numerical claim from the verified notebook
//   notebooks/naive-bayes/01_naive_bayes.ipynb
// or pins a deterministic closed-form identity. Anchors come from the
// notebook's `cell.outputs[].text` after execution; loose-range checks
// substitute for RNG-bound MC quantities (TS mulberry32 vs numpy PCG64
// prevents bit-for-bit reproduction).
//
// Run with: pnpm verify:naive-bayes
// Exits non-zero on first failure. Designed to be CI-friendly.
// =============================================================================

import {
  gaussianNBFit,
  gaussianNBPredict,
  gaussianNBPredictProba,
  multinomialNBFit,
  multinomialNBPredictProba,
  dirichletPosteriorMean,
  bayesRiskGaussianMixture1D,
  plattScalingFit,
  plattScalingApply,
  expectedCalibrationError,
  brierScore,
  logLoss,
  mutualInformationDiscreteSmoothed,
  chowLiuTree,
  mulberry32,
  lgamma,
} from '../naive-bayes';

let pass = 0;
let fail = 0;
const failures: string[] = [];

function ok(name: string, cond: boolean, detail: string): void {
  if (cond) {
    pass++;
    console.log(`  PASS  ${name}  ${detail}`);
  } else {
    fail++;
    failures.push(`${name}: ${detail}`);
    console.error(`  FAIL  ${name}  ${detail}`);
  }
}

function approxEq(a: number, b: number, atol: number, name: string, detail: string): void {
  ok(name, Math.abs(a - b) < atol, `${detail} (|${a.toFixed(6)} − ${b.toFixed(6)}| = ${Math.abs(a - b).toExponential(2)}, atol = ${atol})`);
}

function within(a: number, lo: number, hi: number, name: string, detail: string): void {
  ok(name, a >= lo && a <= hi, `${detail} (${a.toFixed(6)} ∈ [${lo}, ${hi}])`);
}

// -----------------------------------------------------------------------------
// 1. Bayes risk 1D Gaussian mixture — closed-form Φ(−1) ≈ 0.158655.
// -----------------------------------------------------------------------------

console.log('\n[1] Bayes risk on symmetric 1D Gaussian mixture');
const r1 = bayesRiskGaussianMixture1D(1, 1, -1, 1, 0.5);
approxEq(r1, 0.158655, 1e-4, 'bayesRiskGaussianMixture1D symmetric N(±1, 1)', 'Bayes risk = Φ(−1) ≈ 0.158655');

// Asymmetric case: π = 0.7, μ = ±1, σ = 1. Decision boundary shifts, risk drops.
const r2 = bayesRiskGaussianMixture1D(1, 1, -1, 1, 0.7);
within(r2, 0.08, 0.16, 'bayesRiskGaussianMixture1D asymmetric prior', 'asymmetric risk in [0.08, 0.16]');
ok(
  'bayesRiskGaussianMixture1D monotone in separation',
  bayesRiskGaussianMixture1D(2, 1, -2, 1, 0.5) < r1,
  `risk at Δ=4 (${bayesRiskGaussianMixture1D(2, 1, -2, 1, 0.5).toFixed(4)}) < risk at Δ=2 (${r1.toFixed(4)})`,
);

// -----------------------------------------------------------------------------
// 2. Gaussian NB MLE — deterministic 2D fixture matches sklearn-style closed-form.
// -----------------------------------------------------------------------------

console.log('\n[2] Gaussian NB MLE on a small deterministic fixture');
const Xg = new Float64Array([
  -1, -1,
  -1.5, -0.8,
  -0.5, -1.2,
  -1.2, -0.7,
  1, 1,
  1.2, 0.8,
  0.7, 1.3,
]);
const yg = new Int32Array([0, 0, 0, 0, 1, 1, 1]);
const gnb = gaussianNBFit(Xg, yg, 2);
ok('gaussianNBFit returns 2 classes', gnb.classes.length === 2, `classes = [${gnb.classes.join(', ')}]`);
approxEq(gnb.classPriors[0], 4 / 7, 1e-12, 'class prior 0', 'π_0 = 4/7');
approxEq(gnb.classPriors[1], 3 / 7, 1e-12, 'class prior 1', 'π_1 = 3/7');
// Means: class 0 average = (-1.05, -0.925); class 1 average = (0.9667, 1.0333).
approxEq(gnb.means[0], (-1 - 1.5 - 0.5 - 1.2) / 4, 1e-12, 'class 0 mean x', 'μ_00 = (−1 − 1.5 − 0.5 − 1.2)/4');
approxEq(gnb.means[1], (-1 - 0.8 - 1.2 - 0.7) / 4, 1e-12, 'class 0 mean y', 'μ_01 = (−1 − 0.8 − 1.2 − 0.7)/4');
approxEq(gnb.means[2], (1 + 1.2 + 0.7) / 3, 1e-12, 'class 1 mean x', 'μ_10 = (1 + 1.2 + 0.7)/3');
approxEq(gnb.means[3], (1 + 0.8 + 1.3) / 3, 1e-12, 'class 1 mean y', 'μ_11 = (1 + 0.8 + 1.3)/3');

// Predict — every training point should be correctly classified on this clean fixture.
const pred = gaussianNBPredict(Xg, gnb);
let correct = 0;
for (let i = 0; i < yg.length; i++) if (pred[i] === yg[i]) correct++;
ok('gaussianNBPredict perfect train accuracy on clean fixture', correct === yg.length, `${correct}/${yg.length} correct`);

// Predict-proba sums to 1.
const proba = gaussianNBPredictProba(Xg, gnb);
for (let i = 0; i < yg.length; i++) {
  const s = proba[i * 2] + proba[i * 2 + 1];
  approxEq(s, 1.0, 1e-10, `predictProba row ${i} sums to 1`, 'each row sums to one');
}

// -----------------------------------------------------------------------------
// 3. Multinomial NB Laplace posterior identity — matches notebook §6.2.
//    Notebook fixture: counts = [3, 0, 5, 2, 0] (MLE = [0.3, 0, 0.5, 0.2, 0]).
//    With Laplace smoothing, the per-class token probability is
//       (count_k + α) / (Σ_j count_j + K·α).
// -----------------------------------------------------------------------------

console.log('\n[3] Multinomial NB Laplace posterior identity (§6.2 fixture)');
const counts = [3, 0, 5, 2, 0];

const post1 = Array.from(dirichletPosteriorMean(counts, 1.0));
const expected1 = [4 / 15, 1 / 15, 6 / 15, 3 / 15, 1 / 15];
for (let k = 0; k < 5; k++) approxEq(post1[k], expected1[k], 1e-12, `posterior α=1.0 cat ${k}`, 'matches (count_k + 1)/(n + K)');

const post05 = Array.from(dirichletPosteriorMean(counts, 0.5));
const expected05 = [3.5 / 12.5, 0.5 / 12.5, 5.5 / 12.5, 2.5 / 12.5, 0.5 / 12.5];
for (let k = 0; k < 5; k++) approxEq(post05[k], expected05[k], 1e-12, `posterior α=0.5 cat ${k}`, 'matches (count_k + 0.5)/(n + 2.5)');

const post2 = Array.from(dirichletPosteriorMean(counts, 2.0));
const expected2 = [5 / 20, 2 / 20, 7 / 20, 4 / 20, 2 / 20];
for (let k = 0; k < 5; k++) approxEq(post2[k], expected2[k], 1e-12, `posterior α=2.0 cat ${k}`, 'matches (count_k + 2)/(n + 10)');

// Multinomial NB on a tiny corpus (one document per class). predict_proba should be exactly
// the smoothed token-probability ratio applied to a query document.
console.log('\n[3b] multinomialNBFit + multinomialNBPredictProba sanity');
const XmnbTrain = new Float64Array([3, 0, 5, 2, 0, 0, 4, 1, 6, 3]);
const ymnbTrain = new Int32Array([0, 1]);
const mnb = multinomialNBFit(XmnbTrain, ymnbTrain, 5, 1.0);
const XmnbQuery = new Float64Array([1, 0, 2, 1, 0]); // looks more like class 0
const probaQ = multinomialNBPredictProba(XmnbQuery, mnb);
ok('multinomialNBPredictProba sums to 1', Math.abs(probaQ[0] + probaQ[1] - 1) < 1e-10, `sum = ${(probaQ[0] + probaQ[1]).toFixed(6)}`);
ok('multinomialNBPredictProba routes class-0-looking query to class 0', probaQ[0] > probaQ[1], `P(0|x) = ${probaQ[0].toFixed(3)} > P(1|x) = ${probaQ[1].toFixed(3)}`);

// -----------------------------------------------------------------------------
// 4. Chow–Liu maximum-spanning-tree on planted-dependency data (§9.1).
//    Notebook seed 42 with planted near-duplicates X_0↔X_1 (prob 0.8) and
//    X_2↔X_3 (prob 0.8) produces edges {(0,1), (1,4), (4,2), (2,3)} with
//    MI(0,1) ≈ 0.575, MI(2,3) ≈ 0.565. TS mulberry32 will produce a different
//    sample, but the planted-dependency structure dominates random MI by 50×,
//    so the two highest-MI edges are robustly the planted pairs.
// -----------------------------------------------------------------------------

console.log('\n[4] Chow–Liu spanning tree recovers planted dependencies');
function generateChowLiuFixture(seed: number, depProb: number) {
  const rng = mulberry32(seed);
  const n = 500;
  const d = 5;
  const X = new Float64Array(n * d);
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) X[i * d + j] = Math.floor(rng() * 3);
  // Planted: X_1 ≈ X_0 (prob depProb), X_3 ≈ X_2 (prob depProb).
  for (let i = 0; i < n; i++) {
    if (rng() < depProb) X[i * d + 1] = X[i * d + 0];
    if (rng() < depProb) X[i * d + 3] = X[i * d + 2];
  }
  return { X, n, d };
}
const fix = generateChowLiuFixture(42, 0.8);
const cl = chowLiuTree(fix.X, fix.n, fix.d, 1.0);
// Find which edges are in the tree as sets.
const edgeSet = new Set(cl.edges.map(([a, b]) => `${Math.min(a, b)}-${Math.max(a, b)}`));
ok('Chow–Liu recovers planted edge (0,1)', edgeSet.has('0-1'), `edges: ${[...edgeSet].join(', ')}`);
ok('Chow–Liu recovers planted edge (2,3)', edgeSet.has('2-3'), `edges: ${[...edgeSet].join(', ')}`);
within(cl.miMatrix[0 * 5 + 1], 0.4, 0.75, 'MI(X_0, X_1) in planted-dependency range', `MI ∈ [0.4, 0.75]`);
within(cl.miMatrix[2 * 5 + 3], 0.4, 0.75, 'MI(X_2, X_3) in planted-dependency range', `MI ∈ [0.4, 0.75]`);
// Non-planted edges should have tiny MI.
within(cl.miMatrix[0 * 5 + 4], 0, 0.05, 'MI(X_0, X_4) near zero', 'non-planted MI ≤ 0.05');

// -----------------------------------------------------------------------------
// 5. Platt scaling fit + apply on a synthetic well-separated 1D fixture.
//    Generate scores s ~ N(±1, 1) by class; logistic fit recovers a positive slope
//    (negative A in Platt's convention) and an intercept near zero (balanced classes).
// -----------------------------------------------------------------------------

console.log('\n[5] Platt scaling fit + apply');
const rngPlatt = mulberry32(7);
const scores: number[] = [];
const labels: number[] = [];
for (let i = 0; i < 400; i++) {
  const y = i < 200 ? 0 : 1;
  // Box-Muller for Gaussian via mulberry32.
  const u1 = Math.max(1e-9, rngPlatt());
  const u2 = rngPlatt();
  const g = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  scores.push(g + (y === 1 ? 1 : -1));
  labels.push(y);
}
const platt = plattScalingFit(scores, labels);
ok('Platt A < 0 (sigmoid opens upward in score)', platt.A < 0, `A = ${platt.A.toFixed(3)}`);
within(platt.B, -0.5, 0.5, 'Platt B near zero (balanced fixture)', `B = ${platt.B.toFixed(3)}`);
const probsAfterPlatt = Array.from(plattScalingApply(scores, platt));
ok('plattScalingApply output bounded in (0, 1)', probsAfterPlatt.every((p) => p > 0 && p < 1), 'all probs in (0, 1)');

// -----------------------------------------------------------------------------
// 6. Calibration losses on hand-crafted (probs, labels) examples.
// -----------------------------------------------------------------------------

console.log('\n[6] Calibration losses');
// Perfectly calibrated 4 bins of 10 samples each, predicted P = [0, 1/3, 2/3, 1].
const perfectProbs: number[] = [];
const perfectLabels: number[] = [];
for (let b = 0; b < 4; b++) {
  const p = [0, 1 / 3, 2 / 3, 1][b];
  for (let i = 0; i < 30; i++) {
    perfectProbs.push(p);
    perfectLabels.push(Math.random() < p ? 1 : 0); // labels noisy but expected freq matches p
  }
}
// Use deterministic labels so the test is stable: split evenly per bin.
const detProbs: number[] = [];
const detLabels: number[] = [];
for (const p of [0.1, 0.3, 0.6, 0.9]) {
  const nPos = Math.round(30 * p);
  for (let i = 0; i < 30; i++) {
    detProbs.push(p);
    detLabels.push(i < nPos ? 1 : 0);
  }
}
const eceDet = expectedCalibrationError(detProbs, detLabels, 4);
ok('expectedCalibrationError near zero on perfectly calibrated bins', eceDet < 0.02, `ECE = ${eceDet.toFixed(4)}`);

// Brier score on certainty: all probs = 0 but labels are 50/50 → Brier = 0.5.
const certainProbs = [0, 0, 0, 0];
const certainLabels = [0, 1, 0, 1];
approxEq(brierScore(certainProbs, certainLabels), 0.5, 1e-12, 'Brier on (probs=0, labels=01)', 'Brier = mean((0 − [0,1,0,1])²) = 0.5');

// Log-loss reflects the same.
const ll = logLoss([0.5, 0.5, 0.5, 0.5], certainLabels);
approxEq(ll, Math.log(2), 1e-10, 'logLoss on 0.5 baseline', 'log-loss = log(2)');

// -----------------------------------------------------------------------------
// 7. Mutual information identity — I(X; X) ≥ I(X; Y) for any X, Y.
// -----------------------------------------------------------------------------

console.log('\n[7] Mutual information sanity');
const xArr = [0, 1, 2, 0, 1, 2, 0, 1, 2];
const yArr = [0, 0, 0, 1, 1, 1, 2, 2, 2]; // independent of x
const miXY = mutualInformationDiscreteSmoothed(xArr, yArr, 1);
within(miXY, 0, 0.05, 'MI of independent uniform discrete pair is near zero', `MI ≈ 0`);
const miXX = mutualInformationDiscreteSmoothed(xArr, xArr, 1);
ok('MI(X, X) ≥ MI(X, Y) for independent Y', miXX >= miXY, `MI(X, X) = ${miXX.toFixed(3)}, MI(X, Y) = ${miXY.toFixed(3)}`);

// -----------------------------------------------------------------------------
// 8. lgamma identity check — lgamma(n) = log((n-1)!) for small positive integers.
// -----------------------------------------------------------------------------

console.log('\n[8] lgamma identity log Γ(n) = log((n−1)!)');
approxEq(lgamma(1), 0, 1e-10, 'lgamma(1)', 'log 0! = 0');
approxEq(lgamma(2), 0, 1e-10, 'lgamma(2)', 'log 1! = 0');
approxEq(lgamma(5), Math.log(24), 1e-10, 'lgamma(5)', 'log 4! = log 24');
approxEq(lgamma(10), Math.log(362880), 1e-9, 'lgamma(10)', 'log 9! = log 362880');

// -----------------------------------------------------------------------------

console.log(`\n${pass} pass, ${fail} fail`);
if (fail > 0) {
  console.error('\nFailures:');
  for (const f of failures) console.error(`  ${f}`);
}
process.exit(fail === 0 ? 0 : 1);
