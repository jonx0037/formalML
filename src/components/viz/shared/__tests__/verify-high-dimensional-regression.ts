// =============================================================================
// verify-high-dimensional-regression.ts
//
// Numerical regression tests for the high-dimensional-regression viz components
// and any helpers added to src/components/viz/shared/proximalUtils.ts.
// Each test reproduces a numerical claim from the verified notebook
//   notebooks/high-dimensional-regression/01_high_dimensional_regression.ipynb
// and asserts the result lies within the brief-specified tolerance.
//
// Run with: pnpm verify:high-dimensional-regression
// Exits non-zero on first failure batch. Designed to be CI-friendly.
//
// Verification targets (from the topic-shipment plan, populated as viz components ship):
//   1.  softThreshold(z, 0.1) for z in {-2, -1, -0.05, 0, 0.05, 1, 2}
//       → {-1.9, -0.9, 0, 0, 0, 0.9, 1.9}                                        (1e-12)
//   2.  softThresholdScalar matches softThreshold componentwise on a 100-vector  (1e-12)
//   [3-15: populated as viz components are built; one row per viz, 1-3 spot checks per
//          notebook cell. See plans/implement-formalml-topic-validated-pike.md
//          Step 6 for the full inventory.]
// =============================================================================

import { softThreshold, softThresholdScalar } from '../proximalUtils';
import {
  generateDgp1,
  lambdaMax,
  lassoIsta,
  operatorNorm,
  predictMse,
} from '../high-dim-regression';

// -----------------------------------------------------------------------------
// Test plumbing (mirrors verify-local-regression.ts conventions)
// -----------------------------------------------------------------------------

let pass = 0;
let fail = 0;
const failures: string[] = [];

function ok(name: string, condition: boolean, detail: string): void {
  if (condition) {
    pass++;
    console.log(`  PASS  ${name}  ${detail}`);
  } else {
    fail++;
    failures.push(`${name}: ${detail}`);
    console.log(`  FAIL  ${name}  ${detail}`);
  }
}

function approxEq(
  name: string,
  observed: number,
  expected: number,
  tol: number,
  label: string,
): void {
  const gap = Math.abs(observed - expected);
  ok(
    name,
    gap < tol,
    `${label} observed=${observed.toFixed(8)} expected=${expected.toFixed(8)} gap=${gap.toExponential(3)} tol=${tol.toExponential(3)}`,
  );
}

// -----------------------------------------------------------------------------
// Test 1: softThreshold closed form on the §3.1 reference values
// -----------------------------------------------------------------------------

console.log('\n[1] softThreshold — §3.1 closed-form reference values');

{
  const lambda = 0.1;
  const inputs = [-2, -1, -0.05, 0, 0.05, 1, 2];
  const expected = [-1.9, -0.9, 0, 0, 0, 0.9, 1.9];
  const observed = softThreshold(inputs, lambda);
  for (let i = 0; i < inputs.length; i++) {
    approxEq(
      `softThreshold[z=${inputs[i]}]`,
      observed[i],
      expected[i],
      1e-12,
      `S(z, 0.1)`,
    );
  }
}

// -----------------------------------------------------------------------------
// Test 2: softThresholdScalar agreement with softThreshold componentwise
// -----------------------------------------------------------------------------

console.log('\n[2] softThresholdScalar — componentwise agreement with softThreshold');

{
  const lambda = 0.25;
  // Deterministic 100-vector (no Math.random; small LCG suffices for a sanity check).
  const seed = 42;
  let state = seed >>> 0;
  function lcg(): number {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  }
  const xs = Array.from({ length: 100 }, () => 4 * lcg() - 2);
  const vec = softThreshold(xs, lambda);
  let maxGap = 0;
  for (let i = 0; i < xs.length; i++) {
    const scalar = softThresholdScalar(xs[i], lambda);
    maxGap = Math.max(maxGap, Math.abs(scalar - vec[i]));
  }
  ok(
    'softThresholdScalar vs softThreshold (100-vector)',
    maxGap < 1e-12,
    `max gap = ${maxGap.toExponential(3)} (tol 1e-12)`,
  );
}

// -----------------------------------------------------------------------------
// Test 3: DGP-1 generator produces deterministic output for a fixed seed.
// -----------------------------------------------------------------------------

console.log('\n[3] generateDgp1 — deterministic + correct shape');

{
  const opts = { n: 50, p: 30, s: 5, sigma: 0.5, rho: 0.5, seed: 42 };
  const a = generateDgp1(opts);
  const b = generateDgp1(opts);
  // Determinism.
  let maxGap = 0;
  for (let i = 0; i < opts.n; i++) {
    maxGap = Math.max(maxGap, Math.abs(a.y[i] - b.y[i]));
    for (let j = 0; j < opts.p; j++) maxGap = Math.max(maxGap, Math.abs(a.X[i][j] - b.X[i][j]));
  }
  ok('generateDgp1 deterministic', maxGap < 1e-12, `max gap = ${maxGap.toExponential(3)}`);
  // Shape.
  ok('generateDgp1 X shape', a.X.length === opts.n && a.X[0].length === opts.p, `n × p = ${a.X.length} × ${a.X[0].length}`);
  ok('generateDgp1 y length', a.y.length === opts.n, `length = ${a.y.length}`);
  // Sparsity of betaStar.
  let nzBeta = 0;
  for (let j = 0; j < opts.p; j++) if (a.betaStar[j] !== 0) nzBeta++;
  ok('generateDgp1 betaStar sparsity', nzBeta === opts.s, `${nzBeta} nonzero (expected ${opts.s})`);
}

// -----------------------------------------------------------------------------
// Test 4: lambdaMax = ‖Xᵀy/n‖_∞ closed form, verified by direct check that the
// all-zero β̂ satisfies KKT at λ = lambdaMax (i.e., max(|grad|) ≤ λ).
// -----------------------------------------------------------------------------

console.log('\n[4] lambdaMax — KKT all-zero validity');

{
  const opts = { n: 100, p: 50, s: 5, sigma: 0.5, rho: 0.5, seed: 7 };
  const sample = generateDgp1(opts);
  const lMax = lambdaMax(sample.X, sample.y);
  // |Xᵀy/n|_max = lMax exactly.
  let observedMax = 0;
  for (let j = 0; j < opts.p; j++) {
    let xtyj = 0;
    for (let i = 0; i < opts.n; i++) xtyj += sample.X[i][j] * sample.y[i];
    observedMax = Math.max(observedMax, Math.abs(xtyj / opts.n));
  }
  approxEq('lambdaMax matches ‖Xᵀy/n‖_∞', lMax, observedMax, 1e-12, 'closed form');
  ok('lambdaMax positive', lMax > 0, `lMax = ${lMax.toFixed(4)}`);
}

// -----------------------------------------------------------------------------
// Test 5: lassoIsta at λ = lambdaMax produces β̂ = 0 (within ISTA convergence
// tolerance — the all-zero vector is the unique minimizer at λ = lambdaMax).
// -----------------------------------------------------------------------------

console.log('\n[5] lassoIsta at λ_max — all-zero solution');

{
  const opts = { n: 100, p: 50, s: 5, sigma: 0.5, rho: 0.5, seed: 7 };
  const sample = generateDgp1(opts);
  const lMax = lambdaMax(sample.X, sample.y);
  const L = operatorNorm(sample.X);
  // ISTA at exactly λ_max: should produce essentially zero (KKT inactive
  // condition is met everywhere with equality at one coord, which gives β̂ = 0).
  const beta = lassoIsta(sample.X, sample.y, lMax, L, 200);
  let maxAbs = 0;
  for (let j = 0; j < opts.p; j++) maxAbs = Math.max(maxAbs, Math.abs(beta[j]));
  ok('lassoIsta(λ_max) ≈ 0', maxAbs < 0.05, `max |β̂_j| = ${maxAbs.toFixed(4)} (loose tol; ISTA convergence)`);
}

// -----------------------------------------------------------------------------
// Test 6: lassoIsta sparsity at moderate λ — at λ between 0.5·λ_max and 0.7·λ_max
// the active-set size should be ≤ s + a few false-positives. This is the
// substantive sparsity check that drives every viz in the topic.
// -----------------------------------------------------------------------------

console.log('\n[6] lassoIsta — sparsity at moderate λ');

{
  const opts = { n: 100, p: 50, s: 5, sigma: 0.5, rho: 0.5, seed: 7 };
  const sample = generateDgp1(opts);
  const lMax = lambdaMax(sample.X, sample.y);
  const L = operatorNorm(sample.X);
  const lambda = 0.6 * lMax;
  const beta = lassoIsta(sample.X, sample.y, lambda, L, 200);
  let nz = 0;
  for (let j = 0; j < opts.p; j++) if (Math.abs(beta[j]) > 0.01) nz++;
  ok('lassoIsta(0.6·λ_max) sparsity bound', nz <= opts.s + 3, `${nz} nonzero coords (expected ≤ ${opts.s + 3} = s + 3 false positives)`);
}

// -----------------------------------------------------------------------------
// [7-15] Placeholder slots — populated as viz components ship.
// Each new viz adds 1-3 numerical-anchor rows verified against the notebook
// printed cell outputs. Anchor list from the plan's Step 1 table:
//   3.  LassoCV-selected λ on §1 toy ≈ 0.06 (±0.005)        — RidgeVsLassoCoefBars
//   4.  KKT residual-correlation max-deviation < 1e-3        — KktCorrelationHistogram
//   5.  FISTA F* (5000-iter reference) tolerance < 1e-6      — FistaConvergenceTrace
//   6.  FISTA iter-to-1e-6 count ±10%                        — FistaConvergenceTrace
//   7.  Lasso path knot count exact                          — LassoSolutionPath
//   8.  U-curve λ-min and λ-1SE values ±5%                   — LassoUCurve
//   9.  BRT empirical-vs-bound slope match within 0.1        — OracleInequalityRiskBound
//  10.  Population IC at ρ=0.5 ±0.01                         — IrrepresentableConditionViewer
//  11.  CV-selected λ from each selector ±10%                — CvLambdaSelector
//  12.  Adaptive vs vanilla active-set size exact            — RidgeLassoEnetPaths
//  13.  Debiased-lasso signal-coord coverage ∈ [0.90, 0.97]  — DebiasedLassoCoverage
//  14.  Naive post-selection signal-coord coverage ∈ [0.45, 0.75] — DebiasedLassoCoverage
//  15.  OLS-baseline coverage ∈ [0.92, 0.97]                  — DebiasedLassoCoverage
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// Summary
// -----------------------------------------------------------------------------

console.log(`\n${pass} passed, ${fail} failed`);
if (fail > 0) {
  console.log('\nFailures:');
  for (const f of failures) console.log(`  - ${f}`);
  process.exit(1);
}
process.exit(0);
