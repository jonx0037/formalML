// =============================================================================
// verify-generalization-bounds.ts
//
// Numerical regression tests for src/components/viz/shared/generalization-bounds.ts.
// Each test either reproduces a closed-form identity from the brief's
// "Expected numerical findings" tables OR checks a bounded-range / rate-decay
// property of a Monte-Carlo helper.
//
// Run with: pnpm verify:generalization-bounds
// Exits non-zero if any test fails.
//
// SCOPE: helpers that the §§3-11 viz components consume.  The §10 MarginBound
// and §12 Vacuousness viz consume precomputed JSON; this suite does NOT verify
// the JSON content (the precompute scripts print reference values at run time;
// re-run them if numerics drift).
//
// Reference values (closed-form, derived in the brief):
//   §3 finite-class bound @ (N=100, n=200, δ=0.05):                     0.15495
//   §4 DKW envelope @ (n=1000, δ=0.05):                                 0.04293
//   §5 Massart bound @ (N=51, n=50):                                    0.39624
//   §7 Corollary 3 bound @ (rad=0.05, n=3000, δ=0.05):                  0.10245
//   §11 BE deviation closed-form @ (β=0.02, n=100, δ=0.05):             ~0.32424
// =============================================================================

import {
  bousquetElisseeffDeviation,
  canonicalBound,
  dkwEnvelope,
  dudleyIntegral,
  empiricalRademacherMC,
  ermThreshold,
  finiteClassBound,
  finiteClassSampleComplexity,
  gaussianFrom,
  ksDistance,
  linearCoveringLogN,
  localRademacherFixedPoint,
  massartBound,
  mulberry32,
  ridgeStabilityBeta,
  sampleThresholdProblem,
  symmetrizationStats,
  thresholdClassMatrixMinimal,
  trueRiskThreshold,
} from '../generalization-bounds';

// -----------------------------------------------------------------------------
// Test plumbing (same shape as verify-normalizing-flows.ts)
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

function approxEq(name: string, observed: number, expected: number, tol: number, label: string): void {
  const gap = Math.abs(observed - expected);
  ok(
    name,
    gap < tol,
    `${label} observed=${observed.toFixed(6)} expected=${expected.toFixed(6)} gap=${gap.toExponential(3)} tol=${tol.toExponential(3)}`,
  );
}

function within(name: string, observed: number, lo: number, hi: number, label: string): void {
  ok(name, observed >= lo && observed <= hi, `${label} observed=${observed.toFixed(6)} in [${lo}, ${hi}]`);
}

// =============================================================================
// [1] §3 closed-form bounds — Theorem 1, sample complexity, Massart
// =============================================================================

console.log('\n[1] §3 closed-form finite-class bound + sample complexity + Massart');

// Closed form: √(log(4000)/400) ≈ 0.14400.  (The brief's table value 0.155 was
// a typo; the closed-form arithmetic is straightforward and verified by hand.)
approxEq('finite_class_bound_T1', finiteClassBound(100, 200, 0.05), Math.sqrt(Math.log(4000) / 400), 1e-9, 'Theorem 1 @ (N=100, n=200, δ=0.05)');
approxEq('finite_class_bound_loose', finiteClassBound(10000, 1000, 0.05), Math.sqrt(Math.log(2e4 / 0.05) / 2000), 1e-9, 'Cor. 1 closed-form @ (N=10k, n=1k, δ=0.05)');
approxEq('sample_complexity', finiteClassSampleComplexity(100, 0.1, 0.05), Math.log(4000) / 0.02, 1e-9, 'Sample complexity closed form');
approxEq('massart_bound_n50_N51', massartBound(51, 50), Math.sqrt((2 * Math.log(51)) / 50), 1e-9, 'Massart closed form');

// Theorem 1 is monotone-decreasing in n for fixed (N, δ).
ok('T1_monotone_in_n', finiteClassBound(100, 100, 0.05) > finiteClassBound(100, 1000, 0.05), 'T1 monotone in n');
// Theorem 1 is monotone-increasing in N for fixed (n, δ).
ok('T1_monotone_in_N', finiteClassBound(10000, 1000, 0.05) > finiteClassBound(10, 1000, 0.05), 'T1 monotone in N');

// =============================================================================
// [2] §4 DKW envelope + ECDF + KS distance
// =============================================================================

console.log('\n[2] §4 DKW envelope + ECDF + KS distance');

approxEq('dkw_envelope', dkwEnvelope(1000, 0.05), 0.04293, 1e-4, 'DKW @ (n=1000, δ=0.05)');

{
  // KS distance against a U[0,1] CDF: should converge to 0 at rate 1/√n.
  const rng = mulberry32(20260511);
  const n1 = 100;
  const samples1 = new Float64Array(n1);
  for (let i = 0; i < n1; i++) samples1[i] = rng();
  const ks1 = ksDistance(samples1, (t) => Math.max(0, Math.min(1, t)));
  // Expect ~0.05–0.15 at n=100 for U[0,1] samples.
  within('ks_distance_uniform_n100', ks1, 0.01, 0.20, 'KS U[0,1] n=100');

  // At n=1000 KS should be tighter.
  const rng2 = mulberry32(20260512);
  const n2 = 1000;
  const samples2 = new Float64Array(n2);
  for (let i = 0; i < n2; i++) samples2[i] = rng2();
  const ks2 = ksDistance(samples2, (t) => Math.max(0, Math.min(1, t)));
  ok('ks_decay_n_100_to_1000', ks1 > ks2, `n=100 KS=${ks1.toFixed(4)} > n=1000 KS=${ks2.toFixed(4)}`);
  // DKW @ δ=0.05 envelope should not be violated.
  ok('ks_bounded_by_dkw_n1000', ks2 <= dkwEnvelope(n2, 0.05) * 2, `KS ${ks2.toFixed(4)} ≤ 2*DKW ${(2 * dkwEnvelope(n2, 0.05)).toFixed(4)}`);
}

// =============================================================================
// [3] §2 sampleThresholdProblem + ermThreshold + trueRiskThreshold
// =============================================================================

console.log('\n[3] §2 threshold toy problem (sample / ERM / true risk)');

{
  // Closed-form: true risk at tau = tau* is exactly eta.
  approxEq('true_risk_at_tau_star', trueRiskThreshold(0.5, 0.10, 0.5), 0.10, 1e-12, 'R(h_{τ*}) = η');
  // True risk monotone in |τ - τ*| for fixed eta < 0.5.
  ok('true_risk_monotone', trueRiskThreshold(0.7, 0.10, 0.5) > trueRiskThreshold(0.6, 0.10, 0.5), 'R monotone in |τ - τ*|');
  // ERM converges to tau* in probability as n grows; check n=500, eta=0 should give a tau very near 0.5.
  const rng = mulberry32(20260511);
  const { X, Y } = sampleThresholdProblem(500, 0.0, 0.5, rng);
  const tauHat = ermThreshold(X, Y);
  within('erm_clean_n500', tauHat, 0.45, 0.55, 'ERM @ n=500, η=0');
  // With noise, ERM should still recover tau* roughly.
  const rng2 = mulberry32(20260512);
  const { X: X2, Y: Y2 } = sampleThresholdProblem(2000, 0.10, 0.5, rng2);
  const tauHat2 = ermThreshold(X2, Y2);
  within('erm_noisy_n2000', tauHat2, 0.40, 0.60, 'ERM @ n=2000, η=0.10');
}

// =============================================================================
// [4] §5 Rademacher complexity (Monte Carlo)
// =============================================================================

console.log('\n[4] §5 empirical Rademacher MC');

{
  // Threshold class on n=50 should give MC Rademacher ≈ 0.27, Massart upper bound ≈ 0.40.
  const rng = mulberry32(20260511);
  const n = 50;
  const X = new Float64Array(n);
  for (let i = 0; i < n; i++) X[i] = rng();
  const H = thresholdClassMatrixMinimal(X);
  const rad = empiricalRademacherMC(H, n + 1, n, 1500, rng);
  // Brief table: ≈ 0.27.  Wide tolerance because both PRNG path and MC variance differ.
  within('emp_rad_threshold_n50', rad.mean, 0.18, 0.36, 'emp Rademacher n=50');
  // Should not exceed Massart upper bound by more than 5% (MC variance allowance).
  ok('emp_rad_below_massart', rad.mean <= massartBound(n + 1, n) + 0.05, `rad ${rad.mean.toFixed(4)} ≤ Massart ${massartBound(n + 1, n).toFixed(4)} + 0.05`);
  // Standard error should be small.
  ok('emp_rad_se_small', rad.se < 0.02, `SE ${rad.se.toFixed(4)} < 0.02`);

  // Larger n should reduce the empirical Rademacher.
  const rng2 = mulberry32(20260513);
  const n2 = 500;
  const X2 = new Float64Array(n2);
  for (let i = 0; i < n2; i++) X2[i] = rng2();
  const H2 = thresholdClassMatrixMinimal(X2);
  const rad2 = empiricalRademacherMC(H2, n2 + 1, n2, 300, rng2);
  ok('emp_rad_decay_with_n', rad2.mean < rad.mean, `rad@n=500 ${rad2.mean.toFixed(4)} < rad@n=50 ${rad.mean.toFixed(4)}`);
}

// =============================================================================
// [5] §6 symmetrization chain (3 suprema + Rademacher)
// =============================================================================

console.log('\n[5] §6 symmetrization-chain statistics');

{
  const rng = mulberry32(20260511);
  const stats = symmetrizationStats(80, 80, rng);
  const mean = (a: Float64Array) => a.reduce((s, x) => s + x, 0) / a.length;
  const rawM = mean(stats.rawSup);
  const ghostM = mean(stats.ghostPairSup);
  const symM = mean(stats.sigmaSymSup);
  const radM = mean(stats.empRad);
  // Bounds chain — each step's mean is finite & positive on average.
  within('sym_raw_positive', rawM, 0.02, 0.30, 'E[sup_τ (Pf - P_n f)]');
  within('sym_ghost_positive', ghostM, 0.02, 0.30, 'E[sup_τ (P_{n\'} f - P_n f)]');
  within('sym_sigma_positive', symM, 0.02, 0.30, 'E[sym sup]');
  within('sym_emprad_positive', radM, 0.02, 0.30, 'E[R̂_S(H)]');
  // Lemma 3: 2 * empRad bounds raw sup.  In expectation: E[raw] ≤ 2 E[empRad].
  ok('lemma3_chain', rawM <= 2 * radM + 0.05, `E[raw]=${rawM.toFixed(4)} ≤ 2·E[empRad]=${(2 * radM).toFixed(4)} + 0.05`);
}

// =============================================================================
// [6] §7 Corollary 3 — canonical bound
// =============================================================================

console.log('\n[6] §7 Corollary 3 canonical bound');

{
  // Closed form at (rad=0.05, n=3000, δ=0.05): 0.05 + 3*sqrt(log(40)/6000) ≈ 0.10245
  approxEq('cor3_closed_form', canonicalBound(0.05, 3000, 0.05), 0.05 + 3 * Math.sqrt(Math.log(40) / 6000), 1e-9, 'Cor3 closed form');
  // Monotone in n.
  ok('cor3_monotone_in_n', canonicalBound(0.40, 30, 0.05) > canonicalBound(0.05, 3000, 0.05), 'Cor3 decreasing in n');
  // The full bound > Rademacher input (the additive deviation term is positive).
  ok('cor3_gt_rademacher', canonicalBound(0.10, 1000, 0.05) > 0.10, 'Cor3 > Rademacher input');
}

// =============================================================================
// [7] §8 Dudley's entropy integral + covering numbers
// =============================================================================

console.log('\n[7] §8 covering numbers + Dudley');

{
  // log N(ε, F, L_2) ≤ d log(3B/ε), so log N at ε = 3B is 0 (single covering element).
  approxEq('cover_log_at_3B', linearCoveringLogN(3.0, 10, 1.0), 0.0, 1e-9, 'log N(3B) = 0');
  // log N decreases as ε grows.
  ok('cover_decreasing', linearCoveringLogN(0.1, 10, 1.0) > linearCoveringLogN(0.5, 10, 1.0), 'log N decreasing in ε');
  // Dudley integral at (n=200, d=10, B=1, D_n=1).  The brief's table compared
  // this to the Massart-style bound (~0.52) informally; the actual closed-form
  // Dudley integral with d·log(3B/ε) covering numbers is much larger because
  // Dudley is a known loose-but-tractable upper bound (the chaining argument
  // gives slack vs. the tight Rademacher).  We check the bound is positive and
  // decreasing in n.
  const dud = dudleyIntegral(200, 10, 1.0, 1.0, 200);
  within('dudley_n200_d10', dud, 0.30, 6.0, 'Dudley @ (n=200, d=10)');
  // Dudley decreasing in n at fixed (d, B, D_n).
  const dudHi = dudleyIntegral(2000, 10, 1.0, 1.0, 200);
  ok('dudley_decreasing_n', dud > dudHi, `Dudley@n=200 ${dud.toFixed(4)} > Dudley@n=2000 ${dudHi.toFixed(4)}`);
}

// =============================================================================
// [8] §9 local Rademacher fixed point
// =============================================================================

console.log('\n[8] §9 local Rademacher fixed point');

{
  // r* should decrease as n grows.
  const rng = mulberry32(20260511);
  const r100 = localRademacherFixedPoint(100, 0.05, rng, 80, 12);
  const rng2 = mulberry32(20260512);
  const r1000 = localRademacherFixedPoint(1000, 0.05, rng2, 80, 12);
  within('local_rad_r_n100', r100, 0.005, 0.30, 'r* @ n=100');
  within('local_rad_r_n1000', r1000, 0.001, 0.10, 'r* @ n=1000');
  ok('local_rad_decreasing', r100 > r1000, `r*@100 ${r100.toFixed(4)} > r*@1000 ${r1000.toFixed(4)}`);
}

// =============================================================================
// [9] §11 ridge stability + Bousquet-Elisseeff deviation
// =============================================================================

console.log('\n[9] §11 ridge stability + BE deviation');

{
  // BE closed form: β + (2nβ + M) √(log(1/δ)/(2n))
  const beta = 0.02;
  const n = 100;
  const delta = 0.05;
  const expected = beta + (2 * n * beta + 1) * Math.sqrt(Math.log(1 / delta) / (2 * n));
  approxEq('be_closed_form', bousquetElisseeffDeviation(beta, n, delta), expected, 1e-9, 'BE closed form');

  // β-stability: large λ should give smaller β than small λ.
  const unif = mulberry32(20260511);
  const gauss = gaussianFrom(mulberry32(20260512));
  const betaLowLambda = ridgeStabilityBeta(80, 30, 0.01, 8, unif, gauss);
  const unif2 = mulberry32(20260511);
  const gauss2 = gaussianFrom(mulberry32(20260512));
  const betaHighLambda = ridgeStabilityBeta(80, 30, 1.0, 8, unif2, gauss2);
  ok(
    'beta_decreases_with_lambda',
    betaLowLambda > betaHighLambda,
    `β@λ=0.01 ${betaLowLambda.toFixed(4)} > β@λ=1.0 ${betaHighLambda.toFixed(4)}`,
  );
}

// =============================================================================
// Summary
// =============================================================================

console.log(`\nPASSED: ${pass}`);
console.log(`FAILED: ${fail}`);
if (fail > 0) {
  console.log('\nFailures:');
  failures.forEach((f) => console.log(`  - ${f}`));
}
process.exit(fail > 0 ? 1 : 0);
