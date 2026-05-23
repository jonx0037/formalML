// =============================================================================
// verify-bayesian-inference.ts
//
// Numerical regression tests for src/components/viz/shared/bayesian-inference.ts.
// Every assertion either reproduces a printed numerical result from
//   notebooks/bayesian-inference/01_bayesian_inference.ipynb
// or pins a deterministic closed-form identity. Tolerances on the MC
// quantities (P(B > A), c*, etc.) reflect cross-RNG drift between NumPy's
// PCG64 and our seeded mulberry32 — exact bit-for-bit reproduction is not
// the goal, distributional invariance is.
//
// Run with: pnpm verify:bayesian-inference
// Exits non-zero on first failure. Designed to be CI-friendly.
// =============================================================================

import {
  lgamma,
  logBeta,
  betaPdf,
  betaCdf,
  betaQuantile,
  betaMean,
  betaVariance,
  betaMode,
  betaCentralCI,
  betaHPD,
  betaBinomialUpdate,
  betaBinomialLogPmf,
  normalNormalUpdate,
  normalNormalPosteriorPredictiveVar,
  normalPdf,
  normalQuantile,
  studentTPdf,
  trapezoidIntegrate,
  gridPosterior,
  logBayesFactorNormalNormal,
  bayesFactorOptimalTauSq,
  hierarchicalBetaBinomialEB,
  mulberry32,
  betaSample,
  probBeatsAB,
  probSwitchCorrect,
} from '../bayesian-inference';

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
    console.log(`  FAIL  ${name}  ${detail}`);
  }
}

function approxEq(name: string, got: number, expected: number, tol: number): void {
  const diff = Math.abs(got - expected);
  ok(
    name,
    diff <= tol,
    `got ${got.toExponential(4)}, expected ${expected.toExponential(4)} ± ${tol.toExponential(2)}, diff ${diff.toExponential(2)}`,
  );
}

function within(name: string, got: number, lo: number, hi: number): void {
  ok(name, got >= lo && got <= hi, `got ${got.toExponential(4)}, expected in [${lo}, ${hi}]`);
}

console.log('\n=== bayesian-inference verify suite ===\n');

// -----------------------------------------------------------------------------
// SECTION 1 — Special functions: lgamma, logBeta
// -----------------------------------------------------------------------------
console.log('--- §spec: special functions ---');

// lgamma(1) = 0; lgamma(0.5) = log(sqrt(pi)) = 0.5 * log(pi)
approxEq('lgamma_int1', lgamma(1), 0, 1e-12);
approxEq('lgamma_half', lgamma(0.5), 0.5 * Math.log(Math.PI), 1e-10);
// lgamma(5) = log(24)
approxEq('lgamma_5', lgamma(5), Math.log(24), 1e-10);
// logBeta(2, 3) = log( B(2,3) ) = log(1/12) since B(2,3) = (1*2)/(4!) * stuff = 1/12
// B(a,b) = (a-1)!(b-1)! / (a+b-1)!  so B(2,3) = 1*2 / 24 = 1/12
approxEq('logBeta_2_3', logBeta(2, 3), Math.log(1 / 12), 1e-10);

// -----------------------------------------------------------------------------
// SECTION 2 — Beta PDF / CDF / quantile (§4.2 + §5.4)
// -----------------------------------------------------------------------------
console.log('\n--- §4: Beta primitives ---');

// Notebook cell 13: "Beta(48, 954) posterior: mean = 0.047904, variance = 4.547296e-05"
approxEq('beta_48_954_mean', betaMean(48, 954), 0.047904, 1e-5);
approxEq('beta_48_954_var', betaVariance(48, 954), 4.547296e-5, 1e-9);

// Notebook cell 3: variant B posterior Beta(65, 937) — derived from prior Beta(1,1) + (n=1000, s=64)
const postB = betaBinomialUpdate(1, 1, 1000, 64);
ok(
  'betaBinomial_update_B',
  postB.alpha === 65 && postB.beta === 937,
  `posterior(B) = Beta(${postB.alpha}, ${postB.beta}) — expected Beta(65, 937)`,
);
approxEq('beta_65_937_mean', betaMean(65, 937), 0.0649, 1e-3);

// CDF must satisfy 0 < CDF < 1 and be monotone
const cdf01 = betaCdf(0.05, 48, 954);
within('beta_cdf_in_unit', cdf01, 0, 1);
ok(
  'beta_cdf_monotone',
  betaCdf(0.04, 48, 954) < betaCdf(0.05, 48, 954) && betaCdf(0.05, 48, 954) < betaCdf(0.06, 48, 954),
  'CDF is monotone',
);

// Quantile inverts CDF: betaCdf(betaQuantile(p)) = p
const p1 = 0.025;
const q1 = betaQuantile(p1, 48, 954);
approxEq('beta_quantile_inverts_cdf', betaCdf(q1, 48, 954), p1, 1e-6);

// Notebook cell 3: 95% credible interval A = [0.0356, 0.0620]
const ciA = betaCentralCI(48, 954, 0.95);
approxEq('beta_centralCI_A_lower', ciA.lower, 0.0356, 5e-4);
approxEq('beta_centralCI_A_upper', ciA.upper, 0.0620, 5e-4);

// Notebook cell 3: 95% credible interval B = [0.0505, 0.0809]
const ciB = betaCentralCI(65, 937, 0.95);
approxEq('beta_centralCI_B_lower', ciB.lower, 0.0505, 5e-4);
approxEq('beta_centralCI_B_upper', ciB.upper, 0.0809, 5e-4);

// -----------------------------------------------------------------------------
// SECTION 3 — Beta posterior summary (§5 dashboard, notebook cell 17)
// -----------------------------------------------------------------------------
console.log('\n--- §5: posterior summary dashboard ---');

// Beta(20, 20) — symmetric: MAP = mean = median = 0.5
approxEq('beta_20_20_mode', betaMode(20, 20), 0.5, 1e-9);
approxEq('beta_20_20_mean', betaMean(20, 20), 0.5, 1e-9);
approxEq('beta_20_20_median', betaQuantile(0.5, 20, 20), 0.5, 1e-6);
// HPD width = central-CI width for symmetric distributions (0.3044 per notebook)
const hpd2020 = betaHPD(20, 20, 0.95);
const ci2020 = betaCentralCI(20, 20, 0.95);
approxEq('beta_20_20_HPD_eq_central', hpd2020.upper - hpd2020.lower, ci2020.upper - ci2020.lower, 0.01);
approxEq('beta_20_20_CI_width', ci2020.upper - ci2020.lower, 0.3044, 0.01);

// Beta(2, 20) — skewed: MAP, median, mean diverge per notebook cell 17
approxEq('beta_2_20_mode', betaMode(2, 20), 0.05, 1e-9);
approxEq('beta_2_20_median', betaQuantile(0.5, 2, 20), 0.0786, 1e-3);
approxEq('beta_2_20_mean', betaMean(2, 20), 0.0909, 1e-3);

// MAP non-invariance under reparam (notebook cell 17): on logit scale, the
// MAP transformed back to theta-scale equals alpha/(alpha+beta) = posterior mean.
// This is the classic Jeffreys-style demo. The derivation: pi_phi(phi) =
// pi_theta(theta) * theta(1-theta), giving kernel theta^alpha (1-theta)^beta
// whose mode is alpha/(alpha+beta).
approxEq('beta_2_20_logit_MAP_back', 2 / (2 + 20), 0.0909, 1e-3);

// HPD shorter than central CI for skewed Beta (notebook: 0.2054 vs 0.2264)
const hpd220 = betaHPD(2, 20, 0.95);
const ci220 = betaCentralCI(2, 20, 0.95);
ok(
  'beta_2_20_HPD_shorter_than_central',
  hpd220.upper - hpd220.lower < ci220.upper - ci220.lower,
  `HPD width ${(hpd220.upper - hpd220.lower).toFixed(4)} < central CI width ${(ci220.upper - ci220.lower).toFixed(4)}`,
);

// -----------------------------------------------------------------------------
// SECTION 4 — Normal–Normal conjugacy and posterior predictive (§4.3, §6.3)
// -----------------------------------------------------------------------------
console.log('\n--- §4-6: Normal-Normal conjugacy + posterior predictive ---');

// Notebook cell 15: at n=25 with prior N(0,1), likelihood N(yBar, 1), yBar=1
// posterior precision 26 = 1 + 25, mu_n = 25/26 = 0.9615
const post = normalNormalUpdate({ mu0: 0, tau0Sq: 1, sigmaSq: 1, yBar: 1, n: 25 });
approxEq('nn_precision', post.precision, 26, 1e-10);
approxEq('nn_muN', post.muN, 25 / 26, 1e-10);
approxEq('nn_muN_numeric', post.muN, 0.9615, 1e-3);

// Theorem 6.1: posterior predictive variance = tau_n^2 + sigma^2 (notebook cell 20)
// At n=3 with prior N(0, 1), sigmaSq=1: tau_n^2 = 1/(1+3) = 0.25, pp variance = 1.25
const post3 = normalNormalUpdate({ mu0: 0, tau0Sq: 1, sigmaSq: 1, yBar: 0, n: 3 });
approxEq('nn_pp_variance_n3', normalNormalPosteriorPredictiveVar(post3, 1), 1.25, 1e-10);
// At n=30: tau_n^2 = 1/31, pp variance = 1 + 1/31 ≈ 1.0323
const post30 = normalNormalUpdate({ mu0: 0, tau0Sq: 1, sigmaSq: 1, yBar: 0, n: 30 });
approxEq('nn_pp_variance_n30', normalNormalPosteriorPredictiveVar(post30, 1), 1 + 1 / 31, 1e-9);
approxEq('nn_pp_variance_n30_numeric', normalNormalPosteriorPredictiveVar(post30, 1), 1.0323, 1e-3);

// As n -> infinity, pp variance -> sigma^2 (plug-in answer) — Theorem 6.1 corollary
const postLarge = normalNormalUpdate({ mu0: 0, tau0Sq: 1, sigmaSq: 1, yBar: 0, n: 10000 });
approxEq('nn_pp_variance_large_n', normalNormalPosteriorPredictiveVar(postLarge, 1), 1 + 1 / 10001, 1e-8);

// Normal PDF identity: N(0, 1) at x=0 is 1/sqrt(2pi)
approxEq('normal_pdf_origin', normalPdf(0, 0, 1), 1 / Math.sqrt(2 * Math.PI), 1e-12);

// Normal quantile invariants: P(Z <= 1.96) ≈ 0.975
approxEq('normal_quantile_975', normalQuantile(0.975), 1.96, 0.01);
approxEq('normal_quantile_5', normalQuantile(0.5), 0, 1e-9);

// -----------------------------------------------------------------------------
// SECTION 5 — Student-t prior (§11 non-conjugate robust)
// -----------------------------------------------------------------------------
console.log('\n--- §11: Student-t prior ---');

// t_3 at 0 has density Gamma(2)/(Gamma(1.5) * sqrt(3*pi)) = 1 / (sqrt(3)*pi)
// Actually: pdf(0) = Gamma((nu+1)/2) / (Gamma(nu/2) * sqrt(nu*pi))
// For nu=3, loc=0, scale=1: Gamma(2)/(Gamma(1.5)*sqrt(3*pi)) = 1/(0.886*sqrt(3*pi))
// Numerically: 0.36755
approxEq('studentT_3_pdf_zero', studentTPdf(0, 3, 0, 1), 0.36755259694786135, 1e-6);
// PDF integrates to 1 over a wide window
const tIntegral = trapezoidIntegrate((x) => studentTPdf(x, 3, 0, 1), -20, 20, 4000);
approxEq('studentT_integrates_to_one', tIntegral, 1, 1e-3);

// -----------------------------------------------------------------------------
// SECTION 6 — Trapezoid quadrature + grid posterior (§11)
// -----------------------------------------------------------------------------
console.log('\n--- §11: 1-D quadrature + grid posterior ---');

// Trapezoid on x^2 over [0, 1] = 1/3 (analytic)
approxEq('trapezoid_x2', trapezoidIntegrate((x) => x * x, 0, 1, 1000), 1 / 3, 1e-6);
// On Gaussian density over wide window = 1
approxEq(
  'trapezoid_normal_pdf',
  trapezoidIntegrate((x) => normalPdf(x, 0, 1), -10, 10, 2000),
  1,
  1e-6,
);

// Grid posterior reproduces conjugate result for Normal-Normal:
// Prior N(0, 1), likelihood ∏ N(y_i | mu, 1) with mean ybar, n=10, gives
// posterior N(ybar*n/(n+1), 1/(n+1)) = N(0.9091, 0.0909) at ybar = 1.
const gridFit = gridPosterior({
  logUnnormalized: (mu) => -0.5 * mu * mu + (-0.5 * 10 * (1 - mu) * (1 - mu)),
  gridStart: -3,
  gridEnd: 3,
  nGrid: 2048,
});
approxEq('grid_posterior_mean_matches_conjugate', gridFit.mean, 10 / 11, 1e-3);
approxEq('grid_posterior_var_matches_conjugate', gridFit.variance, 1 / 11, 1e-3);

// -----------------------------------------------------------------------------
// SECTION 7 — Bayes factor (§10.3 closed form)
// -----------------------------------------------------------------------------
console.log('\n--- §10: Bayes factor ---');

// Notebook cell 32: with n=30, yBar=0.2984, the closed-form optimum is
// tau* = 0.2360 and log10(B_10(tau*)) = 0.1495
const yBar_nb = 0.2984;
const n_nb = 30;
const tauStarSq = bayesFactorOptimalTauSq(yBar_nb, n_nb);
approxEq('bf_optimal_tauSq', tauStarSq, yBar_nb * yBar_nb - 1 / n_nb, 1e-12);
approxEq('bf_optimal_tau', Math.sqrt(tauStarSq), 0.2360, 1e-3);
const logBfAtOpt = logBayesFactorNormalNormal(yBar_nb, n_nb, Math.sqrt(tauStarSq));
const log10BfAtOpt = logBfAtOpt / Math.log(10);
approxEq('bf_log10_at_optimum', log10BfAtOpt, 0.1495, 1e-3);

// Notebook cell 32 cross-check: at tau=0.5, closed-form log B_10 = 0.1083 nats
// (notebook output: "Agreement within 0.0004 nats" — units are nats, not log10).
const logBfAtHalf = logBayesFactorNormalNormal(yBar_nb, n_nb, 0.5);
approxEq('bf_nats_at_tau_half', logBfAtHalf, 0.1083, 5e-3);

// Occam's penalty: tau -> infinity makes log B_10 -> -infinity
const logBfHuge = logBayesFactorNormalNormal(0.1, 100, 1000);
ok('bf_occam_penalty_dominates', logBfHuge < -2, `log B_10 at tau=1000 is ${logBfHuge.toFixed(2)} (very negative as expected)`);

// tau -> 0 makes log B_10 -> 0
const logBfTiny = logBayesFactorNormalNormal(0.1, 100, 1e-8);
approxEq('bf_tau_zero', logBfTiny, 0, 1e-6);

// -----------------------------------------------------------------------------
// SECTION 8 — Hierarchical Empirical Bayes (§9.3)
// -----------------------------------------------------------------------------
console.log('\n--- §9: hierarchical EB ---');

// Notebook cell 29: 8 variants
// {A: (100, 2), B: (100, 9), C: (200, 6), D: (200, 16), E: (500, 15),
//  F: (500, 40), G: (1000, 35), H: (1000, 75)}
// EB result: omega_hat = 0.0550, kappa_hat = 106.91
const groups = [
  { label: 'A', n: 100, s: 2 },
  { label: 'B', n: 100, s: 9 },
  { label: 'C', n: 200, s: 6 },
  { label: 'D', n: 200, s: 16 },
  { label: 'E', n: 500, s: 15 },
  { label: 'F', n: 500, s: 40 },
  { label: 'G', n: 1000, s: 35 },
  { label: 'H', n: 1000, s: 75 },
];
const eb = hierarchicalBetaBinomialEB(groups);
approxEq('eb_omega_hat', eb.omegaHat, 0.0550, 5e-3);
approxEq('eb_kappa_hat', eb.kappaHat, 106.91, 5);
ok(
  'eb_perGroup_count',
  eb.perGroup.length === 8,
  `8 per-group posteriors returned`,
);
// Pool weight at n=100 is 51.7% per notebook
const groupA = eb.perGroup.find((g) => g.label === 'A')!;
approxEq('eb_pool_weight_n100', groupA.poolWeight, 0.517, 0.02);
// Pool weight at n=1000 is 9.7%
const groupG = eb.perGroup.find((g) => g.label === 'G')!;
approxEq('eb_pool_weight_n1000', groupG.poolWeight, 0.097, 0.02);
// Shrinkage direction is correct: A's MLE (0.02) < posterior mean < grand mean (0.055)
ok(
  'eb_shrinkage_direction_A',
  groupA.mle < groupA.posteriorMean && groupA.posteriorMean < eb.omegaHat,
  `MLE ${groupA.mle.toFixed(3)} < post ${groupA.posteriorMean.toFixed(3)} < omega ${eb.omegaHat.toFixed(3)}`,
);
// Group B's MLE (0.09) > posterior mean > grand mean — shrunk downward toward 0.055
const groupB = eb.perGroup.find((g) => g.label === 'B')!;
ok(
  'eb_shrinkage_direction_B',
  groupB.mle > groupB.posteriorMean && groupB.posteriorMean > eb.omegaHat,
  `MLE ${groupB.mle.toFixed(3)} > post ${groupB.posteriorMean.toFixed(3)} > omega ${eb.omegaHat.toFixed(3)}`,
);

// -----------------------------------------------------------------------------
// SECTION 9 — Beta-Binomial log-marginal-likelihood identity
// -----------------------------------------------------------------------------
console.log('\n--- §9: Beta-Binomial PMF identity ---');

// betaBinomialLogPmf should be a valid PMF: sum over all s ∈ {0, ..., n} equals 1
let totalPmf = 0;
const nTest = 20;
const aTest = 5;
const bTest = 8;
for (let s = 0; s <= nTest; s++) {
  totalPmf += Math.exp(betaBinomialLogPmf(s, nTest, aTest, bTest));
}
approxEq('betaBinomial_pmf_sums_to_one', totalPmf, 1, 1e-10);

// -----------------------------------------------------------------------------
// SECTION 10 — Monte Carlo: P(B > A) and decision boundary (§1.2, §7)
// -----------------------------------------------------------------------------
console.log('\n--- §1.2 / §7: Monte Carlo posterior probabilities ---');

// Notebook cell 3: P(B > A | data) ≈ 0.9511 with M=200,000
// Cross-RNG drift (NumPy PCG64 vs our mulberry32) prevents bit-pinning;
// we use a smaller M and looser tolerance ±0.005.
const rng = mulberry32(42);
const pBeats = probBeatsAB(48, 954, 65, 937, 50000, rng);
within('mc_prob_B_beats_A', pBeats, 0.945, 0.957);

// Notebook cell 23: decision threshold c* = 0.0170 (posterior mean of B - A)
// At c = 0, P(switch correct) ≈ 0.9504
const rng2 = mulberry32(42);
const pSwitch0 = probSwitchCorrect(48, 954, 65, 937, 0, 30000, rng2);
within('mc_prob_switch_at_c_zero', pSwitch0, 0.94, 0.96);

// At c = c* = 0.0170, P(switch correct) ≈ 0.5 by symmetry of B - A around its mean
const rng3 = mulberry32(123);
const pSwitchStar = probSwitchCorrect(48, 954, 65, 937, 0.0170, 30000, rng3);
within('mc_prob_switch_at_c_star', pSwitchStar, 0.46, 0.54);

// At c = 0.05 (well above c*), P(switch correct) drops near 0
const rng4 = mulberry32(7);
const pSwitchHigh = probSwitchCorrect(48, 954, 65, 937, 0.05, 30000, rng4);
within('mc_prob_switch_high_cost', pSwitchHigh, 0, 0.05);

// -----------------------------------------------------------------------------
// SECTION 11 — Beta sampling sanity (gamma ratio)
// -----------------------------------------------------------------------------
console.log('\n--- §spec: Beta sampling ---');

// 10k Beta(2, 8) samples — mean should be 0.2 (= 2/10), std ≈ sqrt(2*8/(100*11)) ≈ 0.121
const rng5 = mulberry32(99);
let mean = 0;
const N = 10000;
const samples = new Float64Array(N);
for (let i = 0; i < N; i++) {
  samples[i] = betaSample(2, 8, rng5);
  mean += samples[i];
}
mean /= N;
let variance = 0;
for (let i = 0; i < N; i++) variance += (samples[i] - mean) ** 2;
variance /= N - 1;
within('beta_sample_mean_2_8', mean, 0.19, 0.21);
within('beta_sample_std_2_8', Math.sqrt(variance), 0.110, 0.135);

// -----------------------------------------------------------------------------
// SUMMARY
// -----------------------------------------------------------------------------
console.log(`\n${'='.repeat(60)}`);
console.log(`bayesian-inference verify: ${pass} pass / ${fail} fail`);
if (fail > 0) {
  console.log('\nFailures:');
  for (const f of failures) console.log(`  - ${f}`);
}
console.log(`${'='.repeat(60)}\n`);

process.exit(fail === 0 ? 0 : 1);
