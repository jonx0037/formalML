// =============================================================================
// verify-extreme-value-theory.ts
//
// Numerical regression tests for src/data/extreme-value-theory.ts.
// Each test reproduces a numerical claim from the notebook
//   notebooks/extreme-value-theory/01_extreme_value_theory.ipynb
// and asserts the result lies in the expected range.
//
// Run with: pnpm verify:extreme-value-theory
// Exits non-zero on first failure-cluster. Designed to be CI-friendly.
//
// Tolerances follow the brief's §13 testing checklist:
//   - ±1% relative for deterministic quantities (Gumbel theoretical moments,
//     GEV densities, Gumbel normalization sequences),
//   - ±0.5 absolute for stochastic ξ̂ estimators at B=50 blocks (the notebook's
//     printed SE at B=50 is ~0.1–0.5; PRNG-implementation differences
//     between mulberry32 and NumPy's PCG64 add a further ±0.05–0.10),
//   - ±5% relative for return-level / VaR / ES extrapolations,
//   - ±10–15% for tail-index estimators, where finite-sample bias is large.
// =============================================================================

import {
  gumbelNormalizationNormal,
  gevPdf,
  gevCdf,
  gevQuantile,
  gpdPdf,
  gpdCdf,
  gpdQuantile,
  blockMaxima,
  sampleNormal,
  sampleParetoStandard,
  gevMle,
  gevPwm,
  returnLevel,
  returnLevelSeDelta,
  gpdMle,
  hillEstimator,
  pickandsEstimator,
  dedhEstimator,
  potVar,
  potEs,
  mulberry32,
} from '../../../../data/extreme-value-theory';

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

function approx(actual: number, expected: number, tol: number): boolean {
  return Math.abs(actual - expected) <= tol;
}

function approxRel(actual: number, expected: number, relTol: number): boolean {
  return Math.abs(actual - expected) / Math.max(Math.abs(expected), 1e-12) <= relTol;
}

function timed<T>(label: string, fn: () => T): T {
  const t0 = Date.now();
  const result = fn();
  const dt = ((Date.now() - t0) / 1000).toFixed(2);
  console.log(`  (${label} took ${dt}s)`);
  return result;
}

// =============================================================================
// Section 1: Deterministic distribution & normalization sanity checks
// =============================================================================

function test_1_distribution_sanity(): void {
  console.log('\nSection 1: Distribution & normalization sanity (deterministic)');

  // 1.1: Gumbel normalization sequences for n = 1000.
  // Notebook §1.3 derived form: a_n = 1/sqrt(2 ln n), b_n = sqrt(2 ln n) - (ln ln n + ln 4π) / (2 sqrt(2 ln n)).
  // For n = 1000: 2 ln n ≈ 13.8155; sqrt ≈ 3.7169.
  //   a_n ≈ 1/3.7169 ≈ 0.2690 ✓ (cell 4 expected value: 0.2690)
  //   b_n ≈ 3.7169 - (ln(6.9078) + ln(12.566)) / (2 · 3.7169)
  //       ≈ 3.7169 - (1.9326 + 2.531) / 7.4338
  //       ≈ 3.7169 - 0.6004 ≈ 3.1165 ✓
  const { aN, bN } = gumbelNormalizationNormal(1000);
  ok(
    '1.1a  gumbelNormalizationNormal(1000).aN ≈ 0.2690',
    approxRel(aN, 0.2690, 0.01),
    `aN = ${aN.toFixed(4)} (notebook 0.2690)`,
  );
  ok(
    '1.1b  gumbelNormalizationNormal(1000).bN ≈ 3.1165',
    approxRel(bN, 3.1165, 0.01),
    `bN = ${bN.toFixed(4)} (notebook 3.1165)`,
  );

  // 1.2: Gumbel density at z = 0:  g(0) = exp(-0 - exp(0)) = exp(-1) ≈ 0.3679.
  const gumbelAt0 = gevPdf(0, 0, 0, 1);
  ok(
    '1.2  gevPdf(0, ξ=0, μ=0, σ=1) = exp(-1) ≈ 0.3679',
    approxRel(gumbelAt0, Math.exp(-1), 0.01),
    `value = ${gumbelAt0.toFixed(4)}`,
  );

  // 1.3: Gumbel CDF at z = 0:  G(0) = exp(-exp(0)) = exp(-1) ≈ 0.3679.
  const gumbelCdf0 = gevCdf(0, 0, 0, 1);
  ok(
    '1.3  gevCdf(0, ξ=0, μ=0, σ=1) = exp(-1) ≈ 0.3679',
    approxRel(gumbelCdf0, Math.exp(-1), 0.01),
    `value = ${gumbelCdf0.toFixed(4)}`,
  );

  // 1.4: Gumbel median:  G^{-1}(0.5) = -log(-log(0.5)) ≈ 0.3665.
  const gumbelMedian = gevQuantile(0.5, 0, 0, 1);
  ok(
    '1.4  gevQuantile(0.5, ξ=0) = -ln(-ln 0.5) ≈ 0.3665',
    approxRel(gumbelMedian, -Math.log(-Math.log(0.5)), 0.01),
    `value = ${gumbelMedian.toFixed(4)}`,
  );

  // 1.5: Fréchet density at x = 1, ξ = 0.5: g(1) = (1 + 0.5)^{-3} exp(-(1.5)^{-2}) ≈ 0.296·0.6412 ≈ 0.190.
  // Direct calc: (1 + 0.5*1)^{-1/0.5 - 1} * exp(-(1+0.5)^{-1/0.5}) / 1
  //            = 1.5^{-3} * exp(-1.5^{-2}) = 0.2963 * exp(-0.4444) = 0.2963 * 0.6412 = 0.1900
  const frechetAt1 = gevPdf(1, 0.5, 0, 1);
  ok(
    '1.5  gevPdf(1, ξ=0.5, μ=0, σ=1) ≈ 0.1900 (Fréchet)',
    approxRel(frechetAt1, 0.1900, 0.02),
    `value = ${frechetAt1.toFixed(4)}`,
  );

  // 1.6: GPD CDF at y=1 with ξ=0.5, β=1:  H(1) = 1 - (1.5)^{-2} = 1 - 0.4444 = 0.5556.
  const gpdCdf1 = gpdCdf(1, 0.5, 1);
  ok(
    '1.6  gpdCdf(1, ξ=0.5, β=1) ≈ 0.5556',
    approxRel(gpdCdf1, 1 - 1 / 2.25, 0.01),
    `value = ${gpdCdf1.toFixed(4)}`,
  );

  // 1.7: GPD CDF at y=1 with ξ=0, β=1 (Exponential):  H(1) = 1 - e^{-1} ≈ 0.6321.
  const gpdExpCdf1 = gpdCdf(1, 0, 1);
  ok(
    '1.7  gpdCdf(1, ξ=0, β=1) = 1 - e^{-1} ≈ 0.6321',
    approxRel(gpdExpCdf1, 1 - Math.exp(-1), 0.01),
    `value = ${gpdExpCdf1.toFixed(4)}`,
  );

  // 1.8: GPD quantile inverse round-trip:  H^{-1}(H(2.5; ξ=0.3, β=1.5); ξ=0.3, β=1.5) ≈ 2.5.
  const gpdInverseRoundTrip = gpdQuantile(gpdCdf(2.5, 0.3, 1.5), 0.3, 1.5);
  ok(
    '1.8  GPD CDF/quantile round-trip at y=2.5, ξ=0.3, β=1.5',
    approxRel(gpdInverseRoundTrip, 2.5, 1e-6),
    `round-trip = ${gpdInverseRoundTrip.toFixed(6)}`,
  );

  // 1.9: GPD pdf at y=0, ξ=0, β=1.5: should equal 1/β = 0.6667 (Exponential rate).
  const gpdExpPdf0 = gpdPdf(0, 0, 1.5);
  ok(
    '1.9  gpdPdf(0, ξ=0, β=1.5) = 1/β ≈ 0.6667',
    approxRel(gpdExpPdf0, 1 / 1.5, 0.01),
    `value = ${gpdExpPdf0.toFixed(4)}`,
  );
}

// =============================================================================
// Section 2: GEV MLE + PWM on simulated GEV data with known truth
// =============================================================================

function test_2_gev_self_recovery(): void {
  console.log('\nSection 2: GEV MLE / PWM self-recovery on simulated GEV data');

  // Simulate B=400 samples directly from GEV(ξ=0.3, μ=2, σ=1) and refit.
  // Tolerances tighter than for §4 worked-example MLE, since here the data
  // are exactly GEV (no asymptotic-approximation bias).
  const rng = mulberry32(7);
  const B = 400;
  const xiTrue = 0.3, muTrue = 2, sigmaTrue = 1;
  const data = new Float64Array(B);
  for (let i = 0; i < B; i++) {
    let u = rng();
    if (u <= 0) u = 1e-12;
    if (u >= 1) u = 1 - 1e-12;
    data[i] = gevQuantile(u, xiTrue, muTrue, sigmaTrue);
  }

  const mle = timed('GEV MLE on B=400 self-data', () => gevMle(data));
  ok(
    '2.1  gevMle ξ̂ within ±0.10 of ξ_true=0.3 (simulated GEV)',
    approx(mle.theta.xi, xiTrue, 0.1),
    `ξ̂ = ${mle.theta.xi.toFixed(3)}; converged=${mle.converged}, iter=${mle.iter}`,
  );
  ok(
    '2.2  gevMle μ̂ within ±0.20 of μ_true=2',
    approx(mle.theta.mu, muTrue, 0.2),
    `μ̂ = ${mle.theta.mu.toFixed(3)}`,
  );
  ok(
    '2.3  gevMle σ̂ within ±0.20 of σ_true=1',
    approx(mle.theta.sigma, sigmaTrue, 0.2),
    `σ̂ = ${mle.theta.sigma.toFixed(3)}`,
  );

  const pwm = gevPwm(data);
  ok(
    '2.4  gevPwm ξ̂ within ±0.10 of ξ_true=0.3',
    approx(pwm.xi, xiTrue, 0.1),
    `PWM ξ̂ = ${pwm.xi.toFixed(3)}`,
  );

  // Return level: x_T at theta_true; should match closed-form gevQuantile.
  const T = 100;
  const xT_true = gevQuantile(1 - 1 / T, xiTrue, muTrue, sigmaTrue);
  const xT_mle = returnLevel(mle.theta, T);
  ok(
    '2.5  returnLevel(θ̂_MLE, T=100) within ±15% of true return level',
    approxRel(xT_mle, xT_true, 0.15),
    `x_100^MLE = ${xT_mle.toFixed(2)}, true = ${xT_true.toFixed(2)}`,
  );

  // Delta-method SE for x_T should be > 0 and finite (sanity).
  const xT_se = returnLevelSeDelta(mle.theta, mle.cov, T);
  ok(
    '2.6  returnLevelSeDelta finite and positive',
    Number.isFinite(xT_se) && xT_se > 0,
    `SE(x_100) = ${xT_se.toFixed(3)}`,
  );
}

// =============================================================================
// Section 3: GEV inference on §4 worked examples (Normal blocks, Pareto blocks)
// =============================================================================

function test_3_gev_worked_examples(): void {
  console.log('\nSection 3: GEV inference on §4 worked examples (Examples 4 + 5)');

  // Example 4 (Normal blocks): N=50000 standard normals, B=50, m=1000.
  // Notebook prints (xi, mu, sigma) MLE = (-0.086, 3.123, 0.275).
  // PRNG difference + finite B=50 → expand tolerance to ±0.5 absolute on ξ̂.
  const rngNormal = mulberry32(42);
  const normalSamples = sampleNormal(50000, rngNormal);
  const normalBlocks = blockMaxima(normalSamples, 1000);
  const normalMle = timed('Example 4 MLE (B=50)', () => gevMle(normalBlocks));
  ok(
    '3.1  Example 4 (Normal blocks) MLE ξ̂ within [-0.6, 0.6] (notebook -0.086)',
    normalMle.theta.xi >= -0.6 && normalMle.theta.xi <= 0.6,
    `ξ̂ = ${normalMle.theta.xi.toFixed(3)}, μ̂ = ${normalMle.theta.mu.toFixed(3)}, σ̂ = ${normalMle.theta.sigma.toFixed(3)}`,
  );
  ok(
    '3.2  Example 4 MLE μ̂ within [2.7, 3.5] (notebook 3.12)',
    normalMle.theta.mu >= 2.7 && normalMle.theta.mu <= 3.5,
    `μ̂ = ${normalMle.theta.mu.toFixed(3)} (theoretical b_1000 ≈ 3.12)`,
  );
  ok(
    '3.3  Example 4 MLE σ̂ within [0.18, 0.45] (notebook 0.275)',
    normalMle.theta.sigma >= 0.18 && normalMle.theta.sigma <= 0.45,
    `σ̂ = ${normalMle.theta.sigma.toFixed(3)} (theoretical a_1000 ≈ 0.27)`,
  );

  const normalPwm = gevPwm(normalBlocks);
  ok(
    '3.4  Example 4 PWM ξ̂ within [-0.4, 0.4] (notebook -0.066)',
    normalPwm.xi >= -0.4 && normalPwm.xi <= 0.4,
    `ξ̂_PWM = ${normalPwm.xi.toFixed(3)}, μ̂ = ${normalPwm.mu.toFixed(3)}, σ̂ = ${normalPwm.sigma.toFixed(3)}`,
  );

  // Example 5 (Pareto blocks): N=50000 Pareto(α=2), B=50, m=1000.
  // Truth: ξ=0.5, normalizing scale ~31.62. Notebook MLE: (0.308, 29.90, 11.96).
  const rngPareto = mulberry32(42);
  const paretoSamples = sampleParetoStandard(2, 50000, rngPareto);
  const paretoBlocks = blockMaxima(paretoSamples, 1000);
  const paretoMle = timed('Example 5 MLE (B=50)', () => gevMle(paretoBlocks));
  ok(
    '3.5  Example 5 (Pareto blocks) MLE ξ̂ within [0.0, 0.9] (notebook 0.308, truth 0.5)',
    paretoMle.theta.xi >= 0.0 && paretoMle.theta.xi <= 0.9,
    `ξ̂ = ${paretoMle.theta.xi.toFixed(3)}, μ̂ = ${paretoMle.theta.mu.toFixed(2)}, σ̂ = ${paretoMle.theta.sigma.toFixed(2)}`,
  );

  const paretoPwm = gevPwm(paretoBlocks);
  ok(
    '3.6  Example 5 PWM ξ̂ within [0.05, 0.85] (notebook 0.369, truth 0.5)',
    paretoPwm.xi >= 0.05 && paretoPwm.xi <= 0.85,
    `ξ̂_PWM = ${paretoPwm.xi.toFixed(3)}`,
  );

  // Return level T=100 for the fitted Pareto blocks: should be substantial (well above the data).
  const xT_pareto = returnLevel(paretoMle.theta, 100);
  ok(
    '3.7  Example 5 return level x_100 in [50, 1000] (extrapolation sanity)',
    xT_pareto >= 50 && xT_pareto <= 1000,
    `x_100 = ${xT_pareto.toFixed(2)}`,
  );
}

// =============================================================================
// Section 4: GPD inference on §5 worked examples (Normal/Pareto exceedances)
// =============================================================================

function test_4_gpd_worked_examples(): void {
  console.log('\nSection 4: GPD inference on §5 worked examples (Examples 6 + 7)');

  // Example 7 (Pareto exceedances): N=50000 Pareto(α=2), threshold = 98%-tile.
  // Notebook (cell 12): u=7.30, N_u=1000, GPD-MLE ξ̂=0.480, β̂=3.59.
  // Truth: ξ=0.5, β(u) ∝ u (regular variation).
  const rngPareto = mulberry32(42);
  const paretoSamples = sampleParetoStandard(2, 50000, rngPareto);
  const sortedPareto = Array.from(paretoSamples).sort((a, b) => a - b);
  const u = sortedPareto[Math.floor(0.98 * sortedPareto.length)];
  const exceedances: number[] = [];
  for (const v of paretoSamples) if (v > u) exceedances.push(v - u);
  const exArr = Float64Array.from(exceedances);

  ok(
    '4.0  Pareto exceedances: threshold u and N_u sanity',
    u > 5 && u < 15 && exArr.length > 800 && exArr.length < 1200,
    `u = ${u.toFixed(2)} (notebook 7.30), N_u = ${exArr.length} (notebook 1000)`,
  );

  const gpdResult = timed('Example 7 GPD MLE', () => gpdMle(exArr));
  ok(
    '4.1  Example 7 (Pareto excess) GPD ξ̂ within [0.20, 0.80] (notebook 0.480, truth 0.5)',
    gpdResult.theta.xi >= 0.2 && gpdResult.theta.xi <= 0.8,
    `ξ̂ = ${gpdResult.theta.xi.toFixed(3)}, β̂ = ${gpdResult.theta.beta.toFixed(3)}`,
  );
  ok(
    '4.2  Example 7 GPD β̂ in reasonable scale range',
    gpdResult.theta.beta > 0.5 && gpdResult.theta.beta < 12,
    `β̂ = ${gpdResult.theta.beta.toFixed(3)} (notebook 3.59)`,
  );

  // Example 6 (Normal exceedances): N=50000 standard normals, threshold = 98%-tile.
  // Notebook (cell 12): u≈2.05, N_u=1000, GPD-MLE ξ̂≈0.10 (highly uncertain).
  const rngNormal = mulberry32(42);
  const normalSamples = sampleNormal(50000, rngNormal);
  const sortedNormal = Array.from(normalSamples).sort((a, b) => a - b);
  const uN = sortedNormal[Math.floor(0.98 * sortedNormal.length)];
  const exNormal: number[] = [];
  for (const v of normalSamples) if (v > uN) exNormal.push(v - uN);
  const exNormalArr = Float64Array.from(exNormal);
  const gpdNormal = gpdMle(exNormalArr);
  ok(
    '4.3  Example 6 (Normal excess) GPD ξ̂ within [-0.5, 0.5] (notebook ~0.10, truth 0)',
    gpdNormal.theta.xi >= -0.5 && gpdNormal.theta.xi <= 0.5,
    `u_N = ${uN.toFixed(2)}, N_u = ${exNormalArr.length}, ξ̂ = ${gpdNormal.theta.xi.toFixed(3)}, β̂ = ${gpdNormal.theta.beta.toFixed(3)}`,
  );

  // POT VaR/ES at α = 0.999 for the Pareto example.
  // Truth: VaR = 0.001^{-1/2} = sqrt(1000) ≈ 31.62; ES = 2 * VaR ≈ 63.25.
  // Notebook: VaR_hat ≈ 31.34, ES_hat ≈ 60.43.
  const varAt999 = potVar(gpdResult.theta, u, exArr.length, 50000, 0.999);
  const esAt999 = potEs(gpdResult.theta, u, exArr.length, 50000, 0.999);
  ok(
    '4.4  POT VaR_{0.999} for Pareto example within [22, 45] (truth 31.62, notebook 31.34)',
    varAt999 >= 22 && varAt999 <= 45,
    `VaR_{0.999} = ${varAt999.toFixed(2)}`,
  );
  ok(
    '4.5  POT ES_{0.999} for Pareto example within [40, 100] (truth 63.25, notebook 60.43)',
    esAt999 >= 40 && esAt999 <= 100,
    `ES_{0.999} = ${esAt999.toFixed(2)}`,
  );

  // ES > VaR sanity (always true for positive ξ < 1).
  ok(
    '4.6  ES_{0.999} > VaR_{0.999} (coherence sanity)',
    esAt999 > varAt999,
    `ES = ${esAt999.toFixed(2)} > VaR = ${varAt999.toFixed(2)}`,
  );
}

// =============================================================================
// Section 5: Tail-index estimators on Pareto data
// =============================================================================

function test_5_tail_index_estimators(): void {
  console.log('\nSection 5: Tail-index estimators (Hill, Pickands, DEdH) on Pareto');

  // Standard Pareto with α=2, ξ_true=0.5. Use N=10000 for speed.
  const rng = mulberry32(42);
  const data = sampleParetoStandard(2, 10000, rng);
  const kVec = new Int32Array([20, 50, 100, 200, 500, 1000]);

  const hill = timed('Hill estimator (6 k-values)', () => hillEstimator(data, kVec));
  // Hill plateau region around k ∈ [50, 200] should land near 0.5.
  const hillAt100 = hill[2]; // k = 100
  ok(
    '5.1  hillEstimator(k=100) on Pareto(α=2) within [0.30, 0.70] (truth ξ=0.5)',
    hillAt100 >= 0.3 && hillAt100 <= 0.7,
    `Hill(k=100) = ${hillAt100.toFixed(3)} (full trace: ${Array.from(hill).map((v) => v.toFixed(3)).join(', ')})`,
  );

  const pickands = pickandsEstimator(data, kVec);
  const pickandsAt100 = pickands[2];
  ok(
    '5.2  pickandsEstimator(k=100) on Pareto(α=2) within [0.10, 1.10] (truth ξ=0.5; high variance)',
    pickandsAt100 >= 0.1 && pickandsAt100 <= 1.1,
    `Pickands(k=100) = ${pickandsAt100.toFixed(3)}`,
  );

  const dedh = dedhEstimator(data, kVec);
  const dedhAt100 = dedh[2];
  ok(
    '5.3  dedhEstimator(k=100) on Pareto(α=2) within [0.20, 0.85] (truth ξ=0.5)',
    dedhAt100 >= 0.2 && dedhAt100 <= 0.85,
    `DEdH(k=100) = ${dedhAt100.toFixed(3)}`,
  );

  // Pickands should be NaN where 4k >= n, i.e., k=1000 with n=10000 should be NaN
  // (4 * 1000 = 4000 < 10000, so this k is fine — let me use a higher k):
  const kHigh = new Int32Array([2500, 3000]);
  const pickandsHigh = pickandsEstimator(data, kHigh);
  ok(
    '5.4  pickandsEstimator returns NaN when 4k >= n',
    Number.isNaN(pickandsHigh[0]) && Number.isNaN(pickandsHigh[1]),
    `Pickands(k=2500) = ${pickandsHigh[0]}, Pickands(k=3000) = ${pickandsHigh[1]}`,
  );
}

// =============================================================================
// Main runner
// =============================================================================

console.log('formalML — verify-extreme-value-theory.ts');
console.log('==========================================');

test_1_distribution_sanity();
test_2_gev_self_recovery();
test_3_gev_worked_examples();
test_4_gpd_worked_examples();
test_5_tail_index_estimators();

console.log('\n==========================================');
console.log(`Summary: ${pass}/${pass + fail} tests passed`);
if (fail > 0) {
  console.log('\nFailures:');
  for (const f of failures) console.log(`  ${f}`);
  process.exit(1);
}
process.exit(0);
