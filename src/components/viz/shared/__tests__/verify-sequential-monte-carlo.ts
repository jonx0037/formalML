// =============================================================================
// verify-sequential-monte-carlo.ts
//
// Numerical regression tests for src/components/viz/shared/sequential-monte-carlo.ts.
// Each assertion reproduces a numerical identity from the verified notebook
//   notebooks/sequential-monte-carlo/01_sequential_monte_carlo.ipynb
// or a closed-form quantity stated in the brief (§§2-9), and asserts the TS
// port satisfies the identity within notebook-derived tolerances.
//
// Tolerances widen modestly to absorb PCG64 ↔ mulberry32 RNG drift; the math
// identities themselves are RNG-independent.
//
// Run with: pnpm verify:sequential-monte-carlo
// Exits non-zero if any assertion fails.
//
// Brief: docs/plans/formalml-sequential-monte-carlo-handoff-brief.md
// =============================================================================

import {
  // log-space helpers
  logSumExp,
  normalizeLogW,
  logSpaceESS,
  logZFromLogW,
  // resampling primitives
  searchSorted,
  multinomialResample,
  systematicResample,
  stratifiedResample,
  residualResample,
  // reweight + adaptive schedule
  geometricPathIncrementalLogWeight,
  projectedEss,
  findNextBeta,
  // cloud statistics
  cloudMean1D,
  cloudStd1D,
  cloudMean,
  cloudCov,
  // IMH cloud-fit
  fitImhProposal,
  mvnLogPdfChol,
  // target densities
  gaussianLogPdf,
  isotropicGaussianLogPdf,
  bananaLogPdf,
  gmmLogLikOne,
  // SMC harness
  smcGaussianBridge,
  // SV particle filter
  simulateSV,
  svEmissionLogPdf,
  bootstrapParticleFilter,
  // PRNG / constants
  mulberry32,
  SMC_SEED,
} from '../sequential-monte-carlo';

// -----------------------------------------------------------------------------
// Assertion helpers (match verify-riemann-hmc.ts style)
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

function within(name: string, observed: number, lo: number, hi: number, label: string): void {
  ok(
    name,
    observed >= lo && observed <= hi,
    `${label} observed=${observed.toFixed(6)} in [${lo}, ${hi}]`,
  );
}

function approx(name: string, actual: number, expected: number, tol: number, label: string): void {
  ok(
    name,
    Math.abs(actual - expected) <= tol,
    `${label} actual=${actual.toFixed(8)} expected=${expected.toFixed(8)} |Δ|≤${tol}`,
  );
}

// =============================================================================
// [1] Log-space helpers — deterministic identities, 1e-10 tolerance
// =============================================================================

console.log('\n[1] Log-space helpers (deterministic algebraic identities)');
{
  // Numeric stability: logSumExp([0, 0, 0]) = log 3.
  approx('1.1  logSumExp([0,0,0]) = log 3', logSumExp([0, 0, 0]), Math.log(3), 1e-12, 'log 3');

  // Identity: logSumExp(x + c) = c + logSumExp(x).
  const x = [0.1, -0.5, 2.3, 1.0];
  const c = 7.0;
  const shifted = x.map((v) => v + c);
  approx(
    '1.2  logSumExp shift-equivariance',
    logSumExp(shifted),
    c + logSumExp(x),
    1e-10,
    'shifted - unshifted',
  );

  // Extreme-magnitude stability: logSumExp([1000, 1001]) = 1001 + log(1 + e^{-1}).
  approx(
    '1.3  logSumExp([1000, 1001]) stable at large magnitude',
    logSumExp([1000, 1001]),
    1001 + Math.log(1 + Math.exp(-1)),
    1e-9,
    'large-x stability',
  );

  // Empty array returns -Infinity.
  ok('1.4  logSumExp([]) = -Infinity', logSumExp([]) === -Infinity, 'empty input');

  // normalizeLogW produces weights summing to 1 in exp-space.
  const lw = [0.0, 1.0, 2.0, 3.0];
  const lwNorm = normalizeLogW(lw);
  const wSum = lwNorm.reduce((s, v) => s + Math.exp(v), 0);
  approx('1.5  Σ exp(normalizeLogW) = 1', wSum, 1.0, 1e-12, 'sum of normalized weights');

  // logSpaceESS matches 1 / Σ w_i^2 (linear-space formula) on hand fixture.
  const lwSimple = [Math.log(0.4), Math.log(0.3), Math.log(0.2), Math.log(0.1)];
  const w = lwSimple.map(Math.exp);
  const directEss = 1 / w.reduce((s, v) => s + v * v, 0);
  approx('1.6  logSpaceESS matches 1/Σw²', logSpaceESS(lwSimple), directEss, 1e-9, 'ESS identity');

  // Edge: uniform log-weights → ESS = N.
  const N = 100;
  const lwUniform = new Array<number>(N).fill(0);
  approx(
    '1.7  ESS at uniform weights equals N',
    logSpaceESS(lwUniform),
    N,
    1e-9,
    `N=${N} uniform`,
  );

  // Edge: degenerate (one weight all mass) → ESS ≈ 1.
  const lwDegen = [0, -1e6, -1e6, -1e6];
  approx('1.8  ESS at degenerate weights ≈ 1', logSpaceESS(lwDegen), 1, 1e-6, 'degenerate ESS');

  // log Ẑ = logSumExp(log_w) - log N. Spot check.
  const lwZ = [Math.log(2), Math.log(2), Math.log(2), Math.log(2)];
  approx(
    '1.9  log Ẑ identity',
    logZFromLogW(lwZ),
    Math.log(2),
    1e-12,
    'logSumExp - log N',
  );
}

// =============================================================================
// [2] searchSorted + resampling primitives — deterministic given u
// =============================================================================

console.log('\n[2] searchSorted and resampling (deterministic for fixed u)');
{
  // searchSorted on a small cumulative array.
  // cum = [0.1, 0.4, 0.7, 1.0]; queries [0.05, 0.15, 0.5, 0.95]
  // → [0, 1, 2, 3]
  const cum = [0.1, 0.4, 0.7, 1.0];
  const q = [0.05, 0.15, 0.5, 0.95];
  const idx = searchSorted(cum, q);
  ok(
    '2.1  searchSorted([.1,.4,.7,1.0], [.05,.15,.5,.95]) = [0,1,2,3]',
    idx.length === 4 && idx[0] === 0 && idx[1] === 1 && idx[2] === 2 && idx[3] === 3,
    `got [${idx.join(',')}]`,
  );

  // Systematic resampling with a forced u0 — bypass RNG by passing constant.
  // Weights = [0.1, 0.2, 0.3, 0.4]; cum = [0.1, 0.3, 0.6, 1.0]
  // With N = 4 and u0 = 0.123: queries = [(0.123+k)/4] = [0.03075, 0.28075, 0.53075, 0.78075]
  // → indices [0, 1, 2, 3].
  const w = [0.1, 0.2, 0.3, 0.4];
  const forcedRng = (() => {
    let called = false;
    return () => {
      if (called) throw new Error('systematic should call rng exactly once');
      called = true;
      return 0.123;
    };
  })();
  const sysIdx = systematicResample(w, forcedRng, 4);
  ok(
    '2.2  systematicResample(w, u0=0.123) deterministic indices',
    sysIdx.length === 4 &&
      sysIdx[0] === 0 &&
      sysIdx[1] === 1 &&
      sysIdx[2] === 2 &&
      sysIdx[3] === 3,
    `got [${sysIdx.join(',')}], expected [0,1,2,3]`,
  );

  // Systematic resampling calls rng exactly once (vs stratified which calls N times).
  let sysRngCalls = 0;
  const sysCounter = () => {
    sysRngCalls++;
    return 0.5;
  };
  systematicResample(w, sysCounter, 100);
  ok('2.3  systematic uses 1 rng() call for any N', sysRngCalls === 1, `rng calls = ${sysRngCalls}`);

  // Multinomial / stratified / residual produce N indices.
  const rng2 = mulberry32(SMC_SEED + 42);
  ok('2.4a multinomialResample returns N indices', multinomialResample(w, rng2, 8).length === 8, 'len');
  ok('2.4b stratifiedResample returns N indices', stratifiedResample(w, rng2, 8).length === 8, 'len');
  ok('2.4c residualResample returns N indices', residualResample(w, rng2, 8).length === 8, 'len');

  // Offspring unbiasedness (Proposition 2): average count per particle ≈ N w_i over M trials.
  // Skewed weights w ∝ exp(2.5 z), test cloud at N = 200, M = 2000.
  const Np = 200;
  const M = 2000;
  const skewedRng = mulberry32(SMC_SEED + 100);
  const lwSkew = new Array<number>(Np);
  for (let i = 0; i < Np; i++) lwSkew[i] = 2.5 * (skewedRng() - 0.5) * 2; // synthetic z
  const wSkew = normalizeLogW(lwSkew).map(Math.exp);
  const expectedCount = wSkew.map((wi) => Np * wi);
  const realizedCount = new Array<number>(Np).fill(0);
  const rngOff = mulberry32(SMC_SEED + 200);
  for (let m = 0; m < M; m++) {
    const idx = systematicResample(wSkew, rngOff, Np);
    for (let i = 0; i < idx.length; i++) realizedCount[idx[i]]++;
  }
  // Compute relative L1 error of mean count vs expected.
  let l1 = 0;
  let expL1 = 0;
  for (let i = 0; i < Np; i++) {
    l1 += Math.abs(realizedCount[i] / M - expectedCount[i]);
    expL1 += expectedCount[i];
  }
  within(
    '2.5  systematic offspring unbiasedness L1/N',
    l1 / expL1,
    0,
    0.04,
    'mean offspring count vs N·w_i, normalized L1',
  );

  // Systematic variance ≤ multinomial variance on the same skewed weights
  // (Theorem 2 ordering).
  const f = wSkew.map((_, i) => Math.sin(i * 0.1)); // bounded test function on particle indices
  const samplesSys: number[] = [];
  const samplesMult: number[] = [];
  const rngSys = mulberry32(SMC_SEED + 300);
  const rngMult = mulberry32(SMC_SEED + 400);
  for (let m = 0; m < 1000; m++) {
    const idxS = systematicResample(wSkew, rngSys, Np);
    const idxM = multinomialResample(wSkew, rngMult, Np);
    let sS = 0;
    let sM = 0;
    for (let i = 0; i < Np; i++) {
      sS += f[idxS[i]];
      sM += f[idxM[i]];
    }
    samplesSys.push(sS / Np);
    samplesMult.push(sM / Np);
  }
  const variance = (arr: number[]) => {
    const mean = arr.reduce((s, v) => s + v, 0) / arr.length;
    return arr.reduce((s, v) => s + (v - mean) * (v - mean), 0) / (arr.length - 1);
  };
  const vSys = variance(samplesSys);
  const vMult = variance(samplesMult);
  ok(
    '2.6  Var(systematic) ≤ Var(multinomial) (Theorem 2)',
    vSys <= vMult,
    `Var_sys=${vSys.toExponential(3)} ≤ Var_mult=${vMult.toExponential(3)}`,
  );
}

// =============================================================================
// [3] Geometric-path reweight (Proposition 3)
// =============================================================================

console.log('\n[3] Geometric-path incremental log-weight (Proposition 3)');
{
  // Closed-form: log α_t = (β_{t+1} - β_t) · log p(y | θ). Prior cancels.
  const logLik = [-1.0, 2.5, 0.3, -0.8, 5.0];
  const deltaBeta = 0.15;
  const w = geometricPathIncrementalLogWeight(deltaBeta, logLik);
  let maxErr = 0;
  for (let i = 0; i < w.length; i++) {
    maxErr = Math.max(maxErr, Math.abs(w[i] - deltaBeta * logLik[i]));
  }
  approx('3.1  geometric reweight = Δβ · logLik element-wise', maxErr, 0, 1e-14, 'max |error|');

  // Sanity: Δβ = 0 gives zero weight increment.
  const zeroInc = geometricPathIncrementalLogWeight(0, logLik);
  const maxZeroAbs = Math.max(...zeroInc.map(Math.abs));
  approx('3.2  Δβ = 0 → zero increment', maxZeroAbs, 0, 1e-14, 'max |inc| at Δβ=0');
}

// =============================================================================
// [4] Adaptive temperature bisection (§9.4)
// =============================================================================

console.log('\n[4] Adaptive temperature bisection (Del Moral-Doucet-Jasra 2012)');
{
  // Construct a cloud with controllable log-likelihood spread. With log L
  // standard-normal at N = 500, the projected ESS at β = 1 (starting from
  // β = 0) collapses well below target ESS = 0.9 N = 450, so bisection
  // returns an interior β.
  const N = 500;
  const rngLogLik = mulberry32(SMC_SEED + 999);
  const logLik = new Array<number>(N);
  // approximate standard normals via Box-Muller-on-pairs
  for (let i = 0; i < N; i += 2) {
    const u1 = Math.max(rngLogLik(), 1e-12);
    const u2 = rngLogLik();
    const r = Math.sqrt(-2 * Math.log(u1));
    const a = r * Math.cos(2 * Math.PI * u2);
    const b = r * Math.sin(2 * Math.PI * u2);
    logLik[i] = a;
    if (i + 1 < N) logLik[i + 1] = b;
  }
  // amplify variance so projection at β=1 collapses ESS
  for (let i = 0; i < N; i++) logLik[i] *= 10;
  const logW0 = new Array<number>(N).fill(0);
  const targetEss = 0.9 * N;
  const result = findNextBeta(0, logW0, logLik, targetEss, 1e-4, 60);
  ok('4.1  findNextBeta returns 0 < β ≤ 1', result.beta > 0 && result.beta <= 1, `β = ${result.beta.toFixed(6)}`);
  // The realized ESS at the returned β should be within tol·N of targetEss
  // (or at β = 1 if ESS doesn't collapse, in which case ess ≥ targetEss).
  if (result.beta < 1) {
    within('4.2  realized ESS hits target within tolerance', result.ess, targetEss - 5, targetEss + 5, 'ESS near target');
  } else {
    ok('4.2  β = 1 means ESS ≥ targetEss', result.ess >= targetEss - 1, `ESS = ${result.ess.toFixed(2)} at β=1`);
  }
  // Convergence within 60 iterations.
  ok('4.3  bisection converges within 60 iters', result.iters <= 60, `iters = ${result.iters}`);

  // Monotonicity: projectedEss is decreasing in β (high target curvature case).
  const essAtBeta = (b: number) => projectedEss(b, 0, logW0, logLik);
  const betas = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0];
  let strictlyDecr = true;
  for (let i = 1; i < betas.length; i++) {
    if (essAtBeta(betas[i]) > essAtBeta(betas[i - 1]) + 1e-9) {
      strictlyDecr = false;
      break;
    }
  }
  ok('4.4  projectedEss is monotone non-increasing in β', strictlyDecr, 'monotone check');
}

// =============================================================================
// [5] Cloud statistics — closed-form weighted moments
// =============================================================================

console.log('\n[5] Cloud statistics (closed-form weighted moments)');
{
  // 1D: uniform weights → arithmetic mean.
  const theta1D = [1, 2, 3, 4, 5];
  const wU = new Array<number>(5).fill(0.2);
  approx('5.1  cloudMean1D uniform → arithmetic mean', cloudMean1D(theta1D, wU), 3, 1e-12, 'mean');
  // Population std of [1..5] with uniform weights = sqrt(2).
  approx('5.2  cloudStd1D uniform → population std', cloudStd1D(theta1D, wU), Math.sqrt(2), 1e-10, 'std');

  // 2D: weighted mean and covariance, hand-computed.
  // theta = [(0,0), (2,2)], w = [0.5, 0.5] → mean (1,1), cov = [[1,1],[1,1]].
  const theta2D = [
    [0, 0],
    [2, 2],
  ];
  const w2 = [0.5, 0.5];
  const mu = cloudMean(theta2D, w2);
  ok('5.3  cloudMean 2D = (1, 1)', mu[0] === 1 && mu[1] === 1, `μ = (${mu[0]}, ${mu[1]})`);
  const cov = cloudCov(theta2D, w2);
  approx('5.4a cov[0][0] = 1', cov[0][0], 1, 1e-12, 'cov(x,x)');
  approx('5.4b cov[0][1] = 1', cov[0][1], 1, 1e-12, 'cov(x,y)');
  approx('5.4c cov[1][0] = 1 (symmetric)', cov[1][0], cov[0][1], 1e-12, 'symmetry');
}

// =============================================================================
// [6] IMH cloud-fit + multivariate normal log-pdf
// =============================================================================

console.log('\n[6] IMH cloud-fit and mvnLogPdfChol');
{
  // Generate a 2D cloud from N(mu, Σ) and verify fit recovers it approximately.
  const trueMu = [3, -1];
  const trueSigmaXX = 2.0;
  const trueSigmaYY = 0.5;
  const N = 5000;
  const rngFit = mulberry32(SMC_SEED + 555);
  const theta: number[][] = new Array(N);
  for (let i = 0; i < N; i++) {
    // Box-Muller pair
    const u1 = Math.max(rngFit(), 1e-12);
    const u2 = rngFit();
    const r = Math.sqrt(-2 * Math.log(u1));
    const z1 = r * Math.cos(2 * Math.PI * u2);
    const z2 = r * Math.sin(2 * Math.PI * u2);
    theta[i] = [
      trueMu[0] + Math.sqrt(trueSigmaXX) * z1,
      trueMu[1] + Math.sqrt(trueSigmaYY) * z2,
    ];
  }
  const wN = new Array<number>(N).fill(1 / N);
  const fit = fitImhProposal(theta, wN, 1e-8);
  approx('6.1  fitImhProposal recovers mu[0]', fit.mu[0], trueMu[0], 0.06, 'μ_x');
  approx('6.2  fitImhProposal recovers mu[1]', fit.mu[1], trueMu[1], 0.06, 'μ_y');
  // Reconstruct Σ from L L^T (lower triangular).
  const L = fit.L;
  const sxx = L[0][0] * L[0][0];
  const syy = L[1][0] * L[1][0] + L[1][1] * L[1][1];
  approx('6.3  Σ_xx ≈ trueSigmaXX', sxx, trueSigmaXX, 0.1, 'cov[0][0]');
  approx('6.4  Σ_yy ≈ trueSigmaYY', syy, trueSigmaYY, 0.05, 'cov[1][1]');

  // mvnLogPdfChol matches scipy formula for diagonal Σ.
  // x = (3, -1) at trueMu → quadratic = 0, logDet = log(SigmaXX * SigmaYY).
  const xAtMu = [3, -1];
  const pdfAtMu = mvnLogPdfChol(xAtMu, trueMu, fit.L, fit.logDet);
  // Sanity: at the fitted mean, the log-pdf should be high (within 0.1 of
  // -log(2π) - 0.5 log det Σ at the empirical (μ̂, Σ̂)).
  const empMu = fit.mu;
  const empLogDet = fit.logDet;
  const dx = [xAtMu[0] - empMu[0], xAtMu[1] - empMu[1]];
  const empQuadApprox = 0.5 * (
    (dx[0] * dx[0]) / sxx + (dx[1] * dx[1]) / syy
  );
  const expectedPdf = -Math.log(2 * Math.PI) - 0.5 * empLogDet - empQuadApprox;
  approx(
    '6.5  mvnLogPdfChol matches isotropic formula at fitted mean',
    pdfAtMu,
    expectedPdf,
    0.1,
    'pdf consistency',
  );
}

// =============================================================================
// [7] Target log-densities — spot-check known values
// =============================================================================

console.log('\n[7] Target log-density spot checks');
{
  // N(0, 1) at 0 = -0.5 log(2π) ≈ -0.9189.
  approx(
    '7.1  gaussianLogPdf(0, 0, 1) = -0.5 log(2π)',
    gaussianLogPdf(0, 0, 1),
    -0.5 * Math.log(2 * Math.PI),
    1e-12,
    'log φ(0)',
  );

  // Banana at θ = (0, 0): log π = -log(2π).
  approx(
    '7.2  bananaLogPdf(0, 0) = -log(2π)',
    bananaLogPdf([0, 0]),
    -Math.log(2 * Math.PI),
    1e-12,
    'log banana at origin',
  );

  // Banana at θ = (1, 1): θ_2 - θ_1^2 = 0, so log π = -1/2 - log(2π).
  approx(
    '7.3  bananaLogPdf(1, 1) = -0.5 - log(2π)',
    bananaLogPdf([1, 1]),
    -0.5 - Math.log(2 * Math.PI),
    1e-12,
    'log banana on ridge',
  );

  // isotropicGaussianLogPdf at origin matches direct calculation.
  const lvl = isotropicGaussianLogPdf([0, 0, 0], [0, 0, 0], 2.0);
  approx(
    '7.4  isotropic 3D Gaussian σ=2 at origin',
    lvl,
    -3 * Math.log(2) - 1.5 * Math.log(2 * Math.PI),
    1e-12,
    'log φ_σ=2(0)',
  );

  // GMM per-obs log-lik: p = 0.5, mu1 = -2, mu2 = 2, y = 0 → balanced mixture.
  // exp evaluates to log(0.5 · φ_(0-(-2)) + 0.5 · φ_(0-2)) where φ_z = N(z; 0, 1).
  // Both terms equal log 0.5 - 2 - 0.5 log(2π).
  const expectedGmm = Math.log(
    0.5 * Math.exp(gaussianLogPdf(0, -2, 1)) + 0.5 * Math.exp(gaussianLogPdf(0, 2, 1)),
  );
  approx(
    '7.5  gmmLogLikOne(y=0, μ=±2, p=.5) matches direct',
    gmmLogLikOne(0, -2, 2, 0.5),
    expectedGmm,
    1e-10,
    'GMM ll',
  );

  // SV emission log-density: y_t ~ N(0, exp(x_t)). At x = 0, y = 0 → N(0, 1) at 0.
  approx(
    '7.6  svEmissionLogPdf(0, 0) = log N(0; 0, 1)',
    svEmissionLogPdf(0, 0),
    -0.5 * Math.log(2 * Math.PI),
    1e-12,
    'SV ll at log-vol 0',
  );
}

// =============================================================================
// [8] End-to-end SMC on Gaussian bridge (§3.4)
//
// The notebook at N = 200, T = 10 prints log Ẑ ≈ +0.58 with mulberry32-free
// PCG64 RNG. Our TS port at N = 2000 with mulberry32 should land |log Ẑ| ≤ 0.4
// with terminal mean within 0.2 and std within 0.2 of (5.0, 1.0).
// =============================================================================

console.log('\n[8] End-to-end SMC on Gaussian bridge — §3.4 cell 7 reference');
{
  const result = smcGaussianBridge({
    N: 2000,
    T: 10,
    mu0: 0,
    sigma0: 2,
    muT: 5,
    sigmaT: 1,
    resampleThreshold: 0.5,
    rwmStep: 0.5,
    rwmSweeps: 5,
    scheme: 'systematic',
    seed: SMC_SEED + 3,
  });
  within('8.1  log Ẑ_T near 0 (truth = 0)', result.logZHat, -0.5, 0.5, 'log Ẑ');
  within('8.2  terminal mean near 5 (truth = 5)', result.terminalMean, 4.7, 5.3, 'terminal μ');
  within('8.3  terminal std near 1 (truth = 1)', result.terminalStd, 0.85, 1.15, 'terminal σ');
  // ESS should never collapse to 1 — the SMC skeleton with resampling keeps it healthy.
  const minEss = Math.min(...result.essHistory.slice(1));
  ok('8.4  min ESS > 100 across path', minEss > 100, `min ESS = ${minEss.toFixed(1)}`);
  // log-Ẑ history is monotone in expectation but path-stochastic;
  // the cumulative path should end near 0.
  ok(
    '8.5  log Ẑ history length matches T+1',
    result.logZHistory.length === 11,
    `len = ${result.logZHistory.length}`,
  );
}

// =============================================================================
// [9] SV particle filter (§6.4)
//
// Bootstrap PF on a short SV simulation. Check that the filter posterior
// tracks the true latent path (RMSE small) and that ESS stays healthy.
// =============================================================================

console.log('\n[9] Bootstrap particle filter on stochastic-volatility (§6.4)');
{
  const data = simulateSV(60, 0.95, 0.2, SMC_SEED + 6);
  const filt = bootstrapParticleFilter(data, 500, 0.95, 0.2, 0.5, SMC_SEED + 7);

  // RMSE of filtered mean vs true latent state.
  let sse = 0;
  for (let t = 0; t < filt.T; t++) {
    const d = filt.mean[t] - data.xTrue[t];
    sse += d * d;
  }
  const rmse = Math.sqrt(sse / filt.T);
  // SV latent state has stationary std ≈ sigma / sqrt(1 - phi^2) = 0.2/sqrt(0.0975) ≈ 0.64.
  // A useless filter (mean ≡ 0) would have RMSE ≈ 0.64. We expect filter RMSE < 0.5.
  within('9.1  bootstrap PF RMSE < 0.5 (uninformative baseline ≈ 0.64)', rmse, 0, 0.5, 'RMSE');

  // ESS should not collapse for too long. Average ESS should be reasonably high.
  const avgEss = filt.ess.reduce((s, v) => s + v, 0) / filt.ess.length;
  within('9.2  average ESS > 100 (of N = 500)', avgEss, 100, 500, 'avg ESS');

  // log p(y_{1:T}) is finite.
  ok('9.3  log Ẑ is finite', isFinite(filt.logZ), `log Ẑ = ${filt.logZ.toFixed(2)}`);
}

// =============================================================================
// Summary
// =============================================================================

console.log(`\n${pass} passed, ${fail} failed`);
if (fail > 0) {
  console.log('\nFailures:');
  for (const f of failures) console.log(`  - ${f}`);
}

process.exit(fail === 0 ? 0 : 1);
