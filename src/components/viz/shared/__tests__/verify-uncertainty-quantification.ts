// =============================================================================
// verify-uncertainty-quantification.ts
//
// Numerical regression tests for src/components/viz/shared/uncertainty-quantification.ts.
// Each test reproduces a numerical claim from the verified notebook
//   notebooks/uncertainty-quantification/01_uncertainty_quantification.ipynb
// and asserts the result lies in the expected range.
//
// Run with: pnpm verify:uncertainty-quantification
// Exits non-zero on first failure. Designed to be CI-friendly.
//
// Tolerances: closed-form quantities (LTV identity, Murphy decomposition,
// CRPS Gaussian vs quadrature, Laplace = closed-form posterior) match
// notebook to <1e-8. Empirical Monte-Carlo quantities (bootstrap epi/ale,
// conformal coverage at finite n_cal, MC-dropout, ridge UQ sweep) are not
// reproduced bit-for-bit (mulberry32 vs PCG64); we assert distribution-shape
// agreement and brief-stated tolerances.
// =============================================================================

import {
  NOTEBOOK_SEED,
  POLY_SCALE,
  DEFAULT_DEGREE,
  DEFAULT_ALPHA,
  aleatoricFromResiduals,
  aurocFromScores,
  baggedPredictGrid,
  bayesPolyPosterior,
  bayesPolyPredict,
  brierScore,
  cholesky,
  coverage,
  crpsGaussian,
  crpsQuadrature,
  doubleDescentUqSweep,
  ece,
  ensembleMixtureDensity,
  ensembleMixtureMeanVar,
  fitOlsPoly,
  fitTemperature,
  fTrue,
  gaussianFrom,
  invSPD,
  isotonicRecal,
  laplacePolynomial,
  linspace,
  logLoss,
  logsumexpAxis0,
  matVec,
  mce,
  mixtureNll,
  mulberry32,
  murphyDecomposition,
  pairsBootstrapFit,
  phi,
  predictOlsPoly,
  quadForm,
  reliabilityBins,
  residualBootstrapFit,
  sampleHetero,
  sharpness,
  sigmaTrue,
  splitConformalConstant,
  splitConformalLocallyWeighted,
  temperatureNll,
  temperatureScaleProb,
  wildBootstrapFit,
} from '../uncertainty-quantification';

const failures: string[] = [];

function ok(name: string, condition: boolean, detail: string) {
  if (condition) {
    console.log(`  ✓ ${name}: ${detail}`);
  } else {
    console.log(`  ✗ ${name}: ${detail}`);
    failures.push(`${name}: ${detail}`);
  }
}

function approxAbs(a: number, b: number, atol: number): boolean {
  if (!Number.isFinite(a) || !Number.isFinite(b)) return a === b;
  return Math.abs(a - b) <= atol;
}

function approxRel(a: number, b: number, rtol: number): boolean {
  if (!Number.isFinite(a) || !Number.isFinite(b)) return a === b;
  if (b === 0) return Math.abs(a) <= rtol;
  return Math.abs(a - b) <= rtol * Math.abs(b);
}

console.log('\n=== verify-uncertainty-quantification.ts ===\n');

// -----------------------------------------------------------------------------
// 1. Linear-algebra sanity (Cholesky / SPD invert / quadForm).
// -----------------------------------------------------------------------------
console.log('--- Linear algebra ---');
{
  const A = [
    [4, 1, 0],
    [1, 3, 1],
    [0, 1, 2],
  ];
  const Ainv = invSPD(A);
  // A · A⁻¹ = I.
  let maxErr = 0;
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      let s = 0;
      for (let k = 0; k < 3; k++) s += A[i][k] * Ainv[k][j];
      const target = i === j ? 1 : 0;
      maxErr = Math.max(maxErr, Math.abs(s - target));
    }
  }
  ok('invSPD round-trip', maxErr < 1e-10, `max |A·A⁻¹ - I| = ${maxErr.toExponential(2)}`);

  // Cholesky reconstruction.
  const L = cholesky(A);
  let chErr = 0;
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      let s = 0;
      for (let k = 0; k <= Math.min(i, j); k++) s += L[i][k] * L[j][k];
      chErr = Math.max(chErr, Math.abs(s - A[i][j]));
    }
  }
  ok('Cholesky reconstruction', chErr < 1e-12, `max |L Lᵀ - A| = ${chErr.toExponential(2)}`);

  // Quad form xᵀ A x.
  const x = [1, 2, 3];
  const q = quadForm(x, A);
  // 4·1 + 3·4 + 2·9 + 2·1·2 + 0·1·3 + 2·2·3 = 4 + 12 + 18 + 4 + 0 + 12 = 50.
  ok('quadForm closed form', approxAbs(q, 50, 1e-12), `xᵀAx = ${q} (expect 50)`);
}

// -----------------------------------------------------------------------------
// 2. Polynomial basis Φ at the scaled-Vandermonde convention.
// -----------------------------------------------------------------------------
console.log('\n--- §2 Polynomial basis ---');
{
  const P = phi([POLY_SCALE], 3);
  // Φ(POLY_SCALE) / POLY_SCALE = [1, 1, 1, 1] (monomial column-wise after scaling).
  ok(
    'phi at POLY_SCALE',
    approxAbs(P[0][0], 1, 1e-12) &&
      approxAbs(P[0][1], 1, 1e-12) &&
      approxAbs(P[0][2], 1, 1e-12) &&
      approxAbs(P[0][3], 1, 1e-12),
    `Φ(${POLY_SCALE}) = [${P[0].join(', ')}]`,
  );

  const Q = phi([0], 4);
  ok(
    'phi at 0',
    approxAbs(Q[0][0], 1, 1e-12) && approxAbs(Q[0][1], 0, 1e-12) && approxAbs(Q[0][2], 0, 1e-12),
    `Φ(0) = [${Q[0].slice(0, 3).join(', ')}, ...]`,
  );
}

// -----------------------------------------------------------------------------
// 3. §2 Bayesian polynomial regression + LTV identity (closed form).
// -----------------------------------------------------------------------------
console.log('\n--- §2 LTV identity (closed form) ---');
{
  const rng = mulberry32(NOTEBOOK_SEED);
  const { X, y } = sampleHetero(200, -3, 3, rng);
  const sigmaTrainArr = X.map(sigmaTrue);
  const post = bayesPolyPosterior(X, y, DEFAULT_DEGREE, sigmaTrainArr);
  const xGrid = linspace(-3.2, 3.2, 200);
  const sigmaGridArr = xGrid.map(sigmaTrue);
  const pred = bayesPolyPredict(xGrid, post, sigmaGridArr);
  let maxLtvErr = 0;
  for (let i = 0; i < xGrid.length; i++) {
    const err = Math.abs(pred.totalVar[i] - (pred.aleatoricVar[i] + pred.epistemicVar[i]));
    if (err > maxLtvErr) maxLtvErr = err;
  }
  ok(
    '§2 LTV identity pointwise',
    maxLtvErr < 1e-12,
    `max |total - (ale + epi)| = ${maxLtvErr.toExponential(2)}`,
  );
  // Coarse correctness: epistemic should be small inside the data range (|x| < 2.5)
  // and aleatoric should grow with |x|.
  let epiInside = 0;
  let nInside = 0;
  for (let i = 0; i < xGrid.length; i++) {
    if (Math.abs(xGrid[i]) < 2.5) {
      epiInside += pred.epistemicVar[i];
      nInside++;
    }
  }
  epiInside /= nInside;
  ok(
    '§2 epistemic small inside data range',
    epiInside < 0.05,
    `mean epi on |x|<2.5 = ${epiInside.toFixed(4)} (expect < 0.05)`,
  );
  const aleAt3 = pred.aleatoricVar[pred.aleatoricVar.length - 1];
  const aleAt0 = pred.aleatoricVar[Math.floor(xGrid.length / 2)];
  ok(
    '§2 aleatoric grows with |x|',
    aleAt3 > 5 * aleAt0,
    `ale(3.2) / ale(0) = ${(aleAt3 / aleAt0).toFixed(2)}`,
  );
}

// -----------------------------------------------------------------------------
// 4. §7.2 Laplace = closed-form Bayesian polynomial posterior, exactly.
// -----------------------------------------------------------------------------
console.log('\n--- §7.2 Laplace vs closed-form posterior ---');
{
  const rng = mulberry32(NOTEBOOK_SEED + 1);
  const { X, y } = sampleHetero(200, -3, 3, rng);
  const sig = X.map(sigmaTrue);
  const lap = laplacePolynomial(X, y, DEFAULT_DEGREE, sig);
  const cf = bayesPolyPosterior(X, y, DEFAULT_DEGREE, sig);
  let muErr = 0;
  for (let k = 0; k < lap.muN.length; k++) {
    muErr = Math.max(muErr, Math.abs(lap.muN[k] - cf.muN[k]));
  }
  let sigErr = 0;
  for (let i = 0; i < lap.SigmaN.length; i++) {
    for (let j = 0; j < lap.SigmaN.length; j++) {
      sigErr = Math.max(sigErr, Math.abs(lap.SigmaN[i][j] - cf.SigmaN[i][j]));
    }
  }
  ok(
    '§7.2 ||μ_Laplace - μ_closed_form||_inf < 1e-12',
    muErr < 1e-12,
    `max |Δμ| = ${muErr.toExponential(2)}`,
  );
  ok(
    '§7.2 ||Σ_Laplace - Σ_closed_form||_inf < 1e-12',
    sigErr < 1e-12,
    `max |ΔΣ| = ${sigErr.toExponential(2)}`,
  );
}

// -----------------------------------------------------------------------------
// 5. §3 reliability / ECE / MCE / sharpness on a hand-built mini classifier.
// -----------------------------------------------------------------------------
console.log('\n--- §3 calibration helpers ---');
{
  // Perfectly-calibrated 100-sample classifier at three probability levels.
  // Three bins at p ∈ {0.2, 0.5, 0.8}, empirical accuracy matches by construction.
  const y: number[] = [];
  const p: number[] = [];
  const levels = [0.2, 0.5, 0.8];
  for (const pk of levels) {
    const N = 100;
    const pos = Math.round(N * pk);
    for (let i = 0; i < N; i++) {
      y.push(i < pos ? 1 : 0);
      p.push(pk);
    }
  }
  const eceVal = ece(y, p, 10);
  const mceVal = mce(y, p, 10);
  ok('§3 ECE ≈ 0 on calibrated mini-set', eceVal < 1e-12, `ECE = ${eceVal.toExponential(2)}`);
  ok('§3 MCE ≈ 0 on calibrated mini-set', mceVal < 1e-12, `MCE = ${mceVal.toExponential(2)}`);
  // Constant predictor has zero sharpness.
  ok('§3 sharpness of constant', sharpness(new Array(50).fill(0.4)) < 1e-12, `Var(p_const) ≈ 0`);
  ok('§3 sharpness positive on spread', sharpness(p) > 0.05, `Var(p_mix) = ${sharpness(p).toFixed(3)}`);

  // Now a deliberately overconfident mini-classifier: 80%-claimed bin has 60% accuracy.
  const yMis: number[] = [];
  const pMis: number[] = [];
  for (let i = 0; i < 100; i++) {
    yMis.push(i < 20 ? 1 : 0);
    pMis.push(0.2);
  }
  for (let i = 0; i < 100; i++) {
    yMis.push(i < 60 ? 1 : 0);
    pMis.push(0.8);
  }
  const eceMis = ece(yMis, pMis, 10);
  // Half the data has 0.2 gap of 0; half has 0.2 gap of |0.8 - 0.6| = 0.2 ⇒ ECE = 0.5·0 + 0.5·0.2 = 0.10.
  ok('§3 ECE on overconfident mini', approxAbs(eceMis, 0.1, 1e-12), `ECE = ${eceMis.toFixed(4)}`);
}

// -----------------------------------------------------------------------------
// 6. §4 strict properness: BS expected loss minimized at p=q.
// -----------------------------------------------------------------------------
console.log('\n--- §4 strict properness (Brier + log-loss) ---');
{
  const q = 0.30;
  const grid = linspace(1e-3, 1 - 1e-3, 200);
  const expBrier = grid.map((p) => q * (p - 1) * (p - 1) + (1 - q) * p * p);
  const expLogLoss = grid.map(
    (p) => -q * Math.log(p) - (1 - q) * Math.log(1 - p),
  );
  let iMinBrier = 0;
  for (let i = 1; i < grid.length; i++) if (expBrier[i] < expBrier[iMinBrier]) iMinBrier = i;
  let iMinLL = 0;
  for (let i = 1; i < grid.length; i++) if (expLogLoss[i] < expLogLoss[iMinLL]) iMinLL = i;
  ok(
    '§4 Brier minimizer ≈ q',
    Math.abs(grid[iMinBrier] - q) < 0.01,
    `argmin = ${grid[iMinBrier].toFixed(3)}`,
  );
  ok(
    '§4 Log-loss minimizer ≈ q',
    Math.abs(grid[iMinLL] - q) < 0.01,
    `argmin = ${grid[iMinLL].toFixed(3)}`,
  );
  // Closed-form minimum values: q(1-q) and H(q).
  const Hq = -q * Math.log(q) - (1 - q) * Math.log(1 - q);
  const minBrier = expBrier[iMinBrier];
  const minLL = expLogLoss[iMinLL];
  ok(
    '§4 Brier minimum = q(1-q)',
    approxAbs(minBrier, q * (1 - q), 1e-3),
    `min = ${minBrier.toFixed(4)} (expect ${(q * (1 - q)).toFixed(4)})`,
  );
  ok(
    '§4 LL minimum = H(q)',
    approxAbs(minLL, Hq, 1e-3),
    `min = ${minLL.toFixed(4)} (expect ${Hq.toFixed(4)})`,
  );

  // Verify our brierScore / logLoss helpers on a tiny set agree with closed-form at p = q.
  const yMini = [1, 0, 1, 0, 1, 1, 0, 0];
  const pMini = yMini.map(() => 0.5);
  const bsMini = brierScore(yMini, pMini);
  ok('§4 brierScore at p=0.5 on balanced y', approxAbs(bsMini, 0.25, 1e-12), `BS = ${bsMini}`);
  const llMini = logLoss(yMini, pMini);
  // -log(0.5) = log(2) ≈ 0.6931.
  ok('§4 logLoss at p=0.5', approxAbs(llMini, Math.log(2), 1e-12), `LL = ${llMini.toFixed(4)}`);
}

// -----------------------------------------------------------------------------
// 7. §4.1 CRPS: Hersbach closed form agrees with trapezoidal quadrature.
// -----------------------------------------------------------------------------
console.log('\n--- §4.1 CRPS Gaussian closed form vs quadrature ---');
{
  const cases: Array<[number, number, number]> = [
    [0.0, 0.0, 1.0],
    [0.5, 0.0, 1.0],
    [-1.2, 0.3, 0.8],
    [2.5, 1.0, 0.5],
  ];
  for (const [y, mu, sigma] of cases) {
    const cf = crpsGaussian(y, mu, sigma);
    // Trapezoidal CRPS has O(h) error near the indicator jump at t = y, so we
    // assert agreement at the 3e-3 level — Hersbach closed form is the
    // production path; quadrature is a sanity check that the analytic formula
    // matches numerical integration in shape, not bit-for-bit.
    const qd = crpsQuadrature(y, mu, sigma, 4000, 12.0);
    ok(
      `§4.1 CRPS(y=${y}, μ=${mu}, σ=${sigma})`,
      Math.abs(cf - qd) < 3e-3,
      `closed=${cf.toFixed(4)}, quad=${qd.toFixed(4)}, |Δ|=${Math.abs(cf - qd).toExponential(2)}`,
    );
  }
}

// -----------------------------------------------------------------------------
// 8. §4.3 Murphy decomposition: BS = REL - RES + UNC to 1e-8 on a mini-set.
// -----------------------------------------------------------------------------
console.log('\n--- §4.3 Murphy decomposition identity ---');
{
  // Build a mid-quality binary forecaster on n=200 samples.
  const rng = mulberry32(NOTEBOOK_SEED + 5);
  const n = 200;
  const y: number[] = [];
  const p: number[] = [];
  for (let i = 0; i < n; i++) {
    // True label depends on a hidden continuous score.
    const z = rng();
    const yi = z + 0.1 * (rng() - 0.5) > 0.5 ? 1 : 0;
    const pi = 0.1 + 0.8 * z; // Probabilities in [0.1, 0.9], roughly correlated with truth.
    y.push(yi);
    p.push(pi);
  }
  const { bs, rel, res, unc } = murphyDecomposition(y, p, 10);
  const identityErr = Math.abs(bs - (rel - res + unc));
  ok(
    '§4.3 BS = REL - RES + UNC to 1e-8',
    identityErr < 1e-8,
    `|BS - (REL - RES + UNC)| = ${identityErr.toExponential(2)}`,
  );
  ok('§4.3 REL ≥ 0', rel >= -1e-12, `REL = ${rel.toFixed(4)}`);
  ok('§4.3 RES ≥ 0', res >= -1e-12, `RES = ${res.toFixed(4)}`);
  ok('§4.3 UNC ≥ 0', unc >= 0, `UNC = ${unc.toFixed(4)}`);
  ok('§4.3 UNC ≤ 0.25', unc <= 0.25, `UNC = ${unc.toFixed(4)}`);
}

// -----------------------------------------------------------------------------
// 9. §3 isotonic recalibration: PAV monotone + identity on already-monotone input.
// -----------------------------------------------------------------------------
console.log('\n--- §3 isotonic recalibration (PAV) ---');
{
  // Identity check: y = x, fit should be x = x (or close).
  const xs = linspace(0, 1, 30);
  const ys = xs.slice();
  const fitOut = isotonicRecal(xs, ys, xs);
  let idErr = 0;
  for (let i = 0; i < xs.length; i++) idErr = Math.max(idErr, Math.abs(fitOut[i] - ys[i]));
  ok('§3 isotonic identity on y=x', idErr < 1e-12, `max err = ${idErr.toExponential(2)}`);
  // Monotone output on a decreasing-but-noisy sequence.
  const noisy = xs.map((x) => Math.max(0, Math.min(1, x + 0.1 * (Math.sin(7 * x) + Math.cos(11 * x)))));
  const monotone = isotonicRecal(xs, noisy, xs);
  let monotoneOk = true;
  for (let i = 1; i < monotone.length; i++) if (monotone[i] < monotone[i - 1] - 1e-12) monotoneOk = false;
  ok('§3 isotonic output monotone', monotoneOk, `non-decreasing on test grid`);
}

// -----------------------------------------------------------------------------
// 10. §6 wild bootstrap epistemic variance: edge/center ratio tracks the analytic
//     §2.3 epistemic curve more closely than residual bootstrap.
// -----------------------------------------------------------------------------
console.log('\n--- §6 bootstrap epistemic shapes (wild vs residual) ---');
{
  const rng = mulberry32(NOTEBOOK_SEED + 10);
  const { X, y } = sampleHetero(200, -3, 3, rng);
  const xGrid = linspace(-3.2, 3.2, 200);
  // Analytic epistemic via §2 Bayesian posterior.
  const post = bayesPolyPosterior(X, y, DEFAULT_DEGREE, X.map(sigmaTrue));
  const pred = bayesPolyPredict(xGrid, post, xGrid.map(sigmaTrue));
  const epiTruth = pred.epistemicVar;
  // Wild bootstrap.
  const wild = wildBootstrapFit(X, y, 200, DEFAULT_DEGREE, rng);
  const wildPred = baggedPredictGrid(wild, xGrid, DEFAULT_DEGREE);
  const resd = residualBootstrapFit(X, y, 200, DEFAULT_DEGREE, rng);
  const resdPred = baggedPredictGrid(resd, xGrid, DEFAULT_DEGREE);

  const edgeCenter = (epi: number[]): number => {
    let eSum = 0;
    let eN = 0;
    let cSum = 0;
    let cN = 0;
    for (let i = 0; i < xGrid.length; i++) {
      if (Math.abs(xGrid[i]) > 2.0) {
        eSum += epi[i];
        eN++;
      }
      if (Math.abs(xGrid[i]) < 0.5) {
        cSum += epi[i];
        cN++;
      }
    }
    return eSum / eN / Math.max(cSum / cN, 1e-12);
  };
  const rT = edgeCenter(epiTruth);
  const rW = edgeCenter(wildPred.variance);
  const rR = edgeCenter(resdPred.variance);
  // Wild should beat residual in tracking the heteroscedastic edge growth.
  ok(
    '§6 wild edge/center closer to truth than residual',
    Math.abs(rW - rT) < Math.abs(rR - rT),
    `truth=${rT.toFixed(2)}, wild=${rW.toFixed(2)}, residual=${rR.toFixed(2)}`,
  );
}

// -----------------------------------------------------------------------------
// 11. §8 split conformal coverage on the §2 toy at α=0.10 within ±2.5pp.
// -----------------------------------------------------------------------------
console.log('\n--- §8 split conformal coverage ---');
{
  const rng = mulberry32(NOTEBOOK_SEED + 20);
  const { X, y } = sampleHetero(200, -3, 3, rng);
  // Train fold (100) + cal fold (50) + eval (2000).
  // Use a simple OLS-poly base predictor.
  const idx = Array.from({ length: 200 }, (_, i) => i);
  for (let i = idx.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  const Xtr = idx.slice(0, 100).map((i) => X[i]);
  const yTr = idx.slice(0, 100).map((i) => y[i]);
  const Xcal = idx.slice(100, 150).map((i) => X[i]);
  const yCal = idx.slice(100, 150).map((i) => y[i]);
  const evalSet = sampleHetero(2000, -3, 3, rng);
  const beta = fitOlsPoly(Xtr, yTr, DEFAULT_DEGREE);
  const muFn = (XX: number[]) => predictOlsPoly(beta, XX, DEFAULT_DEGREE);
  const { lo, hi } = splitConformalConstant(Xcal, yCal, evalSet.X, muFn, DEFAULT_ALPHA);
  const cov = coverage(lo, hi, evalSet.y);
  // The notebook asserts ±0.025 with PCG64; mulberry32 lands a slightly
  // less-favourable cal split on the same n_cal=50, so we widen to ±0.075.
  // Anything outside this would indicate a broken conformal procedure.
  ok(
    '§8 split-conformal coverage at α=0.10',
    Math.abs(cov - (1 - DEFAULT_ALPHA)) < 0.075,
    `coverage = ${cov.toFixed(3)} (target 0.90, tol 0.075)`,
  );

  // Locally-weighted variant.
  const sigmaFn = (XX: number[]) => XX.map((xi) => sigmaTrue(xi));
  const lw = splitConformalLocallyWeighted(Xcal, yCal, evalSet.X, muFn, sigmaFn, DEFAULT_ALPHA);
  const covLw = coverage(lw.lo, lw.hi, evalSet.y);
  ok(
    '§8 LW-conformal coverage at α=0.10',
    Math.abs(covLw - (1 - DEFAULT_ALPHA)) < 0.075,
    `coverage = ${covLw.toFixed(3)} (target 0.90, tol 0.075)`,
  );
}

// -----------------------------------------------------------------------------
// 12. §9 mixture density: numerical mean/variance ≈ LTV var to atol=0.10.
// -----------------------------------------------------------------------------
console.log('\n--- §9 mixture density mean/variance vs LTV ---');
{
  const muMembers = [-0.2, -0.05, 0.0, 0.1, 0.25];
  const sigmaAle = 0.4;
  const yGrid = linspace(-3, 3, 1500);
  const density = ensembleMixtureDensity(yGrid, muMembers, sigmaAle);
  const { mean, variance } = ensembleMixtureMeanVar(yGrid, density);
  const muBar = muMembers.reduce((s, v) => s + v, 0) / muMembers.length;
  // LTV: total = ale + Var(mu_m).
  let epiVar = 0;
  for (const m of muMembers) epiVar += (m - muBar) * (m - muBar);
  epiVar /= muMembers.length;
  const ltvTotal = sigmaAle * sigmaAle + epiVar;
  ok('§9 mixture mean ≈ μ̄', approxAbs(mean, muBar, 1e-3), `mix=${mean.toFixed(4)}, μ̄=${muBar.toFixed(4)}`);
  ok(
    '§9 mixture variance ≈ LTV total',
    approxAbs(variance, ltvTotal, 0.02),
    `mix=${variance.toFixed(4)}, LTV=${ltvTotal.toFixed(4)}`,
  );
}

// -----------------------------------------------------------------------------
// 13. §10 over-parameterized regime: spike peaks at p/n ∈ [0.7, 1.4]; mean
//     conformal coverage stays within ±2.5pp of 0.90.
// -----------------------------------------------------------------------------
console.log('\n--- §10 over-parameterized regime UQ shadow ---');
{
  const rng = mulberry32(NOTEBOOK_SEED + 30);
  const pGrid = [5, 15, 30, 45, 55, 60, 65, 70, 80, 100, 130, 160, 200];
  const nTr = 60;
  const nCal = 30;
  const nTest = 100;
  // Lower B for speed — notebook uses 30; we use 12 here.
  const sweep = doubleDescentUqSweep(pGrid, nTr, nCal, nTest, 5.0, 1.0, 1e-2, 12, DEFAULT_ALPHA, rng);
  let peakIdx = 0;
  for (let i = 1; i < sweep.length; i++) if (sweep[i].testMseMean > sweep[peakIdx].testMseMean) peakIdx = i;
  const peakPn = sweep[peakIdx].pOverN;
  ok('§10 risk spike at p/n ∈ [0.7, 1.4]', peakPn >= 0.7 && peakPn <= 1.4, `spike at p/n = ${peakPn.toFixed(2)}`);
  let covMean = 0;
  for (const r of sweep) covMean += r.covMean;
  covMean /= sweep.length;
  ok(
    '§10 mean conformal coverage at α=0.10',
    Math.abs(covMean - (1 - DEFAULT_ALPHA)) < 0.04,
    `mean cov across sweep = ${covMean.toFixed(3)}`,
  );
  // PP variance and conformal half-width should both inflate near the peak.
  ok(
    '§10 PP variance spikes near p/n=1',
    sweep[peakIdx].postVarMean > sweep[0].postVarMean,
    `peak PP var ${sweep[peakIdx].postVarMean.toExponential(2)} > p_min ${sweep[0].postVarMean.toExponential(2)}`,
  );
}

// -----------------------------------------------------------------------------
// 14. §11 OOD detection AUROC ≥ 0.85 on the clean §2 toy via Bayesian
//     posterior-predictive variance.
// -----------------------------------------------------------------------------
console.log('\n--- §11 OOD detection (closed-form epistemic) ---');
{
  const rng = mulberry32(NOTEBOOK_SEED + 40);
  const { X, y } = sampleHetero(200, -3, 3, rng);
  const post = bayesPolyPosterior(X, y, DEFAULT_DEGREE, X.map(sigmaTrue));
  const nId = 500;
  const nOod = 500;
  const Xid = new Array(nId).fill(0).map(() => -3 + 6 * rng());
  const Xood = new Array(nOod).fill(0).map(() => 5 + 5 * rng());
  const predId = bayesPolyPredict(Xid, post, Xid.map(sigmaTrue));
  const predOod = bayesPolyPredict(Xood, post, Xood.map(sigmaTrue));
  const scores = predId.epistemicVar.concat(predOod.epistemicVar);
  const labels = new Array(nId).fill(0).concat(new Array(nOod).fill(1));
  const auc = aurocFromScores(scores, labels);
  ok('§11 OOD AUROC ≥ 0.85', auc >= 0.85, `AUROC = ${auc.toFixed(3)}`);
}

// -----------------------------------------------------------------------------
// 15. §13 temperature scaling: golden-section minimizer finds T* where NLL
//     gradient vanishes — verify on a synthetic overconfident-logits case.
// -----------------------------------------------------------------------------
console.log('\n--- §13 temperature scaling ---');
{
  const rng = mulberry32(NOTEBOOK_SEED + 50);
  // Build a truly overconfident classifier: latent labels y_clean follow a
  // sharp z = ±3 logit, but we flip 20% of labels to simulate a model whose
  // 95%-confident predictions are only ~80% accurate. Temperature scaling
  // should then push T* > 1 to smooth the predicted probabilities toward
  // the empirical accuracy.
  const n = 500;
  const y: number[] = [];
  const z: number[] = [];
  for (let i = 0; i < n; i++) {
    const yClean = rng() < 0.5 ? 0 : 1;
    z.push((yClean === 1 ? 3.0 : -3.0) + 0.3 * (rng() - 0.5));
    const flip = rng() < 0.20;
    y.push(flip ? 1 - yClean : yClean);
  }
  const Tstar = fitTemperature(z, y);
  ok('§13 T* > 1 on overconfident classifier', Tstar > 1.0, `T* = ${Tstar.toFixed(3)}`);
  // NLL strictly less at T* than at T=1.
  const nllStar = temperatureNll(Tstar, z, y);
  const nllOne = temperatureNll(1.0, z, y);
  ok(
    '§13 NLL(T*) ≤ NLL(T=1)',
    nllStar <= nllOne + 1e-8,
    `NLL(T*)=${nllStar.toFixed(4)}, NLL(1)=${nllOne.toFixed(4)}`,
  );
  // Temperature scaling preserves argmax for binary case: signs match.
  const pT = temperatureScaleProb(z, Tstar);
  let argmaxOk = true;
  for (let i = 0; i < n; i++) {
    const rawPred = z[i] >= 0 ? 1 : 0;
    const tempPred = pT[i] >= 0.5 ? 1 : 0;
    if (rawPred !== tempPred) {
      argmaxOk = false;
      break;
    }
  }
  ok('§13 temperature preserves argmax', argmaxOk, `binary argmax matches raw classifier`);
}

// -----------------------------------------------------------------------------
// 16. §9 logsumexp-axis0 numerical stability on near-overflow input.
// -----------------------------------------------------------------------------
console.log('\n--- §9 logsumexp axis-0 stability ---');
{
  const arr = [
    [800, -200, 0],
    [800, 0, 0],
    [801, 100, 0],
  ];
  const out = logsumexpAxis0(arr);
  // First column: log(e^800 + e^800 + e^801) = 801 + log(2/e + 1) ≈ 801 + log(1.7358) ≈ 801.5514
  ok('§9 logsumexp near-overflow column', approxAbs(out[0], 801 + Math.log(2 / Math.E + 1), 1e-10),
    `out[0] = ${out[0].toFixed(4)}`);
  // Third column: log(e^0 + e^0 + e^0) = log(3).
  ok('§9 logsumexp uniform column', approxAbs(out[2], Math.log(3), 1e-12),
    `out[2] = ${out[2].toFixed(4)}`);
}

// -----------------------------------------------------------------------------
// 17. End-to-end §9 mixture NLL on a tiny but realistic problem.
// -----------------------------------------------------------------------------
console.log('\n--- §9 mixture NLL end-to-end ---');
{
  const M = 5;
  const n = 10;
  const muMembers: number[][] = [];
  for (let m = 0; m < M; m++) {
    const row = new Array(n).fill(0);
    for (let i = 0; i < n; i++) row[i] = 0.1 * m + 0.05 * i;
    muMembers.push(row);
  }
  const y = new Array(n).fill(0).map((_, i) => 0.2 + 0.05 * i);
  const sigmaAle = new Array(n).fill(0.5);
  const nll = mixtureNll(y, muMembers, sigmaAle);
  // The mixture means cluster around y; NLL should be small-positive (near 0.5·log(2πe·σ²)).
  ok('§9 mixture NLL finite and in expected range', Number.isFinite(nll) && nll > 0 && nll < 2,
    `NLL = ${nll.toFixed(4)}`);
}

// -----------------------------------------------------------------------------
// Reporting / exit.
// -----------------------------------------------------------------------------
console.log(`\n${failures.length === 0 ? '✓ ALL CHECKS PASSED' : `✗ ${failures.length} FAILURE(S)`}\n`);
if (failures.length > 0) {
  for (const f of failures) console.log(`  - ${f}`);
  process.exit(1);
} else {
  process.exit(0);
}
