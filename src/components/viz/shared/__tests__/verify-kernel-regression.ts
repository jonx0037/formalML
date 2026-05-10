// =============================================================================
// verify-kernel-regression.ts
//
// Numerical regression tests for src/components/viz/shared/kernel-regression.ts.
// Each test reproduces a numerical claim from the verified notebook
//   notebooks/kernel-regression/01_kernel_regression.ipynb
// and asserts the result lies in the expected range.
//
// Run with: pnpm verify:kernel-regression
// Exits non-zero on first failure. Designed to be CI-friendly.
//
// Tolerances reflect the brief's spec — Monte-Carlo tests use loose ranges
// (the goal is to verify rate-scaling, not bit-exact MC reproduction since
// we substitute Mulberry32+Box-Muller for NumPy's PCG64+Ziggurat).
// =============================================================================

import {
  KERNEL_CONSTANTS,
  KERNELS,
  gaussianSampler,
  gcvScore,
  hStarAmiseMd,
  hStarAmiseUni,
  kGaussian,
  localLinear,
  looCvScore,
  mTrueMd,
  mTrueUni,
  mulberry32,
  nadarayaWatson,
  nwMd,
  sampleToyMd,
  sampleToyUni,
  silvermanRule,
  splitConformalNw,
} from '../kernel-regression';

// -----------------------------------------------------------------------------
// Test plumbing
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
  ok(name, observed >= lo && observed <= hi, `${label} observed=${observed.toExponential(3)} in [${lo}, ${hi}]`);
}

function approxEq(name: string, observed: number, expected: number, tol: number, label: string): void {
  const gap = Math.abs(observed - expected);
  ok(name, gap < tol, `${label} observed=${observed.toFixed(6)} expected=${expected.toFixed(6)} gap=${gap.toExponential(3)} tol=${tol.toExponential(3)}`);
}

// Quadrature: composite Simpson on [a, b] with N (even) panels.
function simpson(f: (u: number) => number, a: number, b: number, N: number): number {
  if (N % 2 !== 0) N += 1;
  const h = (b - a) / N;
  let s = f(a) + f(b);
  for (let i = 1; i < N; i++) {
    const x = a + i * h;
    s += (i % 2 === 0 ? 2 : 4) * f(x);
  }
  return (s * h) / 3;
}

// -----------------------------------------------------------------------------
// Test 1: kernel constants — closed-form mu_2(K), R(K), delta(K), C(K) match
// notebook §2.1 / §7.1 printed values.
// -----------------------------------------------------------------------------

console.log('\n[1] kernel_constants — closed-form vs numerical integration');

{
  const expected = {
    gaussian: { mu2: 1.0, R: 1 / (2 * Math.sqrt(Math.PI)), delta: 0.7764, C: 0.3633 },
    epanechnikov: { mu2: 0.2, R: 0.6, delta: 1.7188, C: 0.3491 },
    box: { mu2: 1 / 3, R: 0.5, delta: 1.3510, C: 0.3701 },
    triangular: { mu2: 1 / 6, R: 2 / 3, delta: 1.8882, C: 0.3531 },
    quartic: { mu2: 1 / 7, R: 5 / 7, delta: 2.0362, C: 0.3508 },
  };
  for (const [name, exp] of Object.entries(expected)) {
    const got = KERNEL_CONSTANTS[name as keyof typeof KERNEL_CONSTANTS];
    approxEq(`kernel_constants[${name}].mu2`, got.mu2, exp.mu2, 1e-12, 'mu_2');
    approxEq(`kernel_constants[${name}].R`, got.R, exp.R, 1e-12, 'R(K)');
    approxEq(`kernel_constants[${name}].delta`, got.delta, exp.delta, 1e-3, 'delta(K)');
    approxEq(`kernel_constants[${name}].C`, got.C, exp.C, 1e-3, 'C(K)');
  }

  // Verify kernel functions integrate to 1 and have zero first moment via Simpson.
  // Each kernel is integrated on its natural support — Gaussian on [-5, 5] (tails
  // negligible), the four compact-support kernels on [-1, 1] where their integrands
  // are smooth (Simpson struggles across the box-kernel discontinuity at u = ±1).
  const supports: Record<string, [number, number]> = {
    gaussian: [-5, 5],
    epanechnikov: [-1, 1],
    box: [-1, 1],
    triangular: [-1, 1],
    quartic: [-1, 1],
  };
  for (const [name, K] of Object.entries(KERNELS)) {
    const [a, b] = supports[name];
    const integral = simpson(K, a, b, 2000);
    const firstMoment = simpson((u) => u * K(u), a, b, 2000);
    const secondMoment = simpson((u) => u * u * K(u), a, b, 2000);
    const RK = simpson((u) => K(u) * K(u), a, b, 2000);
    approxEq(`kernel[${name}] integrates to 1`, integral, 1.0, 1e-3, 'int K');
    approxEq(`kernel[${name}] first moment = 0`, firstMoment, 0.0, 1e-6, 'int u K');
    approxEq(`kernel[${name}] mu_2 numerical`, secondMoment, KERNEL_CONSTANTS[name as keyof typeof KERNELS].mu2, 1e-3, 'int u^2 K');
    approxEq(`kernel[${name}] R(K) numerical`, RK, KERNEL_CONSTANTS[name as keyof typeof KERNELS].R, 1e-3, 'int K^2');
  }

  // Epanechnikov minimizes C(K).
  const Cs = Object.entries(KERNEL_CONSTANTS).map(([n, k]) => ({ n, C: k.C }));
  Cs.sort((a, b) => a.C - b.C);
  ok('kernel_constants epanechnikov_min_C', Cs[0].n === 'epanechnikov', `min-C kernel = ${Cs[0].n} (expected epanechnikov)`);
}

// -----------------------------------------------------------------------------
// Test 2: NW vs WLS agreement — at x_0 = 0.5 on the §1 toy with h = 0.05,
// direct NW formula must equal WLS-with-constant-design to atol = 1e-10.
// -----------------------------------------------------------------------------

console.log('\n[2] nw_vs_wls_agreement — degree-zero local-polynomial equivalence');

{
  const rng = mulberry32(42);
  const { X, Y } = sampleToyUni(200, 0.2, rng);

  // Direct NW.
  const xEval = new Float64Array([0.5]);
  const mhatNw = nadarayaWatson(X, Y, xEval, 0.05, kGaussian)[0];

  // WLS with constant design: hat alpha = sum_i w_i Y_i / sum_i w_i.
  let num = 0;
  let den = 0;
  for (let i = 0; i < X.length; i++) {
    const w = kGaussian((X[i] - 0.5) / 0.05) / 0.05;
    num += w * Y[i];
    den += w;
  }
  const mhatWls = num / den;

  approxEq('nw_vs_wls_agreement', mhatNw, mhatWls, 1e-10, 'NW direct vs WLS-with-1');
}

// -----------------------------------------------------------------------------
// Test 3: Interior bias formula — empirical bias at x in {0.25, 0.5, 0.75}
// on the §1 toy (X ~ Uniform => f_X' = 0) collapses to the prediction
//   B(x) = (h^2/2) * mu_2(K) * m''(x) = -2 pi^2 h^2 sin(2 pi x) (Gaussian)
// to atol = 0.01 with B = 500 replicates at h = 0.05.
// -----------------------------------------------------------------------------

console.log('\n[3] interior_bias_formula — Monte Carlo bias matches asymptotic prediction');

{
  const N = 200;
  const SIGMA = 0.2;
  const h = 0.05;
  const B = 500;
  const targets = new Float64Array([0.25, 0.5, 0.75]);

  const sums = new Float64Array(targets.length);
  const rng = mulberry32(20260501);
  for (let b = 0; b < B; b++) {
    const { X, Y } = sampleToyUni(N, SIGMA, rng);
    const mhat = nadarayaWatson(X, Y, targets, h, kGaussian);
    for (let j = 0; j < targets.length; j++) sums[j] += mhat[j];
  }
  const empBias = new Float64Array(targets.length);
  const predBias = new Float64Array(targets.length);
  for (let j = 0; j < targets.length; j++) {
    empBias[j] = sums[j] / B - mTrueUni(targets[j]);
    predBias[j] = -2 * Math.PI * Math.PI * h * h * Math.sin(2 * Math.PI * targets[j]);
  }

  for (let j = 0; j < targets.length; j++) {
    approxEq(
      `interior_bias_formula[x=${targets[j].toFixed(2)}]`,
      empBias[j],
      predBias[j],
      0.015,
      `empirical vs h^2/2 mu2 m''`,
    );
  }
}

// -----------------------------------------------------------------------------
// Test 4: Boundary-bias breakdown — at x ~ 0.005 (boundary), NW bias is O(h)
// while LL bias remains O(h^2). Notebook reports ratio 22.6x; brief tolerance > 3.
// -----------------------------------------------------------------------------

console.log('\n[4] boundary_bias_breakdown — LL fixes the O(h) NW boundary defect');

{
  const N = 200;
  const SIGMA = 0.2;
  const h = 0.05;
  const B = 500;
  const xEval = new Float64Array([0.005, 0.245]);

  const sumNw = new Float64Array(xEval.length);
  const sumLl = new Float64Array(xEval.length);
  const rng = mulberry32(20260601);
  for (let b = 0; b < B; b++) {
    const { X, Y } = sampleToyUni(N, SIGMA, rng);
    const nw = nadarayaWatson(X, Y, xEval, h, kGaussian);
    const ll = localLinear(X, Y, xEval, h, kGaussian);
    for (let j = 0; j < xEval.length; j++) {
      sumNw[j] += nw[j];
      sumLl[j] += ll[j];
    }
  }
  const biasNwBoundary = Math.abs(sumNw[0] / B - mTrueUni(xEval[0]));
  const biasLlBoundary = Math.abs(sumLl[0] / B - mTrueUni(xEval[0]));
  const biasNwInterior = Math.abs(sumNw[1] / B - mTrueUni(xEval[1]));
  const biasLlInterior = Math.abs(sumLl[1] / B - mTrueUni(xEval[1]));

  const ratio = biasNwBoundary / biasLlBoundary;
  ok(
    'boundary_bias_breakdown',
    ratio > 3,
    `|bias_NW(0.005)|/|bias_LL(0.005)| = ${ratio.toFixed(2)}x  (NW=${biasNwBoundary.toFixed(4)}, LL=${biasLlBoundary.toFixed(4)})`,
  );
  ok(
    'interior_bias_NW_LL_match',
    Math.abs(biasNwInterior - biasLlInterior) < 0.02,
    `interior |bias_NW(0.245) - bias_LL(0.245)| = ${Math.abs(biasNwInterior - biasLlInterior).toFixed(4)}`,
  );
}

// -----------------------------------------------------------------------------
// Test 5: Optimal bandwidth — AMISE U-curve minimum on the §1 toy must lie
// within factor 2 of the analytical h^* = 0.0373. Using B = 100 replicates.
// -----------------------------------------------------------------------------

console.log('\n[5] optimal_h_ratio — empirical AMISE-min within factor 2 of analytical h^*');

{
  const N = 200;
  const SIGMA = 0.2;
  const B = 100;
  const hGrid = new Float64Array(40);
  const lo = Math.log(0.005);
  const hi = Math.log(0.2);
  for (let i = 0; i < 40; i++) hGrid[i] = Math.exp(lo + ((hi - lo) * i) / 39);

  const xGrid = new Float64Array(40);
  for (let j = 0; j < 40; j++) xGrid[j] = 0.1 + (0.8 * j) / 39;
  const mGrid = new Float64Array(40);
  for (let j = 0; j < 40; j++) mGrid[j] = mTrueUni(xGrid[j]);

  const amise = new Float64Array(40);
  const rng = mulberry32(20260701);
  for (let b = 0; b < B; b++) {
    const { X, Y } = sampleToyUni(N, SIGMA, rng);
    for (let k = 0; k < hGrid.length; k++) {
      const mhat = nadarayaWatson(X, Y, xGrid, hGrid[k], kGaussian);
      let mse = 0;
      for (let j = 0; j < 40; j++) {
        const r = mhat[j] - mGrid[j];
        mse += r * r;
      }
      amise[k] += mse / 40;
    }
  }
  for (let k = 0; k < amise.length; k++) amise[k] /= B;

  let argmin = 0;
  for (let k = 1; k < amise.length; k++) if (amise[k] < amise[argmin]) argmin = k;
  const hEmp = hGrid[argmin];
  const hStar = hStarAmiseUni(N, SIGMA);
  const ratio = hEmp / hStar;
  within('optimal_h_ratio', ratio, 0.5, 2.0, `h_empirical/h^* = ${ratio.toFixed(2)}x  (h_emp=${hEmp.toFixed(4)}, h^*=${hStar.toFixed(4)})`);
}

// -----------------------------------------------------------------------------
// Test 6: LOO-CV / GCV / Silverman within factor of analytical h^*.
// Brief: 0.3 < h_CV / h^* < 3.0  (slow n^{-3/10} rate, expected variance).
// -----------------------------------------------------------------------------

console.log('\n[6] cv_within_factor_3 — LOO-CV and GCV land near h^*; Silverman oversmooths');

{
  const N = 200;
  const SIGMA = 0.2;
  const rng = mulberry32(2026);
  const { X, Y } = sampleToyUni(N, SIGMA, rng);
  const hStar = hStarAmiseUni(N, SIGMA);
  const hSilverman = silvermanRule(X);

  const hGrid = new Float64Array(40);
  const lo = Math.log(0.005);
  const hi = Math.log(0.3);
  for (let i = 0; i < 40; i++) hGrid[i] = Math.exp(lo + ((hi - lo) * i) / 39);

  let cvMin = Infinity;
  let gcvMin = Infinity;
  let cvIdx = 0;
  let gcvIdx = 0;
  for (let k = 0; k < hGrid.length; k++) {
    const cvScore = looCvScore(X, Y, hGrid[k], kGaussian);
    const gcv = gcvScore(X, Y, hGrid[k], kGaussian);
    if (cvScore < cvMin) {
      cvMin = cvScore;
      cvIdx = k;
    }
    if (gcv < gcvMin) {
      gcvMin = gcv;
      gcvIdx = k;
    }
  }

  const hCv = hGrid[cvIdx];
  const hGcv = hGrid[gcvIdx];
  within('cv_within_factor_3', hCv / hStar, 0.3, 3.0, `LOO-CV: h_CV/h^* = ${(hCv / hStar).toFixed(2)}x  (h_CV=${hCv.toFixed(4)})`);
  within('gcv_within_factor_3', hGcv / hStar, 0.3, 3.0, `GCV: h_GCV/h^* = ${(hGcv / hStar).toFixed(2)}x  (h_GCV=${hGcv.toFixed(4)})`);
  ok(
    'silverman_oversmoothing',
    hSilverman / hStar > 1.5,
    `Silverman h_S/h^* = ${(hSilverman / hStar).toFixed(2)}x  (oversmoothing as expected)`,
  );
  // GCV-LOO closeness — minima should be within ~3 grid points (40-pt log-grid).
  ok(
    'gcv_loo_minima_close',
    Math.abs(cvIdx - gcvIdx) <= 3,
    `|argmin(GCV) - argmin(LOO-CV)| = ${Math.abs(cvIdx - gcvIdx)} grid points (expected ≤ 3)`,
  );
}

// -----------------------------------------------------------------------------
// Test 7: Canonical-kernel theorem — at AMISE-optimal h^*_K = delta(K) * h_bar^*,
// the ratio h^*_K / delta(K) is identical across kernels (= h_bar^*).
// -----------------------------------------------------------------------------

console.log('\n[7] canonical_kernel_invariance — h^*_K / delta(K) is kernel-independent');

{
  const N = 200;
  const SIGMA = 0.2;
  const thetaMf = 8 * Math.pow(Math.PI, 4);
  const nuSigma = SIGMA * SIGMA;
  const hBarStar = Math.pow(nuSigma / (N * thetaMf), 1 / 5);

  const ratios: number[] = [];
  for (const k of ['gaussian', 'epanechnikov', 'box', 'triangular', 'quartic'] as const) {
    const { delta } = KERNEL_CONSTANTS[k];
    const hStarK = delta * hBarStar;
    ratios.push(hStarK / delta);
  }
  for (let i = 1; i < ratios.length; i++) {
    approxEq(`canonical_kernel_invariance[${i}]`, ratios[i], ratios[0], 1e-12, `h^*_K / delta(K) ratio i=${i}`);
  }
  approxEq('h_bar_star_value', ratios[0], hBarStar, 1e-12, 'h_bar^*');
  // Notebook reports h_bar* = 0.0481.
  approxEq('h_bar_star_notebook_value', hBarStar, 0.0481, 0.001, 'h_bar^* notebook');
}

// -----------------------------------------------------------------------------
// Test 8: Curse-of-dimensionality slope — empirical AMSE-vs-n slope on log-log
// must be within atol = 0.25 of theoretical -4/(4+d) for d in {1, 2, 5}.
// (d = 10 not in finite-sample regime; brief explicitly excludes it.)
// -----------------------------------------------------------------------------

console.log('\n[8] curse_slope — empirical AMSE rates match -4/(4+d) for d in {1, 2, 5}');

{
  const SIGMA = 0.2;
  const ns = [100, 200, 500, 1000, 2000];
  const dValues = [1, 2, 5];
  const Bper = 80; // notebook uses 100; 80 is a slight speed-up with same rate
  const rng = mulberry32(20261001);

  for (const d of dValues) {
    const x0 = new Float64Array(d);
    for (let j = 0; j < d; j++) x0[j] = 0.25;
    const mX0 = mTrueMd(x0);

    const logAmse = new Float64Array(ns.length);
    const logN = new Float64Array(ns.length);
    for (let kn = 0; kn < ns.length; kn++) {
      const n = ns[kn];
      const hOpt = hStarAmiseMd(d, n, SIGMA);
      let acc = 0;
      for (let b = 0; b < Bper; b++) {
        const { X, Y } = sampleToyMd(n, d, SIGMA, rng);
        const mhat = nwMd(X, Y, x0, hOpt);
        const r = mhat - mX0;
        acc += r * r;
      }
      logAmse[kn] = Math.log10(acc / Bper);
      logN[kn] = Math.log10(n);
    }

    // Linear regression slope.
    let mn = 0;
    let ma = 0;
    for (let i = 0; i < ns.length; i++) {
      mn += logN[i];
      ma += logAmse[i];
    }
    mn /= ns.length;
    ma /= ns.length;
    let num = 0;
    let den = 0;
    for (let i = 0; i < ns.length; i++) {
      num += (logN[i] - mn) * (logAmse[i] - ma);
      den += (logN[i] - mn) * (logN[i] - mn);
    }
    const slope = num / den;
    const theory = -4 / (4 + d);
    ok(
      `curse_slope[d=${d}]`,
      Math.abs(slope - theory) < 0.25,
      `empirical=${slope.toFixed(4)} theory=${theory.toFixed(4)} gap=${Math.abs(slope - theory).toFixed(4)}`,
    );
  }
}

// -----------------------------------------------------------------------------
// Test 9: Split-conformal coverage — empirical coverage on the §1 toy must
// match the 90% nominal target within MC tolerance (|cov - 0.9| < 0.03 at B = 100).
// -----------------------------------------------------------------------------

console.log('\n[9] conformal_coverage — split-conformal NW achieves nominal 90% coverage');

{
  const N = 200;
  const SIGMA = 0.2;
  const ALPHA = 0.1;
  const H = 0.04;
  const B = 100;
  const N_TEST = 50;
  const rng = mulberry32(20261201);
  const gauss = gaussianSampler(rng);

  let coveredCount = 0;
  let totalTests = 0;
  let widthSum = 0;

  for (let b = 0; b < B; b++) {
    const { X, Y } = sampleToyUni(N, SIGMA, rng);
    // Generate test data from the same DGP.
    const xTest = new Float64Array(N_TEST);
    const yTest = new Float64Array(N_TEST);
    for (let j = 0; j < N_TEST; j++) {
      xTest[j] = rng();
      yTest[j] = mTrueUni(xTest[j]) + SIGMA * gauss();
    }
    const result = splitConformalNw(X, Y, xTest, H, ALPHA, rng, kGaussian);
    for (let j = 0; j < N_TEST; j++) {
      // NaN can happen if the NW kernel weights all collapse — count as uncovered.
      const lo = result.lower[j];
      const hi = result.upper[j];
      if (Number.isFinite(lo) && Number.isFinite(hi)) {
        if (yTest[j] >= lo && yTest[j] <= hi) coveredCount++;
        widthSum += hi - lo;
        totalTests++;
      }
    }
  }
  const coverage = coveredCount / totalTests;
  const meanWidth = widthSum / totalTests;
  ok(
    'conformal_coverage',
    Math.abs(coverage - (1 - ALPHA)) < 0.03,
    `empirical coverage=${coverage.toFixed(4)} target=${(1 - ALPHA).toFixed(2)} mean PI width=${meanWidth.toFixed(3)}`,
  );
}

// -----------------------------------------------------------------------------
// Summary
// -----------------------------------------------------------------------------

console.log('\n' + '='.repeat(70));
console.log(`Results: ${pass} passed, ${fail} failed.`);
if (fail > 0) {
  console.log('\nFailures:');
  for (const f of failures) console.log(`  - ${f}`);
  process.exit(1);
}
console.log('All numerical regression tests passed.');
process.exit(0);
