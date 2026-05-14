// =============================================================================
// verify-double-descent.ts
//
// Numerical regression tests for src/components/viz/shared/double-descent.ts.
// Each test reproduces a numerical claim from the verified notebook
//   notebooks/double-descent/01_double_descent.ipynb
// and asserts the result lies in the expected range.
//
// Run with: pnpm verify:double-descent
// Exits non-zero on first failure. Designed to be CI-friendly.
//
// Tolerances: closed-form quantities (Hastie 2022 risk, MP support) must match
// notebook to <1e-9 relative error. Empirical Monte-Carlo quantities (excess-
// risk MC, random-feature MC, classical bias-variance) are not reproduced
// bit-for-bit (mulberry32 vs PCG64); we assert wide-tolerance agreement and
// rate / sign correctness.
// =============================================================================

import {
  activations,
  betaFromSVD,
  classicalBiasVariance,
  conditionNumber,
  excessRiskMisspecified,
  excessRiskSweep,
  fillIsotropicGaussian,
  gaussianRng,
  gdTrajectory,
  hastieRisk,
  hastieRiskDecomp,
  legendreVandermonde,
  matVec,
  mpAtomZero,
  mpDensity,
  mpSupport,
  mulberry32,
  randomFeatureMap,
  sampleSphereBetaStar,
  thinSVD,
} from '../double-descent';

const failures: string[] = [];

function ok(name: string, condition: boolean, detail: string) {
  if (condition) {
    console.log(`  ✓ ${name}: ${detail}`);
  } else {
    console.log(`  ✗ ${name}: ${detail}`);
    failures.push(`${name}: ${detail}`);
  }
}

function approxRel(a: number, b: number, rtol: number): boolean {
  if (!Number.isFinite(a) || !Number.isFinite(b)) return a === b;
  if (b === 0) return Math.abs(a) <= rtol;
  return Math.abs(a - b) <= rtol * Math.abs(b);
}

console.log('\n=== verify-double-descent.ts ===\n');

// -----------------------------------------------------------------------------
// 1. Closed-form sanity: MP support endpoints and density at γ = 1
// -----------------------------------------------------------------------------
console.log('--- §5 Marchenko–Pastur (closed form) ---');
{
  const { lamMinus, lamPlus } = mpSupport(0.25);
  ok(
    '§5 MP support γ=0.25',
    approxRel(lamMinus, 0.25, 1e-12) && approxRel(lamPlus, 2.25, 1e-12),
    `λ₋=${lamMinus.toFixed(6)} (expect 0.25), λ₊=${lamPlus.toFixed(6)} (expect 2.25)`,
  );
}
{
  const { lamMinus, lamPlus } = mpSupport(1);
  ok(
    '§5 MP support γ=1',
    Math.abs(lamMinus) < 1e-12 && approxRel(lamPlus, 4, 1e-12),
    `λ₋=${lamMinus.toExponential(2)} (expect 0), λ₊=${lamPlus.toFixed(6)} (expect 4)`,
  );
}
{
  // Density at the bulk midpoint of γ=1 is f(2) = √(2·2)/(2π·1·2) = 1/(2π) ≈ 0.1592
  const f = mpDensity(2, 1);
  ok(
    '§5 MP density at λ=2, γ=1',
    approxRel(f, 1 / (2 * Math.PI), 1e-12),
    `f=${f.toFixed(6)} (expect ${(1 / (2 * Math.PI)).toFixed(6)})`,
  );
}
{
  // Atomic mass at λ=0 for γ=2 is 1 − 1/2 = 0.5.
  ok('§5 MP atom γ=2', mpAtomZero(2) === 0.5, `atom=${mpAtomZero(2)} (expect 0.5)`);
  ok('§5 MP atom γ=0.5', mpAtomZero(0.5) === 0, `atom=${mpAtomZero(0.5)} (expect 0)`);
}

// -----------------------------------------------------------------------------
// 2. Hastie 2022 closed form: §6.2 piecewise risk
// -----------------------------------------------------------------------------
console.log('\n--- §6 Hastie 2022 closed form ---');
{
  // γ=0.5, σ²=1, r²=1 → R = 1 · 0.5 / 0.5 = 1
  const R = hastieRisk(0.5, 1, 1);
  ok('§6 Hastie risk γ=0.5', approxRel(R, 1, 1e-12), `R=${R.toFixed(6)} (expect 1.0)`);
}
{
  // γ=2, σ²=1, r²=1 → R = 1·(1 − 0.5) + 1/1 = 1.5
  const R = hastieRisk(2, 1, 1);
  ok('§6 Hastie risk γ=2', approxRel(R, 1.5, 1e-12), `R=${R.toFixed(6)} (expect 1.5)`);
}
{
  // γ=4, σ²=1, r²=1 → R = 1·(1 − 0.25) + 1/3 = 0.75 + 0.333… ≈ 1.0833
  const R = hastieRisk(4, 1, 1);
  ok('§6 Hastie risk γ=4', approxRel(R, 0.75 + 1 / 3, 1e-12), `R=${R.toFixed(6)} (expect ≈1.0833)`);
}
{
  const R = hastieRisk(1, 1, 1);
  ok('§6 Hastie risk γ=1 is +∞', !Number.isFinite(R) && R > 0, `R=${R}`);
}
{
  // Decomposition: γ=4, r²=1, σ²=1 → bias²=0.75, variance=1/3
  const d = hastieRiskDecomp(4, 1, 1);
  ok(
    '§6 Hastie decomposition γ=4',
    approxRel(d.bias2, 0.75, 1e-12) && approxRel(d.variance, 1 / 3, 1e-12),
    `bias²=${d.bias2.toFixed(4)} (expect 0.75), variance=${d.variance.toFixed(4)} (expect 0.333)`,
  );
}

// -----------------------------------------------------------------------------
// 3. SVD pseudoinverse round-trip
// -----------------------------------------------------------------------------
console.log('\n--- SVD pseudoinverse round-trip ---');
{
  const rng = mulberry32(42);
  const gauss = gaussianRng(rng);
  const n = 30;
  const p = 50;
  const X = new Float64Array(n * p);
  fillIsotropicGaussian(X, n, p, gauss);
  const betaStar = new Float64Array(p);
  for (let j = 0; j < p; j++) betaStar[j] = gauss();
  const y = new Float64Array(n);
  matVec(X, betaStar, n, p, y);
  // No noise: β̂† should interpolate exactly, ‖Xβ̂† − y‖ ≈ 0.
  const svd = thinSVD(X, n, p);
  const betaHat = betaFromSVD(svd, y);
  const yhat = matVec(X, betaHat, n, p);
  let res = 0;
  for (let i = 0; i < n; i++) res += (yhat[i] - y[i]) ** 2;
  ok('SVD pinv interpolates (wide, noiseless)', res < 1e-18, `‖Xβ̂† − y‖²=${res.toExponential(3)} (<1e-18 expected)`);
}
{
  // Tall case: β̂† should equal OLS solution to within precision.
  const rng = mulberry32(7);
  const gauss = gaussianRng(rng);
  const n = 80;
  const p = 20;
  const X = new Float64Array(n * p);
  fillIsotropicGaussian(X, n, p, gauss);
  const betaStar = new Float64Array(p);
  for (let j = 0; j < p; j++) betaStar[j] = gauss();
  const y = new Float64Array(n);
  matVec(X, betaStar, n, p, y);
  for (let i = 0; i < n; i++) y[i] += 0.01 * gauss();
  const svd = thinSVD(X, n, p);
  const betaHat = betaFromSVD(svd, y);
  // For tall noiseless-ish case, ‖β̂† − β*‖ should be tiny (noise≈0.01).
  let dist2 = 0;
  for (let j = 0; j < p; j++) dist2 += (betaHat[j] - betaStar[j]) ** 2;
  ok(
    'SVD pinv recovers β* (tall, low noise)',
    dist2 < 0.05,
    `‖β̂† − β*‖²=${dist2.toFixed(4)} (<0.05 expected)`,
  );
}

// -----------------------------------------------------------------------------
// 4. §1 empirical excess-risk sweep at the threshold (Cell [004])
//
// Notebook printed: p=1 → 1.04, p=50 → 977, p=200 → 1.09 at n=50, σ²=1, SNR=1, B=50.
// We use a smaller B=20 to keep the test fast and assert wide bounds.
// -----------------------------------------------------------------------------
console.log('\n--- §1 empirical excess-risk sweep (Cell [004]) ---');
{
  const rng = mulberry32(123);
  const gauss = gaussianRng(rng);
  const n = 50;
  const pMax = 200;
  const SNR = 1;
  const r = Math.sqrt(SNR); // r² = SNR · σ²; σ = 1
  const betaStar = sampleSphereBetaStar(pMax, r, gauss);
  // Spike value at p=n=50
  const grid = new Int32Array([1, 50, 200]);
  const out = excessRiskSweep({
    n,
    pMax,
    pGrid: grid,
    betaStarFull: betaStar,
    sigma: 1,
    B: 20,
    rng: mulberry32(456),
  });
  const Rp1 = out.mean[0];
  const Rp50 = out.mean[1];
  const Rp200 = out.mean[2];
  ok(
    '§1 left edge p=1',
    Rp1 > 0.5 && Rp1 < 2.5,
    `R(p=1)=${Rp1.toFixed(3)} (notebook 1.04, expect ∈ [0.5, 2.5])`,
  );
  ok(
    '§1 spike p=n=50',
    Rp50 > 200,
    `R(p=50)=${Rp50.toFixed(1)} (notebook 977, expect > 200 — high variance)`,
  );
  ok(
    '§1 floor p=200=4n',
    Rp200 > 0.5 && Rp200 < 3.0,
    `R(p=200)=${Rp200.toFixed(3)} (notebook 1.09, expect ∈ [0.5, 3.0])`,
  );
  // Spike-to-floor ratio
  ok(
    '§1 spike dominates floor by ≥ 100×',
    Rp50 / Rp200 > 100,
    `ratio R(50)/R(200) = ${(Rp50 / Rp200).toFixed(1)} (expect > 100)`,
  );
}

// -----------------------------------------------------------------------------
// 5. §2 classical U-curve minimum at d = 5 (Cell [011])
// -----------------------------------------------------------------------------
console.log('\n--- §2 classical U-curve (Cell [011]) ---');
{
  const rng = mulberry32(13);
  const testGrid = new Float64Array(50);
  for (let i = 0; i < 50; i++) testGrid[i] = -1 + (2 * i) / 49;
  const bv = classicalBiasVariance({
    n: 100,
    sigma: 0.2,
    dMax: 15,
    B: 80,
    testGrid,
    rng,
  });
  // Find argmin
  let argmin = 0;
  let minVal = bv.total[0];
  for (let d = 1; d < bv.total.length; d++) {
    if (bv.total[d] < minVal) {
      minVal = bv.total[d];
      argmin = d;
    }
  }
  ok(
    '§2 classical U min at d ∈ [3, 7]',
    argmin >= 3 && argmin <= 7,
    `argmin = ${argmin} (notebook 5, expect ∈ [3, 7])`,
  );
  // Bias drops by the U-min and stays low at high d. We check the bias² alone
  // (which is the stable component); the variance contribution at high d has
  // outlier-replicate noise with B=80 that we don't assert tightly on.
  ok(
    '§2 bias² → 0 by d = 10',
    bv.bias2[10] < 1e-3,
    `bias²(d=10)=${bv.bias2[10].toExponential(3)} (< 1e-3 expected)`,
  );
  ok(
    '§2 U-minimum total at least 50× smaller than total(d=0)',
    bv.total[argmin] * 50 < bv.total[0],
    `total(d=${argmin})=${bv.total[argmin].toExponential(3)}, total(d=0)=${bv.total[0].toFixed(3)}`,
  );
}

// -----------------------------------------------------------------------------
// 6. §7 sample-wise pathology (Cell [043])
//
// Notebook: R(n=5)=1.0, R(n=49)=210; adding 44 points multiplies risk ~210×.
// Reading from §6 Hastie formula at γ = p/n. p=50 fixed.
//   γ=10 (n=5): R = 1·(1 − 0.1) + 1/9 = 0.9 + 0.111 = 1.011
//   γ=50/49≈1.02 (n=49): R = 1·(1 − 49/50) + 1/(50/49 − 1) = 0.02 + 49 = 49.02
// The MC at n=49 has high variance; we check Hastie formula matches.
// -----------------------------------------------------------------------------
console.log('\n--- §7 sample-wise (Hastie formula) ---');
{
  const R_n5 = hastieRisk(10, 1, 1);
  const R_n49 = hastieRisk(50 / 49, 1, 1);
  ok(
    '§7 Hastie R at n=5 (γ=10)',
    approxRel(R_n5, 0.9 + 1 / 9, 1e-12),
    `R=${R_n5.toFixed(4)} (expect ≈1.0111)`,
  );
  ok('§7 Hastie R at n=49 (γ=50/49)', R_n49 > 40 && R_n49 < 70, `R=${R_n49.toFixed(2)} (expect ~49)`);
  // Empirical at n=49 was 210 (MC artifact of finite n near threshold); we only check the Hastie predicts >> R_n5.
  ok('§7 Hastie n=49 dominates n=5 by ≥ 30×', R_n49 / R_n5 > 30, `ratio = ${(R_n49 / R_n5).toFixed(1)}`);
}

// -----------------------------------------------------------------------------
// 7. §8 random-feature floors at p=200 (Cell [047])
//
// Notebook: linear=0.723, ReLU=1.267, tanh=1.222 at d=20, n=50, B small.
// We assert the qualitative ordering (linear < ReLU, linear < tanh) and floor < 2.
// -----------------------------------------------------------------------------
console.log('\n--- §8 random-feature floors at p=200 (Cell [047]) ---');
{
  const rng = mulberry32(77);
  const gauss = gaussianRng(rng);
  const d = 20;
  const n = 50;
  const p = 200;
  const sigma = 1;
  const SNR = 1;
  const B = 8;
  const betaStar = sampleSphereBetaStar(d, Math.sqrt(SNR), gauss);
  // For each activation, average excess risk at p=200.
  const risks: Record<string, number> = {};
  for (const actName of ['linear', 'relu', 'tanh'] as const) {
    let acc = 0;
    for (let b = 0; b < B; b++) {
      const X = new Float64Array(n * d);
      fillIsotropicGaussian(X, n, d, gauss);
      const W = new Float64Array(p * d);
      fillIsotropicGaussian(W, p, d, gauss);
      const Phi = randomFeatureMap(X, n, d, W, p, activations[actName]);
      const Xte = new Float64Array(1000 * d);
      fillIsotropicGaussian(Xte, 1000, d, gauss);
      const PhiTe = randomFeatureMap(Xte, 1000, d, W, p, activations[actName]);
      // Targets: linear in x.
      const y = new Float64Array(n);
      const yteTrue = new Float64Array(1000);
      for (let i = 0; i < n; i++) {
        let s = 0;
        for (let j = 0; j < d; j++) s += X[i * d + j] * betaStar[j];
        y[i] = s + sigma * gauss();
      }
      for (let i = 0; i < 1000; i++) {
        let s = 0;
        for (let j = 0; j < d; j++) s += Xte[i * d + j] * betaStar[j];
        yteTrue[i] = s;
      }
      // Fit min-norm β on Φ.
      const betaHat = betaFromSVD(thinSVD(Phi, n, p), y);
      // Test prediction
      let mse = 0;
      for (let i = 0; i < 1000; i++) {
        let s = 0;
        for (let j = 0; j < p; j++) s += PhiTe[i * p + j] * betaHat[j];
        const e = s - yteTrue[i];
        mse += e * e;
      }
      acc += mse / 1000;
    }
    risks[actName] = acc / B;
  }
  ok(
    '§8 linear floor < 2',
    risks.linear > 0.1 && risks.linear < 2,
    `R_linear=${risks.linear.toFixed(3)} (notebook 0.723)`,
  );
  ok(
    '§8 ReLU floor < 4',
    risks.relu > 0.3 && risks.relu < 4,
    `R_ReLU=${risks.relu.toFixed(3)} (notebook 1.267)`,
  );
  ok(
    '§8 tanh floor < 4',
    risks.tanh > 0.3 && risks.tanh < 4,
    `R_tanh=${risks.tanh.toFixed(3)} (notebook 1.222)`,
  );
}

// -----------------------------------------------------------------------------
// 8. §9 GD-trajectory limit (Cell [056])
//
// Notebook: ‖Π⊥ β∞‖ ≈ 1e-15 (lemma 1), ‖β∞ − β̂†‖ ≈ 1e-15 (theorem 2) at γ=4.
// Our closed-form trajectory uses t = a very large iteration count; the
// dist-to-min-norm should be < 1e-8 at that grid point.
// -----------------------------------------------------------------------------
console.log('\n--- §9 GD-trajectory limit (Cell [056]) ---');
{
  const rng = mulberry32(99);
  const gauss = gaussianRng(rng);
  const n = 25;
  const p = 100;
  const X = new Float64Array(n * p);
  fillIsotropicGaussian(X, n, p, gauss);
  const betaStar = sampleSphereBetaStar(p, 1, gauss);
  const y = new Float64Array(n);
  matVec(X, betaStar, n, p, y);
  for (let i = 0; i < n; i++) y[i] += 1.0 * gauss();
  const iters = new Int32Array([0, 1, 10, 100, 1000, 10000]);
  const traj = gdTrajectory({ X, y, n, p, betaStar, iters });
  const lastDist = traj.distToMinNorm[traj.iters.length - 1];
  ok(
    '§9 dist(β_t, β̂†) → 0 at t=10000',
    lastDist < 1e-6,
    `dist@t=10000 = ${lastDist.toExponential(3)} (< 1e-6 expected)`,
  );
  const lastNorm = traj.normBeta[traj.iters.length - 1];
  ok(
    '§9 ‖β_∞‖² ≈ ‖β̂†‖²',
    approxRel(lastNorm, traj.betaMinNormNormSq, 1e-6),
    `‖β_∞‖²=${lastNorm.toFixed(4)} vs ‖β̂†‖²=${traj.betaMinNormNormSq.toFixed(4)}`,
  );
  // Training loss should fall by orders of magnitude.
  ok(
    '§9 training loss drops below 1e-6 by t=10000',
    traj.trainLoss[traj.iters.length - 1] < 1e-6,
    `loss@t=10000 = ${traj.trainLoss[traj.iters.length - 1].toExponential(3)}`,
  );
  // First iteration: ‖β_0‖ = 0
  ok('§9 ‖β_0‖² = 0', traj.normBeta[0] === 0, `‖β_0‖² = ${traj.normBeta[0]}`);
}

// -----------------------------------------------------------------------------
// 9. §11 eigenvalue spectra and condition numbers (Cell [067])
//
// Notebook printed: κ_isotropic ≈ 27.7, κ_ReLU ≈ 1190, κ_Legendre ≈ 2.84e5.
// We assert order-of-magnitude bounds.
// -----------------------------------------------------------------------------
console.log('\n--- §11 eigenvalue spectra (Cell [067]) ---');
{
  const rng = mulberry32(42);
  const gauss = gaussianRng(rng);
  const n = 50;
  const p = 100;

  // Isotropic Gaussian
  const Xiso = new Float64Array(n * p);
  fillIsotropicGaussian(Xiso, n, p, gauss);
  const Siso = thinSVD(Xiso, n, p).S;
  const kappaIso = conditionNumber(Siso);
  ok('§11 κ(isotropic) ∈ [3, 200]', kappaIso > 3 && kappaIso < 200, `κ=${kappaIso.toFixed(1)} (notebook 27.7)`);

  // ReLU random features
  const d = 8;
  const Xinp = new Float64Array(n * d);
  fillIsotropicGaussian(Xinp, n, d, gauss);
  const W = new Float64Array(p * d);
  fillIsotropicGaussian(W, p, d, gauss);
  const Phi = randomFeatureMap(Xinp, n, d, W, p, activations.relu);
  const Sphi = thinSVD(Phi, n, p).S;
  const kappaRelu = conditionNumber(Sphi);
  ok(
    '§11 κ(ReLU RF) >> κ(isotropic)',
    kappaRelu > 50 && kappaRelu > kappaIso * 2,
    `κ=${kappaRelu.toFixed(1)} (notebook 1190)`,
  );

  // Legendre polynomial features on uniform X ∈ [-1, 1]
  const Xleg = new Float64Array(n);
  for (let i = 0; i < n; i++) Xleg[i] = 2 * Math.random() - 1; // non-seeded OK for κ ordering
  const Vleg = legendreVandermonde(Xleg, p);
  const Sleg = thinSVD(Vleg, n, p).S;
  const kappaLeg = conditionNumber(Sleg);
  ok(
    '§11 κ(Legendre) >> κ(ReLU)',
    kappaLeg > 1e3 && kappaLeg > kappaRelu,
    `κ=${kappaLeg.toExponential(2)} (notebook 2.84e5)`,
  );
}

// -----------------------------------------------------------------------------
// 10. §6 analytic vs MC at γ=4
//
// Empirically at γ=4 the MC mean was 1.08, analytic Hastie predicts 1.083.
// We check both sides are within 25% — modest sample, modest tolerance.
// -----------------------------------------------------------------------------
console.log('\n--- §6 analytic-vs-MC at γ=4 (Cells [038]-[039]) ---');
{
  const n = 50;
  const p = 200;
  const grid = new Int32Array([p]);
  // Well-specified: β* ∈ ℝᵖ with ‖β*‖²=1.
  const rng = mulberry32(2024);
  const gauss = gaussianRng(rng);
  const betaStar = sampleSphereBetaStar(p, 1, gauss);
  const out = excessRiskSweep({
    n,
    pMax: p,
    pGrid: grid,
    betaStarFull: betaStar,
    sigma: 1,
    B: 20,
    rng: mulberry32(99),
  });
  const Rmc = out.mean[0];
  const Ran = hastieRisk(p / n, 1, 1);
  const rel = Math.abs(Rmc - Ran) / Ran;
  ok(
    '§6 |R_MC − R_analytic| / R_analytic < 0.25 at γ=4',
    rel < 0.25,
    `R_MC=${Rmc.toFixed(4)}, R_analytic=${Ran.toFixed(4)}, rel=${rel.toFixed(3)}`,
  );
}

// -----------------------------------------------------------------------------
// Sanity: excessRiskMisspecified equals ‖β̂ − β*‖² when p == P_max.
// -----------------------------------------------------------------------------
console.log('\n--- excessRiskMisspecified well-specified equivalence ---');
{
  const beta1 = new Float64Array([1, 2, 3]);
  const betaStar = new Float64Array([0.5, 2.5, 2.5]);
  const r = excessRiskMisspecified(beta1, betaStar, 3);
  const expected = 0.25 + 0.25 + 0.25;
  ok('excessRiskMisspecified', approxRel(r, expected, 1e-12), `r=${r}, expect ${expected}`);
}

// -----------------------------------------------------------------------------
// Summary
// -----------------------------------------------------------------------------
console.log(`\n=== summary: ${failures.length} failure(s) ===`);
if (failures.length > 0) {
  for (const f of failures) console.log(`  ✗ ${f}`);
  process.exit(1);
}
process.exit(0);
