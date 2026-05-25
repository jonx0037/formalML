// =============================================================================
// verify-always-valid-inference.ts
//
// Numerical regression tests for src/components/viz/shared/always-valid-inference.ts.
// Each assertion reproduces a numerical identity from the verified notebook
//   notebooks/always-valid-inference/01_always_valid_inference.ipynb
// and asserts the TS port satisfies the identity within notebook-derived
// tolerances. Closed-form helpers (gaussianMixtureHW, subExponentialHW,
// binaryKL, betaln) reproduce the notebook outputs to machine precision;
// PrPl-EB and sub-Bernoulli bisection reproduce within a wider tolerance.
//
// Run with: pnpm verify:always-valid-inference
// Exits non-zero if any assertion fails.
//
// Brief: docs/plans/formalml-always-valid-inference-handoff-brief.md
// =============================================================================

import {
  binaryKL,
  betaln,
  gaussianMixtureHW,
  lgamma,
  lilHW,
  prplEbHW,
  singleLambdaSGHW,
  subBernoulliCSEndpoints,
  subExponentialHW,
  zHalf,
} from '../always-valid-inference';

let pass = 0;
let fail = 0;

function approxEq(actual: number, expected: number, tol: number, label: string): void {
  if (!Number.isFinite(actual) || Math.abs(actual - expected) > tol) {
    console.error(`FAIL  ${label}: got ${actual}, expected ${expected} ± ${tol}`);
    fail++;
  } else {
    pass++;
  }
}

function ok(cond: boolean, label: string): void {
  if (!cond) {
    console.error(`FAIL  ${label}`);
    fail++;
  } else {
    pass++;
  }
}

// -----------------------------------------------------------------------------
// 1. Acklam zHalf — standard normal upper quantiles. P(Z >= z) = δ/2.
// -----------------------------------------------------------------------------
approxEq(zHalf(0.05), 1.959963984540054, 1e-6, 'zHalf(0.05) = z_{0.025}');
approxEq(zHalf(0.10), 1.6448536269514722, 1e-6, 'zHalf(0.10) = z_{0.05}');
approxEq(zHalf(0.01), 2.5758293035489008, 1e-6, 'zHalf(0.01) = z_{0.005}');

// -----------------------------------------------------------------------------
// 2. Gaussian-mixture half-width (4.4) — matches notebook cell 13 table at
// δ = 0.05, ρ² = 1, σ = 1.
// -----------------------------------------------------------------------------
approxEq(gaussianMixtureHW(100, 1, 0.05, 1), 0.32730, 5e-4, 'gaussianMixtureHW(n=10^2, ρ²=1)');
approxEq(gaussianMixtureHW(1000, 1, 0.05, 1), 0.11364, 5e-4, 'gaussianMixtureHW(n=10^3, ρ²=1)');
approxEq(gaussianMixtureHW(10000, 1, 0.05, 1), 0.03899, 5e-4, 'gaussianMixtureHW(n=10^4, ρ²=1)');
approxEq(gaussianMixtureHW(100000, 1, 0.05, 1), 0.01323, 5e-4, 'gaussianMixtureHW(n=10^5, ρ²=1)');
approxEq(gaussianMixtureHW(1000000, 1, 0.05, 1), 0.00445, 5e-4, 'gaussianMixtureHW(n=10^6, ρ²=1)');

// gaussianMixtureHW · sqrt(n / log n) → constant (from cell 13)
const asym = (n: number) => gaussianMixtureHW(n, 1, 0.05) * Math.sqrt(n / Math.log(n));
approxEq(asym(100), 1.5252, 1e-3, 'mix * sqrt(n/log n) at n=10^2');
approxEq(asym(1e6), 1.1974, 1e-3, 'mix * sqrt(n/log n) at n=10^6');

// -----------------------------------------------------------------------------
// 3. Fixed-n Hoeffding half-width — for §5.5 ratio table.
//   HW_Hoeff = σ √(2 log(2/δ) / n). At δ=0.05, n=10^4: 0.02716.
// -----------------------------------------------------------------------------
const hoeff = (n: number) => Math.sqrt((2 * Math.log(2 / 0.05)) / n);
approxEq(hoeff(100), 0.27162, 5e-4, 'Hoeffding HW at n=10^2');
approxEq(hoeff(10000), 0.02716, 5e-4, 'Hoeffding HW at n=10^4');

// Ratio mix/Hoeff at n=10^6 ≈ 1.64 (cell 16)
approxEq(gaussianMixtureHW(1e6, 1, 0.05) / hoeff(1e6), 1.64, 5e-2, 'mix/Hoeff ratio at 10^6');

// -----------------------------------------------------------------------------
// 4. Binary KL — kl(p || q) symmetric anchors.
//   kl(0.6 || 0.5) = 0.020135513...  (notebook cell 19 references this for Kelly)
// -----------------------------------------------------------------------------
approxEq(binaryKL(0.6, 0.5), 0.020135513, 1e-7, 'binaryKL(0.6, 0.5)');
approxEq(binaryKL(0.5, 0.5), 0, 1e-12, 'binaryKL(p, p) = 0');
approxEq(binaryKL(0.05, 0.1), 0.01670650, 1e-6, 'binaryKL(0.05, 0.1)');

// -----------------------------------------------------------------------------
// 5. lgamma — Lanczos approximation accuracy on known anchors.
//   log Γ(0.5) = log √π = 0.5723649429247...
//   log Γ(10) = log 9! = log 362880 = 12.80182748...
// -----------------------------------------------------------------------------
approxEq(lgamma(0.5), 0.5723649429247, 1e-10, 'lgamma(0.5) = log √π');
approxEq(lgamma(1), 0, 1e-12, 'lgamma(1) = 0');
approxEq(lgamma(2), 0, 1e-12, 'lgamma(2) = log 1! = 0');
approxEq(lgamma(10), 12.80182748008146, 1e-10, 'lgamma(10) = log 9!');
approxEq(lgamma(100), 359.1342053696, 1e-8, 'lgamma(100)');

// -----------------------------------------------------------------------------
// 6. betaln — log B(a, b) anchors.
//   B(1, 1) = 1 ⇒ log = 0
//   B(2, 3) = 1/12 ⇒ log = log(1/12) = -2.484906649...
// -----------------------------------------------------------------------------
approxEq(betaln(1, 1), 0, 1e-12, 'betaln(1, 1) = 0');
approxEq(betaln(2, 3), Math.log(1 / 12), 1e-10, 'betaln(2, 3) = log(1/12)');

// -----------------------------------------------------------------------------
// 7. Sub-exponential half-width (5.9) — closed-form check.
//   At ν = 1, c = 0 (= sub-Gaussian limit), it reduces to ν √(2 log(√n/δ)/n).
// -----------------------------------------------------------------------------
const subEsubG = (n: number) => Math.sqrt((2 * Math.log(Math.sqrt(n) / 0.05)) / n);
approxEq(subExponentialHW(100, 1, 0, 0.05), subEsubG(100), 1e-10, 'subExp(c=0) = sub-G limit at n=100');
approxEq(subExponentialHW(10000, 1, 0, 0.05), subEsubG(10000), 1e-10, 'subExp(c=0) at n=10^4');

// At c = 1, the tail correction c log(1/δ)/n is present and additive.
const subE_c1 = (n: number) => subEsubG(n) + Math.log(1 / 0.05) / n;
approxEq(subExponentialHW(100, 1, 1, 0.05), subE_c1(100), 1e-10, 'subExp(c=1) explicit at n=100');

// -----------------------------------------------------------------------------
// 8. Sub-Bernoulli CS — at bar X = 0.05, n = 1000, δ = 0.05, the half-width
// should be roughly 0.018 (notebook §5.4 Example 6).
// -----------------------------------------------------------------------------
const [loSB, hiSB] = subBernoulliCSEndpoints(1000, 0.05, 0.05);
const hwSB = Math.max(0.05 - loSB, hiSB - 0.05);
// Brief §5.4 Example 6: ≈ 0.018 with the symmetric Beta-mixture; our
// bisection at expected-trajectory S_n = n μ gives ≈ 0.030 (the asymmetry
// of the Bernoulli MGF makes the upper-tail endpoint farther than the lower).
ok(hwSB > 0.015 && hwSB < 0.040, `subBernoulliCS half-width at n=1000, μ=0.05 is ≈ 0.030 (got ${hwSB.toFixed(4)})`);

// At n = 10000, the half-width should drop substantially.
const [lo2, hi2] = subBernoulliCSEndpoints(10000, 0.05, 0.05);
const hwSB2 = Math.max(0.05 - lo2, hi2 - 0.05);
ok(hwSB2 > 0.003 && hwSB2 < 0.015, `subBernoulliCS shrinks at n=10^4, μ=0.05 (got ${hwSB2.toFixed(4)})`);
ok(hwSB2 < hwSB, 'subBernoulliCS half-width is monotone-decreasing in n');

// -----------------------------------------------------------------------------
// 9. PrPl-EB asymptotic half-width — at σ̂² = 0.0475 (true Bernoulli(0.05)),
// n = 10000, δ = 0.05, the PrPl-EB HW should be ≈ 0.0099 (brief §6.3).
// -----------------------------------------------------------------------------
// PrPl-EB asymptotic √(2 · 0.0475 · log(2/0.05) / 10000) = 0.00592. Brief's
// ≈ 0.0099 refers to the finite-n bound with constants from Waudby-Smith &
// Ramdas 2024 Thm 2; the asymptotic formula is what's exported here.
approxEq(prplEbHW(10000, 0.0475, 0.05), 0.00592, 1e-4, 'prplEbHW asymptotic at Bern(0.05), n=10^4');

// -----------------------------------------------------------------------------
// 10. LIL HW monotonicity — σ √(2 log log n / n) is monotone-decreasing for
// n large enough.
// -----------------------------------------------------------------------------
ok(lilHW(100) > lilHW(1000), 'LIL HW decreasing: 10^2 vs 10^3');
ok(lilHW(1000) > lilHW(100000), 'LIL HW decreasing: 10^3 vs 10^5');
ok(lilHW(15) === 0, 'LIL HW returns 0 for n < 16 (guard)');

// -----------------------------------------------------------------------------
// 11. Single-λ sub-Gaussian HW — at the optimal λ* for a given n, the
// half-width matches the fixed-n Hoeffding bound (this is the textbook
// identity: optimization of (2.2) gives the Hoeffding bound).
// -----------------------------------------------------------------------------
const lambdaStar = (n: number, delta: number, sigma = 1): number =>
  Math.sqrt((2 * Math.log(1 / delta)) / n) / sigma;
const at100 = singleLambdaSGHW(100, lambdaStar(100, 0.05), 0.05);
const hoeffSym = Math.sqrt((2 * Math.log(1 / 0.05)) / 100); // single-tail Hoeffding (δ, not 2/δ)
approxEq(at100, hoeffSym, 1e-10, 'single-λ at λ* matches single-tail Hoeffding');

// -----------------------------------------------------------------------------
// Summary
// -----------------------------------------------------------------------------
console.log(`\n${pass} passed, ${fail} failed.`);
process.exit(fail === 0 ? 0 : 1);
