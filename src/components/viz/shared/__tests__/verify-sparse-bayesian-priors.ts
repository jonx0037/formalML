// =============================================================================
// verify-sparse-bayesian-priors.ts
//
// Numerical regression tests for the sparse-prior helpers added to
//   src/components/viz/shared/bayesian-ml.ts
//
// Each test reproduces a numerical claim from the verified notebook
//   notebooks/sparse-bayesian-priors/01_sparse_bayesian_priors.ipynb
// and asserts the result lies in the documented tolerance band.
//
// Run with: pnpm verify:sparse-bayesian-priors
// Exits non-zero on first failure.
// =============================================================================

import {
  bayesianLassoShrinkageMarginal,
  HORSESHOE_K,
  horseshoeBetaShrinkageDensity,
  horseshoeMarginalBound,
  horseshoeMarginalDensity,
  horseshoeShrinkageMarginal,
  regularizedHorseshoeLambdaTilde,
  ridgeShrinkageConstant,
  spikeSlabShrinkageMarginal,
} from '../bayesian-ml';

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
    console.error(`  FAIL  ${name}  ${detail}`);
  }
}

function approx(a: number, b: number, tol: number): boolean {
  return Math.abs(a - b) <= tol;
}

console.log('verify-sparse-bayesian-priors\n----------------------------------------');

// -------------------------------------------------------------------------
// §1 four-prior shrinkage profiles
// -------------------------------------------------------------------------

// Notebook §1 assertion: at |y|=8 with τ=1, horseshoe shrinkage < 0.10.
const hsKAtY8 = horseshoeShrinkageMarginal(8.0, 1.0);
ok(
  'horseshoeShrinkageMarginal vanishes at large |y|',
  hsKAtY8 < 0.1,
  `E[κ | y=8, τ=1] = ${hsKAtY8.toFixed(4)} < 0.10`,
);

// Ridge shrinkage at τ=1 is exactly 0.5.
const ridgeK = ridgeShrinkageConstant(1.0);
ok(
  'ridgeShrinkageConstant at τ=1 is 0.5',
  approx(ridgeK, 0.5, 1e-10),
  `1/(1+1²) = ${ridgeK.toFixed(6)}`,
);

// At |y|=0.5 (noise regime), horseshoe shrinks aggressively.
const hsKAtY05 = horseshoeShrinkageMarginal(0.5, 1.0);
ok(
  'horseshoeShrinkageMarginal aggressive at small |y|',
  hsKAtY05 > 0.5,
  `E[κ | y=0.5, τ=1] = ${hsKAtY05.toFixed(4)} > 0.5`,
);

// LASSO keeps a constant shrinkage gap y − E[β|y] → σ²√(2ρ), so its E[κ|y]
// decays only like 1/|y| (horseshoe: like 1/y²). Test that LASSO shrinkage
// stays well above the horseshoe's at |y| = 8.
const lassoKAtY8 = bayesianLassoShrinkageMarginal(8.0, 1.0);
ok(
  'bayesianLassoShrinkageMarginal does not vanish at large |y|',
  lassoKAtY8 > hsKAtY8 + 0.05,
  `LASSO κ=${lassoKAtY8.toFixed(4)} > HS κ=${hsKAtY8.toFixed(4)} + 0.05`,
);

// -------------------------------------------------------------------------
// §2 Polson-Scott marginal bounds
// -------------------------------------------------------------------------

// Carvalho-Polson-Scott (2010) Theorem 1 bounds, rescaled to general τ:
//   (K/(2τ)) log(1 + 4τ²/β²) ≤ p_HS(β) ≤ (K/τ) log(1 + 2τ²/β²),  K = 1/√(2π³).
const K_CPS = 1 / Math.sqrt(2 * Math.PI ** 3);
ok(
  'HORSESHOE_K = 1/√(2π³) ≈ 0.12698',
  approx(HORSESHOE_K, K_CPS, 1e-15) && approx(HORSESHOE_K, 0.12698727, 1e-7),
  `K = ${HORSESHOE_K.toFixed(8)}`,
);
const cpsLow001 = horseshoeMarginalBound(0.01, 1.0, true);
ok(
  'CPS lower bound at β=0.01 is (K/2) log(1 + 4/β²)',
  approx(cpsLow001, (K_CPS / 2) * Math.log(1 + 40000.0), 1e-5),
  `value = ${cpsLow001.toFixed(5)} (expected ${((K_CPS / 2) * Math.log(40001)).toFixed(5)})`,
);
const cpsUp001 = horseshoeMarginalBound(0.01, 1.0, false);
ok(
  'CPS upper bound at β=0.01 is K log(1 + 2/β²)',
  approx(cpsUp001, K_CPS * Math.log(1 + 20000.0), 1e-5),
  `value = ${cpsUp001.toFixed(5)} (expected ${(K_CPS * Math.log(20001)).toFixed(5)})`,
);

// Independent quadrature of p_HS(β) = ∫ N(β; 0, λ²τ²) · 2/(π(1+λ²)) dλ,
// substituting λ = e^u (smooth integrand on u ∈ [-25, 25], trapezoid rule).
function horseshoeDensityQuadrature(beta: number, tau: number): number {
  const M = 40000;
  const lo = -25;
  const hi = 25;
  const h = (hi - lo) / M;
  let acc = 0;
  for (let i = 0; i <= M; i++) {
    const u = lo + i * h;
    const lam = Math.exp(u);
    const sd = lam * tau;
    const z = beta / sd;
    const f =
      (Math.exp(-0.5 * z * z) / (sd * Math.sqrt(2 * Math.PI))) *
      (2 / (Math.PI * (1 + lam * lam))) *
      lam;
    acc += i === 0 || i === M ? 0.5 * f : f;
  }
  return acc * h;
}

// The exact density must sit inside the bounds at several β and τ, and the
// closed form (K/τ) e^x E₁(x) must agree with the quadrature.
for (const tau of [0.3, 1.0, 2.0]) {
  for (const beta of [0.01, 0.1, 0.5, 1, 2, 4, 8, 12]) {
    const pq = horseshoeDensityQuadrature(beta, tau);
    const pc = horseshoeMarginalDensity(beta, tau);
    const lo = horseshoeMarginalBound(beta, tau, true);
    const up = horseshoeMarginalBound(beta, tau, false);
    ok(
      `CPS bounds bracket p_HS at τ=${tau}, β=${beta}`,
      lo <= pq && pq <= up && Math.abs(pc - pq) <= 1e-6 * pq,
      `${lo.toExponential(4)} ≤ ${pq.toExponential(4)} ≤ ${up.toExponential(4)}; closed form ${pc.toExponential(4)}`,
    );
  }
}

// At β=8, τ=1: band [0.00385, 0.00391], exact 0.00385; Laplace(0, 1/√2)
// (ρ = 1) gives 8.63e-6, so the horseshoe tail is ≈ 446× heavier.
const hs8 = horseshoeMarginalDensity(8, 1);
const lap8 = Math.exp(-8 * Math.SQRT2) / Math.SQRT2;
ok(
  'horseshoe vs Laplace at β=8 (MDX: ~0.004 vs ~9e-6, ~450×)',
  approx(hs8, 0.003851, 2e-6) && approx(lap8, 8.63e-6, 1e-8) && approx(hs8 / lap8, 446.3, 0.5),
  `p_HS = ${hs8.toFixed(6)}, Laplace = ${lap8.toExponential(3)}, ratio = ${(hs8 / lap8).toFixed(1)}`,
);

// Tail: β² p_HS(β) → 2Kτ ≈ 0.254τ.
const tail40 = 40 * 40 * horseshoeMarginalDensity(40, 1);
ok(
  'horseshoe tail β² p_HS(β) → 2K ≈ 0.254',
  approx(tail40, 2 * K_CPS, 1e-3),
  `40² · p_HS(40) = ${tail40.toFixed(5)} (2K = ${(2 * K_CPS).toFixed(5)})`,
);

// Mass at zero (MDX §2: "roughly three times" ridge at |β| = 0.01).
const ratio001 = horseshoeMarginalDensity(0.01, 1) / (Math.exp(-0.5e-4) / Math.sqrt(2 * Math.PI));
ok(
  'horseshoe / ridge density at β=0.01 ≈ 2.97',
  approx(ratio001, 2.969, 0.005),
  `ratio = ${ratio001.toFixed(3)}`,
);

// -------------------------------------------------------------------------
// §3 spike-and-slab posterior weights
// -------------------------------------------------------------------------

// At |y|=0.5 with π=0.2, τ_slab=2: posterior should favor spike (κ ≈ 1).
const ssAt05 = spikeSlabShrinkageMarginal(0.5, 0.2, 2.0);
ok(
  'spikeSlabShrinkageMarginal favors spike at small |y|',
  ssAt05 > 0.6,
  `E[κ | y=0.5] = ${ssAt05.toFixed(4)} > 0.6`,
);

// At |y|=5 with π=0.2, τ_slab=2: posterior should favor slab (κ ≈ 1/(1+4)).
const ssAt5 = spikeSlabShrinkageMarginal(5.0, 0.2, 2.0);
const slabKappa = 1 / (1 + 4);
ok(
  'spikeSlabShrinkageMarginal converges to slab at large |y|',
  Math.abs(ssAt5 - slabKappa) < 0.05,
  `E[κ | y=5] = ${ssAt5.toFixed(4)} (slab ceiling = ${slabKappa.toFixed(4)})`,
);

// -------------------------------------------------------------------------
// §4 Beta(1/2, 1/2) horseshoe-shape identity
// -------------------------------------------------------------------------

// Beta(1/2, 1/2) density at κ=0.5 = (1/π) (0.5·0.5)^(-0.5) = (1/π) · 2 ≈ 0.6366.
const beta05 = horseshoeBetaShrinkageDensity(0.5);
ok(
  'horseshoeBetaShrinkageDensity at κ=0.5',
  approx(beta05, 2 / Math.PI, 1e-9),
  `value = ${beta05.toFixed(6)}`,
);

// Beta diverges at endpoints.
const beta01 = horseshoeBetaShrinkageDensity(0.01);
ok(
  'horseshoeBetaShrinkageDensity grows toward κ=0',
  beta01 > 3.0,
  `value at κ=0.01 = ${beta01.toFixed(4)}`,
);

// -------------------------------------------------------------------------
// §6 regularized horseshoe truncation
// -------------------------------------------------------------------------

// At small λ, λ̃ ≈ λ (no truncation effect).
const tildeSmall = regularizedHorseshoeLambdaTilde(0.5, 1.0, 5.0);
ok(
  'regularizedHorseshoeLambdaTilde ≈ λ at small λ',
  Math.abs(tildeSmall - 0.5) < 0.05,
  `λ̃(λ=0.5, τ=1, c=5) = ${tildeSmall.toFixed(4)} (expect ≈ 0.5)`,
);

// At very large λ, λ̃ approaches c/τ (slab cap on joint variance).
const tildeLarge = regularizedHorseshoeLambdaTilde(1000.0, 1.0, 2.0);
ok(
  'regularizedHorseshoeLambdaTilde caps at c/τ',
  Math.abs(tildeLarge * 1.0 - 2.0) < 0.01,
  `λ̃·τ at λ=1000 = ${(tildeLarge * 1).toFixed(4)} (slab ceiling c=2)`,
);

// Joint variance λ̃² τ² is bounded above by c² for any (λ, τ).
const jointVar = tildeLarge * tildeLarge * 1.0 * 1.0;
ok(
  'regularizedHorseshoe joint variance ≤ c²',
  jointVar <= 4.0001,
  `λ̃² τ² = ${jointVar.toFixed(4)} ≤ c² = 4.0`,
);

// -------------------------------------------------------------------------
// Summary
// -------------------------------------------------------------------------

console.log('----------------------------------------');
console.log(`Total: ${pass} pass, ${fail} fail`);

if (fail > 0) {
  console.log('\nFailures:');
  failures.forEach((f) => console.log(`  - ${f}`));
  process.exit(1);
}
process.exit(0);
