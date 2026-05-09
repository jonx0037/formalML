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
  horseshoeBetaShrinkageDensity,
  horseshoeMarginalBound,
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

// LASSO at large |y| keeps shrinking (asymptote ≈ const > 0). Test that
// LASSO doesn't vanish to 0 the way horseshoe does.
const lassoKAtY8 = bayesianLassoShrinkageMarginal(8.0, 1.0);
ok(
  'bayesianLassoShrinkageMarginal does not vanish at large |y|',
  lassoKAtY8 > hsKAtY8 + 0.05,
  `LASSO κ=${lassoKAtY8.toFixed(4)} > HS κ=${hsKAtY8.toFixed(4)} + 0.05`,
);

// -------------------------------------------------------------------------
// §2 Polson-Scott marginal bounds
// -------------------------------------------------------------------------

// CPS Eq. 7 lower bound at β=0.01, τ=1: (1/(2π)) log(1 + 4/0.01²) ≈ 1.475
const cpsLow001 = horseshoeMarginalBound(0.01, 1.0, true);
ok(
  'CPS lower bound at β=0.01',
  approx(cpsLow001, (1 / (2 * Math.PI)) * Math.log(1 + 40000.0), 1e-3),
  `value = ${cpsLow001.toFixed(4)} (expected ≈ ${(
    (1 / (2 * Math.PI)) *
    Math.log(40001)
  ).toFixed(4)})`,
);

// CPS upper bound is exactly 2× lower bound.
const cpsUp001 = horseshoeMarginalBound(0.01, 1.0, false);
ok(
  'CPS upper bound = 2 × lower bound',
  approx(cpsUp001, 2 * cpsLow001, 1e-9),
  `upper/lower = ${(cpsUp001 / cpsLow001).toFixed(4)} (expected 2.0)`,
);

// At β=4 (tail), horseshoe upper bound ≈ (1/π) log(1 + 4/16) = (1/π) log(1.25) ≈ 0.0710.
const hsTailAt4 = horseshoeMarginalBound(4.0, 1.0, false);
ok(
  'CPS upper bound at β=4 ≈ 0.0710',
  approx(hsTailAt4, (1 / Math.PI) * Math.log(1.25), 1e-3),
  `value = ${hsTailAt4.toFixed(4)}`,
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
