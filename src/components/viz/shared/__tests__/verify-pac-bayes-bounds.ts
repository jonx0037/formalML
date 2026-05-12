// =============================================================================
// verify-pac-bayes-bounds.ts
//
// Numerical regression tests for src/components/viz/shared/pac-bayes-bounds.ts.
// Each test either reproduces a closed-form identity OR checks a notebook-
// verified bound value against the corresponding TS certificate formula.
//
// Run with: pnpm verify:pac-bayes-bounds
// Exits non-zero if any test fails.
//
// Notebook-verified inputs (seed 20260511 in NumPy PCG64; values printed by
// notebooks/pac-bayes-bounds/01_pac_bayes_bounds.ipynb §§5.5/7.5/10.2):
//   §10 running example (n=200, δ=0.05):
//     Q_narrow:  KL = 2.28,  E_Q[Rhat] = 0.064  ->  McAllester bound = 0.211
//     Q_broad:   KL = 0.67,  E_Q[Rhat] = 0.167  ->  McAllester bound = 0.299
//     Catoni-optimized closed-form slack at Q_narrow: 0.1148  ->  cert = 0.179
//   §5.5 table @ (n=500, KL=3.0, δ=0.05, p_hat=0.05):
//     McAllester cert = 0.149,  Seeger cert = 0.105
//   §7.5 Bernstein vs Hoeffding @ (n=200, KL=2.2, δ=0.05, p_hat=0.064):
//     Catoni-Hoeffding slack = 0.1148,  Tolstikhin-Seldin slack = 0.1562
//     (K=10 grid for the union over candidate variance bounds)
//   §7.5 Seeger cert @ (n=200, KL=2.2, δ=0.05, p_hat=0.05): 0.14
// =============================================================================

import {
  binaryKL,
  catoniDecomposition,
  catoniHoeffdingSlack,
  catoniOptimalLambda,
  catoniOptimizedCertificate,
  empiricalRiskOnGrid,
  finiteClassUnionBound,
  gaussianKLIsotropic,
  gaussianKLPerCoord,
  gaussianOnGridPosterior,
  gibbsOnGrid,
  klDiscrete,
  klInversionUpper,
  logExpEnvelope,
  logSumExp,
  mcAllesterCertificate,
  mcAllesterSlack,
  mulberry32,
  normalCdf,
  sampleNormalThresholdProblem,
  seegerCertificate,
  thresholdGrid,
  tolstikhinSeldinSlack,
  trueRiskOnGrid,
  variationalFunctional,
} from '../pac-bayes-bounds';

// -----------------------------------------------------------------------------
// Test plumbing (same shape as verify-generalization-bounds.ts)
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
// [1] Closed-form identities: binary-KL, KL inversion, normal CDF
// =============================================================================

console.log('\n[1] Closed-form identities — binary KL, kl-inversion, Φ(x)');

approxEq('binaryKL boundary p=0, q=0.1', binaryKL(0, 0.1), -Math.log(0.9), 1e-14, 'kl(0,0.1)');
approxEq('binaryKL boundary p=0, q=0.5', binaryKL(0, 0.5), -Math.log(0.5), 1e-14, 'kl(0,0.5)');
approxEq('binaryKL boundary p=1, q=0.5', binaryKL(1, 0.5), -Math.log(0.5), 1e-14, 'kl(1,0.5)');
approxEq('binaryKL symmetric kl(0.3||0.3)=0', binaryKL(0.3, 0.3), 0, 1e-14, 'kl identity');
// Coincident-degenerate identity cases — Bernoulli(0) ≡ Bernoulli(0), Bernoulli(1) ≡ Bernoulli(1).
approxEq('binaryKL coincident kl(0||0)=0', binaryKL(0, 0), 0, 1e-14, 'identity at p=q=0');
approxEq('binaryKL coincident kl(1||1)=0', binaryKL(1, 1), 0, 1e-14, 'identity at p=q=1');
// Support-mismatch boundary — Q's mass disappears where P has support.
ok('binaryKL support mismatch kl(0.5||0)=∞', binaryKL(0.5, 0) === Number.POSITIVE_INFINITY, 'kl(0.5,0) infinite');
ok('binaryKL support mismatch kl(0.5||1)=∞', binaryKL(0.5, 1) === Number.POSITIVE_INFINITY, 'kl(0.5,1) infinite');
ok('binaryKL support mismatch kl(0||1)=∞', binaryKL(0, 1) === Number.POSITIVE_INFINITY, 'kl(0,1) infinite');
ok('binaryKL support mismatch kl(1||0)=∞', binaryKL(1, 0) === Number.POSITIVE_INFINITY, 'kl(1,0) infinite');
approxEq(
  'binaryKL kl(0.5||0.6) exact closed form',
  binaryKL(0.5, 0.6),
  0.5 * Math.log(0.5 / 0.6) + 0.5 * Math.log(0.5 / 0.4),
  1e-14,
  'kl(0.5,0.6)',
);
// Sanity: kl(p,q) ≥ 2(p-q)² (Pinsker) is a strict inequality here.
ok(
  'Pinsker inequality on (0.5, 0.6)',
  binaryKL(0.5, 0.6) > 2 * 0.1 * 0.1,
  `kl=${binaryKL(0.5, 0.6).toFixed(6)} > 0.020`,
);

approxEq('klInversionUpper(0, 0.5) closed form', klInversionUpper(0, 0.5), 1 - Math.exp(-0.5), 1e-9, '1 - e^{-0.5}');
approxEq('klInversionUpper(0, 1.0)', klInversionUpper(0, 1.0), 1 - Math.exp(-1.0), 1e-9, '1 - e^{-1}');
approxEq('klInversionUpper vacuous detection', klInversionUpper(0.5, 10), 1, 1e-14, 'huge c returns 1');
approxEq('klInversionUpper round-trip', binaryKL(0.05, klInversionUpper(0.05, 0.0427)), 0.0427, 1e-10, 'kl(p,q*)=c');

approxEq('normalCdf(0) = 0.5', normalCdf(0), 0.5, 1e-6, 'Φ(0)');
approxEq('normalCdf(1.96)', normalCdf(1.96), 0.9750021, 5e-6, 'Φ(1.96)');
approxEq('normalCdf(-1.96)', normalCdf(-1.96), 0.0249979, 5e-6, 'Φ(-1.96)');

// =============================================================================
// [2] Donsker–Varadhan envelope saturation (§3.3 / §3.5)
// =============================================================================

console.log('\n[2] DV envelope saturation: g(Q*) === log E_P[e^f]');

{
  // Take P uniform on a 101-point grid, f some bounded function.
  const K = 101;
  const P = new Float64Array(K).fill(1 / K);
  const grid = thresholdGrid();
  const f = new Float64Array(K);
  for (let i = 0; i < K; i++) f[i] = -50 * (grid[i] - 0.3) ** 2; // peaked at μ*=0.3

  // Build Q* explicitly via gibbsOnGrid with risks = -f (since Gibbs is ∝ P·e^(-λ·risks)
  // and we want Q* ∝ P·e^f, so set λ=1 and risks=-f).
  const negF = new Float64Array(K);
  for (let i = 0; i < K; i++) negF[i] = -f[i];
  const Qstar = gibbsOnGrid(P, negF, 1);

  const envelope = logExpEnvelope(P, f);
  const gAtQstar = variationalFunctional(Qstar, P, f);

  approxEq('§3.5 DV saturation', gAtQstar, envelope, 1e-10, 'g(Q*) - log E_P[e^f]');

  // Build a non-Gibbs Q (uniform) and confirm the gap is strictly positive.
  const gAtP = variationalFunctional(P, P, f);
  ok('§3.4 strict gap when Q ≠ Q*', envelope - gAtP > 1e-3, `envelope - g(P) = ${(envelope - gAtP).toFixed(6)} > 0`);

  // Gibbs distribution sums to 1
  let sum = 0;
  for (let i = 0; i < K; i++) sum += Qstar[i];
  approxEq('gibbsOnGrid sums to 1', sum, 1, 1e-12, 'Σ Q*');
}

// =============================================================================
// [3] Certificate formulas at notebook-verified inputs
// =============================================================================

console.log('\n[3] Certificate formulas at notebook-verified Q_narrow / Q_broad inputs');

// §10.2 verified: Q_narrow has KL=2.28, E_Q[Rhat]=0.064, McAllester bound = 0.211.
approxEq(
  '§10.2 McAllester @ Q_narrow',
  mcAllesterCertificate(0.064, 2.28, 200, 0.05),
  0.211,
  5e-3,
  'McAllester(0.064, 2.28, 200, 0.05)',
);
approxEq(
  '§10.2 McAllester slack only',
  mcAllesterSlack(2.28, 200, 0.05),
  0.147,
  2e-3,
  'slack',
);

// §10.2 verified: Q_broad has KL=0.67, E_Q[Rhat]=0.167, McAllester bound = 0.299.
approxEq(
  '§10.2 McAllester @ Q_broad',
  mcAllesterCertificate(0.167, 0.67, 200, 0.05),
  0.299,
  5e-3,
  'McAllester(0.167, 0.67, 200, 0.05)',
);

// §6.4 Catoni-optimized closed form at Q_narrow.  Notebook prints slack = 0.1148.
approxEq(
  '§6.4 Catoni-Hoeffding slack @ Q_narrow',
  catoniHoeffdingSlack(2.28, 200, 0.05),
  0.1148,
  5e-4,
  'sqrt((KL+log(1/δ))/(2n))',
);
approxEq(
  '§6.4 Catoni-optimized cert @ Q_narrow',
  catoniOptimizedCertificate(0.064, 2.28, 200, 0.05),
  0.179,
  3e-3,
  'cert = 0.064 + slack',
);

// Catoni-optimal λ* closed form (§6.4).
{
  const lambdaStar = catoniOptimalLambda(2.28, 200, 0.05);
  // At this λ*, the linear-form Catoni cert should equal the optimized closed form.
  const decomp = catoniDecomposition(0.064, 2.28, 200, 0.05, lambdaStar);
  approxEq(
    '§6.4 Catoni linear @ λ* equals optimized closed form',
    decomp.total,
    catoniOptimizedCertificate(0.064, 2.28, 200, 0.05),
    1e-9,
    'linear(λ*) === optimized',
  );
  within('§6.4 λ* in plausible range', lambdaStar, 50, 200, 'lambda*');
}

// §5.5 verified table: (n=500, KL=3.0, δ=0.05, p_hat=0.05) gives McAllester=0.149, Seeger=0.105.
approxEq(
  '§5.5 McAllester @ (n=500,KL=3.0,p̂=0.05)',
  mcAllesterCertificate(0.05, 3.0, 500, 0.05),
  0.149,
  2e-3,
  'McAllester',
);
approxEq(
  '§5.5 Seeger @ (n=500,KL=3.0,p̂=0.05)',
  seegerCertificate(0.05, 3.0, 500, 0.05),
  0.1051,
  2e-3,
  'Seeger',
);

// User-prompt: Seeger at (p̂=0.05, KL=2.2, n=200, δ=0.05) ~ 0.14.
approxEq(
  '§7 Seeger @ (n=200,KL=2.2,p̂=0.05)',
  seegerCertificate(0.05, 2.2, 200, 0.05),
  0.14,
  1e-2,
  'Seeger small-n',
);

// =============================================================================
// [4] Tolstikhin–Seldin empirical-Bernstein slack
// =============================================================================

console.log('\n[4] Tolstikhin–Seldin slack at §7.5 setup');

// §7.5 verified: notebook's print line uses Q_narrow's actual KL=2.28 (not the
// §7.5 plot's nominal KL_FIXED_7=2.2 used as a stylized axis label).
// Notebook prints Catoni-Hoeffding slack = 0.1148, Tolstikhin-Seldin slack = 0.1562.
approxEq(
  '§7.5 Catoni-Hoeffding slack @ Q_narrow (KL=2.28)',
  catoniHoeffdingSlack(2.28, 200, 0.05),
  0.1148,
  5e-4,
  'sqrt((2.28+log(20))/400)',
);
approxEq(
  '§7.5 Tolstikhin–Seldin slack @ Q_narrow (KL=2.28, K=10)',
  tolstikhinSeldinSlack(0.064, 2.28, 200, 0.05, 10),
  0.1562,
  5e-3,
  'p̂(1-p̂) variance proxy',
);

// =============================================================================
// [5] Gaussian–Gaussian KL (Proposition 1, §8.2)
// =============================================================================

console.log('\n[5] Gaussian–Gaussian KL closed form');

approxEq('gaussianKL Q==P (μ=0, σ²=1, d=1)', gaussianKLIsotropic(0, 1, 1, 1), 0, 1e-14, 'KL(N(0,1)||N(0,1))');
approxEq('gaussianKL μ=1 σ²=1 σ²_P=1 d=1', gaussianKLIsotropic(1, 1, 1, 1), 0.5, 1e-14, 'KL = 0.5');
approxEq(
  'gaussianKL μ=0 σ²_Q=2 σ²_P=1 d=1',
  gaussianKLIsotropic(0, 2, 1, 1),
  0.5 * (2 - 1 - Math.log(2)),
  1e-14,
  'variance-only',
);
// Sanity: in d=10^4 with ρ_i = 0.5 across all coords, scaling matches per-coord call.
{
  const d = 10_000;
  const muQ = new Float64Array(d);
  const sigmaQ2 = new Float64Array(d).fill(0.5);
  const isoKL = gaussianKLIsotropic(0, 0.5, 1, d);
  const perKL = gaussianKLPerCoord(muQ, sigmaQ2, 1);
  approxEq('per-coord ≡ isotropic when ρ uniform', perKL, isoKL, 1e-9, 'per-coord vs isotropic');
}

// =============================================================================
// [6] Gibbs distribution properties
// =============================================================================

console.log('\n[6] Gibbs-on-grid sanity properties');

{
  const K = 50;
  const P = new Float64Array(K).fill(1 / K);
  const zeroRisks = new Float64Array(K);
  const Qzero = gibbsOnGrid(P, zeroRisks, 100);
  // λ * 0 risk = 0, so Q* ∝ P (uniform).
  let maxGap = 0;
  for (let i = 0; i < K; i++) maxGap = Math.max(maxGap, Math.abs(Qzero[i] - 1 / K));
  approxEq('zero-risks Gibbs ≡ prior', maxGap, 0, 1e-12, 'per-coord gap');

  // Increasing λ on a nonzero risk vector should concentrate Q on argmin.
  const risks = new Float64Array(K);
  for (let i = 0; i < K; i++) risks[i] = (i - 25) * (i - 25) / 100; // U-shape, min at i=25
  const QlowLam = gibbsOnGrid(P, risks, 1);
  const QhighLam = gibbsOnGrid(P, risks, 1000);
  ok(
    'higher λ concentrates on argmin',
    QhighLam[25] > QlowLam[25] && QhighLam[25] > 0.99,
    `Q_high[25]=${QhighLam[25].toFixed(4)}, Q_low[25]=${QlowLam[25].toFixed(4)}`,
  );
}

// =============================================================================
// [7] Running-example data pipeline — soft sanity (cross-RNG, wider tolerance)
// =============================================================================

console.log('\n[7] Running-example data pipeline (mulberry32 seed; cross-RNG tolerance)');

{
  // mulberry32 won't reproduce the notebook's PCG64 seed, but should land near the
  // verified Q_narrow numbers in distribution.  We test for plausible ranges only.
  const rng = mulberry32(20260511 >>> 0);
  const { X, Y } = sampleNormalThresholdProblem(200, 0.05, 0.3, rng);
  const grid = thresholdGrid();
  const risksEmp = empiricalRiskOnGrid(X, Y, grid);

  // ERM threshold should land near μ*=0.3 (within roughly ±0.4 grid steps at n=200).
  let bestJ = 0;
  for (let j = 1; j < grid.length; j++) if (risksEmp[j] < risksEmp[bestJ]) bestJ = j;
  within('ERM threshold near μ*', grid[bestJ], -0.6, 1.0, 'ERM ∈ [-0.6, 1.0]');

  // E_uniform[Rhat] ≈ 0.327 per user prompt (notebook PCG64; mulberry32 likely close).
  let avgRiskUnderUniform = 0;
  for (let j = 0; j < grid.length; j++) avgRiskUnderUniform += risksEmp[j];
  avgRiskUnderUniform /= grid.length;
  within('E_uniform[Rhat] ≈ 0.327', avgRiskUnderUniform, 0.25, 0.45, 'plausible range');

  // True-risk closed form at exactly t=μ*=0.3 should be exactly η.
  const trueRiskAtMuStar = trueRiskOnGrid(new Float64Array([0.3]), 0.05, 0.3)[0];
  approxEq('true risk at h_{μ*} = η', trueRiskAtMuStar, 0.05, 1e-14, 'R(h_{μ*}) exact');

  // Build Q_narrow centered at ERM with σ_Q=0.10 and confirm KL is in plausible range.
  const P = new Float64Array(grid.length).fill(1 / grid.length);
  const Qnarrow = gaussianOnGridPosterior(grid, grid[bestJ], 0.10, P);
  const klNarrow = klDiscrete(Qnarrow, P);
  within('KL(Q_narrow ∥ P) plausible', klNarrow, 1.5, 3.0, 'near notebook 2.28');

  // Cert formulas at this Q_narrow should land in McAllester's typical band.
  const eEmp = 0; // E_Q[Rhat] — recompute
  let er = 0;
  for (let j = 0; j < grid.length; j++) er += Qnarrow[j] * risksEmp[j];
  void eEmp;
  const mcBound = mcAllesterCertificate(er, klNarrow, 200, 0.05);
  within('McAllester @ realized Q_narrow', mcBound, 0.15, 0.30, 'plausible band');
}

// =============================================================================
// [8] Union-bound baseline (§10.1)
// =============================================================================

console.log('\n[8] §10.1 union bound');

// log(2*101/0.05) = log(4040) = 8.304; slack = sqrt(8.304/400) = 0.144.
approxEq(
  '§10.1 finite-class union bound slack',
  finiteClassUnionBound(101, 200, 0.05),
  0.144,
  1e-3,
  'sqrt(log(2|H|/δ)/(2n))',
);

// =============================================================================
// Summary
// =============================================================================

console.log(`\n${'='.repeat(72)}`);
console.log(`verify-pac-bayes-bounds  PASS=${pass}  FAIL=${fail}`);
if (fail > 0) {
  console.log('\nFAILURES:');
  for (const f of failures) console.log(`  - ${f}`);
  process.exit(1);
}
console.log('All PAC-Bayes regression tests pass.');
process.exit(0);

// Silence unused-import warnings for symbols only used transitively above.
void logSumExp;
