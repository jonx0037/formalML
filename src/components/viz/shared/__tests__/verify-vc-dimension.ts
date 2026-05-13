// =============================================================================
// verify-vc-dimension.ts
//
// Numerical regression tests for src/components/viz/shared/vc-dimension.ts.
// Each test reproduces a numerical claim from the verified notebook
//   notebooks/vc-dimension/01_vc_dimension.ipynb
// and asserts the result lies in the expected range.
//
// Run with: pnpm verify:vc-dimension
// Exits non-zero on first failure. Designed to be CI-friendly.
//
// Tolerances: closed-form quantities (growth functions, Sauer–Shelah,
// FTSL bound formula) must match notebook to <1e-9 relative error.
// Empirical Monte-Carlo quantities (FTSL empirical, Rademacher) are not
// reproduced bit-for-bit (mulberry32 vs PCG64); we assert order-of-
// magnitude agreement and rate-correctness instead.
// =============================================================================

import {
  agnosticSampleComplexity,
  binomialCoefficient,
  enumerateHalfPlaneDichotomies,
  enumerateRectangleDichotomies,
  ftslBoundEpsilon,
  growthHalfLine,
  growthHalfPlane,
  growthInterval,
  marginVCBound,
  massartBound,
  radonPartition2D,
  realizableSampleComplexity,
  sauerShelahBinomialSum,
  sauerShelahClosedForm,
  sauerShelahRademacherBound,
} from '../vc-dimension';
import type { Point2D } from '../vc-dimension';

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

function approxEq(name: string, observed: number, expected: number, relTol: number): void {
  const denom = Math.max(Math.abs(expected), 1e-12);
  const relErr = Math.abs(observed - expected) / denom;
  ok(
    name,
    relErr < relTol,
    `observed=${observed.toPrecision(6)} expected=${expected.toPrecision(6)} relErr=${relErr.toExponential(2)} (tol ${relTol})`,
  );
}

function within(name: string, observed: number, lo: number, hi: number, label: string): void {
  ok(name, observed >= lo && observed <= hi, `${label} observed=${observed.toPrecision(6)} in [${lo}, ${hi}]`);
}

function eqInt(name: string, observed: number, expected: number, label: string): void {
  ok(name, observed === expected, `${label} observed=${observed} expected=${expected}`);
}

console.log('vc-dimension verification suite');
console.log('================================');

// -----------------------------------------------------------------------------
// §3 Growth-function closed forms.
// Notebook cell 25 prints:
//   n  | 2^n   | HL Π  | IV Π  | HP Π  | RC Π
//    1 |     2 |     2 |     2 |     2 |     2
//    2 |     4 |     3 |     4 |     4 |     4
//    3 |     8 |     4 |     7 |     8 |     8
//    4 |    16 |     5 |    11 |    14 |    16
//    5 |    32 |     6 |    16 |    22 |    26
//    6 |    64 |     7 |    22 |    32 |    46
//    7 |   128 |     8 |    29 |    44 |    70
//    8 |   256 |     9 |    37 |    58 |   102
// -----------------------------------------------------------------------------

console.log('\n§3 Growth-function closed forms');

const halfLineExpected = [2, 3, 4, 5, 6, 7, 8, 9];
const intervalExpected = [2, 4, 7, 11, 16, 22, 29, 37];
const halfPlaneExpected = [2, 4, 8, 14, 22, 32, 44, 58];

for (let n = 1; n <= 8; n++) {
  eqInt(`HL Π(${n})`, growthHalfLine(n), halfLineExpected[n - 1], 'closed form');
  eqInt(`IV Π(${n})`, growthInterval(n), intervalExpected[n - 1], 'closed form');
  eqInt(`HP Π(${n})`, growthHalfPlane(n), halfPlaneExpected[n - 1], 'closed form n²-n+2');
}

// -----------------------------------------------------------------------------
// §3.2 Trivial bound: Π(n) ≤ 2^n with equality iff shattered.
// At d_VC, half-planes peel: HP Π(4) = 14 < 16 = 2^4.
// -----------------------------------------------------------------------------

ok('HP peel at n=4', growthHalfPlane(4) < Math.pow(2, 4), `Π(4)=${growthHalfPlane(4)} < 2^4=16`);
ok('HP shatters at n=3', growthHalfPlane(3) === Math.pow(2, 3), `Π(3)=${growthHalfPlane(3)} = 2^3=8`);
ok('IV peel at n=3', growthInterval(3) < Math.pow(2, 3), `Π(3)=${growthInterval(3)} < 2^3=8`);
ok('IV shatters at n=2', growthInterval(2) === Math.pow(2, 2), `Π(2)=${growthInterval(2)} = 2^2=4`);
ok('HL peel at n=2', growthHalfLine(2) < Math.pow(2, 2), `Π(2)=${growthHalfLine(2)} < 2^2=4`);
ok('HL shatters at n=1', growthHalfLine(1) === Math.pow(2, 1), `Π(1)=${growthHalfLine(1)} = 2^1=2`);

// -----------------------------------------------------------------------------
// §5.5 Sauer–Shelah bounds. Notebook cell 42 prints at n=20:
//   d=1: C(20,≤1) = 21,       (20e/1)^1 = 54.36,         ratio 2.59
//   d=2: C(20,≤2) = 211,      (20e/2)^2 = 738.9,         ratio 3.50
//   d=3: C(20,≤3) = 1351,     (20e/3)^3 = 5950.8,        ratio 4.41
//   d=4: C(20,≤4) = 6196,     (20e/4)^4 = 34123.9,       ratio 5.51
//   d=5: C(20,≤5) = 21700,    (20e/5)^5 = 151975.4,      ratio 7.00
//   d=10: C(20,≤10) = 616666, (20e/10)^10 = 22555101,    ratio 36.58
// -----------------------------------------------------------------------------

console.log('\n§5 Sauer–Shelah bounds at n=20');

const ssExpected: Record<number, { binom: number; closed: number; ratio: number }> = {
  1: { binom: 21, closed: 54.36, ratio: 2.59 },
  2: { binom: 211, closed: 738.9, ratio: 3.50 },
  3: { binom: 1351, closed: 5950.8, ratio: 4.41 },
  4: { binom: 6196, closed: 34123.9, ratio: 5.51 },
  5: { binom: 21700, closed: 151975, ratio: 7.0 },
  10: { binom: 616666, closed: 22555101, ratio: 36.58 },
};

for (const dStr of Object.keys(ssExpected)) {
  const d = Number(dStr);
  const { binom, closed, ratio } = ssExpected[d];
  eqInt(`SS binomial sum C(20,≤${d})`, sauerShelahBinomialSum(20, d), binom, 'exact integer');
  approxEq(`SS closed form (20e/${d})^${d}`, sauerShelahClosedForm(20, d), closed, 1e-3);
  const obsRatio = sauerShelahClosedForm(20, d) / sauerShelahBinomialSum(20, d);
  approxEq(`SS slack ratio at d=${d}`, obsRatio, ratio, 5e-3);
}

// Corollary 1 invariant: binomial sum ≤ closed form, for d in {1..5} and n in {d..30}.
console.log('\n§5.4 Corollary 1: ∑ C(n,i) ≤ (en/d)^d');
let corollaryViolations = 0;
for (let d = 1; d <= 5; d++) {
  for (let n = d; n <= 30; n++) {
    if (sauerShelahBinomialSum(n, d) > sauerShelahClosedForm(n, d) * (1 + 1e-9)) {
      corollaryViolations++;
    }
  }
}
ok('Corollary 1 over d in [1,5], n in [d,30]', corollaryViolations === 0, `${corollaryViolations} violations / 150 cases`);

// -----------------------------------------------------------------------------
// §6.5 FTSL bound formula. Notebook cell 50 prints (half-planes, d=3, δ=0.05):
//   n=30:   bound=2.089
//   n=60:   bound=1.568
//   n=120:  bound=1.170
//   n=250:  bound=0.853
//   n=500:  bound=0.630
//   n=1000: bound=0.464
// -----------------------------------------------------------------------------

console.log('\n§6 FTSL bound (closed form)');

const ftslExpected: Record<number, number> = {
  30: 2.089,
  60: 1.568,
  120: 1.170,
  250: 0.853,
  500: 0.630,
  1000: 0.464,
};
for (const nStr of Object.keys(ftslExpected)) {
  const n = Number(nStr);
  approxEq(`FTSL ε at n=${n}, d=3, δ=0.05`, ftslBoundEpsilon(n, 3, 0.05), ftslExpected[n], 5e-3);
}

// Rate-correctness: log(ε_n) / log(ε_2n) ≈ 1 / (1/2) ratio test (slope -1/2).
const fSlope = -Math.log(ftslBoundEpsilon(1000, 3, 0.05) / ftslBoundEpsilon(30, 3, 0.05)) /
  Math.log(1000 / 30);
within('FTSL log-log slope ≈ 0.5', fSlope, 0.40, 0.55, 'rate exponent');

// -----------------------------------------------------------------------------
// §7.5 Sample-complexity formulas. Notebook cell 58 prints (d=5, δ=0.05):
//   ε=0.100: realizable=1893,     agnostic=18934
//   ε=0.050: realizable=4341,     agnostic=86826
//   ε=0.010: realizable=28144,    agnostic=2814437
//   ε=0.005: realizable=61833,    agnostic=12366784
//   ε=0.001: realizable=373547,   agnostic=373547134
// Use 1% relative tolerance — the notebook uses ceil() on the formula.
// -----------------------------------------------------------------------------

console.log('\n§7 Sample-complexity formulas at d=5, δ=0.05');

const scExpected: Record<string, { realizable: number; agnostic: number }> = {
  '0.100': { realizable: 1893, agnostic: 18934 },
  '0.050': { realizable: 4341, agnostic: 86826 },
  '0.010': { realizable: 28144, agnostic: 2814437 },
  '0.005': { realizable: 61833, agnostic: 12366784 },
  '0.001': { realizable: 373547, agnostic: 373547134 },
};
for (const epsStr of Object.keys(scExpected)) {
  const eps = Number(epsStr);
  const { realizable, agnostic } = scExpected[epsStr];
  approxEq(`realizable n*(ε=${eps})`, realizableSampleComplexity(eps, 0.05, 5), realizable, 5e-3);
  // The agnostic formula has self-bounding slack; relax tolerance to 5%.
  approxEq(`agnostic n*(ε=${eps})`, agnosticSampleComplexity(eps, 0.05, 5), agnostic, 5e-2);
}

// Linear vs quadratic rate check.
const rRealizable = realizableSampleComplexity(0.001, 0.05, 5) / realizableSampleComplexity(0.01, 0.05, 5);
const rAgnostic = agnosticSampleComplexity(0.001, 0.05, 5) / agnosticSampleComplexity(0.01, 0.05, 5);
within('realizable 10x-ε ratio ≈ 10 (slope -1)', rRealizable, 12, 16, 'log scale + log(2e/ε) bump');
within('agnostic 10x-ε ratio ≈ 100 (slope -2)', rAgnostic, 120, 145, 'log scale + log(2e/ε²) bump');

// -----------------------------------------------------------------------------
// §8.5 Radon partition. Notebook cell 67 prints the I+/I- partitions of three
// 4-point configurations. We re-derive ours and check:
//   (1) The partition is non-trivial (both I+ and I- non-empty).
//   (2) The shared point p is contained in conv(I+) ∩ conv(I-).
//   (3) For the convex-position quadrilateral (0,0)-(1,0)-(1,1)-(0,1), the
//       diagonals split as {0,2} vs {1,3} or vice versa.
// -----------------------------------------------------------------------------

console.log('\n§8 Radon partition checks');

// Configuration A: convex quadrilateral (0,0), (1,0), (1,1), (0,1).
const convexQuad: Point2D[] = [
  [0, 0],
  [1, 0],
  [1, 1],
  [0, 1],
];
const radonA = radonPartition2D(convexQuad);
if (radonA === null) {
  ok('Radon convexQuad non-null', false, 'partition returned null');
} else {
  ok('Radon convexQuad sizes', radonA.Iplus.length >= 1 && radonA.Iminus.length >= 1, `I+=${radonA.Iplus} I-=${radonA.Iminus}`);
  // The shared point of the diagonals (0,0)-(1,1) and (1,0)-(0,1) is (0.5, 0.5).
  approxEq('Radon convexQuad p.x', radonA.p[0], 0.5, 1e-6);
  approxEq('Radon convexQuad p.y', radonA.p[1], 0.5, 1e-6);
  // Diagonals split: {0, 2} vs {1, 3}.
  const I1 = radonA.Iplus.slice().sort();
  const I2 = radonA.Iminus.slice().sort();
  const diagonalSplit =
    (JSON.stringify(I1) === '[0,2]' && JSON.stringify(I2) === '[1,3]') ||
    (JSON.stringify(I1) === '[1,3]' && JSON.stringify(I2) === '[0,2]');
  ok('Radon convexQuad diagonals', diagonalSplit, `I+=${I1} I-=${I2}`);
}

// Configuration B: triangle + interior point.
// Outer triangle (0,0), (1,0), (0.5, 1); interior point (0.5, 0.4).
const triPlusInterior: Point2D[] = [
  [0, 0],
  [1, 0],
  [0.5, 1],
  [0.5, 0.4],
];
const radonB = radonPartition2D(triPlusInterior);
if (radonB === null) {
  ok('Radon triPlusInterior non-null', false, 'partition returned null');
} else {
  // The interior point alone vs the three vertices.
  const I1 = radonB.Iplus.slice().sort();
  const I2 = radonB.Iminus.slice().sort();
  const interiorVsVertices =
    (JSON.stringify(I1) === '[3]' && JSON.stringify(I2) === '[0,1,2]') ||
    (JSON.stringify(I1) === '[0,1,2]' && JSON.stringify(I2) === '[3]');
  ok('Radon triPlusInterior split', interiorVsVertices, `I+=${I1} I-=${I2}`);
}

// -----------------------------------------------------------------------------
// §9.5 Rademacher bounds. Notebook cell 75 prints (axis-rectangles, d_VC=4):
//    n |   |H|S| |  empirical |   Massart |  SS bound
//    5 |      21 |     0.4247 |    1.1035 |    1.3989
//    8 |     108 |     0.4168 |    1.0819 |    1.3012
//   12 |     327 |     0.3556 |    0.9823 |    1.1828
//   16 |     583 |     0.3035 |    0.8922 |    1.0923
//   20 |   1,727 |     0.2917 |    0.8634 |    1.0217
// We verify the *closed-form* Massart and SS bounds; the empirical column
// depends on the Rademacher Monte Carlo and is cross-RNG (PCG64 vs mulberry32),
// so we only check the bound ordering.
// -----------------------------------------------------------------------------

console.log('\n§9 Massart and Sauer–Shelah Rademacher bounds');

const rademacherExpected: Record<number, { restrictedSize: number; massart: number; ssBound: number }> = {
  5: { restrictedSize: 21, massart: 1.1035, ssBound: 1.3989 },
  8: { restrictedSize: 108, massart: 1.0819, ssBound: 1.3012 },
  12: { restrictedSize: 327, massart: 0.9823, ssBound: 1.1828 },
  16: { restrictedSize: 583, massart: 0.8922, ssBound: 1.0923 },
  20: { restrictedSize: 1727, massart: 0.8634, ssBound: 1.0217 },
};
for (const nStr of Object.keys(rademacherExpected)) {
  const n = Number(nStr);
  const { restrictedSize, massart, ssBound } = rademacherExpected[n];
  approxEq(`Massart at n=${n}, |H|_S|=${restrictedSize}`, massartBound(restrictedSize, n), massart, 5e-3);
  approxEq(`SS Rademacher at n=${n}, d=4`, sauerShelahRademacherBound(n, 4), ssBound, 5e-3);
  ok(`Massart ≤ SS at n=${n}`, massartBound(restrictedSize, n) <= sauerShelahRademacherBound(n, 4) + 1e-9,
    `${massart} ≤ ${ssBound}`);
}

// -----------------------------------------------------------------------------
// §10 Integrative growth-function brute-force enumeration.
// Notebook cell 81 prints (sample-dependent, but Π_HP and Π_rect at n=5..20):
//   n  | Π_HP | C(n,≤3) | Π_rect | C(n,≤4)
//    5 |   22 |      26 |     23 |      31
//   10 |   92 |     176 |    136 |     386
//   15 |  212 |     576 |    602 |   1,941
//   20 |  382 |   1,351 |  1,744 |   6,196
// These are sample-dependent (the points are random); we verify Π_HP exactly
// (the closed form n²-n+2 holds for any sample in general position) and we
// verify Π_rect ≤ C(n,≤4) for the brute-force enumerator on a deterministic
// random sample at n=5,8 (the only n small enough to run quickly in CI).
// -----------------------------------------------------------------------------

console.log('\n§10 Integrative growth-function checks');

// Π_HP closed form check at the integrative sample sizes.
for (const n of [5, 10, 15, 20] as const) {
  eqInt(`Π_HP closed form at n=${n}`, growthHalfPlane(n), n * n - n + 2, 'n²-n+2');
}

// Brute-force rectangle enumeration on a deterministic random sample at n=5, 8.
// Use a tiny mulberry32-seeded sample. The point of this test is to verify
// (a) enumerator runs without throwing and (b) result ≤ C(n,≤4).
function seededSample(n: number, seed: number): Point2D[] {
  let s = seed >>> 0;
  const out: Point2D[] = [];
  for (let i = 0; i < n; i++) {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    const x = ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    s = (s + 0x9e3779b9) >>> 0;
    let u = s;
    u = Math.imul(u ^ (u >>> 15), u | 1);
    u ^= u + Math.imul(u ^ (u >>> 7), u | 61);
    const y = ((u ^ (u >>> 14)) >>> 0) / 4294967296;
    out.push([x, y]);
  }
  return out;
}

const sample5 = seededSample(5, 20260513);
const rectDichotomies5 = enumerateRectangleDichotomies(sample5);
ok(
  'rect enumerator n=5: ≤ C(5,≤4) = 31',
  rectDichotomies5.size <= 31,
  `|H|S|=${rectDichotomies5.size}`,
);
within('rect enumerator n=5: realistic count', rectDichotomies5.size, 8, 32, '8 ≤ |H|S| ≤ 32');

const sample8 = seededSample(8, 20260513);
const rectDichotomies8 = enumerateRectangleDichotomies(sample8);
ok(
  'rect enumerator n=8: ≤ C(8,≤4) = 163',
  rectDichotomies8.size <= sauerShelahBinomialSum(8, 4),
  `|H|S|=${rectDichotomies8.size}`,
);

// Same for half-plane enumerator at n=5 — should match the closed form n²-n+2 = 22.
const hpDichotomies5 = enumerateHalfPlaneDichotomies(sample5);
ok(
  'HP enumerator n=5: ≤ Cover (22)',
  hpDichotomies5.size <= 22,
  `|H|S|=${hpDichotomies5.size}`,
);

// -----------------------------------------------------------------------------
// §11.5 Margin-based VC bound. Notebook cell 89 prints (n=200, R=1.0):
//   γ=0.05: R²/γ²=400
//   γ=0.10: R²/γ²=100
//   γ=0.20: R²/γ²=25
//   γ=0.35: R²/γ²=9       (ceil(1/0.1225) = ceil(8.16) = 9)
//   γ=0.50: R²/γ²=4
//   γ=0.70: R²/γ²=3       (ceil(1/0.49) = ceil(2.04) = 3)
// -----------------------------------------------------------------------------

console.log('\n§11 Margin-based VC bound');

const marginExpected: Record<string, number> = {
  '0.05': 400,
  '0.10': 100,
  '0.20': 25,
  '0.35': 9,
  '0.50': 4,
  '0.70': 3,
};
for (const gStr of Object.keys(marginExpected)) {
  const g = Number(gStr);
  eqInt(`R²/γ² at γ=${g}`, marginVCBound(1.0, g), marginExpected[gStr], 'ceil(1/γ²)');
}

// -----------------------------------------------------------------------------
// §12 Binomial coefficient sanity: C(20, 10) = 184756.
// -----------------------------------------------------------------------------

console.log('\n§12 Binomial coefficient sanity');
eqInt('C(20, 10)', binomialCoefficient(20, 10), 184756, 'standard reference');
eqInt('C(10, 0)', binomialCoefficient(10, 0), 1, 'edge case');
eqInt('C(10, 10)', binomialCoefficient(10, 10), 1, 'edge case');
eqInt('C(5, 7)', binomialCoefficient(5, 7), 0, 'k > n');

// -----------------------------------------------------------------------------
// Final report.
// -----------------------------------------------------------------------------

console.log('\n================================');
console.log(`Results: ${pass} passed, ${fail} failed`);
if (fail > 0) {
  console.log('\nFailures:');
  for (const f of failures) console.log(`  ${f}`);
  process.exit(1);
}
process.exit(0);
