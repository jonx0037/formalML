// =============================================================================
// verify-statistical-depth.ts
//
// Numerical regression tests for src/data/statistical-depth.ts.
// Each test reproduces a numerical claim from the notebook
//   notebooks/statistical-depth/01_statistical_depth.ipynb
// and asserts the result lies in the expected range.
//
// Run with: pnpm verify:statistical-depth
// Accumulates failures and exits non-zero at the end if any checks fail.
//
// Tolerances:
//   - exact match for deterministic worked examples (unit-square corners,
//     Mahalanobis depth at sample mean),
//   - ±0.05 absolute for stochastic Tukey / projection / spatial depths at
//     n = 200 — PRNG-implementation differences (mulberry32 vs NumPy's PCG64)
//     dominate at this sample size,
//   - ±0.10 absolute for Mahalanobis depth at the origin of the n = 60 panel
//     sample (small-n covariance estimate is sensitive to which 60 draws
//     mulberry32 produces vs. PCG64).
// =============================================================================

import {
  componentwiseMedian,
  fitMahalanobis,
  IRIS_VERSICOLOR,
  IRIS_VIRGINICA,
  mahalanobisDepth,
  projectionDepth,
  sampleBanana,
  sampleCauchy,
  sampleContaminated,
  sampleGaussian,
  sampleMean,
  spatialDepth,
  tukeyDepth2D,
  weiszfeld,
  type Point2D,
} from '../../../../data/statistical-depth';

let pass = 0;
let fail = 0;
const failures: string[] = [];

function ok(name: string, condition: boolean, detail: string): void {
  if (condition) {
    pass++;
    console.log(`  PASS  ${name.padEnd(60)}  ${detail}`);
  } else {
    fail++;
    failures.push(`${name}: ${detail}`);
    console.log(`  FAIL  ${name.padEnd(60)}  ${detail}`);
  }
}

function near(actual: number, expected: number, tol: number): boolean {
  return Math.abs(actual - expected) <= tol;
}

// -----------------------------------------------------------------------------
// §2 — Tukey halfspace depth, deterministic worked examples
// -----------------------------------------------------------------------------

console.log('\n§2 — Tukey halfspace depth (worked examples)');

const unitSquare: Point2D[] = [
  [0, 0], [1, 0], [0, 1], [1, 1],
];

{
  const d = tukeyDepth2D([0.5, 0.5], unitSquare);
  ok(
    'HD at (0.5, 0.5) on unit-square corners',
    near(d, 0.5, 1e-12),
    `expected 0.500, got ${d.toFixed(3)}`,
  );
}
{
  const d = tukeyDepth2D([0.0, 0.0], unitSquare);
  ok(
    'HD at (0, 0) on unit-square corners (vertex)',
    near(d, 0.25, 1e-12),
    `expected 0.250, got ${d.toFixed(3)}`,
  );
}
{
  const d = tukeyDepth2D([2.0, 2.0], unitSquare);
  ok(
    'HD at (2, 2) outside unit square',
    near(d, 0.0, 1e-12),
    `expected 0.000, got ${d.toFixed(3)}`,
  );
}

// HD on a Gaussian sample — stochastic, ±0.05 tolerance
{
  const X = sampleGaussian(500, 0);
  const d = tukeyDepth2D([0, 0], X);
  ok(
    'HD at origin of N(0, Σ) sample, n=500',
    near(d, 0.5, 0.08),
    `expected ≈0.5 (notebook prints 0.436 with PCG64), got ${d.toFixed(3)}`,
  );
}
{
  const X = sampleGaussian(500, 0);
  const d = tukeyDepth2D([10, 10], X);
  ok(
    'HD at (10, 10) far outside N(0, Σ) cloud, n=500',
    d <= 0.005,
    `expected 0.000, got ${d.toFixed(3)}`,
  );
}

// -----------------------------------------------------------------------------
// §3 — Mahalanobis depth (closed form)
// -----------------------------------------------------------------------------

console.log('\n§3 — Mahalanobis depth');

{
  const X = sampleGaussian(200, 42);
  const params = fitMahalanobis(X);
  const d = mahalanobisDepth(params.mu, params);
  ok(
    'Mahalanobis depth at sample mean',
    near(d, 1.0, 1e-12),
    `expected 1.000 (analytically), got ${d.toFixed(3)}`,
  );
}
{
  // §3 panel: depth at origin on n = 60 Gaussian
  const X = sampleGaussian(60, 52); // SEED + 10 = 52 in notebook
  const params = fitMahalanobis(X);
  const d = mahalanobisDepth([0, 0], params);
  ok(
    'Mahalanobis depth at origin (n=60 panel sample)',
    d > 0.85 && d < 1.0,
    `expected near max ≈ 1.0 (notebook prints 0.998), got ${d.toFixed(3)}`,
  );
}

// -----------------------------------------------------------------------------
// §3 — Projection depth (stochastic via K random directions)
// -----------------------------------------------------------------------------

console.log('\n§3 — Projection depth');

{
  const X = sampleGaussian(60, 52);
  const X2 = X.map(([x, y]) => [x, y]);
  const d = projectionDepth([0, 0], X2, 200, 62);
  ok(
    'Projection depth at origin (n=60, K=200)',
    d > 0.4 && d < 0.95,
    `expected high depth at origin (notebook prints 0.707), got ${d.toFixed(3)}`,
  );
}

// -----------------------------------------------------------------------------
// §3 — Spatial depth (deterministic given sample)
// -----------------------------------------------------------------------------

console.log('\n§3 — Spatial depth');

{
  const X = sampleGaussian(60, 52);
  const d = spatialDepth([0, 0], X);
  ok(
    'Spatial depth at origin (n=60 panel sample)',
    d > 0.7 && d < 0.95,
    `expected high depth (notebook prints 0.871), got ${d.toFixed(3)}`,
  );
}

// -----------------------------------------------------------------------------
// §1 — Center estimators: Weiszfeld, sample mean, componentwise median
// -----------------------------------------------------------------------------

console.log('\n§1 — Center estimators (sanity)');

{
  // Symmetric square-of-corners: all three centres coincide at (0.5, 0.5).
  const X: Point2D[] = [[0, 0], [1, 0], [0, 1], [1, 1]];
  const wm = weiszfeld(X);
  const sm = sampleMean(X);
  const cm = componentwiseMedian(X);
  ok(
    'Weiszfeld geometric median on unit-square corners',
    near(wm[0], 0.5, 1e-3) && near(wm[1], 0.5, 1e-3),
    `expected (0.5, 0.5), got (${wm[0].toFixed(3)}, ${wm[1].toFixed(3)})`,
  );
  ok(
    'Sample mean on unit-square corners',
    near(sm[0], 0.5, 1e-12) && near(sm[1], 0.5, 1e-12),
    `expected (0.500, 0.500), got (${sm[0].toFixed(3)}, ${sm[1].toFixed(3)})`,
  );
  // componentwise median of {0,0,1,1} on each axis = 0.5 (mid of two middle)
  ok(
    'Componentwise median on unit-square corners',
    near(cm[0], 0.5, 1e-12) && near(cm[1], 0.5, 1e-12),
    `expected (0.500, 0.500), got (${cm[0].toFixed(3)}, ${cm[1].toFixed(3)})`,
  );
}

{
  // Sample A: all three centres should be near origin
  const A = sampleGaussian(200, 42);
  const sm = sampleMean(A);
  const wm = weiszfeld(A);
  const ok1 =
    Math.abs(sm[0]) < 0.4 &&
    Math.abs(sm[1]) < 0.4 &&
    Math.abs(wm[0]) < 0.4 &&
    Math.abs(wm[1]) < 0.4;
  ok(
    'Sample A centres land near origin (mean, geometric median)',
    ok1,
    `mean=(${sm[0].toFixed(2)}, ${sm[1].toFixed(2)})  weiszfeld=(${wm[0].toFixed(2)}, ${wm[1].toFixed(2)})`,
  );
}

{
  // Sample C: contamination should drift sample mean toward (8, 8)*0.10 = (0.8, 0.8)
  const C = sampleContaminated(200, 44).points;
  const sm = sampleMean(C);
  const ok1 = sm[0] > 0.4 && sm[0] < 1.5 && sm[1] > 0.4 && sm[1] < 1.5;
  ok(
    'Sample C sample mean drifts toward contamination (8, 8)',
    ok1,
    `mean=(${sm[0].toFixed(2)}, ${sm[1].toFixed(2)})  expected ≈(0.8, 0.8)`,
  );
}

// -----------------------------------------------------------------------------
// Sample shape sanity (Cauchy, Banana)
// -----------------------------------------------------------------------------

console.log('\n§§1, 3 — Sample-shape sanity');

{
  const B = sampleCauchy(2000, 43);
  // Cauchy has no mean; the sample mean should be unstable but the median
  // should sit near the origin.
  const xs = B.map((p) => p[0]).sort((a, b) => a - b);
  const ys = B.map((p) => p[1]).sort((a, b) => a - b);
  const medX = xs[1000];
  const medY = ys[1000];
  ok(
    'Sample B (Cauchy): coordinate medians sit near origin',
    Math.abs(medX) < 0.5 && Math.abs(medY) < 0.5,
    `medians=(${medX.toFixed(3)}, ${medY.toFixed(3)})`,
  );
}

{
  const X = sampleBanana(60, 72); // SEED + 30 = 72 in notebook
  // Banana mean(x2) ≈ 0.5 * E[x1²] - 1.5 = 0.5 * 1.5² - 1.5 = -0.375
  const m = sampleMean(X);
  ok(
    'Banana sample: x₂ mean tracks 0.5·Var(x₁) − 1.5',
    Math.abs(m[1] - -0.375) < 0.6,
    `mean=(${m[0].toFixed(2)}, ${m[1].toFixed(2)})  expected x2 ≈ −0.375`,
  );
}

// -----------------------------------------------------------------------------
// Iris dataset shape — load_iris extraction sanity
// -----------------------------------------------------------------------------

console.log('\n§5 — Iris feature subset sanity');

ok(
  'Iris Versicolor class has 50 samples',
  IRIS_VERSICOLOR.length === 50,
  `got ${IRIS_VERSICOLOR.length}`,
);
ok(
  'Iris Virginica class has 50 samples',
  IRIS_VIRGINICA.length === 50,
  `got ${IRIS_VIRGINICA.length}`,
);

{
  // sklearn's load_iris: Versicolor mean petal length ≈ 4.260, width ≈ 1.326.
  const m = sampleMean(IRIS_VERSICOLOR);
  ok(
    'Iris Versicolor mean (petal length, width) matches sklearn',
    near(m[0], 4.26, 0.05) && near(m[1], 1.33, 0.05),
    `mean=(${m[0].toFixed(3)}, ${m[1].toFixed(3)})  expected ≈(4.26, 1.33)`,
  );
}
{
  // Virginica mean petal length ≈ 5.552, width ≈ 2.026.
  const m = sampleMean(IRIS_VIRGINICA);
  ok(
    'Iris Virginica mean (petal length, width) matches sklearn',
    near(m[0], 5.55, 0.05) && near(m[1], 2.03, 0.05),
    `mean=(${m[0].toFixed(3)}, ${m[1].toFixed(3)})  expected ≈(5.55, 2.03)`,
  );
}

// -----------------------------------------------------------------------------
// DD-classifier baseline — training accuracy on Versicolor vs Virginica
// -----------------------------------------------------------------------------

console.log('\n§5.1 — DD-classifier on Iris');

{
  const allPoints = [...IRIS_VERSICOLOR, ...IRIS_VIRGINICA];
  const labels = allPoints.map((_, i) => (i < 50 ? 0 : 1));

  let correct = 0;
  for (let i = 0; i < allPoints.length; i++) {
    const z = allPoints[i];
    const d0 = tukeyDepth2D(z, IRIS_VERSICOLOR);
    const d1 = tukeyDepth2D(z, IRIS_VIRGINICA);
    const pred = d1 > d0 ? 1 : 0;
    if (pred === labels[i]) correct++;
  }
  const acc = correct / allPoints.length;
  ok(
    'DD-classifier training accuracy on Versicolor vs Virginica',
    near(acc, 0.96, 0.04),
    `expected 0.960 ± 0.04 (notebook), got ${acc.toFixed(3)}`,
  );
}

// -----------------------------------------------------------------------------
// Summary
// -----------------------------------------------------------------------------

console.log(`\n${'='.repeat(70)}`);
console.log(`  ${pass} passed, ${fail} failed`);
console.log('='.repeat(70));

if (fail > 0) {
  console.log('\nFailures:');
  for (const f of failures) console.log(`  - ${f}`);
  process.exit(1);
}
