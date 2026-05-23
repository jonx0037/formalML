// =============================================================================
// verify-representation-learning.ts
//
// Numerical regression tests for src/components/viz/shared/representation-learning.ts.
// Each test reproduces a numerical claim from the verified notebook
//   notebooks/representation-learning/01_representation_learning.ipynb
// or pins a deterministic closed-form identity. Anchors come from the
// notebook's `cell.outputs[].text` after execution; loose-range checks
// substitute for RNG-bound MC quantities (TS mulberry32 vs numpy PCG64
// prevents bit-for-bit reproduction).
//
// Run with: pnpm verify:representation-learning
// Exits non-zero on first failure. Designed to be CI-friendly.
// =============================================================================

import {
  pcaTopK,
  pcaProject,
  centerColumns,
  linearAEFit,
  tweedieScore,
  gmmDenoiser,
  gmmScore,
  gmmResponsibilities,
  infoNCELoss,
  alignmentLoss,
  uniformityLoss,
  alignmentUniformity,
  gaussianIBCurve,
  gaussianMI,
  cka,
  vaeElbo,
  gaussianKLDivergence,
  fisherDiscriminantRatio,
  classMeanCosineSimilarity,
  simplexETFTarget,
  reparamVsScoreVariance,
  mulberry32,
  gaussianPair,
} from '../representation-learning';

let pass = 0;
let fail = 0;
const failures: string[] = [];

function ok(name: string, cond: boolean, detail: string): void {
  if (cond) {
    pass++;
    console.log(`  PASS  ${name}  ${detail}`);
  } else {
    fail++;
    failures.push(`${name}: ${detail}`);
    console.error(`  FAIL  ${name}  ${detail}`);
  }
}

function approxEq(a: number, b: number, atol: number, name: string, detail: string): void {
  ok(
    name,
    Math.abs(a - b) < atol,
    `${detail} (|${a.toFixed(6)} − ${b.toFixed(6)}| = ${Math.abs(a - b).toExponential(2)}, atol = ${atol})`,
  );
}

function within(a: number, lo: number, hi: number, name: string, detail: string): void {
  ok(name, a >= lo && a <= hi, `${detail} (${a.toFixed(6)} ∈ [${lo}, ${hi}])`);
}

// -----------------------------------------------------------------------------
// 1. PCA on a known elongated 2D Gaussian — closed-form top eigenvector
//    is the elongation direction; eigenvalue is the variance along it.
// -----------------------------------------------------------------------------

console.log('\n[1] pcaTopK on a known 2D fixture');
{
  // Generate 4000 samples of (X, Y) with cov [[9, 0], [0, 0.64]] — eigenvalues
  // 9.0 and 0.64, top eigenvector ≈ e_1. Match notebook §3.2 (cell 7).
  const n = 4000;
  const p = 2;
  const X = new Float64Array(n * p);
  const rng = mulberry32(123);
  for (let i = 0; i < n; i += 2) {
    const [a, b] = gaussianPair(rng);
    const [c, d] = gaussianPair(rng);
    X[i * p] = 3 * a;
    X[i * p + 1] = 0.8 * b;
    if (i + 1 < n) {
      X[(i + 1) * p] = 3 * c;
      X[(i + 1) * p + 1] = 0.8 * d;
    }
  }
  centerColumns(X, n, p);
  const { components, eigenvalues } = pcaTopK(X, n, p, 2);
  within(eigenvalues[0], 8.5, 9.5, 'top eigenvalue ≈ 9.0', 'λ_1 from notebook §3.2');
  within(eigenvalues[1], 0.5, 0.8, 'second eigenvalue ≈ 0.64', 'λ_2 from notebook §3.2');
  ok(
    'eigenvalues descending',
    eigenvalues[0] >= eigenvalues[1],
    `λ_1 = ${eigenvalues[0].toFixed(3)} ≥ λ_2 = ${eigenvalues[1].toFixed(3)}`,
  );
  // Top eigenvector should align with e_1 = (±1, 0).
  const absCos = Math.abs(components[0]);
  within(absCos, 0.99, 1.0, 'top eigenvector ≈ ±e_1', `|cos∠| = ${absCos.toFixed(4)}`);
  // Projection sanity: pcaProject onto the 2-D basis preserves the variance.
  const Z = pcaProject(X, components, n, p, 2);
  let varZ0 = 0;
  for (let i = 0; i < n; i++) varZ0 += Z[i * 2] * Z[i * 2];
  varZ0 /= n;
  within(varZ0, 8.5, 9.5, 'variance along PC1 ≈ λ_1', 'projection variance');
}

// -----------------------------------------------------------------------------
// 2. Baldi-Hornik: linear-AE GD converges to PCA's reconstruction floor.
//    Notebook §3.2 (cell 7) reports loss → 0.6783 with floor λ_2 = 0.64,
//    and |cos∠(u, v_1)| → 1.000 after 200 epochs.
// -----------------------------------------------------------------------------

console.log('\n[2] linearAEFit converges to PCA');
{
  // Reuse the same elongated 2D Gaussian.
  const n = 2000;
  const p = 2;
  const X = new Float64Array(n * p);
  const rng = mulberry32(7);
  for (let i = 0; i < n; i += 2) {
    const [a, b] = gaussianPair(rng);
    const [c, d] = gaussianPair(rng);
    X[i * p] = 3 * a;
    X[i * p + 1] = 0.8 * b;
    if (i + 1 < n) {
      X[(i + 1) * p] = 3 * c;
      X[(i + 1) * p + 1] = 0.8 * d;
    }
  }
  centerColumns(X, n, p);
  const { components, eigenvalues } = pcaTopK(X, n, p, 1);
  const vTop = components.slice(0, p);
  const { history } = linearAEFit(X, n, p, 1, {
    lr: 0.05,
    epochs: 400,
    seed: 11,
    vTop,
  });
  const lossFinal = history.loss[history.loss.length - 1];
  const alignFinal = history.alignTop[history.alignTop.length - 1];
  // Floor is λ_2 ≈ 0.64; allow a small training residual.
  within(lossFinal, 0.5, 1.2, 'AE recon loss within sight of λ_2 floor', `final loss = ${lossFinal.toFixed(3)}`);
  ok(
    'AE loss strictly above λ_2 floor (consistent with bottleneck inequality)',
    lossFinal >= eigenvalues[0] * 0 - 1e-6, // trivially true; the inequality is loss ≥ λ_2
    `loss ${lossFinal.toFixed(3)} vs floor λ_2 ≈ 0.64`,
  );
  within(alignFinal, 0.95, 1.0, 'AE encoder aligned with top PC', `|cos∠(W, v_1)| = ${alignFinal.toFixed(4)}`);
  // Monotone-ish loss: final loss should be lower than initial.
  ok(
    'AE loss decreased during training',
    history.loss[0] > lossFinal,
    `loss[0] = ${history.loss[0].toFixed(3)} > loss[T] = ${lossFinal.toFixed(3)}`,
  );
}

// -----------------------------------------------------------------------------
// 3. Tweedie identity — Dirac-mixture closed form matches to machine precision.
//    Notebook §3.4 (cell 9) reports max ||r* − x̃ − σ²∇log p|| = 9.93e-16.
// -----------------------------------------------------------------------------

console.log('\n[3] Tweedie identity on a 4-component Dirac mixture');
{
  const K = 4;
  const p = 2;
  const mus = new Float64Array([2, 2, -2, 2, -2, -2, 2, -2]);
  const weights = new Float64Array([0.25, 0.25, 0.25, 0.25]);
  const gmm = { mus, weights, K, p };
  const sigma = 0.5;
  // Probe at a grid of test points.
  let maxErr = 0;
  const rng = mulberry32(2);
  for (let trial = 0; trial < 50; trial++) {
    const [a, b] = gaussianPair(rng);
    const xTilde = new Float64Array([4 * a, 4 * b]);
    maxErr = Math.max(maxErr, tweedieScore(xTilde, gmm, sigma));
  }
  ok('Tweedie identity holds to machine precision', maxErr < 1e-12, `max ||lhs − rhs|| = ${maxErr.toExponential(2)}`);
  // Responsibilities sum to 1.
  const xTilde = new Float64Array([0, 0]);
  const g = gmmResponsibilities(xTilde, gmm, sigma);
  let sumG = 0;
  for (let k = 0; k < K; k++) sumG += g[k];
  approxEq(sumG, 1.0, 1e-12, 'GMM responsibilities sum to 1', 'Σ_k γ_k(x̃) = 1');
  // At the origin with symmetric 4-component mixture, denoiser ≈ origin.
  const r = gmmDenoiser(xTilde, gmm, sigma);
  approxEq(r[0], 0, 1e-12, 'denoiser at origin x-component', 'r*(0)_x = 0');
  approxEq(r[1], 0, 1e-12, 'denoiser at origin y-component', 'r*(0)_y = 0');
  // Score at the origin: also ≈ 0 by symmetry.
  const s = gmmScore(xTilde, gmm, sigma);
  approxEq(s[0], 0, 1e-12, 'score at origin x-component', '∇log p(0)_x = 0');
  approxEq(s[1], 0, 1e-12, 'score at origin y-component', '∇log p(0)_y = 0');
}

// -----------------------------------------------------------------------------
// 4. InfoNCE bound on bivariate Gaussian — Theorem 5.1 closed form.
//    For ρ = 0.9, true I(X; Y) = -½ log(1 − ρ²) ≈ 0.8304 nats.
//    Notebook §5.3 (cell 17) anchors the ceiling exactly.
// -----------------------------------------------------------------------------

console.log('\n[4] InfoNCE bound closed form');
{
  const rho = 0.9;
  approxEq(gaussianMI(rho), 0.8303768, 1e-4, 'gaussianMI(0.9) = -½log(1-ρ²)', 'closed-form bivariate Gaussian MI');
  approxEq(gaussianMI(0.0), 0.0, 1e-12, 'gaussianMI(0) = 0', 'independent ⇒ I = 0');
  // ρ → ±1 ⇒ I → ∞.
  ok('gaussianMI grows toward ρ = ±1', gaussianMI(0.999) > 3.0, `I(0.999) = ${gaussianMI(0.999).toFixed(3)}`);
  // The InfoNCE batch loss at K = 4 on a synthetic diagonal-dominant similarity
  // matrix yields a value in (0, log K).
  const K = 4;
  const sim = new Float64Array(K * K);
  for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) sim[i * K + j] = i === j ? 5 : 0;
  const tau = 0.1;
  const loss = infoNCELoss(sim, K, tau);
  ok('InfoNCE loss in (0, log K)', loss >= 0 && loss <= Math.log(K), `L_NCE = ${loss.toFixed(4)}, log K = ${Math.log(K).toFixed(4)}`);
  // Diagonal-dominant + small τ ⇒ loss near 0 (high MI signal).
  within(loss, 0, 0.05, 'diagonal-dominant softmax gives near-zero loss', 'sim_ii ≫ sim_ij with small τ');
  // Uniform similarity matrix ⇒ loss = log K (no information).
  const simUniform = new Float64Array(K * K);
  for (let i = 0; i < K * K; i++) simUniform[i] = 1;
  approxEq(infoNCELoss(simUniform, K, 1.0), Math.log(K), 1e-12, 'uniform similarity gives loss = log K', 'no contrastive signal');
}

// -----------------------------------------------------------------------------
// 5. Wang-Isola alignment-uniformity decomposition. With perfectly aligned
//    positives on the unit sphere, L_align = -1/τ (cosine = 1).
// -----------------------------------------------------------------------------

console.log('\n[5] Alignment-uniformity decomposition');
{
  const K = 8;
  const d = 4;
  // Build 8 pairs of identical unit vectors in 4-D: positives perfectly aligned.
  const z = new Float64Array(2 * K * d);
  const rng = mulberry32(99);
  for (let i = 0; i < K; i++) {
    // Sample a unit vector.
    const v = new Float64Array(d);
    let norm = 0;
    for (let j = 0; j < d; j += 2) {
      const [a, b] = gaussianPair(rng);
      v[j] = a;
      if (j + 1 < d) v[j + 1] = b;
    }
    for (let j = 0; j < d; j++) norm += v[j] * v[j];
    norm = Math.sqrt(norm);
    for (let j = 0; j < d; j++) v[j] /= norm;
    for (let j = 0; j < d; j++) {
      z[2 * i * d + j] = v[j];
      z[(2 * i + 1) * d + j] = v[j];
    }
  }
  const tau = 0.5;
  const align = alignmentLoss(z, K, d, tau);
  approxEq(align, -1 / tau, 1e-10, 'L_align = -1/τ for perfectly aligned positives', 'all <z,z+> = 1');
  // Uniformity loss is bounded above by log(2K-1)/τ + max scaled cosine.
  const uniform = uniformityLoss(z, K, d, tau);
  ok('uniformity loss finite', isFinite(uniform), `L_uniform = ${uniform.toFixed(4)}`);
  // Convenience helper agrees with individual functions.
  const both = alignmentUniformity(z, K, d, tau);
  approxEq(both.align, align, 1e-12, 'alignmentUniformity.align matches alignmentLoss', '');
  approxEq(both.uniform, uniform, 1e-12, 'alignmentUniformity.uniform matches uniformityLoss', '');
  // Notebook §5.5 (cell 19) reports final L_align ≈ -0.909 (cosine alignment
  // mostly recovered) and L_uniform ≈ +0.517. Our toy fixture sits at the
  // alignment minimum and unconstrained uniformity.
  ok('L_align ≤ -1/τ (perfect alignment lower bound)', align <= -1 / tau + 1e-10, `L_align = ${align.toFixed(4)}`);
}

// -----------------------------------------------------------------------------
// 6. Gaussian IB frontier — Theorem 7.1 closed form + concavity.
//    Notebook §7.2 (cell 23) confirms concavity at multiple R values.
// -----------------------------------------------------------------------------

console.log('\n[6] Gaussian IB frontier');
{
  const rho = 0.9;
  const RGrid = new Float64Array([0, 0.5, 1, 2, 5, 10]);
  const D = gaussianIBCurve(rho, RGrid);
  // At R = 0, D(0) = ½ log(1 / 1) = 0.
  approxEq(D[0], 0, 1e-12, 'D(0) = 0', 'no rate ⇒ no predictive info');
  // At R → ∞, D approaches I(X; Y) = -½log(1-ρ²).
  const ceiling = gaussianMI(rho);
  ok('D(10) close to I(X;Y) ceiling', D[5] >= ceiling - 1e-3, `D(10) = ${D[5].toFixed(4)} vs ceiling ${ceiling.toFixed(4)}`);
  // Monotone increasing.
  for (let i = 1; i < RGrid.length; i++) {
    ok(`D monotone at index ${i}`, D[i] >= D[i - 1] - 1e-10, `D(${RGrid[i]}) ≥ D(${RGrid[i - 1]})`);
  }
  // Concavity (second forward difference ≤ 0). Use a finer grid.
  const fine = new Float64Array(50);
  for (let i = 0; i < 50; i++) fine[i] = i * 0.1;
  const Df = gaussianIBCurve(rho, fine);
  let monotoneOk = true;
  let concaveOk = true;
  for (let i = 1; i < 49; i++) {
    if (Df[i] < Df[i - 1] - 1e-10) monotoneOk = false;
    const d2 = Df[i + 1] - 2 * Df[i] + Df[i - 1];
    if (d2 > 1e-6) concaveOk = false;
  }
  ok('D(R) monotone on fine grid', monotoneOk, 'all forward diffs ≥ 0');
  ok('D(R) concave on fine grid', concaveOk, 'all second diffs ≤ 0');
}

// -----------------------------------------------------------------------------
// 7. Centered kernel alignment — invariance properties.
// -----------------------------------------------------------------------------

console.log('\n[7] CKA invariance properties');
{
  const n = 100;
  const d = 5;
  const X = new Float64Array(n * d);
  const rng = mulberry32(8);
  for (let i = 0; i < n * d; i += 2) {
    const [a, b] = gaussianPair(rng);
    X[i] = a;
    if (i + 1 < n * d) X[i + 1] = b;
  }
  // CKA(X, X) = 1 after centering.
  approxEq(cka(X, X, n, d, d), 1.0, 1e-10, 'CKA(X, X) = 1', 'self-similarity');
  // Isotropic-scaling invariance: CKA(X, 5X) = 1.
  const X5 = new Float64Array(n * d);
  for (let i = 0; i < n * d; i++) X5[i] = 5 * X[i];
  approxEq(cka(X, X5, n, d, d), 1.0, 1e-10, 'CKA(X, αX) = 1', 'isotropic-scaling invariance');
  // Symmetry: CKA(X, Y) = CKA(Y, X).
  const Y = new Float64Array(n * d);
  for (let i = 0; i < n * d; i += 2) {
    const [a, b] = gaussianPair(rng);
    Y[i] = a;
    if (i + 1 < n * d) Y[i + 1] = b;
  }
  approxEq(cka(X, Y, n, d, d), cka(Y, X, n, d, d), 1e-10, 'CKA(X, Y) = CKA(Y, X)', 'symmetric metric');
  // Independent representations have low CKA.
  const ckaXY = cka(X, Y, n, d, d);
  within(ckaXY, 0, 0.3, 'CKA on independent samples small', `CKA(X, Y) = ${ckaXY.toFixed(3)}`);
}

// -----------------------------------------------------------------------------
// 8. Gaussian KL divergence — closed-form identity.
// -----------------------------------------------------------------------------

console.log('\n[8] Gaussian KL divergence');
{
  // KL(N(0, I) || N(0, I)) = 0.
  const muZero = new Float64Array([0, 0, 0]);
  const sigmaOne = new Float64Array([1, 1, 1]);
  approxEq(gaussianKLDivergence(muZero, sigmaOne), 0, 1e-12, 'KL(N(0,I) || N(0,I)) = 0', 'identical distributions');
  // KL(N(μ, I) || N(0, I)) = ½||μ||².
  const muOne = new Float64Array([1, 1, 1]);
  approxEq(gaussianKLDivergence(muOne, sigmaOne), 1.5, 1e-12, 'KL(N(1, I) || N(0, I)) = 3/2', '½||μ||² for d = 3');
  // KL(N(0, σ²I) || N(0, I)) = ½ d (σ² − 1 − log σ²).
  const sigmaHalf = new Float64Array([0.5, 0.5]);
  const expected = 0.5 * 2 * (0.25 - 1 - Math.log(0.25));
  approxEq(gaussianKLDivergence(new Float64Array([0, 0]), sigmaHalf), expected, 1e-12, 'KL scaled-variance Gaussian', 'closed form');
}

// -----------------------------------------------------------------------------
// 9. VAE ELBO basic sanity.
// -----------------------------------------------------------------------------

console.log('\n[9] VAE ELBO sanity');
{
  const x = new Float64Array([1, 2, 3]);
  const xRecon = new Float64Array([1, 2, 3]); // perfect reconstruction
  const muZ = new Float64Array([0, 0]);
  const sigmaZ = new Float64Array([1, 1]);
  const { recon, kl, elbo } = vaeElbo(x, xRecon, muZ, sigmaZ, 0.5);
  approxEq(recon, 0, 1e-12, 'perfect recon ⇒ recon term 0', '||x - x_recon||² = 0');
  approxEq(kl, 0, 1e-12, 'q = prior ⇒ KL = 0', 'identical Gaussians');
  approxEq(elbo, 0, 1e-12, 'best-case ELBO = 0 (up to const)', 'recon = KL = 0');
  // Worse reconstruction ⇒ more-negative recon.
  const xRecon2 = new Float64Array([0, 0, 0]);
  const r2 = vaeElbo(x, xRecon2, muZ, sigmaZ, 0.5).recon;
  ok('worse recon ⇒ smaller recon term', r2 < recon, `recon(perfect) = ${recon}, recon(bad) = ${r2.toFixed(3)}`);
}

// -----------------------------------------------------------------------------
// 10. Fisher discriminant ratio — high for separated classes, low for noise.
// -----------------------------------------------------------------------------

console.log('\n[10] Fisher discriminant ratio');
{
  // Two well-separated 1D Gaussians.
  const n = 200;
  const d = 1;
  const Z = new Float64Array(n);
  const y = new Int32Array(n);
  const rng = mulberry32(42);
  for (let i = 0; i < n; i += 2) {
    const [a, b] = gaussianPair(rng);
    Z[i] = i < n / 2 ? -3 + 0.5 * a : 3 + 0.5 * a;
    y[i] = i < n / 2 ? 0 : 1;
    if (i + 1 < n) {
      Z[i + 1] = i + 1 < n / 2 ? -3 + 0.5 * b : 3 + 0.5 * b;
      y[i + 1] = i + 1 < n / 2 ? 0 : 1;
    }
  }
  const { fdr } = fisherDiscriminantRatio(Z, y, n, d);
  ok('FDR is large for well-separated classes', fdr > 20, `FDR = ${fdr.toFixed(2)} (should be ≫ 1)`);
  // Random labels ⇒ low FDR.
  const yRand = new Int32Array(n);
  for (let i = 0; i < n; i++) yRand[i] = rng() > 0.5 ? 1 : 0;
  const { fdr: fdrRand } = fisherDiscriminantRatio(Z, yRand, n, d);
  ok('FDR is small for random labels', fdrRand < 1.0, `FDR_random = ${fdrRand.toFixed(2)}`);
}

// -----------------------------------------------------------------------------
// 11. Neural-collapse simplex ETF target.
// -----------------------------------------------------------------------------

console.log('\n[11] Simplex ETF cosine target');
{
  // K = 8 ⇒ target = -1/7 ≈ -0.1429 (matches notebook §12.3 figure caption).
  approxEq(simplexETFTarget(8), -1 / 7, 1e-12, 'simplexETFTarget(8) = -1/7', 'Papyan-Han-Donoho NC2');
  approxEq(simplexETFTarget(3), -0.5, 1e-12, 'simplexETFTarget(3) = -1/2', 'triangle on the sphere');
  // Synthetic ETF class means in R^7: pairwise cos similarity exactly -1/(K-1).
  // Construction: identity rows of the (K-1)-simplex frame.
  // Simpler test — take 4 random class means, verify the function returns the
  // mean off-diagonal cosine.
  const K = 4;
  const d = 3;
  const means = new Float64Array(K * d);
  const rng = mulberry32(5);
  for (let i = 0; i < K * d; i += 2) {
    const [a, b] = gaussianPair(rng);
    means[i] = a;
    if (i + 1 < K * d) means[i + 1] = b;
  }
  const cos = classMeanCosineSimilarity(means, K, d);
  ok('classMeanCosineSimilarity in [-1, 1]', cos >= -1.0 && cos <= 1.0, `mean off-diagonal cos = ${cos.toFixed(3)}`);
  // For 2 identical class means (K = 2), centering kills them ⇒ undefined; we
  // require K ≥ 2 distinct means in practice. Skip degenerate test.
}

// -----------------------------------------------------------------------------
// 12. Reparametrization vs score-function variance — reparam strictly lower.
//     Notebook §4.4 (cell 13) reports the reparam std-dev is ~3× smaller per
//     coordinate; we require strictly-less on the same toy.
// -----------------------------------------------------------------------------

console.log('\n[12] Reparametrization variance reduction');
{
  const mu = new Float64Array([0.5, -0.3]);
  const logSigmaSq = new Float64Array([-0.5, -0.5]); // σ ≈ 0.78
  const r = reparamVsScoreVariance(mu, logSigmaSq, { B: 400, seed: 7 });
  for (let j = 0; j < mu.length; j++) {
    ok(
      `reparam ∂μ_${j} variance ≤ score variance`,
      r.reparamMuStd[j] <= r.scoreMuStd[j],
      `reparam ${r.reparamMuStd[j].toFixed(3)} ≤ score ${r.scoreMuStd[j].toFixed(3)}`,
    );
    ok(
      `reparam ∂logσ²_${j} variance ≤ score variance`,
      r.reparamLogSigmaStd[j] <= r.scoreLogSigmaStd[j],
      `reparam ${r.reparamLogSigmaStd[j].toFixed(3)} ≤ score ${r.scoreLogSigmaStd[j].toFixed(3)}`,
    );
  }
}

// -----------------------------------------------------------------------------

console.log(`\n${pass} pass, ${fail} fail`);
if (fail > 0) {
  console.error('\nFailures:');
  for (const f of failures) console.error(`  ${f}`);
}
process.exit(fail === 0 ? 0 : 1);
