// =============================================================================
// verify-meta-learning.ts
//
// Numerical regression tests for src/components/viz/shared/meta-learning.ts.
// Each assertion reproduces a numerical identity from
//   notebooks/meta-learning/01_meta_learning.ipynb
// or a closed-form quantity stated in the brief, and asserts the TS port
// satisfies the identity within notebook-derived tolerances.
//
// The notebook's PyTorch training cells (cells 15, 25, 26, 35, 44) cannot be
// reproduced exactly in TS — different RNGs, different autograd, different
// numerical conditioning. The verify suite focuses on the *closed-form*
// primitives that have rigid notebook-side reference values, and on the
// algebraic identities (Theorem 6.1's exact match, FOMAML bias formula,
// Bregman-mean optimality) that are RNG-independent.
//
// Run with: pnpm verify:meta-learning
// Exits non-zero if any assertion fails.
//
// Brief: docs/plans/formalml-meta-learning-handoff-brief.md
// =============================================================================

import {
  META_SEED,
  mulberry32,
  sampleSinusoidTask,
  rbfKernel,
  choleskyLower,
  sampleProtoNetTask,
  quadraticGdIterate,
  quadraticMap,
  implicitPriorPrecision,
  mamlOuterGradient1D,
  fomamlOuterGradient1D,
  fomamlBias1D,
  reptileDirection1D,
  reptileImplicitCoeff,
  cnpEncoderForward,
  diagonalGaussianKL,
  protonetClassPrototypes,
  squaredDistance,
  nearestPrototypeSoftmax,
  amitMeirBound,
  fmoSmoothnessConstant,
  fmoVarianceBound,
  convergenceEnvelopeConstant,
} from '../meta-learning';

// -----------------------------------------------------------------------------
// Assertion helpers
// -----------------------------------------------------------------------------

let pass = 0;
let fail = 0;
const failures: string[] = [];

function ok(name: string, condition: boolean, detail: string): void {
  if (condition) {
    pass++;
    console.log(`  ✓ ${name}`);
  } else {
    fail++;
    failures.push(`${name}: ${detail}`);
    console.log(`  ✗ ${name}  —  ${detail}`);
  }
}

function approxEq(name: string, actual: number, expected: number, tol: number): void {
  const diff = Math.abs(actual - expected);
  ok(name, diff < tol, `actual=${actual.toPrecision(8)}, expected=${expected.toPrecision(8)}, |diff|=${diff.toExponential(3)} ≥ ${tol}`);
}

function within(name: string, actual: number, lo: number, hi: number): void {
  ok(name, actual >= lo && actual <= hi, `actual=${actual.toPrecision(6)} not in [${lo}, ${hi}]`);
}

// -----------------------------------------------------------------------------
// SECTION A — PRNG and task samplers (cells 3, 7, 8)
// -----------------------------------------------------------------------------

console.log('\nA. PRNG and task samplers');

{
  const rng = mulberry32(META_SEED);
  // The first uniform draw from mulberry32(20260518) is deterministic.
  const u0 = rng();
  within('A1. mulberry32 first draw in [0,1)', u0, 0, 1);

  // Sinusoid task: A in [0.1, 5], phi in [0, 2π], 5 support points
  const rng2 = mulberry32(META_SEED);
  const task = sampleSinusoidTask(rng2);
  within('A2. sinusoid A in [0.1, 5]', task.A, 0.1, 5.0);
  within('A3. sinusoid phi in [0, 2π]', task.phi, 0, 2 * Math.PI);
  ok('A4. sinusoid K=5 support pts', task.xSupport.length === 5 && task.ySupport.length === 5, `xSupport.length=${task.xSupport.length}`);
  ok('A5. sinusoid dense=200', task.xDense.length === 200 && task.yDense.length === 200, `xDense.length=${task.xDense.length}`);

  // y_support[i] = A sin(x_support[i] + phi) should hold exactly
  const recon = task.A * Math.sin(task.xSupport[0] + task.phi);
  approxEq('A6. sinusoid y = A sin(x+phi) identity', task.ySupport[0], recon, 1e-12);
}

{
  // RBF kernel matrix is symmetric, K(x, x) = 1 on diagonal, off-diagonal ≤ 1
  const x = [0, 1, 2, 3, 4];
  const K = rbfKernel(x, x, 1.0);
  approxEq('A7. RBF diagonal = 1', K[0][0], 1.0, 1e-15);
  approxEq('A8. RBF symmetric K[i][j] = K[j][i]', K[1][3], K[3][1], 1e-15);
  // K(0, 1) at ell=1: exp(-0.5 * 1 / 1) = exp(-0.5)
  approxEq('A9. RBF K(0,1) at ell=1', K[0][1], Math.exp(-0.5), 1e-12);
}

{
  // Cholesky factor: L @ L.T = A
  const A = [
    [4, 2, 0],
    [2, 5, 1],
    [0, 1, 3],
  ];
  const L = choleskyLower(A);
  // Reconstruct
  const Areco: number[][] = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      let s = 0;
      for (let k = 0; k < 3; k++) s += L[i][k] * L[j][k];
      Areco[i][j] = s;
    }
  }
  let maxDiff = 0;
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      maxDiff = Math.max(maxDiff, Math.abs(Areco[i][j] - A[i][j]));
    }
  }
  ok('A10. Cholesky L @ L.T = A within 1e-12', maxDiff < 1e-12, `max|diff|=${maxDiff}`);
}

{
  // ProtoNet task: 5 classes × 5 shot = 25 support, × 15 query = 75 query
  const rng3 = mulberry32(META_SEED + 8);
  const task = sampleProtoNetTask(rng3);
  ok('A11. ProtoNet 25 support pts', task.Xs.length === 25 && task.ys.length === 25, `Xs.length=${task.Xs.length}`);
  ok('A12. ProtoNet 75 query pts', task.Xq.length === 75 && task.yq.length === 75, `Xq.length=${task.Xq.length}`);
  ok('A13. ProtoNet 5 class means', task.means.length === 5 && task.means[0].length === 2, `means.length=${task.means.length}`);
  // Each class mean at distance radius (=2.5) from origin
  for (let c = 0; c < 5; c++) {
    const r = Math.hypot(task.means[c][0], task.means[c][1]);
    approxEq(`A14.${c}. class ${c} mean on disk r=2.5`, r, 2.5, 1e-12);
  }
  // 5 support per class
  const counts = [0, 0, 0, 0, 0];
  for (const lbl of task.ys) counts[lbl]++;
  ok('A15. 5 support per class', counts.every((c) => c === 5), `counts=[${counts.join(',')}]`);
}

// -----------------------------------------------------------------------------
// SECTION B — Theorem 6.1 (Grant et al.): MAML inner loop equals MAP under
// the implicit hyperprior. Tested in closed form on the 1D quadratic.
// -----------------------------------------------------------------------------

console.log('\nB. Theorem 6.1 (Grant et al., 1D Gaussian case)');

(() => {
  const theta0 = 1.0;
  const muMle = 0.0;
  const lambda = 1.0;
  const alpha = 0.1;

  for (const N of [1, 2, 5, 10, 20]) {
    const thetaN = quadraticGdIterate(theta0, muMle, lambda, alpha, N);
    const tauP = implicitPriorPrecision(lambda, alpha, N); // 1/σ_p²
    const sigmaP = 1.0 / Math.sqrt(tauP);
    const thetaMAP = quadraticMap(theta0, muMle, lambda, sigmaP);
    approxEq(`B1.${N}. θ_N = θ_MAP under implicit σ_p²`, thetaN, thetaMAP, 1e-12);

    // Check the precision formula: σ_p^{-2} = λ(1-αλ)^N / (1 - (1-αλ)^N)
    const decay = Math.pow(1 - alpha * lambda, N);
    const tauExpected = (lambda * decay) / (1 - decay);
    approxEq(`B2.${N}. implicit precision formula (6.5)`, tauP, tauExpected, 1e-12);
  }

  // Early-stopping regime: at N=1 the implicit precision exceeds 1/(N*alpha)
  // because the (1-αλ)^N factor only mildly compresses; verify the leading-
  // order approximation σ_p^{-2} ≈ 1/(Nα) is within 50% for N*α*λ small.
  const tauP5 = implicitPriorPrecision(1.0, 0.01, 5); // Nαλ = 0.05
  const tauP5Approx = 1.0 / (5 * 0.01);
  ok('B3. σ_p^{-2} ≈ 1/(Nα) when Nαλ ≪ 1', Math.abs(tauP5 / tauP5Approx - 1) < 0.5, `ratio=${(tauP5 / tauP5Approx).toPrecision(4)}`);

  // Boundary cases
  ok('B4. N=0 → infinite implicit precision', implicitPriorPrecision(1, 0.1, 0) === Infinity, 'N=0');
  ok('B5. αλ=0 → infinite precision (no adaptation)', implicitPriorPrecision(1, 0, 5) === Infinity, `got ${implicitPriorPrecision(1, 0, 5)}`);
})();

// -----------------------------------------------------------------------------
// SECTION C — MAML / FOMAML / Reptile outer-gradient identities
// -----------------------------------------------------------------------------

console.log('\nC. MAML / FOMAML / Reptile (closed-form 1D)');

(() => {
  // Standard 1D test point: distinct support and query quadratics
  const theta0 = 0;
  const muS = 1;
  const muQ = 2;
  const lambdaS = 1;
  const lambdaQ = 1;
  const alpha = 0.1;

  const theta1 = theta0 - alpha * lambdaS * (theta0 - muS); // 0.1
  approxEq('C1. inner step value', theta1, 0.1, 1e-12);

  // MAML = (1-αλ_S) * λ_Q * (θ_1 - μ_Q) = 0.9 * 1 * (0.1 - 2) = -1.71
  const maml = mamlOuterGradient1D(theta0, muS, muQ, lambdaS, lambdaQ, alpha);
  approxEq('C2. MAML outer gradient (2.6) at N=1', maml, -1.71, 1e-12);

  // FOMAML = λ_Q * (θ_1 - μ_Q) = -1.9
  const fomaml = fomamlOuterGradient1D(theta0, muS, muQ, lambdaS, lambdaQ, alpha);
  approxEq('C3. FOMAML outer gradient (3.1)', fomaml, -1.9, 1e-12);

  // FOMAML bias = MAML - FOMAML = 0.19 (per (3.2): -α H_S g_Q evaluates here to +0.19)
  const bias = fomamlBias1D(theta0, muS, muQ, lambdaS, lambdaQ, alpha);
  approxEq('C4. FOMAML bias (3.2) value', bias, 0.19, 1e-12);

  // Bias formula check: -α λ_S λ_Q (θ_1 - μ_Q) — sign convention
  const biasFormula = -alpha * lambdaS * lambdaQ * (theta1 - muQ);
  approxEq('C5. FOMAML bias matches closed form', bias, biasFormula, 1e-12);

  // Bias scales linearly in α
  const biasHalfAlpha = fomamlBias1D(theta0, muS, muQ, lambdaS, lambdaQ, alpha / 2);
  // Doubling/halving α also moves θ_1; the relationship is not exactly halving.
  // The exact statement: at α=0.05, θ_1 = 0.05, bias = -0.05 * 1 * 1 * (0.05 - 2) = 0.0975
  approxEq('C6. FOMAML bias at α/2', biasHalfAlpha, 0.0975, 1e-12);
})();

(() => {
  // Reptile direction θ_N - θ_0 on 1D quadratic
  const theta0 = 1;
  const mu = 0;
  const lambda = 1;
  const alpha = 0.1;
  const dir2 = reptileDirection1D(theta0, mu, lambda, alpha, 2);
  // θ_2 = 0 + 0.81*(1-0) = 0.81; direction = -0.19
  approxEq('C7. Reptile direction at N=2', dir2, -0.19, 1e-12);

  // Reptile implicit-objective coefficient α(N-1)/2
  approxEq('C8. Reptile coeff at N=1 (= 0)', reptileImplicitCoeff(alpha, 1), 0, 1e-15);
  approxEq('C9. Reptile coeff at N=2', reptileImplicitCoeff(alpha, 2), 0.05, 1e-15);
  approxEq('C10. Reptile coeff at N=5', reptileImplicitCoeff(alpha, 5), 0.2, 1e-15);
})();

// -----------------------------------------------------------------------------
// SECTION D — Neural Process forward / KL primitives (cells 31, 34)
// -----------------------------------------------------------------------------

console.log('\nD. Neural Process primitives');

(() => {
  // CNP encoder with identity-ish weights: x_dim=1, y_dim=1, h_dim=2, r_dim=2.
  // Use weights chosen so encoder output is (x, y) directly, then mean-pool.
  const encoderWeights = [
    // first layer: 2 -> 2 (project to (x, y))
    { W: [[1, 0], [0, 1]], b: [0, 0] },
    // second layer: 2 -> 2 (identity, no ReLU since this is the last layer)
    { W: [[1, 0], [0, 1]], b: [0, 0] },
  ];
  // Context: (1, 2), (3, 4), (5, 6) → mean = (3, 4) under identity encoding
  // BUT the first layer applies ReLU after computing W x + b, so y = ReLU(input)
  // which is identity for positive inputs. The aggregator then averages.
  const xCtx = [[1], [3], [5]];
  const yCtx = [[2], [4], [6]];
  const r = cnpEncoderForward(xCtx, yCtx, encoderWeights);
  approxEq('D1. CNP encoder mean-pool x-component', r[0], 3, 1e-12);
  approxEq('D2. CNP encoder mean-pool y-component', r[1], 4, 1e-12);

  // Single-point context: r should equal the encoder output exactly
  const rOne = cnpEncoderForward([[7]], [[8]], encoderWeights);
  approxEq('D3. CNP single-context r = enc(x,y)', rOne[0], 7, 1e-12);
  approxEq('D4. CNP single-context r y-component', rOne[1], 8, 1e-12);
})();

(() => {
  // Diagonal Gaussian KL identities
  approxEq('D5. KL(N(0,1) || N(0,1)) = 0', diagonalGaussianKL([0], [1], [0], [1]), 0, 1e-15);

  // KL(N(1,1) || N(0,1)) = 0.5 * ((1 + 1) / 1 - 1 + 0) = 0.5
  approxEq('D6. KL(N(1,1) || N(0,1)) = 0.5', diagonalGaussianKL([1], [1], [0], [1]), 0.5, 1e-12);

  // KL is non-negative (Gibbs inequality)
  const kl1 = diagonalGaussianKL([1, 2], [0.5, 0.5], [0, 0], [1, 1]);
  ok('D7. KL ≥ 0', kl1 >= 0, `KL=${kl1}`);

  // Multivariate: tr identity for diagonal Σ
  // KL(N(μ, σ²I) || N(0, I)) = ½ (d σ² - d - d log(σ²) + ||μ||²)
  // At μ=0, σ²=4: ½ (2*4 - 2 - 2 log 4) = ½ (8 - 2 - 2 * 1.3863) = ½ (3.2274) = 1.6137
  const klDiag = diagonalGaussianKL([0, 0], [2, 2], [0, 0], [1, 1]);
  approxEq('D8. KL(N(0,4I) || N(0,I)) multi-d', klDiag, 0.5 * (2 * 4 - 2 - 2 * Math.log(4)), 1e-12);
})();

// -----------------------------------------------------------------------------
// SECTION E — Prototypical Networks (Bregman-mean optimality)
// -----------------------------------------------------------------------------

console.log('\nE. Prototypical Networks');

(() => {
  // 3 classes, 4 points per class, embed in 2D. Prototype for class k should
  // be the mean of class-k embeddings.
  const sup: number[][] = [];
  const lbl: number[] = [];
  const classMeans = [[1, 0], [0, 1], [-1, 0]];
  // Deterministic offsets that sum to zero per class
  const offsets = [[0.1, 0.2], [-0.1, -0.2], [0.3, -0.1], [-0.3, 0.1]];
  for (let c = 0; c < 3; c++) {
    for (const [dx, dy] of offsets) {
      sup.push([classMeans[c][0] + dx, classMeans[c][1] + dy]);
      lbl.push(c);
    }
  }
  const protos = protonetClassPrototypes(sup, lbl, 3);
  approxEq('E1. prototype class 0 x-coord = class mean', protos[0][0], classMeans[0][0], 1e-12);
  approxEq('E2. prototype class 1 y-coord = class mean', protos[1][1], classMeans[1][1], 1e-12);
  approxEq('E3. prototype class 2 x-coord = class mean', protos[2][0], classMeans[2][0], 1e-12);

  // Bregman-mean optimality: for any candidate c̃, Σ_i ||φ_i - c̃||² is minimized
  // at c̃ = (1/n) Σ φ_i = the mean. Check with finite differences.
  let sumAtMean = 0;
  for (let i = 0; i < 4; i++) {
    sumAtMean += squaredDistance(sup[i], protos[0]); // class 0 indices 0..3
  }
  let sumAtPerturbed = 0;
  const cTilde = [protos[0][0] + 0.05, protos[0][1] + 0.05];
  for (let i = 0; i < 4; i++) sumAtPerturbed += squaredDistance(sup[i], cTilde);
  ok('E4. Bregman-mean: prototype minimizes Σ d²', sumAtPerturbed > sumAtMean, `mean=${sumAtMean}, perturbed=${sumAtPerturbed}`);
})();

(() => {
  // Nearest-prototype softmax: when query is exactly at prototype k, p_k > p_j (j≠k)
  const protos = [[0, 0], [1, 0], [0, 1]];
  const probs = nearestPrototypeSoftmax([0, 0], protos);
  ok('E5. softmax sum-to-one', Math.abs(probs.reduce((a, b) => a + b, 0) - 1) < 1e-12, `sum=${probs.reduce((a, b) => a + b, 0)}`);
  ok('E6. exact-on-prototype: p_0 > p_1', probs[0] > probs[1], `p_0=${probs[0]}, p_1=${probs[1]}`);
  ok('E7. exact-on-prototype: p_0 > p_2', probs[0] > probs[2], `p_0=${probs[0]}, p_2=${probs[2]}`);
})();

// -----------------------------------------------------------------------------
// SECTION F — Amit-Meir bound (cell 57 closed-form reproduction)
// -----------------------------------------------------------------------------

console.log('\nF. Amit-Meir meta PAC-Bayes bound');

(() => {
  // Cell 57 constants: KL_meta=5, KL_within=2, delta=0.05, r_hat=0.10
  const KL_META = 5.0;
  const KL_WITHIN = 2.0;
  const DELTA = 0.05;
  const R_HAT = 0.10;

  // At T=10, n=25:
  //   across = sqrt((5 + log(40/0.05)) / 18) = sqrt((5 + log 800) / 18)
  //   within = sqrt((2 + log(40*25/0.05)) / 48) = sqrt((2 + log 20000) / 48)
  const components = amitMeirBound(10, 25, KL_META, KL_WITHIN, DELTA, R_HAT);
  const expectedAcross = Math.sqrt((KL_META + Math.log((4 * 10) / DELTA)) / (2 * (10 - 1)));
  const expectedWithin = Math.sqrt((KL_WITHIN + Math.log((4 * 10 * 25) / DELTA)) / (2 * (25 - 1)));
  approxEq('F1. across-task piece (T=10,n=25)', components.across, expectedAcross, 1e-12);
  approxEq('F2. within-task piece (T=10,n=25)', components.within, expectedWithin, 1e-12);
  approxEq('F3. total bound (T=10,n=25)', components.total, R_HAT + expectedAcross + expectedWithin, 1e-12);

  // Concrete numerical value (computed by hand):
  // across = sqrt((5 + 6.6846) / 18) = sqrt(0.64914) ≈ 0.80569
  // within = sqrt((2 + 9.9035) / 48) = sqrt(0.24799) ≈ 0.49799
  // total ≈ 0.10 + 0.80569 + 0.49799 = 1.40369
  approxEq('F4. total bound numerical (T=10,n=25)', components.total, 1.40369, 1e-4);

  // Monotonicity: across-task piece shrinks as T grows
  const c1 = amitMeirBound(10, 25, KL_META, KL_WITHIN, DELTA, R_HAT);
  const c2 = amitMeirBound(100, 25, KL_META, KL_WITHIN, DELTA, R_HAT);
  ok('F5. across-task shrinks in T', c2.across < c1.across, `T=10 across=${c1.across}, T=100 across=${c2.across}`);

  // Within-task piece shrinks as n grows
  const c3 = amitMeirBound(10, 25, KL_META, KL_WITHIN, DELTA, R_HAT);
  const c4 = amitMeirBound(10, 100, KL_META, KL_WITHIN, DELTA, R_HAT);
  ok('F6. within-task shrinks in n', c4.within < c3.within, `n=25 within=${c3.within}, n=100 within=${c4.within}`);

  // 1/√T rate: doubling T-1 should shrink across by ~√2
  const cT10 = amitMeirBound(11, 25, KL_META, KL_WITHIN, DELTA, R_HAT); // T-1=10
  const cT20 = amitMeirBound(21, 25, KL_META, KL_WITHIN, DELTA, R_HAT); // T-1=20
  const ratio = cT10.across / cT20.across;
  // Note: ratio isn't exactly √2 because the log(4T/δ) numerator also changes
  ok('F7. across-task scaling close to 1/√T', Math.abs(ratio - Math.sqrt(2)) < 0.15, `ratio=${ratio.toPrecision(4)} vs √2=${Math.sqrt(2).toPrecision(4)}`);

  // Throws on degenerate T < 2 or n < 2
  let threw = false;
  try {
    amitMeirBound(1, 25, KL_META, KL_WITHIN, DELTA, R_HAT);
  } catch {
    threw = true;
  }
  ok('F8. throws on T<2', threw, 'T=1 should error');
})();

// -----------------------------------------------------------------------------
// SECTION G — FMO smoothness constant (Lemma 8.6, equation 8.4)
// -----------------------------------------------------------------------------

console.log('\nG. FMO constants (Lemma 1, equation 8.4)');

(() => {
  // L=1, ρ=0.5, α=0.01, G_Q=1
  //   L_F = (1+0.01)² * 1 + 0.01 * 0.5 * 1 = 1.0201 + 0.005 = 1.0251
  const LF = fmoSmoothnessConstant(1, 0.5, 0.01, 1);
  approxEq('G1. L_F at small α (8.4)', LF, 1.0251, 1e-12);

  // At α = 0: L_F = L
  const LF0 = fmoSmoothnessConstant(2, 0.5, 0, 3);
  approxEq('G2. L_F at α=0 reduces to L', LF0, 2, 1e-12);

  // Monotone in α: L_F(α=0.1) > L_F(α=0.01)
  const LF01 = fmoSmoothnessConstant(1, 0.5, 0.1, 1);
  ok('G3. L_F monotone in α', LF01 > LF, `L_F(0.01)=${LF}, L_F(0.1)=${LF01}`);

  // Variance bound (8.5): σ_g² + α² σ_H² G_Q² + α² σ_g² σ_H²
  // At σ_g=σ_H=1, α=0.1, G_Q=1: 1 + 0.01*1 + 0.01 = 1.02
  const Vb = fmoVarianceBound(1, 1, 0.1, 1);
  approxEq('G4. variance bound (8.5)', Vb, 1.02, 1e-12);

  // At α=0: variance reduces to σ_g²
  approxEq('G5. variance bound at α=0', fmoVarianceBound(2, 5, 0, 3), 4, 1e-12);

  // C / √K envelope constant: running_min[-1] * √K_max
  const C = convergenceEnvelopeConstant(0.5, 100);
  approxEq('G6. envelope constant C = min[-1] * √K_max', C, 5, 1e-12);
})();

// -----------------------------------------------------------------------------
// Summary
// -----------------------------------------------------------------------------

console.log(`\n========================================`);
console.log(`verify-meta-learning: ${pass} pass, ${fail} fail`);
if (fail > 0) {
  console.log(`\nFailures:`);
  for (const f of failures) console.log(`  - ${f}`);
}
console.log(`========================================\n`);

process.exit(fail === 0 ? 0 : 1);
