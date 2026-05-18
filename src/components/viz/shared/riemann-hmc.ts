// =============================================================================
// riemann-hmc.ts
//
// Shared math/types/palette/sampler module for the T5 riemann-manifold-hmc
// topic. Mirrors notebooks/riemann-manifold-hmc/01_riemann_manifold_hmc.ipynb
// cells 3, 6, 9, 10, 11, 13, 14, 17, 18, 23, 26, 37, 38 in TypeScript.
//
// Three independent groups of functions:
//
//   1. Banana primitives (§§1, 3, 4, 5, 6, 7, 8, 9): log-density, gradient,
//      Fisher metric (Gauss-Newton), metric inverse, ∂_k G_{ij}, Christoffel
//      symbols, riemannian Hamiltonian (component decomposition).
//
//      Christoffel note: bananaChristoffel() is corrected here vs the notebook's
//      banana_christoffel() (cell 10), which has a bracket-formula bug that
//      makes the FD-verification print max |diff| = 27.9 instead of ~1e-9.
//      Hand-derivation comment block above the function. The §3 viz uses this
//      corrected version so the live page demonstrates the brief's "geodesic
//      follows ridge" claim correctly; the static fallback PNG remains as a
//      known-degraded artifact for no-JS readers.
//
//   2. Integrators (§§2, 4, 5):
//        - standardLeapfrogStep    — flat-space Verlet with constant Minv
//        - geodesicRK4Step         — RK4 on ddot θ^k = -Γ^k_{ij} dot θ^i dot θ^j
//        - rmhmcRK4Step            — RK4 on Hamilton's equations (continuous-time
//                                    reference for §4 figure)
//        - generalizedLeapfrogStep — implicit Leimkuhler-Reich integrator (§5.2)
//        - naiveRiemannianLeapfrogStep — controlled-failure baseline (§5.1)
//
//   3. Samplers (§§6, 9, 10):
//        - hmcSample      — Metropolis-corrected standard HMC chain (banana)
//        - rmhmcSample    — Metropolis-corrected RMHMC chain (banana)
//        - lcpRmhmcSample — RMHMC chain for the §10 log-Gaussian Cox process
//
// Conventions:
//   - All matrices are number[][] row-major (matches gaussian-processes.ts).
//   - All vectors are number[] (or Float64Array in the hot-loop scratch
//     buffers inside the samplers, hoisted per CLAUDE.md "Hoist Float64Array
//     allocations" rule).
//   - All RNGs are () => number returning uniforms in [0, 1), seedable via
//     mulberry32. Standard normals via gaussianPair / drawGaussian.
//   - Pure functions, no React / D3 / DOM dependencies.
//
// Brief: docs/plans/formalml-riemann-manifold-hmc-handoff-brief.md
// Notebook: notebooks/riemann-manifold-hmc/01_riemann_manifold_hmc.ipynb
// =============================================================================

import { mulberry32, gaussianPair } from './bayesian-ml';
import {
  choleskyFactor,
  choleskyLogDet,
  solveLowerTriangular,
  solveUpperTriangularT,
  matVec,
} from './gaussian-processes';
import { paletteSGMCMC } from './sgmcmc';

// Re-export RNG helpers so consumers don't have to know which shared module
// they live in.
export { mulberry32, gaussianPair };

// =============================================================================
// PALETTE
// =============================================================================
// Extends paletteSGMCMC with RMHMC-only anchors. The cross-topic RMHMC color
// (olive #bcbd22) is already in paletteSGMCMC.rmhmc — we re-export through
// paletteRMHMC so RMHMC viz components don't have to import sgmcmc.ts.

export const paletteRMHMC = {
  target: paletteSGMCMC.target, //   '#2c3e50' near-black — analytical posterior
  hmc: '#1f77b4', //                  blue   — standard HMC (constant mass)
  rmhmc: paletteSGMCMC.rmhmc, //      '#bcbd22' olive — RMHMC (Fisher metric)
  rmla: '#e377c2', //                 pink   — Riemann-manifold Langevin (§12.1)
  nuts: paletteSGMCMC.nuts, //        '#d62728' red — NUTS reference
  metric: '#9467bd', //               purple — §3 Fisher-metric ellipses
  geodesic: '#17becf', //             teal   — §3 / §5 geodesic / RK4 reference
  naive: '#ff7f0e', //                orange — §5 naive explicit (failure case)
} as const;

export type RMHMCColorKey = keyof typeof paletteRMHMC;

// =============================================================================
// RNG / SAMPLING HELPERS
// =============================================================================

/** Stateful gaussian sampler keeping a half-pair in reserve. */
export function makeGaussian(rng: () => number): () => number {
  let half: number | null = null;
  return () => {
    if (half !== null) {
      const v = half;
      half = null;
      return v;
    }
    const [a, b] = gaussianPair(rng);
    half = b;
    return a;
  };
}

// =============================================================================
// BANANA MATH PRIMITIVES (§§1, 3, 4, 5, 6, 7, 8, 9)
// =============================================================================
// Bayesian model:
//   θ_1 ~ N(0, a²);  θ_2 ~ Uniform (improper)
//   y | θ ~ N(θ_2 + b(θ_1² - a²), 1);  y_obs = 0
// Up to additive constants:
//   log π(θ) = -½ [θ_1²/a² + (θ_2 + b(θ_1² - a²))²]
// Default params: a = b = 1.

export interface BananaParams {
  a: number;
  b: number;
}

export const BANANA_DEFAULT: BananaParams = { a: 1, b: 1 };

/** -log π(θ) up to additive constant. (Returns U(θ) = -log target.) */
export function bananaPotential(theta: number[], p: BananaParams = BANANA_DEFAULT): number {
  const t1 = theta[0];
  const t2 = theta[1];
  const w = t2 + p.b * (t1 * t1 - p.a * p.a);
  return 0.5 * ((t1 * t1) / (p.a * p.a) + w * w);
}

/** log π(θ) up to additive constant. */
export function bananaLogDensity(theta: number[], p: BananaParams = BANANA_DEFAULT): number {
  return -bananaPotential(theta, p);
}

/** ∇ U(θ) = -∇ log π(θ). */
export function bananaGradU(theta: number[], p: BananaParams = BANANA_DEFAULT): number[] {
  const t1 = theta[0];
  const t2 = theta[1];
  const w = t2 + p.b * (t1 * t1 - p.a * p.a);
  // ∂U/∂t1 = t1 / a² + w · 2 b t1
  // ∂U/∂t2 = w
  return [t1 / (p.a * p.a) + 2 * p.b * t1 * w, w];
}

/**
 * Gauss-Newton Fisher metric for the banana model.
 *
 *   G(θ) = J(θ)^T J(θ) + Λ_prior,
 *   J(θ) = (2 b θ_1, 1),
 *   Λ_prior = diag(1/a², 0).
 *
 * Closed form:
 *   G_{11} = 1/a² + 4 b² θ_1²
 *   G_{12} = G_{21} = 2 b θ_1
 *   G_{22} = 1
 *
 * Constant det G = 1/a², which makes the §4 volume term V(θ) = ½ log det G
 * constant on the banana (the §4.3 "V is free pedagogy on banana" remark).
 */
export function bananaMetric(theta: number[], p: BananaParams = BANANA_DEFAULT): number[][] {
  const t1 = theta[0];
  const a2 = p.a * p.a;
  const off = 2 * p.b * t1;
  return [
    [1 / a2 + 4 * p.b * p.b * t1 * t1, off],
    [off, 1],
  ];
}

/**
 * Inverse Fisher metric.
 *
 *   det G = 1/a² (constant)
 *   G^{-1} = a² · [[1, -2 b θ_1], [-2 b θ_1, G_{11}]]
 */
export function bananaMetricInv(theta: number[], p: BananaParams = BANANA_DEFAULT): number[][] {
  const t1 = theta[0];
  const a2 = p.a * p.a;
  const g11 = 1 / a2 + 4 * p.b * p.b * t1 * t1;
  const minusOff = -2 * p.b * t1;
  return [
    [a2, a2 * minusOff],
    [a2 * minusOff, a2 * g11],
  ];
}

/**
 * Metric partial derivatives ∂_k G_{ij}.
 *
 *   Only nonzero entries:
 *     ∂_1 G_{11} = 8 b² θ_1
 *     ∂_1 G_{12} = ∂_1 G_{21} = 2 b
 *   All ∂_2 G_{ij} = 0 (g_{ij} has no θ_2 dependence).
 *
 * Returned as out[k][i][j] = ∂_k G_{ij}.
 */
export function bananaMetricPartials(theta: number[], p: BananaParams = BANANA_DEFAULT): number[][][] {
  const t1 = theta[0];
  const d1_g11 = 8 * p.b * p.b * t1;
  const d1_g12 = 2 * p.b;
  return [
    // k = 0 (∂_1)
    [
      [d1_g11, d1_g12],
      [d1_g12, 0],
    ],
    // k = 1 (∂_2) — all zero
    [
      [0, 0],
      [0, 0],
    ],
  ];
}

/**
 * Christoffel symbols Γ^k_{ij} = ½ g^{kl} (∂_i g_{jl} + ∂_j g_{il} - ∂_l g_{ij}).
 *
 * Hand-derivation for the banana metric:
 *
 *   Let B^l_{ij} := ∂_i g_{jl} + ∂_j g_{il} - ∂_l g_{ij}.
 *
 *   By symmetry (B^l_{ij} = B^l_{ji}) and the banana's sparse partials, the
 *   only nonzero B entries are:
 *     B^1_{11} = ∂_1 g_{11} + ∂_1 g_{11} - ∂_1 g_{11} = ∂_1 g_{11} = 8 b² θ_1
 *     B^1_{12} = B^1_{21} = ∂_1 g_{21} + ∂_2 g_{11} - ∂_1 g_{12}
 *                          = 2b + 0 - 2b = 0
 *     B^1_{22} = ∂_2 g_{21} + ∂_2 g_{21} - ∂_1 g_{22} = 0
 *     B^2_{11} = ∂_1 g_{12} + ∂_1 g_{12} - ∂_2 g_{11} = 2b + 2b - 0 = 4b
 *     B^2_{12} = B^2_{21} = ∂_1 g_{22} + ∂_2 g_{12} - ∂_2 g_{12} = 0
 *     B^2_{22} = ∂_2 g_{22} + ∂_2 g_{22} - ∂_2 g_{22} = 0
 *
 *   Then Γ^k_{ij} = ½ (g^{k1} B^1_{ij} + g^{k2} B^2_{ij}).
 *
 *   Substituting g^{11} = a², g^{12} = g^{21} = -2 a² b θ_1, g^{22} = 1 + 4 a² b² θ_1²:
 *     Γ^1_{11} = ½ (a² · 8b²θ_1 + (-2a²bθ_1) · 4b)
 *              = ½ (8 a² b² θ_1 - 8 a² b² θ_1) = 0
 *     Γ^1_{12} = ½ (a² · 0 + (-2a²bθ_1) · 0) = 0
 *     Γ^1_{22} = ½ (a² · 0 + (-2a²bθ_1) · 0) = 0
 *     Γ^2_{11} = ½ ((-2a²bθ_1) · 8b²θ_1 + (1 + 4a²b²θ_1²) · 4b)
 *              = ½ (-16 a² b³ θ_1² + 4b + 16 a² b³ θ_1²) = 2b
 *     Γ^2_{12} = ½ ((-2a²bθ_1) · 0 + (1 + 4a²b²θ_1²) · 0) = 0
 *     Γ^2_{22} = ½ ((-2a²bθ_1) · 0 + (1 + 4a²b²θ_1²) · 0) = 0
 *
 *   So Γ^2_{11} = 2b is the ONLY nonzero symbol on the banana. The geodesic
 *   equation reduces to
 *
 *     ddot θ_1 = 0
 *     ddot θ_2 = -Γ^2_{11} (dot θ_1)² = -2b (dot θ_1)²
 *
 *   On a ridge-tangent initial velocity (vel_0 = (1, -2b·θ_{1,0})), the
 *   geodesic traces the ridge θ_2 = -(θ_1² - a²) exactly — the curvature
 *   match is d²θ_2/dθ_1² = -2b on both sides.
 *
 *   This is the COMPLETE-MANIFOLD version of the notebook's banana_christoffel,
 *   which mis-encodes B^l_{ij} and prints max |closed - FD| = 27.9 instead of
 *   the ~1e-9 that the verify script asserts.
 *
 * Returned as out[k][i][j] = Γ^k_{ij}.
 */
export function bananaChristoffel(theta: number[], p: BananaParams = BANANA_DEFAULT): number[][][] {
  const two_b = 2 * p.b;
  return [
    // k = 0 (Γ^1): all zero
    [
      [0, 0],
      [0, 0],
    ],
    // k = 1 (Γ^2): only Γ^2_{11} = 2b is nonzero
    [
      [two_b, 0],
      [0, 0],
    ],
  ];
}

/**
 * Finite-difference Christoffel for verification.
 *
 *   Γ^k_{ij} = ½ g^{kl} (∂_i g_{jl} + ∂_j g_{il} - ∂_l g_{ij}),
 *
 * with ∂_l g_{ij} computed via central differences. Used by the verify script.
 */
export function bananaChristoffelFD(theta: number[], p: BananaParams = BANANA_DEFAULT, h: number = 1e-5): number[][][] {
  const d = 2;
  // dG[l][i][j] = (G_{ij}(θ + h e_l) - G_{ij}(θ - h e_l)) / (2 h)
  const dG: number[][][] = [
    [
      [0, 0],
      [0, 0],
    ],
    [
      [0, 0],
      [0, 0],
    ],
  ];
  for (let l = 0; l < d; l++) {
    const thPlus = [theta[0], theta[1]];
    const thMinus = [theta[0], theta[1]];
    thPlus[l] += h;
    thMinus[l] -= h;
    const Gp = bananaMetric(thPlus, p);
    const Gm = bananaMetric(thMinus, p);
    for (let i = 0; i < d; i++) {
      for (let j = 0; j < d; j++) {
        dG[l][i][j] = (Gp[i][j] - Gm[i][j]) / (2 * h);
      }
    }
  }
  const Ginv = bananaMetricInv(theta, p);
  const Gamma: number[][][] = [
    [
      [0, 0],
      [0, 0],
    ],
    [
      [0, 0],
      [0, 0],
    ],
  ];
  for (let k = 0; k < d; k++) {
    for (let i = 0; i < d; i++) {
      for (let j = 0; j < d; j++) {
        let s = 0;
        for (let l = 0; l < d; l++) {
          const bracket = dG[i][j][l] + dG[j][i][l] - dG[l][i][j];
          s += Ginv[k][l] * bracket;
        }
        Gamma[k][i][j] = 0.5 * s;
      }
    }
  }
  return Gamma;
}

/**
 * Banana Riemannian Hamiltonian H = U + K + V with component decomposition.
 *
 *   U(θ) = -log π(θ)
 *   K(θ, p) = ½ p^T G(θ)^{-1} p
 *   V(θ) = ½ log det G(θ) = -log a (constant on banana)
 *
 *   H = U + K + V.
 */
export function bananaHamiltonian(
  theta: number[],
  mom: number[],
  p: BananaParams = BANANA_DEFAULT,
): { U: number; K: number; V: number; H: number } {
  const U = bananaPotential(theta, p);
  const Ginv = bananaMetricInv(theta, p);
  // K = ½ p^T G^{-1} p
  const Gip = [Ginv[0][0] * mom[0] + Ginv[0][1] * mom[1], Ginv[1][0] * mom[0] + Ginv[1][1] * mom[1]];
  const K = 0.5 * (mom[0] * Gip[0] + mom[1] * Gip[1]);
  // det G = 1/a², log det G = -2 log a, V = ½ (-2 log a) = -log a
  const V = -Math.log(p.a);
  return { U, K, V, H: U + K + V };
}

/**
 * ∇_θ K (banana). Components:
 *   [∇_θ K]_k = -½ p^T G^{-1} (∂_k G) G^{-1} p.
 *
 * Uses the closed-form metric partials. Three-vector pipeline: solve x = G^{-1} p,
 * then for each k compute -½ x^T (∂_k G) x.
 */
export function bananaGradKineticBanana(
  theta: number[],
  mom: number[],
  p: BananaParams = BANANA_DEFAULT,
): number[] {
  const Ginv = bananaMetricInv(theta, p);
  const x = [Ginv[0][0] * mom[0] + Ginv[0][1] * mom[1], Ginv[1][0] * mom[0] + Ginv[1][1] * mom[1]];
  const dG = bananaMetricPartials(theta, p);
  const out = [0, 0];
  for (let k = 0; k < 2; k++) {
    let s = 0;
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) {
        s += x[i] * dG[k][i][j] * x[j];
      }
    }
    out[k] = -0.5 * s;
  }
  return out;
}

/**
 * ∇_θ V (banana). V is constant ⇒ gradient is zero. Provided as a function for
 * uniformity with the LCP path.
 */
export function bananaGradVolume(_theta: number[], _p: BananaParams = BANANA_DEFAULT): number[] {
  return [0, 0];
}

// =============================================================================
// 2×2 LINALG FAST PATHS (for banana hot loops)
// =============================================================================

/** Sample p ~ N(0, G) for a 2×2 SPD G via Cholesky (G = L L^T → p = L z). */
export function sampleMomentum2x2(G: number[][], gauss: () => number): number[] {
  // L_00 = sqrt(G_00), L_10 = G_10 / L_00, L_11 = sqrt(G_11 - L_10²)
  const l00 = Math.sqrt(G[0][0]);
  const l10 = G[1][0] / l00;
  const l11 = Math.sqrt(Math.max(G[1][1] - l10 * l10, 1e-12));
  const z0 = gauss();
  const z1 = gauss();
  return [l00 * z0, l10 * z0 + l11 * z1];
}

// =============================================================================
// STANDARD HMC (§§1, 2, 8, 9)
// =============================================================================
// Constant mass matrix M (typically identity), explicit leapfrog. Used as the
// HMC baseline for the banana comparison panels.

/** Single explicit Verlet step with constant Minv (diagonal). */
export function standardLeapfrogStep(
  theta: number[],
  mom: number[],
  eps: number,
  gradU: (theta: number[]) => number[],
  MinvDiag: number[], // d-vector
): { theta: number[]; mom: number[] } {
  const d = theta.length;
  // half-momentum
  const g0 = gradU(theta);
  const pHalf: number[] = new Array(d);
  for (let i = 0; i < d; i++) pHalf[i] = mom[i] - 0.5 * eps * g0[i];
  // position
  const thetaNew: number[] = new Array(d);
  for (let i = 0; i < d; i++) thetaNew[i] = theta[i] + eps * MinvDiag[i] * pHalf[i];
  // final half
  const g1 = gradU(thetaNew);
  const momNew: number[] = new Array(d);
  for (let i = 0; i < d; i++) momNew[i] = pHalf[i] - 0.5 * eps * g1[i];
  return { theta: thetaNew, mom: momNew };
}

/** Metropolis-corrected standard HMC chain on the banana. */
export interface ChainResult {
  samples: number[][]; // n × d
  acceptanceRate: number;
  divergences: number;
  meanPIters: number;
  meanThetaIters: number;
  wallSeconds: number;
}

export function hmcSample(
  theta0: number[],
  nSamples: number,
  eps: number,
  L: number,
  bananaP: BananaParams,
  rng: () => number,
  MinvDiag: number[] = [1, 1],
): ChainResult {
  const t0 = performance.now();
  const gauss = makeGaussian(rng);
  const d = theta0.length;
  const samples: number[][] = [theta0.slice()];
  let theta = theta0.slice();
  let accepts = 0;
  const gradU = (th: number[]) => bananaGradU(th, bananaP);
  for (let n = 1; n < nSamples; n++) {
    // resample momentum p ~ N(0, M) ⇒ p_i = sqrt(M_i) z_i = z_i / sqrt(MinvDiag_i)
    const mom: number[] = new Array(d);
    for (let i = 0; i < d; i++) mom[i] = gauss() / Math.sqrt(MinvDiag[i]);
    // initial H_0 = U + K (no volume term for standard HMC)
    const U0 = bananaPotential(theta, bananaP);
    let K0 = 0;
    for (let i = 0; i < d; i++) K0 += 0.5 * MinvDiag[i] * mom[i] * mom[i];
    const H0 = U0 + K0;
    // L leapfrog steps
    let thN = theta.slice();
    let pN = mom.slice();
    for (let l = 0; l < L; l++) {
      const r = standardLeapfrogStep(thN, pN, eps, gradU, MinvDiag);
      thN = r.theta;
      pN = r.mom;
    }
    // Metropolis
    const U1 = bananaPotential(thN, bananaP);
    let K1 = 0;
    for (let i = 0; i < d; i++) K1 += 0.5 * MinvDiag[i] * pN[i] * pN[i];
    const H1 = U1 + K1;
    const accept = Math.log(rng()) < H0 - H1;
    if (accept) {
      theta = thN;
      accepts++;
    }
    samples.push(theta.slice());
  }
  return {
    samples,
    acceptanceRate: accepts / (nSamples - 1),
    divergences: 0,
    meanPIters: 0,
    meanThetaIters: 0,
    wallSeconds: (performance.now() - t0) / 1000,
  };
}

// =============================================================================
// GEODESIC INTEGRATOR (§3)
// =============================================================================
// Integrates ddot θ^k = -Γ^k_{ij} dot θ^i dot θ^j by RK4 on the first-order
// system (θ, v) with v = dot θ.

export interface GeodesicState {
  theta: number[];
  vel: number[];
}

function geodesicAccel(theta: number[], vel: number[], p: BananaParams): number[] {
  const Gamma = bananaChristoffel(theta, p);
  const acc = [0, 0];
  for (let k = 0; k < 2; k++) {
    let s = 0;
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) {
        s += Gamma[k][i][j] * vel[i] * vel[j];
      }
    }
    acc[k] = -s;
  }
  return acc;
}

export function geodesicRK4Step(s: GeodesicState, dt: number, p: BananaParams = BANANA_DEFAULT): GeodesicState {
  const { theta, vel } = s;
  // k1
  const a1 = geodesicAccel(theta, vel, p);
  // k2
  const th2 = [theta[0] + 0.5 * dt * vel[0], theta[1] + 0.5 * dt * vel[1]];
  const v2 = [vel[0] + 0.5 * dt * a1[0], vel[1] + 0.5 * dt * a1[1]];
  const a2 = geodesicAccel(th2, v2, p);
  // k3
  const th3 = [theta[0] + 0.5 * dt * v2[0], theta[1] + 0.5 * dt * v2[1]];
  const v3 = [vel[0] + 0.5 * dt * a2[0], vel[1] + 0.5 * dt * a2[1]];
  const a3 = geodesicAccel(th3, v3, p);
  // k4
  const th4 = [theta[0] + dt * v3[0], theta[1] + dt * v3[1]];
  const v4 = [vel[0] + dt * a3[0], vel[1] + dt * a3[1]];
  const a4 = geodesicAccel(th4, v4, p);
  // combine
  const thetaNew = [
    theta[0] + (dt / 6) * (vel[0] + 2 * v2[0] + 2 * v3[0] + v4[0]),
    theta[1] + (dt / 6) * (vel[1] + 2 * v2[1] + 2 * v3[1] + v4[1]),
  ];
  const velNew = [
    vel[0] + (dt / 6) * (a1[0] + 2 * a2[0] + 2 * a3[0] + a4[0]),
    vel[1] + (dt / 6) * (a1[1] + 2 * a2[1] + 2 * a3[1] + a4[1]),
  ];
  return { theta: thetaNew, vel: velNew };
}

export function geodesicTrajectory(
  theta0: number[],
  vel0: number[],
  T: number,
  nSteps: number,
  p: BananaParams = BANANA_DEFAULT,
): GeodesicState[] {
  const dt = T / nSteps;
  const path: GeodesicState[] = [{ theta: theta0.slice(), vel: vel0.slice() }];
  let s: GeodesicState = { theta: theta0.slice(), vel: vel0.slice() };
  for (let n = 0; n < nSteps; n++) {
    s = geodesicRK4Step(s, dt, p);
    path.push({ theta: s.theta.slice(), vel: s.vel.slice() });
  }
  return path;
}

// =============================================================================
// CONTINUOUS-TIME RMHMC RK4 REFERENCE (§4)
// =============================================================================
// dot θ = G^{-1} p
// dot p_k = -∂_k U + ½ p^T G^{-1} (∂_k G) G^{-1} p - ½ tr(G^{-1} ∂_k G)
//        = -∂_k U - ∇_θ K_k - ∂_k V
// We use the form (dot θ, dot p) = f(θ, p).

function rmhmcRHS(theta: number[], mom: number[], p: BananaParams): { dTheta: number[]; dMom: number[] } {
  const Ginv = bananaMetricInv(theta, p);
  const dTheta = [Ginv[0][0] * mom[0] + Ginv[0][1] * mom[1], Ginv[1][0] * mom[0] + Ginv[1][1] * mom[1]];
  const gU = bananaGradU(theta, p);
  const gK = bananaGradKineticBanana(theta, mom, p); // contains the +½ p^T G^{-1} (∂G) G^{-1} p
  // Note: bananaGradKineticBanana returns -½ p^T G^{-1} (∂G) G^{-1} p (the dK/dθ
  // contribution to ∇_θ H from differentiating K = ½ p^T G^{-1} p). So
  //   dot p = -∂_k U - (∂_k H from K) - (∂_k V)
  // and (∂_k H from K) is the value bananaGradKineticBanana returns.
  // V is constant on banana ⇒ gV = 0.
  const gV = bananaGradVolume(theta, p);
  const dMom = [-gU[0] - gK[0] - gV[0], -gU[1] - gK[1] - gV[1]];
  return { dTheta, dMom };
}

export function rmhmcRK4Step(
  theta: number[],
  mom: number[],
  dt: number,
  p: BananaParams = BANANA_DEFAULT,
): { theta: number[]; mom: number[] } {
  // k1
  const k1 = rmhmcRHS(theta, mom, p);
  // k2 @ (θ + ½ dt k1_θ, p + ½ dt k1_p)
  const th2 = [theta[0] + 0.5 * dt * k1.dTheta[0], theta[1] + 0.5 * dt * k1.dTheta[1]];
  const p2 = [mom[0] + 0.5 * dt * k1.dMom[0], mom[1] + 0.5 * dt * k1.dMom[1]];
  const k2 = rmhmcRHS(th2, p2, p);
  // k3
  const th3 = [theta[0] + 0.5 * dt * k2.dTheta[0], theta[1] + 0.5 * dt * k2.dTheta[1]];
  const p3 = [mom[0] + 0.5 * dt * k2.dMom[0], mom[1] + 0.5 * dt * k2.dMom[1]];
  const k3 = rmhmcRHS(th3, p3, p);
  // k4
  const th4 = [theta[0] + dt * k3.dTheta[0], theta[1] + dt * k3.dTheta[1]];
  const p4 = [mom[0] + dt * k3.dMom[0], mom[1] + dt * k3.dMom[1]];
  const k4 = rmhmcRHS(th4, p4, p);
  return {
    theta: [
      theta[0] + (dt / 6) * (k1.dTheta[0] + 2 * k2.dTheta[0] + 2 * k3.dTheta[0] + k4.dTheta[0]),
      theta[1] + (dt / 6) * (k1.dTheta[1] + 2 * k2.dTheta[1] + 2 * k3.dTheta[1] + k4.dTheta[1]),
    ],
    mom: [
      mom[0] + (dt / 6) * (k1.dMom[0] + 2 * k2.dMom[0] + 2 * k3.dMom[0] + k4.dMom[0]),
      mom[1] + (dt / 6) * (k1.dMom[1] + 2 * k2.dMom[1] + 2 * k3.dMom[1] + k4.dMom[1]),
    ],
  };
}

// =============================================================================
// GENERALIZED LEAPFROG (§5)
// =============================================================================
// Three-line implicit Leimkuhler-Reich scheme. Two of the three lines need
// fixed-point iteration; the third is explicit. Returns the new state plus
// per-step diagnostics (FP iteration counts, divergence flag).

export interface GLStepResult {
  theta: number[];
  mom: number[];
  pIters: number; // momentum half-step FP iterations
  thetaIters: number; // position step FP iterations
  converged: boolean; // false ⇒ hit iteration cap (divergent)
}

export interface GLStepOpts {
  tol?: number;
  maxIters?: number;
}

const GL_DEFAULTS: Required<GLStepOpts> = { tol: 1e-6, maxIters: 20 };

/**
 * Single generalized leapfrog step on the banana Riemannian Hamiltonian.
 *
 *   p^{n+½} = p^n - (ε/2) ∇_θ H(θ^n, p^{n+½})           [implicit in p]
 *   θ^{n+1} = θ^n + (ε/2) [G^{-1}(θ^n) + G^{-1}(θ^{n+1})] p^{n+½}   [implicit in θ]
 *   p^{n+1} = p^{n+½} - (ε/2) ∇_θ H(θ^{n+1}, p^{n+½})              [explicit]
 *
 * where ∇_θ H = ∇U + ∇V + ∇_θ K (with ∇V = 0 on banana).
 */
export function generalizedLeapfrogStep(
  theta: number[],
  mom: number[],
  eps: number,
  p: BananaParams = BANANA_DEFAULT,
  opts: GLStepOpts = {},
): GLStepResult {
  const { tol, maxIters } = { ...GL_DEFAULTS, ...opts };
  // ─── Step 1: implicit momentum half-step ──────────────────────────────
  // p^{n+½} = b - (ε/2) ∇_θ K(θ^n, p^{n+½}),  b = p^n - (ε/2) [∇U + ∇V]
  const gU = bananaGradU(theta, p);
  const gV = bananaGradVolume(theta, p);
  const b = [mom[0] - 0.5 * eps * (gU[0] + gV[0]), mom[1] - 0.5 * eps * (gU[1] + gV[1])];
  let pHalf = b.slice();
  let pIters = 0;
  let pConverged = false;
  for (let k = 0; k < maxIters; k++) {
    const gK = bananaGradKineticBanana(theta, pHalf, p);
    const pNext = [b[0] - 0.5 * eps * gK[0], b[1] - 0.5 * eps * gK[1]];
    const dx = Math.hypot(pNext[0] - pHalf[0], pNext[1] - pHalf[1]);
    const denom = Math.hypot(pNext[0], pNext[1]) + 1e-12;
    pHalf = pNext;
    pIters = k + 1;
    if (dx / denom < tol) {
      pConverged = true;
      break;
    }
  }
  if (!pConverged) {
    return { theta: theta.slice(), mom: mom.slice(), pIters, thetaIters: 0, converged: false };
  }
  // ─── Step 2: implicit position step ─────────────────────────────────────
  // θ^{n+1} = c + (ε/2) G^{-1}(θ^{n+1}) p^{n+½},  c = θ^n + (ε/2) G^{-1}(θ^n) p^{n+½}
  const GinvN = bananaMetricInv(theta, p);
  const half0 = [
    0.5 * eps * (GinvN[0][0] * pHalf[0] + GinvN[0][1] * pHalf[1]),
    0.5 * eps * (GinvN[1][0] * pHalf[0] + GinvN[1][1] * pHalf[1]),
  ];
  const c = [theta[0] + half0[0], theta[1] + half0[1]];
  let thetaNew = c.slice();
  let thetaIters = 0;
  let thetaConverged = false;
  for (let k = 0; k < maxIters; k++) {
    const GinvK = bananaMetricInv(thetaNew, p);
    const update = [
      0.5 * eps * (GinvK[0][0] * pHalf[0] + GinvK[0][1] * pHalf[1]),
      0.5 * eps * (GinvK[1][0] * pHalf[0] + GinvK[1][1] * pHalf[1]),
    ];
    const next = [c[0] + update[0], c[1] + update[1]];
    const dx = Math.hypot(next[0] - thetaNew[0], next[1] - thetaNew[1]);
    const denom = Math.hypot(next[0], next[1]) + 1e-12;
    thetaNew = next;
    thetaIters = k + 1;
    if (dx / denom < tol) {
      thetaConverged = true;
      break;
    }
  }
  if (!thetaConverged) {
    return { theta: theta.slice(), mom: mom.slice(), pIters, thetaIters, converged: false };
  }
  // ─── Step 3: explicit momentum half-step ───────────────────────────────
  const gU1 = bananaGradU(thetaNew, p);
  const gV1 = bananaGradVolume(thetaNew, p);
  const gK1 = bananaGradKineticBanana(thetaNew, pHalf, p);
  const momNew = [
    pHalf[0] - 0.5 * eps * (gU1[0] + gV1[0] + gK1[0]),
    pHalf[1] - 0.5 * eps * (gU1[1] + gV1[1] + gK1[1]),
  ];
  return { theta: thetaNew, mom: momNew, pIters, thetaIters, converged: true };
}

/**
 * Naive explicit Riemannian leapfrog (the §5.1 controlled-failure baseline).
 * Treats the kinetic term as separable — evaluates ∇_θ K at the CURRENT
 * (θ, p) rather than implicitly, and uses G^{-1}(θ) at the current θ for the
 * position update. Symplecticity fails; the energy error drifts secularly.
 */
export function naiveRiemannianLeapfrogStep(
  theta: number[],
  mom: number[],
  eps: number,
  p: BananaParams = BANANA_DEFAULT,
): { theta: number[]; mom: number[] } {
  const gU0 = bananaGradU(theta, p);
  const gV0 = bananaGradVolume(theta, p);
  const gK0 = bananaGradKineticBanana(theta, mom, p);
  const pHalf = [
    mom[0] - 0.5 * eps * (gU0[0] + gV0[0] + gK0[0]),
    mom[1] - 0.5 * eps * (gU0[1] + gV0[1] + gK0[1]),
  ];
  const Ginv = bananaMetricInv(theta, p);
  const thetaNew = [
    theta[0] + eps * (Ginv[0][0] * pHalf[0] + Ginv[0][1] * pHalf[1]),
    theta[1] + eps * (Ginv[1][0] * pHalf[0] + Ginv[1][1] * pHalf[1]),
  ];
  const gU1 = bananaGradU(thetaNew, p);
  const gV1 = bananaGradVolume(thetaNew, p);
  const gK1 = bananaGradKineticBanana(thetaNew, pHalf, p);
  const momNew = [
    pHalf[0] - 0.5 * eps * (gU1[0] + gV1[0] + gK1[0]),
    pHalf[1] - 0.5 * eps * (gU1[1] + gV1[1] + gK1[1]),
  ];
  return { theta: thetaNew, mom: momNew };
}

// =============================================================================
// RMHMC SAMPLER (§§6, 7, 8, 9)
// =============================================================================

export function rmhmcSample(
  theta0: number[],
  nSamples: number,
  eps: number,
  L: number,
  bananaP: BananaParams,
  rng: () => number,
  opts: GLStepOpts = {},
): ChainResult {
  const t0 = performance.now();
  const gauss = makeGaussian(rng);
  const samples: number[][] = [theta0.slice()];
  let theta = theta0.slice();
  let accepts = 0;
  let divergences = 0;
  let totalPIters = 0;
  let totalThetaIters = 0;
  let totalGLSteps = 0;
  for (let n = 1; n < nSamples; n++) {
    // resample momentum p ~ N(0, G(θ))
    const G = bananaMetric(theta, bananaP);
    const mom = sampleMomentum2x2(G, gauss);
    const H0 = bananaHamiltonian(theta, mom, bananaP).H;
    // L GL steps
    let thN = theta.slice();
    let pN = mom.slice();
    let trajectoryDivergent = false;
    for (let l = 0; l < L; l++) {
      const r = generalizedLeapfrogStep(thN, pN, eps, bananaP, opts);
      totalPIters += r.pIters;
      totalThetaIters += r.thetaIters;
      totalGLSteps++;
      if (!r.converged) {
        trajectoryDivergent = true;
        break;
      }
      thN = r.theta;
      pN = r.mom;
    }
    if (trajectoryDivergent) {
      divergences++;
      samples.push(theta.slice()); // reject
      continue;
    }
    const H1 = bananaHamiltonian(thN, pN, bananaP).H;
    const accept = Math.log(rng()) < H0 - H1;
    if (accept) {
      theta = thN;
      accepts++;
    }
    samples.push(theta.slice());
  }
  return {
    samples,
    acceptanceRate: accepts / (nSamples - 1),
    divergences,
    meanPIters: totalGLSteps > 0 ? totalPIters / totalGLSteps : 0,
    meanThetaIters: totalGLSteps > 0 ? totalThetaIters / totalGLSteps : 0,
    wallSeconds: (performance.now() - t0) / 1000,
  };
}

// =============================================================================
// LCP PRIMITIVES (§10)
// =============================================================================
// Log-Gaussian Cox process on an nx × ny grid. d = nx · ny cells. Latent log
// intensity x ∈ R^d. Likelihood: y_i | x_i ~ Poisson(A · exp(x_i)) with cell
// area A = 1 / d. Prior: x ~ N(μ_x · 1, K) with squared-exponential kernel K.

export interface LCPModel {
  d: number;
  nx: number;
  ny: number;
  cellArea: number;
  muX: number;
  Kprior: number[][]; // prior covariance
  Lambda: number[][]; // prior precision = K^{-1}
  cholLambda: number[][]; // lower Cholesky of Lambda
  yObs: number[]; // observed counts
  xTrue: number[]; // synthetic ground truth
  centers: number[][]; // cell centers in [0, 1]² (length d, each entry [cx, cy])
}

function buildKernel(centers: number[][], rho: number, sigma: number, jitter: number = 1e-6): number[][] {
  const d = centers.length;
  const K: number[][] = Array.from({ length: d }, () => new Array(d).fill(0));
  const sig2 = sigma * sigma;
  for (let i = 0; i < d; i++) {
    for (let j = 0; j < d; j++) {
      const dx = centers[i][0] - centers[j][0];
      const dy = centers[i][1] - centers[j][1];
      const r2 = dx * dx + dy * dy;
      K[i][j] = sig2 * Math.exp(-r2 / (2 * rho * rho));
    }
    K[i][i] += jitter;
  }
  return K;
}

function matInvSPD(L: number[][]): number[][] {
  // Given lower-triangular Cholesky L of A, return A^{-1} via two triangular solves on each column of I.
  const d = L.length;
  const inv: number[][] = Array.from({ length: d }, () => new Array(d).fill(0));
  for (let col = 0; col < d; col++) {
    const e: number[] = new Array(d).fill(0);
    e[col] = 1;
    const y = solveLowerTriangular(L, e);
    const x = solveUpperTriangularT(L, y);
    for (let i = 0; i < d; i++) inv[i][col] = x[i];
  }
  return inv;
}

export function buildLCPModel(
  nx: number,
  ny: number,
  rho: number = 0.25,
  sigma: number = 1,
  muX: number = Math.log(50),
  seed: number = 42,
): LCPModel {
  const d = nx * ny;
  const cellArea = 1 / d;
  // grid centers in [0, 1]²
  const centers: number[][] = [];
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      centers.push([(i + 0.5) / nx, (j + 0.5) / ny]);
    }
  }
  const Kprior = buildKernel(centers, rho, sigma);
  const cholK = choleskyFactor(Kprior);
  const Lambda = matInvSPD(cholK);
  // sample x_true from N(muX·1, K)
  const rng = mulberry32(seed);
  const gauss = makeGaussian(rng);
  const z: number[] = new Array(d);
  for (let i = 0; i < d; i++) z[i] = gauss();
  // x_true = muX + L z (lower-triangular Cholesky of Kprior)
  const xTrue: number[] = new Array(d).fill(muX);
  for (let i = 0; i < d; i++) {
    for (let j = 0; j <= i; j++) {
      xTrue[i] += cholK[i][j] * z[j];
    }
  }
  // sample yObs ~ Poisson(A exp(xTrue))
  const yObs: number[] = new Array(d);
  for (let i = 0; i < d; i++) {
    const lambda = cellArea * Math.exp(xTrue[i]);
    yObs[i] = poissonSample(lambda, rng);
  }
  const cholLambda = choleskyFactor(Lambda);
  return { d, nx, ny, cellArea, muX, Kprior, Lambda, cholLambda, yObs, xTrue, centers };
}

function poissonSample(lambda: number, rng: () => number): number {
  // Knuth's algorithm — adequate for lambda ≤ ~30; for larger lambda use a better method
  if (lambda < 30) {
    const Lexp = Math.exp(-lambda);
    let k = 0;
    let p = 1;
    do {
      k++;
      p *= rng();
    } while (p > Lexp);
    return k - 1;
  }
  // Atkinson 1979 rejection method (approximate for lambda > 30)
  const c = 0.767 - 3.36 / lambda;
  const beta = Math.PI / Math.sqrt(3 * lambda);
  const alpha = beta * lambda;
  const k = Math.log(c) - lambda - Math.log(beta);
  for (let attempt = 0; attempt < 1000; attempt++) {
    const u = rng();
    const x = (alpha - Math.log((1 - u) / u)) / beta;
    const n = Math.floor(x + 0.5);
    if (n < 0) continue;
    const v = rng();
    const y = alpha - beta * x;
    const lhs = y + Math.log(v / (1 + Math.exp(y)) ** 2);
    const rhs = k + n * Math.log(lambda) - logGamma(n + 1);
    if (lhs <= rhs) return n;
  }
  return Math.round(lambda);
}

function logGamma(x: number): number {
  // Lanczos approximation
  const g = 7;
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
    1.5056327351493116e-7,
  ];
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
  x -= 1;
  let a = c[0];
  const t = x + g + 0.5;
  for (let i = 1; i < g + 2; i++) a += c[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

export function lcpLogDensity(x: number[], m: LCPModel): number {
  let ll = 0;
  for (let i = 0; i < m.d; i++) {
    ll += m.yObs[i] * x[i] - m.cellArea * Math.exp(x[i]);
  }
  // -½ (x - μ·1)^T Λ (x - μ·1)
  const xm: number[] = new Array(m.d);
  for (let i = 0; i < m.d; i++) xm[i] = x[i] - m.muX;
  const Lxm = matVec(m.Lambda, xm);
  let quad = 0;
  for (let i = 0; i < m.d; i++) quad += xm[i] * Lxm[i];
  return ll - 0.5 * quad;
}

export function lcpGradLog(x: number[], m: LCPModel): number[] {
  const grad: number[] = new Array(m.d);
  const xm: number[] = new Array(m.d);
  for (let i = 0; i < m.d; i++) xm[i] = x[i] - m.muX;
  const Lxm = matVec(m.Lambda, xm);
  for (let i = 0; i < m.d; i++) {
    grad[i] = m.yObs[i] - m.cellArea * Math.exp(x[i]) - Lxm[i];
  }
  return grad;
}

export function lcpMetric(x: number[], m: LCPModel): number[][] {
  // G(x) = D(x) + Λ, D = diag(A exp(x))
  const G: number[][] = Array.from({ length: m.d }, (_, i) => m.Lambda[i].slice());
  for (let i = 0; i < m.d; i++) G[i][i] += m.cellArea * Math.exp(x[i]);
  return G;
}

function lcpVolumeAndGrad(
  cholG: number[][],
  x: number[],
  m: LCPModel,
): { V: number; gradV: number[] } {
  // V = ½ log det G
  const V = 0.5 * choleskyLogDet(cholG);
  // gradV_k = ½ A exp(x_k) [G^{-1}]_{kk}; compute diag(G^{-1}) via column solves
  const gradV: number[] = new Array(m.d);
  for (let k = 0; k < m.d; k++) {
    const e: number[] = new Array(m.d).fill(0);
    e[k] = 1;
    const y = solveLowerTriangular(cholG, e);
    const xk = solveUpperTriangularT(cholG, y);
    gradV[k] = 0.5 * m.cellArea * Math.exp(x[k]) * xk[k];
  }
  return { V, gradV };
}

function lcpGradKinetic(cholG: number[][], x: number[], mom: number[], m: LCPModel): number[] {
  // [∇_x K]_k = -½ p^T G^{-1} (∂_k G) G^{-1} p
  // ∂_k G = A exp(x_k) e_k e_k^T  ⇒  [∇_x K]_k = -½ A exp(x_k) ([G^{-1} p]_k)²
  const y = solveLowerTriangular(cholG, mom);
  const Ginvp = solveUpperTriangularT(cholG, y);
  const out: number[] = new Array(m.d);
  for (let k = 0; k < m.d; k++) {
    out[k] = -0.5 * m.cellArea * Math.exp(x[k]) * Ginvp[k] * Ginvp[k];
  }
  return out;
}

function lcpSampleMomentum(cholG: number[][], d: number, gauss: () => number): number[] {
  // p ~ N(0, G) ⇒ p = L_G z (L_G lower Cholesky of G)
  const z: number[] = new Array(d);
  for (let i = 0; i < d; i++) z[i] = gauss();
  const p: number[] = new Array(d).fill(0);
  for (let i = 0; i < d; i++) {
    for (let j = 0; j <= i; j++) p[i] += cholG[i][j] * z[j];
  }
  return p;
}

function lcpHamiltonian(x: number[], mom: number[], m: LCPModel, cholG?: number[][]): number {
  const logPi = lcpLogDensity(x, m);
  const cG = cholG ?? choleskyFactor(lcpMetric(x, m));
  const y = solveLowerTriangular(cG, mom);
  let K = 0;
  for (let i = 0; i < m.d; i++) K += 0.5 * y[i] * y[i]; // ½ p^T G^{-1} p = ½ |L^{-1} p|²
  const V = 0.5 * choleskyLogDet(cG);
  return -logPi + K + V;
}

function lcpGLStep(
  x: number[],
  mom: number[],
  eps: number,
  m: LCPModel,
  opts: GLStepOpts,
): GLStepResult {
  const tol = opts.tol ?? GL_DEFAULTS.tol;
  const maxIters = opts.maxIters ?? GL_DEFAULTS.maxIters;
  // Step 1: implicit momentum half-step
  const cholG0 = choleskyFactor(lcpMetric(x, m));
  const gradLog = lcpGradLog(x, m); // ∇ log π = -∇U
  const { gradV } = lcpVolumeAndGrad(cholG0, x, m);
  const b: number[] = new Array(m.d);
  for (let i = 0; i < m.d; i++) b[i] = mom[i] - 0.5 * eps * (-gradLog[i] + gradV[i]); // -∇U = -(-∇log π) — note ∇U = -∇log π
  let pHalf = b.slice();
  let pIters = 0;
  let pConverged = false;
  for (let k = 0; k < maxIters; k++) {
    const gK = lcpGradKinetic(cholG0, x, pHalf, m);
    const pNext: number[] = new Array(m.d);
    let dx = 0;
    let denom = 0;
    for (let i = 0; i < m.d; i++) {
      pNext[i] = b[i] - 0.5 * eps * gK[i];
      dx += (pNext[i] - pHalf[i]) ** 2;
      denom += pNext[i] ** 2;
    }
    pHalf = pNext;
    pIters = k + 1;
    if (Math.sqrt(dx) / (Math.sqrt(denom) + 1e-12) < tol) {
      pConverged = true;
      break;
    }
  }
  if (!pConverged) return { theta: x.slice(), mom: mom.slice(), pIters, thetaIters: 0, converged: false };
  // Step 2: implicit position step
  const y0 = solveLowerTriangular(cholG0, pHalf);
  const Ginvp0 = solveUpperTriangularT(cholG0, y0);
  const c: number[] = new Array(m.d);
  for (let i = 0; i < m.d; i++) c[i] = x[i] + 0.5 * eps * Ginvp0[i];
  let xNew = c.slice();
  let thetaIters = 0;
  let thetaConverged = false;
  let cholGN: number[][] = cholG0;
  for (let k = 0; k < maxIters; k++) {
    cholGN = choleskyFactor(lcpMetric(xNew, m));
    const yK = solveLowerTriangular(cholGN, pHalf);
    const GinvpK = solveUpperTriangularT(cholGN, yK);
    const next: number[] = new Array(m.d);
    let dx = 0;
    let denom = 0;
    for (let i = 0; i < m.d; i++) {
      next[i] = c[i] + 0.5 * eps * GinvpK[i];
      dx += (next[i] - xNew[i]) ** 2;
      denom += next[i] ** 2;
    }
    xNew = next;
    thetaIters = k + 1;
    if (Math.sqrt(dx) / (Math.sqrt(denom) + 1e-12) < tol) {
      thetaConverged = true;
      break;
    }
  }
  if (!thetaConverged) return { theta: x.slice(), mom: mom.slice(), pIters, thetaIters, converged: false };
  // Step 3: explicit momentum half-step
  const gradLog1 = lcpGradLog(xNew, m);
  const { gradV: gradV1 } = lcpVolumeAndGrad(cholGN, xNew, m);
  const gK1 = lcpGradKinetic(cholGN, xNew, pHalf, m);
  const momNew: number[] = new Array(m.d);
  for (let i = 0; i < m.d; i++) momNew[i] = pHalf[i] - 0.5 * eps * (-gradLog1[i] + gradV1[i] + gK1[i]);
  return { theta: xNew, mom: momNew, pIters, thetaIters, converged: true };
}

export function lcpRmhmcSample(
  x0: number[],
  nSamples: number,
  eps: number,
  L: number,
  m: LCPModel,
  rng: () => number,
  opts: GLStepOpts = {},
): ChainResult {
  const t0 = performance.now();
  const gauss = makeGaussian(rng);
  const samples: number[][] = [x0.slice()];
  let x = x0.slice();
  let accepts = 0;
  let divergences = 0;
  let totalPIters = 0;
  let totalThetaIters = 0;
  let totalGLSteps = 0;
  for (let n = 1; n < nSamples; n++) {
    const cholG = choleskyFactor(lcpMetric(x, m));
    const mom = lcpSampleMomentum(cholG, m.d, gauss);
    const H0 = lcpHamiltonian(x, mom, m, cholG);
    let xN = x.slice();
    let pN = mom.slice();
    let trajectoryDivergent = false;
    for (let l = 0; l < L; l++) {
      const r = lcpGLStep(xN, pN, eps, m, opts);
      totalPIters += r.pIters;
      totalThetaIters += r.thetaIters;
      totalGLSteps++;
      if (!r.converged) {
        trajectoryDivergent = true;
        break;
      }
      xN = r.theta;
      pN = r.mom;
    }
    if (trajectoryDivergent) {
      divergences++;
      samples.push(x.slice());
      continue;
    }
    const H1 = lcpHamiltonian(xN, pN, m);
    const accept = Math.log(rng()) < H0 - H1;
    if (accept) {
      x = xN;
      accepts++;
    }
    samples.push(x.slice());
  }
  return {
    samples,
    acceptanceRate: accepts / (nSamples - 1),
    divergences,
    meanPIters: totalGLSteps > 0 ? totalPIters / totalGLSteps : 0,
    meanThetaIters: totalGLSteps > 0 ? totalThetaIters / totalGLSteps : 0,
    wallSeconds: (performance.now() - t0) / 1000,
  };
}

// =============================================================================
// EFFECTIVE SAMPLE SIZE (for §8 ESS/sec panels)
// =============================================================================
// Geyer initial monotone-positive sequence (ArviZ default for split-half ESS).

/** Sample autocorrelation at lag k for a 1D chain. */
function autocorr(chain: number[], maxLag: number): number[] {
  const n = chain.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += chain[i];
  mean /= n;
  const acf: number[] = new Array(maxLag + 1);
  let var0 = 0;
  for (let i = 0; i < n; i++) var0 += (chain[i] - mean) ** 2;
  var0 /= n;
  if (var0 === 0) return new Array(maxLag + 1).fill(0);
  acf[0] = 1;
  for (let k = 1; k <= maxLag; k++) {
    let s = 0;
    for (let i = 0; i < n - k; i++) s += (chain[i] - mean) * (chain[i + k] - mean);
    acf[k] = s / (n * var0);
  }
  return acf;
}

/**
 * Effective sample size via Geyer initial positive sequence (single-chain
 * approximation; for multi-chain we apply per-chain and sum).
 */
export function ess(chain: number[]): number {
  const n = chain.length;
  if (n < 4) return n;
  const maxLag = Math.min(n - 1, 500);
  const acf = autocorr(chain, maxLag);
  // Sum pairs (acf[2k] + acf[2k+1]) until non-positive.
  let tau = -1;
  for (let k = 0; 2 * k + 1 <= maxLag; k++) {
    const sumPair = acf[2 * k] + acf[2 * k + 1];
    if (sumPair <= 0) break;
    tau += 2 * sumPair;
  }
  if (tau < 1) tau = 1;
  return n / tau;
}
