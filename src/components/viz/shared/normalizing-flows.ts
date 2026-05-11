// =============================================================================
// normalizing-flows.ts
//
// Shared math primitives for the normalizing-flows topic's viz components.
// Ported from notebook §4–§6 (AffineCoupling, CouplingFlow, MAFLayer,
// Conv1x1Dense, Conv1x1LU). The TS port replaces the notebook's PyTorch
// neural-network parameterization with deterministic analytic scale/shift
// maps — the viz components illustrate the math, not the training. The
// math identities (block-triangular Jacobian, log-det = sum of diagonals,
// closed-form inverse, log-det additivity across composition) hold
// identically under either parameterization.
//
// All exports are pure functions / immutable classes — no module-level state,
// deterministic outputs for a given seed.
//
// Source-of-truth notebook: notebooks/normalizing-flows/01_normalizing_flows.ipynb
// Brief: docs/plans/formalml-normalizing-flows-handoff-brief.md
// =============================================================================

// -----------------------------------------------------------------------------
// Deterministic RNG: mulberry32 + Box-Muller.
//
// Substitutes for NumPy's PCG64 + Ziggurat in the notebook. Same role
// (seeded standard-normal sampler), different distribution of stream
// values — so monte-carlo agreement is "rate-scaling" not "bit-exact".
// -----------------------------------------------------------------------------

/** Mulberry32 PRNG: 32-bit state, period 2³², good distribution properties. */
export function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box-Muller-based standard-normal sampler. Returns iid N(0, 1) draws. */
export function gaussianRng(rng: () => number): () => number {
  let spare: number | null = null;
  return () => {
    if (spare !== null) {
      const v = spare;
      spare = null;
      return v;
    }
    let u = 0;
    let v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    const mag = Math.sqrt(-2 * Math.log(u));
    spare = mag * Math.sin(2 * Math.PI * v);
    return mag * Math.cos(2 * Math.PI * v);
  };
}

/** Sample n iid d-dimensional standard normals; row-major (n × d). */
export function sampleStandardNormalBatch(n: number, d: number, seed: number): Float64Array[] {
  const rng = mulberry32(seed);
  const g = gaussianRng(rng);
  const out: Float64Array[] = [];
  for (let i = 0; i < n; i++) {
    const row = new Float64Array(d);
    for (let j = 0; j < d; j++) row[j] = g();
    out.push(row);
  }
  return out;
}

// -----------------------------------------------------------------------------
// Mask helpers — checkerboard / alternating split.
// -----------------------------------------------------------------------------

/**
 * Alternating mask used by CouplingFlow.
 *
 * Layer index even → mask = [1, 1, ..., 1, 0, 0, ..., 0] (first d/2 are
 * pass-through). Odd layers → bit-flipped.
 *
 * @param d - data dimension
 * @param layerIdx - 0-indexed layer position in the stack
 */
export function alternatingMask(d: number, layerIdx: number): Float64Array {
  const m = new Float64Array(d);
  const half = Math.floor(d / 2);
  if (layerIdx % 2 === 0) {
    for (let i = 0; i < half; i++) m[i] = 1;
  } else {
    for (let i = half; i < d; i++) m[i] = 1;
  }
  return m;
}

/** Indices where mask == 1 (pass-through, set A). */
export function maskPassThroughIndices(mask: Float64Array): number[] {
  const A: number[] = [];
  for (let i = 0; i < mask.length; i++) if (mask[i] === 1) A.push(i);
  return A;
}

/** Indices where mask == 0 (transformed, set B). */
export function maskTransformedIndices(mask: Float64Array): number[] {
  const B: number[] = [];
  for (let i = 0; i < mask.length; i++) if (mask[i] === 0) B.push(i);
  return B;
}

// -----------------------------------------------------------------------------
// AffineCoupling — RealNVP coupling layer with deterministic analytic s, t.
//
// Forward:   x_A = z_A                                       (pass-through)
//            x_i = z_i * exp(s_i(z_A)) + t_i(z_A)   for i ∈ B
// Inverse:   z_A = x_A
//            z_i = (x_i - t_i(x_A)) * exp(-s_i(x_A))  for i ∈ B
// Log-det:   log|det dx/dz| = Σ_{i ∈ B} s_i(z_A)
//
// Per-dim scale and shift:
//   s_i(z_A) = α * tanh(W_s[i] · z_A + b_s[i])     bounded scalar
//   t_i(z_A) = W_t[i] · z_A + b_t[i]                linear scalar
//
// Layer parameters (Ws, bs, Wt, bt, scaleAmp) are set once at construction
// from a seeded RNG; they are NOT trained — the educational viz illustrates
// the architectural identities under any deterministic choice of s, t.
// -----------------------------------------------------------------------------

export interface AffineCouplingOptions {
  /** Data dimension. */
  d: number;
  /** Binary mask, length d. 1 = pass-through, 0 = transformed. */
  mask: Float64Array;
  /** Seed used to generate deterministic layer parameters. */
  seed?: number;
  /** Amplitude bound on the tanh-clamped scale s_i (default 1.0; matches notebook tanh). */
  scaleAmp?: number;
  /** Magnitude of the random Ws/Wt initialization (default 0.6). */
  paramScale?: number;
}

export class AffineCoupling {
  readonly d: number;
  readonly mask: Float64Array;
  readonly A: number[]; // pass-through indices
  readonly B: number[]; // transformed indices
  readonly scaleAmp: number;

  // Per-i-in-B params. We store as flat arrays indexed by position-in-B.
  // Ws[k] is a |A|-vector for the k-th transformed dim; same for Wt.
  readonly Ws: Float64Array[];
  readonly bs: Float64Array;
  readonly Wt: Float64Array[];
  readonly bt: Float64Array;

  constructor(opts: AffineCouplingOptions) {
    const { d, mask, seed = 1, scaleAmp = 1.0, paramScale = 0.6 } = opts;
    if (mask.length !== d) throw new Error(`mask length ${mask.length} != d ${d}`);
    this.d = d;
    this.mask = mask.slice();
    this.A = maskPassThroughIndices(this.mask);
    this.B = maskTransformedIndices(this.mask);
    this.scaleAmp = scaleAmp;

    const rng = mulberry32(seed);
    const g = gaussianRng(rng);
    const aLen = this.A.length;
    this.Ws = [];
    this.bs = new Float64Array(this.B.length);
    this.Wt = [];
    this.bt = new Float64Array(this.B.length);
    for (let k = 0; k < this.B.length; k++) {
      const ws = new Float64Array(aLen);
      const wt = new Float64Array(aLen);
      for (let j = 0; j < aLen; j++) ws[j] = paramScale * g();
      for (let j = 0; j < aLen; j++) wt[j] = paramScale * g();
      this.Ws.push(ws);
      this.Wt.push(wt);
      this.bs[k] = paramScale * g();
      this.bt[k] = paramScale * g();
    }
  }

  /** Compute s_i(z_A), t_i(z_A) for all i ∈ B. Returns parallel arrays of length |B|. */
  private computeST(zA: Float64Array): { s: Float64Array; t: Float64Array } {
    const k = this.B.length;
    const s = new Float64Array(k);
    const t = new Float64Array(k);
    for (let i = 0; i < k; i++) {
      let dotS = this.bs[i];
      let dotT = this.bt[i];
      for (let j = 0; j < this.A.length; j++) {
        dotS += this.Ws[i][j] * zA[j];
        dotT += this.Wt[i][j] * zA[j];
      }
      s[i] = this.scaleAmp * Math.tanh(dotS);
      t[i] = dotT;
    }
    return { s, t };
  }

  /** Extract the |A| pass-through entries from a length-d vector. */
  private extractA(z: Float64Array): Float64Array {
    const out = new Float64Array(this.A.length);
    for (let j = 0; j < this.A.length; j++) out[j] = z[this.A[j]];
    return out;
  }

  /** Forward pass: returns { x, logDet }. */
  forward(z: Float64Array): { x: Float64Array; logDet: number } {
    const x = new Float64Array(this.d);
    const zA = this.extractA(z);
    const { s, t } = this.computeST(zA);
    // Pass-through copy.
    for (let j = 0; j < this.A.length; j++) x[this.A[j]] = z[this.A[j]];
    // Transformed: x_i = z_i * exp(s_i) + t_i.
    let logDet = 0;
    for (let k = 0; k < this.B.length; k++) {
      const i = this.B[k];
      x[i] = z[i] * Math.exp(s[k]) + t[k];
      logDet += s[k];
    }
    return { x, logDet };
  }

  /** Inverse pass: returns { z, logDet } where logDet = log|det dT^-1/dx| = -Σ s_i(x_A). */
  inverse(x: Float64Array): { z: Float64Array; logDet: number } {
    const z = new Float64Array(this.d);
    const xA = this.extractA(x);
    const { s, t } = this.computeST(xA);
    for (let j = 0; j < this.A.length; j++) z[this.A[j]] = x[this.A[j]];
    let logDet = 0;
    for (let k = 0; k < this.B.length; k++) {
      const i = this.B[k];
      z[i] = (x[i] - t[k]) * Math.exp(-s[k]);
      logDet -= s[k];
    }
    return { z, logDet };
  }

  /** Full d×d Jacobian ∂x/∂z at z. Block-triangular structure (verified by tests). */
  jacobian(z: Float64Array): Float64Array[] {
    const J: Float64Array[] = [];
    for (let i = 0; i < this.d; i++) J.push(new Float64Array(this.d));
    // Pass-through rows are identity (∂x_i/∂z_j = δ_ij for i ∈ A).
    for (const i of this.A) J[i][i] = 1;
    // Transformed rows:
    //   x_i = z_i * exp(s_i(z_A)) + t_i(z_A)
    //   ∂x_i/∂z_i = exp(s_i(z_A))                         (diagonal of B block)
    //   ∂x_i/∂z_j = z_i * exp(s_i) * ∂s_i/∂z_j + ∂t_i/∂z_j  for j ∈ A (lower-left)
    //   ∂x_i/∂z_j = 0                                      for j ∈ B, j ≠ i (off-diag of B block)
    //
    // ∂s_i/∂z_{A[m]} = scaleAmp * sech²(dotS_i) * Ws[i][m]
    // ∂t_i/∂z_{A[m]} = Wt[i][m]
    const zA = this.extractA(z);
    for (let k = 0; k < this.B.length; k++) {
      const i = this.B[k];
      let dotS = this.bs[k];
      for (let j = 0; j < this.A.length; j++) dotS += this.Ws[k][j] * zA[j];
      const expS = Math.exp(this.scaleAmp * Math.tanh(dotS));
      const sech2 = 1 - Math.tanh(dotS) ** 2;
      J[i][i] = expS;
      for (let m = 0; m < this.A.length; m++) {
        const dsdz = this.scaleAmp * sech2 * this.Ws[k][m];
        const dtdz = this.Wt[k][m];
        J[i][this.A[m]] = z[i] * expS * dsdz + dtdz;
      }
    }
    return J;
  }
}

// -----------------------------------------------------------------------------
// CouplingFlow — stack of AffineCoupling layers with alternating masks.
//
// Forward: T(z) = T_K ∘ ... ∘ T_1(z)
// Log-det: sum_k log|det dT_k/dh_{k-1}|  (composition rule, §3.2 eq. 3.4)
// Inverse: T^-1(x) = T_1^-1 ∘ ... ∘ T_K^-1(x)
// -----------------------------------------------------------------------------

export interface CouplingFlowOptions {
  d: number;
  nLayers: number;
  seed?: number;
  scaleAmp?: number;
  paramScale?: number;
}

export class CouplingFlow {
  readonly d: number;
  readonly layers: AffineCoupling[];

  constructor(opts: CouplingFlowOptions) {
    const { d, nLayers, seed = 1, scaleAmp = 1.0, paramScale = 0.6 } = opts;
    this.d = d;
    this.layers = [];
    for (let k = 0; k < nLayers; k++) {
      const mask = alternatingMask(d, k);
      this.layers.push(
        new AffineCoupling({ d, mask, seed: seed + k * 97, scaleAmp, paramScale }),
      );
    }
  }

  /** Forward pass returning final x, total log-det, and per-layer intermediates h_k = T_k(h_{k-1}). */
  forward(z: Float64Array): { x: Float64Array; logDet: number; intermediates: Float64Array[] } {
    let h = z.slice();
    let logDet = 0;
    const intermediates: Float64Array[] = [h.slice()];
    for (const layer of this.layers) {
      const r = layer.forward(h);
      h = r.x;
      logDet += r.logDet;
      intermediates.push(h.slice());
    }
    return { x: h, logDet, intermediates };
  }

  /** Inverse pass: applies layer inverses in reverse order. */
  inverse(x: Float64Array): { z: Float64Array; logDet: number; intermediates: Float64Array[] } {
    let h = x.slice();
    let logDet = 0;
    const intermediates: Float64Array[] = [h.slice()];
    for (let k = this.layers.length - 1; k >= 0; k--) {
      const r = this.layers[k].inverse(h);
      h = r.z;
      logDet += r.logDet;
      intermediates.push(h.slice());
    }
    return { z: h, logDet, intermediates };
  }
}

// -----------------------------------------------------------------------------
// MAFLayer — Masked Autoregressive Flow (Papamakarios et al. 2017).
//
// Forward (z → x, SEQUENTIAL):
//   x_i = z_i * exp(s_i(x_{<i})) + t_i(x_{<i}), i = 0, ..., d-1
// Inverse (x → z, PARALLEL):
//   z_i = (x_i - t_i(x_{<i})) * exp(-s_i(x_{<i}))
// Log-det:
//   log|det dx/dz| = Σ_i s_i(x_{<i})
//
// Per-dim analytic s_i, t_i — for i = 0 they are unconditional scalars
// (bias-only); for i ≥ 1 they are linear+tanh maps of the prefix x_{<i}.
// -----------------------------------------------------------------------------

export interface MAFLayerOptions {
  d: number;
  seed?: number;
  scaleAmp?: number;
  paramScale?: number;
}

export class MAFLayer {
  readonly d: number;
  readonly scaleAmp: number;
  /** Per-dim params: [d] vectors of scalar bias for i=0, and per-dim weight vectors for i ≥ 1. */
  readonly bs: Float64Array;
  readonly bt: Float64Array;
  readonly Ws: Float64Array[]; // Ws[i] has length i (one weight per earlier dim)
  readonly Wt: Float64Array[];

  constructor(opts: MAFLayerOptions) {
    const { d, seed = 1, scaleAmp = 1.0, paramScale = 0.6 } = opts;
    this.d = d;
    this.scaleAmp = scaleAmp;
    const rng = mulberry32(seed);
    const g = gaussianRng(rng);
    this.bs = new Float64Array(d);
    this.bt = new Float64Array(d);
    this.Ws = [];
    this.Wt = [];
    for (let i = 0; i < d; i++) {
      this.bs[i] = paramScale * g();
      this.bt[i] = paramScale * g();
      const ws = new Float64Array(i);
      const wt = new Float64Array(i);
      for (let j = 0; j < i; j++) ws[j] = paramScale * g();
      for (let j = 0; j < i; j++) wt[j] = paramScale * g();
      this.Ws.push(ws);
      this.Wt.push(wt);
    }
  }

  /** Compute s_i, t_i given the prefix x_{<i}. */
  private perDim(prefix: Float64Array, i: number): { s: number; t: number } {
    let dotS = this.bs[i];
    let dotT = this.bt[i];
    for (let j = 0; j < i; j++) {
      dotS += this.Ws[i][j] * prefix[j];
      dotT += this.Wt[i][j] * prefix[j];
    }
    return { s: this.scaleAmp * Math.tanh(dotS), t: dotT };
  }

  /** Forward pass: SEQUENTIAL — x_i depends on x_{<i}. */
  forward(z: Float64Array): { x: Float64Array; logDet: number } {
    const x = new Float64Array(this.d);
    let logDet = 0;
    for (let i = 0; i < this.d; i++) {
      const { s, t } = this.perDim(x, i);
      x[i] = z[i] * Math.exp(s) + t;
      logDet += s;
    }
    return { x, logDet };
  }

  /** Inverse pass: PARALLEL — all z_i computed from x_{<i} in any order. */
  inverse(x: Float64Array): { z: Float64Array; logDet: number } {
    const z = new Float64Array(this.d);
    let logDet = 0;
    for (let i = 0; i < this.d; i++) {
      const { s, t } = this.perDim(x, i);
      z[i] = (x[i] - t) * Math.exp(-s);
      logDet -= s;
    }
    return { z, logDet };
  }

  /** Full d×d Jacobian — lower-triangular by construction (autoregressive). */
  jacobian(z: Float64Array): Float64Array[] {
    // To compute ∂x_i/∂z_j for j ≤ i, we use the chain rule along the
    // sequential forward pass. Cheaper: forward through x, then for each
    // (i, j) with j ≤ i, ∂x_i/∂z_j = exp(s_i) δ_ij + (terms from x_{<i}
    // depending on z_{≤j}).
    //
    // Simplest correct way: finite differences (this is for the verify suite).
    // For the production hot path, derive analytically. We do FD here for
    // clarity; verify-suite uses FD too.
    return finiteDifferenceJacobian((u) => this.forward(u).x, z);
  }
}

// -----------------------------------------------------------------------------
// Conv1x1Dense — 1×1 invertible convolution with dense weight W ∈ R^{C×C}.
//
// On a single C × H × W tensor with batch=1:
//   y_{c, i, j} = Σ_{c'} W_{c, c'} x_{c', i, j}
// Jacobian of full flat operation is block-diagonal with H·W copies of W.
// log|det ∂y/∂x| = H · W · log|det W|.
// -----------------------------------------------------------------------------

export interface Conv1x1Options {
  C: number;
  seed?: number;
  /** Initialization strategy: "orthogonal" (QR of random Gaussian) or "gaussian". */
  init?: 'orthogonal' | 'gaussian';
}

export class Conv1x1Dense {
  readonly C: number;
  readonly W: Float64Array[]; // C × C row-major

  constructor(opts: Conv1x1Options) {
    const { C, seed = 1, init = 'orthogonal' } = opts;
    this.C = C;
    const rng = mulberry32(seed);
    const g = gaussianRng(rng);
    if (init === 'orthogonal') {
      // Build random C×C, then QR-orthogonalize via modified Gram-Schmidt.
      const M: Float64Array[] = [];
      for (let i = 0; i < C; i++) {
        const row = new Float64Array(C);
        for (let j = 0; j < C; j++) row[j] = g();
        M.push(row);
      }
      // QR via column-wise Gram-Schmidt on M^T (since rows of W become columns).
      this.W = gramSchmidtRows(M);
    } else {
      const M: Float64Array[] = [];
      for (let i = 0; i < C; i++) {
        const row = new Float64Array(C);
        for (let j = 0; j < C; j++) row[j] = 0.4 * g();
        M.push(row);
      }
      this.W = M;
    }
  }

  /** Apply W to an input of shape (C,) at a single spatial location. */
  applySingle(x: Float64Array): Float64Array {
    return matVec(this.W, x);
  }

  /** log|det W| via slogdet (LU-based). */
  logDetW(): number {
    return slogdetAbs(this.W);
  }

  /** Forward on a flat C·H·W input (row-major: c then h then w). Returns flat output + logDet. */
  forward(x: Float64Array, H: number, W: number): { y: Float64Array; logDet: number } {
    if (x.length !== this.C * H * W) {
      throw new Error(`Conv1x1Dense.forward: expected length ${this.C * H * W}, got ${x.length}`);
    }
    const y = new Float64Array(x.length);
    for (let h = 0; h < H; h++) {
      for (let w = 0; w < W; w++) {
        // Pull out the C-vector at (h, w), apply W, scatter back.
        const xVec = new Float64Array(this.C);
        for (let c = 0; c < this.C; c++) xVec[c] = x[c * H * W + h * W + w];
        const yVec = this.applySingle(xVec);
        for (let c = 0; c < this.C; c++) y[c * H * W + h * W + w] = yVec[c];
      }
    }
    const logDet = this.logDetW() * H * W;
    return { y, logDet };
  }
}

// -----------------------------------------------------------------------------
// Conv1x1LU — Glow's LU-parameterized 1×1 conv.
//
// W = P · L · (U_strict + diag(exp(s_log)))
// where:
//   P: fixed permutation matrix (C × C)
//   L: lower-triangular with 1s on diagonal, free below
//   U_strict: strictly upper-triangular (zeros on diag and below)
//   exp(s_log): C positive diagonal scale factors
//
// log|det W| = Σ_i s_log[i]   (since det P = ±1, det L = 1, det(U_strict+diag) = Π exp(s_log_i))
//
// This is the load-bearing identity that brings the dense-W's O(C³) log-det
// down to O(C). See §6.2 (4.6.1 in brief).
// -----------------------------------------------------------------------------

export class Conv1x1LU {
  readonly C: number;
  /** Permutation as an index array: P @ x = x[perm]. */
  readonly perm: number[];
  /** Strict lower-triangular C × C (zeros on/above diag). */
  readonly L: Float64Array[];
  /** Strict upper-triangular C × C (zeros on/below diag). */
  readonly U: Float64Array[];
  /** Log-scale parameters: actual diag(U) entries are exp(sLog). */
  readonly sLog: Float64Array;

  constructor(opts: Conv1x1Options) {
    const { C, seed = 1 } = opts;
    this.C = C;
    const rng = mulberry32(seed);
    const g = gaussianRng(rng);

    // Random permutation via Fisher-Yates.
    this.perm = Array.from({ length: C }, (_, i) => i);
    for (let i = C - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      const tmp = this.perm[i];
      this.perm[i] = this.perm[j];
      this.perm[j] = tmp;
    }

    this.L = [];
    this.U = [];
    for (let i = 0; i < C; i++) {
      this.L.push(new Float64Array(C));
      this.U.push(new Float64Array(C));
    }
    for (let i = 0; i < C; i++) {
      for (let j = 0; j < i; j++) this.L[i][j] = 0.1 * g(); // strictly below diag
      for (let j = i + 1; j < C; j++) this.U[i][j] = 0.1 * g(); // strictly above diag
    }
    this.sLog = new Float64Array(C);
    for (let i = 0; i < C; i++) this.sLog[i] = 0.1 * g();
  }

  /** Materialize W = P · L_full · U_full where L_full = L + I, U_full = U_strict + diag(exp(s_log)). */
  buildW(): Float64Array[] {
    const C = this.C;
    // Compute Lfull * Ufull explicitly (both already in this.L, this.U).
    // M = Lfull * Ufull. Lfull[i][j] = L[i][j] for j<i; 1 for j=i; 0 for j>i.
    // Ufull[i][j] = U[i][j] for j>i; exp(sLog[i]) for j=i; 0 for j<i.
    const M: Float64Array[] = [];
    for (let i = 0; i < C; i++) M.push(new Float64Array(C));
    for (let i = 0; i < C; i++) {
      for (let j = 0; j < C; j++) {
        let sum = 0;
        const kMax = Math.min(i, j);
        for (let k = 0; k <= kMax; k++) {
          const Lik = k < i ? this.L[i][k] : 1; // k <= i, and Lfull[i][i] = 1
          const Ukj = k < j ? this.U[k][j] : k === j ? Math.exp(this.sLog[k]) : 0;
          sum += Lik * Ukj;
        }
        M[i][j] = sum;
      }
    }
    // Now apply P from the left: (P · M)[i][j] = M[perm[i]][j].
    const W: Float64Array[] = [];
    for (let i = 0; i < C; i++) W.push(M[this.perm[i]]);
    return W;
  }

  /** log|det W| = Σ s_log_i  (O(C) — the headline identity). */
  logDetW(): number {
    let s = 0;
    for (let i = 0; i < this.C; i++) s += this.sLog[i];
    return s;
  }

  forward(x: Float64Array, H: number, W: number): { y: Float64Array; logDet: number } {
    const Wmat = this.buildW();
    const out = new Float64Array(x.length);
    for (let h = 0; h < H; h++) {
      for (let w = 0; w < W; w++) {
        const xVec = new Float64Array(this.C);
        for (let c = 0; c < this.C; c++) xVec[c] = x[c * H * W + h * W + w];
        const yVec = matVec(Wmat, xVec);
        for (let c = 0; c < this.C; c++) out[c * H * W + h * W + w] = yVec[c];
      }
    }
    const logDet = this.logDetW() * H * W;
    return { y: out, logDet };
  }
}

// -----------------------------------------------------------------------------
// Linear-algebra primitives — pure JS, no external deps.
// -----------------------------------------------------------------------------

/** y = M · x   where M is C × C and x is length C. */
export function matVec(M: Float64Array[], x: Float64Array): Float64Array {
  const C = M.length;
  const y = new Float64Array(C);
  for (let i = 0; i < C; i++) {
    let s = 0;
    for (let j = 0; j < C; j++) s += M[i][j] * x[j];
    y[i] = s;
  }
  return y;
}

/** Row-Gram-Schmidt: returns Q (rows orthonormal) for the given matrix M (rows = input vectors). */
export function gramSchmidtRows(M: Float64Array[]): Float64Array[] {
  const n = M.length;
  const Q: Float64Array[] = [];
  for (let i = 0; i < n; i++) {
    const v = M[i].slice();
    for (let j = 0; j < i; j++) {
      let dot = 0;
      for (let k = 0; k < v.length; k++) dot += v[k] * Q[j][k];
      for (let k = 0; k < v.length; k++) v[k] -= dot * Q[j][k];
    }
    let norm = 0;
    for (let k = 0; k < v.length; k++) norm += v[k] * v[k];
    norm = Math.sqrt(norm);
    if (norm > 1e-14) {
      for (let k = 0; k < v.length; k++) v[k] /= norm;
    }
    Q.push(v);
  }
  return Q;
}

/**
 * log|det M| via LU decomposition with partial pivoting.
 *
 * Returns log|det M|. For singular matrices returns -Infinity; for
 * ill-conditioned matrices may produce NaN. The caller is responsible
 * for handling those edge cases.
 */
export function slogdetAbs(M: Float64Array[]): number {
  const n = M.length;
  // Copy.
  const A: Float64Array[] = [];
  for (let i = 0; i < n; i++) A.push(M[i].slice());
  let logAbsDet = 0;
  for (let k = 0; k < n; k++) {
    // Partial pivot.
    let pivot = k;
    let pivotAbs = Math.abs(A[k][k]);
    for (let i = k + 1; i < n; i++) {
      if (Math.abs(A[i][k]) > pivotAbs) {
        pivot = i;
        pivotAbs = Math.abs(A[i][k]);
      }
    }
    if (pivot !== k) {
      const tmp = A[k];
      A[k] = A[pivot];
      A[pivot] = tmp;
    }
    if (A[k][k] === 0) return -Infinity;
    logAbsDet += Math.log(Math.abs(A[k][k]));
    // Eliminate below.
    for (let i = k + 1; i < n; i++) {
      const factor = A[i][k] / A[k][k];
      for (let j = k; j < n; j++) A[i][j] -= factor * A[k][j];
    }
  }
  return logAbsDet;
}

/**
 * Finite-difference Jacobian — central differences, eps = 1e-6.
 *
 * Returns a (output-dim) × (input-dim) matrix. Used by the verify suite to
 * check closed-form log-dets vs autograd-style numerical Jacobian-determinants.
 */
export function finiteDifferenceJacobian(
  fn: (x: Float64Array) => Float64Array,
  x: Float64Array,
  eps = 1e-6,
): Float64Array[] {
  const fxProbe = fn(x);
  const outDim = fxProbe.length;
  const inDim = x.length;
  const J: Float64Array[] = [];
  for (let i = 0; i < outDim; i++) J.push(new Float64Array(inDim));
  // Hoist the perturbation buffers and the constant 1/(2 eps) out of the loop
  // (Gemini PR #82 review): saves O(d) Float64Array allocations and avoids the
  // hot-path division. Each iteration restores xPlus[j] / xMinus[j] before
  // moving on so the buffer is reused without aliasing across columns.
  const xPlus = x.slice();
  const xMinus = x.slice();
  const invTwoEps = 1 / (2 * eps);
  for (let j = 0; j < inDim; j++) {
    const orig = x[j];
    xPlus[j] = orig + eps;
    xMinus[j] = orig - eps;
    const fPlus = fn(xPlus);
    const fMinus = fn(xMinus);
    for (let i = 0; i < outDim; i++) {
      J[i][j] = (fPlus[i] - fMinus[i]) * invTwoEps;
    }
    xPlus[j] = orig;
    xMinus[j] = orig;
  }
  return J;
}

// -----------------------------------------------------------------------------
// 2-D demo target distributions — used by viz components for pushforward plots.
// -----------------------------------------------------------------------------

/** Sample n points from a 2-D bimodal mixture (centered at ±2, σ = 0.5). */
export function sampleBimodal2D(n: number, seed: number, sigma = 0.5, sep = 2): Float64Array[] {
  const rng = mulberry32(seed);
  const g = gaussianRng(rng);
  const pts: Float64Array[] = [];
  for (let i = 0; i < n; i++) {
    const which = rng() < 0.5 ? -1 : 1;
    pts.push(new Float64Array([which * sep + sigma * g(), sigma * g()]));
  }
  return pts;
}

/** Sample n points from the 2-moons dataset (sklearn defaults: noise σ = 0.05). */
export function sampleTwoMoons(n: number, seed: number, noise = 0.05): Float64Array[] {
  const rng = mulberry32(seed);
  const g = gaussianRng(rng);
  const pts: Float64Array[] = [];
  const nOut = Math.floor(n / 2);
  const nIn = n - nOut;
  // Outer moon: half-circle at (0, 0), radius 1.
  for (let i = 0; i < nOut; i++) {
    const theta = (i / nOut) * Math.PI;
    pts.push(new Float64Array([Math.cos(theta) + noise * g(), Math.sin(theta) + noise * g()]));
  }
  // Inner moon: half-circle at (1, 0.5), radius 1, opposite orientation.
  for (let i = 0; i < nIn; i++) {
    const theta = (i / nIn) * Math.PI;
    pts.push(new Float64Array([1 - Math.cos(theta) + noise * g(), 0.5 - Math.sin(theta) + noise * g()]));
  }
  // Shuffle (Fisher-Yates with rng).
  for (let i = pts.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = pts[i];
    pts[i] = pts[j];
    pts[j] = tmp;
  }
  return pts;
}

/**
 * Standard-normal log-density on R^d:  log p_Z(z) = -d/2 log(2π) - ||z||²/2.
 */
export function logStandardNormal(z: Float64Array): number {
  let sq = 0;
  for (let i = 0; i < z.length; i++) sq += z[i] * z[i];
  return -0.5 * z.length * Math.log(2 * Math.PI) - 0.5 * sq;
}
