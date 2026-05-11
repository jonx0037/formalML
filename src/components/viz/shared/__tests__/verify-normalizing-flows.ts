// =============================================================================
// verify-normalizing-flows.ts
//
// Numerical regression tests for src/components/viz/shared/normalizing-flows.ts.
// Each test reproduces a numerical identity from the verified notebook
//   notebooks/normalizing-flows/01_normalizing_flows.ipynb
// and asserts the TS port satisfies the identity to TS-computable tolerance.
//
// Run with: pnpm verify:normalizing-flows
// Exits non-zero on first set of failures.
//
// SCOPE: 7 closed-form identities computable in pure TS. The brief's §7/§10/§11
// numerical results (trained-flow NLL, KDE-vs-flow inequality, latent moments)
// require PyTorch training and are out-of-scope for this v1 suite. They are
// documented in the brief's Implementation-notes section and asserted in the
// notebook directly.
//
// Notebook reference values for the Jacobian-autograd checks:
//   §4.3 affine coupling Jacobian:        |closed - autograd| = 9.71e-17
//   §6.2 Conv1x1Dense Jacobian:           |closed - autograd| = 3.11e-15
//   §6.2 Conv1x1LU log|det W| vs slogdet: tolerance 1e-12 in notebook
// =============================================================================

import {
  AffineCoupling,
  Conv1x1Dense,
  Conv1x1LU,
  CouplingFlow,
  MAFLayer,
  alternatingMask,
  finiteDifferenceJacobian,
  matVec,
  mulberry32,
  gaussianRng,
  sampleStandardNormalBatch,
  slogdetAbs,
} from '../normalizing-flows';

// -----------------------------------------------------------------------------
// Test plumbing — same shape as verify-kernel-regression.ts.
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
    `${label} observed=${observed.toExponential(4)} expected=${expected.toExponential(4)} gap=${gap.toExponential(3)} tol=${tol.toExponential(3)}`,
  );
}

function within(name: string, observed: number, lo: number, hi: number, label: string): void {
  ok(name, observed >= lo && observed <= hi, `${label} observed=${observed.toExponential(3)} in [${lo}, ${hi}]`);
}

// -----------------------------------------------------------------------------
// Test 1: AffineCoupling Jacobian — closed-form sum of s_i vs full Jacobian
// determinant via finite-difference. Notebook §4.3.
// Tolerance: 1e-6 (loosened from notebook's 1e-10 because central-difference
// FD has truncation error O(eps²) ≈ 1e-12 plus condition-number amplification).
// -----------------------------------------------------------------------------

console.log('\n[1] affine_coupling_jacobian — closed-form Σ s_i vs FD slogdet');

{
  const d = 4;
  const mask = new Float64Array([1, 1, 0, 0]);
  const layer = new AffineCoupling({ d, mask, seed: 20260511, scaleAmp: 1.0, paramScale: 0.6 });
  // Test on 5 different z draws.
  const seeds = [11, 22, 33, 44, 55];
  for (const s of seeds) {
    const zBatch = sampleStandardNormalBatch(1, d, s);
    const z = zBatch[0];
    const closed = layer.forward(z).logDet;
    const J = finiteDifferenceJacobian((u) => layer.forward(u).x, z);
    const fdLogDet = slogdetAbs(J);
    approxEq(`affine_coupling_jacobian[seed=${s}]`, fdLogDet, closed, 1e-6, 'Σ s_i vs FD slogdet');
  }
}

// -----------------------------------------------------------------------------
// Test 2: AffineCoupling round-trip — inverse(forward(z)) ≈ z to high precision.
// Notebook §4.3 assertion (1e-10).
// -----------------------------------------------------------------------------

console.log('\n[2] affine_coupling_roundtrip — inverse(forward(z)) ≈ z');

{
  const d = 4;
  const mask = new Float64Array([1, 1, 0, 0]);
  const layer = new AffineCoupling({ d, mask, seed: 20260511 });
  const zs = sampleStandardNormalBatch(8, d, 99);
  let maxErr = 0;
  let maxLogDetErr = 0;
  for (const z of zs) {
    const { x, logDet: ldFwd } = layer.forward(z);
    const { z: zBack, logDet: ldInv } = layer.inverse(x);
    for (let i = 0; i < d; i++) {
      const e = Math.abs(zBack[i] - z[i]);
      if (e > maxErr) maxErr = e;
    }
    const ldGap = Math.abs(ldFwd + ldInv);
    if (ldGap > maxLogDetErr) maxLogDetErr = ldGap;
  }
  within('affine_coupling_roundtrip_z', maxErr, 0, 1e-12, 'max |inverse(forward(z)) - z|');
  within('affine_coupling_logdet_pair', maxLogDetErr, 0, 1e-12, 'max |ld_fwd + ld_inv|');
}

// -----------------------------------------------------------------------------
// Test 3: CouplingFlow 6-layer round-trip — composition preserves invertibility.
// Notebook §4.4 (1e-10).
// -----------------------------------------------------------------------------

console.log('\n[3] coupling_flow_roundtrip — 6-layer stack inverse(forward(z)) ≈ z');

{
  const d = 2;
  const flow = new CouplingFlow({ d, nLayers: 6, seed: 20260511 });
  const zs = sampleStandardNormalBatch(16, d, 77);
  let maxErr = 0;
  let maxLogDetErr = 0;
  for (const z of zs) {
    const { x, logDet: ldFwd } = flow.forward(z);
    const { z: zBack, logDet: ldInv } = flow.inverse(x);
    for (let i = 0; i < d; i++) {
      const e = Math.abs(zBack[i] - z[i]);
      if (e > maxErr) maxErr = e;
    }
    const ldGap = Math.abs(ldFwd + ldInv);
    if (ldGap > maxLogDetErr) maxLogDetErr = ldGap;
  }
  within('coupling_flow_roundtrip_z', maxErr, 0, 1e-10, 'max |inverse(forward(z)) - z|');
  within('coupling_flow_logdet_pair', maxLogDetErr, 0, 1e-10, 'max |ld_fwd + ld_inv|');
}

// -----------------------------------------------------------------------------
// Test 4: log-det additivity across composition — eq. (3.4).
// Layered sum of log-dets = total log-det from CouplingFlow.forward.
// -----------------------------------------------------------------------------

console.log('\n[4] logdet_additivity — Σ_k log|det dT_k| = log|det dT|');

{
  const d = 2;
  const flow = new CouplingFlow({ d, nLayers: 4, seed: 20260511 });
  const z = sampleStandardNormalBatch(1, d, 13)[0];
  // Full flow's total log-det:
  const ldFull = flow.forward(z).logDet;
  // Layered sum: forward each layer in turn, accumulate per-layer log-dets.
  let h = z.slice();
  let ldLayered = 0;
  for (const layer of flow.layers) {
    const r = layer.forward(h);
    h = r.x;
    ldLayered += r.logDet;
  }
  approxEq('logdet_additivity', ldLayered, ldFull, 1e-12, 'Σ_k ld_k vs ld_full');
}

// -----------------------------------------------------------------------------
// Test 5: Conv1x1Dense closed-form log|det W| vs slogdet vs FD Jacobian.
// Notebook §6.2 (1e-9).
// Tolerance loosened to 1e-5 because FD slogdet at the C·H·W flat level
// accumulates truncation error across the (C·H·W)² Jacobian entries.
// -----------------------------------------------------------------------------

console.log('\n[5] conv1x1_dense_jacobian — closed-form H·W·log|det W| vs FD slogdet');

{
  const C = 4;
  const H = 2;
  const W = 2;
  const conv = new Conv1x1Dense({ C, seed: 20260511, init: 'gaussian' });
  // 1) Closed-form against slogdet.
  const logDetWDirect = conv.logDetW();
  // 2) Build same W matrix explicitly and check.
  const slogdet = slogdetAbs([conv.W[0].slice(), conv.W[1].slice(), conv.W[2].slice(), conv.W[3].slice()]);
  approxEq('conv1x1_dense_logdetW_vs_slogdet', logDetWDirect, slogdet, 1e-12, 'logDetW vs slogdet(W)');

  // 3) Forward on a random input, compare H·W·log|det W| vs FD slogdet of the full operator.
  const xs = sampleStandardNormalBatch(1, C * H * W, 42);
  const x = xs[0];
  const { logDet: closedFlat } = conv.forward(x, H, W);
  const J = finiteDifferenceJacobian((u) => conv.forward(u, H, W).y, x);
  const fdSlog = slogdetAbs(J);
  approxEq('conv1x1_dense_flat_logdet', fdSlog, closedFlat, 1e-5, 'closed H·W·log|det W| vs FD slogdet');
}

// -----------------------------------------------------------------------------
// Test 6: Conv1x1LU closed-form log|det W| = Σ s_log vs slogdet of materialized W.
// Notebook §6.2 (1e-12).
// -----------------------------------------------------------------------------

console.log('\n[6] conv1x1_lu_logdet — Σ s_log = log|det (P·L·(U+diag(exp s)))|');

{
  for (const C of [4, 6, 8]) {
    const conv = new Conv1x1LU({ C, seed: 20260511 + C });
    const closed = conv.logDetW(); // = Σ s_log
    const W = conv.buildW();
    const slogdet = slogdetAbs(W);
    approxEq(`conv1x1_lu_logdetW[C=${C}]`, closed, slogdet, 1e-10, 'Σ s_log vs slogdet(P·L·U)');
  }
}

// -----------------------------------------------------------------------------
// Test 7: MAFLayer triangular Jacobian — diagonal = exp(s_i), strict upper = 0.
// Notebook §5.2 (strict-upper max < 1e-12).
// -----------------------------------------------------------------------------

console.log('\n[7] maf_layer_triangular — Jacobian is lower-triangular with diag = exp(s_i)');

{
  const d = 4;
  const maf = new MAFLayer({ d, seed: 20260511 });
  const z = sampleStandardNormalBatch(1, d, 23)[0];
  // Round-trip first.
  const { x, logDet: ldFwd } = maf.forward(z);
  const { z: zBack, logDet: ldInv } = maf.inverse(x);
  let maxRtErr = 0;
  for (let i = 0; i < d; i++) {
    const e = Math.abs(zBack[i] - z[i]);
    if (e > maxRtErr) maxRtErr = e;
  }
  within('maf_roundtrip', maxRtErr, 0, 1e-12, 'max |inverse(forward(z)) - z|');
  within('maf_logdet_pair', Math.abs(ldFwd + ldInv), 0, 1e-12, '|ld_fwd + ld_inv|');

  // Triangular structure via FD Jacobian.
  const J = finiteDifferenceJacobian((u) => maf.forward(u).x, z);
  let strictUpperMax = 0;
  for (let i = 0; i < d; i++) {
    for (let j = i + 1; j < d; j++) {
      const e = Math.abs(J[i][j]);
      if (e > strictUpperMax) strictUpperMax = e;
    }
  }
  within('maf_strict_upper', strictUpperMax, 0, 1e-6, 'max |J[i,j]| for j > i');

  // log|det J| from FD vs closed-form Σ s_i.
  const slogdet = slogdetAbs(J);
  approxEq('maf_logdet_jacobian', slogdet, ldFwd, 1e-5, 'FD slogdet vs closed Σ s_i');
}

// -----------------------------------------------------------------------------
// Final summary.
// -----------------------------------------------------------------------------

console.log('\n' + '='.repeat(72));
console.log(`PASSED: ${pass}    FAILED: ${fail}`);
if (fail > 0) {
  console.log('\nFAILURES:');
  for (const f of failures) console.log(`  - ${f}`);
  process.exit(1);
}
console.log('All checks pass.');
process.exit(0);
