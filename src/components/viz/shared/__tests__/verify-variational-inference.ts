// =============================================================================
// verify-variational-inference.ts
//
// Numerical checks for shared/reverseKLFit.ts, which drives the reverse-KL
// curve in ForwardVsReverseKL (variational-inference §1.3) and
// ForwardReverseKLExplorer (kl-divergence).
//   (1) reference values for an equal mixture of N(±2, 1): covering optimum
//       N(0, 2.05²) with KL(q ∥ p) = 0.227; the mode-locked N(2, 1) scores 0.633;
//   (2) the minimizer is global: no point of a fine brute-force (μ, σ) grid
//       beats it, across separations, weights and the explorer presets;
//   (3) the covering → mode-locked switch for w₁ = 0.5 lies in (5.5, 5.75),
//       and the component defaults (Δ = 6) open on a mode-locked fit;
//   (4) the quadrature agrees with a plain trapezoid on a wide θ grid.
//
// Run with: pnpm verify:variational-inference
// =============================================================================

import {
  fitReverseKL,
  forwardKLFit,
  lockedComponent,
  mixtureLogPdf,
  reverseKL,
  type MixtureComponent,
} from '../reverseKLFit';

let pass = 0;
let fail = 0;
function check(name: string, ok: boolean, detail = ''): void {
  if (ok) pass++;
  else {
    fail++;
    console.error(`FAIL  ${name}${detail ? ` — ${detail}` : ''}`);
  }
}
const close = (a: number, b: number, tol: number) => Math.abs(a - b) <= tol;

const bimodal = (sep: number, w1: number, s1 = 1, s2 = 1): MixtureComponent[] => [
  { w: w1, mu: -sep / 2, sigma: s1 },
  { w: 1 - w1, mu: sep / 2, sigma: s2 },
];

// (1) Reference values at Δ = 4, w₁ = 0.5.
{
  const comps = bimodal(4, 0.5);
  const { best } = fitReverseKL(comps);
  check('Δ=4: q_R centered at 0', close(best.mu, 0, 1e-4), best.mu.toFixed(5));
  check('Δ=4: q_R σ = 2.047', close(best.sigma, 2.047, 1e-3), best.sigma.toFixed(4));
  check('Δ=4: KL(q_R ∥ p) = 0.2265', close(best.kl, 0.2265, 5e-4), best.kl.toFixed(4));
  const lock = reverseKL(2, 1, comps);
  check('Δ=4: mode-locked N(2,1) KL = 0.633', close(lock, 0.6327, 5e-4), lock.toFixed(4));
  console.log(`Δ=4, w₁=0.5: before (hardcoded lock-on) N(2.00, 1.00²) KL ${lock.toFixed(3)} → after N(${best.mu.toFixed(2)}, ${best.sigma.toFixed(2)}²) KL ${best.kl.toFixed(3)}`);
}

// (4) Quadrature vs a plain trapezoid on a wide θ grid.
for (const [mu, sigma, sep] of [[0, 2, 4], [3, 1, 6], [-1, 0.5, 3], [0.5, 4, 8]]) {
  const comps = bimodal(sep, 0.4);
  const a = mu - 14 * sigma;
  const b = mu + 14 * sigma;
  const n = 40000;
  const h = (b - a) / n;
  let s = 0;
  for (let i = 0; i <= n; i++) {
    const x = a + i * h;
    const z = (x - mu) / sigma;
    const logq = -0.5 * z * z - Math.log(sigma) - 0.5 * Math.log(2 * Math.PI);
    const term = Math.exp(logq) * (logq - mixtureLogPdf(x, comps));
    s += (i === 0 || i === n ? 0.5 : 1) * term;
  }
  const ref = s * h;
  const got = reverseKL(mu, sigma, comps);
  // 1e-5 absolute: the panels print KL to 3 decimals. The worst case is a
  // wide q (σ = 4) over unit-width modes, where the z-step is 0.3 in θ.
  check(`quadrature N(${mu},${sigma}²) vs Δ=${sep}`, close(got, ref, 1e-5), `${got} vs ${ref}`);
}

// (2) Global optimality against a brute-force grid.
function bruteMin(comps: MixtureComponent[]): { mu: number; sigma: number; kl: number } {
  let lo = Infinity;
  let hi = -Infinity;
  for (const c of comps) {
    lo = Math.min(lo, c.mu - 3);
    hi = Math.max(hi, c.mu + 3);
  }
  let best = { mu: 0, sigma: 1, kl: Infinity };
  for (let mu = lo; mu <= hi + 1e-9; mu += 0.05) {
    for (let k = 0; k <= 120; k++) {
      const sigma = 0.15 * Math.pow(6 / 0.15, k / 120);
      const kl = reverseKL(mu, sigma, comps);
      if (kl < best.kl) best = { mu, sigma, kl };
    }
  }
  return best;
}
const CASES: [string, MixtureComponent[]][] = [
  ['Δ=1 w=.5', bimodal(1, 0.5)],
  ['Δ=3 w=.5', bimodal(3, 0.5)],
  ['Δ=4 w=.5', bimodal(4, 0.5)],
  ['Δ=5.5 w=.5', bimodal(5.5, 0.5)],
  ['Δ=6 w=.5', bimodal(6, 0.5)],
  ['Δ=8 w=.5', bimodal(8, 0.5)],
  ['Δ=4 w=.3', bimodal(4, 0.3)],
  ['Δ=5 w=.8', bimodal(5, 0.8)],
  ['explorer asym s=4', [{ w: 0.7, mu: -2, sigma: 0.8 }, { w: 0.3, mu: 2, sigma: 1.2 }]],
  ['explorer tri s=4', [{ w: 0.35, mu: -4, sigma: 0.8 }, { w: 0.3, mu: 0, sigma: 0.8 }, { w: 0.35, mu: 4, sigma: 0.8 }]],
  ['explorer tri s=6', [{ w: 0.35, mu: -6, sigma: 0.8 }, { w: 0.3, mu: 0, sigma: 0.8 }, { w: 0.35, mu: 6, sigma: 0.8 }]],
];
for (const [label, comps] of CASES) {
  const { best } = fitReverseKL(comps);
  const brute = bruteMin(comps);
  check(`${label}: minimizer ≤ brute-force grid`, best.kl <= brute.kl + 1e-9, `fit ${best.kl.toFixed(6)} vs grid ${brute.kl.toFixed(6)}`);
}

// (3) Switch location and defaults.
{
  const lockedAt = (sep: number) => {
    const c = bimodal(sep, 0.5);
    return lockedComponent(fitReverseKL(c).best, c) >= 0;
  };
  check('w=.5: Δ=5.5 covers', !lockedAt(5.5));
  check('w=.5: Δ=5.75 locks', lockedAt(5.75));
  let a = 5.5;
  let b = 5.75;
  for (let i = 0; i < 25; i++) {
    const m = (a + b) / 2;
    if (lockedAt(m)) b = m;
    else a = m;
  }
  console.log(`switch (w₁ = 0.5, unit modes): Δ ≈ ${b.toFixed(3)}`);

  const c6 = bimodal(6, 0.5);
  const r6 = fitReverseKL(c6);
  check('default Δ=6: q_R locked onto a mode', lockedComponent(r6.best, c6) >= 0);
  check('default Δ=6: right mode by convention', r6.best.mu > 0, r6.best.mu.toFixed(3));
  const cover6 = r6.minima.find((m) => lockedComponent(m, c6) < 0);
  check('default Δ=6: covering fit is a worse local min', !!cover6 && cover6.kl > r6.best.kl);
  const f6 = forwardKLFit(c6);
  check('default Δ=6: q_F = N(0, √10²)', close(f6.mu, 0, 1e-12) && close(f6.sigma, Math.sqrt(10), 1e-12));
  console.log(
    `default Δ=6, w₁=0.5: q_R = N(${r6.best.mu.toFixed(2)}, ${r6.best.sigma.toFixed(2)}²) KL ${r6.best.kl.toFixed(3)}; ` +
      `covering local min N(0, ${cover6?.sigma.toFixed(2)}²) KL ${cover6?.kl.toFixed(3)}; q_F = N(0, ${f6.sigma.toFixed(2)}²)`,
  );
}

console.log(`\n${pass} passed, ${fail} failed`);
if (fail > 0) process.exit(1);
