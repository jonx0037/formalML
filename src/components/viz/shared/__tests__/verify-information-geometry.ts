// =============================================================================
// verify-information-geometry.ts
//
// Numerical checks for fisherRaoGeodesicGaussian (the Levi-Civita curve in
// DualGeometryExplorer) against src/content/topics/information-geometry.mdx:
//   (1) the curve starts at the first endpoint and ends at the second (1e-6);
//   (2) its Fisher length equals Rao's closed-form distance
//       √2·arccosh(1 + (Δμ² + 2Δσ²)/(4σ₁σ₂)), so it is the minimizing geodesic;
//   (3) on the explorer's default endpoints it is shorter than the e- and
//       m-geodesics (3.0487 < 3.1001 < 3.1550).
//
// Run with: pnpm verify:information-geometry
// =============================================================================

import {
  fisherRaoDistanceGaussian,
  fisherRaoGeodesicGaussian,
} from '../manifoldGeometry';

let fail = 0;
function check(name: string, ok: boolean, detail = ''): void {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? `  (${detail})` : ''}`);
  if (!ok) fail++;
}

type Pt = { x: number; y: number };

/** Fisher length of a polyline under ds² = (dμ² + 2 dσ²)/σ² (midpoint σ). */
function fisherLength(pts: Pt[]): number {
  let L = 0;
  for (let i = 1; i < pts.length; i++) {
    const dm = pts[i].x - pts[i - 1].x;
    const ds = pts[i].y - pts[i - 1].y;
    const sm = 0.5 * (pts[i].y + pts[i - 1].y);
    L += Math.sqrt(dm * dm + 2 * ds * ds) / sm;
  }
  return L;
}

const N = 20000;
const cases: [number, number, number, number][] = [
  [-1.5, 0.5, 1.5, 2.0], // explorer + notebook default
  [1.5, 2.0, -1.5, 0.5], // reversed
  [-2.8, 3.3, 2.8, 0.15], // extreme drag corners
  [0.3, 0.4, 0.3, 2.5], // vertical (Δμ = 0)
  [0.3, 0.4, 0.3 + 5e-7, 2.5], // near-vertical, vertical branch
  [0.3, 0.4, 0.3 + 5e-5, 2.5], // near-vertical, arc branch
  [-1, 1, 1, 1], // equal σ
];

for (const [m1, s1, m2, s2] of cases) {
  const tag = `(${m1}, ${s1}) → (${m2}, ${s2})`;
  const pts = fisherRaoGeodesicGaussian(m1, s1, m2, s2, N);
  const a = pts[0];
  const b = pts[pts.length - 1];
  const endErr = Math.max(
    Math.abs(a.x - m1), Math.abs(a.y - s1), Math.abs(b.x - m2), Math.abs(b.y - s2)
  );
  check(`endpoints ${tag}`, endErr < 1e-6, `max err ${endErr.toExponential(2)}`);
  const L = fisherLength(pts);
  const d = fisherRaoDistanceGaussian(m1, s1, m2, s2);
  const rel = Math.abs(L - d) / d;
  check(`length = d_FR ${tag}`, rel < 1e-6, `L=${L.toFixed(6)} d=${d.toFixed(6)}`);
  check(`σ > 0 along ${tag}`, pts.every((p) => p.y > 0));
}

// Default endpoints: Levi-Civita is shorter than e- and m-geodesics
{
  const [m1, s1, m2, s2] = cases[0];
  const lc = fisherLength(fisherRaoGeodesicGaussian(m1, s1, m2, s2, N));
  const e: Pt[] = [];
  const m: Pt[] = [];
  const n1 = [m1 / s1 ** 2, -1 / (2 * s1 ** 2)];
  const n2 = [m2 / s2 ** 2, -1 / (2 * s2 ** 2)];
  const x1 = [m1, m1 * m1 + s1 * s1];
  const x2 = [m2, m2 * m2 + s2 * s2];
  for (let i = 0; i <= N; i++) {
    const t = i / N;
    const h1 = n1[0] + t * (n2[0] - n1[0]);
    const h2 = n1[1] + t * (n2[1] - n1[1]);
    const v = -1 / (2 * h2);
    e.push({ x: h1 * v, y: Math.sqrt(v) });
    const u1 = x1[0] + t * (x2[0] - x1[0]);
    const u2 = x1[1] + t * (x2[1] - x1[1]);
    m.push({ x: u1, y: Math.sqrt(u2 - u1 * u1) });
  }
  const le = fisherLength(e);
  const lm = fisherLength(m);
  check('default: d_FR ≈ 3.04866', Math.abs(lc - 3.04866) < 1e-5, lc.toFixed(5));
  check('default: LC < m < e', lc < lm && lm < le,
    `LC=${lc.toFixed(4)} m=${lm.toFixed(4)} e=${le.toFixed(4)}`);
}

console.log(fail === 0 ? '\nAll checks passed.' : `\n${fail} check(s) failed.`);
if (fail > 0) process.exit(1);
