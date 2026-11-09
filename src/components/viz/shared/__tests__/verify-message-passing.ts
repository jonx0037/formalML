// =============================================================================
// verify-message-passing.ts
//
// Numerical checks for the OverSmoothingAnalyzer panels against Theorem 6 of
// src/content/topics/message-passing.mdx:
//   (1) Ẽ(H^(ℓ)) ≤ |μ₂|^{2ℓ} Ẽ(H^(0)) at every layer, on every preset and on
//       every FoSR-rewired variant the component can show;
//   (2) Ẽ(H^(ℓ)) → 0 on every connected preset (30 layers);
//   (3) on a regular graph Ẽ/Ẽ₀ = E/E₀, so the Definition-7 energy agrees.
// It also prints the before/after table (unnormalized E/E₀ vs Ẽ/Ẽ₀ at ℓ = 30).
//
// Run with: pnpm verify:message-passing
// =============================================================================

import {
  barbellGraph,
  completeGraph,
  createRng,
  cycleGraph,
  fosrRewire,
  gcnSmoothingTrace,
  gridGraph,
  karateClubGraph,
  pathGraph,
  petersenGraph,
  propagateFeatures,
  type Graph,
} from '../graphTheory';

let pass = 0;
let fail = 0;
function check(name: string, ok: boolean, detail = ''): void {
  if (ok) pass++;
  else {
    fail++;
    console.error(`FAIL  ${name}${detail ? ` — ${detail}` : ''}`);
  }
}

// Mirrors OverSmoothingAnalyzer.tsx (seed 42, d = 4, 30 layers).
const MAX_LAYERS = 30;
const FEATURE_DIM = 4;
function generateH0(n: number): number[][] {
  const rng = createRng(42);
  return Array.from({ length: n }, () =>
    Array.from({ length: FEATURE_DIM }, () => rng() * 2 - 1),
  );
}

const FAMILIES: [string, () => Graph, boolean][] = [
  ['Path(10)', () => pathGraph(10), false],
  ['Cycle(10)', () => cycleGraph(10), true],
  ['Grid 3x3', () => gridGraph(3), false],
  ['Barbell(5)', () => barbellGraph(5), false],
  ['Petersen', petersenGraph, true],
  ['Complete(8)', () => completeGraph(8), true],
  ['Karate Club', karateClubGraph, false],
];

function checkBound(label: string, g: Graph) {
  const prop = propagateFeatures(g, generateH0(g.n), { architecture: 'gcn', layers: MAX_LAYERS });
  const tr = gcnSmoothingTrace(g, prop.features);
  const E0 = tr.energy[0];
  let worst = -Infinity;
  for (let ell = 0; ell <= MAX_LAYERS; ell++) {
    const bound = tr.mu2 ** (2 * ell) * E0;
    worst = Math.max(worst, tr.energy[ell] - bound);
  }
  check(`${label}: Ẽ(H^ℓ) ≤ |μ₂|^2ℓ Ẽ₀ for ℓ ≤ ${MAX_LAYERS}`, worst <= 1e-9 * E0, `max excess ${worst.toExponential(2)}`);
  check(`${label}: Ẽ ≥ 0`, tr.energy.every((e) => e >= -1e-12));
  return { ...tr, E: prop.dirichletEnergy, prop };
}

console.log('graph         |μ₂|     γ_old(λ₂ L_sym)  γ_new   E/E₀@30     Ẽ/Ẽ₀@30     |μ₂|^60     depth old→new');
for (const [label, build, regular] of FAMILIES) {
  const g = build();
  const r = checkBound(label, g);
  const { prop } = r;
  const eRatio = r.E[MAX_LAYERS] / r.E[0];
  const tRatio = r.energy[MAX_LAYERS] / r.energy[0];
  check(`${label}: Ẽ/Ẽ₀ < 1e-2 by ℓ = 30`, tRatio < 1e-2, tRatio.toExponential(2));
  if (regular) {
    let maxRel = 0;
    for (let ell = 0; ell <= MAX_LAYERS; ell++) {
      const a = r.E[ell] / r.E[0];
      const b = r.energy[ell] / r.energy[0];
      if (b < 1e-10) break; // below this both sit at the float floor
      maxRel = Math.max(maxRel, Math.abs(a - b) / Math.max(a, 1e-300));
    }
    check(`${label} (regular): E/E₀ = Ẽ/Ẽ₀`, maxRel < 1e-6, maxRel.toExponential(2));
  }
  const at = (arr: number[], l: number) => (arr[l] / arr[0]).toExponential(1);
  console.log(`  ${label}: E/E₀ @ℓ=10,20,30 = ${at(r.E, 10)}, ${at(r.E, 20)}, ${at(r.E, 30)}; Ẽ/Ẽ₀ = ${at(r.energy, 10)}, ${at(r.energy, 20)}, ${at(r.energy, 30)}`);
  console.log(
    `${label.padEnd(13)} ${r.mu2.toFixed(4)}   ${prop.spectralGap.toFixed(4).padEnd(16)} ${r.spectralGap.toFixed(4)}  ` +
      `${eRatio.toExponential(2).padEnd(11)} ${tRatio.toExponential(2).padEnd(11)} ${(r.mu2 ** 60).toExponential(2).padEnd(11)} ` +
      `${prop.overSmoothingDepth}→${r.overSmoothingDepth}`,
  );
  for (let k = 2; k <= 10; k++) {
    checkBound(`${label} +${k} FoSR`, fosrRewire(g, k).graph);
  }
}

console.log(`\n${pass} passed, ${fail} failed`);
if (fail > 0) process.exit(1);
