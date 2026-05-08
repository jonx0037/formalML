// =============================================================================
// CompleteRecipeMorpher.tsx
//
// §2 The Ma–Chen–Fox (2015) complete-recipe lens. The single template
//   dθ_t = -[D(θ) + Q(θ)] ∇H(θ) dt + Γ(θ) dt + √(2 D(θ)) dW_t
// instantiates SGLD, SGHMC, and RSGLD by different choices of (D, Q):
//   - SGLD:  D = I,           Q = 0    → overdamped Langevin
//   - SGHMC: D = diag(0, C),  Q = [[0, -I], [I, 0]]  → underdamped Langevin
//   - RSGLD: D = G⁻¹(θ),      Q = 0    → Riemann-manifold Langevin
//
// Two panels:
//   (a) The (D, Q) matrices for the selected method, rendered as a small
//       coefficient table with skew-symmetric / positive-definite annotations.
//   (b) A 2D phase portrait on a curved target (an anisotropic Gaussian)
//       running the corresponding chain to show the dynamics induced by
//       each (D, Q) choice. SGLD: random-walk; SGHMC: ballistic with
//       momentum; RSGLD: preconditioned Langevin.
//
// Controls:
//   - Method selector (SGLD / SGHMC / RSGLD)
//   - Resample button
//
// Computation: in-browser, ~3k-step chain ≈ instantaneous.
// Static fallback: /images/topics/stochastic-gradient-mcmc/02_complete_recipe.png.
// =============================================================================

import { useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import { mulberry32 } from './shared/bayesian-ml';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  anisotropic2DGradAndLogDensity,
  paletteSGMCMC,
  rsgldDiagonalChain,
  sghmcChain,
  sgldChain,
} from './shared/sgmcmc';

const KAPPA = 25;
const N_STEPS = 3000;
const ETA_BY_METHOD = { SGLD: 0.05, SGHMC: 0.1, RSGLD: 0.05 } as const;
const C_SGHMC = 0.5;

type Method = 'SGLD' | 'SGHMC' | 'RSGLD';

const FIGURE_PATH = '/images/topics/stochastic-gradient-mcmc/02_complete_recipe.png';
const ALT =
  'Two-panel figure showing the Ma–Chen–Fox complete-recipe template instantiated as SGLD, SGHMC, and RSGLD by different (D, Q) choices. Panel (a): the matrices D and Q for the selected method. Panel (b): the corresponding chain trajectory on a curved 2D target.';
const CAPTION =
  'Figure 2. The Ma–Chen–Fox (2015) complete-recipe template. Different (D, Q) choices yield SGLD (random-walk Langevin), SGHMC (ballistic with momentum), or RSGLD (preconditioned Langevin). The matrices in panel (a) drive the dynamics in panel (b); skew-symmetric Q produces rotation, positive-definite D produces diffusion.';
const ARIA = 'Figure 2: Interactive complete-recipe (D, Q) morpher';

interface CachedRun {
  trajectories: { SGLD: Float32Array[]; SGHMC: Float32Array[]; RSGLD: Float32Array[] };
  chainSeed: number;
}

function runAllMethods(chainSeed: number): CachedRun {
  const target = anisotropic2DGradAndLogDensity(KAPPA);
  const start = new Float32Array([0, 4]);
  const sgld = sgldChain(target.gradU, start, N_STEPS, ETA_BY_METHOD.SGLD, mulberry32(chainSeed * 991 + 11));
  const sghmc = sghmcChain(target.gradU, start, N_STEPS, ETA_BY_METHOD.SGHMC, C_SGHMC, mulberry32(chainSeed * 991 + 12));
  // RSGLD with diagonal G⁻¹ matching the target Hessian inverse: G⁻¹ = diag(1, κ)
  const gInvDiag = (_t: Float32Array) => new Float32Array([1, KAPPA]);
  const divGInv = (_t: Float32Array) => new Float32Array([0, 0]);
  const rsgld = rsgldDiagonalChain(
    target.gradU,
    gInvDiag,
    divGInv,
    start,
    N_STEPS,
    ETA_BY_METHOD.RSGLD,
    mulberry32(chainSeed * 991 + 13),
  );
  return { trajectories: { SGLD: sgld, SGHMC: sghmc, RSGLD: rsgld }, chainSeed };
}

export default function CompleteRecipeMorpher() {
  const { ref, width } = useResizeObserver<HTMLElement>();
  const [method, setMethod] = useState<Method>('SGLD');
  const [chainSeed, setChainSeed] = useState(11);
  const [run, setRun] = useState<CachedRun | null>(null);
  const idRef = useRef(0);

  useEffect(() => {
    const id = ++idRef.current;
    setRun(null);
    // Defer to next tick so the loading state can render.
    setTimeout(() => {
      if (id !== idRef.current) return;
      setRun(runAllMethods(chainSeed));
    }, 0);
  }, [chainSeed]);

  return (
    <figure
      ref={ref as React.RefObject<HTMLElement>}
      aria-label={ARIA}
      style={{ width: '100%', margin: '2rem auto', display: 'flex', flexDirection: 'column', alignItems: 'center' }}
    >
      <div style={{ width: '100%', maxWidth: '880px' }}>
        {!run ? (
          <Fallback label={'Computing chain trajectories…'} />
        ) : (
          <>
            <Controls method={method} onMethodChange={setMethod} onReseed={() => setChainSeed((s) => s + 1)} />
            <Panels run={run} method={method} width={width} />
          </>
        )}
      </div>
      <figcaption style={{ fontSize: '0.875rem', lineHeight: 1.5, color: 'var(--color-text-muted, #666)', marginTop: '0.75rem', textAlign: 'left', maxWidth: '880px', padding: '0 0.5rem' }}>{CAPTION}</figcaption>
    </figure>
  );
}

function Fallback({ label }: { label: string }) {
  return (
    <div style={{ position: 'relative', width: '100%' }}>
      <img src={FIGURE_PATH} alt={ALT} loading="lazy" style={{ width: '100%', height: 'auto', backgroundColor: 'var(--color-surface, transparent)', borderRadius: '0.25rem', boxShadow: '0 1px 2px var(--color-shadow, rgba(0, 0, 0, 0.04))', opacity: 0.55 }} />
      <div role="status" aria-live="polite" style={{ position: 'absolute', left: '50%', top: '50%', transform: 'translate(-50%, -50%)', padding: '0.5rem 0.875rem', background: 'var(--color-surface, #fff)', border: '1px solid var(--color-border, #ccc)', borderRadius: '4px', fontSize: '0.875rem', color: 'var(--color-text, #333)', boxShadow: '0 2px 6px rgba(0,0,0,0.08)' }}>{label}</div>
    </div>
  );
}

function Controls({ method, onMethodChange, onReseed }: { method: Method; onMethodChange: (m: Method) => void; onReseed: () => void }) {
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.75rem 1.25rem', alignItems: 'center', fontSize: '0.875rem', color: 'var(--color-text, #333)', marginBottom: '0.75rem' }}>
      <span>Method:</span>
      {(['SGLD', 'SGHMC', 'RSGLD'] as Method[]).map((m) => (
        <button
          key={m}
          type="button"
          onClick={() => onMethodChange(m)}
          aria-pressed={method === m}
          style={{
            padding: '0.3rem 0.7rem',
            border: '1px solid var(--color-border, #ccc)',
            borderRadius: '4px',
            background: method === m ? colorOfMethod(m) : 'var(--color-surface, #fff)',
            color: method === m ? '#fff' : 'var(--color-text, #333)',
            cursor: 'pointer',
            fontSize: '0.875rem',
            fontWeight: method === m ? 600 : 400,
          }}
        >
          {m}
        </button>
      ))}
      <button type="button" onClick={onReseed} style={{ padding: '0.3rem 0.7rem', border: '1px solid var(--color-border, #ccc)', borderRadius: '4px', background: 'var(--color-surface, #fff)', color: 'var(--color-text, #333)', cursor: 'pointer', fontSize: '0.875rem' }}>
        Resample
      </button>
    </div>
  );
}

function colorOfMethod(m: Method): string {
  return m === 'SGLD' ? paletteSGMCMC.sgld : m === 'SGHMC' ? paletteSGMCMC.sghmc : paletteSGMCMC.rsgld;
}

function Panels({ run, method, width }: { run: CachedRun; method: Method; width: number }) {
  const layoutWidth = Math.max(width, 320);
  const stack = layoutWidth < 720;
  const panelW = stack ? layoutWidth : (layoutWidth - 16) / 2;
  const panelH = 280;
  return (
    <div style={{ display: 'flex', flexDirection: stack ? 'column' : 'row', gap: stack ? '12px' : '16px', alignItems: stack ? 'center' : 'flex-start' }}>
      <Panel title="(a) (D, Q) matrices" width={panelW} height={panelH}>
        <DQMatrices method={method} width={panelW} height={panelH} />
      </Panel>
      <Panel title={`(b) Phase portrait on κ = ${KAPPA} target`} width={panelW} height={panelH}>
        <PhasePortrait chain={run.trajectories[method]} method={method} width={panelW} height={panelH} />
      </Panel>
    </div>
  );
}

function Panel({ title, width, height, children }: { title: string; width: number; height: number; children: React.ReactNode }) {
  return (
    <div style={{ width, display: 'flex', flexDirection: 'column' }}>
      <div style={{ fontSize: '0.825rem', fontWeight: 600, color: 'var(--color-text, #333)', marginBottom: '0.25rem' }}>{title}</div>
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img">{children}</svg>
    </div>
  );
}

function DQMatrices({ method, width, height }: { method: Method; width: number; height: number }) {
  const dim = method === 'SGHMC' ? 4 : 2;
  const [D, Q, label] = useMemo(() => {
    if (method === 'SGLD') {
      return [
        [
          [1, 0],
          [0, 1],
        ],
        [
          [0, 0],
          [0, 0],
        ],
        'Overdamped Langevin: D = I, Q = 0',
      ];
    }
    if (method === 'SGHMC') {
      const C = C_SGHMC;
      // 4x4 D = diag(0, 0, C, C), Q = [[0 0 -1 0],[0 0 0 -1],[1 0 0 0],[0 1 0 0]]
      return [
        [
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, C, 0],
          [0, 0, 0, C],
        ],
        [
          [0, 0, -1, 0],
          [0, 0, 0, -1],
          [1, 0, 0, 0],
          [0, 1, 0, 0],
        ],
        `Underdamped Langevin: D = diag(0, C·I_d), Q = [[0, -I],[I, 0]], C = ${C}`,
      ];
    }
    // RSGLD with G⁻¹ = diag(1, κ)
    return [
      [
        [1, 0],
        [0, KAPPA],
      ],
      [
        [0, 0],
        [0, 0],
      ],
      `Riemann-manifold Langevin: D = G⁻¹(θ) = diag(1, κ), Q = 0, κ = ${KAPPA}`,
    ];
  }, [method]);

  const margin = { top: 26, right: 12, bottom: 28, left: 12 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const matW = Math.min(innerW * 0.42, 160);
  const matH = Math.min(innerH * 0.7, 160);
  const cellSize = Math.min(matW / dim, matH / dim);
  const totalMatW = cellSize * dim;
  const dx = innerW / 2;
  const dxOffsetD = dx - matW;
  const dxOffsetQ = dx + (matW - totalMatW) / 2 + 8;

  return (
    <g transform={`translate(${margin.left},${margin.top})`}>
      <text x={innerW / 2} y={-12} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--color-text, #333)">
        {label}
      </text>
      <text x={dxOffsetD + totalMatW / 2} y={-2} textAnchor="middle" fontSize={11} fontWeight={600} fill={paletteSGMCMC.langevin}>D (positive semi-def.)</text>
      <text x={dxOffsetQ + totalMatW / 2} y={-2} textAnchor="middle" fontSize={11} fontWeight={600} fill={paletteSGMCMC.sghmc}>Q (skew-symmetric)</text>
      {/* D matrix */}
      {(D as number[][]).map((row, i) =>
        row.map((v, j) => (
          <g key={`D-${i}-${j}`} transform={`translate(${dxOffsetD + j * cellSize}, ${10 + i * cellSize})`}>
            <rect width={cellSize} height={cellSize} fill={v === 0 ? 'transparent' : 'rgba(31,119,180,0.15)'} stroke="var(--color-border, #ccc)" strokeWidth={0.5} />
            <text x={cellSize / 2} y={cellSize / 2 + 4} textAnchor="middle" fontSize={11} fill="var(--color-text, #333)">
              {fmtCell(v)}
            </text>
          </g>
        )),
      )}
      {/* Q matrix */}
      {(Q as number[][]).map((row, i) =>
        row.map((v, j) => (
          <g key={`Q-${i}-${j}`} transform={`translate(${dxOffsetQ + j * cellSize}, ${10 + i * cellSize})`}>
            <rect
              width={cellSize}
              height={cellSize}
              fill={v === 0 ? 'transparent' : v > 0 ? 'rgba(23,190,207,0.18)' : 'rgba(214,39,40,0.18)'}
              stroke="var(--color-border, #ccc)"
              strokeWidth={0.5}
            />
            <text x={cellSize / 2} y={cellSize / 2 + 4} textAnchor="middle" fontSize={11} fill="var(--color-text, #333)">
              {fmtCell(v)}
            </text>
          </g>
        )),
      )}
      <text x={innerW / 2} y={innerH + 16} textAnchor="middle" fontSize={10} fill="var(--color-text-muted, #666)">
        f(θ) = -(D + Q) ∇H(θ) + Γ(θ); Γ = ∂_j (D + Q)_{'{ij}'}
      </text>
    </g>
  );
}

function fmtCell(v: number): string {
  if (v === 0) return '0';
  if (Number.isInteger(v)) return String(v);
  return v.toFixed(2);
}

function PhasePortrait({ chain, method, width, height }: { chain: Float32Array[]; method: Method; width: number; height: number }) {
  const margin = { top: 8, right: 12, bottom: 28, left: 36 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const xS = d3.scaleLinear().domain([-3, 3]).range([0, innerW]);
  const yS = d3.scaleLinear().domain([-12, 12]).range([innerH, 0]);

  // Anisotropic Gaussian iso contours
  const contours = useMemo(() => {
    const grid = 50;
    const xMin = -3, xMax = 3, yMin = -12, yMax = 12;
    const dx = (xMax - xMin) / grid;
    const dy = (yMax - yMin) / grid;
    const mahal = (a: number, b: number) => a * a + (b * b) / KAPPA;
    const targets = [2.3, 6.18, 11.83];
    const segs: string[][] = targets.map(() => []);
    for (let i = 0; i < grid; i++)
      for (let j = 0; j < grid; j++) {
        const x = xMin + i * dx;
        const y = yMin + j * dy;
        targets.forEach((t, k) => {
          const v00 = mahal(x, y) - t;
          const v10 = mahal(x + dx, y) - t;
          const v01 = mahal(x, y + dy) - t;
          const v11 = mahal(x + dx, y + dy) - t;
          const c = (v00 > 0 ? 1 : 0) + (v10 > 0 ? 2 : 0) + (v11 > 0 ? 4 : 0) + (v01 > 0 ? 8 : 0);
          if (c === 0 || c === 15) return;
          const lerp = (a: number, b: number) => a / (a - b);
          const eB = { x: x + lerp(v00, v10) * dx, y };
          const eR = { x: x + dx, y: y + lerp(v10, v11) * dy };
          const eT = { x: x + lerp(v01, v11) * dx, y: y + dy };
          const eL = { x, y: y + lerp(v00, v01) * dy };
          const seg = (a: { x: number; y: number }, b: { x: number; y: number }) =>
            segs[k].push(`M${xS(a.x)},${yS(a.y)}L${xS(b.x)},${yS(b.y)}`);
          switch (c) {
            case 1: case 14: seg(eB, eL); break;
            case 2: case 13: seg(eB, eR); break;
            case 4: case 11: seg(eR, eT); break;
            case 8: case 7: seg(eL, eT); break;
            case 3: case 12: seg(eL, eR); break;
            case 6: case 9: seg(eB, eT); break;
            case 5: seg(eL, eT); seg(eB, eR); break;
            case 10: seg(eL, eB); seg(eT, eR); break;
          }
        });
      }
    return segs.map((s) => s.join(' '));
  }, [xS, yS]);

  const tracePath = useMemo(
    () =>
      d3.line<Float32Array>()
        .x((p) => xS(p[0]))
        .y((p) => yS(p[1]))
        .defined((p) => p[0] >= -3 && p[0] <= 3 && p[1] >= -12 && p[1] <= 12)(chain) ?? '',
    [chain, xS, yS],
  );

  return (
    <g transform={`translate(${margin.left},${margin.top})`}>
      <rect x={0} y={0} width={innerW} height={innerH} fill="var(--color-surface, #fafafa)" />
      {contours.map((d, i) => <path key={i} d={d} fill="none" stroke={paletteSGMCMC.target} strokeWidth={0.5} opacity={0.45} />)}
      <path d={tracePath} fill="none" stroke={colorOfMethod(method)} strokeWidth={0.6} opacity={0.85} />
      <line x1={0} x2={innerW} y1={innerH} y2={innerH} stroke="var(--color-text-muted, #888)" />
      <line x1={0} x2={0} y1={0} y2={innerH} stroke="var(--color-text-muted, #888)" />
      <text x={innerW / 2} y={innerH + 18} textAnchor="middle" fontSize={10} fill="var(--color-text, #333)">θ_1</text>
      <text transform={`translate(${-26}, ${innerH / 2}) rotate(-90)`} textAnchor="middle" fontSize={10} fill="var(--color-text, #333)">θ_2</text>
    </g>
  );
}
