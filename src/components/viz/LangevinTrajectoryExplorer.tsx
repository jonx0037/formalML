// =============================================================================
// LangevinTrajectoryExplorer.tsx
//
// §5 Overdamped Langevin dynamics. The continuous-time SDE
//   dθ_t = -∇H(θ_t) dt + √2 dW_t
// discretized as Euler–Maruyama on the running 2D toy: a two-component
// Gaussian mixture at (±2, 0) with σ² = 0.5. The chain demonstrates that
// gradient-flow + calibrated noise samples from π ∝ exp(-H), and that a
// well-tuned η resolves both modes.
//
// Two panels:
//   (a) 2D phase portrait — GMM iso-density contours + chain trajectory.
//       At small η the chain is locally well-mixed; at large η the chain
//       jumps between modes (or escapes to large radius).
//   (b) Histogram of θ_1 marginal vs analytic π(θ_1) = 0.5·N(-2, 0.5)
//       + 0.5·N(2, 0.5). At sufficient T and well-chosen η, the chain's
//       marginal converges to the bimodal target.
//
// Controls:
//   - η slider (0.005..0.1, default 0.05). Larger η → faster mode-hopping
//     but more discretization bias.
//   - T slider (3k..20k, default 8k). Longer chains tighten the histogram.
//   - Resample button.
//
// Computation: in-browser via shared/sgmcmc.ts. One ~8k-step chain ≈ 0.5 s.
// Static fallback: /images/topics/stochastic-gradient-mcmc/05_overdamped_langevin.png.
// =============================================================================

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import { mulberry32 } from './shared/bayesian-ml';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  gmm2DGradAndLogDensity,
  histogram,
  paletteSGMCMC,
  sgldChain,
} from './shared/sgmcmc';

const DEFAULT_ETA = 0.05;
const DEFAULT_T = 8000;
const BURN_FRAC = 0.1;

const FIGURE_PATH = '/images/topics/stochastic-gradient-mcmc/05_overdamped_langevin.png';
const ALT =
  'Two-panel figure showing overdamped Langevin dynamics on a two-component Gaussian mixture: (a) chain trajectory tracing both modes; (b) θ_1 marginal histogram matching the bimodal target density.';
const CAPTION =
  'Figure 5. Overdamped Langevin dynamics on a 2D Gaussian mixture target. Gradient flow + √2 dW noise samples from π ∝ exp(-H). Slide η to see fast vs slow mode-hopping; longer T tightens the marginal histogram against the analytic π(θ_1).';
const ARIA = 'Figure 5: Interactive Langevin chain on 2D GMM';

interface CachedRun {
  chain: Float32Array[];
  burn: number;
  eta: number;
  T: number;
  chainSeed: number;
}

const yieldToBrowser = () => new Promise<void>((res) => setTimeout(res, 0));

async function runChain(eta: number, T: number, chainSeed: number): Promise<CachedRun> {
  const target = gmm2DGradAndLogDensity();
  await yieldToBrowser();
  const rng = mulberry32(chainSeed * 991 + 41);
  const start = new Float32Array([0.0, 0.0]); // between modes
  const chain = sgldChain(target.gradU, start, T, eta, rng);
  return { chain, burn: Math.floor(T * BURN_FRAC), eta, T, chainSeed };
}

type CompState =
  | { kind: 'idle' }
  | { kind: 'sampling'; reason: 'first' | 'param' | 'reseed' }
  | { kind: 'ready'; run: CachedRun };

export default function LangevinTrajectoryExplorer() {
  const { ref, width } = useResizeObserver<HTMLElement>();
  const [state, setState] = useState<CompState>({ kind: 'idle' });
  const [eta, setEta] = useState(DEFAULT_ETA);
  const [T, setT] = useState(DEFAULT_T);
  const [chainSeed, setChainSeed] = useState(11);
  const ticketRef = useRef(0);

  useEffect(() => {
    let alive = true;
    setState({ kind: 'sampling', reason: 'first' });
    const ticket = ++ticketRef.current;
    runChain(DEFAULT_ETA, DEFAULT_T, 11).then((run) => {
      if (alive && ticket === ticketRef.current) setState({ kind: 'ready', run });
    });
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const launch = useCallback((e: number, t: number, seed: number, reason: 'param' | 'reseed') => {
    const ticket = ++ticketRef.current;
    setState({ kind: 'sampling', reason });
    runChain(e, t, seed).then((run) => {
      if (ticket === ticketRef.current) setState({ kind: 'ready', run });
    });
  }, []);

  const onEtaCommit = (next: number) => {
    if (Math.abs(next - eta) < 1e-6) return;
    setEta(next);
    if (state.kind === 'ready') launch(next, T, chainSeed, 'param');
  };
  const onTCommit = (next: number) => {
    if (next === T) return;
    setT(next);
    if (state.kind === 'ready') launch(eta, next, chainSeed, 'param');
  };
  const onReseed = () => {
    const next = chainSeed + 1;
    setChainSeed(next);
    if (state.kind === 'ready') launch(eta, T, next, 'reseed');
  };

  const isBlocked = state.kind !== 'ready';
  const ready = state.kind === 'ready' ? state : null;

  return (
    <figure
      ref={ref as React.RefObject<HTMLElement>}
      aria-label={ARIA}
      style={{ width: '100%', margin: '2rem auto', display: 'flex', flexDirection: 'column', alignItems: 'center' }}
    >
      <div style={{ width: '100%', maxWidth: '880px' }}>
        {!ready ? (
          <Fallback label={state.kind === 'sampling' && state.reason === 'first' ? 'Running Langevin chain (~0.5 s)…' : 'Re-running chain…'} />
        ) : (
          <>
            <Controls eta={eta} T={T} onEtaCommit={onEtaCommit} onTCommit={onTCommit} onReseed={onReseed} isBlocked={isBlocked} />
            <Panels run={ready.run} width={width} />
          </>
        )}
      </div>
      <figcaption
        style={{ fontSize: '0.875rem', lineHeight: 1.5, color: 'var(--color-text-muted, #666)', marginTop: '0.75rem', textAlign: 'left', maxWidth: '880px', padding: '0 0.5rem' }}
      >
        {CAPTION}
      </figcaption>
    </figure>
  );
}

function Fallback({ label }: { label: string }) {
  return (
    <div style={{ position: 'relative', width: '100%' }}>
      <img src={FIGURE_PATH} alt={ALT} loading="lazy" style={{ width: '100%', height: 'auto', backgroundColor: 'var(--color-surface, transparent)', borderRadius: '0.25rem', boxShadow: '0 1px 2px var(--color-shadow, rgba(0, 0, 0, 0.04))', opacity: 0.55 }} />
      <div role="status" aria-live="polite" style={{ position: 'absolute', left: '50%', top: '50%', transform: 'translate(-50%, -50%)', padding: '0.5rem 0.875rem', background: 'var(--color-surface, #fff)', border: '1px solid var(--color-border, #ccc)', borderRadius: '4px', fontSize: '0.875rem', color: 'var(--color-text, #333)', boxShadow: '0 2px 6px rgba(0,0,0,0.08)' }}>
        {label}
      </div>
    </div>
  );
}

function Controls({
  eta,
  T,
  onEtaCommit,
  onTCommit,
  onReseed,
  isBlocked,
}: {
  eta: number;
  T: number;
  onEtaCommit: (n: number) => void;
  onTCommit: (n: number) => void;
  onReseed: () => void;
  isBlocked: boolean;
}) {
  const [le, setLe] = useState(eta);
  const [lt, setLt] = useState(T);
  useEffect(() => setLe(eta), [eta]);
  useEffect(() => setLt(T), [T]);
  const ctlStyle = { display: 'flex', flexWrap: 'wrap' as const, gap: '0.75rem 1.25rem', alignItems: 'center', fontSize: '0.875rem', color: 'var(--color-text, #333)', marginBottom: '0.75rem' };
  return (
    <div style={ctlStyle}>
      <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}>
        <span>η:</span>
        <input type="range" min={0.005} max={0.1} step={0.005} value={le}
          onChange={(e) => setLe(Number(e.target.value))}
          onMouseUp={(e) => onEtaCommit(Number((e.target as HTMLInputElement).value))}
          onTouchEnd={(e) => onEtaCommit(Number((e.target as HTMLInputElement).value))}
          onKeyUp={(e) => onEtaCommit(Number((e.target as HTMLInputElement).value))}
          style={{ width: '120px' }} disabled={isBlocked} aria-label="Step size eta" />
        <span style={{ fontVariantNumeric: 'tabular-nums', minWidth: '4em' }}>{le.toFixed(3)}</span>
      </label>
      <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}>
        <span>T:</span>
        <input type="range" min={3000} max={20000} step={1000} value={lt}
          onChange={(e) => setLt(Number(e.target.value))}
          onMouseUp={(e) => onTCommit(Number((e.target as HTMLInputElement).value))}
          onTouchEnd={(e) => onTCommit(Number((e.target as HTMLInputElement).value))}
          onKeyUp={(e) => onTCommit(Number((e.target as HTMLInputElement).value))}
          style={{ width: '120px' }} disabled={isBlocked} aria-label="Chain length T" />
        <span style={{ fontVariantNumeric: 'tabular-nums', minWidth: '4em' }}>{lt}</span>
      </label>
      <button type="button" onClick={onReseed} disabled={isBlocked}
        style={{ padding: '0.3rem 0.7rem', border: '1px solid var(--color-border, #ccc)', borderRadius: '4px', background: isBlocked ? 'var(--color-surface-muted, #f0f0f0)' : 'var(--color-surface, #fff)', color: 'var(--color-text, #333)', cursor: isBlocked ? 'not-allowed' : 'pointer', fontSize: '0.875rem' }}>
        Resample chain
      </button>
      {isBlocked && <span style={{ color: 'var(--color-text-muted, #888)', fontStyle: 'italic' }}>sampling…</span>}
    </div>
  );
}

function Panels({ run, width }: { run: CachedRun; width: number }) {
  const layoutWidth = Math.max(width, 320);
  const stack = layoutWidth < 720;
  const panelW = stack ? layoutWidth : (layoutWidth - 16) / 2;
  const panelH = 280;
  return (
    <div style={{ display: 'flex', flexDirection: stack ? 'column' : 'row', gap: stack ? '12px' : '16px', alignItems: stack ? 'center' : 'flex-start' }}>
      <Panel title="(a) Phase portrait on 2D GMM" width={panelW} height={panelH}>
        <PhasePortrait chain={run.chain} burn={run.burn} width={panelW} height={panelH} />
      </Panel>
      <Panel title="(b) θ_1 marginal vs π(θ_1)" width={panelW} height={panelH}>
        <Marginal chain={run.chain} burn={run.burn} width={panelW} height={panelH} />
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

function PhasePortrait({ chain, burn, width, height }: { chain: Float32Array[]; burn: number; width: number; height: number }) {
  const margin = { top: 8, right: 12, bottom: 28, left: 36 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const xS = d3.scaleLinear().domain([-5, 5]).range([0, innerW]);
  const yS = d3.scaleLinear().domain([-3, 3]).range([innerH, 0]);

  const target = useMemo(() => gmm2DGradAndLogDensity(), []);
  const contours = useMemo(() => {
    const grid = 70;
    const xMin = -5, xMax = 5, yMin = -3, yMax = 3;
    const dx = (xMax - xMin) / grid;
    const dy = (yMax - yMin) / grid;
    let lpMax = -Infinity;
    for (let i = 0; i <= grid; i++) for (let j = 0; j <= grid; j++) {
      const lp = target.logDensity(xMin + i * dx, yMin + j * dy);
      if (lp > lpMax) lpMax = lp;
    }
    const levels = [-0.5, -1.5, -3, -5];
    const segs: string[][] = levels.map(() => []);
    for (let i = 0; i < grid; i++) for (let j = 0; j < grid; j++) {
      const x = xMin + i * dx, y = yMin + j * dy;
      const lp00 = target.logDensity(x, y) - lpMax;
      const lp10 = target.logDensity(x + dx, y) - lpMax;
      const lp01 = target.logDensity(x, y + dy) - lpMax;
      const lp11 = target.logDensity(x + dx, y + dy) - lpMax;
      levels.forEach((thr, k) => {
        const v00 = lp00 - thr, v10 = lp10 - thr, v01 = lp01 - thr, v11 = lp11 - thr;
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
  }, [target, xS, yS]);

  const tracePath = useMemo(() => {
    const pts = chain.slice(burn);
    return d3.line<Float32Array>()
      .x((p) => xS(p[0]))
      .y((p) => yS(p[1]))
      .defined((p) => p[0] >= -5 && p[0] <= 5 && p[1] >= -3 && p[1] <= 3)(pts) ?? '';
  }, [chain, burn, xS, yS]);

  return (
    <g transform={`translate(${margin.left},${margin.top})`}>
      <rect x={0} y={0} width={innerW} height={innerH} fill="var(--color-surface, #fafafa)" />
      {contours.map((d, i) => <path key={i} d={d} fill="none" stroke={paletteSGMCMC.target} strokeWidth={0.5} opacity={0.45} />)}
      <path d={tracePath} fill="none" stroke={paletteSGMCMC.langevin} strokeWidth={0.4} opacity={0.7} />
      <line x1={0} x2={innerW} y1={innerH} y2={innerH} stroke="var(--color-text-muted, #888)" />
      <line x1={0} x2={0} y1={0} y2={innerH} stroke="var(--color-text-muted, #888)" />
      <text x={innerW / 2} y={innerH + 18} textAnchor="middle" fontSize={10} fill="var(--color-text, #333)">θ_1</text>
      <text transform={`translate(${-26}, ${innerH / 2}) rotate(-90)`} textAnchor="middle" fontSize={10} fill="var(--color-text, #333)">θ_2</text>
    </g>
  );
}

function Marginal({ chain, burn, width, height }: { chain: Float32Array[]; burn: number; width: number; height: number }) {
  const margin = { top: 8, right: 12, bottom: 28, left: 36 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const lo = -5, hi = 5;
  const nBins = 40;

  const trace = useMemo(() => {
    const out = new Float32Array(chain.length - burn);
    for (let i = 0; i < out.length; i++) out[i] = chain[i + burn][0];
    return out;
  }, [chain, burn]);
  const hist = useMemo(() => histogram(trace, lo, hi, nBins), [trace]);

  const truePdf = (x: number) => {
    const sigma = Math.sqrt(0.5);
    const phi = (mu: number) => Math.exp(-((x - mu) ** 2) / (2 * sigma * sigma)) / (Math.sqrt(2 * Math.PI) * sigma);
    return 0.5 * phi(-2) + 0.5 * phi(2);
  };

  const xS = d3.scaleLinear().domain([lo, hi]).range([0, innerW]);
  const refPts = Array.from({ length: 200 }, (_, i) => lo + (i * (hi - lo)) / 199);
  const yMax = Math.max(
    Math.max(...Array.from(hist.density)),
    Math.max(...refPts.map(truePdf)),
  ) * 1.15;
  const yS = d3.scaleLinear().domain([0, yMax]).range([innerH, 0]);
  const refPath = refPts.map((x, i) => `${i === 0 ? 'M' : 'L'}${xS(x)},${yS(truePdf(x))}`).join(' ');
  const binWidth = (hi - lo) / nBins;

  return (
    <g transform={`translate(${margin.left},${margin.top})`}>
      <rect x={0} y={0} width={innerW} height={innerH} fill="var(--color-surface, #fafafa)" />
      {Array.from(hist.centers).map((c, i) => (
        <rect
          key={i}
          x={xS(c - binWidth / 2)}
          y={yS(hist.density[i])}
          width={Math.max(0, xS(c + binWidth / 2) - xS(c - binWidth / 2) - 1)}
          height={Math.max(0, innerH - yS(hist.density[i]))}
          fill={paletteSGMCMC.langevin}
          opacity={0.55}
        />
      ))}
      <path d={refPath} fill="none" stroke={paletteSGMCMC.target} strokeWidth={1.6} strokeDasharray="4 3" />
      <line x1={0} x2={innerW} y1={innerH} y2={innerH} stroke="var(--color-text-muted, #888)" />
      <line x1={0} x2={0} y1={0} y2={innerH} stroke="var(--color-text-muted, #888)" />
      {[-4, -2, 0, 2, 4].map((t) => (
        <text key={t} x={xS(t)} y={innerH + 16} textAnchor="middle" fontSize={10} fill="var(--color-text, #333)">{t}</text>
      ))}
      <text x={innerW / 2} y={innerH + 24} textAnchor="middle" fontSize={10} fill="var(--color-text, #333)">θ_1</text>
      <text transform={`translate(${-26}, ${innerH / 2}) rotate(-90)`} textAnchor="middle" fontSize={10} fill="var(--color-text, #333)">density</text>
    </g>
  );
}
