// =============================================================================
// NealsFunnelSGMCMC.tsx
//
// §10 Riemann-manifold preconditioning, on Neal's funnel: v ~ N(0, σ_v²),
// x | v ~ N(0, e^v). The funnel's curvature in x is e^{-v}, so a single
// fixed Euclidean step size cannot serve both v = 0 (broad) and v = -5
// (narrow at e^{-(-5)/2} ≈ σ_x = 0.08).
//
// The §10 fix is to give parameter space a state-dependent metric
// G⁻¹(v, x) = diag(1, e^v); the divergence ∇·G⁻¹ is identically zero
// (verified analytically in §10.4), so RSGLD on the funnel is just
//   v_{n+1} = v_n - η · ∂_v H + √(2η) · ξ_v
//   x_{n+1} = x_n - η · e^v · ∂_x H + √(2η · e^v) · ξ_x
// which absorbs the local curvature into the metric.
//
// Three panels (matching the static figure):
//   (a) Vanilla SGLD trajectory: stalls in the wide neck, never reaches v < -2.
//   (b) RSGLD trajectory: covers the full funnel uniformly.
//   (c) Marginals of v vs the true π(v) = N(0, σ_v²).
//
// Interactive controls:
//   - η slider (0.005..0.05, default 0.02). Smaller η helps vanilla SGLD
//     reach narrower regions of the funnel but at the cost of mixing speed.
//   - Chain length T (5k..30k, default 15k).
//   - Resample button.
//
// Computation: in-browser via shared/sgmcmc.ts. Two ~15k-step chains ≈ 1 s.
// Static fallback: /images/topics/stochastic-gradient-mcmc/10_rsgld_funnel.png.
// =============================================================================

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import { mulberry32 } from './shared/bayesian-ml';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  funnelDiagMetric,
  funnelGradAndLogDensity,
  histogram,
  paletteSGMCMC,
  rsgldDiagonalChain,
  sgldChain,
} from './shared/sgmcmc';

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

const SIGMA_V = 3.0;
const DEFAULT_ETA = 0.02;
const DEFAULT_T = 15000;
const BURN_FRAC = 0.2;
const START: [number, number] = [2.0, 1.0];

const FIGURE_PATH = '/images/topics/stochastic-gradient-mcmc/10_rsgld_funnel.png';
const ALT =
  'Three-panel figure of Neal\'s funnel with σ_v = 3. (a) Vanilla SGLD trajectory stalls in the wide neck and never reaches v < -2 — the small e^v scaling crushes the chain when v drops. (b) RSGLD with G⁻¹ = diag(1, e^v) covers the full funnel uniformly. (c) Marginal of v: RSGLD recovers π(v) = N(0, 9); vanilla SGLD\'s histogram is heavily biased toward v > 0.';
const CAPTION =
  'Figure 10. Riemann-manifold SGLD on Neal\'s funnel. Vanilla SGLD\'s isotropic noise cannot resolve both the wide neck (v ≈ 0) and the narrow throat (v < -2); RSGLD\'s state-dependent metric G⁻¹(v, x) = diag(1, e^v) absorbs the curvature directly. The fix is dual to PP\'s non-centered reparameterization — change the metric instead of the coordinates. Slide η to see vanilla SGLD\'s reach extend or shrink as the step size shrinks.';
const ARIA = 'Figure 10: Interactive Neal\'s funnel with vanilla SGLD vs RSGLD';

// -----------------------------------------------------------------------------
// Computation
// -----------------------------------------------------------------------------

interface CachedRun {
  sgld: Float32Array[];
  rsgld: Float32Array[];
  burn: number;
  eta: number;
  T: number;
  chainSeed: number;
}

const yieldToBrowser = (): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, 0));

async function runChains(eta: number, T: number, chainSeed: number): Promise<CachedRun> {
  const target = funnelGradAndLogDensity(SIGMA_V);
  const metric = funnelDiagMetric();
  const burn = Math.floor(T * BURN_FRAC);

  await yieldToBrowser();
  const rngS = mulberry32(chainSeed * 991 + 501);
  const sgld = sgldChain(target.gradU, new Float32Array(START), T, eta, rngS);

  await yieldToBrowser();
  const rngR = mulberry32(chainSeed * 991 + 502);
  const rsgld = rsgldDiagonalChain(
    target.gradU,
    metric.gInvDiag,
    metric.divGInv,
    new Float32Array(START),
    T,
    eta,
    rngR,
  );

  return { sgld, rsgld, burn, eta, T, chainSeed };
}

// -----------------------------------------------------------------------------
// State
// -----------------------------------------------------------------------------

type CompState =
  | { kind: 'idle' }
  | { kind: 'sampling'; reason: 'first' | 'param' | 'reseed' }
  | { kind: 'ready'; run: CachedRun };

// -----------------------------------------------------------------------------
// Component
// -----------------------------------------------------------------------------

export default function NealsFunnelSGMCMC() {
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
    runChains(DEFAULT_ETA, DEFAULT_T, 11).then((run) => {
      if (alive && ticket === ticketRef.current) {
        setState({ kind: 'ready', run });
      }
    });
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const launch = useCallback(
    (e: number, t: number, seed: number, reason: 'param' | 'reseed') => {
      const ticket = ++ticketRef.current;
      setState({ kind: 'sampling', reason });
      runChains(e, t, seed).then((run) => {
        if (ticket === ticketRef.current) {
          setState({ kind: 'ready', run });
        }
      });
    },
    [],
  );

  const handleEtaCommit = (next: number) => {
    if (Math.abs(next - eta) < 1e-6) return;
    setEta(next);
    if (state.kind === 'ready') launch(next, T, chainSeed, 'param');
  };
  const handleTCommit = (next: number) => {
    if (next === T) return;
    setT(next);
    if (state.kind === 'ready') launch(eta, next, chainSeed, 'param');
  };
  const handleReseed = () => {
    const next = chainSeed + 1;
    setChainSeed(next);
    if (state.kind === 'ready') launch(eta, T, next, 'reseed');
  };

  const isBlocked = state.kind === 'sampling' || state.kind === 'idle';
  const ready = state.kind === 'ready' ? state : null;

  return (
    <figure
      ref={ref as React.RefObject<HTMLElement>}
      aria-label={ARIA}
      style={{
        width: '100%',
        margin: '2rem auto',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
      }}
    >
      <div style={{ width: '100%', maxWidth: '880px' }}>
        {!ready ? (
          <FallbackWithBanner
            label={
              state.kind === 'sampling' && state.reason === 'first'
                ? 'Running SGLD + RSGLD chains on funnel (~2 s)…'
                : state.kind === 'sampling'
                  ? 'Re-running funnel chains…'
                  : 'Loading…'
            }
          />
        ) : (
          <>
            <Controls
              eta={eta}
              T={T}
              onEtaCommit={handleEtaCommit}
              onTCommit={handleTCommit}
              onReseed={handleReseed}
              isBlocked={isBlocked}
            />
            <Panels run={ready.run} width={width} />
          </>
        )}
      </div>
      <figcaption
        style={{
          fontSize: '0.875rem',
          lineHeight: 1.5,
          color: 'var(--color-text-muted, #666)',
          marginTop: '0.75rem',
          textAlign: 'left',
          maxWidth: '880px',
          padding: '0 0.5rem',
        }}
      >
        {CAPTION}
      </figcaption>
    </figure>
  );
}

function FallbackWithBanner({ label }: { label: string }) {
  return (
    <div style={{ position: 'relative', width: '100%' }}>
      <img
        src={FIGURE_PATH}
        alt={ALT}
        loading="lazy"
        style={{
          width: '100%',
          height: 'auto',
          backgroundColor: 'var(--color-surface, transparent)',
          borderRadius: '0.25rem',
          boxShadow: '0 1px 2px var(--color-shadow, rgba(0, 0, 0, 0.04))',
          opacity: 0.55,
        }}
      />
      <div
        role="status"
        aria-live="polite"
        style={{
          position: 'absolute',
          left: '50%',
          top: '50%',
          transform: 'translate(-50%, -50%)',
          padding: '0.5rem 0.875rem',
          background: 'var(--color-surface, #fff)',
          border: '1px solid var(--color-border, #ccc)',
          borderRadius: '4px',
          fontSize: '0.875rem',
          color: 'var(--color-text, #333)',
          boxShadow: '0 2px 6px rgba(0,0,0,0.08)',
        }}
      >
        {label}
      </div>
    </div>
  );
}

// -----------------------------------------------------------------------------
// Controls
// -----------------------------------------------------------------------------

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
  onEtaCommit: (next: number) => void;
  onTCommit: (next: number) => void;
  onReseed: () => void;
  isBlocked: boolean;
}) {
  const [localEta, setLocalEta] = useState(eta);
  const [localT, setLocalT] = useState(T);
  useEffect(() => setLocalEta(eta), [eta]);
  useEffect(() => setLocalT(T), [T]);
  return (
    <div
      style={{
        display: 'flex',
        flexWrap: 'wrap',
        gap: '0.75rem 1.25rem',
        alignItems: 'center',
        fontSize: '0.875rem',
        color: 'var(--color-text, #333)',
        marginBottom: '0.75rem',
      }}
    >
      <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}>
        <span>η:</span>
        <input
          type="range"
          min={0.005}
          max={0.05}
          step={0.005}
          value={localEta}
          onChange={(e) => setLocalEta(Number(e.target.value))}
          onMouseUp={(e) => onEtaCommit(Number((e.target as HTMLInputElement).value))}
          onTouchEnd={(e) => onEtaCommit(Number((e.target as HTMLInputElement).value))}
          onKeyUp={(e) => onEtaCommit(Number((e.target as HTMLInputElement).value))}
          style={{ width: '120px' }}
          disabled={isBlocked}
          aria-label="Step size eta"
        />
        <span style={{ fontVariantNumeric: 'tabular-nums', minWidth: '4em' }}>
          {localEta.toFixed(3)}
        </span>
      </label>
      <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}>
        <span>T:</span>
        <input
          type="range"
          min={5000}
          max={30000}
          step={1000}
          value={localT}
          onChange={(e) => setLocalT(Number(e.target.value))}
          onMouseUp={(e) => onTCommit(Number((e.target as HTMLInputElement).value))}
          onTouchEnd={(e) => onTCommit(Number((e.target as HTMLInputElement).value))}
          onKeyUp={(e) => onTCommit(Number((e.target as HTMLInputElement).value))}
          style={{ width: '120px' }}
          disabled={isBlocked}
          aria-label="Chain length T"
        />
        <span style={{ fontVariantNumeric: 'tabular-nums', minWidth: '4em' }}>{localT}</span>
      </label>
      <button
        type="button"
        onClick={onReseed}
        disabled={isBlocked}
        style={{
          padding: '0.3rem 0.7rem',
          border: '1px solid var(--color-border, #ccc)',
          borderRadius: '4px',
          background: isBlocked ? 'var(--color-surface-muted, #f0f0f0)' : 'var(--color-surface, #fff)',
          color: 'var(--color-text, #333)',
          cursor: isBlocked ? 'not-allowed' : 'pointer',
          fontSize: '0.875rem',
        }}
      >
        Resample chain
      </button>
      {isBlocked && (
        <span style={{ color: 'var(--color-text-muted, #888)', fontStyle: 'italic' }}>
          sampling…
        </span>
      )}
    </div>
  );
}

// -----------------------------------------------------------------------------
// Panels
// -----------------------------------------------------------------------------

function Panels({ run, width }: { run: CachedRun; width: number }) {
  const layoutWidth = Math.max(width, 320);
  const stack = layoutWidth < 720;
  const panelW = stack ? layoutWidth : (layoutWidth - 32) / 3;
  const panelH = 260;

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: stack ? 'column' : 'row',
        gap: stack ? '12px' : '16px',
        alignItems: stack ? 'center' : 'flex-start',
      }}
    >
      <Panel title={`(a) Vanilla SGLD, η = ${run.eta.toFixed(3)}`} width={panelW} height={panelH}>
        <FunnelTraj
          chain={run.sgld}
          burn={run.burn}
          stroke={paletteSGMCMC.sgld}
          width={panelW}
          height={panelH}
        />
      </Panel>
      <Panel title={`(b) RSGLD, η = ${run.eta.toFixed(3)}`} width={panelW} height={panelH}>
        <FunnelTraj
          chain={run.rsgld}
          burn={run.burn}
          stroke={paletteSGMCMC.rsgld}
          width={panelW}
          height={panelH}
        />
      </Panel>
      <Panel title="(c) Marginal of v vs π(v)" width={panelW} height={panelH}>
        <MarginalV
          sgld={run.sgld}
          rsgld={run.rsgld}
          burn={run.burn}
          width={panelW}
          height={panelH}
        />
      </Panel>
    </div>
  );
}

function Panel({
  title,
  width,
  height,
  children,
}: {
  title: string;
  width: number;
  height: number;
  children: React.ReactNode;
}) {
  return (
    <div style={{ width, display: 'flex', flexDirection: 'column' }}>
      <div
        style={{
          fontSize: '0.825rem',
          fontWeight: 600,
          color: 'var(--color-text, #333)',
          marginBottom: '0.25rem',
        }}
      >
        {title}
      </div>
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img">
        {children}
      </svg>
    </div>
  );
}

// -----------------------------------------------------------------------------
// Funnel-trajectory panel
// -----------------------------------------------------------------------------

function FunnelTraj({
  chain,
  burn,
  stroke,
  width,
  height,
}: {
  chain: Float32Array[];
  burn: number;
  stroke: string;
  width: number;
  height: number;
}) {
  const margin = { top: 8, right: 12, bottom: 28, left: 36 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  const xS = d3.scaleLinear().domain([-7, 7]).range([0, innerW]);
  const yS = d3.scaleLinear().domain([-15, 15]).range([innerH, 0]);

  // Funnel iso-density contours via marching squares on log-density.
  const target = useMemo(() => funnelGradAndLogDensity(SIGMA_V), []);
  const contours = useMemo(() => {
    const grid = 60;
    const vMin = -7,
      vMax = 7,
      xMin = -15,
      xMax = 15;
    const dv = (vMax - vMin) / grid;
    const dx = (xMax - xMin) / grid;
    // First pass: find max log density over grid for normalization.
    let lpMax = -Infinity;
    for (let i = 0; i <= grid; i++) {
      for (let j = 0; j <= grid; j++) {
        const lp = target.logDensity(vMin + i * dv, xMin + j * dx);
        if (Number.isFinite(lp) && lp > lpMax) lpMax = lp;
      }
    }
    // Iso-density levels (relative log density): 5 levels exponentially spaced.
    const levels = [-1, -3, -6, -9, -12];
    const segs: string[][] = levels.map(() => []);
    for (let i = 0; i < grid; i++) {
      for (let j = 0; j < grid; j++) {
        const v = vMin + i * dv;
        const x = xMin + j * dx;
        const lp00 = target.logDensity(v, x) - lpMax;
        const lp10 = target.logDensity(v + dv, x) - lpMax;
        const lp01 = target.logDensity(v, x + dx) - lpMax;
        const lp11 = target.logDensity(v + dv, x + dx) - lpMax;
        levels.forEach((thr, k) => {
          const v00 = lp00 - thr;
          const v10 = lp10 - thr;
          const v01 = lp01 - thr;
          const v11 = lp11 - thr;
          const c =
            (v00 > 0 ? 1 : 0) + (v10 > 0 ? 2 : 0) + (v11 > 0 ? 4 : 0) + (v01 > 0 ? 8 : 0);
          if (c === 0 || c === 15) return;
          const lerp = (a: number, b: number) => a / (a - b);
          const eB = { x: v + lerp(v00, v10) * dv, y: x };
          const eR = { x: v + dv, y: x + lerp(v10, v11) * dx };
          const eT = { x: v + lerp(v01, v11) * dv, y: x + dx };
          const eL = { x: v, y: x + lerp(v00, v01) * dx };
          const seg = (a: { x: number; y: number }, b: { x: number; y: number }) =>
            segs[k].push(`M${xS(a.x)},${yS(a.y)}L${xS(b.x)},${yS(b.y)}`);
          switch (c) {
            case 1:
            case 14:
              seg(eB, eL);
              break;
            case 2:
            case 13:
              seg(eB, eR);
              break;
            case 4:
            case 11:
              seg(eR, eT);
              break;
            case 8:
            case 7:
              seg(eL, eT);
              break;
            case 3:
            case 12:
              seg(eL, eR);
              break;
            case 6:
            case 9:
              seg(eB, eT);
              break;
            case 5:
              seg(eL, eT);
              seg(eB, eR);
              break;
            case 10:
              seg(eL, eB);
              seg(eT, eR);
              break;
          }
        });
      }
    }
    return segs.map((s) => s.join(' '));
  }, [target, xS, yS]);

  const tracePath = useMemo(() => {
    const pts = chain.slice(burn);
    return (
      d3
        .line<Float32Array>()
        .x((p) => xS(p[0]))
        .y((p) => yS(p[1]))
        .defined((p) => p[0] >= -7 && p[0] <= 7 && p[1] >= -15 && p[1] <= 15)(pts) ?? ''
    );
  }, [chain, burn, xS, yS]);

  return (
    <g transform={`translate(${margin.left},${margin.top})`}>
      <rect x={0} y={0} width={innerW} height={innerH} fill="var(--color-surface, #fafafa)" />
      {contours.map((d, i) => (
        <path
          key={i}
          d={d}
          fill="none"
          stroke={paletteSGMCMC.target}
          strokeWidth={0.5}
          opacity={0.4}
        />
      ))}
      <path d={tracePath} fill="none" stroke={stroke} strokeWidth={0.4} opacity={0.65} />
      <circle cx={xS(START[0])} cy={yS(START[1])} r={3.5} fill="black" />
      <line x1={0} x2={innerW} y1={innerH} y2={innerH} stroke="var(--color-text-muted, #888)" />
      <line x1={0} x2={0} y1={0} y2={innerH} stroke="var(--color-text-muted, #888)" />
      <text x={innerW / 2} y={innerH + 18} textAnchor="middle" fontSize={10} fill="var(--color-text, #333)">
        v
      </text>
      <text
        transform={`translate(${-26}, ${innerH / 2}) rotate(-90)`}
        textAnchor="middle"
        fontSize={10}
        fill="var(--color-text, #333)"
      >
        x
      </text>
    </g>
  );
}

// -----------------------------------------------------------------------------
// Marginal-of-v panel
// -----------------------------------------------------------------------------

function MarginalV({
  sgld,
  rsgld,
  burn,
  width,
  height,
}: {
  sgld: Float32Array[];
  rsgld: Float32Array[];
  burn: number;
  width: number;
  height: number;
}) {
  const margin = { top: 8, right: 12, bottom: 28, left: 40 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  const lo = -9;
  const hi = 9;
  const nBins = 36;

  const sgldVs = useMemo(() => {
    const out = new Float32Array(sgld.length - burn);
    for (let i = 0; i < out.length; i++) out[i] = sgld[i + burn][0];
    return out;
  }, [sgld, burn]);
  const rsgldVs = useMemo(() => {
    const out = new Float32Array(rsgld.length - burn);
    for (let i = 0; i < out.length; i++) out[i] = rsgld[i + burn][0];
    return out;
  }, [rsgld, burn]);

  const sgldHist = useMemo(() => histogram(sgldVs, lo, hi, nBins), [sgldVs]);
  const rsgldHist = useMemo(() => histogram(rsgldVs, lo, hi, nBins), [rsgldVs]);

  const truePdf = (v: number) =>
    Math.exp(-(v * v) / (2 * SIGMA_V * SIGMA_V)) / (Math.sqrt(2 * Math.PI) * SIGMA_V);

  const xS = d3.scaleLinear().domain([lo, hi]).range([0, innerW]);
  const yMax =
    Math.max(
      ...Array.from(sgldHist.density),
      ...Array.from(rsgldHist.density),
      ...Array.from({ length: 200 }, (_, i) => truePdf(lo + ((hi - lo) * i) / 199)),
    ) * 1.15;
  const yS = d3.scaleLinear().domain([0, yMax]).range([innerH, 0]);

  const truePath = (() => {
    const pts: string[] = [];
    for (let i = 0; i < 200; i++) {
      const v = lo + ((hi - lo) * i) / 199;
      pts.push(`${i === 0 ? 'M' : 'L'}${xS(v)},${yS(truePdf(v))}`);
    }
    return pts.join(' ');
  })();

  const binWidth = (hi - lo) / nBins;

  return (
    <g transform={`translate(${margin.left},${margin.top})`}>
      <rect x={0} y={0} width={innerW} height={innerH} fill="var(--color-surface, #fafafa)" />
      {/* SGLD bars */}
      {Array.from(sgldHist.centers).map((c, i) => (
        <rect
          key={`s-${i}`}
          x={xS(c - binWidth / 2)}
          y={yS(sgldHist.density[i])}
          width={Math.max(0, xS(c + binWidth / 2) - xS(c - binWidth / 2) - 1)}
          height={Math.max(0, innerH - yS(sgldHist.density[i]))}
          fill={paletteSGMCMC.sgld}
          opacity={0.55}
        />
      ))}
      {/* RSGLD bars */}
      {Array.from(rsgldHist.centers).map((c, i) => (
        <rect
          key={`r-${i}`}
          x={xS(c - binWidth / 2)}
          y={yS(rsgldHist.density[i])}
          width={Math.max(0, xS(c + binWidth / 2) - xS(c - binWidth / 2) - 1)}
          height={Math.max(0, innerH - yS(rsgldHist.density[i]))}
          fill={paletteSGMCMC.rsgld}
          opacity={0.55}
        />
      ))}
      {/* True π(v) */}
      <path d={truePath} fill="none" stroke={paletteSGMCMC.target} strokeWidth={1.6} strokeDasharray="4 3" />
      <line x1={0} x2={innerW} y1={innerH} y2={innerH} stroke="var(--color-text-muted, #888)" />
      <line x1={0} x2={0} y1={0} y2={innerH} stroke="var(--color-text-muted, #888)" />
      {[-6, -3, 0, 3, 6].map((t) => (
        <text key={t} x={xS(t)} y={innerH + 16} textAnchor="middle" fontSize={10} fill="var(--color-text, #333)">
          {t}
        </text>
      ))}
      <text x={innerW / 2} y={innerH + 24} textAnchor="middle" fontSize={10} fill="var(--color-text, #333)">
        v
      </text>
      <text
        transform={`translate(${-30}, ${innerH / 2}) rotate(-90)`}
        textAnchor="middle"
        fontSize={10}
        fill="var(--color-text, #333)"
      >
        density
      </text>
      <g transform={`translate(${innerW - 86}, 8)`}>
        <rect x={0} y={-6} width={84} height={56} fill="var(--color-surface, #fff)" stroke="var(--color-border, #ccc)" />
        <rect x={4} y={1} width={12} height={6} fill={paletteSGMCMC.sgld} opacity={0.55} />
        <text x={20} y={7} fontSize={10} fill="var(--color-text, #333)">SGLD</text>
        <rect x={4} y={15} width={12} height={6} fill={paletteSGMCMC.rsgld} opacity={0.55} />
        <text x={20} y={21} fontSize={10} fill="var(--color-text, #333)">RSGLD</text>
        <line x1={4} x2={16} y1={32} y2={32} stroke={paletteSGMCMC.target} strokeWidth={2} strokeDasharray="3 2" />
        <text x={20} y={35} fontSize={10} fill="var(--color-text, #333)">π(v)</text>
      </g>
    </g>
  );
}
