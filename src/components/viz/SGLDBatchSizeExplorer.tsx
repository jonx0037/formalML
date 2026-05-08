// =============================================================================
// SGLDBatchSizeExplorer.tsx
//
// §6 Stochastic-Gradient Langevin Dynamics. Three panels on the toy Bayesian
// linear regression (N = 100, D = 2, σ_noise = 0.5, τ_prior = 5):
//
//   (a) Full-batch SGLD chain (B = N) — clean Langevin trajectory; the chain
//       traces the closed-form posterior contours with Brownian jitter.
//   (b) Mini-batch SGLD chain at user-set B — same dynamics with the
//       N/B-rescaled gradient estimator. Wider chain spread quantifies the
//       mini-batch noise injection.
//   (c) Running estimate of E_π[θ_1]: the constant-η chain stalls at a
//       positive bias floor (the §8 O(η + 1/B) bound made visible), while a
//       Welling–Teh-scheduled chain's running mean decays to zero.
//
// Interactive controls:
//   - log₁₀ η slider (-4..-2, default ≈ -3.3) — step size. Smaller η → less
//     jitter, slower mixing; larger η → more jitter, larger §8 bias.
//   - B slider (5..100, default 10) — mini-batch size. Halving B doubles the
//     gradient-estimator variance (§8 panel (b)).
//   - Chain length T (1k..30k, default 4k) — applies to all three panels.
//   - Resample button — re-rolls the chain seed (data seed is fixed so the
//     posterior contours stay still).
//
// Computation: in-browser via shared/sgmcmc.ts. No precompute, no JSON fetch.
// Static fallback: /images/topics/stochastic-gradient-mcmc/06_sgld.png.
// =============================================================================

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import { mulberry32 } from './shared/bayesian-ml';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  blrGradFull,
  blrGradMinibatch,
  blrPosterior,
  makeBLRDataset,
  paletteSGMCMC,
  runningMean,
  sgldChain,
  sgldChainScheduled,
  wellingTehSchedule,
} from './shared/sgmcmc';

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

const DATA_SEED = 41;
const DEFAULT_LOG10_ETA = -3.3; // η ≈ 5e-4
const DEFAULT_B = 10;
const DEFAULT_T = 4000;
const T_MAX_RUNNING = 20000; // longer chain for panel (c) so the schedule has room to decay

const FIGURE_PATH = '/images/topics/stochastic-gradient-mcmc/06_sgld.png';
const ALT =
  'Three-panel figure on Bayesian linear regression posterior. Panel (a): full-batch SGLD chain traces closed-form posterior contours with mild Brownian jitter. Panel (b): mini-batch SGLD chain at the same step size — wider spread reflects mini-batch gradient noise. Panel (c): running mean estimate of θ₁ — constant-η chain stalls at a positive bias floor while the Welling–Teh-scheduled chain decays to zero.';
const CAPTION =
  'Figure 6. SGLD on Bayesian linear regression. (a) Full-batch SGLD; (b) mini-batch SGLD with the N/B-rescaled estimator — same dynamics, wider spread. (c) Running estimate of E_π[θ₁]: the constant-η chain stalls at the §8 bias floor, the Welling–Teh-scheduled chain decays to zero. Slide η to trade jitter for mixing speed; slide B to see the §8 1/B scaling come alive in panel (b).';
const ARIA = 'Figure 6: Interactive SGLD chain on Bayesian linear regression';

// -----------------------------------------------------------------------------
// Cached computation
// -----------------------------------------------------------------------------

interface CachedDataset {
  spec: ReturnType<typeof makeBLRDataset>;
  posterior: ReturnType<typeof blrPosterior>;
  trueTheta1: number;
}

interface CachedChains {
  fullBatch: Float32Array[]; // B = N
  miniBatch: Float32Array[]; // user B
  miniBatchLong: Float32Array[]; // for running-mean panel
  scheduleLong: Float32Array[]; // Welling–Teh schedule, for panel (c)
  eta: number;
  B: number;
  T: number;
  chainSeed: number;
}

function makeDataset(): CachedDataset {
  const spec = makeBLRDataset(DATA_SEED);
  const posterior = blrPosterior(spec);
  return { spec, posterior, trueTheta1: spec.trueTheta[0] };
}

const yieldToBrowser = (): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, 0));

async function runChains(
  ds: CachedDataset,
  eta: number,
  B: number,
  T: number,
  chainSeed: number,
): Promise<CachedChains> {
  await yieldToBrowser();
  const theta0 = new Float32Array([
    ds.posterior.muPost[0] + 0.4,
    ds.posterior.muPost[1] + 0.4,
  ]);

  // Panel (a): full-batch SGLD — gradient is deterministic, only Brownian noise.
  const fullGrad = blrGradFull(ds.spec);
  const rngFull = mulberry32(chainSeed * 991 + 7);
  const fullBatch = sgldChain(fullGrad, theta0, T, eta, rngFull);

  // Panel (b): mini-batch SGLD at user B.
  const rngMiniB = mulberry32(chainSeed * 991 + 13);
  const miniGrad = blrGradMinibatch(ds.spec, B, rngMiniB);
  const miniBatch = sgldChain(miniGrad, theta0, T, eta, rngMiniB);

  // Panel (c): two long chains — constant-η and Welling–Teh schedule.
  await yieldToBrowser();
  const rngLongConst = mulberry32(chainSeed * 991 + 17);
  const longGrad = blrGradMinibatch(ds.spec, B, rngLongConst);
  const miniBatchLong = sgldChain(longGrad, theta0, T_MAX_RUNNING, eta, rngLongConst);

  await yieldToBrowser();
  const rngLongSched = mulberry32(chainSeed * 991 + 19);
  const schedGrad = blrGradMinibatch(ds.spec, B, rngLongSched);
  const schedule = wellingTehSchedule(8e-3, 50, 0.55);
  const scheduleLong = sgldChainScheduled(schedGrad, theta0, T_MAX_RUNNING, schedule, rngLongSched);

  return { fullBatch, miniBatch, miniBatchLong, scheduleLong, eta, B, T, chainSeed };
}

// -----------------------------------------------------------------------------
// State
// -----------------------------------------------------------------------------

type CompState =
  | { kind: 'idle' }
  | { kind: 'sampling'; ds: CachedDataset; reason: 'first' | 'param' | 'reseed' }
  | { kind: 'ready'; ds: CachedDataset; chains: CachedChains };

// -----------------------------------------------------------------------------
// Component
// -----------------------------------------------------------------------------

export default function SGLDBatchSizeExplorer() {
  const { ref, width } = useResizeObserver<HTMLElement>();
  const [state, setState] = useState<CompState>({ kind: 'idle' });
  const [logEta, setLogEta] = useState(DEFAULT_LOG10_ETA);
  const [B, setB] = useState(DEFAULT_B);
  const [T, setT] = useState(DEFAULT_T);
  const [chainSeed, setChainSeed] = useState(11);
  const ticketRef = useRef(0);

  // First-hydration: build dataset, run initial chains.
  useEffect(() => {
    let alive = true;
    const ds = makeDataset();
    setState({ kind: 'sampling', ds, reason: 'first' });
    const ticket = ++ticketRef.current;
    runChains(ds, Math.pow(10, DEFAULT_LOG10_ETA), DEFAULT_B, DEFAULT_T, 11).then((chains) => {
      if (alive && ticket === ticketRef.current) {
        setState({ kind: 'ready', ds, chains });
      }
    });
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const launch = useCallback(
    (ds: CachedDataset, eta: number, b: number, t: number, seed: number, reason: 'param' | 'reseed') => {
      const ticket = ++ticketRef.current;
      setState({ kind: 'sampling', ds, reason });
      runChains(ds, eta, b, t, seed).then((chains) => {
        if (ticket === ticketRef.current) {
          setState({ kind: 'ready', ds, chains });
        }
      });
    },
    [],
  );

  const handleLogEtaCommit = (next: number) => {
    if (Math.abs(next - logEta) < 1e-6) return;
    setLogEta(next);
    if (state.kind === 'ready') launch(state.ds, Math.pow(10, next), B, T, chainSeed, 'param');
  };
  const handleBCommit = (next: number) => {
    if (next === B) return;
    setB(next);
    if (state.kind === 'ready') launch(state.ds, Math.pow(10, logEta), next, T, chainSeed, 'param');
  };
  const handleTCommit = (next: number) => {
    if (next === T) return;
    setT(next);
    if (state.kind === 'ready') launch(state.ds, Math.pow(10, logEta), B, next, chainSeed, 'param');
  };
  const handleReseed = () => {
    const next = chainSeed + 1;
    setChainSeed(next);
    if (state.kind === 'ready') launch(state.ds, Math.pow(10, logEta), B, T, next, 'reseed');
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
                ? 'Running initial SGLD chains (~2 s)…'
                : state.kind === 'sampling'
                  ? 'Re-running SGLD chains…'
                  : 'Loading…'
            }
          />
        ) : (
          <>
            <Controls
              logEta={logEta}
              B={B}
              T={T}
              onLogEtaCommit={handleLogEtaCommit}
              onBCommit={handleBCommit}
              onTCommit={handleTCommit}
              onReseed={handleReseed}
              isBlocked={isBlocked}
            />
            <Panels ds={ready.ds} chains={ready.chains} width={width} />
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
  logEta,
  B,
  T,
  onLogEtaCommit,
  onBCommit,
  onTCommit,
  onReseed,
  isBlocked,
}: {
  logEta: number;
  B: number;
  T: number;
  onLogEtaCommit: (next: number) => void;
  onBCommit: (next: number) => void;
  onTCommit: (next: number) => void;
  onReseed: () => void;
  isBlocked: boolean;
}) {
  const [localEta, setLocalEta] = useState(logEta);
  const [localB, setLocalB] = useState(B);
  const [localT, setLocalT] = useState(T);
  useEffect(() => setLocalEta(logEta), [logEta]);
  useEffect(() => setLocalB(B), [B]);
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
        <span>log₁₀ η:</span>
        <input
          type="range"
          min={-4}
          max={-2}
          step={0.1}
          value={localEta}
          onChange={(e) => setLocalEta(Number(e.target.value))}
          onMouseUp={(e) => onLogEtaCommit(Number((e.target as HTMLInputElement).value))}
          onTouchEnd={(e) => onLogEtaCommit(Number((e.target as HTMLInputElement).value))}
          onKeyUp={(e) => onLogEtaCommit(Number((e.target as HTMLInputElement).value))}
          style={{ width: '120px' }}
          disabled={isBlocked}
          aria-label="SGLD step size log10 eta"
        />
        <span style={{ fontVariantNumeric: 'tabular-nums', minWidth: '4em' }}>
          η={Math.pow(10, localEta).toExponential(1)}
        </span>
      </label>
      <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}>
        <span>B:</span>
        <input
          type="range"
          min={5}
          max={100}
          step={5}
          value={localB}
          onChange={(e) => setLocalB(Number(e.target.value))}
          onMouseUp={(e) => onBCommit(Number((e.target as HTMLInputElement).value))}
          onTouchEnd={(e) => onBCommit(Number((e.target as HTMLInputElement).value))}
          onKeyUp={(e) => onBCommit(Number((e.target as HTMLInputElement).value))}
          style={{ width: '120px' }}
          disabled={isBlocked}
          aria-label="Mini-batch size B"
        />
        <span style={{ fontVariantNumeric: 'tabular-nums', minWidth: '2.5em' }}>{localB}</span>
      </label>
      <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}>
        <span>T:</span>
        <input
          type="range"
          min={1000}
          max={10000}
          step={500}
          value={localT}
          onChange={(e) => setLocalT(Number(e.target.value))}
          onMouseUp={(e) => onTCommit(Number((e.target as HTMLInputElement).value))}
          onTouchEnd={(e) => onTCommit(Number((e.target as HTMLInputElement).value))}
          onKeyUp={(e) => onTCommit(Number((e.target as HTMLInputElement).value))}
          style={{ width: '120px' }}
          disabled={isBlocked}
          aria-label="Chain length T"
        />
        <span style={{ fontVariantNumeric: 'tabular-nums', minWidth: '3.5em' }}>{localT}</span>
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

function Panels({
  ds,
  chains,
  width,
}: {
  ds: CachedDataset;
  chains: CachedChains;
  width: number;
}) {
  const layoutWidth = Math.max(width, 320);
  const stack = layoutWidth < 720;
  const trajW = stack ? layoutWidth : (layoutWidth - 16) / 2;
  const trajH = 240;
  const runW = layoutWidth;
  const runH = 200;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      <div
        style={{
          display: 'flex',
          flexDirection: stack ? 'column' : 'row',
          gap: stack ? '12px' : '16px',
          alignItems: stack ? 'center' : 'flex-start',
        }}
      >
        <Panel
          title={`(a) Full-batch SGLD (B = ${ds.spec.N})`}
          width={trajW}
          height={trajH}
        >
          <Trajectory
            chain={chains.fullBatch.slice(0, chains.T)}
            ds={ds}
            stroke={paletteSGMCMC.langevin}
            width={trajW}
            height={trajH}
          />
        </Panel>
        <Panel title={`(b) Mini-batch SGLD (B = ${chains.B})`} width={trajW} height={trajH}>
          <Trajectory
            chain={chains.miniBatch.slice(0, chains.T)}
            ds={ds}
            stroke={paletteSGMCMC.sgld}
            width={trajW}
            height={trajH}
          />
        </Panel>
      </div>
      <Panel
        title="(c) Running mean of θ₁: constant-η stalls at bias floor; Welling–Teh schedule decays to truth"
        width={runW}
        height={runH}
      >
        <RunningMeanPanel
          ds={ds}
          chains={chains}
          width={runW}
          height={runH}
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
// Trajectory panel
// -----------------------------------------------------------------------------

function Trajectory({
  chain,
  ds,
  stroke,
  width,
  height,
}: {
  chain: Float32Array[];
  ds: CachedDataset;
  stroke: string;
  width: number;
  height: number;
}) {
  const margin = { top: 8, right: 12, bottom: 28, left: 36 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  // Chain extents combined with posterior 3σ for stable bounds across resamples.
  const mu = ds.posterior.muPost;
  const s00 = Math.sqrt(ds.posterior.sigmaPost[0]);
  const s11 = Math.sqrt(ds.posterior.sigmaPost[3]);
  const x0Min = mu[0] - 4 * s00;
  const x0Max = mu[0] + 4 * s00;
  const x1Min = mu[1] - 4 * s11;
  const x1Max = mu[1] + 4 * s11;

  const xS = d3.scaleLinear().domain([x0Min, x0Max]).range([0, innerW]);
  const yS = d3.scaleLinear().domain([x1Min, x1Max]).range([innerH, 0]);

  // Posterior contours via Mahalanobis levels of the analytic Gaussian.
  const sigmaPostInv = ds.posterior.sigmaPostInv;
  const mahalanobis = (a: number, b: number) => {
    const da = a - mu[0];
    const db = b - mu[1];
    return (
      sigmaPostInv[0] * da * da +
      2 * sigmaPostInv[1] * da * db +
      sigmaPostInv[3] * db * db
    );
  };
  const contourPaths = useMemo(() => {
    // 1, 2, 3 std contours: m² = 2.30, 6.18, 11.83 (chi-square 2-df CDF)
    const targets = [2.3, 6.18, 11.83];
    const grid = 60;
    const out: string[] = [];
    const dx = (x0Max - x0Min) / grid;
    const dy = (x1Max - x1Min) / grid;
    for (const t of targets) {
      // Walk the contour as marching squares — minimal hand roll.
      const segs: { x1: number; y1: number; x2: number; y2: number }[] = [];
      for (let i = 0; i < grid; i++) {
        for (let j = 0; j < grid; j++) {
          const x = x0Min + i * dx;
          const y = x1Min + j * dy;
          const v00 = mahalanobis(x, y) - t;
          const v10 = mahalanobis(x + dx, y) - t;
          const v01 = mahalanobis(x, y + dy) - t;
          const v11 = mahalanobis(x + dx, y + dy) - t;
          // 4-bit code
          const c =
            (v00 > 0 ? 1 : 0) +
            (v10 > 0 ? 2 : 0) +
            (v11 > 0 ? 4 : 0) +
            (v01 > 0 ? 8 : 0);
          if (c === 0 || c === 15) continue;
          const lerp = (a: number, b: number) => a / (a - b);
          const lx = (a: number, b: number) => x + lerp(a, b) * dx;
          const ly = (a: number, b: number) => y + lerp(a, b) * dy;
          // Bottom (v00, v10), right (v10, v11), top (v01, v11), left (v00, v01)
          const eB = { x: lx(v00, v10), y };
          const eR = { x: x + dx, y: ly(v10, v11) };
          const eT = { x: lx(v01, v11), y: y + dy };
          const eL = { x, y: ly(v00, v01) };
          const push = (a: { x: number; y: number }, b: { x: number; y: number }) =>
            segs.push({ x1: a.x, y1: a.y, x2: b.x, y2: b.y });
          switch (c) {
            case 1:
            case 14:
              push(eB, eL);
              break;
            case 2:
            case 13:
              push(eB, eR);
              break;
            case 4:
            case 11:
              push(eR, eT);
              break;
            case 8:
            case 7:
              push(eL, eT);
              break;
            case 3:
            case 12:
              push(eL, eR);
              break;
            case 6:
            case 9:
              push(eB, eT);
              break;
            case 5:
              push(eL, eT);
              push(eB, eR);
              break;
            case 10:
              push(eL, eB);
              push(eT, eR);
              break;
          }
        }
      }
      const path = segs
        .map((s) => `M${xS(s.x1)},${yS(s.y1)}L${xS(s.x2)},${yS(s.y2)}`)
        .join(' ');
      out.push(path);
    }
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [width, height, ds]);

  // Chain trajectory line
  const traceLine = useMemo(() => {
    const line = d3
      .line<Float32Array>()
      .x((p) => xS(p[0]))
      .y((p) => yS(p[1]))
      .defined((p) => p[0] >= x0Min && p[0] <= x0Max && p[1] >= x1Min && p[1] <= x1Max);
    return line(chain) ?? '';
  }, [chain, xS, yS, x0Min, x0Max, x1Min, x1Max]);

  return (
    <g transform={`translate(${margin.left},${margin.top})`}>
      <rect x={0} y={0} width={innerW} height={innerH} fill="var(--color-surface, #fafafa)" />
      {contourPaths.map((d, i) => (
        <path
          key={i}
          d={d}
          fill="none"
          stroke={paletteSGMCMC.target}
          strokeWidth={0.6}
          opacity={0.45}
        />
      ))}
      <path d={traceLine} fill="none" stroke={stroke} strokeWidth={0.6} opacity={0.7} />
      <circle cx={xS(mu[0])} cy={yS(mu[1])} r={4} fill={paletteSGMCMC.target} />
      <line x1={0} x2={innerW} y1={innerH} y2={innerH} stroke="var(--color-text-muted, #888)" />
      <line x1={0} x2={0} y1={0} y2={innerH} stroke="var(--color-text-muted, #888)" />
      <text
        x={innerW / 2}
        y={innerH + 18}
        textAnchor="middle"
        fontSize={10}
        fill="var(--color-text, #333)"
      >
        θ₁ (intercept)
      </text>
      <text
        transform={`translate(${-26}, ${innerH / 2}) rotate(-90)`}
        textAnchor="middle"
        fontSize={10}
        fill="var(--color-text, #333)"
      >
        θ₂ (slope)
      </text>
    </g>
  );
}

// -----------------------------------------------------------------------------
// Running-mean panel
// -----------------------------------------------------------------------------

function RunningMeanPanel({
  ds,
  chains,
  width,
  height,
}: {
  ds: CachedDataset;
  chains: CachedChains;
  width: number;
  height: number;
}) {
  const margin = { top: 12, right: 100, bottom: 30, left: 50 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  const muTrue = ds.posterior.muPost[0];

  const runConst = useMemo(() => {
    const trace = new Float32Array(chains.miniBatchLong.length);
    for (let i = 0; i < trace.length; i++) trace[i] = chains.miniBatchLong[i][0];
    return runningMean(trace);
  }, [chains.miniBatchLong]);

  const runSched = useMemo(() => {
    const trace = new Float32Array(chains.scheduleLong.length);
    for (let i = 0; i < trace.length; i++) trace[i] = chains.scheduleLong[i][0];
    return runningMean(trace);
  }, [chains.scheduleLong]);

  const errConst = useMemo(() => {
    const out = new Float32Array(runConst.length);
    for (let i = 0; i < out.length; i++) out[i] = Math.abs(runConst[i] - muTrue);
    return out;
  }, [runConst, muTrue]);
  const errSched = useMemo(() => {
    const out = new Float32Array(runSched.length);
    for (let i = 0; i < out.length; i++) out[i] = Math.abs(runSched[i] - muTrue);
    return out;
  }, [runSched, muTrue]);

  const N = errConst.length;
  // Drop the first 50 to avoid the warm-up spike eating the y-range.
  const start = 50;
  const yFloor = 1e-4;
  const yMax = Math.max(...errConst.slice(start), ...errSched.slice(start), 1e-3) * 1.1;

  const xS = d3
    .scaleLog()
    .domain([Math.max(start, 1), N - 1])
    .range([0, innerW]);
  const yS = d3
    .scaleLog()
    .domain([yFloor, yMax])
    .range([innerH, 0]);

  // Powers-of-ten ticks only — d3's default log-tick density crowds at this width.
  const xTicks = [1e2, 1e3, 1e4].filter((t) => t >= xS.domain()[0] && t <= xS.domain()[1]);
  const yTicks = [1e-4, 1e-3, 1e-2, 1e-1].filter((t) => t >= yS.domain()[0] && t <= yS.domain()[1]);
  const fmtPow = (v: number) => {
    const e = Math.round(Math.log10(v));
    return `10^${e}`;
  };

  const lineConst = d3
    .line<number>()
    .x((_, i) => xS(i + start))
    .y((v) => yS(Math.max(v, yFloor)));
  const lineSched = d3
    .line<number>()
    .x((_, i) => xS(i + start))
    .y((v) => yS(Math.max(v, yFloor)));

  const dataConst = Array.from(errConst.slice(start));
  const dataSched = Array.from(errSched.slice(start));

  return (
    <g transform={`translate(${margin.left},${margin.top})`}>
      <rect x={0} y={0} width={innerW} height={innerH} fill="var(--color-surface, #fafafa)" />
      {/* x grid */}
      {xTicks.map((t) => (
        <g key={`xg-${t}`}>
          <line
            x1={xS(t)}
            x2={xS(t)}
            y1={0}
            y2={innerH}
            stroke="var(--color-text-muted, #888)"
            strokeOpacity={0.15}
          />
          <text
            x={xS(t)}
            y={innerH + 16}
            textAnchor="middle"
            fontSize={10}
            fill="var(--color-text, #333)"
          >
            {fmtPow(t)}
          </text>
        </g>
      ))}
      {/* y grid */}
      {yTicks.map((t) => (
        <g key={`yg-${t}`}>
          <line
            x1={0}
            x2={innerW}
            y1={yS(t)}
            y2={yS(t)}
            stroke="var(--color-text-muted, #888)"
            strokeOpacity={0.15}
          />
          <text
            x={-6}
            y={yS(t) + 3}
            textAnchor="end"
            fontSize={10}
            fill="var(--color-text, #333)"
          >
            {fmtPow(t)}
          </text>
        </g>
      ))}
      <path d={lineConst(dataConst) ?? ''} fill="none" stroke={paletteSGMCMC.sgld} strokeWidth={1.5} />
      <path d={lineSched(dataSched) ?? ''} fill="none" stroke={paletteSGMCMC.langevin} strokeWidth={1.5} />
      <line x1={0} x2={innerW} y1={innerH} y2={innerH} stroke="var(--color-text-muted, #888)" />
      <line x1={0} x2={0} y1={0} y2={innerH} stroke="var(--color-text-muted, #888)" />
      <text
        x={innerW / 2}
        y={innerH + 28}
        textAnchor="middle"
        fontSize={10}
        fill="var(--color-text, #333)"
      >
        iteration n (log)
      </text>
      <text
        transform={`translate(${-38}, ${innerH / 2}) rotate(-90)`}
        textAnchor="middle"
        fontSize={10}
        fill="var(--color-text, #333)"
      >
        |running θ₁ − μ₁| (log)
      </text>
      {/* Legend */}
      <g transform={`translate(${innerW + 8}, 12)`}>
        <rect x={0} y={-6} width={92} height={42} fill="var(--color-surface, #fff)" stroke="var(--color-border, #ccc)" />
        <line x1={4} x2={20} y1={4} y2={4} stroke={paletteSGMCMC.sgld} strokeWidth={2} />
        <text x={24} y={7} fontSize={10} fill="var(--color-text, #333)">
          constant η
        </text>
        <line x1={4} x2={20} y1={22} y2={22} stroke={paletteSGMCMC.langevin} strokeWidth={2} />
        <text x={24} y={25} fontSize={10} fill="var(--color-text, #333)">
          WT schedule
        </text>
      </g>
    </g>
  );
}
