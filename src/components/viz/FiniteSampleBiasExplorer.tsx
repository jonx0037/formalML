// =============================================================================
// FiniteSampleBiasExplorer.tsx
//
// §8 The Vollmer–Zygalakis–Teh (2016) finite-sample bias bound made empirical.
// Three log-log panels on the §6 BLR posterior:
//
//   (a) Bias vs η at fixed B = 64. Slope-+1 reference line shows the
//       O(η) bias contribution from §8.1's Euler–Maruyama discretization.
//   (b) Bias vs B at fixed η = 1e-4. Slope-−1 reference line shows the
//       O(1/B) contribution from §8.1's mini-batch noise injection.
//   (c) The collapse: bias vs (η + ĉ/B) on log axes — both sweeps
//       collapse onto a single line whose slope is the linear fit.
//
// The test statistic is φ(θ) = (θ_1 - μ_post[0])²; under π, E_π[φ] =
// Σ_post[0,0]. Mini-batch noise is simulated synthetically: full gradient
// + N(0, σ_g²/B · I). This decouples η from B in a way the natural N/B-
// rescaled estimator does not (cf. notebook cell 16).
//
// Interactive controls:
//   - Chain length T (3000..15000, default 6000) — longer chains tighten
//     the bias estimate at proportional runtime cost.
//   - σ_g slider (1.0..10.0, default 5.0) — synthetic gradient-noise scale.
//     Larger σ_g makes the §8 1/B contribution dominant; smaller σ_g
//     makes the §8 η contribution dominant.
//   - Resample button — re-rolls all chain seeds.
//
// Computation: in-browser via shared/sgmcmc.ts. 14 chains per run (7
// η-sweep + 7 B-sweep). At T = 6000 ≈ 2 s on commodity laptops.
// Static fallback: /images/topics/stochastic-gradient-mcmc/08_vzt_bias.png.
// =============================================================================

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import { mulberry32 } from './shared/bayesian-ml';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  blrGradSyntheticNoise,
  blrPosterior,
  makeBLRDataset,
  paletteSGMCMC,
  sgldChain,
} from './shared/sgmcmc';

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

const DATA_SEED = 41;
// η range stays below the Euler–Maruyama stability threshold for the BLR
// Hessian (max eigenvalue ≈ 530 ⇒ η_max ≈ 3.7e-3). The notebook uses 5e-3
// as well but with 30,000-step chains; in-browser at 6,000 steps the
// occasional divergence inflates the bias estimate, so we cap at 2e-3.
const ETA_GRID = [5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3] as const;
const B_GRID = [4, 8, 16, 32, 64, 128, 256] as const;
const B_FIXED = 64;
const ETA_FIXED = 1e-4;
const DEFAULT_T = 10000;
const DEFAULT_BURN_FRAC = 0.2;
const DEFAULT_SIGMA_G = 5.0;

const FIGURE_PATH = '/images/topics/stochastic-gradient-mcmc/08_vzt_bias.png';
const ALT =
  'Three-panel log-log figure on the Vollmer–Zygalakis–Teh bias bound. (a) Empirical bias vs step size η at fixed B = 64 — slope matches +1 (O(η)). (b) Bias vs batch size B at fixed η = 10⁻⁴ — slope matches −1 (O(1/B)). (c) Combined collapse: bias against η + ĉ/B fits the line bias ≈ Ĉ₁(η + ĉ/B), confirming the additive structure of Theorem 8.1.';
const CAPTION =
  'Figure 8. The Vollmer–Zygalakis–Teh (2016) bound made empirical. (a) Slope +1 in η, (b) slope −1 in B, (c) bias ≈ Ĉ₁ · (η + ĉ/B) for an empirically-fit ratio ĉ. Slide σ_g to shift the η/B trade-off; longer T tightens the estimate.';
const ARIA = 'Figure 8: Interactive VZT finite-sample bias verification';

// -----------------------------------------------------------------------------
// Computation
// -----------------------------------------------------------------------------

interface CachedDataset {
  spec: ReturnType<typeof makeBLRDataset>;
  posterior: ReturnType<typeof blrPosterior>;
  phiTrue: number;
  thetaInit: Float32Array;
}

interface BiasGrid {
  biasEta: Float32Array; // length |ETA_GRID|, at B = B_FIXED
  biasB: Float32Array; // length |B_GRID|, at η = ETA_FIXED
  T: number;
  burn: number;
  sigmaG: number;
  chainSeed: number;
  C1Hat: number; // bias ≈ C1 · (η + c/B)
  cRatio: number; // ĉ = C2 / C1
}

function makeDataset(): CachedDataset {
  const spec = makeBLRDataset(DATA_SEED);
  const posterior = blrPosterior(spec);
  const phiTrue = posterior.sigmaPost[0]; // Σ_post[0, 0]
  const thetaInit = new Float32Array([posterior.muPost[0], posterior.muPost[1]]);
  return { spec, posterior, phiTrue, thetaInit };
}

function biasOfChain(chain: Float32Array[], burn: number, mu0: number, phiTrue: number): number {
  let sum = 0;
  let count = 0;
  for (let n = burn; n < chain.length; n++) {
    const v = chain[n][0];
    // Skip diverged samples — the EM scheme can blow up near the stability
    // threshold and contaminate the bias estimate with NaN/Infinity.
    if (!Number.isFinite(v)) continue;
    const d = v - mu0;
    sum += d * d;
    count++;
  }
  if (count === 0) return NaN;
  return Math.abs(sum / count - phiTrue);
}

const yieldToBrowser = (): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, 0));

async function runBiasSweep(
  ds: CachedDataset,
  T: number,
  burnFrac: number,
  sigmaG: number,
  chainSeed: number,
): Promise<BiasGrid> {
  const burn = Math.floor(T * burnFrac);
  const biasEta = new Float32Array(ETA_GRID.length);
  const biasB = new Float32Array(B_GRID.length);

  // η sweep at fixed B = 64.
  for (let i = 0; i < ETA_GRID.length; i++) {
    await yieldToBrowser();
    const rng = mulberry32(chainSeed * 977 + 200 + i);
    const grad = blrGradSyntheticNoise(ds.spec, sigmaG, B_FIXED, rng);
    const chain = sgldChain(grad, ds.thetaInit, T, ETA_GRID[i], rng);
    biasEta[i] = biasOfChain(chain, burn, ds.posterior.muPost[0], ds.phiTrue);
  }

  // B sweep at fixed η = 1e-4.
  for (let i = 0; i < B_GRID.length; i++) {
    await yieldToBrowser();
    const rng = mulberry32(chainSeed * 977 + 300 + i);
    const grad = blrGradSyntheticNoise(ds.spec, sigmaG, B_GRID[i], rng);
    const chain = sgldChain(grad, ds.thetaInit, T, ETA_FIXED, rng);
    biasB[i] = biasOfChain(chain, burn, ds.posterior.muPost[0], ds.phiTrue);
  }

  // Joint linear fit: bias ≈ C1 · η + C2 · (1/B). Least squares on stacked rows
  // (skip any rows where the chain diverged and bias is NaN).
  let AtA00 = 0,
    AtA01 = 0,
    AtA11 = 0,
    Aty0 = 0,
    Aty1 = 0;
  const accumulate = (a0: number, a1: number, yv: number) => {
    if (!Number.isFinite(yv)) return;
    AtA00 += a0 * a0;
    AtA01 += a0 * a1;
    AtA11 += a1 * a1;
    Aty0 += a0 * yv;
    Aty1 += a1 * yv;
  };
  for (let i = 0; i < ETA_GRID.length; i++) accumulate(ETA_GRID[i], 1 / B_FIXED, biasEta[i]);
  for (let i = 0; i < B_GRID.length; i++) accumulate(ETA_FIXED, 1 / B_GRID[i], biasB[i]);
  const det = AtA00 * AtA11 - AtA01 * AtA01;
  let C1Hat = NaN,
    cRatio = NaN;
  if (Number.isFinite(det) && Math.abs(det) > 1e-30) {
    C1Hat = (AtA11 * Aty0 - AtA01 * Aty1) / det;
    const C2Hat = (-AtA01 * Aty0 + AtA00 * Aty1) / det;
    cRatio = C2Hat / Math.max(C1Hat, 1e-12);
  }

  return { biasEta, biasB, T, burn, sigmaG, chainSeed, C1Hat, cRatio };
}

// -----------------------------------------------------------------------------
// State
// -----------------------------------------------------------------------------

type CompState =
  | { kind: 'idle' }
  | { kind: 'sampling'; ds: CachedDataset; reason: 'first' | 'param' | 'reseed' }
  | { kind: 'ready'; ds: CachedDataset; grid: BiasGrid };

// -----------------------------------------------------------------------------
// Component
// -----------------------------------------------------------------------------

export default function FiniteSampleBiasExplorer() {
  const { ref, width } = useResizeObserver<HTMLElement>();
  const [state, setState] = useState<CompState>({ kind: 'idle' });
  const [T, setT] = useState(DEFAULT_T);
  const [sigmaG, setSigmaG] = useState(DEFAULT_SIGMA_G);
  const [chainSeed, setChainSeed] = useState(11);
  const ticketRef = useRef(0);

  useEffect(() => {
    let alive = true;
    const ds = makeDataset();
    setState({ kind: 'sampling', ds, reason: 'first' });
    const ticket = ++ticketRef.current;
    runBiasSweep(ds, DEFAULT_T, DEFAULT_BURN_FRAC, DEFAULT_SIGMA_G, 11).then((grid) => {
      if (alive && ticket === ticketRef.current) {
        setState({ kind: 'ready', ds, grid });
      }
    });
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const launch = useCallback(
    (ds: CachedDataset, t: number, sg: number, seed: number, reason: 'param' | 'reseed') => {
      const ticket = ++ticketRef.current;
      setState({ kind: 'sampling', ds, reason });
      runBiasSweep(ds, t, DEFAULT_BURN_FRAC, sg, seed).then((grid) => {
        if (ticket === ticketRef.current) {
          setState({ kind: 'ready', ds, grid });
        }
      });
    },
    [],
  );

  const handleTCommit = (next: number) => {
    if (next === T) return;
    setT(next);
    if (state.kind === 'ready') launch(state.ds, next, sigmaG, chainSeed, 'param');
  };
  const handleSigmaGCommit = (next: number) => {
    if (Math.abs(next - sigmaG) < 1e-3) return;
    setSigmaG(next);
    if (state.kind === 'ready') launch(state.ds, T, next, chainSeed, 'param');
  };
  const handleReseed = () => {
    const next = chainSeed + 1;
    setChainSeed(next);
    if (state.kind === 'ready') launch(state.ds, T, sigmaG, next, 'reseed');
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
                ? 'Running 13 SGLD chains across η × B grid (~5 s)…'
                : state.kind === 'sampling'
                  ? 'Re-running η × B sweep…'
                  : 'Loading…'
            }
          />
        ) : (
          <>
            <Controls
              T={T}
              sigmaG={sigmaG}
              onTCommit={handleTCommit}
              onSigmaGCommit={handleSigmaGCommit}
              onReseed={handleReseed}
              isBlocked={isBlocked}
            />
            <Panels grid={ready.grid} width={width} />
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
  T,
  sigmaG,
  onTCommit,
  onSigmaGCommit,
  onReseed,
  isBlocked,
}: {
  T: number;
  sigmaG: number;
  onTCommit: (next: number) => void;
  onSigmaGCommit: (next: number) => void;
  onReseed: () => void;
  isBlocked: boolean;
}) {
  const [localT, setLocalT] = useState(T);
  const [localSigma, setLocalSigma] = useState(sigmaG);
  useEffect(() => setLocalT(T), [T]);
  useEffect(() => setLocalSigma(sigmaG), [sigmaG]);
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
        <span>T:</span>
        <input
          type="range"
          min={3000}
          max={15000}
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
      <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}>
        <span>σ_g:</span>
        <input
          type="range"
          min={1.0}
          max={10.0}
          step={0.5}
          value={localSigma}
          onChange={(e) => setLocalSigma(Number(e.target.value))}
          onMouseUp={(e) => onSigmaGCommit(Number((e.target as HTMLInputElement).value))}
          onTouchEnd={(e) => onSigmaGCommit(Number((e.target as HTMLInputElement).value))}
          onKeyUp={(e) => onSigmaGCommit(Number((e.target as HTMLInputElement).value))}
          style={{ width: '120px' }}
          disabled={isBlocked}
          aria-label="Synthetic gradient noise scale sigma_g"
        />
        <span style={{ fontVariantNumeric: 'tabular-nums', minWidth: '2.5em' }}>
          {localSigma.toFixed(1)}
        </span>
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

function Panels({ grid, width }: { grid: BiasGrid; width: number }) {
  const layoutWidth = Math.max(width, 320);
  const stack = layoutWidth < 720;
  const panelW = stack ? layoutWidth : (layoutWidth - 32) / 3;
  const panelH = 240;

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: stack ? 'column' : 'row',
        gap: stack ? '12px' : '16px',
        alignItems: stack ? 'center' : 'flex-start',
      }}
    >
      <Panel title={`(a) bias vs η (B = ${B_FIXED})`} width={panelW} height={panelH}>
        <BiasVsParam
          xs={ETA_GRID as unknown as number[]}
          ys={Array.from(grid.biasEta)}
          slope={1}
          width={panelW}
          height={panelH}
          xLabel="step size η (log)"
          referenceLabel="slope +1 (O(η))"
        />
      </Panel>
      <Panel title={`(b) bias vs B (η = ${ETA_FIXED.toExponential(0)})`} width={panelW} height={panelH}>
        <BiasVsParam
          xs={B_GRID as unknown as number[]}
          ys={Array.from(grid.biasB)}
          slope={-1}
          width={panelW}
          height={panelH}
          xLabel="batch size B (log)"
          referenceLabel="slope −1 (O(1/B))"
        />
      </Panel>
      <Panel title={`(c) collapse: bias vs (η + ĉ/B), ĉ ≈ ${grid.cRatio.toExponential(1)}`} width={panelW} height={panelH}>
        <CollapsePanel grid={grid} width={panelW} height={panelH} />
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
// Bias-vs-parameter log-log scatter
// -----------------------------------------------------------------------------

function BiasVsParam({
  xs,
  ys,
  slope,
  width,
  height,
  xLabel,
  referenceLabel,
}: {
  xs: number[];
  ys: number[];
  slope: number;
  width: number;
  height: number;
  xLabel: string;
  referenceLabel: string;
}) {
  const margin = { top: 8, right: 12, bottom: 28, left: 44 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  const positiveYs = ys.filter((y) => y > 0);
  const yMin = positiveYs.length > 0 ? Math.min(...positiveYs) * 0.7 : 1e-5;
  const yMax = Math.max(...ys, 1e-3) * 1.5;
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);

  const xS = d3.scaleLog().domain([xMin * 0.8, xMax * 1.2]).range([0, innerW]);
  const yS = d3.scaleLog().domain([yMin, yMax]).range([innerH, 0]);

  // Reference line: anchor at the median index of the empirical curve.
  const med = Math.floor(xs.length / 2);
  const xRef = xs[med];
  const yRef = ys[med] > 0 ? ys[med] : yMax / 10;
  const refX = xS.domain();
  const refY = refX.map((x) => yRef * Math.pow(x / xRef, slope));

  const fmtPow = (v: number) => `10^${Math.round(Math.log10(v))}`;

  return (
    <g transform={`translate(${margin.left},${margin.top})`}>
      <rect x={0} y={0} width={innerW} height={innerH} fill="var(--color-surface, #fafafa)" />
      {/* x grid (log decades) */}
      {decadeTicks(xMin * 0.8, xMax * 1.2).map((t) => (
        <g key={`xg-${t}`}>
          <line x1={xS(t)} x2={xS(t)} y1={0} y2={innerH} stroke="var(--color-text-muted, #888)" strokeOpacity={0.15} />
          <text x={xS(t)} y={innerH + 16} textAnchor="middle" fontSize={10} fill="var(--color-text, #333)">
            {fmtPow(t)}
          </text>
        </g>
      ))}
      {decadeTicks(yMin, yMax).map((t) => (
        <g key={`yg-${t}`}>
          <line x1={0} x2={innerW} y1={yS(t)} y2={yS(t)} stroke="var(--color-text-muted, #888)" strokeOpacity={0.15} />
          <text x={-6} y={yS(t) + 3} textAnchor="end" fontSize={10} fill="var(--color-text, #333)">
            {fmtPow(t)}
          </text>
        </g>
      ))}
      {/* Reference slope line */}
      <line
        x1={xS(refX[0])}
        x2={xS(refX[1])}
        y1={yS(refY[0])}
        y2={yS(refY[1])}
        stroke={paletteSGMCMC.biasBound}
        strokeWidth={1.4}
        strokeDasharray="4 3"
      />
      {/* Empirical points + connecting line */}
      <path
        d={d3
          .line<[number, number]>()
          .x((d) => xS(d[0]))
          .y((d) => yS(Math.max(d[1], yMin)))(xs.map((x, i) => [x, ys[i]] as [number, number])) ?? ''}
        fill="none"
        stroke={paletteSGMCMC.sgld}
        strokeWidth={1.6}
      />
      {xs.map((x, i) => (
        <circle key={i} cx={xS(x)} cy={yS(Math.max(ys[i], yMin))} r={4} fill={paletteSGMCMC.sgld} />
      ))}
      {/* Axes */}
      <line x1={0} x2={innerW} y1={innerH} y2={innerH} stroke="var(--color-text-muted, #888)" />
      <line x1={0} x2={0} y1={0} y2={innerH} stroke="var(--color-text-muted, #888)" />
      <text x={innerW / 2} y={innerH + 24} textAnchor="middle" fontSize={10} fill="var(--color-text, #333)">
        {xLabel}
      </text>
      <text transform={`translate(${-34}, ${innerH / 2}) rotate(-90)`} textAnchor="middle" fontSize={10} fill="var(--color-text, #333)">
        |E[φ] − π(φ)| (log)
      </text>
      <text x={innerW - 6} y={14} textAnchor="end" fontSize={10} fill={paletteSGMCMC.biasBound}>
        {referenceLabel}
      </text>
    </g>
  );
}

// -----------------------------------------------------------------------------
// Collapse panel
// -----------------------------------------------------------------------------

function CollapsePanel({ grid, width, height }: { grid: BiasGrid; width: number; height: number }) {
  const margin = { top: 8, right: 12, bottom: 28, left: 44 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  const cRatio = grid.cRatio;
  const C1 = grid.C1Hat;

  // Combined x: η + cRatio/B for all 14 samples (η-sweep at B=64; B-sweep at η=ETA_FIXED).
  const etaPoints: { x: number; y: number }[] = ETA_GRID.map((eta, i) => ({
    x: eta + cRatio / B_FIXED,
    y: grid.biasEta[i],
  }));
  const bPoints: { x: number; y: number }[] = B_GRID.map((B, i) => ({
    x: ETA_FIXED + cRatio / B,
    y: grid.biasB[i],
  }));

  const allX = [...etaPoints, ...bPoints].map((p) => p.x);
  const allY = [...etaPoints, ...bPoints].map((p) => p.y).filter((y) => y > 0);
  const xMin = Math.max(Math.min(...allX) * 0.7, 1e-7);
  const xMax = Math.max(...allX) * 1.3;
  const yMin = (allY.length > 0 ? Math.min(...allY) : 1e-5) * 0.7;
  const yMax = Math.max(...allY, 1e-3) * 1.5;

  const xS = d3.scaleLog().domain([xMin, xMax]).range([0, innerW]);
  const yS = d3.scaleLog().domain([yMin, yMax]).range([innerH, 0]);

  const fitX = [xMin, xMax];
  const fitY = fitX.map((x) => Math.max(C1 * x, yMin));

  const fmtPow = (v: number) => `10^${Math.round(Math.log10(v))}`;

  return (
    <g transform={`translate(${margin.left},${margin.top})`}>
      <rect x={0} y={0} width={innerW} height={innerH} fill="var(--color-surface, #fafafa)" />
      {decadeTicks(xMin, xMax).map((t) => (
        <g key={`xg-${t}`}>
          <line x1={xS(t)} x2={xS(t)} y1={0} y2={innerH} stroke="var(--color-text-muted, #888)" strokeOpacity={0.15} />
          <text x={xS(t)} y={innerH + 16} textAnchor="middle" fontSize={10} fill="var(--color-text, #333)">
            {fmtPow(t)}
          </text>
        </g>
      ))}
      {decadeTicks(yMin, yMax).map((t) => (
        <g key={`yg-${t}`}>
          <line x1={0} x2={innerW} y1={yS(t)} y2={yS(t)} stroke="var(--color-text-muted, #888)" strokeOpacity={0.15} />
          <text x={-6} y={yS(t) + 3} textAnchor="end" fontSize={10} fill="var(--color-text, #333)">
            {fmtPow(t)}
          </text>
        </g>
      ))}
      {/* Linear fit through origin in (η + ĉ/B, bias) on log axes ⇒ a slope-+1 line. */}
      <line
        x1={xS(fitX[0])}
        x2={xS(fitX[1])}
        y1={yS(fitY[0])}
        y2={yS(fitY[1])}
        stroke={paletteSGMCMC.biasBound}
        strokeWidth={1.4}
        strokeDasharray="4 3"
      />
      {/* η sweep — circles in SGLD purple */}
      {etaPoints.map((p, i) => (
        <circle
          key={`e-${i}`}
          cx={xS(p.x)}
          cy={yS(Math.max(p.y, yMin))}
          r={4}
          fill={paletteSGMCMC.sgld}
        />
      ))}
      {/* B sweep — squares in SGHMC teal */}
      {bPoints.map((p, i) => (
        <rect
          key={`b-${i}`}
          x={xS(p.x) - 3.5}
          y={yS(Math.max(p.y, yMin)) - 3.5}
          width={7}
          height={7}
          fill={paletteSGMCMC.sghmc}
        />
      ))}
      <line x1={0} x2={innerW} y1={innerH} y2={innerH} stroke="var(--color-text-muted, #888)" />
      <line x1={0} x2={0} y1={0} y2={innerH} stroke="var(--color-text-muted, #888)" />
      <text x={innerW / 2} y={innerH + 24} textAnchor="middle" fontSize={10} fill="var(--color-text, #333)">
        η + ĉ/B (log)
      </text>
      <text transform={`translate(${-34}, ${innerH / 2}) rotate(-90)`} textAnchor="middle" fontSize={10} fill="var(--color-text, #333)">
        |E[φ] − π(φ)| (log)
      </text>
      {/* Legend */}
      <g transform={`translate(${innerW - 90}, 10)`}>
        <rect x={0} y={-6} width={86} height={42} fill="var(--color-surface, #fff)" stroke="var(--color-border, #ccc)" />
        <circle cx={10} cy={4} r={4} fill={paletteSGMCMC.sgld} />
        <text x={20} y={7} fontSize={10} fill="var(--color-text, #333)">η sweep</text>
        <rect x={6} y={18} width={7} height={7} fill={paletteSGMCMC.sghmc} />
        <text x={20} y={25} fontSize={10} fill="var(--color-text, #333)">B sweep</text>
      </g>
    </g>
  );
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

function decadeTicks(lo: number, hi: number): number[] {
  const lo10 = Math.floor(Math.log10(lo));
  const hi10 = Math.ceil(Math.log10(hi));
  const out: number[] = [];
  for (let e = lo10; e <= hi10; e++) {
    const v = Math.pow(10, e);
    if (v >= lo && v <= hi) out.push(v);
  }
  return out;
}
