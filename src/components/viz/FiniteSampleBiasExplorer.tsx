// =============================================================================
// FiniteSampleBiasExplorer.tsx
//
// §8.5 The Vollmer–Zygalakis–Teh (2016) finite-sample bias made empirical.
// SGLD on the §6 BLR posterior with synthetic gradient noise N(0, σ_g²/B · I),
// which decouples η from B. On this Gaussian target the stationary bias is
// known exactly (eq. 8.3, `sgldStationaryBiasExact`), so every panel compares
// measurements against that prediction:
//
//   (a) Bias vs η at fixed B = 64, with the exact curve (slope → +1 as η → 0,
//       steeper near the stability limit 2/λ_max).
//   (b) Bias vs B at fixed η = 1e-4, with the exact curve and its B → ∞
//       discretization floor. The 1/B term carries a factor η, so the slope
//       approaches −1 only where σ_g²/(Bλ) dominates.
//   (c) Measured vs exact bias for all 13 grid points (line: measured = exact).
//
// Test statistic φ(θ) = (θ_1 − μ_post[0])²; π(φ) = Σ_post[0,0]. Each grid
// point averages N_REP independent chains started from exact posterior draws;
// error bars are ±2 standard errors across chains. Mirrors notebook cell 16
// (4,000 chains × 20,000 steps there; fewer here, so wider error bars).
//
// Interactive controls:
//   - Chain length T (3000..15000, default 10000).
//   - σ_g slider (0..400, default 225 — the §6 mini-batch estimator's noise
//     level; σ_g = 0 leaves only the discretization bias, flat in B).
//   - Resample button — re-rolls all chain seeds.
//
// Computation: in-browser via shared/sgmcmc.ts (running sums, no chain
// storage); ≈ 2–3 s at the defaults.
// Static fallback: /images/topics/stochastic-gradient-mcmc/08_vzt_bias.png.
// =============================================================================

import { useCallback, useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { mulberry32 } from './shared/bayesian-ml';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  blrPosterior,
  makeBLRDataset,
  paletteSGMCMC,
  sgldStationaryBiasExact,
  sgldSyntheticNoisePhiMeans,
} from './shared/sgmcmc';

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

const DATA_SEED = 41;
// η range stays below the Euler–Maruyama stability threshold for the BLR
// Hessian (max eigenvalue ≈ 570 ⇒ 2/λ_max ≈ 3.5e-3); same grid as the notebook.
const ETA_GRID = [5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3] as const;
const B_GRID = [4, 8, 16, 32, 64, 128, 256] as const;
const B_FIXED = 64;
const ETA_FIXED = 1e-4;
const N_REP = 400;
const DEFAULT_T = 10000;
const DEFAULT_BURN_FRAC = 0.2;
const DEFAULT_SIGMA_G = 225;

const FIGURE_PATH = '/images/topics/stochastic-gradient-mcmc/08_vzt_bias.png';
const ALT =
  'Three-panel log-log figure comparing measured SGLD bias with the exact stationary bias on the Gaussian §6 posterior (σ_g = 225, 4,000 chains of 20,000 steps per point). (a) Bias vs step size η at B = 64: fitted slope +1.12 ± 0.01, matching the exact +1.12; the curve steepens above slope 1 near the stability limit. (b) Bias vs batch size B at η = 10⁻⁴: slope −0.77 ± 0.01 against the exact −0.76, flattening toward the discretization floor at large B. (c) Measured against exact bias for all 13 grid points, all on the line measured = exact.';
const CAPTION =
  'Figure 8. SGLD bias on the §6 posterior with synthetic gradient noise σ_g²/B · I, measured over independent chains (±2 SE) against the exact stationary bias (8.3). (a) Linear in η until the stability limit bends it upward. (b) A 1/B term (carrying a factor η) above a discretization floor; the slope nears −1 only while the gradient-noise term dominates. (c) Measured vs exact. Slide σ_g to 0 and the B curve collapses onto the floor.';
const ARIA = 'Figure 8: Interactive VZT finite-sample bias verification';

// -----------------------------------------------------------------------------
// Computation
// -----------------------------------------------------------------------------

interface CachedDataset {
  posterior: ReturnType<typeof blrPosterior>;
  phiTrue: number;
}

interface BiasGrid {
  biasEta: Float64Array; // length |ETA_GRID|, at B = B_FIXED
  seEta: Float64Array;
  biasB: Float64Array; // length |B_GRID|, at η = ETA_FIXED
  seB: Float64Array;
  T: number;
  sigmaG: number;
  chainSeed: number;
}

function makeDataset(): CachedDataset {
  const spec = makeBLRDataset(DATA_SEED);
  const posterior = blrPosterior(spec);
  return { posterior, phiTrue: posterior.sigmaPost[0] };
}

function meanAndSe(xs: Float64Array): [number, number] {
  const n = xs.length;
  let m = 0;
  for (let i = 0; i < n; i++) m += xs[i];
  m /= n;
  let v = 0;
  for (let i = 0; i < n; i++) v += (xs[i] - m) * (xs[i] - m);
  v /= n - 1;
  return [m, Math.sqrt(v / n)];
}

/** Least-squares slope of log y on log x; NaN unless every y is positive. */
function logLogSlope(xs: readonly number[], ys: ArrayLike<number>): number {
  const n = xs.length;
  let sx = 0,
    sy = 0,
    sxx = 0,
    sxy = 0;
  for (let i = 0; i < n; i++) {
    if (!(ys[i] > 0)) return NaN;
    const lx = Math.log(xs[i]);
    const ly = Math.log(ys[i]);
    sx += lx;
    sy += ly;
    sxx += lx * lx;
    sxy += lx * ly;
  }
  return (n * sxy - sx * sy) / (n * sxx - sx * sx);
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
  const biasEta = new Float64Array(ETA_GRID.length);
  const seEta = new Float64Array(ETA_GRID.length);
  const biasB = new Float64Array(B_GRID.length);
  const seB = new Float64Array(B_GRID.length);

  for (let i = 0; i < ETA_GRID.length; i++) {
    await yieldToBrowser();
    const rng = mulberry32(chainSeed * 977 + 200 + i);
    const m = sgldSyntheticNoisePhiMeans(ds.posterior, ETA_GRID[i], B_FIXED, sigmaG, N_REP, T, burn, rng);
    const [mean, se] = meanAndSe(m);
    biasEta[i] = mean - ds.phiTrue;
    seEta[i] = se;
  }
  for (let i = 0; i < B_GRID.length; i++) {
    await yieldToBrowser();
    const rng = mulberry32(chainSeed * 977 + 300 + i);
    const m = sgldSyntheticNoisePhiMeans(ds.posterior, ETA_FIXED, B_GRID[i], sigmaG, N_REP, T, burn, rng);
    const [mean, se] = meanAndSe(m);
    biasB[i] = mean - ds.phiTrue;
    seB[i] = se;
  }
  return { biasEta, seEta, biasB, seB, T, sigmaG, chainSeed };
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
                ? `Running ${N_REP} SGLD chains at each of 13 (η, B) points (~3 s)…`
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
            <Panels grid={ready.grid} ds={ready.ds} width={width} />
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
          min={0}
          max={400}
          step={25}
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
          {localSigma.toFixed(0)}
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

function Panels({ grid, ds, width }: { grid: BiasGrid; ds: CachedDataset; width: number }) {
  const layoutWidth = Math.max(width, 320);
  const stack = layoutWidth < 720;
  const panelW = stack ? layoutWidth : (layoutWidth - 32) / 3;
  const panelH = 240;

  const exactAt = (eta: number, B: number) =>
    sgldStationaryBiasExact(ds.posterior, eta, B, grid.sigmaG);
  const exactEta = ETA_GRID.map((eta) => exactAt(eta, B_FIXED));
  const exactB = B_GRID.map((B) => exactAt(ETA_FIXED, B));
  // A measured slope is reported only when every point clears 2 SE; otherwise
  // the smallest biases are within noise of zero and the log-log fit is noise.
  const resolved = (b: Float64Array, se: Float64Array) => b.every((v, i) => v > 2 * se[i]);
  const slopeEta = resolved(grid.biasEta, grid.seEta) ? logLogSlope(ETA_GRID, grid.biasEta) : NaN;
  const slopeB = resolved(grid.biasB, grid.seB) ? logLogSlope(B_GRID, grid.biasB) : NaN;
  const fmtSlope = (v: number) =>
    Number.isFinite(v) ? `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(2)}` : 'unresolved (a point is within 2 SE of 0)';

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: stack ? 'column' : 'row',
        gap: stack ? '12px' : '16px',
        alignItems: stack ? 'center' : 'flex-start',
      }}
    >
      <Panel
        title={`(a) bias vs η (B = ${B_FIXED}): slope ${fmtSlope(slopeEta)} (exact ${fmtSlope(logLogSlope(ETA_GRID, exactEta))})`}
        width={panelW}
        height={panelH}
      >
        <BiasVsParam
          xs={ETA_GRID as unknown as number[]}
          ys={Array.from(grid.biasEta)}
          ses={Array.from(grid.seEta)}
          exact={(eta) => exactAt(eta, B_FIXED)}
          width={panelW}
          height={panelH}
          xLabel="step size η (log)"
          legendBottom
        />
      </Panel>
      <Panel
        title={`(b) bias vs B (η = ${ETA_FIXED.toExponential(0)}): slope ${fmtSlope(slopeB)} (exact ${fmtSlope(logLogSlope(B_GRID, exactB))})`}
        width={panelW}
        height={panelH}
      >
        <BiasVsParam
          xs={B_GRID as unknown as number[]}
          ys={Array.from(grid.biasB)}
          ses={Array.from(grid.seB)}
          exact={(B) => exactAt(ETA_FIXED, B)}
          floor={exactAt(ETA_FIXED, Infinity)}
          width={panelW}
          height={panelH}
          xLabel="batch size B (log)"
        />
      </Panel>
      <Panel title="(c) measured vs exact bias, all 13 points" width={panelW} height={panelH}>
        <ParityPanel
          etaPts={exactEta.map((x, i) => ({ x, y: grid.biasEta[i], se: grid.seEta[i] }))}
          bPts={exactB.map((x, i) => ({ x, y: grid.biasB[i], se: grid.seB[i] }))}
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
          minHeight: '2.4em',
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

const MARGIN = { top: 8, right: 12, bottom: 28, left: 44 };
const fmtPow = (v: number) => `10^${Math.round(Math.log10(v))}`;

function Axes({
  xS,
  yS,
  innerW,
  innerH,
  xLabel,
  yLabel,
}: {
  xS: d3.ScaleLogarithmic<number, number>;
  yS: d3.ScaleLogarithmic<number, number>;
  innerW: number;
  innerH: number;
  xLabel: string;
  yLabel: string;
}) {
  const [x0, x1] = xS.domain();
  const [y0, y1] = yS.domain();
  return (
    <>
      <rect x={0} y={0} width={innerW} height={innerH} fill="var(--color-surface, #fafafa)" />
      {decadeTicks(x0, x1).map((t) => (
        <g key={`xg-${t}`}>
          <line x1={xS(t)} x2={xS(t)} y1={0} y2={innerH} stroke="var(--color-text-muted, #888)" strokeOpacity={0.15} />
          <text x={xS(t)} y={innerH + 16} textAnchor="middle" fontSize={10} fill="var(--color-text, #333)">
            {fmtPow(t)}
          </text>
        </g>
      ))}
      {decadeTicks(y0, y1).map((t) => (
        <g key={`yg-${t}`}>
          <line x1={0} x2={innerW} y1={yS(t)} y2={yS(t)} stroke="var(--color-text-muted, #888)" strokeOpacity={0.15} />
          <text x={-6} y={yS(t) + 3} textAnchor="end" fontSize={10} fill="var(--color-text, #333)">
            {fmtPow(t)}
          </text>
        </g>
      ))}
      <line x1={0} x2={innerW} y1={innerH} y2={innerH} stroke="var(--color-text-muted, #888)" />
      <line x1={0} x2={0} y1={0} y2={innerH} stroke="var(--color-text-muted, #888)" />
      <text x={innerW / 2} y={innerH + 26} textAnchor="middle" fontSize={10} fill="var(--color-text, #333)">
        {xLabel}
      </text>
      <text transform={`translate(${-34}, ${innerH / 2}) rotate(-90)`} textAnchor="middle" fontSize={10} fill="var(--color-text, #333)">
        {yLabel}
      </text>
    </>
  );
}

/** Points with ±2 SE bars; non-positive values (possible when σ_g is small and
 *  the bias is within noise of zero) are dropped from the log axes. */
function ErrorPoints({
  pts,
  xS,
  yS,
  color,
  square = false,
}: {
  pts: { x: number; y: number; se: number }[];
  xS: d3.ScaleLogarithmic<number, number>;
  yS: d3.ScaleLogarithmic<number, number>;
  color: string;
  square?: boolean;
}) {
  const [yLo, yHi] = yS.domain();
  return (
    <>
      {pts.map((p, i) => {
        if (!Number.isFinite(p.y)) return null;
        // Points below the axis floor (bias within noise of zero, or negative)
        // are drawn hollow on the bottom edge.
        const clamped = p.y < yLo;
        const yv = Math.min(Math.max(p.y, yLo), yHi);
        const hi = p.y + 2 * p.se;
        return (
          <g key={i}>
            {hi > yLo && (
              <line
                x1={xS(p.x)}
                x2={xS(p.x)}
                y1={yS(Math.min(Math.max(p.y - 2 * p.se, yLo), yHi))}
                y2={yS(Math.min(hi, yHi))}
                stroke={color}
                strokeWidth={1.2}
              />
            )}
            {square ? (
              <rect
                x={xS(p.x) - 3.5}
                y={yS(yv) - 3.5}
                width={7}
                height={7}
                fill={clamped ? 'none' : color}
                stroke={color}
              />
            ) : (
              <circle cx={xS(p.x)} cy={yS(yv)} r={4} fill={clamped ? 'none' : color} stroke={color} />
            )}
          </g>
        );
      })}
    </>
  );
}

function BiasVsParam({
  xs,
  ys,
  ses,
  exact,
  floor,
  width,
  height,
  xLabel,
  legendBottom = false,
}: {
  xs: number[];
  ys: number[];
  ses: number[];
  exact: (x: number) => number;
  floor?: number;
  width: number;
  height: number;
  xLabel: string;
  legendBottom?: boolean;
}) {
  const innerW = width - MARGIN.left - MARGIN.right;
  const innerH = height - MARGIN.top - MARGIN.bottom;
  const xLo = Math.min(...xs) * 0.8;
  const xHi = Math.max(...xs) * 1.2;
  const fine = d3.range(61).map((k) => xLo * Math.pow(xHi / xLo, k / 60));
  const exactFine = fine.map(exact);
  // The y-range follows the exact curve, so a noisy point near zero cannot
  // stretch the log axis; such points are drawn hollow on the bottom edge.
  const ref = [...exactFine, ...(floor ? [floor] : [])];
  const yLo = Math.min(...ref) * 0.3;
  const yHi = Math.max(...ref, ...ys.filter((y) => Number.isFinite(y))) * 1.6;
  const xS = d3.scaleLog().domain([xLo, xHi]).range([0, innerW]);
  const yS = d3.scaleLog().domain([yLo, yHi]).range([innerH, 0]);
  const exactPath =
    d3
      .line<number>()
      .x((x) => xS(x))
      .y((_, k) => yS(exactFine[k]))(fine) ?? '';

  return (
    <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
      <Axes xS={xS} yS={yS} innerW={innerW} innerH={innerH} xLabel={xLabel} yLabel="E[φ] − π(φ) (log)" />
      {floor !== undefined && floor > 0 && (
        <>
          <line x1={0} x2={innerW} y1={yS(floor)} y2={yS(floor)} stroke={paletteSGMCMC.target} strokeWidth={1} strokeDasharray="2 3" />
          <text x={6} y={yS(floor) - 4} fontSize={9.5} fill={paletteSGMCMC.target}>
            B → ∞ floor (discretization)
          </text>
        </>
      )}
      <path d={exactPath} fill="none" stroke={paletteSGMCMC.biasBound} strokeWidth={1.5} strokeDasharray="5 3" />
      <ErrorPoints pts={xs.map((x, i) => ({ x, y: ys[i], se: ses[i] }))} xS={xS} yS={yS} color={paletteSGMCMC.sgld} />
      <text x={innerW - 6} y={legendBottom ? innerH - 22 : 14} textAnchor="end" fontSize={10} fill={paletteSGMCMC.biasBound}>
        exact (8.3)
      </text>
      <text x={innerW - 6} y={legendBottom ? innerH - 9 : 27} textAnchor="end" fontSize={10} fill={paletteSGMCMC.sgld}>
        measured ± 2 SE
      </text>
    </g>
  );
}

function ParityPanel({
  etaPts,
  bPts,
  width,
  height,
}: {
  etaPts: { x: number; y: number; se: number }[];
  bPts: { x: number; y: number; se: number }[];
  width: number;
  height: number;
}) {
  const innerW = width - MARGIN.left - MARGIN.right;
  const innerH = height - MARGIN.top - MARGIN.bottom;
  const all = [...etaPts, ...bPts];
  const lo = Math.min(...all.map((p) => p.x)) * 0.3;
  const hi = Math.max(...all.map((p) => p.x), ...all.map((p) => p.y).filter((y) => Number.isFinite(y))) * 1.6;
  const xS = d3.scaleLog().domain([lo, hi]).range([0, innerW]);
  const yS = d3.scaleLog().domain([lo, hi]).range([innerH, 0]);
  return (
    <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
      <Axes xS={xS} yS={yS} innerW={innerW} innerH={innerH} xLabel="exact bias (8.3) (log)" yLabel="measured bias (log)" />
      <line x1={xS(lo)} x2={xS(hi)} y1={yS(lo)} y2={yS(hi)} stroke={paletteSGMCMC.biasBound} strokeWidth={1.4} strokeDasharray="5 3" />
      <ErrorPoints pts={etaPts} xS={xS} yS={yS} color={paletteSGMCMC.sgld} />
      <ErrorPoints pts={bPts} xS={xS} yS={yS} color={paletteSGMCMC.sghmc} square />
      <g transform="translate(8, 10)">
        <rect x={0} y={-6} width={92} height={42} fill="var(--color-surface, #fff)" stroke="var(--color-border, #ccc)" />
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
