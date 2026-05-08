// =============================================================================
// BrownianPathExplorer.tsx
//
// §3 Itô SDE preliminaries — visualizing standard 1D Brownian motion. The
// canonical building block for every SDE in the topic. Multiple sample paths
// at a chosen step size, with the theoretical ±√t envelope overlaid.
//
// Two panels:
//   (a) K Brownian sample paths W_t on [0, T_MAX] at uniform spacing dt.
//       The ±√t envelope (E[W_t] = 0, std(W_t) = √t) is overlaid as
//       dashed lines. The paths are visibly continuous-but-rough — no
//       differentiable structure.
//   (b) Histogram of W_T over many independent sample paths vs the
//       analytic N(0, T) pdf — empirical confirmation of the marginal.
//
// Controls:
//   - K (number of paths to display, 5..40, default 12)
//   - dt slider (0.001..0.05, default 0.01) — step size for the
//     discretization. Smaller dt → smoother-looking paths; the
//     continuous-time process is the dt → 0 limit.
//   - Resample button.
//
// Computation: in-browser via shared/sgmcmc.ts.brownianPath. K paths × T_MAX/dt
// steps each ≈ instantaneous.
// Static fallback: /images/topics/stochastic-gradient-mcmc/03_brownian.png.
// =============================================================================

import { useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import { mulberry32 } from './shared/bayesian-ml';
import { useResizeObserver } from './shared/useResizeObserver';
import { brownianPath, brownianTerminalValues, histogram, paletteSGMCMC } from './shared/sgmcmc';

const T_MAX = 1.0;
const N_HIST_PATHS = 5000;
const DEFAULT_K = 12;
const DEFAULT_DT = 0.01;

const FIGURE_PATH = '/images/topics/stochastic-gradient-mcmc/03_brownian.png';
const ALT =
  'Two-panel figure of standard 1D Brownian motion. (a) Sample paths against the ±√t envelope. (b) Histogram of W_T at the terminal time vs the analytic N(0, T) density.';
const CAPTION =
  'Figure 3. Standard Brownian motion: the canonical source of continuous-time noise behind every SDE in the topic. Variance grows as t (not t²), so the typical magnitude of W_t is √t — what the dashed envelope tracks. Sample paths are continuous but nowhere differentiable; the SDE machinery in §§4–7 is the workaround for that.';
const ARIA = 'Figure 3: Interactive Brownian motion paths';

interface CachedRun {
  paths: { t: Float32Array; w: Float32Array }[];
  histAtT: ReturnType<typeof histogram>;
  K: number;
  dt: number;
  chainSeed: number;
}

function runPaths(K: number, dt: number, chainSeed: number): CachedRun {
  const paths: { t: Float32Array; w: Float32Array }[] = [];
  for (let k = 0; k < K; k++) {
    const rng = mulberry32(chainSeed * 991 + 31 + k);
    paths.push(brownianPath(T_MAX, dt, rng));
  }
  // Histogram of W_T over many independent paths. Use the specialized
  // terminal-only helper — generating full paths just to keep the last
  // value would allocate 2 · N_HIST_PATHS Float32Arrays per slider commit.
  const rngHist = mulberry32(chainSeed * 991 + 1000);
  const wT = brownianTerminalValues(T_MAX, dt, N_HIST_PATHS, rngHist);
  const histAtT = histogram(wT, -3, 3, 30);
  return { paths, histAtT, K, dt, chainSeed };
}

export default function BrownianPathExplorer() {
  const { ref, width } = useResizeObserver<HTMLElement>();
  const [K, setK] = useState(DEFAULT_K);
  const [dt, setDt] = useState(DEFAULT_DT);
  const [chainSeed, setChainSeed] = useState(11);
  const [run, setRun] = useState<CachedRun | null>(null);
  const idRef = useRef(0);

  useEffect(() => {
    const id = ++idRef.current;
    setRun(null);
    setTimeout(() => {
      if (id !== idRef.current) return;
      setRun(runPaths(K, dt, chainSeed));
    }, 0);
  }, [K, dt, chainSeed]);

  return (
    <figure
      ref={ref as React.RefObject<HTMLElement>}
      aria-label={ARIA}
      style={{ width: '100%', margin: '2rem auto', display: 'flex', flexDirection: 'column', alignItems: 'center' }}
    >
      <div style={{ width: '100%', maxWidth: '880px' }}>
        {!run ? (
          <Fallback label="Computing Brownian paths…" />
        ) : (
          <>
            <Controls K={K} dt={dt} onKCommit={setK} onDtCommit={setDt} onReseed={() => setChainSeed((s) => s + 1)} />
            <Panels run={run} width={width} />
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

function Controls({ K, dt, onKCommit, onDtCommit, onReseed }: { K: number; dt: number; onKCommit: (n: number) => void; onDtCommit: (n: number) => void; onReseed: () => void }) {
  const [lk, setLk] = useState(K);
  const [ld, setLd] = useState(dt);
  useEffect(() => setLk(K), [K]);
  useEffect(() => setLd(dt), [dt]);
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.75rem 1.25rem', alignItems: 'center', fontSize: '0.875rem', color: 'var(--color-text, #333)', marginBottom: '0.75rem' }}>
      <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}>
        <span>K paths:</span>
        <input type="range" min={5} max={40} step={1} value={lk}
          onChange={(e) => setLk(Number(e.target.value))}
          onMouseUp={(e) => onKCommit(Number((e.target as HTMLInputElement).value))}
          onTouchEnd={(e) => onKCommit(Number((e.target as HTMLInputElement).value))}
          onKeyUp={(e) => onKCommit(Number((e.target as HTMLInputElement).value))}
          style={{ width: '120px' }} aria-label="Number of paths K" />
        <span style={{ fontVariantNumeric: 'tabular-nums', minWidth: '2.5em' }}>{lk}</span>
      </label>
      <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}>
        <span>dt:</span>
        <input type="range" min={0.001} max={0.05} step={0.001} value={ld}
          onChange={(e) => setLd(Number(e.target.value))}
          onMouseUp={(e) => onDtCommit(Number((e.target as HTMLInputElement).value))}
          onTouchEnd={(e) => onDtCommit(Number((e.target as HTMLInputElement).value))}
          onKeyUp={(e) => onDtCommit(Number((e.target as HTMLInputElement).value))}
          style={{ width: '120px' }} aria-label="Step size dt" />
        <span style={{ fontVariantNumeric: 'tabular-nums', minWidth: '4em' }}>{ld.toFixed(3)}</span>
      </label>
      <button type="button" onClick={onReseed} style={{ padding: '0.3rem 0.7rem', border: '1px solid var(--color-border, #ccc)', borderRadius: '4px', background: 'var(--color-surface, #fff)', color: 'var(--color-text, #333)', cursor: 'pointer', fontSize: '0.875rem' }}>
        Resample
      </button>
    </div>
  );
}

function Panels({ run, width }: { run: CachedRun; width: number }) {
  const layoutWidth = Math.max(width, 320);
  const stack = layoutWidth < 720;
  const panelW = stack ? layoutWidth : (layoutWidth - 16) / 2;
  const panelH = 260;
  return (
    <div style={{ display: 'flex', flexDirection: stack ? 'column' : 'row', gap: stack ? '12px' : '16px', alignItems: stack ? 'center' : 'flex-start' }}>
      <Panel title="(a) Brownian sample paths with ±√t envelope" width={panelW} height={panelH}>
        <PathsPanel paths={run.paths} width={panelW} height={panelH} />
      </Panel>
      <Panel title={`(b) Histogram of W_T (T = ${T_MAX})`} width={panelW} height={panelH}>
        <HistPanel hist={run.histAtT} width={panelW} height={panelH} />
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

function PathsPanel({ paths, width, height }: { paths: { t: Float32Array; w: Float32Array }[]; width: number; height: number }) {
  const margin = { top: 8, right: 12, bottom: 28, left: 36 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  // Avoid `Math.max(...flatMap(...))` here: with K up to 40 paths and
  // dt down to 0.001 the spread expands to ~40k arguments, which spills
  // through the JS call-stack limit on some engines and allocates a chain
  // of intermediate arrays. The explicit loop is O(K · pathLen) with no
  // intermediate allocations.
  const yAbs = useMemo(() => {
    let mx = 3;
    for (const p of paths) {
      for (let i = 0; i < p.w.length; i++) {
        const a = Math.abs(p.w[i]);
        if (a > mx) mx = a;
      }
    }
    return mx * 1.05;
  }, [paths]);
  const xS = d3.scaleLinear().domain([0, T_MAX]).range([0, innerW]);
  const yS = d3.scaleLinear().domain([-yAbs, yAbs]).range([innerH, 0]);

  const linePathOf = (p: { t: Float32Array; w: Float32Array }) =>
    Array.from(p.w).map((wv, i) => `${i === 0 ? 'M' : 'L'}${xS(p.t[i])},${yS(wv)}`).join(' ');

  const envelopeUpper = Array.from({ length: 80 }, (_, i) => {
    const t = (i / 79) * T_MAX;
    return `${i === 0 ? 'M' : 'L'}${xS(t)},${yS(Math.sqrt(t))}`;
  }).join(' ');
  const envelopeLower = Array.from({ length: 80 }, (_, i) => {
    const t = (i / 79) * T_MAX;
    return `${i === 0 ? 'M' : 'L'}${xS(t)},${yS(-Math.sqrt(t))}`;
  }).join(' ');

  return (
    <g transform={`translate(${margin.left},${margin.top})`}>
      <rect x={0} y={0} width={innerW} height={innerH} fill="var(--color-surface, #fafafa)" />
      <line x1={0} x2={innerW} y1={yS(0)} y2={yS(0)} stroke="var(--color-text-muted, #888)" strokeOpacity={0.3} />
      {paths.map((p, k) => (
        <path key={k} d={linePathOf(p)} fill="none" stroke={paletteSGMCMC.brownian} strokeWidth={0.7} opacity={0.55} />
      ))}
      <path d={envelopeUpper} fill="none" stroke={paletteSGMCMC.target} strokeWidth={1.4} strokeDasharray="4 3" />
      <path d={envelopeLower} fill="none" stroke={paletteSGMCMC.target} strokeWidth={1.4} strokeDasharray="4 3" />
      <line x1={0} x2={innerW} y1={innerH} y2={innerH} stroke="var(--color-text-muted, #888)" />
      <line x1={0} x2={0} y1={0} y2={innerH} stroke="var(--color-text-muted, #888)" />
      <text x={innerW / 2} y={innerH + 18} textAnchor="middle" fontSize={10} fill="var(--color-text, #333)">t</text>
      <text transform={`translate(${-26}, ${innerH / 2}) rotate(-90)`} textAnchor="middle" fontSize={10} fill="var(--color-text, #333)">W_t</text>
      <text x={innerW - 6} y={14} textAnchor="end" fontSize={10} fill={paletteSGMCMC.target}>±√t envelope</text>
    </g>
  );
}

function HistPanel({ hist, width, height }: { hist: ReturnType<typeof histogram>; width: number; height: number }) {
  const margin = { top: 8, right: 12, bottom: 28, left: 36 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const lo = -3, hi = 3;
  const xS = d3.scaleLinear().domain([lo, hi]).range([0, innerW]);
  const truePdf = (x: number) => Math.exp(-(x * x) / (2 * T_MAX)) / Math.sqrt(2 * Math.PI * T_MAX);
  const refPts = Array.from({ length: 200 }, (_, i) => lo + ((hi - lo) * i) / 199);
  const yMax = Math.max(Math.max(...Array.from(hist.density)), Math.max(...refPts.map(truePdf))) * 1.15;
  const yS = d3.scaleLinear().domain([0, yMax]).range([innerH, 0]);
  const refPath = refPts.map((x, i) => `${i === 0 ? 'M' : 'L'}${xS(x)},${yS(truePdf(x))}`).join(' ');
  const binWidth = (hi - lo) / hist.centers.length;

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
          fill={paletteSGMCMC.brownian}
          opacity={0.6}
        />
      ))}
      <path d={refPath} fill="none" stroke={paletteSGMCMC.target} strokeWidth={1.6} strokeDasharray="4 3" />
      <line x1={0} x2={innerW} y1={innerH} y2={innerH} stroke="var(--color-text-muted, #888)" />
      <line x1={0} x2={0} y1={0} y2={innerH} stroke="var(--color-text-muted, #888)" />
      {[-2, -1, 0, 1, 2].map((t) => (
        <text key={t} x={xS(t)} y={innerH + 16} textAnchor="middle" fontSize={10} fill="var(--color-text, #333)">{t}</text>
      ))}
      <text x={innerW / 2} y={innerH + 24} textAnchor="middle" fontSize={10} fill="var(--color-text, #333)">W_T</text>
      <text transform={`translate(${-26}, ${innerH / 2}) rotate(-90)`} textAnchor="middle" fontSize={10} fill="var(--color-text, #333)">density</text>
      <text x={innerW - 6} y={14} textAnchor="end" fontSize={10} fill={paletteSGMCMC.target}>N(0, T)</text>
    </g>
  );
}
