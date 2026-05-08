// =============================================================================
// SGLDvsSGHMCAutocorrelation.tsx
//
// §7 SGHMC's mixing advantage on anisotropic targets, made interactive. The
// running target is N(0, diag(1, κ)) at κ = 100 — the long thin Gaussian
// where overdamped Langevin's mixing time scales as O(κ) and underdamped
// Langevin's as O(√κ).
//
// Three panels:
//   (a) Trajectories on the κ = 100 Gaussian over N_TRAJ steps for SGLD and
//       SGHMC starting at (0, 8). SGHMC carries momentum and ballistically
//       traverses the long axis; SGLD random-walks step-by-step.
//   (b) Single-coordinate (θ_2) autocorrelation functions for both chains
//       on a long run (N_LONG steps post-burn). The faster ACF decay of
//       SGHMC at well-tuned C is the §7.6 mixing advantage made empirical.
//   (c) IAT(θ_2) vs friction C for SGHMC — the U-shape predicted by §7.6.
//       Too low C → near-Hamiltonian, weak ergodicity. Too high C →
//       approaching overdamped Langevin, slow mixing. Sweet spot at C ≈ √M.
//
// Interactive controls:
//   - Friction C slider (selects from the §7 C_grid). The chosen C drives
//     panel (a)'s SGHMC trajectory and panel (b)'s SGHMC ACF; panel (c)
//     highlights the selected C on the precomputed U-curve.
//   - Chain length N_LONG slider (5k..20k). Affects ACF stability and IAT
//     estimate precision.
//   - Resample button — re-rolls all chain seeds.
//
// Computation: in-browser via shared/sgmcmc.ts. 8 SGHMC chains for the C
// sweep (panel c) + 1 SGLD long chain + 2 trajectory chains. At default
// N_LONG = 8000 ≈ 2 s.
// Static fallback: /images/topics/stochastic-gradient-mcmc/07_sghmc.png.
// =============================================================================

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import { mulberry32 } from './shared/bayesian-ml';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  anisotropic2DGradAndLogDensity,
  autocorr,
  extractCoord,
  integratedAutocorrTime,
  paletteSGMCMC,
  sghmcChain,
  sgldChain,
} from './shared/sgmcmc';

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

const KAPPA = 100;
const C_GRID = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0] as const;
const DEFAULT_C_INDEX = 3; // C = 0.5 (the U-curve sweet spot)
const N_TRAJ = 1500;
const DEFAULT_N_LONG = 8000;
const TRAJ_BURN = 0;
const LONG_BURN_FRAC = 0.2;
const ACF_MAX_LAG = 300;
const ETA_SGLD = 0.05;
const ETA_SGHMC = 0.1;
const TRAJ_START: [number, number] = [0.0, 8.0];
const LONG_START: [number, number] = [0.0, 0.0];

const FIGURE_PATH = '/images/topics/stochastic-gradient-mcmc/07_sghmc.png';
const ALT =
  'Three-panel figure on the anisotropic 2D Gaussian N(0, diag(1, 100)). (a) Trajectories: SGLD random-walks slowly along the long axis while SGHMC traverses ballistically. (b) Autocorrelation functions: SGHMC decorrelates faster at well-tuned friction C. (c) IAT vs C: U-shape with optimal C around √M.';
const CAPTION =
  'Figure 7. SGHMC mixes faster than SGLD on anisotropic targets. (a) Trajectories on the κ = 100 Gaussian; SGHMC carries momentum through the long axis. (b) The ACF of θ_2 decorrelates an order of magnitude faster for SGHMC at well-tuned C. (c) The friction–IAT U-curve: too little C = non-ergodic, too much = overdamped. Slide C to find the minimum.';
const ARIA = 'Figure 7: Interactive SGLD vs SGHMC mixing comparison';

// -----------------------------------------------------------------------------
// Computation
// -----------------------------------------------------------------------------

interface CachedRun {
  sgldTraj: Float32Array[];
  sghmcTrajPerC: Float32Array[][]; // one for each C in C_GRID
  sgldACF: Float32Array;
  sghmcACFPerC: Float32Array[]; // one for each C
  iatPerC: Float32Array;
  iatSGLD: number;
  cIndex: number;
  nLong: number;
  chainSeed: number;
}

const yieldToBrowser = (): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, 0));

async function runAll(nLong: number, chainSeed: number): Promise<CachedRun> {
  const target = anisotropic2DGradAndLogDensity(KAPPA);
  const burn = Math.floor(nLong * LONG_BURN_FRAC);

  // SGLD long chain (single)
  await yieldToBrowser();
  const rngSGLD = mulberry32(chainSeed * 977 + 71);
  const sgldLong = sgldChain(
    target.gradU,
    new Float32Array(LONG_START),
    nLong,
    ETA_SGLD,
    rngSGLD,
  );
  const sgldTrace = extractCoord(sgldLong.slice(burn), 1);
  const sgldACF = autocorr(sgldTrace, ACF_MAX_LAG);
  const iatSGLD = integratedAutocorrTime(sgldTrace);

  // SGLD trajectory (short, from start = (0, 8))
  await yieldToBrowser();
  const rngSGLDTraj = mulberry32(chainSeed * 977 + 61);
  const sgldTraj = sgldChain(
    target.gradU,
    new Float32Array(TRAJ_START),
    N_TRAJ,
    ETA_SGLD,
    rngSGLDTraj,
  );

  // SGHMC: one trajectory + one long chain per C ∈ C_GRID
  const sghmcTrajPerC: Float32Array[][] = [];
  const sghmcACFPerC: Float32Array[] = [];
  const iatPerC = new Float32Array(C_GRID.length);
  for (let i = 0; i < C_GRID.length; i++) {
    await yieldToBrowser();
    const rngTraj = mulberry32(chainSeed * 977 + 62 + i);
    const traj = sghmcChain(
      target.gradU,
      new Float32Array(TRAJ_START),
      N_TRAJ,
      ETA_SGHMC,
      C_GRID[i],
      rngTraj,
    );
    sghmcTrajPerC.push(traj);

    const rngLong = mulberry32(chainSeed * 977 + 80 + i);
    const long = sghmcChain(
      target.gradU,
      new Float32Array(LONG_START),
      nLong,
      ETA_SGHMC,
      C_GRID[i],
      rngLong,
    );
    const trace = extractCoord(long.slice(burn), 1);
    sghmcACFPerC.push(autocorr(trace, ACF_MAX_LAG));
    iatPerC[i] = integratedAutocorrTime(trace);
  }

  return {
    sgldTraj,
    sghmcTrajPerC,
    sgldACF,
    sghmcACFPerC,
    iatPerC,
    iatSGLD,
    cIndex: 0, // placeholder; consumer overrides
    nLong,
    chainSeed,
  };
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

export default function SGLDvsSGHMCAutocorrelation() {
  const { ref, width } = useResizeObserver<HTMLElement>();
  const [state, setState] = useState<CompState>({ kind: 'idle' });
  const [cIndex, setCIndex] = useState(DEFAULT_C_INDEX);
  const [nLong, setNLong] = useState(DEFAULT_N_LONG);
  const [chainSeed, setChainSeed] = useState(11);
  const ticketRef = useRef(0);

  useEffect(() => {
    let alive = true;
    setState({ kind: 'sampling', reason: 'first' });
    const ticket = ++ticketRef.current;
    runAll(DEFAULT_N_LONG, 11).then((run) => {
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
    (n: number, seed: number, reason: 'param' | 'reseed') => {
      const ticket = ++ticketRef.current;
      setState({ kind: 'sampling', reason });
      runAll(n, seed).then((run) => {
        if (ticket === ticketRef.current) {
          setState({ kind: 'ready', run });
        }
      });
    },
    [],
  );

  const handleNLongCommit = (next: number) => {
    if (next === nLong) return;
    setNLong(next);
    if (state.kind === 'ready') launch(next, chainSeed, 'param');
  };
  const handleReseed = () => {
    const next = chainSeed + 1;
    setChainSeed(next);
    if (state.kind === 'ready') launch(nLong, next, 'reseed');
  };
  // C slider does NOT trigger a chain re-run — all C values are precomputed.

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
                ? 'Running SGLD + 8 SGHMC chains over C grid (~3 s)…'
                : state.kind === 'sampling'
                  ? 'Re-running chain ensemble…'
                  : 'Loading…'
            }
          />
        ) : (
          <>
            <Controls
              cIndex={cIndex}
              nLong={nLong}
              onCIndexChange={setCIndex}
              onNLongCommit={handleNLongCommit}
              onReseed={handleReseed}
              isBlocked={isBlocked}
            />
            <Panels run={ready.run} cIndex={cIndex} width={width} />
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
  cIndex,
  nLong,
  onCIndexChange,
  onNLongCommit,
  onReseed,
  isBlocked,
}: {
  cIndex: number;
  nLong: number;
  onCIndexChange: (next: number) => void;
  onNLongCommit: (next: number) => void;
  onReseed: () => void;
  isBlocked: boolean;
}) {
  const [localN, setLocalN] = useState(nLong);
  useEffect(() => setLocalN(nLong), [nLong]);
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
        <span>SGHMC C:</span>
        <input
          type="range"
          min={0}
          max={C_GRID.length - 1}
          step={1}
          value={cIndex}
          onChange={(e) => onCIndexChange(Number(e.target.value))}
          style={{ width: '160px' }}
          aria-label="SGHMC friction C (selects from precomputed grid)"
        />
        <span style={{ fontVariantNumeric: 'tabular-nums', minWidth: '4em' }}>
          C = {C_GRID[cIndex]}
        </span>
      </label>
      <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}>
        <span>N (long chains):</span>
        <input
          type="range"
          min={5000}
          max={20000}
          step={1000}
          value={localN}
          onChange={(e) => setLocalN(Number(e.target.value))}
          onMouseUp={(e) => onNLongCommit(Number((e.target as HTMLInputElement).value))}
          onTouchEnd={(e) => onNLongCommit(Number((e.target as HTMLInputElement).value))}
          onKeyUp={(e) => onNLongCommit(Number((e.target as HTMLInputElement).value))}
          style={{ width: '120px' }}
          disabled={isBlocked}
          aria-label="Long-chain length N"
        />
        <span style={{ fontVariantNumeric: 'tabular-nums', minWidth: '4em' }}>{localN}</span>
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

function Panels({ run, cIndex, width }: { run: CachedRun; cIndex: number; width: number }) {
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
      <Panel title={`(a) Trajectories on κ = ${KAPPA} Gaussian`} width={panelW} height={panelH}>
        <Trajectories
          sgldTraj={run.sgldTraj}
          sghmcTraj={run.sghmcTrajPerC[cIndex]}
          width={panelW}
          height={panelH}
        />
      </Panel>
      <Panel
        title={`(b) ACF of θ_2  (IAT_SGLD ≈ ${run.iatSGLD.toFixed(0)}, IAT_SGHMC ≈ ${run.iatPerC[cIndex].toFixed(0)})`}
        width={panelW}
        height={panelH}
      >
        <ACFCurves
          sgldACF={run.sgldACF}
          sghmcACF={run.sghmcACFPerC[cIndex]}
          width={panelW}
          height={panelH}
        />
      </Panel>
      <Panel title="(c) IAT vs friction C" width={panelW} height={panelH}>
        <IATCurve iatPerC={run.iatPerC} cIndex={cIndex} width={panelW} height={panelH} />
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
// Trajectories panel
// -----------------------------------------------------------------------------

function Trajectories({
  sgldTraj,
  sghmcTraj,
  width,
  height,
}: {
  sgldTraj: Float32Array[];
  sghmcTraj: Float32Array[];
  width: number;
  height: number;
}) {
  const margin = { top: 8, right: 12, bottom: 28, left: 36 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  const xS = d3.scaleLinear().domain([-4, 4]).range([0, innerW]);
  const yS = d3.scaleLinear().domain([-22, 22]).range([innerH, 0]);

  // Density contours of N(0, diag(1, κ)): m² levels at 2.30, 6.18, 11.83.
  const contours = useMemo(() => {
    const targets = [2.3, 6.18, 11.83];
    const grid = 50;
    const out: string[] = [];
    const xMin = -4,
      xMax = 4,
      yMin = -22,
      yMax = 22;
    const dx = (xMax - xMin) / grid;
    const dy = (yMax - yMin) / grid;
    const mahal = (a: number, b: number) => a * a + (b * b) / KAPPA;
    for (const t of targets) {
      const segs: string[] = [];
      for (let i = 0; i < grid; i++) {
        for (let j = 0; j < grid; j++) {
          const x = xMin + i * dx;
          const y = yMin + j * dy;
          const v00 = mahal(x, y) - t;
          const v10 = mahal(x + dx, y) - t;
          const v01 = mahal(x, y + dy) - t;
          const v11 = mahal(x + dx, y + dy) - t;
          const c =
            (v00 > 0 ? 1 : 0) + (v10 > 0 ? 2 : 0) + (v11 > 0 ? 4 : 0) + (v01 > 0 ? 8 : 0);
          if (c === 0 || c === 15) continue;
          const lerp = (a: number, b: number) => a / (a - b);
          const eB = { x: x + lerp(v00, v10) * dx, y };
          const eR = { x: x + dx, y: y + lerp(v10, v11) * dy };
          const eT = { x: x + lerp(v01, v11) * dx, y: y + dy };
          const eL = { x, y: y + lerp(v00, v01) * dy };
          const seg = (a: { x: number; y: number }, b: { x: number; y: number }) =>
            segs.push(`M${xS(a.x)},${yS(a.y)}L${xS(b.x)},${yS(b.y)}`);
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
        }
      }
      out.push(segs.join(' '));
    }
    return out;
  }, [innerW, innerH]);

  const sgldPath = useMemo(
    () =>
      d3
        .line<Float32Array>()
        .x((p) => xS(p[0]))
        .y((p) => yS(p[1]))
        .defined((p) => p[0] >= -4 && p[0] <= 4 && p[1] >= -22 && p[1] <= 22)(sgldTraj) ?? '',
    [sgldTraj, xS, yS],
  );
  const sghmcPath = useMemo(
    () =>
      d3
        .line<Float32Array>()
        .x((p) => xS(p[0]))
        .y((p) => yS(p[1]))
        .defined((p) => p[0] >= -4 && p[0] <= 4 && p[1] >= -22 && p[1] <= 22)(sghmcTraj) ?? '',
    [sghmcTraj, xS, yS],
  );

  return (
    <g transform={`translate(${margin.left},${margin.top})`}>
      <rect x={0} y={0} width={innerW} height={innerH} fill="var(--color-surface, #fafafa)" />
      {contours.map((d, i) => (
        <path
          key={i}
          d={d}
          fill="none"
          stroke={paletteSGMCMC.target}
          strokeWidth={0.6}
          opacity={0.45}
        />
      ))}
      <path d={sgldPath} fill="none" stroke={paletteSGMCMC.sgld} strokeWidth={0.6} opacity={0.7} />
      <path
        d={sghmcPath}
        fill="none"
        stroke={paletteSGMCMC.sghmc}
        strokeWidth={0.6}
        opacity={0.85}
      />
      <circle cx={xS(TRAJ_START[0])} cy={yS(TRAJ_START[1])} r={3} fill="black" />
      <line x1={0} x2={innerW} y1={innerH} y2={innerH} stroke="var(--color-text-muted, #888)" />
      <line x1={0} x2={0} y1={0} y2={innerH} stroke="var(--color-text-muted, #888)" />
      <text x={innerW / 2} y={innerH + 18} textAnchor="middle" fontSize={10} fill="var(--color-text, #333)">
        θ_1 (short axis)
      </text>
      <text
        transform={`translate(${-26}, ${innerH / 2}) rotate(-90)`}
        textAnchor="middle"
        fontSize={10}
        fill="var(--color-text, #333)"
      >
        θ_2 (long axis)
      </text>
      {/* Legend */}
      <g transform={`translate(${innerW - 78}, 8)`}>
        <rect x={0} y={-6} width={74} height={42} fill="var(--color-surface, #fff)" stroke="var(--color-border, #ccc)" />
        <line x1={4} x2={20} y1={4} y2={4} stroke={paletteSGMCMC.sgld} strokeWidth={2} />
        <text x={24} y={7} fontSize={10} fill="var(--color-text, #333)">SGLD</text>
        <line x1={4} x2={20} y1={22} y2={22} stroke={paletteSGMCMC.sghmc} strokeWidth={2} />
        <text x={24} y={25} fontSize={10} fill="var(--color-text, #333)">SGHMC</text>
      </g>
    </g>
  );
}

// -----------------------------------------------------------------------------
// ACF panel
// -----------------------------------------------------------------------------

function ACFCurves({
  sgldACF,
  sghmcACF,
  width,
  height,
}: {
  sgldACF: Float32Array;
  sghmcACF: Float32Array;
  width: number;
  height: number;
}) {
  const margin = { top: 8, right: 12, bottom: 28, left: 36 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  const xS = d3.scaleLinear().domain([0, sgldACF.length - 1]).range([0, innerW]);
  const yS = d3.scaleLinear().domain([-0.2, 1.0]).range([innerH, 0]);

  const acfPath = (acf: Float32Array) =>
    Array.from(acf)
      .map((v, i) => `${i === 0 ? 'M' : 'L'}${xS(i)},${yS(v)}`)
      .join(' ');
  const sgldLine = acfPath(sgldACF);
  const sghmcLine = acfPath(sghmcACF);

  return (
    <g transform={`translate(${margin.left},${margin.top})`}>
      <rect x={0} y={0} width={innerW} height={innerH} fill="var(--color-surface, #fafafa)" />
      <line x1={0} x2={innerW} y1={yS(0)} y2={yS(0)} stroke="var(--color-text-muted, #888)" strokeOpacity={0.6} />
      <line x1={0} x2={innerW} y1={yS(0.05)} y2={yS(0.05)} stroke="var(--color-text-muted, #888)" strokeDasharray="3 3" strokeOpacity={0.4} />
      <path d={sgldLine} fill="none" stroke={paletteSGMCMC.sgld} strokeWidth={1.6} />
      <path d={sghmcLine} fill="none" stroke={paletteSGMCMC.sghmc} strokeWidth={1.6} />
      <line x1={0} x2={innerW} y1={innerH} y2={innerH} stroke="var(--color-text-muted, #888)" />
      <line x1={0} x2={0} y1={0} y2={innerH} stroke="var(--color-text-muted, #888)" />
      {/* x ticks */}
      {[0, 100, 200, 300].filter((t) => t <= sgldACF.length - 1).map((t) => (
        <g key={t}>
          <line x1={xS(t)} x2={xS(t)} y1={innerH} y2={innerH + 4} stroke="var(--color-text-muted, #888)" />
          <text x={xS(t)} y={innerH + 16} textAnchor="middle" fontSize={10} fill="var(--color-text, #333)">
            {t}
          </text>
        </g>
      ))}
      {/* y ticks */}
      {[0, 0.5, 1.0].map((t) => (
        <g key={t}>
          <text x={-6} y={yS(t) + 3} textAnchor="end" fontSize={10} fill="var(--color-text, #333)">
            {t.toFixed(1)}
          </text>
        </g>
      ))}
      <text x={innerW / 2} y={innerH + 24} textAnchor="middle" fontSize={10} fill="var(--color-text, #333)">
        lag
      </text>
      <text
        transform={`translate(${-26}, ${innerH / 2}) rotate(-90)`}
        textAnchor="middle"
        fontSize={10}
        fill="var(--color-text, #333)"
      >
        ACF(θ_2)
      </text>
    </g>
  );
}

// -----------------------------------------------------------------------------
// IAT vs C panel
// -----------------------------------------------------------------------------

function IATCurve({
  iatPerC,
  cIndex,
  width,
  height,
}: {
  iatPerC: Float32Array;
  cIndex: number;
  width: number;
  height: number;
}) {
  const margin = { top: 8, right: 12, bottom: 28, left: 44 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  const xS = d3.scaleLog().domain([C_GRID[0] * 0.7, C_GRID[C_GRID.length - 1] * 1.3]).range([0, innerW]);
  const yMin = Math.max(1, Math.min(...Array.from(iatPerC).filter((v) => v > 0)) * 0.7);
  const yMax = Math.max(...Array.from(iatPerC), 10) * 1.3;
  const yS = d3.scaleLog().domain([yMin, yMax]).range([innerH, 0]);

  const fmtPow = (v: number) => `10^${Math.round(Math.log10(v))}`;

  return (
    <g transform={`translate(${margin.left},${margin.top})`}>
      <rect x={0} y={0} width={innerW} height={innerH} fill="var(--color-surface, #fafafa)" />
      {/* C grid lines */}
      {C_GRID.map((c) => (
        <line
          key={`c-${c}`}
          x1={xS(c)}
          x2={xS(c)}
          y1={0}
          y2={innerH}
          stroke="var(--color-text-muted, #888)"
          strokeOpacity={0.1}
        />
      ))}
      {/* y grid (decades) */}
      {decadeTicks(yMin, yMax).map((t) => (
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
      {/* x ticks at each C */}
      {C_GRID.filter((_, i) => i % 2 === 0).map((c) => (
        <text
          key={`x-${c}`}
          x={xS(c)}
          y={innerH + 16}
          textAnchor="middle"
          fontSize={10}
          fill="var(--color-text, #333)"
        >
          {c < 1 ? c.toFixed(2) : c.toFixed(1)}
        </text>
      ))}
      {/* IAT curve */}
      <path
        d={C_GRID.map((c, i) => `${i === 0 ? 'M' : 'L'}${xS(c)},${yS(Math.max(iatPerC[i], yMin))}`).join(' ')}
        fill="none"
        stroke={paletteSGMCMC.sghmc}
        strokeWidth={1.6}
      />
      {C_GRID.map((c, i) => (
        <circle
          key={`pt-${c}`}
          cx={xS(c)}
          cy={yS(Math.max(iatPerC[i], yMin))}
          r={i === cIndex ? 6 : 3.5}
          fill={i === cIndex ? paletteSGMCMC.biasBound : paletteSGMCMC.sghmc}
          stroke={i === cIndex ? 'var(--color-surface, #fff)' : undefined}
          strokeWidth={i === cIndex ? 1.5 : 0}
        />
      ))}
      <line x1={0} x2={innerW} y1={innerH} y2={innerH} stroke="var(--color-text-muted, #888)" />
      <line x1={0} x2={0} y1={0} y2={innerH} stroke="var(--color-text-muted, #888)" />
      <text x={innerW / 2} y={innerH + 24} textAnchor="middle" fontSize={10} fill="var(--color-text, #333)">
        friction C (log)
      </text>
      <text
        transform={`translate(${-34}, ${innerH / 2}) rotate(-90)`}
        textAnchor="middle"
        fontSize={10}
        fill="var(--color-text, #333)"
      >
        IAT(θ_2) (log)
      </text>
    </g>
  );
}

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
