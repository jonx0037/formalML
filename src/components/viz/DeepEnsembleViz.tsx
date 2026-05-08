// =============================================================================
// DeepEnsembleViz.tsx
//
// §5 Deep ensembles as a function-space posterior proxy. Three panels on a
// 60×60 Two Moons grid:
//   (a) Ensemble predictive MEAN (RdBu_r heatmap) — visually similar to §§1/3/4.
//       When a member is highlighted, panel (a) swaps to that member's
//       full predictive heatmap so the reader can see what one "vote" looks like.
//   (b) Ensemble predictive STANDARD DEVIATION (viridis heatmap) — noticeably
//       brighter off-distribution than the §3/§4 single-mode methods.
//   (c) All K members' 0.5-probability contours overlaid (paletteBNN.ensemble,
//       translucent). Click any contour to highlight that member.
//
// Interactive controls:
//   - K slider (2–20, default 10) — subsets the cached pool of POOL_SIZE=20
//     pre-trained MLPs. <100 ms per drag tick (no retraining).
//   - Member-highlight click on panel (c) — clicking near a contour highlights
//     that ensemble member; click empty space to clear.
//   - Resample-seed button — re-rolls seeds and retrains the full pool.
// =============================================================================

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  deepEnsembleTraining,
  makeMoonsData,
  paletteBNN,
  type DeepEnsembleResult,
  type MLPArchSpec,
  type TrainingData,
  type TrainingSpec,
} from './shared/bayesian-ml';
import {
  buildGridFlat,
  canvasFromColor,
  DEFAULT_GRID,
  isoContourPath,
  mathRowMajorToCanvas,
  meanOverK,
  stdOverK,
} from './shared/bnn-grid-render';

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

const ARCH: MLPArchSpec = {
  inputDim: 2,
  hiddenDims: [32, 32, 32],
  outputDim: 1,
  activation: 'relu',
};
const TRAINING_BASE = (seed: number): TrainingSpec => ({
  lr: 0.01,
  weightDecay: 1e-4,
  epochs: 80,
  optimizer: 'adam',
  seed,
});
const POOL_SIZE = 20;
const N_TRAIN = 300;
const TRAIN_NOISE = 0.2;
const { res: GRID_RES, xMin: GRID_X_MIN, xMax: GRID_X_MAX, yMin: GRID_Y_MIN, yMax: GRID_Y_MAX } =
  DEFAULT_GRID;

const FIGURE_PATH = '/images/topics/bayesian-neural-networks/05_deep_ensemble_two_moons.png';
const ALT =
  'Three panels on Two Moons data: panel (a) deep-ensemble predictive mean RdBu_r heatmap; panel (b) deep-ensemble predictive standard deviation viridis heatmap, noticeably brighter off-distribution than single-mode methods; panel (c) all K = 10 ensemble members 0.5-probability contours overlaid in translucent orange, fanning out wider off-distribution than single-mode samples.';
const CAPTION =
  'Figure 5. Deep ensembles. (a) Ensemble mean — robust across methods, similar to §§1/3/4. (b) Predictive standard deviation — noticeably brighter off-distribution, reflecting genuine multi-mode coverage. (c) K decision boundaries fan out wider than single-mode samples — visual confirmation that ensembles cover function-space modes the §3/§4 single-mode methods miss. Drag K to see the diminishing-returns curve; click any boundary in (c) to inspect that member’s full predictive in (a); resample seeds to confirm diversity is robust.';
const ARIA = 'Figure 5: Interactive deep-ensemble predictive on Two Moons';

// -----------------------------------------------------------------------------
// Cached pool
// -----------------------------------------------------------------------------

interface CachedPool {
  data: TrainingData;
  ensemble: DeepEnsembleResult;
  gridProbs: Float32Array[];
  poolSize: number;
  seed: number;
}

const yieldToBrowser = (): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, 0));

async function trainAndCache(seed: number): Promise<CachedPool> {
  await yieldToBrowser();
  const data = makeMoonsData(N_TRAIN, TRAIN_NOISE, seed);
  const ensemble = deepEnsembleTraining(ARCH, TRAINING_BASE(seed), POOL_SIZE, data);
  const grid = buildGridFlat(DEFAULT_GRID);
  const gridProbs = ensemble.predictOnGrid(grid, GRID_RES * GRID_RES);
  return { data, ensemble, gridProbs, poolSize: POOL_SIZE, seed };
}

// -----------------------------------------------------------------------------
// State machine
// -----------------------------------------------------------------------------

type CompState =
  | { kind: 'idle' }
  | { kind: 'training'; reason: 'first' | 'reseed' }
  | { kind: 'ready'; cached: CachedPool };

// -----------------------------------------------------------------------------
// Component
// -----------------------------------------------------------------------------

export default function DeepEnsembleViz() {
  const { ref, width } = useResizeObserver<HTMLElement>();
  const [state, setState] = useState<CompState>({ kind: 'idle' });
  const [K, setK] = useState(10);
  const [seed, setSeed] = useState(7);
  const [highlight, setHighlight] = useState<number | null>(null);
  const ticketRef = useRef(0);

  useEffect(() => {
    let alive = true;
    setState({ kind: 'training', reason: 'first' });
    trainAndCache(seed).then((cached) => {
      if (alive) setState({ kind: 'ready', cached });
    });
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const retrain = useCallback((newSeed: number) => {
    const t = ++ticketRef.current;
    setState({ kind: 'training', reason: 'reseed' });
    trainAndCache(newSeed).then((cached) => {
      if (t === ticketRef.current) setState({ kind: 'ready', cached });
    });
  }, []);

  const handleReseed = () => {
    const next = seed + 1;
    setSeed(next);
    setHighlight(null);
    retrain(next);
  };

  const isBlocked = state.kind === 'training' || state.kind === 'idle';
  const cached = state.kind === 'ready' ? state.cached : null;
  // Clamp highlight to current K so an old highlight doesn't survive a K decrease.
  const effectiveHighlight = highlight !== null && highlight < K ? highlight : null;

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
        {!cached ? (
          <FallbackWithBanner
            label={
              state.kind === 'training' && state.reason === 'first'
                ? `Training pool of ${POOL_SIZE} MLPs (~10 s)…`
                : state.kind === 'training' && state.reason === 'reseed'
                  ? 'Resampling seeds and retraining pool…'
                  : 'Loading…'
            }
          />
        ) : (
          <>
            <Controls
              K={K}
              highlight={effectiveHighlight}
              onKChange={(next) => {
                setK(next);
                if (highlight !== null && highlight >= next) setHighlight(null);
              }}
              onReseed={handleReseed}
              onClearHighlight={() => setHighlight(null)}
              isBlocked={isBlocked}
            />
            <Panels
              cached={cached}
              K={K}
              highlight={effectiveHighlight}
              onPickMember={setHighlight}
              width={width}
            />
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

// -----------------------------------------------------------------------------
// Loading fallback
// -----------------------------------------------------------------------------

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
  K,
  highlight,
  onKChange,
  onReseed,
  onClearHighlight,
  isBlocked,
}: {
  K: number;
  highlight: number | null;
  onKChange: (next: number) => void;
  onReseed: () => void;
  onClearHighlight: () => void;
  isBlocked: boolean;
}) {
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
        <span>K (members):</span>
        <input
          type="range"
          min={2}
          max={POOL_SIZE}
          step={1}
          value={K}
          onChange={(e) => onKChange(Number(e.target.value))}
          style={{ width: '140px' }}
          disabled={isBlocked}
          aria-label="Ensemble size K"
        />
        <span style={{ fontVariantNumeric: 'tabular-nums', minWidth: '1.5em' }}>{K}</span>
      </label>
      {highlight !== null && (
        <span
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '0.3rem',
            color: paletteBNN.ensemble,
          }}
        >
          <strong>Member {highlight}</strong>
          <button
            type="button"
            onClick={onClearHighlight}
            style={{
              padding: '0.15rem 0.45rem',
              border: '1px solid var(--color-border, #ccc)',
              borderRadius: '3px',
              background: 'var(--color-surface, #fff)',
              fontSize: '0.75rem',
              cursor: 'pointer',
            }}
          >
            clear
          </button>
        </span>
      )}
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
        Resample seeds
      </button>
      {isBlocked && (
        <span style={{ color: 'var(--color-text-muted, #888)', fontStyle: 'italic' }}>
          training…
        </span>
      )}
    </div>
  );
}

// -----------------------------------------------------------------------------
// Panels
// -----------------------------------------------------------------------------

function Panels({
  cached,
  K,
  highlight,
  onPickMember,
  width,
}: {
  cached: CachedPool;
  K: number;
  highlight: number | null;
  onPickMember: (k: number | null) => void;
  width: number;
}) {
  const layoutWidth = Math.max(width, 320);
  const stack = layoutWidth < 720;
  const panelW = stack ? layoutWidth : (layoutWidth - 32) / 3;
  const panelH = 240;

  // Mean / std over the first K members.
  const mean = useMemo(() => meanOverK(cached.gridProbs, K), [cached.gridProbs, K]);
  const std = useMemo(() => stdOverK(cached.gridProbs, K), [cached.gridProbs, K]);

  // Heatmap: highlighted member if set, otherwise ensemble mean.
  const meanPanelArr =
    highlight !== null && highlight < cached.gridProbs.length
      ? cached.gridProbs[highlight]
      : mean;

  const meanURL = useMemo(
    () =>
      canvasFromColor(
        mathRowMajorToCanvas(meanPanelArr, GRID_RES),
        GRID_RES,
        d3.scaleSequential(d3.interpolateRdBu).domain([1, 0]),
      ),
    [meanPanelArr],
  );
  const stdURL = useMemo(() => {
    const sMax = Math.max(0.001, ...std);
    const scale = d3.scaleSequential(d3.interpolateViridis).domain([0, sMax]);
    return canvasFromColor(mathRowMajorToCanvas(std, GRID_RES), GRID_RES, scale);
  }, [std]);

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
        title={highlight !== null ? `(a) Member ${highlight} predictive` : '(a) Ensemble mean'}
        width={panelW}
        height={panelH}
      >
        <Heatmap imgURL={meanURL} width={panelW} height={panelH} data={cached.data} />
      </Panel>
      <Panel title="(b) Ensemble std" width={panelW} height={panelH}>
        <Heatmap
          imgURL={stdURL}
          width={panelW}
          height={panelH}
          data={cached.data}
          dataStroke="rgba(255,255,255,0.85)"
        />
      </Panel>
      <Panel
        title={
          highlight !== null
            ? `(c) ${K} boundaries — member ${highlight} highlighted`
            : `(c) ${K} ensemble boundaries (click to inspect)`
        }
        width={panelW}
        height={panelH}
      >
        <Boundaries
          gridProbs={cached.gridProbs}
          K={K}
          highlight={highlight}
          onPickMember={onPickMember}
          width={panelW}
          height={panelH}
          data={cached.data}
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

function Heatmap({
  imgURL,
  width,
  height,
  data,
  dataStroke = 'rgba(0,0,0,0.7)',
}: {
  imgURL: string;
  width: number;
  height: number;
  data: TrainingData;
  dataStroke?: string;
}) {
  const xS = d3.scaleLinear().domain([GRID_X_MIN, GRID_X_MAX]).range([0, width]);
  const yS = d3.scaleLinear().domain([GRID_Y_MIN, GRID_Y_MAX]).range([height, 0]);
  return (
    <>
      <image
        href={imgURL}
        x={0}
        y={0}
        width={width}
        height={height}
        preserveAspectRatio="none"
      />
      {Array.from({ length: data.n }).map((_, i) => (
        <circle
          key={i}
          cx={xS(data.X[i * 2])}
          cy={yS(data.X[i * 2 + 1])}
          r={1.7}
          fill={data.y[i] === 0 ? paletteBNN.data : '#ffffff'}
          stroke={dataStroke}
          strokeWidth={0.5}
          opacity={0.85}
        />
      ))}
    </>
  );
}

// -----------------------------------------------------------------------------
// Boundaries with click-to-pick — each member gets a wide invisible hit-path
// stacked behind its visible thin one so the click target is forgiving.
// -----------------------------------------------------------------------------

function Boundaries({
  gridProbs,
  K,
  highlight,
  onPickMember,
  width,
  height,
  data,
}: {
  gridProbs: Float32Array[];
  K: number;
  highlight: number | null;
  onPickMember: (k: number | null) => void;
  width: number;
  height: number;
  data: TrainingData;
}) {
  const xS = d3.scaleLinear().domain([GRID_X_MIN, GRID_X_MAX]).range([0, width]);
  const yS = d3.scaleLinear().domain([GRID_Y_MIN, GRID_Y_MAX]).range([height, 0]);
  const paths = useMemo(() => {
    const out: string[] = [];
    for (let k = 0; k < K; k++) {
      out.push(isoContourPath(gridProbs[k], GRID_RES, width, height, 0.5));
    }
    return out;
  }, [gridProbs, K, width, height]);

  return (
    <>
      <rect
        x={0}
        y={0}
        width={width}
        height={height}
        fill="var(--color-surface, #fafafa)"
        onClick={() => onPickMember(null)}
        style={{ cursor: 'pointer' }}
      />
      {paths.map((d, k) => (
        <g key={k}>
          {/* Wide invisible hit target for forgiving click */}
          <path
            d={d}
            fill="none"
            stroke="transparent"
            strokeWidth={10}
            style={{ cursor: 'pointer' }}
            onClick={(e) => {
              e.stopPropagation();
              onPickMember(highlight === k ? null : k);
            }}
            aria-label={`Highlight ensemble member ${k}`}
          />
          {/* Visible stroke */}
          <path
            d={d}
            fill="none"
            stroke={paletteBNN.ensemble}
            strokeWidth={highlight === k ? 2.5 : 1.4}
            opacity={highlight === null ? 0.55 : highlight === k ? 1 : 0.2}
            style={{ pointerEvents: 'none' }}
          />
        </g>
      ))}
      {Array.from({ length: data.n }).map((_, i) => (
        <circle
          key={i}
          cx={xS(data.X[i * 2])}
          cy={yS(data.X[i * 2 + 1])}
          r={2}
          fill={data.y[i] === 0 ? paletteBNN.data : '#ffffff'}
          stroke={data.y[i] === 0 ? '#ffffff' : paletteBNN.data}
          strokeWidth={0.6}
          style={{ pointerEvents: 'none' }}
        />
      ))}
    </>
  );
}
