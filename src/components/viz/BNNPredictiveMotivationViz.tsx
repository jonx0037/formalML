// =============================================================================
// BNNPredictiveMotivationViz.tsx
//
// §1 motivation: point estimate vs Bayesian-ensemble predictive on Two Moons.
// Three panels:
//   (a) Single-MLP predicted-probability heatmap (RdBu_r) — sharp, confident,
//       arbitrary far from data.
//   (b) Two Moons data + K MLPs' 0.5-probability contours overlaid.
//   (c) Predictive-variance heatmap across the K MLPs (viridis) — dark near
//       data, bright in the off-distribution corners.
//
// All computation is in-browser. Pre-trains a pool of POOL_SIZE MLPs at the
// default noise on first hydration (~5–7 s), caches them, and re-renders all
// three panels in <100 ms on K-slider interactions. Noise-slider commit and
// resample-seed button trigger a fresh training pass and use the pre-hydration
// PNG as the loading-state placeholder.
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

// -----------------------------------------------------------------------------
// Constants — architecture, training spec, grid
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
const POOL_SIZE = 10;
const N_TRAIN = 300;
const GRID_RES = 60;
const GRID_X_MIN = -2.5;
const GRID_X_MAX = 2.5;
const GRID_Y_MIN = -2.0;
const GRID_Y_MAX = 2.0;

const FIGURE_PATH = '/images/topics/bayesian-neural-networks/01_point_vs_bayesian_predictive.png';
const ALT =
  'Three panels on Two Moons data: panel (a) point-estimate predicted-probability heatmap with a sharp confident decision surface that is correct near the data but arbitrary far from it; panel (b) five independently-trained MLPs’ 0.5-probability contours overlaid as red lines, agreeing tightly near the data and fanning out far from any training point; panel (c) a viridis heatmap of predictive variance computed across the five MLP predictions, dark near the data and bright in the off-distribution corners.';
const CAPTION =
  'Figure 1. The ensemble preview. (a) A single trained MLP confidently predicts everywhere — including regions far from the training data. (b) K MLPs trained from different seeds agree on the data and disagree off it — disagreement among independently-trained models is itself a kind of uncertainty quantification. (c) The variance over the input space recovers the desideratum: the model is uncertain where it lacks data. Drag the noise slider to retrain on a noisier dataset; vary K to see how few members are needed before the variance pattern stabilizes.';
const ARIA = 'Figure 1: Point-estimate vs. Bayesian-ensemble predictive on Two Moons';

// -----------------------------------------------------------------------------
// Grid + ensemble caching
// -----------------------------------------------------------------------------

interface CachedEnsemble {
  data: TrainingData;
  ensemble: DeepEnsembleResult;
  /** Per-member probabilities on the grid in math-coordinate row-major order
   *  (gridProbs[k][i * GRID_RES + j] = σ(f_k(x_i, y_j)) for i,j ∈ [0, GRID_RES)). */
  gridProbs: Float32Array[];
  noise: number;
  seed: number;
}

function buildGridFlat(): Float32Array {
  const out = new Float32Array(GRID_RES * GRID_RES * 2);
  for (let i = 0; i < GRID_RES; i++) {
    const x = GRID_X_MIN + (i / (GRID_RES - 1)) * (GRID_X_MAX - GRID_X_MIN);
    for (let j = 0; j < GRID_RES; j++) {
      const y = GRID_Y_MIN + (j / (GRID_RES - 1)) * (GRID_Y_MAX - GRID_Y_MIN);
      const idx = (i * GRID_RES + j) * 2;
      out[idx] = x;
      out[idx + 1] = y;
    }
  }
  return out;
}

const yieldToBrowser = (): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, 0));

async function trainAndCache(noise: number, seed: number): Promise<CachedEnsemble> {
  await yieldToBrowser();
  const data = makeMoonsData(N_TRAIN, noise, seed);
  const ensemble = deepEnsembleTraining(ARCH, TRAINING_BASE(seed), POOL_SIZE, data);
  const grid = buildGridFlat();
  const gridProbs = ensemble.predictOnGrid(grid, GRID_RES * GRID_RES);
  return { data, ensemble, gridProbs, noise, seed };
}

// -----------------------------------------------------------------------------
// Component state machine
// -----------------------------------------------------------------------------

type CompState =
  | { kind: 'idle' }
  | { kind: 'training'; reason: 'first' | 'noise' | 'reseed' }
  | { kind: 'ready'; cached: CachedEnsemble };

// -----------------------------------------------------------------------------
// Heatmap → canvas helpers
// -----------------------------------------------------------------------------

/**
 * Convert a per-grid Float32Array (math-coord row-major: index i*GRID_RES + j)
 * to a canvas-row-major Float32Array (top-row first, x increasing left→right).
 * Canvas y is flipped relative to math y.
 */
function mathRowMajorToCanvas(probs: Float32Array, gridRes: number): Float32Array {
  const out = new Float32Array(gridRes * gridRes);
  for (let i = 0; i < gridRes; i++) {
    for (let j = 0; j < gridRes; j++) {
      const canvasRow = gridRes - 1 - j;
      out[canvasRow * gridRes + i] = probs[i * gridRes + j];
    }
  }
  return out;
}

function canvasFromColor(
  values: Float32Array,
  gridRes: number,
  colorMap: (v: number) => string,
): string {
  const canvas = document.createElement('canvas');
  canvas.width = gridRes;
  canvas.height = gridRes;
  const ctx = canvas.getContext('2d');
  if (!ctx) return '';
  const imgData = ctx.createImageData(gridRes, gridRes);
  for (let r = 0; r < gridRes; r++) {
    for (let c = 0; c < gridRes; c++) {
      const v = values[r * gridRes + c];
      const rgb = d3.color(colorMap(v))?.rgb();
      const idx = (r * gridRes + c) * 4;
      if (rgb) {
        imgData.data[idx + 0] = rgb.r;
        imgData.data[idx + 1] = rgb.g;
        imgData.data[idx + 2] = rgb.b;
        imgData.data[idx + 3] = 255;
      }
    }
  }
  ctx.putImageData(imgData, 0, 0);
  return canvas.toDataURL();
}

// -----------------------------------------------------------------------------
// Component
// -----------------------------------------------------------------------------

export default function BNNPredictiveMotivationViz() {
  const { ref, width } = useResizeObserver<HTMLElement>();
  const [state, setState] = useState<CompState>({ kind: 'idle' });
  const [noise, setNoise] = useState(0.2);
  const [K, setK] = useState(5);
  const [seed, setSeed] = useState(0);
  const cancelRef = useRef(0);

  // First-hydration kickoff. Empty deps — runs once.
  useEffect(() => {
    let alive = true;
    setState({ kind: 'training', reason: 'first' });
    trainAndCache(noise, seed).then((cached) => {
      if (alive) setState({ kind: 'ready', cached });
    });
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const retrain = useCallback(
    (newNoise: number, newSeed: number, reason: 'noise' | 'reseed') => {
      const ticket = ++cancelRef.current;
      setState({ kind: 'training', reason });
      trainAndCache(newNoise, newSeed).then((cached) => {
        if (ticket === cancelRef.current) setState({ kind: 'ready', cached });
      });
    },
    [],
  );

  const handleNoiseCommit = (next: number) => {
    if (next === noise) return;
    setNoise(next);
    retrain(next, seed, 'noise');
  };
  const handleReseed = () => {
    const next = seed + 1;
    setSeed(next);
    retrain(noise, next, 'reseed');
  };

  const isTraining = state.kind === 'training' || state.kind === 'idle';
  const cached = state.kind === 'ready' ? state.cached : null;

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
                ? `Training ${POOL_SIZE} MLPs (~5 s)…`
                : state.kind === 'training' && state.reason === 'noise'
                  ? 'Retraining at new noise level…'
                  : state.kind === 'training' && state.reason === 'reseed'
                    ? 'Resampling seeds…'
                    : 'Loading…'
            }
          />
        ) : (
          <>
            <Controls
              noise={noise}
              K={K}
              onNoiseCommit={handleNoiseCommit}
              onKChange={setK}
              onReseed={handleReseed}
              isTraining={isTraining}
            />
            <Panels cached={cached} K={K} width={width} />
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
// Loading fallback — renders the static PNG (without nesting <figure>) plus a
// thin status banner so the reader knows training is in progress.
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
  noise,
  K,
  onNoiseCommit,
  onKChange,
  onReseed,
  isTraining,
}: {
  noise: number;
  K: number;
  onNoiseCommit: (next: number) => void;
  onKChange: (next: number) => void;
  onReseed: () => void;
  isTraining: boolean;
}) {
  // Track noise locally so the slider feels live; only commit (= retrain) on
  // mouseup / touchend / change-end, not on every drag tick.
  const [localNoise, setLocalNoise] = useState(noise);
  // Keep localNoise synced when parent's noise changes (e.g., reseed doesn't change it).
  useEffect(() => {
    setLocalNoise(noise);
  }, [noise]);
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
        <span>Noise:</span>
        <input
          type="range"
          min={0.05}
          max={0.4}
          step={0.01}
          value={localNoise}
          onChange={(e) => setLocalNoise(Number(e.target.value))}
          onMouseUp={(e) => onNoiseCommit(Number((e.target as HTMLInputElement).value))}
          onTouchEnd={(e) => onNoiseCommit(Number((e.target as HTMLInputElement).value))}
          onKeyUp={(e) => onNoiseCommit(Number((e.target as HTMLInputElement).value))}
          disabled={isTraining}
          style={{ width: '120px' }}
          aria-label="Two Moons noise"
        />
        <span style={{ fontVariantNumeric: 'tabular-nums', minWidth: '2.5em' }}>
          {localNoise.toFixed(2)}
        </span>
      </label>
      <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}>
        <span>K (members):</span>
        <input
          type="range"
          min={2}
          max={POOL_SIZE}
          step={1}
          value={K}
          onChange={(e) => onKChange(Number(e.target.value))}
          style={{ width: '100px' }}
          aria-label="Number of ensemble members shown"
        />
        <span style={{ fontVariantNumeric: 'tabular-nums', minWidth: '1.5em' }}>{K}</span>
      </label>
      <button
        type="button"
        onClick={onReseed}
        disabled={isTraining}
        style={{
          padding: '0.3rem 0.7rem',
          border: '1px solid var(--color-border, #ccc)',
          borderRadius: '4px',
          background: isTraining ? 'var(--color-surface-muted, #f0f0f0)' : 'var(--color-surface, #fff)',
          color: 'var(--color-text, #333)',
          cursor: isTraining ? 'not-allowed' : 'pointer',
          fontSize: '0.875rem',
        }}
      >
        Resample seeds
      </button>
      {isTraining && (
        <span style={{ color: 'var(--color-text-muted, #888)', fontStyle: 'italic' }}>
          training…
        </span>
      )}
    </div>
  );
}

// -----------------------------------------------------------------------------
// Panel layout
// -----------------------------------------------------------------------------

function Panels({
  cached,
  K,
  width,
}: {
  cached: CachedEnsemble;
  K: number;
  width: number;
}) {
  const layoutWidth = Math.max(width, 320);
  // Three panels side-by-side on wide screens, stacked under 720px.
  const stack = layoutWidth < 720;
  const panelW = stack ? layoutWidth : (layoutWidth - 32) / 3;
  const panelH = 240;

  // Memoize derived data so K-slider doesn't recompute the canvas image URLs.
  const variance = useMemo(() => varianceOverK(cached.gridProbs, K), [cached.gridProbs, K]);
  // Panel (a) shows the FIRST member's predictive (the "point estimate" feel),
  // not the mean — per brief §1 panel (a) intent.
  const point = cached.gridProbs[0];

  // Heatmap data URLs (memoized — same K-independent dependencies).
  const pointURL = useMemo(
    () =>
      canvasFromColor(
        mathRowMajorToCanvas(point, GRID_RES),
        GRID_RES,
        d3.scaleSequential(d3.interpolateRdBu).domain([1, 0]),
      ),
    [point],
  );
  const varURL = useMemo(() => {
    const vMax = Math.max(0.001, ...variance);
    const scale = d3.scaleSequential(d3.interpolateViridis).domain([0, vMax]);
    return canvasFromColor(mathRowMajorToCanvas(variance, GRID_RES), GRID_RES, scale);
  }, [variance]);

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: stack ? 'column' : 'row',
        gap: stack ? '12px' : '16px',
        alignItems: stack ? 'center' : 'flex-start',
      }}
    >
      <Panel title="(a) One MLP — confident everywhere" width={panelW} height={panelH}>
        <Heatmap
          imgURL={pointURL}
          width={panelW}
          height={panelH}
          showData={cached.data}
        />
      </Panel>
      <Panel title={`(b) K=${K} decision boundaries`} width={panelW} height={panelH}>
        <Boundaries
          gridProbs={cached.gridProbs}
          K={K}
          width={panelW}
          height={panelH}
          data={cached.data}
        />
      </Panel>
      <Panel title="(c) Predictive variance" width={panelW} height={panelH}>
        <Heatmap
          imgURL={varURL}
          width={panelW}
          height={panelH}
          showData={cached.data}
          dataStroke="rgba(255,255,255,0.85)"
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
// Heatmap (panels a, c): renders the precomputed canvas image scaled to panel,
// then overlays training-data points.
// -----------------------------------------------------------------------------

function Heatmap({
  imgURL,
  width,
  height,
  showData,
  dataStroke = 'rgba(0,0,0,0.7)',
}: {
  imgURL: string;
  width: number;
  height: number;
  showData?: TrainingData;
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
      {showData &&
        Array.from({ length: showData.n }).map((_, i) => (
          <circle
            key={i}
            cx={xS(showData.X[i * 2])}
            cy={yS(showData.X[i * 2 + 1])}
            r={1.7}
            fill={showData.y[i] === 0 ? paletteBNN.data : '#ffffff'}
            stroke={dataStroke}
            strokeWidth={0.5}
            opacity={0.85}
          />
        ))}
    </>
  );
}

// -----------------------------------------------------------------------------
// Decision boundary panel (b): K contour lines at p=0.5 over the data.
// -----------------------------------------------------------------------------

function Boundaries({
  gridProbs,
  K,
  width,
  height,
  data,
}: {
  gridProbs: Float32Array[];
  K: number;
  width: number;
  height: number;
  data: TrainingData;
}) {
  const xS = d3.scaleLinear().domain([GRID_X_MIN, GRID_X_MAX]).range([0, width]);
  const yS = d3.scaleLinear().domain([GRID_Y_MIN, GRID_Y_MAX]).range([height, 0]);
  const cellW = width / (GRID_RES - 1);
  const cellH = height / (GRID_RES - 1);
  // d3.contours wants canvas-row-major (top-left origin). Convert each member's
  // grid, run marching squares at the 0.5 threshold, and emit one path per member.
  const paths = useMemo(() => {
    const out: string[] = [];
    const contour = d3.contours().size([GRID_RES, GRID_RES]).thresholds([0.5]);
    for (let k = 0; k < K; k++) {
      const arr = Array.from(mathRowMajorToCanvas(gridProbs[k], GRID_RES));
      const polys = contour(arr);
      // d3.contours returns a MultiPolygon in [col, row] grid-cell coordinates;
      // map each ring to SVG pixel coordinates.
      const segments: string[] = [];
      for (const polygon of polys[0]?.coordinates ?? []) {
        for (const ring of polygon) {
          if (ring.length === 0) continue;
          const pts = ring.map(([col, row]) => {
            const px = col * cellW;
            const py = row * cellH;
            return `${px.toFixed(2)},${py.toFixed(2)}`;
          });
          segments.push(`M ${pts.join(' L ')}`);
        }
      }
      out.push(segments.join(' '));
    }
    return out;
    // cellW, cellH derive from width/height; include them so resize re-paths.
  }, [gridProbs, K, cellW, cellH]);

  return (
    <>
      <rect x={0} y={0} width={width} height={height} fill="var(--color-surface, #fafafa)" />
      {paths.map((d, k) => (
        <path
          key={k}
          d={d}
          fill="none"
          stroke={paletteBNN.ensemble}
          strokeWidth={1.4}
          opacity={0.55}
        />
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
        />
      ))}
    </>
  );
}

// -----------------------------------------------------------------------------
// Pure helpers — pointwise mean / variance over the first K predictions.
// -----------------------------------------------------------------------------

function meanOverK(gridProbs: Float32Array[], K: number): Float32Array {
  const n = gridProbs[0].length;
  const out = new Float32Array(n);
  for (let k = 0; k < K; k++) {
    const p = gridProbs[k];
    for (let i = 0; i < n; i++) out[i] += p[i];
  }
  for (let i = 0; i < n; i++) out[i] /= K;
  return out;
}

function varianceOverK(gridProbs: Float32Array[], K: number): Float32Array {
  const n = gridProbs[0].length;
  const mean = meanOverK(gridProbs, K);
  const out = new Float32Array(n);
  for (let k = 0; k < K; k++) {
    const p = gridProbs[k];
    for (let i = 0; i < n; i++) {
      const d = p[i] - mean[i];
      out[i] += d * d;
    }
  }
  for (let i = 0; i < n; i++) out[i] /= Math.max(K - 1, 1);
  return out;
}
