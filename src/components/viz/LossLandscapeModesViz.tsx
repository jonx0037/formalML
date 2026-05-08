// =============================================================================
// LossLandscapeModesViz.tsx
//
// §2 weight-space posterior — the loss landscape is multi-modal. Two panels:
//   (a) PCA scatter of N trained MLP weight vectors projected to their first
//       two principal components (dual PCA via the K×K Gram matrix). Each
//       point colored by final training loss. Two endpoints (A, B) are
//       highlighted; clicking the scatter swaps the "next" endpoint.
//   (b) Loss-along-interpolation profile L_WD((1−t)·w_A + t·w_B) for
//       t ∈ [0, 1] on an M-point grid. Annotates the trained-model floor
//       (dashed horizontal line at min(L(w_A), L(w_B))) and the peak barrier.
//
// Interactive controls:
//   - N slider (5–30, default 20). Subsets the cached pool of POOL_SIZE = 30
//     pre-trained MLPs and re-runs PCA. <500 ms per change.
//   - M slider (20–200, default 100). Resolution of the interpolation grid.
//     Loss recomputation is O(M · n_train · p_per_layer) ≈ O(M · 633k) ≈ 30 ms
//     at M = 100.
//   - Click on panel (a) — selects the closest point as the next endpoint.
//     The "next-click sets" cycle alternates A → B → A → B …; a label in the
//     control bar shows which endpoint will be assigned next.
//
// Pre-trains POOL_SIZE = 30 MLPs on first hydration (~15 s blocking,
// the heaviest first-hydration cost in the topic). The PNG fallback is shown
// throughout via the loading-state pattern from CLAUDE.md.
// =============================================================================

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  BCE_EPSILON,
  deepEnsembleTraining,
  makeMoonsData,
  mlpForward,
  mlpLayout,
  paletteBNN,
  pcaProject2D,
  type MLPArchSpec,
  type TrainingData,
  type TrainingSpec,
  type PCAProject2DResult,
} from './shared/bayesian-ml';

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
const POOL_SIZE = 30;
const N_TRAIN = 300;
const TRAIN_NOISE = 0.2;

const FIGURE_PATH = '/images/topics/bayesian-neural-networks/02_loss_landscape_modes.png';
const ALT =
  'Two panels: panel (a) PCA scatter of trained MLP weight vectors projected to their first two principal components, color-coded by final training loss; panel (b) loss along a linear interpolation between two trained models, rising from the trained-model floor to a peak barrier and back down — a non-convex ridge separating two modes.';
const CAPTION =
  'Figure 2. The loss landscape is genuinely multi-modal. (a) N MLPs from independent seeds project into discrete clusters in PCA space — each cluster a permutation/scaling class of §2.4’s hidden-unit symmetry. (b) Linearly interpolating between two trained models passes through a barrier of strictly higher loss. Click any two points in (a) to set the endpoints; a single Gaussian (Laplace, mean-field VI) cannot bridge that ridge — which is why §5 (deep ensembles) and §§6–7 (SG-MCMC) exist.';
const ARIA = 'Figure 2: PCA-projected weight-space modes and the loss barrier between them';

// -----------------------------------------------------------------------------
// Cached pool
// -----------------------------------------------------------------------------

interface CachedPool {
  data: TrainingData;
  weights: Float32Array[]; // POOL_SIZE entries, each of dim p
  finalLosses: number[];
  pDim: number;
  seed: number;
}

const yieldToBrowser = (): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, 0));

async function trainAndCache(seed: number): Promise<CachedPool> {
  await yieldToBrowser();
  const data = makeMoonsData(N_TRAIN, TRAIN_NOISE, seed);
  const ensemble = deepEnsembleTraining(ARCH, TRAINING_BASE(seed), POOL_SIZE, data);
  return {
    data,
    weights: ensemble.weights,
    finalLosses: ensemble.finalLosses,
    pDim: ensemble.weights[0].length,
    seed,
  };
}

// -----------------------------------------------------------------------------
// Loss-along-interpolation: BCE + λ/2 ‖w‖² evaluated at each t
// -----------------------------------------------------------------------------

function computeLossProfile(
  data: TrainingData,
  wA: Float32Array,
  wB: Float32Array,
  M: number,
  weightDecay: number,
): { tGrid: Float32Array; loss: Float32Array } {
  const layout = mlpLayout(ARCH);
  const tGrid = new Float32Array(M);
  const loss = new Float32Array(M);
  const wT = new Float32Array(layout.pDim);
  for (let mi = 0; mi < M; mi++) {
    const t = M === 1 ? 0 : mi / (M - 1);
    tGrid[mi] = t;
    for (let i = 0; i < layout.pDim; i++) wT[i] = (1 - t) * wA[i] + t * wB[i];
    const probs = mlpForward(data.X, data.n, wT, ARCH, layout, null, null);
    let bce = 0;
    for (let i = 0; i < data.n; i++) {
      const p = Math.min(Math.max(probs[i], BCE_EPSILON), 1 - BCE_EPSILON);
      bce += -data.y[i] * Math.log(p) - (1 - data.y[i]) * Math.log(1 - p);
    }
    bce /= data.n;
    let l2 = 0;
    for (let i = 0; i < layout.pDim; i++) l2 += wT[i] * wT[i];
    loss[mi] = bce + 0.5 * weightDecay * l2;
  }
  return { tGrid, loss };
}

function pickFurthestPair(scores: Array<[number, number]>): [number, number] {
  let bestI = 0;
  let bestJ = 1;
  let bestD = -1;
  for (let i = 0; i < scores.length; i++) {
    for (let j = i + 1; j < scores.length; j++) {
      const dx = scores[i][0] - scores[j][0];
      const dy = scores[i][1] - scores[j][1];
      const d = dx * dx + dy * dy;
      if (d > bestD) {
        bestD = d;
        bestI = i;
        bestJ = j;
      }
    }
  }
  return [bestI, bestJ];
}

// -----------------------------------------------------------------------------
// State machine
// -----------------------------------------------------------------------------

type CompState =
  | { kind: 'idle' }
  | { kind: 'training' }
  | { kind: 'ready'; cached: CachedPool };

// -----------------------------------------------------------------------------
// Component
// -----------------------------------------------------------------------------

export default function LossLandscapeModesViz() {
  const { ref, width } = useResizeObserver<HTMLElement>();
  const [state, setState] = useState<CompState>({ kind: 'idle' });
  const [N, setN] = useState(20);
  const [M, setM] = useState(100);
  const [endpoints, setEndpoints] = useState<[number, number]>([0, 1]);
  const [nextSlot, setNextSlot] = useState<0 | 1>(0);
  const seedRef = useRef(0);
  const initialEndpointsPicked = useRef(false);

  useEffect(() => {
    let alive = true;
    setState({ kind: 'training' });
    trainAndCache(seedRef.current).then((cached) => {
      if (alive) setState({ kind: 'ready', cached });
    });
    return () => {
      alive = false;
    };
  }, []);

  const handlePoolClick = useCallback(
    (idx: number) => {
      setEndpoints((cur) => {
        const next: [number, number] = [...cur];
        next[nextSlot] = idx;
        return next;
      });
      setNextSlot((s) => (s === 0 ? 1 : 0));
    },
    [nextSlot],
  );

  const cached = state.kind === 'ready' ? state.cached : null;

  // PCA over first N.
  const pca = useMemo<PCAProject2DResult | null>(() => {
    if (!cached) return null;
    const sub = cached.weights.slice(0, N);
    return pcaProject2D(sub, 1);
  }, [cached, N]);

  // First-pca: pick the furthest pair as default endpoints.
  // After that: clamp + repick if user shrinks N below an existing endpoint.
  useEffect(() => {
    if (!pca) return;
    if (!initialEndpointsPicked.current) {
      initialEndpointsPicked.current = true;
      setEndpoints(pickFurthestPair(pca.scores));
      return;
    }
    setEndpoints((cur) => {
      if (cur[0] < N && cur[1] < N) return cur;
      return pickFurthestPair(pca.scores);
    });
  }, [pca, N]);

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
          <FallbackWithBanner label={`Training pool of ${POOL_SIZE} MLPs (~15 s)…`} />
        ) : pca ? (
          <>
            <Controls
              N={N}
              M={M}
              endpoints={endpoints}
              nextSlot={nextSlot}
              onNChange={setN}
              onMChange={setM}
              onSetNextSlot={setNextSlot}
            />
            <Panels
              cached={cached}
              pca={pca}
              N={N}
              M={M}
              endpoints={endpoints}
              onPickPoint={handlePoolClick}
              width={width}
            />
          </>
        ) : null}
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

const ENDPOINT_COLORS: [string, string] = ['#d62728', '#1f77b4']; // red (A), blue (B)

function Controls({
  N,
  M,
  endpoints,
  nextSlot,
  onNChange,
  onMChange,
  onSetNextSlot,
}: {
  N: number;
  M: number;
  endpoints: [number, number];
  nextSlot: 0 | 1;
  onNChange: (next: number) => void;
  onMChange: (next: number) => void;
  onSetNextSlot: (s: 0 | 1) => void;
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
        <span>N (models):</span>
        <input
          type="range"
          min={5}
          max={POOL_SIZE}
          step={1}
          value={N}
          onChange={(e) => onNChange(Number(e.target.value))}
          style={{ width: '120px' }}
          aria-label="Ensemble size N"
        />
        <span style={{ fontVariantNumeric: 'tabular-nums', minWidth: '2em' }}>{N}</span>
      </label>
      <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}>
        <span>M (grid):</span>
        <input
          type="range"
          min={20}
          max={200}
          step={5}
          value={M}
          onChange={(e) => onMChange(Number(e.target.value))}
          style={{ width: '120px' }}
          aria-label="Interpolation grid resolution M"
        />
        <span style={{ fontVariantNumeric: 'tabular-nums', minWidth: '2.5em' }}>{M}</span>
      </label>
      <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}>
        <span>Endpoints:</span>
        <button
          type="button"
          onClick={() => onSetNextSlot(0)}
          style={{
            padding: '0.2rem 0.55rem',
            borderRadius: '3px',
            border: `1px solid ${ENDPOINT_COLORS[0]}`,
            background:
              nextSlot === 0 ? ENDPOINT_COLORS[0] : 'var(--color-surface, #fff)',
            color: nextSlot === 0 ? '#fff' : ENDPOINT_COLORS[0],
            fontSize: '0.825rem',
            cursor: 'pointer',
          }}
        >
          A = {endpoints[0]}
        </button>
        <button
          type="button"
          onClick={() => onSetNextSlot(1)}
          style={{
            padding: '0.2rem 0.55rem',
            borderRadius: '3px',
            border: `1px solid ${ENDPOINT_COLORS[1]}`,
            background:
              nextSlot === 1 ? ENDPOINT_COLORS[1] : 'var(--color-surface, #fff)',
            color: nextSlot === 1 ? '#fff' : ENDPOINT_COLORS[1],
            fontSize: '0.825rem',
            cursor: 'pointer',
          }}
        >
          B = {endpoints[1]}
        </button>
      </span>
      <span style={{ color: 'var(--color-text-muted, #888)', fontStyle: 'italic' }}>
        Next click sets {nextSlot === 0 ? 'A' : 'B'}
      </span>
    </div>
  );
}

// -----------------------------------------------------------------------------
// Panels
// -----------------------------------------------------------------------------

function Panels({
  cached,
  pca,
  N,
  M,
  endpoints,
  onPickPoint,
  width,
}: {
  cached: CachedPool;
  pca: PCAProject2DResult;
  N: number;
  M: number;
  endpoints: [number, number];
  onPickPoint: (idx: number) => void;
  width: number;
}) {
  const layoutWidth = Math.max(width, 320);
  const stack = layoutWidth < 720;
  const panelW = stack ? layoutWidth : (layoutWidth - 32) / 2;
  const panelH = 280;

  // Loss profile recomputed when (endpoints, N, M, cached) change.
  const profile = useMemo(() => {
    const a = endpoints[0] < N ? endpoints[0] : 0;
    const b = endpoints[1] < N ? endpoints[1] : Math.min(1, N - 1);
    return computeLossProfile(cached.data, cached.weights[a], cached.weights[b], M, 1e-4);
  }, [cached, endpoints, N, M]);

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: stack ? 'column' : 'row',
        gap: stack ? '12px' : '24px',
        alignItems: stack ? 'center' : 'flex-start',
      }}
    >
      <PanelFrame title="(a) PCA of trained weight vectors" width={panelW} height={panelH}>
        <ScatterPanel
          pca={pca}
          finalLosses={cached.finalLosses}
          N={N}
          endpoints={endpoints}
          onPickPoint={onPickPoint}
          width={panelW}
          height={panelH}
        />
      </PanelFrame>
      <PanelFrame title="(b) Loss along interpolation" width={panelW} height={panelH}>
        <ProfilePanel
          tGrid={profile.tGrid}
          loss={profile.loss}
          floorLoss={Math.min(profile.loss[0], profile.loss[profile.loss.length - 1])}
          width={panelW}
          height={panelH}
        />
      </PanelFrame>
    </div>
  );
}

function PanelFrame({
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
// Scatter panel — PCA scores, color by training loss, click to pick endpoint
// -----------------------------------------------------------------------------

function ScatterPanel({
  pca,
  finalLosses,
  N,
  endpoints,
  onPickPoint,
  width,
  height,
}: {
  pca: PCAProject2DResult;
  finalLosses: number[];
  N: number;
  endpoints: [number, number];
  onPickPoint: (idx: number) => void;
  width: number;
  height: number;
}) {
  const margin = { top: 12, right: 12, bottom: 32, left: 40 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  const xs = pca.scores.map((s) => s[0]);
  const ys = pca.scores.map((s) => s[1]);
  const xExt = d3.extent(xs) as [number, number];
  const yExt = d3.extent(ys) as [number, number];
  // Pad a touch
  const padX = ((xExt[1] ?? 1) - (xExt[0] ?? 0)) * 0.1 || 0.5;
  const padY = ((yExt[1] ?? 1) - (yExt[0] ?? 0)) * 0.1 || 0.5;
  const xScale = d3.scaleLinear().domain([xExt[0]! - padX, xExt[1]! + padX]).range([0, innerW]);
  const yScale = d3.scaleLinear().domain([yExt[0]! - padY, yExt[1]! + padY]).range([innerH, 0]);
  const lossSubset = finalLosses.slice(0, N);
  const lossExt = d3.extent(lossSubset) as [number, number];
  const colorScale = d3
    .scaleSequential(d3.interpolateViridis)
    .domain([lossExt[1]! + 1e-9, lossExt[0]!]); // higher loss → darker

  return (
    <g transform={`translate(${margin.left},${margin.top})`}>
      <rect x={0} y={0} width={innerW} height={innerH} fill="var(--color-surface, #fafafa)" />
      {/* Points */}
      {pca.scores.map(([x, y], i) => {
        const isA = i === endpoints[0];
        const isB = i === endpoints[1];
        return (
          <circle
            key={i}
            cx={xScale(x)}
            cy={yScale(y)}
            r={isA || isB ? 7 : 4.5}
            fill={
              isA ? ENDPOINT_COLORS[0] : isB ? ENDPOINT_COLORS[1] : colorScale(finalLosses[i])
            }
            stroke={isA || isB ? '#fff' : 'var(--color-text-muted, #888)'}
            strokeWidth={isA || isB ? 2 : 0.6}
            style={{ cursor: 'pointer' }}
            onClick={() => onPickPoint(i)}
            aria-label={`Trained model ${i}, loss ${finalLosses[i].toFixed(3)}`}
          />
        );
      })}
      {/* Axes — light, no ticks needed for PC scores */}
      <line x1={0} y1={innerH} x2={innerW} y2={innerH} stroke="var(--color-text-muted, #888)" />
      <line x1={0} y1={0} x2={0} y2={innerH} stroke="var(--color-text-muted, #888)" />
      <text
        x={innerW / 2}
        y={innerH + 24}
        textAnchor="middle"
        fontSize={11}
        fill="var(--color-text, #333)"
      >
        PC₁
      </text>
      <text
        transform={`translate(${-26}, ${innerH / 2}) rotate(-90)`}
        textAnchor="middle"
        fontSize={11}
        fill="var(--color-text, #333)"
      >
        PC₂
      </text>
    </g>
  );
}

// -----------------------------------------------------------------------------
// Profile panel — line plot with floor + barrier annotations
// -----------------------------------------------------------------------------

function ProfilePanel({
  tGrid,
  loss,
  floorLoss,
  width,
  height,
}: {
  tGrid: Float32Array;
  loss: Float32Array;
  floorLoss: number;
  width: number;
  height: number;
}) {
  const margin = { top: 18, right: 16, bottom: 32, left: 48 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  const lossArr = Array.from(loss);
  let peakIdx = 0;
  for (let i = 1; i < lossArr.length; i++) if (lossArr[i] > lossArr[peakIdx]) peakIdx = i;
  const peakLoss = lossArr[peakIdx];
  const yMax = peakLoss * 1.05;
  const yMin = Math.min(0, floorLoss * 0.9);
  const xScale = d3.scaleLinear().domain([0, 1]).range([0, innerW]);
  const yScale = d3.scaleLinear().domain([yMin, yMax]).range([innerH, 0]);

  const line = d3
    .line<number>()
    .x((_, i) => xScale(tGrid[i]))
    .y((v) => yScale(v))
    .curve(d3.curveMonotoneX);

  return (
    <g transform={`translate(${margin.left},${margin.top})`}>
      <rect x={0} y={0} width={innerW} height={innerH} fill="var(--color-surface, #fafafa)" />
      {/* Floor reference */}
      <line
        x1={0}
        x2={innerW}
        y1={yScale(floorLoss)}
        y2={yScale(floorLoss)}
        stroke="var(--color-text-muted, #999)"
        strokeDasharray="3 3"
      />
      <text
        x={innerW - 4}
        y={yScale(floorLoss) - 4}
        textAnchor="end"
        fontSize={10}
        fill="var(--color-text-muted, #888)"
      >
        floor {floorLoss.toFixed(3)}
      </text>
      {/* Loss curve */}
      <path
        d={line(lossArr) ?? ''}
        fill="none"
        stroke={paletteBNN.point}
        strokeWidth={2}
      />
      {/* Peak marker */}
      <circle cx={xScale(tGrid[peakIdx])} cy={yScale(peakLoss)} r={4} fill="#d62728" />
      <text
        x={xScale(tGrid[peakIdx])}
        y={yScale(peakLoss) - 6}
        textAnchor="middle"
        fontSize={10}
        fill="#d62728"
      >
        peak {peakLoss.toFixed(3)}
      </text>
      {/* Endpoint markers (t=0 = A, t=1 = B) */}
      <circle cx={xScale(0)} cy={yScale(loss[0])} r={5} fill={ENDPOINT_COLORS[0]} stroke="#fff" strokeWidth={1.5} />
      <circle cx={xScale(1)} cy={yScale(loss[loss.length - 1])} r={5} fill={ENDPOINT_COLORS[1]} stroke="#fff" strokeWidth={1.5} />
      {/* Axes */}
      <g transform={`translate(0, ${innerH})`}>
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t} transform={`translate(${xScale(t)}, 0)`}>
            <line y2={4} stroke="var(--color-text, #333)" />
            <text y={16} fontSize={10} textAnchor="middle" fill="var(--color-text, #333)">
              {t.toFixed(2)}
            </text>
          </g>
        ))}
      </g>
      <g>
        {[yMin, (yMin + yMax) / 2, yMax].map((v) => (
          <g key={v} transform={`translate(0, ${yScale(v)})`}>
            <line x2={-4} stroke="var(--color-text, #333)" />
            <text
              x={-8}
              dy="0.32em"
              fontSize={10}
              textAnchor="end"
              fill="var(--color-text, #333)"
            >
              {v.toFixed(2)}
            </text>
          </g>
        ))}
      </g>
      <text
        x={innerW / 2}
        y={innerH + 24}
        textAnchor="middle"
        fontSize={11}
        fill="var(--color-text, #333)"
      >
        t  (interpolation parameter)
      </text>
      <text
        transform={`translate(${-36}, ${innerH / 2}) rotate(-90)`}
        textAnchor="middle"
        fontSize={11}
        fill="var(--color-text, #333)"
      >
        L_WD
      </text>
    </g>
  );
}
