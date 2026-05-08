// =============================================================================
// MCDropoutBNNViz.tsx
//
// §4 MC-dropout as approximate variational inference. Three panels on a
// 60×60 Two Moons grid:
//   (a) MC-dropout predictive mean (T forward passes averaged) — RdBu_r heatmap.
//       Visually identical to the §1/§3 mean prediction; the takeaway is that
//       dropout at test time leaves the *mean* unchanged.
//   (b) MC-dropout predictive standard deviation across T samples — viridis.
//       Bright off-distribution, dark near data; "flatter" than §3 Laplace
//       per Foong et al. (Rem 4.4).
//   (c) 20 sampled 0.5-probability contours from the T forward passes,
//       overlaid as translucent green lines (paletteBNN.dropout).
//
// Interactive controls:
//   - Dropout rate slider (0.0–0.5, default 0.10) — retrains the dropout MLP
//     (~5s blocking); commits on mouseup/touchend/keyup, not during drag.
//   - Forward-pass count slider T (10–200, default 100) — cheap re-prediction
//     using the cached trained model (<100 ms). No retraining.
//   - Toggle: dropout-on-test (default on). When off, the network does one
//     deterministic forward pass — pedagogically reveals that the predictive
//     *mean* is the standard deterministic prediction; only the variance
//     requires test-time stochasticity.
// =============================================================================

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  makeMoonsData,
  mcDropoutInference,
  paletteBNN,
  type MCDropoutResult,
  type MLPArchSpec,
  type TrainedMLP,
  type TrainingData,
  type TrainingSpec,
} from './shared/bayesian-ml';
import {
  buildGridFlat,
  canvasFromColor,
  DEFAULT_GRID,
  isoContourPath,
  mathRowMajorToCanvas,
  maxFinite,
  meanOverK,
  stdOverK,
} from './shared/bnn-grid-render';

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

const baseArch = (dropoutP: number): MLPArchSpec => ({
  inputDim: 2,
  hiddenDims: [32, 32, 32],
  outputDim: 1,
  activation: 'relu',
  dropoutP,
});
const TRAINING_BASE = (seed: number): TrainingSpec => ({
  lr: 0.01,
  weightDecay: 1e-4,
  epochs: 80,
  optimizer: 'adam',
  seed,
});
const N_TRAIN = 300;
const TRAIN_NOISE = 0.2;
const SAMPLED_CONTOURS_PER_PANEL = 20;
const { res: GRID_RES, xMin: GRID_X_MIN, xMax: GRID_X_MAX, yMin: GRID_Y_MIN, yMax: GRID_Y_MAX } =
  DEFAULT_GRID;

const FIGURE_PATH = '/images/topics/bayesian-neural-networks/04_mcdropout_two_moons.png';
const ALT =
  'Three panels on Two Moons data: panel (a) MC-dropout predictive mean RdBu_r heatmap visually similar to a deterministic predictor; panel (b) MC-dropout predictive standard deviation viridis heatmap with flatter off-distribution std than a full-Hessian Laplace; panel (c) twenty MC-dropout sampled 0.5-probability contours overlaid in translucent green.';
const CAPTION =
  'Figure 4. MC-dropout. (a) The predictive mean is essentially the point-estimate predictor — turning dropout off at test confirms it. (b) The predictive standard deviation captures epistemic uncertainty qualitatively but is flatter off-distribution than Laplace. (c) Sampled decision boundaries fan out narrowly. Drag the dropout-rate slider to retrain (1−ρ from 0 retains a deterministic predictor; large 1−ρ over-regularizes); slide T to trade compute for variance smoothness; toggle dropout-on-test to compare deterministic vs MC predictions side by side.';
const ARIA = 'Figure 4: Interactive MC-dropout predictive on Two Moons';

// -----------------------------------------------------------------------------
// Cached per-rate state
// -----------------------------------------------------------------------------

interface CachedRun {
  data: TrainingData;
  inference: MCDropoutResult;
  trainedModel: TrainedMLP;
  /** T forward passes cached at the current (T, predictSeed) pair. */
  predsMC: Float32Array[];
  /** Single deterministic forward pass through the trained weights (no dropout). */
  predDet: Float32Array;
  T: number;
  predictSeed: number;
  dropoutP: number;
  seed: number;
}

const yieldToBrowser = (): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, 0));

async function trainAndPredict(
  dropoutP: number,
  T: number,
  seed: number,
  predictSeed: number,
): Promise<CachedRun> {
  await yieldToBrowser();
  const data = makeMoonsData(N_TRAIN, TRAIN_NOISE, seed);
  const inference = mcDropoutInference(baseArch(dropoutP), TRAINING_BASE(seed), data);
  const grid = buildGridFlat(DEFAULT_GRID);
  const predsMC = inference.predict(grid, GRID_RES * GRID_RES, T, predictSeed);
  const predDet = inference.trainedModel.forward(grid, GRID_RES * GRID_RES);
  return {
    data,
    inference,
    trainedModel: inference.trainedModel,
    predsMC,
    predDet,
    T,
    predictSeed,
    dropoutP,
    seed,
  };
}

/** Cheap re-prediction: T forward passes through the *cached* trained model. */
async function repredictOnly(prev: CachedRun, T: number, predictSeed: number): Promise<CachedRun> {
  await yieldToBrowser();
  const grid = buildGridFlat(DEFAULT_GRID);
  const predsMC = prev.inference.predict(grid, GRID_RES * GRID_RES, T, predictSeed);
  return { ...prev, predsMC, T, predictSeed };
}

// -----------------------------------------------------------------------------
// State machine
// -----------------------------------------------------------------------------

type CompState =
  | { kind: 'idle' }
  | { kind: 'training'; reason: 'first' | 'rate' }
  | { kind: 'predicting'; reason: 'T' | 'reseed'; cached: CachedRun }
  | { kind: 'ready'; cached: CachedRun };

// -----------------------------------------------------------------------------
// Component
// -----------------------------------------------------------------------------

export default function MCDropoutBNNViz() {
  const { ref, width } = useResizeObserver<HTMLElement>();
  const [state, setState] = useState<CompState>({ kind: 'idle' });
  const [dropoutP, setDropoutP] = useState(0.1);
  const [T, setT] = useState(100);
  const [predictSeed, setPredictSeed] = useState(0);
  const [dropoutAtTest, setDropoutAtTest] = useState(true);
  const [seed] = useState(42);
  const trainTicketRef = useRef(0);
  const predictTicketRef = useRef(0);

  // First-hydration kickoff.
  useEffect(() => {
    let alive = true;
    setState({ kind: 'training', reason: 'first' });
    trainAndPredict(dropoutP, T, seed, predictSeed).then((cached) => {
      if (alive) setState({ kind: 'ready', cached });
    });
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const retrain = useCallback(
    (newDropoutP: number) => {
      const ticket = ++trainTicketRef.current;
      setState({ kind: 'training', reason: 'rate' });
      trainAndPredict(newDropoutP, T, seed, predictSeed).then((cached) => {
        if (ticket === trainTicketRef.current) setState({ kind: 'ready', cached });
      });
    },
    [T, seed, predictSeed],
  );

  const repredict = useCallback(
    (prev: CachedRun, newT: number, newPredictSeed: number, reason: 'T' | 'reseed') => {
      const ticket = ++predictTicketRef.current;
      setState({ kind: 'predicting', reason, cached: prev });
      repredictOnly(prev, newT, newPredictSeed).then((cached) => {
        if (ticket === predictTicketRef.current) setState({ kind: 'ready', cached });
      });
    },
    [],
  );

  const handleDropoutCommit = (next: number) => {
    if (Math.abs(next - dropoutP) < 1e-6) return;
    setDropoutP(next);
    retrain(next);
  };
  const handleTCommit = (next: number) => {
    if (next === T) return;
    setT(next);
    if (state.kind === 'ready' || state.kind === 'predicting') {
      repredict(state.cached, next, predictSeed, 'T');
    }
  };
  const handleReseed = () => {
    const next = predictSeed + 1;
    setPredictSeed(next);
    if (state.kind === 'ready' || state.kind === 'predicting') {
      repredict(state.cached, T, next, 'reseed');
    }
  };

  const isBlocked = state.kind === 'training' || state.kind === 'idle';
  const cached =
    state.kind === 'ready' || state.kind === 'predicting' ? state.cached : null;

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
                ? 'Training MC-dropout MLP (~5 s)…'
                : state.kind === 'training' && state.reason === 'rate'
                  ? 'Retraining at new dropout rate…'
                  : 'Loading…'
            }
          />
        ) : (
          <>
            <Controls
              dropoutP={dropoutP}
              T={T}
              dropoutAtTest={dropoutAtTest}
              onDropoutCommit={handleDropoutCommit}
              onTCommit={handleTCommit}
              onToggleAtTest={setDropoutAtTest}
              onReseed={handleReseed}
              isBlocked={isBlocked}
              isPredicting={state.kind === 'predicting'}
            />
            <Panels
              cached={cached}
              dropoutAtTest={dropoutAtTest}
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
  dropoutP,
  T,
  dropoutAtTest,
  onDropoutCommit,
  onTCommit,
  onToggleAtTest,
  onReseed,
  isBlocked,
  isPredicting,
}: {
  dropoutP: number;
  T: number;
  dropoutAtTest: boolean;
  onDropoutCommit: (next: number) => void;
  onTCommit: (next: number) => void;
  onToggleAtTest: (next: boolean) => void;
  onReseed: () => void;
  isBlocked: boolean;
  isPredicting: boolean;
}) {
  const [localDrop, setLocalDrop] = useState(dropoutP);
  const [localT, setLocalT] = useState(T);
  useEffect(() => setLocalDrop(dropoutP), [dropoutP]);
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
        <span>Dropout (1−ρ):</span>
        <input
          type="range"
          min={0}
          max={0.5}
          step={0.01}
          value={localDrop}
          onChange={(e) => setLocalDrop(Number(e.target.value))}
          onMouseUp={(e) => onDropoutCommit(Number((e.target as HTMLInputElement).value))}
          onTouchEnd={(e) => onDropoutCommit(Number((e.target as HTMLInputElement).value))}
          onKeyUp={(e) => onDropoutCommit(Number((e.target as HTMLInputElement).value))}
          disabled={isBlocked}
          style={{ width: '120px' }}
          aria-label="Dropout rate"
        />
        <span style={{ fontVariantNumeric: 'tabular-nums', minWidth: '2.5em' }}>
          {localDrop.toFixed(2)}
        </span>
      </label>
      <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}>
        <span>T (samples):</span>
        {/* T re-prediction is cheap (<100 ms per tick), so commit live on
            every onChange — no mouseup-only debounce. The dropout-rate slider
            above keeps the mouseup-commit pattern because retraining is ~5 s. */}
        <input
          type="range"
          min={10}
          max={200}
          step={5}
          value={localT}
          onChange={(e) => {
            const next = Number(e.target.value);
            setLocalT(next);
            onTCommit(next);
          }}
          disabled={isBlocked}
          style={{ width: '120px' }}
          aria-label="Forward-pass count T"
        />
        <span style={{ fontVariantNumeric: 'tabular-nums', minWidth: '2em' }}>{localT}</span>
      </label>
      <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}>
        <input
          type="checkbox"
          checked={dropoutAtTest}
          onChange={(e) => onToggleAtTest(e.target.checked)}
          disabled={isBlocked}
        />
        <span>Dropout on at test</span>
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
        Resample masks
      </button>
      {(isBlocked || isPredicting) && (
        <span style={{ color: 'var(--color-text-muted, #888)', fontStyle: 'italic' }}>
          {isBlocked ? 'training…' : 'sampling…'}
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
  dropoutAtTest,
  width,
}: {
  cached: CachedRun;
  dropoutAtTest: boolean;
  width: number;
}) {
  const layoutWidth = Math.max(width, 320);
  const stack = layoutWidth < 720;
  const panelW = stack ? layoutWidth : (layoutWidth - 32) / 3;
  const panelH = 240;

  // Mean and std are functions of (predsMC, T) when dropout-at-test is on.
  // When it's off, panel (a) is the deterministic predictor and panel (b) is
  // identically zero (no variance without test-time stochasticity).
  const mean = useMemo(
    () => (dropoutAtTest ? meanOverK(cached.predsMC, cached.T) : cached.predDet),
    [dropoutAtTest, cached.predsMC, cached.T, cached.predDet],
  );
  const std = useMemo(
    () =>
      dropoutAtTest
        ? stdOverK(cached.predsMC, cached.T)
        : new Float32Array(GRID_RES * GRID_RES),
    [dropoutAtTest, cached.predsMC, cached.T],
  );

  const meanURL = useMemo(
    () =>
      canvasFromColor(
        mathRowMajorToCanvas(mean, GRID_RES),
        GRID_RES,
        d3.scaleSequential(d3.interpolateRdBu).domain([1, 0]),
      ),
    [mean],
  );
  const stdURL = useMemo(() => {
    const sMax = maxFinite(std, 0.001);
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
        title={dropoutAtTest ? '(a) MC-dropout mean' : '(a) Deterministic prediction'}
        width={panelW}
        height={panelH}
      >
        <Heatmap imgURL={meanURL} width={panelW} height={panelH} data={cached.data} />
      </Panel>
      <Panel
        title={dropoutAtTest ? '(b) MC-dropout std' : '(b) No variance (deterministic)'}
        width={panelW}
        height={panelH}
      >
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
          dropoutAtTest
            ? `(c) ${Math.min(SAMPLED_CONTOURS_PER_PANEL, cached.T)} sampled boundaries`
            : '(c) Single deterministic boundary'
        }
        width={panelW}
        height={panelH}
      >
        <Boundaries
          predsMC={cached.predsMC}
          predDet={cached.predDet}
          T={cached.T}
          dropoutAtTest={dropoutAtTest}
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

function Boundaries({
  predsMC,
  predDet,
  T,
  dropoutAtTest,
  width,
  height,
  data,
}: {
  predsMC: Float32Array[];
  predDet: Float32Array;
  T: number;
  dropoutAtTest: boolean;
  width: number;
  height: number;
  data: TrainingData;
}) {
  const xS = d3.scaleLinear().domain([GRID_X_MIN, GRID_X_MAX]).range([0, width]);
  const yS = d3.scaleLinear().domain([GRID_Y_MIN, GRID_Y_MAX]).range([height, 0]);
  const paths = useMemo(() => {
    if (!dropoutAtTest) {
      return [isoContourPath(predDet, GRID_RES, width, height, 0.5)];
    }
    const n = Math.min(SAMPLED_CONTOURS_PER_PANEL, T);
    const out: string[] = [];
    for (let i = 0; i < n; i++) out.push(isoContourPath(predsMC[i], GRID_RES, width, height, 0.5));
    return out;
  }, [dropoutAtTest, predsMC, predDet, T, width, height]);

  return (
    <>
      <rect x={0} y={0} width={width} height={height} fill="var(--color-surface, #fafafa)" />
      {paths.map((d, k) => (
        <path
          key={k}
          d={d}
          fill="none"
          stroke={paletteBNN.dropout}
          strokeWidth={dropoutAtTest ? 1.2 : 1.8}
          opacity={dropoutAtTest ? 0.45 : 1}
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
