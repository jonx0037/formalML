// =============================================================================
// LaplaceBNNViz.tsx
//
// §3 Laplace approximation. Three panels on a 60×60 Two Moons grid:
//   (a) Predictive mean from S Laplace-sampled weight vectors — RdBu_r heatmap.
//       Visually similar to the §1 point estimate (Laplace doesn't change the
//       MAP, only adds a Gaussian envelope around it).
//   (b) Predictive standard deviation across the S samples — viridis. Bright
//       off-distribution, dark near data.
//   (c) S sampled 0.5-probability decision boundaries overlaid (paletteBNN.laplace).
//
// Interactive controls:
//   - Prior-scale slider τ² (10⁻¹–10³, log scale, default 10²). Updates λ = 1/τ²
//     and recomputes the last-layer Hessian + Cholesky in milliseconds. The
//     MAP is held fixed at the default τ² for interactivity (CLAUDE.md note:
//     under reasonable τ² the MAP shift is small).
//   - Sample-count slider S (10–100, default 30) — cheap re-sampling from the
//     cached Cholesky factor.
//   - Resample button — re-rolls the sampling seed.
//
// Implementation: lastLayerLaplaceFromMAP from bayesian-ml.ts. The MAP is
// trained once on first hydration (~5 s); subsequent τ² and S changes only
// hit the closed-form Hessian recomputation and back-substitution sampling.
// =============================================================================

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  lastLayerLaplaceFromMAP,
  makeMoonsData,
  mlpForward,
  mlpLayout,
  mlpTrain,
  paletteBNN,
  type LaplaceResult,
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

const ARCH: MLPArchSpec = {
  inputDim: 2,
  hiddenDims: [32, 32, 32],
  outputDim: 1,
  activation: 'relu',
};
const TRAINING_BASE = (seed: number, weightDecay: number): TrainingSpec => ({
  lr: 0.01,
  weightDecay,
  epochs: 80,
  optimizer: 'adam',
  seed,
});
const N_TRAIN = 300;
const TRAIN_NOISE = 0.2;
const DEFAULT_TAU2 = 1e2;
const DEFAULT_S = 30;
const SAMPLED_CONTOURS_PER_PANEL = 20;
const { res: GRID_RES, xMin: GRID_X_MIN, xMax: GRID_X_MAX, yMin: GRID_Y_MIN, yMax: GRID_Y_MAX } =
  DEFAULT_GRID;

const FIGURE_PATH = '/images/topics/bayesian-neural-networks/03_laplace_two_moons.png';
const ALT =
  'Three panels on Two Moons data: panel (a) Laplace-BNN predictive mean RdBu_r heatmap visually indistinguishable from the §1 point-estimate predictive; panel (b) Laplace predictive standard deviation viridis heatmap with dark regions hugging the data and bright regions far from any training point; panel (c) sampled Laplace 0.5-probability decision boundaries overlaid in translucent blue.';
const CAPTION =
  'Figure 3. The Laplace BNN. (a) The predictive mean is the point estimate — Laplace doesn’t move the MAP. (b) The predictive standard deviation grows away from the data, recovering the §1 desideratum from a single trained model’s local Gaussian. (c) Sampled decision boundaries fan out within one mode of the loss landscape, missing the multi-mode structure §5 will recover. Slide τ² (prior scale, log axis) to widen or sharpen the posterior; slide S to trade compute for sampling smoothness; reseed for a fresh sample set.';
const ARIA = 'Figure 3: Interactive Laplace BNN predictive on Two Moons';

// -----------------------------------------------------------------------------
// Cached MAP (trained once)
// -----------------------------------------------------------------------------

interface CachedMAP {
  data: TrainingData;
  map: TrainedMLP;
  seed: number;
}

const yieldToBrowser = (): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, 0));

async function trainMAP(seed: number, defaultLambda: number): Promise<CachedMAP> {
  await yieldToBrowser();
  const data = makeMoonsData(N_TRAIN, TRAIN_NOISE, seed);
  const map = mlpTrain(ARCH, TRAINING_BASE(seed, defaultLambda), data);
  return { data, map, seed };
}

/** Forward each Laplace-sampled weight vector through the prediction grid. */
function predictGridFromSamples(
  samples: Float32Array[],
  data: TrainingData,
  gridFlat: Float32Array,
  gridN: number,
): Float32Array[] {
  void data;
  const layout = mlpLayout(ARCH);
  const out: Float32Array[] = [];
  for (let s = 0; s < samples.length; s++) {
    out.push(mlpForward(gridFlat, gridN, samples[s], ARCH, layout, null, null));
  }
  return out;
}

// -----------------------------------------------------------------------------
// State
// -----------------------------------------------------------------------------

type CompState =
  | { kind: 'idle' }
  | { kind: 'training' }
  | { kind: 'ready'; cached: CachedMAP };

// -----------------------------------------------------------------------------
// Component
// -----------------------------------------------------------------------------

export default function LaplaceBNNViz() {
  const { ref, width } = useResizeObserver<HTMLElement>();
  const [state, setState] = useState<CompState>({ kind: 'idle' });
  const [tau2, setTau2] = useState(DEFAULT_TAU2);
  const [S, setS] = useState(DEFAULT_S);
  const [sampleSeed, setSampleSeed] = useState(0);
  const seedRef = useRef(7);

  useEffect(() => {
    let alive = true;
    setState({ kind: 'training' });
    trainMAP(seedRef.current, 1 / DEFAULT_TAU2).then((cached) => {
      if (alive) setState({ kind: 'ready', cached });
    });
    return () => {
      alive = false;
    };
  }, []);

  const cached = state.kind === 'ready' ? state.cached : null;

  // Recompute Hessian (closed form) when τ² changes — fast (~ms).
  const laplace = useMemo<LaplaceResult | null>(() => {
    if (!cached) return null;
    return lastLayerLaplaceFromMAP(cached.map, cached.data, 1 / tau2);
  }, [cached, tau2]);

  // Sample S weight vectors from the Cholesky factor.
  const samples = useMemo<Float32Array[] | null>(() => {
    if (!laplace) return null;
    return laplace.sampleWeights(S, sampleSeed * 31 + 1);
  }, [laplace, S, sampleSeed]);

  // Forward each sampled weight through the prediction grid (cheap: S × ~700k flops).
  const gridProbs = useMemo<Float32Array[] | null>(() => {
    if (!cached || !samples) return null;
    const grid = buildGridFlat(DEFAULT_GRID);
    return predictGridFromSamples(samples, cached.data, grid, GRID_RES * GRID_RES);
  }, [cached, samples]);

  const handleReseed = useCallback(() => setSampleSeed((s) => s + 1), []);

  const isBlocked = state.kind === 'training' || state.kind === 'idle';

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
        {!cached || !laplace || !gridProbs ? (
          <FallbackWithBanner label="Training MAP for Laplace approximation (~5 s)…" />
        ) : (
          <>
            <Controls
              tau2={tau2}
              S={S}
              onTau2Change={setTau2}
              onSChange={setS}
              onReseed={handleReseed}
              conditionNumber={laplace.conditionNumber}
              isBlocked={isBlocked}
            />
            <Panels cached={cached} gridProbs={gridProbs} S={S} width={width} />
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
  tau2,
  S,
  onTau2Change,
  onSChange,
  onReseed,
  conditionNumber,
  isBlocked,
}: {
  tau2: number;
  S: number;
  onTau2Change: (next: number) => void;
  onSChange: (next: number) => void;
  onReseed: () => void;
  conditionNumber: number;
  isBlocked: boolean;
}) {
  // Slider value is log10(τ²); maps -1..3 → 10⁻¹..10³.
  const logVal = Math.log10(tau2);
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
        <span>log₁₀ τ²:</span>
        <input
          type="range"
          min={-1}
          max={3}
          step={0.1}
          value={logVal}
          onChange={(e) => onTau2Change(Math.pow(10, Number(e.target.value)))}
          style={{ width: '140px' }}
          disabled={isBlocked}
          aria-label="Prior scale (log10 tau-squared)"
        />
        <span style={{ fontVariantNumeric: 'tabular-nums', minWidth: '4em' }}>
          τ²={tau2.toExponential(1)}
        </span>
      </label>
      <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}>
        <span>S (samples):</span>
        <input
          type="range"
          min={10}
          max={100}
          step={5}
          value={S}
          onChange={(e) => onSChange(Number(e.target.value))}
          style={{ width: '120px' }}
          disabled={isBlocked}
          aria-label="Number of Laplace samples"
        />
        <span style={{ fontVariantNumeric: 'tabular-nums', minWidth: '2em' }}>{S}</span>
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
        Resample
      </button>
      <span style={{ color: 'var(--color-text-muted, #888)', fontSize: '0.8rem' }}>
        κ(H) = {conditionNumber.toExponential(1)}
      </span>
    </div>
  );
}

// -----------------------------------------------------------------------------
// Panels
// -----------------------------------------------------------------------------

function Panels({
  cached,
  gridProbs,
  S,
  width,
}: {
  cached: CachedMAP;
  gridProbs: Float32Array[];
  S: number;
  width: number;
}) {
  const layoutWidth = Math.max(width, 320);
  const stack = layoutWidth < 720;
  const panelW = stack ? layoutWidth : (layoutWidth - 32) / 3;
  const panelH = 240;

  const mean = useMemo(() => meanOverK(gridProbs, S), [gridProbs, S]);
  const std = useMemo(() => stdOverK(gridProbs, S), [gridProbs, S]);

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
      <Panel title="(a) Laplace mean" width={panelW} height={panelH}>
        <Heatmap imgURL={meanURL} width={panelW} height={panelH} data={cached.data} />
      </Panel>
      <Panel title="(b) Laplace std" width={panelW} height={panelH}>
        <Heatmap
          imgURL={stdURL}
          width={panelW}
          height={panelH}
          data={cached.data}
          dataStroke="rgba(255,255,255,0.85)"
        />
      </Panel>
      <Panel title={`(c) ${Math.min(SAMPLED_CONTOURS_PER_PANEL, S)} sampled boundaries`} width={panelW} height={panelH}>
        <Boundaries
          gridProbs={gridProbs}
          S={S}
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
      <image href={imgURL} x={0} y={0} width={width} height={height} preserveAspectRatio="none" />
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
  gridProbs,
  S,
  width,
  height,
  data,
}: {
  gridProbs: Float32Array[];
  S: number;
  width: number;
  height: number;
  data: TrainingData;
}) {
  const xS = d3.scaleLinear().domain([GRID_X_MIN, GRID_X_MAX]).range([0, width]);
  const yS = d3.scaleLinear().domain([GRID_Y_MIN, GRID_Y_MAX]).range([height, 0]);
  const paths = useMemo(() => {
    const n = Math.min(SAMPLED_CONTOURS_PER_PANEL, S);
    const out: string[] = [];
    for (let i = 0; i < n; i++) out.push(isoContourPath(gridProbs[i], GRID_RES, width, height, 0.5));
    return out;
  }, [gridProbs, S, width, height]);

  return (
    <>
      <rect x={0} y={0} width={width} height={height} fill="var(--color-surface, #fafafa)" />
      {paths.map((d, k) => (
        <path key={k} d={d} fill="none" stroke={paletteBNN.laplace} strokeWidth={1.2} opacity={0.5} />
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
