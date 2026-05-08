// =============================================================================
// SGHMCBNNViz.tsx
//
// §7 Stochastic-Gradient HMC. Mirrors the §6 SGLD viz with momentum + friction.
// Four panels — three spatial + one diagnostic — on a 60×60 Two Moons grid.
//
// Pedagogical hook: side-by-side comparison toggle overlays the §6 SGLD chain's
// autocorrelation function on panel (d), so the reader can see SGHMC's faster
// mixing (steeper ACF decay) directly. The toggle runs an SGLD chain at the
// same η for the comparison and caches it.
//
// Interactive controls:
//   - log₁₀ η slider (-7..-4, default ≈ -4.5 → η ≈ 3e-5).
//   - friction c slider (0.01..0.3, default 0.05).
//   - Samples slider S.
//   - Compare-with-SGLD toggle.
//   - Resample chain button.
// =============================================================================

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  makeMoonsData,
  mlpForward,
  mlpLayout,
  mlpTrain,
  paletteBNN,
  sgMCMCBNNTraining,
  type MLPArchSpec,
  type SGMCMCResult,
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

const ARCH: MLPArchSpec = {
  inputDim: 2,
  hiddenDims: [32, 32, 32],
  outputDim: 1,
  activation: 'relu',
};
const TRAINING_BASE = (seed: number): TrainingSpec => ({
  lr: 0.01,
  weightDecay: 1e-2,
  epochs: 80,
  optimizer: 'adam',
  seed,
});
const N_TRAIN = 300;
const TRAIN_NOISE = 0.2;
const DEFAULT_LOG10_ETA = -4.5;
const DEFAULT_FRICTION = 0.05;
const DEFAULT_S = 200;
const BURN_IN = 80;
const SAMPLED_CONTOURS_PER_PANEL = 20;
const { res: GRID_RES, xMin: GRID_X_MIN, xMax: GRID_X_MAX, yMin: GRID_Y_MIN, yMax: GRID_Y_MAX } =
  DEFAULT_GRID;

const FIGURE_PATH = '/images/topics/bayesian-neural-networks/07_sghmc_two_moons.png';
const ALT =
  'Three spatial panels and one diagnostic panel: (a) SGHMC predictive mean on Two Moons; (b) SGHMC predictive standard deviation; (c) sampled 0.5-probability decision boundaries from SGHMC chain in red; (d) single-component weight trace plus autocorrelation function with optional SGLD overlay for mixing comparison.';
const CAPTION =
  'Figure 7. SGHMC and the momentum-induced mixing speedup. (a, b) The predictive distribution is similar to SGLD’s at the same iteration budget. (c) Sampled decision boundaries. (d) The autocorrelation function decays faster for SGHMC than SGLD — the visual signature of why momentum helps: each effective sample takes fewer iterations of wall-clock to produce. Toggle the SGLD overlay on (d) to see the side-by-side mixing comparison.';
const ARIA = 'Figure 7: Interactive SGHMC predictive on Two Moons';

interface CachedMAP {
  data: TrainingData;
  map: TrainedMLP;
}

interface CachedChain {
  result: SGMCMCResult;
  gridProbs: Float32Array[];
  eta: number;
  friction: number;
  S: number;
  chainSeed: number;
}

interface CachedSGLDCompare {
  acf: number[];
  eta: number;
  S: number;
  chainSeed: number;
}

const yieldToBrowser = (): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, 0));

async function trainMAP(seed: number): Promise<CachedMAP> {
  await yieldToBrowser();
  const data = makeMoonsData(N_TRAIN, TRAIN_NOISE, seed);
  const map = mlpTrain(ARCH, TRAINING_BASE(seed), data);
  return { data, map };
}

async function runSGHMCChain(
  cachedMAP: CachedMAP,
  eta: number,
  friction: number,
  S: number,
  chainSeed: number,
): Promise<CachedChain> {
  await yieldToBrowser();
  const result = sgMCMCBNNTraining(ARCH, TRAINING_BASE(chainSeed), cachedMAP.data, {
    method: 'SGHMC',
    eta,
    batchSize: cachedMAP.data.n,
    burnIn: BURN_IN,
    samples: S,
    thin: 1,
    friction,
    warmStart: cachedMAP.map.weights,
    seed: chainSeed,
  });
  const layout = mlpLayout(ARCH);
  const grid = buildGridFlat(DEFAULT_GRID);
  const gridN = GRID_RES * GRID_RES;
  const gridProbs: Float32Array[] = [];
  for (let s = 0; s < result.weights.length; s++) {
    gridProbs.push(mlpForward(grid, gridN, result.weights[s], ARCH, layout, null, null));
  }
  return { result, gridProbs, eta, friction, S, chainSeed };
}

async function runSGLDForCompare(
  cachedMAP: CachedMAP,
  eta: number,
  S: number,
  chainSeed: number,
): Promise<CachedSGLDCompare> {
  await yieldToBrowser();
  // SGLD step size scaling: SGLD's η has different units than SGHMC's. The
  // SGLD chain converges with a much smaller η, so we use η_sgld = η/100 as
  // a rule-of-thumb to put both chains in a working regime simultaneously.
  const etaSGLD = eta / 100;
  const result = sgMCMCBNNTraining(ARCH, TRAINING_BASE(chainSeed + 31), cachedMAP.data, {
    method: 'SGLD',
    eta: etaSGLD,
    batchSize: cachedMAP.data.n,
    burnIn: BURN_IN,
    samples: S,
    thin: 1,
    warmStart: cachedMAP.map.weights,
    seed: chainSeed + 31,
  });
  return { acf: result.autocorrelation, eta: etaSGLD, S, chainSeed };
}

type CompState =
  | { kind: 'idle' }
  | { kind: 'training-map' }
  | { kind: 'sampling'; cachedMAP: CachedMAP; reason: 'first' | 'eta' | 'friction' | 'S' | 'reseed' }
  | { kind: 'ready'; cachedMAP: CachedMAP; cachedChain: CachedChain };

export default function SGHMCBNNViz() {
  const { ref, width } = useResizeObserver<HTMLElement>();
  const [state, setState] = useState<CompState>({ kind: 'idle' });
  const [logEta, setLogEta] = useState(DEFAULT_LOG10_ETA);
  const [friction, setFriction] = useState(DEFAULT_FRICTION);
  const [S, setS] = useState(DEFAULT_S);
  const [chainSeed, setChainSeed] = useState(13);
  const [compareSGLD, setCompareSGLD] = useState(false);
  const [sgldCompare, setSgldCompare] = useState<CachedSGLDCompare | null>(null);
  const ticketRef = useRef(0);
  const sgldTicketRef = useRef(0);
  const seedRef = useRef(7);

  useEffect(() => {
    let alive = true;
    setState({ kind: 'training-map' });
    trainMAP(seedRef.current).then((cachedMAP) => {
      if (!alive) return;
      setState({ kind: 'sampling', cachedMAP, reason: 'first' });
      const ticket = ++ticketRef.current;
      runSGHMCChain(cachedMAP, Math.pow(10, DEFAULT_LOG10_ETA), DEFAULT_FRICTION, DEFAULT_S, chainSeed).then(
        (cachedChain) => {
          if (alive && ticket === ticketRef.current) {
            setState({ kind: 'ready', cachedMAP, cachedChain });
          }
        },
      );
    });
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const launchChain = useCallback(
    (
      cachedMAP: CachedMAP,
      eta: number,
      fric: number,
      samples: number,
      seed: number,
      reason: 'eta' | 'friction' | 'S' | 'reseed',
    ) => {
      const ticket = ++ticketRef.current;
      setState({ kind: 'sampling', cachedMAP, reason });
      runSGHMCChain(cachedMAP, eta, fric, samples, seed).then((cachedChain) => {
        if (ticket === ticketRef.current) setState({ kind: 'ready', cachedMAP, cachedChain });
      });
    },
    [],
  );

  // When user toggles compareSGLD on, run an SGLD chain in the background.
  useEffect(() => {
    if (!compareSGLD) return;
    if (state.kind !== 'ready') return;
    const ticket = ++sgldTicketRef.current;
    runSGLDForCompare(state.cachedMAP, Math.pow(10, logEta), S, chainSeed).then((cmp) => {
      if (ticket === sgldTicketRef.current) setSgldCompare(cmp);
    });
  }, [compareSGLD, state, logEta, S, chainSeed]);

  const handleLogEtaCommit = (next: number) => {
    if (Math.abs(next - logEta) < 1e-6) return;
    setLogEta(next);
    if (state.kind === 'ready')
      launchChain(state.cachedMAP, Math.pow(10, next), friction, S, chainSeed, 'eta');
  };
  const handleFrictionCommit = (next: number) => {
    if (Math.abs(next - friction) < 1e-6) return;
    setFriction(next);
    if (state.kind === 'ready')
      launchChain(state.cachedMAP, Math.pow(10, logEta), next, S, chainSeed, 'friction');
  };
  const handleSCommit = (next: number) => {
    if (next === S) return;
    setS(next);
    if (state.kind === 'ready')
      launchChain(state.cachedMAP, Math.pow(10, logEta), friction, next, chainSeed, 'S');
  };
  const handleReseed = () => {
    const next = chainSeed + 1;
    setChainSeed(next);
    if (state.kind === 'ready')
      launchChain(state.cachedMAP, Math.pow(10, logEta), friction, S, next, 'reseed');
  };

  const isBlocked = state.kind === 'training-map' || state.kind === 'sampling' || state.kind === 'idle';
  const cachedMAP = state.kind === 'ready' || state.kind === 'sampling' ? state.cachedMAP : null;
  const cachedChain = state.kind === 'ready' ? state.cachedChain : null;

  return (
    <figure
      ref={ref as React.RefObject<HTMLElement>}
      aria-label={ARIA}
      style={{ width: '100%', margin: '2rem auto', display: 'flex', flexDirection: 'column', alignItems: 'center' }}
    >
      <div style={{ width: '100%', maxWidth: '880px' }}>
        {!cachedMAP || !cachedChain ? (
          <FallbackWithBanner
            label={
              state.kind === 'training-map'
                ? 'Training MAP for SGHMC warm-start (~5 s)…'
                : state.kind === 'sampling' && state.reason === 'first'
                  ? 'Running initial SGHMC chain (~3 s)…'
                  : state.kind === 'sampling'
                    ? 'Re-running SGHMC chain…'
                    : 'Loading…'
            }
          />
        ) : (
          <>
            <Controls
              logEta={logEta}
              friction={friction}
              S={S}
              compareSGLD={compareSGLD}
              onLogEtaCommit={handleLogEtaCommit}
              onFrictionCommit={handleFrictionCommit}
              onSCommit={handleSCommit}
              onCompareToggle={setCompareSGLD}
              onReseed={handleReseed}
              isBlocked={isBlocked}
            />
            <Panels
              cachedMAP={cachedMAP}
              cachedChain={cachedChain}
              compareSGLD={compareSGLD}
              sgldCompare={sgldCompare}
              S={S}
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

function Controls({
  logEta,
  friction,
  S,
  compareSGLD,
  onLogEtaCommit,
  onFrictionCommit,
  onSCommit,
  onCompareToggle,
  onReseed,
  isBlocked,
}: {
  logEta: number;
  friction: number;
  S: number;
  compareSGLD: boolean;
  onLogEtaCommit: (next: number) => void;
  onFrictionCommit: (next: number) => void;
  onSCommit: (next: number) => void;
  onCompareToggle: (next: boolean) => void;
  onReseed: () => void;
  isBlocked: boolean;
}) {
  const [localEta, setLocalEta] = useState(logEta);
  const [localFric, setLocalFric] = useState(friction);
  const [localS, setLocalS] = useState(S);
  useEffect(() => setLocalEta(logEta), [logEta]);
  useEffect(() => setLocalFric(friction), [friction]);
  useEffect(() => setLocalS(S), [S]);
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
          min={-7}
          max={-4}
          step={0.1}
          value={localEta}
          onChange={(e) => setLocalEta(Number(e.target.value))}
          onMouseUp={(e) => onLogEtaCommit(Number((e.target as HTMLInputElement).value))}
          onTouchEnd={(e) => onLogEtaCommit(Number((e.target as HTMLInputElement).value))}
          onKeyUp={(e) => onLogEtaCommit(Number((e.target as HTMLInputElement).value))}
          style={{ width: '120px' }}
          disabled={isBlocked}
          aria-label="SGHMC step size log10 eta"
        />
        <span style={{ fontVariantNumeric: 'tabular-nums', minWidth: '4em' }}>
          η={Math.pow(10, localEta).toExponential(1)}
        </span>
      </label>
      <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}>
        <span>c (friction):</span>
        <input
          type="range"
          min={0.01}
          max={0.3}
          step={0.01}
          value={localFric}
          onChange={(e) => setLocalFric(Number(e.target.value))}
          onMouseUp={(e) => onFrictionCommit(Number((e.target as HTMLInputElement).value))}
          onTouchEnd={(e) => onFrictionCommit(Number((e.target as HTMLInputElement).value))}
          onKeyUp={(e) => onFrictionCommit(Number((e.target as HTMLInputElement).value))}
          style={{ width: '110px' }}
          disabled={isBlocked}
          aria-label="SGHMC friction c"
        />
        <span style={{ fontVariantNumeric: 'tabular-nums', minWidth: '2.5em' }}>
          {localFric.toFixed(2)}
        </span>
      </label>
      <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}>
        <span>S (samples):</span>
        <input
          type="range"
          min={50}
          max={400}
          step={10}
          value={localS}
          onChange={(e) => setLocalS(Number(e.target.value))}
          onMouseUp={(e) => onSCommit(Number((e.target as HTMLInputElement).value))}
          onTouchEnd={(e) => onSCommit(Number((e.target as HTMLInputElement).value))}
          onKeyUp={(e) => onSCommit(Number((e.target as HTMLInputElement).value))}
          style={{ width: '120px' }}
          disabled={isBlocked}
          aria-label="Number of SGHMC samples"
        />
        <span style={{ fontVariantNumeric: 'tabular-nums', minWidth: '2.5em' }}>{localS}</span>
      </label>
      <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}>
        <input
          type="checkbox"
          checked={compareSGLD}
          onChange={(e) => onCompareToggle(e.target.checked)}
          disabled={isBlocked}
        />
        <span>Compare with SGLD</span>
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
        <span style={{ color: 'var(--color-text-muted, #888)', fontStyle: 'italic' }}>sampling…</span>
      )}
    </div>
  );
}

function Panels({
  cachedMAP,
  cachedChain,
  compareSGLD,
  sgldCompare,
  S,
  width,
}: {
  cachedMAP: CachedMAP;
  cachedChain: CachedChain;
  compareSGLD: boolean;
  sgldCompare: CachedSGLDCompare | null;
  S: number;
  width: number;
}) {
  const layoutWidth = Math.max(width, 320);
  const stack = layoutWidth < 720;
  const spatialW = stack ? layoutWidth : (layoutWidth - 32) / 3;
  const spatialH = 220;
  const diagW = layoutWidth;
  const diagH = 180;

  const effectiveS = Math.min(S, cachedChain.gridProbs.length);
  const mean = useMemo(() => meanOverK(cachedChain.gridProbs, effectiveS), [cachedChain.gridProbs, effectiveS]);
  const std = useMemo(() => stdOverK(cachedChain.gridProbs, effectiveS), [cachedChain.gridProbs, effectiveS]);
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
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      <div
        style={{
          display: 'flex',
          flexDirection: stack ? 'column' : 'row',
          gap: stack ? '12px' : '16px',
          alignItems: stack ? 'center' : 'flex-start',
        }}
      >
        <Panel title="(a) SGHMC mean" width={spatialW} height={spatialH}>
          <Heatmap imgURL={meanURL} width={spatialW} height={spatialH} data={cachedMAP.data} />
        </Panel>
        <Panel title="(b) SGHMC std" width={spatialW} height={spatialH}>
          <Heatmap imgURL={stdURL} width={spatialW} height={spatialH} data={cachedMAP.data} dataStroke="rgba(255,255,255,0.85)" />
        </Panel>
        <Panel title={`(c) ${Math.min(SAMPLED_CONTOURS_PER_PANEL, effectiveS)} sampled boundaries`} width={spatialW} height={spatialH}>
          <Boundaries
            gridProbs={cachedChain.gridProbs}
            S={effectiveS}
            width={spatialW}
            height={spatialH}
            data={cachedMAP.data}
            stroke={paletteBNN.sghmc}
          />
        </Panel>
      </div>
      <Panel
        title={
          compareSGLD
            ? '(d) Trace + ACF (SGHMC red, SGLD purple — fewer iterations to mix)'
            : '(d) Trace + autocorrelation'
        }
        width={diagW}
        height={diagH}
      >
        <TraceACF
          trace={cachedChain.result.weightTrace}
          acf={cachedChain.result.autocorrelation}
          compareACF={compareSGLD ? sgldCompare?.acf ?? null : null}
          width={diagW}
          height={diagH}
          stroke={paletteBNN.sghmc}
          compareStroke={paletteBNN.sgld}
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
  stroke,
}: {
  gridProbs: Float32Array[];
  S: number;
  width: number;
  height: number;
  data: TrainingData;
  stroke: string;
}) {
  const xS = d3.scaleLinear().domain([GRID_X_MIN, GRID_X_MAX]).range([0, width]);
  const yS = d3.scaleLinear().domain([GRID_Y_MIN, GRID_Y_MAX]).range([height, 0]);
  const paths = useMemo(() => {
    const n = Math.min(SAMPLED_CONTOURS_PER_PANEL, S);
    const step = Math.max(1, Math.floor(S / n));
    const out: string[] = [];
    for (let i = 0; i < n; i++) {
      out.push(isoContourPath(gridProbs[i * step], GRID_RES, width, height, 0.5));
    }
    return out;
  }, [gridProbs, S, width, height]);

  return (
    <>
      <rect x={0} y={0} width={width} height={height} fill="var(--color-surface, #fafafa)" />
      {paths.map((d, k) => (
        <path key={k} d={d} fill="none" stroke={stroke} strokeWidth={1.2} opacity={0.45} />
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

function TraceACF({
  trace,
  acf,
  compareACF,
  width,
  height,
  stroke,
  compareStroke,
}: {
  trace: number[];
  acf: number[];
  compareACF: number[] | null;
  width: number;
  height: number;
  stroke: string;
  compareStroke: string;
}) {
  const halfW = (width - 24) / 2;
  const margin = { top: 18, right: 14, bottom: 28, left: 40 };
  const innerW = halfW - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  const tExt = d3.extent(trace) as [number, number];
  const traceX = d3.scaleLinear().domain([0, trace.length - 1]).range([0, innerW]);
  const traceY = d3.scaleLinear().domain([tExt[0]!, tExt[1]!]).range([innerH, 0]);
  const traceLine = d3
    .line<number>()
    .x((_, i) => traceX(i))
    .y((v) => traceY(v));

  const maxLag = Math.max(acf.length, compareACF?.length ?? 0);
  const acfX = d3.scaleLinear().domain([0, maxLag - 1]).range([0, innerW]);
  const acfY = d3.scaleLinear().domain([-0.2, 1]).range([innerH, 0]);

  return (
    <g>
      <g transform={`translate(${margin.left},${margin.top})`}>
        <rect x={-margin.left + 4} y={-margin.top + 4} width={halfW - 8} height={height - 8} fill="var(--color-surface, #fafafa)" />
        <path d={traceLine(trace) ?? ''} fill="none" stroke={stroke} strokeWidth={1.1} opacity={0.85} />
        <line x1={0} x2={innerW} y1={innerH} y2={innerH} stroke="var(--color-text-muted, #888)" />
        <line x1={0} x2={0} y1={0} y2={innerH} stroke="var(--color-text-muted, #888)" />
        <text x={innerW / 2} y={innerH + 18} textAnchor="middle" fontSize={10} fill="var(--color-text, #333)">
          iteration
        </text>
        <text transform={`translate(${-30}, ${innerH / 2}) rotate(-90)`} textAnchor="middle" fontSize={10} fill="var(--color-text, #333)">
          weight[0]
        </text>
        <text x={innerW / 2} y={-6} textAnchor="middle" fontSize={11} fontWeight={600} fill="var(--color-text, #333)">
          Trace
        </text>
      </g>
      <g transform={`translate(${halfW + 24 + margin.left},${margin.top})`}>
        <rect x={-margin.left + 4} y={-margin.top + 4} width={halfW - 8} height={height - 8} fill="var(--color-surface, #fafafa)" />
        <line x1={0} x2={innerW} y1={acfY(0)} y2={acfY(0)} stroke="var(--color-text-muted, #888)" strokeDasharray="3 3" />
        {acf.map((v, k) => (
          <line key={`s-${k}`} x1={acfX(k)} x2={acfX(k)} y1={acfY(0)} y2={acfY(v)} stroke={stroke} strokeWidth={1.5} />
        ))}
        {compareACF &&
          compareACF.map((v, k) => (
            <line
              key={`c-${k}`}
              x1={acfX(k) + 1.2}
              x2={acfX(k) + 1.2}
              y1={acfY(0)}
              y2={acfY(v)}
              stroke={compareStroke}
              strokeWidth={1.2}
              opacity={0.7}
            />
          ))}
        <line x1={0} x2={innerW} y1={innerH} y2={innerH} stroke="var(--color-text-muted, #888)" />
        <line x1={0} x2={0} y1={0} y2={innerH} stroke="var(--color-text-muted, #888)" />
        <text x={innerW / 2} y={innerH + 18} textAnchor="middle" fontSize={10} fill="var(--color-text, #333)">
          lag
        </text>
        <text transform={`translate(${-30}, ${innerH / 2}) rotate(-90)`} textAnchor="middle" fontSize={10} fill="var(--color-text, #333)">
          ACF
        </text>
        <text x={innerW / 2} y={-6} textAnchor="middle" fontSize={11} fontWeight={600} fill="var(--color-text, #333)">
          Autocorrelation
        </text>
      </g>
    </g>
  );
}
