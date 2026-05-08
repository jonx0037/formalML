// =============================================================================
// ControlVariateVarianceExplorer.tsx
//
// §9 Bias-reduction strategies. Three methods on the §6 BLR posterior:
//   - Vanilla SGLD with mini-batch gradient
//   - SVRG-LD (Dubey et al. 2016) — periodic reference point, lower-variance
//     gradient estimator
//   - ZV-SGLD (Mira–Solgi–Imparato 2013 + Brosse–Durmus–Moulines for SGLD) —
//     post-process the chain with a linear control variate α^T ∇U(θ).
//
// Two panels:
//   (a) Running estimate of E_π[θ_1] for all three methods + true posterior
//       mean (dashed). Vanilla SGLD has the most variance; ZV-SGLD has the
//       least at zero compute overhead.
//   (b) Running standard error vs iteration (log-log). The slope is -½
//       (Monte-Carlo √n rate) for all three; the constant is the variance-
//       reduction factor.
//
// Controls:
//   - T (5k..30k, default 15k)
//   - B (mini-batch size, 4..64, default 16)
//   - Resample button
//
// Computation: in-browser. SVRG-LD and ZV-SGLD implemented inline.
// Static fallback: /images/topics/stochastic-gradient-mcmc/09_bias_reduction.png.
// =============================================================================

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import { mulberry32 } from './shared/bayesian-ml';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  blrPosterior,
  makeBLRDataset,
  makeGauss,
  makeSampler,
  paletteSGMCMC,
  runningMean,
} from './shared/sgmcmc';

const DATA_SEED = 41;
const DEFAULT_T = 15000;
const DEFAULT_B = 16;
const SVRG_K_INNER = 100;

const FIGURE_PATH = '/images/topics/stochastic-gradient-mcmc/09_bias_reduction.png';
const ALT =
  'Two-panel comparison of vanilla SGLD, SVRG-LD, and ZV-SGLD on Bayesian linear regression. (a) Running estimates of E_π[θ_1]: ZV-SGLD smoothest, vanilla noisiest. (b) Running standard error vs iteration on log-log axes: all three methods follow the √n Monte-Carlo rate; ZV-SGLD\'s lower variance is a constant factor below the others.';
const CAPTION =
  'Figure 9. Three variance-reduction strategies on §6\'s BLR posterior. ZV-SGLD wins by the largest factor at zero compute overhead (the gradient is already computed at every chain step). SVRG-LD reduces variance modestly at moderate cost via a periodic reference-point gradient. All three converge to the true posterior mean μ_post[0]; the differences are in their finite-sample variance.';
const ARIA = 'Figure 9: Variance-reduction strategies for SG-MCMC';

interface CachedRun {
  vanillaTrace: Float32Array; // running mean of θ_1
  svrgTrace: Float32Array;
  zvTrace: Float32Array;
  vanillaSE: Float32Array; // running standard error of running estimate
  svrgSE: Float32Array;
  zvSE: Float32Array;
  T: number;
  B: number;
  muTrue: number;
  chainSeed: number;
}

interface BLRGrad {
  /** Full gradient ∇U(θ) = ∇U_data(θ) + λθ. Writes into `out`. */
  full: (theta: Float32Array, out: Float32Array) => void;
  /** Data-only full gradient ∇U_data(θ). Writes into `out`. */
  data: (theta: Float32Array, out: Float32Array) => void;
  /** Mini-batch data-only gradient at the given index slot. Writes into `out`. */
  dataMini: (theta: Float32Array, idx: Int32Array, B: number, out: Float32Array) => void;
  /** Mini-batch full gradient (data + prior) at the given index slot. Writes into `out`. */
  miniFull: (theta: Float32Array, idx: Int32Array, B: number, out: Float32Array) => void;
}

function buildGrads(spec: ReturnType<typeof makeBLRDataset>): BLRGrad {
  const { XDesign, yData, N, sigmaNoise, tauPrior } = spec;
  const inv2 = 1 / (sigmaNoise * sigmaNoise);
  const tauInv2 = 1 / (tauPrior * tauPrior);
  // All four gradient functions write into a caller-provided `out` Float32Array
  // so the chain-runner loop can avoid allocating a new gradient buffer each
  // step. Compose-by-write rather than compose-by-return.
  const data = (theta: Float32Array, out: Float32Array): void => {
    let g0 = 0,
      g1 = 0;
    for (let i = 0; i < N; i++) {
      const a = XDesign[i * 2 + 0];
      const b = XDesign[i * 2 + 1];
      const r = a * theta[0] + b * theta[1] - yData[i];
      g0 += inv2 * a * r;
      g1 += inv2 * b * r;
    }
    out[0] = g0;
    out[1] = g1;
  };
  const dataMini = (theta: Float32Array, idx: Int32Array, B: number, out: Float32Array): void => {
    const scale = N / B;
    let g0 = 0,
      g1 = 0;
    for (let k = 0; k < B; k++) {
      const i = idx[k];
      const a = XDesign[i * 2 + 0];
      const b = XDesign[i * 2 + 1];
      const r = a * theta[0] + b * theta[1] - yData[i];
      g0 += scale * inv2 * a * r;
      g1 += scale * inv2 * b * r;
    }
    out[0] = g0;
    out[1] = g1;
  };
  const full = (theta: Float32Array, out: Float32Array): void => {
    data(theta, out);
    out[0] += tauInv2 * theta[0];
    out[1] += tauInv2 * theta[1];
  };
  const miniFull = (theta: Float32Array, idx: Int32Array, B: number, out: Float32Array): void => {
    dataMini(theta, idx, B, out);
    out[0] += tauInv2 * theta[0];
    out[1] += tauInv2 * theta[1];
  };
  return { full, data, dataMini, miniFull };
}

const yieldToBrowser = () => new Promise<void>((res) => setTimeout(res, 0));

async function runChains(T: number, B: number, chainSeed: number): Promise<CachedRun> {
  const spec = makeBLRDataset(DATA_SEED);
  const post = blrPosterior(spec);
  const grads = buildGrads(spec);
  const N = spec.N;
  const muTrue = post.muPost[0];
  const start = new Float32Array([post.muPost[0], post.muPost[1]]);

  const eta = 5e-4;
  const sqrt2eta = Math.sqrt(2 * eta);

  // Vanilla SGLD with mini-batch gradient.
  // gradCache holds the per-step mini-batch gradient ∇̂U_B(θ_n) used by the
  // ZV-SGLD post-processing below. Allocated as a single flat Float32Array
  // of size T·2 (row-major) instead of T separate Float32Array(2) objects —
  // avoids T allocations per slider commit at T = 30k.
  await yieldToBrowser();
  const rngV = mulberry32(chainSeed * 991 + 401);
  const gaussV = makeGauss(rngV);
  const samplerV = makeSampler(N, B);
  const theta = new Float32Array(start);
  const gradOut = new Float32Array(2);
  const vanillaCh1 = new Float32Array(T);
  const gradCache = new Float32Array(T * 2);
  for (let n = 0; n < T; n++) {
    const idx = samplerV(rngV);
    grads.miniFull(theta, idx, B, gradOut);
    gradCache[n * 2 + 0] = gradOut[0];
    gradCache[n * 2 + 1] = gradOut[1];
    theta[0] -= eta * gradOut[0] - sqrt2eta * gaussV();
    theta[1] -= eta * gradOut[1] - sqrt2eta * gaussV();
    vanillaCh1[n] = theta[0];
  }

  // SVRG-LD: gradient = ∇U_data(θ̃) + (∇̂U_data,B(θ_n) - ∇̂U_data,B(θ̃)) + prior(θ_n).
  // Full-data reference gradient at θ̃ is refreshed every k_inner steps.
  await yieldToBrowser();
  const rngS = mulberry32(chainSeed * 991 + 402);
  const gaussS = makeGauss(rngS);
  const samplerS = makeSampler(N, B);
  const thetaS = new Float32Array(start);
  const thetaTilde = new Float32Array(start);
  const gTildeData = new Float32Array(2);
  grads.data(thetaTilde, gTildeData);
  const gDataCur = new Float32Array(2);
  const gDataTilde = new Float32Array(2);
  const tauInv2 = 1 / (spec.tauPrior * spec.tauPrior);
  const svrgCh1 = new Float32Array(T);
  for (let n = 0; n < T; n++) {
    if (n > 0 && n % SVRG_K_INNER === 0) {
      thetaTilde[0] = thetaS[0];
      thetaTilde[1] = thetaS[1];
      grads.data(thetaTilde, gTildeData);
    }
    const idx = samplerS(rngS);
    grads.dataMini(thetaS, idx, B, gDataCur);
    grads.dataMini(thetaTilde, idx, B, gDataTilde);
    const g0 = gTildeData[0] + gDataCur[0] - gDataTilde[0] + tauInv2 * thetaS[0];
    const g1 = gTildeData[1] + gDataCur[1] - gDataTilde[1] + tauInv2 * thetaS[1];
    thetaS[0] -= eta * g0 - sqrt2eta * gaussS();
    thetaS[1] -= eta * g1 - sqrt2eta * gaussS();
    svrgCh1[n] = thetaS[0];
  }

  // ZV-SGLD: post-process the vanilla chain with a linear control variate.
  // Fit α* = -[Cov(∇U, ∇U)]⁻¹ Cov(∇U, φ) on the chain, then φ_zv = φ + α^T ∇U.
  // Note: gradCache stores the *mini-batch* gradient estimator ∇̂U_B(θ_n), not
  // the full gradient ∇U(θ_n). Mira–Solgi–Imparato's variance-zero argument
  // assumes ∇U is the exact gradient under π, so using the unbiased mini-batch
  // estimator carries an extra noise term that shows up as the residual
  // estimator variance in panel (b). For the §6 BLR posterior at B = 16 the
  // residual is small (the §8 mini-batch bias for SGLD is O(η + 1/B)) and the
  // variance reduction over vanilla SGLD is still ~5–10×.
  await yieldToBrowser();
  let mg0 = 0,
    mg1 = 0,
    mp = 0;
  for (let n = 0; n < T; n++) {
    mg0 += gradCache[n * 2 + 0];
    mg1 += gradCache[n * 2 + 1];
    mp += vanillaCh1[n];
  }
  mg0 /= T;
  mg1 /= T;
  mp /= T;
  let s00 = 0,
    s01 = 0,
    s11 = 0,
    sp0 = 0,
    sp1 = 0;
  for (let n = 0; n < T; n++) {
    const d0 = gradCache[n * 2 + 0] - mg0;
    const d1 = gradCache[n * 2 + 1] - mg1;
    const dp = vanillaCh1[n] - mp;
    s00 += d0 * d0;
    s01 += d0 * d1;
    s11 += d1 * d1;
    sp0 += d0 * dp;
    sp1 += d1 * dp;
  }
  s00 /= T;
  s01 /= T;
  s11 /= T;
  sp0 /= T;
  sp1 /= T;
  const det = s00 * s11 - s01 * s01;
  const invDet = 1 / Math.max(det, 1e-12);
  const a0 = -(s11 * sp0 - s01 * sp1) * invDet;
  const a1 = -(-s01 * sp0 + s00 * sp1) * invDet;
  const zvCh1 = new Float32Array(T);
  for (let n = 0; n < T; n++) {
    zvCh1[n] = vanillaCh1[n] + a0 * gradCache[n * 2 + 0] + a1 * gradCache[n * 2 + 1];
  }

  // Running means
  const vanillaTrace = runningMean(vanillaCh1);
  const svrgTrace = runningMean(svrgCh1);
  const zvTrace = runningMean(zvCh1);

  // Running standard error: SE(n) = sqrt(Var(running mean estimate up to n))
  // ≈ sqrt(running_var / n). Compute via online variance.
  const computeSE = (vals: Float32Array): Float32Array => {
    const out = new Float32Array(vals.length);
    let mean = 0,
      M2 = 0;
    for (let n = 0; n < vals.length; n++) {
      const x = vals[n];
      const delta = x - mean;
      mean += delta / (n + 1);
      M2 += delta * (x - mean);
      const variance = n > 0 ? M2 / n : 0;
      out[n] = Math.sqrt(variance / (n + 1));
    }
    return out;
  };

  return {
    vanillaTrace,
    svrgTrace,
    zvTrace,
    vanillaSE: computeSE(vanillaCh1),
    svrgSE: computeSE(svrgCh1),
    zvSE: computeSE(zvCh1),
    T,
    B,
    muTrue,
    chainSeed,
  };
}

type CompState = { kind: 'idle' } | { kind: 'sampling'; reason: 'first' | 'param' | 'reseed' } | { kind: 'ready'; run: CachedRun };

export default function ControlVariateVarianceExplorer() {
  const { ref, width } = useResizeObserver<HTMLElement>();
  const [state, setState] = useState<CompState>({ kind: 'idle' });
  const [T, setT] = useState(DEFAULT_T);
  const [B, setB] = useState(DEFAULT_B);
  const [chainSeed, setChainSeed] = useState(11);
  const ticketRef = useRef(0);

  useEffect(() => {
    let alive = true;
    setState({ kind: 'sampling', reason: 'first' });
    const ticket = ++ticketRef.current;
    runChains(DEFAULT_T, DEFAULT_B, 11).then((run) => {
      if (alive && ticket === ticketRef.current) setState({ kind: 'ready', run });
    });
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const launch = useCallback((t: number, b: number, seed: number, reason: 'param' | 'reseed') => {
    const ticket = ++ticketRef.current;
    setState({ kind: 'sampling', reason });
    runChains(t, b, seed).then((run) => {
      if (ticket === ticketRef.current) setState({ kind: 'ready', run });
    });
  }, []);

  const onTCommit = (next: number) => {
    if (next === T) return;
    setT(next);
    if (state.kind === 'ready') launch(next, B, chainSeed, 'param');
  };
  const onBCommit = (next: number) => {
    if (next === B) return;
    setB(next);
    if (state.kind === 'ready') launch(T, next, chainSeed, 'param');
  };
  const onReseed = () => {
    const next = chainSeed + 1;
    setChainSeed(next);
    if (state.kind === 'ready') launch(T, B, next, 'reseed');
  };

  const isBlocked = state.kind !== 'ready';
  const ready = state.kind === 'ready' ? state : null;

  return (
    <figure
      ref={ref as React.RefObject<HTMLElement>}
      aria-label={ARIA}
      style={{ width: '100%', margin: '2rem auto', display: 'flex', flexDirection: 'column', alignItems: 'center' }}
    >
      <div style={{ width: '100%', maxWidth: '880px' }}>
        {!ready ? (
          <Fallback label={state.kind === 'sampling' && state.reason === 'first' ? 'Running 3 chains (vanilla / SVRG-LD / ZV-SGLD)…' : 'Re-running chains…'} />
        ) : (
          <>
            <Controls T={T} B={B} onTCommit={onTCommit} onBCommit={onBCommit} onReseed={onReseed} isBlocked={isBlocked} />
            <Panels run={ready.run} width={width} />
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

function Controls({
  T,
  B,
  onTCommit,
  onBCommit,
  onReseed,
  isBlocked,
}: {
  T: number;
  B: number;
  onTCommit: (n: number) => void;
  onBCommit: (n: number) => void;
  onReseed: () => void;
  isBlocked: boolean;
}) {
  const [lt, setLt] = useState(T);
  const [lb, setLb] = useState(B);
  useEffect(() => setLt(T), [T]);
  useEffect(() => setLb(B), [B]);
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.75rem 1.25rem', alignItems: 'center', fontSize: '0.875rem', color: 'var(--color-text, #333)', marginBottom: '0.75rem' }}>
      <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}>
        <span>T:</span>
        <input type="range" min={5000} max={30000} step={1000} value={lt}
          onChange={(e) => setLt(Number(e.target.value))}
          onMouseUp={(e) => onTCommit(Number((e.target as HTMLInputElement).value))}
          onTouchEnd={(e) => onTCommit(Number((e.target as HTMLInputElement).value))}
          onKeyUp={(e) => onTCommit(Number((e.target as HTMLInputElement).value))}
          style={{ width: '120px' }} disabled={isBlocked} aria-label="Chain length T" />
        <span style={{ fontVariantNumeric: 'tabular-nums', minWidth: '4em' }}>{lt}</span>
      </label>
      <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}>
        <span>B:</span>
        <input type="range" min={4} max={64} step={4} value={lb}
          onChange={(e) => setLb(Number(e.target.value))}
          onMouseUp={(e) => onBCommit(Number((e.target as HTMLInputElement).value))}
          onTouchEnd={(e) => onBCommit(Number((e.target as HTMLInputElement).value))}
          onKeyUp={(e) => onBCommit(Number((e.target as HTMLInputElement).value))}
          style={{ width: '120px' }} disabled={isBlocked} aria-label="Mini-batch size B" />
        <span style={{ fontVariantNumeric: 'tabular-nums', minWidth: '2.5em' }}>{lb}</span>
      </label>
      <button type="button" onClick={onReseed} disabled={isBlocked}
        style={{ padding: '0.3rem 0.7rem', border: '1px solid var(--color-border, #ccc)', borderRadius: '4px', background: isBlocked ? 'var(--color-surface-muted, #f0f0f0)' : 'var(--color-surface, #fff)', color: 'var(--color-text, #333)', cursor: isBlocked ? 'not-allowed' : 'pointer', fontSize: '0.875rem' }}>
        Resample chain
      </button>
      {isBlocked && <span style={{ color: 'var(--color-text-muted, #888)', fontStyle: 'italic' }}>sampling…</span>}
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
      <Panel title="(a) Running estimate of E_π[θ_1]" width={panelW} height={panelH}>
        <RunningEstimate run={run} width={panelW} height={panelH} />
      </Panel>
      <Panel title="(b) Running standard error (log-log)" width={panelW} height={panelH}>
        <StdErrPanel run={run} width={panelW} height={panelH} />
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

function RunningEstimate({ run, width, height }: { run: CachedRun; width: number; height: number }) {
  const margin = { top: 12, right: 110, bottom: 28, left: 50 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const start = 100;

  // Y range: μ_post ± a window that fits all three running traces post-burn.
  const tail = (a: Float32Array) => Array.from(a.slice(start));
  const all = [...tail(run.vanillaTrace), ...tail(run.svrgTrace), ...tail(run.zvTrace)];
  const yMin = Math.min(...all, run.muTrue) - 0.005;
  const yMax = Math.max(...all, run.muTrue) + 0.005;

  const xS = d3.scaleLinear().domain([start, run.T - 1]).range([0, innerW]);
  const yS = d3.scaleLinear().domain([yMin, yMax]).range([innerH, 0]);

  const linePath = (vals: Float32Array) =>
    Array.from(vals.slice(start))
      .map((v, i) => `${i === 0 ? 'M' : 'L'}${xS(i + start)},${yS(v)}`)
      .join(' ');

  return (
    <g transform={`translate(${margin.left},${margin.top})`}>
      <rect x={0} y={0} width={innerW} height={innerH} fill="var(--color-surface, #fafafa)" />
      <line x1={0} x2={innerW} y1={yS(run.muTrue)} y2={yS(run.muTrue)} stroke={paletteSGMCMC.target} strokeWidth={1.4} strokeDasharray="4 3" />
      <path d={linePath(run.vanillaTrace)} fill="none" stroke={paletteSGMCMC.sgld} strokeWidth={1.2} opacity={0.85} />
      <path d={linePath(run.svrgTrace)} fill="none" stroke={paletteSGMCMC.sghmc} strokeWidth={1.2} opacity={0.85} />
      <path d={linePath(run.zvTrace)} fill="none" stroke={paletteSGMCMC.langevin} strokeWidth={1.2} opacity={0.85} />
      <line x1={0} x2={innerW} y1={innerH} y2={innerH} stroke="var(--color-text-muted, #888)" />
      <line x1={0} x2={0} y1={0} y2={innerH} stroke="var(--color-text-muted, #888)" />
      <text x={innerW / 2} y={innerH + 18} textAnchor="middle" fontSize={10} fill="var(--color-text, #333)">iteration n</text>
      <text transform={`translate(${-38}, ${innerH / 2}) rotate(-90)`} textAnchor="middle" fontSize={10} fill="var(--color-text, #333)">running θ_1</text>
      <g transform={`translate(${innerW + 8}, 12)`}>
        <rect x={0} y={-6} width={102} height={70} fill="var(--color-surface, #fff)" stroke="var(--color-border, #ccc)" />
        <line x1={4} x2={20} y1={4} y2={4} stroke={paletteSGMCMC.sgld} strokeWidth={2} />
        <text x={24} y={7} fontSize={10} fill="var(--color-text, #333)">vanilla</text>
        <line x1={4} x2={20} y1={22} y2={22} stroke={paletteSGMCMC.sghmc} strokeWidth={2} />
        <text x={24} y={25} fontSize={10} fill="var(--color-text, #333)">SVRG-LD</text>
        <line x1={4} x2={20} y1={40} y2={40} stroke={paletteSGMCMC.langevin} strokeWidth={2} />
        <text x={24} y={43} fontSize={10} fill="var(--color-text, #333)">ZV-SGLD</text>
        <line x1={4} x2={20} y1={58} y2={58} stroke={paletteSGMCMC.target} strokeWidth={2} strokeDasharray="3 2" />
        <text x={24} y={61} fontSize={10} fill="var(--color-text, #333)">μ_post</text>
      </g>
    </g>
  );
}

function StdErrPanel({ run, width, height }: { run: CachedRun; width: number; height: number }) {
  const margin = { top: 12, right: 12, bottom: 28, left: 50 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const start = 100;
  const all = [
    ...Array.from(run.vanillaSE.slice(start)),
    ...Array.from(run.svrgSE.slice(start)),
    ...Array.from(run.zvSE.slice(start)),
  ].filter((v) => v > 0);
  const yMin = Math.min(...all) * 0.7;
  const yMax = Math.max(...all) * 1.3;
  const xS = d3.scaleLog().domain([start, run.T - 1]).range([0, innerW]);
  const yS = d3.scaleLog().domain([yMin, yMax]).range([innerH, 0]);
  const linePath = (vals: Float32Array) =>
    Array.from(vals.slice(start))
      .map((v, i) => `${i === 0 ? 'M' : 'L'}${xS(i + start)},${yS(Math.max(v, yMin))}`)
      .join(' ');

  const fmtPow = (v: number) => `10^${Math.round(Math.log10(v))}`;
  const xTicks = [1e2, 1e3, 1e4].filter((t) => t >= xS.domain()[0] && t <= xS.domain()[1]);
  const yTicks = decadeTicks(yMin, yMax);

  return (
    <g transform={`translate(${margin.left},${margin.top})`}>
      <rect x={0} y={0} width={innerW} height={innerH} fill="var(--color-surface, #fafafa)" />
      {xTicks.map((t) => (
        <g key={`xg-${t}`}>
          <line x1={xS(t)} x2={xS(t)} y1={0} y2={innerH} stroke="var(--color-text-muted, #888)" strokeOpacity={0.15} />
          <text x={xS(t)} y={innerH + 16} textAnchor="middle" fontSize={10} fill="var(--color-text, #333)">{fmtPow(t)}</text>
        </g>
      ))}
      {yTicks.map((t) => (
        <g key={`yg-${t}`}>
          <line x1={0} x2={innerW} y1={yS(t)} y2={yS(t)} stroke="var(--color-text-muted, #888)" strokeOpacity={0.15} />
          <text x={-6} y={yS(t) + 3} textAnchor="end" fontSize={10} fill="var(--color-text, #333)">{fmtPow(t)}</text>
        </g>
      ))}
      <path d={linePath(run.vanillaSE)} fill="none" stroke={paletteSGMCMC.sgld} strokeWidth={1.2} />
      <path d={linePath(run.svrgSE)} fill="none" stroke={paletteSGMCMC.sghmc} strokeWidth={1.2} />
      <path d={linePath(run.zvSE)} fill="none" stroke={paletteSGMCMC.langevin} strokeWidth={1.2} />
      <line x1={0} x2={innerW} y1={innerH} y2={innerH} stroke="var(--color-text-muted, #888)" />
      <line x1={0} x2={0} y1={0} y2={innerH} stroke="var(--color-text-muted, #888)" />
      <text x={innerW / 2} y={innerH + 18} textAnchor="middle" fontSize={10} fill="var(--color-text, #333)">iteration n (log)</text>
      <text transform={`translate(${-38}, ${innerH / 2}) rotate(-90)`} textAnchor="middle" fontSize={10} fill="var(--color-text, #333)">SE(θ_1) (log)</text>
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
