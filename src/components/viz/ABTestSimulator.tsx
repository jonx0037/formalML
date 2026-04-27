import { useEffect, useMemo, useRef, useState, type CSSProperties } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { gaussianSampler, mulberry32 } from './shared/nonparametric-ml';

// =============================================================================
// ABTestSimulator — embedded in §8 after Theorem 12.
//
// Three-panel grid demonstrating randomization-inference exactness vs Welch
// t size distortion, plus the CUPED variance-reduction story.
//
//   Top-left  : two group histograms (control A / treatment B), means
//               marked. Shows the metric's shape directly so the
//               heavy-tail / skewness story is visceral.
//
//   Top-right : randomization distribution of the chosen statistic over B
//               label-swap permutations, with the observed value, mean
//               reference, and shaded rejection region. Phipson-Smyth
//               (1+#extreme)/(1+B) p-value.
//
//   Bottom    : type-I error tracker — fires K = 1000 datasets under H₀ and
//               reports empirical rejection rate of Welch t vs the
//               randomization test at the chosen α. Updates incrementally
//               via setTimeout chunks to keep the UI responsive; can be
//               aborted at any time.
//
// Default scenario: log-normal revenue (σ_log = 2), n = 80 per group,
// effect Δ = 0. At these parameters the notebook reports Welch t empirical
// rejection ≈ 0.030 (under-rejects) vs randomization ≈ 0.052 (nominal).
// The component reproduces this within Monte Carlo sampling error.
//
// CUPED: when enabled, generates a pre-period covariate X with correlation
// ρ_{X,Y} to the metric, and applies the symmetric adjustment
// Y_cuped = Y - β(X - X̄), β = Cov(X,Y)/Var(X). The randomization is run on
// Y_cuped; exactness carries through (Remark 6) and variance shrinks by
// (1 - ρ²), translating to power gain.
// =============================================================================

const SM_BREAKPOINT = 640;
const MD_BREAKPOINT = 880;

const BLUE = '#2563EB';
const RED = '#DC2626';
const GREEN = '#059669';
const AMBER = '#D97706';
const PURPLE = '#7C3AED';
const SLATE = '#475569';
const LIGHT_BLUE = '#DBEAFE';
const LIGHT_RED = '#FEE2E2';

type Metric = 'normal' | 'lognormal' | 'bernoulli' | 'contaminated' | 'cauchy';

const METRICS: { value: Metric; label: string }[] = [
  { value: 'normal', label: 'Normal(0, 1)' },
  { value: 'lognormal', label: 'Log-normal (σ_log = 2)' },
  { value: 'bernoulli', label: 'Bernoulli(p = 0.05)' },
  { value: 'contaminated', label: 'Contaminated normal (ε = 0.1, scale 6)' },
  { value: 'cauchy', label: 'Cauchy(0, 1) — capped' },
];

type Stat = 'meanDiff' | 'cuped';

const fmt = (x: number, digits = 3) => x.toFixed(digits);

function sampleMetric(rng: () => number, metric: Metric, n: number, shift: number): Float64Array {
  const out = new Float64Array(n);
  if (metric === 'normal') {
    const g = gaussianSampler(rng);
    for (let i = 0; i < n; i++) out[i] = g() + shift;
    return out;
  }
  if (metric === 'lognormal') {
    const g = gaussianSampler(rng);
    for (let i = 0; i < n; i++) out[i] = Math.exp(2 * g()) + shift;
    return out;
  }
  if (metric === 'bernoulli') {
    const p = Math.max(0, Math.min(1, 0.05 + shift));
    for (let i = 0; i < n; i++) out[i] = rng() < p ? 1 : 0;
    return out;
  }
  if (metric === 'contaminated') {
    const g = gaussianSampler(rng);
    for (let i = 0; i < n; i++) {
      const u = rng();
      out[i] = (u < 0.1 ? g() * 6 : g()) + shift;
    }
    return out;
  }
  // cauchy: capped at ±50 to keep histogram readable.
  for (let i = 0; i < n; i++) {
    let c = Math.tan(Math.PI * (rng() - 0.5));
    if (Math.abs(c) > 50) c = Math.sign(c) * 50;
    out[i] = c + shift;
  }
  return out;
}

function sampleCovariate(rng: () => number, Y: Float64Array, rho: number): Float64Array {
  const g = gaussianSampler(rng);
  const muY = arrayMean(Y);
  const sdY = Math.sqrt(arrayVar(Y, muY));
  // Generate X = ρ · standardized(Y) + sqrt(1 - ρ²) · ε
  const out = new Float64Array(Y.length);
  for (let i = 0; i < Y.length; i++) {
    const z = sdY > 1e-12 ? (Y[i] - muY) / sdY : 0;
    out[i] = rho * z + Math.sqrt(Math.max(0, 1 - rho * rho)) * g();
  }
  return out;
}

function arrayMean(a: Float64Array | number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i];
  return s / a.length;
}

function arrayVar(a: Float64Array | number[], mu: number): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - mu;
    s += d * d;
  }
  return s / Math.max(1, a.length - 1);
}

// CUPED-adjusted Y values: Y - β(X - X̄), β = Cov(X, Y)/Var(X) computed
// from the pooled (X, Y).
function cupedAdjust(Y: Float64Array, X: Float64Array): Float64Array {
  const muX = arrayMean(X);
  const muY = arrayMean(Y);
  let cov = 0;
  let varX = 0;
  for (let i = 0; i < X.length; i++) {
    const dx = X[i] - muX;
    cov += dx * (Y[i] - muY);
    varX += dx * dx;
  }
  const beta = varX > 1e-12 ? cov / varX : 0;
  const out = new Float64Array(Y.length);
  for (let i = 0; i < Y.length; i++) out[i] = Y[i] - beta * (X[i] - muX);
  return out;
}

function welchT(A: Float64Array, B: Float64Array): { t: number; df: number; p: number } {
  const muA = arrayMean(A);
  const muB = arrayMean(B);
  const vA = arrayVar(A, muA);
  const vB = arrayVar(B, muB);
  const seA2 = vA / A.length;
  const seB2 = vB / B.length;
  const denom = Math.sqrt(seA2 + seB2);
  const t = denom < 1e-12 ? 0 : (muA - muB) / denom;
  const num = (seA2 + seB2) ** 2;
  const denomDF = (seA2 ** 2) / (A.length - 1) + (seB2 ** 2) / (B.length - 1);
  const df = denomDF < 1e-12 ? 1 : num / denomDF;
  // Two-sided p under t_df via the regularized incomplete beta function:
  //   P(|T| ≥ t) = I_{df/(df+t²)}(df/2, 1/2).
  // Heavier tails than the Normal at small df are essential here — using
  // 2·(1 − Φ(|t|)) understates the true Welch p-value at small df, which
  // would mask the size distortion this component is meant to surface.
  const x = df / (df + t * t);
  const p = incompleteBeta(x, df / 2, 0.5);
  return { t, df, p };
}

// Lanczos log-Γ (relative error < 1e-13 for z > 0).
function logGamma(z: number): number {
  const g = [
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
  ];
  if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - logGamma(1 - z);
  z -= 1;
  let x = 0.99999999999980993;
  for (let i = 0; i < g.length; i++) x += g[i] / (z + i + 1);
  const tt = z + g.length - 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(tt) - tt + Math.log(x);
}

// Continued-fraction expansion of I_x(a, b), Numerical Recipes §6.4.
function betacf(a: number, b: number, x: number): number {
  const MAXIT = 200;
  const EPS = 3e-7;
  const FPMIN = 1e-30;
  const qab = a + b;
  const qap = a + 1;
  const qam = a - 1;
  let c = 1;
  let d = 1 - (qab * x) / qap;
  if (Math.abs(d) < FPMIN) d = FPMIN;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= MAXIT; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    h *= d * c;
    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < EPS) break;
  }
  return h;
}

// Regularized incomplete beta function I_x(a, b) ∈ [0, 1].
function incompleteBeta(x: number, a: number, b: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const lnBeta = logGamma(a) + logGamma(b) - logGamma(a + b);
  const bt = Math.exp(-lnBeta + a * Math.log(x) + b * Math.log(1 - x));
  if (x < (a + 1) / (a + b + 2)) {
    return (bt * betacf(a, b, x)) / a;
  }
  return 1 - (bt * betacf(b, a, 1 - x)) / b;
}

function permutationP(
  pool: Float64Array,
  nA: number,
  observed: number,
  B: number,
  rng: () => number,
): number {
  const N = pool.length;
  const work = Float64Array.from(pool);
  // Centring is at zero by construction: under H₀^sharp (Theorem 12) the
  // randomization distribution of mean(A) − mean(B) is symmetric about 0,
  // so the rejection criterion is |stat| ≥ |observed|.
  let nExtreme = 0;
  for (let s = 0; s < B; s++) {
    for (let i = N - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      const tmp = work[i]; work[i] = work[j]; work[j] = tmp;
    }
    const mA = meanRange(work, 0, nA);
    const mB = meanRange(work, nA, N);
    const stat = mA - mB;
    if (Math.abs(stat) >= Math.abs(observed) - 1e-12) nExtreme++;
  }
  return (1 + nExtreme) / (1 + B);
}

function meanRange(arr: Float64Array, lo: number, hi: number): number {
  let s = 0;
  for (let i = lo; i < hi; i++) s += arr[i];
  return s / (hi - lo);
}

interface TrackerState {
  running: boolean;
  done: number;
  total: number;
  welchReject: number;
  permReject: number;
}

const initialTracker: TrackerState = {
  running: false,
  done: 0,
  total: 0,
  welchReject: 0,
  permReject: 0,
};

export default function ABTestSimulator() {
  const { ref: containerRef, width: containerWidth } =
    useResizeObserver<HTMLDivElement>();

  const [metric, setMetric] = useState<Metric>('lognormal');
  const [delta, setDelta] = useState(0);
  const [n, setN] = useState(80);
  const [B, setB] = useState(2000);
  const [alpha, setAlpha] = useState(0.05);
  const [seed, setSeed] = useState(42);
  const [useCuped, setUseCuped] = useState(false);
  const [rho, setRho] = useState(0.7);
  const [tracker, setTracker] = useState<TrackerState>(initialTracker);
  const trackerAbort = useRef(false);

  const sample = useMemo(() => {
    const rng = mulberry32(seed);
    const A = sampleMetric(rng, metric, n, 0);
    const B = sampleMetric(rng, metric, n, delta);
    let Aw = A;
    let Bw = B;
    if (useCuped) {
      const Xa = sampleCovariate(rng, A, rho);
      const Xb = sampleCovariate(rng, B, rho);
      // Pool covariates so β is computed once.
      const X = new Float64Array(n + n);
      X.set(Xa, 0);
      X.set(Xb, n);
      const Y = new Float64Array(n + n);
      Y.set(A, 0);
      Y.set(B, n);
      const Yc = cupedAdjust(Y, X);
      Aw = Yc.subarray(0, n);
      Bw = Yc.subarray(n, n + n);
    }
    return { A: Aw, B: Bw, raw: { A, B } };
  }, [metric, delta, n, useCuped, rho, seed]);

  const obsStat = arrayMean(sample.A) - arrayMean(sample.B);

  const permP = useMemo(() => {
    const rng = mulberry32(seed + 17);
    const pool = new Float64Array(2 * n);
    pool.set(sample.A, 0);
    pool.set(sample.B, n);
    return permutationP(pool, n, obsStat, B, rng);
  }, [sample, n, B, seed, obsStat]);

  const welch = useMemo(() => welchT(sample.A, sample.B), [sample]);

  const isStacked = (containerWidth || 0) < SM_BREAKPOINT;
  const isMid = (containerWidth || 0) < MD_BREAKPOINT;
  const topPanelW = isStacked
    ? containerWidth || 0
    : Math.max(280, Math.floor(((containerWidth || 720) - 16) / 2));
  const bottomPanelW = isStacked ? containerWidth || 0 : containerWidth || 720;
  const TOP_H = 240;
  const BOT_H = 200;

  // ── Top-left: group histograms ──────────────────────────────────────────
  const histRef = useRef<SVGSVGElement>(null);
  useEffect(() => {
    const svg = d3.select(histRef.current);
    if (!histRef.current || topPanelW <= 0) return;
    svg.selectAll('*').remove();
    const margin = { top: 28, right: 12, bottom: 30, left: 36 };
    const w = topPanelW - margin.left - margin.right;
    const h = TOP_H - margin.top - margin.bottom;
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const all = [...sample.A, ...sample.B];
    const xLo = d3.min(all) ?? 0;
    const xHi = d3.max(all) ?? 1;
    const span = xHi - xLo || 1;
    const xScale = d3.scaleLinear().domain([xLo - 0.05 * span, xHi + 0.05 * span]).range([0, w]);
    const nBins = 26;
    const binner = d3.bin().domain(xScale.domain() as [number, number]).thresholds(nBins);
    const binsA = binner(Array.from(sample.A));
    const binsB = binner(Array.from(sample.B));
    const yMax = Math.max(d3.max(binsA, (b) => b.length) ?? 1, d3.max(binsB, (b) => b.length) ?? 1) * 1.1;
    const yScale = d3.scaleLinear().domain([0, yMax]).range([h, 0]);

    g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(xScale).ticks(5)).selectAll('text').style('fill', 'var(--color-text)').style('font-size', '10px');
    g.append('g').call(d3.axisLeft(yScale).ticks(4)).selectAll('text').style('fill', 'var(--color-text)').style('font-size', '10px');
    g.selectAll('path.domain, .tick line').style('stroke', 'var(--color-border)');

    const drawBins = (bins: d3.Bin<number, number>[], color: string, label: string) => {
      g.selectAll(`.${label}`)
        .data(bins)
        .enter()
        .append('rect')
        .attr('class', label)
        .attr('x', (b) => xScale(b.x0 ?? 0))
        .attr('y', (b) => yScale(b.length))
        .attr('width', (b) => Math.max(0, xScale(b.x1 ?? 0) - xScale(b.x0 ?? 0) - 1))
        .attr('height', (b) => h - yScale(b.length))
        .style('fill', color)
        .style('opacity', 0.55);
    };
    drawBins(binsA, BLUE, 'binA');
    drawBins(binsB, RED, 'binB');

    const meanA = arrayMean(sample.A);
    const meanB = arrayMean(sample.B);
    const meanLine = (x: number, color: string) =>
      g.append('line').attr('x1', xScale(x)).attr('x2', xScale(x)).attr('y1', 0).attr('y2', h).style('stroke', color).style('stroke-width', 1.6).style('stroke-dasharray', '3,3');
    meanLine(meanA, BLUE);
    meanLine(meanB, RED);

    svg
      .append('text')
      .attr('x', margin.left + w / 2)
      .attr('y', 16)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text)')
      .style('font-size', '11.5px')
      .style('font-family', 'var(--font-sans)')
      .style('font-weight', 600)
      .text(`Group histograms  (n = ${n} per arm)`);

    g.append('text')
      .attr('x', 4)
      .attr('y', 12)
      .style('fill', BLUE)
      .style('font-size', '10.5px')
      .style('font-family', 'var(--font-mono)')
      .text(`A  X̄ = ${fmt(meanA, 3)}`);
    g.append('text')
      .attr('x', 4)
      .attr('y', 26)
      .style('fill', RED)
      .style('font-size', '10.5px')
      .style('font-family', 'var(--font-mono)')
      .text(`B  Ȳ = ${fmt(meanB, 3)}${useCuped ? ' (CUPED)' : ''}`);
  }, [sample, topPanelW, n, useCuped]);

  // ── Top-right: randomization histogram ──────────────────────────────────
  const permRef = useRef<SVGSVGElement>(null);
  const permDist = useMemo(() => {
    const rng = mulberry32(seed + 99);
    const pool = new Float64Array(2 * n);
    pool.set(sample.A, 0);
    pool.set(sample.B, n);
    const work = Float64Array.from(pool);
    const samples = new Float64Array(B);
    for (let s = 0; s < B; s++) {
      for (let i = work.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        const tmp = work[i]; work[i] = work[j]; work[j] = tmp;
      }
      samples[s] = meanRange(work, 0, n) - meanRange(work, n, 2 * n);
    }
    return samples;
  }, [sample, n, B, seed]);

  useEffect(() => {
    const svg = d3.select(permRef.current);
    if (!permRef.current || topPanelW <= 0) return;
    svg.selectAll('*').remove();
    const margin = { top: 28, right: 12, bottom: 32, left: 36 };
    const w = topPanelW - margin.left - margin.right;
    const h = TOP_H - margin.top - margin.bottom;
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const arr = Array.from(permDist);
    const xLo = Math.min(d3.min(arr) ?? 0, obsStat);
    const xHi = Math.max(d3.max(arr) ?? 0, obsStat);
    const span = (xHi - xLo) || 1;
    const xScale = d3.scaleLinear().domain([xLo - 0.05 * span, xHi + 0.05 * span]).range([0, w]);
    const binner = d3.bin().domain(xScale.domain() as [number, number]).thresholds(40);
    const bins = binner(arr);
    const yMax = (d3.max(bins, (b) => b.length) ?? 1) * 1.1;
    const yScale = d3.scaleLinear().domain([0, yMax]).range([h, 0]);

    g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(xScale).ticks(5)).selectAll('text').style('fill', 'var(--color-text)').style('font-size', '10px');
    g.append('g').call(d3.axisLeft(yScale).ticks(4)).selectAll('text').style('fill', 'var(--color-text)').style('font-size', '10px');
    g.selectAll('path.domain, .tick line').style('stroke', 'var(--color-border)');

    const sortedAbsDevs = arr.map((v) => Math.abs(v - 0)).sort((a, b) => a - b);
    const critIdx = Math.floor((1 - alpha) * sortedAbsDevs.length);
    const critDev = sortedAbsDevs[Math.min(critIdx, sortedAbsDevs.length - 1)];

    g.selectAll('rect.bar')
      .data(bins)
      .enter()
      .append('rect')
      .attr('x', (b) => xScale(b.x0 ?? 0))
      .attr('y', (b) => yScale(b.length))
      .attr('width', (b) => Math.max(0, xScale(b.x1 ?? 0) - xScale(b.x0 ?? 0) - 1))
      .attr('height', (b) => h - yScale(b.length))
      .style('fill', (b) => {
        const c = ((b.x0 ?? 0) + (b.x1 ?? 0)) / 2;
        return Math.abs(c) >= critDev ? LIGHT_RED : LIGHT_BLUE;
      })
      .style('stroke', (b) => {
        const c = ((b.x0 ?? 0) + (b.x1 ?? 0)) / 2;
        return Math.abs(c) >= critDev ? RED : BLUE;
      })
      .style('stroke-width', 0.6);

    g.append('line')
      .attr('x1', xScale(0))
      .attr('x2', xScale(0))
      .attr('y1', 0)
      .attr('y2', h)
      .style('stroke', SLATE)
      .style('stroke-dasharray', '2,3');

    g.append('line')
      .attr('x1', xScale(obsStat))
      .attr('x2', xScale(obsStat))
      .attr('y1', 0)
      .attr('y2', h)
      .style('stroke', RED)
      .style('stroke-width', 2)
      .style('stroke-dasharray', '4,3');

    svg
      .append('text')
      .attr('x', margin.left + w / 2)
      .attr('y', 16)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text)')
      .style('font-size', '11.5px')
      .style('font-family', 'var(--font-sans)')
      .style('font-weight', 600)
      .text(`Randomization distribution  (B = ${B.toLocaleString()})`);

    svg
      .append('text')
      .attr('x', margin.left + w / 2)
      .attr('y', TOP_H - 8)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text-secondary)')
      .style('font-size', '10.5px')
      .style('font-family', 'var(--font-mono)')
      .text(
        `T_obs = ${fmt(obsStat, 3)}    rand. p = ${fmt(permP, 4)}    Welch p = ${fmt(welch.p, 4)}`,
      );
  }, [permDist, topPanelW, obsStat, B, alpha, permP, welch]);

  // ── Bottom: type-I tracker ──────────────────────────────────────────────
  const trackerRef = useRef<SVGSVGElement>(null);
  useEffect(() => {
    const svg = d3.select(trackerRef.current);
    if (!trackerRef.current || bottomPanelW <= 0) return;
    svg.selectAll('*').remove();
    const margin = { top: 26, right: 16, bottom: 30, left: 50 };
    const w = bottomPanelW - margin.left - margin.right;
    const h = BOT_H - margin.top - margin.bottom;
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const xScale = d3.scaleLinear().domain([0, Math.max(1, tracker.total)]).range([0, w]);
    const yScale = d3.scaleLinear().domain([0, Math.max(0.15, alpha * 3)]).range([h, 0]);

    g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(xScale).ticks(5)).selectAll('text').style('fill', 'var(--color-text)').style('font-size', '10px');
    g.append('g').call(d3.axisLeft(yScale).ticks(4).tickFormat(d3.format('.0%'))).selectAll('text').style('fill', 'var(--color-text)').style('font-size', '10px');
    g.selectAll('path.domain, .tick line').style('stroke', 'var(--color-border)');

    g.append('line')
      .attr('x1', 0)
      .attr('x2', w)
      .attr('y1', yScale(alpha))
      .attr('y2', yScale(alpha))
      .style('stroke', SLATE)
      .style('stroke-dasharray', '2,3');
    g.append('text')
      .attr('x', w - 4)
      .attr('y', yScale(alpha) - 4)
      .attr('text-anchor', 'end')
      .style('font-size', '10px')
      .style('fill', SLATE)
      .text(`α = ${alpha.toFixed(2)}`);

    if (tracker.done > 0) {
      const wRate = tracker.welchReject / tracker.done;
      const pRate = tracker.permReject / tracker.done;
      const dot = (x: number, color: string) =>
        g.append('circle').attr('cx', xScale(x)).attr('cy', yScale(x === 0 ? 0 : 0)).attr('r', 0).style('fill', color);
      void dot;
      g.append('rect')
        .attr('x', xScale(tracker.done) - 4)
        .attr('y', yScale(wRate) - 8)
        .attr('width', 8)
        .attr('height', 16)
        .style('fill', RED);
      g.append('rect')
        .attr('x', xScale(tracker.done) - 4)
        .attr('y', yScale(pRate) - 8)
        .attr('width', 8)
        .attr('height', 16)
        .style('fill', BLUE);
      g.append('text')
        .attr('x', xScale(tracker.done) + 8)
        .attr('y', yScale(wRate) + 4)
        .style('font-size', '10.5px')
        .style('font-family', 'var(--font-mono)')
        .style('fill', RED)
        .text(`Welch ${fmt(wRate, 3)}`);
      g.append('text')
        .attr('x', xScale(tracker.done) + 8)
        .attr('y', yScale(pRate) + 4)
        .style('font-size', '10.5px')
        .style('font-family', 'var(--font-mono)')
        .style('fill', BLUE)
        .text(`rand. ${fmt(pRate, 3)}`);
    }

    svg
      .append('text')
      .attr('x', margin.left + w / 2)
      .attr('y', 14)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text)')
      .style('font-size', '11.5px')
      .style('font-family', 'var(--font-sans)')
      .style('font-weight', 600)
      .text(
        tracker.running
          ? `Type-I tracker — ${tracker.done.toLocaleString()} / ${tracker.total.toLocaleString()} datasets`
          : tracker.done > 0
            ? `Type-I tracker — completed ${tracker.done.toLocaleString()} / ${tracker.total.toLocaleString()}`
            : 'Type-I tracker — click "Run" to fire K H₀ datasets',
      );
  }, [tracker, alpha, bottomPanelW]);

  const runTracker = (K: number) => {
    if (tracker.running) return;
    trackerAbort.current = false;
    setTracker({ running: true, done: 0, total: K, welchReject: 0, permReject: 0 });
    const rng = mulberry32(seed + 999);
    let done = 0;
    let welchReject = 0;
    let permReject = 0;
    const chunk = () => {
      if (trackerAbort.current) {
        setTracker((t) => ({ ...t, running: false }));
        return;
      }
      const stepSize = 25;
      const stop = Math.min(K, done + stepSize);
      for (let s = done; s < stop; s++) {
        // Generate H0 data: both groups from the same metric, no shift.
        const A = sampleMetric(rng, metric, n, 0);
        const B = sampleMetric(rng, metric, n, 0);
        const Aw = useCuped ? null : A;
        const Bw = useCuped ? null : B;
        let A_ = A;
        let B_ = B;
        if (useCuped) {
          const Xa = sampleCovariate(rng, A, rho);
          const Xb = sampleCovariate(rng, B, rho);
          const X = new Float64Array(2 * n);
          X.set(Xa, 0); X.set(Xb, n);
          const Y = new Float64Array(2 * n);
          Y.set(A, 0); Y.set(B, n);
          const Yc = cupedAdjust(Y, X);
          A_ = Yc.subarray(0, n);
          B_ = Yc.subarray(n, 2 * n);
        }
        // Welch
        const w = welchT(A_, B_);
        if (w.p <= alpha) welchReject++;
        // Randomization with B_perm = 200 (cheap, sufficient for size estimation)
        const obs = arrayMean(A_) - arrayMean(B_);
        const pool = new Float64Array(2 * n);
        pool.set(A_, 0); pool.set(B_, n);
        const p = permutationP(pool, n, obs, 200, rng);
        if (p <= alpha) permReject++;
        void Aw; void Bw;
      }
      done = stop;
      setTracker({ running: done < K, done, total: K, welchReject, permReject });
      if (done < K) {
        // Yield to the browser so the UI stays responsive.
        setTimeout(chunk, 0);
      }
    };
    setTimeout(chunk, 0);
  };

  return (
    <div ref={containerRef} className="not-prose">
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: isMid ? '1fr' : 'repeat(3, minmax(0, 1fr))',
          gap: '0.5rem 1rem',
          marginBottom: '0.75rem',
          fontFamily: 'var(--font-sans)',
          fontSize: '12px',
        }}
      >
        <label style={ctrlRow()}>
          <span style={ctrlLab()}>metric Y</span>
          <select value={metric} onChange={(e) => setMetric(e.target.value as Metric)} style={selStyle()}>
            {METRICS.map((m) => (<option key={m.value} value={m.value}>{m.label}</option>))}
          </select>
        </label>
        <label style={ctrlRow()}>
          <span style={ctrlLab()}>effect Δ</span>
          <input type="range" min={-1} max={1} step={0.05} value={delta} onChange={(e) => setDelta(Number(e.target.value))} style={{ flex: 1 }} aria-label="treatment effect Δ" />
          <span style={ctrlVal()}>{delta.toFixed(2)}</span>
        </label>
        <label style={ctrlRow()}>
          <span style={ctrlLab()}>n / arm</span>
          <input type="range" min={10} max={500} step={10} value={n} onChange={(e) => setN(Number(e.target.value))} style={{ flex: 1 }} aria-label="users per arm" />
          <span style={ctrlVal()}>{n}</span>
        </label>
        <label style={ctrlRow()}>
          <span style={ctrlLab()}>α</span>
          <input type="range" min={0.01} max={0.20} step={0.01} value={alpha} onChange={(e) => setAlpha(Number(e.target.value))} style={{ flex: 1 }} aria-label="significance α" />
          <span style={ctrlVal()}>{alpha.toFixed(2)}</span>
        </label>
        <label style={ctrlRow()}>
          <span style={ctrlLab()}>B perms</span>
          <input type="range" min={500} max={5000} step={100} value={B} onChange={(e) => setB(Number(e.target.value))} style={{ flex: 1 }} aria-label="number of permutations" />
          <span style={ctrlVal()}>{B.toLocaleString()}</span>
        </label>
        <label style={ctrlRow()}>
          <span style={ctrlLab()}>CUPED</span>
          <input type="checkbox" checked={useCuped} onChange={(e) => setUseCuped(e.target.checked)} aria-label="enable CUPED adjustment" style={{ marginRight: '0.5rem' }} />
          {useCuped && (
            <>
              <span style={{ ...ctrlLab(), minWidth: '2em' }}>ρ</span>
              <input type="range" min={0} max={0.95} step={0.05} value={rho} onChange={(e) => setRho(Number(e.target.value))} style={{ flex: 1 }} aria-label="ρ_XY for CUPED" />
              <span style={ctrlVal()}>{rho.toFixed(2)}</span>
            </>
          )}
        </label>
      </div>

      <div
        style={{
          display: 'flex',
          flexDirection: isStacked ? 'column' : 'row',
          gap: isStacked ? '0.75rem' : '0.5rem',
          marginBottom: '0.5rem',
        }}
      >
        <svg ref={histRef} width={topPanelW} height={TOP_H} role="img" aria-label="Group A and Group B histograms with means" />
        <svg ref={permRef} width={topPanelW} height={TOP_H} role="img" aria-label="Randomization distribution histogram with rejection region and observed value" />
      </div>

      <div
        style={{
          marginBottom: '0.5rem',
          display: 'flex',
          gap: '0.6rem',
          flexWrap: 'wrap',
          fontFamily: 'var(--font-sans)',
        }}
      >
        <button type="button" onClick={() => runTracker(1000)} disabled={tracker.running} style={btn(true, tracker.running)}>
          ▶ Run type-I tracker  (K = 1000)
        </button>
        <button type="button" onClick={() => runTracker(500)} disabled={tracker.running} style={btn(false, tracker.running)}>
          ▶ Run K = 500
        </button>
        <button type="button" onClick={() => { trackerAbort.current = true; }} disabled={!tracker.running} style={btn(false, !tracker.running)}>
          ■ Abort
        </button>
        <button type="button" onClick={() => setSeed((s) => (s + 1) % 999983)} style={btn(false, false)}>
          ↻ resample data
        </button>
      </div>

      <svg ref={trackerRef} width={bottomPanelW} height={BOT_H} role="img" aria-label="Type-I error tracker comparing Welch t and randomization rejection rates" />

      <p
        style={{
          fontSize: '11px',
          color: 'var(--color-text-secondary)',
          fontFamily: 'var(--font-sans)',
          marginTop: '0.25rem',
          fontStyle: 'italic',
        }}
      >
        At log-normal Y with n = 80, σ_log = 2 and Δ = 0, run the tracker to see the Welch-t bar settle around <strong>0.030</strong> (under-rejecting) while the randomization bar stabilizes at <strong>~0.05</strong>. Toggle CUPED to confirm exactness still holds (Remark 6); raise Δ to feel the power gain.
      </p>
    </div>
  );
}

const ctrlRow = (): CSSProperties => ({ display: 'flex', alignItems: 'center', gap: '0.5rem' });
const ctrlLab = (): CSSProperties => ({ fontSize: '11px', color: 'var(--color-text-secondary)', minWidth: '4.5em' });
const ctrlVal = (): CSSProperties => ({ fontSize: '11px', fontFamily: 'var(--font-mono)', minWidth: '3em', textAlign: 'right', color: 'var(--color-text)' });
const selStyle = (): CSSProperties => ({ flex: 1, fontSize: '11.5px', padding: '3px 6px', background: 'var(--color-surface)', color: 'var(--color-text)', border: '1px solid var(--color-border)', borderRadius: 4 });
const btn = (primary: boolean, disabled: boolean): CSSProperties => ({
  fontSize: '11px',
  padding: '5px 12px',
  background: primary && !disabled ? BLUE : 'var(--color-surface)',
  color: primary && !disabled ? 'white' : 'var(--color-text)',
  border: '1px solid var(--color-border)',
  borderRadius: 4,
  cursor: disabled ? 'not-allowed' : 'pointer',
  opacity: disabled ? 0.5 : 1,
});
