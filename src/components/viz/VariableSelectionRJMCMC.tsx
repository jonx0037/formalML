// =============================================================================
// VariableSelectionRJMCMC.tsx
//
// §9 of reversible-jump-mcmc. Head-to-head of RJ-MCMC (5000 iter) against the
// 256-model enumeration gold standard at p = 8. Liang et al. (2008) closed-form
// g-prior marginal likelihood; Scott–Berger multiplicity-correction prior.
//
// Display-vs-committed: chain reruns only on slider release.
//
// Static fallback: /images/topics/reversible-jump-mcmc/fig09_variable_selection.png
// =============================================================================
import { useCallback, useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';

function mulberry32(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function randn(rng: () => number): number {
  const u = 1 - rng();
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
function lgamma(x: number): number {
  const cof = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
  let y = x;
  let tmp = x + 5.5;
  tmp -= (x + 0.5) * Math.log(tmp);
  let ser = 1.000000000190015;
  for (const c of cof) ser += c / ++y;
  return -tmp + Math.log((2.5066282746310005 * ser) / x);
}
function logChoose(n: number, k: number): number {
  return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1);
}

// Solve A β = b for symmetric positive-definite A via Cholesky + back-sub.
// Returns null if not PD; otherwise β.
function choleskySolve(A: number[][], b: number[]): number[] | null {
  const n = b.length;
  const L: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      if (i === j) {
        if (s <= 0) return null;
        L[i][j] = Math.sqrt(s);
      } else {
        L[i][j] = s / L[j][j];
      }
    }
  }
  const yv = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let s = b[i];
    for (let k = 0; k < i; k++) s -= L[i][k] * yv[k];
    yv[i] = s / L[i][i];
  }
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = yv[i];
    for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k];
    x[i] = s / L[i][i];
  }
  return x;
}

function olsR2(X: number[][], y: number[], cols: number[]): number {
  const n = y.length;
  const k = cols.length;
  if (k === 0) return 0;
  // Build XtX and Xty for the active columns
  const XtX: number[][] = Array.from({ length: k }, () => new Array(k).fill(0));
  const Xty = new Array(k).fill(0);
  for (let i = 0; i < n; i++) {
    for (let a = 0; a < k; a++) {
      const xia = X[i][cols[a]];
      Xty[a] += xia * y[i];
      for (let b = 0; b < k; b++) XtX[a][b] += xia * X[i][cols[b]];
    }
  }
  const beta = choleskySolve(XtX, Xty);
  if (!beta) return 0;
  let rss = 0;
  let tss = 0;
  const yMean = y.reduce((a, b) => a + b, 0) / n;
  for (let i = 0; i < n; i++) {
    let yhat = 0;
    for (let a = 0; a < k; a++) yhat += X[i][cols[a]] * beta[a];
    rss += (y[i] - yhat) ** 2;
    tss += (y[i] - yMean) ** 2;
  }
  return tss > 1e-12 ? 1 - rss / tss : 0;
}

const P = 8;
const N = 80;
const TRUE_ACTIVE = [0, 2, 5];
const TRUE_BETAS = [2.5, 0, -1.8, 0, 0, 1.5, 0, 0];

function generateData(snr: number, seed: number): { X: number[][]; y: number[] } {
  const rng = mulberry32(seed);
  const X: number[][] = [];
  for (let i = 0; i < N; i++) {
    const row = new Array(P);
    for (let j = 0; j < P; j++) row[j] = randn(rng);
    X.push(row);
  }
  // Center columns
  for (let j = 0; j < P; j++) {
    let s = 0;
    for (let i = 0; i < N; i++) s += X[i][j];
    const m = s / N;
    for (let i = 0; i < N; i++) X[i][j] -= m;
  }
  // Scale true betas by snr-derived factor: snr controls noise std
  const noise = 1 / snr;
  const y = new Array(N).fill(0);
  for (let i = 0; i < N; i++) {
    let yi = 0;
    for (let j = 0; j < P; j++) yi += X[i][j] * TRUE_BETAS[j];
    y[i] = yi + noise * randn(rng);
  }
  // Center y
  const yMean = y.reduce((a, b) => a + b, 0) / N;
  for (let i = 0; i < N; i++) y[i] -= yMean;
  return { X, y };
}

interface EnumResult {
  inclProbs: number[];
  sizeProbs: number[];
  topModels: { gamma: string; logPost: number }[];
}
function enumerate(X: number[][], y: number[], g: number): EnumResult {
  const logPosts: { gamma: string; logPost: number; cols: number[]; card: number }[] = [];
  for (let id = 0; id < 1 << P; id++) {
    const cols: number[] = [];
    let gamma = '';
    for (let j = 0; j < P; j++) {
      if ((id >> j) & 1) cols.push(j);
      gamma += (id >> j) & 1 ? '1' : '0';
    }
    const card = cols.length;
    let logMarg = 0;
    if (card > 0) {
      const R2 = olsR2(X, y, cols);
      logMarg = ((N - card - 1) / 2) * Math.log(1 + g) - ((N - 1) / 2) * Math.log(1 + g * (1 - R2));
    }
    const logPrior = -logChoose(P, card) - Math.log(P + 1);
    logPosts.push({ gamma, logPost: logMarg + logPrior, cols, card });
  }
  const maxL = Math.max(...logPosts.map((p) => p.logPost));
  let sum = 0;
  for (const p of logPosts) sum += Math.exp(p.logPost - maxL);
  const logZ = maxL + Math.log(sum);
  for (const p of logPosts) p.logPost = Math.exp(p.logPost - logZ);

  const inclProbs = new Array(P).fill(0);
  const sizeProbs = new Array(P + 1).fill(0);
  for (const p of logPosts) {
    sizeProbs[p.card] += p.logPost;
    for (const c of p.cols) inclProbs[c] += p.logPost;
  }
  const top = logPosts.slice().sort((a, b) => b.logPost - a.logPost).slice(0, 5).map((p) => ({ gamma: p.gamma, logPost: p.logPost }));
  return { inclProbs, sizeProbs, topModels: top };
}

interface ChainResult {
  inclProbs: number[];
  sizeProbs: number[];
  topModels: { gamma: string; logPost: number }[];
  acceptRate: number;
  uniqueVisited: number;
}
function runChain(X: number[][], y: number[], g: number, T: number, burn: number, seed: number): ChainResult {
  const rng = mulberry32(seed);
  let gammaArr = new Uint8Array(P);
  // Cache of log marginal likelihood (kept as small number to avoid map of bitstring strings)
  const cache = new Map<number, number>();
  const computeLogMarg = (gammaBits: Uint8Array): number => {
    let id = 0;
    for (let j = 0; j < P; j++) if (gammaBits[j]) id |= 1 << j;
    if (cache.has(id)) return cache.get(id)!;
    const cols: number[] = [];
    for (let j = 0; j < P; j++) if (gammaBits[j]) cols.push(j);
    const card = cols.length;
    let logMarg = 0;
    if (card > 0) {
      const R2 = olsR2(X, y, cols);
      logMarg = ((N - card - 1) / 2) * Math.log(1 + g) - ((N - 1) / 2) * Math.log(1 + g * (1 - R2));
    }
    const logPrior = -logChoose(P, card) - Math.log(P + 1);
    const v = logMarg + logPrior;
    cache.set(id, v);
    return v;
  };

  let currLP = computeLogMarg(gammaArr);
  let accepts = 0;
  const inclCounts = new Array(P).fill(0);
  const sizeCounts = new Array(P + 1).fill(0);
  const visitCounts = new Map<string, number>();

  for (let t = 0; t < T; t++) {
    const j = Math.floor(rng() * P);
    const propArr = gammaArr.slice();
    propArr[j] = 1 - propArr[j];
    const propLP = computeLogMarg(propArr);
    if (Math.log(rng()) < propLP - currLP) {
      gammaArr = propArr;
      currLP = propLP;
      accepts++;
    }
    if (t >= burn) {
      let card = 0;
      let gString = '';
      for (let k = 0; k < P; k++) {
        if (gammaArr[k]) {
          card++;
          inclCounts[k]++;
        }
        gString += gammaArr[k] ? '1' : '0';
      }
      sizeCounts[card]++;
      visitCounts.set(gString, (visitCounts.get(gString) ?? 0) + 1);
    }
  }
  const samples = T - burn;
  const inclProbs = inclCounts.map((c) => c / samples);
  const sizeProbs = sizeCounts.map((c) => c / samples);
  const top = Array.from(visitCounts.entries()).sort((a, b) => b[1] - a[1]).slice(0, 5).map((e) => ({ gamma: e[0], logPost: e[1] / samples }));
  return {
    inclProbs,
    sizeProbs,
    topModels: top,
    acceptRate: accepts / T,
    uniqueVisited: visitCounts.size,
  };
}

interface Committed {
  snr: number;
  T: number;
  g: number;
}
const DEFAULTS: Committed = { snr: 2, T: 5000, g: N };

export default function VariableSelectionRJMCMC() {
  const [dSnr, setDSnr] = useState(DEFAULTS.snr);
  const [dT, setDT] = useState(DEFAULTS.T);
  const [dG, setDG] = useState(DEFAULTS.g);
  const [committed, setCommitted] = useState<Committed>(DEFAULTS);

  const commit = useCallback((next: Partial<Committed>) => {
    setCommitted((c) => ({ ...c, ...next }));
  }, []);

  const result = useMemo(() => {
    const { X, y } = generateData(committed.snr, 11);
    const enumR = enumerate(X, y, committed.g);
    const burn = Math.floor(committed.T * 0.2);
    const chain = runChain(X, y, committed.g, committed.T, burn, 17);
    let maxInclErr = 0;
    for (let j = 0; j < P; j++) {
      const e = Math.abs(enumR.inclProbs[j] - chain.inclProbs[j]);
      if (e > maxInclErr) maxInclErr = e;
    }
    return { enumR, chain, maxInclErr };
  }, [committed]);

  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const W = containerWidth || 800;
  const isMobile = W < 640;
  const panelW = isMobile ? W : (W - 24) / 2;
  const panelH = 300;

  const inclRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelW <= 0) return;
      const M = { top: 20, right: 18, bottom: 36, left: 40 };
      const w = panelW - M.left - M.right;
      const h = panelH - M.top - M.bottom;
      const g = svg.append('g').attr('transform', `translate(${M.left},${M.top})`);
      const js = d3.range(P);
      const x = d3.scaleBand().domain(js.map((j) => j.toString())).range([0, w]).padding(0.15);
      const xSub = d3.scaleBand().domain(['enum', 'chain']).range([0, x.bandwidth()]).padding(0.08);
      const y = d3.scaleLinear().domain([0, 1]).range([h, 0]);

      // Truth highlight rect
      for (const j of TRUE_ACTIVE) {
        g.append('rect')
          .attr('x', x(j.toString()) ?? 0)
          .attr('y', 0)
          .attr('width', x.bandwidth())
          .attr('height', h)
          .style('fill', 'var(--color-accent)')
          .style('opacity', 0.08);
      }
      for (const j of js) {
        g.append('rect')
          .attr('x', (x(j.toString()) ?? 0) + (xSub('enum') ?? 0))
          .attr('y', y(result.enumR.inclProbs[j]))
          .attr('width', xSub.bandwidth())
          .attr('height', h - y(result.enumR.inclProbs[j]))
          .style('fill', 'var(--color-text-secondary)');
        g.append('rect')
          .attr('x', (x(j.toString()) ?? 0) + (xSub('chain') ?? 0))
          .attr('y', y(result.chain.inclProbs[j]))
          .attr('width', xSub.bandwidth())
          .attr('height', h - y(result.chain.inclProbs[j]))
          .style('fill', 'var(--color-accent)');
      }
      g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x)).selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.append('g').call(d3.axisLeft(y).ticks(5, '.1f')).selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.append('text').attr('x', w / 2).attr('y', h + 28).attr('text-anchor', 'middle').style('font-size', '12px').style('font-family', 'var(--font-sans)').style('fill', 'var(--color-text)').text('covariate j (shaded = true active)');
      // Legend
      const lg = g.append('g').attr('transform', `translate(${w - 100},${0})`);
      lg.append('rect').attr('width', 12).attr('height', 12).style('fill', 'var(--color-text-secondary)');
      lg.append('text').attr('x', 16).attr('y', 10).style('font-size', '11px').style('fill', 'var(--color-text)').text('enumeration');
      lg.append('rect').attr('width', 12).attr('height', 12).attr('y', 16).style('fill', 'var(--color-accent)');
      lg.append('text').attr('x', 16).attr('y', 26).style('font-size', '11px').style('fill', 'var(--color-text)').text('chain (RJ-MH)');
    },
    [result, panelW, panelH],
  );

  const sizeRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelW <= 0) return;
      const M = { top: 20, right: 18, bottom: 36, left: 40 };
      const w = panelW - M.left - M.right;
      const h = panelH - M.top - M.bottom;
      const g = svg.append('g').attr('transform', `translate(${M.left},${M.top})`);
      const ks = d3.range(P + 1);
      const x = d3.scaleBand().domain(ks.map((k) => k.toString())).range([0, w]).padding(0.15);
      const xSub = d3.scaleBand().domain(['enum', 'chain']).range([0, x.bandwidth()]).padding(0.08);
      const yMax = Math.max(...result.enumR.sizeProbs, ...result.chain.sizeProbs) * 1.15;
      const y = d3.scaleLinear().domain([0, yMax]).range([h, 0]);

      for (const k of ks) {
        g.append('rect')
          .attr('x', (x(k.toString()) ?? 0) + (xSub('enum') ?? 0))
          .attr('y', y(result.enumR.sizeProbs[k]))
          .attr('width', xSub.bandwidth())
          .attr('height', h - y(result.enumR.sizeProbs[k]))
          .style('fill', 'var(--color-text-secondary)');
        g.append('rect')
          .attr('x', (x(k.toString()) ?? 0) + (xSub('chain') ?? 0))
          .attr('y', y(result.chain.sizeProbs[k]))
          .attr('width', xSub.bandwidth())
          .attr('height', h - y(result.chain.sizeProbs[k]))
          .style('fill', 'var(--color-accent)');
      }
      // Truth vertical line
      g.append('line')
        .attr('x1', (x(TRUE_ACTIVE.length.toString()) ?? 0) + x.bandwidth() / 2)
        .attr('x2', (x(TRUE_ACTIVE.length.toString()) ?? 0) + x.bandwidth() / 2)
        .attr('y1', 0)
        .attr('y2', h)
        .style('stroke', 'var(--color-text)')
        .style('stroke-dasharray', '3 3');
      g.append('text')
        .attr('x', (x(TRUE_ACTIVE.length.toString()) ?? 0) + x.bandwidth() / 2 + 4)
        .attr('y', 12)
        .style('font-size', '10px')
        .style('font-family', 'var(--font-mono)')
        .style('fill', 'var(--color-text)')
        .text(`|S★| = ${TRUE_ACTIVE.length}`);

      g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x)).selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.append('g').call(d3.axisLeft(y).ticks(5, '.2f')).selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.append('text').attr('x', w / 2).attr('y', h + 28).attr('text-anchor', 'middle').style('font-size', '12px').style('font-family', 'var(--font-sans)').style('fill', 'var(--color-text)').text('|S| (model cardinality)');
    },
    [result, panelW, panelH],
  );

  const sliderRow = (
    label: string,
    value: number,
    setVal: (n: number) => void,
    commitFn: (v: number) => void,
    min: number,
    max: number,
    step: number,
    fmt: (n: number) => string,
    aria: string,
  ) => (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 12,
        marginBottom: 6,
        fontFamily: 'var(--font-sans)',
        fontSize: 12,
        color: 'var(--color-text)',
      }}
    >
      <label style={{ minWidth: 110 }}>
        {label} = <strong>{fmt(value)}</strong>
      </label>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => setVal(parseFloat(e.target.value))}
        onMouseUp={() => commitFn(value)}
        onTouchEnd={() => commitFn(value)}
        onKeyUp={() => commitFn(value)}
        style={{ flex: 1 }}
        aria-label={aria}
      />
    </div>
  );

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <div style={{ marginBottom: 8 }}>
        {sliderRow('SNR (1/σ)', dSnr, setDSnr, (v) => commit({ snr: v }), 0.5, 5, 0.25, (n) => n.toFixed(2), 'Signal-to-noise ratio')}
        {sliderRow('g', dG, setDG, (v) => commit({ g: v }), 10, 200, 5, (n) => n.toFixed(0), 'g-prior g')}
        {sliderRow('chain length T', dT, setDT, (v) => commit({ T: Math.round(v) }), 1000, 10000, 500, (n) => n.toFixed(0), 'Chain length T')}
      </div>
      <div
        style={{
          marginBottom: 8,
          fontFamily: 'var(--font-mono)',
          fontSize: 11,
          color: 'var(--color-text-secondary)',
        }}
      >
        max |π_enum − π_chain| = {result.maxInclErr.toFixed(4)} · accept {(result.chain.acceptRate * 100).toFixed(1)}% ·
        unique models visited {result.chain.uniqueVisited} / 256 = {(result.chain.uniqueVisited / 256 * 100).toFixed(1)}%
      </div>
      <div
        style={{
          display: 'flex',
          flexDirection: isMobile ? 'column' : 'row',
          gap: 24,
          alignItems: 'flex-start',
        }}
      >
        <div style={{ width: isMobile ? '100%' : panelW }}>
          <svg ref={inclRef} width={panelW} height={panelH} />
        </div>
        <div style={{ width: isMobile ? '100%' : panelW }}>
          <svg ref={sizeRef} width={panelW} height={panelH} />
        </div>
      </div>
    </div>
  );
}
