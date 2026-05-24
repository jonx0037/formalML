import { useState, useMemo } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';

// T3 toy: 6 documents, 3 topics, graded-purity p(y|x).
// I(X;Y) = 0.470 nats (per brief §5.3); saturation rate R^dagger = log 3 ≈ 1.099 nats.
const N_DOCS = 6;
const N_Y = 3;
const K_CLUSTERS = 4; // one spare; should self-disable

const T3_P_YgX: number[][] = [
  [0.85, 0.1, 0.05],
  [0.85, 0.1, 0.05],
  [0.1, 0.85, 0.05],
  [0.1, 0.85, 0.05],
  [0.15, 0.15, 0.7],
  [0.15, 0.15, 0.7],
];
const T3_P_x = 1 / N_DOCS;

// Numerically-stable log-sum-exp (vectorized over an array)
function logSumExp(logs: number[]): number {
  let m = -Infinity;
  for (const v of logs) if (v > m) m = v;
  if (!isFinite(m)) return m;
  let s = 0;
  for (const v of logs) s += Math.exp(v - m);
  return m + Math.log(s);
}

// KL(p || q) for discrete distributions, with q clipped at 1e-300
function klDiv(p: number[], q: number[]): number {
  let s = 0;
  for (let i = 0; i < p.length; i++) {
    if (p[i] > 0) {
      const qi = q[i] > 1e-300 ? q[i] : 1e-300;
      s += p[i] * Math.log(p[i] / qi);
    }
  }
  return s;
}

// Mutual information I(X;T) in nats, from log_p(t|x) and p_x uniform
function muiXT(logPtgx: number[][], pt: number[]): number {
  // I(X;T) = sum_{x,t} p(x) p(t|x) [log p(t|x) - log p(t)]
  let s = 0;
  for (let x = 0; x < N_DOCS; x++) {
    for (let t = 0; t < K_CLUSTERS; t++) {
      const ptgx = Math.exp(logPtgx[x][t]);
      if (ptgx > 0 && pt[t] > 0) {
        s += T3_P_x * ptgx * (logPtgx[x][t] - Math.log(pt[t]));
      }
    }
  }
  return s;
}

// Mutual information I(T;Y) in nats from p(t,y)
function muiTY(pTY: number[][], pt: number[], py: number[]): number {
  let s = 0;
  for (let t = 0; t < K_CLUSTERS; t++) {
    for (let y = 0; y < N_Y; y++) {
      if (pTY[t][y] > 0 && pt[t] > 0 && py[y] > 0) {
        s += pTY[t][y] * Math.log(pTY[t][y] / (pt[t] * py[y]));
      }
    }
  }
  return s;
}

// Effective cluster count K_eff(β) = exp(H(p(t)))
function effectiveClusters(pt: number[]): number {
  let H = 0;
  for (const p of pt) if (p > 0) H -= p * Math.log(p);
  return Math.exp(H);
}

// Mulberry32 — deterministic seedable PRNG
function mulberry32(seed: number) {
  let state = seed >>> 0;
  return function next() {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Random softmax initialization: small log-weight scale to perturb the trivial encoder
function randomSoftmaxInit(rng: () => number, sigma = 0.3): number[][] {
  const logP: number[][] = [];
  for (let x = 0; x < N_DOCS; x++) {
    const logits = Array.from({ length: K_CLUSTERS }, () => sigma * (rng() - 0.5) * 2);
    const lse = logSumExp(logits);
    logP.push(logits.map((l) => l - lse));
  }
  return logP;
}

// One sweep of the IB iteration starting from logPtgx, holding (q, r) implicit.
type IBState = {
  logPtgx: number[][]; // shape [N_DOCS][K_CLUSTERS], rows sum to 1 in prob-space
  pt: number[]; // shape [K_CLUSTERS]
  pYgt: number[][]; // shape [K_CLUSTERS][N_Y]
  IXT: number; // in nats
  ITY: number; // in nats
  K_eff: number;
  F_beta: number;
  iters: number;
};

function runIB(beta: number, init: number[][], maxIters = 120, tol = 1e-8): IBState {
  let logPtgx = init.map((row) => row.slice());

  // Compute p(t) and p(y|t) from logPtgx
  const computeMarginals = (lp: number[][]) => {
    const pt = new Array(K_CLUSTERS).fill(0);
    const pTY = Array.from({ length: K_CLUSTERS }, () => new Array(N_Y).fill(0));
    for (let x = 0; x < N_DOCS; x++) {
      for (let t = 0; t < K_CLUSTERS; t++) {
        const ptgx = Math.exp(lp[x][t]);
        const w = T3_P_x * ptgx;
        pt[t] += w;
        for (let y = 0; y < N_Y; y++) pTY[t][y] += w * T3_P_YgX[x][y];
      }
    }
    const pYgt = pTY.map((row, t) => row.map((v) => (pt[t] > 0 ? v / pt[t] : 0)));
    return { pt, pTY, pYgt };
  };

  let last_F = Infinity;
  let iters = 0;
  // py is fixed marginal of Y under T3
  const py = new Array(N_Y).fill(0);
  for (let x = 0; x < N_DOCS; x++)
    for (let y = 0; y < N_Y; y++) py[y] += T3_P_x * T3_P_YgX[x][y];

  for (let it = 0; it < maxIters; it++) {
    const { pt, pTY, pYgt } = computeMarginals(logPtgx);

    // Update log p(t|x) ∝ log p(t) - β * D_KL(p(y|x) || p(y|t)), normalize via logsumexp
    const newLog: number[][] = [];
    for (let x = 0; x < N_DOCS; x++) {
      const log_pt = pt.map((p) => (p > 1e-300 ? Math.log(p) : Math.log(1e-300)));
      const logits = new Array(K_CLUSTERS).fill(0);
      for (let t = 0; t < K_CLUSTERS; t++) {
        const kl = klDiv(T3_P_YgX[x], pYgt[t]);
        logits[t] = log_pt[t] - beta * kl;
      }
      const lse = logSumExp(logits);
      newLog.push(logits.map((l) => l - lse));
    }
    logPtgx = newLog;

    // Compute F_β for convergence check
    const m = computeMarginals(logPtgx);
    const F_beta = muiXT(logPtgx, m.pt) - beta * muiTY(m.pTY, m.pt, py);
    iters = it + 1;
    if (Math.abs(last_F - F_beta) < tol * (Math.abs(last_F) + 1e-12)) break;
    last_F = F_beta;
  }

  const { pt, pTY, pYgt } = computeMarginals(logPtgx);
  const IXT = muiXT(logPtgx, pt);
  const ITY = muiTY(pTY, pt, py);
  return { logPtgx, pt, pYgt, IXT, ITY, K_eff: effectiveClusters(pt), F_beta: IXT - beta * ITY, iters };
}

// Annealing sweep across log-spaced β, warm-starting each next β from the previous
function annealSweep(betas: number[], seed = 7): IBState[] {
  const rng = mulberry32(seed);
  const trace: IBState[] = [];
  let init = randomSoftmaxInit(rng, 0.3);
  for (const b of betas) {
    const s = runIB(b, init, 120, 1e-9);
    trace.push(s);
    // Warm-start; add a tiny perturbation to break exact saddle-point degeneracy
    init = s.logPtgx.map((row) => {
      const perturbed = row.map((v) => v + 0.01 * (rng() - 0.5));
      const lse = logSumExp(perturbed);
      return perturbed.map((v) => v - lse);
    });
  }
  return trace;
}

export default function T3IBCurveTracer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();

  // Precompute a 40-point log-spaced β sweep, annealed
  const betas = useMemo(() => {
    const lo = Math.log(0.3);
    const hi = Math.log(100);
    return Array.from({ length: 40 }, (_, i) => Math.exp(lo + (hi - lo) * (i / 39)));
  }, []);
  const trace = useMemo(() => annealSweep(betas, 7), [betas]);
  const [betaIdx, setBetaIdx] = useState<number>(28);

  const W = Math.max(containerWidth || 760, 320);
  const isMobile = W < 720;
  const panelW = isMobile ? W : Math.min((W - 24) / 2, 460);
  const panelH = Math.round(panelW * 0.78);

  const beta = betas[betaIdx];
  const cur = trace[betaIdx];

  // Saturation lines from brief
  const IXY = 0.470;
  const Rdagger = Math.log(3); // = 1.0986 nats

  // Panel (a): IB plane
  const planeRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const margin = { top: 28, right: 18, bottom: 44, left: 56 };
      const innerW = panelW - margin.left - margin.right;
      const innerH = panelH - margin.top - margin.bottom;
      if (innerW <= 0 || innerH <= 0) return;
      svg.attr('viewBox', `0 0 ${panelW} ${panelH}`).attr('width', '100%').attr('height', panelH);

      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      // Domain from the trace + a bit of headroom
      const maxIXT = Math.max(Rdagger * 1.15, d3.max(trace, (s) => s.IXT) ?? 1.2);
      const maxITY = Math.max(IXY * 1.2, d3.max(trace, (s) => s.ITY) ?? 0.6);
      const xScale = d3.scaleLinear().domain([0, maxIXT]).range([0, innerW]);
      const yScale = d3.scaleLinear().domain([0, maxITY]).range([innerH, 0]);

      // DPI ceiling at I(X;Y) = 0.470
      g.append('line')
        .attr('x1', xScale(0)).attr('x2', xScale(maxIXT))
        .attr('y1', yScale(IXY)).attr('y2', yScale(IXY))
        .style('stroke', 'var(--color-text-secondary)')
        .attr('stroke-dasharray', '4 3').attr('stroke-width', 1.0);
      g.append('text')
        .attr('x', xScale(maxIXT) - 6).attr('y', yScale(IXY) - 5)
        .attr('text-anchor', 'end')
        .style('fill', 'var(--color-text-secondary)').attr('font-size', 10)
        .text(`I(X;Y) = ${IXY.toFixed(3)} nats`);

      // Saturation rate vertical
      g.append('line')
        .attr('x1', xScale(Rdagger)).attr('x2', xScale(Rdagger))
        .attr('y1', yScale(0)).attr('y2', yScale(maxITY))
        .style('stroke', 'var(--color-text-secondary)')
        .attr('stroke-dasharray', '3 4').attr('stroke-width', 0.9);
      g.append('text')
        .attr('x', xScale(Rdagger) + 6).attr('y', yScale(maxITY) + 12)
        .attr('text-anchor', 'start')
        .style('fill', 'var(--color-text-secondary)').attr('font-size', 10)
        .text(`R† = log 3 = ${Rdagger.toFixed(3)} nats`);

      // Diagonal R = I(T;Y)
      g.append('line')
        .attr('x1', xScale(0)).attr('x2', xScale(Math.min(maxIXT, maxITY)))
        .attr('y1', yScale(0)).attr('y2', yScale(Math.min(maxIXT, maxITY)))
        .style('stroke', 'var(--color-border)')
        .attr('stroke-dasharray', '3 3').attr('stroke-width', 1.0);

      // Axes
      const xAxis = d3.axisBottom(xScale).ticks(5);
      const yAxis = d3.axisLeft(yScale).ticks(5);
      const xg = g.append('g').attr('transform', `translate(0,${innerH})`).call(xAxis);
      const yg = g.append('g').call(yAxis);
      xg.selectAll('text').style('fill', 'var(--color-text)').attr('font-size', 10);
      yg.selectAll('text').style('fill', 'var(--color-text)').attr('font-size', 10);
      xg.selectAll('line, path').style('stroke', 'var(--color-border)');
      yg.selectAll('line, path').style('stroke', 'var(--color-border)');

      g.append('text')
        .attr('x', innerW / 2).attr('y', innerH + 32).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)').attr('font-size', 11)
        .text('I(X;T) (nats)');
      g.append('text')
        .attr('transform', `translate(-40,${innerH / 2}) rotate(-90)`).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)').attr('font-size', 11)
        .text('I(T;Y) (nats)');
      g.append('text')
        .attr('x', innerW / 2).attr('y', -10).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)').attr('font-size', 11)
        .text('IB curve traced by annealing across 40 β values (T3 toy)');

      // Plot the trace as a path
      const lineGen = d3.line<IBState>().x((d) => xScale(d.IXT)).y((d) => yScale(d.ITY));
      g.append('path')
        .attr('d', lineGen(trace) ?? '')
        .style('fill', 'none').style('stroke', 'var(--color-accent)').attr('stroke-width', 2);

      // Trace markers
      g.selectAll('.trace-dot').data(trace).enter()
        .append('circle')
        .attr('cx', (d) => xScale(d.IXT)).attr('cy', (d) => yScale(d.ITY))
        .attr('r', 2.2)
        .style('fill', 'var(--color-accent)')
        .attr('opacity', 0.6);

      // Highlight the active β operating point
      g.append('circle')
        .attr('cx', xScale(cur.IXT)).attr('cy', yScale(cur.ITY))
        .attr('r', 7)
        .style('fill', '#D97706')
        .attr('stroke', 'var(--color-text)').attr('stroke-width', 1.4);
      g.append('text')
        .attr('x', xScale(cur.IXT) + 10).attr('y', yScale(cur.ITY) - 8)
        .style('fill', 'var(--color-text)').attr('font-size', 11).attr('font-weight', 600)
        .text(`β = ${beta.toFixed(2)}`);
      g.append('text')
        .attr('x', xScale(cur.IXT) + 10).attr('y', yScale(cur.ITY) + 6)
        .style('fill', 'var(--color-text-secondary)').attr('font-size', 10)
        .text(`(${cur.IXT.toFixed(3)}, ${cur.ITY.toFixed(3)}) nats`);
    },
    [betaIdx, panelW, panelH, trace],
  );

  // Panel (b): I(X;T), I(T;Y) vs β on log axis
  const sweepRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const margin = { top: 28, right: 18, bottom: 44, left: 52 };
      const innerW = panelW - margin.left - margin.right;
      const innerH = panelH - margin.top - margin.bottom;
      if (innerW <= 0 || innerH <= 0) return;
      svg.attr('viewBox', `0 0 ${panelW} ${panelH}`).attr('width', '100%').attr('height', panelH);

      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const xScale = d3.scaleLog().domain([betas[0], betas[betas.length - 1]]).range([0, innerW]);
      const ymax = Math.max(Rdagger * 1.15, d3.max(trace, (s) => s.IXT) ?? 1.2);
      const yScale = d3.scaleLinear().domain([0, ymax]).range([innerH, 0]);

      // Axes
      const xAxis = d3.axisBottom(xScale).ticks(6, '.1f');
      const yAxis = d3.axisLeft(yScale).ticks(5);
      const xg = g.append('g').attr('transform', `translate(0,${innerH})`).call(xAxis);
      const yg = g.append('g').call(yAxis);
      xg.selectAll('text').style('fill', 'var(--color-text)').attr('font-size', 10);
      yg.selectAll('text').style('fill', 'var(--color-text)').attr('font-size', 10);
      xg.selectAll('line, path').style('stroke', 'var(--color-border)');
      yg.selectAll('line, path').style('stroke', 'var(--color-border)');

      g.append('text')
        .attr('x', innerW / 2).attr('y', innerH + 32).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)').attr('font-size', 11)
        .text('β (log scale)');
      g.append('text')
        .attr('transform', `translate(-38,${innerH / 2}) rotate(-90)`).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)').attr('font-size', 11)
        .text('information (nats)');
      g.append('text')
        .attr('x', innerW / 2).attr('y', -10).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)').attr('font-size', 11)
        .text('I(X;T) and I(T;Y) vs β');

      // I(X;T) line
      const lineIXT = d3.line<IBState>().x((d, i) => xScale(betas[i])).y((d) => yScale(d.IXT));
      g.append('path')
        .attr('d', lineIXT(trace) ?? '')
        .style('fill', 'none').style('stroke', 'var(--color-accent)').attr('stroke-width', 2);
      g.append('text')
        .attr('x', xScale(betas[betas.length - 1])).attr('y', yScale(trace[trace.length - 1].IXT) - 8)
        .attr('text-anchor', 'end')
        .style('fill', 'var(--color-accent)').attr('font-size', 11)
        .text('I(X;T)');

      // I(T;Y) line
      const lineITY = d3.line<IBState>().x((d, i) => xScale(betas[i])).y((d) => yScale(d.ITY));
      g.append('path')
        .attr('d', lineITY(trace) ?? '')
        .style('fill', 'none').style('stroke', '#D97706').attr('stroke-width', 2);
      g.append('text')
        .attr('x', xScale(betas[betas.length - 1])).attr('y', yScale(trace[trace.length - 1].ITY) - 8)
        .attr('text-anchor', 'end')
        .style('fill', '#D97706').attr('font-size', 11)
        .text('I(T;Y)');

      // Active β cursor
      g.append('line')
        .attr('x1', xScale(beta)).attr('x2', xScale(beta))
        .attr('y1', yScale(0)).attr('y2', yScale(ymax))
        .style('stroke', 'var(--color-text)').attr('stroke-dasharray', '3 3').attr('stroke-width', 1.2);

      // DPI ceiling
      g.append('line')
        .attr('x1', xScale(betas[0])).attr('x2', xScale(betas[betas.length - 1]))
        .attr('y1', yScale(IXY)).attr('y2', yScale(IXY))
        .style('stroke', '#D97706').attr('stroke-dasharray', '2 4').attr('stroke-width', 0.9)
        .attr('opacity', 0.5);
    },
    [betaIdx, panelW, panelH, trace, betas, beta],
  );

  return (
    <div ref={containerRef} style={{ padding: '1rem 0' }}>
      <div style={{ marginBottom: '0.75rem' }}>
        <label
          htmlFor="t3-beta-slider"
          style={{ fontSize: '0.85rem', color: 'var(--color-text-secondary)', display: 'block' }}
        >
          β = <strong style={{ color: 'var(--color-text)' }}>{beta.toFixed(2)}</strong>
          {'   '}
          <span style={{ color: 'var(--color-text-secondary)' }}>
            (idx {betaIdx + 1}/{betas.length} · I(X;T) = {cur.IXT.toFixed(3)} nats · I(T;Y) = {cur.ITY.toFixed(3)} nats · K_eff = {cur.K_eff.toFixed(2)})
          </span>
        </label>
        <input
          id="t3-beta-slider"
          type="range"
          min={0}
          max={betas.length - 1}
          step={1}
          value={betaIdx}
          onChange={(e) => setBetaIdx(parseInt(e.target.value, 10))}
          aria-label="Beta index along the log-spaced sweep"
          style={{ width: '100%' }}
        />
      </div>
      <div
        style={{
          display: 'flex',
          flexDirection: isMobile ? 'column' : 'row',
          gap: '1rem',
          alignItems: 'flex-start',
        }}
      >
        <div style={{ flex: 1, minWidth: 0 }}>
          <svg ref={planeRef} role="img" aria-label="T3 IB curve in the information plane" />
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <svg ref={sweepRef} role="img" aria-label="I(X;T) and I(T;Y) vs beta on log axis" />
        </div>
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.75rem' }}>
        The sweep is β-annealed (each new β warm-starts from the previous β's converged encoder). Saturation point sits at (R†, I(X;Y)) = (log 3, 0.470) nats; the spare fourth cluster self-disables. Two phase transitions are visible in the right panel as steepness changes in I(T;Y).
      </p>
    </div>
  );
}
