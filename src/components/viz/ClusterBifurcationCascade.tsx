import { useState, useMemo } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';

// T3 toy — same as T3IBCurveTracer. Kept inline per brief's "no algorithm helpers"
// directive for shared modules.
const N_DOCS = 6;
const N_Y = 3;
const K_CLUSTERS = 4;

const T3_P_YgX: number[][] = [
  [0.85, 0.1, 0.05],
  [0.85, 0.1, 0.05],
  [0.1, 0.85, 0.05],
  [0.1, 0.85, 0.05],
  [0.15, 0.15, 0.7],
  [0.15, 0.15, 0.7],
];
const T3_P_x = 1 / N_DOCS;

function logSumExp(logs: number[]): number {
  let m = -Infinity;
  for (const v of logs) if (v > m) m = v;
  if (!isFinite(m)) return m;
  let s = 0;
  for (const v of logs) s += Math.exp(v - m);
  return m + Math.log(s);
}

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

function effectiveClusters(pt: number[]): number {
  let H = 0;
  for (const p of pt) if (p > 0) H -= p * Math.log(p);
  return Math.exp(H);
}

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

function randomSoftmaxInit(rng: () => number, sigma = 0.3): number[][] {
  const logP: number[][] = [];
  for (let x = 0; x < N_DOCS; x++) {
    const logits = Array.from({ length: K_CLUSTERS }, () => sigma * (rng() - 0.5) * 2);
    const lse = logSumExp(logits);
    logP.push(logits.map((l) => l - lse));
  }
  return logP;
}

type IBState = {
  logPtgx: number[][];
  pt: number[];
  pYgt: number[][];
  K_eff: number;
};

function runIB(beta: number, init: number[][], maxIters = 120, tol = 1e-8): IBState {
  let logPtgx = init.map((row) => row.slice());

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
  const py = new Array(N_Y).fill(0);
  for (let x = 0; x < N_DOCS; x++)
    for (let y = 0; y < N_Y; y++) py[y] += T3_P_x * T3_P_YgX[x][y];

  for (let it = 0; it < maxIters; it++) {
    const { pt, pYgt } = computeMarginals(logPtgx);
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

    // Convergence check via F_β
    const m = computeMarginals(logPtgx);
    let IXT = 0;
    for (let x = 0; x < N_DOCS; x++)
      for (let t = 0; t < K_CLUSTERS; t++) {
        const ptgx = Math.exp(logPtgx[x][t]);
        if (ptgx > 0 && m.pt[t] > 0)
          IXT += T3_P_x * ptgx * (logPtgx[x][t] - Math.log(m.pt[t]));
      }
    let ITY = 0;
    for (let t = 0; t < K_CLUSTERS; t++)
      for (let y = 0; y < N_Y; y++) {
        const pTY = m.pt[t] * m.pYgt[t][y];
        if (pTY > 0 && m.pt[t] > 0 && py[y] > 0)
          ITY += pTY * Math.log(pTY / (m.pt[t] * py[y]));
      }
    const F_beta = IXT - beta * ITY;
    if (Math.abs(last_F - F_beta) < tol * (Math.abs(last_F) + 1e-12)) break;
    last_F = F_beta;
  }

  const { pt, pYgt } = computeMarginals(logPtgx);
  return { logPtgx, pt, pYgt, K_eff: effectiveClusters(pt) };
}

function annealSweep(betas: number[], seed = 7): IBState[] {
  const rng = mulberry32(seed);
  const trace: IBState[] = [];
  let init = randomSoftmaxInit(rng, 0.3);
  for (const b of betas) {
    const s = runIB(b, init, 120, 1e-9);
    trace.push(s);
    init = s.logPtgx.map((row) => {
      const perturbed = row.map((v) => v + 0.01 * (rng() - 0.5));
      const lse = logSumExp(perturbed);
      return perturbed.map((v) => v - lse);
    });
  }
  return trace;
}

const CLUSTER_COLORS = ['#0F6E56', '#534AB7', '#D97706', '#0EA5E9'];

export default function ClusterBifurcationCascade() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
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

  // Panel (a): K_eff(β) staircase on log-β
  const staircaseRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const margin = { top: 28, right: 18, bottom: 44, left: 48 };
      const innerW = panelW - margin.left - margin.right;
      const innerH = panelH - margin.top - margin.bottom;
      if (innerW <= 0 || innerH <= 0) return;
      svg.attr('viewBox', `0 0 ${panelW} ${panelH}`).attr('width', '100%').attr('height', panelH);

      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const xScale = d3.scaleLog().domain([betas[0], betas[betas.length - 1]]).range([0, innerW]);
      const yScale = d3.scaleLinear().domain([0.8, K_CLUSTERS + 0.2]).range([innerH, 0]);

      // Horizontal guides at K_eff = 1, 2, 3, 4
      [1, 2, 3, 4].forEach((k) => {
        g.append('line')
          .attr('x1', xScale(betas[0])).attr('x2', xScale(betas[betas.length - 1]))
          .attr('y1', yScale(k)).attr('y2', yScale(k))
          .style('stroke', 'var(--color-border)')
          .attr('stroke-dasharray', '2 4').attr('stroke-width', 0.9)
          .attr('opacity', 0.6);
        g.append('text')
          .attr('x', xScale(betas[0]) + 4).attr('y', yScale(k) - 3)
          .style('fill', 'var(--color-text-secondary)').attr('font-size', 9)
          .text(`K_eff = ${k}`);
      });

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
        .attr('transform', `translate(-34,${innerH / 2}) rotate(-90)`).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)').attr('font-size', 11)
        .text('K_eff(β) = exp(H(p(t)))');
      g.append('text')
        .attr('x', innerW / 2).attr('y', -10).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)').attr('font-size', 11)
        .text('Effective cluster count across the β sweep');

      // Plot the staircase
      const line = d3.line<IBState>().x((d, i) => xScale(betas[i])).y((d) => yScale(d.K_eff));
      g.append('path')
        .attr('d', line(trace) ?? '')
        .style('fill', 'none').style('stroke', 'var(--color-accent)').attr('stroke-width', 2);
      g.selectAll('.k-dot').data(trace).enter()
        .append('circle')
        .attr('cx', (d, i) => xScale(betas[i])).attr('cy', (d) => yScale(d.K_eff))
        .attr('r', 2.4)
        .style('fill', 'var(--color-accent)')
        .attr('opacity', 0.7);

      // Active β cursor
      g.append('line')
        .attr('x1', xScale(beta)).attr('x2', xScale(beta))
        .attr('y1', yScale(0.8)).attr('y2', yScale(K_CLUSTERS + 0.2))
        .style('stroke', '#D97706').attr('stroke-dasharray', '4 3').attr('stroke-width', 1.4);
      g.append('circle')
        .attr('cx', xScale(beta)).attr('cy', yScale(cur.K_eff))
        .attr('r', 6)
        .style('fill', '#D97706')
        .attr('stroke', 'var(--color-text)').attr('stroke-width', 1.2);
      g.append('text')
        .attr('x', xScale(beta) + 8).attr('y', yScale(cur.K_eff) - 8)
        .style('fill', 'var(--color-text)').attr('font-size', 11).attr('font-weight', 600)
        .text(`β = ${beta.toFixed(2)}`);
      g.append('text')
        .attr('x', xScale(beta) + 8).attr('y', yScale(cur.K_eff) + 8)
        .style('fill', 'var(--color-text-secondary)').attr('font-size', 10)
        .text(`K_eff = ${cur.K_eff.toFixed(2)}`);
    },
    [betaIdx, panelW, panelH, trace, betas, beta],
  );

  // Panel (b): 6×4 encoder heatmap p(t|x) at active β
  const heatmapRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const margin = { top: 50, right: 16, bottom: 56, left: 64 };
      const innerW = panelW - margin.left - margin.right;
      const innerH = panelH - margin.top - margin.bottom;
      if (innerW <= 0 || innerH <= 0) return;
      svg.attr('viewBox', `0 0 ${panelW} ${panelH}`).attr('width', '100%').attr('height', panelH);

      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const cellW = innerW / N_DOCS;
      const cellH = innerH / K_CLUSTERS;

      // Cells
      for (let x = 0; x < N_DOCS; x++) {
        for (let t = 0; t < K_CLUSTERS; t++) {
          const p = Math.exp(cur.logPtgx[x][t]);
          g.append('rect')
            .attr('x', x * cellW + 1)
            .attr('y', t * cellH + 1)
            .attr('width', cellW - 2)
            .attr('height', cellH - 2)
            .style('fill', CLUSTER_COLORS[t])
            .attr('opacity', Math.max(0.05, p));
          // Label opacity numeric inside cell for non-trivial cells
          if (p > 0.1) {
            g.append('text')
              .attr('x', x * cellW + cellW / 2)
              .attr('y', t * cellH + cellH / 2 + 4)
              .attr('text-anchor', 'middle')
              .style('fill', p > 0.6 ? '#FFFFFF' : 'var(--color-text)')
              .attr('font-size', 10)
              .text(p.toFixed(2));
          }
        }
      }

      // Column labels (documents)
      const docTopics = ['(A)', '(A)', '(B)', '(B)', '(C)', '(C)'];
      for (let x = 0; x < N_DOCS; x++) {
        g.append('text')
          .attr('x', x * cellW + cellW / 2).attr('y', -24)
          .attr('text-anchor', 'middle').attr('font-size', 11)
          .style('fill', 'var(--color-text)')
          .text(`x${x}`);
        g.append('text')
          .attr('x', x * cellW + cellW / 2).attr('y', -10)
          .attr('text-anchor', 'middle').attr('font-size', 9)
          .style('fill', 'var(--color-text-secondary)')
          .text(docTopics[x]);
      }

      // Row labels (clusters)
      for (let t = 0; t < K_CLUSTERS; t++) {
        g.append('text')
          .attr('x', -8).attr('y', t * cellH + cellH / 2 + 4)
          .attr('text-anchor', 'end').attr('font-size', 11)
          .style('fill', 'var(--color-text)')
          .text(`t = ${t}`);
        // Cluster marginal next to label
        g.append('text')
          .attr('x', innerW + 6).attr('y', t * cellH + cellH / 2 + 4)
          .attr('text-anchor', 'start').attr('font-size', 9)
          .style('fill', 'var(--color-text-secondary)')
          .text(`p(t) = ${cur.pt[t].toFixed(2)}`);
      }

      g.append('text')
        .attr('x', innerW / 2).attr('y', innerH + 26).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)').attr('font-size', 11)
        .text('document x  (A: docs 0–1 topic-0 pure · B: 2–3 topic-1 · C: 4–5 topic-2)');
      g.append('text')
        .attr('x', innerW / 2).attr('y', -38).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)').attr('font-size', 11)
        .text(`Encoder p(t | x) at β = ${beta.toFixed(2)}  ·  K_eff = ${cur.K_eff.toFixed(2)}`);
    },
    [betaIdx, panelW, panelH, trace, cur],
  );

  return (
    <div ref={containerRef} style={{ padding: '1rem 0' }}>
      <div style={{ marginBottom: '0.75rem' }}>
        <label
          htmlFor="bifurcation-beta"
          style={{ fontSize: '0.85rem', color: 'var(--color-text-secondary)', display: 'block' }}
        >
          β = <strong style={{ color: 'var(--color-text)' }}>{beta.toFixed(2)}</strong>
          {'   '}
          <span style={{ color: 'var(--color-text-secondary)' }}>
            (idx {betaIdx + 1}/{betas.length} · K_eff = {cur.K_eff.toFixed(2)})
          </span>
        </label>
        <input
          id="bifurcation-beta"
          type="range"
          min={0}
          max={betas.length - 1}
          step={1}
          value={betaIdx}
          onChange={(e) => setBetaIdx(parseInt(e.target.value, 10))}
          aria-label="Beta index"
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
          <svg ref={staircaseRef} role="img" aria-label="K_eff staircase across the beta sweep" />
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <svg ref={heatmapRef} role="img" aria-label="Encoder p(t|x) heatmap at the active beta" />
        </div>
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.75rem' }}>
        At small β, all six documents share one cluster (K_eff ≈ 1). Crossing the first phase transition splits the cluster structure — first the topic-2 pair separates from the topic-0/topic-1 group (purity 0.70 is the least correlated), then the topic-0 and topic-1 pairs separate at a higher β. The fourth allocated cluster self-disables. At large β, assignments approach one-hot and K_eff stabilizes near 3.
      </p>
    </div>
  );
}
