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
  [0.65, 0.3, 0.05], // docs 0, 1: mostly topic 0, heavy topic-1 overlap
  [0.65, 0.3, 0.05],
  [0.3, 0.65, 0.05], // docs 2, 3: the mirror image
  [0.3, 0.65, 0.05],
  [0.1, 0.1, 0.8], // docs 4, 5: sharply topic 2
  [0.1, 0.1, 0.8],
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

// Perplexity exp H(p(t)) of the cluster marginal (§5.4, eq. 5.2) — shown only for contrast:
// it scores a uniform trivial encoder as k and counts duplicate clusters twice.
function perplexity(pt: number[]): number {
  let H = 0;
  for (const p of pt) if (p > 0) H -= p * Math.log(p);
  return Math.exp(H);
}

// First split, predicted (§5.4, eq. 5.5): the trivial encoder loses stability at β = 1/λ₂(C), where
// C(x, x') = Σ_y p(y|x) p(x'|y) is the two-step chain x → y → x'. Power iteration on p-mean-zero
// vectors removes λ₁ = 1 (constant eigenvector); C is symmetric here because p(x) is uniform.
const T3_BETA_SPLIT = (() => {
  const py = new Array(N_Y).fill(0);
  for (let x = 0; x < N_DOCS; x++) for (let y = 0; y < N_Y; y++) py[y] += T3_P_x * T3_P_YgX[x][y];
  const C = Array.from({ length: N_DOCS }, (_, x) =>
    Array.from({ length: N_DOCS }, (_, xp) => {
      let s = 0;
      for (let y = 0; y < N_Y; y++) s += (T3_P_YgX[x][y] * T3_P_x * T3_P_YgX[xp][y]) / py[y];
      return s;
    }),
  );
  let v = Array.from({ length: N_DOCS }, (_, x) => Math.cos(1 + 2.3 * x));
  let lambda = 0;
  for (let it = 0; it < 500; it++) {
    const mean = v.reduce((a, b) => a + T3_P_x * b, 0);
    v = v.map((vi) => vi - mean);
    const w = C.map((row) => row.reduce((a, c, xp) => a + c * v[xp], 0));
    const norm = Math.hypot(...w);
    lambda = norm / Math.hypot(...v);
    v = w.map((wi) => wi / norm);
  }
  return 1 / lambda;
})();

// Two-cluster plateau: a hard split into blocks of mass 1/3 and 2/3 has exp I(X;T) = exp H(1/3, 2/3)
const PLATEAU = Math.exp(-(1 / 3) * Math.log(1 / 3) - (2 / 3) * Math.log(2 / 3));

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
  K_eff: number; // exp I(X;T): effective number of distinguishable clusters (§5.4, eq. 5.3)
  perplexity: number; // exp H(p(t)), for contrast
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
    if (it > 20 && Math.abs(last_F - F_beta) < tol * (Math.abs(last_F) + 1e-12)) break;
    last_F = F_beta;
  }

  const { pt, pYgt } = computeMarginals(logPtgx);
  let IXT = 0;
  for (let x = 0; x < N_DOCS; x++)
    for (let t = 0; t < K_CLUSTERS; t++) {
      const ptgx = Math.exp(logPtgx[x][t]);
      if (ptgx > 0 && pt[t] > 0) IXT += T3_P_x * ptgx * (logPtgx[x][t] - Math.log(pt[t]));
    }
  return { logPtgx, pt, pYgt, K_eff: Math.exp(IXT), perplexity: perplexity(pt) };
}

function annealSweep(betas: number[], seed = 7): IBState[] {
  const rng = mulberry32(seed);
  const trace: IBState[] = [];
  let init = randomSoftmaxInit(rng, 0.3);
  for (const b of betas) {
    // Generous cap: convergence slows sharply near each phase transition
    const s = runIB(b, init, 3000, 1e-12);
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

  // Panel (a): K_eff(β) = exp I(X;T) on log-β, with the perplexity for contrast and the predicted split
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

      // Horizontal guides at 1 cluster, the two-cluster plateau exp H(1/3, 2/3), and 3 clusters
      [
        { k: 1, label: '1' },
        { k: PLATEAU, label: `exp H(⅓, ⅔) = ${PLATEAU.toFixed(2)}` },
        { k: 3, label: '3' },
      ].forEach(({ k, label }) => {
        g.append('line')
          .attr('x1', xScale(betas[0])).attr('x2', xScale(betas[betas.length - 1]))
          .attr('y1', yScale(k)).attr('y2', yScale(k))
          .style('stroke', 'var(--color-border)')
          .attr('stroke-dasharray', '2 4').attr('stroke-width', 0.9)
          .attr('opacity', 0.6);
        g.append('text')
          .attr('x', xScale(betas[0]) + 4).attr('y', yScale(k) - 3)
          .style('fill', 'var(--color-text-secondary)').attr('font-size', 9)
          .text(label);
      });

      // Predicted first split β = 1/λ₂
      g.append('line')
        .attr('x1', xScale(T3_BETA_SPLIT)).attr('x2', xScale(T3_BETA_SPLIT))
        .attr('y1', yScale(0.8)).attr('y2', yScale(K_CLUSTERS + 0.2))
        .style('stroke', 'var(--color-text-secondary)')
        .attr('stroke-dasharray', '6 3').attr('stroke-width', 1.1);
      g.append('text')
        .attr('x', xScale(T3_BETA_SPLIT) + 4).attr('y', yScale(K_CLUSTERS + 0.05))
        .style('fill', 'var(--color-text-secondary)').attr('font-size', 9)
        .text(`predicted split 1/λ₂ = ${T3_BETA_SPLIT.toFixed(2)}`);

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
        .text('K_eff(β) = exp I(X;T)');
      g.append('text')
        .attr('x', innerW / 2).attr('y', -10).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)').attr('font-size', 11)
        .text('Distinguishable clusters across the β sweep');

      // Perplexity exp H(p(t)) for contrast (dotted)
      const perpLine = d3.line<IBState>().x((d, i) => xScale(betas[i])).y((d) => yScale(d.perplexity));
      g.append('path')
        .attr('d', perpLine(trace) ?? '')
        .style('fill', 'none').style('stroke', 'var(--color-text-secondary)')
        .attr('stroke-dasharray', '1.5 3').attr('stroke-width', 1.4);
      g.append('text')
        .attr('x', xScale(betas[betas.length - 1]) - 4).attr('y', yScale(trace[trace.length - 1].perplexity) - 5)
        .attr('text-anchor', 'end')
        .style('fill', 'var(--color-text-secondary)').attr('font-size', 9)
        .text('perplexity exp H(p(t))');

      // Plot K_eff = exp I(X;T)
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
        .text('document x  (A: docs 0–1 mostly topic 0 · B: 2–3 mostly topic 1 · C: 4–5 sharply topic 2)');
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
          <svg ref={staircaseRef} role="img" aria-label="Effective cluster count exp I(X;T) across the beta sweep" />
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <svg ref={heatmapRef} role="img" aria-label="Encoder p(t|x) heatmap at the active beta" />
        </div>
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.75rem' }}>
        K_eff = exp I(X;T) counts distinguishable clusters. At small β every document has the same assignment (K_eff = 1). The trivial encoder loses stability at the predicted β = 1/λ₂ ≈ {T3_BETA_SPLIT.toFixed(2)}, where the topic-2 pair — the documents farthest from the rest — splits off; K_eff then levels off near {PLATEAU.toFixed(2)}, not 2, because that cluster holds a third of the documents. Topics 0 and 1 separate near β ≈ 7.8, a value that has to be measured, and K_eff climbs to 3. All four clusters stay in use: the topic-2 pair is spread over two that make identical predictions, which the dotted perplexity counts twice.
      </p>
    </div>
  );
}
