import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { gaussianPair, mulberry32 } from './shared/bayesian-ml';

// =============================================================================
// BayesianGMMOccam — §4 Bishop §10.2 Bayesian GMM with ARD
// =============================================================================
// CAVI for Bayesian GMM on a synthetic 2D dataset (4 well-separated clusters
// at (±3, ±3), 100 points each). Sliders for K_max and α₀.
//
// At α₀ < 1 the Dirichlet prior is sparse over components, and the variational
// posterior collapses unused components to weights ≈ α₀ / N — Theorem 3.
//
// The CAVI updates (Bishop §10.2) on a 2D Bayesian GMM with Normal-Wishart
// prior on (μ_k, Λ_k) and Dirichlet on π. Implemented at low precision (no
// Wishart on Λ — we use isotropic Gaussian likelihood per component for
// browser speed). The qualitative ARD behavior is preserved.
// =============================================================================

const PANEL_HEIGHT = 360;
const SCATTER_MARGIN = { top: 22, right: 18, bottom: 36, left: 36 };
const BAR_MARGIN = { top: 22, right: 18, bottom: 36, left: 36 };

// Dataset constants
const CLUSTER_CENTERS: [number, number][] = [
  [3.0, 3.0],
  [-3.0, 3.0],
  [3.0, -3.0],
  [-3.0, -3.0],
];
const CLUSTER_STD = [0.5, 0.6, 0.65, 0.7];
const POINTS_PER_CLUSTER = 100;

const CLUSTER_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'];

interface DataPoint {
  x: number;
  y: number;
  trueCluster: number;
}

function generateDataset(seed: number): DataPoint[] {
  const rng = mulberry32(seed);
  let buf: number | null = null;
  const draw = (): number => {
    if (buf !== null) {
      const v = buf;
      buf = null;
      return v;
    }
    const [a, b] = gaussianPair(rng);
    buf = b;
    return a;
  };
  const points: DataPoint[] = [];
  for (let c = 0; c < CLUSTER_CENTERS.length; c++) {
    const [cx, cy] = CLUSTER_CENTERS[c];
    const sd = CLUSTER_STD[c];
    for (let i = 0; i < POINTS_PER_CLUSTER; i++) {
      points.push({ x: cx + sd * draw(), y: cy + sd * draw(), trueCluster: c });
    }
  }
  return points;
}

const DATASET = generateDataset(42424242);

// CAVI for Bayesian GMM with K_max components, isotropic-Gaussian likelihood
// (variance σ² treated as fixed per CAVI step but updated). The full
// Normal-Wishart hyperprior collapses to Normal-InverseGamma in 2D isotropic;
// we run a simplified version that captures the ARD weight collapse cleanly.
interface CAVIResult {
  weights: number[];
  means: [number, number][];
  variances: number[];
  responsibilities: number[][]; // n × K_max
  alphaPosterior: number[];
  elbo: number;
  iterations: number;
}

function runCAVI(
  data: DataPoint[],
  Kmax: number,
  alpha0: number,
  seed: number,
  maxIter: number = 80,
): CAVIResult {
  const n = data.length;
  // Initialize means via random data points
  const rng = mulberry32(seed);
  const means: [number, number][] = [];
  const used = new Set<number>();
  for (let k = 0; k < Kmax; k++) {
    let idx = Math.floor(rng() * n);
    let tries = 0;
    while (used.has(idx) && tries < 20) {
      idx = Math.floor(rng() * n);
      tries += 1;
    }
    used.add(idx);
    means.push([data[idx].x + 0.05 * (rng() - 0.5), data[idx].y + 0.05 * (rng() - 0.5)]);
  }
  const variances = new Array<number>(Kmax).fill(1.0);
  const weights = new Array<number>(Kmax).fill(1 / Kmax);

  // Hyperprior: Normal-InverseGamma (each component) with weak prior
  const m0: [number, number] = [0, 0];
  const beta0 = 0.1;
  const a0 = 2.0;
  const b0 = 1.0;
  let alpha = new Array<number>(Kmax).fill(alpha0);
  let r = new Array<Array<number>>(n).fill([]).map(() => new Array<number>(Kmax).fill(1 / Kmax));
  let prevELBO = -Infinity;
  let iterUsed = 0;

  for (let iter = 0; iter < maxIter; iter++) {
    iterUsed = iter + 1;
    // E-step: responsibilities ∝ exp(E[log π_k] + E[log p(x | params_k)])
    // E[log π_k] = digamma(α_k) − digamma(Σ α_j)
    const sumAlpha = alpha.reduce((s, v) => s + v, 0);
    const elogPi = alpha.map((a) => digamma(a) - digamma(sumAlpha));
    for (let i = 0; i < n; i++) {
      const logr = new Array<number>(Kmax).fill(0);
      for (let k = 0; k < Kmax; k++) {
        const dx = data[i].x - means[k][0];
        const dy = data[i].y - means[k][1];
        const d2 = dx * dx + dy * dy;
        // Isotropic Gaussian log-density up to a constant
        logr[k] = elogPi[k] - Math.log(2 * Math.PI * variances[k]) - 0.5 * d2 / variances[k];
      }
      // Softmax normalize
      let maxLr = -Infinity;
      for (let k = 0; k < Kmax; k++) if (logr[k] > maxLr) maxLr = logr[k];
      let denom = 0;
      const rk = new Array<number>(Kmax);
      for (let k = 0; k < Kmax; k++) {
        rk[k] = Math.exp(logr[k] - maxLr);
        denom += rk[k];
      }
      for (let k = 0; k < Kmax; k++) r[i][k] = rk[k] / denom;
    }
    // M-step
    const Nk = new Array<number>(Kmax).fill(0);
    for (let i = 0; i < n; i++) for (let k = 0; k < Kmax; k++) Nk[k] += r[i][k];
    const newAlpha = alpha.map((_, k) => alpha0 + Nk[k]);
    const newMeans: [number, number][] = [];
    const newVariances: number[] = [];
    for (let k = 0; k < Kmax; k++) {
      if (Nk[k] < 1e-6) {
        newMeans.push(means[k]);
        newVariances.push(variances[k]);
        continue;
      }
      let sumX = 0;
      let sumY = 0;
      for (let i = 0; i < n; i++) {
        sumX += r[i][k] * data[i].x;
        sumY += r[i][k] * data[i].y;
      }
      const xbar = sumX / Nk[k];
      const ybar = sumY / Nk[k];
      // Posterior mean: shrinkage to m0 weighted by β0
      const muX = (beta0 * m0[0] + Nk[k] * xbar) / (beta0 + Nk[k]);
      const muY = (beta0 * m0[1] + Nk[k] * ybar) / (beta0 + Nk[k]);
      newMeans.push([muX, muY]);
      // Variance update (Normal-InverseGamma posterior mean)
      let sse = 0;
      for (let i = 0; i < n; i++) {
        const dx = data[i].x - muX;
        const dy = data[i].y - muY;
        sse += r[i][k] * (dx * dx + dy * dy);
      }
      const aN = a0 + Nk[k];
      const bN = b0 + 0.5 * sse + 0.5 * (beta0 * Nk[k] / (beta0 + Nk[k])) * ((xbar - m0[0]) ** 2 + (ybar - m0[1]) ** 2);
      newVariances.push(bN / Math.max(aN - 1, 1e-3));
    }
    // Compute approximate ELBO for convergence check
    let elbo = 0;
    for (let i = 0; i < n; i++) {
      let acc = 0;
      for (let k = 0; k < Kmax; k++) {
        const dx = data[i].x - newMeans[k][0];
        const dy = data[i].y - newMeans[k][1];
        const d2 = dx * dx + dy * dy;
        const logp = -Math.log(2 * Math.PI * newVariances[k]) - 0.5 * d2 / newVariances[k];
        if (r[i][k] > 1e-12) {
          acc += r[i][k] * (elogPi[k] + logp - Math.log(r[i][k]));
        }
      }
      elbo += acc;
    }
    alpha = newAlpha;
    for (let k = 0; k < Kmax; k++) {
      means[k] = newMeans[k];
      variances[k] = newVariances[k];
    }
    if (Math.abs(elbo - prevELBO) < 0.001) break;
    prevELBO = elbo;
  }

  const sumAlphaFinal = alpha.reduce((s, v) => s + v, 0);
  const finalWeights = alpha.map((a) => a / sumAlphaFinal);
  for (let k = 0; k < Kmax; k++) weights[k] = finalWeights[k];

  return {
    weights: finalWeights,
    means: means as [number, number][],
    variances,
    responsibilities: r,
    alphaPosterior: alpha,
    elbo: prevELBO,
    iterations: iterUsed,
  };
}

// Stirling-approximation digamma
function digamma(x: number): number {
  let result = 0;
  let z = x;
  while (z < 6) {
    result -= 1 / z;
    z += 1;
  }
  result += Math.log(z) - 0.5 / z;
  const z2 = z * z;
  result -= 1 / (12 * z2);
  result += 1 / (120 * z2 * z2);
  return result;
}

export default function BayesianGMMOccam() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [Kmax, setKmax] = useState(10);
  const [alpha0, setAlpha0] = useState(0.1);

  const result = useMemo(() => runCAVI(DATASET, Kmax, alpha0, 1234567), [Kmax, alpha0]);

  const isMobile = containerWidth > 0 && containerWidth < 640;
  const panelW = isMobile ? containerWidth : Math.max(0, Math.floor(containerWidth / 2) - 8);

  const scatterRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelW <= 0) return;
      const W = panelW;
      const H = PANEL_HEIGHT;
      const w = W - SCATTER_MARGIN.left - SCATTER_MARGIN.right;
      const h = H - SCATTER_MARGIN.top - SCATTER_MARGIN.bottom;
      const g = svg.append('g').attr('transform', `translate(${SCATTER_MARGIN.left},${SCATTER_MARGIN.top})`);
      const xScale = d3.scaleLinear().domain([-5, 5]).range([0, w]);
      const yScale = d3.scaleLinear().domain([-5, 5]).range([h, 0]);
      g.append('g')
        .attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(xScale).ticks(5))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '10px');
      g.append('g')
        .call(d3.axisLeft(yScale).ticks(5))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '10px');
      // Points colored by argmax responsibility
      DATASET.forEach((p, i) => {
        let argMax = 0;
        let mr = -1;
        for (let k = 0; k < Kmax; k++) {
          if (result.responsibilities[i][k] > mr) {
            mr = result.responsibilities[i][k];
            argMax = k;
          }
        }
        g.append('circle')
          .attr('cx', xScale(p.x))
          .attr('cy', yScale(p.y))
          .attr('r', 2.4)
          .attr('fill', CLUSTER_COLORS[argMax % CLUSTER_COLORS.length])
          .attr('fill-opacity', 0.7);
      });
      // Component centers + 2σ ellipses for active components
      const activeThresh = 0.01;
      for (let k = 0; k < Kmax; k++) {
        const isActive = result.weights[k] >= activeThresh;
        const [cx, cy] = result.means[k];
        const sd = Math.sqrt(result.variances[k]);
        const color = CLUSTER_COLORS[k % CLUSTER_COLORS.length];
        if (isActive) {
          // 2-sigma circle
          g.append('circle')
            .attr('cx', xScale(cx))
            .attr('cy', yScale(cy))
            .attr('r', Math.abs(xScale(2 * sd) - xScale(0)))
            .attr('fill', 'none')
            .attr('stroke', color)
            .attr('stroke-width', 1.6)
            .attr('stroke-dasharray', '4,3');
          g.append('circle')
            .attr('cx', xScale(cx))
            .attr('cy', yScale(cy))
            .attr('r', 5)
            .attr('fill', color);
        } else {
          // Inactive cross
          g.append('text')
            .attr('x', xScale(cx))
            .attr('y', yScale(cy))
            .attr('text-anchor', 'middle')
            .attr('dy', '0.35em')
            .style('fill', 'var(--color-text-secondary)')
            .style('font-size', '10px')
            .text('×');
        }
      }
    },
    [result, panelW, Kmax],
  );

  const barRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelW <= 0) return;
      const W = panelW;
      const H = PANEL_HEIGHT;
      const w = W - BAR_MARGIN.left - BAR_MARGIN.right;
      const h = H - BAR_MARGIN.top - BAR_MARGIN.bottom;
      const g = svg.append('g').attr('transform', `translate(${BAR_MARGIN.left},${BAR_MARGIN.top})`);
      const xScale = d3
        .scaleBand<number>()
        .domain(d3.range(Kmax))
        .range([0, w])
        .padding(0.15);
      const yScale = d3.scaleLinear().domain([0.0001, 1]).range([h, 0]).clamp(true);
      const yScaleLog = d3.scaleLog().domain([0.0001, 1]).range([h, 0]).clamp(true);
      g.append('g')
        .attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(xScale).tickFormat((v) => `${v}`))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '10px');
      g.append('g')
        .call(
          d3
            .axisLeft(yScaleLog)
            .tickValues([0.0001, 0.001, 0.01, 0.1, 1.0])
            .tickFormat(d3.format('~g')),
        )
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '10px');
      g.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', -h / 2)
        .attr('y', -28)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '11px')
        .text('Posterior weight (log scale)');
      g.append('text')
        .attr('x', w / 2)
        .attr('y', h + 28)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '11px')
        .text('Component k');
      // Bars
      for (let k = 0; k < Kmax; k++) {
        const ww = xScale.bandwidth();
        const x = xScale(k) ?? 0;
        const v = Math.max(result.weights[k], 0.0001);
        const yTop = yScaleLog(v);
        g.append('rect')
          .attr('x', x)
          .attr('y', yTop)
          .attr('width', ww)
          .attr('height', h - yTop)
          .attr('fill', CLUSTER_COLORS[k % CLUSTER_COLORS.length])
          .attr('fill-opacity', result.weights[k] >= 0.01 ? 0.85 : 0.3);
      }
      // Reference lines: 1/K_max and active threshold
      const uniformY = yScaleLog(1 / Kmax);
      g.append('line').attr('x1', 0).attr('x2', w).attr('y1', uniformY).attr('y2', uniformY).style('stroke', '#666').style('stroke-dasharray', '4,3').style('stroke-width', 1);
      g.append('text')
        .attr('x', w - 4)
        .attr('y', uniformY - 4)
        .attr('text-anchor', 'end')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '10px')
        .text(`uniform 1/K_max = ${(1 / Kmax).toFixed(2)}`);
      const threshY = yScaleLog(0.01);
      g.append('line').attr('x1', 0).attr('x2', w).attr('y1', threshY).attr('y2', threshY).style('stroke', '#c0504d').style('stroke-dasharray', '4,3').style('stroke-width', 1);
      g.append('text')
        .attr('x', 4)
        .attr('y', threshY - 4)
        .style('fill', '#c0504d')
        .style('font-size', '10px')
        .text('active threshold 0.01');
      void yScale;
    },
    [result, panelW, Kmax],
  );

  const activeCount = result.weights.filter((w) => w >= 0.01).length;

  return (
    <div ref={containerRef} className="my-6">
      <div
        style={{
          background: 'var(--color-surface)',
          border: '1px solid var(--color-border)',
          borderRadius: 8,
          padding: 16,
          fontFamily: 'var(--font-sans)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12, marginBottom: 8 }}>
          <div>
            <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--color-text)' }}>
              Bayesian GMM ARD: {activeCount} active / {Kmax} components — α₀ = {alpha0.toFixed(2)}
            </div>
            <div style={{ fontSize: 12, color: 'var(--color-text-secondary)' }}>
              4 well-separated clusters; CAVI converges in {result.iterations} iterations.
            </div>
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, fontSize: 12, color: 'var(--color-text-secondary)' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              K_max
              <input type="range" min={3} max={12} step={1} value={Kmax} onChange={(e) => setKmax(parseInt(e.target.value, 10))} style={{ width: 100 }} />
              <span style={{ fontFamily: 'var(--font-mono)', minWidth: 16, color: 'var(--color-text)' }}>{Kmax}</span>
            </label>
            <label style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              α₀
              <input type="range" min={0.01} max={2.0} step={0.01} value={alpha0} onChange={(e) => setAlpha0(parseFloat(e.target.value))} style={{ width: 100 }} />
              <span style={{ fontFamily: 'var(--font-mono)', minWidth: 36, color: 'var(--color-text)' }}>{alpha0.toFixed(2)}</span>
            </label>
          </div>
        </div>
        <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: 16 }}>
          <svg ref={scatterRef} width={panelW} height={PANEL_HEIGHT} role="img" aria-label="2D scatter plot of cluster data with active component centers and 2σ ellipses." />
          <svg ref={barRef} width={panelW} height={PANEL_HEIGHT} role="img" aria-label="Bar chart of posterior mixture weights on log scale showing active vs collapsed components." />
        </div>
      </div>
    </div>
  );
}
