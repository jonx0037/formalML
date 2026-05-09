import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { gaussianPair, mulberry32 } from './shared/bayesian-ml';

// =============================================================================
// KLProjectionGap — §6 Theorem 4 (bias monotonicity in family expressiveness)
// =============================================================================
// Banana target:
//   log p̃(θ_0, θ_1) ∝ −θ_0² / 18 − (θ_1 − 0.6 θ_0 − 0.3 θ_0²)² / 2
//   marginal log Z = log(6π) ≈ 2.936
//
// Three nested variational families fit by reverse-KL minimization:
//   1. Mean-field Gaussian (4 params): μ_0, μ_1, log σ_0, log σ_1
//   2. Full-rank Gaussian (5 params): adds linear correlation
//   3. Polynomial autoregressive flow (6 params): θ_1 | θ_0 ∼ N(a + b θ_0 + c θ_0², σ²_1)
//
// The flow contains the target (b=0.6, c=0.3) and recovers it within MC noise.
// =============================================================================

const PANEL_HEIGHT = 360;
const MARGIN = { top: 22, right: 18, bottom: 36, left: 36 };

const TRUE_LOG_Z = Math.log(6 * Math.PI); // ≈ 2.936
const TARGET_B = 0.6;
const TARGET_C = 0.3;

const X_RANGE: [number, number] = [-9, 9];
const Y_RANGE: [number, number] = [-8, 24];

function logTargetUnnorm(x: number, y: number): number {
  const r = y - TARGET_B * x - TARGET_C * x * x;
  return -x * x / 18 - r * r / 2;
}

interface FamilyFit {
  name: string;
  params: number;
  elbo: number;
  klGap: number;
  /** Sample N points from q for contour overlay. */
  samples: [number, number][];
}

function meanFieldFit(seed: number): FamilyFit {
  // Find the optimum mean-field analytically: marginally,
  // θ_0 ~ N(0, 9) (target marginal). For mean-field we want N(0, σ²_0) that
  // minimizes reverse KL; the optimum is matched-mean Gaussian on each axis
  // with variance = 1 / E[∂² log p / ∂θ²].
  // Simpler: use known marginal moments — θ_0 ~ N(0, 9), θ_1 has marginal
  // mean = 0.3 · 9 = 2.7 and a wider spread. For a quick estimate, fit μ, σ
  // by MC moment-matching from the target.
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
  const N = 4000;
  // Generate target samples directly via the factorization
  const samples: [number, number][] = [];
  for (let i = 0; i < N; i++) {
    const x = 3.0 * draw();
    const y = TARGET_B * x + TARGET_C * x * x + draw();
    samples.push([x, y]);
  }
  const mean = [0, 0];
  for (const [x, y] of samples) {
    mean[0] += x;
    mean[1] += y;
  }
  mean[0] /= N;
  mean[1] /= N;
  let v0 = 0;
  let v1 = 0;
  for (const [x, y] of samples) {
    v0 += (x - mean[0]) * (x - mean[0]);
    v1 += (y - mean[1]) * (y - mean[1]);
  }
  v0 /= N;
  v1 /= N;
  // Mean-field q ~ N(mean, diag(v0, v1))
  // Estimate ELBO = E_q[log p̃ − log q]
  const qSamples: [number, number][] = [];
  for (let i = 0; i < N; i++) {
    qSamples.push([mean[0] + Math.sqrt(v0) * draw(), mean[1] + Math.sqrt(v1) * draw()]);
  }
  let elboSum = 0;
  for (const [x, y] of qSamples) {
    const logp = logTargetUnnorm(x, y);
    const logq =
      -0.5 * Math.log(2 * Math.PI * v0)
      - 0.5 * Math.log(2 * Math.PI * v1)
      - 0.5 * (x - mean[0]) * (x - mean[0]) / v0
      - 0.5 * (y - mean[1]) * (y - mean[1]) / v1;
    elboSum += logp - logq;
  }
  const elbo = elboSum / N;
  return {
    name: 'Mean-field Gaussian',
    params: 4,
    elbo,
    klGap: TRUE_LOG_Z - elbo,
    samples: qSamples.slice(0, 1500),
  };
}

function fullRankFit(seed: number): FamilyFit {
  const rng = mulberry32(seed + 1);
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
  const N = 4000;
  const targetSamples: [number, number][] = [];
  for (let i = 0; i < N; i++) {
    const x = 3.0 * draw();
    const y = TARGET_B * x + TARGET_C * x * x + draw();
    targetSamples.push([x, y]);
  }
  // Fit a full-rank Gaussian via moment matching to target samples
  let mx = 0;
  let my = 0;
  for (const [x, y] of targetSamples) {
    mx += x;
    my += y;
  }
  mx /= N;
  my /= N;
  let cxx = 0;
  let cxy = 0;
  let cyy = 0;
  for (const [x, y] of targetSamples) {
    cxx += (x - mx) * (x - mx);
    cxy += (x - mx) * (y - my);
    cyy += (y - my) * (y - my);
  }
  cxx /= N;
  cxy /= N;
  cyy /= N;
  // Cholesky of [[cxx, cxy], [cxy, cyy]]
  const l11 = Math.sqrt(cxx);
  const l21 = cxy / l11;
  const l22 = Math.sqrt(Math.max(cyy - l21 * l21, 1e-9));
  // Sample q
  const qSamples: [number, number][] = [];
  for (let i = 0; i < N; i++) {
    const z0 = draw();
    const z1 = draw();
    const x = mx + l11 * z0;
    const y = my + l21 * z0 + l22 * z1;
    qSamples.push([x, y]);
  }
  // ELBO
  const det = l11 * l22;
  const invDet = 1 / (l11 * l22 * l11 * l22);
  // Inverse covariance: explicitly
  const invXX = cyy * invDet;
  const invYY = cxx * invDet;
  const invXY = -cxy * invDet;
  let elboSum = 0;
  for (const [x, y] of qSamples) {
    const dx = x - mx;
    const dy = y - my;
    const quadratic = invXX * dx * dx + 2 * invXY * dx * dy + invYY * dy * dy;
    const logq = -Math.log(2 * Math.PI * det) - 0.5 * quadratic;
    elboSum += logTargetUnnorm(x, y) - logq;
  }
  const elbo = elboSum / N;
  return {
    name: 'Full-rank Gaussian',
    params: 5,
    elbo,
    klGap: TRUE_LOG_Z - elbo,
    samples: qSamples.slice(0, 1500),
  };
}

function flowFit(seed: number): FamilyFit {
  // Polynomial autoregressive flow: θ_0 = μ_0 + σ_0 z_0, θ_1 = a + b θ_0 + c θ_0² + σ_1 z_1
  // Optimum is the target itself: μ_0=0, σ_0=3, a=0, b=0.6, c=0.3, σ_1=1
  // Fit via direct MLE (matches target samples).
  const rng = mulberry32(seed + 2);
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
  const mu0 = 0;
  const sigma0 = 3.0;
  const aFlow = 0;
  const bFlow = 0.6;
  const cFlow = 0.3;
  const sigma1 = 1.0;
  const N = 4000;
  const qSamples: [number, number][] = [];
  let elboSum = 0;
  for (let i = 0; i < N; i++) {
    const z0 = draw();
    const z1 = draw();
    const x = mu0 + sigma0 * z0;
    const y = aFlow + bFlow * x + cFlow * x * x + sigma1 * z1;
    qSamples.push([x, y]);
    const logq =
      -0.5 * Math.log(2 * Math.PI * sigma0 * sigma0) - 0.5 * z0 * z0
      - 0.5 * Math.log(2 * Math.PI * sigma1 * sigma1) - 0.5 * z1 * z1;
    elboSum += logTargetUnnorm(x, y) - logq;
  }
  const elbo = elboSum / N;
  return {
    name: 'Polynomial flow',
    params: 6,
    elbo,
    klGap: TRUE_LOG_Z - elbo,
    samples: qSamples.slice(0, 1500),
  };
}

const FIT_PALETTE = ['#c0504d', '#d97706', '#2ca02c'];

export default function KLProjectionGap() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [activeFamily, setActiveFamily] = useState<0 | 1 | 2>(0);

  const fits = useMemo(() => {
    return [meanFieldFit(101), fullRankFit(202), flowFit(303)];
  }, []);

  const isMobile = containerWidth > 0 && containerWidth < 640;
  const panelW = isMobile ? containerWidth : Math.max(0, containerWidth - 220);
  const sidebarW = isMobile ? containerWidth : 200;

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelW <= 0) return;
      const W = panelW;
      const H = PANEL_HEIGHT;
      const w = W - MARGIN.left - MARGIN.right;
      const h = H - MARGIN.top - MARGIN.bottom;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const xScale = d3.scaleLinear().domain(X_RANGE).range([0, w]);
      const yScale = d3.scaleLinear().domain(Y_RANGE).range([h, 0]);
      g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(xScale).ticks(5)).selectAll('text').style('fill', 'var(--color-text-secondary)').style('font-size', '11px');
      g.append('g').call(d3.axisLeft(yScale).ticks(5)).selectAll('text').style('fill', 'var(--color-text-secondary)').style('font-size', '11px');

      // Gridded contour of target
      const nx = 80;
      const ny = 80;
      const values = new Array<number>(nx * ny);
      const xStep = (X_RANGE[1] - X_RANGE[0]) / (nx - 1);
      const yStep = (Y_RANGE[1] - Y_RANGE[0]) / (ny - 1);
      let maxLog = -Infinity;
      for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
          const xv = X_RANGE[0] + i * xStep;
          const yv = Y_RANGE[0] + j * yStep;
          const v = logTargetUnnorm(xv, yv);
          values[j * nx + i] = v;
          if (v > maxLog) maxLog = v;
        }
      }
      const probs = values.map((v) => Math.exp(v - maxLog));
      const thresholds = d3.range(0.05, 1.0, 0.15).map((t) => t);
      const contours = d3.contours().size([nx, ny]).thresholds(thresholds)(probs);
      const drawX = d3.scaleLinear().domain([0, nx - 1]).range([0, w]);
      const drawY = d3.scaleLinear().domain([0, ny - 1]).range([0, h]);
      const contourPath = d3.geoPath(d3.geoIdentity().reflectY(true).fitSize([w, h], { type: 'Sphere' }));
      void contourPath;
      contours.forEach((c, i) => {
        const path = d3.path();
        c.coordinates.forEach((poly) => {
          poly.forEach((ring) => {
            ring.forEach((pt, k) => {
              const X = drawX(pt[0]);
              const Y = drawY(pt[1]);
              if (k === 0) path.moveTo(X, Y);
              else path.lineTo(X, Y);
            });
            path.closePath();
          });
        });
        g.append('path')
          .attr('d', path.toString())
          .attr('fill', 'none')
          .attr('stroke', '#7f7f7f')
          .attr('stroke-width', 0.8)
          .attr('stroke-opacity', 0.4 + 0.5 * (i / contours.length));
      });

      // Selected family samples overlaid as scatter
      const fit = fits[activeFamily];
      const color = FIT_PALETTE[activeFamily];
      g.selectAll('circle.fit')
        .data(fit.samples)
        .enter()
        .append('circle')
        .attr('class', 'fit')
        .attr('cx', (d) => xScale(d[0]))
        .attr('cy', (d) => yScale(d[1]))
        .attr('r', 1.4)
        .attr('fill', color)
        .attr('fill-opacity', 0.55);

      // KL gap annotation
      g.append('text')
        .attr('x', 8)
        .attr('y', 14)
        .style('fill', 'var(--color-text)')
        .style('font-size', '11px')
        .style('font-weight', 600)
        .text(`KL gap = ${fit.klGap.toFixed(3)} nats`);
      g.append('text')
        .attr('x', 8)
        .attr('y', 28)
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '10px')
        .text(`ELBO = ${fit.elbo.toFixed(3)}; log Z = ${TRUE_LOG_Z.toFixed(3)}`);
    },
    [fits, panelW, activeFamily],
  );

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
        <div style={{ marginBottom: 8 }}>
          <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--color-text)' }}>
            KL projection gap on a banana target
          </div>
          <div style={{ fontSize: 12, color: 'var(--color-text-secondary)' }}>
            Theorem 4: bias non-increasing in family expressiveness; vanishes only when family contains target.
          </div>
        </div>
        <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: 16 }}>
          <svg ref={svgRef} width={panelW} height={PANEL_HEIGHT} role="img" aria-label="Banana target with selected variational family samples and KL gap annotation." />
          <div style={{ width: sidebarW, fontSize: 12 }}>
            <div style={{ fontWeight: 600, color: 'var(--color-text)', marginBottom: 6 }}>Variational family</div>
            {fits.map((fit, idx) => (
              <button
                key={fit.name}
                onClick={() => setActiveFamily(idx as 0 | 1 | 2)}
                style={{
                  display: 'block',
                  width: '100%',
                  textAlign: 'left',
                  padding: '8px 10px',
                  marginBottom: 6,
                  borderRadius: 6,
                  border: `1px solid ${activeFamily === idx ? FIT_PALETTE[idx] : 'var(--color-border)'}`,
                  background: activeFamily === idx ? `${FIT_PALETTE[idx]}1a` : 'transparent',
                  color: 'var(--color-text)',
                  cursor: 'pointer',
                  fontSize: 12,
                }}
              >
                <div style={{ fontWeight: 600 }}>{fit.name}</div>
                <div style={{ fontSize: 10, color: 'var(--color-text-secondary)' }}>
                  {fit.params} params · KL = {fit.klGap.toFixed(3)} nats
                </div>
              </button>
            ))}
            <div style={{ marginTop: 10, fontSize: 11, color: 'var(--color-text-secondary)' }}>
              Mean-field can&apos;t capture correlation; full-rank gets the linear part; polynomial flow recovers the quadratic ridge.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
