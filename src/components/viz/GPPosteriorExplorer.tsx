import { useState, useMemo, useEffect, useRef, useCallback } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';

const MARGIN = { top: 30, right: 16, bottom: 40, left: 56 };

const COLORS = {
  mean: '#2563eb',
  band: 'rgba(37, 99, 235, 0.18)',
  observation: '#dc2626',
  priorSample: '#888888',
} as const;

const SAMPLE_COLORS = [
  d3.schemeTableau10[0],
  d3.schemeTableau10[1],
  d3.schemeTableau10[2],
];

// ─── Kernel functions ───

function rbfKernel(x1: number, x2: number, l: number): number {
  return Math.exp(-((x1 - x2) ** 2) / (2 * l * l));
}

function matern32Kernel(x1: number, x2: number, l: number): number {
  const r = Math.abs(x1 - x2);
  const s = (Math.sqrt(3) * r) / l;
  return (1 + s) * Math.exp(-s);
}

function linearKernel(x1: number, x2: number): number {
  return x1 * x2;
}

// ─── Linear algebra utilities ───

/** Cholesky decomposition — returns lower triangular L such that A = L L^T */
function cholesky(A: number[][]): number[][] {
  const n = A.length;
  const L: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = 0;
      for (let k = 0; k < j; k++) sum += L[i][k] * L[j][k];
      if (i === j) {
        const diag = A[i][i] - sum;
        L[i][j] = Math.sqrt(Math.max(diag, 1e-10));
      } else {
        L[i][j] = (A[i][j] - sum) / L[j][j];
      }
    }
  }
  return L;
}

/** Solve L x = b where L is lower triangular */
function forwardSolve(L: number[][], b: number[]): number[] {
  const n = b.length;
  const x = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let j = 0; j < i; j++) sum += L[i][j] * x[j];
    x[i] = (b[i] - sum) / L[i][i];
  }
  return x;
}

/** Solve L^T x = b where L is lower triangular */
function backwardSolve(L: number[][], b: number[]): number[] {
  const n = b.length;
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let sum = 0;
    for (let j = i + 1; j < n; j++) sum += L[j][i] * x[j];
    x[i] = (b[i] - sum) / L[i][i];
  }
  return x;
}

// ─── Seeded PRNG ───

/** Simple LCG for deterministic random numbers */
function lcg(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s * 1664525 + 1013904223) | 0;
    return ((s >>> 0) / 0xffffffff);
  };
}

/** Box-Muller normal using a seeded uniform source */
function seededNormals(n: number, rng: () => number): number[] {
  const out: number[] = [];
  for (let i = 0; i < n; i += 2) {
    const u1 = Math.max(rng(), 1e-10);
    const u2 = rng();
    const r = Math.sqrt(-2 * Math.log(u1));
    out.push(r * Math.cos(2 * Math.PI * u2));
    if (i + 1 < n) out.push(r * Math.sin(2 * Math.PI * u2));
  }
  return out.slice(0, n);
}

// ─── Kernel dispatch ───

type KernelType = 'rbf' | 'matern32' | 'linear';

function evalKernel(type: KernelType, x1: number, x2: number, l: number): number {
  switch (type) {
    case 'rbf': return rbfKernel(x1, x2, l);
    case 'matern32': return matern32Kernel(x1, x2, l);
    case 'linear': return linearKernel(x1, x2);
  }
}

// ─── Build kernel matrix ───

function buildKernelMatrix(
  xs: number[],
  ys: number[],
  type: KernelType,
  l: number,
): number[][] {
  const n = xs.length;
  const m = ys.length;
  const K: number[][] = Array.from({ length: n }, () => new Array(m).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      K[i][j] = evalKernel(type, xs[i], ys[j], l);
    }
  }
  return K;
}

// ─── GP computation result types ───

interface GPPriorResult {
  priorSamples: number[][];
  posteriorMean: null;
  posteriorStd: null;
  posteriorSamples: null;
}

interface GPPosteriorResult {
  priorSamples: null;
  posteriorMean: number[];
  posteriorStd: number[];
  posteriorSamples: number[][];
}

type GPResult = GPPriorResult | GPPosteriorResult;

// ─── Component ───

export default function GPPosteriorExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();

  const [kernel, setKernel] = useState<KernelType>('rbf');
  const [lengthScale, setLengthScale] = useState(1.0);
  const [noiseVariance, setNoiseVariance] = useState(0.1);
  const [observations, setObservations] = useState<{ x: number; y: number }[]>([]);

  const svgRef = useRef<SVGSVGElement>(null);

  const canvasWidth = Math.max(containerWidth - 32, 300);
  const canvasHeight = Math.min(Math.max(canvasWidth * 0.5, 250), 420);
  const plotW = canvasWidth;
  const plotH = canvasHeight;

  // Test grid
  const testGrid = useMemo(() => {
    const n = 150;
    const out: number[] = [];
    for (let i = 0; i < n; i++) {
      out.push(-5 + (10 * i) / (n - 1));
    }
    return out;
  }, []);

  // Reduced grid for posterior sample generation
  const sampleGrid = useMemo(() => {
    const n = 50;
    const out: number[] = [];
    for (let i = 0; i < n; i++) {
      out.push(-5 + (10 * i) / (n - 1));
    }
    return out;
  }, []);

  // ─── GP computation ───
  const gpResult = useMemo<GPResult>(() => {
    if (observations.length === 0) {
      // Prior samples
      const nSamples = 5;
      const K = buildKernelMatrix(testGrid, testGrid, kernel, lengthScale);
      // Add jitter
      for (let i = 0; i < testGrid.length; i++) K[i][i] += 1e-8;
      let L: number[][];
      try {
        L = cholesky(K);
      } catch {
        // Fallback: return flat zero samples
        return {
          priorSamples: Array.from({ length: nSamples }, () => new Array(testGrid.length).fill(0)),
          posteriorMean: null,
          posteriorStd: null,
          posteriorSamples: null,
        };
      }

      const samples: number[][] = [];
      for (let s = 0; s < nSamples; s++) {
        const rng = lcg(42 + s * 1337);
        const z = seededNormals(testGrid.length, rng);
        // sample = L * z
        const sample = new Array(testGrid.length).fill(0);
        for (let i = 0; i < testGrid.length; i++) {
          for (let j = 0; j <= i; j++) {
            sample[i] += L[i][j] * z[j];
          }
        }
        samples.push(sample);
      }
      return {
        priorSamples: samples,
        posteriorMean: null,
        posteriorStd: null,
        posteriorSamples: null,
      };
    }

    // Posterior computation
    const n = observations.length;
    const xTrain = observations.map(o => o.x);
    const yTrain = observations.map(o => o.y);

    // K_train: n×n
    const Ktrain = buildKernelMatrix(xTrain, xTrain, kernel, lengthScale);
    for (let i = 0; i < n; i++) {
      Ktrain[i][i] += noiseVariance + 1e-8;
    }

    let L: number[][];
    try {
      L = cholesky(Ktrain);
    } catch {
      return {
        priorSamples: null,
        posteriorMean: new Array(testGrid.length).fill(0),
        posteriorStd: new Array(testGrid.length).fill(1),
        posteriorSamples: [],
      };
    }

    // alpha = L^T \ (L \ y)
    const alpha = backwardSolve(L, forwardSolve(L, yTrain));

    // Posterior mean and variance at each test point
    const posteriorMean = new Array(testGrid.length).fill(0);
    const posteriorStd = new Array(testGrid.length).fill(0);

    for (let i = 0; i < testGrid.length; i++) {
      const kStar: number[] = new Array(n);
      for (let j = 0; j < n; j++) {
        kStar[j] = evalKernel(kernel, testGrid[i], xTrain[j], lengthScale);
      }
      // Mean
      let mean = 0;
      for (let j = 0; j < n; j++) mean += kStar[j] * alpha[j];
      posteriorMean[i] = mean;

      // Variance
      const v = forwardSolve(L, kStar);
      let vDot = 0;
      for (let j = 0; j < n; j++) vDot += v[j] * v[j];
      const kxx = evalKernel(kernel, testGrid[i], testGrid[i], lengthScale);
      posteriorStd[i] = Math.sqrt(Math.max(kxx - vDot, 0));
    }

    // Posterior samples on reduced grid, then interpolate
    const nSamp = 3;
    const sg = sampleGrid;
    const sgLen = sg.length;

    // Posterior mean on sample grid
    const sgMean = new Array(sgLen).fill(0);
    for (let i = 0; i < sgLen; i++) {
      const kStar: number[] = new Array(n);
      for (let j = 0; j < n; j++) {
        kStar[j] = evalKernel(kernel, sg[i], xTrain[j], lengthScale);
      }
      for (let j = 0; j < n; j++) sgMean[i] += kStar[j] * alpha[j];
    }

    // Build posterior covariance on sample grid: K** - K*n Ktrain^{-1} Kn*
    const Kss = buildKernelMatrix(sg, sg, kernel, lengthScale);
    const Kns = buildKernelMatrix(sg, xTrain, kernel, lengthScale); // sgLen × n

    // V = L \ K_ns^T  →  we solve L * V[j] = Kns[i] for each test point
    // Actually: for each row i of Kns, solve L * v_i = Kns[i]
    const V: number[][] = [];
    for (let i = 0; i < sgLen; i++) {
      V.push(forwardSolve(L, Kns[i]));
    }

    // Posterior covariance = Kss - V * V^T
    const postCov: number[][] = Array.from({ length: sgLen }, () => new Array(sgLen).fill(0));
    for (let i = 0; i < sgLen; i++) {
      for (let j = 0; j <= i; j++) {
        let vvt = 0;
        for (let k = 0; k < n; k++) vvt += V[i][k] * V[j][k];
        postCov[i][j] = Kss[i][j] - vvt;
        postCov[j][i] = postCov[i][j];
      }
    }
    // Add jitter
    for (let i = 0; i < sgLen; i++) postCov[i][i] += 1e-8;

    let Lpost: number[][];
    try {
      Lpost = cholesky(postCov);
    } catch {
      return {
        priorSamples: null,
        posteriorMean,
        posteriorStd,
        posteriorSamples: [],
      };
    }

    const posteriorSamples: number[][] = [];
    for (let s = 0; s < nSamp; s++) {
      const rng = lcg(123 + s * 997);
      const z = seededNormals(sgLen, rng);
      const sampleReduced = new Array(sgLen).fill(0);
      for (let i = 0; i < sgLen; i++) {
        for (let j = 0; j <= i; j++) {
          sampleReduced[i] += Lpost[i][j] * z[j];
        }
        sampleReduced[i] += sgMean[i];
      }

      // Interpolate from sampleGrid to testGrid using linear interpolation
      const interp = d3.scaleLinear()
        .domain(sg)
        .range(sampleReduced)
        .clamp(true);

      const fullSample = testGrid.map(x => interp(x) as number);
      posteriorSamples.push(fullSample);
    }

    return {
      priorSamples: null,
      posteriorMean,
      posteriorStd,
      posteriorSamples,
    };
  }, [observations, kernel, lengthScale, noiseVariance, testGrid, sampleGrid]);

  // ─── D3 rendering ───

  useEffect(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    if (plotW <= 0 || plotH <= 0) return;

    const innerW = plotW - MARGIN.left - MARGIN.right;
    const innerH = plotH - MARGIN.top - MARGIN.bottom;
    if (innerW <= 0 || innerH <= 0) return;

    const xScale = d3.scaleLinear().domain([-5, 5]).range([MARGIN.left, plotW - MARGIN.right]);

    // Determine y domain
    let yMin = -3;
    let yMax = 3;

    if (gpResult.priorSamples) {
      const allVals = gpResult.priorSamples.flat();
      yMin = Math.min(yMin, d3.min(allVals)! - 0.5);
      yMax = Math.max(yMax, d3.max(allVals)! + 0.5);
    }
    if (gpResult.posteriorMean) {
      const means = gpResult.posteriorMean;
      const stds = gpResult.posteriorStd!;
      const lo = means.map((m, i) => m - 2.5 * stds[i]);
      const hi = means.map((m, i) => m + 2.5 * stds[i]);
      yMin = Math.min(yMin, d3.min(lo)! - 0.3);
      yMax = Math.max(yMax, d3.max(hi)! + 0.3);

      if (gpResult.posteriorSamples) {
        const sampleVals = gpResult.posteriorSamples.flat();
        if (sampleVals.length > 0) {
          yMin = Math.min(yMin, d3.min(sampleVals)! - 0.3);
          yMax = Math.max(yMax, d3.max(sampleVals)! + 0.3);
        }
      }
    }
    if (observations.length > 0) {
      const obsY = observations.map(o => o.y);
      yMin = Math.min(yMin, d3.min(obsY)! - 0.5);
      yMax = Math.max(yMax, d3.max(obsY)! + 0.5);
    }

    const yScale = d3.scaleLinear().domain([yMin, yMax]).range([plotH - MARGIN.bottom, MARGIN.top]);

    // Axes
    svg.append('g')
      .attr('transform', `translate(0,${plotH - MARGIN.bottom})`)
      .call(d3.axisBottom(xScale).ticks(5).tickSize(3))
      .call(g => {
        g.selectAll('text')
          .style('font-size', '9px')
          .style('font-family', 'var(--font-mono)')
          .style('fill', 'var(--color-text-secondary)');
        g.select('.domain').style('stroke', 'var(--color-border)');
        g.selectAll('.tick line').style('stroke', 'var(--color-border)');
      });

    svg.append('g')
      .attr('transform', `translate(${MARGIN.left},0)`)
      .call(d3.axisLeft(yScale).ticks(5).tickSize(3))
      .call(g => {
        g.selectAll('text')
          .style('font-size', '9px')
          .style('font-family', 'var(--font-mono)')
          .style('fill', 'var(--color-text-secondary)');
        g.select('.domain').style('stroke', 'var(--color-border)');
        g.selectAll('.tick line').style('stroke', 'var(--color-border)');
      });

    // Title
    svg.append('text')
      .attr('x', MARGIN.left + 4)
      .attr('y', MARGIN.top - 10)
      .style('font-size', '11px')
      .style('font-family', 'var(--font-sans)')
      .style('font-weight', '600')
      .style('fill', 'var(--color-text)')
      .text('GP Posterior');

    const lineGen = d3.line<number>()
      .x((_, i) => xScale(testGrid[i]))
      .y(d => yScale(d))
      .defined(d => isFinite(d));

    if (gpResult.priorSamples) {
      // Prior mode: 5 thin gray lines
      for (const sample of gpResult.priorSamples) {
        svg.append('path')
          .datum(sample)
          .attr('d', lineGen)
          .style('fill', 'none')
          .style('stroke', COLORS.priorSample)
          .style('stroke-width', 1.2)
          .style('opacity', 0.4);
      }
    }

    if (gpResult.posteriorMean) {
      const mean = gpResult.posteriorMean;
      const std = gpResult.posteriorStd!;

      // Confidence band: mean ± 2σ
      const areaGen = d3.area<number>()
        .x((_, i) => xScale(testGrid[i]))
        .y0((_, i) => yScale(mean[i] - 2 * std[i]))
        .y1((_, i) => yScale(mean[i] + 2 * std[i]))
        .defined((_, i) => isFinite(mean[i]) && isFinite(std[i]));

      svg.append('path')
        .datum(testGrid)
        .attr('d', areaGen)
        .style('fill', COLORS.band)
        .style('stroke', 'none');

      // Mean line
      svg.append('path')
        .datum(mean)
        .attr('d', lineGen)
        .style('fill', 'none')
        .style('stroke', COLORS.mean)
        .style('stroke-width', 2);

      // Posterior samples
      if (gpResult.posteriorSamples) {
        for (let s = 0; s < gpResult.posteriorSamples.length; s++) {
          svg.append('path')
            .datum(gpResult.posteriorSamples[s])
            .attr('d', lineGen)
            .style('fill', 'none')
            .style('stroke', SAMPLE_COLORS[s % SAMPLE_COLORS.length])
            .style('stroke-width', 1.2)
            .style('opacity', 0.6);
        }
      }

      // Observations
      for (const obs of observations) {
        svg.append('circle')
          .attr('cx', xScale(obs.x))
          .attr('cy', yScale(obs.y))
          .attr('r', 5)
          .style('fill', COLORS.observation)
          .style('stroke', '#fff')
          .style('stroke-width', 1.5);
      }
    }

    // Legend
    const legendX = plotW - MARGIN.right - 120;
    let legendY = MARGIN.top + 4;

    if (gpResult.posteriorMean) {
      // Band legend
      svg.append('rect')
        .attr('x', legendX).attr('y', legendY - 5)
        .attr('width', 16).attr('height', 10)
        .style('fill', COLORS.band).style('stroke', COLORS.mean).style('stroke-width', 0.5);
      svg.append('text')
        .attr('x', legendX + 20).attr('y', legendY + 3)
        .style('font-size', '9px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)')
        .text('Mean ± 2σ');
      legendY += 16;

      // Samples legend
      svg.append('line')
        .attr('x1', legendX).attr('x2', legendX + 16).attr('y1', legendY).attr('y2', legendY)
        .style('stroke', SAMPLE_COLORS[0]).style('stroke-width', 1.2).style('opacity', 0.6);
      svg.append('text')
        .attr('x', legendX + 20).attr('y', legendY + 3)
        .style('font-size', '9px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)')
        .text('Samples');
      legendY += 16;

      // Observations legend
      svg.append('circle')
        .attr('cx', legendX + 8).attr('cy', legendY)
        .attr('r', 4)
        .style('fill', COLORS.observation).style('stroke', '#fff').style('stroke-width', 1);
      svg.append('text')
        .attr('x', legendX + 20).attr('y', legendY + 3)
        .style('font-size', '9px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)')
        .text('Observations');
    } else {
      // Prior legend
      svg.append('line')
        .attr('x1', legendX).attr('x2', legendX + 16).attr('y1', legendY).attr('y2', legendY)
        .style('stroke', COLORS.priorSample).style('stroke-width', 1.2).style('opacity', 0.4);
      svg.append('text')
        .attr('x', legendX + 20).attr('y', legendY + 3)
        .style('font-size', '9px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)')
        .text('Prior samples');
    }
  }, [gpResult, testGrid, observations, plotW, plotH]);

  // ─── Click handler ───

  const handleSvgClick = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    if (observations.length >= 10) return;

    const svg = svgRef.current;
    if (!svg) return;

    const rect = svg.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;

    // Check if click is within plot area
    if (px < MARGIN.left || px > plotW - MARGIN.right) return;
    if (py < MARGIN.top || py > plotH - MARGIN.bottom) return;

    const xScale = d3.scaleLinear().domain([-5, 5]).range([MARGIN.left, plotW - MARGIN.right]);

    // Recompute yScale to match rendering
    let yMin = -3;
    let yMax = 3;
    if (gpResult.priorSamples) {
      const allVals = gpResult.priorSamples.flat();
      yMin = Math.min(yMin, d3.min(allVals)! - 0.5);
      yMax = Math.max(yMax, d3.max(allVals)! + 0.5);
    }
    if (gpResult.posteriorMean) {
      const means = gpResult.posteriorMean;
      const stds = gpResult.posteriorStd!;
      const lo = means.map((m, i) => m - 2.5 * stds[i]);
      const hi = means.map((m, i) => m + 2.5 * stds[i]);
      yMin = Math.min(yMin, d3.min(lo)! - 0.3);
      yMax = Math.max(yMax, d3.max(hi)! + 0.3);
      if (gpResult.posteriorSamples) {
        const sampleVals = gpResult.posteriorSamples.flat();
        if (sampleVals.length > 0) {
          yMin = Math.min(yMin, d3.min(sampleVals)! - 0.3);
          yMax = Math.max(yMax, d3.max(sampleVals)! + 0.3);
        }
      }
    }
    if (observations.length > 0) {
      const obsY = observations.map(o => o.y);
      yMin = Math.min(yMin, d3.min(obsY)! - 0.5);
      yMax = Math.max(yMax, d3.max(obsY)! + 0.5);
    }
    const yScale = d3.scaleLinear().domain([yMin, yMax]).range([plotH - MARGIN.bottom, MARGIN.top]);

    const x = xScale.invert(px);
    const y = yScale.invert(py);

    setObservations(prev => [...prev, { x, y }]);
  }, [observations, gpResult, plotW, plotH]);

  // ─── Render ───

  return (
    <div
      ref={containerRef}
      style={{
        border: '1px solid var(--color-border)',
        borderRadius: '8px',
        padding: '16px',
        background: 'var(--color-surface)',
        marginTop: '1.5rem',
        marginBottom: '1.5rem',
      }}
    >
      <div style={{
        fontFamily: 'var(--font-sans)',
        fontSize: '13px',
        fontWeight: 600,
        marginBottom: '4px',
        color: 'var(--color-text)',
      }}>
        GP Posterior Explorer
      </div>

      <div style={{
        fontSize: '11px',
        fontFamily: 'var(--font-mono)',
        color: 'var(--color-text-secondary)',
        marginBottom: '10px',
      }}>
        Click to add observations (max 10)
      </div>

      <svg
        ref={svgRef}
        width={plotW}
        height={plotH}
        style={{
          border: '1px solid var(--color-border)',
          borderRadius: '6px',
          background: 'var(--color-muted-bg)',
          cursor: observations.length < 10 ? 'crosshair' : 'default',
          display: 'block',
        }}
        onClick={handleSvgClick}
      />

      {/* Controls */}
      <div style={{
        marginTop: '10px',
        display: 'flex',
        flexWrap: 'wrap',
        gap: '14px',
        alignItems: 'center',
      }}>
        {/* Kernel selector */}
        <fieldset style={{
          border: 'none',
          padding: 0,
          margin: 0,
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          fontSize: '12px',
          fontFamily: 'var(--font-mono)',
          color: 'var(--color-text-secondary)',
        }}>
          <legend style={{
            fontSize: '12px',
            fontFamily: 'var(--font-mono)',
            color: 'var(--color-text-secondary)',
            padding: 0,
            float: 'left',
            marginRight: '6px',
            lineHeight: '1.5',
          }}>
            Kernel:
          </legend>
          {([
            { value: 'rbf' as const, label: 'RBF' },
            { value: 'matern32' as const, label: 'Matern 3/2' },
            { value: 'linear' as const, label: 'Linear' },
          ]).map(({ value, label }) => (
            <label key={value} style={{
              display: 'flex',
              alignItems: 'center',
              gap: '3px',
              fontSize: '11px',
              cursor: 'pointer',
            }}>
              <input
                type="radio"
                name="gp-kernel"
                value={value}
                checked={kernel === value}
                onChange={() => setKernel(value)}
                style={{ accentColor: 'var(--color-accent)' }}
              />
              {label}
            </label>
          ))}
        </fieldset>

        {/* Length scale */}
        <label style={{
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          fontSize: '12px',
          fontFamily: 'var(--font-mono)',
          color: 'var(--color-text-secondary)',
          opacity: kernel === 'linear' ? 0.4 : 1,
        }}>
          ℓ = {lengthScale.toFixed(2)}
          <input
            type="range"
            min={0.1}
            max={5.0}
            step={0.1}
            value={lengthScale}
            onChange={e => setLengthScale(parseFloat(e.target.value))}
            disabled={kernel === 'linear'}
            style={{ width: '80px', accentColor: 'var(--color-accent)' }}
          />
        </label>

        {/* Noise variance */}
        <label style={{
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          fontSize: '12px',
          fontFamily: 'var(--font-mono)',
          color: 'var(--color-text-secondary)',
        }}>
          σ² = {noiseVariance.toFixed(2)}
          <input
            type="range"
            min={0.01}
            max={1.0}
            step={0.01}
            value={noiseVariance}
            onChange={e => setNoiseVariance(parseFloat(e.target.value))}
            style={{ width: '80px', accentColor: 'var(--color-accent)' }}
          />
        </label>

        {/* Clear button */}
        <button
          onClick={() => setObservations([])}
          disabled={observations.length === 0}
          style={{
            padding: '4px 12px',
            fontSize: '11px',
            fontFamily: 'var(--font-mono)',
            border: '1px solid var(--color-border)',
            borderRadius: '4px',
            background: 'var(--color-surface)',
            color: 'var(--color-text)',
            cursor: observations.length === 0 ? 'default' : 'pointer',
            opacity: observations.length === 0 ? 0.4 : 1,
          }}
        >
          Clear
        </button>

        {/* Observation count */}
        <span style={{
          fontSize: '11px',
          fontFamily: 'var(--font-mono)',
          color: 'var(--color-text-secondary)',
        }}>
          {observations.length} observation{observations.length !== 1 ? 's' : ''}
        </span>
      </div>
    </div>
  );
}
