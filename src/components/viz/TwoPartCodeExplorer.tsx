import { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { logStar } from './shared/informationTheory';

const HEIGHT = 340;
const SM_BREAKPOINT = 640;
const MARGIN = { top: 30, right: 20, bottom: 50, left: 55 };
const BITS_PER_COEFF = 8;

// Seeded LCG PRNG (same pattern as ConvergenceModesDemo)
function lcg(index: number, seed: number): number {
  let s = (seed * 2147483647 + index * 16807 + 12345) & 0x7fffffff;
  s = (s * 16807 + 12345) & 0x7fffffff;
  s = (s * 16807 + 12345) & 0x7fffffff;
  return (s & 0x7fffffff) / 0x7fffffff;
}

function seededNormal(i: number, seed: number): number {
  const u1 = Math.max(lcg(i * 2, seed), 1e-10);
  const u2 = lcg(i * 2 + 1, seed);
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// True polynomial: y = 1 - 0.5x + 2x^2 - 1.5x^3
const TRUE_COEFFS = [1, -0.5, 2, -1.5];

function evalPoly(coeffs: number[], x: number): number {
  let y = 0;
  for (let i = 0; i < coeffs.length; i++) {
    y += coeffs[i] * Math.pow(x, i);
  }
  return y;
}

function generateData(n: number, sigma: number, seed: number) {
  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i < n; i++) {
    const x = -1 + (2 * i) / (n - 1);
    const yTrue = evalPoly(TRUE_COEFFS, x);
    const noise = seededNormal(i, seed) * sigma;
    xs.push(x);
    ys.push(yTrue + noise);
  }
  return { xs, ys };
}

// Polynomial least-squares fit via normal equations (Vandermonde)
function polyFit(xs: number[], ys: number[], degree: number): number[] {
  const n = xs.length;
  const k = degree + 1;
  // Build X^T X and X^T y
  const XtX: number[][] = Array.from({ length: k }, () => new Array(k).fill(0));
  const Xty: number[] = new Array(k).fill(0);
  for (let i = 0; i < n; i++) {
    const powers: number[] = [1];
    for (let j = 1; j < k; j++) powers.push(powers[j - 1] * xs[i]);
    for (let r = 0; r < k; r++) {
      for (let c = 0; c < k; c++) {
        XtX[r][c] += powers[r] * powers[c];
      }
      Xty[r] += powers[r] * ys[i];
    }
  }
  // Solve via Gaussian elimination with partial pivoting
  const A = XtX.map((row, i) => [...row, Xty[i]]);
  for (let col = 0; col < k; col++) {
    let maxRow = col;
    for (let row = col + 1; row < k; row++) {
      if (Math.abs(A[row][col]) > Math.abs(A[maxRow][col])) maxRow = row;
    }
    [A[col], A[maxRow]] = [A[maxRow], A[col]];
    if (Math.abs(A[col][col]) < 1e-15) continue;
    for (let row = col + 1; row < k; row++) {
      const factor = A[row][col] / A[col][col];
      for (let j = col; j <= k; j++) A[row][j] -= factor * A[col][j];
    }
  }
  const coeffs = new Array(k).fill(0);
  for (let i = k - 1; i >= 0; i--) {
    let sum = A[i][k];
    for (let j = i + 1; j < k; j++) sum -= A[i][j] * coeffs[j];
    coeffs[i] = Math.abs(A[i][i]) > 1e-15 ? sum / A[i][i] : 0;
  }
  return coeffs;
}

function computeRSS(xs: number[], ys: number[], coeffs: number[]): number {
  let rss = 0;
  for (let i = 0; i < xs.length; i++) {
    const diff = ys[i] - evalPoly(coeffs, xs[i]);
    rss += diff * diff;
  }
  return rss;
}

const SAMPLE_SIZES = [20, 50, 100, 200];

export default function TwoPartCodeExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const svgRef = useRef<SVGSVGElement>(null);

  const [maxDegree, setMaxDegree] = useState(12);
  const [noiseLevel, setNoiseLevel] = useState(0.3);
  const [sampleSizeIdx, setSampleSizeIdx] = useState(1); // 50
  const [seed, setSeed] = useState(42);

  const n = SAMPLE_SIZES[sampleSizeIdx];
  const data = useMemo(() => generateData(n, noiseLevel, seed), [n, noiseLevel, seed]);

  const fits = useMemo(() => {
    const results: { degree: number; coeffs: number[]; rss: number; modelLen: number; dataLen: number; totalLen: number }[] = [];
    for (let d = 0; d <= maxDegree; d++) {
      const coeffs = polyFit(data.xs, data.ys, d);
      const rss = computeRSS(data.xs, data.ys, coeffs);
      const sigmaHat2 = Math.max(rss / n, 1e-15);
      const modelLen = logStar(d + 1) + (d + 1) * BITS_PER_COEFF;
      const dataLen = (n / 2) * Math.log2(2 * Math.PI * Math.E * sigmaHat2);
      const totalLen = modelLen + dataLen;
      results.push({ degree: d, coeffs, rss, modelLen, dataLen, totalLen });
    }
    return results;
  }, [data, maxDegree, n]);

  const optimalDegree = useMemo(() => {
    let bestIdx = 0;
    for (let i = 1; i < fits.length; i++) {
      if (fits[i].totalLen < fits[bestIdx].totalLen) bestIdx = i;
    }
    return fits[bestIdx].degree;
  }, [fits]);

  const isSmall = containerWidth < SM_BREAKPOINT;
  const totalWidth = Math.max(containerWidth, 300);
  const leftW = isSmall ? totalWidth : Math.floor(totalWidth * 0.5);
  const rightW = isSmall ? totalWidth : totalWidth - leftW;
  const svgHeight = isSmall ? HEIGHT * 2 + 40 : HEIGHT;

  const drawChart = useCallback(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    if (totalWidth <= 0) return;

    // --- Left panel: Data + polynomial fits ---
    const lW = leftW - MARGIN.left - MARGIN.right;
    const lH = HEIGHT - MARGIN.top - MARGIN.bottom;
    const leftG = svg.append('g')
      .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    const xExtent = d3.extent(data.xs) as [number, number];
    const yExtent = d3.extent(data.ys) as [number, number];
    const yPad = (yExtent[1] - yExtent[0]) * 0.15;
    const xScale = d3.scaleLinear().domain([xExtent[0] - 0.05, xExtent[1] + 0.05]).range([0, lW]);
    const yScale = d3.scaleLinear().domain([yExtent[0] - yPad, yExtent[1] + yPad]).range([lH, 0]);

    // Axes
    leftG.append('g')
      .attr('transform', `translate(0,${lH})`)
      .call(d3.axisBottom(xScale).ticks(5))
      .selectAll('text').style('fill', 'var(--color-text-secondary, #666)');
    leftG.append('g')
      .call(d3.axisLeft(yScale).ticks(5))
      .selectAll('text').style('fill', 'var(--color-text-secondary, #666)');
    leftG.selectAll('.domain, .tick line').style('stroke', 'var(--color-text-secondary, #999)');

    leftG.append('text')
      .attr('x', lW / 2).attr('y', lH + 40)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text-secondary, #666)')
      .style('font-size', '12px')
      .text('x');
    leftG.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -lH / 2).attr('y', -40)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text-secondary, #666)')
      .style('font-size', '12px')
      .text('y');

    // Data points
    leftG.selectAll('.data-pt')
      .data(data.xs.map((x, i) => ({ x, y: data.ys[i] })))
      .join('circle')
      .attr('cx', d => xScale(d.x))
      .attr('cy', d => yScale(d.y))
      .attr('r', 2.5)
      .style('fill', 'var(--color-text-secondary, #999)')
      .style('opacity', 0.6);

    // True polynomial (dashed)
    const xFine = d3.range(xExtent[0], xExtent[1], 0.01);
    const trueLine = d3.line<number>()
      .x(x => xScale(x))
      .y(x => yScale(evalPoly(TRUE_COEFFS, x)));
    leftG.append('path')
      .datum(xFine)
      .attr('d', trueLine)
      .style('fill', 'none')
      .style('stroke', '#10b981')
      .style('stroke-width', 1.5)
      .style('stroke-dasharray', '6,3')
      .style('opacity', 0.7);

    // MDL-optimal fit (solid purple)
    const optFit = fits[optimalDegree];
    const optLine = d3.line<number>()
      .x(x => xScale(x))
      .y(x => yScale(evalPoly(optFit.coeffs, x)));
    leftG.append('path')
      .datum(xFine)
      .attr('d', optLine)
      .style('fill', 'none')
      .style('stroke', '#7c3aed')
      .style('stroke-width', 2.5);

    // Legend
    const legY = 10;
    leftG.append('line').attr('x1', 8).attr('y1', legY).attr('x2', 28).attr('y2', legY)
      .style('stroke', '#10b981').style('stroke-width', 1.5).style('stroke-dasharray', '6,3');
    leftG.append('text').attr('x', 32).attr('y', legY + 4)
      .style('font-size', '10px').style('fill', 'var(--color-text-secondary, #666)')
      .text('True (degree 3)');
    leftG.append('line').attr('x1', 8).attr('y1', legY + 16).attr('x2', 28).attr('y2', legY + 16)
      .style('stroke', '#7c3aed').style('stroke-width', 2.5);
    leftG.append('text').attr('x', 32).attr('y', legY + 20)
      .style('font-size', '10px').style('fill', 'var(--color-text-secondary, #666)')
      .text(`MDL (degree ${optimalDegree})`);

    leftG.append('text')
      .attr('x', lW / 2).attr('y', -12)
      .attr('text-anchor', 'middle')
      .style('font-size', '12px')
      .style('fill', 'var(--color-text-secondary, #666)')
      .style('font-weight', '600')
      .text('Data & Polynomial Fits');

    // --- Right panel: Two-part code decomposition ---
    const rOffset = isSmall ? 0 : leftW;
    const rYOffset = isSmall ? HEIGHT + 20 : 0;
    const rW = (isSmall ? totalWidth : rightW) - MARGIN.left - MARGIN.right;
    const rH = HEIGHT - MARGIN.top - MARGIN.bottom;
    const rightG = svg.append('g')
      .attr('transform', `translate(${rOffset + MARGIN.left},${rYOffset + MARGIN.top})`);

    const maxTotal = Math.max(...fits.map(f => f.totalLen));
    const xBar = d3.scaleBand<number>()
      .domain(fits.map(f => f.degree))
      .range([0, rW])
      .padding(0.25);
    const yBar = d3.scaleLinear().domain([0, maxTotal * 1.1]).range([rH, 0]);

    // Axes
    rightG.append('g')
      .attr('transform', `translate(0,${rH})`)
      .call(d3.axisBottom(xBar).tickValues(fits.filter((_, i) => i % Math.max(1, Math.floor(fits.length / 10)) === 0).map(f => f.degree)))
      .selectAll('text').style('fill', 'var(--color-text-secondary, #666)');
    rightG.append('g')
      .call(d3.axisLeft(yBar).ticks(5))
      .selectAll('text').style('fill', 'var(--color-text-secondary, #666)');
    rightG.selectAll('.domain, .tick line').style('stroke', 'var(--color-text-secondary, #999)');

    rightG.append('text')
      .attr('x', rW / 2).attr('y', rH + 40)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text-secondary, #666)')
      .style('font-size', '12px')
      .text('Model degree d');
    rightG.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -rH / 2).attr('y', -40)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text-secondary, #666)')
      .style('font-size', '12px')
      .text('Code length (bits)');

    // Stacked bars: model (blue) + data (red)
    fits.forEach(f => {
      const bw = xBar.bandwidth();
      const bx = xBar(f.degree)!;
      // Data code length (bottom)
      const dataH = Math.max(0, rH - yBar(Math.max(0, f.dataLen)));
      rightG.append('rect')
        .attr('x', bx).attr('y', rH - dataH)
        .attr('width', bw).attr('height', dataH)
        .style('fill', '#ef4444')
        .style('opacity', 0.7);
      // Model code length (stacked on top)
      const modelH = Math.max(0, rH - yBar(f.modelLen));
      rightG.append('rect')
        .attr('x', bx).attr('y', rH - dataH - modelH)
        .attr('width', bw).attr('height', modelH)
        .style('fill', '#3b82f6')
        .style('opacity', 0.7);
    });

    // Total line (purple)
    const totalLine = d3.line<typeof fits[0]>()
      .x(d => (xBar(d.degree)! + xBar.bandwidth() / 2))
      .y(d => yBar(d.totalLen));
    rightG.append('path')
      .datum(fits)
      .attr('d', totalLine)
      .style('fill', 'none')
      .style('stroke', '#7c3aed')
      .style('stroke-width', 2.5);
    fits.forEach(f => {
      rightG.append('circle')
        .attr('cx', xBar(f.degree)! + xBar.bandwidth() / 2)
        .attr('cy', yBar(f.totalLen))
        .attr('r', 2.5)
        .style('fill', '#7c3aed');
    });

    // Optimal degree marker
    const optX = xBar(optimalDegree)! + xBar.bandwidth() / 2;
    rightG.append('line')
      .attr('x1', optX).attr('y1', 0)
      .attr('x2', optX).attr('y2', rH)
      .style('stroke', '#10b981')
      .style('stroke-width', 2)
      .style('stroke-dasharray', '4,3');
    rightG.append('text')
      .attr('x', optX).attr('y', -4)
      .attr('text-anchor', 'middle')
      .style('font-size', '10px')
      .style('fill', '#10b981')
      .style('font-weight', '600')
      .text(`d* = ${optimalDegree}`);

    // Legend
    const legX = rW - 120;
    rightG.append('rect').attr('x', legX).attr('y', 8).attr('width', 12).attr('height', 12).style('fill', '#3b82f6').style('opacity', 0.7);
    rightG.append('text').attr('x', legX + 16).attr('y', 18).style('font-size', '10px').style('fill', 'var(--color-text-secondary, #666)').text('L(M)');
    rightG.append('rect').attr('x', legX).attr('y', 24).attr('width', 12).attr('height', 12).style('fill', '#ef4444').style('opacity', 0.7);
    rightG.append('text').attr('x', legX + 16).attr('y', 34).style('font-size', '10px').style('fill', 'var(--color-text-secondary, #666)').text('L(D|M)');
    rightG.append('line').attr('x1', legX).attr('y1', 46).attr('x2', legX + 12).attr('y2', 46).style('stroke', '#7c3aed').style('stroke-width', 2.5);
    rightG.append('text').attr('x', legX + 16).attr('y', 50).style('font-size', '10px').style('fill', 'var(--color-text-secondary, #666)').text('Total');

    rightG.append('text')
      .attr('x', rW / 2).attr('y', -12)
      .attr('text-anchor', 'middle')
      .style('font-size', '12px')
      .style('fill', 'var(--color-text-secondary, #666)')
      .style('font-weight', '600')
      .text('Two-Part Code Decomposition');

  }, [containerWidth, data, fits, optimalDegree, isSmall, totalWidth, leftW, rightW]);

  useEffect(() => {
    drawChart();
  }, [drawChart]);

  return (
    <div ref={containerRef} className="my-6 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 p-4">
      <div className="mb-3 flex flex-wrap items-center gap-4">
        <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
          Max degree:
          <input
            type="range" min={5} max={20} step={1} value={maxDegree}
            onChange={e => setMaxDegree(Number(e.target.value))}
            className="ml-2 w-24 align-middle"
          />
          <span className="ml-1 text-xs font-mono">{maxDegree}</span>
        </label>
        <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
          Noise σ:
          <input
            type="range" min={0.1} max={1.0} step={0.05} value={noiseLevel}
            onChange={e => setNoiseLevel(Number(e.target.value))}
            className="ml-2 w-24 align-middle"
          />
          <span className="ml-1 text-xs font-mono">{noiseLevel.toFixed(2)}</span>
        </label>
        <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
          n:
          <select
            value={sampleSizeIdx}
            onChange={e => setSampleSizeIdx(Number(e.target.value))}
            className="ml-2 rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-2 py-1 text-sm text-gray-900 dark:text-gray-100"
          >
            {SAMPLE_SIZES.map((s, i) => (
              <option key={s} value={i}>{s}</option>
            ))}
          </select>
        </label>
        <button
          onClick={() => setSeed(s => s + 1)}
          className="rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-3 py-1 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700"
        >
          Regenerate data
        </button>
      </div>
      <svg ref={svgRef} width={totalWidth} height={svgHeight} />
    </div>
  );
}
