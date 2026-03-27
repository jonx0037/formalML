import { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { bicScore, aicScore, mdlScore } from './shared/informationTheory';

const HEIGHT = 340;
const SM_BREAKPOINT = 640;
const MARGIN = { top: 30, right: 20, bottom: 50, left: 55 };

// Seeded LCG PRNG
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

function evalPoly(coeffs: number[], x: number): number {
  let y = 0;
  for (let i = 0; i < coeffs.length; i++) y += coeffs[i] * Math.pow(x, i);
  return y;
}

// Polynomial least-squares fit via normal equations
function polyFit(xs: number[], ys: number[], degree: number): number[] {
  const n = xs.length;
  const k = degree + 1;
  const XtX: number[][] = Array.from({ length: k }, () => new Array(k).fill(0));
  const Xty: number[] = new Array(k).fill(0);
  for (let i = 0; i < n; i++) {
    const powers: number[] = [1];
    for (let j = 1; j < k; j++) powers.push(powers[j - 1] * xs[i]);
    for (let r = 0; r < k; r++) {
      for (let c = 0; c < k; c++) XtX[r][c] += powers[r] * powers[c];
      Xty[r] += powers[r] * ys[i];
    }
  }
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

function computeLogLik(xs: number[], ys: number[], coeffs: number[]): number {
  const n = xs.length;
  let rss = 0;
  for (let i = 0; i < n; i++) {
    const diff = ys[i] - evalPoly(coeffs, xs[i]);
    rss += diff * diff;
  }
  const sigmaHat2 = Math.max(rss / n, 1e-15);
  return -n / 2 * (Math.log(2 * Math.PI * sigmaHat2) + 1);
}

function generateTrueCoeffs(degree: number): number[] {
  const presets: Record<number, number[]> = {
    1: [1, 2],
    2: [1, -0.5, 2],
    3: [1, -0.5, 2, -1.5],
    4: [1, -0.5, 2, -1.5, 0.8],
    5: [1, -0.5, 2, -1.5, 0.8, -0.3],
  };
  return presets[degree] || presets[3];
}

const MAX_DEGREE = 12;
const CONSISTENCY_NS = [20, 50, 100, 200, 500];
const N_SIMS = 30;

export default function MDLModelSelectionExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const svgRef = useRef<SVGSVGElement>(null);

  const [trueDegree, setTrueDegree] = useState(3);
  const [noiseLevel, setNoiseLevel] = useState(0.3);
  const [displayN, setDisplayN] = useState(100);

  const trueCoeffs = useMemo(() => generateTrueCoeffs(trueDegree), [trueDegree]);

  // Criterion curves for a single dataset
  const criterionData = useMemo(() => {
    const xs: number[] = [];
    const ys: number[] = [];
    for (let i = 0; i < displayN; i++) {
      const x = -1 + (2 * i) / (displayN - 1);
      xs.push(x);
      ys.push(evalPoly(trueCoeffs, x) + seededNormal(i, 42) * noiseLevel);
    }
    const results: { degree: number; mdl: number; bic: number; aic: number }[] = [];
    for (let d = 0; d <= MAX_DEGREE; d++) {
      const coeffs = polyFit(xs, ys, d);
      const logLik = computeLogLik(xs, ys, coeffs);
      const k = d + 1;
      results.push({
        degree: d,
        mdl: mdlScore(logLik, displayN, k),
        bic: bicScore(logLik, displayN, k),
        aic: aicScore(logLik, k),
      });
    }
    return results;
  }, [trueCoeffs, noiseLevel, displayN]);

  // Consistency simulation
  const consistencyData = useMemo(() => {
    const results: { n: number; mdlRate: number; bicRate: number; aicRate: number }[] = [];
    for (const ni of CONSISTENCY_NS) {
      let mdlCorrect = 0, bicCorrect = 0, aicCorrect = 0;
      for (let sim = 0; sim < N_SIMS; sim++) {
        const xs: number[] = [];
        const ys: number[] = [];
        for (let i = 0; i < ni; i++) {
          const x = -1 + (2 * i) / (ni - 1);
          xs.push(x);
          ys.push(evalPoly(trueCoeffs, x) + seededNormal(i, sim * 1000 + 7) * noiseLevel);
        }
        let bestMDL = Infinity, bestBIC = Infinity, bestAIC = Infinity;
        let mdlDeg = 0, bicDeg = 0, aicDeg = 0;
        for (let d = 0; d <= MAX_DEGREE; d++) {
          const coeffs = polyFit(xs, ys, d);
          const logLik = computeLogLik(xs, ys, coeffs);
          const k = d + 1;
          const m = mdlScore(logLik, ni, k);
          const b = bicScore(logLik, ni, k);
          const a = aicScore(logLik, k);
          if (m < bestMDL) { bestMDL = m; mdlDeg = d; }
          if (b < bestBIC) { bestBIC = b; bicDeg = d; }
          if (a < bestAIC) { bestAIC = a; aicDeg = d; }
        }
        if (mdlDeg === trueDegree) mdlCorrect++;
        if (bicDeg === trueDegree) bicCorrect++;
        if (aicDeg === trueDegree) aicCorrect++;
      }
      results.push({
        n: ni,
        mdlRate: mdlCorrect / N_SIMS,
        bicRate: bicCorrect / N_SIMS,
        aicRate: aicCorrect / N_SIMS,
      });
    }
    return results;
  }, [trueCoeffs, trueDegree, noiseLevel]);

  const isSmall = containerWidth < SM_BREAKPOINT;
  const totalWidth = Math.max(containerWidth, 300);
  const leftW = isSmall ? totalWidth : Math.floor(totalWidth * 0.5);
  const rightW = isSmall ? totalWidth : totalWidth - leftW;
  const svgHeight = isSmall ? HEIGHT * 2 + 40 : HEIGHT;

  const drawChart = useCallback(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    if (totalWidth <= 0) return;

    const lW = leftW - MARGIN.left - MARGIN.right;
    const lH = HEIGHT - MARGIN.top - MARGIN.bottom;

    // --- Left panel: Criterion curves ---
    const g1 = svg.append('g')
      .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    const xScale = d3.scaleLinear().domain([0, MAX_DEGREE]).range([0, lW]);
    const allVals = criterionData.flatMap(d => [d.mdl, d.bic, d.aic]);
    const yMin = Math.min(...allVals) * 0.95;
    const yMax = Math.max(...allVals) * 1.05;
    const yScale = d3.scaleLinear().domain([yMin, yMax]).range([lH, 0]);

    // Criterion lines
    const colors = { mdl: '#7c3aed', bic: '#3b82f6', aic: '#f59e0b' };
    const labels = { mdl: 'MDL', bic: 'BIC', aic: 'AIC' };
    for (const key of ['aic', 'bic', 'mdl'] as const) {
      const line = d3.line<typeof criterionData[0]>()
        .x(d => xScale(d.degree))
        .y(d => yScale(d[key]));
      g1.append('path')
        .datum(criterionData)
        .attr('d', line)
        .style('fill', 'none')
        .style('stroke', colors[key])
        .style('stroke-width', key === 'mdl' ? 2.5 : 2)
        .style('stroke-dasharray', key === 'aic' ? '4,3' : 'none');
      criterionData.forEach(d => {
        g1.append('circle')
          .attr('cx', xScale(d.degree)).attr('cy', yScale(d[key]))
          .attr('r', 2).style('fill', colors[key]);
      });
    }

    // True degree marker
    g1.append('line')
      .attr('x1', xScale(trueDegree)).attr('y1', 0)
      .attr('x2', xScale(trueDegree)).attr('y2', lH)
      .style('stroke', '#10b981').style('stroke-width', 1.5).style('stroke-dasharray', '4,3');
    g1.append('text')
      .attr('x', xScale(trueDegree)).attr('y', -4)
      .attr('text-anchor', 'middle')
      .style('font-size', '10px').style('fill', '#10b981').style('font-weight', '600')
      .text(`true d = ${trueDegree}`);

    // Find and annotate minima
    for (const key of ['mdl', 'bic', 'aic'] as const) {
      let bestIdx = 0;
      for (let i = 1; i < criterionData.length; i++) {
        if (criterionData[i][key] < criterionData[bestIdx][key]) bestIdx = i;
      }
      g1.append('circle')
        .attr('cx', xScale(criterionData[bestIdx].degree))
        .attr('cy', yScale(criterionData[bestIdx][key]))
        .attr('r', 5)
        .style('fill', 'none').style('stroke', colors[key]).style('stroke-width', 2);
    }

    g1.append('g').attr('transform', `translate(0,${lH})`).call(d3.axisBottom(xScale).ticks(MAX_DEGREE).tickFormat(d3.format('d')))
      .selectAll('text').style('fill', 'var(--color-text-secondary, #666)');
    g1.append('g').call(d3.axisLeft(yScale).ticks(5))
      .selectAll('text').style('fill', 'var(--color-text-secondary, #666)');
    g1.selectAll('.domain, .tick line').style('stroke', 'var(--color-text-secondary, #999)');

    g1.append('text').attr('x', lW / 2).attr('y', lH + 40).attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text-secondary, #666)').style('font-size', '12px').text('Model degree d');
    g1.append('text').attr('transform', 'rotate(-90)').attr('x', -lH / 2).attr('y', -40)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text-secondary, #666)').style('font-size', '12px').text('Criterion value');
    g1.append('text').attr('x', lW / 2).attr('y', -12).attr('text-anchor', 'middle')
      .style('font-size', '12px').style('fill', 'var(--color-text-secondary, #666)').style('font-weight', '600')
      .text('MDL vs BIC vs AIC');

    // Legend
    let ly = 8;
    for (const key of ['mdl', 'bic', 'aic'] as const) {
      g1.append('line').attr('x1', lW - 80).attr('y1', ly).attr('x2', lW - 60).attr('y2', ly)
        .style('stroke', colors[key]).style('stroke-width', 2)
        .style('stroke-dasharray', key === 'aic' ? '4,3' : 'none');
      g1.append('text').attr('x', lW - 56).attr('y', ly + 4)
        .style('font-size', '10px').style('fill', 'var(--color-text-secondary, #666)').text(labels[key]);
      ly += 16;
    }

    // --- Right panel: Consistency simulation ---
    const rOffset = isSmall ? 0 : leftW;
    const rYOffset = isSmall ? HEIGHT + 20 : 0;
    const rW = (isSmall ? totalWidth : rightW) - MARGIN.left - MARGIN.right;
    const rH = HEIGHT - MARGIN.top - MARGIN.bottom;
    const g2 = svg.append('g')
      .attr('transform', `translate(${rOffset + MARGIN.left},${rYOffset + MARGIN.top})`);

    const xBand = d3.scaleBand<number>()
      .domain(CONSISTENCY_NS)
      .range([0, rW])
      .padding(0.2);
    const yRate = d3.scaleLinear().domain([0, 1.05]).range([rH, 0]);
    const groupW = xBand.bandwidth() / 3;

    consistencyData.forEach(d => {
      const bx = xBand(d.n)!;
      // MDL bar
      g2.append('rect').attr('x', bx).attr('y', yRate(d.mdlRate))
        .attr('width', groupW).attr('height', rH - yRate(d.mdlRate))
        .style('fill', colors.mdl).style('opacity', 0.8);
      // BIC bar
      g2.append('rect').attr('x', bx + groupW).attr('y', yRate(d.bicRate))
        .attr('width', groupW).attr('height', rH - yRate(d.bicRate))
        .style('fill', colors.bic).style('opacity', 0.8);
      // AIC bar
      g2.append('rect').attr('x', bx + groupW * 2).attr('y', yRate(d.aicRate))
        .attr('width', groupW).attr('height', rH - yRate(d.aicRate))
        .style('fill', colors.aic).style('opacity', 0.8);
    });

    g2.append('g').attr('transform', `translate(0,${rH})`).call(d3.axisBottom(xBand))
      .selectAll('text').style('fill', 'var(--color-text-secondary, #666)');
    g2.append('g').call(d3.axisLeft(yRate).ticks(5).tickFormat(d3.format('.0%')))
      .selectAll('text').style('fill', 'var(--color-text-secondary, #666)');
    g2.selectAll('.domain, .tick line').style('stroke', 'var(--color-text-secondary, #999)');

    g2.append('text').attr('x', rW / 2).attr('y', rH + 40).attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text-secondary, #666)').style('font-size', '12px').text('Sample size n');
    g2.append('text').attr('transform', 'rotate(-90)').attr('x', -rH / 2).attr('y', -40)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text-secondary, #666)').style('font-size', '12px').text('P(correct model)');
    g2.append('text').attr('x', rW / 2).attr('y', -12).attr('text-anchor', 'middle')
      .style('font-size', '12px').style('fill', 'var(--color-text-secondary, #666)').style('font-weight', '600')
      .text(`Consistency (${N_SIMS} trials per n)`);

    // Legend
    ly = 8;
    for (const key of ['mdl', 'bic', 'aic'] as const) {
      g2.append('rect').attr('x', rW - 80).attr('y', ly - 5).attr('width', 12).attr('height', 12)
        .style('fill', colors[key]).style('opacity', 0.8);
      g2.append('text').attr('x', rW - 64).attr('y', ly + 4)
        .style('font-size', '10px').style('fill', 'var(--color-text-secondary, #666)').text(labels[key]);
      ly += 16;
    }

  }, [containerWidth, criterionData, consistencyData, trueDegree, isSmall, totalWidth, leftW, rightW]);

  useEffect(() => {
    drawChart();
  }, [drawChart]);

  return (
    <div ref={containerRef} className="my-6 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 p-4">
      <div className="mb-3 flex flex-wrap items-center gap-4">
        <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
          True degree:
          <select
            value={trueDegree}
            onChange={e => setTrueDegree(Number(e.target.value))}
            className="ml-2 rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-2 py-1 text-sm text-gray-900 dark:text-gray-100"
          >
            {[1, 2, 3, 4, 5].map(d => (
              <option key={d} value={d}>{d}</option>
            ))}
          </select>
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
          Display n:
          <input
            type="range" min={20} max={500} step={10} value={displayN}
            onChange={e => setDisplayN(Number(e.target.value))}
            className="ml-2 w-28 align-middle"
          />
          <span className="ml-1 text-xs font-mono">{displayN}</span>
        </label>
      </div>
      <svg ref={svgRef} width={totalWidth} height={svgHeight} />
    </div>
  );
}
