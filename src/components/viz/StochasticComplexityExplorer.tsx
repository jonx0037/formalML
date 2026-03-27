import { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { bernoulliNML, bernoulliCOMP, logBinomial, logGamma } from './shared/informationTheory';

const PANEL_HEIGHT = 320;
const SM_BREAKPOINT = 640;
const MARGIN = { top: 30, right: 15, bottom: 50, left: 55 };

type ComparisonCode = 'bayes-uniform' | 'bayes-jeffreys' | 'plugin-mle';

const COMPARISON_OPTIONS: { value: ComparisonCode; label: string }[] = [
  { value: 'bayes-uniform', label: 'Bayes (uniform prior)' },
  { value: 'bayes-jeffreys', label: 'Bayes (Jeffreys prior)' },
  { value: 'plugin-mle', label: 'Plug-in MLE' },
];

// Bayes predictive probability for Bernoulli
// P_Bayes(k | n) = C(n,k) * B(k + a, n - k + b) / B(a, b) where B is the beta function
// For uniform prior (a=b=1): P(k|n) = C(n,k) * k! * (n-k)! / (n+1)!
// For Jeffreys prior (a=b=0.5): P(k|n) = C(n,k) * Γ(k+0.5)*Γ(n-k+0.5) / (Γ(0.5)^2 * Γ(n+1))
function bayesLogProb(k: number, n: number, prior: 'uniform' | 'jeffreys'): number {
  const logBinom = logBinomial(n, k); // in log2
  if (prior === 'uniform') {
    // log2(B(k+1, n-k+1) / B(1,1)) = log2(k! * (n-k)! / (n+1)!)
    // Using logGamma: ln(Γ(k+1)) + ln(Γ(n-k+1)) - ln(Γ(n+2))
    const logBeta = (logGamma(k + 1) + logGamma(n - k + 1) - logGamma(n + 2)) / Math.LN2;
    return logBinom + logBeta;
  } else {
    // Jeffreys: B(k+0.5, n-k+0.5) / B(0.5, 0.5)
    const logBetaNum = (logGamma(k + 0.5) + logGamma(n - k + 0.5) - logGamma(n + 1)) / Math.LN2;
    const logBetaDen = (logGamma(0.5) + logGamma(0.5) - logGamma(1)) / Math.LN2;
    return logBinom + logBetaNum - logBetaDen;
  }
}

function pluginMLELogProb(k: number, n: number): number {
  // Plug-in MLE: p(k|n) = C(n,k) * theta_hat^k * (1 - theta_hat)^(n-k)
  // where theta_hat = k/n is the maximum-likelihood estimate for the Bernoulli parameter.
  // This is the maximized likelihood for the observed k. It does not define
  // a normalized distribution over all possible k, but is useful as a baseline code.
  if (n === 0) return 0;
  const thetaHat = k / n;
  const logBinom = logBinomial(n, k);
  let logLik = 0;
  if (k > 0) logLik += k * Math.log2(thetaHat);
  if (k < n) logLik += (n - k) * Math.log2(1 - thetaHat);
  return logBinom + logLik;
}

function comparisonLogProb(k: number, n: number, code: ComparisonCode): number {
  if (code === 'bayes-uniform') return bayesLogProb(k, n, 'uniform');
  if (code === 'bayes-jeffreys') return bayesLogProb(k, n, 'jeffreys');
  return pluginMLELogProb(k, n);
}

export default function StochasticComplexityExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const svgRef = useRef<SVGSVGElement>(null);

  const [sampleSize, setSampleSize] = useState(20);
  const [compCode, setCompCode] = useState<ComparisonCode>('bayes-uniform');

  const comp = useMemo(() => bernoulliCOMP(sampleSize), [sampleSize]);

  const nmlData = useMemo(() => {
    const result: { k: number; nmlProb: number; nmlLogProb: number; compLogProb: number }[] = [];
    for (let k = 0; k <= sampleSize; k++) {
      const nml = bernoulliNML(k, sampleSize, comp);
      const cLogProb = comparisonLogProb(k, sampleSize, compCode);
      result.push({ k, nmlProb: nml.prob, nmlLogProb: nml.logProb, compLogProb: cLogProb });
    }
    return result;
  }, [sampleSize, comp, compCode]);

  // Regret = log2(MLE / code) = -log2(code) - (-log2(MLE)) = -compLogProb - (-nmlLogProb for NML is log2(COMP))
  // NML regret = log2(COMP) for all k (constant)
  // Comparison regret = log2(MLE(k)) - compLogProb(k)
  const regretData = useMemo(() => {
    const nmlRegret = Math.log2(comp);
    return nmlData.map(d => {
      // MLE log prob = logBinomial + k*log2(k/n) + (n-k)*log2((n-k)/n)
      let mleLogProb = logBinomial(sampleSize, d.k);
      if (d.k > 0) mleLogProb += d.k * Math.log2(d.k / sampleSize);
      if (d.k < sampleSize) mleLogProb += (sampleSize - d.k) * Math.log2((sampleSize - d.k) / sampleSize);
      const compRegret = mleLogProb - d.compLogProb;
      return { k: d.k, nmlRegret, compRegret };
    });
  }, [nmlData, comp, sampleSize]);

  // COMP growth data
  const compGrowth = useMemo(() => {
    const points: { n: number; exact: number; asymptotic: number }[] = [];
    for (let ni = 2; ni <= 50; ni++) {
      const c = bernoulliCOMP(ni);
      points.push({
        n: ni,
        exact: Math.log2(c),
        asymptotic: 0.5 * Math.log2(ni / (2 * Math.PI)),
      });
    }
    return points;
  }, []);

  const isSmall = containerWidth < SM_BREAKPOINT;
  const totalWidth = Math.max(containerWidth, 300);
  const panelW = isSmall ? totalWidth : Math.floor(totalWidth / 3);
  const svgHeight = isSmall ? PANEL_HEIGHT * 3 + 40 : PANEL_HEIGHT;

  const drawChart = useCallback(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    if (totalWidth <= 0) return;

    const w = panelW - MARGIN.left - MARGIN.right;
    const h = PANEL_HEIGHT - MARGIN.top - MARGIN.bottom;

    // --- Panel 1: NML vs Bernoulli(0.5) ---
    const g1 = svg.append('g')
      .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    const xBand = d3.scaleBand<number>()
      .domain(nmlData.map(d => d.k))
      .range([0, w])
      .padding(0.15);

    // Bernoulli(0.5) for comparison
    const bernProbs = nmlData.map(d => {
      let lp = logBinomial(sampleSize, d.k);
      lp += sampleSize * Math.log2(0.5); // theta=0.5
      return Math.pow(2, lp);
    });

    const maxProb = Math.max(...nmlData.map(d => d.nmlProb), ...bernProbs) * 1.1;
    const yProb = d3.scaleLinear().domain([0, maxProb]).range([h, 0]);

    // Bernoulli bars (blue, behind)
    const barW = xBand.bandwidth() / 2;
    nmlData.forEach((d, i) => {
      g1.append('rect')
        .attr('x', xBand(d.k)!)
        .attr('y', yProb(bernProbs[i]))
        .attr('width', barW)
        .attr('height', h - yProb(bernProbs[i]))
        .style('fill', '#3b82f6')
        .style('opacity', 0.6);
    });
    // NML bars (purple, front)
    nmlData.forEach(d => {
      g1.append('rect')
        .attr('x', xBand(d.k)! + barW)
        .attr('y', yProb(d.nmlProb))
        .attr('width', barW)
        .attr('height', h - yProb(d.nmlProb))
        .style('fill', '#7c3aed')
        .style('opacity', 0.7);
    });

    // Axes - show subset of ticks for readability
    const tickStep = sampleSize <= 15 ? 1 : sampleSize <= 30 ? 5 : 10;
    g1.append('g')
      .attr('transform', `translate(0,${h})`)
      .call(d3.axisBottom(xBand).tickValues(nmlData.filter(d => d.k % tickStep === 0).map(d => d.k)))
      .selectAll('text').style('fill', 'var(--color-text-secondary, #666)').style('font-size', '9px');
    g1.append('g')
      .call(d3.axisLeft(yProb).ticks(4))
      .selectAll('text').style('fill', 'var(--color-text-secondary, #666)');
    g1.selectAll('.domain, .tick line').style('stroke', 'var(--color-text-secondary, #999)');

    g1.append('text').attr('x', w / 2).attr('y', h + 40).attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text-secondary, #666)').style('font-size', '12px').text('k (successes)');
    g1.append('text').attr('transform', 'rotate(-90)').attr('x', -h / 2).attr('y', -42)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text-secondary, #666)').style('font-size', '12px').text('Probability');
    g1.append('text').attr('x', w / 2).attr('y', -12).attr('text-anchor', 'middle')
      .style('font-size', '12px').style('fill', 'var(--color-text-secondary, #666)').style('font-weight', '600')
      .text('NML vs Bernoulli(0.5)');

    // Legend
    g1.append('rect').attr('x', w - 100).attr('y', 4).attr('width', 10).attr('height', 10).style('fill', '#3b82f6').style('opacity', 0.6);
    g1.append('text').attr('x', w - 86).attr('y', 13).style('font-size', '9px').style('fill', 'var(--color-text-secondary, #666)').text('Bern(0.5)');
    g1.append('rect').attr('x', w - 100).attr('y', 18).attr('width', 10).attr('height', 10).style('fill', '#7c3aed').style('opacity', 0.7);
    g1.append('text').attr('x', w - 86).attr('y', 27).style('font-size', '9px').style('fill', 'var(--color-text-secondary, #666)').text('NML');

    // COMP annotation
    g1.append('text').attr('x', 4).attr('y', h - 4)
      .style('font-size', '10px').style('fill', '#7c3aed').style('font-weight', '500')
      .text(`COMP = ${comp.toFixed(2)}`);

    // --- Panel 2: Regret ---
    const p2X = isSmall ? 0 : panelW;
    const p2Y = isSmall ? PANEL_HEIGHT + 20 : 0;
    const g2 = svg.append('g')
      .attr('transform', `translate(${p2X + MARGIN.left},${p2Y + MARGIN.top})`);

    const xRegret = d3.scaleLinear().domain([0, sampleSize]).range([0, w]);
    const maxRegret = Math.max(
      Math.log2(comp) * 1.3,
      Math.max(...regretData.map(d => d.compRegret)) * 1.1
    );
    const yRegret = d3.scaleLinear().domain([Math.min(0, ...regretData.map(d => d.compRegret)), maxRegret]).range([h, 0]);

    // NML regret (horizontal line)
    g2.append('line')
      .attr('x1', 0).attr('y1', yRegret(Math.log2(comp)))
      .attr('x2', w).attr('y2', yRegret(Math.log2(comp)))
      .style('stroke', '#7c3aed')
      .style('stroke-width', 2.5);

    // Comparison code regret
    const compLine = d3.line<typeof regretData[0]>()
      .x(d => xRegret(d.k))
      .y(d => yRegret(d.compRegret));
    g2.append('path')
      .datum(regretData)
      .attr('d', compLine)
      .style('fill', 'none')
      .style('stroke', '#f59e0b')
      .style('stroke-width', 2);

    g2.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(xRegret).ticks(5))
      .selectAll('text').style('fill', 'var(--color-text-secondary, #666)');
    g2.append('g').call(d3.axisLeft(yRegret).ticks(5))
      .selectAll('text').style('fill', 'var(--color-text-secondary, #666)');
    g2.selectAll('.domain, .tick line').style('stroke', 'var(--color-text-secondary, #999)');

    g2.append('text').attr('x', w / 2).attr('y', h + 40).attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text-secondary, #666)').style('font-size', '12px').text('k (successes)');
    g2.append('text').attr('transform', 'rotate(-90)').attr('x', -h / 2).attr('y', -42)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text-secondary, #666)').style('font-size', '12px').text('Regret (bits)');
    g2.append('text').attr('x', w / 2).attr('y', -12).attr('text-anchor', 'middle')
      .style('font-size', '12px').style('fill', 'var(--color-text-secondary, #666)').style('font-weight', '600')
      .text('Minimax Regret');

    // Annotation
    g2.append('text').attr('x', w - 4).attr('y', yRegret(Math.log2(comp)) - 8)
      .attr('text-anchor', 'end')
      .style('font-size', '10px').style('fill', '#7c3aed').style('font-weight', '500')
      .text(`NML: constant = ${Math.log2(comp).toFixed(2)}`);

    // Legend
    g2.append('line').attr('x1', 4).attr('y1', 8).attr('x2', 24).attr('y2', 8)
      .style('stroke', '#7c3aed').style('stroke-width', 2.5);
    g2.append('text').attr('x', 28).attr('y', 12).style('font-size', '9px').style('fill', 'var(--color-text-secondary, #666)').text('NML');
    g2.append('line').attr('x1', 4).attr('y1', 22).attr('x2', 24).attr('y2', 22)
      .style('stroke', '#f59e0b').style('stroke-width', 2);
    g2.append('text').attr('x', 28).attr('y', 26).style('font-size', '9px').style('fill', 'var(--color-text-secondary, #666)')
      .text(COMPARISON_OPTIONS.find(o => o.value === compCode)?.label ?? '');

    // --- Panel 3: COMP growth ---
    const p3X = isSmall ? 0 : panelW * 2;
    const p3Y = isSmall ? (PANEL_HEIGHT + 20) * 2 : 0;
    const g3 = svg.append('g')
      .attr('transform', `translate(${p3X + MARGIN.left},${p3Y + MARGIN.top})`);

    const xComp = d3.scaleLinear().domain([2, 50]).range([0, w]);
    const maxComp = Math.max(...compGrowth.map(d => d.exact)) * 1.1;
    const yComp = d3.scaleLinear().domain([0, maxComp]).range([h, 0]);

    // Exact COMP
    const exactLine = d3.line<typeof compGrowth[0]>()
      .x(d => xComp(d.n)).y(d => yComp(d.exact));
    g3.append('path')
      .datum(compGrowth)
      .attr('d', exactLine)
      .style('fill', 'none')
      .style('stroke', '#7c3aed')
      .style('stroke-width', 2.5);

    // Asymptotic approximation
    const asymLine = d3.line<typeof compGrowth[0]>()
      .x(d => xComp(d.n)).y(d => yComp(d.asymptotic));
    g3.append('path')
      .datum(compGrowth)
      .attr('d', asymLine)
      .style('fill', 'none')
      .style('stroke', '#f59e0b')
      .style('stroke-width', 2)
      .style('stroke-dasharray', '6,3');

    // Current n marker
    const curComp = compGrowth.find(d => d.n === sampleSize);
    if (curComp) {
      g3.append('circle')
        .attr('cx', xComp(curComp.n)).attr('cy', yComp(curComp.exact))
        .attr('r', 5).style('fill', '#7c3aed').style('stroke', '#fff').style('stroke-width', 2);
    }

    g3.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(xComp).ticks(5))
      .selectAll('text').style('fill', 'var(--color-text-secondary, #666)');
    g3.append('g').call(d3.axisLeft(yComp).ticks(5))
      .selectAll('text').style('fill', 'var(--color-text-secondary, #666)');
    g3.selectAll('.domain, .tick line').style('stroke', 'var(--color-text-secondary, #999)');

    g3.append('text').attr('x', w / 2).attr('y', h + 40).attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text-secondary, #666)').style('font-size', '12px').text('Sample size n');
    g3.append('text').attr('transform', 'rotate(-90)').attr('x', -h / 2).attr('y', -42)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text-secondary, #666)').style('font-size', '12px').text('log₂ COMP(n)');
    g3.append('text').attr('x', w / 2).attr('y', -12).attr('text-anchor', 'middle')
      .style('font-size', '12px').style('fill', 'var(--color-text-secondary, #666)').style('font-weight', '600')
      .text('Parametric Complexity Growth');

    g3.append('line').attr('x1', 4).attr('y1', 8).attr('x2', 24).attr('y2', 8)
      .style('stroke', '#7c3aed').style('stroke-width', 2.5);
    g3.append('text').attr('x', 28).attr('y', 12).style('font-size', '9px').style('fill', 'var(--color-text-secondary, #666)').text('Exact');
    g3.append('line').attr('x1', 4).attr('y1', 22).attr('x2', 24).attr('y2', 22)
      .style('stroke', '#f59e0b').style('stroke-width', 2).style('stroke-dasharray', '6,3');
    g3.append('text').attr('x', 28).attr('y', 26).style('font-size', '9px').style('fill', 'var(--color-text-secondary, #666)').text('½ log₂(n/2π)');

  }, [containerWidth, nmlData, regretData, compGrowth, comp, sampleSize, compCode, isSmall, totalWidth, panelW]);

  useEffect(() => {
    drawChart();
  }, [drawChart]);

  return (
    <div ref={containerRef} className="my-6 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 p-4">
      <div className="mb-3 flex flex-wrap items-center gap-4">
        <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
          Sample size n:
          <input
            type="range" min={5} max={50} step={1} value={sampleSize}
            onChange={e => setSampleSize(Number(e.target.value))}
            className="ml-2 w-28 align-middle"
          />
          <span className="ml-1 text-xs font-mono">{sampleSize}</span>
        </label>
        <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
          Comparison code:
          <select
            value={compCode}
            onChange={e => setCompCode(e.target.value as ComparisonCode)}
            className="ml-2 rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-2 py-1 text-sm text-gray-900 dark:text-gray-100"
          >
            {COMPARISON_OPTIONS.map(o => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>
        </label>
      </div>
      <svg ref={svgRef} width={totalWidth} height={svgHeight} />
    </div>
  );
}
