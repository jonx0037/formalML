import { useState, useMemo, useCallback } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { rateDistortionGaussian } from './shared/informationTheory';

const HEIGHT = 340;
const SM_BREAKPOINT = 640;
const MARGIN = { top: 30, right: 20, bottom: 50, left: 55 };

const SIGMA2_OPTIONS = [0.5, 1, 2, 4];

function vaeOperatingPoint(beta: number, sigma2: number) {
  // Operating point on Gaussian R(D) where β weights the KL term in the β-VAE ELBO.
  // For R(D) = ½ log₂(σ²/D), the optimal D at a given β is D = σ² / (1 + β).
  const D = sigma2 / (1 + beta);
  const R = rateDistortionGaussian(sigma2, D);
  return { D, R };
}

export default function VAERateDistortionExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();

  const [beta, setBeta] = useState(1);
  const [sigma2, setSigma2] = useState(1);

  const isSmall = containerWidth < SM_BREAKPOINT;
  const leftW = isSmall ? containerWidth : Math.floor(containerWidth * 0.5);
  const rightW = isSmall ? containerWidth : containerWidth - leftW;
  const svgHeight = isSmall ? HEIGHT * 2 + 20 : HEIGHT;

  // R(D) curve points
  const rdCurve = useMemo(() => {
    const pts: { D: number; R: number }[] = [];
    for (let i = 1; i <= 200; i++) {
      const D = (i / 200) * sigma2;
      pts.push({ D, R: rateDistortionGaussian(sigma2, D) });
    }
    return pts;
  }, [sigma2]);

  // Operating points for multiple betas
  const betaPoints = useMemo(() => {
    const betas = [0.1, 0.2, 0.5, 1, 2, 5, 10];
    return betas.map(b => ({ beta: b, ...vaeOperatingPoint(b, sigma2) }));
  }, [sigma2]);

  // Rate and distortion as functions of beta
  const tradeoffCurve = useMemo(() => {
    const pts: { beta: number; rate: number; distortion: number }[] = [];
    for (let i = 1; i <= 100; i++) {
      const b = 0.1 * Math.pow(100, i / 100);
      const { D, R } = vaeOperatingPoint(b, sigma2);
      pts.push({ beta: b, rate: R, distortion: D });
    }
    return pts;
  }, [sigma2]);

  const currentOp = useMemo(() => vaeOperatingPoint(beta, sigma2), [beta, sigma2]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (containerWidth <= 0) return;

      const w1 = leftW - MARGIN.left - MARGIN.right;
      const h = HEIGHT - MARGIN.top - MARGIN.bottom;

      // --- Left: R(D) curve with VAE operating points ---
      const g1 = svg.append('g')
        .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const maxR = rateDistortionGaussian(sigma2, sigma2 * 0.01) * 1.05;
      const xScale = d3.scaleLinear().domain([0, sigma2 * 1.05]).range([0, w1]);
      const yScale = d3.scaleLinear().domain([0, maxR]).range([h, 0]);

      // Achievable region shading
      const areaAbove = d3.area<{ D: number; R: number }>()
        .x(d => xScale(d.D))
        .y0(d => yScale(d.R))
        .y1(0);
      g1.append('path')
        .datum(rdCurve)
        .attr('d', areaAbove)
        .style('fill', '#10b981')
        .style('fill-opacity', 0.06);

      // R(D) curve
      const lineGen = d3.line<{ D: number; R: number }>()
        .x(d => xScale(d.D)).y(d => yScale(d.R));
      g1.append('path')
        .datum(rdCurve)
        .attr('d', lineGen)
        .style('fill', 'none')
        .style('stroke', '#0F6E56')
        .style('stroke-width', 2.5);

      // Operating points
      betaPoints.forEach(pt => {
        const isCurrent = Math.abs(pt.beta - beta) < 0.05;
        g1.append('circle')
          .attr('cx', xScale(pt.D)).attr('cy', yScale(pt.R))
          .attr('r', isCurrent ? 8 : 4)
          .style('fill', isCurrent ? '#534AB7' : '#999')
          .style('stroke', isCurrent ? '#fff' : 'none')
          .style('stroke-width', isCurrent ? 2 : 0)
          .style('opacity', isCurrent ? 1 : 0.6);

        if (!isCurrent && pt.R > 0 && yScale(pt.R) > 15) {
          g1.append('text')
            .attr('x', xScale(pt.D) + 8).attr('y', yScale(pt.R) + 4)
            .style('font-size', '9px').style('fill', '#999')
            .text(`β=${pt.beta}`);
        }
      });

      // Current point label
      if (currentOp.R > 0) {
        g1.append('circle')
          .attr('cx', xScale(currentOp.D)).attr('cy', yScale(currentOp.R))
          .attr('r', 8)
          .style('fill', '#534AB7').style('stroke', '#fff').style('stroke-width', 2);
        g1.append('text')
          .attr('x', xScale(currentOp.D)).attr('y', yScale(currentOp.R) - 14)
          .attr('text-anchor', 'middle')
          .style('font-size', '10px').style('fill', '#534AB7').style('font-weight', '600')
          .text(`(${currentOp.D.toFixed(2)}, ${currentOp.R.toFixed(2)})`);
      }

      // Axes
      g1.append('g').attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(xScale).ticks(5))
        .selectAll('text').style('fill', 'var(--color-text-secondary, #666)');
      g1.append('g').call(d3.axisLeft(yScale).ticks(5))
        .selectAll('text').style('fill', 'var(--color-text-secondary, #666)');
      g1.selectAll('.domain, .tick line').style('stroke', 'var(--color-text-secondary, #999)');

      g1.append('text').attr('x', w1 / 2).attr('y', h + 40).attr('text-anchor', 'middle')
        .style('font-size', '12px').style('fill', 'var(--color-text-secondary, #666)')
        .text('Distortion D (reconstruction error)');
      g1.append('text').attr('transform', 'rotate(-90)').attr('x', -h / 2).attr('y', -42)
        .attr('text-anchor', 'middle')
        .style('font-size', '12px').style('fill', 'var(--color-text-secondary, #666)')
        .text('Rate R (KL divergence, bits)');
      g1.append('text').attr('x', w1 / 2).attr('y', -12).attr('text-anchor', 'middle')
        .style('font-size', '12px').style('fill', 'var(--color-text-secondary, #666)').style('font-weight', '600')
        .text('Gaussian R(D) with VAE Operating Points');

      // --- Right: Rate/Distortion vs β ---
      const rOffset = isSmall ? 0 : leftW;
      const rYOffset = isSmall ? HEIGHT + 10 : 0;
      const w2 = (isSmall ? containerWidth : rightW) - MARGIN.left - MARGIN.right;
      const g2 = svg.append('g')
        .attr('transform', `translate(${rOffset + MARGIN.left},${rYOffset + MARGIN.top})`);

      const xBeta = d3.scaleLog().domain([0.1, 10]).range([0, w2]);
      const maxVal = Math.max(
        ...tradeoffCurve.filter(p => p.beta >= 0.1 && p.beta <= 10).map(p => Math.max(p.rate, p.distortion))
      ) * 1.1;
      const yVal = d3.scaleLinear().domain([0, maxVal]).range([h, 0]);

      // Rate curve
      const rateLine = d3.line<typeof tradeoffCurve[0]>()
        .x(d => xBeta(d.beta)).y(d => yVal(d.rate));
      const distLine = d3.line<typeof tradeoffCurve[0]>()
        .x(d => xBeta(d.beta)).y(d => yVal(d.distortion));

      const filtered = tradeoffCurve.filter(p => p.beta >= 0.1 && p.beta <= 10);

      g2.append('path')
        .datum(filtered)
        .attr('d', rateLine)
        .style('fill', 'none').style('stroke', '#534AB7').style('stroke-width', 2);
      g2.append('path')
        .datum(filtered)
        .attr('d', distLine)
        .style('fill', 'none').style('stroke', '#D97706').style('stroke-width', 2);

      // β = 1 dashed line
      g2.append('line')
        .attr('x1', xBeta(1)).attr('y1', 0)
        .attr('x2', xBeta(1)).attr('y2', h)
        .style('stroke', '#999').style('stroke-dasharray', '4,3').style('stroke-width', 1.5);
      g2.append('text')
        .attr('x', xBeta(1)).attr('y', -3)
        .attr('text-anchor', 'middle')
        .style('font-size', '10px').style('fill', '#999')
        .text('β = 1 (VAE)');

      // Region labels
      g2.append('text')
        .attr('x', xBeta(0.3)).attr('y', 14)
        .attr('text-anchor', 'middle')
        .style('font-size', '9px').style('fill', '#D97706').style('font-weight', '600')
        .text('Reconstruction');
      g2.append('text')
        .attr('x', xBeta(0.3)).attr('y', 25)
        .attr('text-anchor', 'middle')
        .style('font-size', '9px').style('fill', '#D97706')
        .text('priority');
      g2.append('text')
        .attr('x', xBeta(4)).attr('y', 14)
        .attr('text-anchor', 'middle')
        .style('font-size', '9px').style('fill', '#534AB7').style('font-weight', '600')
        .text('Compression');
      g2.append('text')
        .attr('x', xBeta(4)).attr('y', 25)
        .attr('text-anchor', 'middle')
        .style('font-size', '9px').style('fill', '#534AB7')
        .text('priority');

      // Current β marker
      if (beta >= 0.1 && beta <= 10) {
        const curRate = rateDistortionGaussian(sigma2, currentOp.D);
        g2.append('circle')
          .attr('cx', xBeta(beta)).attr('cy', yVal(curRate))
          .attr('r', 6).style('fill', '#534AB7').style('stroke', '#fff').style('stroke-width', 2);
        g2.append('circle')
          .attr('cx', xBeta(beta)).attr('cy', yVal(currentOp.D))
          .attr('r', 6).style('fill', '#D97706').style('stroke', '#fff').style('stroke-width', 2);
      }

      // Legend
      g2.append('line')
        .attr('x1', w2 - 100).attr('y1', h - 30)
        .attr('x2', w2 - 80).attr('y2', h - 30)
        .style('stroke', '#534AB7').style('stroke-width', 2);
      g2.append('text')
        .attr('x', w2 - 75).attr('y', h - 26)
        .style('font-size', '10px').style('fill', '#534AB7').text('Rate (KL)');
      g2.append('line')
        .attr('x1', w2 - 100).attr('y1', h - 14)
        .attr('x2', w2 - 80).attr('y2', h - 14)
        .style('stroke', '#D97706').style('stroke-width', 2);
      g2.append('text')
        .attr('x', w2 - 75).attr('y', h - 10)
        .style('font-size', '10px').style('fill', '#D97706').text('Distortion');

      // Axes
      g2.append('g').attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(xBeta).ticks(5, '.1f'))
        .selectAll('text').style('fill', 'var(--color-text-secondary, #666)');
      g2.append('g').call(d3.axisLeft(yVal).ticks(5))
        .selectAll('text').style('fill', 'var(--color-text-secondary, #666)');
      g2.selectAll('.domain, .tick line').style('stroke', 'var(--color-text-secondary, #999)');

      g2.append('text').attr('x', w2 / 2).attr('y', h + 40).attr('text-anchor', 'middle')
        .style('font-size', '12px').style('fill', 'var(--color-text-secondary, #666)')
        .text('β (rate-distortion trade-off)');
      g2.append('text').attr('x', w2 / 2).attr('y', -12).attr('text-anchor', 'middle')
        .style('font-size', '12px').style('fill', 'var(--color-text-secondary, #666)').style('font-weight', '600')
        .text('Rate & Distortion vs β');
    },
    [containerWidth, rdCurve, betaPoints, tradeoffCurve, beta, sigma2, currentOp, isSmall, leftW, rightW]
  );

  const handleBeta = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setBeta(parseFloat(e.target.value));
  }, []);

  return (
    <div ref={containerRef} className="my-6 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 p-4">
      <div className="mb-3 flex flex-wrap items-center gap-4">
        <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
          β:
          <input
            type="range"
            min={0.1}
            max={10}
            step={0.05}
            value={beta}
            onChange={handleBeta}
            className="ml-2 w-32 align-middle"
          />
          <span className="ml-1 text-xs font-mono">{beta.toFixed(2)}</span>
        </label>
        <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
          σ²:
          <select
            value={sigma2}
            onChange={e => setSigma2(Number(e.target.value))}
            className="ml-2 rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-2 py-1 text-sm text-gray-900 dark:text-gray-100"
          >
            {SIGMA2_OPTIONS.map(v => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
        </label>
      </div>
      <svg ref={svgRef} width={containerWidth} height={svgHeight} />
    </div>
  );
}
