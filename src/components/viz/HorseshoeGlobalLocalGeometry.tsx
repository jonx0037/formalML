import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  bayesianLassoShrinkageMarginal,
  horseshoeMarginalBound,
  horseshoeShrinkageMarginal,
} from './shared/bayesian-ml';

// =============================================================================
// HorseshoeGlobalLocalGeometry — §5 horseshoe vs Bayesian LASSO geometric
// content. Two panels:
//   (a) prior tail densities on log scale for ridge / Bayesian LASSO / horseshoe
//   (b) posterior shrinkage profile E[κ | y] for the same three priors
//
// Sliders: τ ∈ [0.2, 3], comparison toggle (LASSO vs horseshoe vs regularized).
// =============================================================================

const PANEL_HEIGHT = 320;
const MARGIN = { top: 22, right: 16, bottom: 50, left: 56 };
const N_GRID = 200;
const N_Y = 80;

const RIDGE_COLOR = '#7f7f7f';
const LASSO_COLOR = '#7b3c10';
const HS_COLOR = '#1f4e79';

function gaussianPdf(x: number, sigma: number): number {
  return Math.exp(-0.5 * (x / sigma) ** 2) / (sigma * Math.sqrt(2 * Math.PI));
}
function laplacePdf(x: number, scale: number): number {
  return Math.exp(-Math.abs(x) / scale) / (2 * scale);
}

export default function HorseshoeGlobalLocalGeometry() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [tau, setTau] = useState(1.0);

  const data = useMemo(() => {
    const betaGrid = new Array<number>(N_GRID);
    const ridge: number[] = new Array(N_GRID);
    const lasso: number[] = new Array(N_GRID);
    const horseshoeLow: number[] = new Array(N_GRID);
    const horseshoeUp: number[] = new Array(N_GRID);
    for (let i = 0; i < N_GRID; i++) {
      const b = -8 + (16 * i) / (N_GRID - 1);
      betaGrid[i] = b;
      ridge[i] = gaussianPdf(b, tau);
      lasso[i] = laplacePdf(b, tau / Math.sqrt(2));
      horseshoeLow[i] = horseshoeMarginalBound(b, tau, true);
      horseshoeUp[i] = horseshoeMarginalBound(b, tau, false);
    }
    const yGrid = new Array<number>(N_Y);
    const shrinkRidge: number[] = new Array(N_Y);
    const shrinkLasso: number[] = new Array(N_Y);
    const shrinkHs: number[] = new Array(N_Y);
    for (let i = 0; i < N_Y; i++) {
      const y = 0.05 + (8 * i) / (N_Y - 1);
      yGrid[i] = y;
      shrinkRidge[i] = 1 / (1 + tau * tau);
      shrinkLasso[i] = bayesianLassoShrinkageMarginal(y, tau);
      shrinkHs[i] = horseshoeShrinkageMarginal(y, tau);
    }
    return {
      betaGrid,
      ridge,
      lasso,
      horseshoeLow,
      horseshoeUp,
      yGrid,
      shrinkRidge,
      shrinkLasso,
      shrinkHs,
    };
  }, [tau]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const w = width || 760;
      const h = PANEL_HEIGHT;
      svg.attr('width', w).attr('height', h);
      const innerW = (w - MARGIN.left - MARGIN.right - 40) / 2;
      const innerH = h - MARGIN.top - MARGIN.bottom;

      // Panel (a) — prior tail densities, log-scale y
      const gA = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);
      const xA = d3.scaleLinear().domain([-8, 8]).range([0, innerW]);
      const yA = d3.scaleLog().domain([1e-6, 1]).range([innerH, 0]).clamp(true);
      gA.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xA).ticks(5));
      gA.append('g').call(d3.axisLeft(yA).ticks(5, '~g'));
      gA.append('text')
        .attr('x', innerW / 2)
        .attr('y', innerH + 36)
        .attr('text-anchor', 'middle')
        .style('font-size', '11px')
        .text('β');
      gA.append('text')
        .attr('x', -innerH / 2)
        .attr('y', -42)
        .attr('text-anchor', 'middle')
        .attr('transform', 'rotate(-90)')
        .style('font-size', '11px')
        .text('p(β) — log scale');
      gA.append('text')
        .attr('x', 0)
        .attr('y', -8)
        .style('font-size', '11px')
        .style('font-weight', '600')
        .text('(a) prior tail decay');

      const drawTrace = (
        parent: d3.Selection<SVGGElement, unknown, null, undefined>,
        data: number[],
        color: string,
        x: d3.ScaleLinear<number, number>,
        y: d3.ScaleContinuousNumeric<number, number>,
        gridArr: number[],
        opts: { width?: number; dash?: string } = {},
      ) => {
        const lineGen = d3
          .line<number>()
          .x((_, i) => x(gridArr[i]))
          .y((d) => y(Math.max(d, 1e-12)))
          .curve(d3.curveMonotoneX);
        parent
          .append('path')
          .datum(data)
          .attr('fill', 'none')
          .attr('stroke', color)
          .attr('stroke-width', opts.width ?? 2)
          .attr('stroke-dasharray', opts.dash ?? '')
          .attr('d', lineGen);
      };

      drawTrace(gA, data.ridge, RIDGE_COLOR, xA, yA, data.betaGrid);
      drawTrace(gA, data.lasso, LASSO_COLOR, xA, yA, data.betaGrid);

      // Horseshoe band (between CPS lower and upper bounds)
      const areaGen = d3
        .area<number>()
        .x((_, i) => xA(data.betaGrid[i]))
        .y0((_, i) => yA(Math.max(data.horseshoeLow[i], 1e-12)))
        .y1((_, i) => yA(Math.max(data.horseshoeUp[i], 1e-12)));
      gA.append('path')
        .datum(data.horseshoeUp)
        .attr('fill', HS_COLOR)
        .attr('opacity', 0.18)
        .attr('d', areaGen);
      drawTrace(
        gA,
        data.horseshoeUp.map((u, i) => Math.sqrt(u * data.horseshoeLow[i])),
        HS_COLOR,
        xA,
        yA,
        data.betaGrid,
        { width: 2.4 },
      );

      // Panel (b) — posterior shrinkage profile
      const gB = svg
        .append('g')
        .attr('transform', `translate(${MARGIN.left + innerW + 50},${MARGIN.top})`);
      const xB = d3.scaleLinear().domain([0, 8]).range([0, innerW]);
      const yB = d3.scaleLinear().domain([0, 1.05]).range([innerH, 0]).clamp(true);
      gB.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xB).ticks(5));
      gB.append('g').call(d3.axisLeft(yB).ticks(5));
      gB.append('text')
        .attr('x', innerW / 2)
        .attr('y', innerH + 36)
        .attr('text-anchor', 'middle')
        .style('font-size', '11px')
        .text('|y_j|');
      gB.append('text')
        .attr('x', -innerH / 2)
        .attr('y', -42)
        .attr('text-anchor', 'middle')
        .attr('transform', 'rotate(-90)')
        .style('font-size', '11px')
        .text('E[κ | y]');
      gB.append('text')
        .attr('x', 0)
        .attr('y', -8)
        .style('font-size', '11px')
        .style('font-weight', '600')
        .text('(b) posterior shrinkage');

      drawTrace(gB, data.shrinkRidge, RIDGE_COLOR, xB, yB, data.yGrid);
      drawTrace(gB, data.shrinkLasso, LASSO_COLOR, xB, yB, data.yGrid);
      drawTrace(gB, data.shrinkHs, HS_COLOR, xB, yB, data.yGrid, { width: 2.6 });

      // Panel (b) legend
      const legend = gB.append('g').attr('transform', `translate(${innerW - 195}, 4)`);
      const legendItems = [
        { color: RIDGE_COLOR, label: 'Ridge' },
        { color: LASSO_COLOR, label: 'Bayesian LASSO' },
        { color: HS_COLOR, label: 'Horseshoe' },
      ];
      legendItems.forEach((it, i) => {
        const row = legend.append('g').attr('transform', `translate(0, ${i * 16})`);
        row
          .append('line')
          .attr('x1', 0)
          .attr('x2', 18)
          .attr('y1', 5)
          .attr('y2', 5)
          .attr('stroke', it.color)
          .attr('stroke-width', 2);
        row
          .append('text')
          .attr('x', 22)
          .attr('y', 9)
          .style('font-size', '10.5px')
          .text(it.label);
      });
    },
    [data, width],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', maxWidth: 760 }}>
      <div style={{ display: 'flex', gap: 18, marginBottom: 8, fontSize: 13, alignItems: 'center' }}>
        <label>
          τ: {tau.toFixed(2)}{' '}
          <input
            type="range"
            min={0.2}
            max={3.0}
            step={0.05}
            value={tau}
            onChange={(e) => setTau(Number(e.target.value))}
            style={{ width: 160 }}
          />
        </label>
      </div>
      <svg ref={svgRef} role="img" aria-label="Horseshoe vs Bayesian LASSO geometry" />
    </div>
  );
}
