import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  doubleDescentUqSweep,
  mulberry32,
} from './shared/uncertainty-quantification';

// =============================================================================
// OverparamRegimeUQ — §10
//
// Four-panel sweep of test MSE, posterior-predictive variance, conformal
// half-width, and marginal conformal coverage vs p/n. Slider for ridge λ
// (commit-on-release); the sweep is fixed at n_tr = 60, n_cal = 30, n_test = 100,
// B = 8 replicates per p value for in-browser speed.
// Static fallback: public/images/topics/uncertainty-quantification/fig_10_uq_shadows.png
// =============================================================================

const HEIGHT = 560;
const N_TR = 60;
const N_CAL = 30;
const N_TEST = 100;
const ALPHA = 0.10;
const SNR = 5;
const SIGMA = 1;
const P_GRID = [5, 15, 25, 35, 45, 55, 60, 65, 75, 90, 110, 140, 175, 200];

export default function OverparamRegimeUQ() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [logLambdaDisplay, setLogLambdaDisplay] = useState(-2);
  const [logLambdaCommitted, setLogLambdaCommitted] = useState(-2);

  const sweep = useMemo(() => {
    const lambda = Math.pow(10, logLambdaCommitted);
    const rng = mulberry32(20260514 + 30 + Math.round(logLambdaCommitted * 100));
    return doubleDescentUqSweep(P_GRID, N_TR, N_CAL, N_TEST, SNR, SIGMA, lambda, 8, ALPHA, rng);
  }, [logLambdaCommitted]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const margin = { top: 32, right: 32, bottom: 56, left: 64 };
      const panelW = (w - margin.left - margin.right - 24) / 2;
      const panelH = (HEIGHT - margin.top - margin.bottom - 24) / 2;
      svg.selectAll('*').remove();
      if (panelW <= 0) return;
      svg.attr('viewBox', `0 0 ${w} ${HEIGHT}`);

      const pOverN = sweep.map((s) => s.pOverN);
      const xScale = d3.scaleLog().domain([Math.min(...pOverN), Math.max(...pOverN)]).range([0, panelW]);

      const drawPanel = (
        col: number, row: number, title: string, values: number[],
        ses: number[], color: string, yLog: boolean, targetLine?: number,
      ) => {
        const g = svg.append('g').attr('transform',
          `translate(${margin.left + col * (panelW + 24)},${margin.top + row * (panelH + 24)})`);
        const finite = values.filter((v) => Number.isFinite(v) && v > 0);
        const yDomain = yLog
          ? [Math.max(1e-3, Math.min(...finite) * 0.6), Math.max(...finite) * 1.4]
          : [Math.min(...values) - 0.05, Math.max(...values) + 0.05];
        const yScale = (yLog ? d3.scaleLog() : d3.scaleLinear())
          .domain(yDomain).range([panelH, 0]).nice();
        const lineGen = d3.line<number>()
          .x((_, i) => xScale(pOverN[i]))
          .y((d) => yScale(Math.max(yLog ? 1e-3 : -Infinity, d)));
        // Error bars.
        for (let i = 0; i < values.length; i++) {
          if (!Number.isFinite(values[i])) continue;
          g.append('line').attr('x1', xScale(pOverN[i])).attr('x2', xScale(pOverN[i]))
            .attr('y1', yScale(Math.max(yLog ? 1e-3 : -Infinity, values[i] - ses[i])))
            .attr('y2', yScale(Math.max(yLog ? 1e-3 : -Infinity, values[i] + ses[i])))
            .style('stroke', '#f97316').attr('stroke-width', 1);
        }
        g.append('path').datum(values).attr('d', lineGen).attr('fill', 'none')
          .style('stroke', color).attr('stroke-width', 1.8);
        g.selectAll('.dot').data(values).enter().append('circle').attr('class', 'dot')
          .attr('cx', (_, i) => xScale(pOverN[i]))
          .attr('cy', (d) => yScale(Math.max(yLog ? 1e-3 : -Infinity, d as number)))
          .attr('r', 3).style('fill', color);
        // Threshold marker at p/n = 1.
        g.append('line').attr('x1', xScale(1)).attr('x2', xScale(1))
          .attr('y1', 0).attr('y2', panelH).style('stroke', '#94a3b8')
          .attr('stroke-dasharray', '4 3').attr('stroke-width', 1.2);
        if (targetLine !== undefined) {
          g.append('line').attr('x1', 0).attr('x2', panelW)
            .attr('y1', yScale(targetLine)).attr('y2', yScale(targetLine))
            .style('stroke', '#94a3b8').attr('stroke-dasharray', '4 3').attr('stroke-width', 1.2);
        }
        g.append('g').attr('transform', `translate(0,${panelH})`)
          .call(d3.axisBottom(xScale).ticks(5, '.1f'))
          .selectAll('text').style('fill', 'var(--color-text)');
        g.append('g').call(d3.axisLeft(yScale).ticks(5, yLog ? '.1e' : '.2f'))
          .selectAll('text').style('fill', 'var(--color-text)');
        g.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');
        g.append('text').attr('x', panelW / 2).attr('y', -8).attr('text-anchor', 'middle')
          .style('font-size', '12px').style('fill', 'var(--color-text)').text(title);
        if (row === 1) {
          g.append('text').attr('x', panelW / 2).attr('y', panelH + 36)
            .attr('text-anchor', 'middle').style('font-size', '11px')
            .style('fill', 'var(--color-text)').text('p / n (log)');
        }
      };

      drawPanel(0, 0, '(a) test MSE', sweep.map((s) => s.testMseMean),
        sweep.map((s) => s.testMseSe), '#dc2626', true);
      drawPanel(1, 0, '(b) PP variance', sweep.map((s) => s.postVarMean),
        sweep.map((s) => s.postVarSe), '#8b5cf6', true);
      drawPanel(0, 1, '(c) conformal half-width q̂', sweep.map((s) => s.qHatMean),
        sweep.map((s) => s.qHatSe), '#06b6d4', true);
      drawPanel(1, 1, '(d) marginal conformal coverage',
        sweep.map((s) => s.covMean), sweep.map((s) => s.covSe),
        '#f59e0b', false, 1 - ALPHA);
    },
    [containerWidth, sweep],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: 'var(--font-sans)' }}>
      <svg ref={svgRef} width="100%" height={HEIGHT} role="img"
        aria-label="Four-panel sweep over p/n: test MSE, posterior-predictive variance, conformal half-width, marginal conformal coverage." />
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', marginTop: '0.75rem',
        fontSize: '13px', color: 'var(--color-text)' }}>
        <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', minWidth: 240 }}>
          <span>ridge log₁₀(λ): <strong>{logLambdaDisplay.toFixed(1)}</strong> (λ = {Math.pow(10, logLambdaDisplay).toExponential(1)})</span>
          <input type="range" min={-4} max={1} step={0.5} value={logLambdaDisplay}
            onChange={(e) => setLogLambdaDisplay(parseFloat(e.target.value))}
            onMouseUp={() => setLogLambdaCommitted(logLambdaDisplay)}
            onTouchEnd={() => setLogLambdaCommitted(logLambdaDisplay)}
            onKeyUp={() => setLogLambdaCommitted(logLambdaDisplay)}
            aria-label="Ridge log-lambda" />
        </label>
      </div>
      <p style={{ fontSize: '12px', color: 'var(--color-text-secondary)', marginTop: '0.4rem' }}>
        Test risk (a), posterior-predictive variance (b), and conformal half-width (c) all spike at p/n ≈ 1.
        Conformal marginal coverage (d) is flat at the target 1 − α — coverage is theorem-protected, width
        follows predictor quality. Drag λ to see ridge regularization smooth the spike.
      </p>
    </div>
  );
}
