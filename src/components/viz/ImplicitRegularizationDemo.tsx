import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  fillIsotropicGaussian,
  gaussianRng,
  gdTrajectory,
  matVec,
  mulberry32,
  sampleSphereBetaStar,
} from './shared/double-descent';

// =============================================================================
// ImplicitRegularizationDemo — §9.3 (Figure 10)
//
// Four-panel GD trajectory: (a) training loss; (b) ‖β_t‖² toward ‖β̂†‖²;
// (c) distance to β̂†; (d) test risk. Iteration grid is log-spaced; the
// trajectory uses the SVD-basis closed form, not actual GD iteration.
// Static fallback: 10_gd_trajectory_4panel.png
// =============================================================================

const HEIGHT = 460;
const ITER_GRID = (() => {
  const result: number[] = [0];
  for (const exp of [0, 1, 2, 3]) {
    for (let mant = 1; mant < 10; mant++) {
      const v = Math.round(mant * Math.pow(10, exp));
      if (v > 0 && v <= 10000) result.push(v);
    }
  }
  result.push(10000);
  return Array.from(new Set(result)).sort((a, b) => a - b);
})();

export default function ImplicitRegularizationDemo() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [gammaCommitted, setGammaCommitted] = useState(4);
  const [gammaDisplay, setGammaDisplay] = useState(4);
  const [snrLog, setSnrLog] = useState(0);
  const [snrLogCommitted, setSnrLogCommitted] = useState(0);
  const [etaFrac, setEtaFrac] = useState(0.9);

  const trajectory = useMemo(() => {
    const gamma = gammaCommitted;
    const n = 25;
    const p = Math.max(n + 1, Math.round(gamma * n));
    const sigma = 1;
    const r = Math.sqrt(Math.pow(10, snrLogCommitted));
    const rng = mulberry32(2024);
    const gauss = gaussianRng(rng);
    const X = new Float64Array(n * p);
    fillIsotropicGaussian(X, n, p, gauss);
    const betaStar = sampleSphereBetaStar(p, r, gauss);
    const y = new Float64Array(n);
    matVec(X, betaStar, n, p, y);
    for (let i = 0; i < n; i++) y[i] += sigma * gauss();
    const iters = new Int32Array(ITER_GRID);
    const traj = gdTrajectory({ X, y, n, p, betaStar, iters, etaFraction: etaFrac });
    return traj;
  }, [gammaCommitted, snrLogCommitted, etaFrac]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const margin = { top: 28, right: 16, bottom: 36, left: 56 };
      const gap = 28;
      const panelW = (w - margin.left - margin.right - gap) / 2;
      const panelH = (HEIGHT - margin.top - margin.bottom - gap) / 2;
      svg.selectAll('*').remove();
      if (panelW <= 0) return;
      svg.attr('viewBox', `0 0 ${w} ${HEIGHT}`);

      const drawPanel = (
        col: number, row: number, title: string, yLabel: string,
        ys: Float64Array, logY: boolean, refLine?: number,
      ) => {
        const offX = margin.left + col * (panelW + gap);
        const offY = margin.top + row * (panelH + gap);
        const g = svg.append('g').attr('transform', `translate(${offX},${offY})`);
        const valid = Array.from(ys).filter((v) => Number.isFinite(v) && (!logY || v > 0));
        const yMin = logY ? Math.max(1e-30, Math.min(...valid, refLine ?? Infinity)) : Math.min(...valid, refLine ?? Infinity);
        const yMax = Math.max(...valid, refLine ?? -Infinity);
        const xMin = Math.max(1, trajectory.iters[0]);
        const xMax = trajectory.iters[trajectory.iters.length - 1];
        const xScale = d3.scaleLog().domain([xMin, xMax]).range([0, panelW]);
        const yScale = logY
          ? d3.scaleLog().domain([yMin * 0.7, yMax * 1.4]).range([panelH, 0])
          : d3.scaleLinear().domain([yMin - (yMax - yMin) * 0.1, yMax + (yMax - yMin) * 0.1]).range([panelH, 0]);
        g.append('g').attr('transform', `translate(0,${panelH})`).call(d3.axisBottom(xScale).ticks(4, '~g'))
          .selectAll('text').style('fill', 'var(--color-text)').style('font-size', '10px');
        g.append('g').call(d3.axisLeft(yScale).ticks(4, '~g'))
          .selectAll('text').style('fill', 'var(--color-text)').style('font-size', '10px');
        g.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');
        g.append('text').attr('x', panelW / 2).attr('y', -8).attr('text-anchor', 'middle')
          .style('fill', 'var(--color-text)').style('font-size', '11px').style('font-weight', '600').text(title);
        g.append('text').attr('x', -44).attr('y', panelH / 2)
          .attr('transform', `rotate(-90, -44, ${panelH / 2})`).attr('text-anchor', 'middle')
          .style('fill', 'var(--color-text)').style('font-size', '10px').text(yLabel);

        if (refLine !== undefined && refLine > 0) {
          g.append('line').attr('x1', 0).attr('x2', panelW)
            .attr('y1', yScale(refLine)).attr('y2', yScale(refLine))
            .style('stroke', '#9B1D20').attr('stroke-dasharray', '5 3').attr('opacity', 0.7);
        }

        const data: [number, number][] = Array.from(trajectory.iters).map((t, i) => [Math.max(1, t), ys[i]]);
        const lineGen = d3.line<[number, number]>().x((d) => xScale(d[0]))
          .y((d) => yScale(Math.max(logY ? yMin * 0.7 : -Infinity, d[1])))
          .defined((d) => Number.isFinite(d[1]) && (!logY || d[1] > 0));
        g.append('path').datum(data).attr('d', lineGen).attr('fill', 'none')
          .style('stroke', 'var(--color-accent)').attr('stroke-width', 2);
      };

      drawPanel(0, 0, '(a) training loss', 'loss', trajectory.trainLoss, true);
      drawPanel(1, 0, '(b) ‖β_t‖²', 'norm²', trajectory.normBeta, false, trajectory.betaMinNormNormSq);
      drawPanel(0, 1, '(c) ‖β_t − β̂†‖', 'distance', trajectory.distToMinNorm, true);
      drawPanel(1, 1, '(d) test risk', 'risk', trajectory.testRisk, false);

      // x-axis label
      svg.append('text').attr('x', (margin.left + (panelW * 2 + gap)) / 2 + margin.left / 2)
        .attr('y', HEIGHT - 6).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)').style('font-size', '11px').text('iteration t (log)');
    },
    [containerWidth, trajectory],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: 'var(--font-sans)' }}>
      <svg ref={svgRef} width="100%" height={HEIGHT} role="img"
        aria-label="Four-panel gradient descent trajectory: train loss, coefficient norm, distance to β̂†, test risk." />
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', marginTop: '0.75rem',
        fontSize: '13px', color: 'var(--color-text)' }}>
        <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', minWidth: 180 }}>
          <span>γ = p/n: <strong>{gammaDisplay.toFixed(1)}</strong></span>
          <input type="range" min={1.5} max={10} step={0.5} value={gammaDisplay}
            onChange={(e) => setGammaDisplay(parseFloat(e.target.value))}
            onMouseUp={() => setGammaCommitted(gammaDisplay)}
            onTouchEnd={() => setGammaCommitted(gammaDisplay)}
            onKeyUp={() => setGammaCommitted(gammaDisplay)} aria-label="Aspect ratio gamma" />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', minWidth: 180 }}>
          <span>SNR: <strong>{Math.pow(10, snrLog).toFixed(2)}</strong></span>
          <input type="range" min={-1} max={1} step={0.05} value={snrLog}
            onChange={(e) => setSnrLog(parseFloat(e.target.value))}
            onMouseUp={() => setSnrLogCommitted(snrLog)} onTouchEnd={() => setSnrLogCommitted(snrLog)}
            onKeyUp={() => setSnrLogCommitted(snrLog)} aria-label="SNR" />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', minWidth: 180 }}>
          <span>η fraction of 1/s²_max: <strong>{etaFrac.toFixed(2)}</strong></span>
          <input type="range" min={0.05} max={1.99} step={0.05} value={etaFrac}
            onChange={(e) => setEtaFrac(parseFloat(e.target.value))}
            aria-label="Learning rate as fraction of 1/s_max²" />
        </label>
      </div>
    </div>
  );
}
