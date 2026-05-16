import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  crossFitDmlPlr,
  gaussianRng,
  mulberry32,
  paletteSemi,
  ROBINSON_NOISE_SD,
  ROBINSON_THETA_0,
  sampleRobinson,
  type RobinsonNuisanceFitter,
} from './shared/semiparametric-inference';

// =============================================================================
// NuisanceRateThreshold — §8
// Coverage of nominal-95% Wald CI vs the nuisance error-rate exponent α.
// Nuisance = oracle + N(0, n^{-2α}) noise. Sweep α; vertical line at α=1/4.
// Bottom: |bias(θ̂)| vs α on log scale.
// =============================================================================

const HEIGHT_TOP = 280;
const HEIGHT_BOT = 200;
const SM_BREAKPOINT = 640;
const N = 1000;
const ALPHA_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50];

function noisyOracleFitter(
  gOracle: Float64Array,
  mOracle: Float64Array,
  fullX: Float64Array,
  fullN: number,
  p: number,
  alpha: number,
  rng: () => number,
): RobinsonNuisanceFitter {
  const noiseSd = Math.pow(fullN, -alpha);
  const key = (X: Float64Array, i: number) => {
    let k = '';
    for (let j = 0; j < p; j++) k += X[i * p + j].toString() + ':';
    return k;
  };
  const map = new Map<string, number>();
  for (let i = 0; i < fullN; i++) map.set(key(fullX, i), i);
  const gauss = gaussianRng(rng);
  return (_Xtr, _Dtr, _Ytr, _ntr, Xte, nte) => {
    const gPred = new Float64Array(nte);
    const mPred = new Float64Array(nte);
    for (let i = 0; i < nte; i++) {
      const k = key(Xte, i);
      const idx = map.get(k) ?? 0;
      gPred[i] = gOracle[idx] + noiseSd * gauss();
      mPred[i] = mOracle[idx] + noiseSd * gauss();
    }
    return { gPred, mPred };
  };
}

export default function NuisanceRateThreshold() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [displayB, setDisplayB] = useState(100);
  const [committedB, setCommittedB] = useState(100);

  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;

  const data = useMemo(() => {
    const coverage: number[] = [];
    const bias: number[] = [];
    for (const alpha of ALPHA_GRID) {
      let inCI = 0;
      const thetas: number[] = [];
      for (let b = 0; b < committedB; b++) {
        const rng = mulberry32(20260515 + b * 31 + Math.round(alpha * 1000) * 7);
        const sample = sampleRobinson(N, ROBINSON_THETA_0, rng);
        const fitter = noisyOracleFitter(
          sample.gOracle, sample.mOracle, sample.X, sample.n, 5, alpha, rng,
        );
        const r = crossFitDmlPlr(sample.X, sample.D, sample.Y, 5, fitter, rng);
        thetas.push(r.theta);
        if (Math.abs(r.theta - ROBINSON_THETA_0) <= 1.96 * r.se) inCI += 1;
      }
      coverage.push(inCI / committedB);
      const mu = thetas.reduce((a, b) => a + b, 0) / thetas.length;
      bias.push(Math.abs(mu - ROBINSON_THETA_0));
    }
    return { coverage, bias };
  }, [committedB]);

  const refTop = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const margin = { top: 24, right: 14, bottom: 36, left: 56 };
      const W = (isMobile ? w : Math.min(w, 720)) - margin.left - margin.right;
      const H = HEIGHT_TOP - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${(isMobile ? w : Math.min(w, 720))} ${HEIGHT_TOP}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);
      const xScale = d3.scaleLinear().domain([0.05, 0.55]).range([0, W]);
      const yScale = d3.scaleLinear().domain([0, 1.0]).range([H, 0]);
      g.append('g').attr('transform', `translate(0, ${H})`).call(d3.axisBottom(xScale).ticks(6));
      g.append('g').call(d3.axisLeft(yScale).ticks(5, '.0%'));
      // Nominal 95% line.
      g.append('line').attr('x1', 0).attr('x2', W)
        .attr('y1', yScale(0.95)).attr('y2', yScale(0.95))
        .style('stroke', paletteSemi.theoryLine).style('stroke-width', 1)
        .style('stroke-dasharray', '4,3');
      g.append('text').attr('x', W).attr('y', yScale(0.95) - 4).attr('text-anchor', 'end')
        .style('fill', paletteSemi.theoryLine)
        .style('font-family', 'var(--font-mono)').style('font-size', 11)
        .text('nominal 95%');
      // Threshold line α = 0.25.
      g.append('line').attr('x1', xScale(0.25)).attr('x2', xScale(0.25))
        .attr('y1', 0).attr('y2', H)
        .style('stroke', paletteSemi.bias).style('stroke-width', 1.5)
        .style('stroke-dasharray', '5,4');
      g.append('text').attr('x', xScale(0.25) + 6).attr('y', 16)
        .style('fill', paletteSemi.bias)
        .style('font-family', 'var(--font-mono)').style('font-size', 11)
        .text('α = 1/4 threshold');
      // Coverage curve.
      const lineGen = d3.line<number>().x((_d, i) => xScale(ALPHA_GRID[i])).y((d) => yScale(d));
      g.append('path').datum(data.coverage).attr('d', lineGen)
        .style('fill', 'none').style('stroke', paletteSemi.oneStep).style('stroke-width', 2);
      g.selectAll(null).data(data.coverage).join('circle')
        .attr('cx', (_d, i) => xScale(ALPHA_GRID[i]))
        .attr('cy', (d) => yScale(d))
        .attr('r', 3).style('fill', paletteSemi.oneStep);
      g.append('text').attr('x', 0).attr('y', -10)
        .style('fill', 'var(--color-text)')
        .style('font-family', 'var(--font-mono)').style('font-size', 12)
        .text('Empirical coverage of nominal-95% Wald CI vs nuisance rate exponent α');
      g.append('text').attr('x', W / 2).attr('y', H + 30).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-family', 'var(--font-mono)').style('font-size', 11)
        .text('α (so that ‖η̂ − η‖ ≈ n^{−α})');
    },
    [data, containerWidth, isMobile],
  );

  const refBot = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const margin = { top: 16, right: 14, bottom: 36, left: 56 };
      const W = (isMobile ? w : Math.min(w, 720)) - margin.left - margin.right;
      const H = HEIGHT_BOT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${(isMobile ? w : Math.min(w, 720))} ${HEIGHT_BOT}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);
      const xScale = d3.scaleLinear().domain([0.05, 0.55]).range([0, W]);
      const yMin = Math.max(1e-4, Math.min(...data.bias.filter((v) => v > 0)));
      const yMax = Math.max(1e-2, Math.max(...data.bias));
      const yScale = d3.scaleLog().domain([yMin * 0.5, yMax * 2]).range([H, 0]);
      g.append('g').attr('transform', `translate(0, ${H})`).call(d3.axisBottom(xScale).ticks(6));
      g.append('g').call(d3.axisLeft(yScale).ticks(4, '.0e'));
      g.append('line').attr('x1', xScale(0.25)).attr('x2', xScale(0.25))
        .attr('y1', 0).attr('y2', H)
        .style('stroke', paletteSemi.bias).style('stroke-width', 1.5)
        .style('stroke-dasharray', '5,4');
      const lineGen = d3.line<number>().x((_d, i) => xScale(ALPHA_GRID[i])).y((d) => yScale(Math.max(d, yMin * 0.6)));
      g.append('path').datum(data.bias).attr('d', lineGen)
        .style('fill', 'none').style('stroke', paletteSemi.bias).style('stroke-width', 2);
      g.selectAll(null).data(data.bias).join('circle')
        .attr('cx', (_d, i) => xScale(ALPHA_GRID[i]))
        .attr('cy', (d) => yScale(Math.max(d, yMin * 0.6)))
        .attr('r', 3).style('fill', paletteSemi.bias);
      g.append('text').attr('x', 0).attr('y', -2)
        .style('fill', 'var(--color-text)')
        .style('font-family', 'var(--font-mono)').style('font-size', 12)
        .text('|empirical bias of θ̂| (log scale)');
      g.append('text').attr('x', W / 2).attr('y', H + 30).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-family', 'var(--font-mono)').style('font-size', 11)
        .text('α');
    },
    [data, containerWidth, isMobile],
  );

  return (
    <div ref={containerRef} className="viz-container" style={{ width: '100%' }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginBottom: 12 }}>
        <label style={{ fontFamily: 'var(--font-mono)', fontSize: 13 }}>
          MC replications B per α: <strong>{displayB}</strong>
          <input
            type="range"
            min={50}
            max={200}
            step={50}
            value={displayB}
            onChange={(e) => setDisplayB(+e.target.value)}
            onMouseUp={() => setCommittedB(displayB)}
            onTouchEnd={() => setCommittedB(displayB)}
            onKeyUp={() => setCommittedB(displayB)}
            aria-label="Monte Carlo replications per α"
            style={{ width: '100%', marginTop: 4 }}
          />
        </label>
      </div>
      <svg ref={refTop} style={{ width: '100%', height: HEIGHT_TOP, display: 'block' }} />
      <svg ref={refBot} style={{ width: '100%', height: HEIGHT_BOT, display: 'block' }} />
    </div>
  );
}
