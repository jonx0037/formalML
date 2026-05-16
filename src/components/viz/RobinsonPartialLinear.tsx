import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  crossFitDmlPlr,
  linearRobinsonNuisanceFitter,
  makeOracleRobinsonNuisanceFitter,
  mulberry32,
  paletteSemi,
  polyDeg3RobinsonNuisanceFitter,
  ROBINSON_NOISE_SD,
  ROBINSON_THETA_0,
  sampleRobinson,
  stddev,
} from './shared/semiparametric-inference';

// =============================================================================
// RobinsonPartialLinear — §9.2
// Robinson partial-linear regression with three nuisance options:
//   (a) linear-controls, (b) polynomial-degree-3, (c) oracle.
// Top: histogram of θ̂ across MC replications, color-coded by nuisance type.
// Bottom: convergence to BKRW bound — empirical SD vs theoretical SD at
// representative n values.
// =============================================================================

const HEIGHT_TOP = 300;
const HEIGHT_BOT = 220;
const SM_BREAKPOINT = 640;
const N = 1000;
const BKRW_BOUND = ROBINSON_NOISE_SD / (ROBINSON_NOISE_SD * Math.sqrt(N));
const N_GRID = [200, 500, 1000, 2000];

type NuisanceChoice = 'linear' | 'poly3' | 'oracle';

export default function RobinsonPartialLinear() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [choice, setChoice] = useState<NuisanceChoice>('poly3');
  const [displayB, setDisplayB] = useState(100);
  const [committedB, setCommittedB] = useState(100);

  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;

  const topData = useMemo(() => {
    const thetas: number[] = [];
    for (let b = 0; b < committedB; b++) {
      const rng = mulberry32(20260515 + b * 13);
      const sample = sampleRobinson(N, ROBINSON_THETA_0, rng);
      let fitter;
      if (choice === 'linear') fitter = linearRobinsonNuisanceFitter;
      else if (choice === 'poly3') fitter = polyDeg3RobinsonNuisanceFitter;
      else fitter = makeOracleRobinsonNuisanceFitter(sample.gOracle, sample.mOracle, sample.X, sample.n, 5);
      const r = crossFitDmlPlr(sample.X, sample.D, sample.Y, 5, fitter, rng);
      thetas.push(r.theta);
    }
    return thetas;
  }, [choice, committedB]);

  const bottomData = useMemo(() => {
    // SD convergence vs n at fixed B=60 for speed.
    const B = 60;
    const result: { n: number; sdLinear: number; sdPoly3: number; sdOracle: number; theoreticalSD: number }[] = [];
    for (const n of N_GRID) {
      const linearThetas: number[] = [];
      const polyThetas: number[] = [];
      const oracleThetas: number[] = [];
      for (let b = 0; b < B; b++) {
        const rng = mulberry32(20260515 + b * 29 + n);
        const sample = sampleRobinson(n, ROBINSON_THETA_0, rng);
        const lr = crossFitDmlPlr(sample.X, sample.D, sample.Y, 5, linearRobinsonNuisanceFitter, rng);
        const pr = crossFitDmlPlr(sample.X, sample.D, sample.Y, 5, polyDeg3RobinsonNuisanceFitter, rng);
        const or = crossFitDmlPlr(
          sample.X, sample.D, sample.Y, 5,
          makeOracleRobinsonNuisanceFitter(sample.gOracle, sample.mOracle, sample.X, sample.n, 5),
          rng,
        );
        linearThetas.push(lr.theta);
        polyThetas.push(pr.theta);
        oracleThetas.push(or.theta);
      }
      result.push({
        n,
        sdLinear: stddev(Float64Array.from(linearThetas)),
        sdPoly3: stddev(Float64Array.from(polyThetas)),
        sdOracle: stddev(Float64Array.from(oracleThetas)),
        theoreticalSD: ROBINSON_NOISE_SD / (ROBINSON_NOISE_SD * Math.sqrt(n)),
      });
    }
    return result;
  }, []);

  const refTop = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const W0 = isMobile ? w : Math.min(w, 720);
      const margin = { top: 24, right: 14, bottom: 36, left: 60 };
      const W = W0 - margin.left - margin.right;
      const H = HEIGHT_TOP - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${W0} ${HEIGHT_TOP}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);
      const lo = Math.min(...topData, 0.85);
      const hi = Math.max(...topData, 1.15);
      const xScale = d3.scaleLinear().domain([lo, hi]).range([0, W]).nice();
      const bins = d3.bin().domain(xScale.domain() as [number, number]).thresholds(28)(topData);
      const yMax = d3.max(bins, (d) => d.length)! * 1.1 + 1;
      const yScale = d3.scaleLinear().domain([0, yMax]).range([H, 0]).nice();
      g.append('g').attr('transform', `translate(0, ${H})`).call(d3.axisBottom(xScale).ticks(5));
      g.append('g').call(d3.axisLeft(yScale).ticks(4));
      const colorMap: Record<NuisanceChoice, string> = {
        linear: paletteSemi.linear, poly3: paletteSemi.dml, oracle: paletteSemi.theoryLine,
      };
      g.selectAll(null).data(bins).join('rect')
        .attr('x', (d) => xScale(d.x0!) + 1)
        .attr('y', (d) => yScale(d.length))
        .attr('width', (d) => Math.max(xScale(d.x1!) - xScale(d.x0!) - 1, 0))
        .attr('height', (d) => H - yScale(d.length))
        .style('fill', colorMap[choice]).style('opacity', 0.75);
      g.append('line').attr('x1', xScale(ROBINSON_THETA_0)).attr('x2', xScale(ROBINSON_THETA_0))
        .attr('y1', 0).attr('y2', H)
        .style('stroke', paletteSemi.theoryLine).style('stroke-width', 1.5)
        .style('stroke-dasharray', '4,3');
      const mu = d3.mean(topData)!;
      const sd = stddev(Float64Array.from(topData));
      g.append('text').attr('x', 0).attr('y', -8)
        .style('fill', 'var(--color-text)')
        .style('font-family', 'var(--font-mono)').style('font-size', 12)
        .text(`${choice} nuisance: mean=${mu.toFixed(4)}, SD=${sd.toFixed(4)} (BKRW ≈ ${BKRW_BOUND.toFixed(4)})`);
      g.append('text').attr('x', xScale(ROBINSON_THETA_0) + 4).attr('y', 14)
        .style('fill', paletteSemi.theoryLine)
        .style('font-family', 'var(--font-mono)').style('font-size', 11)
        .text('θ_0 = 1');
    },
    [topData, choice, containerWidth, isMobile],
  );

  const refBot = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const W0 = isMobile ? w : Math.min(w, 720);
      const margin = { top: 24, right: 14, bottom: 36, left: 60 };
      const W = W0 - margin.left - margin.right;
      const H = HEIGHT_BOT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${W0} ${HEIGHT_BOT}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);

      const all = bottomData.flatMap((d) => [d.sdLinear, d.sdPoly3, d.sdOracle, d.theoreticalSD]);
      const xScale = d3.scaleLog().domain([N_GRID[0] * 0.9, N_GRID[N_GRID.length - 1] * 1.1]).range([0, W]);
      const yScale = d3.scaleLog().domain([Math.min(...all) * 0.8, Math.max(...all) * 1.2]).range([H, 0]);
      g.append('g').attr('transform', `translate(0, ${H})`).call(d3.axisBottom(xScale).ticks(5, '.0f'));
      g.append('g').call(d3.axisLeft(yScale).ticks(4, '.3f'));
      const series = [
        { key: 'theoreticalSD', color: paletteSemi.theoryLine, label: 'BKRW bound', dash: true },
        { key: 'sdOracle', color: paletteSemi.oneStep, label: 'oracle', dash: false },
        { key: 'sdPoly3', color: paletteSemi.dml, label: 'poly-deg-3', dash: false },
        { key: 'sdLinear', color: paletteSemi.linear, label: 'linear', dash: false },
      ] as const;
      series.forEach((s) => {
        const ys = bottomData.map((d) => d[s.key as keyof typeof d] as number);
        const path = g.append('path').datum(ys).attr('d',
          d3.line<number>().x((_d, i) => xScale(N_GRID[i])).y((d) => yScale(d)) as unknown as string)
          .style('fill', 'none').style('stroke', s.color).style('stroke-width', 2);
        if (s.dash) path.style('stroke-dasharray', '5,4');
        ys.forEach((y, i) => {
          g.append('circle').attr('cx', xScale(N_GRID[i])).attr('cy', yScale(y))
            .attr('r', 3).style('fill', s.color);
        });
      });
      // Legend.
      const lgX = W - 130;
      series.forEach((s, i) => {
        const y = i * 16 + 8;
        const line = g.append('line').attr('x1', lgX).attr('x2', lgX + 22)
          .attr('y1', y).attr('y2', y)
          .style('stroke', s.color).style('stroke-width', 2);
        if (s.dash) line.style('stroke-dasharray', '5,4');
        g.append('text').attr('x', lgX + 28).attr('y', y + 4)
          .style('fill', 'var(--color-text)')
          .style('font-family', 'var(--font-mono)').style('font-size', 11).text(s.label);
      });
      g.append('text').attr('x', 0).attr('y', -8)
        .style('fill', 'var(--color-text)')
        .style('font-family', 'var(--font-mono)').style('font-size', 12)
        .text('SD of θ̂ vs n (B=60, log-log) — all three nuisance choices approach the bound');
    },
    [bottomData, containerWidth, isMobile],
  );

  return (
    <div ref={containerRef} className="viz-container" style={{ width: '100%' }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 12 }}>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 13 }}>Nuisance:</span>
          {(['linear', 'poly3', 'oracle'] as NuisanceChoice[]).map((c) => (
            <label key={c} style={{ fontFamily: 'var(--font-mono)', fontSize: 13 }}>
              <input type="radio" name="nuisance" value={c} checked={choice === c} onChange={() => setChoice(c)} />{' '}
              {c}
            </label>
          ))}
        </div>
        <label style={{ fontFamily: 'var(--font-mono)', fontSize: 13 }}>
          MC replications B (top panel only): <strong>{displayB}</strong>
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
            aria-label="Monte Carlo replications"
            style={{ width: '100%', marginTop: 4 }}
          />
        </label>
      </div>
      <svg ref={refTop} style={{ width: '100%', height: HEIGHT_TOP, display: 'block' }} />
      <svg ref={refBot} style={{ width: '100%', height: HEIGHT_BOT, display: 'block' }} />
    </div>
  );
}
