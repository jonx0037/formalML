import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  completeCaseMean,
  ipwMean,
  marEfficiencyBound,
  mulberry32,
  oneStepMar,
  paletteSemi,
  sampleMar,
} from './shared/semiparametric-inference';

// =============================================================================
// EfficiencyBoundExplorer — §4
// RMSE-vs-n trajectories on log-log axes for three MAR estimators
// (complete-case, IPW, AIPW one-step), with the BKRW bound √(V_eff/n) overlay.
// Truth-substituted nuisances throughout. Recreates the structure of the
// §4 cell 30 figure.
// =============================================================================

const HEIGHT = 360;
const SM_BREAKPOINT = 640;
const N_GRID = [100, 250, 500, 1000, 2500, 5000];
const PSI_0 = 1 + 0.25 + 2 / 3; // 1.9167 — true MAR mean

export default function EfficiencyBoundExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [displayB, setDisplayB] = useState(100);
  const [committedB, setCommittedB] = useState(100);

  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;

  const data = useMemo(() => {
    const ccRmse: number[] = [];
    const ipwRmse: number[] = [];
    const aipwRmse: number[] = [];
    const bound: number[] = [];
    // Compute V_eff once via a large reference sample.
    const refRng = mulberry32(99999);
    const refSample = sampleMar(5000, refRng);
    const { Veff } = marEfficiencyBound(refSample.piOracle, refSample.mOracle, 1);
    for (const n of N_GRID) {
      const ccs = new Float64Array(committedB);
      const ipws = new Float64Array(committedB);
      const aipws = new Float64Array(committedB);
      for (let b = 0; b < committedB; b++) {
        const rng = mulberry32(20260515 + b * 31 + n * 7);
        const s = sampleMar(n, rng);
        ccs[b] = completeCaseMean(s.R, s.Y) - PSI_0;
        ipws[b] = ipwMean(s.R, s.Y, s.piOracle) - PSI_0;
        const { psi } = oneStepMar(s.R, s.Y, s.mOracle, s.piOracle);
        aipws[b] = psi - PSI_0;
      }
      const rmse = (arr: Float64Array) => {
        let ss = 0;
        for (let i = 0; i < arr.length; i++) ss += arr[i] * arr[i];
        return Math.sqrt(ss / arr.length);
      };
      ccRmse.push(rmse(ccs));
      ipwRmse.push(rmse(ipws));
      aipwRmse.push(rmse(aipws));
      bound.push(Math.sqrt(Veff / n));
    }
    return { ccRmse, ipwRmse, aipwRmse, bound, Veff };
  }, [committedB]);

  const ref = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const W = (isMobile ? w : Math.min(w, 800)) - 70 - 14;
      const H = HEIGHT - 36 - 30;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${(isMobile ? w : Math.min(w, 800))} ${HEIGHT}`);
      const g = svg.append('g').attr('transform', `translate(70, 30)`);

      const allY = [...data.ccRmse, ...data.ipwRmse, ...data.aipwRmse, ...data.bound];
      const xScale = d3.scaleLog().base(10).domain([N_GRID[0] * 0.9, N_GRID[N_GRID.length - 1] * 1.1]).range([0, W]);
      const yScale = d3.scaleLog().base(10).domain([Math.min(...allY) * 0.8, Math.max(...allY) * 1.2]).range([H, 0]);

      g.append('g').attr('transform', `translate(0, ${H})`)
        .call(d3.axisBottom(xScale).ticks(6, '.0f'));
      g.append('g').call(d3.axisLeft(yScale).ticks(5, '.3f'));
      g.append('text').attr('x', W / 2).attr('y', H + 32).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-family', 'var(--font-mono)').style('font-size', 11)
        .text('sample size n (log)');
      g.append('text').attr('x', -H / 2).attr('y', -45).attr('text-anchor', 'middle')
        .attr('transform', 'rotate(-90)')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-family', 'var(--font-mono)').style('font-size', 11)
        .text('RMSE (log)');

      const lineGen = (ys: number[]) =>
        d3.line<number>().x((_d, i) => xScale(N_GRID[i])).y((d) => yScale(d))(ys) ?? '';

      // BKRW bound (dashed, black).
      g.append('path').attr('d', lineGen(data.bound))
        .style('fill', 'none').style('stroke', paletteSemi.theoryLine)
        .style('stroke-width', 2).style('stroke-dasharray', '5,4');
      // Complete-case (grey).
      g.append('path').attr('d', lineGen(data.ccRmse))
        .style('fill', 'none').style('stroke', paletteSemi.completeCase).style('stroke-width', 2);
      // IPW (brown).
      g.append('path').attr('d', lineGen(data.ipwRmse))
        .style('fill', 'none').style('stroke', paletteSemi.ipw).style('stroke-width', 2);
      // AIPW (red).
      g.append('path').attr('d', lineGen(data.aipwRmse))
        .style('fill', 'none').style('stroke', paletteSemi.oneStep).style('stroke-width', 2.4);

      // Points.
      const markers = (ys: number[], color: string) => {
        g.selectAll(null).data(ys).join('circle')
          .attr('cx', (_d, i) => xScale(N_GRID[i]))
          .attr('cy', (d) => yScale(d))
          .attr('r', 3.5).style('fill', color);
      };
      markers(data.ccRmse, paletteSemi.completeCase);
      markers(data.ipwRmse, paletteSemi.ipw);
      markers(data.aipwRmse, paletteSemi.oneStep);
      markers(data.bound, paletteSemi.theoryLine);

      // Legend.
      const lgX = W - 170;
      const lgY = 8;
      const legend = [
        { c: paletteSemi.theoryLine, t: `BKRW bound √(V_eff/n) — V_eff≈${data.Veff.toFixed(2)}`, dash: true },
        { c: paletteSemi.oneStep, t: 'AIPW (oracle)', dash: false },
        { c: paletteSemi.ipw, t: 'IPW (oracle π)', dash: false },
        { c: paletteSemi.completeCase, t: 'Complete-case', dash: false },
      ];
      legend.forEach((L, i) => {
        const y = lgY + i * 18;
        const line = g.append('line').attr('x1', lgX).attr('x2', lgX + 24)
          .attr('y1', y).attr('y2', y)
          .style('stroke', L.c).style('stroke-width', 2);
        if (L.dash) line.style('stroke-dasharray', '5,4');
        g.append('text').attr('x', lgX + 30).attr('y', y + 4)
          .style('fill', 'var(--color-text)')
          .style('font-family', 'var(--font-mono)').style('font-size', 11).text(L.t);
      });
    },
    [data, containerWidth, isMobile],
  );

  return (
    <div ref={containerRef} className="viz-container" style={{ width: '100%' }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginBottom: 12 }}>
        <label style={{ fontFamily: 'var(--font-mono)', fontSize: 13 }}>
          Monte-Carlo replications B: <strong>{displayB}</strong>
          <input
            type="range"
            min={20}
            max={300}
            step={20}
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
      <svg ref={ref} style={{ width: '100%', height: HEIGHT, display: 'block' }} />
    </div>
  );
}
