// =============================================================================
// ParticleFilterExplorer.tsx
//
// §6.4 of sequential-monte-carlo. Interactive bootstrap particle filter on a
// stochastic-volatility state-space model:
//
//   x_t = φ x_{t-1} + σ ε_t,    ε_t ~ N(0, 1)
//   y_t = exp(x_t / 2) η_t,     η_t ~ N(0, 1)
//
// Three panels:
//   (A) Truth latent x_t and observations y_t over time.
//   (B) Filter posterior mean (line) with 95% credible band (shaded), overlaid
//       on the true latent path.
//   (C) Per-step ESS, with the τN resampling threshold marked.
//
// Reader-discoverable behaviours:
//   - High σ: latent variance is large, observations more informative, ESS healthy.
//   - High φ: latent is persistent, posterior mean tracks slowly when σ_emission is wide.
//   - Small N: ESS collapses on sharp observations; filter posterior gets coarse.
//
// Pure TS via shared/sequential-monte-carlo.ts → simulateSV + bootstrapParticleFilter.
// At default (N = 500, T = 100) the filter takes ≈ 80-150 ms. Sliders commit-on-release.
//
// Static fallback: /images/topics/sequential-monte-carlo/06_sv_particle_filter.png
// =============================================================================

import { useCallback, useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import {
  simulateSV,
  bootstrapParticleFilter,
  paletteSMC,
  SMC_SEED,
} from './shared/sequential-monte-carlo';

interface Committed {
  N: number;
  T: number;
  phi: number;
  sigma: number;
}

function runFilter(c: Committed) {
  const data = simulateSV(c.T, c.phi, c.sigma, SMC_SEED + 6);
  const filt = bootstrapParticleFilter(data, c.N, c.phi, c.sigma, 0.5, SMC_SEED + 7);
  return { data, filt };
}

export default function ParticleFilterExplorer() {
  const [dN, setDN] = useState(500);
  const [dT, setDT] = useState(100);
  const [dPhi, setDPhi] = useState(0.95);
  const [dSigma, setDSigma] = useState(0.2);
  const [committed, setCommitted] = useState<Committed>({
    N: 500,
    T: 100,
    phi: 0.95,
    sigma: 0.2,
  });

  const commit = useCallback(
    (patch: Partial<Committed>) => setCommitted((c) => ({ ...c, ...patch })),
    [],
  );

  const result = useMemo(() => runFilter(committed), [committed]);
  const { data, filt } = result;

  // RMSE for diagnostic
  const rmse = useMemo(() => {
    let s = 0;
    for (let t = 0; t < filt.T; t++) {
      const d = filt.mean[t] - data.xTrue[t];
      s += d * d;
    }
    return Math.sqrt(s / filt.T);
  }, [filt, data]);

  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const width = containerWidth || 760;
  const panelGap = 14;
  const topH = 220;
  const bottomH = 160;
  const margin = { top: 22, right: 14, bottom: 32, left: 48 };
  const panelW = (width - panelGap) / 2;
  const innerWtop = panelW - margin.left - margin.right;
  const innerHtop = topH - margin.top - margin.bottom;
  const innerWbot = width - margin.left - margin.right;
  const innerHbot = bottomH - margin.top - margin.bottom;

  const topSvgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (width <= 0) return;

      // ---- Panel A: truth path + observations
      const gA = svg.append('g').attr('transform', 'translate(0, 0)');
      const innerA = gA.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);
      const xA = d3.scaleLinear().domain([0, filt.T - 1]).range([0, innerWtop]);
      const yExt = d3.extent([...data.xTrue, ...data.y]) as [number, number];
      const yA = d3
        .scaleLinear()
        .domain([Math.min(yExt[0], -3), Math.max(yExt[1], 3)])
        .range([innerHtop, 0]);
      innerA
        .append('g')
        .attr('transform', `translate(0, ${innerHtop})`)
        .call(d3.axisBottom(xA).ticks(5))
        .call((g) => g.selectAll('text').style('fill', 'var(--color-text)').style('font-size', '10px'))
        .call((g) => g.selectAll('line, path').style('stroke', 'var(--color-muted)'));
      innerA
        .append('g')
        .call(d3.axisLeft(yA).ticks(5))
        .call((g) => g.selectAll('text').style('fill', 'var(--color-text)').style('font-size', '10px'))
        .call((g) => g.selectAll('line, path').style('stroke', 'var(--color-muted)'));
      innerA
        .append('text')
        .attr('x', innerWtop / 2)
        .attr('y', innerHtop + 24)
        .attr('text-anchor', 'middle')
        .style('font-size', '10px')
        .style('fill', 'var(--color-text-secondary)')
        .text('t');
      // observations
      for (let t = 0; t < filt.T; t++) {
        innerA
          .append('circle')
          .attr('cx', xA(t))
          .attr('cy', yA(data.y[t]))
          .attr('r', 1.6)
          .style('fill', paletteSMC.accent)
          .style('opacity', 0.6);
      }
      // truth latent
      const truthLn = d3
        .line<number>()
        .x((_, i) => xA(i))
        .y((x) => yA(x));
      innerA
        .append('path')
        .datum(data.xTrue)
        .attr('d', truthLn)
        .style('fill', 'none')
        .style('stroke', paletteSMC.target)
        .style('stroke-width', 1.6);
      gA.append('text')
        .attr('x', margin.left)
        .attr('y', 13)
        .style('font-size', '11px')
        .style('font-weight', '600')
        .style('fill', 'var(--color-text)')
        .text('(A) truth latent x_t (red) and observations y_t (dots)');

      // ---- Panel B: filter posterior with 95% band
      const gB = svg.append('g').attr('transform', `translate(${panelW + panelGap}, 0)`);
      const innerB = gB.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);
      const xExt = [0, filt.T - 1] as [number, number];
      const yBExt = d3.extent([...filt.q025, ...filt.q975, ...data.xTrue]) as [number, number];
      const xB = d3.scaleLinear().domain(xExt).range([0, innerWtop]);
      const yB = d3
        .scaleLinear()
        .domain([Math.min(yBExt[0], -3), Math.max(yBExt[1], 3)])
        .range([innerHtop, 0]);
      innerB
        .append('g')
        .attr('transform', `translate(0, ${innerHtop})`)
        .call(d3.axisBottom(xB).ticks(5))
        .call((g) => g.selectAll('text').style('fill', 'var(--color-text)').style('font-size', '10px'))
        .call((g) => g.selectAll('line, path').style('stroke', 'var(--color-muted)'));
      innerB
        .append('g')
        .call(d3.axisLeft(yB).ticks(5))
        .call((g) => g.selectAll('text').style('fill', 'var(--color-text)').style('font-size', '10px'))
        .call((g) => g.selectAll('line, path').style('stroke', 'var(--color-muted)'));
      innerB
        .append('text')
        .attr('x', innerWtop / 2)
        .attr('y', innerHtop + 24)
        .attr('text-anchor', 'middle')
        .style('font-size', '10px')
        .style('fill', 'var(--color-text-secondary)')
        .text('t');
      // 95% band
      const areaGen = d3
        .area<number>()
        .x((_, i) => xB(i))
        .y0((_, i) => yB(filt.q025[i]))
        .y1((_, i) => yB(filt.q975[i]));
      innerB
        .append('path')
        .datum(filt.q975)
        .attr('d', areaGen)
        .style('fill', paletteSMC.cloud)
        .style('opacity', 0.25);
      // filter mean
      const meanLn = d3
        .line<number>()
        .x((_, i) => xB(i))
        .y((m) => yB(m));
      innerB
        .append('path')
        .datum(filt.mean)
        .attr('d', meanLn)
        .style('fill', 'none')
        .style('stroke', paletteSMC.cloud)
        .style('stroke-width', 1.6);
      // truth
      innerB
        .append('path')
        .datum(data.xTrue)
        .attr('d', meanLn)
        .style('fill', 'none')
        .style('stroke', paletteSMC.target)
        .style('stroke-width', 1.2)
        .style('stroke-dasharray', '4 3');
      gB.append('text')
        .attr('x', margin.left)
        .attr('y', 13)
        .style('font-size', '11px')
        .style('font-weight', '600')
        .style('fill', 'var(--color-text)')
        .text('(B) filter mean (blue) ± 95% CI vs truth (red dashed)');
    },
    [width, panelW, innerWtop, innerHtop, result, filt, data],
  );

  const bottomSvgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (width <= 0) return;
      const inner = svg
        .append('g')
        .attr('transform', `translate(${margin.left}, ${margin.top})`);
      const x = d3.scaleLinear().domain([0, filt.T - 1]).range([0, innerWbot]);
      const y = d3.scaleLinear().domain([0, committed.N * 1.05]).range([innerHbot, 0]);
      inner
        .append('g')
        .attr('transform', `translate(0, ${innerHbot})`)
        .call(d3.axisBottom(x).ticks(6))
        .call((g) => g.selectAll('text').style('fill', 'var(--color-text)').style('font-size', '10px'))
        .call((g) => g.selectAll('line, path').style('stroke', 'var(--color-muted)'));
      inner
        .append('g')
        .call(d3.axisLeft(y).ticks(4))
        .call((g) => g.selectAll('text').style('fill', 'var(--color-text)').style('font-size', '10px'))
        .call((g) => g.selectAll('line, path').style('stroke', 'var(--color-muted)'));
      inner
        .append('text')
        .attr('x', innerWbot / 2)
        .attr('y', innerHbot + 24)
        .attr('text-anchor', 'middle')
        .style('font-size', '10px')
        .style('fill', 'var(--color-text-secondary)')
        .text('t');
      inner
        .append('text')
        .attr('transform', `translate(${-34}, ${innerHbot / 2}) rotate(-90)`)
        .attr('text-anchor', 'middle')
        .style('font-size', '10px')
        .style('fill', 'var(--color-text-secondary)')
        .text('ESS');
      // tau N reference
      inner
        .append('line')
        .attr('x1', 0)
        .attr('x2', innerWbot)
        .attr('y1', y(0.5 * committed.N))
        .attr('y2', y(0.5 * committed.N))
        .style('stroke', paletteSMC.accent)
        .style('stroke-width', 1)
        .style('stroke-dasharray', '4 3');
      const ln = d3
        .line<number>()
        .x((_, i) => x(i))
        .y((e) => y(e));
      inner
        .append('path')
        .datum(filt.ess)
        .attr('d', ln)
        .style('fill', 'none')
        .style('stroke', paletteSMC.cloud)
        .style('stroke-width', 1.4);
      svg
        .append('text')
        .attr('x', margin.left)
        .attr('y', 13)
        .style('font-size', '11px')
        .style('font-weight', '600')
        .style('fill', 'var(--color-text)')
        .text('(C) per-step ESS');
    },
    [width, innerWbot, innerHbot, filt, committed.N],
  );

  const isStale =
    committed.N !== dN || committed.T !== dT || committed.phi !== dPhi || committed.sigma !== dSigma;

  return (
    <div ref={containerRef} className="w-full">
      <div
        className="mb-3 flex flex-wrap items-center gap-x-4 gap-y-2 text-xs"
        style={{ color: 'var(--color-text)' }}
      >
        <label className="inline-flex items-center gap-2">
          <span>
            N = <span className="font-mono">{dN}</span>
          </span>
          <input
            type="range"
            min={100}
            max={1000}
            step={50}
            value={dN}
            onChange={(e) => setDN(parseInt(e.target.value, 10))}
            onMouseUp={(e) => commit({ N: parseInt((e.target as HTMLInputElement).value, 10) })}
            onTouchEnd={(e) => commit({ N: parseInt((e.target as HTMLInputElement).value, 10) })}
            onKeyUp={(e) => commit({ N: parseInt((e.target as HTMLInputElement).value, 10) })}
            className="w-28"
            aria-label="Particle count N"
          />
        </label>
        <label className="inline-flex items-center gap-2">
          <span>
            T = <span className="font-mono">{dT}</span>
          </span>
          <input
            type="range"
            min={30}
            max={200}
            step={10}
            value={dT}
            onChange={(e) => setDT(parseInt(e.target.value, 10))}
            onMouseUp={(e) => commit({ T: parseInt((e.target as HTMLInputElement).value, 10) })}
            onTouchEnd={(e) => commit({ T: parseInt((e.target as HTMLInputElement).value, 10) })}
            onKeyUp={(e) => commit({ T: parseInt((e.target as HTMLInputElement).value, 10) })}
            className="w-28"
            aria-label="Observation length T"
          />
        </label>
        <label className="inline-flex items-center gap-2">
          <span>
            φ = <span className="font-mono">{dPhi.toFixed(2)}</span>
          </span>
          <input
            type="range"
            min={0.5}
            max={0.99}
            step={0.01}
            value={dPhi}
            onChange={(e) => setDPhi(parseFloat(e.target.value))}
            onMouseUp={(e) => commit({ phi: parseFloat((e.target as HTMLInputElement).value) })}
            onTouchEnd={(e) => commit({ phi: parseFloat((e.target as HTMLInputElement).value) })}
            onKeyUp={(e) => commit({ phi: parseFloat((e.target as HTMLInputElement).value) })}
            className="w-28"
            aria-label="Persistence φ"
          />
        </label>
        <label className="inline-flex items-center gap-2">
          <span>
            σ = <span className="font-mono">{dSigma.toFixed(2)}</span>
          </span>
          <input
            type="range"
            min={0.1}
            max={0.5}
            step={0.05}
            value={dSigma}
            onChange={(e) => setDSigma(parseFloat(e.target.value))}
            onMouseUp={(e) => commit({ sigma: parseFloat((e.target as HTMLInputElement).value) })}
            onTouchEnd={(e) => commit({ sigma: parseFloat((e.target as HTMLInputElement).value) })}
            onKeyUp={(e) => commit({ sigma: parseFloat((e.target as HTMLInputElement).value) })}
            className="w-28"
            aria-label="Innovation σ"
          />
        </label>
        {isStale && (
          <span className="text-[10px] italic" style={{ color: 'var(--color-text-secondary)' }}>
            release the slider to recompute
          </span>
        )}
      </div>
      <svg
        ref={topSvgRef}
        width={width}
        height={topH}
        viewBox={`0 0 ${width} ${topH}`}
        role="img"
        aria-label="Truth latent path with observations, and the bootstrap PF posterior with 95% credible band."
      />
      <svg
        ref={bottomSvgRef}
        width={width}
        height={bottomH}
        viewBox={`0 0 ${width} ${bottomH}`}
        role="img"
        aria-label="Per-step effective sample size for the bootstrap particle filter."
      />
      <div className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-xs sm:grid-cols-3" style={{ color: 'var(--color-text)' }}>
        <div className="font-mono">RMSE(filter mean vs truth) = {rmse.toFixed(3)}</div>
        <div className="font-mono">avg ESS = {(filt.ess.reduce((s, v) => s + v, 0) / filt.ess.length).toFixed(1)}</div>
        <div className="font-mono">log p̂(y_{`{1:T}`}) = {filt.logZ.toFixed(2)}</div>
      </div>
    </div>
  );
}
