import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { mulberry32, gaussianFrom } from './shared/always-valid-inference';

// -----------------------------------------------------------------------------
// Viz 2.1 — Ville's inequality in action.
//
// Top panel: log M_n^sG(λ) for 30 H_0 (blue) and 30 H_1 (red) sample paths,
//            horizontal threshold at log(1/δ) ≈ 3.0.
// Bottom panel: empirical P(sup_{m≤n} M_m ≥ 1/δ) vs n under H_0 (stays below δ
//               — the Ville guarantee) and H_1 (climbs to 1 — the power).
//
// Sliders (commit-on-release for the heavy MC):
//   λ ∈ [0.05, 1.5], μ_1 ∈ [0.0, 0.6], n_max ∈ {200, 500, 1000, 2000}, δ.
//
// Annotation: λ* = μ_1/σ² and the GRO log-growth rate μ_1²/(2σ²).
// -----------------------------------------------------------------------------

const SM_BREAKPOINT = 640;
const PANEL_H = 260;
const MARGIN = { top: 28, right: 14, bottom: 38, left: 56 };
const B_REPLICATES = 800;
const N_PATHS_SHOWN = 30;
const N_MAX_OPTIONS = [200, 500, 1000, 2000];
const SIGMA = 1;

const COLOR_H0 = '#356fa3';
const COLOR_H1 = '#c44545';
const COLOR_THRESH = 'var(--color-accent)';

interface VilleResult {
  logPathsH0: Float64Array[];
  logPathsH1: Float64Array[];
  crossingH0: Float64Array; // running max-crossing fraction over n
  crossingH1: Float64Array;
  nGrid: number[];
}

function simulate(lambda: number, mu1: number, nmax: number, delta: number): VilleResult {
  const rng0 = mulberry32(20260525);
  const rng1 = mulberry32(20260601);
  const g0 = gaussianFrom(rng0);
  const g1 = gaussianFrom(rng1);
  const logThr = Math.log(1 / delta);

  const logPathsH0: Float64Array[] = [];
  const logPathsH1: Float64Array[] = [];
  // for crossing fraction: maintain "has crossed by step n" booleans per replicate
  const crossedH0 = new Uint8Array(B_REPLICATES);
  const crossedH1 = new Uint8Array(B_REPLICATES);
  const crossingH0 = new Float64Array(nmax);
  const crossingH1 = new Float64Array(nmax);

  for (let b = 0; b < B_REPLICATES; b++) {
    const keep = b < N_PATHS_SHOWN;
    const path0 = keep ? new Float64Array(nmax) : null;
    const path1 = keep ? new Float64Array(nmax) : null;

    let logM0 = 0;
    let logM1 = 0;
    for (let n = 1; n <= nmax; n++) {
      // log M_n^sG(λ) = λ S_n − n λ² σ²/2
      // Increment: λ X_n − λ² σ²/2
      const x0 = g0(); // H_0: X ~ N(0,1)
      const x1 = g1() + mu1; // H_1: X ~ N(μ_1, 1)
      logM0 += lambda * x0 - 0.5 * lambda * lambda * SIGMA * SIGMA;
      logM1 += lambda * x1 - 0.5 * lambda * lambda * SIGMA * SIGMA;
      if (path0) path0[n - 1] = logM0;
      if (path1) path1[n - 1] = logM1;
      if (!crossedH0[b] && logM0 >= logThr) crossedH0[b] = 1;
      if (!crossedH1[b] && logM1 >= logThr) crossedH1[b] = 1;
      crossingH0[n - 1] += crossedH0[b];
      crossingH1[n - 1] += crossedH1[b];
    }
    if (path0) logPathsH0.push(path0);
    if (path1) logPathsH1.push(path1);
  }
  for (let n = 0; n < nmax; n++) {
    crossingH0[n] /= B_REPLICATES;
    crossingH1[n] /= B_REPLICATES;
  }
  const nGrid = Array.from({ length: nmax }, (_, i) => i + 1);
  return { logPathsH0, logPathsH1, crossingH0, crossingH1, nGrid };
}

export default function VilleInequalityExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();

  const [displayLambda, setDisplayLambda] = useState(0.3);
  const [committedLambda, setCommittedLambda] = useState(0.3);
  const [displayMu, setDisplayMu] = useState(0.3);
  const [committedMu, setCommittedMu] = useState(0.3);
  const [displayNmaxIdx, setDisplayNmaxIdx] = useState(2); // 1000
  const [committedNmaxIdx, setCommittedNmaxIdx] = useState(2);
  const [delta, setDelta] = useState(0.05);

  const committedNmax = N_MAX_OPTIONS[committedNmaxIdx];
  const displayNmax = N_MAX_OPTIONS[displayNmaxIdx];

  const sim = useMemo(
    () => simulate(committedLambda, committedMu, committedNmax, delta),
    [committedLambda, committedMu, committedNmax, delta],
  );

  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const panelW = Math.max(280, Math.min(containerWidth || 720, 820));

  // KL log-growth at the current committed (λ, μ_1, σ=1):
  // d/dn E[log M_n] = λ μ_1 − λ²/2
  const oneStepGrowth = committedLambda * committedMu - 0.5 * committedLambda * committedLambda;
  // GRO at λ* = μ_1/σ² is μ_1²/(2σ²)
  const groRate = (committedMu * committedMu) / 2;
  const lambdaStar = committedMu; // / σ² = 1

  const logThr = Math.log(1 / delta);

  const pathsRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const innerW = panelW - MARGIN.left - MARGIN.right;
      const innerH = PANEL_H - MARGIN.top - MARGIN.bottom;
      if (innerW <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const xScale = d3.scaleLog().domain([1, displayNmax]).range([0, innerW]);
      // y-range adaptive: include the threshold and the typical drift range
      const driftAtNmax = committedLambda * committedMu * displayNmax - 0.5 * committedLambda * committedLambda * displayNmax;
      const yMax = Math.max(logThr + 1, driftAtNmax + 5, 5);
      const yMin = Math.min(-yMax, -5);
      const yScale = d3.scaleLinear().domain([yMin, yMax]).range([innerH, 0]);

      g.append('g')
        .attr('transform', `translate(0,${innerH})`)
        .call(d3.axisBottom(xScale).ticks(4, '~s'))
        .selectAll('text')
        .style('fill', 'var(--color-text)');
      g.append('g')
        .call(d3.axisLeft(yScale).ticks(5))
        .selectAll('text')
        .style('fill', 'var(--color-text)');

      // threshold line
      g.append('line')
        .attr('x1', 0)
        .attr('x2', innerW)
        .attr('y1', yScale(logThr))
        .attr('y2', yScale(logThr))
        .style('stroke', COLOR_THRESH)
        .attr('stroke-width', 1.4)
        .attr('stroke-dasharray', '6 3');
      g.append('text')
        .attr('x', innerW - 5)
        .attr('y', yScale(logThr) - 4)
        .attr('text-anchor', 'end')
        .style('font-family', 'var(--font-mono)')
        .style('font-size', '11px')
        .style('fill', COLOR_THRESH)
        .text(`log(1/δ) = ${logThr.toFixed(2)}`);

      // zero line
      g.append('line')
        .attr('x1', 0)
        .attr('x2', innerW)
        .attr('y1', yScale(0))
        .attr('y2', yScale(0))
        .style('stroke', 'var(--color-text-secondary)')
        .attr('stroke-width', 0.6)
        .attr('opacity', 0.5);

      const pathGen = d3
        .line<number>()
        .x((_, i) => xScale(i + 1))
        .y((v) => yScale(v));

      for (const p of sim.logPathsH0) {
        g.append('path')
          .datum(Array.from(p.subarray(0, displayNmax)) as number[])
          .attr('d', pathGen as never)
          .style('fill', 'none')
          .style('stroke', COLOR_H0)
          .attr('stroke-width', 0.7)
          .attr('opacity', 0.55);
      }
      for (const p of sim.logPathsH1) {
        g.append('path')
          .datum(Array.from(p.subarray(0, displayNmax)) as number[])
          .attr('d', pathGen as never)
          .style('fill', 'none')
          .style('stroke', COLOR_H1)
          .attr('stroke-width', 0.7)
          .attr('opacity', 0.55);
      }

      g.append('text')
        .attr('x', 0)
        .attr('y', -10)
        .style('fill', 'var(--color-text)')
        .style('font-family', 'var(--font-sans)')
        .style('font-size', '12px')
        .style('font-weight', 600)
        .text(`log M_n^sG(λ) — 30 H_0 (blue), 30 H_1 (red)`);
      g.append('text')
        .attr('x', innerW / 2)
        .attr('y', innerH + 30)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-family', 'var(--font-sans)')
        .style('font-size', '11px')
        .text('n (log scale)');
    },
    [sim, displayNmax, committedLambda, committedMu, delta, panelW],
  );

  const crossingRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const innerW = panelW - MARGIN.left - MARGIN.right;
      const innerH = PANEL_H - MARGIN.top - MARGIN.bottom;
      if (innerW <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const xScale = d3.scaleLog().domain([1, displayNmax]).range([0, innerW]);
      const yScale = d3.scaleLinear().domain([0, 1]).range([innerH, 0]);

      g.append('g')
        .attr('transform', `translate(0,${innerH})`)
        .call(d3.axisBottom(xScale).ticks(4, '~s'))
        .selectAll('text')
        .style('fill', 'var(--color-text)');
      g.append('g')
        .call(d3.axisLeft(yScale).tickFormat(d3.format('.0%')))
        .selectAll('text')
        .style('fill', 'var(--color-text)');

      // δ reference
      g.append('line')
        .attr('x1', 0)
        .attr('x2', innerW)
        .attr('y1', yScale(delta))
        .attr('y2', yScale(delta))
        .style('stroke', COLOR_THRESH)
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '4 3');
      g.append('text')
        .attr('x', 8)
        .attr('y', yScale(delta) - 4)
        .style('font-family', 'var(--font-mono)')
        .style('font-size', '11px')
        .style('fill', COLOR_THRESH)
        .text(`Ville bound: δ = ${delta.toFixed(2)}`);

      const xy0 = d3
        .line<number>()
        .x((_, i) => xScale(i + 1))
        .y((v) => yScale(v));

      g.append('path')
        .datum(Array.from(sim.crossingH0.subarray(0, displayNmax)) as number[])
        .attr('d', xy0 as never)
        .style('fill', 'none')
        .style('stroke', COLOR_H0)
        .attr('stroke-width', 1.8);
      g.append('path')
        .datum(Array.from(sim.crossingH1.subarray(0, displayNmax)) as number[])
        .attr('d', xy0 as never)
        .style('fill', 'none')
        .style('stroke', COLOR_H1)
        .attr('stroke-width', 1.8);

      g.append('text')
        .attr('x', 0)
        .attr('y', -10)
        .style('fill', 'var(--color-text)')
        .style('font-family', 'var(--font-sans)')
        .style('font-size', '12px')
        .style('font-weight', 600)
        .text(`Empirical P(sup_{m≤n} M_m ≥ 1/δ) — H_0 stays ≤ δ (Ville), H_1 → 1`);
      g.append('text')
        .attr('x', innerW / 2)
        .attr('y', innerH + 30)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-family', 'var(--font-sans)')
        .style('font-size', '11px')
        .text('n (log scale)');
    },
    [sim, displayNmax, delta, panelW],
  );

  const maxH0Cross = Math.max(...Array.from(sim.crossingH0.subarray(0, displayNmax)));
  const maxH1Cross = Math.max(...Array.from(sim.crossingH1.subarray(0, displayNmax)));

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: 'var(--font-sans)' }}>
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '14px',
          padding: '10px 0',
          fontSize: '13px',
        }}
      >
        <label style={{ display: 'flex', flexDirection: 'column', minWidth: '180px' }}>
          <span>
            Tilt λ = <strong>{displayLambda.toFixed(2)}</strong>
            {Math.abs(displayLambda - lambdaStar) < 0.02 ? ' = λ★' : ''}
          </span>
          <input
            type="range"
            min={0.05}
            max={1.5}
            step={0.01}
            value={displayLambda}
            aria-label="Tilt parameter lambda"
            onChange={(e) => setDisplayLambda(Number(e.target.value))}
            onMouseUp={(e) => setCommittedLambda(Number((e.target as HTMLInputElement).value))}
            onTouchEnd={(e) => setCommittedLambda(Number((e.target as HTMLInputElement).value))}
            onKeyUp={(e) => setCommittedLambda(Number((e.target as HTMLInputElement).value))}
          />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', minWidth: '180px' }}>
          <span>
            Alt mean μ₁ = <strong>{displayMu.toFixed(2)}</strong>
          </span>
          <input
            type="range"
            min={0}
            max={0.6}
            step={0.01}
            value={displayMu}
            aria-label="Alternative-hypothesis mean"
            onChange={(e) => setDisplayMu(Number(e.target.value))}
            onMouseUp={(e) => setCommittedMu(Number((e.target as HTMLInputElement).value))}
            onTouchEnd={(e) => setCommittedMu(Number((e.target as HTMLInputElement).value))}
            onKeyUp={(e) => setCommittedMu(Number((e.target as HTMLInputElement).value))}
          />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', minWidth: '160px' }}>
          <span>
            n_max = <strong>{displayNmax}</strong>
          </span>
          <input
            type="range"
            min={0}
            max={N_MAX_OPTIONS.length - 1}
            step={1}
            value={displayNmaxIdx}
            aria-label="Sample-size horizon"
            onChange={(e) => setDisplayNmaxIdx(Number(e.target.value))}
            onMouseUp={(e) => setCommittedNmaxIdx(Number((e.target as HTMLInputElement).value))}
            onTouchEnd={(e) => setCommittedNmaxIdx(Number((e.target as HTMLInputElement).value))}
            onKeyUp={(e) => setCommittedNmaxIdx(Number((e.target as HTMLInputElement).value))}
          />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', minWidth: '160px' }}>
          <span>
            δ = <strong>{delta.toFixed(3)}</strong>
          </span>
          <input
            type="range"
            min={0.01}
            max={0.20}
            step={0.005}
            value={delta}
            aria-label="Confidence level"
            onChange={(e) => setDelta(Number(e.target.value))}
          />
        </label>
        <div
          style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '12px',
            color: 'var(--color-text-secondary)',
            padding: '6px 10px',
            border: '1px solid var(--color-border)',
            borderRadius: '4px',
            minWidth: '300px',
          }}
        >
          λ★ = μ₁/σ² = {lambdaStar.toFixed(2)},
          {' '}drift λμ₁ − λ²/2 = {oneStepGrowth.toFixed(4)},
          {' '}GRO μ₁²/(2σ²) = {groRate.toFixed(4)}.
          <br />
          Empirical max crossing: H₀ = {maxH0Cross.toFixed(3)}, H₁ = {maxH1Cross.toFixed(3)}.
        </div>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
        <svg ref={pathsRef} width={panelW} height={PANEL_H} style={{ maxWidth: '100%', display: 'block' }} />
        <svg ref={crossingRef} width={panelW} height={PANEL_H} style={{ maxWidth: '100%', display: 'block' }} />
      </div>
    </div>
  );
}
