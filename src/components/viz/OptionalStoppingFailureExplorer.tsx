import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { mulberry32, gaussianFrom, zHalf } from './shared/always-valid-inference';

// -----------------------------------------------------------------------------
// Viz 1.1 — Optional-stopping coverage failure of the fixed-n Gaussian CI.
//
// Two panels:
//   (left)  Cumulative empirical false-rejection rate of C_n^fixed under the
//           aggressive-analyst stopping rule, plotted as a step function in
//           n_max, with the nominal δ as a horizontal reference line.
//   (right) 200 sample paths of bar X_n on log-x from n=10 to n=n_max, with
//           the shrinking ±z_{δ/2}/√n band shaded; trajectories that exit
//           highlighted in red, others in gray.
//
// Sliders are commit-on-release for the heavy MC (B=2000 trajectories of
// length n_max). The committed n_max also drives the right panel.
// -----------------------------------------------------------------------------

const SM_BREAKPOINT = 640;
const MARGIN = { top: 24, right: 14, bottom: 38, left: 50 };
const PANEL_H = 320;
const N_PATHS_SHOWN = 200;
const B_REPLICATES = 2000;
const N_MAX_OPTIONS = [100, 200, 500, 1000, 2000, 5000, 10000];

const COLOR_RATE = 'var(--color-accent)';
const COLOR_NULL = 'var(--color-text-secondary)';
const COLOR_EXIT = '#c44545';
const COLOR_OK = '#9aa3ad';

interface SimResult {
  // false-rejection rate at each n_max in N_MAX_OPTIONS
  rateByNmax: { nmax: number; rate: number }[];
  // for the trajectory panel: 200 paths of bar X_n, indexed n=1..currentNmax
  paths: Float64Array[];
  // boolean: did this path exit the band by currentNmax?
  exitedAt: Int32Array; // index >=0 of first exit, or -1
  // shared n-grid for both panels
  nGrid: number[];
}

function simulate(committedNmax: number, delta: number): SimResult {
  const rng = mulberry32(20260525);
  const gauss = gaussianFrom(rng);
  const z = zHalf(delta);

  // Pre-compute z/√n for n=1..committedNmax
  const halfWidth = new Float64Array(committedNmax + 1);
  for (let n = 1; n <= committedNmax; n++) {
    halfWidth[n] = z / Math.sqrt(n);
  }

  // For the rate plot we only need the FIRST-EXIT time per replicate. For the
  // trajectory plot we keep the first N_PATHS_SHOWN paths in full.
  const firstExit = new Int32Array(B_REPLICATES);
  firstExit.fill(-1);
  const keptPaths: Float64Array[] = [];

  for (let b = 0; b < B_REPLICATES; b++) {
    let cumsum = 0;
    let path: Float64Array | null = null;
    if (b < N_PATHS_SHOWN) {
      path = new Float64Array(committedNmax);
      keptPaths.push(path);
    }
    for (let n = 1; n <= committedNmax; n++) {
      cumsum += gauss();
      const mean = cumsum / n;
      if (path !== null) path[n - 1] = mean;
      if (firstExit[b] === -1 && Math.abs(mean) >= halfWidth[n]) {
        firstExit[b] = n;
        // continue drawing for path completion (not all 2000 paths kept)
        if (path === null) {
          // skip the rest of this replicate — we only needed first exit
          break;
        }
      }
    }
  }

  // For each n_max option in N_MAX_OPTIONS that is ≤ committedNmax,
  // compute the fraction of replicates with firstExit > 0 AND ≤ n_max.
  const rateByNmax: { nmax: number; rate: number }[] = [];
  for (const nmax of N_MAX_OPTIONS) {
    if (nmax > committedNmax) continue;
    let count = 0;
    for (let b = 0; b < B_REPLICATES; b++) {
      const e = firstExit[b];
      if (e > 0 && e <= nmax) count++;
    }
    rateByNmax.push({ nmax, rate: count / B_REPLICATES });
  }

  // Mark which of the kept paths exited:
  const exitedAt = new Int32Array(N_PATHS_SHOWN);
  for (let i = 0; i < N_PATHS_SHOWN; i++) exitedAt[i] = firstExit[i];

  // Build the n-grid for x-axes. Log-x for the trajectory panel uses
  // n in [10, committedNmax]; the rate panel uses N_MAX_OPTIONS directly.
  const nGrid: number[] = [];
  for (let n = 1; n <= committedNmax; n++) nGrid.push(n);

  return { rateByNmax, paths: keptPaths, exitedAt, nGrid };
}

export default function OptionalStoppingFailureExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();

  const [displayNmaxIdx, setDisplayNmaxIdx] = useState(6); // 10000
  const [committedNmaxIdx, setCommittedNmaxIdx] = useState(6);
  const [displayDelta, setDisplayDelta] = useState(0.05);
  const [committedDelta, setCommittedDelta] = useState(0.05);

  const committedNmax = N_MAX_OPTIONS[committedNmaxIdx];
  const displayNmax = N_MAX_OPTIONS[displayNmaxIdx];

  const sim = useMemo(() => simulate(committedNmax, committedDelta), [committedNmax, committedDelta]);

  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const w = Math.max(280, Math.min(containerWidth || 720, 820));
  const panelW = isMobile ? w : Math.floor(w / 2) - 8;

  const ratePanelRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const innerW = panelW - MARGIN.left - MARGIN.right;
      const innerH = PANEL_H - MARGIN.top - MARGIN.bottom;
      if (innerW <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const xScale = d3.scaleLog().domain([50, 10000]).range([0, innerW]);
      const yScale = d3.scaleLinear().domain([0, 1]).range([innerH, 0]);

      // axes
      g.append('g')
        .attr('transform', `translate(0,${innerH})`)
        .call(d3.axisBottom(xScale).ticks(4, '~s'))
        .selectAll('text')
        .style('fill', 'var(--color-text)');
      g.append('g')
        .call(d3.axisLeft(yScale).tickFormat(d3.format('.0%')))
        .selectAll('text')
        .style('fill', 'var(--color-text)');

      // nominal δ reference line
      g.append('line')
        .attr('x1', 0)
        .attr('x2', innerW)
        .attr('y1', yScale(committedDelta))
        .attr('y2', yScale(committedDelta))
        .style('stroke', COLOR_NULL)
        .attr('stroke-width', 1.2)
        .attr('stroke-dasharray', '4 3');
      g.append('text')
        .attr('x', innerW - 4)
        .attr('y', yScale(committedDelta) - 4)
        .attr('text-anchor', 'end')
        .style('font-family', 'var(--font-mono)')
        .style('font-size', '11px')
        .style('fill', 'var(--color-text-secondary)')
        .text(`nominal δ = ${committedDelta.toFixed(2)}`);

      // step function of rate vs n_max
      const line = d3
        .line<{ nmax: number; rate: number }>()
        .x((d) => xScale(d.nmax))
        .y((d) => yScale(d.rate))
        .curve(d3.curveStepAfter);

      g.append('path')
        .datum(sim.rateByNmax)
        .attr('d', line as never)
        .style('fill', 'none')
        .style('stroke', COLOR_RATE)
        .attr('stroke-width', 2);

      // markers
      g.append('g')
        .selectAll('circle')
        .data(sim.rateByNmax)
        .join('circle')
        .attr('cx', (d) => xScale(d.nmax))
        .attr('cy', (d) => yScale(d.rate))
        .attr('r', 3.5)
        .style('fill', COLOR_RATE);

      // titles + axis labels
      g.append('text')
        .attr('x', 0)
        .attr('y', -8)
        .style('fill', 'var(--color-text)')
        .style('font-family', 'var(--font-sans)')
        .style('font-size', '12px')
        .style('font-weight', 600)
        .text('Empirical false-rejection rate vs analyst patience');

      g.append('text')
        .attr('x', innerW / 2)
        .attr('y', innerH + 30)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-family', 'var(--font-sans)')
        .style('font-size', '11px')
        .text('n_max (log scale)');
    },
    [sim, committedDelta, panelW],
  );

  const pathsPanelRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const innerW = panelW - MARGIN.left - MARGIN.right;
      const innerH = PANEL_H - MARGIN.top - MARGIN.bottom;
      if (innerW <= 0 || sim.paths.length === 0) return;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const z = zHalf(committedDelta);
      const yMax = Math.max(0.6, z / Math.sqrt(10));
      const xScale = d3.scaleLog().domain([10, displayNmax]).range([0, innerW]);
      const yScale = d3.scaleLinear().domain([-yMax, yMax]).range([innerH, 0]);

      // shrinking band (shaded)
      const bandData: { n: number; lo: number; hi: number }[] = [];
      for (let n = 10; n <= displayNmax; n = Math.max(n + 1, Math.floor(n * 1.05))) {
        const hw = z / Math.sqrt(n);
        bandData.push({ n, lo: -hw, hi: hw });
      }
      const area = d3
        .area<{ n: number; lo: number; hi: number }>()
        .x((d) => xScale(d.n))
        .y0((d) => yScale(d.lo))
        .y1((d) => yScale(d.hi))
        .curve(d3.curveMonotoneX);
      g.append('path')
        .datum(bandData)
        .attr('d', area as never)
        .style('fill', COLOR_NULL)
        .style('opacity', 0.12);

      // axes
      g.append('g')
        .attr('transform', `translate(0,${innerH})`)
        .call(d3.axisBottom(xScale).ticks(4, '~s'))
        .selectAll('text')
        .style('fill', 'var(--color-text)');
      g.append('g')
        .call(d3.axisLeft(yScale).ticks(5))
        .selectAll('text')
        .style('fill', 'var(--color-text)');

      // zero line
      g.append('line')
        .attr('x1', 0)
        .attr('x2', innerW)
        .attr('y1', yScale(0))
        .attr('y2', yScale(0))
        .style('stroke', 'var(--color-text-secondary)')
        .attr('stroke-width', 1)
        .attr('opacity', 0.5);

      // trajectories — clipped to displayNmax
      const pathGen = d3
        .line<number>()
        .x((_, i) => xScale(i + 1))
        .y((v) => yScale(v))
        .defined((_, i) => i + 1 >= 10 && i + 1 <= displayNmax);

      const N_DRAW = Math.min(sim.paths.length, N_PATHS_SHOWN);
      const okPaths = [];
      const exitPaths = [];
      for (let i = 0; i < N_DRAW; i++) {
        const e = sim.exitedAt[i];
        if (e > 0 && e <= displayNmax) exitPaths.push(i);
        else okPaths.push(i);
      }

      // Draw non-exiters first (so exiters are on top, in red)
      for (const i of okPaths) {
        const path = sim.paths[i];
        g.append('path')
          .datum(Array.from(path) as number[])
          .attr('d', pathGen as never)
          .style('fill', 'none')
          .style('stroke', COLOR_OK)
          .attr('stroke-width', 0.6)
          .attr('opacity', 0.45);
      }
      for (const i of exitPaths) {
        const path = sim.paths[i];
        g.append('path')
          .datum(Array.from(path) as number[])
          .attr('d', pathGen as never)
          .style('fill', 'none')
          .style('stroke', COLOR_EXIT)
          .attr('stroke-width', 0.7)
          .attr('opacity', 0.6);
      }

      // band-edge lines
      g.append('path')
        .datum(bandData)
        .attr(
          'd',
          d3
            .line<{ n: number; hi: number }>()
            .x((d) => xScale(d.n))
            .y((d) => yScale(d.hi)) as never,
        )
        .style('fill', 'none')
        .style('stroke', COLOR_NULL)
        .attr('stroke-width', 1.2)
        .attr('stroke-dasharray', '2 3');
      g.append('path')
        .datum(bandData)
        .attr(
          'd',
          d3
            .line<{ n: number; lo: number }>()
            .x((d) => xScale(d.n))
            .y((d) => yScale(d.lo)) as never,
        )
        .style('fill', 'none')
        .style('stroke', COLOR_NULL)
        .attr('stroke-width', 1.2)
        .attr('stroke-dasharray', '2 3');

      // title + labels
      g.append('text')
        .attr('x', 0)
        .attr('y', -8)
        .style('fill', 'var(--color-text)')
        .style('font-family', 'var(--font-sans)')
        .style('font-size', '12px')
        .style('font-weight', 600)
        .text(`${N_DRAW} trajectories — ${exitPaths.length} exit, ${okPaths.length} survive`);
      g.append('text')
        .attr('x', innerW / 2)
        .attr('y', innerH + 30)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-family', 'var(--font-sans)')
        .style('font-size', '11px')
        .text('n (log scale)');
    },
    [sim, displayNmax, committedDelta, panelW],
  );

  const rateAtCommitted = sim.rateByNmax.find((d) => d.nmax === committedNmax)?.rate ?? 0;

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: 'var(--font-sans)' }}>
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '16px',
          alignItems: 'center',
          padding: '12px 0',
          fontSize: '13px',
        }}
      >
        <label style={{ display: 'flex', flexDirection: 'column', minWidth: '180px' }}>
          <span>
            Horizon n_max = <strong>{displayNmax.toLocaleString()}</strong>
          </span>
          <input
            type="range"
            min={0}
            max={N_MAX_OPTIONS.length - 1}
            step={1}
            value={displayNmaxIdx}
            aria-label="Sample-size horizon"
            onChange={(e) => setDisplayNmaxIdx(Number(e.target.value))}
            onMouseUp={(e) =>
              setCommittedNmaxIdx(Number((e.target as HTMLInputElement).value))
            }
            onTouchEnd={(e) =>
              setCommittedNmaxIdx(Number((e.target as HTMLInputElement).value))
            }
            onKeyUp={(e) =>
              setCommittedNmaxIdx(Number((e.target as HTMLInputElement).value))
            }
          />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', minWidth: '180px' }}>
          <span>
            Nominal δ = <strong>{displayDelta.toFixed(3)}</strong>
          </span>
          <input
            type="range"
            min={0.01}
            max={0.2}
            step={0.005}
            value={displayDelta}
            aria-label="Nominal false-rejection level"
            onChange={(e) => setDisplayDelta(Number(e.target.value))}
            onMouseUp={(e) => setCommittedDelta(Number((e.target as HTMLInputElement).value))}
            onTouchEnd={(e) => setCommittedDelta(Number((e.target as HTMLInputElement).value))}
            onKeyUp={(e) => setCommittedDelta(Number((e.target as HTMLInputElement).value))}
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
          }}
        >
          P(false rejection by n_max) ≈ <strong>{rateAtCommitted.toFixed(3)}</strong>
          {' '}vs nominal δ = {committedDelta.toFixed(3)}
        </div>
      </div>
      <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: '14px' }}>
        <svg ref={ratePanelRef} width={panelW} height={PANEL_H} style={{ maxWidth: '100%', display: 'block' }} />
        <svg ref={pathsPanelRef} width={panelW} height={PANEL_H} style={{ maxWidth: '100%', display: 'block' }} />
      </div>
    </div>
  );
}
