import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { mulberry32, subBernoulliCSEndpoints, prplEbHW } from './shared/always-valid-inference';

// -----------------------------------------------------------------------------
// Viz 6.2 — PrPl-EB beats sub-Bernoulli Beta-mixture on Bernoulli(μ) streams.
//
// Side-by-side on a single Bernoulli(μ) stream of length n_max:
//   (left) sub-Bernoulli Beta-mixture CS endpoints over time
//   (right) PrPl-EB asymptotic half-width over time
// Sub-panel: ratio over time, settling to ≈ 0.5 for small μ.
// -----------------------------------------------------------------------------

const SM_BREAKPOINT = 640;
const PANEL_H = 260;
const MARGIN = { top: 26, right: 14, bottom: 36, left: 50 };
const N_MAX = 5000;
const N_GRID_LEN = 60;

const COLOR_BETA = '#7c3aed';
const COLOR_EB = 'var(--color-accent)';

function buildGrid(): number[] {
  const out: number[] = [];
  const lo = Math.log10(50);
  const hi = Math.log10(N_MAX);
  for (let i = 0; i < N_GRID_LEN; i++) out.push(Math.round(10 ** (lo + ((hi - lo) * i) / (N_GRID_LEN - 1))));
  return Array.from(new Set(out)).sort((a, b) => a - b);
}

interface Sim {
  ng: number[];
  betaHW: number[];
  ebHW: number[];
  ratio: number[];
}

function simulate(mu: number, delta: number): Sim {
  // Generate one Bernoulli(μ) stream of length N_MAX
  const rng = mulberry32(20261101);
  const stream = new Int8Array(N_MAX);
  for (let n = 0; n < N_MAX; n++) stream[n] = rng() < mu ? 1 : 0;

  const ng = buildGrid();
  const betaHW: number[] = [];
  const ebHW: number[] = [];
  const ratio: number[] = [];

  for (const n of ng) {
    let S = 0;
    let SS = 0;
    for (let i = 0; i < n; i++) {
      S += stream[i];
      SS += stream[i] * stream[i];
    }
    const mean = S / n;
    const variance = Math.max(SS / n - mean * mean, 1e-4);
    // Beta-mixture endpoints — fed by observed S directly (NOT expected μ).
    // For viz we use subBernoulliCSEndpoints with muObs = mean (gives the
    // realized HW at this n's data, not the expected-trajectory bound).
    if (mean > 0 && mean < 1) {
      const [lo, hi] = subBernoulliCSEndpoints(n, mean, delta);
      betaHW.push(Math.max(mean - lo, hi - mean));
    } else {
      betaHW.push(NaN);
    }
    ebHW.push(prplEbHW(n, variance, delta));
    const last = betaHW[betaHW.length - 1];
    ratio.push(Number.isFinite(last) && last > 0 ? ebHW[ebHW.length - 1] / last : NaN);
  }
  return { ng, betaHW, ebHW, ratio };
}

function renderPanel(
  svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
  panelW: number,
  ng: number[],
  series: { values: number[]; color: string; label: string }[],
  title: string,
): void {
  svg.selectAll('*').remove();
  const innerW = panelW - MARGIN.left - MARGIN.right;
  const innerH = PANEL_H - MARGIN.top - MARGIN.bottom;
  if (innerW <= 0) return;
  const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);
  const xScale = d3.scaleLog().domain([Math.min(...ng), Math.max(...ng)]).range([0, innerW]);
  const allValues = series.flatMap((s) => s.values).filter(Number.isFinite);
  if (allValues.length === 0) return;
  const yMax = Math.max(...allValues) * 1.1;
  const yMin = Math.max(Math.min(...allValues) * 0.5, 1e-4);
  const yScale = d3.scaleLog().domain([yMin, yMax]).range([innerH, 0]);

  g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xScale).ticks(4, '~s')).selectAll('text').style('fill', 'var(--color-text)');
  g.append('g').call(d3.axisLeft(yScale).ticks(4, '~s')).selectAll('text').style('fill', 'var(--color-text)');

  series.forEach((s) => {
    const pts: [number, number][] = ng.map((n, i) => [n, s.values[i]]).filter(([, v]) => Number.isFinite(v) && v > 0) as [number, number][];
    const line = d3.line<[number, number]>().x((d) => xScale(d[0])).y((d) => yScale(d[1]));
    g.append('path').datum(pts).attr('d', line as never).style('fill', 'none').style('stroke', s.color).attr('stroke-width', 2);
    g.append('text').attr('x', innerW - 4).attr('y', 12 + series.indexOf(s) * 14).attr('text-anchor', 'end').style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', s.color).text(s.label);
  });

  g.append('text').attr('x', 0).attr('y', -10).style('fill', 'var(--color-text)').style('font-family', 'var(--font-sans)').style('font-size', '12px').style('font-weight', 600).text(title);
}

export default function EmpiricalBernsteinVsBetaMixture() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [displayMu, setDisplayMu] = useState(0.05);
  const [committedMu, setCommittedMu] = useState(0.05);
  const [delta, setDelta] = useState(0.05);

  const sim = useMemo(() => simulate(committedMu, delta), [committedMu, delta]);

  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const w = Math.max(280, Math.min(containerWidth || 720, 820));
  const panelW = isMobile ? w : Math.floor(w / 2) - 8;

  const leftRef = useD3<SVGSVGElement>(
    (svg) =>
      renderPanel(svg, panelW, sim.ng, [
        { values: sim.betaHW, color: COLOR_BETA, label: 'Beta-mixture (5.14)' },
        { values: sim.ebHW, color: COLOR_EB, label: 'PrPl-EB (6.7)' },
      ], `Half-widths on Bern(${committedMu})`),
    [sim, panelW, committedMu],
  );

  const rightRef = useD3<SVGSVGElement>(
    (svg) =>
      renderPanel(svg, panelW, sim.ng, [
        { values: sim.ratio, color: 'var(--color-text)', label: 'EB / Beta-mix' },
      ], 'PrPl-EB / Beta-mixture ratio'),
    [sim, panelW],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: 'var(--font-sans)' }}>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '14px', padding: '10px 0', fontSize: '13px' }}>
        <label style={{ display: 'flex', flexDirection: 'column', minWidth: '180px' }}>
          <span>Bernoulli mean μ = <strong>{displayMu.toFixed(2)}</strong></span>
          <input
            type="range"
            min={0.01}
            max={0.50}
            step={0.01}
            value={displayMu}
            aria-label="Bernoulli mean"
            onChange={(e) => setDisplayMu(Number(e.target.value))}
            onMouseUp={(e) => setCommittedMu(Number((e.target as HTMLInputElement).value))}
            onTouchEnd={(e) => setCommittedMu(Number((e.target as HTMLInputElement).value))}
            onKeyUp={(e) => setCommittedMu(Number((e.target as HTMLInputElement).value))}
          />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', minWidth: '140px' }}>
          <span>δ = <strong>{delta.toFixed(3)}</strong></span>
          <input type="range" min={0.01} max={0.10} step={0.01} value={delta} aria-label="Confidence level" onChange={(e) => setDelta(Number(e.target.value))} />
        </label>
      </div>
      <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: '14px' }}>
        <svg ref={leftRef} width={panelW} height={PANEL_H} style={{ maxWidth: '100%', display: 'block' }} />
        <svg ref={rightRef} width={panelW} height={PANEL_H} style={{ maxWidth: '100%', display: 'block' }} />
      </div>
    </div>
  );
}
