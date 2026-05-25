import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { mulberry32, gaussianFrom, gaussianMixtureHW, zHalf } from './shared/always-valid-inference';

// -----------------------------------------------------------------------------
// Viz 4.2 — Mixture CS fixes the peeking problem.
//
// Side-by-side comparison of cumulative empirical false-rejection rate under
// the aggressive-analyst stopping rule:
//   (left)  fixed-n Gaussian band — climbs past 60% by n_max = 10,000
//   (right) Gaussian-mixture CS (4.4) — flat at ≈ 0.03 below δ = 0.05
//
// Slider for ρ² controls the mixture; this is the closing-the-loop figure
// on §1's peeking demonstration.
// -----------------------------------------------------------------------------

const SM_BREAKPOINT = 640;
const MARGIN = { top: 26, right: 14, bottom: 38, left: 50 };
const PANEL_H = 280;
const N_MAX = 10000;
const B_REPLICATES = 2000;
const N_GRID = [50, 100, 200, 500, 1000, 2000, 5000, 10000];
const SIGMA = 1;

const COLOR_FIXED = '#c44545';
const COLOR_MIX = 'var(--color-accent)';
const COLOR_NULL = 'var(--color-text-secondary)';

interface SimData {
  fixedRate: { n: number; rate: number }[];
  mixRate: { n: number; rate: number }[];
}

function simulate(delta: number, rhoSq: number): SimData {
  const rng = mulberry32(20260801);
  const gauss = gaussianFrom(rng);
  const z = zHalf(delta);
  // For each replicate: track whether bar X_n has ever exited the fixed-n band
  // and whether it has ever exited the mixture CS by step n.
  const firstExitFixed = new Int32Array(B_REPLICATES).fill(-1);
  const firstExitMix = new Int32Array(B_REPLICATES).fill(-1);

  for (let b = 0; b < B_REPLICATES; b++) {
    let s = 0;
    for (let n = 1; n <= N_MAX; n++) {
      s += gauss();
      const mean = s / n;
      const halfFixed = z / Math.sqrt(n);
      const halfMix = gaussianMixtureHW(n, rhoSq, delta, SIGMA);
      if (firstExitFixed[b] === -1 && Math.abs(mean) >= halfFixed) firstExitFixed[b] = n;
      if (firstExitMix[b] === -1 && Math.abs(mean) >= halfMix) firstExitMix[b] = n;
      // continue to maxN to collect both stat trajectories
    }
  }

  const fixedRate: { n: number; rate: number }[] = [];
  const mixRate: { n: number; rate: number }[] = [];
  for (const nmax of N_GRID) {
    let cf = 0;
    let cm = 0;
    for (let b = 0; b < B_REPLICATES; b++) {
      if (firstExitFixed[b] > 0 && firstExitFixed[b] <= nmax) cf++;
      if (firstExitMix[b] > 0 && firstExitMix[b] <= nmax) cm++;
    }
    fixedRate.push({ n: nmax, rate: cf / B_REPLICATES });
    mixRate.push({ n: nmax, rate: cm / B_REPLICATES });
  }

  return { fixedRate, mixRate };
}

function renderPanel(
  svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
  panelW: number,
  data: { n: number; rate: number }[],
  delta: number,
  color: string,
  title: string,
): void {
  svg.selectAll('*').remove();
  const innerW = panelW - MARGIN.left - MARGIN.right;
  const innerH = PANEL_H - MARGIN.top - MARGIN.bottom;
  if (innerW <= 0) return;
  const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

  const xScale = d3.scaleLog().domain([50, 10000]).range([0, innerW]);
  const yScale = d3.scaleLinear().domain([0, 1]).range([innerH, 0]);

  g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xScale).ticks(4, '~s')).selectAll('text').style('fill', 'var(--color-text)');
  g.append('g').call(d3.axisLeft(yScale).tickFormat(d3.format('.0%'))).selectAll('text').style('fill', 'var(--color-text)');

  // δ line
  g.append('line').attr('x1', 0).attr('x2', innerW).attr('y1', yScale(delta)).attr('y2', yScale(delta)).style('stroke', COLOR_NULL).attr('stroke-width', 1).attr('stroke-dasharray', '4 3');
  g.append('text').attr('x', innerW - 4).attr('y', yScale(delta) - 4).attr('text-anchor', 'end').style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', COLOR_NULL).text(`δ = ${delta.toFixed(2)}`);

  const line = d3.line<{ n: number; rate: number }>().x((d) => xScale(d.n)).y((d) => yScale(d.rate)).curve(d3.curveStepAfter);
  g.append('path').datum(data).attr('d', line as never).style('fill', 'none').style('stroke', color).attr('stroke-width', 2);
  g.append('g').selectAll('circle').data(data).join('circle').attr('cx', (d) => xScale(d.n)).attr('cy', (d) => yScale(d.rate)).attr('r', 3.5).style('fill', color);

  g.append('text').attr('x', 0).attr('y', -10).style('fill', 'var(--color-text)').style('font-family', 'var(--font-sans)').style('font-size', '12px').style('font-weight', 600).text(title);
  g.append('text').attr('x', innerW / 2).attr('y', innerH + 30).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text('n_max (log scale)');
}

export default function MixtureCSFixesPeeking() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();

  const [displayRhoSq, setDisplayRhoSq] = useState(1);
  const [committedRhoSq, setCommittedRhoSq] = useState(1);
  const [delta, setDelta] = useState(0.05);

  const sim = useMemo(() => simulate(delta, committedRhoSq), [delta, committedRhoSq]);

  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const w = Math.max(280, Math.min(containerWidth || 720, 820));
  const panelW = isMobile ? w : Math.floor(w / 2) - 8;

  const leftRef = useD3<SVGSVGElement>(
    (svg) => renderPanel(svg, panelW, sim.fixedRate, delta, COLOR_FIXED, 'Fixed-n Gaussian band (anytime-invalid)'),
    [sim, delta, panelW],
  );
  const rightRef = useD3<SVGSVGElement>(
    (svg) => renderPanel(svg, panelW, sim.mixRate, delta, COLOR_MIX, `Gaussian mixture (ρ² = ${committedRhoSq})`),
    [sim, delta, panelW, committedRhoSq],
  );

  const maxFixed = Math.max(...sim.fixedRate.map((d) => d.rate));
  const maxMix = Math.max(...sim.mixRate.map((d) => d.rate));

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: 'var(--font-sans)' }}>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '14px', padding: '10px 0', fontSize: '13px' }}>
        <label style={{ display: 'flex', flexDirection: 'column', minWidth: '200px' }}>
          <span>Mixing ρ² = <strong>{displayRhoSq.toFixed(2)}</strong></span>
          <input
            type="range"
            min={0.1}
            max={10}
            step={0.1}
            value={displayRhoSq}
            aria-label="Mixing prior variance"
            onChange={(e) => setDisplayRhoSq(Number(e.target.value))}
            onMouseUp={(e) => setCommittedRhoSq(Number((e.target as HTMLInputElement).value))}
            onTouchEnd={(e) => setCommittedRhoSq(Number((e.target as HTMLInputElement).value))}
            onKeyUp={(e) => setCommittedRhoSq(Number((e.target as HTMLInputElement).value))}
          />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', minWidth: '160px' }}>
          <span>δ = <strong>{delta.toFixed(3)}</strong></span>
          <input type="range" min={0.01} max={0.20} step={0.005} value={delta} aria-label="Confidence level" onChange={(e) => setDelta(Number(e.target.value))} />
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
          Max false-rejection over n ∈ [50, 10000]:{' '}
          fixed-n = <strong style={{ color: COLOR_FIXED }}>{maxFixed.toFixed(3)}</strong>,{' '}
          mixture = <strong style={{ color: COLOR_MIX }}>{maxMix.toFixed(3)}</strong>{' '}
          (target δ = {delta.toFixed(2)}).
        </div>
      </div>
      <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: '14px' }}>
        <svg ref={leftRef} width={panelW} height={PANEL_H} style={{ maxWidth: '100%', display: 'block' }} />
        <svg ref={rightRef} width={panelW} height={PANEL_H} style={{ maxWidth: '100%', display: 'block' }} />
      </div>
    </div>
  );
}
