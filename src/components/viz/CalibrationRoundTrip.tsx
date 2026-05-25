import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { mulberry32 } from './shared/always-valid-inference';

// -----------------------------------------------------------------------------
// Viz 8.2 — Calibration round-trip.
//
// (left) histogram of P = min(1, 1/E) from "betting-wealth-like" e-values
//        under H_0 — looks ≥ Uniform(0,1) (valid p-values, Lemma 8.7).
// (right) Histograms of three p-to-e calibrators applied to uniform p-values:
//        Vovk-κ for two κ values, plus Shafer logarithmic. Means should
//        equal 1 (Theorem 8.9).
// -----------------------------------------------------------------------------

const SM_BREAKPOINT = 640;
const PANEL_H = 260;
const MARGIN = { top: 26, right: 14, bottom: 36, left: 50 };
const N_SAMPLES = 8000;

const COLOR_HIST = 'var(--color-accent)';

function vovkCalibrator(p: number, kappa: number): number {
  if (p <= 0) return Infinity;
  if (p >= 1) return 0;
  return kappa * Math.pow(p, kappa - 1);
}

function shaferCalibrator(p: number): number {
  if (p <= 0) return Infinity;
  if (p >= 1) return 0;
  const lp = Math.log(p);
  return (1 - p + p * lp) / (p * lp * lp);
}

function simulate(kappa: number): { pSamples: number[]; eVovk: number[]; eVovk2: number[]; eShafer: number[]; meanVovk: number; meanVovk2: number; meanShafer: number } {
  const rng = mulberry32(20261301);
  // Generate uniform null p-values
  const pSamples: number[] = [];
  for (let i = 0; i < N_SAMPLES; i++) pSamples.push(rng());

  const eVovk: number[] = pSamples.map((p) => Math.min(vovkCalibrator(p, kappa), 1e6));
  const eVovk2: number[] = pSamples.map((p) => Math.min(vovkCalibrator(p, 1 / 3), 1e6));
  const eShafer: number[] = pSamples.map((p) => Math.min(shaferCalibrator(p), 1e6));

  const meanVovk = eVovk.reduce((s, v) => s + v, 0) / N_SAMPLES;
  const meanVovk2 = eVovk2.reduce((s, v) => s + v, 0) / N_SAMPLES;
  const meanShafer = eShafer.reduce((s, v) => s + v, 0) / N_SAMPLES;

  return { pSamples, eVovk, eVovk2, eShafer, meanVovk, meanVovk2, meanShafer };
}

function renderHistogram(
  svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
  panelW: number,
  data: number[],
  bounds: [number, number],
  xLabel: string,
  title: string,
  refLines: { x: number; label: string; color: string }[] = [],
): void {
  svg.selectAll('*').remove();
  const innerW = panelW - MARGIN.left - MARGIN.right;
  const innerH = PANEL_H - MARGIN.top - MARGIN.bottom;
  if (innerW <= 0) return;
  const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

  const xScale = d3.scaleLinear().domain(bounds).range([0, innerW]).clamp(true);
  const filtered = data.filter((v) => v >= bounds[0] && v <= bounds[1]);
  const bins = d3.bin().domain(bounds).thresholds(30)(filtered);
  const yMax = Math.max(...bins.map((b) => b.length));
  const yScale = d3.scaleLinear().domain([0, yMax * 1.1]).range([innerH, 0]);

  g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xScale).ticks(5)).selectAll('text').style('fill', 'var(--color-text)');
  g.append('g').call(d3.axisLeft(yScale).ticks(4)).selectAll('text').style('fill', 'var(--color-text)');

  g.append('g').selectAll('rect').data(bins).join('rect')
    .attr('x', (d) => xScale(d.x0 ?? 0) + 1)
    .attr('y', (d) => yScale(d.length))
    .attr('width', (d) => Math.max(xScale(d.x1 ?? 0) - xScale(d.x0 ?? 0) - 1, 0))
    .attr('height', (d) => innerH - yScale(d.length))
    .style('fill', COLOR_HIST)
    .style('opacity', 0.75);

  refLines.forEach((rl) => {
    if (rl.x >= bounds[0] && rl.x <= bounds[1]) {
      g.append('line').attr('x1', xScale(rl.x)).attr('x2', xScale(rl.x)).attr('y1', 0).attr('y2', innerH).style('stroke', rl.color).attr('stroke-width', 1.2).attr('stroke-dasharray', '4 3');
      g.append('text').attr('x', xScale(rl.x) + 3).attr('y', 10).style('font-family', 'var(--font-mono)').style('font-size', '10px').style('fill', rl.color).text(rl.label);
    }
  });

  g.append('text').attr('x', 0).attr('y', -10).style('fill', 'var(--color-text)').style('font-family', 'var(--font-sans)').style('font-size', '12px').style('font-weight', 600).text(title);
  g.append('text').attr('x', innerW / 2).attr('y', innerH + 28).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text(xLabel);
}

export default function CalibrationRoundTrip() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [kappa, setKappa] = useState(0.5);

  const sim = useMemo(() => simulate(kappa), [kappa]);

  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const w = Math.max(280, Math.min(containerWidth || 720, 820));
  const panelW = isMobile ? w : Math.floor(w / 2) - 8;

  const leftRef = useD3<SVGSVGElement>(
    (svg) => renderHistogram(svg, panelW, sim.pSamples, [0, 1], 'p ~ Uniform(0,1)', 'p-value histogram (null)'),
    [sim, panelW],
  );
  const rightRef = useD3<SVGSVGElement>(
    (svg) =>
      renderHistogram(
        svg,
        panelW,
        sim.eVovk,
        [0, 5],
        'e = f(p)',
        `Vovk-κ calibrator (κ=${kappa.toFixed(2)}): E[e] ≈ ${sim.meanVovk.toFixed(3)}`,
        [{ x: 1, label: 'E = 1', color: 'var(--color-accent)' }],
      ),
    [sim, panelW, kappa],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: 'var(--font-sans)' }}>
      <div style={{ display: 'flex', gap: '14px', padding: '10px 0', fontSize: '13px', flexWrap: 'wrap' }}>
        <label style={{ display: 'flex', flexDirection: 'column', minWidth: '200px' }}>
          <span>Vovk κ = <strong>{kappa.toFixed(2)}</strong></span>
          <input type="range" min={0.1} max={0.9} step={0.05} value={kappa} aria-label="Vovk calibrator parameter" onChange={(e) => setKappa(Number(e.target.value))} />
        </label>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--color-text-secondary)', padding: '6px 10px', border: '1px solid var(--color-border)', borderRadius: '4px' }}>
          Mean e-value under null (Theorem 8.9 says E[E] = 1):{' '}
          Vovk-κ: <strong>{sim.meanVovk.toFixed(3)}</strong>,{' '}
          Vovk-1/3: <strong>{sim.meanVovk2.toFixed(3)}</strong>,{' '}
          Shafer: <strong>{sim.meanShafer.toFixed(3)}</strong>.
        </div>
      </div>
      <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: '14px' }}>
        <svg ref={leftRef} width={panelW} height={PANEL_H} style={{ maxWidth: '100%', display: 'block' }} />
        <svg ref={rightRef} width={panelW} height={PANEL_H} style={{ maxWidth: '100%', display: 'block' }} />
      </div>
    </div>
  );
}
