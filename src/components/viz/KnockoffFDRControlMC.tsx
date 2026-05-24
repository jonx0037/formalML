import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';

// Locked outputs from cell 33 (fixed-X §7) and cell 38 (model-X §8).
type Anchor = { q: number; fdp: number; sdFdp: number; power: number; nRej: number };

const FIXED_X: Anchor[] = [
  { q: 0.05, fdp: 0.0, sdFdp: 0.0, power: 0.0, nRej: 0.0 },
  { q: 0.1, fdp: 0.0184, sdFdp: 0.1063, power: 0.014, nRej: 0.36 },
  { q: 0.2, fdp: 0.0929, sdFdp: 0.1896, power: 0.117, nRej: 2.03 },
];
const FIXED_X_META = { n: 500, p: 200, s0: 10, beta: 2.5, design: 'identity', stat: 'LCD at λ_kf = 0.01 σ √(2 log p / n)' };

const MODEL_X: Anchor[] = [
  { q: 0.05, fdp: 0.0, sdFdp: 0.0, power: 0.0, nRej: 0.0 },
  { q: 0.1, fdp: 0.0594, sdFdp: 0.083, power: 1.0, nRej: 10.74 },
  { q: 0.2, fdp: 0.1588, sdFdp: 0.1446, power: 1.0, nRej: 12.32 },
];
const MODEL_X_META = { n: 200, p: 500, s0: 10, beta: 3.5, design: 'AR(1) ρ = 0.3', stat: 'LCD on augmented lasso, CV-tuned λ' };

// Interpolate FDP and power at arbitrary q ∈ [0.05, 0.25]. Linear within [0.05, 0.2];
// extrapolate gently outside.
function interp(anchors: Anchor[], q: number, key: 'fdp' | 'power'): number {
  const qs = anchors.map((a) => a.q);
  const ys = anchors.map((a) => a[key]);
  if (q <= qs[0]) return ys[0];
  if (q >= qs[qs.length - 1]) return Math.min(1, ys[ys.length - 1] + ((q - qs[qs.length - 1]) / (qs[qs.length - 1] - qs[qs.length - 2])) * (ys[ys.length - 1] - ys[ys.length - 2]));
  for (let i = 1; i < qs.length; i++) {
    if (q <= qs[i]) {
      const t = (q - qs[i - 1]) / (qs[i] - qs[i - 1]);
      return ys[i - 1] + t * (ys[i] - ys[i - 1]);
    }
  }
  return ys[ys.length - 1];
}

export default function KnockoffFDRControlMC() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [regime, setRegime] = useState<'fixed' | 'model'>('model');
  const [q, setQ] = useState(0.1);

  const anchors = regime === 'fixed' ? FIXED_X : MODEL_X;
  const meta = regime === 'fixed' ? FIXED_X_META : MODEL_X_META;
  const fdpAtQ = interp(anchors, q, 'fdp');
  const powerAtQ = interp(anchors, q, 'power');
  const isControlled = fdpAtQ <= q + 1e-6;

  const W = Math.max(280, Math.min(containerWidth || 720, 720));
  const isMobile = W < 640;
  const PANEL_W = isMobile ? W : (W - 16) / 2;
  const PANEL_H = Math.round(PANEL_W * 0.72);
  const Mg = { t: 20, r: 18, b: 38, l: 50 };

  const fdrRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const iw = PANEL_W - Mg.l - Mg.r;
      const ih = PANEL_H - Mg.t - Mg.b;
      if (iw <= 0 || ih <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${Mg.l},${Mg.t})`);
      const x = d3.scaleLinear().domain([0, 0.3]).range([0, iw]);
      const y = d3.scaleLinear().domain([0, 0.3]).range([ih, 0]);

      // y = q reference line
      const refLine = d3.line<[number, number]>().x((d) => x(d[0])).y((d) => y(d[1]));
      g.append('path').datum([[0, 0], [0.3, 0.3]] as [number, number][]).attr('d', refLine).attr('fill', 'none').style('stroke', 'var(--color-text)').attr('stroke-dasharray', '4 3').attr('opacity', 0.6);
      g.append('text').attr('x', iw - 4).attr('y', y(0.3) + 14).attr('text-anchor', 'end').style('fill', 'var(--color-text-secondary)').style('font-size', '0.68rem').text('y = q');

      // anchor MC points with error bars
      for (const a of anchors) {
        g.append('line').attr('x1', x(a.q)).attr('x2', x(a.q)).attr('y1', y(Math.min(0.3, a.fdp + a.sdFdp / Math.sqrt(100)))).attr('y2', y(Math.max(0, a.fdp - a.sdFdp / Math.sqrt(100)))).style('stroke', 'var(--color-accent)').attr('stroke-width', 1.5).attr('opacity', 0.6);
        g.append('circle').attr('cx', x(a.q)).attr('cy', y(a.fdp)).attr('r', 4).style('fill', 'var(--color-accent)');
      }
      // smooth curve through anchors
      const sweep = d3.range(0.05, 0.301, 0.01);
      const line = d3.line<number>().x((qv) => x(qv)).y((qv) => y(interp(anchors, qv, 'fdp')));
      g.append('path').datum(sweep as unknown as number[]).attr('d', line).attr('fill', 'none').style('stroke', 'var(--color-accent)').attr('stroke-width', 1.4).attr('opacity', 0.7);

      // current q marker
      g.append('line').attr('x1', x(q)).attr('x2', x(q)).attr('y1', 0).attr('y2', ih).style('stroke', 'var(--color-text)').attr('stroke-dasharray', '2 4').attr('opacity', 0.6);
      g.append('circle').attr('cx', x(q)).attr('cy', y(fdpAtQ)).attr('r', 5).style('fill', 'var(--color-text)');

      g.append('g').attr('transform', `translate(0,${ih})`).call(d3.axisBottom(x).ticks(5)).selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(y).ticks(5)).selectAll('text').style('fill', 'var(--color-text)');
      g.selectAll('.domain, line').style('stroke', 'var(--color-border)');
      g.append('text').attr('x', iw / 2).attr('y', ih + 28).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-size', '0.7rem').text('nominal q');
      g.append('text').attr('x', 8).attr('y', 10).style('fill', 'var(--color-text-secondary)').style('font-size', '0.72rem').text(`empirical FDR (${regime === 'fixed' ? 'fixed-X' : 'model-X'})`);
    },
    [PANEL_W, PANEL_H, anchors, q, regime, fdpAtQ],
  );

  const powerRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const iw = PANEL_W - Mg.l - Mg.r;
      const ih = PANEL_H - Mg.t - Mg.b;
      if (iw <= 0 || ih <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${Mg.l},${Mg.t})`);
      const x = d3.scaleLinear().domain([0, 0.3]).range([0, iw]);
      const y = d3.scaleLinear().domain([0, 1]).range([ih, 0]);

      for (const a of anchors) {
        g.append('circle').attr('cx', x(a.q)).attr('cy', y(a.power)).attr('r', 4).style('fill', 'var(--color-accent)');
      }
      const sweep = d3.range(0.05, 0.301, 0.01);
      const line = d3.line<number>().x((qv) => x(qv)).y((qv) => y(interp(anchors, qv, 'power')));
      g.append('path').datum(sweep as unknown as number[]).attr('d', line).attr('fill', 'none').style('stroke', 'var(--color-accent)').attr('stroke-width', 1.4).attr('opacity', 0.7);

      g.append('line').attr('x1', x(q)).attr('x2', x(q)).attr('y1', 0).attr('y2', ih).style('stroke', 'var(--color-text)').attr('stroke-dasharray', '2 4').attr('opacity', 0.6);
      g.append('circle').attr('cx', x(q)).attr('cy', y(powerAtQ)).attr('r', 5).style('fill', 'var(--color-text)');

      g.append('g').attr('transform', `translate(0,${ih})`).call(d3.axisBottom(x).ticks(5)).selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(y).ticks(5)).selectAll('text').style('fill', 'var(--color-text)');
      g.selectAll('.domain, line').style('stroke', 'var(--color-border)');
      g.append('text').attr('x', iw / 2).attr('y', ih + 28).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-size', '0.7rem').text('nominal q');
      g.append('text').attr('x', 8).attr('y', 10).style('fill', 'var(--color-text-secondary)').style('font-size', '0.72rem').text('empirical power');
    },
    [PANEL_W, PANEL_H, anchors, q, powerAtQ],
  );

  return (
    <div ref={containerRef} className="not-prose my-6 w-full">
      <div className="flex flex-wrap items-end gap-4 text-sm" style={{ color: 'var(--color-text-secondary)' }}>
        <label className="flex items-center gap-2">
          <span>regime</span>
          <select value={regime} onChange={(e) => setRegime(e.target.value as 'fixed' | 'model')} className="rounded border px-2 py-1" style={{ background: 'var(--color-surface)', borderColor: 'var(--color-border)', color: 'var(--color-text)' }}>
            <option value="fixed">fixed-X (§7)</option>
            <option value="model">model-X (§8)</option>
          </select>
        </label>
        <label className="flex flex-col">
          <span>q = <span className="tabular-nums" style={{ color: 'var(--color-text)' }}>{q.toFixed(2)}</span></span>
          <input type="range" min={0.05} max={0.3} step={0.01} value={q} onChange={(e) => setQ(parseFloat(e.target.value))} aria-label="Nominal FDR level q" className="w-56" />
        </label>
        <div className="flex flex-col text-xs">
          <span>n = {meta.n} · p = {meta.p} · s₀ = {meta.s0} · β★ = {meta.beta}</span>
          <span>{meta.design} · {meta.stat}</span>
        </div>
      </div>
      <div className={isMobile ? 'mt-2' : 'mt-2 flex gap-4'}>
        <svg ref={fdrRef} width={PANEL_W} height={PANEL_H} role="img" aria-label="Empirical FDR vs nominal q" />
        <svg ref={powerRef} width={PANEL_W} height={PANEL_H} role="img" aria-label="Empirical power vs nominal q" />
      </div>
      <p className="mt-1 text-xs" style={{ color: 'var(--color-text-secondary)' }}>
        At q = {q.toFixed(2)}, empirical FDR ≈ {fdpAtQ.toFixed(3)} ({isControlled ? <span style={{ color: 'var(--color-accent)' }}>controlled</span> : <span style={{ color: '#dc2626' }}>over budget</span>}) with power ≈ {powerAtQ.toFixed(3)}. Anchor points are the
        Monte Carlo means from the §7 and §8 notebook runs (B = 100 replicates each); the connecting curves are linear
        interpolations between locked anchors. The FDR≤q guarantee holds in both regimes; power separation reflects the
        joint impact of signal strength, design correlation, and W-statistic choice.
      </p>
    </div>
  );
}
