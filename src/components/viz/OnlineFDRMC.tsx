import { useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';

// Locked notebook outputs from cell 50 (§11.5 Monte Carlo, B = 100 streams).
type Row = { proc: 'α-investing' | 'LORD' | 'SAFFRON'; t: number; eFdp: number; pTail: number; eRej: number };
const ROWS: Row[] = [
  { proc: 'α-investing', t: 100, eFdp: 0.0333, pTail: 0.04, eRej: 0.38 },
  { proc: 'α-investing', t: 250, eFdp: 0.0333, pTail: 0.04, eRej: 0.38 },
  { proc: 'α-investing', t: 500, eFdp: 0.0333, pTail: 0.04, eRej: 0.38 },
  { proc: 'α-investing', t: 750, eFdp: 0.0333, pTail: 0.04, eRej: 0.38 },
  { proc: 'α-investing', t: 1000, eFdp: 0.0333, pTail: 0.04, eRej: 0.38 },
  { proc: 'LORD', t: 100, eFdp: 0.0308, pTail: 0.08, eRej: 1.56 },
  { proc: 'LORD', t: 250, eFdp: 0.0238, pTail: 0.15, eRej: 4.26 },
  { proc: 'LORD', t: 500, eFdp: 0.0272, pTail: 0.24, eRej: 8.88 },
  { proc: 'LORD', t: 750, eFdp: 0.0336, pTail: 0.31, eRej: 13.99 },
  { proc: 'LORD', t: 1000, eFdp: 0.0341, pTail: 0.33, eRej: 19.6 },
  { proc: 'SAFFRON', t: 100, eFdp: 0.02, pTail: 0.02, eRej: 0.96 },
  { proc: 'SAFFRON', t: 250, eFdp: 0.02, pTail: 0.02, eRej: 1.57 },
  { proc: 'SAFFRON', t: 500, eFdp: 0.0117, pTail: 0.03, eRej: 2.34 },
  { proc: 'SAFFRON', t: 750, eFdp: 0.0083, pTail: 0.03, eRej: 3.0 },
  { proc: 'SAFFRON', t: 1000, eFdp: 0.0083, pTail: 0.03, eRej: 3.58 },
];

const PROCS = [
  { name: 'α-investing' as const, color: '#dc2626' },
  { name: 'LORD' as const, color: '#16a34a' },
  { name: 'SAFFRON' as const, color: '#1d4ed8' },
];
const TS = [100, 250, 500, 750, 1000];
const ALPHA = 0.05;

export default function OnlineFDRMC() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [active, setActive] = useState<Record<string, boolean>>({ 'α-investing': true, LORD: true, SAFFRON: true });
  const [highlightT, setHighlightT] = useState<number>(1000);

  const W = Math.max(280, Math.min(containerWidth || 720, 720));
  const isMobile = W < 640;
  const PANEL_W = isMobile ? W : (W - 16) / 2;
  const PANEL_H = Math.round(PANEL_W * 0.72);
  const Mg = { t: 20, r: 16, b: 38, l: 50 };

  function rowsFor(proc: string): Row[] {
    return ROWS.filter((r) => r.proc === proc);
  }

  const fdrRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const iw = PANEL_W - Mg.l - Mg.r;
      const ih = PANEL_H - Mg.t - Mg.b;
      if (iw <= 0 || ih <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${Mg.l},${Mg.t})`);
      const x = d3.scaleLinear().domain([0, 1100]).range([0, iw]);
      const y = d3.scaleLinear().domain([0, 0.08]).range([ih, 0]);

      g.append('line').attr('x1', 0).attr('x2', iw).attr('y1', y(ALPHA)).attr('y2', y(ALPHA)).style('stroke', 'var(--color-text)').attr('stroke-dasharray', '4 3').attr('opacity', 0.5);
      g.append('text').attr('x', iw - 4).attr('y', y(ALPHA) - 4).attr('text-anchor', 'end').style('fill', 'var(--color-text-secondary)').style('font-size', '0.68rem').text(`α = ${ALPHA}`);

      for (const p of PROCS) {
        if (!active[p.name]) continue;
        const rows = rowsFor(p.name);
        const line = d3.line<Row>().x((r) => x(r.t)).y((r) => y(r.eFdp));
        g.append('path').datum(rows).attr('d', line).attr('fill', 'none').attr('stroke', p.color).attr('stroke-width', 1.6);
        for (const r of rows) {
          g.append('circle').attr('cx', x(r.t)).attr('cy', y(r.eFdp)).attr('r', r.t === highlightT ? 5 : 3).attr('fill', p.color);
        }
      }

      g.append('g').attr('transform', `translate(0,${ih})`).call(d3.axisBottom(x).tickValues(TS)).selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(y).ticks(4)).selectAll('text').style('fill', 'var(--color-text)');
      g.selectAll('.domain, line').style('stroke', 'var(--color-border)');
      g.append('text').attr('x', iw / 2).attr('y', ih + 28).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-size', '0.7rem').text('stream position t');
      g.append('text').attr('x', 8).attr('y', 10).style('fill', 'var(--color-text-secondary)').style('font-size', '0.72rem').text('E[FDP] (B = 100 streams)');
    },
    [PANEL_W, PANEL_H, active, highlightT],
  );

  const rejRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const iw = PANEL_W - Mg.l - Mg.r;
      const ih = PANEL_H - Mg.t - Mg.b;
      if (iw <= 0 || ih <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${Mg.l},${Mg.t})`);
      const x = d3.scaleLinear().domain([0, 1100]).range([0, iw]);
      const y = d3.scaleLinear().domain([0, 22]).range([ih, 0]);

      for (const p of PROCS) {
        if (!active[p.name]) continue;
        const rows = rowsFor(p.name);
        const line = d3.line<Row>().x((r) => x(r.t)).y((r) => y(r.eRej));
        g.append('path').datum(rows).attr('d', line).attr('fill', 'none').attr('stroke', p.color).attr('stroke-width', 1.6);
        for (const r of rows) {
          g.append('circle').attr('cx', x(r.t)).attr('cy', y(r.eRej)).attr('r', r.t === highlightT ? 5 : 3).attr('fill', p.color);
        }
      }

      g.append('g').attr('transform', `translate(0,${ih})`).call(d3.axisBottom(x).tickValues(TS)).selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(y).ticks(5)).selectAll('text').style('fill', 'var(--color-text)');
      g.selectAll('.domain, line').style('stroke', 'var(--color-border)');
      g.append('text').attr('x', iw / 2).attr('y', ih + 28).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-size', '0.7rem').text('stream position t');
      g.append('text').attr('x', 8).attr('y', 10).style('fill', 'var(--color-text-secondary)').style('font-size', '0.72rem').text('E[#rej] (B = 100 streams)');
    },
    [PANEL_W, PANEL_H, active, highlightT],
  );

  return (
    <div ref={containerRef} className="not-prose my-6 w-full">
      <div className="flex flex-wrap items-end gap-3 text-sm" style={{ color: 'var(--color-text-secondary)' }}>
        {PROCS.map((p) => (
          <label key={p.name} className="flex items-center gap-1">
            <input type="checkbox" checked={active[p.name]} onChange={(e) => setActive({ ...active, [p.name]: e.target.checked })} />
            <span style={{ color: p.color }}>■</span>
            <span>{p.name}</span>
          </label>
        ))}
        <label className="flex flex-col">
          <span>highlight t = <span className="tabular-nums" style={{ color: 'var(--color-text)' }}>{highlightT}</span></span>
          <select value={highlightT} onChange={(e) => setHighlightT(parseInt(e.target.value, 10))} className="rounded border px-2 py-1" style={{ background: 'var(--color-surface)', borderColor: 'var(--color-border)', color: 'var(--color-text)' }}>
            {TS.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className={isMobile ? 'mt-2' : 'mt-2 flex gap-4'}>
        <svg ref={fdrRef} width={PANEL_W} height={PANEL_H} role="img" aria-label="Monte Carlo E[FDP] vs stream position" />
        <svg ref={rejRef} width={PANEL_W} height={PANEL_H} role="img" aria-label="Monte Carlo E[#rej] vs stream position" />
      </div>
      <div className="mt-1 text-xs" style={{ color: 'var(--color-text-secondary)' }}>
        At t = {highlightT}:{' '}
        {PROCS.filter((p) => active[p.name]).map((p) => {
          const r = ROWS.find((row) => row.proc === p.name && row.t === highlightT)!;
          return (
            <span key={p.name} className="mr-3">
              <span style={{ color: p.color }}>■</span> {p.name}: E[FDP] = {r.eFdp.toFixed(3)}, E[#rej] = {r.eRej.toFixed(2)}
            </span>
          );
        })}
      </div>
      <p className="mt-1 text-xs" style={{ color: 'var(--color-text-secondary)' }}>
        All three procedures hold E[FDP] under α = 0.05 across every stream-position checkpoint — Theorem 11.2 / 11.3 verified
        in finite samples. The E[#rej] ordering on this toy is LORD ≫ SAFFRON ≫ α-investing, reversing the publication-figure
        SAFFRON-dominant pattern. The §11.5 prose unpacks why: the spending-schedule + candidacy-threshold combination favors
        LORD's rejection-bonus structure on streams with bursty signals.
      </p>
    </div>
  );
}
