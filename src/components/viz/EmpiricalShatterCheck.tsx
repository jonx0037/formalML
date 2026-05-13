import { useEffect, useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { paletteVC, sauerShelahBinomialSum } from './shared/vc-dimension';

// =============================================================================
// EmpiricalShatterCheck — §10.5 four-panel "money shot"
//
// Combines all four protocols on half-planes and axis-rectangles:
//   Panel A: empirical Π(n) vs Sauer–Shelah ceilings
//   Panel B: FTSL envelope vs empirical worst-case gap
//   Panel C: realizable sample-complexity, empirical vs theoretical
//   Panel D: empirical Rademacher vs Sauer–Shelah Rademacher at n = 20
// Data fetched from /sample-data/vc-dimension/integrative_monte_carlo.json
// Static fallback: public/images/topics/vc-dimension/10_integrative_money_shot.png
// =============================================================================

const HEIGHT = 380;
const JSON_URL = '/sample-data/vc-dimension/integrative_monte_carlo.json';

type PanelARow = { n: number; Pi: number; SS_ceiling_d3?: number; SS_ceiling_d4?: number };
type PanelBRow = { n: number; bound: number; empirical: number };
type PanelCRow = { eps: number; n_emp_HP: number | null; n_th_HP: number; n_emp_rect: number | null; n_th_rect: number };
type Payload = {
  config: { replicates_protocol_2: number; trials_protocol_3: number; B_rademacher: number; n_test: number; delta: number };
  panel_a_growth: { half_plane: PanelARow[]; rectangle: PanelARow[] };
  panel_b_ftsl_vs_empirical: { half_plane: PanelBRow[]; rectangle: PanelBRow[] };
  panel_c_sample_complexity: PanelCRow[];
  panel_d_rademacher_at_n_20: {
    n: number;
    half_plane: { empirical: number; restricted_size: number; ss_bound: number };
    rectangle: { empirical: number; restricted_size: number; ss_bound: number };
  };
};

function renderPanelA(
  svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
  w: number,
  payload: Payload,
) {
  const margin = { top: 28, right: 14, bottom: 36, left: 50 };
  const W = w - margin.left - margin.right;
  const H = HEIGHT - margin.top - margin.bottom;
  svg.selectAll('*').remove();
  if (W <= 0) return;
  svg.attr('viewBox', `0 0 ${w} ${HEIGHT}`);
  svg.append('text').attr('x', w / 2).attr('y', 16).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '12px').style('font-weight', 'bold').text('Panel A: Π(n) vs Sauer–Shelah ceiling');
  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
  const rowsHP = payload.panel_a_growth.half_plane;
  const rowsRect = payload.panel_a_growth.rectangle;
  const ns = rowsHP.map((r) => r.n);
  const yMax = Math.max(...rowsRect.map((r) => r.SS_ceiling_d4 || sauerShelahBinomialSum(r.n, 4)));
  const x = d3.scaleLinear().domain([Math.min(...ns), Math.max(...ns)]).range([0, W]);
  const y = d3.scaleLog().domain([1, yMax * 1.2]).range([H, 0]);
  g.append('g').attr('transform', `translate(0,${H})`).call(d3.axisBottom(x).ticks(rowsHP.length));
  g.append('g').call(d3.axisLeft(y).ticks(5, '~s'));
  g.append('text').attr('x', W / 2).attr('y', H + 28).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '10px').text('n');
  g.append('text').attr('transform', 'rotate(-90)').attr('x', -H / 2).attr('y', -36).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '10px').text('Π(n) (log)');
  const lineHP = d3.line<PanelARow>().x((r) => x(r.n)).y((r) => y(Math.max(1, r.Pi)));
  const lineSSHP = d3.line<PanelARow>().x((r) => x(r.n)).y((r) => y(Math.max(1, r.SS_ceiling_d3 || sauerShelahBinomialSum(r.n, 3))));
  const lineRect = d3.line<PanelARow>().x((r) => x(r.n)).y((r) => y(Math.max(1, r.Pi)));
  const lineSSRect = d3.line<PanelARow>().x((r) => x(r.n)).y((r) => y(Math.max(1, r.SS_ceiling_d4 || sauerShelahBinomialSum(r.n, 4))));
  g.append('path').datum(rowsHP).attr('d', lineHP).attr('fill', 'none').attr('stroke', paletteVC.primary).attr('stroke-width', 2.2);
  g.append('path').datum(rowsHP).attr('d', lineSSHP).attr('fill', 'none').attr('stroke', paletteVC.primary).attr('stroke-width', 1.4).attr('stroke-dasharray', '4 3');
  g.append('path').datum(rowsRect).attr('d', lineRect).attr('fill', 'none').attr('stroke', paletteVC.emp).attr('stroke-width', 2.2);
  g.append('path').datum(rowsRect).attr('d', lineSSRect).attr('fill', 'none').attr('stroke', paletteVC.emp).attr('stroke-width', 1.4).attr('stroke-dasharray', '4 3');
  rowsHP.forEach((r) => g.append('circle').attr('cx', x(r.n)).attr('cy', y(r.Pi)).attr('r', 3).attr('fill', paletteVC.primary));
  rowsRect.forEach((r) => g.append('circle').attr('cx', x(r.n)).attr('cy', y(r.Pi)).attr('r', 3).attr('fill', paletteVC.emp));
  const legend = g.append('g').attr('transform', `translate(${Math.max(4, W - 130)}, 4)`);
  ['half-plane', 'SS d=3', 'rect', 'SS d=4'].forEach((lbl, i) => {
    const color = i < 2 ? paletteVC.primary : paletteVC.emp;
    const dash = i % 2 === 1 ? '4 3' : '';
    legend.append('line').attr('x1', 0).attr('x2', 14).attr('y1', i * 12 + 4).attr('y2', i * 12 + 4).attr('stroke', color).attr('stroke-width', 1.6).attr('stroke-dasharray', dash);
    legend.append('text').attr('x', 18).attr('y', i * 12 + 7).style('fill', 'var(--color-text)').style('font-size', '9px').text(lbl);
  });
}

function renderPanelB(
  svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
  w: number,
  payload: Payload,
) {
  const margin = { top: 28, right: 14, bottom: 36, left: 50 };
  const W = w - margin.left - margin.right;
  const H = HEIGHT - margin.top - margin.bottom;
  svg.selectAll('*').remove();
  if (W <= 0) return;
  svg.attr('viewBox', `0 0 ${w} ${HEIGHT}`);
  svg.append('text').attr('x', w / 2).attr('y', 16).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '12px').style('font-weight', 'bold').text('Panel B: FTSL envelope vs empirical gap');
  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
  const rowsHP = payload.panel_b_ftsl_vs_empirical.half_plane;
  const rowsRect = payload.panel_b_ftsl_vs_empirical.rectangle;
  const all = [...rowsHP, ...rowsRect];
  const ns = all.map((r) => r.n);
  const allY = all.flatMap((r) => [r.bound, r.empirical]).filter((v) => v > 1e-5);
  const yMin = Math.max(1e-3, Math.min(...allY) * 0.5);
  const yMax = Math.max(...allY) * 1.2;
  const x = d3.scaleLog().domain([Math.min(...ns) * 0.7, Math.max(...ns) * 1.4]).range([0, W]);
  const y = d3.scaleLog().domain([yMin, yMax]).range([H, 0]);
  g.append('g').attr('transform', `translate(0,${H})`).call(d3.axisBottom(x).ticks(5, '~s'));
  g.append('g').call(d3.axisLeft(y).ticks(5, '~g'));
  g.append('text').attr('x', W / 2).attr('y', H + 28).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '10px').text('n (log)');
  g.append('text').attr('transform', 'rotate(-90)').attr('x', -H / 2).attr('y', -36).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '10px').text('gap (log)');
  const lineBoundHP = d3.line<PanelBRow>().x((r) => x(r.n)).y((r) => y(r.bound));
  const lineEmpHP = d3.line<PanelBRow>().x((r) => x(r.n)).y((r) => y(Math.max(1e-5, r.empirical)));
  const lineBoundR = d3.line<PanelBRow>().x((r) => x(r.n)).y((r) => y(r.bound));
  const lineEmpR = d3.line<PanelBRow>().x((r) => x(r.n)).y((r) => y(Math.max(1e-5, r.empirical)));
  g.append('path').datum(rowsHP).attr('d', lineBoundHP).attr('fill', 'none').attr('stroke', paletteVC.primary).attr('stroke-width', 2);
  g.append('path').datum(rowsHP).attr('d', lineEmpHP).attr('fill', 'none').attr('stroke', paletteVC.primary).attr('stroke-width', 1.6).attr('stroke-dasharray', '5 3');
  g.append('path').datum(rowsRect).attr('d', lineBoundR).attr('fill', 'none').attr('stroke', paletteVC.emp).attr('stroke-width', 2);
  g.append('path').datum(rowsRect).attr('d', lineEmpR).attr('fill', 'none').attr('stroke', paletteVC.emp).attr('stroke-width', 1.6).attr('stroke-dasharray', '5 3');
}

function renderPanelC(
  svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
  w: number,
  payload: Payload,
) {
  const margin = { top: 28, right: 14, bottom: 36, left: 50 };
  const W = w - margin.left - margin.right;
  const H = HEIGHT - margin.top - margin.bottom;
  svg.selectAll('*').remove();
  if (W <= 0) return;
  svg.attr('viewBox', `0 0 ${w} ${HEIGHT}`);
  svg.append('text').attr('x', w / 2).attr('y', 16).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '12px').style('font-weight', 'bold').text('Panel C: realizable n*(ε)');
  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
  const rows = payload.panel_c_sample_complexity;
  const epsValues = rows.map((r) => r.eps);
  const all = rows.flatMap((r) => [r.n_emp_HP || r.n_th_HP, r.n_th_HP, r.n_emp_rect || r.n_th_rect, r.n_th_rect]).filter((v) => v > 0);
  const x = d3.scaleLog().domain([Math.min(...epsValues) * 0.7, Math.max(...epsValues) * 1.4]).range([0, W]);
  const y = d3.scaleLog().domain([Math.min(...all) * 0.5, Math.max(...all) * 1.4]).range([H, 0]);
  g.append('g').attr('transform', `translate(0,${H})`).call(d3.axisBottom(x).ticks(3, '~g'));
  g.append('g').call(d3.axisLeft(y).ticks(5, '~s'));
  g.append('text').attr('x', W / 2).attr('y', H + 28).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '10px').text('ε (log)');
  g.append('text').attr('transform', 'rotate(-90)').attr('x', -H / 2).attr('y', -36).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '10px').text('n*(ε) (log)');
  // Plot empirical and theoretical for both classes.
  const lineThHP = d3.line<PanelCRow>().x((r) => x(r.eps)).y((r) => y(r.n_th_HP));
  const lineThR = d3.line<PanelCRow>().x((r) => x(r.eps)).y((r) => y(r.n_th_rect));
  g.append('path').datum(rows).attr('d', lineThHP).attr('fill', 'none').attr('stroke', paletteVC.primary).attr('stroke-width', 2);
  g.append('path').datum(rows).attr('d', lineThR).attr('fill', 'none').attr('stroke', paletteVC.emp).attr('stroke-width', 2);
  rows.forEach((r) => {
    if (r.n_emp_HP) g.append('circle').attr('cx', x(r.eps)).attr('cy', y(r.n_emp_HP)).attr('r', 3).attr('fill', paletteVC.primary);
    if (r.n_emp_rect) g.append('circle').attr('cx', x(r.eps)).attr('cy', y(r.n_emp_rect)).attr('r', 3).attr('fill', paletteVC.emp);
  });
}

function renderPanelD(
  svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
  w: number,
  payload: Payload,
) {
  const margin = { top: 28, right: 14, bottom: 36, left: 50 };
  const W = w - margin.left - margin.right;
  const H = HEIGHT - margin.top - margin.bottom;
  svg.selectAll('*').remove();
  if (W <= 0) return;
  svg.attr('viewBox', `0 0 ${w} ${HEIGHT}`);
  svg.append('text').attr('x', w / 2).attr('y', 16).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '12px').style('font-weight', 'bold').text(`Panel D: Rademacher at n=${payload.panel_d_rademacher_at_n_20.n}`);
  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
  const d = payload.panel_d_rademacher_at_n_20;
  const bars = [
    { lbl: 'HP emp', val: d.half_plane.empirical, color: paletteVC.primary },
    { lbl: 'HP SS', val: d.half_plane.ss_bound, color: paletteVC.primary, dash: '4 3' },
    { lbl: 'rect emp', val: d.rectangle.empirical, color: paletteVC.emp },
    { lbl: 'rect SS', val: d.rectangle.ss_bound, color: paletteVC.emp, dash: '4 3' },
  ];
  const x = d3.scaleBand().domain(bars.map((b) => b.lbl)).range([0, W]).padding(0.2);
  const yMax = Math.max(...bars.map((b) => b.val)) * 1.2;
  const y = d3.scaleLinear().domain([0, yMax]).range([H, 0]);
  g.append('g').attr('transform', `translate(0,${H})`).call(d3.axisBottom(x));
  g.append('g').call(d3.axisLeft(y).ticks(5));
  bars.forEach((b) => {
    g.append('rect').attr('x', x(b.lbl) || 0).attr('y', y(b.val)).attr('width', x.bandwidth()).attr('height', H - y(b.val)).attr('fill', b.color).attr('opacity', b.dash ? 0.4 : 0.85);
    g.append('text').attr('x', (x(b.lbl) || 0) + x.bandwidth() / 2).attr('y', y(b.val) - 4).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '10px').text(b.val.toFixed(3));
  });
}

export default function EmpiricalShatterCheck() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [payload, setPayload] = useState<Payload | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    fetch(JSON_URL)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(setPayload)
      .catch((e) => setErr(String(e)));
  }, []);

  const isMobile = containerWidth > 0 && containerWidth < 720;
  const colW = useMemo(() => Math.max(280, ((containerWidth || 720) - 24) / (isMobile ? 1 : 2)), [containerWidth, isMobile]);

  const refA = useD3<SVGSVGElement>((svg) => { if (payload) renderPanelA(svg, colW, payload); }, [payload, colW]);
  const refB = useD3<SVGSVGElement>((svg) => { if (payload) renderPanelB(svg, colW, payload); }, [payload, colW]);
  const refC = useD3<SVGSVGElement>((svg) => { if (payload) renderPanelC(svg, colW, payload); }, [payload, colW]);
  const refD = useD3<SVGSVGElement>((svg) => { if (payload) renderPanelD(svg, colW, payload); }, [payload, colW]);

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      {err && <p style={{ color: paletteVC.emp }}>Could not load precomputed Monte Carlo: {err}</p>}
      {!payload && !err && <p style={{ fontSize: '0.85rem', color: 'var(--color-text-secondary)' }}>Loading integrative Monte Carlo (4 protocols)...</p>}
      {payload && (
        <>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
            <svg ref={refA} width={colW} height={HEIGHT} role="img" aria-label="Panel A: empirical growth versus Sauer–Shelah ceiling" />
            <svg ref={refB} width={colW} height={HEIGHT} role="img" aria-label="Panel B: FTSL envelope versus empirical gap" />
            <svg ref={refC} width={colW} height={HEIGHT} role="img" aria-label="Panel C: realizable sample complexity" />
            <svg ref={refD} width={colW} height={HEIGHT} role="img" aria-label="Panel D: empirical Rademacher versus Sauer–Shelah at n=20" />
          </div>
          <p style={{ fontSize: '0.85rem', color: 'var(--color-text-secondary)', marginTop: '0.4rem' }}>
            Four-panel integrative Monte Carlo. Panel A: empirical Π(n) below Sauer–Shelah ceilings. Panel B: FTSL envelope (solid) versus empirical worst-case gap (dashed). Panel C: realizable sample-complexity, empirical points versus theoretical curves. Panel D: empirical Rademacher (solid) versus Sauer–Shelah Rademacher bound (faded) at n = {payload.panel_d_rademacher_at_n_20.n}. Bounds loose by ~10× but rate-correct across every protocol.
          </p>
        </>
      )}
    </div>
  );
}
