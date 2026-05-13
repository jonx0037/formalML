import { useEffect, useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { ftslBoundEpsilon, paletteVC } from './shared/vc-dimension';

// =============================================================================
// FTSLBoundExplorer — §6.5
//
// Two-panel:
//   (a) FTSL closed-form bound vs n for d in {1, 3, 5, 10}, slope -1/2 on log-log
//   (b) Empirical worst-case gap on half-planes (cached from precompute_ftsl.py)
//       overlaid against FTSL bound at d=3, delta=0.05.
// Reader controls: sliders for n, d, delta.
// Static fallback: public/images/topics/vc-dimension/06_ftsl_bound_vs_empirical.png
// =============================================================================

const HEIGHT = 380;
const JSON_URL = '/sample-data/vc-dimension/ftsl_envelope.json';

type EmpiricalRow = {
  n: number;
  bound: number;
  gap_mean: number;
  gap_std: number;
  gap_samples: number[];
};

type Payload = {
  config: { d_VC_half_plane: number; delta: number; replicates: number };
  ftsl_curves: Record<string, Array<{ n: number; epsilon: number }>>;
  empirical_half_planes: EmpiricalRow[];
};

export default function FTSLBoundExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [payload, setPayload] = useState<Payload | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [n, setN] = useState(120);
  const [d, setD] = useState(3);
  const [delta, setDelta] = useState(0.05);

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
  const half = useMemo(() => Math.max(280, ((containerWidth || 720) - 24) / (isMobile ? 1 : 2)), [containerWidth, isMobile]);

  const cursorBound = ftslBoundEpsilon(n, d, delta);

  const refA = useD3<SVGSVGElement>(
    (svg) => {
      const margin = { top: 22, right: 30, bottom: 40, left: 56 };
      const W = half - margin.left - margin.right;
      const H = HEIGHT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${half} ${HEIGHT}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const nGrid = Array.from({ length: 991 }, (_, i) => i + 10);
      const dValues = [1, 3, 5, 10];
      const x = d3.scaleLog().domain([10, 1000]).range([0, W]);
      const yMax = ftslBoundEpsilon(10, 10, delta);
      const yMin = ftslBoundEpsilon(1000, 1, delta);
      const y = d3.scaleLog().domain([yMin * 0.7, yMax * 1.1]).range([H, 0]);
      g.append('g').attr('transform', `translate(0,${H})`).call(d3.axisBottom(x).ticks(5, '~s'));
      g.append('g').call(d3.axisLeft(y).ticks(5, '~g'));
      g.append('text').attr('x', W / 2).attr('y', H + 30).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '11px').text('sample size n (log)');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -H / 2).attr('y', -42).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '11px').text('FTSL ε (log)');

      const palette = [paletteVC.primary, paletteVC.alt, paletteVC.highlight, paletteVC.accent];
      dValues.forEach((dd, i) => {
        const line = d3.line<number>().x((nn) => x(nn)).y((nn) => y(ftslBoundEpsilon(nn, dd, delta)));
        g.append('path').datum(nGrid).attr('d', line).attr('fill', 'none').attr('stroke', palette[i]).attr('stroke-width', 2.0).attr('stroke-dasharray', dd === d ? '' : '3 3');
      });
      // Cursor at (n, cursorBound)
      g.append('circle').attr('cx', x(n)).attr('cy', y(cursorBound)).attr('r', 5).attr('fill', paletteVC.emp).attr('stroke', 'var(--color-bg)').attr('stroke-width', 1.5);

      const legend = g.append('g').attr('transform', `translate(${Math.max(8, W - 100)}, 6)`);
      dValues.forEach((dd, i) => {
        legend.append('line').attr('x1', 0).attr('x2', 14).attr('y1', i * 14 + 6).attr('y2', i * 14 + 6).attr('stroke', palette[i]).attr('stroke-width', 2);
        legend.append('text').attr('x', 18).attr('y', i * 14 + 10).style('fill', 'var(--color-text)').style('font-size', '10px').text(`d=${dd}`);
      });
    },
    [n, d, delta, cursorBound, half, containerWidth],
  );

  const refB = useD3<SVGSVGElement>(
    (svg) => {
      const margin = { top: 22, right: 30, bottom: 40, left: 56 };
      const W = half - margin.left - margin.right;
      const H = HEIGHT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0 || !payload) return;
      svg.attr('viewBox', `0 0 ${half} ${HEIGHT}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const empRows = payload.empirical_half_planes;
      const ns = empRows.map((r) => r.n);
      const allY = empRows.flatMap((r) => [r.bound, r.gap_mean]).filter((v) => v > 0);
      const yMin = Math.max(1e-3, Math.min(...allY) * 0.5);
      const yMax = Math.max(...allY) * 1.3;
      const x = d3.scaleLog().domain([Math.min(...ns) * 0.7, Math.max(...ns) * 1.4]).range([0, W]);
      const y = d3.scaleLog().domain([yMin, yMax]).range([H, 0]);
      g.append('g').attr('transform', `translate(0,${H})`).call(d3.axisBottom(x).ticks(5, '~s'));
      g.append('g').call(d3.axisLeft(y).ticks(5, '~g'));
      g.append('text').attr('x', W / 2).attr('y', H + 30).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '11px').text('sample size n (log)');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -H / 2).attr('y', -42).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '11px').text('gap (log)');

      const lineBound = d3.line<EmpiricalRow>().x((r) => x(r.n)).y((r) => y(r.bound));
      const lineEmp = d3.line<EmpiricalRow>().x((r) => x(r.n)).y((r) => y(r.gap_mean));
      g.append('path').datum(empRows).attr('d', lineBound).attr('fill', 'none').attr('stroke', paletteVC.primary).attr('stroke-width', 2.4);
      g.append('path').datum(empRows).attr('d', lineEmp).attr('fill', 'none').attr('stroke', paletteVC.emp).attr('stroke-width', 2.0).attr('stroke-dasharray', '5 3');

      empRows.forEach((r) => {
        g.append('circle').attr('cx', x(r.n)).attr('cy', y(r.gap_mean)).attr('r', 3).attr('fill', paletteVC.emp);
      });

      const legend = g.append('g').attr('transform', `translate(${Math.max(8, W - 140)}, 6)`);
      legend.append('line').attr('x1', 0).attr('x2', 14).attr('y1', 6).attr('y2', 6).attr('stroke', paletteVC.primary).attr('stroke-width', 2.4);
      legend.append('text').attr('x', 18).attr('y', 10).style('fill', 'var(--color-text)').style('font-size', '10px').text('FTSL bound');
      legend.append('line').attr('x1', 0).attr('x2', 14).attr('y1', 20).attr('y2', 20).attr('stroke', paletteVC.emp).attr('stroke-width', 2).attr('stroke-dasharray', '5 3');
      legend.append('text').attr('x', 18).attr('y', 24).style('fill', 'var(--color-text)').style('font-size', '10px').text('empirical (HP, d=3)');
    },
    [payload, half, containerWidth],
  );

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      {err && <p style={{ color: paletteVC.emp }}>Could not load precomputed data: {err}</p>}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
        <svg ref={refA} width={half} height={HEIGHT} role="img" aria-label="FTSL bound versus sample size for selected VC dimensions" />
        <svg ref={refB} width={half} height={HEIGHT} role="img" aria-label="FTSL bound versus empirical generalization gap on half-planes" />
      </div>
      <div style={{ marginTop: '0.6rem', display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '0.4rem 0.8rem', fontSize: '0.85rem', alignItems: 'center' }}>
        <label htmlFor="ftsl-n">n: {n}</label>
        <input id="ftsl-n" type="range" min={10} max={1000} value={n} onChange={(e) => setN(Number(e.target.value))} aria-label="Sample size n" />
        <label htmlFor="ftsl-d">d_VC: {d}</label>
        <input id="ftsl-d" type="range" min={1} max={10} value={d} onChange={(e) => setD(Number(e.target.value))} aria-label="VC dimension d" />
        <label htmlFor="ftsl-delta">δ: {delta.toFixed(3)}</label>
        <input id="ftsl-delta" type="range" min={0.01} max={0.5} step={0.005} value={delta} onChange={(e) => setDelta(Number(e.target.value))} aria-label="Confidence parameter delta" />
      </div>
      <p style={{ fontSize: '0.85rem', color: 'var(--color-text-secondary)', marginTop: '0.4rem' }}>
        At n = {n}, d = {d}, δ = {delta.toFixed(2)}: FTSL bound ε = {cursorBound.toFixed(3)}. The bound has slope −1/2 in n on log-log, just like the empirical gap (right panel).
      </p>
    </div>
  );
}
