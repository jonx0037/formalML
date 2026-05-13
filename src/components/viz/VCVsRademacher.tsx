import { useEffect, useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { paletteVC } from './shared/vc-dimension';

// =============================================================================
// VCVsRademacher — §9.5
//
// Empirical Rademacher complexity vs Massart bound vs Sauer–Shelah Rademacher
// bound on axis-rectangles. Data is precomputed in
//   notebooks/vc-dimension/precompute_rademacher.py
// and lives at /sample-data/vc-dimension/rademacher_axis_rectangles.json.
// Static fallback: public/images/topics/vc-dimension/09_vc_vs_rademacher.png
// =============================================================================

const HEIGHT = 400;
const JSON_URL = '/sample-data/vc-dimension/rademacher_axis_rectangles.json';

type Row = {
  n: number;
  restricted_size: number;
  empirical: number;
  empirical_std: number;
  massart: number;
  ss_bound: number;
};

type Payload = {
  config: { ns: number[]; d_VC: number; B_rademacher: number };
  rows: Row[];
};

export default function VCVsRademacher() {
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

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const margin = { top: 22, right: 28, bottom: 48, left: 60 };
      const W = w - margin.left - margin.right;
      const H = HEIGHT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0 || !payload) return;
      svg.attr('viewBox', `0 0 ${w} ${HEIGHT}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const rows = payload.rows;
      const xExt: [number, number] = [Math.min(...rows.map((r) => r.n)) - 1, Math.max(...rows.map((r) => r.n)) + 1];
      const yMax = Math.max(...rows.map((r) => r.ss_bound)) * 1.15;
      const x = d3.scaleLinear().domain(xExt).range([0, W]);
      const y = d3.scaleLinear().domain([0, yMax]).range([H, 0]);
      g.append('g').attr('transform', `translate(0,${H})`).call(d3.axisBottom(x).ticks(rows.length));
      g.append('g').call(d3.axisLeft(y).ticks(6));
      g.append('text').attr('x', W / 2).attr('y', H + 32).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '12px').text('sample size n');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -H / 2).attr('y', -46).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '12px').text('complexity');

      const lineEmp = d3.line<Row>().x((r) => x(r.n)).y((r) => y(r.empirical));
      const lineMassart = d3.line<Row>().x((r) => x(r.n)).y((r) => y(r.massart));
      const lineSS = d3.line<Row>().x((r) => x(r.n)).y((r) => y(r.ss_bound));

      g.append('path').datum(rows).attr('d', lineSS).attr('fill', 'none').attr('stroke', paletteVC.emp).attr('stroke-width', 2).attr('stroke-dasharray', '5 3');
      g.append('path').datum(rows).attr('d', lineMassart).attr('fill', 'none').attr('stroke', paletteVC.alt).attr('stroke-width', 2).attr('stroke-dasharray', '4 4');
      g.append('path').datum(rows).attr('d', lineEmp).attr('fill', 'none').attr('stroke', paletteVC.primary).attr('stroke-width', 2.6);

      rows.forEach((r) => {
        g.append('circle').attr('cx', x(r.n)).attr('cy', y(r.empirical)).attr('r', 4).attr('fill', paletteVC.primary);
        g.append('text').attr('x', x(r.n)).attr('y', y(r.empirical) - 8).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-size', '10px').text(`|H|_S|=${r.restricted_size}`);
      });

      const legend = g.append('g').attr('transform', `translate(${Math.max(8, W - 240)}, 6)`);
      const items = [
        { label: 'empirical Rademacher', color: paletteVC.primary, dash: '' },
        { label: 'Massart on actual |H|_S|', color: paletteVC.alt, dash: '4 4' },
        { label: 'Sauer–Shelah Rademacher (d=4)', color: paletteVC.emp, dash: '5 3' },
      ];
      items.forEach((it, i) => {
        legend.append('line').attr('x1', 0).attr('x2', 18).attr('y1', i * 16 + 6).attr('y2', i * 16 + 6).attr('stroke', it.color).attr('stroke-width', 2.4).attr('stroke-dasharray', it.dash);
        legend.append('text').attr('x', 22).attr('y', i * 16 + 10).style('fill', 'var(--color-text)').style('font-size', '10px').text(it.label);
      });
    },
    [payload, containerWidth],
  );

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      {err && <p style={{ color: paletteVC.emp }}>Could not load precomputed data: {err}</p>}
      {!payload && !err && <p style={{ fontSize: '0.85rem', color: 'var(--color-text-secondary)' }}>Loading precomputed Rademacher Monte Carlo...</p>}
      {payload && (
        <>
          <svg ref={svgRef} width="100%" height={HEIGHT} role="img" aria-label="Empirical Rademacher versus Massart bound versus Sauer–Shelah bound on axis-rectangles" />
          <p style={{ fontSize: '0.85rem', color: 'var(--color-text-secondary)', marginTop: '0.4rem' }}>
            Empirical Rademacher complexity (blue) sits below both upper bounds at every n. Massart on the actual restricted-class size (orange) tracks the empirical closely; the Sauer–Shelah Rademacher bound (red) is the worst-case envelope. {payload.config.B_rademacher} Rademacher draws per n.
          </p>
        </>
      )}
    </div>
  );
}
