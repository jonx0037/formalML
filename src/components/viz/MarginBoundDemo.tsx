import { useEffect, useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';

// =============================================================================
// MarginBoundDemo — §10.  Consumes precomputed JSON from precompute_margin_bound.py.
//
//   Left  — SVM decision boundary on two-moons with margin contours and points.
//   Right — empirical margin loss vs γ alongside the Theorem 6 bound, with their
//           sum and the median-positive-margin reference line.
// =============================================================================

const HEIGHT = 360;
const SM_BREAKPOINT = 1000;
const PAYLOAD_URL = '/sample-data/generalization-bounds/margin_bound.json';

type Payload = {
  config: { n: number; delta: number; C: number; rbf_gamma: number; seed: number };
  train: { X: [number, number][]; y: number[] };
  decision_grid: { xs: number[]; ys: number[]; Z: number[][] };
  margins: {
    gamma_grid: number[];
    emp_margin_loss: number[];
    bound: number[];
    w_norm: number;
    R_kernel: number;
    median_positive_margin: number;
    optimal_gamma: number;
  };
};

export default function MarginBoundDemo() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [payload, setPayload] = useState<Payload | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    fetch(PAYLOAD_URL)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json() as Promise<Payload>;
      })
      .then((p) => { if (alive) setPayload(p); })
      .catch((e) => { if (alive) setError(String(e)); });
    return () => { alive = false; };
  }, []);

  const isStacked = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const panelWidth = isStacked ? containerWidth : Math.floor(containerWidth / 2);

  const leftRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (!payload || panelWidth <= 0) return;
      const margin = { top: 32, right: 12, bottom: 40, left: 50 };
      const w = panelWidth - margin.left - margin.right;
      const h = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const xs = payload.decision_grid.xs;
      const ys = payload.decision_grid.ys;
      const Z = payload.decision_grid.Z;
      const x = d3.scaleLinear().domain([xs[0], xs[xs.length - 1]]).range([0, w]);
      const y = d3.scaleLinear().domain([ys[0], ys[ys.length - 1]]).range([h, 0]);
      g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(5));
      g.append('g').call(d3.axisLeft(y).ticks(5));
      g.append('text').attr('x', w / 2).attr('y', -12).attr('text-anchor', 'middle')
        .style('font-size', '12px').style('font-weight', '600').style('fill', 'var(--color-text)')
        .text(`RBF SVM on two-moons (n=${payload.config.n}, C=${payload.config.C})`);

      // Background fill: positive (+) light blue, negative (-) light red.
      const cellW = w / (xs.length - 1);
      const cellH = h / (ys.length - 1);
      for (let i = 0; i < ys.length - 1; i++) {
        for (let j = 0; j < xs.length - 1; j++) {
          const z = Z[i][j];
          g.append('rect')
            .attr('x', j * cellW).attr('y', (ys.length - 2 - i) * cellH)
            .attr('width', cellW + 0.5).attr('height', cellH + 0.5)
            .style('fill', z > 0 ? '#dbeafe' : '#fecaca')
            .attr('fill-opacity', 0.45)
            .attr('stroke', 'none');
        }
      }

      // Contour lines at z = -1, 0, +1 (rough — using zero-crossing of cells).
      function drawContour(level: number, stroke: string, dash?: string) {
        const path: d3.Selection<SVGPathElement, unknown, null, undefined> = g.append('path')
          .attr('fill', 'none').style('stroke', stroke).style('stroke-width', 1.5);
        if (dash) path.style('stroke-dasharray', dash);
        const lines: [number, number, number, number][] = [];
        for (let i = 0; i < ys.length - 1; i++) {
          for (let j = 0; j < xs.length - 1; j++) {
            const z00 = Z[i][j], z01 = Z[i][j + 1];
            // Horizontal interpolation within this row
            if ((z00 - level) * (z01 - level) < 0) {
              const t = (level - z00) / (z01 - z00);
              const xi = j + t;
              const xs0 = x(xs[0] + xi * (xs[xs.length - 1] - xs[0]) / (xs.length - 1));
              const ys0 = y(ys[0] + (ys.length - 1 - i) * (ys[ys.length - 1] - ys[0]) / (ys.length - 1));
              lines.push([xs0, ys0, xs0 + 1, ys0]);
            }
          }
        }
        let dStr = '';
        lines.forEach(([x1, y1]) => { dStr += `M${x1},${y1}l1,0 `; });
        path.attr('d', dStr);
      }
      drawContour(0, '#0f172a');
      drawContour(1, '#0f172a', '4 3');
      drawContour(-1, '#0f172a', '4 3');

      // Training points
      payload.train.X.forEach((pt, i) => {
        g.append('circle').attr('cx', x(pt[0])).attr('cy', y(pt[1])).attr('r', 4)
          .style('fill', payload.train.y[i] === 1 ? '#1e40af' : '#dc2626')
          .style('stroke', 'white').style('stroke-width', 1);
      });
    },
    [payload, panelWidth],
  );

  const rightRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (!payload || panelWidth <= 0) return;
      const margin = { top: 32, right: 16, bottom: 40, left: 56 };
      const w = panelWidth - margin.left - margin.right;
      const h = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const m = payload.margins;
      const xMax = m.gamma_grid[m.gamma_grid.length - 1];
      const yMax = Math.max(
        d3.max(m.bound) ?? 1,
        d3.max(m.emp_margin_loss) ?? 1,
      );
      const x = d3.scaleLinear().domain([m.gamma_grid[0], xMax]).range([0, w]);
      const y = d3.scaleLinear().domain([0, yMax * 1.05]).range([h, 0]);
      g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(5));
      g.append('g').call(d3.axisLeft(y).ticks(5));
      g.append('text').attr('x', w / 2).attr('y', h + 32).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('margin γ');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -h / 2).attr('y', -42).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('loss / bound');
      g.append('text').attr('x', w / 2).attr('y', -12).attr('text-anchor', 'middle')
        .style('font-size', '12px').style('font-weight', '600').style('fill', 'var(--color-text)')
        .text(`Theorem 6 trade-off (‖w‖=${m.w_norm.toFixed(2)}, optimal γ≈${m.optimal_gamma.toFixed(2)})`);

      const empLine = d3.line<number>().x((_, i) => x(m.gamma_grid[i])).y((v) => y(v));
      const sumPts = m.emp_margin_loss.map((v, i) => v + m.bound[i]);
      g.append('path').datum(m.emp_margin_loss).attr('fill', 'none').style('stroke', '#3b82f6').style('stroke-width', 2).attr('d', empLine);
      g.append('path').datum(m.bound).attr('fill', 'none').style('stroke', '#ef4444').style('stroke-width', 2).style('stroke-dasharray', '4 3').attr('d', empLine);
      g.append('path').datum(sumPts).attr('fill', 'none').style('stroke', '#7c3aed').style('stroke-width', 1.5).style('stroke-dasharray', '2 2').attr('d', empLine);

      // Median positive-margin marker
      const xMed = x(m.median_positive_margin);
      g.append('line').attr('x1', xMed).attr('x2', xMed).attr('y1', 0).attr('y2', h)
        .style('stroke', '#0f172a').style('stroke-dasharray', '3 3');
      g.append('text').attr('x', xMed + 4).attr('y', 14).style('font-size', '10px').style('fill', '#0f172a')
        .text(`median margin = ${m.median_positive_margin.toFixed(3)}`);

      // Legend
      const lg = g.append('g').attr('transform', `translate(${w - 200},${30})`);
      lg.append('line').attr('x1', 0).attr('x2', 18).attr('y1', 0).attr('y2', 0).style('stroke', '#3b82f6').style('stroke-width', 2);
      lg.append('text').attr('x', 22).attr('y', 4).style('font-size', '11px').style('fill', 'var(--color-text)').text('empirical margin loss');
      lg.append('line').attr('x1', 0).attr('x2', 18).attr('y1', 18).attr('y2', 18).style('stroke', '#ef4444').style('stroke-width', 2).style('stroke-dasharray', '4 3');
      lg.append('text').attr('x', 22).attr('y', 22).style('font-size', '11px').style('fill', 'var(--color-text)').text('Theorem 6 bound');
      lg.append('line').attr('x1', 0).attr('x2', 18).attr('y1', 36).attr('y2', 36).style('stroke', '#7c3aed').style('stroke-width', 1.5).style('stroke-dasharray', '2 2');
      lg.append('text').attr('x', 22).attr('y', 40).style('font-size', '11px').style('fill', 'var(--color-text)').text('sum (total)');
    },
    [payload, panelWidth],
  );

  return (
    <figure ref={containerRef} className="my-8 not-prose">
      {payload ? (
        <div style={{ display: 'grid', gridTemplateColumns: isStacked ? '1fr' : '1fr 1fr', gap: 8 }}>
          <svg ref={leftRef} width={panelWidth || 360} height={HEIGHT} role="img" aria-label="SVM decision boundary on two-moons." />
          <svg ref={rightRef} width={panelWidth || 360} height={HEIGHT} role="img" aria-label="Margin loss + Theorem 6 bound vs γ." />
        </div>
      ) : (
        <div style={{
          height: HEIGHT, display: 'flex', alignItems: 'center', justifyContent: 'center',
          color: 'var(--color-text-secondary)', fontSize: 13,
        }}>
          {error ? `failed to load: ${error}` : 'loading margin-bound payload…'}
        </div>
      )}
    </figure>
  );
}
