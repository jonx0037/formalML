import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  affineCriterionSurface,
  identityMat,
  rankDiagnostic,
  type Mat,
} from './shared/generalized-method-of-moments';

// -----------------------------------------------------------------------------
// Viz 3.2 — Identification rank inspector.
//
// The reader slides an "angle" parameter that interpolates the design matrix
// A between full-rank (the running example) and rank-deficient (two columns
// proportional). Two panels:
//
//   Panel A: criterion surface contours under identity weighting and a fixed
//   sample-mean vector (no MC). As A degrades, the elliptical basin elongates
//   into a ridge — visually showing identification failure.
//
//   Panel B: log-scale bar of the two eigenvalues of AᵀA. The smaller
//   eigenvalue drops toward zero as the angle approaches degeneracy. A
//   shaded "weak identification" zone (smallest eigenvalue < 0.1) marks
//   the danger region.
// -----------------------------------------------------------------------------

const GRID_RES = 45;
const PANEL_H = 280;
const BAR_H = 130;
const MARGIN = { top: 22, right: 18, bottom: 40, left: 56 };

// Smoothly interpolate the running-example A toward rank deficiency by
// collapsing the second column toward the first as angle → 1.
function makeAngledA(angle: number): Mat {
  // angle ∈ [0, 1]; 0 = original A, 1 = both columns are the first column.
  const A0: Mat = [
    [1, 0],
    [0, 1],
    [1, 1],
    [2, -1],
  ];
  const A1: Mat = [
    [1, 1],
    [0, 0],
    [1, 1],
    [2, 2],
  ];
  const A: Mat = [];
  for (let i = 0; i < 4; i++) {
    A.push([(1 - angle) * A0[i][0] + angle * A1[i][0], (1 - angle) * A0[i][1] + angle * A1[i][1]]);
  }
  return A;
}

export default function IdentificationRankInspector() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [angle, setAngle] = useState(0.0);
  const A = useMemo(() => makeAngledA(angle), [angle]);
  const diag = useMemo(() => rankDiagnostic(A), [A]);

  // Fixed X̄ = A · θ_0 at θ_0 = (1, 1). Keeps the contour "centered" at the
  // population minimum so the shape change is the only thing the reader sees.
  const Xbar = useMemo(() => {
    return [A[0][0] + A[0][1], A[1][0] + A[1][1], A[2][0] + A[2][1], A[3][0] + A[3][1]];
  }, [A]);

  const surface = useMemo(
    () => affineCriterionSurface(A, Xbar, identityMat(4), [-0.5, 2.5], [-0.5, 2.5], GRID_RES, 200),
    [A, Xbar],
  );

  const w = Math.max(280, Math.min(containerWidth || 720, 820));

  const surfaceRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const innerW = w - MARGIN.left - MARGIN.right;
      const innerH = PANEL_H - MARGIN.top - MARGIN.bottom;
      if (innerW <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const xScale = d3.scaleLinear().domain([-0.5, 2.5]).range([0, innerW]);
      const yScale = d3.scaleLinear().domain([-0.5, 2.5]).range([innerH, 0]);

      const flat: number[] = [];
      for (let i = 0; i < GRID_RES; i++) for (let j = 0; j < GRID_RES; j++) flat.push(surface.J[i][j]);
      const Jmax = d3.max(flat) ?? 1;
      const Jmin = d3.min(flat) ?? 0;
      const span = Math.max(Jmax - Jmin, 1e-9);
      const levels: number[] = [];
      for (let k = 1; k <= 8; k++) levels.push(Jmin + Math.pow(k / 8, 2.0) * span);

      const contours = d3.contours().size([GRID_RES, GRID_RES]).thresholds(levels)(flat);
      const colorScale = d3.scaleSequential(d3.interpolateBlues).domain([Jmax, Jmin]);
      const cellW = innerW / (GRID_RES - 1);
      const cellH = innerH / (GRID_RES - 1);

      g.append('g')
        .attr('transform', `translate(0, ${innerH}) scale(${cellW}, ${-cellH})`)
        .selectAll('path')
        .data(contours)
        .enter()
        .append('path')
        .attr('d', d3.geoPath() as never)
        .style('fill', (d) => colorScale(d.value))
        .style('stroke', 'var(--color-border)')
        .style('stroke-width', 0.5 / Math.min(cellW, cellH))
        .style('opacity', 0.85);

      g.append('g').attr('transform', `translate(0, ${innerH})`).call(d3.axisBottom(xScale).ticks(5)).selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(yScale).ticks(5)).selectAll('text').style('fill', 'var(--color-text)');

      // θ₀ cross.
      const x0 = xScale(1);
      const y0 = yScale(1);
      g.append('line').attr('x1', x0 - 8).attr('x2', x0 + 8).attr('y1', y0).attr('y2', y0).style('stroke', 'var(--color-text)').style('stroke-width', 1.6);
      g.append('line').attr('x1', x0).attr('x2', x0).attr('y1', y0 - 8).attr('y2', y0 + 8).style('stroke', 'var(--color-text)').style('stroke-width', 1.6);
      g.append('text').attr('x', x0 + 10).attr('y', y0 - 6).style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', 'var(--color-text)').text('θ₀');

      // Axis labels.
      g.append('text').attr('x', innerW / 2).attr('y', innerH + 30).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text('θ_1');
      g.append('text').attr('x', -innerH / 2).attr('y', -42).attr('text-anchor', 'middle').attr('transform', 'rotate(-90)').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text('θ_2');
    },
    [surface, w],
  );

  const barRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const innerW = w - MARGIN.left - MARGIN.right;
      const innerH = BAR_H - 18;
      if (innerW <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},6)`);

      const eigs = diag.eigvals;
      const yScale = d3.scaleLog().domain([1e-6, Math.max(diag.largest * 2, 100)]).range([innerH, 0]);

      g.append('g').call(d3.axisLeft(yScale).ticks(4, '~s')).selectAll('text').style('fill', 'var(--color-text)');

      // Weak-identification shaded zone (smallest eigenvalue < 0.1).
      g.append('rect')
        .attr('x', 0)
        .attr('y', yScale(0.1))
        .attr('width', innerW)
        .attr('height', innerH - yScale(0.1))
        .style('fill', 'var(--color-accent)')
        .attr('opacity', 0.12);
      g.append('text')
        .attr('x', innerW - 10)
        .attr('y', yScale(0.1) + 14)
        .attr('text-anchor', 'end')
        .style('font-family', 'var(--font-mono)')
        .style('font-size', '11px')
        .style('fill', 'var(--color-accent)')
        .text('weak-identification zone (λ_min < 0.1)');

      const barW = innerW * 0.25;
      eigs.forEach((lambda, i) => {
        const cx = innerW * (0.2 + i * 0.4);
        const safe = Math.max(lambda, 1e-7);
        g.append('rect')
          .attr('x', cx - barW / 2)
          .attr('y', yScale(safe))
          .attr('width', barW)
          .attr('height', innerH - yScale(safe))
          .style('fill', i === 0 ? 'var(--color-accent)' : '#9aa3ad');
        g.append('text')
          .attr('x', cx)
          .attr('y', yScale(safe) - 6)
          .attr('text-anchor', 'middle')
          .style('font-family', 'var(--font-mono)')
          .style('font-size', '11px')
          .style('fill', 'var(--color-text)')
          .text(`λ${i + 1} = ${lambda.toExponential(2)}`);
      });

      g.append('text').attr('x', -innerH / 2).attr('y', -42).attr('text-anchor', 'middle').attr('transform', 'rotate(-90)').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text('eigenvalue of AᵀA (log)');
    },
    [diag, w],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: 'var(--font-sans)' }}>
      <div style={{ display: 'flex', gap: '14px', padding: '10px 0', fontSize: '13px', flexWrap: 'wrap' }}>
        <label style={{ display: 'flex', flexDirection: 'column', minWidth: '260px' }}>
          <span>
            collapse angle = <strong>{angle.toFixed(2)}</strong>
            <span style={{ color: 'var(--color-text-secondary)', marginLeft: 8 }}>
              {angle < 0.3 ? '(full rank)' : angle < 0.85 ? '(near rank-deficient)' : '(rank ≈ 1)'}
            </span>
          </span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.02}
            value={angle}
            aria-label="Design-matrix collapse angle"
            onChange={(e) => setAngle(Number(e.target.value))}
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
            alignSelf: 'flex-end',
          }}
        >
          condition number κ(AᵀA) = {diag.conditionNumber.toExponential(2)} &nbsp;·&nbsp;
          effective rank = {diag.effectiveRank}
        </div>
      </div>
      <svg ref={surfaceRef} width={w} height={PANEL_H} style={{ maxWidth: '100%', display: 'block' }} aria-label="Criterion surface under degrading identification" />
      <svg ref={barRef} width={w} height={BAR_H} style={{ maxWidth: '100%', display: 'block', marginTop: 8 }} aria-label="Eigenvalues of AᵀA" />
    </div>
  );
}
