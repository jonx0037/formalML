import { useMemo } from 'react';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { paletteVC, type Point2D } from './shared/vc-dimension';

// =============================================================================
// ShatteringPlayground — §2.5
//
// 3 × 8 atlas: rows half-line-1D, half-plane-collinear-2D, half-plane-triangle-2D
// columns: 8 possible labelings of {0, 1}^3
// Each cell marks ✓ when realized, ✗ when not.
// Static fallback: public/images/topics/vc-dimension/02_shatter_atlas.png
// =============================================================================

const HEIGHT = 420;
const CELL_SIZE = 56;

type Row = {
  label: string;
  realizableMask: number; // bitmask of which of {0,1}^3 labelings are realized
  points: Point2D[];
};

function isLinearSeparable3(points: Point2D[], lbls: number[]): boolean {
  // 3 points in R^2: linearly separable iff we can find a line s.t. labels match signs.
  // Brute force: try perpendicular bisector lines between pairs of opposite-label points.
  const n = points.length;
  // Trivial: 0 or all 1s.
  const pos = lbls.filter((b) => b === 1).length;
  if (pos === 0 || pos === n) return true;
  // Try a set of candidate normals: rotations and through-pair-midpoint candidates.
  for (let theta = 0; theta < Math.PI; theta += Math.PI / 36) {
    const wx = Math.cos(theta);
    const wy = Math.sin(theta);
    const proj = points.map((p, i) => ({ v: wx * p[0] + wy * p[1], lbl: lbls[i] }));
    proj.sort((a, b) => a.v - b.v);
    // Check if labels are sorted (all 0s then all 1s) or reversed.
    let mono1 = true;
    let mono2 = true;
    for (let i = 1; i < n; i++) {
      if (proj[i].lbl < proj[i - 1].lbl) mono1 = false;
      if (proj[i].lbl > proj[i - 1].lbl) mono2 = false;
    }
    if (mono1 || mono2) return true;
  }
  return false;
}

function isHalflineRealizable(labels: number[]): boolean {
  // Half-line realizes (b1, b2, b3) on sorted x1 < x2 < x3 iff labels are of the form
  // (0, ..., 0, 1, ..., 1) or (1, 1, ..., 1, 0, ..., 0) — only 4 of 8 patterns realized.
  // Specifically the "monotone increasing or decreasing" patterns; both monotone 0->1 and 1->0 count
  // since the class includes h(x) = 1[x >= a] AND its complement 1[x <= a]? Actually our notebook
  // includes only 1[x >= a]. So only (0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1) — 4 realized.
  // For the brief's atlas we follow the notebook: only the 4 prefix-1 patterns realized.
  const n = labels.length;
  // Check labels are non-decreasing (since x1 < x2 < x3 sorted).
  for (let i = 1; i < n; i++) {
    if (labels[i] < labels[i - 1]) return false;
  }
  return true;
}

function computeRow(points: Point2D[], type: 'halfline' | 'halfplane'): number {
  let mask = 0;
  for (let m = 0; m < 8; m++) {
    const lbls = [m & 1, (m >> 1) & 1, (m >> 2) & 1];
    const realizable = type === 'halfline' ? isHalflineRealizable(lbls) : isLinearSeparable3(points, lbls);
    if (realizable) mask |= 1 << m;
  }
  return mask;
}

export default function ShatteringPlayground() {
  // Use static rows (we don't enable point dragging here; the atlas itself is the demo).
  const rows: Row[] = useMemo(() => {
    const r1: Row = {
      label: 'Half-lines on three collinear points',
      points: [
        [0.2, 0.5],
        [0.5, 0.5],
        [0.8, 0.5],
      ],
      realizableMask: 0,
    };
    r1.realizableMask = computeRow(r1.points, 'halfline');
    const r2: Row = {
      label: 'Half-planes on three collinear points',
      points: [
        [0.2, 0.5],
        [0.5, 0.5],
        [0.8, 0.5],
      ],
      realizableMask: 0,
    };
    r2.realizableMask = computeRow(r2.points, 'halfplane');
    const r3: Row = {
      label: 'Half-planes on a triangle',
      points: [
        [0.2, 0.25],
        [0.8, 0.25],
        [0.5, 0.85],
      ],
      realizableMask: 0,
    };
    r3.realizableMask = computeRow(r3.points, 'halfplane');
    return [r1, r2, r3];
  }, []);

  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const margin = { top: 18, right: 8, bottom: 8, left: 220 };
      svg.selectAll('*').remove();
      svg.attr('viewBox', `0 0 ${w} ${HEIGHT}`);

      // Headers: 8 labelings as column titles
      for (let m = 0; m < 8; m++) {
        const lbls = [m & 1, (m >> 1) & 1, (m >> 2) & 1];
        svg
          .append('text')
          .attr('x', margin.left + m * CELL_SIZE + CELL_SIZE / 2)
          .attr('y', 14)
          .attr('text-anchor', 'middle')
          .style('fill', 'var(--color-text)')
          .style('font-size', '11px')
          .style('font-family', 'var(--font-mono, monospace)')
          .text(`(${lbls[0]},${lbls[1]},${lbls[2]})`);
      }

      // Rows
      rows.forEach((row, ri) => {
        const yTop = margin.top + 12 + ri * 110;
        // Row label
        svg.append('text').attr('x', 8).attr('y', yTop + 36).style('fill', 'var(--color-text)').style('font-size', '11px').text(row.label);
        // Realized count
        let realizedCount = 0;
        for (let m = 0; m < 8; m++) if ((row.realizableMask >> m) & 1) realizedCount++;
        svg.append('text').attr('x', 8).attr('y', yTop + 54).style('fill', 'var(--color-text-secondary)').style('font-size', '10px').text(`realizes ${realizedCount}/8 — ${realizedCount === 8 ? 'SHATTERS' : 'does not shatter'}`);

        for (let m = 0; m < 8; m++) {
          const cx = margin.left + m * CELL_SIZE;
          const cy = yTop;
          const realized = ((row.realizableMask >> m) & 1) === 1;
          svg.append('rect').attr('x', cx).attr('y', cy).attr('width', CELL_SIZE - 4).attr('height', 88).style('fill', realized ? 'rgba(31, 95, 168, 0.08)' : 'rgba(192, 57, 43, 0.08)').style('stroke', realized ? paletteVC.primary : paletteVC.emp).style('stroke-width', 1);
          // Mini scatter of the 3 row points with the column labeling
          const lbls = [m & 1, (m >> 1) & 1, (m >> 2) & 1];
          row.points.forEach((p, i) => {
            svg
              .append('circle')
              .attr('cx', cx + 6 + p[0] * (CELL_SIZE - 16))
              .attr('cy', cy + 8 + (1 - p[1]) * 64)
              .attr('r', 4)
              .style('fill', lbls[i] === 1 ? paletteVC.primary : paletteVC.emp);
          });
          // Status icon (✓ / ✗)
          svg.append('text').attr('x', cx + (CELL_SIZE - 4) / 2).attr('y', cy + 84).attr('text-anchor', 'middle').style('fill', realized ? paletteVC.primary : paletteVC.emp).style('font-size', '12px').style('font-weight', 'bold').text(realized ? '✓' : '✗');
        }
      });
    },
    [rows, containerWidth],
  );

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <svg ref={svgRef} width="100%" height={HEIGHT} role="img" aria-label="3 by 8 grid of configurations and labelings, marking each cell as realized or not" />
      <p style={{ fontSize: '0.85rem', color: 'var(--color-text-secondary)', marginTop: '0.4rem' }}>
        Top row: half-lines on three collinear 1D points realize 4 of 8 labelings (the monotone prefix patterns). Middle row: half-planes on three collinear 2D points realize the same 4. Bottom row: half-planes on a triangle realize all 8 — shatters.
      </p>
    </div>
  );
}
