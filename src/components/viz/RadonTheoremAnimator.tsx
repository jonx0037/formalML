import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { paletteVC, radonPartition2D, type Point2D } from './shared/vc-dimension';

// =============================================================================
// RadonTheoremAnimator — §8.5
//
// Three 4-point configurations in R^2; convex hulls of Radon partition shaded;
// shared intersection point marked. Reader can drag any point in any panel.
// Static fallback: public/images/topics/vc-dimension/08_radon_partitions.png
// =============================================================================

const HEIGHT = 320;

const INITIAL_CONFIGS: Point2D[][] = [
  // Convex-position quadrilateral
  [
    [0.2, 0.2],
    [0.8, 0.2],
    [0.8, 0.8],
    [0.2, 0.8],
  ],
  // Triangle + interior point
  [
    [0.2, 0.2],
    [0.8, 0.2],
    [0.5, 0.9],
    [0.5, 0.42],
  ],
  // Near-collinear configuration
  [
    [0.15, 0.4],
    [0.45, 0.45],
    [0.75, 0.5],
    [0.5, 0.85],
  ],
];

const TITLES = [
  'Convex-position quadrilateral',
  'One interior point',
  'Near-collinear configuration',
];

function renderPanel(
  svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
  w: number,
  points: Point2D[],
  title: string,
  onDrag: (idx: number, p: Point2D) => void,
) {
  const margin = { top: 28, right: 16, bottom: 16, left: 16 };
  const W = w - margin.left - margin.right;
  const H = HEIGHT - margin.top - margin.bottom;
  svg.selectAll('*').remove();
  if (W <= 0) return;
  svg.attr('viewBox', `0 0 ${w} ${HEIGHT}`);
  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

  const x = d3.scaleLinear().domain([0, 1]).range([0, W]);
  const y = d3.scaleLinear().domain([0, 1]).range([H, 0]);

  svg.append('text').attr('x', w / 2).attr('y', 18).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '12px').style('font-weight', 'bold').text(title);

  g.append('rect').attr('x', 0).attr('y', 0).attr('width', W).attr('height', H).attr('fill', 'transparent').attr('stroke', 'var(--color-border)');

  const result = radonPartition2D(points);
  if (result) {
    // Shade convex hulls of I+ and I-
    const hullPath = (idxs: number[]) => {
      const pts = idxs.map((i) => [x(points[i][0]), y(points[i][1])] as [number, number]);
      // Centroid sort for a 4-point hull (degenerate cases: |I+| or |I-| in {1, 2, 3}).
      if (pts.length < 3) {
        // Line: use a thick line.
        return null;
      }
      const cx = pts.reduce((s, p) => s + p[0], 0) / pts.length;
      const cy = pts.reduce((s, p) => s + p[1], 0) / pts.length;
      const sorted = [...pts].sort((a, b) => Math.atan2(a[1] - cy, a[0] - cx) - Math.atan2(b[1] - cy, b[0] - cx));
      return d3.line()(sorted) + 'Z';
    };
    const plusPath = hullPath(result.Iplus);
    const minusPath = hullPath(result.Iminus);
    if (plusPath) g.append('path').attr('d', plusPath).attr('fill', paletteVC.primary).attr('fill-opacity', 0.16).attr('stroke', paletteVC.primary).attr('stroke-width', 1.4);
    if (minusPath) g.append('path').attr('d', minusPath).attr('fill', paletteVC.emp).attr('fill-opacity', 0.16).attr('stroke', paletteVC.emp).attr('stroke-width', 1.4);
    // If one side has only 2 points (a line), draw a segment instead.
    if (!plusPath && result.Iplus.length === 2) {
      const [a, b] = result.Iplus;
      g.append('line').attr('x1', x(points[a][0])).attr('y1', y(points[a][1])).attr('x2', x(points[b][0])).attr('y2', y(points[b][1])).attr('stroke', paletteVC.primary).attr('stroke-width', 1.6);
    }
    if (!minusPath && result.Iminus.length === 2) {
      const [a, b] = result.Iminus;
      g.append('line').attr('x1', x(points[a][0])).attr('y1', y(points[a][1])).attr('x2', x(points[b][0])).attr('y2', y(points[b][1])).attr('stroke', paletteVC.emp).attr('stroke-width', 1.6);
    }
    // Shared point
    g.append('circle').attr('cx', x(result.p[0])).attr('cy', y(result.p[1])).attr('r', 5).attr('fill', paletteVC.highlight).attr('stroke', 'var(--color-bg)').attr('stroke-width', 1.5);
    g.append('text').attr('x', x(result.p[0]) + 8).attr('y', y(result.p[1]) - 8).style('fill', paletteVC.highlight).style('font-size', '10px').text('p');
  }

  // Points
  points.forEach((pt, i) => {
    const isPlus = result?.Iplus.includes(i) ?? false;
    const isMinus = result?.Iminus.includes(i) ?? false;
    const color = isPlus ? paletteVC.primary : isMinus ? paletteVC.emp : 'var(--color-text-secondary)';
    g.append('circle')
      .attr('cx', x(pt[0]))
      .attr('cy', y(pt[1]))
      .attr('r', 9)
      .style('fill', color)
      .style('cursor', 'grab')
      .style('stroke', 'var(--color-bg)')
      .style('stroke-width', 1.5)
      .call(
        d3
          .drag<SVGCircleElement, unknown>()
          .on('drag', function (event) {
            const nx = Math.max(0.02, Math.min(0.98, x.invert(event.x)));
            const ny = Math.max(0.02, Math.min(0.98, y.invert(event.y)));
            onDrag(i, [nx, ny]);
          }),
      );
    g.append('text').attr('x', x(pt[0])).attr('y', y(pt[1]) + 3).attr('text-anchor', 'middle').style('fill', 'white').style('font-size', '10px').style('font-weight', 'bold').style('pointer-events', 'none').text(String(i));
  });

  // Status text
  svg.append('text').attr('x', w / 2).attr('y', HEIGHT - 4).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-size', '10px').text(
    result ? `I+ = {${result.Iplus.join(',')}}, I- = {${result.Iminus.join(',')}}` : 'degenerate configuration',
  );
}

export default function RadonTheoremAnimator() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [configs, setConfigs] = useState<Point2D[][]>(INITIAL_CONFIGS);

  const updatePoint = (panelIdx: number) => (ptIdx: number, p: Point2D) => {
    setConfigs((prev) => {
      const next = prev.map((conf) => conf.slice()) as Point2D[][];
      next[panelIdx][ptIdx] = p;
      return next;
    });
  };

  const isMobile = containerWidth > 0 && containerWidth < 720;
  const colWidth = useMemo(() => Math.max(220, ((containerWidth || 720) - 24) / (isMobile ? 1 : 3)), [containerWidth, isMobile]);

  const r1 = useD3<SVGSVGElement>((svg) => renderPanel(svg, colWidth, configs[0], TITLES[0], updatePoint(0)), [configs, colWidth]);
  const r2 = useD3<SVGSVGElement>((svg) => renderPanel(svg, colWidth, configs[1], TITLES[1], updatePoint(1)), [configs, colWidth]);
  const r3 = useD3<SVGSVGElement>((svg) => renderPanel(svg, colWidth, configs[2], TITLES[2], updatePoint(2)), [configs, colWidth]);

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
        <svg ref={r1} width={colWidth} height={HEIGHT} role="img" aria-label="Radon partition for convex-position quadrilateral" />
        <svg ref={r2} width={colWidth} height={HEIGHT} role="img" aria-label="Radon partition for triangle with interior point" />
        <svg ref={r3} width={colWidth} height={HEIGHT} role="img" aria-label="Radon partition for near-collinear configuration" />
      </div>
      <p style={{ fontSize: '0.85rem', color: 'var(--color-text-secondary)', marginTop: '0.4rem' }}>
        Drag any point in any panel; the Radon partition reassigns automatically. The shared point p (teal) always lies inside both convex hulls — that's what makes the corresponding labeling unrealizable by a half-plane.
      </p>
    </div>
  );
}
