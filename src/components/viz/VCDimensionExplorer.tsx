import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { paletteVC } from './shared/vc-dimension';

// =============================================================================
// VCDimensionExplorer — §4.5
//
// Four destructive-direction panels: half-lines (monotonicity), intervals
// (connectivity), half-planes (convex-hull), axis-rectangles (bounding-box).
// Each panel shows the smallest unrealizable labeling for the corresponding class.
// Static fallback: public/images/topics/vc-dimension/04_destructive_direction.png
// =============================================================================

const HEIGHT = 360;
const SM_BREAKPOINT = 720;

type Panel = {
  title: string;
  obstruction: string;
  points: Array<[number, number]>;
  labels: number[];
  unrealizable: string;
};

const PANELS: Panel[] = [
  {
    title: 'Half-lines: monotonicity',
    obstruction: 'no threshold realizes (1, 0)',
    points: [
      [0.25, 0.5],
      [0.75, 0.5],
    ],
    labels: [1, 0],
    unrealizable: '(1, 0)',
  },
  {
    title: 'Intervals: connectivity',
    obstruction: 'no interval covers endpoints but not middle',
    points: [
      [0.2, 0.5],
      [0.5, 0.5],
      [0.8, 0.5],
    ],
    labels: [1, 0, 1],
    unrealizable: '(1, 0, 1)',
  },
  {
    title: 'Half-planes: convex hull',
    obstruction: 'XOR labeling on unit-square corners',
    points: [
      [0.2, 0.2],
      [0.8, 0.2],
      [0.8, 0.8],
      [0.2, 0.8],
    ],
    labels: [1, 0, 1, 0],
    unrealizable: '(1, 0, 1, 0)',
  },
  {
    title: 'Axis-rectangles: bounding box',
    obstruction: 'origin labeled 0, four cardinals labeled 1',
    points: [
      [0.5, 0.85],
      [0.85, 0.5],
      [0.5, 0.15],
      [0.15, 0.5],
      [0.5, 0.5],
    ],
    labels: [1, 1, 1, 1, 0],
    unrealizable: '(1, 1, 1, 1, 0)',
  },
];

function renderPanel(
  svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
  w: number,
  panel: Panel,
) {
  const margin = { top: 36, right: 14, bottom: 32, left: 14 };
  const H = HEIGHT - margin.top - margin.bottom;
  const W = w - margin.left - margin.right;
  svg.selectAll('*').remove();
  if (W <= 0) return;
  svg.attr('viewBox', `0 0 ${w} ${HEIGHT}`);
  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

  g.append('rect').attr('x', 0).attr('y', 0).attr('width', W).attr('height', H).attr('fill', 'transparent').attr('stroke', 'var(--color-border)').attr('stroke-width', 1);

  const x = d3.scaleLinear().domain([0, 1]).range([0, W]);
  const y = d3.scaleLinear().domain([0, 1]).range([H, 0]);

  // Title
  svg.append('text').attr('x', w / 2).attr('y', 14).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '12px').style('font-weight', 'bold').text(panel.title);
  svg.append('text').attr('x', w / 2).attr('y', 28).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-size', '10px').text(panel.obstruction);

  // Points
  panel.points.forEach(([px, py], i) => {
    const fill = panel.labels[i] === 1 ? paletteVC.primary : paletteVC.emp;
    g.append('circle').attr('cx', x(px)).attr('cy', y(py)).attr('r', 9).style('fill', fill).style('stroke', 'var(--color-bg)').style('stroke-width', 1.5);
    g.append('text').attr('x', x(px)).attr('y', y(py) + 3).attr('text-anchor', 'middle').style('fill', 'white').style('font-size', '11px').style('font-weight', 'bold').text(String(panel.labels[i]));
  });

  // Unrealizable label
  svg.append('text').attr('x', w / 2).attr('y', HEIGHT - 8).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '11px').text(`unrealizable: ${panel.unrealizable}`);
}

export default function VCDimensionExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const colWidth = useMemo(() => Math.max(160, ((containerWidth || 720) - 24) / (isMobile ? 2 : 4)), [containerWidth, isMobile]);

  const r1 = useD3<SVGSVGElement>((svg) => renderPanel(svg, colWidth, PANELS[0]), [colWidth]);
  const r2 = useD3<SVGSVGElement>((svg) => renderPanel(svg, colWidth, PANELS[1]), [colWidth]);
  const r3 = useD3<SVGSVGElement>((svg) => renderPanel(svg, colWidth, PANELS[2]), [colWidth]);
  const r4 = useD3<SVGSVGElement>((svg) => renderPanel(svg, colWidth, PANELS[3]), [colWidth]);

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <div style={{ display: 'grid', gridTemplateColumns: isMobile ? '1fr 1fr' : '1fr 1fr 1fr 1fr', gap: '8px' }}>
        <svg ref={r1} width={colWidth} height={HEIGHT} role="img" aria-label="Destructive direction for half-lines" />
        <svg ref={r2} width={colWidth} height={HEIGHT} role="img" aria-label="Destructive direction for intervals" />
        <svg ref={r3} width={colWidth} height={HEIGHT} role="img" aria-label="Destructive direction for half-planes" />
        <svg ref={r4} width={colWidth} height={HEIGHT} role="img" aria-label="Destructive direction for axis-rectangles" />
      </div>
      <p style={{ fontSize: '0.85rem', color: 'var(--color-text-secondary)', marginTop: '0.4rem' }}>
        Blue circles labeled 1; red circles labeled 0. Each panel shows the smallest unrealizable labeling for its class — the geometric obstruction that caps the VC dimension at d_VC.
      </p>
    </div>
  );
}
