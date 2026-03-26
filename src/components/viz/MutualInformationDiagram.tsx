import { useState, useMemo, useCallback } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { dimensionColors } from './shared/colorScales';
import {
  entropy,
  jointEntropy,
  conditionalEntropy,
  mutualInformation,
  marginal,
} from './shared/informationTheory';

// ── Constants ────────────────────────────────────────────────────────

const VENN_HEIGHT = 340;
const HEAT_HEIGHT = 340;
const SM_BREAKPOINT = 640;
const MARGIN = { top: 20, right: 20, bottom: 30, left: 40 };

const TEAL = dimensionColors[0];
const PURPLE = dimensionColors[1];
const AMBER = '#D97706';

type Preset = 'independent' | 'correlated' | 'noisy' | 'custom';
type GridSize = 2 | 3 | 4;

const fmt = (x: number) => x.toFixed(3);

function makeIndependent(nx: number, ny: number): number[][] {
  const val = 1 / (nx * ny);
  return Array.from({ length: nx }, () => new Array(ny).fill(val));
}

function makeCorrelated(n: number): number[][] {
  // Perfectly correlated: p(x,y) = 1/n if x=y, else 0
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 / n : 0))
  );
}

function makeNoisy(n: number): number[][] {
  // Noisy channel: high on diagonal, some off-diagonal mass
  const diag = 0.7 / n;
  const off = 0.3 / (n * n - n);
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? diag : off))
  );
}

// ── Component ────────────────────────────────────────────────────────

export default function MutualInformationDiagram() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();

  const [nx, setNx] = useState<GridSize>(3);
  const [ny, setNy] = useState<GridSize>(3);
  const [preset, setPreset] = useState<Preset>('noisy');
  const [joint, setJoint] = useState(() => makeNoisy(3));
  const [hoveredRegion, setHoveredRegion] = useState<string | null>(null);

  const isStacked = containerWidth > 0 && containerWidth < SM_BREAKPOINT;

  const handlePreset = useCallback((p: Preset) => {
    setPreset(p);
    if (p === 'independent') {
      setJoint(makeIndependent(nx, ny));
    } else {
      // Correlated/noisy presets require a square grid
      const n = Math.min(nx, ny) as GridSize;
      if (nx !== n) setNx(n);
      if (ny !== n) setNy(n);
      if (p === 'correlated') setJoint(makeCorrelated(n));
      else if (p === 'noisy') setJoint(makeNoisy(n));
    }
  }, [nx, ny]);

  const handleGridChange = useCallback((axis: 'x' | 'y', val: GridSize) => {
    if (axis === 'x') setNx(val);
    else setNy(val);
    const newNx = axis === 'x' ? val : nx;
    const newNy = axis === 'y' ? val : ny;
    setJoint(makeIndependent(newNx, newNy));
    setPreset('custom');
  }, [nx, ny]);

  // Click to adjust joint distribution cells
  const handleCellClick = useCallback((i: number, j: number, shift: boolean) => {
    setPreset('custom');
    setJoint((prev) => {
      const updated = prev.map((row) => [...row]);
      const delta = shift ? -0.02 : 0.02;
      updated[i][j] = Math.max(0.001, updated[i][j] + delta);
      // Normalize
      const total = updated.flat().reduce((s, v) => s + v, 0);
      return updated.map((row) => row.map((v) => v / total));
    });
  }, []);

  // ── Computed quantities ─────────────────────────────────────────
  const quantities = useMemo(() => {
    const pX = marginal(joint, 'y');
    const pY = marginal(joint, 'x');
    const hX = entropy(pX);
    const hY = entropy(pY);
    const hXY = jointEntropy(joint);
    const hYgX = conditionalEntropy(joint);
    const hXgY = hXY - hY;
    const mi = mutualInformation(joint);
    return { pX, pY, hX, hY, hXY, hYgX, hXgY, mi };
  }, [joint]);

  // ── Venn diagram ────────────────────────────────────────────────
  const vennWidth = isStacked ? containerWidth : Math.floor(containerWidth * 0.5);

  const vennRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (vennWidth <= 0) return;

      const { hX, hY, mi, hYgX, hXgY, hXY } = quantities;
      const w = vennWidth;
      const h = VENN_HEIGHT;
      const cx = w / 2;
      const cy = h / 2;

      // Scale radii proportional to sqrt(entropy)
      const maxR = Math.min(w, h) * 0.32;
      const scale = maxR / Math.max(Math.sqrt(hX), Math.sqrt(hY), 0.01);
      const rX = Math.max(10, scale * Math.sqrt(hX));
      const rY = Math.max(10, scale * Math.sqrt(hY));

      // Overlap separation based on MI
      const maxMI = Math.min(hX, hY);
      const overlapFrac = maxMI > 0 ? mi / maxMI : 0;
      const maxSep = rX + rY;
      const minSep = Math.abs(rX - rY);
      const sep = maxSep - overlapFrac * (maxSep - minSep);

      const cxLeft = cx - sep / 2;
      const cxRight = cx + sep / 2;

      const g = svg.append('g');

      // Bounding rectangle for H(X,Y)
      g.append('rect')
        .attr('x', cx - maxR - 30)
        .attr('y', cy - maxR - 20)
        .attr('width', 2 * maxR + 60)
        .attr('height', 2 * maxR + 40)
        .attr('rx', 8)
        .attr('fill', 'none')
        .attr('stroke', 'var(--color-border)')
        .attr('stroke-dasharray', '4,3');

      g.append('text')
        .attr('x', cx + maxR + 28)
        .attr('y', cy - maxR - 6)
        .attr('text-anchor', 'end')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '11px')
        .text(`H(X,Y) = ${fmt(hXY)}`);

      // H(X) circle
      g.append('circle')
        .attr('cx', cxLeft)
        .attr('cy', cy)
        .attr('r', rX)
        .attr('fill', TEAL)
        .attr('fill-opacity', hoveredRegion === 'hX' ? 0.35 : 0.2)
        .attr('stroke', TEAL)
        .attr('stroke-width', 2);

      // H(Y) circle
      g.append('circle')
        .attr('cx', cxRight)
        .attr('cy', cy)
        .attr('r', rY)
        .attr('fill', PURPLE)
        .attr('fill-opacity', hoveredRegion === 'hY' ? 0.35 : 0.2)
        .attr('stroke', PURPLE)
        .attr('stroke-width', 2);

      // Labels
      const labelOffX = Math.min(rX * 0.5, sep * 0.3);

      // H(X|Y) — left crescent
      g.append('text')
        .attr('x', cxLeft - labelOffX)
        .attr('y', cy - 10)
        .attr('text-anchor', 'middle')
        .style('fill', TEAL)
        .style('font-size', '11px')
        .style('font-weight', hoveredRegion === 'hXgY' ? '700' : '500')
        .text('H(X|Y)')
        .style('cursor', 'pointer')
        .on('mouseenter', () => setHoveredRegion('hXgY'))
        .on('mouseleave', () => setHoveredRegion(null));

      g.append('text')
        .attr('x', cxLeft - labelOffX)
        .attr('y', cy + 8)
        .attr('text-anchor', 'middle')
        .style('fill', TEAL)
        .style('font-size', '10px')
        .text(fmt(hXgY));

      // I(X;Y) — overlap
      g.append('text')
        .attr('x', cx)
        .attr('y', cy - 10)
        .attr('text-anchor', 'middle')
        .style('fill', AMBER)
        .style('font-size', '11px')
        .style('font-weight', hoveredRegion === 'mi' ? '700' : '600')
        .text('I(X;Y)')
        .style('cursor', 'pointer')
        .on('mouseenter', () => setHoveredRegion('mi'))
        .on('mouseleave', () => setHoveredRegion(null));

      g.append('text')
        .attr('x', cx)
        .attr('y', cy + 8)
        .attr('text-anchor', 'middle')
        .style('fill', AMBER)
        .style('font-size', '10px')
        .text(fmt(mi));

      // H(Y|X) — right crescent
      g.append('text')
        .attr('x', cxRight + labelOffX)
        .attr('y', cy - 10)
        .attr('text-anchor', 'middle')
        .style('fill', PURPLE)
        .style('font-size', '11px')
        .style('font-weight', hoveredRegion === 'hYgX' ? '700' : '500')
        .text('H(Y|X)')
        .style('cursor', 'pointer')
        .on('mouseenter', () => setHoveredRegion('hYgX'))
        .on('mouseleave', () => setHoveredRegion(null));

      g.append('text')
        .attr('x', cxRight + labelOffX)
        .attr('y', cy + 8)
        .attr('text-anchor', 'middle')
        .style('fill', PURPLE)
        .style('font-size', '10px')
        .text(fmt(hYgX));

      // Circle labels
      g.append('text')
        .attr('x', cxLeft)
        .attr('y', cy + rX + 18)
        .attr('text-anchor', 'middle')
        .style('fill', TEAL)
        .style('font-size', '12px')
        .style('font-weight', '600')
        .text(`H(X) = ${fmt(hX)}`);

      g.append('text')
        .attr('x', cxRight)
        .attr('y', cy + rY + 18)
        .attr('text-anchor', 'middle')
        .style('fill', PURPLE)
        .style('font-size', '12px')
        .style('font-weight', '600')
        .text(`H(Y) = ${fmt(hY)}`);
    },
    [vennWidth, quantities, hoveredRegion]
  );

  // ── Heatmap + marginals ─────────────────────────────────────────
  const heatWidth = isStacked ? containerWidth : containerWidth - vennWidth;

  const heatRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (heatWidth <= 0) return;

      const { pX, pY } = quantities;
      const margH = 40; // height for marginal bar chart
      const w = heatWidth - MARGIN.left - MARGIN.right;
      const h = HEAT_HEIGHT - MARGIN.top - MARGIN.bottom - margH;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const rows = joint.length;
      const cols = joint[0].length;
      const cellW = (w - margH) / cols; // reserve space for y-marginal
      const cellH = h / rows;

      const maxP = Math.max(...joint.flat());
      const colorScale = d3.scaleSequential(d3.interpolateBlues).domain([0, maxP]);

      // Joint distribution heatmap
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          g.append('rect')
            .attr('x', j * cellW)
            .attr('y', i * cellH)
            .attr('width', cellW - 1)
            .attr('height', cellH - 1)
            .attr('fill', colorScale(joint[i][j]))
            .attr('rx', 2)
            .style('cursor', 'pointer')
            .on('click', (event: MouseEvent) => {
              handleCellClick(i, j, event.shiftKey);
            });

          // Cell value
          g.append('text')
            .attr('x', j * cellW + cellW / 2)
            .attr('y', i * cellH + cellH / 2)
            .attr('text-anchor', 'middle')
            .attr('dominant-baseline', 'central')
            .style('fill', joint[i][j] > maxP * 0.6 ? '#fff' : 'var(--color-text)')
            .style('font-size', '10px')
            .text(joint[i][j].toFixed(3));
        }
      }

      // X-axis labels
      for (let j = 0; j < cols; j++) {
        g.append('text')
          .attr('x', j * cellW + cellW / 2)
          .attr('y', rows * cellH + 14)
          .attr('text-anchor', 'middle')
          .style('fill', 'var(--color-text-secondary)')
          .style('font-size', '11px')
          .text(`y${j + 1}`);
      }

      // Y-axis labels
      for (let i = 0; i < rows; i++) {
        g.append('text')
          .attr('x', -10)
          .attr('y', i * cellH + cellH / 2)
          .attr('text-anchor', 'end')
          .attr('dominant-baseline', 'central')
          .style('fill', 'var(--color-text-secondary)')
          .style('font-size', '11px')
          .text(`x${i + 1}`);
      }

      // p(x) marginal bars (right of heatmap)
      const margXScale = d3.scaleLinear().domain([0, 1]).range([0, margH]);
      for (let i = 0; i < rows; i++) {
        g.append('rect')
          .attr('x', cols * cellW + 4)
          .attr('y', i * cellH + 2)
          .attr('width', margXScale(pX[i]))
          .attr('height', cellH - 4)
          .attr('fill', TEAL)
          .attr('opacity', 0.7)
          .attr('rx', 1);
      }

      // p(y) marginal bars (below heatmap)
      const margYScale = d3.scaleLinear().domain([0, 1]).range([0, margH]);
      for (let j = 0; j < cols; j++) {
        g.append('rect')
          .attr('x', j * cellW + 2)
          .attr('y', rows * cellH + 22)
          .attr('width', cellW - 4)
          .attr('height', margYScale(pY[j]))
          .attr('fill', PURPLE)
          .attr('opacity', 0.7)
          .attr('rx', 1);
      }

      // Title
      g.append('text')
        .attr('x', (cols * cellW) / 2)
        .attr('y', -6)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)')
        .style('font-size', '12px')
        .style('font-weight', '500')
        .text('p(x, y)  — click cells to adjust (shift-click to decrease)');
    },
    [heatWidth, joint, quantities]
  );

  return (
    <div ref={containerRef} className="w-full">
      {/* Controls */}
      <div className="flex flex-wrap items-center gap-3 mb-4 text-sm">
        <label className="flex items-center gap-1.5" style={{ color: 'var(--color-text-secondary)' }}>
          |X|:
          <select
            value={nx}
            onChange={(e) => handleGridChange('x', Number(e.target.value) as GridSize)}
            className="rounded border px-2 py-1 text-sm"
            style={{ borderColor: 'var(--color-border)', backgroundColor: 'var(--color-bg)', color: 'var(--color-text)' }}
          >
            {[2, 3, 4].map((n) => <option key={n} value={n}>{n}</option>)}
          </select>
        </label>

        <label className="flex items-center gap-1.5" style={{ color: 'var(--color-text-secondary)' }}>
          |Y|:
          <select
            value={ny}
            onChange={(e) => handleGridChange('y', Number(e.target.value) as GridSize)}
            className="rounded border px-2 py-1 text-sm"
            style={{ borderColor: 'var(--color-border)', backgroundColor: 'var(--color-bg)', color: 'var(--color-text)' }}
          >
            {[2, 3, 4].map((n) => <option key={n} value={n}>{n}</option>)}
          </select>
        </label>

        {(['independent', 'correlated', 'noisy', 'custom'] as Preset[]).map((p) => (
          <button
            key={p}
            onClick={() => p !== 'custom' && handlePreset(p)}
            disabled={p === 'custom'}
            className={`rounded px-3 py-1 text-sm font-medium transition-colors ${
              preset === p ? 'ring-1' : ''
            }`}
            style={{
              backgroundColor: preset === p ? 'var(--color-definition-bg)' : 'transparent',
              color: preset === p ? TEAL : 'var(--color-text-secondary)',
              borderColor: 'var(--color-border)',
              border: '1px solid var(--color-border)',
            }}
          >
            {p.charAt(0).toUpperCase() + p.slice(1)}
          </button>
        ))}
      </div>

      {/* Panels */}
      <div className={`flex ${isStacked ? 'flex-col' : 'flex-row'} gap-2`}>
        <svg ref={vennRef} width={vennWidth} height={VENN_HEIGHT} />
        <svg ref={heatRef} width={heatWidth} height={HEAT_HEIGHT} />
      </div>

      {/* Formula highlight on hover */}
      {hoveredRegion && (
        <div
          className="mt-2 px-3 py-2 rounded text-sm"
          style={{ backgroundColor: 'var(--color-definition-bg)', color: 'var(--color-text)' }}
        >
          {hoveredRegion === 'mi' && `I(X;Y) = H(X) − H(X|Y) = H(Y) − H(Y|X) = ${fmt(quantities.mi)} bits`}
          {hoveredRegion === 'hXgY' && `H(X|Y) = H(X) − I(X;Y) = ${fmt(quantities.hXgY)} bits`}
          {hoveredRegion === 'hYgX' && `H(Y|X) = H(Y) − I(X;Y) = ${fmt(quantities.hYgX)} bits`}
        </div>
      )}
    </div>
  );
}
