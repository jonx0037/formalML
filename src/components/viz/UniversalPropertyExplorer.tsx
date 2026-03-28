import { useState, useEffect, useRef, useMemo, useId } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { productMediatingSet, coproductMediatingSet } from './shared/categoryTheory';

type ConstructionType = 'product' | 'coproduct' | 'initial' | 'terminal';
type CategoryType = 'set' | 'vec' | 'poset';
interface NodeLayout { id: string; label: string; x: number; y: number; r: number }
interface ArrowLayout {
  id: string; label: string; source: string; target: string;
  stroke: string; dashed: boolean; curve: number;
}

const SVG_HEIGHT = 340;
const SM_BREAKPOINT = 640;
const NODE_R = 24;
const NODE_R_SM = 20;
const ARROW_HEAD = 8;

function getConcreteData(cat: CategoryType, construction: ConstructionType) {
  if (cat === 'set') {
    if (construction === 'product') {
      const f = new Map([['x', '1'], ['y', '2']]);
      const g = new Map([['x', 'a'], ['y', 'b']]);
      const h = productMediatingSet(['x', 'y'], f, g);
      return {
        A: '{1, 2}', B: '{a, b}', universal: '{(1,a),(1,b),(2,a),(2,b)}',
        Z: '{x, y}', f: 'x\u21a61, y\u21a62', g: 'x\u21a6a, y\u21a6b',
        h: [...h.entries()].map(([k, [a, b]]) => `${k}\u21a6(${a},${b})`).join(', '),
      };
    }
    if (construction === 'coproduct') {
      const f = new Map([['1', 'x'], ['2', 'y']]);
      const g = new Map([['a', 'x'], ['b', 'y']]);
      const hMap = coproductMediatingSet(f, g);
      return {
        A: '{1, 2}', B: '{a, b}', universal: '{1, 2, a, b}',
        Z: '{x, y}', f: '1\u21a6x, 2\u21a6y', g: 'a\u21a6x, b\u21a6y',
        h: [...hMap.entries()].map(([k, v]) => `${k}\u21a6${v}`).join(', '),
      };
    }
    return { A: '', B: '', universal: '', Z: '{x, y}', f: '', g: '', h: '' };
  }
  if (cat === 'vec') {
    if (construction === 'product') {
      return {
        A: '\u211d\u00b2', B: '\u211d\u00b3', universal: '\u211d\u2075 = \u211d\u00b2 \u2295 \u211d\u00b3',
        Z: '\u211d', f: '[1, 0]\u1d40', g: '[1, 0, 0]\u1d40',
        h: '[1, 0, 1, 0, 0]\u1d40',
      };
    }
    if (construction === 'coproduct') {
      return {
        A: '\u211d\u00b2', B: '\u211d\u00b3', universal: '\u211d\u2075 = \u211d\u00b2 \u2295 \u211d\u00b3',
        Z: '\u211d', f: '[1, 1]', g: '[1, 1, 1]',
        h: '[1, 1, 1, 1, 1]',
      };
    }
    return { A: '', B: '', universal: '', Z: '\u211d', f: '', g: '', h: '' };
  }
  // poset
  if (construction === 'product') {
    return {
      A: 'a', B: 'b', universal: 'a \u2227 b',
      Z: 'z', f: 'z \u2264 a', g: 'z \u2264 b',
      h: 'z \u2264 a\u2227b',
    };
  }
  if (construction === 'coproduct') {
    return {
      A: 'a', B: 'b', universal: 'a \u2228 b',
      Z: 'z', f: 'a \u2264 z', g: 'b \u2264 z',
      h: 'a\u2228b \u2264 z',
    };
  }
  return { A: '', B: '', universal: '', Z: 'z', f: '', g: '', h: '' };
}

function computeLayout(
  construction: ConstructionType,
  w: number,
  h: number,
  isSmall: boolean,
): { nodes: NodeLayout[]; arrows: ArrowLayout[]; note?: string } {
  const r = isSmall ? NODE_R_SM : NODE_R;
  const cx = w / 2;
  const cy = h / 2;
  const spread = Math.min(w * 0.35, 160);
  const vSpread = Math.min(h * 0.32, 120);

  if (construction === 'product') {
    const nodes: NodeLayout[] = [
      { id: 'A', label: 'A', x: cx - spread, y: cy - vSpread * 0.3, r },
      { id: 'B', label: 'B', x: cx + spread, y: cy - vSpread * 0.3, r },
      { id: 'P', label: 'A\u00d7B', x: cx, y: cy - vSpread, r: r + 4 },
      { id: 'Z', label: 'Z', x: cx, y: cy + vSpread, r },
    ];
    const arrows: ArrowLayout[] = [
      { id: 'pi1', label: '\u03c0\u2081', source: 'P', target: 'A', stroke: 'var(--color-text)', dashed: false, curve: 0 },
      { id: 'pi2', label: '\u03c0\u2082', source: 'P', target: 'B', stroke: 'var(--color-text)', dashed: false, curve: 0 },
      { id: 'f', label: 'f', source: 'Z', target: 'A', stroke: '#3b82f6', dashed: false, curve: 20 },
      { id: 'g', label: 'g', source: 'Z', target: 'B', stroke: '#3b82f6', dashed: false, curve: -20 },
      { id: 'h', label: 'h = \u27e8f, g\u27e9', source: 'Z', target: 'P', stroke: '#ef4444', dashed: true, curve: 0 },
    ];
    return { nodes, arrows };
  }

  if (construction === 'coproduct') {
    const nodes: NodeLayout[] = [
      { id: 'A', label: 'A', x: cx - spread, y: cy - vSpread * 0.3, r },
      { id: 'B', label: 'B', x: cx + spread, y: cy - vSpread * 0.3, r },
      { id: 'C', label: 'A\u2294B', x: cx, y: cy - vSpread, r: r + 4 },
      { id: 'Z', label: 'Z', x: cx, y: cy + vSpread, r },
    ];
    const arrows: ArrowLayout[] = [
      { id: 'i1', label: '\u03b9\u2081', source: 'A', target: 'C', stroke: 'var(--color-text)', dashed: false, curve: 0 },
      { id: 'i2', label: '\u03b9\u2082', source: 'B', target: 'C', stroke: 'var(--color-text)', dashed: false, curve: 0 },
      { id: 'f', label: 'f', source: 'A', target: 'Z', stroke: '#3b82f6', dashed: false, curve: -20 },
      { id: 'g', label: 'g', source: 'B', target: 'Z', stroke: '#3b82f6', dashed: false, curve: 20 },
      { id: 'h', label: 'h = [f, g]', source: 'C', target: 'Z', stroke: '#ef4444', dashed: true, curve: 0 },
    ];
    return { nodes, arrows };
  }

  if (construction === 'initial') {
    const nodes: NodeLayout[] = [
      { id: 'I', label: '\u2205', x: cx - spread * 0.8, y: cy, r },
      { id: 'Z', label: 'Z', x: cx + spread * 0.8, y: cy, r },
    ];
    const arrows: ArrowLayout[] = [
      { id: 'bang', label: '!', source: 'I', target: 'Z', stroke: '#ef4444', dashed: true, curve: 0 },
    ];
    return { nodes, arrows, note: 'For every object Z, there exists exactly one morphism \u2205 \u2192 Z' };
  }

  // terminal
  const nodes: NodeLayout[] = [
    { id: 'Z', label: 'Z', x: cx - spread * 0.8, y: cy, r },
    { id: 'T', label: '*', x: cx + spread * 0.8, y: cy, r },
  ];
  const arrows: ArrowLayout[] = [
    { id: 'bang', label: '!', source: 'Z', target: 'T', stroke: '#ef4444', dashed: true, curve: 0 },
  ];
  return { nodes, arrows, note: 'For every object Z, there exists exactly one morphism Z \u2192 *' };
}

function getEquation(construction: ConstructionType): string {
  switch (construction) {
    case 'product':
      return '\u03c0\u2081 \u2218 h = f   and   \u03c0\u2082 \u2218 h = g   where   h = \u27e8f, g\u27e9';
    case 'coproduct':
      return 'h \u2218 \u03b9\u2081 = f   and   h \u2218 \u03b9\u2082 = g   where   h = [f, g]';
    case 'initial':
      return '\u2203! morphism  ! : \u2205 \u2192 Z   for every object Z';
    case 'terminal':
      return '\u2203! morphism  ! : Z \u2192 *   for every object Z';
  }
}

function arrowPath(
  sx: number, sy: number, tx: number, ty: number,
  sr: number, tr: number, curve: number,
): { path: string; labelX: number; labelY: number; angle: number } {
  const dx = tx - sx;
  const dy = ty - sy;
  const dist = Math.sqrt(dx * dx + dy * dy) || 1;
  const ux = dx / dist;
  const uy = dy / dist;

  // Start and end, offset by radii
  const x1 = sx + ux * (sr + 4);
  const y1 = sy + uy * (sr + 4);
  const x2 = tx - ux * (tr + ARROW_HEAD + 2);
  const y2 = ty - uy * (tr + ARROW_HEAD + 2);

  // Control point for quadratic bezier
  const mx = (x1 + x2) / 2 - uy * curve;
  const my = (y1 + y2) / 2 + ux * curve;

  const path = `M ${x1} ${y1} Q ${mx} ${my} ${x2} ${y2}`;

  // Label position at t=0.5 of quadratic bezier, offset perpendicular
  const labelX = 0.25 * x1 + 0.5 * mx + 0.25 * x2;
  const labelY = 0.25 * y1 + 0.5 * my + 0.25 * y2;

  const angle = Math.atan2(y2 - y1, x2 - x1) * (180 / Math.PI);

  return { path, labelX, labelY, angle };
}

export default function UniversalPropertyExplorer() {
  const { ref: containerRef, width: containerWidth } =
    useResizeObserver<HTMLDivElement>();
  const instanceId = useId().replace(/:/g, '');

  const svgRef = useRef<SVGSVGElement>(null);

  const [constructionType, setConstructionType] = useState<ConstructionType>('product');
  const [categoryType, setCategoryType] = useState<CategoryType>('set');
  const [showConcreteValues, setShowConcreteValues] = useState(false);
  const [mediatingVisible, setMediatingVisible] = useState(false);

  const isSmall = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const svgWidth = containerWidth > 0 ? containerWidth - 32 : 500;

  const layout = useMemo(
    () => computeLayout(constructionType, svgWidth, SVG_HEIGHT, isSmall),
    [constructionType, svgWidth, isSmall],
  );

  const concrete = useMemo(
    () => getConcreteData(categoryType, constructionType),
    [categoryType, constructionType],
  );

  // Reset mediating arrow animation on construction change
  useEffect(() => {
    setMediatingVisible(false);
    const timer = setTimeout(() => setMediatingVisible(true), 400);
    return () => clearTimeout(timer);
  }, [constructionType, categoryType]);

  useEffect(() => {
    if (!svgRef.current || svgWidth <= 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const { nodes, arrows, note } = layout;
    const nodeMap = new Map(nodes.map((n) => [n.id, n]));

    // Arrowhead markers
    const defs = svg.append('defs');
    const markerColors = [
      { suffix: 'black', color: 'var(--color-text)' },
      { suffix: 'blue', color: '#3b82f6' },
      { suffix: 'red', color: '#ef4444' },
    ];
    for (const { suffix, color } of markerColors) {
      defs.append('marker')
        .attr('id', `arrow-${suffix}-${instanceId}`)
        .attr('viewBox', '0 0 10 10')
        .attr('refX', 10)
        .attr('refY', 5)
        .attr('markerWidth', ARROW_HEAD)
        .attr('markerHeight', ARROW_HEAD)
        .attr('orient', 'auto-start-reverse')
        .append('path')
        .attr('d', 'M 0 0 L 10 5 L 0 10 z')
        .style('fill', color);
    }

    const arrowGroup = svg.append('g');
    const nodeGroup = svg.append('g');
    const labelGroup = svg.append('g');

    // Draw arrows
    for (const arrow of arrows) {
      const sn = nodeMap.get(arrow.source);
      const tn = nodeMap.get(arrow.target);
      if (!sn || !tn) continue;

      const isMediating = arrow.id === 'h' || arrow.id === 'bang';
      if (isMediating && !mediatingVisible) continue;

      const { path, labelX, labelY } = arrowPath(
        sn.x, sn.y, tn.x, tn.y, sn.r, tn.r, arrow.curve,
      );

      const markerSuffix = arrow.stroke === '#ef4444'
        ? 'red'
        : arrow.stroke === '#3b82f6'
          ? 'blue'
          : 'black';
      const markerId = `arrow-${markerSuffix}-${instanceId}`;

      const pathEl = arrowGroup.append('path')
        .attr('d', path)
        .style('fill', 'none')
        .style('stroke', arrow.stroke)
        .style('stroke-width', '2')
        .attr('marker-end', `url(#${markerId})`);

      if (arrow.dashed) {
        pathEl.style('stroke-dasharray', '6,4');
      }

      // Animate mediating arrow
      if (isMediating) {
        const totalLength = (pathEl.node() as SVGPathElement).getTotalLength();
        pathEl
          .style('stroke-dasharray', `${totalLength}`)
          .style('stroke-dashoffset', `${totalLength}`)
          .style('opacity', '0')
          .transition()
          .duration(600)
          .style('opacity', '1')
          .style('stroke-dashoffset', '0')
          .on('end', function () {
            d3.select(this).style('stroke-dasharray', '6,4');
          });
      }

      // Arrow label
      const perpOffset = arrow.curve >= 0 ? -10 : 10;
      labelGroup.append('text')
        .attr('x', labelX)
        .attr('y', labelY + perpOffset)
        .attr('text-anchor', 'middle')
        .attr('font-size', isSmall ? 11 : 13)
        .attr('font-weight', 600)
        .attr('font-style', 'italic')
        .style('fill', arrow.stroke === 'var(--color-text)' ? 'var(--color-text-secondary)' : arrow.stroke)
        .text(arrow.label);
    }

    // Draw nodes
    for (const node of nodes) {
      nodeGroup.append('circle')
        .attr('cx', node.x)
        .attr('cy', node.y)
        .attr('r', node.r)
        .style('fill', 'var(--color-bg)')
        .style('stroke', 'var(--color-text)')
        .style('stroke-width', '2');

      nodeGroup.append('text')
        .attr('x', node.x)
        .attr('y', node.y + (isSmall ? 4 : 5))
        .attr('text-anchor', 'middle')
        .attr('font-size', isSmall ? 13 : 15)
        .attr('font-weight', 700)
        .style('fill', 'var(--color-text)')
        .text(node.label);
    }

    // Note text for initial/terminal
    if (note) {
      svg.append('text')
        .attr('x', svgWidth / 2)
        .attr('y', SVG_HEIGHT - 24)
        .attr('text-anchor', 'middle')
        .attr('font-size', isSmall ? 11 : 13)
        .style('fill', 'var(--color-text-secondary)')
        .text(note);
    }

    // Concrete value annotations
    if (showConcreteValues && (constructionType === 'product' || constructionType === 'coproduct')) {
      const annotFontSize = isSmall ? 9 : 10;
      for (const node of nodes) {
        let detail = '';
        if (node.id === 'A') detail = concrete.A;
        else if (node.id === 'B') detail = concrete.B;
        else if (node.id === 'P' || node.id === 'C') detail = concrete.universal;
        else if (node.id === 'Z') detail = concrete.Z;

        if (detail) {
          svg.append('text')
            .attr('x', node.x)
            .attr('y', node.y + node.r + 14)
            .attr('text-anchor', 'middle')
            .attr('font-size', annotFontSize)
            .style('fill', 'var(--color-text-secondary)')
            .text(detail);
        }
      }
    }
  }, [layout, svgWidth, isSmall, mediatingVisible, showConcreteValues, concrete, constructionType]);

  const constructionButtons: { key: ConstructionType; label: string }[] = [
    { key: 'product', label: 'Product' },
    { key: 'coproduct', label: 'Coproduct' },
    { key: 'initial', label: 'Initial' },
    { key: 'terminal', label: 'Terminal' },
  ];

  const showCategoryControls = constructionType === 'product' || constructionType === 'coproduct';

  return (
    <div
      ref={containerRef}
      className="w-full rounded-lg border p-4"
      style={{
        borderColor: 'var(--color-border)',
        backgroundColor: 'var(--color-surface)',
      }}
    >
      {/* Controls */}
      <div className="mb-4 flex flex-wrap items-center gap-3">
        {/* Construction type toggle */}
        <div className="flex rounded border" style={{ borderColor: 'var(--color-border)' }}>
          {constructionButtons.map(({ key, label }) => (
            <button
              key={key}
              className="px-3 py-1 text-sm font-medium"
              style={{
                backgroundColor: constructionType === key ? 'var(--color-text)' : 'var(--color-bg)',
                color: constructionType === key ? 'var(--color-bg)' : 'var(--color-text)',
                borderRight: key !== 'terminal' ? '1px solid var(--color-border)' : 'none',
              }}
              onClick={() => setConstructionType(key)}
            >
              {label}
            </button>
          ))}
        </div>

        {/* Category dropdown (only for product/coproduct) */}
        {showCategoryControls && (
          <label className="flex items-center gap-1.5 text-sm" style={{ color: 'var(--color-text)' }}>
            <span className="font-medium">Category:</span>
            <select
              className="rounded border px-2 py-1 text-sm"
              style={{
                borderColor: 'var(--color-border)',
                backgroundColor: 'var(--color-bg)',
                color: 'var(--color-text)',
              }}
              value={categoryType}
              onChange={(e) => setCategoryType(e.target.value as CategoryType)}
            >
              <option value="set">Set</option>
              <option value="vec">Vec</option>
              <option value="poset">Poset</option>
            </select>
          </label>
        )}

        {/* Concrete values checkbox */}
        {showCategoryControls && (
          <label className="flex items-center gap-1.5 text-sm" style={{ color: 'var(--color-text)' }}>
            <input
              type="checkbox"
              checked={showConcreteValues}
              onChange={(e) => setShowConcreteValues(e.target.checked)}
            />
            <span>Show concrete values</span>
          </label>
        )}
      </div>

      {/* SVG Diagram */}
      <svg
        ref={svgRef}
        width={svgWidth}
        height={SVG_HEIGHT}
        style={{
          border: '1px solid var(--color-border)',
          borderRadius: 8,
          backgroundColor: 'var(--color-bg)',
          display: 'block',
          margin: '0 auto',
        }}
      />

      {/* Equation display */}
      <div
        className="mt-3 rounded px-4 py-2 text-center"
        style={{
          backgroundColor: 'var(--color-bg)',
          border: '1px solid var(--color-border)',
          fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
          fontSize: isSmall ? 12 : 14,
          color: 'var(--color-text)',
          letterSpacing: '0.02em',
        }}
      >
        {getEquation(constructionType)}
      </div>

      {/* Concrete mapping table */}
      {showConcreteValues && showCategoryControls && (
        <div
          className="mt-2 rounded px-4 py-2 text-sm"
          style={{
            backgroundColor: 'var(--color-bg)',
            border: '1px solid var(--color-border)',
            color: 'var(--color-text-secondary)',
          }}
        >
          <div className="flex flex-wrap gap-x-6 gap-y-1">
            <span><strong style={{ color: '#3b82f6' }}>f:</strong> {concrete.f}</span>
            <span><strong style={{ color: '#3b82f6' }}>g:</strong> {concrete.g}</span>
            <span><strong style={{ color: '#ef4444' }}>h:</strong> {concrete.h}</span>
          </div>
        </div>
      )}

      {/* Legend */}
      <div className="mt-3 flex flex-wrap gap-4 text-xs" style={{ color: 'var(--color-text-secondary)' }}>
        {([
          ['var(--color-text)', false, 'Structural morphisms'],
          ['#3b82f6', false, 'User-defined morphisms'],
          ['#ef4444', true, 'Unique mediating morphism'],
        ] as const).map(([c, dashed, label]) => (
          <span key={label}>
            <span style={{ display: 'inline-block', width: 16, height: 0, borderTop: `2px ${dashed ? 'dashed' : 'solid'} ${c}`, verticalAlign: 'middle', marginRight: 4 }} />
            {label}
          </span>
        ))}
      </div>
    </div>
  );
}
