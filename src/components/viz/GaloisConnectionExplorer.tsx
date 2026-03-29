import { useState, useMemo, useEffect, useRef, useId } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  getGaloisPresets,
  checkGaloisConnection,
  closureOperator,
} from './shared/categoryTheory';

// ─── Constants ───

const SM_BREAKPOINT = 640;
const SVG_H = 380;
const NODE_R = 16;
const MATH_FONT = 'KaTeX_Math, Georgia, serif';

const COLORS = {
  leftAdj: '#3b82f6',    // blue — f (left adjoint)
  rightAdj: '#8b5cf6',   // purple — g (right adjoint)
  closure: '#f59e0b',    // amber — closure operator gf
  kernel: '#06b6d4',     // cyan — kernel operator fg
  valid: '#22c55e',
  invalid: '#ef4444',
  bg: '#f8fafc',
  text: '#1e293b',
  muted: '#94a3b8',
  node: '#e2e8f0',
  selected: '#3b82f6',
  closed: '#f59e0b',
  edge: '#cbd5e1',
};

// ─── Helpers ───

/** Compute layer (rank) for each element as the length of the longest path from a minimum. */
function computeLayers(elements: string[], morphisms: { source: string; target: string }[]): Map<string, number> {
  const layers = new Map<string, number>();

  // Build adjacency list (non-reflexive edges)
  const outEdges = new Map<string, string[]>();
  const inDegree = new Map<string, number>();
  elements.forEach((e) => { outEdges.set(e, []); inDegree.set(e, 0); });

  morphisms.forEach((m) => {
    if (m.source !== m.target) {
      outEdges.get(m.source)?.push(m.target);
      inDegree.set(m.target, (inDegree.get(m.target) ?? 0) + 1);
    }
  });

  // BFS longest-path (Kahn's algorithm variant): assign each node the
  // length of the longest path reaching it, so all paths are respected.
  const queue = elements.filter((e) => (inDegree.get(e) ?? 0) === 0);
  queue.forEach((e) => layers.set(e, 0));

  while (queue.length > 0) {
    const curr = queue.shift()!;
    const currLayer = layers.get(curr) ?? 0;
    for (const next of outEdges.get(curr) ?? []) {
      const newLayer = currLayer + 1;
      const existing = layers.get(next);
      if (existing === undefined || newLayer > existing) {
        layers.set(next, newLayer);
      }
      // Decrement in-degree; enqueue when all predecessors processed
      const remaining = (inDegree.get(next) ?? 1) - 1;
      inDegree.set(next, remaining);
      if (remaining === 0) {
        queue.push(next);
      }
    }
  }

  // Ensure all elements have a layer (fallback for cycles or disconnected)
  elements.forEach((e) => {
    if (!layers.has(e)) layers.set(e, 0);
  });

  return layers;
}

// ─── Component ───

export default function GaloisConnectionExplorer() {
  const presets = useMemo(() => getGaloisPresets(), []);
  const [presetIdx, setPresetIdx] = useState(0);
  const [selectedP, setSelectedP] = useState<string | null>(null);
  const [selectedQ, setSelectedQ] = useState<string | null>(null);
  const [showClosure, setShowClosure] = useState(false);

  const { ref: containerRef, width: cw } = useResizeObserver<HTMLDivElement>();
  const svgRef = useRef<SVGSVGElement>(null);
  const uid = useId().replace(/:/g, '');
  const isNarrow = cw < SM_BREAKPOINT;

  const data = useMemo(() => presets[presetIdx].build(), [presets, presetIdx]);

  // Reset selection on preset change
  useEffect(() => {
    const left = data.gc.leftPoset.objects;
    setSelectedP(left[0] ?? null);
    setSelectedQ(null);
  }, [data]);

  // Validation
  const validation = useMemo(() => checkGaloisConnection(data.gc), [data]);

  // Closure/kernel
  const closureP = useMemo(() => {
    if (!selectedP) return null;
    return closureOperator(data.gc, selectedP);
  }, [data, selectedP]);

  const closedElements = useMemo(() => {
    if (!showClosure) return new Set<string>();
    const closed = new Set<string>();
    for (const p of data.gc.leftPoset.objects) {
      const cl = closureOperator(data.gc, p);
      if (cl === p) closed.add(p);
    }
    return closed;
  }, [data, showClosure]);

  // f(p) and g(q) highlights
  const fp = useMemo(() => {
    if (!selectedP) return null;
    return data.gc.leftAdjoint.onObjects.get(selectedP) ?? null;
  }, [data, selectedP]);

  const gq = useMemo(() => {
    if (!selectedQ) return null;
    return data.gc.rightAdjoint.onObjects.get(selectedQ) ?? null;
  }, [data, selectedQ]);

  // ─── D3 Rendering ───
  useEffect(() => {
    if (!svgRef.current || cw < 100) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const w = cw;
    const h = SVG_H;
    const margin = { top: 40, right: 20, bottom: 40, left: 20 };

    // Defs
    const defs = svg.append('defs');
    const addMarker = (id: string, color: string) => {
      defs.append('marker')
        .attr('id', `${uid}-${id}`)
        .attr('viewBox', '0 0 10 10')
        .attr('refX', 8).attr('refY', 5)
        .attr('markerWidth', 5).attr('markerHeight', 5)
        .attr('orient', 'auto-start-reverse')
        .append('path').attr('d', 'M 0 0 L 10 5 L 0 10 z').attr('fill', color);
    };
    addMarker('left', COLORS.leftAdj);
    addMarker('right', COLORS.rightAdj);

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
    const innerW = w - margin.left - margin.right;
    const innerH = h - margin.top - margin.bottom;

    // Two Hasse diagrams: left poset and right poset
    const panelW = innerW * 0.35;
    const gapW = innerW * 0.3;
    const leftCx = panelW / 2;
    const rightCx = innerW - panelW / 2;

    // Labels
    g.append('text').attr('x', leftCx).attr('y', -12)
      .attr('text-anchor', 'middle')
      .style('font-family', MATH_FONT).style('font-size', '14px')
      .style('font-weight', 'bold').style('fill', COLORS.leftAdj)
      .text('P');
    g.append('text').attr('x', rightCx).attr('y', -12)
      .attr('text-anchor', 'middle')
      .style('font-family', MATH_FONT).style('font-size', '14px')
      .style('font-weight', 'bold').style('fill', COLORS.rightAdj)
      .text('Q');

    // Position nodes in Hasse diagram
    const drawHasse = (
      cat: { objects: string[]; morphisms: { source: string; target: string; label: string }[] },
      centerX: number,
      width: number,
      selected: string | null,
      highlighted: string | null,
      closedSet: Set<string>,
      onClick: (el: string) => void,
    ) => {
      const layers = computeLayers(cat.objects, cat.morphisms);
      const maxLayer = Math.max(...Array.from(layers.values()), 0);

      // Group elements by layer
      const byLayer = new Map<number, string[]>();
      cat.objects.forEach((obj) => {
        const layer = layers.get(obj) ?? 0;
        if (!byLayer.has(layer)) byLayer.set(layer, []);
        byLayer.get(layer)!.push(obj);
      });

      // Position nodes
      const nodePositions = new Map<string, { x: number; y: number }>();
      byLayer.forEach((elems, layer) => {
        const n = elems.length;
        elems.forEach((el, i) => {
          const x = centerX + (n === 1 ? 0 : (i - (n - 1) / 2) * Math.min(width / (n + 1), 40));
          const y = innerH - (maxLayer === 0 ? innerH / 2 : (layer / maxLayer) * (innerH - 40) + 20);
          nodePositions.set(el, { x, y });
        });
      });

      // Draw covering edges (upward)
      const coverings = cat.morphisms.filter((m) => {
        if (m.source === m.target) return false;
        const sLayer = layers.get(m.source) ?? 0;
        const tLayer = layers.get(m.target) ?? 0;
        return tLayer === sLayer + 1;
      });

      coverings.forEach((m) => {
        const s = nodePositions.get(m.source);
        const t = nodePositions.get(m.target);
        if (!s || !t) return;
        g.append('line')
          .attr('x1', s.x).attr('y1', s.y - NODE_R)
          .attr('x2', t.x).attr('y2', t.y + NODE_R)
          .attr('stroke', COLORS.edge)
          .attr('stroke-width', 1.5);
      });

      // Draw nodes
      cat.objects.forEach((obj) => {
        const pos = nodePositions.get(obj);
        if (!pos) return;
        const isSel = obj === selected;
        const isHL = obj === highlighted;
        const isClosed = closedSet.has(obj);

        let fill = COLORS.node;
        let stroke = COLORS.text;
        let sw = 1.5;
        if (isSel) { fill = COLORS.selected; stroke = COLORS.selected; sw = 2.5; }
        else if (isHL) { fill = COLORS.rightAdj; stroke = COLORS.rightAdj; sw = 2.5; }
        else if (isClosed) { fill = COLORS.closed; stroke = COLORS.closed; sw = 2; }

        g.append('circle')
          .attr('cx', pos.x).attr('cy', pos.y).attr('r', NODE_R)
          .attr('fill', fill).attr('stroke', stroke).attr('stroke-width', sw)
          .style('cursor', 'pointer')
          .on('click', () => onClick(obj));

        g.append('text')
          .attr('x', pos.x).attr('y', pos.y + 4)
          .attr('text-anchor', 'middle')
          .style('font-family', MATH_FONT).style('font-size', '11px')
          .style('fill', (isSel || isHL || isClosed) ? '#fff' : COLORS.text)
          .style('pointer-events', 'none')
          .text(obj);
      });

      return nodePositions;
    };

    const leftPositions = drawHasse(
      data.gc.leftPoset, leftCx, panelW, selectedP, gq, closedElements,
      (el) => { setSelectedP(el); setSelectedQ(null); },
    );
    const rightPositions = drawHasse(
      data.gc.rightPoset, rightCx, panelW, selectedQ, fp, new Set(),
      (el) => { setSelectedQ(el); setSelectedP(null); },
    );

    // Draw connecting arrows for selected elements
    if (selectedP && fp) {
      const sPos = leftPositions.get(selectedP);
      const tPos = rightPositions.get(fp);
      if (sPos && tPos) {
        g.append('path')
          .attr('d', `M ${sPos.x + NODE_R + 4} ${sPos.y} Q ${innerW / 2} ${sPos.y - 20} ${tPos.x - NODE_R - 4} ${tPos.y}`)
          .attr('fill', 'none')
          .attr('stroke', COLORS.leftAdj)
          .attr('stroke-width', 2)
          .attr('stroke-dasharray', '6,3')
          .attr('marker-end', `url(#${uid}-left)`);

        g.append('text')
          .attr('x', innerW / 2).attr('y', Math.min(sPos.y, tPos.y) - 25)
          .attr('text-anchor', 'middle')
          .style('font-family', MATH_FONT).style('font-size', '12px')
          .style('fill', COLORS.leftAdj)
          .text(`f(${selectedP}) = ${fp}`);
      }

      // Also show gf(p) = closure
      if (closureP) {
        const clPos = leftPositions.get(closureP);
        if (clPos && sPos) {
          g.append('text')
            .attr('x', leftCx).attr('y', innerH + 15)
            .attr('text-anchor', 'middle')
            .style('font-family', MATH_FONT).style('font-size', '11px')
            .style('fill', COLORS.closure)
            .text(`g∘f(${selectedP}) = ${closureP}${closureP === selectedP ? ' (closed)' : ''}`);
        }
      }
    }

    if (selectedQ && gq) {
      const sPos = rightPositions.get(selectedQ);
      const tPos = leftPositions.get(gq);
      if (sPos && tPos) {
        g.append('path')
          .attr('d', `M ${sPos.x - NODE_R - 4} ${sPos.y} Q ${innerW / 2} ${sPos.y + 20} ${tPos.x + NODE_R + 4} ${tPos.y}`)
          .attr('fill', 'none')
          .attr('stroke', COLORS.rightAdj)
          .attr('stroke-width', 2)
          .attr('stroke-dasharray', '6,3')
          .attr('marker-end', `url(#${uid}-right)`);

        g.append('text')
          .attr('x', innerW / 2).attr('y', Math.max(sPos.y, tPos.y) + 25)
          .attr('text-anchor', 'middle')
          .style('font-family', MATH_FONT).style('font-size', '12px')
          .style('fill', COLORS.rightAdj)
          .text(`g(${selectedQ}) = ${gq}`);
      }
    }

    // f ⊣ g label in the middle
    g.append('text')
      .attr('x', innerW / 2).attr('y', innerH / 2)
      .attr('text-anchor', 'middle')
      .style('font-family', MATH_FONT).style('font-size', '18px')
      .style('fill', COLORS.text)
      .text('f ⊣ g');

  }, [cw, data, selectedP, selectedQ, fp, gq, closureP, closedElements, uid, isNarrow]);

  return (
    <div ref={containerRef} style={{ margin: '1.5rem 0' }}>
      {/* Controls */}
      <div style={{
        display: 'flex', flexWrap: 'wrap', gap: '0.75rem', alignItems: 'center',
        marginBottom: '0.75rem', padding: '0.5rem 0.75rem',
        background: COLORS.bg, borderRadius: '8px', fontSize: '13px',
      }}>
        <label style={{ fontWeight: 600, color: COLORS.text }}>Example</label>
        <select
          value={presetIdx}
          onChange={(e) => setPresetIdx(+e.target.value)}
          style={{ padding: '4px 8px', borderRadius: '4px', border: `1px solid ${COLORS.muted}` }}
        >
          {presets.map((p, i) => (
            <option key={p.name} value={i}>{p.name}</option>
          ))}
        </select>

        <label style={{ display: 'flex', alignItems: 'center', gap: '4px', marginLeft: 'auto', cursor: 'pointer' }}>
          <input
            type="checkbox" checked={showClosure}
            onChange={(e) => setShowClosure(e.target.checked)}
          />
          <span style={{ color: COLORS.closure, fontWeight: 600, fontSize: '12px' }}>Show closed elements</span>
        </label>
      </div>

      {/* SVG */}
      <svg role="img" aria-label="Galois connection explorer visualization"
        ref={svgRef}
        width={cw}
        height={SVG_H}
        style={{ display: 'block', background: 'white', borderRadius: '8px', border: `1px solid ${COLORS.node}` }}
      />

      {/* Info */}
      <div style={{
        display: 'flex', flexDirection: isNarrow ? 'column' : 'row', gap: '0.5rem',
        marginTop: '0.5rem', fontSize: '12px',
      }}>
        <div style={{ flex: 2, padding: '0.5rem 0.75rem', background: COLORS.bg, borderRadius: '6px' }}>
          <strong style={{ color: COLORS.text }}>{presets[presetIdx].name}</strong>
          <p style={{ margin: '0.25rem 0 0', color: COLORS.muted, lineHeight: 1.4 }}>{data.description}</p>
        </div>
        <div style={{ flex: 1, padding: '0.5rem 0.75rem', background: COLORS.bg, borderRadius: '6px' }}>
          <strong style={{ color: COLORS.text }}>Verification</strong>
          <p style={{ margin: '0.25rem 0 0', color: validation.valid ? COLORS.valid : COLORS.invalid }}>
            f(p) ≤ q ⟺ p ≤ g(q) : {validation.valid ? '✓ valid' : `✗ ${validation.violations.length} violation(s)`}
          </p>
          {selectedP && <p style={{ margin: '0.25rem 0 0', color: COLORS.muted }}>
            Click elements to see f and g mappings
          </p>}
        </div>
      </div>
    </div>
  );
}
