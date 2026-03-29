import { useState, useEffect, useRef, useId, useMemo } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';

// ─── Constants ───

const SVG_H = 340;
const SM_BREAKPOINT = 640;
const MATH_FONT = 'KaTeX_Math, Georgia, serif';

const COLORS = {
  functor: '#8b5cf6',    // purple — comonad W
  counit: '#ef4444',     // red — extraction ε
  comult: '#f59e0b',     // amber — duplication δ
  focus: '#3b82f6',      // blue — focused element
  neighbor: '#22c55e',   // green — neighbors
  extend: '#8b5cf6',     // purple — extend result
  bg: '#f8fafc',
  text: '#1e293b',
  muted: '#94a3b8',
  node: '#e2e8f0',
  edge: '#cbd5e1',
};

// ─── Comonad data ───

interface ComonadDisplay {
  name: string;
  wDesc: string;
  epsDesc: string;
  deltaDesc: string;
  context: string;
}

const COMONAD_DISPLAYS: ComonadDisplay[] = [
  {
    name: 'Stream',
    wDesc: 'W(X) = Xᴺ  (infinite sequence with focus)',
    epsDesc: 'ε(s) = s₀  (extract head)',
    deltaDesc: 'δ(s) = [s, shift(s), shift²(s), ...]  (stream of all shifts)',
    context: 'Signal processing — each position sees the entire future stream',
  },
  {
    name: 'Neighborhood',
    wDesc: 'W(v) = (v, features(N(v)))  (node with neighborhood)',
    epsDesc: 'ε(v, ctx) = feature(v)  (extract focus node)',
    deltaDesc: 'δ(v, ctx) = (v, {(u, N(u)) | u ∈ N(v)})  (nested neighborhoods)',
    context: 'GNN message passing — each node sees its local neighborhood',
  },
  {
    name: 'Store',
    wDesc: 'W(X) = (S → X) × S  (lookup function + position)',
    epsDesc: 'ε(f, s) = f(s)  (evaluate at position)',
    deltaDesc: 'δ(f, s) = (λs′. (f, s′), s)  (all possible refocusings)',
    context: 'Cellular automata — each cell sees the entire grid through a lens',
  },
  {
    name: 'Environment',
    wDesc: 'W(X) = E × X  (value paired with environment)',
    epsDesc: 'ε(e, x) = x  (extract value, discard environment)',
    deltaDesc: 'δ(e, x) = (e, (e, x))  (duplicate environment)',
    context: 'Contextual computation — dual of Reader monad',
  },
];

// ─── Stream tape data ───

function fibStream(n: number): number[] {
  const s = [1, 1];
  for (let i = 2; i < n; i++) s.push(s[i - 1] + s[i - 2]);
  return s;
}

// ─── Graph data ───

const GRAPH_NODES = [
  { id: 'v₁', x: 0.2, y: 0.3, feature: 3 },
  { id: 'v₂', x: 0.5, y: 0.15, feature: 7 },
  { id: 'v₃', x: 0.8, y: 0.3, feature: 2 },
  { id: 'v₄', x: 0.35, y: 0.7, feature: 5 },
  { id: 'v₅', x: 0.65, y: 0.7, feature: 1 },
];

const GRAPH_EDGES: [string, string][] = [
  ['v₁', 'v₂'], ['v₁', 'v₄'],
  ['v₂', 'v₃'], ['v₂', 'v₄'], ['v₂', 'v₅'],
  ['v₃', 'v₅'],
  ['v₄', 'v₅'],
];

function getNeighbors(nodeId: string): string[] {
  const result: string[] = [];
  for (const [a, b] of GRAPH_EDGES) {
    if (a === nodeId) result.push(b);
    if (b === nodeId) result.push(a);
  }
  return result;
}

type AggFn = 'sum' | 'mean' | 'max';

function aggregateNeighbors(nodeId: string, aggFn: AggFn): number {
  const neighbors = getNeighbors(nodeId);
  const focusFeature = GRAPH_NODES.find((n) => n.id === nodeId)?.feature ?? 0;
  const neighborFeatures = neighbors.map(
    (nId) => GRAPH_NODES.find((n) => n.id === nId)?.feature ?? 0,
  );
  const allFeatures = [focusFeature, ...neighborFeatures];
  switch (aggFn) {
    case 'sum':
      return allFeatures.reduce((a, b) => a + b, 0);
    case 'mean':
      return Math.round((allFeatures.reduce((a, b) => a + b, 0) / allFeatures.length) * 10) / 10;
    case 'max':
      return Math.max(...allFeatures);
  }
}

// ─── Component ───

export default function ComonadExplorer() {
  const [comonadIdx, setComonadIdx] = useState(0);
  const [streamFocus, setStreamFocus] = useState(4); // center of 9
  const [graphFocus, setGraphFocus] = useState('v₂');
  const [aggFn, setAggFn] = useState<AggFn>('sum');
  const [showExtend, setShowExtend] = useState(false);

  const { ref: containerRef, width: cw } = useResizeObserver<HTMLDivElement>();
  const svgRef = useRef<SVGSVGElement>(null);
  const uid = useId().replace(/:/g, '');
  const isNarrow = cw < SM_BREAKPOINT;

  const display = COMONAD_DISPLAYS[comonadIdx];
  const isStream = comonadIdx === 0;
  const isGraph = comonadIdx === 1;

  const streamData = useMemo(() => fibStream(15), []);

  // Extended values for all graph nodes
  const extendedValues = useMemo(() => {
    const result = new Map<string, number>();
    for (const n of GRAPH_NODES) {
      result.set(n.id, aggregateNeighbors(n.id, aggFn));
    }
    return result;
  }, [aggFn]);

  // ─── D3: Stream tape ───
  useEffect(() => {
    if (!svgRef.current || cw < 100 || !isStream) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const w = cw;
    const h = SVG_H;
    const margin = { top: 20, right: 20, bottom: 20, left: 20 };
    const iw = w - margin.left - margin.right;
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const cellW = Math.min(60, iw / 9);
    const cellH = 50;
    const tapeY = 60;
    const visibleStart = Math.max(0, streamFocus - 4);
    const visibleCells = Math.min(9, streamData.length - visibleStart);

    // Tape label
    g.append('text')
      .attr('x', iw / 2)
      .attr('y', 20)
      .attr('text-anchor', 'middle')
      .style('font-size', '13px')
      .style('fill', COLORS.text)
      .text('Stream: Fibonacci sequence with focus');

    // Draw cells
    for (let i = 0; i < visibleCells; i++) {
      const idx = visibleStart + i;
      const cx = (iw - visibleCells * cellW) / 2 + i * cellW;
      const isFocused = idx === streamFocus;

      g.append('rect')
        .attr('x', cx)
        .attr('y', tapeY)
        .attr('width', cellW)
        .attr('height', cellH)
        .attr('rx', 4)
        .style('fill', isFocused ? '#dbeafe' : '#fff')
        .style('stroke', isFocused ? COLORS.focus : COLORS.edge)
        .style('stroke-width', isFocused ? 2.5 : 1)
        .style('cursor', 'pointer')
        .on('click', () => setStreamFocus(idx));

      g.append('text')
        .attr('x', cx + cellW / 2)
        .attr('y', tapeY + cellH / 2 + 5)
        .attr('text-anchor', 'middle')
        .style('font-family', MATH_FONT)
        .style('font-size', '16px')
        .style('fill', isFocused ? COLORS.focus : COLORS.text)
        .style('font-weight', isFocused ? 'bold' : 'normal')
        .style('pointer-events', 'none')
        .text(streamData[idx]);

      // Index label
      g.append('text')
        .attr('x', cx + cellW / 2)
        .attr('y', tapeY - 6)
        .attr('text-anchor', 'middle')
        .style('font-size', '10px')
        .style('fill', COLORS.muted)
        .text(`s${idx}`);
    }

    // Focus cursor
    const focusCx = (iw - visibleCells * cellW) / 2 + (streamFocus - visibleStart) * cellW + cellW / 2;
    g.append('text')
      .attr('x', focusCx)
      .attr('y', tapeY + cellH + 22)
      .attr('text-anchor', 'middle')
      .style('font-size', '12px')
      .style('fill', COLORS.focus)
      .text('▲ focus');

    // Extract result
    const extractY = tapeY + cellH + 50;
    g.append('text')
      .attr('x', iw / 2)
      .attr('y', extractY)
      .attr('text-anchor', 'middle')
      .style('font-family', MATH_FONT)
      .style('font-size', '14px')
      .style('fill', COLORS.counit)
      .text(`ε(stream) = ${streamData[streamFocus]}  (extract focused element)`);

    // Extend result if toggled
    if (showExtend) {
      const extendY = extractY + 30;
      // Running average of window [focus-1, focus, focus+1]
      const lo = Math.max(0, streamFocus - 1);
      const hi = Math.min(streamData.length - 1, streamFocus + 1);
      const windowVals = streamData.slice(lo, hi + 1);
      const avg = (windowVals.reduce((a, b) => a + b, 0) / windowVals.length).toFixed(1);

      g.append('text')
        .attr('x', iw / 2)
        .attr('y', extendY)
        .attr('text-anchor', 'middle')
        .style('font-family', MATH_FONT)
        .style('font-size', '14px')
        .style('fill', COLORS.extend)
        .text(`extend(avg₃)(stream) → new stream with values computed from local windows`);

      // Show extend results for visible cells
      for (let i = 0; i < visibleCells; i++) {
        const idx = visibleStart + i;
        const cx = (iw - visibleCells * cellW) / 2 + i * cellW;
        const lo2 = Math.max(0, idx - 1);
        const hi2 = Math.min(streamData.length - 1, idx + 1);
        const wVals = streamData.slice(lo2, hi2 + 1);
        const localAvg = (wVals.reduce((a, b) => a + b, 0) / wVals.length).toFixed(1);

        g.append('rect')
          .attr('x', cx)
          .attr('y', extendY + 10)
          .attr('width', cellW)
          .attr('height', 30)
          .attr('rx', 4)
          .style('fill', idx === streamFocus ? '#ede9fe' : '#faf5ff')
          .style('stroke', COLORS.extend)
          .style('stroke-width', 0.5);

        g.append('text')
          .attr('x', cx + cellW / 2)
          .attr('y', extendY + 30)
          .attr('text-anchor', 'middle')
          .style('font-family', MATH_FONT)
          .style('font-size', '13px')
          .style('fill', COLORS.extend)
          .text(localAvg);
      }
    }

  }, [cw, uid, isStream, streamFocus, streamData, showExtend]);

  // ─── D3: Graph neighborhood ───
  useEffect(() => {
    if (!svgRef.current || cw < 100 || !isGraph) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const w = cw;
    const h = SVG_H;
    const margin = { top: 20, right: 20, bottom: 20, left: 20 };
    const iw = w - margin.left - margin.right;
    const ih = h - margin.top - margin.bottom;
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const defs = svg.append('defs');
    defs.append('marker')
      .attr('id', `${uid}-graph-arrow`)
      .attr('viewBox', '0 0 10 10')
      .attr('refX', 20).attr('refY', 5)
      .attr('markerWidth', 5).attr('markerHeight', 5)
      .attr('orient', 'auto')
      .append('path').attr('d', 'M0,0 L10,5 L0,10 Z').style('fill', COLORS.edge);

    const neighbors = getNeighbors(graphFocus);
    const nodeR = 24;

    // Draw edges
    for (const [a, b] of GRAPH_EDGES) {
      const na = GRAPH_NODES.find((n) => n.id === a)!;
      const nb = GRAPH_NODES.find((n) => n.id === b)!;
      const isNeighborEdge =
        (a === graphFocus || b === graphFocus) ||
        (neighbors.includes(a) && neighbors.includes(b));
      const highlighted = a === graphFocus || b === graphFocus;

      g.append('line')
        .attr('x1', na.x * iw)
        .attr('y1', na.y * ih)
        .attr('x2', nb.x * iw)
        .attr('y2', nb.y * ih)
        .style('stroke', highlighted ? COLORS.neighbor : COLORS.edge)
        .style('stroke-width', highlighted ? 2 : 1)
        .style('opacity', highlighted ? 1 : 0.4);
    }

    // Draw nodes
    for (const node of GRAPH_NODES) {
      const isFocused = node.id === graphFocus;
      const isNeighbor = neighbors.includes(node.id);
      const nx = node.x * iw;
      const ny = node.y * ih;

      let fillColor = COLORS.node;
      let strokeColor = COLORS.muted;
      if (isFocused) { fillColor = '#dbeafe'; strokeColor = COLORS.focus; }
      else if (isNeighbor) { fillColor = '#dcfce7'; strokeColor = COLORS.neighbor; }

      g.append('circle')
        .attr('cx', nx).attr('cy', ny)
        .attr('r', nodeR)
        .style('fill', fillColor)
        .style('stroke', strokeColor)
        .style('stroke-width', isFocused ? 3 : isNeighbor ? 2 : 1)
        .style('cursor', 'pointer')
        .on('click', () => setGraphFocus(node.id));

      // Feature value
      const displayVal = showExtend ? extendedValues.get(node.id) : node.feature;
      g.append('text')
        .attr('x', nx).attr('y', ny + 5)
        .attr('text-anchor', 'middle')
        .style('font-family', MATH_FONT)
        .style('font-size', '16px')
        .style('fill', showExtend ? COLORS.extend : COLORS.text)
        .style('font-weight', isFocused ? 'bold' : 'normal')
        .style('pointer-events', 'none')
        .text(displayVal ?? node.feature);

      // Node label
      g.append('text')
        .attr('x', nx).attr('y', ny - nodeR - 6)
        .attr('text-anchor', 'middle')
        .style('font-size', '11px')
        .style('fill', isFocused ? COLORS.focus : COLORS.muted)
        .style('pointer-events', 'none')
        .text(node.id);
    }

    // Extract label
    g.append('text')
      .attr('x', iw / 2)
      .attr('y', ih - 5)
      .attr('text-anchor', 'middle')
      .style('font-family', MATH_FONT)
      .style('font-size', '13px')
      .style('fill', COLORS.counit)
      .text(`ε(${graphFocus}) = ${GRAPH_NODES.find((n) => n.id === graphFocus)?.feature}  (extract focus feature)`);

    if (showExtend) {
      g.append('text')
        .attr('x', iw / 2)
        .attr('y', ih + 15)
        .attr('text-anchor', 'middle')
        .style('font-size', '12px')
        .style('fill', COLORS.extend)
        .text(`extend(${aggFn})(${graphFocus}) = ${extendedValues.get(graphFocus)}  — applied at every node`);
    }

  }, [cw, uid, isGraph, graphFocus, showExtend, aggFn, extendedValues]);

  // Reset showExtend on comonad change
  useEffect(() => { setShowExtend(false); }, [comonadIdx]);

  return (
    <div
      ref={containerRef}
      style={{
        margin: '1.5rem 0',
        background: '#ffffff',
        border: '1px solid #e2e8f0',
        borderRadius: '8px',
        overflow: 'hidden',
      }}
    >
      {/* Controls */}
      <div
        style={{
          padding: '12px 16px',
          borderBottom: '1px solid #e2e8f0',
          display: 'flex',
          flexWrap: 'wrap',
          gap: '12px',
          alignItems: 'center',
          background: COLORS.bg,
        }}
      >
        <label style={{ fontSize: '13px', fontWeight: 600, color: COLORS.text }}>
          Comonad:
          <select
            value={comonadIdx}
            onChange={(e) => setComonadIdx(Number(e.target.value))}
            style={{
              marginLeft: '6px', padding: '4px 8px', borderRadius: '4px',
              border: '1px solid #cbd5e1', fontSize: '13px',
            }}
          >
            {COMONAD_DISPLAYS.map((c, i) => (
              <option key={c.name} value={i}>{c.name}</option>
            ))}
          </select>
        </label>

        {isGraph && (
          <label style={{ fontSize: '13px', fontWeight: 600, color: COLORS.text }}>
            Aggregation:
            <select
              value={aggFn}
              onChange={(e) => setAggFn(e.target.value as AggFn)}
              style={{
                marginLeft: '6px', padding: '4px 8px', borderRadius: '4px',
                border: '1px solid #cbd5e1', fontSize: '13px',
              }}
            >
              <option value="sum">Sum</option>
              <option value="mean">Mean</option>
              <option value="max">Max</option>
            </select>
          </label>
        )}

        {(isStream || isGraph) && (
          <button
            onClick={() => setShowExtend(!showExtend)}
            style={{
              padding: '4px 14px', borderRadius: '4px',
              border: '1px solid #cbd5e1',
              background: showExtend ? COLORS.extend : '#fff',
              color: showExtend ? '#fff' : COLORS.text,
              fontSize: '13px', cursor: 'pointer',
            }}
          >
            {showExtend ? 'Hide extend' : 'Extend'}
          </button>
        )}
      </div>

      {/* SVG area for Stream and Graph */}
      {(isStream || isGraph) && (
        <svg ref={svgRef} width={cw} height={SVG_H} />
      )}

      {/* Text description for Store and Environment */}
      {!isStream && !isGraph && (
        <div style={{ padding: '24px 16px', textAlign: 'center' }}>
          <div style={{ fontSize: '16px', fontFamily: MATH_FONT, color: COLORS.text, marginBottom: '12px' }}>
            {display.wDesc}
          </div>
          <div style={{ fontSize: '14px', color: COLORS.counit, marginBottom: '6px' }}>
            {display.epsDesc}
          </div>
          <div style={{ fontSize: '14px', color: COLORS.comult, marginBottom: '12px' }}>
            {display.deltaDesc}
          </div>
          <div style={{ fontSize: '13px', color: COLORS.muted }}>
            {display.context}
          </div>
        </div>
      )}

      {/* Description footer */}
      <div
        style={{
          padding: '10px 16px',
          borderTop: '1px solid #e2e8f0',
          background: COLORS.bg,
        }}
      >
        <div style={{ fontSize: '12px', color: COLORS.muted }}>
          <strong style={{ color: COLORS.text }}>{display.name} comonad:</strong>{' '}
          {display.context}
        </div>
        <div style={{ fontSize: '12px', fontFamily: MATH_FONT, color: COLORS.muted, marginTop: '4px' }}>
          ε: {display.epsDesc} &nbsp;|&nbsp; δ: {display.deltaDesc}
        </div>
      </div>
    </div>
  );
}
