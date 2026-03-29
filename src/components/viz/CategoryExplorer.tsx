import { useState, useMemo, useCallback, useEffect, useRef, useId } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  getCategoryPresets,
  compositionTable,
  checkAssociativity,
  checkIdentity,
  type Category,
  type Morphism,
} from './shared/categoryTheory';

// ─── Layout constants ───

const SM_BREAKPOINT = 640;
const GRAPH_PANEL_HEIGHT = 380;
const NODE_RADIUS = 24;
const LOOP_RADIUS = 16;

// ─── Helpers ───

interface MorphismLink { source: string; target: string; label: string; index: number; total: number; }

function morphismLinks(cat: Category): MorphismLink[] {
  const nonId = cat.morphisms.filter((m) => !m.isIdentity);
  const groups = new Map<string, Morphism[]>();
  for (const m of nonId) {
    const key = `${m.source}->${m.target}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key)!.push(m);
  }
  const links: MorphismLink[] = [];
  for (const [, morphisms] of groups) {
    const total = morphisms.length;
    morphisms.forEach((m, i) => {
      links.push({ source: m.source, target: m.target, label: m.label, index: i, total });
    });
  }
  return links;
}

/** Compute a curved path for a directed edge between two points. */
function arcPath(
  sx: number,
  sy: number,
  tx: number,
  ty: number,
  curvature: number,
): string {
  const dx = tx - sx;
  const dy = ty - sy;
  const dr = Math.sqrt(dx * dx + dy * dy) / curvature;
  return `M${sx},${sy} A${dr},${dr} 0 0,1 ${tx},${ty}`;
}

/** Compute a self-loop path at a given position. */
function loopPath(cx: number, cy: number): string {
  const r = LOOP_RADIUS;
  const topY = cy - NODE_RADIUS - 2;
  return (
    `M${cx - r * 0.5},${topY}` +
    ` C${cx - r * 1.5},${topY - r * 2} ${cx + r * 1.5},${topY - r * 2} ${cx + r * 0.5},${topY}`
  );
}

// ─── Node colors ───

const NODE_COLORS = [
  '#3b82f6', '#8b5cf6', '#06b6d4', '#f59e0b', '#ef4444',
  '#10b981', '#ec4899', '#6366f1',
];

// ─── Component ───

export default function CategoryExplorer() {
  const { ref: containerRef, width: containerWidth } =
    useResizeObserver<HTMLDivElement>();
  const instanceId = useId().replace(/:/g, '');

  const graphSvgRef = useRef<SVGSVGElement>(null);

  // ─── State ───

  const presets = useMemo(() => getCategoryPresets(), []);
  const [presetIndex, setPresetIndex] = useState(0);
  const [category, setCategory] = useState<Category>(() => presets[0].build());
  const [showIdentities, setShowIdentities] = useState(true);
  const [showCompTable, setShowCompTable] = useState(true);

  // Force simulation refs
  const positionsRef = useRef<Map<string, { x: number; y: number }>>(new Map());
  const simRef = useRef<d3.Simulation<
    { id: string; x: number; y: number },
    undefined
  > | null>(null);
  const [tick, setTick] = useState(0);

  // ─── Derived computations ───

  const assocResult = useMemo(() => checkAssociativity(category), [category]);
  const identResult = useMemo(() => checkIdentity(category), [category]);
  const compTable = useMemo(() => compositionTable(category), [category]);
  const links = useMemo(() => morphismLinks(category), [category]);

  // ─── Layout calculations ───

  const isNarrow = containerWidth > 0 && containerWidth < SM_BREAKPOINT;

  const graphPanelWidth = useMemo(() => {
    if (!containerWidth) return 400;
    if (isNarrow) return containerWidth - 16;
    return showCompTable
      ? Math.floor((containerWidth - 32) * 0.6)
      : containerWidth - 16;
  }, [containerWidth, isNarrow, showCompTable]);

  // ─── Force simulation management ───

  const initSimulation = useCallback(
    (cat: Category) => {
      if (simRef.current) simRef.current.stop();

      const n = cat.objects.length;
      const cx = graphPanelWidth / 2;
      const cy = GRAPH_PANEL_HEIGHT / 2;
      const spreadR = Math.min(graphPanelWidth, GRAPH_PANEL_HEIGHT) * 0.25;

      const nodes = cat.objects.map((id, i) => ({
        id,
        x: cx + Math.cos((2 * Math.PI * i) / n - Math.PI / 2) * spreadR,
        y: cy + Math.sin((2 * Math.PI * i) / n - Math.PI / 2) * spreadR,
      }));

      const posMap = new Map<string, { x: number; y: number }>();
      for (const nd of nodes) posMap.set(nd.id, nd);
      positionsRef.current = posMap;

      const simLinks = links.map((l) => ({
        source: l.source,
        target: l.target,
      }));

      const sim = d3
        .forceSimulation(nodes)
        .force(
          'link',
          d3
            .forceLink<{ id: string; x: number; y: number }, { source: string; target: string }>(simLinks)
            .id((d) => d.id)
            .distance(120),
        )
        .force('charge', d3.forceManyBody().strength(-400))
        .force('center', d3.forceCenter(cx, cy))
        .force('collide', d3.forceCollide(NODE_RADIUS + 20))
        .alphaDecay(0.04)
        .on('tick', () => {
          const pad = NODE_RADIUS + 10;
          for (const node of nodes) {
            node.x = Math.max(pad, Math.min(graphPanelWidth - pad, node.x));
            node.y = Math.max(pad, Math.min(GRAPH_PANEL_HEIGHT - pad, node.y));
          }
          setTick((t) => t + 1);
        });

      simRef.current = sim;
    },
    [graphPanelWidth, links],
  );

  // Re-init when category changes
  useEffect(() => {
    initSimulation(category);
    return () => {
      if (simRef.current) simRef.current.stop();
    };
  }, [category, initSimulation]);

  // ─── Interaction handlers ───

  const handlePresetChange = useCallback(
    (idx: number) => {
      setPresetIndex(idx);
      setCategory(presets[idx].build());
    },
    [presets],
  );

  // ─── Graph panel rendering ───

  useEffect(() => {
    if (!graphSvgRef.current || graphPanelWidth === 0) return;

    const svg = d3.select(graphSvgRef.current);
    svg.selectAll('*').remove();

    const positions = positionsRef.current;
    if (positions.size !== category.objects.length) return;

    // Defs: arrowhead marker
    const defs = svg.append('defs');
    defs
      .append('marker')
      .attr('id', `cat-arrow-${instanceId}`)
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', NODE_RADIUS + 10)
      .attr('refY', 0)
      .attr('markerWidth', 8)
      .attr('markerHeight', 8)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-4L10,0L0,4Z')
      .style('fill', 'var(--color-text-secondary)');

    // Loop arrowhead (smaller refX)
    defs
      .append('marker')
      .attr('id', `cat-loop-${instanceId}`)
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 6)
      .attr('refY', 0)
      .attr('markerWidth', 7)
      .attr('markerHeight', 7)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-4L10,0L0,4Z')
      .style('fill', 'var(--color-text-secondary)');

    // Draw morphism edges
    const edgeG = svg.append('g');
    for (const link of links) {
      const sp = positions.get(link.source);
      const tp = positions.get(link.target);
      if (!sp || !tp) continue;

      // Curvature: more curve for parallel edges
      const curvature = link.total > 1 ? 0.8 + link.index * 0.4 : 1.2;

      edgeG
        .append('path')
        .attr('d', arcPath(sp.x, sp.y, tp.x, tp.y, curvature))
        .attr('fill', 'none')
        .attr('marker-end', `url(#cat-arrow-${instanceId})`)
        .style('stroke', 'var(--color-text-secondary)')
        .style('stroke-width', '2')
        .style('stroke-opacity', '0.7');

      // Edge label at midpoint
      const mx = (sp.x + tp.x) / 2;
      const my = (sp.y + tp.y) / 2;
      // Offset label perpendicular to the edge
      const dx = tp.x - sp.x;
      const dy = tp.y - sp.y;
      const len = Math.sqrt(dx * dx + dy * dy) || 1;
      const offset = link.total > 1 ? 14 + link.index * 12 : 14;
      const nx = -dy / len;
      const ny = dx / len;

      edgeG
        .append('text')
        .attr('x', mx + nx * offset)
        .attr('y', my + ny * offset + 4)
        .attr('text-anchor', 'middle')
        .attr('font-size', 12)
        .attr('font-weight', 600)
        .attr('font-style', 'italic')
        .attr('pointer-events', 'none')
        .style('fill', 'var(--color-text)')
        .text(link.label);
    }

    // Draw identity loops if enabled
    if (showIdentities) {
      const loopG = svg.append('g');
      for (const obj of category.objects) {
        const pos = positions.get(obj);
        if (!pos) continue;
        loopG
          .append('path')
          .attr('d', loopPath(pos.x, pos.y))
          .attr('fill', 'none')
          .attr('marker-end', `url(#cat-loop-${instanceId})`)
          .style('stroke', 'var(--color-text-secondary)')
          .style('stroke-width', '1.5')
          .style('stroke-opacity', '0.5')
          .style('stroke-dasharray', '4,3');

        // Loop label
        loopG
          .append('text')
          .attr('x', pos.x)
          .attr('y', pos.y - NODE_RADIUS - LOOP_RADIUS * 2 - 6)
          .attr('text-anchor', 'middle')
          .attr('font-size', 10)
          .attr('pointer-events', 'none')
          .style('fill', 'var(--color-text-secondary)')
          .text(category.identity(obj));
      }
    }

    // Draw nodes
    const nodeG = svg.append('g');
    category.objects.forEach((obj, i) => {
      const pos = positions.get(obj);
      if (!pos) return;

      const color = NODE_COLORS[i % NODE_COLORS.length];

      nodeG
        .append('circle')
        .attr('cx', pos.x)
        .attr('cy', pos.y)
        .attr('r', NODE_RADIUS)
        .style('fill', color)
        .style('stroke', 'var(--color-text)')
        .style('stroke-width', '2')
        .style('cursor', 'default');

      nodeG
        .append('text')
        .attr('x', pos.x)
        .attr('y', pos.y + 5)
        .attr('text-anchor', 'middle')
        .attr('font-size', 14)
        .attr('font-weight', 700)
        .attr('pointer-events', 'none')
        .style('fill', 'white')
        .text(obj);
    });

    // Panel label
    svg
      .append('text')
      .attr('x', graphPanelWidth / 2)
      .attr('y', 18)
      .attr('text-anchor', 'middle')
      .attr('font-size', 12)
      .attr('font-weight', 600)
      .style('fill', 'var(--color-text-secondary)')
      .text('Category diagram');
  }, [
    category, graphPanelWidth, showIdentities, links, tick,
  ]);

  // ─── Composition table rendering ───

  const compTableEl = useMemo(() => {
    if (!showCompTable) return null;
    const { labels, table } = compTable;
    const idLabels = new Set(
      category.morphisms.filter((m) => m.isIdentity).map((m) => m.label),
    );
    const hdrStyle = {
      borderColor: 'var(--color-border)',
      color: 'var(--color-text)',
      backgroundColor: 'var(--color-surface)',
    };
    return (
      <div className="overflow-auto">
        <p className="mb-2 text-center text-xs font-semibold" style={{ color: 'var(--color-text-secondary)' }}>
          Composition table (g ∘ f)
        </p>
        <table className="border-collapse text-xs" style={{ borderColor: 'var(--color-border)' }}>
          <thead>
            <tr>
              <th className="border px-2 py-1 text-center font-semibold" style={{ ...hdrStyle, color: 'var(--color-text-secondary)' }}>g ∖ f</th>
              {labels.map((f) => (
                <th key={f} className="border px-2 py-1 text-center font-semibold" style={hdrStyle}>{f}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {labels.map((g, gi) => (
              <tr key={g}>
                <td className="border px-2 py-1 text-center font-semibold" style={hdrStyle}>{g}</td>
                {table[gi].map((val, fi) => {
                  const isId = val !== null && idLabels.has(val);
                  return (
                    <td key={fi} className="border px-2 py-1 text-center" style={{
                      borderColor: 'var(--color-border)',
                      color: val ? 'var(--color-text)' : 'var(--color-text-secondary)',
                      backgroundColor: isId ? 'rgba(59, 130, 246, 0.08)' : 'var(--color-bg)',
                    }}>
                      {val ?? '—'}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }, [showCompTable, compTable, category]);

  // ─── Axiom checker ───

  const axiomPanel = useMemo(() => {
    const badge = (ok: boolean) => ({
      backgroundColor: ok ? '#dcfce7' : '#fef2f2',
      color: ok ? '#166534' : '#991b1b',
    });
    return (
      <div className="mt-4 flex flex-wrap gap-x-8 gap-y-2 rounded-lg border px-4 py-3 text-sm"
        style={{ borderColor: 'var(--color-border)', backgroundColor: 'var(--color-surface)' }}>
        <div className="flex items-center gap-2">
          <span className="inline-flex h-5 w-5 items-center justify-center rounded-full text-xs font-bold" style={badge(assocResult.valid)}>
            {assocResult.valid ? '✓' : '✗'}
          </span>
          <span style={{ color: 'var(--color-text)' }}>
            <strong>Associativity:</strong>{' '}
            {assocResult.valid ? 'All composable triples' : `Violation: (${assocResult.violations[0].join(', ')})`}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="inline-flex h-5 w-5 items-center justify-center rounded-full text-xs font-bold" style={badge(identResult.valid)}>
            {identResult.valid ? '✓' : '✗'}
          </span>
          <span style={{ color: 'var(--color-text)' }}>
            <strong>Identity law:</strong>{' '}
            {identResult.valid ? 'All objects' : `Missing for: ${identResult.violations.join(', ')}`}
          </span>
        </div>
      </div>
    );
  }, [assocResult, identResult]);

  // ─── Render ───

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
        {/* Preset dropdown */}
        <label
          className="flex items-center gap-1.5 text-sm"
          style={{ color: 'var(--color-text)' }}
        >
          <span className="font-medium">Category:</span>
          <select
            className="rounded border px-2 py-1 text-sm"
            style={{
              borderColor: 'var(--color-border)',
              backgroundColor: 'var(--color-bg)',
              color: 'var(--color-text)',
            }}
            value={presetIndex}
            onChange={(e) => handlePresetChange(Number(e.target.value))}
          >
            {presets.map((p, i) => (
              <option key={p.name} value={i}>
                {p.name}
              </option>
            ))}
          </select>
        </label>

        {/* Show identities checkbox */}
        <label
          className="flex items-center gap-1.5 text-sm"
          style={{ color: 'var(--color-text)' }}
        >
          <input
            type="checkbox"
            checked={showIdentities}
            onChange={(e) => setShowIdentities(e.target.checked)}
            className="accent-blue-500"
          />
          <span className="font-medium">Identities</span>
        </label>

        {/* Show composition table checkbox */}
        <label
          className="flex items-center gap-1.5 text-sm"
          style={{ color: 'var(--color-text)' }}
        >
          <input
            type="checkbox"
            checked={showCompTable}
            onChange={(e) => setShowCompTable(e.target.checked)}
            className="accent-blue-500"
          />
          <span className="font-medium">Composition table</span>
        </label>
      </div>

      {/* Main panels */}
      <div
        className={isNarrow ? 'flex flex-col gap-4' : 'flex gap-4'}
        style={{ alignItems: 'flex-start' }}
      >
        {/* Left: D3 force graph */}
        <div style={{ flexShrink: 0 }}>
          <svg role="img" aria-label="Category explorer visualization"
            ref={graphSvgRef}
            width={graphPanelWidth}
            height={GRAPH_PANEL_HEIGHT}
            style={{
              border: '1px solid var(--color-border)',
              borderRadius: 8,
              backgroundColor: 'var(--color-bg)',
            }}
          />
        </div>

        {/* Right: composition table */}
        {showCompTable && !isNarrow && (
          <div style={{ flexShrink: 1, minWidth: 0, overflow: 'auto' }}>
            {compTableEl}
          </div>
        )}
      </div>

      {/* Composition table below on narrow screens */}
      {showCompTable && isNarrow && (
        <div className="mt-4">{compTableEl}</div>
      )}

      {/* Axiom checker */}
      {axiomPanel}

      {/* Legend */}
      <div
        className="mt-3 flex flex-wrap gap-4 text-xs"
        style={{ color: 'var(--color-text-secondary)' }}
      >
        <span>Objects = colored circles</span>
        <span>Morphisms = directed arrows</span>
        <span>Dashed loops = identity morphisms</span>
        <span>Table cell = g ∘ f (or — if not composable)</span>
      </div>
    </div>
  );
}
