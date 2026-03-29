import { useState, useMemo, useEffect, useRef, useId } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  type Category,
  type Adjunction,
  getAdjunctionPresets,
  checkTriangleIdentities,
  homSetBijection,
} from './shared/categoryTheory';

// ─── Constants ───

const SM_BREAKPOINT = 640;
const SVG_H = 360;
const NODE_R = 20;
const STEP_MS = 500;
const MATH_FONT = 'KaTeX_Math, Georgia, serif';

const COLORS = {
  functorF: '#3b82f6',    // blue — left adjoint
  functorG: '#8b5cf6',    // purple — right adjoint
  unit: '#f59e0b',        // amber — unit η
  counit: '#ef4444',      // red — counit ε
  valid: '#22c55e',       // green
  invalid: '#ef4444',
  bg: '#f8fafc',
  text: '#1e293b',
  muted: '#94a3b8',
  node: '#e2e8f0',
  bijLine: '#06b6d4',     // cyan — bijection lines
};

// ─── Types ───

interface PresetData {
  adj: Adjunction;
  sourceCategory: Category;
  targetCategory: Category;
  description: string;
}

type AnimPhase = 'idle' | 'zig1' | 'zig2' | 'zag1' | 'zag2' | 'done';

// ─── Component ───

export default function AdjunctionExplorer() {
  const presets = useMemo(() => getAdjunctionPresets(), []);
  const [presetIdx, setPresetIdx] = useState(0);
  const [selectedA, setSelectedA] = useState<string | null>(null);
  const [selectedB, setSelectedB] = useState<string | null>(null);
  const [animPhase, setAnimPhase] = useState<AnimPhase>('idle');

  const { ref: containerRef, width: cw } = useResizeObserver<HTMLDivElement>();
  const svgRef = useRef<SVGSVGElement>(null);
  const timersRef = useRef<ReturnType<typeof setTimeout>[]>([]);
  const uid = useId().replace(/:/g, '');
  const isNarrow = cw < SM_BREAKPOINT;

  // Build preset data
  const data = useMemo<PresetData>(() => {
    const p = presets[presetIdx].build();
    return p;
  }, [presets, presetIdx]);

  // Set default selected objects when preset changes
  useEffect(() => {
    const srcObjs = data.sourceCategory.objects;
    const tgtObjs = data.targetCategory.objects;
    setSelectedA(srcObjs[0] ?? null);
    setSelectedB(tgtObjs[0] ?? null);
    setAnimPhase('idle');
  }, [data]);

  // Cleanup timers
  useEffect(() => {
    return () => {
      timersRef.current.forEach(clearTimeout);
    };
  }, []);

  // Compute derived data
  const triangleResult = useMemo(() => {
    return checkTriangleIdentities(data.adj, data.sourceCategory, data.targetCategory);
  }, [data]);

  const homBij = useMemo(() => {
    if (!selectedA || !selectedB) return null;
    return homSetBijection(data.adj, selectedA, selectedB, data.sourceCategory, data.targetCategory);
  }, [data, selectedA, selectedB]);

  // Triangle animation
  const startTriangleAnim = () => {
    timersRef.current.forEach(clearTimeout);
    timersRef.current = [];
    setAnimPhase('zig1');
    const t1 = setTimeout(() => setAnimPhase('zig2'), STEP_MS);
    const t2 = setTimeout(() => setAnimPhase('zag1'), STEP_MS * 2);
    const t3 = setTimeout(() => setAnimPhase('zag2'), STEP_MS * 3);
    const t4 = setTimeout(() => setAnimPhase('done'), STEP_MS * 4);
    timersRef.current = [t1, t2, t3, t4];
  };

  // ─── D3 Rendering ───
  useEffect(() => {
    if (!svgRef.current || cw < 100) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const w = cw;
    const h = SVG_H;
    const margin = { top: 30, right: 20, bottom: 40, left: 20 };
    const innerW = w - margin.left - margin.right;

    // Defs: arrow markers
    const defs = svg.append('defs');

    const addMarker = (id: string, color: string) => {
      defs.append('marker')
        .attr('id', `${uid}-${id}`)
        .attr('viewBox', '0 0 10 10')
        .attr('refX', 8).attr('refY', 5)
        .attr('markerWidth', 6).attr('markerHeight', 6)
        .attr('orient', 'auto-start-reverse')
        .append('path')
        .attr('d', 'M 0 0 L 10 5 L 0 10 z')
        .attr('fill', color);
    };

    addMarker('arrow-f', COLORS.functorF);
    addMarker('arrow-g', COLORS.functorG);
    addMarker('arrow-bij', COLORS.bijLine);
    addMarker('arrow-unit', COLORS.unit);
    addMarker('arrow-counit', COLORS.counit);
    addMarker('arrow-text', COLORS.text);

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    // Layout: three columns — source | middle | target
    const colW = innerW / 3;
    const srcX = colW * 0.5;
    const tgtX = colW * 2.5;
    const midX = colW * 1.5;

    // Draw category boxes
    const drawCatBox = (x: number, label: string, color: string) => {
      const boxW = colW * 0.85;
      const boxH = h - margin.top - margin.bottom - 20;
      g.append('rect')
        .attr('x', x - boxW / 2)
        .attr('y', 0)
        .attr('width', boxW)
        .attr('height', boxH)
        .attr('rx', 8)
        .attr('fill', 'none')
        .attr('stroke', COLORS.muted)
        .attr('stroke-width', 1.5)
        .attr('stroke-dasharray', '4,3');
      g.append('text')
        .attr('x', x)
        .attr('y', -8)
        .attr('text-anchor', 'middle')
        .style('font-family', MATH_FONT)
        .style('font-size', '14px')
        .style('font-weight', 'bold')
        .style('fill', color)
        .text(label);
    };

    const srcLabel = data.adj.leftAdjoint.source === data.sourceCategory ? '𝒞' : 'Source';
    const tgtLabel = data.adj.leftAdjoint.target === data.targetCategory ? '𝒟' : 'Target';
    drawCatBox(srcX, srcLabel, COLORS.functorF);
    drawCatBox(tgtX, tgtLabel, COLORS.functorG);

    // Position objects within boxes
    const boxH = h - margin.top - margin.bottom - 20;
    const positionObjects = (objects: string[], centerX: number) => {
      const n = objects.length;
      return objects.map((obj, i) => ({
        label: obj,
        x: centerX,
        y: (boxH / (n + 1)) * (i + 1),
      }));
    };

    const srcNodes = positionObjects(data.sourceCategory.objects, srcX);
    const tgtNodes = positionObjects(data.targetCategory.objects, tgtX);

    // Draw nodes with click-to-select
    const drawNodes = (
      nodes: { label: string; x: number; y: number }[],
      highlight: string | null | undefined,
      onClick: (label: string) => void,
    ) => {
      nodes.forEach((n) => {
        const isHighlighted = n.label === highlight;
        g.append('circle')
          .attr('cx', n.x).attr('cy', n.y).attr('r', NODE_R)
          .attr('fill', isHighlighted ? COLORS.unit : COLORS.node)
          .attr('stroke', isHighlighted ? COLORS.unit : COLORS.text)
          .attr('stroke-width', isHighlighted ? 2.5 : 1.5)
          .style('cursor', 'pointer')
          .on('click', () => onClick(n.label));
        g.append('text')
          .attr('x', n.x).attr('y', n.y + 4)
          .attr('text-anchor', 'middle')
          .style('font-family', MATH_FONT)
          .style('font-size', '13px')
          .style('fill', isHighlighted ? '#fff' : COLORS.text)
          .style('pointer-events', 'none')
          .text(n.label);
      });
    };

    drawNodes(srcNodes, selectedA, (label) => setSelectedA(label));
    drawNodes(tgtNodes, selectedB, (label) => setSelectedB(label));

    // Draw functor arrows (F: left→right, G: right→left)
    const F = data.adj.leftAdjoint;
    const G = data.adj.rightAdjoint;

    // F arrows — highlight during zig phases (zig1: Fη, zig2: εF)
    srcNodes.forEach((sn) => {
      const fImage = F.onObjects.get(sn.label);
      if (!fImage) return;
      const tn = tgtNodes.find((t) => t.label === fImage);
      if (!tn) return;
      const isZigActive = animPhase === 'zig1' || animPhase === 'zig2';
      g.append('path')
        .attr('d', `M ${sn.x + NODE_R + 2} ${sn.y - 4} Q ${midX} ${sn.y - 30} ${tn.x - NODE_R - 2} ${tn.y - 4}`)
        .attr('fill', 'none')
        .attr('stroke', animPhase === 'zig1' ? COLORS.unit : COLORS.functorF)
        .attr('stroke-width', isZigActive ? 3 : 1.8)
        .attr('marker-end', `url(#${uid}-arrow-f)`)
        .attr('opacity', isZigActive ? 1 : 0.7);
    });

    // G arrows — highlight during zag phases (zag1: ηG, zag2: Gε)
    tgtNodes.forEach((tn) => {
      const gImage = G.onObjects.get(tn.label);
      if (!gImage) return;
      const sn = srcNodes.find((s) => s.label === gImage);
      if (!sn) return;
      const isZagActive = animPhase === 'zag1' || animPhase === 'zag2';
      g.append('path')
        .attr('d', `M ${tn.x - NODE_R - 2} ${tn.y + 4} Q ${midX} ${tn.y + 30} ${sn.x + NODE_R + 2} ${sn.y + 4}`)
        .attr('fill', 'none')
        .attr('stroke', animPhase === 'zag2' ? COLORS.counit : COLORS.functorG)
        .attr('stroke-width', isZagActive ? 3 : 1.8)
        .attr('marker-end', `url(#${uid}-arrow-g)`)
        .attr('opacity', isZagActive ? 1 : 0.7);
    });

    // ⊣ symbol in the middle (green when animation completes)
    g.append('text')
      .attr('x', midX).attr('y', boxH / 2)
      .attr('text-anchor', 'middle')
      .style('font-family', MATH_FONT)
      .style('font-size', '24px')
      .style('fill', animPhase === 'done' ? COLORS.valid : COLORS.text)
      .text('⊣');

    // F and G labels
    g.append('text')
      .attr('x', midX).attr('y', boxH / 2 - 24)
      .attr('text-anchor', 'middle')
      .style('font-family', MATH_FONT)
      .style('font-size', '14px')
      .style('font-weight', 'bold')
      .style('fill', COLORS.functorF)
      .text('F');
    g.append('text')
      .attr('x', midX).attr('y', boxH / 2 + 32)
      .attr('text-anchor', 'middle')
      .style('font-family', MATH_FONT)
      .style('font-size', '14px')
      .style('font-weight', 'bold')
      .style('fill', COLORS.functorG)
      .text('G');

    // Hom-set bijection display (bottom of middle column)
    if (homBij && selectedA && selectedB) {
      const bijY = boxH + 10;
      const bijLeftX = midX - 60;
      const bijRightX = midX + 60;

      g.append('text')
        .attr('x', bijLeftX).attr('y', bijY)
        .attr('text-anchor', 'middle')
        .style('font-family', MATH_FONT)
        .style('font-size', '11px')
        .style('fill', COLORS.text)
        .text(`Hom(F(${selectedA}), ${selectedB})`);

      g.append('text')
        .attr('x', bijRightX).attr('y', bijY)
        .attr('text-anchor', 'middle')
        .style('font-family', MATH_FONT)
        .style('font-size', '11px')
        .style('fill', COLORS.text)
        .text(`Hom(${selectedA}, G(${selectedB}))`);

      g.append('text')
        .attr('x', midX).attr('y', bijY)
        .attr('text-anchor', 'middle')
        .style('font-family', MATH_FONT)
        .style('font-size', '14px')
        .style('fill', COLORS.bijLine)
        .text('≅');

      // List elements and bijection lines
      homBij.leftHomSet.forEach((l, i) => {
        const y = bijY + 16 + i * 14;
        g.append('text')
          .attr('x', bijLeftX).attr('y', y)
          .attr('text-anchor', 'middle')
          .style('font-family', MATH_FONT)
          .style('font-size', '10px')
          .style('fill', COLORS.muted)
          .text(l);

        const r = homBij.bijection.get(l);
        if (r) {
          const ri = homBij.rightHomSet.indexOf(r);
          const ry = bijY + 16 + ri * 14;
          g.append('line')
            .attr('x1', bijLeftX + 30).attr('y1', y - 3)
            .attr('x2', bijRightX - 30).attr('y2', ry - 3)
            .attr('stroke', COLORS.bijLine)
            .attr('stroke-width', 1)
            .attr('opacity', 0.5);
        }
      });

      homBij.rightHomSet.forEach((r, i) => {
        const y = bijY + 16 + i * 14;
        g.append('text')
          .attr('x', bijRightX).attr('y', y)
          .attr('text-anchor', 'middle')
          .style('font-family', MATH_FONT)
          .style('font-size', '10px')
          .style('fill', COLORS.muted)
          .text(r);
      });
    }
  }, [cw, data, selectedA, selectedB, uid, homBij, animPhase]);

  const animating = animPhase !== 'idle' && animPhase !== 'done';

  return (
    <div ref={containerRef} style={{ margin: '1.5rem 0' }}>
      {/* Controls */}
      <div style={{
        display: 'flex',
        flexWrap: 'wrap',
        gap: '0.75rem',
        alignItems: 'center',
        marginBottom: '0.75rem',
        padding: '0.5rem 0.75rem',
        background: COLORS.bg,
        borderRadius: '8px',
        fontSize: '13px',
      }}>
        <label style={{ fontWeight: 600, color: COLORS.text }}>Adjunction</label>
        <select
          value={presetIdx}
          onChange={(e) => setPresetIdx(+e.target.value)}
          style={{ padding: '4px 8px', borderRadius: '4px', border: `1px solid ${COLORS.muted}` }}
        >
          {presets.map((p, i) => (
            <option key={p.name} value={i}>{p.name}</option>
          ))}
        </select>

        {data.sourceCategory.objects.length > 0 && (
          <>
            <label style={{ fontWeight: 600, color: COLORS.text, marginLeft: '0.5rem' }}>A</label>
            <select
              value={selectedA ?? ''}
              onChange={(e) => setSelectedA(e.target.value)}
              style={{ padding: '4px 8px', borderRadius: '4px', border: `1px solid ${COLORS.muted}` }}
            >
              {data.sourceCategory.objects.map((o) => (
                <option key={o} value={o}>{o}</option>
              ))}
            </select>

            <label style={{ fontWeight: 600, color: COLORS.text, marginLeft: '0.5rem' }}>B</label>
            <select
              value={selectedB ?? ''}
              onChange={(e) => setSelectedB(e.target.value)}
              style={{ padding: '4px 8px', borderRadius: '4px', border: `1px solid ${COLORS.muted}` }}
            >
              {data.targetCategory.objects.map((o) => (
                <option key={o} value={o}>{o}</option>
              ))}
            </select>
          </>
        )}

        <button
          onClick={startTriangleAnim}
          disabled={animating}
          style={{
            marginLeft: 'auto',
            padding: '4px 12px',
            borderRadius: '4px',
            border: `1px solid ${COLORS.unit}`,
            background: animating ? COLORS.muted : 'white',
            color: animating ? 'white' : COLORS.unit,
            fontWeight: 600,
            cursor: animating ? 'not-allowed' : 'pointer',
            fontSize: '12px',
          }}
        >
          {animating ? 'Animating…' : '▶ Triangle Identities'}
        </button>
      </div>

      {/* SVG */}
      <svg
        ref={svgRef}
        width={cw}
        height={SVG_H}
        style={{ display: 'block', background: 'white', borderRadius: '8px', border: `1px solid ${COLORS.node}` }}
      />

      {/* Info panel */}
      <div style={{
        display: 'flex',
        flexDirection: isNarrow ? 'column' : 'row',
        gap: '1rem',
        marginTop: '0.75rem',
        fontSize: '13px',
      }}>
        <div style={{ flex: 1, padding: '0.75rem', background: COLORS.bg, borderRadius: '8px' }}>
          <strong style={{ color: COLORS.text }}>Description</strong>
          <p style={{ margin: '0.5rem 0 0', color: COLORS.muted, lineHeight: 1.5 }}>{data.description}</p>
        </div>
        <div style={{ flex: 1, padding: '0.75rem', background: COLORS.bg, borderRadius: '8px' }}>
          <strong style={{ color: COLORS.text }}>Triangle Identities</strong>
          <p style={{ margin: '0.5rem 0 0', color: triangleResult.firstTriangle ? COLORS.valid : COLORS.invalid }}>
            ε<sub>F</sub> ∘ Fη = id<sub>F</sub> : {triangleResult.firstTriangle ? '✓' : '✗'}
          </p>
          <p style={{ margin: '0.25rem 0 0', color: triangleResult.secondTriangle ? COLORS.valid : COLORS.invalid }}>
            Gε ∘ η<sub>G</sub> = id<sub>G</sub> : {triangleResult.secondTriangle ? '✓' : '✗'}
          </p>
          {triangleResult.violations.length > 0 && (
            <p style={{ margin: '0.25rem 0 0', color: COLORS.invalid, fontSize: '11px' }}>
              {triangleResult.violations[0]}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
