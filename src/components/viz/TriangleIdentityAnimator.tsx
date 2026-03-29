import { useState, useMemo, useEffect, useRef, useId } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { getAdjunctionPresets } from './shared/categoryTheory';

// ─── Constants ───

const SM_BREAKPOINT = 640;
const SVG_H = 300;
const STEP_MS = 600;
const MATH_FONT = 'KaTeX_Math, Georgia, serif';

const COLORS = {
  unit: '#f59e0b',      // amber — F(η) and η
  counit: '#ef4444',    // red — ε and G(ε)
  identity: '#94a3b8',  // gray — dashed id
  valid: '#22c55e',
  bg: '#f8fafc',
  text: '#1e293b',
  muted: '#94a3b8',
  node: '#e2e8f0',
  nodeActive: '#3b82f6',
};

type Phase = 'idle' | 'step1' | 'step2' | 'done';

// ─── Component ───

export default function TriangleIdentityAnimator() {
  const presets = useMemo(() => getAdjunctionPresets(), []);
  const [presetIdx, setPresetIdx] = useState(0);
  const [phase, setPhase] = useState<Phase>('idle');
  const [speed, setSpeed] = useState(1);

  const { ref: containerRef, width: cw } = useResizeObserver<HTMLDivElement>();
  const svgRef = useRef<SVGSVGElement>(null);
  const timersRef = useRef<ReturnType<typeof setTimeout>[]>([]);
  const uid = useId().replace(/:/g, '');
  const isNarrow = cw < SM_BREAKPOINT;

  const data = useMemo(() => presets[presetIdx].build(), [presets, presetIdx]);

  // Cleanup timers
  useEffect(() => {
    return () => { timersRef.current.forEach(clearTimeout); };
  }, []);

  // Reset on preset change
  useEffect(() => { setPhase('idle'); }, [presetIdx]);

  const animate = () => {
    timersRef.current.forEach(clearTimeout);
    timersRef.current = [];
    const ms = STEP_MS / speed;
    setPhase('step1');
    const t1 = setTimeout(() => setPhase('step2'), ms);
    const t2 = setTimeout(() => setPhase('done'), ms * 2);
    timersRef.current = [t1, t2];
  };

  // ─── D3 Rendering ───
  useEffect(() => {
    if (!svgRef.current || cw < 100) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const w = cw;
    const h = SVG_H;
    const margin = { top: 20, bottom: 20 };

    // Defs
    const defs = svg.append('defs');
    const addMarker = (id: string, color: string) => {
      defs.append('marker')
        .attr('id', `${uid}-${id}`)
        .attr('viewBox', '0 0 10 10')
        .attr('refX', 8).attr('refY', 5)
        .attr('markerWidth', 6).attr('markerHeight', 6)
        .attr('orient', 'auto-start-reverse')
        .append('path').attr('d', 'M 0 0 L 10 5 L 0 10 z').attr('fill', color);
    };
    addMarker('unit', COLORS.unit);
    addMarker('counit', COLORS.counit);
    addMarker('id', COLORS.identity);

    const g = svg.append('g').attr('transform', `translate(0,${margin.top})`);
    const innerH = h - margin.top - margin.bottom;

    // Pick representative objects
    const A = data.sourceCategory.objects[0] ?? 'A';
    const B = data.targetCategory.objects[0] ?? 'B';
    const F = data.adj.leftAdjoint;
    const G = data.adj.rightAdjoint;
    const FA = F.onObjects.get(A) ?? `F(${A})`;

    // Two triangles side by side
    const triW = isNarrow ? w * 0.45 : w * 0.4;
    const gap = isNarrow ? w * 0.1 : w * 0.2;

    const drawTriangle = (
      cx: number,
      topLabel: string,
      midLabel: string,
      botLabel: string,
      sideUpLabel: string,
      sideDownLabel: string,
      diagLabel: string,
      leftTriangle: boolean,
    ) => {
      const topY = 20;
      const nodeR = 18;

      // Positions: triangle with top-left, mid-top-right, bottom-left
      // For left triangle: F(A) top-left, FGF(A) top-right, F(A) bottom
      // Actually: three nodes in a triangle shape
      const leftX = cx - triW * 0.35;
      const rightX = cx + triW * 0.35;
      const bottomX = cx;
      const bottomY = innerH * 0.75;

      // Node positions
      const n1 = { x: leftX, y: topY, label: topLabel };        // start
      const n2 = { x: rightX, y: topY, label: midLabel };       // middle (top-right)
      const n3 = { x: bottomX, y: bottomY, label: botLabel };   // end (bottom)

      // Draw nodes
      [n1, n2, n3].forEach((n) => {
        const isActive = (phase === 'step1' && n === n1) ||
          (phase === 'step2' && n === n2) ||
          (phase === 'done' && n === n3);
        g.append('circle')
          .attr('cx', n.x).attr('cy', n.y).attr('r', nodeR)
          .attr('fill', isActive ? COLORS.nodeActive : COLORS.node)
          .attr('stroke', isActive ? COLORS.nodeActive : COLORS.text)
          .attr('stroke-width', isActive ? 2.5 : 1.5);
        g.append('text')
          .attr('x', n.x).attr('y', n.y + 4)
          .attr('text-anchor', 'middle')
          .style('font-family', MATH_FONT)
          .style('font-size', '11px')
          .style('fill', isActive ? '#fff' : COLORS.text)
          .text(n.label);
      });

      // Side up: n1 -> n2 (unit/F(η) or η)
      const upOpacity = phase === 'step1' || phase === 'step2' || phase === 'done' ? 1 : 0.3;
      const upColor = phase === 'step1' ? COLORS.unit : (phase !== 'idle' ? COLORS.unit : COLORS.muted);
      g.append('line')
        .attr('x1', n1.x + nodeR).attr('y1', n1.y)
        .attr('x2', n2.x - nodeR).attr('y2', n2.y)
        .attr('stroke', upColor)
        .attr('stroke-width', phase === 'step1' ? 3 : 1.8)
        .attr('marker-end', `url(#${uid}-unit)`)
        .attr('opacity', upOpacity);
      g.append('text')
        .attr('x', (n1.x + n2.x) / 2).attr('y', n1.y - 10)
        .attr('text-anchor', 'middle')
        .style('font-family', MATH_FONT)
        .style('font-size', '12px')
        .style('fill', COLORS.unit)
        .text(sideUpLabel);

      // Side down: n2 -> n3 (counit/ε)
      const downOpacity = phase === 'step2' || phase === 'done' ? 1 : 0.3;
      const downColor = phase === 'step2' ? COLORS.counit : (phase === 'done' ? COLORS.counit : COLORS.muted);
      g.append('line')
        .attr('x1', n2.x - nodeR * 0.5).attr('y1', n2.y + nodeR)
        .attr('x2', n3.x + nodeR * 0.5).attr('y2', n3.y - nodeR)
        .attr('stroke', downColor)
        .attr('stroke-width', phase === 'step2' ? 3 : 1.8)
        .attr('marker-end', `url(#${uid}-counit)`)
        .attr('opacity', downOpacity);
      g.append('text')
        .attr('x', (n2.x + n3.x) / 2 + 12).attr('y', (n2.y + n3.y) / 2)
        .attr('text-anchor', 'start')
        .style('font-family', MATH_FONT)
        .style('font-size', '12px')
        .style('fill', COLORS.counit)
        .text(sideDownLabel);

      // Diagonal (identity): n1 -> n3, dashed
      g.append('line')
        .attr('x1', n1.x - nodeR * 0.3).attr('y1', n1.y + nodeR)
        .attr('x2', n3.x - nodeR * 0.3).attr('y2', n3.y - nodeR)
        .attr('stroke', phase === 'done' ? COLORS.valid : COLORS.identity)
        .attr('stroke-width', phase === 'done' ? 2.5 : 1.5)
        .attr('stroke-dasharray', phase === 'done' ? 'none' : '5,4')
        .attr('marker-end', `url(#${uid}-id)`)
        .attr('opacity', phase === 'done' ? 1 : 0.5);
      g.append('text')
        .attr('x', n1.x - 18).attr('y', (n1.y + n3.y) / 2)
        .attr('text-anchor', 'end')
        .style('font-family', MATH_FONT)
        .style('font-size', '12px')
        .style('fill', phase === 'done' ? COLORS.valid : COLORS.identity)
        .text(diagLabel);

      // Checkmark when done
      if (phase === 'done') {
        g.append('text')
          .attr('x', cx).attr('y', bottomY + 30)
          .attr('text-anchor', 'middle')
          .style('font-size', '16px')
          .style('fill', COLORS.valid)
          .text('= identity ✓');
      }

      // Title
      g.append('text')
        .attr('x', cx).attr('y', innerH + 5)
        .attr('text-anchor', 'middle')
        .style('font-family', MATH_FONT)
        .style('font-size', '12px')
        .style('font-weight', 'bold')
        .style('fill', COLORS.text)
        .text(leftTriangle ? 'First Triangle' : 'Second Triangle');
    };

    const leftCx = gap / 2 + triW / 2;
    const rightCx = w - gap / 2 - triW / 2;

    drawTriangle(leftCx, `F(${A})`, `FGF(${A})`, `F(${A})`, `F(η_${A})`, `ε_{F(${A})}`, `id`, true);
    drawTriangle(rightCx, `G(${B})`, `GFG(${B})`, `G(${B})`, `η_{G(${B})}`, `G(ε_${B})`, `id`, false);
  }, [cw, data, phase, uid, isNarrow]);

  const animating = phase === 'step1' || phase === 'step2';

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

        <label style={{ fontWeight: 600, color: COLORS.text, marginLeft: '0.5rem' }}>Speed</label>
        <input
          type="range" min="0.5" max="3" step="0.5" value={speed}
          onChange={(e) => setSpeed(+e.target.value)}
          style={{ width: '80px' }}
        />
        <span style={{ color: COLORS.muted, fontSize: '11px' }}>{speed}×</span>

        <button
          onClick={animate}
          disabled={animating}
          style={{
            marginLeft: 'auto', padding: '4px 12px', borderRadius: '4px',
            border: `1px solid ${COLORS.unit}`,
            background: animating ? COLORS.muted : 'white',
            color: animating ? 'white' : COLORS.unit,
            fontWeight: 600, cursor: animating ? 'not-allowed' : 'pointer', fontSize: '12px',
          }}
        >
          {animating ? 'Animating…' : '▶ Animate'}
        </button>
      </div>

      {/* SVG */}
      <svg
        ref={svgRef}
        width={cw}
        height={SVG_H}
        style={{ display: 'block', background: 'white', borderRadius: '8px', border: `1px solid ${COLORS.node}` }}
      />

      {/* Equations */}
      <div style={{
        display: 'flex', flexDirection: isNarrow ? 'column' : 'row', gap: '0.5rem',
        marginTop: '0.5rem', fontSize: '12px',
      }}>
        <div style={{
          flex: 1, padding: '0.5rem 0.75rem', background: COLORS.bg, borderRadius: '6px',
          fontFamily: MATH_FONT, color: phase === 'done' ? COLORS.valid : COLORS.text,
        }}>
          ε<sub>F(A)</sub> ∘ F(η<sub>A</sub>) = id<sub>F(A)</sub>
        </div>
        <div style={{
          flex: 1, padding: '0.5rem 0.75rem', background: COLORS.bg, borderRadius: '6px',
          fontFamily: MATH_FONT, color: phase === 'done' ? COLORS.valid : COLORS.text,
        }}>
          G(ε<sub>B</sub>) ∘ η<sub>G(B)</sub> = id<sub>G(B)</sub>
        </div>
      </div>
    </div>
  );
}
