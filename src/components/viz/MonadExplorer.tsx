import { useState, useMemo, useEffect, useRef, useId } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  getMonadPresets,
  checkMonadLaws,
} from './shared/categoryTheory';

// ─── Constants ───

const SM_BREAKPOINT = 640;
const SVG_H = 320;
const MATH_FONT = 'KaTeX_Math, Georgia, serif';

const COLORS = {
  functor: '#3b82f6',     // blue — endofunctor T
  unit: '#f59e0b',        // amber — unit η
  mult: '#8b5cf6',        // purple — multiplication μ
  valid: '#22c55e',       // green — law holds
  invalid: '#ef4444',     // red — law fails
  bg: '#f8fafc',
  text: '#1e293b',
  muted: '#94a3b8',
  node: '#e2e8f0',
  active: '#3b82f6',
  highlight: '#dbeafe',
};

// ─── Monad display data ───

interface MonadDisplay {
  name: string;
  tDesc: string;
  etaDesc: string;
  muDesc: string;
  effect: string;
  concreteExamples: {
    input: string;
    etaResult: string;
    tOfT: string;
    muResult: string;
  }[];
  adjunction: string;
}

const MONAD_DISPLAYS: MonadDisplay[] = [
  {
    name: 'Maybe',
    tDesc: 'T(X) = X ∪ {⊥}',
    etaDesc: 'η(x) = Just(x)',
    muDesc: 'μ(Just(Just(x))) = Just(x),  μ(Just(Nothing)) = Nothing',
    effect: 'Partiality — computations that may fail',
    concreteExamples: [
      { input: '3', etaResult: 'Just(3)', tOfT: 'Just(Just(3))', muResult: 'Just(3)' },
      { input: '—', etaResult: 'Nothing', tOfT: 'Just(Nothing)', muResult: 'Nothing' },
    ],
    adjunction: 'Free ⊣ Forgetful : Set* ↔ Set',
  },
  {
    name: 'List',
    tDesc: 'T(X) = X* (free monoid)',
    etaDesc: 'η(x) = [x]',
    muDesc: 'μ([[a,b],[c]]) = [a,b,c]  (concat)',
    effect: 'Nondeterminism — computations with multiple outcomes',
    concreteExamples: [
      { input: 'a', etaResult: '[a]', tOfT: '[[a]]', muResult: '[a]' },
      { input: '[a,b]', etaResult: '[[a,b]]', tOfT: '[[[a,b]]]', muResult: '[[a,b]]' },
    ],
    adjunction: 'Free ⊣ Forgetful : Mon ↔ Set',
  },
  {
    name: 'Giry',
    tDesc: 'T(X) = Dist(X)',
    etaDesc: 'η(x) = δₓ  (Dirac delta)',
    muDesc: 'μ(Φ) = ∫ p dΦ(p)  (integrate over distributions)',
    effect: 'Probability — computations with stochastic outcomes',
    concreteExamples: [
      { input: 's₁', etaResult: 'δ_{s₁} = [1, 0, 0]', tOfT: 'Dist(Dist({s₁,s₂,s₃}))', muResult: 'E[p] ∈ Dist({s₁,s₂,s₃})' },
    ],
    adjunction: 'Meas → Meas  (endofunctor on measurable spaces)',
  },
  {
    name: 'Reader',
    tDesc: 'T(X) = (E → X)',
    etaDesc: 'η(x) = λe. x  (constant function)',
    muDesc: 'μ(f) = λe. f(e)(e)  (diagonal)',
    effect: 'Environment — computations reading shared config',
    concreteExamples: [
      { input: '42', etaResult: 'λe. 42', tOfT: 'λe₁. (λe₂. 42)', muResult: 'λe. 42' },
    ],
    adjunction: '(— × E) ⊣ (E → —)',
  },
];

// ─── Component ───

export default function MonadExplorer() {
  const presets = useMemo(() => getMonadPresets(), []);
  const [presetIdx, setPresetIdx] = useState(0);
  const [showLaws, setShowLaws] = useState(false);
  const [showAdj, setShowAdj] = useState(false);
  const [exampleIdx, setExampleIdx] = useState(0);

  const { ref: containerRef, width: cw } = useResizeObserver<HTMLDivElement>();
  const svgRef = useRef<SVGSVGElement>(null);
  const uid = useId().replace(/:/g, '');
  const isNarrow = cw < SM_BREAKPOINT;

  const display = MONAD_DISPLAYS[presetIdx];

  // Build preset for law checking
  const presetData = useMemo(() => {
    if (presetIdx < presets.length) {
      const { monad, category } = presets[presetIdx].build();
      return { monad, category };
    }
    return null;
  }, [presets, presetIdx]);

  const lawResult = useMemo(() => {
    if (!presetData) return null;
    return checkMonadLaws(presetData.monad, presetData.category);
  }, [presetData]);

  // ─── D3 rendering: abstract diagram ───
  useEffect(() => {
    if (!svgRef.current || cw < 100) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const w = cw;
    const h = SVG_H;
    const mx = w / 2;
    const my = h / 2;

    // Defs for arrow markers
    const defs = svg.append('defs');
    const markerIds = ['arrow-t', 'arrow-eta', 'arrow-mu'];
    const markerColors = [COLORS.functor, COLORS.unit, COLORS.mult];
    markerIds.forEach((id, i) => {
      defs
        .append('marker')
        .attr('id', `${uid}-${id}`)
        .attr('viewBox', '0 0 10 10')
        .attr('refX', 8)
        .attr('refY', 5)
        .attr('markerWidth', 6)
        .attr('markerHeight', 6)
        .attr('orient', 'auto')
        .append('path')
        .attr('d', 'M0,0 L10,5 L0,10 Z')
        .style('fill', markerColors[i]);
    });

    const g = svg.append('g');

    // Draw category C as a rounded rectangle
    const catW = Math.min(180, w * 0.35);
    const catH = 120;
    const catX = mx - catW / 2;
    const catY = my - catH / 2 - 20;

    g.append('rect')
      .attr('x', catX)
      .attr('y', catY)
      .attr('width', catW)
      .attr('height', catH)
      .attr('rx', 12)
      .style('fill', COLORS.bg)
      .style('stroke', COLORS.muted)
      .style('stroke-width', 1.5);

    g.append('text')
      .attr('x', catX + catW / 2)
      .attr('y', catY + 20)
      .attr('text-anchor', 'middle')
      .style('font-family', MATH_FONT)
      .style('font-size', '16px')
      .style('font-style', 'italic')
      .style('fill', COLORS.text)
      .text('C');

    // Endofunctor T as a loop arrow on C
    const loopCx = catX + catW + 30;
    const loopCy = catY + catH / 2;
    const loopR = 35;

    g.append('path')
      .attr('d', `M ${catX + catW} ${catY + catH / 2 - 15}
        C ${loopCx + loopR} ${loopCy - loopR * 1.5},
          ${loopCx + loopR} ${loopCy + loopR * 1.5},
          ${catX + catW} ${catY + catH / 2 + 15}`)
      .style('fill', 'none')
      .style('stroke', COLORS.functor)
      .style('stroke-width', 2.5)
      .attr('marker-end', `url(#${uid}-arrow-t)`);

    g.append('text')
      .attr('x', loopCx + loopR + 8)
      .attr('y', loopCy + 5)
      .style('font-family', MATH_FONT)
      .style('font-size', '18px')
      .style('font-weight', 'bold')
      .style('font-style', 'italic')
      .style('fill', COLORS.functor)
      .text('T');

    // Unit η arrow: small upward arrow inside C
    const etaX = catX + catW * 0.3;
    const etaY1 = catY + catH - 15;
    const etaY2 = catY + catH * 0.45;

    g.append('line')
      .attr('x1', etaX)
      .attr('y1', etaY1)
      .attr('x2', etaX)
      .attr('y2', etaY2)
      .style('stroke', COLORS.unit)
      .style('stroke-width', 2)
      .attr('marker-end', `url(#${uid}-arrow-eta)`);

    g.append('text')
      .attr('x', etaX - 12)
      .attr('y', (etaY1 + etaY2) / 2 + 4)
      .style('font-family', MATH_FONT)
      .style('font-size', '14px')
      .style('font-style', 'italic')
      .style('fill', COLORS.unit)
      .text('η');

    // Multiplication μ arrow: downward arrow inside C
    const muX = catX + catW * 0.7;
    const muY1 = catY + catH * 0.45;
    const muY2 = catY + catH - 15;

    g.append('line')
      .attr('x1', muX)
      .attr('y1', muY1)
      .attr('x2', muX)
      .attr('y2', muY2)
      .style('stroke', COLORS.mult)
      .style('stroke-width', 2)
      .attr('marker-end', `url(#${uid}-arrow-mu)`);

    g.append('text')
      .attr('x', muX + 8)
      .attr('y', (muY1 + muY2) / 2 + 4)
      .style('font-family', MATH_FONT)
      .style('font-size', '14px')
      .style('font-style', 'italic')
      .style('fill', COLORS.mult)
      .text('μ');

    // Labels below
    const labelY = catY + catH + 35;
    g.append('text')
      .attr('x', mx)
      .attr('y', labelY)
      .attr('text-anchor', 'middle')
      .style('font-family', MATH_FONT)
      .style('font-size', '13px')
      .style('fill', COLORS.text)
      .text(`(T, η, μ) — ${display.name} monad`);

    g.append('text')
      .attr('x', mx)
      .attr('y', labelY + 20)
      .attr('text-anchor', 'middle')
      .style('font-size', '12px')
      .style('fill', COLORS.muted)
      .text(display.tDesc);

  }, [cw, uid, presetIdx, display]);

  // Reset example index when monad changes
  useEffect(() => {
    setExampleIdx(0);
  }, [presetIdx]);

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
      {/* ─── Controls ─── */}
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
          Monad:
          <select
            value={presetIdx}
            onChange={(e) => setPresetIdx(Number(e.target.value))}
            style={{
              marginLeft: '6px',
              padding: '4px 8px',
              borderRadius: '4px',
              border: '1px solid #cbd5e1',
              fontSize: '13px',
            }}
          >
            {MONAD_DISPLAYS.map((m, i) => (
              <option key={m.name} value={i}>
                {m.name}
              </option>
            ))}
          </select>
        </label>

        <label
          style={{ fontSize: '13px', color: COLORS.text, display: 'flex', alignItems: 'center', gap: '4px' }}
        >
          <input
            type="checkbox"
            checked={showLaws}
            onChange={(e) => setShowLaws(e.target.checked)}
          />
          Show monad laws
        </label>

        <label
          style={{ fontSize: '13px', color: COLORS.text, display: 'flex', alignItems: 'center', gap: '4px' }}
        >
          <input
            type="checkbox"
            checked={showAdj}
            onChange={(e) => setShowAdj(e.target.checked)}
          />
          Show adjunction
        </label>
      </div>

      {/* ─── Abstract diagram ─── */}
      <svg role="img" aria-label="Monad explorer visualization" ref={svgRef} width={cw} height={SVG_H} />

      {/* ─── Monad laws panel ─── */}
      {showLaws && (
        <div
          style={{
            padding: '12px 16px',
            borderTop: '1px solid #e2e8f0',
            background: '#fefce8',
          }}
        >
          <div style={{ fontSize: '14px', fontWeight: 600, marginBottom: '8px', color: COLORS.text }}>
            Monad Laws
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '16px' }}>
            <LawCard
              name="Associativity"
              formula="μ ∘ Tμ = μ ∘ μ_T"
              description="Flattening two layers then one is the same as flattening one then two"
              holds={lawResult?.associativity ?? true}
            />
            <LawCard
              name="Left Unit"
              formula="μ ∘ η_T = id"
              description="Wrapping then flattening is the identity"
              holds={lawResult?.leftUnit ?? true}
            />
            <LawCard
              name="Right Unit"
              formula="μ ∘ Tη = id"
              description="Applying T to wrapping, then flattening, is the identity"
              holds={lawResult?.rightUnit ?? true}
            />
          </div>
        </div>
      )}

      {/* ─── Adjunction panel ─── */}
      {showAdj && (
        <div
          style={{
            padding: '12px 16px',
            borderTop: '1px solid #e2e8f0',
            background: '#f0f9ff',
          }}
        >
          <div style={{ fontSize: '14px', fontWeight: 600, marginBottom: '4px', color: COLORS.text }}>
            Generating Adjunction
          </div>
          <div style={{ fontSize: '13px', fontFamily: MATH_FONT, color: COLORS.text }}>
            {display.adjunction}
          </div>
          <div style={{ fontSize: '12px', color: COLORS.muted, marginTop: '4px' }}>
            T = GF, η from adjunction unit, μ = GεF from counit
          </div>
        </div>
      )}

      {/* ─── Concrete examples ─── */}
      <div
        style={{
          padding: '12px 16px',
          borderTop: '1px solid #e2e8f0',
        }}
      >
        <div style={{ fontSize: '14px', fontWeight: 600, marginBottom: '8px', color: COLORS.text }}>
          Concrete Example — {display.name}
        </div>
        <div style={{ fontSize: '12px', color: COLORS.muted, marginBottom: '8px' }}>
          {display.effect}
        </div>

        {display.concreteExamples.length > 1 && (
          <div style={{ marginBottom: '8px' }}>
            {display.concreteExamples.map((_, i) => (
              <button
                key={i}
                onClick={() => setExampleIdx(i)}
                style={{
                  padding: '2px 10px',
                  marginRight: '6px',
                  borderRadius: '4px',
                  border: '1px solid #cbd5e1',
                  background: exampleIdx === i ? COLORS.active : '#fff',
                  color: exampleIdx === i ? '#fff' : COLORS.text,
                  fontSize: '12px',
                  cursor: 'pointer',
                }}
              >
                Example {i + 1}
              </button>
            ))}
          </div>
        )}

        {display.concreteExamples[exampleIdx] && (
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: isNarrow ? '1fr' : 'repeat(4, 1fr)',
              gap: '8px',
            }}
          >
            <StepBox label="Input x" value={display.concreteExamples[exampleIdx].input} color={COLORS.text} />
            <StepBox label="η(x)" value={display.concreteExamples[exampleIdx].etaResult} color={COLORS.unit} />
            <StepBox label="T(T(x))" value={display.concreteExamples[exampleIdx].tOfT} color={COLORS.functor} />
            <StepBox label="μ(T(T(x)))" value={display.concreteExamples[exampleIdx].muResult} color={COLORS.mult} />
          </div>
        )}

        <div style={{ fontSize: '12px', color: COLORS.muted, marginTop: '8px' }}>
          <span style={{ fontFamily: MATH_FONT }}>η</span>: {display.etaDesc} &nbsp;|&nbsp;{' '}
          <span style={{ fontFamily: MATH_FONT }}>μ</span>: {display.muDesc}
        </div>
      </div>
    </div>
  );
}

// ─── Sub-components ───

function LawCard({
  name,
  formula,
  description,
  holds,
}: {
  name: string;
  formula: string;
  description: string;
  holds: boolean;
}) {
  return (
    <div
      style={{
        flex: '1 1 180px',
        padding: '8px 12px',
        background: '#fff',
        borderRadius: '6px',
        border: `1px solid ${holds ? '#bbf7d0' : '#fecaca'}`,
      }}
    >
      <div style={{ fontSize: '13px', fontWeight: 600, color: holds ? '#166534' : '#991b1b' }}>
        {holds ? '✓' : '✗'} {name}
      </div>
      <div style={{ fontSize: '12px', fontFamily: MATH_FONT, color: '#334155', marginTop: '2px' }}>
        {formula}
      </div>
      <div style={{ fontSize: '11px', color: '#64748b', marginTop: '2px' }}>
        {description}
      </div>
    </div>
  );
}

function StepBox({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div
      style={{
        padding: '8px 10px',
        background: COLORS.bg,
        borderRadius: '6px',
        border: `1px solid ${color}33`,
        borderLeft: `3px solid ${color}`,
      }}
    >
      <div style={{ fontSize: '11px', fontWeight: 600, color, marginBottom: '2px' }}>{label}</div>
      <div style={{ fontSize: '13px', fontFamily: MATH_FONT, color: COLORS.text }}>{value}</div>
    </div>
  );
}
