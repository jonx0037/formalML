import { useState, useEffect, useRef, useId } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import type { MarkovKernel } from './shared/types';

// ─── Constants ───

const SVG_H = 300;
const STEP_MS = 600;
const MATH_FONT = 'KaTeX_Math, Georgia, serif';

const COLORS = {
  arrow: '#3b82f6',       // blue — Kleisli arrows
  compose: '#8b5cf6',     // purple — composition
  mult: '#f59e0b',        // amber — multiplication μ
  state: '#e2e8f0',
  active: '#3b82f6',
  text: '#1e293b',
  muted: '#94a3b8',
  bg: '#f8fafc',
  bar1: '#3b82f6',
  bar2: '#8b5cf6',
  bar3: '#f59e0b',
};

// ─── Monad-specific Kleisli data ───

interface KleisliPreset {
  name: string;
  monadName: string;
  description: string;
  steps: { label: string; from: string; to: string; detail: string }[];
}

const KLEISLI_PRESETS: KleisliPreset[] = [
  {
    name: 'Maybe — Partial function pipeline',
    monadName: 'Maybe',
    description: 'Kleisli arrows A → T(B) model partial functions. Composition short-circuits on Nothing.',
    steps: [
      { label: 'f', from: 'ℤ', to: 'Maybe(ℤ)', detail: 'reciprocal: n ↦ Just(1/n) if n≠0, Nothing if n=0' },
      { label: 'g', from: 'ℤ', to: 'Maybe(ℤ)', detail: 'sqrt: n ↦ Just(√n) if n≥0, Nothing if n<0' },
    ],
  },
  {
    name: 'List — Nondeterministic pipeline',
    monadName: 'List',
    description: 'Kleisli arrows A → [B] model nondeterministic computations. Composition is concatMap.',
    steps: [
      { label: 'f', from: 'ℤ', to: '[ℤ]', detail: 'divisors: n ↦ [d | d divides n]' },
      { label: 'g', from: 'ℤ', to: '[ℤ]', detail: 'plusMinus: n ↦ [n, −n]' },
    ],
  },
  {
    name: 'Giry — Markov chain',
    monadName: 'Giry',
    description: 'Kleisli arrows A → Dist(B) are Markov kernels. Composition = Chapman-Kolmogorov.',
    steps: [
      { label: 'K₁', from: '{s₁,s₂,s₃}', to: 'Dist({s₁,s₂,s₃})', detail: 'Transition: row-stochastic matrix P₁' },
      { label: 'K₂', from: '{s₁,s₂,s₃}', to: 'Dist({s₁,s₂,s₃})', detail: 'Transition: row-stochastic matrix P₂' },
    ],
  },
];

// Markov chain data for Giry preset
const MARKOV_KERNELS: MarkovKernel[] = [
  {
    states: ['s₁', 's₂', 's₃'],
    transitionMatrix: [
      [0.2, 0.5, 0.3],
      [0.1, 0.6, 0.3],
      [0.4, 0.2, 0.4],
    ],
    label: 'K₁',
  },
  {
    states: ['s₁', 's₂', 's₃'],
    transitionMatrix: [
      [0.7, 0.2, 0.1],
      [0.3, 0.4, 0.3],
      [0.1, 0.3, 0.6],
    ],
    label: 'K₂',
  },
];

function multiplyMatrices(A: number[][], B: number[][]): number[][] {
  const n = A.length;
  const m = B[0].length;
  const k = B.length;
  const result: number[][] = Array.from({ length: n }, () => Array(m).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      for (let l = 0; l < k; l++) {
        result[i][j] += A[i][l] * B[l][j];
      }
    }
  }
  return result;
}

// ─── Component ───

export default function KleisliExplorer() {
  const [presetIdx, setPresetIdx] = useState(2); // Start with Giry
  const [animStep, setAnimStep] = useState(-1); // -1 = idle
  const [showAdjunction, setShowAdjunction] = useState(false);

  const { ref: containerRef, width: cw } = useResizeObserver<HTMLDivElement>();
  const svgRef = useRef<SVGSVGElement>(null);
  const timersRef = useRef<ReturnType<typeof setTimeout>[]>([]);
  const uid = useId().replace(/:/g, '');

  const preset = KLEISLI_PRESETS[presetIdx];
  const isGiry = preset.monadName === 'Giry';

  // Composed Markov kernel
  const composedMatrix = isGiry
    ? multiplyMatrices(MARKOV_KERNELS[0].transitionMatrix, MARKOV_KERNELS[1].transitionMatrix)
    : null;

  // Cleanup timers
  useEffect(() => {
    return () => { timersRef.current.forEach(clearTimeout); };
  }, []);

  const animate = () => {
    timersRef.current.forEach(clearTimeout);
    setAnimStep(0);
    const t1 = setTimeout(() => setAnimStep(1), STEP_MS);
    const t2 = setTimeout(() => setAnimStep(2), STEP_MS * 2);
    const t3 = setTimeout(() => setAnimStep(3), STEP_MS * 3);
    timersRef.current = [t1, t2, t3];
  };

  // ─── D3: Composition diagram ───
  useEffect(() => {
    if (!svgRef.current || cw < 100) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const w = cw;
    const h = SVG_H;
    const margin = { top: 30, right: 30, bottom: 40, left: 30 };
    const iw = w - margin.left - margin.right;
    const ih = h - margin.top - margin.bottom;

    const defs = svg.append('defs');
    ['kleisli-arr', 'kleisli-compose'].forEach((id, i) => {
      defs.append('marker')
        .attr('id', `${uid}-${id}`)
        .attr('viewBox', '0 0 10 10')
        .attr('refX', 8).attr('refY', 5)
        .attr('markerWidth', 6).attr('markerHeight', 6)
        .attr('orient', 'auto')
        .append('path')
        .attr('d', 'M0,0 L10,5 L0,10 Z')
        .style('fill', i === 0 ? COLORS.arrow : COLORS.compose);
    });

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    // Kleisli composition: A -f-> TB -Tg-> T²C -μ-> TC
    const nodeLabels = ['A', 'TB', 'T²C', 'TC'];
    const stepLabels = ['f', 'Tg', 'μ_C'];
    const stepColors = [COLORS.arrow, COLORS.arrow, COLORS.mult];
    const nodeCount = nodeLabels.length;
    const spacing = iw / (nodeCount - 1);
    const cy = ih * 0.4;

    // Nodes
    nodeLabels.forEach((label, i) => {
      const cx = i * spacing;
      const isActive = animStep >= i;

      g.append('circle')
        .attr('cx', cx).attr('cy', cy)
        .attr('r', 22)
        .style('fill', isActive ? '#dbeafe' : COLORS.state)
        .style('stroke', isActive ? COLORS.active : COLORS.muted)
        .style('stroke-width', isActive ? 2 : 1);

      g.append('text')
        .attr('x', cx).attr('y', cy + 5)
        .attr('text-anchor', 'middle')
        .style('font-family', MATH_FONT)
        .style('font-size', '14px')
        .style('font-style', 'italic')
        .style('fill', COLORS.text)
        .text(label);
    });

    // Arrows between nodes
    stepLabels.forEach((label, i) => {
      const x1 = i * spacing + 26;
      const x2 = (i + 1) * spacing - 26;
      const isActive = animStep >= i + 1;
      const color = isActive ? stepColors[i] : COLORS.muted;

      g.append('line')
        .attr('x1', x1).attr('y1', cy)
        .attr('x2', x2).attr('y2', cy)
        .style('stroke', color)
        .style('stroke-width', isActive ? 2.5 : 1.5)
        .attr('marker-end', `url(#${uid}-kleisli-${i < 2 ? 'arr' : 'compose'})`);

      g.append('text')
        .attr('x', (x1 + x2) / 2)
        .attr('y', cy - 16)
        .attr('text-anchor', 'middle')
        .style('font-family', MATH_FONT)
        .style('font-size', '13px')
        .style('font-style', 'italic')
        .style('fill', color)
        .text(label);
    });

    // Composed arrow (curved, below)
    if (animStep >= 3) {
      const x1 = 26;
      const x2 = (nodeCount - 1) * spacing - 26;
      const curveY = cy + 60;

      g.append('path')
        .attr('d', `M ${x1} ${cy + 26} Q ${(x1 + x2) / 2} ${curveY + 20}, ${x2} ${cy + 26}`)
        .style('fill', 'none')
        .style('stroke', COLORS.compose)
        .style('stroke-width', 2.5)
        .style('stroke-dasharray', '6,3')
        .attr('marker-end', `url(#${uid}-kleisli-compose)`);

      g.append('text')
        .attr('x', (x1 + x2) / 2)
        .attr('y', curveY + 16)
        .attr('text-anchor', 'middle')
        .style('font-family', MATH_FONT)
        .style('font-size', '13px')
        .style('font-style', 'italic')
        .style('fill', COLORS.compose)
        .text('g >=> f = μ ∘ Tg ∘ f');
    }

    // Title
    g.append('text')
      .attr('x', iw / 2)
      .attr('y', ih + 10)
      .attr('text-anchor', 'middle')
      .style('font-size', '12px')
      .style('fill', COLORS.muted)
      .text('Kleisli composition: (g >=> f)(a) = μ(T(g)(f(a)))');

  }, [cw, uid, animStep, preset]);

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
          Monad:
          <select
            value={presetIdx}
            onChange={(e) => { setPresetIdx(Number(e.target.value)); setAnimStep(-1); }}
            style={{
              marginLeft: '6px',
              padding: '4px 8px',
              borderRadius: '4px',
              border: '1px solid #cbd5e1',
              fontSize: '13px',
            }}
          >
            {KLEISLI_PRESETS.map((p, i) => (
              <option key={p.name} value={i}>{p.name}</option>
            ))}
          </select>
        </label>

        <button
          onClick={animate}
          style={{
            padding: '4px 14px',
            borderRadius: '4px',
            border: '1px solid #cbd5e1',
            background: COLORS.active,
            color: '#fff',
            fontSize: '13px',
            cursor: 'pointer',
          }}
        >
          Animate composition
        </button>

        <label
          style={{ fontSize: '13px', color: COLORS.text, display: 'flex', alignItems: 'center', gap: '4px' }}
        >
          <input
            type="checkbox"
            checked={showAdjunction}
            onChange={(e) => setShowAdjunction(e.target.checked)}
          />
          Show Kleisli adjunction
        </label>
      </div>

      {/* Composition diagram */}
      <svg ref={svgRef} width={cw} height={SVG_H} />

      {/* Kleisli steps detail */}
      <div
        style={{
          padding: '12px 16px',
          borderTop: '1px solid #e2e8f0',
        }}
      >
        <div style={{ fontSize: '13px', color: COLORS.muted, marginBottom: '8px' }}>
          {preset.description}
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
          {preset.steps.map((step, i) => (
            <div
              key={i}
              style={{
                flex: '1 1 200px',
                padding: '8px 10px',
                background: COLORS.bg,
                borderRadius: '6px',
                borderLeft: `3px solid ${i === 0 ? COLORS.arrow : COLORS.compose}`,
              }}
            >
              <div style={{ fontSize: '13px', fontWeight: 600, fontFamily: MATH_FONT, color: COLORS.text }}>
                {step.label}: {step.from} → {step.to}
              </div>
              <div style={{ fontSize: '12px', color: COLORS.muted, marginTop: '2px' }}>
                {step.detail}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Markov chain detail for Giry */}
      {isGiry && (
        <div style={{ padding: '12px 16px', borderTop: '1px solid #e2e8f0' }}>
          <div style={{ fontSize: '14px', fontWeight: 600, marginBottom: '8px', color: COLORS.text }}>
            Markov Kernels — Chapman-Kolmogorov as Kleisli Composition
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '16px' }}>
            {MARKOV_KERNELS.map((k, ki) => (
              <MatrixDisplay key={ki} kernel={k} />
            ))}
            {composedMatrix && (
              <MatrixDisplay
                kernel={{ states: MARKOV_KERNELS[0].states, transitionMatrix: composedMatrix, label: 'K₂ >=> K₁' }}
                isResult
              />
            )}
          </div>
          <div style={{ fontSize: '12px', color: COLORS.muted, marginTop: '8px' }}>
            Kleisli composition of Markov kernels = matrix multiplication of transition matrices
          </div>
        </div>
      )}

      {/* Kleisli adjunction */}
      {showAdjunction && (
        <div style={{ padding: '12px 16px', borderTop: '1px solid #e2e8f0', background: '#f0f9ff' }}>
          <div style={{ fontSize: '14px', fontWeight: 600, marginBottom: '4px', color: COLORS.text }}>
            Kleisli Adjunction F_T ⊣ G_T
          </div>
          <div style={{ fontSize: '13px', fontFamily: MATH_FONT, color: COLORS.text }}>
            F_T: C → C_T sends A to A, f: A → B to η_B ∘ f
          </div>
          <div style={{ fontSize: '13px', fontFamily: MATH_FONT, color: COLORS.text }}>
            G_T: C_T → C sends A to TA, (f: A → TB) to μ_B ∘ T(f)
          </div>
          <div style={{ fontSize: '12px', color: COLORS.muted, marginTop: '4px' }}>
            G_T F_T = T — the Kleisli adjunction recovers the monad
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Matrix display ───

function MatrixDisplay({ kernel, isResult = false }: { kernel: MarkovKernel; isResult?: boolean }) {
  const barColors = [COLORS.bar1, COLORS.bar2, COLORS.bar3];
  return (
    <div
      style={{
        flex: '1 1 160px',
        padding: '8px 10px',
        background: isResult ? '#f0fdf4' : COLORS.bg,
        borderRadius: '6px',
        border: `1px solid ${isResult ? '#bbf7d0' : '#e2e8f0'}`,
      }}
    >
      <div style={{ fontSize: '13px', fontWeight: 600, fontFamily: MATH_FONT, color: COLORS.text, marginBottom: '6px' }}>
        {kernel.label}
      </div>
      <table style={{ fontSize: '12px', borderCollapse: 'collapse', width: '100%' }}>
        <thead>
          <tr>
            <th style={{ padding: '2px 4px', color: COLORS.muted }}></th>
            {kernel.states.map((s) => (
              <th key={s} style={{ padding: '2px 4px', fontFamily: MATH_FONT, color: COLORS.text }}>{s}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {kernel.transitionMatrix.map((row, i) => (
            <tr key={i}>
              <td style={{ padding: '2px 4px', fontFamily: MATH_FONT, fontWeight: 600, color: barColors[i] }}>
                {kernel.states[i]}
              </td>
              {row.map((val, j) => (
                <td key={j} style={{ padding: '2px 4px', textAlign: 'center', color: COLORS.text }}>
                  {val.toFixed(2)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
