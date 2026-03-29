import { useState, useEffect, useRef, useId } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';

// ─── Constants ───

const SM_BREAKPOINT = 640;
const SVG_H = 320;
const MATH_FONT = 'KaTeX_Math, Georgia, serif';

const COLORS = {
  primal: '#3b82f6',
  dual: '#8b5cf6',
  gap: '#f59e0b',
  encoder: '#3b82f6',
  decoder: '#8b5cf6',
  recon: '#ef4444',
  query: '#3b82f6',
  key: '#8b5cf6',
  value: '#f59e0b',
  free: '#3b82f6',
  constrained: '#8b5cf6',
  penalty: '#ef4444',
  bg: '#f8fafc',
  text: '#1e293b',
  muted: '#94a3b8',
  node: '#e2e8f0',
  active: '#3b82f6',
};

const TAB_NAMES = [
  'Lagrangian Duality',
  'Encoder-Decoder',
  'Tensor-Hom (Attention)',
  'Regularization',
] as const;

type Tab = 0 | 1 | 2 | 3;

// ─── Component ───

export default function AdjunctionMLExplorer() {
  const [activeTab, setActiveTab] = useState<Tab>(0);

  return (
    <div style={{ margin: '1.5rem 0' }}>
      {/* Tab bar */}
      <div style={{
        display: 'flex', flexWrap: 'wrap', gap: '2px',
        marginBottom: '0.75rem', borderRadius: '8px',
        overflow: 'hidden', border: `1px solid ${COLORS.node}`,
      }}>
        {TAB_NAMES.map((name, i) => (
          <button
            key={name}
            onClick={() => setActiveTab(i as Tab)}
            style={{
              flex: 1, padding: '8px 12px', border: 'none',
              background: activeTab === i ? COLORS.active : COLORS.bg,
              color: activeTab === i ? '#fff' : COLORS.text,
              fontWeight: activeTab === i ? 700 : 500,
              fontSize: '12px', cursor: 'pointer',
              minWidth: '120px',
            }}
          >
            {name}
          </button>
        ))}
      </div>

      {/* Active tab */}
      {activeTab === 0 && <LagrangianDualityTab />}
      {activeTab === 1 && <EncoderDecoderTab />}
      {activeTab === 2 && <TensorHomTab />}
      {activeTab === 3 && <RegularizationTab />}
    </div>
  );
}

// ─── Tab 1: Lagrangian Duality ───

function LagrangianDualityTab() {
  const [constraint, setConstraint] = useState(1.5);
  const { ref: containerRef, width: cw } = useResizeObserver<HTMLDivElement>();
  const svgRef = useRef<SVGSVGElement>(null);
  const uid = useId().replace(/:/g, '');

  useEffect(() => {
    if (!svgRef.current || cw < 100) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const w = cw;
    const h = SVG_H;
    const margin = { top: 30, right: 30, bottom: 40, left: 50 };
    const iw = w - margin.left - margin.right;
    const ih = h - margin.top - margin.bottom;

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    // Problem: min x² s.t. x >= b (constraint parameter)
    // Primal: f* = b² when b >= 0, f* = 0 when b < 0
    // Dual: d* = max_{λ>=0} min_x (x² + λ(b - x)) = max_{λ>=0} (λb - λ²/4)
    // Dual optimal: λ* = 2b when b >= 0, λ* = 0 when b < 0

    const b = constraint;
    const primalOpt = b >= 0 ? b * b : 0;
    const dualOpt = b >= 0 ? b * b : 0; // strong duality for convex

    const xScale = d3.scaleLinear().domain([-2, 4]).range([0, iw]);
    const yScale = d3.scaleLinear().domain([0, 10]).range([ih, 0]);

    // Axes
    g.append('g').attr('transform', `translate(0,${ih})`).call(d3.axisBottom(xScale).ticks(6))
      .selectAll('text').style('font-size', '10px');
    g.append('g').call(d3.axisLeft(yScale).ticks(5))
      .selectAll('text').style('font-size', '10px');

    g.append('text').attr('x', iw / 2).attr('y', ih + 35)
      .attr('text-anchor', 'middle').style('font-size', '12px').style('fill', COLORS.text)
      .text('x');
    g.append('text').attr('x', -35).attr('y', ih / 2)
      .attr('text-anchor', 'middle').attr('transform', `rotate(-90, -35, ${ih / 2})`)
      .style('font-size', '12px').style('fill', COLORS.text)
      .text('f(x)');

    // Primal objective: x²
    const xs = d3.range(-2, 4, 0.05);
    const line = d3.line<number>()
      .x((x) => xScale(x))
      .y((x) => yScale(Math.min(x * x, 10)));

    g.append('path')
      .datum(xs)
      .attr('d', line)
      .attr('fill', 'none')
      .attr('stroke', COLORS.primal)
      .attr('stroke-width', 2);

    // Constraint region
    g.append('rect')
      .attr('x', 0).attr('y', 0)
      .attr('width', Math.max(0, xScale(b)))
      .attr('height', ih)
      .attr('fill', COLORS.penalty)
      .attr('opacity', 0.08);

    // Constraint line
    g.append('line')
      .attr('x1', xScale(b)).attr('y1', 0)
      .attr('x2', xScale(b)).attr('y2', ih)
      .attr('stroke', COLORS.penalty)
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '5,3');

    g.append('text')
      .attr('x', xScale(b) + 5).attr('y', 15)
      .style('font-family', MATH_FONT).style('font-size', '11px')
      .style('fill', COLORS.penalty)
      .text(`x ≥ ${b.toFixed(1)}`);

    // Primal optimum
    const pOptX = Math.max(0, b);
    g.append('circle')
      .attr('cx', xScale(pOptX)).attr('cy', yScale(primalOpt))
      .attr('r', 6)
      .attr('fill', COLORS.primal)
      .attr('stroke', 'white').attr('stroke-width', 2);

    // Dual optimum
    g.append('circle')
      .attr('cx', xScale(pOptX)).attr('cy', yScale(dualOpt))
      .attr('r', 4)
      .attr('fill', COLORS.dual)
      .attr('stroke', 'white').attr('stroke-width', 2);

    // Labels
    g.append('text')
      .attr('x', iw - 5).attr('y', 15)
      .attr('text-anchor', 'end')
      .style('font-family', MATH_FONT).style('font-size', '12px')
      .style('fill', COLORS.primal)
      .text(`f* = ${primalOpt.toFixed(2)}`);

    g.append('text')
      .attr('x', iw - 5).attr('y', 30)
      .attr('text-anchor', 'end')
      .style('font-family', MATH_FONT).style('font-size', '12px')
      .style('fill', COLORS.dual)
      .text(`d* = ${dualOpt.toFixed(2)}`);

    const dualityGap = primalOpt - dualOpt;
    g.append('text')
      .attr('x', iw - 5).attr('y', 45)
      .attr('text-anchor', 'end')
      .style('font-family', MATH_FONT).style('font-size', '12px')
      .style('fill', dualityGap < 0.01 ? COLORS.gap : COLORS.penalty)
      .text(`gap = ${dualityGap.toFixed(2)}${dualityGap < 0.01 ? ' (strong duality!)' : ''}`);

  }, [cw, constraint, uid]);

  return (
    <div ref={containerRef}>
      <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', marginBottom: '0.5rem', fontSize: '13px' }}>
        <label style={{ fontWeight: 600, color: COLORS.text }}>Constraint bound b</label>
        <input type="range" min="-1" max="3" step="0.1" value={constraint}
          onChange={(e) => setConstraint(+e.target.value)} style={{ width: '150px' }} />
        <span style={{ color: COLORS.muted }}>{constraint.toFixed(1)}</span>
      </div>
      <svg ref={svgRef} width={cw} height={SVG_H}
        style={{ display: 'block', background: 'white', borderRadius: '8px', border: `1px solid ${COLORS.node}` }} />
      <p style={{ fontSize: '12px', color: COLORS.muted, marginTop: '0.5rem' }}>
        Convex program: min x² s.t. x ≥ b. Strong duality holds (convex QP). The Galois connection collapses to an equality: f* = d*.
      </p>
    </div>
  );
}

// ─── Tab 2: Encoder-Decoder ───

function EncoderDecoderTab() {
  const { ref: containerRef, width: cw } = useResizeObserver<HTMLDivElement>();
  const svgRef = useRef<SVGSVGElement>(null);
  const uid = useId().replace(/:/g, '');

  useEffect(() => {
    if (!svgRef.current || cw < 100) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const w = cw;
    const h = SVG_H;
    const margin = { top: 30, right: 20, bottom: 30, left: 20 };

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
    const iw = w - margin.left - margin.right;
    const ih = h - margin.top - margin.bottom;

    // Generate 2D data points
    const points = [
      [1, 2], [2, 3], [3, 2.5], [1.5, 3.5], [2.5, 1.5], [3.5, 3], [0.5, 1], [4, 2],
    ];

    // PCA: project to first principal component (roughly the diagonal)
    const meanX = d3.mean(points, (p) => p[0])!;
    const meanY = d3.mean(points, (p) => p[1])!;

    // Simple projection onto y = x line (45 degrees)
    const project = (p: number[]): number => (p[0] + p[1]) / Math.sqrt(2);
    const reconstruct = (z: number): [number, number] => [z / Math.sqrt(2), z / Math.sqrt(2)];

    const latentValues = points.map(project);
    const reconstructed = latentValues.map(reconstruct);

    // Three panels: input, latent, reconstruction
    const panelW = iw / 3.5;
    const panelGap = (iw - panelW * 3) / 2;

    // Scales for 2D panels
    const xScale = d3.scaleLinear().domain([-0.5, 5]).range([0, panelW]);
    const yScale = d3.scaleLinear().domain([-0.5, 5]).range([ih, 0]);
    // Scale for latent (1D)
    const zScale = d3.scaleLinear().domain([0, 5]).range([ih, 0]);

    const panels = [
      { cx: 0, label: '𝒳 (Input)', color: COLORS.encoder },
      { cx: panelW + panelGap, label: '𝒵 (Latent)', color: COLORS.gap },
      { cx: (panelW + panelGap) * 2, label: '𝒳̂ (Reconstructed)', color: COLORS.decoder },
    ];

    panels.forEach((panel) => {
      g.append('text')
        .attr('x', panel.cx + panelW / 2).attr('y', -8)
        .attr('text-anchor', 'middle')
        .style('font-family', MATH_FONT).style('font-size', '13px')
        .style('font-weight', 'bold').style('fill', panel.color)
        .text(panel.label);
    });

    // Input points
    points.forEach((p) => {
      g.append('circle')
        .attr('cx', panels[0].cx + xScale(p[0]))
        .attr('cy', yScale(p[1]))
        .attr('r', 5)
        .attr('fill', COLORS.encoder)
        .attr('opacity', 0.7);
    });

    // Latent points (1D vertical)
    latentValues.forEach((z) => {
      g.append('circle')
        .attr('cx', panels[1].cx + panelW / 2)
        .attr('cy', zScale(z))
        .attr('r', 5)
        .attr('fill', COLORS.gap)
        .attr('opacity', 0.7);
    });

    // Latent line
    g.append('line')
      .attr('x1', panels[1].cx + panelW / 2).attr('y1', 0)
      .attr('x2', panels[1].cx + panelW / 2).attr('y2', ih)
      .attr('stroke', COLORS.muted).attr('stroke-width', 1).attr('stroke-dasharray', '3,3');

    // Reconstructed points
    reconstructed.forEach((p) => {
      g.append('circle')
        .attr('cx', panels[2].cx + xScale(p[0]))
        .attr('cy', yScale(p[1]))
        .attr('r', 5)
        .attr('fill', COLORS.decoder)
        .attr('opacity', 0.7);
    });

    // Show reconstruction error (lines from original to reconstructed)
    points.forEach((p, i) => {
      const r = reconstructed[i];
      // Offset for side-by-side display
      g.append('line')
        .attr('x1', panels[0].cx + xScale(p[0]))
        .attr('y1', yScale(p[1]))
        .attr('x2', panels[0].cx + xScale(r[0]))
        .attr('y2', yScale(r[1]))
        .attr('stroke', COLORS.recon)
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '2,2')
        .attr('opacity', 0.5);
    });

    // Arrows between panels
    const arrowY = ih / 2;
    // Encode arrow
    g.append('text')
      .attr('x', panels[0].cx + panelW + panelGap / 2).attr('y', arrowY - 10)
      .attr('text-anchor', 'middle')
      .style('font-family', MATH_FONT).style('font-size', '12px')
      .style('fill', COLORS.encoder)
      .text('Encode (F)');
    g.append('text')
      .attr('x', panels[0].cx + panelW + panelGap / 2).attr('y', arrowY + 5)
      .attr('text-anchor', 'middle').style('font-size', '18px').style('fill', COLORS.encoder)
      .text('→');

    // Decode arrow
    g.append('text')
      .attr('x', panels[1].cx + panelW + panelGap / 2).attr('y', arrowY - 10)
      .attr('text-anchor', 'middle')
      .style('font-family', MATH_FONT).style('font-size', '12px')
      .style('fill', COLORS.decoder)
      .text('Decode (G)');
    g.append('text')
      .attr('x', panels[1].cx + panelW + panelGap / 2).attr('y', arrowY + 5)
      .attr('text-anchor', 'middle').style('font-size', '18px').style('fill', COLORS.decoder)
      .text('→');

    // Unit label
    const reconError = d3.mean(points.map((p, i) => {
      const r = reconstructed[i];
      return Math.sqrt((p[0] - r[0]) ** 2 + (p[1] - r[1]) ** 2);
    }))!;

    g.append('text')
      .attr('x', iw / 2).attr('y', ih + 20)
      .attr('text-anchor', 'middle')
      .style('font-family', MATH_FONT).style('font-size', '12px')
      .style('fill', COLORS.recon)
      .text(`Unit η (reconstruction error): avg ‖x − Dec(Enc(x))‖ = ${reconError.toFixed(2)}`);

  }, [cw, uid]);

  return (
    <div ref={containerRef}>
      <svg ref={svgRef} width={cw} height={SVG_H}
        style={{ display: 'block', background: 'white', borderRadius: '8px', border: `1px solid ${COLORS.node}` }} />
      <p style={{ fontSize: '12px', color: COLORS.muted, marginTop: '0.5rem' }}>
        PCA as autoencoder: 2D→1D→2D. The unit η measures reconstruction error. A perfect autoencoder (η = id) means the adjunction is an equivalence.
      </p>
    </div>
  );
}

// ─── Tab 3: Tensor-Hom (Attention) ───

function TensorHomTab() {
  const { ref: containerRef, width: cw } = useResizeObserver<HTMLDivElement>();
  const isNarrow = cw < SM_BREAKPOINT;

  // Small 2x2 example
  const Q = [[1, 0], [0, 1]]; // Identity for simplicity
  const K = [[1, 2], [3, 4]];
  const scores = Q.map((qRow) => K[0].map((_, j) =>
    qRow.reduce((sum, qi, k) => sum + qi * K[k][j], 0),
  ));

  return (
    <div ref={containerRef}>
      <div style={{
        display: 'flex', flexDirection: isNarrow ? 'column' : 'row',
        gap: '1rem', padding: '1rem',
        background: 'white', borderRadius: '8px', border: `1px solid ${COLORS.node}`,
      }}>
        {/* Uncurried form */}
        <div style={{ flex: 1 }}>
          <h4 style={{ fontFamily: MATH_FONT, color: COLORS.query, fontSize: '14px', margin: '0 0 0.5rem' }}>
            Uncurried: Q ⊗ K → scores
          </h4>
          <div style={{ fontFamily: 'monospace', fontSize: '12px', lineHeight: 1.6, color: COLORS.text }}>
            <div>Q = [{Q.map((r) => `[${r.join(', ')}]`).join(', ')}]</div>
            <div>K = [{K.map((r) => `[${r.join(', ')}]`).join(', ')}]</div>
            <div style={{ marginTop: '0.5rem', color: COLORS.gap }}>
              Q·K<sup>T</sup> = [{scores.map((r) => `[${r.join(', ')}]`).join(', ')}]
            </div>
            <div style={{ marginTop: '0.5rem', color: COLORS.muted, fontSize: '11px' }}>
              Bilinear map: takes Q and K simultaneously
            </div>
          </div>
        </div>

        {/* Bijection arrow */}
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: '24px', color: COLORS.gap, fontFamily: MATH_FONT,
          padding: '0 0.5rem',
        }}>
          ≅
        </div>

        {/* Curried form */}
        <div style={{ flex: 1 }}>
          <h4 style={{ fontFamily: MATH_FONT, color: COLORS.decoder, fontSize: '14px', margin: '0 0 0.5rem' }}>
            Curried: Q → Hom(K, scores)
          </h4>
          <div style={{ fontFamily: 'monospace', fontSize: '12px', lineHeight: 1.6, color: COLORS.text }}>
            <div>q₁ = [{Q[0].join(', ')}] ↦ (k ↦ q₁·k)</div>
            <div style={{ marginLeft: '1.5rem', color: COLORS.decoder }}>
              → linear map sending [{K[0].join(', ')}] to {scores[0][0]}, [{K[1].join(', ')}] to {scores[0][1]}
            </div>
            <div style={{ marginTop: '0.25rem' }}>q₂ = [{Q[1].join(', ')}] ↦ (k ↦ q₂·k)</div>
            <div style={{ marginLeft: '1.5rem', color: COLORS.decoder }}>
              → linear map sending [{K[0].join(', ')}] to {scores[1][0]}, [{K[1].join(', ')}] to {scores[1][1]}
            </div>
            <div style={{ marginTop: '0.5rem', color: COLORS.muted, fontSize: '11px' }}>
              Each query maps to a function on keys
            </div>
          </div>
        </div>
      </div>
      <p style={{ fontSize: '12px', color: COLORS.muted, marginTop: '0.5rem' }}>
        Tensor-hom adjunction: Hom(Q ⊗ K, V) ≅ Hom(Q, Hom(K, V)). Attention computes in the uncurried form (Q·K<sup>T</sup>), but the curried interpretation reveals each query as a function on keys.
      </p>
    </div>
  );
}

// ─── Tab 4: Regularization ───

function RegularizationTab() {
  const [lambda, setLambda] = useState(0.5);
  const { ref: containerRef, width: cw } = useResizeObserver<HTMLDivElement>();
  const svgRef = useRef<SVGSVGElement>(null);
  const uid = useId().replace(/:/g, '');

  useEffect(() => {
    if (!svgRef.current || cw < 100) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const w = cw;
    const h = SVG_H;
    const margin = { top: 30, right: 30, bottom: 40, left: 50 };
    const iw = w - margin.left - margin.right;
    const ih = h - margin.top - margin.bottom;

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    // Problem: min_w (w-3)² + λw²
    // Solution: w* = 3 / (1 + 2λ)
    const wOpt = 3 / (1 + 2 * lambda);

    const xScale = d3.scaleLinear().domain([-1, 5]).range([0, iw]);
    const yScale = d3.scaleLinear().domain([0, 15]).range([ih, 0]);

    // Axes
    g.append('g').attr('transform', `translate(0,${ih})`).call(d3.axisBottom(xScale).ticks(6))
      .selectAll('text').style('font-size', '10px');
    g.append('g').call(d3.axisLeft(yScale).ticks(5))
      .selectAll('text').style('font-size', '10px');

    g.append('text').attr('x', iw / 2).attr('y', ih + 35)
      .attr('text-anchor', 'middle').style('font-size', '12px').style('fill', COLORS.text)
      .text('w');

    const xs = d3.range(-1, 5, 0.05);

    // Loss function (w-3)²
    const lossLine = d3.line<number>()
      .x((x) => xScale(x))
      .y((x) => yScale(Math.min((x - 3) ** 2, 15)));
    g.append('path').datum(xs).attr('d', lossLine)
      .attr('fill', 'none').attr('stroke', COLORS.free).attr('stroke-width', 2).attr('opacity', 0.6);

    // Penalty λw²
    const penaltyLine = d3.line<number>()
      .x((x) => xScale(x))
      .y((x) => yScale(Math.min(lambda * x * x, 15)));
    g.append('path').datum(xs).attr('d', penaltyLine)
      .attr('fill', 'none').attr('stroke', COLORS.penalty).attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '5,3').attr('opacity', 0.6);

    // Total: (w-3)² + λw²
    const totalLine = d3.line<number>()
      .x((x) => xScale(x))
      .y((x) => yScale(Math.min((x - 3) ** 2 + lambda * x * x, 15)));
    g.append('path').datum(xs).attr('d', totalLine)
      .attr('fill', 'none').attr('stroke', COLORS.constrained).attr('stroke-width', 2.5);

    // Optimal points
    // Unconstrained optimum at w=3
    g.append('circle')
      .attr('cx', xScale(3)).attr('cy', yScale(0))
      .attr('r', 5).attr('fill', COLORS.free).attr('stroke', 'white').attr('stroke-width', 2);
    g.append('text')
      .attr('x', xScale(3) + 8).attr('y', yScale(0) - 5)
      .style('font-family', MATH_FONT).style('font-size', '11px').style('fill', COLORS.free)
      .text('w=3 (free)');

    // Regularized optimum
    const regLoss = (wOpt - 3) ** 2 + lambda * wOpt * wOpt;
    g.append('circle')
      .attr('cx', xScale(wOpt)).attr('cy', yScale(regLoss))
      .attr('r', 6).attr('fill', COLORS.constrained).attr('stroke', 'white').attr('stroke-width', 2);
    g.append('text')
      .attr('x', xScale(wOpt) + 8).attr('y', yScale(regLoss) - 5)
      .style('font-family', MATH_FONT).style('font-size', '11px').style('fill', COLORS.constrained)
      .text(`w*=${wOpt.toFixed(2)} (reg.)`);

    // Legend
    const legendY = 10;
    [
      { color: COLORS.free, label: '(w−3)² (loss)', dash: '' },
      { color: COLORS.penalty, label: `λw² (penalty, λ=${lambda.toFixed(1)})`, dash: '5,3' },
      { color: COLORS.constrained, label: 'Total (regularized)', dash: '' },
    ].forEach((item, i) => {
      g.append('line')
        .attr('x1', 10).attr('y1', legendY + i * 16)
        .attr('x2', 30).attr('y2', legendY + i * 16)
        .attr('stroke', item.color).attr('stroke-width', 2)
        .attr('stroke-dasharray', item.dash);
      g.append('text')
        .attr('x', 35).attr('y', legendY + i * 16 + 4)
        .style('font-size', '10px').style('fill', COLORS.text)
        .text(item.label);
    });

  }, [cw, lambda, uid]);

  return (
    <div ref={containerRef}>
      <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', marginBottom: '0.5rem', fontSize: '13px' }}>
        <label style={{ fontWeight: 600, color: COLORS.text }}>λ (regularization)</label>
        <input type="range" min="0" max="5" step="0.1" value={lambda}
          onChange={(e) => setLambda(+e.target.value)} style={{ width: '150px' }} />
        <span style={{ color: COLORS.muted }}>{lambda.toFixed(1)}</span>
      </div>
      <svg ref={svgRef} width={cw} height={SVG_H}
        style={{ display: 'block', background: 'white', borderRadius: '8px', border: `1px solid ${COLORS.node}` }} />
      <p style={{ fontSize: '12px', color: COLORS.muted, marginTop: '0.5rem' }}>
        L2 regularization as unit insertion: the "free" optimum (w=3) is pulled toward zero by the penalty.
        As λ→∞, w*→0 (maximally constrained). As λ→0, w*→3 (unconstrained).
      </p>
    </div>
  );
}
