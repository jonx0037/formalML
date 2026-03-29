import { useState, useEffect, useRef, useId } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';

// ─── Constants ───

const SM_BREAKPOINT = 640;
const SVG_H = 320;
const MATH_FONT = 'KaTeX_Math, Georgia, serif';

const COLORS = {
  prior: '#3b82f6',       // blue
  likelihood: '#8b5cf6',  // purple
  posterior: '#f59e0b',    // amber
  forward: '#3b82f6',     // blue — forward pass
  backward: '#ef4444',    // red — backward gradients
  layer: '#8b5cf6',       // purple — layers
  gnnNode: '#22c55e',     // green — GNN nodes
  gnnEdge: '#94a3b8',     // gray
  gnnFocus: '#3b82f6',    // blue
  particle: '#8b5cf6',    // purple — particles
  bg: '#f8fafc',
  text: '#1e293b',
  muted: '#94a3b8',
  node: '#e2e8f0',
  active: '#3b82f6',
};

const TAB_NAMES = [
  'Bayesian Inference',
  'Backpropagation',
  'GNN Message Passing',
  'Probabilistic Programming',
] as const;

type Tab = 0 | 1 | 2 | 3;

// ─── Component ───

export default function MonadMLExplorer() {
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
              fontWeight: activeTab === i ? 600 : 400,
              fontSize: '13px', cursor: 'pointer',
              minWidth: '120px',
            }}
          >
            {name}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {activeTab === 0 && <BayesianInferenceTab />}
      {activeTab === 1 && <BackpropagationTab />}
      {activeTab === 2 && <GNNMessagePassingTab />}
      {activeTab === 3 && <ProbabilisticProgrammingTab />}
    </div>
  );
}

// ─── Tab 1: Bayesian Inference (Giry Monad) ───

function BayesianInferenceTab() {
  const [observation, setObservation] = useState(1); // 0, 1, or 2 index
  const { ref: containerRef, width: cw } = useResizeObserver<HTMLDivElement>();
  const svgRef = useRef<SVGSVGElement>(null);
  const uid = useId().replace(/:/g, '');

  // 3-state system
  const states = ['θ₁', 'θ₂', 'θ₃'];
  const prior = [0.33, 0.34, 0.33];
  const likelihoods = [
    [0.7, 0.2, 0.1], // P(data|θ₁)
    [0.1, 0.6, 0.3], // P(data|θ₂)
    [0.2, 0.3, 0.5], // P(data|θ₃)
  ];

  // Compute posterior via Bayes' rule (Kleisli composition)
  const lik = likelihoods.map((row) => row[observation]);
  const evidence = prior.reduce((s, p, i) => s + p * lik[i], 0);
  const posterior = prior.map((p, i) => (p * lik[i]) / evidence);

  useEffect(() => {
    if (!svgRef.current || cw < 100) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const w = cw;
    const h = SVG_H;
    const margin = { top: 30, right: 20, bottom: 40, left: 40 };
    const iw = w - margin.left - margin.right;
    const ih = h - margin.top - margin.bottom;
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const barGroups = [
      { label: 'Prior π', data: prior, color: COLORS.prior },
      { label: `Likelihood P(x${observation + 1}|θ)`, data: lik, color: COLORS.likelihood },
      { label: 'Posterior π′', data: posterior, color: COLORS.posterior },
    ];

    const groupW = iw / barGroups.length;
    const barW = Math.min(30, groupW / (states.length + 1));
    const yScale = d3.scaleLinear().domain([0, 1]).range([ih, 0]);

    barGroups.forEach((group, gi) => {
      const gx = gi * groupW + groupW * 0.1;

      // Group label
      g.append('text')
        .attr('x', gx + (groupW * 0.8) / 2)
        .attr('y', -10)
        .attr('text-anchor', 'middle')
        .style('font-size', '12px')
        .style('font-weight', 600)
        .style('fill', group.color)
        .text(group.label);

      // Bars
      group.data.forEach((val, bi) => {
        const bx = gx + bi * (barW + 4);
        const by = yScale(val);

        g.append('rect')
          .attr('x', bx)
          .attr('y', by)
          .attr('width', barW)
          .attr('height', ih - by)
          .attr('rx', 2)
          .style('fill', group.color)
          .style('opacity', 0.8);

        g.append('text')
          .attr('x', bx + barW / 2)
          .attr('y', by - 4)
          .attr('text-anchor', 'middle')
          .style('font-size', '10px')
          .style('fill', group.color)
          .text(val.toFixed(2));

        g.append('text')
          .attr('x', bx + barW / 2)
          .attr('y', ih + 14)
          .attr('text-anchor', 'middle')
          .style('font-family', MATH_FONT)
          .style('font-size', '11px')
          .style('fill', COLORS.text)
          .text(states[bi]);
      });

      // Arrow between groups
      if (gi < barGroups.length - 1) {
        const arrowX = (gi + 1) * groupW;
        g.append('text')
          .attr('x', arrowX)
          .attr('y', ih / 2)
          .attr('text-anchor', 'middle')
          .style('font-size', '18px')
          .style('fill', COLORS.muted)
          .text('→');
      }
    });

    // Y axis
    g.append('line')
      .attr('x1', 0).attr('y1', 0)
      .attr('x2', 0).attr('y2', ih)
      .style('stroke', COLORS.muted).style('stroke-width', 0.5);

  }, [cw, uid, observation, prior, lik, posterior]);

  return (
    <div ref={containerRef} style={{ background: '#fff', border: '1px solid #e2e8f0', borderRadius: '8px', overflow: 'hidden' }}>
      <div style={{ padding: '10px 16px', borderBottom: '1px solid #e2e8f0', background: COLORS.bg }}>
        <label style={{ fontSize: '13px', color: COLORS.text }}>
          Observation:
          <select
            value={observation}
            onChange={(e) => setObservation(Number(e.target.value))}
            style={{ marginLeft: '6px', padding: '4px 8px', borderRadius: '4px', border: '1px solid #cbd5e1', fontSize: '13px' }}
          >
            <option value={0}>x₁ (favors θ₁)</option>
            <option value={1}>x₂ (favors θ₂)</option>
            <option value={2}>x₃ (favors θ₃)</option>
          </select>
        </label>
      </div>
      <svg ref={svgRef} width={cw} height={SVG_H} />
      <div style={{ padding: '8px 16px', borderTop: '1px solid #e2e8f0', fontSize: '12px', color: COLORS.muted }}>
        <strong style={{ color: COLORS.text }}>Giry monad:</strong> Prior = Giry unit (δ), Likelihood = Kleisli arrow, Posterior = Kleisli composition (Chapman-Kolmogorov). Marginalization = Giry multiplication μ.
      </div>
    </div>
  );
}

// ─── Tab 2: Backpropagation (Continuation Monad) ───

function BackpropagationTab() {
  const [inputX, setInputX] = useState(2.0);
  const { ref: containerRef, width: cw } = useResizeObserver<HTMLDivElement>();
  const uid = useId().replace(/:/g, '');

  // Simple 2-layer: x -> f(x) = 3x+1 -> g(h) = h² -> L(y) = (y-10)²
  const h = 3 * inputX + 1;
  const y = h * h;
  const loss = (y - 10) * (y - 10);

  // Gradients (chain rule = Kleisli composition in Cont monad)
  const dL_dy = 2 * (y - 10);
  const dy_dh = 2 * h;
  const dh_dx = 3;
  const dL_dh = dL_dy * dy_dh;
  const dL_dx = dL_dh * dh_dx;

  return (
    <div ref={containerRef} style={{ background: '#fff', border: '1px solid #e2e8f0', borderRadius: '8px', overflow: 'hidden' }}>
      <div style={{ padding: '10px 16px', borderBottom: '1px solid #e2e8f0', background: COLORS.bg }}>
        <label style={{ fontSize: '13px', color: COLORS.text }}>
          Input x:
          <input
            type="range"
            min={-3}
            max={5}
            step={0.1}
            value={inputX}
            onChange={(e) => setInputX(Number(e.target.value))}
            style={{ marginLeft: '8px', width: '160px', verticalAlign: 'middle' }}
          />
          <span style={{ marginLeft: '8px', fontFamily: MATH_FONT }}>{inputX.toFixed(1)}</span>
        </label>
      </div>

      {/* Network diagram as styled divs */}
      <div style={{ padding: '16px', display: 'flex', flexWrap: 'wrap', gap: '8px', justifyContent: 'center', alignItems: 'center' }}>
        <NetworkNode label="x" value={inputX.toFixed(2)} color={COLORS.text} />
        <Arrow label="f" sublabel="3x+1" color={COLORS.forward} />
        <NetworkNode label="h" value={h.toFixed(2)} color={COLORS.forward} />
        <Arrow label="g" sublabel="h²" color={COLORS.forward} />
        <NetworkNode label="y" value={y.toFixed(2)} color={COLORS.forward} />
        <Arrow label="L" sublabel="(y−10)²" color={COLORS.forward} />
        <NetworkNode label="Loss" value={loss.toFixed(2)} color={COLORS.backward} />
      </div>

      {/* Gradients */}
      <div style={{ padding: '12px 16px', borderTop: '1px solid #e2e8f0', background: '#fef2f2' }}>
        <div style={{ fontSize: '13px', fontWeight: 600, marginBottom: '6px', color: COLORS.backward }}>
          Backward Pass (Kleisli composition in Cont monad)
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '12px', fontSize: '13px', fontFamily: MATH_FONT }}>
          <span style={{ color: COLORS.backward }}>∂L/∂y = {dL_dy.toFixed(2)}</span>
          <span style={{ color: COLORS.muted }}>×</span>
          <span style={{ color: COLORS.backward }}>∂y/∂h = {dy_dh.toFixed(2)}</span>
          <span style={{ color: COLORS.muted }}>×</span>
          <span style={{ color: COLORS.backward }}>∂h/∂x = {dh_dx.toFixed(2)}</span>
          <span style={{ color: COLORS.muted }}>=</span>
          <span style={{ color: COLORS.backward, fontWeight: 700 }}>∂L/∂x = {dL_dx.toFixed(2)}</span>
        </div>
      </div>

      <div style={{ padding: '8px 16px', borderTop: '1px solid #e2e8f0', fontSize: '12px', color: COLORS.muted }}>
        <strong style={{ color: COLORS.text }}>Continuation monad:</strong> Each layer f wraps as λk. k(f(x)). Backpropagation = Kleisli composition — the chain rule is functoriality of CPS.
      </div>
    </div>
  );
}

function NetworkNode({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div style={{
      padding: '8px 12px', borderRadius: '8px', border: `2px solid ${color}`,
      background: '#fff', textAlign: 'center', minWidth: '60px',
    }}>
      <div style={{ fontSize: '11px', fontWeight: 600, color }}>{label}</div>
      <div style={{ fontSize: '14px', fontFamily: MATH_FONT, color: COLORS.text }}>{value}</div>
    </div>
  );
}

function Arrow({ label, sublabel, color }: { label: string; sublabel: string; color: string }) {
  return (
    <div style={{ textAlign: 'center', padding: '0 4px' }}>
      <div style={{ fontSize: '11px', fontFamily: MATH_FONT, color }}>{label}</div>
      <div style={{ fontSize: '18px', color: COLORS.muted }}>→</div>
      <div style={{ fontSize: '10px', color: COLORS.muted }}>{sublabel}</div>
    </div>
  );
}

// ─── Tab 3: GNN Message Passing (Neighborhood Comonad) ───

function GNNMessagePassingTab() {
  const [focusNode, setFocusNode] = useState('v₂');
  const [aggFn, setAggFn] = useState<'sum' | 'mean' | 'max'>('sum');
  const [showExtended, setShowExtended] = useState(false);
  const { ref: containerRef, width: cw } = useResizeObserver<HTMLDivElement>();
  const svgRef = useRef<SVGSVGElement>(null);
  const uid = useId().replace(/:/g, '');

  const nodes = [
    { id: 'v₁', x: 0.15, y: 0.35, feature: 3 },
    { id: 'v₂', x: 0.5, y: 0.15, feature: 7 },
    { id: 'v₃', x: 0.85, y: 0.35, feature: 2 },
    { id: 'v₄', x: 0.3, y: 0.75, feature: 5 },
    { id: 'v₅', x: 0.7, y: 0.75, feature: 1 },
  ];

  const edges: [string, string][] = [
    ['v₁', 'v₂'], ['v₁', 'v₄'],
    ['v₂', 'v₃'], ['v₂', 'v₄'], ['v₂', 'v₅'],
    ['v₃', 'v₅'], ['v₄', 'v₅'],
  ];

  const getNeighbors = (id: string) => {
    const result: string[] = [];
    for (const [a, b] of edges) {
      if (a === id) result.push(b);
      if (b === id) result.push(a);
    }
    return result;
  };

  const aggregate = (id: string) => {
    const nbrs = getNeighbors(id);
    const self = nodes.find((n) => n.id === id)!.feature;
    const vals = [self, ...nbrs.map((nid) => nodes.find((n) => n.id === nid)!.feature)];
    switch (aggFn) {
      case 'sum': return vals.reduce((a, b) => a + b, 0);
      case 'mean': return Math.round((vals.reduce((a, b) => a + b, 0) / vals.length) * 10) / 10;
      case 'max': return Math.max(...vals);
    }
  };

  useEffect(() => {
    if (!svgRef.current || cw < 100) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const w = cw;
    const h = SVG_H;
    const margin = { top: 20, right: 20, bottom: 30, left: 20 };
    const iw = w - margin.left - margin.right;
    const ih = h - margin.top - margin.bottom;
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const neighbors = getNeighbors(focusNode);
    const nodeR = 22;

    // Edges
    for (const [a, b] of edges) {
      const na = nodes.find((n) => n.id === a)!;
      const nb = nodes.find((n) => n.id === b)!;
      const hl = a === focusNode || b === focusNode;
      g.append('line')
        .attr('x1', na.x * iw).attr('y1', na.y * ih)
        .attr('x2', nb.x * iw).attr('y2', nb.y * ih)
        .style('stroke', hl ? COLORS.gnnFocus : COLORS.gnnEdge)
        .style('stroke-width', hl ? 2 : 1)
        .style('opacity', hl ? 1 : 0.4);
    }

    // Nodes
    for (const node of nodes) {
      const isFocus = node.id === focusNode;
      const isNbr = neighbors.includes(node.id);
      const nx = node.x * iw;
      const ny = node.y * ih;
      const fill = isFocus ? '#dbeafe' : isNbr ? '#dcfce7' : COLORS.node;
      const stroke = isFocus ? COLORS.gnnFocus : isNbr ? COLORS.gnnNode : COLORS.muted;

      g.append('circle')
        .attr('cx', nx).attr('cy', ny).attr('r', nodeR)
        .style('fill', fill).style('stroke', stroke)
        .style('stroke-width', isFocus ? 3 : isNbr ? 2 : 1)
        .style('cursor', 'pointer')
        .on('click', () => setFocusNode(node.id));

      const displayVal = showExtended ? aggregate(node.id) : node.feature;
      g.append('text')
        .attr('x', nx).attr('y', ny + 5)
        .attr('text-anchor', 'middle')
        .style('font-family', MATH_FONT)
        .style('font-size', '15px')
        .style('fill', showExtended ? '#8b5cf6' : COLORS.text)
        .style('font-weight', isFocus ? 'bold' : 'normal')
        .style('pointer-events', 'none')
        .text(displayVal);

      g.append('text')
        .attr('x', nx).attr('y', ny - nodeR - 6)
        .attr('text-anchor', 'middle')
        .style('font-size', '11px')
        .style('fill', isFocus ? COLORS.gnnFocus : COLORS.muted)
        .style('pointer-events', 'none')
        .text(node.id);
    }

    // Label
    g.append('text')
      .attr('x', iw / 2).attr('y', ih + 10)
      .attr('text-anchor', 'middle')
      .style('font-size', '12px')
      .style('fill', COLORS.muted)
      .text(showExtended ? `After extend(${aggFn}) — coKleisli extension applied at every node` : `Click a node to set focus. Neighbors highlighted in green.`);

  }, [cw, uid, focusNode, aggFn, showExtended]);

  return (
    <div ref={containerRef} style={{ background: '#fff', border: '1px solid #e2e8f0', borderRadius: '8px', overflow: 'hidden' }}>
      <div style={{ padding: '10px 16px', borderBottom: '1px solid #e2e8f0', background: COLORS.bg, display: 'flex', flexWrap: 'wrap', gap: '12px', alignItems: 'center' }}>
        <label style={{ fontSize: '13px', color: COLORS.text }}>
          Aggregation:
          <select value={aggFn} onChange={(e) => setAggFn(e.target.value as any)}
            style={{ marginLeft: '6px', padding: '4px 8px', borderRadius: '4px', border: '1px solid #cbd5e1', fontSize: '13px' }}>
            <option value="sum">Sum</option>
            <option value="mean">Mean</option>
            <option value="max">Max</option>
          </select>
        </label>
        <button onClick={() => setShowExtended(!showExtended)}
          style={{
            padding: '4px 14px', borderRadius: '4px', border: '1px solid #cbd5e1',
            background: showExtended ? '#8b5cf6' : '#fff',
            color: showExtended ? '#fff' : COLORS.text, fontSize: '13px', cursor: 'pointer',
          }}>
          {showExtended ? 'Show original' : 'Apply extend'}
        </button>
      </div>
      <svg ref={svgRef} width={cw} height={SVG_H} />
      <div style={{ padding: '8px 16px', borderTop: '1px solid #e2e8f0', fontSize: '12px', color: COLORS.muted }}>
        <strong style={{ color: COLORS.text }}>Neighborhood comonad:</strong> CoKleisli arrow = aggregation function (extracts features from neighborhood). Extend = apply aggregation at every node simultaneously. One GNN layer = one coKleisli extension.
      </div>
    </div>
  );
}

// ─── Tab 4: Probabilistic Programming (Kleisli Category) ───

function ProbabilisticProgrammingTab() {
  const [nParticles] = useState(50);
  const { ref: containerRef, width: cw } = useResizeObserver<HTMLDivElement>();
  const svgRef = useRef<SVGSVGElement>(null);
  const uid = useId().replace(/:/g, '');

  // Simple probabilistic program: sample from prior → transform → condition
  const stages = ['Prior', 'Transform', 'Condition', 'Posterior'];

  useEffect(() => {
    if (!svgRef.current || cw < 100) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const w = cw;
    const h = SVG_H;
    const margin = { top: 30, right: 20, bottom: 30, left: 20 };
    const iw = w - margin.left - margin.right;
    const ih = h - margin.top - margin.bottom;
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const stageW = iw / stages.length;

    // Stage labels
    stages.forEach((name, i) => {
      g.append('text')
        .attr('x', i * stageW + stageW / 2)
        .attr('y', -8)
        .attr('text-anchor', 'middle')
        .style('font-size', '12px')
        .style('font-weight', 600)
        .style('fill', COLORS.text)
        .text(name);

      // Kleisli arrow label
      if (i < stages.length - 1) {
        g.append('text')
          .attr('x', (i + 0.5) * stageW + stageW / 2)
          .attr('y', -8)
          .attr('text-anchor', 'middle')
          .style('font-family', MATH_FONT)
          .style('font-size', '11px')
          .style('fill', COLORS.particle)
          .text(i === 0 ? 'K₁' : i === 1 ? 'K₂' : 'K₃');
      }
    });

    // Seed random
    const rng = d3.randomNormal(0, 1);
    const particles: { stage: number; y: number; opacity: number }[] = [];

    // Generate particles through stages
    for (let p = 0; p < nParticles; p++) {
      // Prior: normal(0, 1)
      const x0 = rng();
      particles.push({ stage: 0, y: x0, opacity: 0.6 });

      // Transform: x + noise
      const x1 = x0 * 0.8 + rng() * 0.3;
      particles.push({ stage: 1, y: x1, opacity: 0.6 });

      // Condition: weight by proximity to 0.5
      const weight = Math.exp(-((x1 - 0.5) ** 2) / 0.5);
      particles.push({ stage: 2, y: x1, opacity: weight * 0.7 + 0.1 });

      // Posterior: resampled around 0.5
      if (weight > 0.3) {
        particles.push({ stage: 3, y: x1 + rng() * 0.1, opacity: 0.8 });
      }
    }

    const yScale = d3.scaleLinear().domain([-3, 3]).range([ih, 0]);

    // Draw particles
    for (const p of particles) {
      const cx = p.stage * stageW + stageW / 2 + (Math.random() - 0.5) * stageW * 0.4;
      g.append('circle')
        .attr('cx', cx)
        .attr('cy', yScale(p.y))
        .attr('r', 3)
        .style('fill', COLORS.particle)
        .style('opacity', p.opacity);
    }

    // Stage dividers
    for (let i = 1; i < stages.length; i++) {
      g.append('line')
        .attr('x1', i * stageW).attr('y1', 0)
        .attr('x2', i * stageW).attr('y2', ih)
        .style('stroke', COLORS.node).style('stroke-dasharray', '4,3');
    }

    // Y axis labels
    [-2, -1, 0, 1, 2].forEach((v) => {
      g.append('text')
        .attr('x', -4).attr('y', yScale(v) + 4)
        .attr('text-anchor', 'end')
        .style('font-size', '10px')
        .style('fill', COLORS.muted)
        .text(v);
    });

  }, [cw, uid, nParticles]);

  return (
    <div ref={containerRef} style={{ background: '#fff', border: '1px solid #e2e8f0', borderRadius: '8px', overflow: 'hidden' }}>
      <svg ref={svgRef} width={cw} height={SVG_H} />
      <div style={{ padding: '12px 16px', borderTop: '1px solid #e2e8f0' }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginBottom: '8px' }}>
          <PipelineStep label="K₁: Sample" detail="Draw from prior N(0,1)" color={COLORS.prior} />
          <PipelineStep label="K₂: Transform" detail="Apply stochastic function" color={COLORS.likelihood} />
          <PipelineStep label="K₃: Condition" detail="Weight by observation" color={COLORS.posterior} />
        </div>
        <div style={{ fontSize: '12px', color: COLORS.muted }}>
          <strong style={{ color: COLORS.text }}>Kleisli category:</strong> Monadic bind = sequential sampling. Join = marginalizing intermediate variables. The entire program is a single composed Kleisli arrow.
        </div>
      </div>
    </div>
  );
}

function PipelineStep({ label, detail, color }: { label: string; detail: string; color: string }) {
  return (
    <div style={{
      flex: '1 1 140px', padding: '6px 10px', background: COLORS.bg,
      borderRadius: '6px', borderLeft: `3px solid ${color}`,
    }}>
      <div style={{ fontSize: '12px', fontWeight: 600, fontFamily: MATH_FONT, color }}>{label}</div>
      <div style={{ fontSize: '11px', color: COLORS.muted }}>{detail}</div>
    </div>
  );
}
