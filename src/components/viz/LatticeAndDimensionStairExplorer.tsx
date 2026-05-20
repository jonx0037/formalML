// =============================================================================
// LatticeAndDimensionStairExplorer.tsx
//
// §3 of reversible-jump-mcmc. Two complementary structural views of X:
// (A) the variable-selection lattice with a sample 6-hop RJ trajectory
// drawn as arrows along edges (uniform-random walk, seeded for reproducibility);
// (B) the mixture-order dimension-stair d_K = 3K - 1 with split/merge arrows.
//
// Both panels are graph-only — no MCMC, no Jacobians.
// Static fallback: /images/topics/reversible-jump-mcmc/fig03_state_space_structure.png
// =============================================================================
import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';

function mulberry32(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

interface LNode {
  id: string;
  card: number;
  x: number;
  y: number;
}
function buildLattice(p: number): { nodes: LNode[]; idxOf: (id: string) => LNode } {
  const total = 1 << p;
  const nodes: LNode[] = [];
  const layerCount = new Array(p + 1).fill(0);
  const layerIndex = new Array(total).fill(0);
  for (let id = 0; id < total; id++) {
    let card = 0;
    let bits = '';
    for (let j = 0; j < p; j++) {
      const b = (id >> j) & 1;
      card += b;
      bits += b;
    }
    layerIndex[id] = layerCount[card];
    layerCount[card]++;
    nodes.push({ id: bits, card, x: 0, y: 0 });
  }
  for (let i = 0; i < total; i++) {
    const k = nodes[i].card;
    nodes[i].x = (layerIndex[i] + 0.5) / layerCount[k];
    nodes[i].y = k / p;
  }
  const idxOf = (id: string) => nodes.find((n) => n.id === id)!;
  return { nodes, idxOf };
}

function sampleRandomWalk(p: number, steps: number, seed: number): string[] {
  const rng = mulberry32(seed);
  let curr = new Array(p).fill(0);
  const path: string[] = [curr.join('')];
  for (let t = 0; t < steps; t++) {
    const j = Math.floor(rng() * p);
    const next = curr.slice();
    next[j] = 1 - next[j];
    curr = next;
    path.push(curr.join(''));
  }
  return path;
}

export default function LatticeAndDimensionStairExplorer() {
  const [p, setP] = useState(4);
  const [Kmax, setKmax] = useState(6);
  const [steps, setSteps] = useState(6);

  const lattice = useMemo(() => buildLattice(p), [p]);
  const walk = useMemo(() => sampleRandomWalk(p, steps, 7), [p, steps]);

  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const W = containerWidth || 800;
  const isMobile = W < 640;
  const panelW = isMobile ? W : (W - 24) / 2;
  const panelH = 320;

  const latticeRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelW <= 0) return;
      const M = { top: 16, right: 18, bottom: 32, left: 40 };
      const w = panelW - M.left - M.right;
      const h = panelH - M.top - M.bottom;
      const g = svg.append('g').attr('transform', `translate(${M.left},${M.top})`);
      const xS = d3.scaleLinear().domain([0, 1]).range([0, w]);
      const yS = d3.scaleLinear().domain([0, 1]).range([h, 0]);

      // Edges between layers (up-only to avoid duplication)
      const edges: { s: string; t: string }[] = [];
      for (const n of lattice.nodes) {
        for (let j = 0; j < p; j++) {
          if (n.id[j] === '0') {
            const t = n.id.substring(0, j) + '1' + n.id.substring(j + 1);
            edges.push({ s: n.id, t });
          }
        }
      }
      g.selectAll('line.edge')
        .data(edges)
        .join('line')
        .attr('class', 'edge')
        .attr('x1', (e) => xS(lattice.idxOf(e.s).x))
        .attr('y1', (e) => yS(lattice.idxOf(e.s).y))
        .attr('x2', (e) => xS(lattice.idxOf(e.t).x))
        .attr('y2', (e) => yS(lattice.idxOf(e.t).y))
        .style('stroke', 'var(--color-border)')
        .attr('stroke-width', 1);

      // Trajectory arrows
      svg.append('defs').append('marker')
        .attr('id', 'arrow-traj')
        .attr('viewBox', '0 -3 6 6')
        .attr('refX', 5)
        .attr('refY', 0)
        .attr('markerWidth', 5)
        .attr('markerHeight', 5)
        .attr('orient', 'auto')
        .append('path')
        .attr('d', 'M0,-3L6,0L0,3')
        .style('fill', 'var(--color-accent)');

      for (let i = 0; i < walk.length - 1; i++) {
        const s = lattice.idxOf(walk[i]);
        const t = lattice.idxOf(walk[i + 1]);
        g.append('line')
          .attr('x1', xS(s.x))
          .attr('y1', yS(s.y))
          .attr('x2', xS(t.x))
          .attr('y2', yS(t.y))
          .style('stroke', 'var(--color-accent)')
          .attr('stroke-width', 2)
          .attr('marker-end', 'url(#arrow-traj)');
      }

      const visited = new Set(walk);
      const r = Math.max(4, Math.min(9, 16 - 2 * p));
      g.selectAll('circle.node')
        .data(lattice.nodes)
        .join('circle')
        .attr('class', 'node')
        .attr('cx', (d) => xS(d.x))
        .attr('cy', (d) => yS(d.y))
        .attr('r', r)
        .style('fill', (d) =>
          visited.has(d.id) ? 'var(--color-accent)' : 'var(--color-surface)',
        )
        .style('stroke', 'var(--color-text)')
        .attr('stroke-width', 1);

      // Dimension labels along left axis
      for (let k = 0; k <= p; k++) {
        g.append('text')
          .attr('x', -8)
          .attr('y', yS(k / p))
          .attr('dy', '0.32em')
          .attr('text-anchor', 'end')
          .style('font-size', '11px')
          .style('font-family', 'var(--font-mono)')
          .style('fill', 'var(--color-text-secondary)')
          .text(`d=${k}`);
      }
      g.append('text')
        .attr('x', w / 2)
        .attr('y', h + 24)
        .attr('text-anchor', 'middle')
        .style('font-size', '12px')
        .style('font-family', 'var(--font-sans)')
        .style('fill', 'var(--color-text)')
        .text(`p = ${p}, ${steps}-hop random walk on lattice edges`);
    },
    [lattice, walk, p, steps, panelW, panelH],
  );

  const stairRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelW <= 0) return;
      const M = { top: 16, right: 18, bottom: 36, left: 40 };
      const w = panelW - M.left - M.right;
      const h = panelH - M.top - M.bottom;
      const g = svg.append('g').attr('transform', `translate(${M.left},${M.top})`);
      const Ks = d3.range(1, Kmax + 1);
      const dims = Ks.map((K) => 3 * K - 1);
      const x = d3.scaleLinear().domain([0.5, Kmax + 0.5]).range([0, w]);
      const y = d3.scaleLinear().domain([0, Math.max(...dims) * 1.15]).range([h, 0]);

      // Step plot
      const line = d3
        .line<number>()
        .x((K) => x(K))
        .y((K) => y(3 * K - 1))
        .curve(d3.curveStepAfter);
      g.append('path').datum(Ks).attr('d', line).style('fill', 'none').style('stroke', 'var(--color-border)').attr('stroke-width', 1.5);
      g.selectAll('circle')
        .data(Ks)
        .join('circle')
        .attr('cx', (K) => x(K))
        .attr('cy', (K) => y(3 * K - 1))
        .attr('r', 6)
        .style('fill', 'var(--color-accent)');
      g.selectAll('text.dim')
        .data(Ks)
        .join('text')
        .attr('class', 'dim')
        .attr('x', (K) => x(K))
        .attr('y', (K) => y(3 * K - 1) - 10)
        .attr('text-anchor', 'middle')
        .style('font-family', 'var(--font-mono)')
        .style('font-size', '11px')
        .style('fill', 'var(--color-text)')
        .text((K) => `d=${3 * K - 1}`);

      // Split/merge arrows: curved
      svg.append('defs').append('marker')
        .attr('id', 'arrow-stair')
        .attr('viewBox', '0 -3 6 6')
        .attr('refX', 5)
        .attr('refY', 0)
        .attr('markerWidth', 5)
        .attr('markerHeight', 5)
        .attr('orient', 'auto')
        .append('path')
        .attr('d', 'M0,-3L6,0L0,3')
        .style('fill', 'var(--color-text-secondary)');

      for (let i = 0; i < Ks.length - 1; i++) {
        const K1 = Ks[i];
        const K2 = Ks[i + 1];
        const xMid = (x(K1) + x(K2)) / 2;
        const yMid = (y(3 * K1 - 1) + y(3 * K2 - 1)) / 2;
        // Split arrow up
        g.append('path')
          .attr(
            'd',
            `M ${x(K1) + 8} ${y(3 * K1 - 1)} Q ${xMid} ${yMid - 18} ${x(K2) - 8} ${y(3 * K2 - 1)}`,
          )
          .style('fill', 'none')
          .style('stroke', 'var(--color-text-secondary)')
          .attr('stroke-width', 1)
          .attr('marker-end', 'url(#arrow-stair)');
        // Merge arrow down
        g.append('path')
          .attr(
            'd',
            `M ${x(K2) - 8} ${y(3 * K2 - 1)} Q ${xMid} ${yMid + 18} ${x(K1) + 8} ${y(3 * K1 - 1)}`,
          )
          .style('fill', 'none')
          .style('stroke', 'var(--color-text-secondary)')
          .attr('stroke-width', 1)
          .attr('stroke-dasharray', '3 3')
          .attr('marker-end', 'url(#arrow-stair)');
      }

      g.append('g')
        .attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(x).ticks(Kmax))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)');
      g.append('g')
        .call(d3.axisLeft(y).ticks(6))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)');
      g.append('text')
        .attr('x', w / 2)
        .attr('y', h + 30)
        .attr('text-anchor', 'middle')
        .style('font-size', '12px')
        .style('font-family', 'var(--font-sans)')
        .style('fill', 'var(--color-text)')
        .text('K (number of mixture components)');
      g.append('text')
        .attr('x', -h / 2)
        .attr('y', -28)
        .attr('text-anchor', 'middle')
        .attr('transform', 'rotate(-90)')
        .style('font-size', '12px')
        .style('font-family', 'var(--font-sans)')
        .style('fill', 'var(--color-text)')
        .text('d_K = 3K - 1');
      // Legend
      g.append('text')
        .attr('x', w - 8)
        .attr('y', 14)
        .attr('text-anchor', 'end')
        .style('font-size', '10px')
        .style('font-family', 'var(--font-sans)')
        .style('fill', 'var(--color-text-secondary)')
        .text('solid: split (+3); dashed: merge (−3)');
    },
    [Kmax, panelW, panelH],
  );

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <div
        style={{
          display: 'flex',
          flexDirection: isMobile ? 'column' : 'row',
          gap: 24,
          alignItems: 'flex-start',
        }}
      >
        <div style={{ width: isMobile ? '100%' : panelW }}>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 12,
              marginBottom: 6,
              fontFamily: 'var(--font-sans)',
              fontSize: 13,
              color: 'var(--color-text)',
              flexWrap: 'wrap',
            }}
          >
            <label htmlFor="p3-slider" style={{ minWidth: 70 }}>
              p = <strong>{p}</strong>
            </label>
            <input
              id="p3-slider"
              type="range"
              min={3}
              max={6}
              step={1}
              value={p}
              onChange={(e) => setP(parseInt(e.target.value, 10))}
              style={{ flex: 1, minWidth: 80 }}
              aria-label="Lattice dimension p"
            />
            <label htmlFor="steps-slider" style={{ minWidth: 70 }}>
              steps = <strong>{steps}</strong>
            </label>
            <input
              id="steps-slider"
              type="range"
              min={1}
              max={20}
              step={1}
              value={steps}
              onChange={(e) => setSteps(parseInt(e.target.value, 10))}
              style={{ flex: 1, minWidth: 80 }}
              aria-label="Random walk steps"
            />
          </div>
          <svg ref={latticeRef} width={panelW} height={panelH} />
        </div>
        <div style={{ width: isMobile ? '100%' : panelW }}>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 12,
              marginBottom: 6,
              fontFamily: 'var(--font-sans)',
              fontSize: 13,
              color: 'var(--color-text)',
            }}
          >
            <label htmlFor="kmax-slider" style={{ minWidth: 70 }}>
              K_max = <strong>{Kmax}</strong>
            </label>
            <input
              id="kmax-slider"
              type="range"
              min={2}
              max={10}
              step={1}
              value={Kmax}
              onChange={(e) => setKmax(parseInt(e.target.value, 10))}
              style={{ flex: 1 }}
              aria-label="Maximum number of mixture components"
            />
          </div>
          <svg ref={stairRef} width={panelW} height={panelH} />
        </div>
      </div>
    </div>
  );
}
