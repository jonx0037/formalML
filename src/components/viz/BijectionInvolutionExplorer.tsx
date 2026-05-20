// =============================================================================
// BijectionInvolutionExplorer.tsx
//
// §6 of reversible-jump-mcmc. Theorem 2 visualization: the bijection T and its
// inverse T^(-1), paired exactly. Round-trip recovers source to machine precision;
// |det J_T| · |det J_T^(-1)| = 1.
//
// Uses the same T(β, u) = (β - αu, β + αu) bijection as §5. Slider for α;
// toggle for round-trip overlay.
//
// Static fallback: /images/topics/reversible-jump-mcmc/fig06_reversibility.png
// =============================================================================
import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';

export default function BijectionInvolutionExplorer() {
  const [alpha, setAlpha] = useState(1.0);
  const [showRoundtrip, setShowRoundtrip] = useState(true);

  const detJfwd = Math.abs(2 * alpha);
  const detJinv = 1 / detJfwd;

  // 100 random source points for the round-trip max-error check
  const samples = useMemo(() => {
    const pts: { beta: number; u: number }[] = [];
    let s = 23;
    const rand = () => {
      s = (s * 1664525 + 1013904223) >>> 0;
      return s / 4294967296;
    };
    for (let i = 0; i < 100; i++) {
      pts.push({ beta: (rand() - 0.5) * 6, u: 0.2 + rand() * 3 });
    }
    return pts;
  }, []);

  const fwd = (p: { beta: number; u: number }) => ({ x: p.beta - alpha * p.u, y: p.beta + alpha * p.u });
  const inv = (q: { x: number; y: number }) => ({ beta: (q.x + q.y) / 2, u: (q.y - q.x) / (2 * alpha) });

  // Round-trip max deviation
  const maxDev = useMemo(() => {
    let m = 0;
    for (const p of samples) {
      const r = inv(fwd(p));
      const d = Math.max(Math.abs(r.beta - p.beta), Math.abs(r.u - p.u));
      if (d > m) m = d;
    }
    return m;
  }, [samples, alpha]);

  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const W = containerWidth || 800;
  const isMobile = W < 640;
  const panelW = isMobile ? W : (W - 24) / 2;
  const panelH = 320;

  const fwdRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelW <= 0) return;
      const M = { top: 24, right: 18, bottom: 32, left: 40 };
      const w = panelW - M.left - M.right;
      const h = panelH - M.top - M.bottom;
      const g = svg.append('g').attr('transform', `translate(${M.left},${M.top})`);
      const x = d3.scaleLinear().domain([-4, 4]).range([0, w]);
      const y = d3.scaleLinear().domain([0, 4]).range([h, 0]);

      // Source samples
      g.selectAll('circle.src')
        .data(samples)
        .join('circle')
        .attr('class', 'src')
        .attr('cx', (p) => x(p.beta))
        .attr('cy', (p) => y(p.u))
        .attr('r', 2.5)
        .style('fill', 'var(--color-accent)')
        .attr('opacity', 0.65);

      // Round-trip overlay (T then T^-1) — should match source if α > 0
      if (showRoundtrip) {
        g.selectAll('circle.rt')
          .data(samples.map((p) => inv(fwd(p))))
          .join('circle')
          .attr('class', 'rt')
          .attr('cx', (p) => x(p.beta))
          .attr('cy', (p) => y(p.u))
          .attr('r', 1.5)
          .style('fill', 'var(--color-text)')
          .attr('opacity', 0.9);
      }

      g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(5)).selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.append('g').call(d3.axisLeft(y).ticks(5)).selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.append('text').attr('x', w / 2).attr('y', h + 28).attr('text-anchor', 'middle').style('font-size', '12px').style('font-family', 'var(--font-sans)').style('fill', 'var(--color-text)').text('β');
      g.append('text').attr('x', -h / 2).attr('y', -28).attr('text-anchor', 'middle').attr('transform', 'rotate(-90)').style('font-size', '12px').style('font-family', 'var(--font-sans)').style('fill', 'var(--color-text)').text('u');
      g.append('text').attr('x', w / 2).attr('y', -8).attr('text-anchor', 'middle').style('font-size', '13px').style('font-family', 'var(--font-sans)').style('fill', 'var(--color-text)').text(`Source (β, u) — |det J_T| = ${detJfwd.toFixed(2)}`);
    },
    [samples, alpha, showRoundtrip, panelW, panelH],
  );

  const tgtRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelW <= 0) return;
      const M = { top: 24, right: 18, bottom: 32, left: 40 };
      const w = panelW - M.left - M.right;
      const h = panelH - M.top - M.bottom;
      const g = svg.append('g').attr('transform', `translate(${M.left},${M.top})`);
      const x = d3.scaleLinear().domain([-7, 7]).range([0, w]);
      const y = d3.scaleLinear().domain([-7, 7]).range([h, 0]);

      g.selectAll('circle.tgt')
        .data(samples.map(fwd))
        .join('circle')
        .attr('class', 'tgt')
        .attr('cx', (p) => x(p.x))
        .attr('cy', (p) => y(p.y))
        .attr('r', 2.5)
        .style('fill', 'var(--color-accent)')
        .attr('opacity', 0.65);

      g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(5)).selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.append('g').call(d3.axisLeft(y).ticks(5)).selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.append('text').attr('x', w / 2).attr('y', h + 28).attr('text-anchor', 'middle').style('font-size', '12px').style('font-family', 'var(--font-sans)').style('fill', 'var(--color-text)').text('β₁ = β − α u');
      g.append('text').attr('x', -h / 2).attr('y', -28).attr('text-anchor', 'middle').attr('transform', 'rotate(-90)').style('font-size', '12px').style('font-family', 'var(--font-sans)').style('fill', 'var(--color-text)').text('β₂ = β + α u');
      g.append('text').attr('x', w / 2).attr('y', -8).attr('text-anchor', 'middle').style('font-size', '13px').style('font-family', 'var(--font-sans)').style('fill', 'var(--color-text)').text(`Target T(β, u) — |det J_T^(-1)| = ${detJinv.toFixed(3)}`);
    },
    [samples, alpha, panelW, panelH],
  );

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 12,
          marginBottom: 12,
          fontFamily: 'var(--font-sans)',
          fontSize: 13,
          color: 'var(--color-text)',
          flexWrap: 'wrap',
        }}
      >
        <label htmlFor="alpha-rev" style={{ minWidth: 110 }}>
          α = <strong>{alpha.toFixed(2)}</strong>
        </label>
        <input
          id="alpha-rev"
          type="range"
          min={0.25}
          max={3.0}
          step={0.05}
          value={alpha}
          onChange={(e) => setAlpha(parseFloat(e.target.value))}
          style={{ flex: 1, minWidth: 120 }}
          aria-label="Bijection scale alpha"
        />
        <label style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
          <input type="checkbox" checked={showRoundtrip} onChange={(e) => setShowRoundtrip(e.target.checked)} aria-label="Show round-trip overlay" />
          Show T⁻¹∘T overlay
        </label>
        <span
          style={{
            minWidth: 160,
            fontFamily: 'var(--font-mono)',
            color: 'var(--color-text-secondary)',
            fontSize: 11,
          }}
        >
          |det J_T| · |det J_T⁻¹| = {(detJfwd * detJinv).toFixed(6)} · max round-trip error: {maxDev.toExponential(2)}
        </span>
      </div>
      <div
        style={{
          display: 'flex',
          flexDirection: isMobile ? 'column' : 'row',
          gap: 24,
          alignItems: 'flex-start',
        }}
      >
        <div style={{ width: isMobile ? '100%' : panelW }}>
          <svg ref={fwdRef} width={panelW} height={panelH} />
        </div>
        <div style={{ width: isMobile ? '100%' : panelW }}>
          <svg ref={tgtRef} width={panelW} height={panelH} />
        </div>
      </div>
    </div>
  );
}
