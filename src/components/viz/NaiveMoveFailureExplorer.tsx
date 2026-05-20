// =============================================================================
// NaiveMoveFailureExplorer.tsx
//
// §4 of reversible-jump-mcmc. Pedagogical schematic of the measure-zero
// submanifold problem and the auxiliary-variable fix. Two panels:
// (A) 150 source samples on the line beta_2 = 0 — Lebesgue measure zero in R^2;
// (B) same 150 samples paired with Gaussian auxiliary u — fills R^2.
//
// Slider controls the auxiliary proposal scale sigma_u; both panels render
// side-by-side so the reader can compare the naive measure-zero image against
// the auxiliary-variable cloud at the same horizontal axis.
// Static fallback: /images/topics/reversible-jump-mcmc/fig04_naive_move_failure.png
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
function randn(rng: () => number): number {
  const u = 1 - rng();
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

interface Sample {
  beta1: number;
  beta2: number;
}

export default function NaiveMoveFailureExplorer() {
  const [sigmaU, setSigmaU] = useState(1.0);

  const samples = useMemo(() => {
    const rng = mulberry32(4);
    const naive: Sample[] = [];
    const aux: Sample[] = [];
    for (let i = 0; i < 150; i++) {
      const b1 = randn(rng);
      naive.push({ beta1: b1, beta2: 0 });
      aux.push({ beta1: b1, beta2: sigmaU * randn(rng) });
    }
    return { naive, aux };
  }, [sigmaU]);

  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const W = containerWidth || 800;
  const isMobile = W < 640;
  const panelW = isMobile ? W : (W - 24) / 2;
  const panelH = 320;

  const ranges: [number, number] = [-4, 4];

  function renderPanel(svg: d3.Selection<SVGSVGElement, unknown, null, undefined>, data: Sample[], title: string) {
    svg.selectAll('*').remove();
    if (panelW <= 0) return;
    const M = { top: 24, right: 18, bottom: 32, left: 40 };
    const w = panelW - M.left - M.right;
    const h = panelH - M.top - M.bottom;
    const g = svg.append('g').attr('transform', `translate(${M.left},${M.top})`);
    const x = d3.scaleLinear().domain(ranges).range([0, w]);
    const y = d3.scaleLinear().domain(ranges).range([h, 0]);

    // Grid lines at 0
    g.append('line').attr('x1', x(0)).attr('x2', x(0)).attr('y1', 0).attr('y2', h).style('stroke', 'var(--color-border)').attr('stroke-dasharray', '2 3');
    g.append('line').attr('x1', 0).attr('x2', w).attr('y1', y(0)).attr('y2', y(0)).style('stroke', 'var(--color-border)').attr('stroke-dasharray', '2 3');

    // Samples
    g.selectAll('circle.s')
      .data(data)
      .join('circle')
      .attr('class', 's')
      .attr('cx', (d) => x(d.beta1))
      .attr('cy', (d) => y(d.beta2))
      .attr('r', 2.5)
      .style('fill', 'var(--color-accent)')
      .attr('opacity', 0.65);

    g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(5)).selectAll('text').style('fill', 'var(--color-text-secondary)');
    g.append('g').call(d3.axisLeft(y).ticks(5)).selectAll('text').style('fill', 'var(--color-text-secondary)');
    g.append('text').attr('x', w / 2).attr('y', h + 28).attr('text-anchor', 'middle').style('font-size', '12px').style('font-family', 'var(--font-sans)').style('fill', 'var(--color-text)').text('β₁');
    g.append('text').attr('x', -h / 2).attr('y', -28).attr('text-anchor', 'middle').attr('transform', 'rotate(-90)').style('font-size', '12px').style('font-family', 'var(--font-sans)').style('fill', 'var(--color-text)').text('β₂');
    g.append('text').attr('x', w / 2).attr('y', -8).attr('text-anchor', 'middle').style('font-size', '13px').style('font-family', 'var(--font-sans)').style('fill', 'var(--color-text)').text(title);
  }

  const naiveRef = useD3<SVGSVGElement>((svg) => renderPanel(svg, samples.naive, 'Naive deterministic move'), [samples.naive, panelW, panelH]);
  const auxRef = useD3<SVGSVGElement>((svg) => renderPanel(svg, samples.aux, `Aux-variable fix (σ_u = ${sigmaU.toFixed(2)})`), [samples.aux, sigmaU, panelW, panelH]);

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
        }}
      >
        <label htmlFor="sigma-u" style={{ minWidth: 110 }}>
          σ_u = <strong>{sigmaU.toFixed(2)}</strong>
        </label>
        <input
          id="sigma-u"
          type="range"
          min={0.1}
          max={3.0}
          step={0.05}
          value={sigmaU}
          onChange={(e) => setSigmaU(parseFloat(e.target.value))}
          style={{ flex: 1 }}
          aria-label="Auxiliary proposal scale sigma_u"
        />
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
          <svg ref={naiveRef} width={panelW} height={panelH} />
          <div style={{ marginTop: 4, fontSize: 11, fontFamily: 'var(--font-sans)', color: 'var(--color-text-secondary)', textAlign: 'center' }}>
            Image: line β₂ = 0 — Lebesgue measure ZERO in ℝ².
          </div>
        </div>
        <div style={{ width: isMobile ? '100%' : panelW }}>
          <svg ref={auxRef} width={panelW} height={panelH} />
          <div style={{ marginTop: 4, fontSize: 11, fontFamily: 'var(--font-sans)', color: 'var(--color-text-secondary)', textAlign: 'center' }}>
            Image: scatter cloud — non-degenerate density on ℝ².
          </div>
        </div>
      </div>
    </div>
  );
}
