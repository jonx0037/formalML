import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { mulberry32 } from './shared/bayesian-ml';

// =============================================================================
// R2D2DecompositionExplorer — §8 Zhang-Reich-Bondell variance-decomposition
// prior. R² ~ Beta(a, b), φ ~ Dirichlet(ξ, …, ξ), and β_j | R², φ_j, σ ~
// N(0, σ² · R²/(1-R²) · φ_j).
//
// Two panels:
//   (a) implied marginal p(β_j) for the user-controlled (a, b, ξ).
//   (b) Dirichlet allocation: expected φ_j sorted, log scale.
//
// Sliders: a, b, ξ.
// =============================================================================

const PANEL_HEIGHT = 320;
const MARGIN = { top: 22, right: 16, bottom: 50, left: 56 };
const N_GRID = 200;
const N_SAMPLES = 8000;
const P_DIM = 30;

const COLOR = '#2e7baa';

function gaussianPair01(rng: () => number): [number, number] {
  const u1 = Math.max(rng(), 1e-12);
  const u2 = rng();
  const r = Math.sqrt(-2 * Math.log(u1));
  return [r * Math.cos(2 * Math.PI * u2), r * Math.sin(2 * Math.PI * u2)];
}

function sampleBeta(a: number, b: number, rng: () => number): number {
  // Marsaglia–Tsang via Gamma sampling: X ~ Gamma(a), Y ~ Gamma(b), Beta = X/(X+Y).
  // Use simple cheng/marsaglia for Gamma; for small shape (<1) fall back to scaled rejection.
  function gamma(shape: number): number {
    if (shape < 1) {
      const u = Math.max(rng(), 1e-12);
      return gamma(shape + 1) * Math.pow(u, 1 / shape);
    }
    const d = shape - 1 / 3;
    const c = 1 / Math.sqrt(9 * d);
    while (true) {
      const [z] = gaussianPair01(rng);
      let v = 1 + c * z;
      if (v <= 0) continue;
      v = v * v * v;
      const u = rng();
      if (u < 1 - 0.0331 * z * z * z * z) return d * v;
      if (Math.log(u) < 0.5 * z * z + d * (1 - v + Math.log(v))) return d * v;
    }
  }
  const x = gamma(a);
  const y = gamma(b);
  return x / (x + y);
}

function sampleDirichlet(alpha: number[], rng: () => number): number[] {
  // Sample Gamma(α_i, 1) and normalize.
  const out = new Array<number>(alpha.length);
  let s = 0;
  for (let i = 0; i < alpha.length; i++) {
    function gamma(shape: number): number {
      if (shape < 1) {
        const u = Math.max(rng(), 1e-12);
        return gamma(shape + 1) * Math.pow(u, 1 / shape);
      }
      const d = shape - 1 / 3;
      const c = 1 / Math.sqrt(9 * d);
      while (true) {
        const [z] = gaussianPair01(rng);
        let v = 1 + c * z;
        if (v <= 0) continue;
        v = v * v * v;
        const u = rng();
        if (u < 1 - 0.0331 * z * z * z * z) return d * v;
        if (Math.log(u) < 0.5 * z * z + d * (1 - v + Math.log(v))) return d * v;
      }
    }
    out[i] = gamma(alpha[i]);
    s += out[i];
  }
  for (let i = 0; i < out.length; i++) out[i] /= s;
  return out;
}

export default function R2D2DecompositionExplorer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [aR2, setAR2] = useState(0.5);
  const [bR2, setBR2] = useState(5.0);
  const [xi, setXi] = useState(0.5);

  const data = useMemo(() => {
    const rng = mulberry32(20260509);
    // Sample β draws from R2-D2 prior at p = P_DIM, σ = 1.
    const betaSamples: number[] = new Array(N_SAMPLES);
    let phiSums = new Array<number>(P_DIM).fill(0);
    for (let s = 0; s < N_SAMPLES; s++) {
      const r2 = sampleBeta(aR2, bR2, rng);
      const phi = sampleDirichlet(new Array(P_DIM).fill(xi), rng);
      const j = Math.floor(rng() * P_DIM);
      const variance = (r2 / Math.max(1 - r2, 1e-9)) * phi[j];
      const [z] = gaussianPair01(rng);
      betaSamples[s] = z * Math.sqrt(variance);
      phi.sort((a, b) => b - a);
      for (let k = 0; k < P_DIM; k++) phiSums[k] += phi[k];
    }
    phiSums = phiSums.map((s) => s / N_SAMPLES);

    // KDE on betaGrid
    const betaGrid = new Array<number>(N_GRID);
    for (let i = 0; i < N_GRID; i++) {
      betaGrid[i] = -3 + (6 * i) / (N_GRID - 1);
    }
    const bw = 0.06;
    const density = new Array<number>(N_GRID);
    for (let i = 0; i < N_GRID; i++) {
      const x = betaGrid[i];
      let acc = 0;
      for (let s = 0; s < N_SAMPLES; s++) {
        const u = (x - betaSamples[s]) / bw;
        acc += Math.exp(-0.5 * u * u);
      }
      density[i] = acc / (N_SAMPLES * bw * Math.sqrt(2 * Math.PI));
    }

    return { betaGrid, density, phiSums };
  }, [aR2, bR2, xi]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const w = width || 760;
      const h = PANEL_HEIGHT;
      svg.attr('width', w).attr('height', h);
      const panelW = (w - MARGIN.left - MARGIN.right - 60) / 2;
      const innerH = h - MARGIN.top - MARGIN.bottom;

      // Panel (a)
      const gA = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);
      const xA = d3.scaleLinear().domain([-3, 3]).range([0, panelW]);
      const yMax = Math.max(d3.max(data.density) ?? 1, 0.5);
      const yA = d3.scaleLinear().domain([0, yMax * 1.1]).range([innerH, 0]);
      gA.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xA).ticks(5));
      gA.append('g').call(d3.axisLeft(yA).ticks(5));
      gA.append('text')
        .attr('x', panelW / 2)
        .attr('y', innerH + 36)
        .attr('text-anchor', 'middle')
        .style('font-size', '11px')
        .text('β');
      gA.append('text')
        .attr('x', 0)
        .attr('y', -8)
        .style('font-size', '11px')
        .style('font-weight', '600')
        .text('(a) implied p(β) under R2-D2');
      const lineA = d3
        .line<number>()
        .x((_, i) => xA(data.betaGrid[i]))
        .y((d) => yA(d))
        .curve(d3.curveMonotoneX);
      gA.append('path')
        .datum(data.density)
        .attr('fill', 'none')
        .attr('stroke', COLOR)
        .attr('stroke-width', 2.4)
        .attr('d', lineA);

      // Panel (b): Dirichlet allocation sorted, bar chart
      const gB = svg
        .append('g')
        .attr('transform', `translate(${MARGIN.left + panelW + 60},${MARGIN.top})`);
      const xB = d3.scaleLinear().domain([0, P_DIM]).range([0, panelW]);
      const yB = d3
        .scaleLog()
        .domain([Math.max(d3.min(data.phiSums) ?? 1e-3, 1e-4), 1])
        .range([innerH, 0]);
      gB.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xB).ticks(5));
      gB.append('g').call(d3.axisLeft(yB).ticks(5, '~g'));
      gB.append('text')
        .attr('x', panelW / 2)
        .attr('y', innerH + 36)
        .attr('text-anchor', 'middle')
        .style('font-size', '11px')
        .text('rank (sorted)');
      gB.append('text')
        .attr('x', 0)
        .attr('y', -8)
        .style('font-size', '11px')
        .style('font-weight', '600')
        .text('(b) Dirichlet allocation E[φ_j]');
      data.phiSums.forEach((v, i) => {
        gB.append('rect')
          .attr('x', xB(i) + 1)
          .attr('y', yB(Math.max(v, 1e-4)))
          .attr('width', xB(1) - xB(0) - 2)
          .attr('height', innerH - yB(Math.max(v, 1e-4)))
          .attr('fill', COLOR)
          .attr('opacity', 0.75);
      });
    },
    [data, width],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', maxWidth: 760 }}>
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: 14,
          marginBottom: 8,
          alignItems: 'center',
          fontSize: 12.5,
        }}
      >
        <label>
          a: {aR2.toFixed(2)}{' '}
          <input
            type="range"
            min={0.1}
            max={5.0}
            step={0.05}
            value={aR2}
            onChange={(e) => setAR2(Number(e.target.value))}
            style={{ width: 110 }}
          />
        </label>
        <label>
          b: {bR2.toFixed(2)}{' '}
          <input
            type="range"
            min={0.5}
            max={10.0}
            step={0.05}
            value={bR2}
            onChange={(e) => setBR2(Number(e.target.value))}
            style={{ width: 110 }}
          />
        </label>
        <label>
          ξ: {xi.toFixed(2)}{' '}
          <input
            type="range"
            min={0.1}
            max={3.0}
            step={0.05}
            value={xi}
            onChange={(e) => setXi(Number(e.target.value))}
            style={{ width: 110 }}
          />
        </label>
      </div>
      <svg ref={svgRef} role="img" aria-label="R2-D2 prior decomposition" />
      <div style={{ fontSize: 11.5, marginTop: 6, color: '#555' }}>
        E[R²] = a/(a+b) ≈ {(aR2 / (aR2 + bR2)).toFixed(3)}; small ξ concentrates Dirichlet on a few
        coefficients (sparse), large ξ spreads it uniformly (dense).
      </div>
    </div>
  );
}
