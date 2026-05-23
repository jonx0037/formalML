import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import {
  betaBinomialUpdate,
  betaMean,
  betaSample,
  mulberry32,
  COLOR_VARIANT_A,
  COLOR_VARIANT_B,
  COLOR_BAND,
} from './shared/bayesian-inference';

// =============================================================================
// DecisionBoundaryExplorer — §7 interactive.
//
// Two-panel decision-theory dashboard for the A/B-test running example.
// Left: posterior expected utility under "stay" (horizontal at E[θ_A|y])
// and "switch" (linearly decreasing line E[θ_B|y] - c). The crossing point
// c* = E[θ_B - θ_A | y] is the Bayes decision boundary.
// Right: posterior probability P(θ_B - c > θ_A | y) as a function of c,
// computed by Monte Carlo from the joint posterior — at c=0 it equals the
// §1.2 P(B > A) ≈ 0.95; at c=c* it crosses 0.5.
//
// Commit-on-release: the Monte Carlo (M=20,000 per cost grid point × 40
// grid points = 800,000 Beta draws) runs only when the reader releases the
// data sliders, to keep dragging responsive. The display value updates the
// label live.
// =============================================================================

const COST_N = 40;

interface MCResults {
  costGrid: number[];
  pSwitch: number[];
  cStar: number;
  meanA: number;
  meanB: number;
}

function monteCarloDecisionCurve(
  alphaA: number,
  betaA: number,
  alphaB: number,
  betaB: number,
  cMax: number,
  M: number,
  seed: number,
): MCResults {
  const rng = mulberry32(seed);
  // Pre-sample all (theta_A, theta_B) once for consistency across cost grid.
  const tA = new Float64Array(M);
  const tB = new Float64Array(M);
  for (let i = 0; i < M; i++) {
    tA[i] = betaSample(alphaA, betaA, rng);
    tB[i] = betaSample(alphaB, betaB, rng);
  }
  const costGrid: number[] = [];
  const pSwitch: number[] = [];
  for (let g = 0; g < COST_N; g++) {
    const c = (g * cMax) / (COST_N - 1);
    let count = 0;
    for (let i = 0; i < M; i++) if (tB[i] - c > tA[i]) count++;
    costGrid.push(c);
    pSwitch.push(count / M);
  }
  const meanA = betaMean(alphaA, betaA);
  const meanB = betaMean(alphaB, betaB);
  return { costGrid, pSwitch, cStar: meanB - meanA, meanA, meanB };
}

export default function DecisionBoundaryExplorer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [nA, setNA] = useState(1000);
  const [sA, setSA] = useState(47);
  const [nB, setNB] = useState(1000);
  const [sB, setSB] = useState(64);
  const [committed, setCommitted] = useState({ nA: 1000, sA: 47, nB: 1000, sB: 64 });

  const W = Math.max(360, Math.min(width || 720, 800));
  const H = 360;
  const margin = { top: 28, right: 14, bottom: 36, left: 50 };
  const colW = (W - margin.left - margin.right) / 2 - 18;

  const postA = useMemo(
    () => betaBinomialUpdate(1, 1, committed.nA, committed.sA),
    [committed],
  );
  const postB = useMemo(
    () => betaBinomialUpdate(1, 1, committed.nB, committed.sB),
    [committed],
  );
  const cMax = useMemo(() => 0.06, []);
  const mc = useMemo(
    () => monteCarloDecisionCurve(postA.alpha, postA.beta, postB.alpha, postB.beta, cMax, 20000, 42),
    [postA, postB, cMax],
  );

  const ref = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      svg.attr('viewBox', `0 0 ${W} ${H}`).attr('width', W).attr('height', H);
      const innerH = H - margin.top - margin.bottom;

      // ---------- Left panel: expected utility lines ----------
      const gL = svg.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);
      const xL = d3.scaleLinear().domain([0, cMax]).range([0, colW]);
      const yL = d3
        .scaleLinear()
        .domain([Math.min(mc.meanA, mc.meanB - cMax) - 0.005, Math.max(mc.meanA, mc.meanB) + 0.005])
        .range([innerH, 0]);
      // "stay" — constant horizontal line at meanA
      gL.append('line')
        .attr('x1', 0)
        .attr('x2', colW)
        .attr('y1', yL(mc.meanA))
        .attr('y2', yL(mc.meanA))
        .attr('stroke', COLOR_VARIANT_A)
        .attr('stroke-width', 2);
      // "switch" — meanB - c
      gL.append('line')
        .attr('x1', xL(0))
        .attr('x2', xL(cMax))
        .attr('y1', yL(mc.meanB))
        .attr('y2', yL(mc.meanB - cMax))
        .attr('stroke', COLOR_VARIANT_B)
        .attr('stroke-width', 2);
      // c* marker
      gL.append('line')
        .attr('x1', xL(mc.cStar))
        .attr('x2', xL(mc.cStar))
        .attr('y1', 0)
        .attr('y2', innerH)
        .attr('stroke', '#000')
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '4 3');
      gL.append('text')
        .attr('x', xL(mc.cStar))
        .attr('y', -6)
        .attr('text-anchor', 'middle')
        .attr('font-size', 10)
        .style('fill', 'var(--color-text)')
        .text(`c* ≈ ${mc.cStar.toFixed(4)}`);
      // Shaded regions
      gL.append('rect')
        .attr('x', xL(0))
        .attr('y', 0)
        .attr('width', xL(mc.cStar))
        .attr('height', innerH)
        .attr('fill', COLOR_VARIANT_B)
        .attr('opacity', 0.06);
      gL.append('rect')
        .attr('x', xL(mc.cStar))
        .attr('y', 0)
        .attr('width', xL(cMax) - xL(mc.cStar))
        .attr('height', innerH)
        .attr('fill', COLOR_VARIANT_A)
        .attr('opacity', 0.06);
      gL.append('g').attr('transform', `translate(0, ${innerH})`).call(d3.axisBottom(xL).ticks(5));
      gL.append('g').call(d3.axisLeft(yL).ticks(5));
      gL.append('text')
        .attr('x', colW / 2)
        .attr('y', innerH + 28)
        .attr('text-anchor', 'middle')
        .attr('font-size', 11)
        .style('fill', 'var(--color-text)')
        .text('switching cost c');
      gL.append('text')
        .attr('x', 4)
        .attr('y', yL(mc.meanA) - 6)
        .attr('font-size', 9)
        .style('fill', COLOR_VARIANT_A)
        .text('E[U(stay) | y] = E[θ_A|y]');
      gL.append('text')
        .attr('x', xL(cMax * 0.6))
        .attr('y', yL(mc.meanB - cMax * 0.6) - 6)
        .attr('font-size', 9)
        .style('fill', COLOR_VARIANT_B)
        .text('E[U(switch) | y] = E[θ_B|y] - c');

      // ---------- Right panel: P(switch correct | c) ----------
      const gR = svg.append('g').attr('transform', `translate(${margin.left + colW + 36}, ${margin.top})`);
      const xR = d3.scaleLinear().domain([0, cMax]).range([0, colW]);
      const yR = d3.scaleLinear().domain([0, 1]).range([innerH, 0]);
      gR.append('path')
        .attr(
          'd',
          d3
            .line<number>()
            .x((_, i) => xR(mc.costGrid[i]))
            .y((p) => yR(p))
            .curve(d3.curveMonotoneX)(mc.pSwitch) ?? '',
        )
        .attr('fill', 'none')
        .attr('stroke', COLOR_BAND)
        .attr('stroke-width', 2);
      gR.append('line')
        .attr('x1', 0)
        .attr('x2', colW)
        .attr('y1', yR(0.5))
        .attr('y2', yR(0.5))
        .attr('stroke', 'var(--color-text)')
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '3 3');
      gR.append('line')
        .attr('x1', xR(mc.cStar))
        .attr('x2', xR(mc.cStar))
        .attr('y1', 0)
        .attr('y2', innerH)
        .attr('stroke', '#000')
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '4 3');
      gR.append('g').attr('transform', `translate(0, ${innerH})`).call(d3.axisBottom(xR).ticks(5));
      gR.append('g').call(d3.axisLeft(yR).ticks(5).tickFormat(d3.format('.0%')));
      gR.append('text')
        .attr('x', colW / 2)
        .attr('y', innerH + 28)
        .attr('text-anchor', 'middle')
        .attr('font-size', 11)
        .style('fill', 'var(--color-text)')
        .text('switching cost c');
      gR.append('text')
        .attr('x', colW / 2)
        .attr('y', -10)
        .attr('text-anchor', 'middle')
        .attr('font-size', 11)
        .attr('font-weight', 600)
        .style('fill', 'var(--color-text)')
        .text('P(switch is correct | data, c)');
    },
    [mc, W, H, cMax],
  );

  const commit = () => setCommitted({ nA, sA, nB, sB });

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <svg ref={ref} role="img" aria-label="Decision boundary explorer" />
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(2, minmax(180px, 1fr))',
          gap: '0.75rem',
          marginTop: '0.5rem',
          fontSize: '0.85rem',
        }}
      >
        <label>
          n_A: <strong>{nA}</strong>, s_A: <strong>{sA}</strong>
          <input
            type="range"
            min={50}
            max={5000}
            step={50}
            value={nA}
            onChange={(e) => {
              const v = Number(e.target.value);
              setNA(v);
              if (sA > v) setSA(v);
            }}
            onMouseUp={commit}
            onTouchEnd={commit}
            onKeyUp={commit}
            aria-label="n_A"
            style={{ width: '100%' }}
          />
          <input
            type="range"
            min={0}
            max={nA}
            step={1}
            value={sA}
            onChange={(e) => setSA(Number(e.target.value))}
            onMouseUp={commit}
            onTouchEnd={commit}
            onKeyUp={commit}
            aria-label="s_A"
            style={{ width: '100%' }}
          />
        </label>
        <label>
          n_B: <strong>{nB}</strong>, s_B: <strong>{sB}</strong>
          <input
            type="range"
            min={50}
            max={5000}
            step={50}
            value={nB}
            onChange={(e) => {
              const v = Number(e.target.value);
              setNB(v);
              if (sB > v) setSB(v);
            }}
            onMouseUp={commit}
            onTouchEnd={commit}
            onKeyUp={commit}
            aria-label="n_B"
            style={{ width: '100%' }}
          />
          <input
            type="range"
            min={0}
            max={nB}
            step={1}
            value={sB}
            onChange={(e) => setSB(Number(e.target.value))}
            onMouseUp={commit}
            onTouchEnd={commit}
            onKeyUp={commit}
            aria-label="s_B"
            style={{ width: '100%' }}
          />
        </label>
      </div>
      <p style={{ opacity: 0.7, marginTop: '0.5rem', fontSize: '0.85rem' }}>
        E[θ_A|y] ≈ {mc.meanA.toFixed(4)}, E[θ_B|y] ≈ {mc.meanB.toFixed(4)}, c* ≈ {mc.cStar.toFixed(4)}.
        Monte Carlo recomputes when you release the slider.
      </p>
    </div>
  );
}
