import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  RUNNING_EXAMPLE_A,
  RUNNING_EXAMPLE_THETA0,
  confidenceEllipse2D,
  laplaceTypeMHAffine,
  mulberry32,
  sampleSensorObservations,
  twoStepAffineGMM,
  type Mat,
} from './shared/generalized-method-of-moments';

// -----------------------------------------------------------------------------
// Viz 13.1 — Laplace-type estimator (Chernozhukov-Hong 2003).
//
// Treat the efficient-GMM criterion as a quasi-log-likelihood:
//   π*(θ) ∝ exp(-J_n(θ, Ω̂⁻¹)/2) · π(θ).
// 2D random-walk Metropolis on the running example. The MH cloud should
// match the frequentist sandwich 95% confidence ellipse asymptotically.
//
// Two panels:
//   Panel A — 2D MH samples scatter, overlaid with the frequentist sandwich
//     95% ellipse. Should agree at large n.
//   Panel B — Marginal posterior densities for θ_1 and θ_2, with the
//     Gaussian-asymptotic-equivalent overlaid.
//
// Compute: MH iter ≈ 4000 + 1000 warmup × O(L·k) ≈ 200k flops. Fast.
// -----------------------------------------------------------------------------

const N_OPTIONS = [50, 100, 200, 500];
const N_SAMPLES = 3000;
const WARMUP = 600;
const PANEL_H = 320;
const MARGIN_H = 200;
const MARGIN = { top: 22, right: 18, bottom: 42, left: 60 };

const COLOR_MH = 'var(--color-accent)';
const COLOR_FREQ = '#d09a3e';

function buildOmegaSandwichEllipse(twoStep: ReturnType<typeof twoStepAffineGMM>, n: number) {
  // 95% ellipse from frequentist sandwich variance at the two-step estimate.
  // V_freq is asymptotic: divide by n for the sampling-distribution scale.
  const V_scaled: Mat = twoStep.V2.map((r) => r.map((v) => v / n));
  return confidenceEllipse2D(V_scaled, 0.05);
}

export default function LaplaceTypeEstimatorComparison() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [nIdx, setNIdx] = useState(2); // n = 200
  const [nIdxDisplay, setNIdxDisplay] = useState(2);
  const [seed, setSeed] = useState(7777);

  const n = N_OPTIONS[nIdx];

  const state = useMemo(() => {
    const A = RUNNING_EXAMPLE_A;
    const theta0 = RUNNING_EXAMPLE_THETA0;
    const sigma = [1, 1, 1, 1];
    const rng = mulberry32(seed);
    const X = sampleSensorObservations(rng, theta0, A, n, sigma);
    const twoStep = twoStepAffineGMM(X, A);
    const rngMH = mulberry32(seed + 1);
    const lte = laplaceTypeMHAffine(X, A, rngMH, N_SAMPLES, WARMUP, 0.15);
    const freqEllipse = buildOmegaSandwichEllipse(twoStep, n);
    return { twoStep, lte, freqEllipse };
  }, [n, seed]);

  const w = Math.max(280, Math.min(containerWidth || 720, 820));

  const scatterRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const innerW = w - MARGIN.left - MARGIN.right;
      const innerH = PANEL_H - MARGIN.top - MARGIN.bottom;
      if (innerW <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      // Choose window centered on θ̂^(2) with span = max(MH spread, freq ellipse) × 1.5.
      const samples = state.lte.samples;
      const xs = samples.map((s) => s[0]);
      const ys = samples.map((s) => s[1]);
      const xExt = d3.extent(xs) as [number, number];
      const yExt = d3.extent(ys) as [number, number];
      const span = Math.max(xExt[1] - xExt[0], yExt[1] - yExt[0]) * 0.6;
      const center = state.twoStep.theta2;
      const xScale = d3.scaleLinear().domain([center[0] - span, center[0] + span]).range([0, innerW]);
      const yScale = d3.scaleLinear().domain([center[1] - span, center[1] + span]).range([innerH, 0]);

      g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xScale).ticks(5).tickFormat(d3.format('.3~f'))).selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(yScale).ticks(5).tickFormat(d3.format('.3~f'))).selectAll('text').style('fill', 'var(--color-text)');

      // MH samples (thinned for visual clarity).
      const thin = Math.max(1, Math.floor(samples.length / 600));
      for (let i = 0; i < samples.length; i += thin) {
        g.append('circle').attr('cx', xScale(samples[i][0])).attr('cy', yScale(samples[i][1])).attr('r', 1.6).style('fill', COLOR_MH).attr('opacity', 0.4);
      }

      // Frequentist sandwich 95% ellipse.
      const cx = xScale(state.twoStep.theta2[0]);
      const cy = yScale(state.twoStep.theta2[1]);
      const rx = Math.abs(xScale(state.twoStep.theta2[0] + state.freqEllipse.rx) - cx);
      const ry = Math.abs(yScale(state.twoStep.theta2[1] + state.freqEllipse.ry) - cy);
      const angleDeg = (-state.freqEllipse.angle * 180) / Math.PI;
      g.append('ellipse').attr('cx', cx).attr('cy', cy).attr('rx', rx).attr('ry', ry).attr('transform', `rotate(${angleDeg}, ${cx}, ${cy})`).attr('fill', 'none').attr('stroke', COLOR_FREQ).attr('stroke-width', 2);

      // θ_0 cross.
      const x0 = xScale(RUNNING_EXAMPLE_THETA0[0]);
      const y0 = yScale(RUNNING_EXAMPLE_THETA0[1]);
      g.append('line').attr('x1', x0 - 6).attr('x2', x0 + 6).attr('y1', y0).attr('y2', y0).style('stroke', 'var(--color-text)').style('stroke-width', 1.4);
      g.append('line').attr('x1', x0).attr('x2', x0).attr('y1', y0 - 6).attr('y2', y0 + 6).style('stroke', 'var(--color-text)').style('stroke-width', 1.4);
      g.append('text').attr('x', x0 + 8).attr('y', y0 - 4).style('font-family', 'var(--font-mono)').style('font-size', '10px').style('fill', 'var(--color-text)').text('θ₀');

      // Legend.
      const legend = g.append('g').attr('transform', `translate(${innerW - 240}, 4)`);
      legend.append('circle').attr('cx', 8).attr('cy', 8).attr('r', 4).style('fill', COLOR_MH).attr('opacity', 0.4);
      legend.append('text').attr('x', 18).attr('y', 12).style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', COLOR_MH).text('LTE Metropolis cloud');
      legend.append('rect').attr('x', 0).attr('y', 22).attr('width', 14).attr('height', 14).attr('fill', 'none').style('stroke', COLOR_FREQ).style('stroke-width', 2);
      legend.append('text').attr('x', 18).attr('y', 34).style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', COLOR_FREQ).text('Frequentist 95% sandwich ellipse');

      g.append('text').attr('x', innerW / 2).attr('y', innerH + 32).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text('θ_1');
      g.append('text').attr('x', -innerH / 2).attr('y', -44).attr('text-anchor', 'middle').attr('transform', 'rotate(-90)').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text('θ_2');
    },
    [state, w],
  );

  const marginalsRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const innerW = w - MARGIN.left - MARGIN.right;
      const innerH = MARGIN_H - MARGIN.top - MARGIN.bottom;
      if (innerW <= 0) return;
      const half = innerW / 2 - 10;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const samples = state.lte.samples;
      // For θ_1 and θ_2 separately:
      for (let dim = 0; dim < 2; dim++) {
        const xValues = samples.map((s) => s[dim]);
        const ext = d3.extent(xValues) as [number, number];
        const pad = (ext[1] - ext[0]) * 0.2;
        const xScale = d3.scaleLinear().domain([ext[0] - pad, ext[1] + pad]).range([0, half]);
        const binner = d3.bin().domain(xScale.domain() as [number, number]).thresholds(28);
        const bins = binner(xValues);
        const yMax = d3.max(bins, (b) => b.length) ?? 0;
        const yScale = d3.scaleLinear().domain([0, yMax * 1.2]).range([innerH, 0]);
        const offsetX = dim === 0 ? 0 : half + 20;
        const panel = g.append('g').attr('transform', `translate(${offsetX}, 0)`);

        panel.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xScale).ticks(4).tickFormat(d3.format('.3~f'))).selectAll('text').style('fill', 'var(--color-text)').style('font-size', '10px');
        panel.append('g').call(d3.axisLeft(yScale).ticks(4)).selectAll('text').style('fill', 'var(--color-text)').style('font-size', '10px');

        bins.forEach((b) => {
          if (b.x0 === undefined || b.x1 === undefined) return;
          panel.append('rect').attr('x', xScale(b.x0)).attr('y', yScale(b.length)).attr('width', Math.max(0, xScale(b.x1) - xScale(b.x0) - 1)).attr('height', innerH - yScale(b.length)).style('fill', COLOR_MH).style('opacity', 0.7);
        });

        // Gaussian asymptotic overlay.
        const mu = state.twoStep.theta2[dim];
        const sd = Math.sqrt(state.twoStep.V2[dim][dim] / n);
        const pdf: [number, number][] = [];
        const xMin = xScale.domain()[0];
        const xMax = xScale.domain()[1];
        for (let i = 0; i <= 100; i++) {
          const x = xMin + (i / 100) * (xMax - xMin);
          const d = (1 / (sd * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * ((x - mu) / sd) ** 2);
          // Rescale to match histogram count: bin_width * n * pdf
          const binWidth = (xMax - xMin) / 28;
          pdf.push([x, d * binWidth * samples.length]);
        }
        const lineGen = d3.line<[number, number]>().x((p) => xScale(p[0])).y((p) => yScale(p[1])).curve(d3.curveBasis);
        panel.append('path').attr('d', lineGen(pdf)).attr('fill', 'none').attr('stroke', COLOR_FREQ).attr('stroke-width', 2);

        panel.append('text').attr('x', half / 2).attr('y', innerH + 28).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-mono)').style('font-size', '11px').text(`θ_${dim + 1} marginal`);
      }
    },
    [state, w, n],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: 'var(--font-sans)' }}>
      <div style={{ display: 'flex', gap: '14px', padding: '10px 0', fontSize: '13px', flexWrap: 'wrap' }}>
        <label style={{ display: 'flex', flexDirection: 'column', minWidth: '180px' }}>
          <span>n = <strong>{N_OPTIONS[nIdxDisplay]}</strong></span>
          <input type="range" min={0} max={N_OPTIONS.length - 1} step={1} value={nIdxDisplay} aria-label="Sample size" onChange={(e) => setNIdxDisplay(Number(e.target.value))} onMouseUp={(e) => setNIdx(Number((e.target as HTMLInputElement).value))} onTouchEnd={(e) => setNIdx(Number((e.target as HTMLInputElement).value))} onKeyUp={(e) => setNIdx(Number((e.target as HTMLInputElement).value))} />
        </label>
        <button type="button" onClick={() => setSeed((s) => s + 1)} style={{ padding: '6px 10px', fontFamily: 'var(--font-mono)', fontSize: '12px', border: '1px solid var(--color-border)', background: 'var(--color-bg)', color: 'var(--color-text)', borderRadius: '4px', cursor: 'pointer', alignSelf: 'flex-end' }}>new seed</button>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--color-text-secondary)', padding: '6px 10px', border: '1px solid var(--color-border)', borderRadius: '4px', alignSelf: 'flex-end' }}>
          MH accept rate = <strong style={{ color: 'var(--color-text)' }}>{(state.lte.acceptanceRate * 100).toFixed(1)}%</strong> · θ̄ = ({state.lte.thetaBar[0].toFixed(3)}, {state.lte.thetaBar[1].toFixed(3)})
        </div>
      </div>
      <svg ref={scatterRef} width={w} height={PANEL_H} style={{ maxWidth: '100%', display: 'block' }} aria-label="LTE Metropolis cloud and frequentist sandwich ellipse" />
      <svg ref={marginalsRef} width={w} height={MARGIN_H} style={{ maxWidth: '100%', display: 'block', marginTop: 8 }} aria-label="Marginal posterior densities for θ_1 and θ_2" />
    </div>
  );
}
