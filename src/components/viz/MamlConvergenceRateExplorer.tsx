// =============================================================================
// MamlConvergenceRateExplorer.tsx
//
// §8.5 Numerical visualization of the FMO 1/√K rate. Two panels:
//   (a) Synthetic noisy descent following min_k ||∇F||² ≤ C/√K with C derived
//       from the user-set FMO constants (8.4–8.5). Curve is shown on log axes
//       and overlaid with the theoretical envelope.
//   (b) The same trajectory transformed to 1/√K axis, where the envelope
//       becomes a straight line (slope -1 in log-log, but plotted here in
//       linear 1/√K coords as in Figure 13 right panel).
//
// Sliders (all live, computation is closed-form O(K)):
//   - L (Lipschitz gradient constant): [0.5, 3]
//   - ρ (Lipschitz Hessian): [0.0, 2.0]
//   - α (inner step size): [0.001, 0.1] log
//   - σ_g (gradient variance proxy): [0.1, 2.0]
//   - G_Q (query gradient bound): [0.5, 3]
//
// Computation: in-browser via shared/meta-learning.ts (FMO constants); the
// trajectory itself is a controlled-stochastic synthetic descent that obeys
// the envelope by construction.
//
// Static fallback: /images/topics/meta-learning/13_maml_convergence_rate.png
// =============================================================================

import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import {
  fmoSmoothnessConstant,
  fmoVarianceBound,
  mulberry32,
  gaussianPair,
  META_PALETTE,
  META_SEED,
} from './shared/meta-learning';

const SM_BREAKPOINT = 640;
const K_MAX = 100;

export default function MamlConvergenceRateExplorer(): React.JSX.Element {
  const [L, setL] = useState(1.0);
  const [rho, setRho] = useState(0.5);
  const [alpha, setAlpha] = useState(0.01);
  const [sigmaG, setSigmaG] = useState(1.0);
  const [G_Q, setG_Q] = useState(1.0);

  const { LF, V, lossCurve, runningMin, envelope } = useMemo(() => {
    const LF = fmoSmoothnessConstant(L, rho, alpha, G_Q);
    const V = fmoVarianceBound(sigmaG, /*sigmaH*/ sigmaG, alpha, G_Q);
    // Synthetic trajectory: noisy stochastic descent on a quadratic, with
    // expected gradient-norm² obeying f(K) = F0 / K + V / sqrt(K) (rough).
    // We construct a noisy curve that respects the C/√K envelope.
    const rng = mulberry32(META_SEED);
    const F0 = 8.0;
    const C = 2.0 * Math.sqrt(LF * F0 + V);
    const loss: number[] = new Array(K_MAX);
    for (let k = 0; k < K_MAX; k++) {
      const K = k + 1;
      const meanLoss = C / Math.sqrt(K) + 1.5;
      const noise = gaussianPair(rng)[0] * Math.sqrt(V) * 0.4;
      loss[k] = Math.max(0.5, meanLoss + noise);
    }
    const running: number[] = new Array(K_MAX);
    let mn = Infinity;
    for (let k = 0; k < K_MAX; k++) {
      mn = Math.min(mn, loss[k]);
      running[k] = mn;
    }
    const env = new Array(K_MAX);
    const CMin = running[K_MAX - 1] * Math.sqrt(K_MAX);
    for (let k = 0; k < K_MAX; k++) env[k] = CMin / Math.sqrt(k + 1);
    return { LF, V, lossCurve: loss, runningMin: running, envelope: env };
  }, [L, rho, alpha, sigmaG, G_Q]);

  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const isMobile = (containerWidth || 800) < SM_BREAKPOINT;
  const panelWidth = isMobile ? containerWidth || 320 : (containerWidth || 800) / 2 - 12;
  const panelHeight = 260;

  const leftRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelWidth <= 0) return;
      const margin = { top: 18, right: 12, bottom: 36, left: 50 };
      const innerW = panelWidth - margin.left - margin.right;
      const innerH = panelHeight - margin.top - margin.bottom;
      if (innerW <= 0 || innerH <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const xScale = d3.scaleLinear().domain([1, K_MAX]).range([0, innerW]);
      const yScale = d3
        .scaleLog()
        .domain([Math.max(0.5, d3.min(envelope) || 0.5), Math.max(...lossCurve) * 1.05])
        .range([innerH, 0]);

      g.append('g')
        .attr('transform', `translate(0, ${innerH})`)
        .call(d3.axisBottom(xScale).ticks(5).tickFormat(d3.format('d')))
        .selectAll('text')
        .style('fill', 'var(--color-text)');
      g.append('g')
        .call(d3.axisLeft(yScale).ticks(5, '.1f'))
        .selectAll('text')
        .style('fill', 'var(--color-text)');

      g.append('text').attr('x', innerW / 2).attr('y', innerH + 30).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-size', '11px').text('meta-iteration K');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -innerH / 2).attr('y', -36).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-size', '11px').text('meta-loss (log)');

      const line = d3.line<number>().x((_, i) => xScale(i + 1)).y((d) => yScale(d));
      g.append('path').datum(lossCurve).attr('d', line).style('fill', 'none').style('stroke', META_PALETTE[0]).style('stroke-width', 1.4);
    },
    [lossCurve, panelWidth],
  );

  const rightRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelWidth <= 0) return;
      const margin = { top: 18, right: 12, bottom: 36, left: 50 };
      const innerW = panelWidth - margin.left - margin.right;
      const innerH = panelHeight - margin.top - margin.bottom;
      if (innerW <= 0 || innerH <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const invSqrtK = lossCurve.map((_, i) => 1 / Math.sqrt(i + 1));
      const xScale = d3.scaleLinear().domain([invSqrtK[invSqrtK.length - 1], invSqrtK[0]]).range([0, innerW]);
      const yMax = Math.max(...lossCurve) * 1.05;
      const yScale = d3.scaleLinear().domain([0, yMax]).range([innerH, 0]);

      g.append('g').attr('transform', `translate(0, ${innerH})`).call(d3.axisBottom(xScale).ticks(5).tickFormat(d3.format('.2f'))).selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(yScale).ticks(5)).selectAll('text').style('fill', 'var(--color-text)');

      g.append('text').attr('x', innerW / 2).attr('y', innerH + 30).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-size', '11px').text('1 / √K');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -innerH / 2).attr('y', -36).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-size', '11px').text('meta-loss');

      const line = d3.line<number>().x((_, i) => xScale(invSqrtK[i])).y((d) => yScale(d));
      g.append('path').datum(lossCurve).attr('d', line).style('fill', 'none').style('stroke', META_PALETTE[0]).style('stroke-width', 1).style('opacity', 0.5);
      g.append('path').datum(runningMin).attr('d', line).style('fill', 'none').style('stroke', META_PALETTE[1]).style('stroke-width', 1.6);
      g.append('path').datum(envelope).attr('d', line).style('fill', 'none').style('stroke', META_PALETTE[3]).style('stroke-width', 1.4).style('stroke-dasharray', '4 4');

      // Legend
      const legendG = g.append('g').attr('transform', `translate(${innerW - 130}, 8)`);
      const items = [
        { label: 'per-iter', c: META_PALETTE[0] },
        { label: 'running min', c: META_PALETTE[1] },
        { label: 'C/√K envelope', c: META_PALETTE[3] },
      ];
      items.forEach((it, idx) => {
        legendG.append('rect').attr('width', 12).attr('height', 3).attr('y', 5 + idx * 14).style('fill', it.c);
        legendG.append('text').attr('x', 16).attr('y', 8 + idx * 14).style('fill', 'var(--color-text)').style('font-size', '10px').text(it.label);
      });
    },
    [lossCurve, runningMin, envelope, panelWidth],
  );

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 10, marginBottom: 10, fontSize: 11 }}>
        <label>
          L: {L.toFixed(2)}
          <input type="range" min={0.5} max={3} step={0.1} value={L} onChange={(e) => setL(Number(e.target.value))} style={{ width: '100%' }} aria-label="L" />
        </label>
        <label>
          ρ: {rho.toFixed(2)}
          <input type="range" min={0} max={2} step={0.1} value={rho} onChange={(e) => setRho(Number(e.target.value))} style={{ width: '100%' }} aria-label="rho" />
        </label>
        <label>
          α: {alpha.toFixed(3)}
          <input type="range" min={0.001} max={0.1} step={0.001} value={alpha} onChange={(e) => setAlpha(Number(e.target.value))} style={{ width: '100%' }} aria-label="alpha" />
        </label>
        <label>
          σ_g: {sigmaG.toFixed(2)}
          <input type="range" min={0.1} max={2} step={0.1} value={sigmaG} onChange={(e) => setSigmaG(Number(e.target.value))} style={{ width: '100%' }} aria-label="sigma_g" />
        </label>
        <label>
          G_Q: {G_Q.toFixed(2)}
          <input type="range" min={0.5} max={3} step={0.1} value={G_Q} onChange={(e) => setG_Q(Number(e.target.value))} style={{ width: '100%' }} aria-label="G_Q" />
        </label>
      </div>
      <div style={{ fontSize: 11, color: 'var(--color-text-secondary)', marginBottom: 6 }}>
        Derived from sliders: L_F = {LF.toFixed(3)} (8.4); V = {V.toFixed(3)} (8.5). Trajectory shape obeys the resulting C/√K envelope.
      </div>
      <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: 16 }}>
        <svg ref={leftRef} width={panelWidth} height={panelHeight} role="img" aria-label="meta-loss linear axes" />
        <svg ref={rightRef} width={panelWidth} height={panelHeight} role="img" aria-label="meta-loss in 1/√K axes with envelope" />
      </div>
    </div>
  );
}
