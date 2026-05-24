import { useState, useMemo } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';

// Scalar source X ~ N(0, 1).
// RD with squared-error distortion: R(D) = 0.5 ln(1/D), D ∈ (0, 1].
// Gaussian IB at correlation ρ: D_pred(R) = I(X;Y) − I(T;Y), where
//   I(X;Y) = −0.5 ln(1 − ρ²),
//   I(T;Y) = −0.5 ln(1 − ρ²(1 − e^{−2R})).
function rdDistortion(R: number): number {
  return Math.exp(-2 * R);
}
function ibDistortion(R: number, rho: number): number {
  const rho2 = rho * rho;
  // I(X;Y) − I(T;Y) in nats
  const inner = 1 - rho2 * (1 - Math.exp(-2 * R));
  return 0.5 * Math.log(inner / (1 - rho2));
}

export default function RateDistortionVsIB() {
  const [rho, setRho] = useState<number>(0.9);
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();

  const W = Math.max(containerWidth || 760, 320);
  const isMobile = W < 720;
  const panelW = isMobile ? W : Math.min((W - 24) / 2, 460);
  const panelH = Math.round(panelW * 0.78);

  // R grid for both panels — go up to where D_pred saturates close to 0
  const Rs = useMemo(() => d3.range(0.02, 3.0, 0.02), []);
  const I_XY = -0.5 * Math.log(1 - rho * rho);

  const referenceRates = [0.2, 0.5, 1.0, 2.0];

  // Panel (a): R(D) — Gaussian rate-distortion (axes: D, R)
  const rdRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const margin = { top: 28, right: 18, bottom: 44, left: 56 };
      const innerW = panelW - margin.left - margin.right;
      const innerH = panelH - margin.top - margin.bottom;
      if (innerW <= 0 || innerH <= 0) return;
      svg.attr('viewBox', `0 0 ${panelW} ${panelH}`).attr('width', '100%').attr('height', panelH);

      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const xScale = d3.scaleLinear().domain([0, 1]).range([0, innerW]);
      const yScale = d3.scaleLinear().domain([0, 3]).range([innerH, 0]);

      const xAxis = d3.axisBottom(xScale).ticks(6);
      const yAxis = d3.axisLeft(yScale).ticks(5);
      const xg = g.append('g').attr('transform', `translate(0,${innerH})`).call(xAxis);
      const yg = g.append('g').call(yAxis);
      xg.selectAll('text').style('fill', 'var(--color-text)').attr('font-size', 10);
      yg.selectAll('text').style('fill', 'var(--color-text)').attr('font-size', 10);
      xg.selectAll('line, path').style('stroke', 'var(--color-border)');
      yg.selectAll('line, path').style('stroke', 'var(--color-border)');

      g.append('text')
        .attr('x', innerW / 2).attr('y', innerH + 32).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)').attr('font-size', 11)
        .text('D — squared-error distortion');
      g.append('text')
        .attr('transform', `translate(-40,${innerH / 2}) rotate(-90)`).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)').attr('font-size', 11)
        .text('R = I(X; X̂) (nats)');
      g.append('text')
        .attr('x', innerW / 2).attr('y', -10).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)').attr('font-size', 11)
        .text('(a) Gaussian R(D) — scalar source, squared-error');

      // R(D) curve: D from rdDistortion(R) for R ∈ [Rs]
      const rdPoints = Rs.map((R) => ({ D: rdDistortion(R), R }));
      const line = d3.line<{ D: number; R: number }>()
        .x((d) => xScale(d.D)).y((d) => yScale(d.R));
      g.append('path')
        .attr('d', line(rdPoints) ?? '')
        .style('fill', 'none').style('stroke', 'var(--color-accent)').attr('stroke-width', 2);

      // Reference markers at the 4 brief-table rates
      referenceRates.forEach((R) => {
        const D = rdDistortion(R);
        g.append('circle')
          .attr('cx', xScale(D)).attr('cy', yScale(R))
          .attr('r', 4.5)
          .style('fill', '#D97706')
          .attr('stroke', 'var(--color-text)').attr('stroke-width', 1);
        g.append('text')
          .attr('x', xScale(D) + 8).attr('y', yScale(R) + 3)
          .style('fill', 'var(--color-text)').attr('font-size', 10)
          .text(`R = ${R.toFixed(2)}, D = ${D.toFixed(4)}`);
      });

      g.append('text')
        .attr('x', innerW / 2).attr('y', innerH - 6)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)').attr('font-size', 10)
        .text('Lagrangian: ℱ_s^RD = I(X; X̂) + s · E[d(X, X̂)]');
    },
    [panelW, panelH, Rs],
  );

  // Panel (b): R(D_pred) — Gaussian IB
  const ibRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const margin = { top: 28, right: 18, bottom: 44, left: 56 };
      const innerW = panelW - margin.left - margin.right;
      const innerH = panelH - margin.top - margin.bottom;
      if (innerW <= 0 || innerH <= 0) return;
      svg.attr('viewBox', `0 0 ${panelW} ${panelH}`).attr('width', '100%').attr('height', panelH);

      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      // D_pred domain: [0, I(X;Y)]
      const xScale = d3.scaleLinear().domain([0, I_XY * 1.05]).range([0, innerW]);
      const yScale = d3.scaleLinear().domain([0, 3]).range([innerH, 0]);

      const xAxis = d3.axisBottom(xScale).ticks(5);
      const yAxis = d3.axisLeft(yScale).ticks(5);
      const xg = g.append('g').attr('transform', `translate(0,${innerH})`).call(xAxis);
      const yg = g.append('g').call(yAxis);
      xg.selectAll('text').style('fill', 'var(--color-text)').attr('font-size', 10);
      yg.selectAll('text').style('fill', 'var(--color-text)').attr('font-size', 10);
      xg.selectAll('line, path').style('stroke', 'var(--color-border)');
      yg.selectAll('line, path').style('stroke', 'var(--color-border)');

      g.append('text')
        .attr('x', innerW / 2).attr('y', innerH + 32).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)').attr('font-size', 11)
        .text('D_pred = I(X;Y) − I(T;Y) (nats)');
      g.append('text')
        .attr('transform', `translate(-40,${innerH / 2}) rotate(-90)`).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)').attr('font-size', 11)
        .text('R = I(X;T) (nats)');
      g.append('text')
        .attr('x', innerW / 2).attr('y', -10).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)').attr('font-size', 11)
        .text(`(b) Gaussian IB R(D_pred) at ρ = ${rho.toFixed(2)}`);

      // IB curve
      const ibPoints = Rs.map((R) => ({ D: Math.max(0, ibDistortion(R, rho)), R }));
      const line = d3.line<{ D: number; R: number }>()
        .x((d) => xScale(d.D)).y((d) => yScale(d.R));
      g.append('path')
        .attr('d', line(ibPoints) ?? '')
        .style('fill', 'none').style('stroke', 'var(--color-accent)').attr('stroke-width', 2);

      // I(X;Y) saturation vertical
      g.append('line')
        .attr('x1', xScale(I_XY)).attr('x2', xScale(I_XY))
        .attr('y1', yScale(0)).attr('y2', yScale(3))
        .style('stroke', 'var(--color-text-secondary)')
        .attr('stroke-dasharray', '3 4').attr('stroke-width', 1);
      g.append('text')
        .attr('x', xScale(I_XY) + 4).attr('y', 14)
        .style('fill', 'var(--color-text-secondary)').attr('font-size', 10)
        .text(`I(X;Y) = ${I_XY.toFixed(3)} nats`);

      // Reference markers
      referenceRates.forEach((R) => {
        const D = ibDistortion(R, rho);
        if (D < 0 || D > I_XY * 1.05) return;
        g.append('circle')
          .attr('cx', xScale(D)).attr('cy', yScale(R))
          .attr('r', 4.5)
          .style('fill', '#D97706')
          .attr('stroke', 'var(--color-text)').attr('stroke-width', 1);
        g.append('text')
          .attr('x', xScale(D) + 8).attr('y', yScale(R) + 3)
          .style('fill', 'var(--color-text)').attr('font-size', 10)
          .text(`R = ${R.toFixed(2)}, D_pred = ${D.toFixed(4)}`);
      });

      g.append('text')
        .attr('x', innerW / 2).attr('y', innerH - 6)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)').attr('font-size', 10)
        .text('Lagrangian: ℱ_β = I(X;T) − β · I(T;Y)');
    },
    [rho, panelW, panelH, Rs, I_XY],
  );

  return (
    <div ref={containerRef} style={{ padding: '1rem 0' }}>
      <div style={{ marginBottom: '0.75rem' }}>
        <label
          htmlFor="rd-vs-ib-rho"
          style={{ fontSize: '0.85rem', color: 'var(--color-text-secondary)', display: 'block' }}
        >
          Correlation ρ ={' '}
          <strong style={{ color: 'var(--color-text)' }}>{rho.toFixed(2)}</strong>{'   '}
          <span style={{ color: 'var(--color-text-secondary)' }}>
            (I(X;Y) = {I_XY.toFixed(3)} nats — the IB saturation rate)
          </span>
        </label>
        <input
          id="rd-vs-ib-rho"
          type="range"
          min={0.1}
          max={0.99}
          step={0.01}
          value={rho}
          onChange={(e) => setRho(parseFloat(e.target.value))}
          aria-label="Correlation rho for the IB panel"
          style={{ width: '100%' }}
        />
      </div>
      <div
        style={{
          display: 'flex',
          flexDirection: isMobile ? 'column' : 'row',
          gap: '1rem',
          alignItems: 'flex-start',
        }}
      >
        <div style={{ flex: 1, minWidth: 0 }}>
          <svg ref={rdRef} role="img" aria-label="Gaussian rate-distortion curve" />
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <svg ref={ibRef} role="img" aria-label="Gaussian IB curve with predictive distortion" />
        </div>
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.75rem' }}>
        Both curves are convex decreasing in (D, R) — structurally identical shape. The substantive difference is the distortion's <em>provenance</em>: RD takes <code>d(X, X̂)</code> as input, fixed before the algorithm runs; IB derives <code>D_pred = D_KL(p(y|x) ‖ p(y|t))</code> from the encoder iterate itself. The side-by-side at ρ = 0.9 reproduces the brief's §11.1 table: at R = 0.20, RD has D ≈ 0.670 vs IB D_pred ≈ 0.675; at R = 2.00, RD ≈ 0.018 vs IB ≈ 0.038.
      </p>
    </div>
  );
}
