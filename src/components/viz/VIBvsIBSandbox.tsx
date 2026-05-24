import { useState, useEffect } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';

// Payload shape — keep in sync with notebooks/information-bottleneck/precompute_vib_sandbox.py
type IBPoint = { beta: number; I_XT: number; I_TY: number; L_VIB?: number };

type VIBPayload = {
  _metadata: { p_dim: number; rho_design: number[]; I_XY_nats: number; beta_c: (number | null)[]; seed: number };
  vib_trajectory: IBPoint[];
  ib_at_vib_betas: IBPoint[];
  variational_gap: number[];
  ib_curve_dense: IBPoint[];
  gd_trajectory_beta5: {
    beta: number;
    n_steps: number;
    lr: number;
    I_XT: number[];
    I_TY: number[];
    peak_I_XT: number;
    final_I_XT: number;
    compression_delta: number;
  };
};

export default function VIBvsIBSandbox() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [payload, setPayload] = useState<VIBPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [betaIdx, setBetaIdx] = useState<number>(20);

  useEffect(() => {
    fetch('/sample-data/information-bottleneck/vib_sandbox.json')
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((d: VIBPayload) => setPayload(d))
      .catch((e: Error) => setError(e.message));
  }, []);

  const W = Math.max(containerWidth || 760, 320);
  const isMobile = W < 720;
  const panelW = isMobile ? W : Math.min((W - 24) / 2, 460);
  const panelH = Math.round(panelW * 0.78);

  const cur = payload?.vib_trajectory[betaIdx];
  const ibCur = payload?.ib_at_vib_betas[betaIdx];
  const gap = payload?.variational_gap[betaIdx];

  // Panel (a): IB curve + VIB trajectory in (I(X;T), I(T;Y)) plane
  const planeRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (!payload) return;
      const margin = { top: 28, right: 18, bottom: 44, left: 56 };
      const innerW = panelW - margin.left - margin.right;
      const innerH = panelH - margin.top - margin.bottom;
      if (innerW <= 0 || innerH <= 0) return;
      svg.attr('viewBox', `0 0 ${panelW} ${panelH}`).attr('width', '100%').attr('height', panelH);

      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const allIXT = [
        ...payload.ib_curve_dense.map((p) => p.I_XT),
        ...payload.vib_trajectory.map((p) => p.I_XT),
      ];
      const allITY = [
        ...payload.ib_curve_dense.map((p) => p.I_TY),
        ...payload.vib_trajectory.map((p) => p.I_TY),
      ];
      const maxIXT = Math.min(10, Math.max(...allIXT) * 1.05);
      const maxITY = Math.max(payload._metadata.I_XY_nats * 1.1, Math.max(...allITY) * 1.05);
      const xScale = d3.scaleLinear().domain([0, maxIXT]).range([0, innerW]);
      const yScale = d3.scaleLinear().domain([0, maxITY]).range([innerH, 0]);

      // DPI ceiling
      g.append('line')
        .attr('x1', xScale(0)).attr('x2', xScale(maxIXT))
        .attr('y1', yScale(payload._metadata.I_XY_nats)).attr('y2', yScale(payload._metadata.I_XY_nats))
        .style('stroke', 'var(--color-text-secondary)')
        .attr('stroke-dasharray', '4 3').attr('stroke-width', 1);
      g.append('text')
        .attr('x', xScale(maxIXT) - 6).attr('y', yScale(payload._metadata.I_XY_nats) - 4)
        .attr('text-anchor', 'end')
        .style('fill', 'var(--color-text-secondary)').attr('font-size', 10)
        .text(`I(X;Y) = ${payload._metadata.I_XY_nats.toFixed(3)} nats`);

      // Axes
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
        .text('I(X;T) (nats)');
      g.append('text')
        .attr('transform', `translate(-40,${innerH / 2}) rotate(-90)`).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)').attr('font-size', 11)
        .text('I(T;Y) (nats)');
      g.append('text')
        .attr('x', innerW / 2).attr('y', -10).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)').attr('font-size', 11)
        .text('VIB trajectory vs. true IB curve (4-D Gaussian toy)');

      // IB curve (dense)
      const lineIB = d3.line<IBPoint>().x((p) => xScale(p.I_XT)).y((p) => yScale(p.I_TY));
      g.append('path')
        .attr('d', lineIB(payload.ib_curve_dense) ?? '')
        .style('fill', 'none').style('stroke', 'var(--color-accent)').attr('stroke-width', 2);

      // VIB trajectory
      const lineVIB = d3.line<IBPoint>().x((p) => xScale(p.I_XT)).y((p) => yScale(p.I_TY));
      g.append('path')
        .attr('d', lineVIB(payload.vib_trajectory) ?? '')
        .style('fill', 'none').style('stroke', '#D97706').attr('stroke-width', 2);
      g.selectAll('.vib-dot').data(payload.vib_trajectory).enter()
        .append('circle')
        .attr('cx', (p) => xScale(p.I_XT)).attr('cy', (p) => yScale(p.I_TY))
        .attr('r', 2.2)
        .style('fill', '#D97706')
        .attr('opacity', 0.6);

      // Legend
      g.append('rect').attr('x', innerW - 200).attr('y', 4).attr('width', 196).attr('height', 38)
        .style('fill', 'var(--color-surface)').attr('opacity', 0.9)
        .style('stroke', 'var(--color-border)').attr('stroke-width', 1);
      g.append('line').attr('x1', innerW - 192).attr('x2', innerW - 174).attr('y1', 16).attr('y2', 16)
        .style('stroke', 'var(--color-accent)').attr('stroke-width', 2.5);
      g.append('text').attr('x', innerW - 170).attr('y', 20)
        .style('fill', 'var(--color-text)').attr('font-size', 10)
        .text('closed-form Gaussian IB');
      g.append('line').attr('x1', innerW - 192).attr('x2', innerW - 174).attr('y1', 32).attr('y2', 32)
        .style('stroke', '#D97706').attr('stroke-width', 2.5);
      g.append('text').attr('x', innerW - 170).attr('y', 36)
        .style('fill', 'var(--color-text)').attr('font-size', 10)
        .text('linear-Gaussian VIB');

      // Active β marker on both curves
      if (cur && ibCur) {
        g.append('line')
          .attr('x1', xScale(cur.I_XT)).attr('y1', yScale(cur.I_TY))
          .attr('x2', xScale(ibCur.I_XT)).attr('y2', yScale(ibCur.I_TY))
          .style('stroke', 'var(--color-text)').attr('stroke-dasharray', '3 3').attr('stroke-width', 1);
        g.append('circle').attr('cx', xScale(cur.I_XT)).attr('cy', yScale(cur.I_TY)).attr('r', 6)
          .style('fill', '#D97706').attr('stroke', 'var(--color-text)').attr('stroke-width', 1.4);
        g.append('circle').attr('cx', xScale(ibCur.I_XT)).attr('cy', yScale(ibCur.I_TY)).attr('r', 6)
          .style('fill', 'var(--color-accent)').attr('stroke', 'var(--color-text)').attr('stroke-width', 1.4);
        g.append('text')
          .attr('x', xScale(cur.I_XT) + 8).attr('y', yScale(cur.I_TY) + 14)
          .style('fill', 'var(--color-text)').attr('font-size', 10)
          .text(`β = ${cur.beta.toFixed(2)}`);
      }
    },
    [betaIdx, panelW, panelH, payload, cur?.I_XT, cur?.I_TY, ibCur?.I_XT, ibCur?.I_TY],
  );

  // Panel (b): predictiveness gap vs β
  const gapRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (!payload) return;
      const margin = { top: 28, right: 18, bottom: 44, left: 56 };
      const innerW = panelW - margin.left - margin.right;
      const innerH = panelH - margin.top - margin.bottom;
      if (innerW <= 0 || innerH <= 0) return;
      svg.attr('viewBox', `0 0 ${panelW} ${panelH}`).attr('width', '100%').attr('height', panelH);

      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const betas = payload.vib_trajectory.map((p) => p.beta);
      const xScale = d3.scaleLog().domain([betas[0], betas[betas.length - 1]]).range([0, innerW]);
      const maxGap = Math.max(...payload.variational_gap) * 1.1;
      const yScale = d3.scaleLinear().domain([0, maxGap]).range([innerH, 0]);

      const xAxis = d3.axisBottom(xScale).ticks(6, '.1f');
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
        .text('β (log scale)');
      g.append('text')
        .attr('transform', `translate(-40,${innerH / 2}) rotate(-90)`).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)').attr('font-size', 11)
        .text('I(T;Y)_IB − I(T;Y)_VIB (nats)');
      g.append('text')
        .attr('x', innerW / 2).attr('y', -10).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)').attr('font-size', 11)
        .text('Variational gap in predictiveness');

      // β_c verticals
      const phaseColors = ['#534AB7', '#D97706', '#DC2626'];
      payload._metadata.beta_c.slice(0, 3).forEach((bc, i) => {
        if (bc && bc >= betas[0] && bc <= betas[betas.length - 1]) {
          g.append('line')
            .attr('x1', xScale(bc)).attr('x2', xScale(bc))
            .attr('y1', yScale(0)).attr('y2', yScale(maxGap))
            .style('stroke', phaseColors[i]).attr('stroke-dasharray', '3 4')
            .attr('stroke-width', 1).attr('opacity', 0.5);
        }
      });

      const lineGap = d3.line<{ beta: number; gap: number }>()
        .x((p) => xScale(p.beta)).y((p) => yScale(p.gap));
      const gapData = betas.map((b, i) => ({ beta: b, gap: payload.variational_gap[i] }));
      g.append('path')
        .attr('d', lineGap(gapData) ?? '')
        .style('fill', 'none').style('stroke', '#D97706').attr('stroke-width', 2);
      g.selectAll('.gap-dot').data(gapData).enter()
        .append('circle')
        .attr('cx', (p) => xScale(p.beta)).attr('cy', (p) => yScale(p.gap))
        .attr('r', 2.5)
        .style('fill', '#D97706')
        .attr('opacity', 0.6);

      // Active β cursor
      if (cur && gap !== undefined) {
        g.append('line')
          .attr('x1', xScale(cur.beta)).attr('x2', xScale(cur.beta))
          .attr('y1', yScale(0)).attr('y2', yScale(maxGap))
          .style('stroke', 'var(--color-text)').attr('stroke-dasharray', '3 3').attr('stroke-width', 1.2);
        g.append('circle')
          .attr('cx', xScale(cur.beta)).attr('cy', yScale(gap)).attr('r', 6)
          .style('fill', '#D97706').attr('stroke', 'var(--color-text)').attr('stroke-width', 1.4);
        g.append('text')
          .attr('x', xScale(cur.beta) + 8).attr('y', yScale(gap) - 8)
          .style('fill', 'var(--color-text)').attr('font-size', 10)
          .text(`gap = ${gap.toFixed(3)} nats`);
      }
    },
    [betaIdx, panelW, panelH, payload, cur?.beta, gap],
  );

  return (
    <div ref={containerRef} style={{ padding: '1rem 0', minHeight: 280 }}>
      {error && (
        <p style={{ fontSize: '0.9rem', color: '#DC2626' }}>
          Failed to load VIB precompute: {error}. Run <code>notebooks/information-bottleneck/precompute_vib_sandbox.py</code> to regenerate.
        </p>
      )}
      {!payload && !error && (
        <p style={{ fontSize: '0.85rem', color: 'var(--color-text-secondary)' }}>
          Loading VIB precompute (40 L-BFGS-B optima)…
        </p>
      )}
      {payload && cur && (
        <>
          <div style={{ marginBottom: '0.75rem' }}>
            <label htmlFor="vib-beta-idx" style={{ fontSize: '0.85rem', color: 'var(--color-text-secondary)', display: 'block' }}>
              β = <strong style={{ color: 'var(--color-text)' }}>{cur.beta.toFixed(2)}</strong>
              {'   '}
              <span style={{ color: 'var(--color-text-secondary)' }}>
                VIB: ({cur.I_XT.toFixed(3)}, {cur.I_TY.toFixed(3)}) · IB: ({ibCur!.I_XT.toFixed(3)}, {ibCur!.I_TY.toFixed(3)}) · gap = {gap!.toFixed(3)} nats
              </span>
            </label>
            <input
              id="vib-beta-idx"
              type="range"
              min={0}
              max={payload.vib_trajectory.length - 1}
              step={1}
              value={betaIdx}
              onChange={(e) => setBetaIdx(parseInt(e.target.value, 10))}
              aria-label="Beta index"
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
              <svg ref={planeRef} role="img" aria-label="VIB trajectory vs IB curve" />
            </div>
            <div style={{ flex: 1, minWidth: 0 }}>
              <svg ref={gapRef} role="img" aria-label="Variational gap in predictiveness" />
            </div>
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.75rem' }}>
            The VIB trajectory sits strictly inside the closed-form IB curve at every β. The gap is small at small β (both near origin), peaks at intermediate β, then closes at large β as both saturate at I(X;Y) = {payload._metadata.I_XY_nats.toFixed(3)} nats. The structural reason: VIB fixes the prior q(t) = N(0, I), so the rate term overshoots the true I(X;T) by exactly D_KL(p_T ‖ q), which is non-zero unless Σ_T = I.
          </p>
        </>
      )}
    </div>
  );
}
