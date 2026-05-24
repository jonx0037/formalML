import { useState, useMemo } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';

// Same 4-D toy as GaussianIBCurveExplorer
const RHOS = [0.95, 0.7, 0.3, 0.0];
const LAMBDAS = RHOS.map((r) => 1 - r * r);
const BETA_C = LAMBDAS.map((l) => (l < 1 ? 1 / (1 - l) : Infinity));
const I_XY = -0.5 * RHOS.reduce((s, r) => s + Math.log(1 - r * r), 0); // 1.548 nats
const DIM_COLORS = ['#0F6E56', '#534AB7', '#D97706', '#0EA5E9'];

// Per-direction contributions to I(X;T) and I(T;Y)
function perDirectionContributions(beta: number): { IXT: number[]; ITY: number[] } {
  const IXT = new Array(4).fill(0);
  const ITY = new Array(4).fill(0);
  for (let i = 0; i < 4; i++) {
    if (beta > BETA_C[i]) {
      const lam = LAMBDAS[i];
      IXT[i] = 0.5 * Math.log(((beta - 1) * (1 - lam)) / lam);
      ITY[i] = 0.5 * Math.log((beta - 1) / (beta * lam));
    }
  }
  return { IXT, ITY };
}

export default function DimensionAllocationExplorer() {
  const [beta, setBeta] = useState<number>(20.0);
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();

  const W = Math.max(containerWidth || 760, 320);
  const isMobile = W < 720;
  const panelW = isMobile ? W : Math.min((W - 24) / 2, 460);
  const panelH = Math.round(panelW * 0.78);

  // Precompute stacked decomposition over β sweep
  const betas = useMemo(() => {
    const lo = Math.log(BETA_C[0] * 0.9);
    const hi = Math.log(200);
    return Array.from({ length: 300 }, (_, i) => Math.exp(lo + (hi - lo) * (i / 299)));
  }, []);
  const trace = useMemo(() => betas.map((b) => ({ beta: b, ...perDirectionContributions(b) })), [betas]);

  const cur = perDirectionContributions(beta);
  const curIXTTotal = cur.IXT.reduce((s, v) => s + v, 0);
  const curITYTotal = cur.ITY.reduce((s, v) => s + v, 0);

  const renderStacked = (
    svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
    yKey: 'IXT' | 'ITY',
    titleText: string,
    ymax: number,
  ) => {
    svg.selectAll('*').remove();
    const margin = { top: 28, right: 18, bottom: 44, left: 56 };
    const innerW = panelW - margin.left - margin.right;
    const innerH = panelH - margin.top - margin.bottom;
    if (innerW <= 0 || innerH <= 0) return;
    svg.attr('viewBox', `0 0 ${panelW} ${panelH}`).attr('width', '100%').attr('height', panelH);

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const xScale = d3.scaleLog().domain([betas[0], betas[betas.length - 1]]).range([0, innerW]);
    const yScale = d3.scaleLinear().domain([0, ymax]).range([innerH, 0]);

    // Build per-direction cumulative bands
    const cumulative: number[][] = trace.map((d) => {
      const vals = d[yKey];
      const cum = [0];
      for (let i = 0; i < 4; i++) cum.push(cum[i] + vals[i]);
      return cum;
    });

    // Area generator per direction
    for (let i = 0; i < 4; i++) {
      const area = d3
        .area<number[]>()
        .x((_, idx) => xScale(betas[idx]))
        .y0((cum) => yScale(cum[i]))
        .y1((cum) => yScale(cum[i + 1]));
      g.append('path')
        .attr('d', area(cumulative) ?? '')
        .style('fill', DIM_COLORS[i])
        .attr('opacity', 0.65);
    }

    // Phase-transition verticals
    const phaseColors = ['#534AB7', '#D97706', '#DC2626'];
    [0, 1, 2].forEach((i) => {
      if (BETA_C[i] >= betas[0] && BETA_C[i] <= betas[betas.length - 1]) {
        g.append('line')
          .attr('x1', xScale(BETA_C[i])).attr('x2', xScale(BETA_C[i]))
          .attr('y1', yScale(0)).attr('y2', yScale(ymax))
          .style('stroke', phaseColors[i]).attr('stroke-dasharray', '3 4')
          .attr('stroke-width', 1.2).attr('opacity', 0.8);
      }
    });

    // Saturation ceiling for I(T;Y) panel
    if (yKey === 'ITY') {
      g.append('line')
        .attr('x1', xScale(betas[0])).attr('x2', xScale(betas[betas.length - 1]))
        .attr('y1', yScale(I_XY)).attr('y2', yScale(I_XY))
        .style('stroke', 'var(--color-text-secondary)')
        .attr('stroke-dasharray', '4 3').attr('stroke-width', 1.0);
      g.append('text')
        .attr('x', xScale(betas[betas.length - 1]) - 4).attr('y', yScale(I_XY) - 4)
        .attr('text-anchor', 'end')
        .style('fill', 'var(--color-text-secondary)').attr('font-size', 10)
        .text(`I(X;Y) = ${I_XY.toFixed(3)} nats`);
    }

    // Axes
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
      .text(`${yKey === 'IXT' ? 'I(X;T)' : 'I(T;Y)'} (nats)`);
    g.append('text')
      .attr('x', innerW / 2).attr('y', -10).attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text-secondary)').attr('font-size', 11)
      .text(titleText);

    // Active β cursor
    g.append('line')
      .attr('x1', xScale(beta)).attr('x2', xScale(beta))
      .attr('y1', yScale(0)).attr('y2', yScale(ymax))
      .style('stroke', 'var(--color-text)').attr('stroke-dasharray', '3 3').attr('stroke-width', 1.4);
  };

  const ixtMax = 5.0;
  const ityMax = I_XY * 1.05;

  const ixtRef = useD3<SVGSVGElement>(
    (svg) => renderStacked(svg, 'IXT', 'Per-direction stack of I(X;T)', ixtMax),
    [beta, panelW, panelH, trace, betas],
  );
  const ityRef = useD3<SVGSVGElement>(
    (svg) => renderStacked(svg, 'ITY', 'Per-direction stack of I(T;Y)', ityMax),
    [beta, panelW, panelH, trace, betas],
  );

  return (
    <div ref={containerRef} style={{ padding: '1rem 0' }}>
      <div style={{ marginBottom: '0.75rem' }}>
        <label
          htmlFor="dim-alloc-beta"
          style={{ fontSize: '0.85rem', color: 'var(--color-text-secondary)', display: 'block' }}
        >
          β = <strong style={{ color: 'var(--color-text)' }}>{beta.toFixed(2)}</strong>
          {'   '}
          <span style={{ color: 'var(--color-text-secondary)' }}>
            ({cur.IXT.filter((v) => v > 0).length} active direction{cur.IXT.filter((v) => v > 0).length === 1 ? '' : 's'} · I(X;T) = {curIXTTotal.toFixed(3)} nats · I(T;Y) = {curITYTotal.toFixed(3)} nats)
          </span>
        </label>
        <input
          id="dim-alloc-beta"
          type="range"
          min={1.05}
          max={50}
          step={0.05}
          value={beta}
          onChange={(e) => setBeta(parseFloat(e.target.value))}
          aria-label="Beta"
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
          <svg ref={ixtRef} role="img" aria-label="Per-direction stack of I(X;T)" />
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <svg ref={ityRef} role="img" aria-label="Per-direction stack of I(T;Y)" />
        </div>
      </div>

      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '0.75rem',
          marginTop: '0.6rem',
          fontSize: '0.78rem',
          color: 'var(--color-text-secondary)',
        }}
      >
        {[0, 1, 2, 3].map((i) => (
          <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
            <span
              style={{
                display: 'inline-block',
                width: 12,
                height: 12,
                background: DIM_COLORS[i],
                opacity: 0.7,
                borderRadius: 2,
              }}
            />
            <span>
              dir {i + 1}: ρ = {RHOS[i].toFixed(2)}, λ = {LAMBDAS[i].toFixed(3)}, β_c ={' '}
              {isFinite(BETA_C[i]) ? BETA_C[i].toFixed(2) : '∞'}
            </span>
          </div>
        ))}
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.75rem' }}>
        Direction 1 dominates the predictiveness budget: it saturates at −½ log λ_1 ≈ 1.164 nats (75.2% of I(X;Y)). Direction 2 contributes ≈ 0.337 nats (21.8%), direction 3 ≈ 0.047 nats (3.0%). Direction 4 (ρ_4 = 0) never activates.
      </p>
    </div>
  );
}
