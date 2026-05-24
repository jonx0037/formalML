import { useState, useMemo } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';

// 4-D Gaussian toy from §6.4: designed canonical correlations ρ = (0.95, 0.70, 0.30, 0)
// Eigenvalues of M = Σ_{X|Y} Σ_X^{-1}: λ_i = 1 - ρ_i^2
const RHOS = [0.95, 0.7, 0.3, 0.0];
const LAMBDAS = RHOS.map((r) => 1 - r * r); // [0.0975, 0.51, 0.91, 1.0]
const BETA_C = LAMBDAS.map((l) => (l < 1 ? 1 / (1 - l) : Infinity)); // [1.108, 2.041, 11.111, Inf]
const I_XY = -0.5 * RHOS.reduce((s, r) => s + Math.log(1 - r * r), 0); // 1.548 nats

// Closed form for active directions per §7
function gaussianIBAtBeta(beta: number): { IXT: number; ITY: number; activeCount: number } {
  let IXT = 0, ITY = 0, active = 0;
  for (let i = 0; i < 4; i++) {
    if (beta > BETA_C[i]) {
      const lam = LAMBDAS[i];
      IXT += 0.5 * Math.log(((beta - 1) * (1 - lam)) / lam);
      ITY += 0.5 * Math.log((beta - 1) / (beta * lam));
      active++;
    }
  }
  return { IXT, ITY, activeCount: active };
}

export default function GaussianIBCurveExplorer() {
  const [beta, setBeta] = useState<number>(5.0);
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();

  const W = Math.max(containerWidth || 760, 320);
  const isMobile = W < 720;
  const panelW = isMobile ? W : Math.min((W - 24) / 2, 460);
  const panelH = Math.round(panelW * 0.78);

  // Trace curve over [β_c,1, 200] log-spaced
  const trace = useMemo(() => {
    const lo = Math.log(BETA_C[0] * 1.001);
    const hi = Math.log(200);
    return Array.from({ length: 300 }, (_, i) => {
      const b = Math.exp(lo + (hi - lo) * (i / 299));
      return { beta: b, ...gaussianIBAtBeta(b) };
    });
  }, []);

  const cur = gaussianIBAtBeta(beta);

  // Phase-transition markers at β_c values
  const phaseMarkers = BETA_C.filter((b) => isFinite(b)).map((b) => ({ beta: b, ...gaussianIBAtBeta(b * 1.0001) }));

  // Panel (a): IB plane
  const planeRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const margin = { top: 28, right: 18, bottom: 44, left: 56 };
      const innerW = panelW - margin.left - margin.right;
      const innerH = panelH - margin.top - margin.bottom;
      if (innerW <= 0 || innerH <= 0) return;
      svg.attr('viewBox', `0 0 ${panelW} ${panelH}`).attr('width', '100%').attr('height', panelH);

      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const maxIXT = 5.0;
      const maxITY = I_XY * 1.1;
      const xScale = d3.scaleLinear().domain([0, maxIXT]).range([0, innerW]);
      const yScale = d3.scaleLinear().domain([0, maxITY]).range([innerH, 0]);

      // DPI ceiling at I(X;Y)
      g.append('line')
        .attr('x1', xScale(0)).attr('x2', xScale(maxIXT))
        .attr('y1', yScale(I_XY)).attr('y2', yScale(I_XY))
        .style('stroke', 'var(--color-text-secondary)')
        .attr('stroke-dasharray', '4 3').attr('stroke-width', 1.0);
      g.append('text')
        .attr('x', xScale(maxIXT) - 6).attr('y', yScale(I_XY) - 5)
        .attr('text-anchor', 'end')
        .style('fill', 'var(--color-text-secondary)').attr('font-size', 10)
        .text(`I(X;Y) = ${I_XY.toFixed(3)} nats`);

      // Diagonal
      g.append('line')
        .attr('x1', xScale(0)).attr('x2', xScale(Math.min(maxIXT, maxITY)))
        .attr('y1', yScale(0)).attr('y2', yScale(Math.min(maxIXT, maxITY)))
        .style('stroke', 'var(--color-border)')
        .attr('stroke-dasharray', '3 3').attr('stroke-width', 1.0);

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
        .text('Closed-form Gaussian IB curve (4-D toy)');

      // The curve
      const line = d3.line<{ IXT: number; ITY: number }>()
        .x((d) => xScale(d.IXT)).y((d) => yScale(d.ITY));
      g.append('path')
        .attr('d', line(trace) ?? '')
        .style('fill', 'none').style('stroke', 'var(--color-accent)').attr('stroke-width', 2);

      // Phase-transition markers
      const phaseColors = ['#534AB7', '#D97706', '#DC2626'];
      phaseMarkers.forEach((p, i) => {
        g.append('circle')
          .attr('cx', xScale(p.IXT)).attr('cy', yScale(p.ITY))
          .attr('r', 4)
          .style('fill', phaseColors[i])
          .attr('opacity', 0.85);
        g.append('text')
          .attr('x', xScale(p.IXT) + 6).attr('y', yScale(p.ITY) + 4 + i * 12)
          .style('fill', phaseColors[i]).attr('font-size', 9)
          .text(`β_c,${i + 1} = ${BETA_C[i].toFixed(2)}`);
      });

      // Current β operating point
      g.append('circle')
        .attr('cx', xScale(cur.IXT)).attr('cy', yScale(cur.ITY))
        .attr('r', 7)
        .style('fill', '#D97706')
        .attr('stroke', 'var(--color-text)').attr('stroke-width', 1.4);
      g.append('text')
        .attr('x', xScale(cur.IXT) - 10).attr('y', yScale(cur.ITY) - 12)
        .attr('text-anchor', 'end')
        .style('fill', 'var(--color-text)').attr('font-size', 11).attr('font-weight', 600)
        .text(`β = ${beta.toFixed(2)}`);
      g.append('text')
        .attr('x', xScale(cur.IXT) - 10).attr('y', yScale(cur.ITY))
        .attr('text-anchor', 'end')
        .style('fill', 'var(--color-text-secondary)').attr('font-size', 10)
        .text(`(${cur.IXT.toFixed(3)}, ${cur.ITY.toFixed(3)}) nats`);
      g.append('text')
        .attr('x', xScale(cur.IXT) - 10).attr('y', yScale(cur.ITY) + 12)
        .attr('text-anchor', 'end')
        .style('fill', 'var(--color-text-secondary)').attr('font-size', 10)
        .text(`${cur.activeCount} active direction${cur.activeCount === 1 ? '' : 's'}`);
    },
    [beta, panelW, panelH, trace, cur.IXT, cur.ITY, cur.activeCount],
  );

  // Panel (b): I(X;T), I(T;Y) vs β log axis
  const sweepRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const margin = { top: 28, right: 18, bottom: 44, left: 52 };
      const innerW = panelW - margin.left - margin.right;
      const innerH = panelH - margin.top - margin.bottom;
      if (innerW <= 0 || innerH <= 0) return;
      svg.attr('viewBox', `0 0 ${panelW} ${panelH}`).attr('width', '100%').attr('height', panelH);

      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const xScale = d3.scaleLog().domain([1.0, 200]).range([0, innerW]);
      const ymax = Math.max(5.0, trace[trace.length - 1].IXT * 1.05);
      const yScale = d3.scaleLinear().domain([0, ymax]).range([innerH, 0]);

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
        .attr('transform', `translate(-38,${innerH / 2}) rotate(-90)`).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)').attr('font-size', 11)
        .text('information (nats)');
      g.append('text')
        .attr('x', innerW / 2).attr('y', -10).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)').attr('font-size', 11)
        .text('I(X;T) and I(T;Y) vs β');

      // Phase-transition vertical lines
      const phaseColors = ['#534AB7', '#D97706', '#DC2626'];
      [0, 1, 2].forEach((i) => {
        if (BETA_C[i] >= 1.0 && BETA_C[i] <= 200) {
          g.append('line')
            .attr('x1', xScale(BETA_C[i])).attr('x2', xScale(BETA_C[i]))
            .attr('y1', yScale(0)).attr('y2', yScale(ymax))
            .style('stroke', phaseColors[i]).attr('stroke-dasharray', '3 4')
            .attr('stroke-width', 1.0).attr('opacity', 0.6);
        }
      });

      // Trace lines
      const lineIXT = d3.line<{ beta: number; IXT: number; ITY: number }>()
        .x((d) => xScale(d.beta)).y((d) => yScale(d.IXT));
      const lineITY = d3.line<{ beta: number; IXT: number; ITY: number }>()
        .x((d) => xScale(d.beta)).y((d) => yScale(d.ITY));
      g.append('path').attr('d', lineIXT(trace) ?? '')
        .style('fill', 'none').style('stroke', 'var(--color-accent)').attr('stroke-width', 2);
      g.append('path').attr('d', lineITY(trace) ?? '')
        .style('fill', 'none').style('stroke', '#D97706').attr('stroke-width', 2);

      // Labels at right edge
      const last = trace[trace.length - 1];
      g.append('text')
        .attr('x', xScale(last.beta) - 4).attr('y', yScale(last.IXT) - 6)
        .attr('text-anchor', 'end')
        .style('fill', 'var(--color-accent)').attr('font-size', 11).text('I(X;T)');
      g.append('text')
        .attr('x', xScale(last.beta) - 4).attr('y', yScale(last.ITY) - 6)
        .attr('text-anchor', 'end')
        .style('fill', '#D97706').attr('font-size', 11).text('I(T;Y)');

      // DPI ceiling
      g.append('line')
        .attr('x1', xScale(1)).attr('x2', xScale(200))
        .attr('y1', yScale(I_XY)).attr('y2', yScale(I_XY))
        .style('stroke', '#D97706').attr('stroke-dasharray', '2 4')
        .attr('stroke-width', 0.9).attr('opacity', 0.5);

      // Active β cursor
      g.append('line')
        .attr('x1', xScale(beta)).attr('x2', xScale(beta))
        .attr('y1', yScale(0)).attr('y2', yScale(ymax))
        .style('stroke', 'var(--color-text)').attr('stroke-dasharray', '3 3').attr('stroke-width', 1.2);
    },
    [beta, panelW, panelH, trace],
  );

  return (
    <div ref={containerRef} style={{ padding: '1rem 0' }}>
      <div style={{ marginBottom: '0.75rem' }}>
        <label
          htmlFor="gauss-ib-beta"
          style={{ fontSize: '0.85rem', color: 'var(--color-text-secondary)', display: 'block' }}
        >
          β = <strong style={{ color: 'var(--color-text)' }}>{beta.toFixed(2)}</strong>
          {'   '}
          <span style={{ color: 'var(--color-text-secondary)' }}>
            (I(X;T) = {cur.IXT.toFixed(3)} nats · I(T;Y) = {cur.ITY.toFixed(3)} nats · {cur.activeCount} active direction{cur.activeCount === 1 ? '' : 's'})
          </span>
        </label>
        <input
          id="gauss-ib-beta"
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
          <svg ref={planeRef} role="img" aria-label="Closed-form Gaussian IB curve" />
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <svg ref={sweepRef} role="img" aria-label="I(X;T) and I(T;Y) vs beta" />
        </div>
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.75rem' }}>
        Phase transitions at β_c = (1.108, 2.041, 11.111) activate the three nonzero canonical-correlation directions. The fourth (ρ_4 = 0) never activates. Saturation at I(X;Y) = 1.548 nats. Try β = 1.5 → (0.766, 0.615), β = 2.5 → (1.498, 0.990), β = 5.0 → (2.479, 1.278), β = 15 → (3.894, 1.444) — the four §7.4 verification points.
      </p>
    </div>
  );
}
