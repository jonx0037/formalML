import { useState, useMemo } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';

// Scalar-Gaussian IB curve in bits (cf. brief §2.4, §6 closed form):
//   I_Y*(R) = -(1/2) log2(1 - rho^2 (1 - 2^(-2 R)))
function ibCurveValue(R: number, rho: number): number {
  const rho2 = rho * rho;
  return -0.5 * Math.log2(1 - rho2 * (1 - Math.pow(2, -2 * R)));
}

// Closed-form (R*, I_Y*) operating point as a function of beta on the scalar-Gaussian.
// At beta <= beta_c = 1/rho^2: trivial encoder, returns (0, 0).
// Otherwise: lambda = 1 - rho^2; alpha^2 = (beta*(1-lambda) - 1)/lambda;
//   R     = (1/2) log2(1 + alpha^2)            in bits
//   I_Y*  = (1/2) log2(1 + alpha^2 * (1-lambda) / lambda) ... but cleaner via the §7 form.
// Equivalent and numerically nicer: use the §7 piecewise formulas directly (in nats then convert).
//   R    = (1/2) ln((beta - 1) * (1 - lambda) / lambda)   nats
//   I_Y* = (1/2) ln((beta - 1) / (beta * lambda))         nats
function operatingPoint(beta: number, rho: number): { R: number; IY: number } {
  const lambda = 1 - rho * rho;
  const beta_c = 1 / (1 - lambda); // = 1 / rho^2
  if (beta <= beta_c) return { R: 0, IY: 0 };
  const ln2 = Math.LN2;
  const R_nats = 0.5 * Math.log(((beta - 1) * (1 - lambda)) / lambda);
  const IY_nats = 0.5 * Math.log((beta - 1) / (beta * lambda));
  return { R: R_nats / ln2, IY: IY_nats / ln2 };
}

const RHO_PRESETS = [0.35, 0.55, 0.8, 0.95];

export default function GaussianIBCurveSchematic() {
  const [rho, setRho] = useState<number>(0.8);
  const [beta, setBeta] = useState<number>(4.0);
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();

  const W = Math.max(containerWidth || 760, 320);
  const isMobile = W < 640;

  // Two side-by-side panels on desktop; stacked on mobile.
  const panelW = isMobile ? W : Math.min((W - 24) / 2, 460);
  const panelH = Math.round(panelW * 0.78);

  const beta_c = 1 / (rho * rho);
  const ceiling = -0.5 * Math.log2(1 - rho * rho); // I(X;Y) in bits
  const op = useMemo(() => operatingPoint(beta, rho), [beta, rho]);

  // Panel (a): the four-rho gallery
  const galleryRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const margin = { top: 22, right: 18, bottom: 44, left: 52 };
      const innerW = panelW - margin.left - margin.right;
      const innerH = panelH - margin.top - margin.bottom;
      if (innerW <= 0 || innerH <= 0) return;

      svg.attr('viewBox', `0 0 ${panelW} ${panelH}`).attr('width', '100%').attr('height', panelH);

      // Choose axes so the largest rho's ceiling is comfortably below the top.
      const maxCeiling = -0.5 * Math.log2(1 - 0.95 * 0.95);
      const xScale = d3.scaleLinear().domain([0, 4.0]).range([0, innerW]);
      const yScale = d3.scaleLinear().domain([0, maxCeiling * 1.05]).range([innerH, 0]);

      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      // Diagonal I(T;Y) ≤ I(X;T)
      g.append('line')
        .attr('x1', xScale(0))
        .attr('x2', xScale(maxCeiling))
        .attr('y1', yScale(0))
        .attr('y2', yScale(maxCeiling))
        .style('stroke', 'var(--color-border)')
        .attr('stroke-dasharray', '3 3')
        .attr('stroke-width', 1);

      // Axes
      const xAxis = d3.axisBottom(xScale).ticks(5);
      const yAxis = d3.axisLeft(yScale).ticks(5);
      const xAxisG = g.append('g').attr('transform', `translate(0,${innerH})`).call(xAxis);
      const yAxisG = g.append('g').call(yAxis);
      xAxisG.selectAll('text').style('fill', 'var(--color-text)').attr('font-size', 10);
      yAxisG.selectAll('text').style('fill', 'var(--color-text)').attr('font-size', 10);
      xAxisG.selectAll('line, path').style('stroke', 'var(--color-border)');
      yAxisG.selectAll('line, path').style('stroke', 'var(--color-border)');

      g.append('text')
        .attr('x', innerW / 2)
        .attr('y', innerH + 32)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)')
        .attr('font-size', 11)
        .text('R = I(X;T) (bits)');
      g.append('text')
        .attr('transform', `translate(-38,${innerH / 2}) rotate(-90)`)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)')
        .attr('font-size', 11)
        .text('I(T;Y) (bits)');

      g.append('text')
        .attr('x', innerW / 2)
        .attr('y', -6)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .attr('font-size', 11)
        .text('Scalar-Gaussian IB curves at four correlations');

      const palette: Record<number, string> = {
        0.35: '#0EA5E9',
        0.55: '#534AB7',
        0.8: '#0F6E56',
        0.95: '#D97706',
      };

      // For each preset rho, sample the curve and the ceiling.
      const Rs = d3.range(0, 4.05, 0.02);
      const lineGen = d3
        .line<number>()
        .x((R) => xScale(R))
        .y((R, _i, arr) => {
          // arr is the same Rs; we close over the current rho via map below
          return yScale(0);
        });

      RHO_PRESETS.forEach((rhoPreset) => {
        const ymax = -0.5 * Math.log2(1 - rhoPreset * rhoPreset);
        const path = d3
          .line<number>()
          .x((R) => xScale(R))
          .y((R) => yScale(ibCurveValue(R, rhoPreset)));

        g.append('path')
          .attr('d', path(Rs) ?? '')
          .style('fill', 'none')
          .style('stroke', palette[rhoPreset])
          .attr('stroke-width', rhoPreset === rho ? 2.5 : 1.4)
          .attr('opacity', rhoPreset === rho ? 1.0 : 0.55);

        // Ceiling
        g.append('line')
          .attr('x1', xScale(0))
          .attr('x2', xScale(4.0))
          .attr('y1', yScale(ymax))
          .attr('y2', yScale(ymax))
          .style('stroke', palette[rhoPreset])
          .attr('stroke-dasharray', '2 4')
          .attr('stroke-width', 0.9)
          .attr('opacity', rhoPreset === rho ? 0.8 : 0.3);

        // Label rho near the ceiling endpoint
        g.append('text')
          .attr('x', xScale(3.9))
          .attr('y', yScale(ymax) - 4)
          .attr('text-anchor', 'end')
          .style('fill', palette[rhoPreset])
          .attr('font-size', 11)
          .attr('font-weight', rhoPreset === rho ? 600 : 400)
          .attr('opacity', rhoPreset === rho ? 1.0 : 0.55)
          .text(`ρ = ${rhoPreset.toFixed(2)}`);
      });
      // Use line generator to silence unused-var
      void lineGen;
    },
    [panelW, panelH, rho],
  );

  // Panel (b): rho curve + tangent at beta
  const tangentRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const margin = { top: 22, right: 18, bottom: 44, left: 52 };
      const innerW = panelW - margin.left - margin.right;
      const innerH = panelH - margin.top - margin.bottom;
      if (innerW <= 0 || innerH <= 0) return;

      svg.attr('viewBox', `0 0 ${panelW} ${panelH}`).attr('width', '100%').attr('height', panelH);

      const Rmax = Math.max(3.0, op.R + 1.0);
      const xScale = d3.scaleLinear().domain([0, Rmax]).range([0, innerW]);
      const yScale = d3.scaleLinear().domain([0, ceiling * 1.1]).range([innerH, 0]);

      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      // Diagonal
      g.append('line')
        .attr('x1', xScale(0))
        .attr('x2', xScale(Math.min(Rmax, ceiling)))
        .attr('y1', yScale(0))
        .attr('y2', yScale(Math.min(Rmax, ceiling)))
        .style('stroke', 'var(--color-border)')
        .attr('stroke-dasharray', '3 3')
        .attr('stroke-width', 1);

      // Ceiling
      g.append('line')
        .attr('x1', xScale(0))
        .attr('x2', xScale(Rmax))
        .attr('y1', yScale(ceiling))
        .attr('y2', yScale(ceiling))
        .style('stroke', 'var(--color-text-secondary)')
        .attr('stroke-dasharray', '4 3')
        .attr('stroke-width', 1.0);
      g.append('text')
        .attr('x', xScale(Rmax) - 6)
        .attr('y', yScale(ceiling) - 5)
        .attr('text-anchor', 'end')
        .style('fill', 'var(--color-text-secondary)')
        .attr('font-size', 10)
        .text(`I(X;Y) = ${ceiling.toFixed(3)} bits`);

      // Axes
      const xAxis = d3.axisBottom(xScale).ticks(5);
      const yAxis = d3.axisLeft(yScale).ticks(5);
      const xAxisG = g.append('g').attr('transform', `translate(0,${innerH})`).call(xAxis);
      const yAxisG = g.append('g').call(yAxis);
      xAxisG.selectAll('text').style('fill', 'var(--color-text)').attr('font-size', 10);
      yAxisG.selectAll('text').style('fill', 'var(--color-text)').attr('font-size', 10);
      xAxisG.selectAll('line, path').style('stroke', 'var(--color-border)');
      yAxisG.selectAll('line, path').style('stroke', 'var(--color-border)');

      g.append('text')
        .attr('x', innerW / 2)
        .attr('y', innerH + 32)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)')
        .attr('font-size', 11)
        .text('R = I(X;T) (bits)');
      g.append('text')
        .attr('transform', `translate(-38,${innerH / 2}) rotate(-90)`)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)')
        .attr('font-size', 11)
        .text('I(T;Y) (bits)');

      g.append('text')
        .attr('x', innerW / 2)
        .attr('y', -6)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .attr('font-size', 11)
        .text(`ρ = ${rho.toFixed(2)} curve with tangent at the β operating point`);

      // Curve
      const Rs = d3.range(0, Rmax + 0.02, 0.02);
      const path = d3
        .line<number>()
        .x((R) => xScale(R))
        .y((R) => yScale(ibCurveValue(R, rho)));
      g.append('path')
        .attr('d', path(Rs) ?? '')
        .style('fill', 'none')
        .style('stroke', 'var(--color-accent)')
        .attr('stroke-width', 2);

      // Tangent line at the operating point (slope = 1/beta by KKT identity 2.5)
      if (beta > beta_c) {
        const slope = 1 / beta;
        const Rstart = Math.max(0, op.R - 1.4);
        const Rend = Math.min(Rmax, op.R + 1.4);
        g.append('line')
          .attr('x1', xScale(Rstart))
          .attr('x2', xScale(Rend))
          .attr('y1', yScale(op.IY - slope * (op.R - Rstart)))
          .attr('y2', yScale(op.IY + slope * (Rend - op.R)))
          .style('stroke', '#D97706')
          .attr('stroke-width', 1.6)
          .attr('opacity', 0.85);

        // Operating point
        g.append('circle')
          .attr('cx', xScale(op.R))
          .attr('cy', yScale(op.IY))
          .attr('r', 6)
          .style('fill', '#D97706')
          .attr('stroke', 'var(--color-text)')
          .attr('stroke-width', 1.2);
        g.append('text')
          .attr('x', xScale(op.R) + 10)
          .attr('y', yScale(op.IY) - 8)
          .style('fill', 'var(--color-text)')
          .attr('font-size', 11)
          .attr('font-weight', 600)
          .text(`β = ${beta.toFixed(2)}`);
        g.append('text')
          .attr('x', xScale(op.R) + 10)
          .attr('y', yScale(op.IY) + 6)
          .style('fill', 'var(--color-text-secondary)')
          .attr('font-size', 10)
          .text(`(${op.R.toFixed(3)}, ${op.IY.toFixed(3)}) bits`);
        g.append('text')
          .attr('x', xScale(op.R) + 10)
          .attr('y', yScale(op.IY) + 20)
          .style('fill', 'var(--color-text-secondary)')
          .attr('font-size', 10)
          .text(`slope = 1/β = ${slope.toFixed(3)}`);
      } else {
        g.append('circle')
          .attr('cx', xScale(0))
          .attr('cy', yScale(0))
          .attr('r', 6)
          .style('fill', '#888')
          .attr('stroke', 'var(--color-text)')
          .attr('stroke-width', 1.2);
        g.append('text')
          .attr('x', xScale(0) + 14)
          .attr('y', yScale(0) - 10)
          .style('fill', 'var(--color-text-secondary)')
          .attr('font-size', 11)
          .text(`β ≤ β_c = ${beta_c.toFixed(3)}  →  T = const`);
      }
    },
    [panelW, panelH, rho, beta, ceiling, beta_c, op.R, op.IY],
  );

  return (
    <div ref={containerRef} style={{ padding: '1rem 0' }}>
      <div
        style={{
          display: 'flex',
          flexDirection: isMobile ? 'column' : 'row',
          gap: '1.5rem',
          marginBottom: '0.75rem',
        }}
      >
        <div style={{ flex: 1 }}>
          <label
            htmlFor="ib-schem-rho"
            style={{ fontSize: '0.85rem', color: 'var(--color-text-secondary)', display: 'block' }}
          >
            Correlation ρ ={' '}
            <span style={{ color: 'var(--color-text)', fontWeight: 600 }}>{rho.toFixed(2)}</span>
            {'   '}
            <span style={{ color: 'var(--color-text-secondary)' }}>(β_c = 1/ρ² = {beta_c.toFixed(3)})</span>
          </label>
          <input
            id="ib-schem-rho"
            type="range"
            min={0.1}
            max={0.99}
            step={0.01}
            value={rho}
            onChange={(e) => setRho(parseFloat(e.target.value))}
            aria-label="Correlation rho"
            style={{ width: '100%' }}
          />
        </div>
        <div style={{ flex: 1 }}>
          <label
            htmlFor="ib-schem-beta"
            style={{ fontSize: '0.85rem', color: 'var(--color-text-secondary)', display: 'block' }}
          >
            β ={' '}
            <span style={{ color: 'var(--color-text)', fontWeight: 600 }}>{beta.toFixed(2)}</span>
            {'   '}
            <span style={{ color: 'var(--color-text-secondary)' }}>(slope at operating point = 1/β = {(1 / beta).toFixed(3)})</span>
          </label>
          <input
            id="ib-schem-beta"
            type="range"
            min={Math.max(0.5, beta_c * 0.5)}
            max={30}
            step={0.1}
            value={beta}
            onChange={(e) => setBeta(parseFloat(e.target.value))}
            aria-label="Lagrange multiplier beta"
            style={{ width: '100%' }}
          />
        </div>
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
          <svg ref={galleryRef} role="img" aria-label="Scalar-Gaussian IB curves at four correlations" />
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <svg ref={tangentRef} role="img" aria-label="Single-rho IB curve with tangent at the beta operating point" />
        </div>
      </div>

      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.75rem' }}>
        Drag β past β_c = 1/ρ² to escape the trivial encoder; the tangent line's slope is always 1/β, which is the KKT identity (2.5). At ρ = 0.8 the §2.4 reference operating points sit at (0.254, 0.152), (1.208, 0.529), and (2.145, 0.674) bits for β ∈ {`{1.8, 4, 12}`}.
      </p>
    </div>
  );
}
