import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import { gaussianIBCurve, gaussianMI, paletteRL } from './shared/representation-learning';

// =============================================================================
// IBLagrangianFrontier — §7.2 companion. Closed-form Gaussian IB frontier
// D(R) = (1/2) log [e^{2R} / (ρ² + (1 − ρ²) e^{2R})] for bivariate Gaussian
// (X, Y) with correlation ρ. The reader sweeps ρ and watches the curve trace
// out between the diagonal (R ≈ 0, every rate bit buys a sufficiency bit) and
// the data-processing ceiling I(X; Y) = -½ log(1 − ρ²).
// =============================================================================

const R_MIN = 0;
const R_MAX = 4;
const N_POINTS = 200;

export default function IBLagrangianFrontier() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [rho, setRho] = useState(0.9);
  const [rTarget, setRTarget] = useState(1.0);

  const W = Math.max(360, Math.min(width || 720, 720));
  const H = 360;
  const margin = { top: 22, right: 22, bottom: 42, left: 56 };
  const innerW = W - margin.left - margin.right;
  const innerH = H - margin.top - margin.bottom;

  const curve = useMemo(() => {
    const Rs = new Float64Array(N_POINTS);
    for (let i = 0; i < N_POINTS; i++) Rs[i] = R_MIN + ((R_MAX - R_MIN) * i) / (N_POINTS - 1);
    const Ds = gaussianIBCurve(rho, Rs);
    return { Rs, Ds };
  }, [rho]);

  const ceiling = useMemo(() => gaussianMI(rho), [rho]);

  const dAtTarget = useMemo(() => {
    const Ds = gaussianIBCurve(rho, new Float64Array([rTarget]));
    return Ds[0];
  }, [rho, rTarget]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.attr('viewBox', `0 0 ${W} ${H}`).selectAll('*').remove();
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const x = d3.scaleLinear().domain([R_MIN, R_MAX]).range([0, innerW]);
      const yMax = Math.max(0.05, ceiling * 1.15);
      const y = d3.scaleLinear().domain([0, yMax]).range([innerH, 0]);

      g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(x).ticks(5));
      g.append('g').call(d3.axisLeft(y).ticks(6));
      g.append('text')
        .attr('x', innerW / 2)
        .attr('y', innerH + 32)
        .attr('text-anchor', 'middle')
        .attr('font-size', 11)
        .style('fill', 'currentColor')
        .text('Rate R = I(X; T)  (nats)');
      g.append('text')
        .attr('x', -innerH / 2)
        .attr('y', -42)
        .attr('text-anchor', 'middle')
        .attr('transform', 'rotate(-90)')
        .attr('font-size', 11)
        .style('fill', 'currentColor')
        .text('Sufficiency D = I(T; Y)  (nats)');

      // Data-processing ceiling.
      g.append('line')
        .attr('x1', 0)
        .attr('x2', innerW)
        .attr('y1', y(ceiling))
        .attr('y2', y(ceiling))
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '4 3')
        .attr('opacity', 0.6)
        .style('stroke', paletteRL.optimum);
      g.append('text')
        .attr('x', innerW - 6)
        .attr('y', y(ceiling) - 4)
        .attr('text-anchor', 'end')
        .attr('font-size', 10)
        .style('fill', paletteRL.optimum)
        .text(`I(X; Y) = -½log(1−ρ²) = ${ceiling.toFixed(3)}`);

      // Diagonal y = R.
      g.append('line')
        .attr('x1', x(0))
        .attr('x2', x(Math.min(R_MAX, yMax)))
        .attr('y1', y(0))
        .attr('y2', y(Math.min(R_MAX, yMax)))
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '2 3')
        .attr('opacity', 0.4)
        .style('stroke', paletteRL.data);

      // IB curve.
      const line = d3
        .line<number>()
        .x((_d, i) => x(curve.Rs[i]))
        .y((d) => y(d));
      g.append('path')
        .datum(Array.from(curve.Ds))
        .attr('fill', 'none')
        .attr('stroke-width', 2.4)
        .attr('d', line)
        .style('stroke', paletteRL.signal);

      // Target point.
      g.append('line')
        .attr('x1', x(rTarget))
        .attr('x2', x(rTarget))
        .attr('y1', y(0))
        .attr('y2', y(dAtTarget))
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '3 3')
        .attr('opacity', 0.6)
        .style('stroke', paletteRL.highlight);
      g.append('circle')
        .attr('cx', x(rTarget))
        .attr('cy', y(dAtTarget))
        .attr('r', 6)
        .attr('stroke-width', 1.5)
        .style('fill', paletteRL.highlight)
        .style('stroke', paletteRL.highlight)
        .style('fill-opacity', 0.65);
    },
    [curve, ceiling, rho, rTarget, dAtTarget, W, H, innerW, innerH],
  );

  return (
    <div ref={containerRef} className="not-prose my-6">
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-2 text-sm">
        <label className="flex flex-col">
          <span>Correlation ρ = {rho.toFixed(2)}</span>
          <input
            type="range"
            min={0.05}
            max={0.99}
            step={0.01}
            value={rho}
            onChange={(e) => setRho(Number(e.target.value))}
            aria-label="Bivariate-Gaussian correlation"
          />
        </label>
        <label className="flex flex-col">
          <span>Rate R = {rTarget.toFixed(2)} nats</span>
          <input
            type="range"
            min={R_MIN}
            max={R_MAX}
            step={0.01}
            value={rTarget}
            onChange={(e) => setRTarget(Number(e.target.value))}
            aria-label="Target rate"
          />
        </label>
      </div>
      <svg ref={svgRef} className="w-full" style={{ maxHeight: H }} role="img" aria-label="Gaussian information bottleneck frontier"/>
      <div className="text-xs opacity-75 mt-1 text-center">
        At ρ = {rho.toFixed(2)} and R = {rTarget.toFixed(2)}: D(R) = <strong>{dAtTarget.toFixed(3)}</strong> nats; ceiling I(X; Y) = <strong>{ceiling.toFixed(3)}</strong> nats. The curve is concave; the asymptote is the data-processing limit.
      </div>
    </div>
  );
}
