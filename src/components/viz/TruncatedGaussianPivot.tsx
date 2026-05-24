import { useState, useMemo } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { normalCdf } from './shared/pac-bayes-bounds';

// Truncated Gaussian: 𝒩(μ, σ²) restricted to [ℓ, u].
const SIGMA = 1;
const L = -2;
const U = 2;

function tnPdf(x: number, mu: number): number {
  if (x < L || x > U) return 0;
  const z = (x - mu) / SIGMA;
  const phi = Math.exp(-0.5 * z * z) / (Math.sqrt(2 * Math.PI) * SIGMA);
  const denom = normalCdf((U - mu) / SIGMA) - normalCdf((L - mu) / SIGMA);
  return denom > 1e-12 ? phi / denom : 0;
}

function tnCdf(x: number, mu: number): number {
  if (x <= L) return 0;
  if (x >= U) return 1;
  const num = normalCdf((x - mu) / SIGMA) - normalCdf((L - mu) / SIGMA);
  const denom = normalCdf((U - mu) / SIGMA) - normalCdf((L - mu) / SIGMA);
  return denom > 1e-12 ? num / denom : 0;
}

export default function TruncatedGaussianPivot() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [mu, setMu] = useState(0);
  const [t, setT] = useState(0.6);

  const pivot = useMemo(() => tnCdf(t, mu), [t, mu]);

  const W = Math.max(280, Math.min(containerWidth || 720, 720));
  const isMobile = W < 640;
  const PANEL_W = isMobile ? W : (W - 16) / 2;
  const PANEL_H = Math.round(PANEL_W * 0.72);
  const M = { t: 18, r: 14, b: 36, l: 42 };

  const xs = useMemo(() => d3.range(L - 1, U + 1.0001, 0.02), []);

  const pdfRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const iw = PANEL_W - M.l - M.r;
      const ih = PANEL_H - M.t - M.b;
      if (iw <= 0 || ih <= 0) return;
      const x = d3.scaleLinear().domain([L - 1, U + 1]).range([0, iw]);
      const y = d3.scaleLinear().domain([0, 1.2]).range([ih, 0]);
      const g = svg.append('g').attr('transform', `translate(${M.l},${M.t})`);

      // unconditional density (gray)
      const uncondPdf = (xv: number) => Math.exp(-0.5 * ((xv - mu) / SIGMA) ** 2) / (Math.sqrt(2 * Math.PI) * SIGMA);
      const lineUnc = d3
        .line<number>()
        .x((v) => x(v))
        .y((v) => y(uncondPdf(v)));
      g.append('path')
        .datum(xs as unknown as number[])
        .attr('d', lineUnc)
        .attr('fill', 'none')
        .style('stroke', 'var(--color-text-secondary)')
        .attr('stroke-width', 1.4)
        .attr('stroke-dasharray', '4 3');

      // truncated density (accent), shaded between L and U
      const tnPoints = xs.filter((xv) => xv >= L && xv <= U);
      const area = d3
        .area<number>()
        .x((v) => x(v))
        .y0(y(0))
        .y1((v) => y(tnPdf(v, mu)));
      g.append('path')
        .datum(tnPoints as unknown as number[])
        .attr('d', area)
        .style('fill', 'var(--color-accent)')
        .attr('opacity', 0.22);
      const lineTn = d3
        .line<number>()
        .x((v) => x(v))
        .y((v) => y(tnPdf(v, mu)));
      g.append('path')
        .datum(tnPoints as unknown as number[])
        .attr('d', lineTn)
        .attr('fill', 'none')
        .style('stroke', 'var(--color-accent)')
        .attr('stroke-width', 1.8);

      // truncation lines
      for (const v of [L, U]) {
        g.append('line')
          .attr('x1', x(v))
          .attr('x2', x(v))
          .attr('y1', 0)
          .attr('y2', ih)
          .style('stroke', 'var(--color-border)')
          .attr('stroke-dasharray', '3 3');
      }
      // observed-value marker
      g.append('line')
        .attr('x1', x(t))
        .attr('x2', x(t))
        .attr('y1', 0)
        .attr('y2', ih)
        .style('stroke', 'var(--color-text)')
        .attr('stroke-width', 1.8);
      g.append('text')
        .attr('x', x(t) + 4)
        .attr('y', 12)
        .style('fill', 'var(--color-text)')
        .style('font-family', 'var(--font-sans)')
        .style('font-size', '0.72rem')
        .text(`η⊤y = ${t.toFixed(2)}`);

      // axes
      g.append('g').attr('transform', `translate(0,${ih})`).call(d3.axisBottom(x).ticks(5)).selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(y).ticks(4)).selectAll('text').style('fill', 'var(--color-text)');
      g.selectAll('.domain, line').style('stroke', 'var(--color-border)');
      g.append('text')
        .attr('x', iw / 2)
        .attr('y', ih + 28)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '0.7rem')
        .text('η⊤y');
      g.append('text')
        .attr('x', 8)
        .attr('y', 8)
        .style('fill', 'var(--color-text-secondary)')
        .style('font-family', 'var(--font-sans)')
        .style('font-size', '0.72rem')
        .text('density');
    },
    [PANEL_W, PANEL_H, mu, t, xs],
  );

  const cdfRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const iw = PANEL_W - M.l - M.r;
      const ih = PANEL_H - M.t - M.b;
      if (iw <= 0 || ih <= 0) return;
      const x = d3.scaleLinear().domain([L - 1, U + 1]).range([0, iw]);
      const y = d3.scaleLinear().domain([0, 1]).range([ih, 0]);
      const g = svg.append('g').attr('transform', `translate(${M.l},${M.t})`);

      // truncated CDF
      const cdfPts = xs.map((xv) => ({ x: xv, y: tnCdf(xv, mu) }));
      const line = d3
        .line<{ x: number; y: number }>()
        .x((d) => x(d.x))
        .y((d) => y(d.y));
      g.append('path')
        .datum(cdfPts)
        .attr('d', line)
        .attr('fill', 'none')
        .style('stroke', 'var(--color-accent)')
        .attr('stroke-width', 1.8);

      // pivot value horizontal/vertical guides
      g.append('line')
        .attr('x1', 0)
        .attr('x2', x(t))
        .attr('y1', y(pivot))
        .attr('y2', y(pivot))
        .style('stroke', 'var(--color-text)')
        .attr('stroke-dasharray', '3 3');
      g.append('line')
        .attr('x1', x(t))
        .attr('x2', x(t))
        .attr('y1', ih)
        .attr('y2', y(pivot))
        .style('stroke', 'var(--color-text)')
        .attr('stroke-dasharray', '3 3');
      g.append('circle').attr('cx', x(t)).attr('cy', y(pivot)).attr('r', 4).style('fill', 'var(--color-text)');

      // truncation lines
      for (const v of [L, U]) {
        g.append('line')
          .attr('x1', x(v))
          .attr('x2', x(v))
          .attr('y1', 0)
          .attr('y2', ih)
          .style('stroke', 'var(--color-border)')
          .attr('stroke-dasharray', '3 3');
      }
      g.append('g').attr('transform', `translate(0,${ih})`).call(d3.axisBottom(x).ticks(5)).selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(y).ticks(4)).selectAll('text').style('fill', 'var(--color-text)');
      g.selectAll('.domain, line').style('stroke', 'var(--color-border)');
      g.append('text')
        .attr('x', iw / 2)
        .attr('y', ih + 28)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '0.7rem')
        .text('η⊤y');
      g.append('text')
        .attr('x', 8)
        .attr('y', 8)
        .style('fill', 'var(--color-text-secondary)')
        .style('font-family', 'var(--font-sans)')
        .style('font-size', '0.72rem')
        .text(`pivot = F(η⊤y) = ${pivot.toFixed(3)}`);
    },
    [PANEL_W, PANEL_H, mu, t, pivot, xs],
  );

  return (
    <div ref={containerRef} className="not-prose my-6 w-full">
      <div className="flex flex-col gap-4 sm:flex-row sm:flex-wrap sm:items-end">
        <label className="flex flex-col text-sm" style={{ color: 'var(--color-text-secondary)' }}>
          <span>
            η⊤μ ={' '}
            <span className="tabular-nums" style={{ color: 'var(--color-text)' }}>
              {mu.toFixed(2)}
            </span>{' '}
            (unknown mean)
          </span>
          <input
            type="range"
            min={-2.5}
            max={2.5}
            step={0.05}
            value={mu}
            onChange={(e) => setMu(parseFloat(e.target.value))}
            aria-label="Unknown mean of the underlying Gaussian"
            className="w-56"
          />
        </label>
        <label className="flex flex-col text-sm" style={{ color: 'var(--color-text-secondary)' }}>
          <span>
            η⊤y ={' '}
            <span className="tabular-nums" style={{ color: 'var(--color-text)' }}>
              {t.toFixed(2)}
            </span>{' '}
            (observed value)
          </span>
          <input
            type="range"
            min={L + 0.02}
            max={U - 0.02}
            step={0.02}
            value={t}
            onChange={(e) => setT(parseFloat(e.target.value))}
            aria-label="Observed contrast value"
            className="w-56"
          />
        </label>
      </div>
      <div className={isMobile ? 'mt-2' : 'mt-2 flex gap-4'}>
        <svg ref={pdfRef} width={PANEL_W} height={PANEL_H} role="img" aria-label="Truncated-Gaussian density" />
        <svg ref={cdfRef} width={PANEL_W} height={PANEL_H} role="img" aria-label="Truncated-Gaussian CDF and pivot value" />
      </div>
      <p className="mt-1 text-xs" style={{ color: 'var(--color-text-secondary)' }}>
        Solid teal: truncated 𝒩(η⊤μ, 1) restricted to [V⁻, V⁺] = [{L}, {U}]. Dashed gray: the unconditional Gaussian. Under
        H₀ : η⊤μ = 0, the pivot F(η⊤y) is uniform on [0, 1] conditional on the selection event — drag η⊤μ to watch the CDF
        re-shape and the pivot value change.
      </p>
    </div>
  );
}
