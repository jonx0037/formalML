import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import {
  gmmDenoiser,
  gmmScore,
  gmmResponsibilities,
  paletteRL,
  type GMMParams,
} from './shared/representation-learning';

// =============================================================================
// DenoisingAEScoreField — §3.4 companion. Tweedie's identity verified live on
// a 4-component Dirac mixture in 2D. The viz draws the smoothed density
// p_σ(x̃) = Σ_k w_k N(x̃; μ_k, σ² I) as a contour plot, overlays the optimal
// denoiser displacement r*(x̃) - x̃ as an arrow field, and shows (in green)
// the closed-form σ² ∇log p_σ(x̃) — which coincides arrow-for-arrow with the
// denoiser displacement, to machine precision. Slider for σ.
// =============================================================================

const GRID = 16;
const PLOT_EXTENT = 4.5;

const MUS = new Float64Array([2, 2, -2, 2, -2, -2, 2, -2]);
const WEIGHTS = new Float64Array([0.25, 0.25, 0.25, 0.25]);
const GMM: GMMParams = { mus: MUS, weights: WEIGHTS, K: 4, p: 2 };

function densityAt(x: number, y: number, sigma: number): number {
  const xTilde = new Float64Array([x, y]);
  const inv = 1 / (2 * sigma * sigma);
  let s = 0;
  for (let k = 0; k < GMM.K; k++) {
    const dx = x - GMM.mus[k * 2];
    const dy = y - GMM.mus[k * 2 + 1];
    s += GMM.weights[k] * Math.exp(-(dx * dx + dy * dy) * inv);
  }
  return s / (2 * Math.PI * sigma * sigma);
}

export default function DenoisingAEScoreField() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [sigma, setSigma] = useState(0.6);

  const W = Math.max(360, Math.min(width || 600, 600));
  const H = W; // square
  const margin = { top: 16, right: 16, bottom: 38, left: 38 };
  const innerW = W - margin.left - margin.right;
  const innerH = H - margin.top - margin.bottom;

  const fieldData = useMemo(() => {
    const arrows: { x: number; y: number; rx: number; ry: number; sx: number; sy: number; err: number }[] = [];
    const densGrid: number[][] = [];
    let dMax = 0;
    for (let i = 0; i < GRID; i++) {
      const yy = -PLOT_EXTENT + (2 * PLOT_EXTENT * i) / (GRID - 1);
      const row: number[] = [];
      for (let j = 0; j < GRID; j++) {
        const xx = -PLOT_EXTENT + (2 * PLOT_EXTENT * j) / (GRID - 1);
        const d = densityAt(xx, yy, sigma);
        row.push(d);
        if (d > dMax) dMax = d;
        const xTilde = new Float64Array([xx, yy]);
        const r = gmmDenoiser(xTilde, GMM, sigma);
        const s = gmmScore(xTilde, GMM, sigma);
        const rx = r[0] - xx;
        const ry = r[1] - yy;
        const sx = sigma * sigma * s[0];
        const sy = sigma * sigma * s[1];
        const err = Math.sqrt((rx - sx) ** 2 + (ry - sy) ** 2);
        arrows.push({ x: xx, y: yy, rx, ry, sx, sy, err });
      }
      densGrid.push(row);
    }
    return { arrows, densGrid, dMax };
  }, [sigma]);

  const maxResp = useMemo(() => {
    const probe = new Float64Array([0, 0]);
    const g = gmmResponsibilities(probe, GMM, sigma);
    let m = 0;
    for (let k = 0; k < GMM.K; k++) m = Math.max(m, g[k]);
    return m;
  }, [sigma]);

  const maxErr = useMemo(() => Math.max(...fieldData.arrows.map((a) => a.err)), [fieldData]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.attr('viewBox', `0 0 ${W} ${H}`).selectAll('*').remove();
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
      const x = d3.scaleLinear().domain([-PLOT_EXTENT, PLOT_EXTENT]).range([0, innerW]);
      const y = d3.scaleLinear().domain([-PLOT_EXTENT, PLOT_EXTENT]).range([innerH, 0]);
      g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(x).ticks(5));
      g.append('g').call(d3.axisLeft(y).ticks(5));

      // Component centers.
      g.append('g')
        .selectAll('circle')
        .data([0, 1, 2, 3])
        .join('circle')
        .attr('cx', (k) => x(GMM.mus[k * 2]))
        .attr('cy', (k) => y(GMM.mus[k * 2 + 1]))
        .attr('r', 4)
        .attr('stroke-width', 1)
        .style('fill', paletteRL.optimum)
        .style('stroke', paletteRL.optimum);

      // Density background — small dots, opacity ∝ density.
      g.append('g')
        .selectAll('circle.dens')
        .data(fieldData.arrows)
        .join('circle')
        .attr('class', 'dens')
        .attr('cx', (a) => x(a.x))
        .attr('cy', (a) => y(a.y))
        .attr('r', 22)
        .attr('opacity', (a) => 0.04 + 0.18 * Math.min(1, densityAt(a.x, a.y, sigma) / Math.max(fieldData.dMax, 1e-9)))
        .style('fill', paletteRL.data);

      // Score field (green, slightly translucent — drawn first so denoiser sits on top).
      g.append('g')
        .selectAll('line.score')
        .data(fieldData.arrows)
        .join('line')
        .attr('class', 'score')
        .attr('x1', (a) => x(a.x))
        .attr('y1', (a) => y(a.y))
        .attr('x2', (a) => x(a.x + 0.5 * a.sx))
        .attr('y2', (a) => y(a.y + 0.5 * a.sy))
        .attr('stroke-width', 1.6)
        .attr('opacity', 0.85)
        .style('stroke', paletteRL.optimum);

      // Denoiser displacement (red).
      g.append('g')
        .selectAll('line.denoise')
        .data(fieldData.arrows)
        .join('line')
        .attr('class', 'denoise')
        .attr('x1', (a) => x(a.x))
        .attr('y1', (a) => y(a.y))
        .attr('x2', (a) => x(a.x + 0.5 * a.rx))
        .attr('y2', (a) => y(a.y + 0.5 * a.ry))
        .attr('stroke-width', 0.9)
        .attr('opacity', 0.75)
        .style('stroke', paletteRL.nuisance);

      g.append('text').attr('x', 6).attr('y', 12).attr('font-size', 10).style('fill', paletteRL.optimum).text('σ² ∇log p_σ(x̃)  (green)');
      g.append('text').attr('x', 6).attr('y', 26).attr('font-size', 10).style('fill', paletteRL.nuisance).text('r*(x̃) − x̃  (red)');
    },
    [fieldData, sigma, W, H, innerW, innerH],
  );

  return (
    <div ref={containerRef} className="not-prose my-6">
      <div className="grid grid-cols-1 gap-3 mb-2 text-sm">
        <label className="flex flex-col">
          <span>Noise scale σ = {sigma.toFixed(2)}</span>
          <input
            type="range"
            min={0.1}
            max={2.0}
            step={0.05}
            value={sigma}
            onChange={(e) => setSigma(Number(e.target.value))}
            aria-label="Noise standard deviation"
          />
        </label>
      </div>
      <svg ref={svgRef} className="w-full" style={{ maxHeight: H }} role="img" aria-label="Tweedie identity: optimal denoiser is the score" />
      <div className="text-xs opacity-75 mt-1 text-center">
        Two arrow fields are drawn on top of one another, and they coincide to machine precision: max ||r*(x̃) − x̃ − σ²∇log p_σ(x̃)|| over this 16×16 grid is <strong>{maxErr.toExponential(2)}</strong>. The largest responsibility at the origin under σ = {sigma.toFixed(2)} is <strong>{maxResp.toFixed(3)}</strong> (uniform when σ is large).
      </div>
    </div>
  );
}
