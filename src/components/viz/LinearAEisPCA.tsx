import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import {
  centerColumns,
  pcaTopK,
  linearAEFit,
  paletteRL,
  mulberry32,
  gaussianPair,
} from './shared/representation-learning';

// =============================================================================
// LinearAEisPCA — §3.2 companion. The reader sees an elongated 2D Gaussian
// scatter, the closed-form top PCA eigenvector (solid green), and the
// AE-recovered bottleneck direction (dashed orange) at a slider-controlled
// epoch index. A side panel tracks the reconstruction loss against the
// Baldi-Hornik floor λ_2 and the |cos∠(W, v_top)| convergence.
// =============================================================================

const N_SAMPLES = 800;
const MAX_EPOCH = 400;

interface Fixture {
  X: Float64Array;
  vTop: Float64Array;
  lambda1: number;
  lambda2: number;
}

function buildFixture(seed: number): Fixture {
  const rng = mulberry32(seed);
  const X = new Float64Array(N_SAMPLES * 2);
  for (let i = 0; i < N_SAMPLES; i += 2) {
    const [a, b] = gaussianPair(rng);
    const [c, d] = gaussianPair(rng);
    X[i * 2] = 3 * a;
    X[i * 2 + 1] = 0.8 * b;
    if (i + 1 < N_SAMPLES) {
      X[(i + 1) * 2] = 3 * c;
      X[(i + 1) * 2 + 1] = 0.8 * d;
    }
  }
  centerColumns(X, N_SAMPLES, 2);
  const { components, eigenvalues } = pcaTopK(X, N_SAMPLES, 2, 2);
  return { X, vTop: components.slice(0, 2), lambda1: eigenvalues[0], lambda2: eigenvalues[1] };
}

export default function LinearAEisPCA() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [epochDisplay, setEpochDisplay] = useState(50);
  const [epochCommitted, setEpochCommitted] = useState(50);
  const [lr, setLr] = useState(0.05);

  const W = Math.max(360, Math.min(width || 720, 720));
  const H = 380;
  const margin = { top: 20, right: 22, bottom: 42, left: 50 };
  const splitX = Math.round((W - margin.left - margin.right) * 0.55);
  const innerH = H - margin.top - margin.bottom;

  const fixture = useMemo(() => buildFixture(13), []);

  const aeResult = useMemo(() => {
    return linearAEFit(fixture.X, N_SAMPLES, 2, 1, {
      lr,
      epochs: MAX_EPOCH,
      seed: 11,
      vTop: fixture.vTop,
    });
  }, [fixture, lr]);

  // Encoder direction at the chosen epoch — replay the trajectory to get W at that epoch.
  // linearAEFit returns only the final W. For a "scrub through epochs" feel, refit at the
  // selected epoch budget instead. Cheap because each fit is O(epochs · n).
  const trimmedAe = useMemo(() => {
    return linearAEFit(fixture.X, N_SAMPLES, 2, 1, {
      lr,
      epochs: Math.max(1, epochCommitted),
      seed: 11,
      vTop: fixture.vTop,
    });
  }, [fixture, lr, epochCommitted]);

  const cosAlign = aeResult.history.alignTop[epochCommitted - 1] ?? 0;
  const lossAt = aeResult.history.loss[epochCommitted - 1] ?? aeResult.history.loss[0];

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.attr('viewBox', `0 0 ${W} ${H}`).selectAll('*').remove();
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      // -- Left panel: scatter + PCA / AE directions --
      const xMax = 1.2 * d3.max(fixture.X, (_, i) => Math.abs(i % 2 === 0 ? fixture.X[i] : 0))!;
      const yMax = 1.2 * d3.max(fixture.X, (_, i) => Math.abs(i % 2 === 1 ? fixture.X[i] : 0))!;
      const lim = Math.max(xMax, yMax, 4);
      const xs = d3.scaleLinear().domain([-lim, lim]).range([0, splitX]);
      const ys = d3.scaleLinear().domain([-lim, lim]).range([innerH, 0]);
      g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xs).ticks(4));
      g.append('g').call(d3.axisLeft(ys).ticks(4));

      // Scatter — sample 300 points to keep DOM light.
      const stride = Math.max(1, Math.floor(N_SAMPLES / 300));
      g.append('g')
        .selectAll('circle')
        .data(d3.range(0, N_SAMPLES, stride))
        .join('circle')
        .attr('cx', (i) => xs(fixture.X[i * 2]))
        .attr('cy', (i) => ys(fixture.X[i * 2 + 1]))
        .attr('r', 1.6)
        .attr('opacity', 0.5)
        .style('fill', paletteRL.data);

      // Top PCA direction (solid green).
      const arrow = (v0: number, v1: number, color: string, dash: string | null) => {
        const len = lim * 0.95;
        g.append('line')
          .attr('x1', xs(-len * v0))
          .attr('y1', ys(-len * v1))
          .attr('x2', xs(len * v0))
          .attr('y2', ys(len * v1))
          .attr('stroke-width', 2.4)
          .attr('stroke-dasharray', dash ?? 'none')
          .style('stroke', color);
      };
      arrow(fixture.vTop[0], fixture.vTop[1], paletteRL.optimum, null);
      // AE-recovered direction at current epoch (dashed orange).
      let norm = 0;
      for (let j = 0; j < 2; j++) norm += trimmedAe.W[j] * trimmedAe.W[j];
      norm = Math.sqrt(norm);
      if (norm > 1e-9) arrow(trimmedAe.W[0] / norm, trimmedAe.W[1] / norm, paletteRL.highlight, '6 4');

      g.append('text').attr('x', 6).attr('y', 14).attr('font-size', 10).style('fill', paletteRL.optimum).text('top PCA eigenvector');
      g.append('text').attr('x', 6).attr('y', 28).attr('font-size', 10).style('fill', paletteRL.highlight).text(`AE bottleneck (epoch ${epochCommitted})`);

      // -- Right panel: loss vs epoch + alignment curve --
      const rg = svg.append('g').attr('transform', `translate(${margin.left + splitX + 28},${margin.top})`);
      const rW = W - margin.left - margin.right - splitX - 28;
      const xe = d3.scaleLinear().domain([0, MAX_EPOCH]).range([0, rW]);
      const lossMax = Math.max(0.05, aeResult.history.loss[0] * 1.05);
      const yL = d3.scaleLinear().domain([0, lossMax]).range([innerH / 2 - 10, 0]);
      const yA = d3.scaleLinear().domain([0, 1.0]).range([innerH - 10, innerH / 2 + 10]);

      rg.append('g').attr('transform', `translate(0,${innerH / 2 - 10})`).call(d3.axisBottom(xe).ticks(4));
      rg.append('g').call(d3.axisLeft(yL).ticks(4));
      rg.append('g').attr('transform', `translate(0,${innerH - 10})`).call(d3.axisBottom(xe).ticks(4));
      rg.append('g').attr('transform', `translate(0,${innerH / 2 + 10})`).call(d3.axisLeft(yA).ticks(3));

      rg.append('text').attr('x', rW / 2).attr('y', -4).attr('text-anchor', 'middle').attr('font-size', 10).style('fill', 'currentColor').text('reconstruction loss');
      rg.append('text').attr('x', rW / 2).attr('y', innerH / 2 + 6).attr('text-anchor', 'middle').attr('font-size', 10).style('fill', 'currentColor').text('|cos∠(W, v_top)|');

      // Baldi-Hornik floor.
      rg.append('line').attr('x1', 0).attr('x2', rW).attr('y1', yL(fixture.lambda2)).attr('y2', yL(fixture.lambda2)).attr('stroke-width', 1).attr('stroke-dasharray', '4 3').attr('opacity', 0.7).style('stroke', paletteRL.optimum);
      rg.append('text').attr('x', rW - 4).attr('y', yL(fixture.lambda2) - 3).attr('text-anchor', 'end').attr('font-size', 9).style('fill', paletteRL.optimum).text(`λ_2 = ${fixture.lambda2.toFixed(2)}`);

      const lineL = d3.line<number>().x((_d, i) => xe(i)).y((d) => yL(d));
      rg.append('path').datum(Array.from(aeResult.history.loss)).attr('fill', 'none').attr('stroke-width', 1.8).attr('d', lineL).style('stroke', paletteRL.signal);

      const lineA = d3.line<number>().x((_d, i) => xe(i)).y((d) => yA(d));
      rg.append('path').datum(Array.from(aeResult.history.alignTop)).attr('fill', 'none').attr('stroke-width', 1.8).attr('d', lineA).style('stroke', paletteRL.nuisance);

      // Marker at current epoch.
      rg.append('line').attr('x1', xe(epochCommitted)).attr('x2', xe(epochCommitted)).attr('y1', 0).attr('y2', innerH - 10).attr('stroke-width', 1).attr('stroke-dasharray', '2 3').attr('opacity', 0.55).style('stroke', paletteRL.highlight);
    },
    [fixture, aeResult, trimmedAe, epochCommitted, W, H, innerH, splitX],
  );

  return (
    <div ref={containerRef} className="not-prose my-6">
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-2 text-sm">
        <label className="flex flex-col">
          <span>Epoch = {epochDisplay}</span>
          <input
            type="range"
            min={1}
            max={MAX_EPOCH}
            step={1}
            value={epochDisplay}
            onChange={(e) => setEpochDisplay(Number(e.target.value))}
            onMouseUp={(e) => setEpochCommitted(Number((e.target as HTMLInputElement).value))}
            onTouchEnd={(e) => setEpochCommitted(Number((e.target as HTMLInputElement).value))}
            onKeyUp={(e) => setEpochCommitted(Number((e.target as HTMLInputElement).value))}
            aria-label="Training epoch"
          />
        </label>
        <label className="flex flex-col">
          <span>Learning rate = {lr.toFixed(3)}</span>
          <input
            type="range"
            min={0.005}
            max={0.2}
            step={0.005}
            value={lr}
            onChange={(e) => setLr(Number(e.target.value))}
            aria-label="Learning rate"
          />
        </label>
      </div>
      <svg ref={svgRef} className="w-full" style={{ maxHeight: H }} role="img" aria-label="Linear AE converges to PCA"/>
      <div className="text-xs opacity-75 mt-1 text-center">
        At epoch {epochCommitted}: loss = <strong>{lossAt.toFixed(3)}</strong> (Baldi-Hornik floor λ_2 = <strong>{fixture.lambda2.toFixed(3)}</strong>); |cos∠(W, v_top)| = <strong>{cosAlign.toFixed(3)}</strong>. Move the epoch slider to scrub through training and watch the bottleneck direction align with the top eigenvector.
      </div>
    </div>
  );
}
