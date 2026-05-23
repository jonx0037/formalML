import { useEffect, useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import { paletteRL } from './shared/representation-learning';

// =============================================================================
// VAELatentSpaceExplorer — §4.3 companion. Loads
//   /sample-data/representation-learning/vae_decoder.json
// (encoder posterior means, scatter labels, decoder x-grid for a regular
// z-grid) and renders the latent space. The reader picks (z₁, z₂) by clicking;
// the side panel displays the decoder's mean reconstruction at that latent.
// =============================================================================

interface Payload {
  K: number;
  D: number;
  Z: number;
  beta: number;
  z_range: number;
  grid: number;
  scatter_mu: number[][];
  scatter_labels: number[];
  grid_z: number[][];
  grid_x: number[][];
}

export default function VAELatentSpaceExplorer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [payload, setPayload] = useState<Payload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [picked, setPicked] = useState<{ z0: number; z1: number } | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch('/sample-data/representation-learning/vae_decoder.json')
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((j: Payload) => {
        if (!cancelled) setPayload(j);
      })
      .catch((e) => {
        if (!cancelled) setError(String(e));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const W = Math.max(360, Math.min(width || 720, 720));
  const H = 420;
  const margin = { top: 22, right: 22, bottom: 36, left: 50 };
  const splitX = Math.round((W - margin.left - margin.right) * 0.6);
  const innerH = H - margin.top - margin.bottom;

  const colorScale = useMemo(() => {
    if (!payload) return null;
    return d3.scaleOrdinal<number, string>().domain(d3.range(payload.K)).range(d3.schemeTableau10);
  }, [payload]);

  // Decode at the nearest grid point for the picked latent (cheap; no JS-side neural net).
  const decoded = useMemo(() => {
    if (!payload || !picked) return null;
    let bestIdx = 0;
    let bestDist = Infinity;
    for (let i = 0; i < payload.grid_z.length; i++) {
      const dz0 = payload.grid_z[i][0] - picked.z0;
      const dz1 = payload.grid_z[i][1] - picked.z1;
      const d2 = dz0 * dz0 + dz1 * dz1;
      if (d2 < bestDist) {
        bestDist = d2;
        bestIdx = i;
      }
    }
    return { idx: bestIdx, x: payload.grid_x[bestIdx], z: payload.grid_z[bestIdx] };
  }, [payload, picked]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.attr('viewBox', `0 0 ${W} ${H}`).selectAll('*').remove();
      if (!payload || !colorScale) return;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
      const r = payload.z_range;
      const xs = d3.scaleLinear().domain([-r, r]).range([0, splitX]);
      const ys = d3.scaleLinear().domain([-r, r]).range([innerH, 0]);
      g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xs).ticks(4));
      g.append('g').call(d3.axisLeft(ys).ticks(4));

      // Click target.
      g.append('rect')
        .attr('x', 0).attr('y', 0).attr('width', splitX).attr('height', innerH)
        .attr('fill', 'transparent')
        .style('cursor', 'crosshair')
        .on('click', (event: MouseEvent) => {
          const [mx, my] = d3.pointer(event);
          setPicked({ z0: xs.invert(mx), z1: ys.invert(my) });
        });

      // Scatter.
      g.append('g')
        .selectAll('circle')
        .data(payload.scatter_mu)
        .join('circle')
        .attr('cx', (d) => xs(d[0]))
        .attr('cy', (d) => ys(d[1]))
        .attr('r', 2)
        .attr('opacity', 0.7)
        .style('fill', (_d, i) => colorScale(payload.scatter_labels[i])!);

      // Selected point.
      if (picked) {
        g.append('circle')
          .attr('cx', xs(picked.z0))
          .attr('cy', ys(picked.z1))
          .attr('r', 7)
          .attr('stroke-width', 2)
          .attr('fill', 'none')
          .style('stroke', paletteRL.highlight);
      }
      g.append('text').attr('x', splitX / 2).attr('y', -4).attr('text-anchor', 'middle').attr('font-size', 10).style('fill', 'currentColor').text(`latent z (click anywhere)`);

      // -- Right: decoded bar chart for the picked z --
      const rg = svg.append('g').attr('transform', `translate(${margin.left + splitX + 28},${margin.top})`);
      const rW = W - margin.left - margin.right - splitX - 28;
      if (decoded) {
        const D = decoded.x.length;
        const xb = d3.scaleBand<number>().domain(d3.range(D)).range([0, rW]).padding(0.15);
        const ymax = Math.max(...decoded.x.map(Math.abs)) * 1.1;
        const yb = d3.scaleLinear().domain([-ymax, ymax]).range([innerH, 0]);
        rg.append('g').attr('transform', `translate(0,${yb(0)})`).call(d3.axisBottom(xb).tickValues(d3.range(D)));
        rg.append('g').call(d3.axisLeft(yb).ticks(4));
        rg.append('text').attr('x', rW / 2).attr('y', -4).attr('text-anchor', 'middle').attr('font-size', 10).style('fill', 'currentColor').text(`decoded x = g_θ(z = [${decoded.z[0].toFixed(2)}, ${decoded.z[1].toFixed(2)}])`);
        rg.selectAll('rect.bar')
          .data(decoded.x)
          .join('rect')
          .attr('class', 'bar')
          .attr('x', (_v, i) => xb(i)!)
          .attr('y', (v) => (v >= 0 ? yb(v) : yb(0)))
          .attr('width', xb.bandwidth())
          .attr('height', (v) => Math.abs(yb(v) - yb(0)))
          .style('fill', paletteRL.signal)
          .style('fill-opacity', 0.85);
      } else {
        rg.append('text').attr('x', rW / 2).attr('y', innerH / 2).attr('text-anchor', 'middle').attr('font-size', 11).style('fill', 'currentColor').attr('opacity', 0.55).text('click a latent point to decode');
      }
    },
    [payload, colorScale, picked, decoded, W, H, innerH, splitX],
  );

  return (
    <div ref={containerRef} className="not-prose my-6">
      {error ? (
        <div className="text-sm text-red-600">Failed to load VAE decoder: {error}</div>
      ) : (
        <svg ref={svgRef} className="w-full" style={{ maxHeight: H }} role="img" aria-label="VAE latent space explorer"/>
      )}
      {payload ? (
        <div className="text-xs opacity-75 mt-1 text-center">
          β = {payload.beta}, latent dim = {payload.Z}, encoder posterior means over {payload.scatter_mu.length} training samples in {payload.K} classes. Click anywhere in the latent space to see the decoder's mean reconstruction at that z.
        </div>
      ) : (
        <div className="text-xs opacity-60 mt-1 text-center">Loading VAE decoder…</div>
      )}
    </div>
  );
}
