import { useEffect, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import { paletteRL } from './shared/representation-learning';

// =============================================================================
// SufficiencyToContrastiveBridge — §12.4 companion. Loads
//   /sample-data/representation-learning/simclr_embeddings.json
// (h, z, and raw class-mean cosines from the §6 SimCLR run on the
// 4-label × 4-color fixture) and shows the three-way convergence:
//
//   • raw X class-mean cosine     (sufficient-statistic baseline)
//   • SimCLR encoder h cosine     (self-supervised representation)
//   • SimCLR projector z cosine   (post-information-bottleneck output)
//   • dashed line: simplex-ETF target -1/(K-1)
//
// The three bars cluster near the ETF target — the topic's closing thesis.
// =============================================================================

interface Payload {
  K_labels: number;
  h_mean_cosine: number;
  z_mean_cosine: number;
  x_mean_cosine: number;
  etf_target: number;
}

export default function SufficiencyToContrastiveBridge() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [payload, setPayload] = useState<Payload | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch('/sample-data/representation-learning/simclr_embeddings.json')
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

  const W = Math.max(360, Math.min(width || 600, 600));
  const H = 320;
  const margin = { top: 22, right: 24, bottom: 56, left: 50 };
  const innerW = W - margin.left - margin.right;
  const innerH = H - margin.top - margin.bottom;

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.attr('viewBox', `0 0 ${W} ${H}`).selectAll('*').remove();
      if (!payload) return;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
      const bars = [
        { label: 'raw X', value: payload.x_mean_cosine, kind: 'data' },
        { label: 'SimCLR h (encoder)', value: payload.h_mean_cosine, kind: 'signal' },
        { label: 'SimCLR z (projector)', value: payload.z_mean_cosine, kind: 'nuisance' },
      ];
      const xs = d3.scaleBand<string>().domain(bars.map((b) => b.label)).range([0, innerW]).padding(0.25);
      const yMin = Math.min(payload.etf_target - 0.15, ...bars.map((b) => b.value)) - 0.05;
      const yMax = Math.max(0.05, ...bars.map((b) => b.value)) + 0.1;
      const ys = d3.scaleLinear().domain([yMin, yMax]).range([innerH, 0]);

      g.append('g').attr('transform', `translate(0,${ys(0)})`).call(d3.axisBottom(xs).tickSize(0).tickPadding(8));
      g.selectAll('g.tick text').attr('font-size', 10);
      g.append('g').call(d3.axisLeft(ys).ticks(6));
      g.append('text').attr('x', -innerH / 2).attr('y', -38).attr('text-anchor', 'middle').attr('transform', 'rotate(-90)').attr('font-size', 11).style('fill', 'currentColor').text('Mean off-diagonal cosine of class means');

      // ETF target reference line.
      g.append('line').attr('x1', 0).attr('x2', innerW).attr('y1', ys(payload.etf_target)).attr('y2', ys(payload.etf_target)).attr('stroke-width', 1.4).attr('stroke-dasharray', '5 3').attr('opacity', 0.85).style('stroke', paletteRL.optimum);
      g.append('text').attr('x', innerW - 4).attr('y', ys(payload.etf_target) - 4).attr('text-anchor', 'end').attr('font-size', 10).style('fill', paletteRL.optimum).text(`simplex ETF target = -1/(K-1) = ${payload.etf_target.toFixed(3)}`);

      // Bars (rooted at zero; negative values extend downward).
      g.selectAll('rect.bar')
        .data(bars)
        .join('rect')
        .attr('class', 'bar')
        .attr('x', (d) => xs(d.label)!)
        .attr('y', (d) => (d.value >= 0 ? ys(d.value) : ys(0)))
        .attr('width', xs.bandwidth())
        .attr('height', (d) => Math.abs(ys(d.value) - ys(0)))
        .style('fill', (d) => paletteRL[d.kind as keyof typeof paletteRL])
        .style('fill-opacity', 0.85);

      // Value labels.
      g.selectAll('text.val')
        .data(bars)
        .join('text')
        .attr('class', 'val')
        .attr('x', (d) => xs(d.label)! + xs.bandwidth() / 2)
        .attr('y', (d) => ys(d.value) - 6)
        .attr('text-anchor', 'middle')
        .attr('font-size', 11)
        .style('fill', 'currentColor')
        .text((d) => d.value.toFixed(3));
    },
    [payload, W, H, innerW, innerH],
  );

  return (
    <div ref={containerRef} className="not-prose my-6">
      {error ? (
        <div className="text-sm text-red-600">Failed to load SimCLR embeddings: {error}</div>
      ) : (
        <svg ref={svgRef} className="w-full" style={{ maxHeight: H }} role="img" aria-label="Sufficiency-to-contrastive bridge"/>
      )}
      {payload ? (
        <div className="text-xs opacity-75 mt-1 text-center">
          K = {payload.K_labels} latent classes. The raw-data class-mean cosine (from the Gaussian-mixture sufficient statistic), the encoder h cosine, and the projector z cosine all sit near the simplex-ETF target <strong>{payload.etf_target.toFixed(3)}</strong> — the three-way convergence of §12.4. Classical sufficiency, supervised neural collapse, and unsupervised contrastive learning all arrive at the same geometric optimum under progressively weaker information assumptions.
        </div>
      ) : (
        <div className="text-xs opacity-60 mt-1 text-center">Loading SimCLR embeddings…</div>
      )}
    </div>
  );
}
