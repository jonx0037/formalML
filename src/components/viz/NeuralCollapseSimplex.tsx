import { useEffect, useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import { paletteRL, simplexETFTarget } from './shared/representation-learning';

// =============================================================================
// NeuralCollapseSimplex — §12.3 companion. Loads
//   /sample-data/representation-learning/neural_collapse_trajectory.json
// (per-epoch NC1 and NC2 from a supervised 8-class classifier trained to
// interpolation) and shows three panels:
//   • NC1: tr(Σ_W)/tr(Σ_B) collapsing to ~0
//   • NC2: mean off-diagonal cosine of class means converging to -1/(K-1)
//   • Final cosine-similarity matrix (heatmap) of class means
// =============================================================================

interface Payload {
  K: number;
  epochs: number;
  etf_target: number;
  nc1_history: number[];
  nc2_history: number[];
  acc_history: number[];
  loss_history: number[];
  final_cosine_matrix: number[][];
  final_features_2d: number[][];
  labels: number[];
}

export default function NeuralCollapseSimplex() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [payload, setPayload] = useState<Payload | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch('/sample-data/representation-learning/neural_collapse_trajectory.json')
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
  const H = 400;
  const margin = { top: 22, right: 22, bottom: 38, left: 50 };
  const colW = (W - margin.left - margin.right - 24) / 3;
  const innerH = H - margin.top - margin.bottom;

  const target = useMemo(() => (payload ? simplexETFTarget(payload.K) : -1 / 7), [payload]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.attr('viewBox', `0 0 ${W} ${H}`).selectAll('*').remove();
      if (!payload) return;

      // -- Panel 1: NC1 = tr(Σ_W)/tr(Σ_B) collapse --
      const g1 = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
      const xe = d3.scaleLinear().domain([0, payload.epochs - 1]).range([0, colW]);
      const nc1Max = Math.max(0.5, Math.max(...payload.nc1_history.slice(0, 20)) * 1.1);
      const yNC1 = d3.scaleLog().domain([Math.max(1e-3, Math.min(...payload.nc1_history)), nc1Max]).range([innerH, 0]).clamp(true);
      g1.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xe).ticks(4));
      g1.append('g').call(d3.axisLeft(yNC1).ticks(4, '~g'));
      g1.append('text').attr('x', colW / 2).attr('y', -6).attr('text-anchor', 'middle').attr('font-size', 10).style('fill', paletteRL.signal).text('NC1 = tr(Σ_W) / tr(Σ_B)');
      const line1 = d3.line<number>().x((_d, i) => xe(i)).y((d) => yNC1(Math.max(d, 1e-3)));
      g1.append('path').datum(payload.nc1_history).attr('fill', 'none').attr('stroke-width', 1.6).attr('d', line1).style('stroke', paletteRL.signal);

      // -- Panel 2: NC2 = mean off-diag cosine of class means → ETF target --
      const g2 = svg.append('g').attr('transform', `translate(${margin.left + colW + 12},${margin.top})`);
      const nc2Min = Math.min(target - 0.1, Math.min(...payload.nc2_history));
      const nc2Max = Math.max(payload.nc2_history[0] + 0.1, 0.2);
      const yNC2 = d3.scaleLinear().domain([nc2Min, nc2Max]).range([innerH, 0]);
      g2.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xe).ticks(4));
      g2.append('g').call(d3.axisLeft(yNC2).ticks(5));
      g2.append('text').attr('x', colW / 2).attr('y', -6).attr('text-anchor', 'middle').attr('font-size', 10).style('fill', paletteRL.nuisance).text(`NC2 = mean off-diag cos → -1/(K-1) = ${target.toFixed(3)}`);
      // Target line.
      g2.append('line').attr('x1', 0).attr('x2', colW).attr('y1', yNC2(target)).attr('y2', yNC2(target)).attr('stroke-width', 1).attr('stroke-dasharray', '4 3').attr('opacity', 0.7).style('stroke', paletteRL.optimum);
      const line2 = d3.line<number>().x((_d, i) => xe(i)).y((d) => yNC2(d));
      g2.append('path').datum(payload.nc2_history).attr('fill', 'none').attr('stroke-width', 1.6).attr('d', line2).style('stroke', paletteRL.nuisance);

      // -- Panel 3: final cosine matrix heatmap --
      const g3 = svg.append('g').attr('transform', `translate(${margin.left + 2 * colW + 24},${margin.top})`);
      const M = payload.final_cosine_matrix;
      const cellW = (colW - 16) / payload.K;
      const cellH = (innerH - 16) / payload.K;
      const color = d3.scaleLinear<string>().domain([-0.5, 0, 1]).range([paletteRL.nuisance, '#ffffff', paletteRL.signal]).clamp(true);
      g3.append('g')
        .selectAll('rect')
        .data(M.flatMap((row, i) => row.map((v, j) => ({ i, j, v }))))
        .join('rect')
        .attr('x', (d) => 8 + d.j * cellW)
        .attr('y', (d) => 8 + d.i * cellH)
        .attr('width', cellW)
        .attr('height', cellH)
        .attr('stroke-width', 0.4)
        .attr('stroke', '#fff')
        .style('fill', (d) => color(d.v));
      g3.append('text').attr('x', colW / 2).attr('y', -6).attr('text-anchor', 'middle').attr('font-size', 10).style('fill', 'currentColor').text(`final cos matrix (K = ${payload.K})`);
      // Legend.
      g3.append('text').attr('x', 8).attr('y', innerH - 2).attr('font-size', 8).style('fill', paletteRL.nuisance).text('-0.5');
      g3.append('text').attr('x', colW - 8).attr('y', innerH - 2).attr('text-anchor', 'end').attr('font-size', 8).style('fill', paletteRL.signal).text('+1.0');
    },
    [payload, target, W, H, innerH, colW],
  );

  return (
    <div ref={containerRef} className="not-prose my-6">
      {error ? (
        <div className="text-sm text-red-600">Failed to load neural-collapse trajectory: {error}</div>
      ) : (
        <svg ref={svgRef} className="w-full" style={{ maxHeight: H }} role="img" aria-label="Neural collapse trajectory + final cosine matrix"/>
      )}
      {payload ? (
        <div className="text-xs opacity-75 mt-1 text-center">
          Trained {payload.K}-way classifier for {payload.epochs} epochs. Final NC1 = <strong>{payload.nc1_history[payload.nc1_history.length - 1].toExponential(2)}</strong> (within-class variance collapse); final NC2 = <strong>{payload.nc2_history[payload.nc2_history.length - 1].toFixed(3)}</strong> vs ETF target <strong>{target.toFixed(3)}</strong>. The simplex ETF is the minimum-energy K-class configuration on the sphere — supervised cross-entropy converges to it from any initialization.
        </div>
      ) : (
        <div className="text-xs opacity-60 mt-1 text-center">Loading neural-collapse trajectory…</div>
      )}
    </div>
  );
}
