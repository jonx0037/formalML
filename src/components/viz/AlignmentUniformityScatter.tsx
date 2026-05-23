import { useEffect, useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import { paletteRL } from './shared/representation-learning';

// =============================================================================
// AlignmentUniformityScatter — §5.5 / §12.2 companion. Loads
//   /sample-data/representation-learning/alignment_uniformity_trajectory.json
// (per-epoch Wang-Isola L_align and L_uniform from a small contrastive run)
// and renders the trajectory in the (L_align, L_uniform) plane plus two
// per-epoch line plots. Successful training drives both metrics toward the
// joint minimizer in the lower-left.
// =============================================================================

interface Payload {
  seed: number;
  N_pairs: number;
  tau: number;
  epochs: number;
  record_every: number;
  align_history: number[];
  uniform_history: number[];
  nce_history: number[];
}

export default function AlignmentUniformityScatter() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [payload, setPayload] = useState<Payload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [epoch, setEpoch] = useState(0);

  useEffect(() => {
    let cancelled = false;
    fetch('/sample-data/representation-learning/alignment_uniformity_trajectory.json')
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((j: Payload) => {
        if (!cancelled) {
          setPayload(j);
          setEpoch(j.align_history.length - 1);
        }
      })
      .catch((e) => {
        if (!cancelled) setError(String(e));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const W = Math.max(360, Math.min(width || 720, 720));
  const H = 360;
  const margin = { top: 22, right: 24, bottom: 42, left: 56 };
  const splitX = Math.round((W - margin.left - margin.right) * 0.55);
  const innerH = H - margin.top - margin.bottom;

  const bounds = useMemo(() => {
    if (!payload) return null;
    const ah = payload.align_history;
    const uh = payload.uniform_history;
    return {
      aMin: Math.min(...ah) - 0.1,
      aMax: Math.max(...ah) + 0.1,
      uMin: Math.min(...uh) - 0.1,
      uMax: Math.max(...uh) + 0.1,
    };
  }, [payload]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.attr('viewBox', `0 0 ${W} ${H}`).selectAll('*').remove();
      if (!payload || !bounds) return;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      // -- Left: (L_align, L_uniform) trajectory scatter --
      const xs = d3.scaleLinear().domain([bounds.aMin, bounds.aMax]).range([0, splitX]);
      const ys = d3.scaleLinear().domain([bounds.uMin, bounds.uMax]).range([innerH, 0]);
      g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xs).ticks(5));
      g.append('g').call(d3.axisLeft(ys).ticks(5));
      g.append('text').attr('x', splitX / 2).attr('y', innerH + 32).attr('text-anchor', 'middle').attr('font-size', 11).style('fill', 'currentColor').text('L_align');
      g.append('text').attr('x', -innerH / 2).attr('y', -42).attr('text-anchor', 'middle').attr('transform', 'rotate(-90)').attr('font-size', 11).style('fill', 'currentColor').text('L_uniform');

      // Trajectory path.
      const path = d3.line<number>().x((_d, i) => xs(payload.align_history[i])).y((_d, i) => ys(payload.uniform_history[i]));
      g.append('path').datum(payload.align_history).attr('fill', 'none').attr('stroke-width', 1.6).attr('d', path).style('stroke', paletteRL.data);
      // Epoch dots with color by training progress (early = red, late = green).
      g.append('g')
        .selectAll('circle')
        .data(payload.align_history)
        .join('circle')
        .attr('cx', (_d, i) => xs(payload.align_history[i]))
        .attr('cy', (_d, i) => ys(payload.uniform_history[i]))
        .attr('r', 2.5)
        .style('fill', (_d, i) => d3.interpolateViridis(i / Math.max(1, payload.align_history.length - 1)))
        .style('fill-opacity', 0.75);
      // Selected epoch.
      const aSel = payload.align_history[epoch];
      const uSel = payload.uniform_history[epoch];
      g.append('circle').attr('cx', xs(aSel)).attr('cy', ys(uSel)).attr('r', 6).attr('stroke-width', 1.5).style('fill', paletteRL.highlight).style('stroke', paletteRL.highlight).style('fill-opacity', 0.85);

      // -- Right: align + uniform vs epoch line plots --
      const rg = svg.append('g').attr('transform', `translate(${margin.left + splitX + 28},${margin.top})`);
      const rW = W - margin.left - margin.right - splitX - 28;
      const xe = d3.scaleLinear().domain([0, payload.align_history.length - 1]).range([0, rW]);
      // Two stacked axes — top half is L_align, bottom half is L_uniform.
      const yTopH = innerH / 2 - 8;
      const yBotH = innerH - innerH / 2 - 8;
      const yA = d3.scaleLinear().domain([bounds.aMin, bounds.aMax]).range([yTopH, 0]);
      const yU = d3.scaleLinear().domain([bounds.uMin, bounds.uMax]).range([innerH, innerH / 2 + 8]);
      rg.append('g').attr('transform', `translate(0,${yTopH})`).call(d3.axisBottom(xe).ticks(4));
      rg.append('g').call(d3.axisLeft(yA).ticks(3));
      rg.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xe).ticks(4));
      rg.append('g').attr('transform', `translate(0,${innerH / 2 + 8})`).call(d3.axisLeft(yU).ticks(3));
      rg.append('text').attr('x', rW / 2).attr('y', -4).attr('text-anchor', 'middle').attr('font-size', 10).style('fill', paletteRL.signal).text('L_align (negative = aligned)');
      rg.append('text').attr('x', rW / 2).attr('y', innerH / 2 + 4).attr('text-anchor', 'middle').attr('font-size', 10).style('fill', paletteRL.nuisance).text('L_uniform');

      const lineA = d3.line<number>().x((_d, i) => xe(i)).y((d) => yA(d));
      rg.append('path').datum(payload.align_history).attr('fill', 'none').attr('stroke-width', 1.6).attr('d', lineA).style('stroke', paletteRL.signal);
      const lineU = d3.line<number>().x((_d, i) => xe(i)).y((d) => yU(d));
      rg.append('path').datum(payload.uniform_history).attr('fill', 'none').attr('stroke-width', 1.6).attr('d', lineU).style('stroke', paletteRL.nuisance);
      // Epoch marker.
      rg.append('line').attr('x1', xe(epoch)).attr('x2', xe(epoch)).attr('y1', 0).attr('y2', innerH).attr('stroke-width', 1).attr('stroke-dasharray', '3 3').attr('opacity', 0.55).style('stroke', paletteRL.highlight);
    },
    [payload, bounds, epoch, W, H, innerH, splitX],
  );

  return (
    <div ref={containerRef} className="not-prose my-6">
      <div className="flex flex-wrap items-center gap-3 mb-2 text-sm">
        {payload ? (
          <label className="flex flex-col">
            <span>Training epoch = {epoch * payload.record_every} / {payload.epochs}</span>
            <input
              type="range"
              min={0}
              max={payload.align_history.length - 1}
              step={1}
              value={epoch}
              onChange={(e) => setEpoch(Number(e.target.value))}
              aria-label="Training epoch"
            />
          </label>
        ) : (
          <span className="opacity-60">Loading trajectory…</span>
        )}
      </div>
      {error ? (
        <div className="text-sm text-red-600">Failed to load trajectory: {error}</div>
      ) : (
        <svg ref={svgRef} className="w-full" style={{ maxHeight: H }} role="img" aria-label="Wang-Isola alignment-uniformity trajectory"/>
      )}
      {payload && (
        <div className="text-xs opacity-75 mt-1 text-center">
          At epoch {epoch * payload.record_every}: L<sub>align</sub> = <strong>{payload.align_history[epoch].toFixed(3)}</strong>, L<sub>uniform</sub> = <strong>{payload.uniform_history[epoch].toFixed(3)}</strong>, L<sub>NCE</sub> = <strong>{payload.nce_history[epoch].toFixed(3)}</strong>. Successful contrastive training pushes the point toward the lower-left (low alignment + low uniformity = both objectives minimized).
        </div>
      )}
    </div>
  );
}
