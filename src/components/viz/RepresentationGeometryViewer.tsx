import { useEffect, useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import { paletteRL } from './shared/representation-learning';

// =============================================================================
// RepresentationGeometryViewer — §1.2 / §12.1 companion. Loads
//   /sample-data/representation-learning/simclr_embeddings.json
// (raw / encoder-h / projector-z 2-D projections of the §6.2 label×color
// fixture). The reader toggles between {raw, PCA, encoder h, projector z}
// representations and between {label, color} colorings, observing how the
// contrastive loss reshapes the geometry around the label axis while
// (partially) preserving or discarding the color axis depending on which
// representation is being viewed.
// =============================================================================

type ViewKey = 'raw_2d' | 'h_2d' | 'z_2d';
type ColorBy = 'label' | 'color';

interface Payload {
  K_labels: number;
  K_colors: number;
  raw_2d: number[][];
  h_2d: number[][];
  z_2d: number[][];
  labels: number[];
  colors: number[];
}

const VIEW_LABEL: Record<ViewKey, string> = {
  raw_2d: 'raw X (PCA-2)',
  h_2d: 'SimCLR encoder h (PCA-2)',
  z_2d: 'SimCLR projector z (PCA-2)',
};

export default function RepresentationGeometryViewer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [payload, setPayload] = useState<Payload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<ViewKey>('raw_2d');
  const [colorBy, setColorBy] = useState<ColorBy>('label');

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
  const H = 480;
  const margin = { top: 22, right: 22, bottom: 36, left: 50 };
  const innerW = W - margin.left - margin.right;
  const innerH = H - margin.top - margin.bottom;

  const colorScale = useMemo(() => {
    if (!payload) return null;
    const K = colorBy === 'label' ? payload.K_labels : payload.K_colors;
    return d3.scaleOrdinal<number, string>().domain(d3.range(K)).range(d3.schemeTableau10);
  }, [payload, colorBy]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.attr('viewBox', `0 0 ${W} ${H}`).selectAll('*').remove();
      if (!payload || !colorScale) return;
      const pts = payload[view];
      const groups = colorBy === 'label' ? payload.labels : payload.colors;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
      const xExt = d3.extent(pts.map((p) => p[0])) as [number, number];
      const yExt = d3.extent(pts.map((p) => p[1])) as [number, number];
      const xp = (xExt[1] - xExt[0]) * 0.1;
      const yp = (yExt[1] - yExt[0]) * 0.1;
      const xs = d3.scaleLinear().domain([xExt[0] - xp, xExt[1] + xp]).range([0, innerW]);
      const ys = d3.scaleLinear().domain([yExt[0] - yp, yExt[1] + yp]).range([innerH, 0]);
      g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xs).ticks(4));
      g.append('g').call(d3.axisLeft(ys).ticks(4));

      g.append('g')
        .selectAll('circle')
        .data(pts)
        .join('circle')
        .attr('cx', (d) => xs(d[0]))
        .attr('cy', (d) => ys(d[1]))
        .attr('r', 2.5)
        .attr('opacity', 0.7)
        .style('fill', (_d, i) => colorScale(groups[i])!);
      g.append('text').attr('x', innerW / 2).attr('y', -6).attr('text-anchor', 'middle').attr('font-size', 11).style('fill', 'currentColor').text(`${VIEW_LABEL[view]}, colored by ${colorBy}`);
    },
    [payload, colorScale, view, colorBy, W, H, innerW, innerH],
  );

  return (
    <div ref={containerRef} className="not-prose my-6">
      <div className="flex flex-wrap items-center gap-3 mb-2 text-sm">
        <div>
          <span className="opacity-70 mr-1">view:</span>
          {(['raw_2d', 'h_2d', 'z_2d'] as ViewKey[]).map((k) => (
            <button
              key={k}
              type="button"
              className={`px-2 py-0.5 text-xs rounded border mr-1 ${view === k ? 'bg-gray-800 text-white' : 'bg-transparent'}`}
              onClick={() => setView(k)}
              aria-pressed={view === k}
            >
              {VIEW_LABEL[k]}
            </button>
          ))}
        </div>
        <div>
          <span className="opacity-70 mr-1">color by:</span>
          {(['label', 'color'] as ColorBy[]).map((k) => (
            <button
              key={k}
              type="button"
              className={`px-2 py-0.5 text-xs rounded border mr-1 ${colorBy === k ? 'bg-gray-800 text-white' : 'bg-transparent'}`}
              onClick={() => setColorBy(k)}
              aria-pressed={colorBy === k}
            >
              {k}
            </button>
          ))}
        </div>
      </div>
      {error ? (
        <div className="text-sm text-red-600">Failed to load embeddings: {error}</div>
      ) : (
        <svg ref={svgRef} className="w-full" style={{ maxHeight: H }} role="img" aria-label="Representation geometry viewer"/>
      )}
      {payload ? (
        <div className="text-xs opacity-75 mt-1 text-center">
          {payload.labels.length} samples, {payload.K_labels} labels × {payload.K_colors} colors. Toggle between representations and colorings: the encoder h preserves both label and color structure; the projector z compresses out the color axis (the projection-head observation of §6.2).
        </div>
      ) : (
        <div className="text-xs opacity-60 mt-1 text-center">Loading embeddings…</div>
      )}
    </div>
  );
}
