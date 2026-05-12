import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { mulberry32, symmetrizationStats } from './shared/generalization-bounds';

// =============================================================================
// SymmetrizationVisualizer — §6.  Three histograms of the symmetrization chain:
//   1. raw sup_τ (Pf − P_n f)
//   2. ghost-paired sup_τ (P_n' f − P_n f)
//   3. σ-symmetrized sup_τ (1/n) Σ σ_i (h(X'_i) − h(X_i))
// with 2 R̂_S(H) reference line on the third panel.
//
// Commit-on-release n slider; B = 80 replicates by default.
// =============================================================================

const HEIGHT = 280;
const SM_BREAKPOINT = 1100;
const DEFAULT_B = 80;
const DEFAULT_N = 80;

export default function SymmetrizationVisualizer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [displayN, setDisplayN] = useState(DEFAULT_N);
  const [committedN, setCommittedN] = useState(DEFAULT_N);
  const [seed, setSeed] = useState(20260511);

  const isStacked = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const panelWidth = isStacked ? containerWidth : Math.floor(containerWidth / 3);

  const stats = useMemo(() => {
    const rng = mulberry32(seed);
    return symmetrizationStats(committedN, DEFAULT_B, rng);
  }, [committedN, seed]);

  const mean = (a: Float64Array) => a.reduce((s, x) => s + x, 0) / a.length;

  function renderHistogram(
    svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
    data: Float64Array,
    color: string,
    title: string,
    extra: { line?: { x: number; label: string; color: string } } = {},
  ) {
    svg.selectAll('*').remove();
    if (panelWidth <= 0) return;
    const margin = { top: 32, right: 12, bottom: 36, left: 40 };
    const w = panelWidth - margin.left - margin.right;
    const h = HEIGHT - margin.top - margin.bottom;
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const lo = Math.min(0, d3.min(data) ?? 0);
    const hi = Math.max(d3.max(data) ?? 0.3, 0.3);
    const x = d3.scaleLinear().domain([lo, hi]).range([0, w]);
    const bins = d3.bin().domain(x.domain() as [number, number]).thresholds(18)(Array.from(data));
    const y = d3.scaleLinear().domain([0, d3.max(bins, (b) => b.length) ?? 1]).range([h, 0]);

    g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(4));
    g.append('g').call(d3.axisLeft(y).ticks(4));
    g.append('text').attr('x', w / 2).attr('y', -12).attr('text-anchor', 'middle')
      .style('font-size', '11px').style('font-weight', '600').style('fill', 'var(--color-text)').text(title);

    const m = mean(data);
    g.selectAll('rect.bin').data(bins).enter().append('rect')
      .attr('x', (d) => x(d.x0 ?? 0) + 1)
      .attr('width', (d) => Math.max(0, x(d.x1 ?? 0) - x(d.x0 ?? 0) - 2))
      .attr('y', (d) => y(d.length))
      .attr('height', (d) => h - y(d.length))
      .style('fill', color)
      .attr('fill-opacity', 0.75);

    g.append('line').attr('x1', x(m)).attr('x2', x(m)).attr('y1', 0).attr('y2', h)
      .style('stroke', '#1f2937').style('stroke-width', 2);
    g.append('text').attr('x', x(m) + 4).attr('y', 14).style('font-size', '10px').style('fill', '#1f2937')
      .text(`mean ${m.toFixed(3)}`);

    if (extra.line) {
      g.append('line').attr('x1', x(extra.line.x)).attr('x2', x(extra.line.x)).attr('y1', 0).attr('y2', h)
        .style('stroke', extra.line.color).style('stroke-dasharray', '4 3').style('stroke-width', 1.5);
      g.append('text').attr('x', x(extra.line.x) + 4).attr('y', 30)
        .style('font-size', '10px').style('fill', extra.line.color).text(extra.line.label);
    }
  }

  const empRadMean = mean(stats.empRad);
  const ref1 = useD3<SVGSVGElement>(
    (svg) => renderHistogram(svg, stats.rawSup, '#3b82f6', 'sup_τ (Pf − P_n f)'),
    [stats, panelWidth],
  );
  const ref2 = useD3<SVGSVGElement>(
    (svg) => renderHistogram(svg, stats.ghostPairSup, '#10b981', 'sup_τ (P_n′ f − P_n f)'),
    [stats, panelWidth],
  );
  const ref3 = useD3<SVGSVGElement>(
    (svg) => renderHistogram(svg, stats.sigmaSymSup, '#f59e0b', 'sup_τ (1/n) Σ σᵢ (h(X′ᵢ) − h(Xᵢ))', {
      line: { x: 2 * empRadMean, label: `2 R̂_S(H) ≈ ${(2 * empRadMean).toFixed(3)}`, color: '#7c2d12' },
    }),
    [stats, panelWidth, empRadMean],
  );

  return (
    <figure ref={containerRef} className="my-8 not-prose">
      <div style={{ display: 'grid', gridTemplateColumns: isStacked ? '1fr' : '1fr 1fr 1fr', gap: 8 }}>
        <svg ref={ref1} width={panelWidth || 260} height={HEIGHT} role="img" aria-label="Raw sup (Pf − Pn f) histogram." />
        <svg ref={ref2} width={panelWidth || 260} height={HEIGHT} role="img" aria-label="Ghost-paired sup histogram." />
        <svg ref={ref3} width={panelWidth || 260} height={HEIGHT} role="img" aria-label="Sigma-symmetrized sup histogram with 2 Rademacher reference line." />
      </div>
      <figcaption style={{ marginTop: 12, display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'center' }}>
        <label style={{ fontSize: 12, color: 'var(--color-text-secondary)' }}>
          n:
          <input
            type="range" min={20} max={300} step={10} value={displayN}
            onChange={(e) => setDisplayN(parseInt(e.target.value, 10))}
            onMouseUp={() => setCommittedN(displayN)}
            onTouchEnd={() => setCommittedN(displayN)}
            onKeyUp={() => setCommittedN(displayN)}
            style={{ marginLeft: 8, verticalAlign: 'middle' }}
            aria-label="sample size"
          />
          <span style={{ marginLeft: 6 }}>{displayN}</span>
        </label>
        <button
          type="button"
          onClick={() => setSeed((s) => s + 1)}
          style={{
            fontSize: 12, padding: '4px 10px', borderRadius: 6,
            border: '1px solid var(--color-border)',
            background: 'var(--color-surface)', color: 'var(--color-text)',
            cursor: 'pointer',
          }}
        >
          re-roll
        </button>
      </figcaption>
    </figure>
  );
}
