import { useMemo } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { paletteVC, discretizedHalflineRestrictedSize } from './shared/vc-dimension';

// =============================================================================
// CardinalityBlowup — §1.5
//
// Log-log: |H_k| (red, diverging) vs |H_k|_S| (blue, plateau at n+1) for the
// discretized half-line class. Static curves; reader compares the two scales.
// Static fallback: public/images/topics/vc-dimension/01_motivation_card_blowup.png
// =============================================================================

const HEIGHT = 360;
const SAMPLE = 20;

export default function CardinalityBlowup() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();

  const { data, plateauN } = useMemo(() => {
    const ks: number[] = [];
    for (let exp = 1; exp <= 5; exp += 0.1) ks.push(Math.round(Math.pow(10, exp)));
    const uniqueKs = Array.from(new Set(ks)).sort((a, b) => a - b);
    // Deterministic sample of 20 evenly-spaced points on [0.025, 0.975].
    const samplePoints: number[] = [];
    for (let i = 0; i < SAMPLE; i++) samplePoints.push(0.025 + (i + 0.5) * 0.95 / SAMPLE);
    const series = uniqueKs.map((k) => ({
      k,
      full: k, // |H_k| = k thresholds
      restricted: discretizedHalflineRestrictedSize(k, samplePoints),
    }));
    return { data: series, plateauN: SAMPLE + 1 };
  }, []);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const margin = { top: 22, right: 28, bottom: 48, left: 64 };
      const W = w - margin.left - margin.right;
      const H = HEIGHT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${w} ${HEIGHT}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const x = d3.scaleLog().domain([10, 1e5]).range([0, W]);
      const yMax = Math.max(1e5, plateauN * 1.5);
      const y = d3.scaleLog().domain([1, yMax]).range([H, 0]);

      g.append('g').attr('transform', `translate(0,${H})`).call(d3.axisBottom(x).ticks(5, '~s'));
      g.append('g').call(d3.axisLeft(y).ticks(6, '~s'));
      g.append('text').attr('x', W / 2).attr('y', H + 36).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '12px').text('number of thresholds k (log scale)');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -H / 2).attr('y', -48).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '12px').text('cardinality (log scale)');

      const lineFull = d3.line<typeof data[number]>().x((d) => x(d.k)).y((d) => y(d.full));
      const lineR = d3.line<typeof data[number]>().x((d) => x(d.k)).y((d) => y(d.restricted));

      g.append('path').datum(data).attr('d', lineFull).attr('fill', 'none').attr('stroke', paletteVC.emp).attr('stroke-width', 2.4);
      g.append('path').datum(data).attr('d', lineR).attr('fill', 'none').attr('stroke', paletteVC.primary).attr('stroke-width', 2.4);

      // Plateau annotation
      g.append('line').attr('x1', 0).attr('x2', W).attr('y1', y(plateauN)).attr('y2', y(plateauN)).attr('stroke', paletteVC.primary).attr('stroke-dasharray', '3 4').attr('opacity', 0.4);
      g.append('text').attr('x', W - 6).attr('y', y(plateauN) - 6).attr('text-anchor', 'end').style('fill', paletteVC.primary).style('font-size', '11px').text(`n + 1 = ${plateauN}`);

      const legend = g.append('g').attr('transform', `translate(${Math.max(8, W - 240)}, 8)`);
      const items = [
        { label: '|H_k| (full class)', color: paletteVC.emp },
        { label: '|H_k|_S| (restricted to n = 20 points)', color: paletteVC.primary },
      ];
      items.forEach((it, i) => {
        legend.append('line').attr('x1', 0).attr('x2', 18).attr('y1', i * 16 + 6).attr('y2', i * 16 + 6).attr('stroke', it.color).attr('stroke-width', 2.4);
        legend.append('text').attr('x', 24).attr('y', i * 16 + 10).style('fill', 'var(--color-text)').style('font-size', '11px').text(it.label);
      });
    },
    [data, plateauN, containerWidth],
  );

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <svg ref={svgRef} width="100%" height={HEIGHT} role="img" aria-label="Log-log plot of full versus restricted cardinality for discretized half-lines" />
      <p style={{ fontSize: '0.85rem', color: 'var(--color-text-secondary)', marginTop: '0.4rem' }}>
        Adding more thresholds inflates |H_k| (red) without adding any new dichotomies once every gap is covered — the restricted cardinality |H_k|_S| (blue) saturates at n + 1.
      </p>
    </div>
  );
}
