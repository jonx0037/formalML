import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  mulberry32,
  robinsonDGP,
} from './shared/causal-inference-methods';

// =============================================================================
// PropensityOverlapDiagnostic — §3.2 (positivity / overlap).
//
// Two-panel histogram of the propensity e(X) by treatment arm. A steepness
// slider scales the propensity logit (s ∈ [0.5, 5]). Larger s pushes mass
// toward {0, 1} and worsens overlap. Shaded regions in [0, δ] ∪ [1-δ, 1] mark
// where positivity is "practically violated" (default δ = 0.05).
//
// Static fallback: public/images/topics/causal-inference-methods/03_positivity_overlap.png
// =============================================================================

const HEIGHT = 340;
const N_SAMPLE = 1500;
const N_BINS = 28;
const SM_BREAKPOINT = 640;

function histogram(values: Float64Array, mask: Int8Array, target: 0 | 1, bins: number): number[] {
  const out = new Array<number>(bins).fill(0);
  for (let i = 0; i < values.length; i++) {
    if (mask[i] !== target) continue;
    const b = Math.min(Math.floor(values[i] * bins), bins - 1);
    out[b]++;
  }
  return out;
}

export default function PropensityOverlapDiagnostic() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [steepDisplay, setSteepDisplay] = useState(1.5);
  const [steepness, setSteepness] = useState(1.5);
  const [trim, setTrim] = useState(0.05);

  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;

  const { eVals, D, fracTrimmed } = useMemo(() => {
    const rng = mulberry32(20260512 + Math.round(steepness * 100));
    const sample = robinsonDGP(N_SAMPLE, 10, 1.0, rng, { propensitySteepness: steepness });
    let trimCount = 0;
    for (let i = 0; i < N_SAMPLE; i++) {
      if (sample.ePropensity[i] < trim || sample.ePropensity[i] > 1 - trim) trimCount++;
    }
    return { eVals: sample.ePropensity, D: sample.D, fracTrimmed: trimCount / N_SAMPLE };
  }, [steepness, trim]);

  const treatedHist = useMemo(() => histogram(eVals, D, 1, N_BINS), [eVals, D]);
  const controlHist = useMemo(() => histogram(eVals, D, 0, N_BINS), [eVals, D]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const margin = { top: 20, right: 24, bottom: 40, left: 56 };
      const W = w - margin.left - margin.right;
      const H = HEIGHT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${w} ${HEIGHT}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);
      const maxCount = Math.max(d3.max(treatedHist)!, d3.max(controlHist)!);
      const xScale = d3.scaleLinear().domain([0, 1]).range([0, W]);
      const yScale = d3.scaleLinear().domain([0, maxCount * 1.05 + 1]).range([H, 0]).nice();
      g.append('g').attr('transform', `translate(0, ${H})`).call(d3.axisBottom(xScale).ticks(6));
      g.append('g').call(d3.axisLeft(yScale).ticks(5));
      // Positivity-violation shaded regions.
      g.append('rect')
        .attr('x', xScale(0)).attr('y', 0)
        .attr('width', xScale(trim) - xScale(0)).attr('height', H)
        .style('fill', '#c0504d').style('opacity', 0.08);
      g.append('rect')
        .attr('x', xScale(1 - trim)).attr('y', 0)
        .attr('width', xScale(1) - xScale(1 - trim)).attr('height', H)
        .style('fill', '#c0504d').style('opacity', 0.08);
      // Histograms.
      const barWidth = (W / N_BINS) * 0.85;
      for (let b = 0; b < N_BINS; b++) {
        const left = b / N_BINS;
        const xb = xScale(left);
        // Control bar (left half).
        g.append('rect')
          .attr('x', xb).attr('y', yScale(controlHist[b]))
          .attr('width', barWidth / 2).attr('height', H - yScale(controlHist[b]))
          .style('fill', '#1f4e79').style('opacity', 0.7);
        // Treated bar (right half).
        g.append('rect')
          .attr('x', xb + barWidth / 2).attr('y', yScale(treatedHist[b]))
          .attr('width', barWidth / 2).attr('height', H - yScale(treatedHist[b]))
          .style('fill', '#c0504d').style('opacity', 0.7);
      }
      // Labels.
      g.append('text').attr('x', W / 2).attr('y', H + 32).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)').style('font-size', '11px')
        .text('Propensity e(X) = Pr(D = 1 | X)');
      g.append('text').attr('transform', `translate(-40, ${H / 2}) rotate(-90)`)
        .attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '11px')
        .text('Count per bin');
      // Legend.
      const lg = g.append('g').attr('transform', `translate(${W - 150}, 4)`);
      lg.append('rect').attr('x', 0).attr('y', 0).attr('width', 14).attr('height', 10).style('fill', '#1f4e79').style('opacity', 0.7);
      lg.append('text').attr('x', 18).attr('y', 10).style('fill', 'var(--color-text)').style('font-size', '10px').text('Control (D = 0)');
      lg.append('rect').attr('x', 0).attr('y', 20).attr('width', 14).attr('height', 10).style('fill', '#c0504d').style('opacity', 0.7);
      lg.append('text').attr('x', 18).attr('y', 30).style('fill', 'var(--color-text)').style('font-size', '10px').text('Treated (D = 1)');
    },
    [containerWidth, treatedHist, controlHist, trim],
  );

  const commitSteepness = () => setSteepness(steepDisplay);

  return (
    <div ref={containerRef} style={{ marginBlock: '1.25rem' }}>
      <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: '1.5rem', marginBottom: '0.75rem', flexWrap: 'wrap' }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.9rem', color: 'var(--color-text)' }}>
          Steepness s =
          <input
            type="range" min={0.5} max={5} step={0.1} value={steepDisplay}
            onChange={(e) => setSteepDisplay(parseFloat(e.target.value))}
            onMouseUp={commitSteepness}
            onTouchEnd={commitSteepness}
            onKeyUp={commitSteepness}
            aria-label="Propensity steepness"
            style={{ width: 180 }}
          />
          <span style={{ fontFamily: 'var(--font-mono)' }}>{steepDisplay.toFixed(1)}</span>
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.9rem', color: 'var(--color-text)' }}>
          Trim threshold δ =
          <input
            type="range" min={0} max={0.2} step={0.01} value={trim}
            onChange={(e) => setTrim(parseFloat(e.target.value))}
            aria-label="Trim threshold"
            style={{ width: 160 }}
          />
          <span style={{ fontFamily: 'var(--font-mono)' }}>{trim.toFixed(2)}</span>
        </label>
        <span style={{ fontSize: '0.85rem', color: 'var(--color-text-secondary)' }}>
          Fraction trimmed: <span style={{ fontFamily: 'var(--font-mono)' }}>{(fracTrimmed * 100).toFixed(1)}%</span>
        </span>
      </div>
      <svg ref={svgRef} style={{ width: '100%', height: HEIGHT }} />
    </div>
  );
}
