import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import { reparamVsScoreVariance, paletteRL } from './shared/representation-learning';

// =============================================================================
// ReparametrizationTrickDiagram — §4.4 companion. For a tiny single-example
// VAE loss f(z) = ½ ||z||² with q(z; μ, σ²) = N(μ, σ² I), the reparam
// estimator computes ∇_μ f via z = μ + σ ⊙ ε and the score-function
// estimator computes the same via the log-derivative trick. The reader sets
// (μ, log σ²) and the batch size B, the viz runs the MC variance comparison
// and plots per-coordinate empirical std-dev side by side. Notebook §4.4
// (cell 13) reports a 3-10× reparam advantage; this viz reproduces that.
// =============================================================================

export default function ReparametrizationTrickDiagram() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [muDisplay, setMuDisplay] = useState(0.5);
  const [muCommitted, setMuCommitted] = useState(0.5);
  const [logSigmaSqDisplay, setLogSigmaSqDisplay] = useState(-0.5);
  const [logSigmaSqCommitted, setLogSigmaSqCommitted] = useState(-0.5);
  const [B, setB] = useState(200);

  const W = Math.max(360, Math.min(width || 720, 720));
  const H = 360;
  const margin = { top: 24, right: 24, bottom: 60, left: 56 };
  const innerW = W - margin.left - margin.right;
  const innerH = H - margin.top - margin.bottom;

  const data = useMemo(() => {
    const mu = new Float64Array([muCommitted, -muCommitted]);
    const logSigmaSq = new Float64Array([logSigmaSqCommitted, logSigmaSqCommitted]);
    const r = reparamVsScoreVariance(mu, logSigmaSq, { B, seed: 7 });
    return [
      { label: '∂L/∂μ₀  reparam', value: r.reparamMuStd[0], kind: 'reparam' },
      { label: '∂L/∂μ₀  score', value: r.scoreMuStd[0], kind: 'score' },
      { label: '∂L/∂μ₁  reparam', value: r.reparamMuStd[1], kind: 'reparam' },
      { label: '∂L/∂μ₁  score', value: r.scoreMuStd[1], kind: 'score' },
      { label: '∂L/∂lnσ²₀  reparam', value: r.reparamLogSigmaStd[0], kind: 'reparam' },
      { label: '∂L/∂lnσ²₀  score', value: r.scoreLogSigmaStd[0], kind: 'score' },
      { label: '∂L/∂lnσ²₁  reparam', value: r.reparamLogSigmaStd[1], kind: 'reparam' },
      { label: '∂L/∂lnσ²₁  score', value: r.scoreLogSigmaStd[1], kind: 'score' },
    ];
  }, [muCommitted, logSigmaSqCommitted, B]);

  const ratio = useMemo(() => {
    const reparamMean = (data[0].value + data[2].value + data[4].value + data[6].value) / 4;
    const scoreMean = (data[1].value + data[3].value + data[5].value + data[7].value) / 4;
    return reparamMean > 0 ? scoreMean / reparamMean : Infinity;
  }, [data]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.attr('viewBox', `0 0 ${W} ${H}`).selectAll('*').remove();
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
      const ymax = d3.max(data, (d) => d.value)!;
      const ys = d3.scaleLinear().domain([0, ymax * 1.15]).range([innerH, 0]);
      const xs = d3.scaleBand<string>().domain(data.map((d) => d.label)).range([0, innerW]).padding(0.18);

      g.append('g').attr('transform', `translate(0,${innerH})`).call(
        d3.axisBottom(xs).tickSize(0).tickPadding(6),
      );
      g.selectAll('g.tick text').attr('font-size', 8).attr('transform', 'rotate(-30)').attr('text-anchor', 'end');
      g.append('g').call(d3.axisLeft(ys).ticks(6));

      g.append('text')
        .attr('x', -innerH / 2)
        .attr('y', -42)
        .attr('text-anchor', 'middle')
        .attr('transform', 'rotate(-90)')
        .attr('font-size', 11)
        .style('fill', 'currentColor')
        .text('Empirical std-dev across B MC samples');

      g.selectAll('rect.bar')
        .data(data)
        .join('rect')
        .attr('class', 'bar')
        .attr('x', (d) => xs(d.label)!)
        .attr('y', (d) => ys(d.value))
        .attr('width', xs.bandwidth())
        .attr('height', (d) => innerH - ys(d.value))
        .style('fill', (d) => (d.kind === 'reparam' ? paletteRL.signal : paletteRL.nuisance))
        .style('fill-opacity', 0.85);

      g.selectAll('text.val')
        .data(data)
        .join('text')
        .attr('class', 'val')
        .attr('x', (d) => xs(d.label)! + xs.bandwidth() / 2)
        .attr('y', (d) => ys(d.value) - 4)
        .attr('text-anchor', 'middle')
        .attr('font-size', 9)
        .style('fill', 'currentColor')
        .text((d) => d.value.toFixed(2));
    },
    [data, W, H, innerW, innerH],
  );

  return (
    <div ref={containerRef} className="not-prose my-6">
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-2 text-sm">
        <label className="flex flex-col">
          <span>μ = ±{muDisplay.toFixed(2)}</span>
          <input
            type="range"
            min={0}
            max={2}
            step={0.05}
            value={muDisplay}
            onChange={(e) => setMuDisplay(Number(e.target.value))}
            onMouseUp={(e) => setMuCommitted(Number((e.target as HTMLInputElement).value))}
            onTouchEnd={(e) => setMuCommitted(Number((e.target as HTMLInputElement).value))}
            onKeyUp={(e) => setMuCommitted(Number((e.target as HTMLInputElement).value))}
            aria-label="Encoder mean"
          />
        </label>
        <label className="flex flex-col">
          <span>log σ² = {logSigmaSqDisplay.toFixed(2)} (σ ≈ {Math.exp(0.5 * logSigmaSqDisplay).toFixed(2)})</span>
          <input
            type="range"
            min={-2}
            max={1}
            step={0.05}
            value={logSigmaSqDisplay}
            onChange={(e) => setLogSigmaSqDisplay(Number(e.target.value))}
            onMouseUp={(e) => setLogSigmaSqCommitted(Number((e.target as HTMLInputElement).value))}
            onTouchEnd={(e) => setLogSigmaSqCommitted(Number((e.target as HTMLInputElement).value))}
            onKeyUp={(e) => setLogSigmaSqCommitted(Number((e.target as HTMLInputElement).value))}
            aria-label="Encoder log variance"
          />
        </label>
        <label className="flex flex-col">
          <span>MC batch B = {B}</span>
          <input
            type="range"
            min={50}
            max={1000}
            step={50}
            value={B}
            onChange={(e) => setB(Number(e.target.value))}
            aria-label="Monte Carlo batch size"
          />
        </label>
      </div>
      <svg ref={svgRef} className="w-full" style={{ maxHeight: H }} role="img" aria-label="Reparametrization vs score-function variance"/>
      <div className="text-xs opacity-75 mt-1 text-center">
        Blue: reparametrization gradient std-dev. Red: score-function (REINFORCE) std-dev. Mean reduction factor on this fixture: <strong>{ratio.toFixed(2)}×</strong>. This is the empirical reason the trick is non-negotiable for VAEs: without it, the gradient noise drowns the signal even on a single example.
      </div>
    </div>
  );
}
