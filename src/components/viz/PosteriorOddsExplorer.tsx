import { useState, useMemo } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import { paletteNB } from './shared/naive-bayes';

// =============================================================================
// PosteriorOddsExplorer — §1.2 companion. The reader controls the class prior
// π = P(Y = 1) and the likelihood ratio LR = p(x | Y = 1) / p(x | Y = 0); the
// viewer plots P(Y = 1 | x) = πLR / (πLR + (1 − π)) as a function of LR on a
// log axis, marking the decision threshold P(Y = 1 | x) = 1/2.
//
// Bayes' rule reframed for classification: the prior gives the starting point
// at LR = 1, the LR slides the posterior up or down, and the threshold is the
// LR at which the two classes are equally probable.
// =============================================================================

const LR_MIN = 0.01;
const LR_MAX = 100;
const N_POINTS = 200;

function posteriorAtLR(prior: number, lr: number): number {
  const num = prior * lr;
  return num / (num + (1 - prior));
}

export default function PosteriorOddsExplorer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [prior, setPrior] = useState(0.3);
  const [lr, setLR] = useState(2.5);

  const W = Math.max(360, Math.min(width || 720, 720));
  const H = 320;
  const margin = { top: 18, right: 20, bottom: 38, left: 52 };
  const innerW = W - margin.left - margin.right;
  const innerH = H - margin.top - margin.bottom;

  const curve = useMemo(() => {
    const xs: number[] = [];
    const ys: number[] = [];
    const logMin = Math.log(LR_MIN);
    const logMax = Math.log(LR_MAX);
    for (let i = 0; i < N_POINTS; i++) {
      const l = Math.exp(logMin + ((logMax - logMin) * i) / (N_POINTS - 1));
      xs.push(l);
      ys.push(posteriorAtLR(prior, l));
    }
    return { xs, ys };
  }, [prior]);

  const decisionLR = useMemo(() => (1 - prior) / Math.max(prior, 1e-9), [prior]);

  const postAtCurrent = posteriorAtLR(prior, lr);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.attr('viewBox', `0 0 ${W} ${H}`).selectAll('*').remove();
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const x = d3.scaleLog().domain([LR_MIN, LR_MAX]).range([0, innerW]);
      const y = d3.scaleLinear().domain([0, 1]).range([innerH, 0]);

      // Threshold band at 0.5.
      g.append('line')
        .attr('x1', 0)
        .attr('x2', innerW)
        .attr('y1', y(0.5))
        .attr('y2', y(0.5))
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '4 3')
        .attr('opacity', 0.55)
        .style('stroke', paletteNB.data);

      g.append('g')
        .attr('transform', `translate(0,${innerH})`)
        .call(d3.axisBottom(x).ticks(6, '~g'));
      g.append('g').call(d3.axisLeft(y).ticks(6));

      // Axis labels.
      g.append('text')
        .attr('x', innerW / 2)
        .attr('y', innerH + 30)
        .attr('text-anchor', 'middle')
        .attr('font-size', 12)
        .style('fill', 'currentColor')
        .text('Likelihood ratio p(x|Y=1) / p(x|Y=0)  (log scale)');
      g.append('text')
        .attr('transform', `translate(-38, ${innerH / 2}) rotate(-90)`)
        .attr('text-anchor', 'middle')
        .attr('font-size', 12)
        .style('fill', 'currentColor')
        .text('P(Y=1 | x)');

      // Curve.
      const line = d3
        .line<number>()
        .x((_, i) => x(curve.xs[i]))
        .y((d) => y(d));
      g.append('path')
        .datum(curve.ys)
        .attr('fill', 'none')
        .attr('stroke-width', 2.2)
        .attr('d', line)
        .style('stroke', paletteNB.pos);

      // Decision-threshold LR vertical line.
      g.append('line')
        .attr('x1', x(decisionLR))
        .attr('x2', x(decisionLR))
        .attr('y1', 0)
        .attr('y2', innerH)
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '3 4')
        .attr('opacity', 0.7)
        .style('stroke', paletteNB.highlight);
      g.append('text')
        .attr('x', x(decisionLR))
        .attr('y', 12)
        .attr('text-anchor', 'middle')
        .attr('font-size', 11)
        .style('fill', paletteNB.highlight)
        .text(`LR* = ${decisionLR.toFixed(2)}`);

      // Current-LR marker.
      g.append('circle')
        .attr('cx', x(lr))
        .attr('cy', y(postAtCurrent))
        .attr('r', 6)
        .attr('stroke-width', 1.5)
        .style('fill', paletteNB.neg)
        .style('stroke', paletteNB.neg)
        .style('fill-opacity', 0.6);
    },
    [curve, decisionLR, lr, postAtCurrent, W, H, innerW, innerH],
  );

  return (
    <div ref={containerRef} className="not-prose my-6">
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-2 text-sm">
        <label className="flex flex-col">
          <span>Prior π = P(Y = 1) = {prior.toFixed(2)}</span>
          <input
            type="range"
            min={0.05}
            max={0.95}
            step={0.01}
            value={prior}
            onChange={(e) => setPrior(Number(e.target.value))}
            aria-label="Class prior"
          />
        </label>
        <label className="flex flex-col">
          <span>Likelihood ratio LR = {lr.toFixed(2)}</span>
          <input
            type="range"
            min={Math.log(LR_MIN)}
            max={Math.log(LR_MAX)}
            step={0.01}
            value={Math.log(lr)}
            onChange={(e) => setLR(Math.exp(Number(e.target.value)))}
            aria-label="Likelihood ratio"
          />
        </label>
      </div>
      <svg ref={svgRef} className="w-full" style={{ maxHeight: H }} role="img" aria-label="Posterior probability vs likelihood ratio"/>
      <div className="text-xs opacity-75 mt-1 text-center">
        At (π = {prior.toFixed(2)}, LR = {lr.toFixed(2)}): P(Y = 1 | x) =
        <strong> {postAtCurrent.toFixed(3)}</strong>. The decision threshold (P = 1/2) lies at LR* = (1 − π) / π = <strong>{decisionLR.toFixed(2)}</strong>.
      </div>
    </div>
  );
}
