import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  crossFitVarianceEif,
  mulberry32,
  paletteSemi,
  plugInVarianceInSample,
  polyDeg2VarianceFitter,
  sampleUq,
} from './shared/semiparametric-inference';

// =============================================================================
// EIFCorrectionForVariance — §11
// Two histograms across MC replications:
//   - in-sample plug-in σ̂² (biased downward by over-fitting)
//   - cross-fit one-step σ̂² (centered at the true σ² = 1)
// True σ² = 1 from the §11 DGP.
// =============================================================================

const HEIGHT = 340;
const SM_BREAKPOINT = 640;
const N = 500;
const TRUE_SIGMA2 = 1.0;

export default function EIFCorrectionForVariance() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [displayB, setDisplayB] = useState(80);
  const [committedB, setCommittedB] = useState(80);

  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;

  const data = useMemo(() => {
    const inSample = new Float64Array(committedB);
    const crossFit = new Float64Array(committedB);
    for (let b = 0; b < committedB; b++) {
      const rng = mulberry32(20260515 + b * 11);
      const sample = sampleUq(N, rng);
      inSample[b] = plugInVarianceInSample(sample.X, sample.Y, sample.n, 5, polyDeg2VarianceFitter);
      crossFit[b] = crossFitVarianceEif(sample.X, sample.Y, sample.n, 5, 5, polyDeg2VarianceFitter, rng).sigma2;
    }
    return { inSample, crossFit };
  }, [committedB]);

  const ref = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const W0 = isMobile ? w : Math.min(w, 760);
      const margin = { top: 28, right: 18, bottom: 36, left: 56 };
      const W = W0 - margin.left - margin.right;
      const H = HEIGHT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${W0} ${HEIGHT}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);

      const all = [...Array.from(data.inSample), ...Array.from(data.crossFit)];
      const lo = Math.min(...all, 0.6) - 0.05;
      const hi = Math.max(...all, 1.3) + 0.05;
      const xScale = d3.scaleLinear().domain([lo, hi]).range([0, W]).nice();
      const bins = d3.bin().domain(xScale.domain() as [number, number]).thresholds(26);
      const binsIn = bins(Array.from(data.inSample));
      const binsCf = bins(Array.from(data.crossFit));
      const yMax = Math.max(d3.max(binsIn, (d) => d.length)!, d3.max(binsCf, (d) => d.length)!) * 1.1 + 1;
      const yScale = d3.scaleLinear().domain([0, yMax]).range([H, 0]).nice();
      g.append('g').attr('transform', `translate(0, ${H})`).call(d3.axisBottom(xScale).ticks(6));
      g.append('g').call(d3.axisLeft(yScale).ticks(5));

      const drawBins = (bs: d3.Bin<number, number>[], color: string) => {
        g.selectAll(null).data(bs).join('rect')
          .attr('x', (d) => xScale(d.x0!) + 1)
          .attr('y', (d) => yScale(d.length))
          .attr('width', (d) => Math.max(xScale(d.x1!) - xScale(d.x0!) - 1, 0))
          .attr('height', (d) => H - yScale(d.length))
          .style('fill', color).style('opacity', 0.6);
      };
      drawBins(binsIn, paletteSemi.plugIn);
      drawBins(binsCf, paletteSemi.oneStep);

      // True σ².
      g.append('line').attr('x1', xScale(TRUE_SIGMA2)).attr('x2', xScale(TRUE_SIGMA2))
        .attr('y1', 0).attr('y2', H)
        .style('stroke', paletteSemi.theoryLine).style('stroke-width', 1.5)
        .style('stroke-dasharray', '4,3');
      g.append('text').attr('x', xScale(TRUE_SIGMA2) + 4).attr('y', 14)
        .style('fill', paletteSemi.theoryLine)
        .style('font-family', 'var(--font-mono)').style('font-size', 11)
        .text('σ² = 1');

      const muIn = d3.mean(Array.from(data.inSample))!;
      const muCf = d3.mean(Array.from(data.crossFit))!;
      // Mean ticks.
      g.append('line').attr('x1', xScale(muIn)).attr('x2', xScale(muIn))
        .attr('y1', 0).attr('y2', H)
        .style('stroke', paletteSemi.plugIn).style('stroke-width', 1.5);
      g.append('line').attr('x1', xScale(muCf)).attr('x2', xScale(muCf))
        .attr('y1', 0).attr('y2', H)
        .style('stroke', paletteSemi.oneStep).style('stroke-width', 1.5);

      g.append('text').attr('x', 0).attr('y', -12)
        .style('fill', 'var(--color-text)')
        .style('font-family', 'var(--font-mono)').style('font-size', 12)
        .text('In-sample plug-in σ̂² (blue) vs cross-fit one-step σ̂² (red), B replications');
      g.append('text').attr('x', 0).attr('y', 0)
        .style('fill', 'var(--color-text-secondary)')
        .style('font-family', 'var(--font-mono)').style('font-size', 11)
        .text(`mean(in-sample) = ${muIn.toFixed(3)}, mean(cross-fit) = ${muCf.toFixed(3)} → bias gap ≈ ${(muCf - muIn).toFixed(3)}`);
    },
    [data, containerWidth, isMobile],
  );

  return (
    <div ref={containerRef} className="viz-container" style={{ width: '100%' }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginBottom: 12 }}>
        <label style={{ fontFamily: 'var(--font-mono)', fontSize: 13 }}>
          MC replications B: <strong>{displayB}</strong>
          <input
            type="range"
            min={40}
            max={150}
            step={10}
            value={displayB}
            onChange={(e) => setDisplayB(+e.target.value)}
            onMouseUp={() => setCommittedB(displayB)}
            onTouchEnd={() => setCommittedB(displayB)}
            onKeyUp={() => setCommittedB(displayB)}
            aria-label="Monte Carlo replications"
            style={{ width: '100%', marginTop: 4 }}
          />
        </label>
      </div>
      <svg ref={ref} style={{ width: '100%', height: HEIGHT, display: 'block' }} />
    </div>
  );
}
