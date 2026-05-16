import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  linearFit,
  logisticFit,
  marCorrectOutcomeFeatures,
  marCorrectPropensityFeatures,
  mulberry32,
  oneStepMar,
  paletteSemi,
  sampleMar,
  sigmoid,
  tmleMar,
} from './shared/semiparametric-inference';

// =============================================================================
// TMLEFluctuation — §6
// Left: scatter of (one-step ψ̂, TMLE ψ̂) pairs across MC replications hugging y=x.
// Right: trajectory of (ε, ψ̂, EIF mean) across the targeting step — two points
// (pre-target at ε=0, post-target at ε̂) showing the EIF correction collapses
// to 0 and ψ̂ adjusts accordingly.
// =============================================================================

const HEIGHT = 320;
const SM_BREAKPOINT = 640;
const N = 1000;
const PSI_0 = 1 + 0.25 + 2 / 3;

function fitParametricNuisances(
  X: Float64Array,
  R: Int8Array,
  Y: Float64Array,
  n: number,
): { mHat: Float64Array; piHat: Float64Array } {
  const obsIdx: number[] = [];
  for (let i = 0; i < n; i++) if (R[i] === 1) obsIdx.push(i);
  const Xobs = new Float64Array(obsIdx.length * 3);
  const Yobs = new Float64Array(obsIdx.length);
  for (let k = 0; k < obsIdx.length; k++) {
    const i = obsIdx[k];
    Xobs[k * 3] = X[i * 3];
    Xobs[k * 3 + 1] = X[i * 3 + 1];
    Xobs[k * 3 + 2] = X[i * 3 + 2];
    Yobs[k] = Y[i];
  }
  const Fobs = marCorrectOutcomeFeatures(Xobs, obsIdx.length);
  const betaM = linearFit(Fobs, Yobs, obsIdx.length, 4);
  const Fall = marCorrectOutcomeFeatures(X, n);
  const mHat = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let m = betaM[0];
    for (let j = 0; j < 4; j++) m += betaM[j + 1] * Fall[i * 4 + j];
    mHat[i] = m;
  }
  const Fp = marCorrectPropensityFeatures(X, n);
  const betaPi = logisticFit(Fp, R, n, 3);
  const piHat = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let z = betaPi[0];
    for (let j = 0; j < 3; j++) z += betaPi[j + 1] * Fp[i * 3 + j];
    piHat[i] = Math.max(sigmoid(z), 1e-6);
  }
  return { mHat, piHat };
}

export default function TMLEFluctuation() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [displayB, setDisplayB] = useState(100);
  const [committedB, setCommittedB] = useState(100);

  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;

  const data = useMemo(() => {
    const pairs: { oneStep: number; tmle: number }[] = [];
    let singleTrace: { epsilon: number; psi: number; eifMean: number }[] = [];
    let singleEpsilon = 0;
    for (let b = 0; b < committedB; b++) {
      const rng = mulberry32(20260515 + b * 19);
      const sample = sampleMar(N, rng);
      const { mHat, piHat } = fitParametricNuisances(sample.X, sample.R, sample.Y, N);
      const os = oneStepMar(sample.R, sample.Y, mHat, piHat);
      const tm = tmleMar(sample.R, sample.Y, mHat, piHat);
      pairs.push({ oneStep: os.psi, tmle: tm.psi });
      if (b === 0) {
        singleTrace = tm.trace;
        singleEpsilon = tm.epsilon;
      }
    }
    return { pairs, singleTrace, singleEpsilon };
  }, [committedB]);

  const refL = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const panelW = isMobile ? w : w / 2 - 8;
      const margin = { top: 22, right: 18, bottom: 36, left: 56 };
      const W = panelW - margin.left - margin.right;
      const H = HEIGHT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${panelW} ${HEIGHT}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);

      const allX = data.pairs.map((p) => p.oneStep);
      const allY = data.pairs.map((p) => p.tmle);
      const lo = Math.min(...allX, ...allY) - 0.01;
      const hi = Math.max(...allX, ...allY) + 0.01;
      const xScale = d3.scaleLinear().domain([lo, hi]).range([0, W]).nice();
      const yScale = d3.scaleLinear().domain([lo, hi]).range([H, 0]).nice();
      g.append('g').attr('transform', `translate(0, ${H})`).call(d3.axisBottom(xScale).ticks(5));
      g.append('g').call(d3.axisLeft(yScale).ticks(5));
      // y = x diagonal.
      g.append('line').attr('x1', xScale(lo)).attr('x2', xScale(hi))
        .attr('y1', yScale(lo)).attr('y2', yScale(hi))
        .style('stroke', paletteSemi.theoryLine).style('stroke-width', 1)
        .style('stroke-dasharray', '4,3');
      // Scatter.
      g.selectAll(null).data(data.pairs).join('circle')
        .attr('cx', (d) => xScale(d.oneStep))
        .attr('cy', (d) => yScale(d.tmle))
        .attr('r', 2.6).style('fill', paletteSemi.tmle).style('opacity', 0.65);
      g.append('text').attr('x', W / 2).attr('y', H + 30).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-family', 'var(--font-mono)').style('font-size', 11)
        .text('one-step ψ̂');
      g.append('text').attr('x', -H / 2).attr('y', -45).attr('text-anchor', 'middle')
        .attr('transform', 'rotate(-90)')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-family', 'var(--font-mono)').style('font-size', 11)
        .text('TMLE ψ̂');
      g.append('text').attr('x', 0).attr('y', -8)
        .style('fill', 'var(--color-text)')
        .style('font-family', 'var(--font-mono)').style('font-size', 12)
        .text(`Scatter (one-step, TMLE) over B=${data.pairs.length} draws — hugs y=x`);
    },
    [data, containerWidth, isMobile],
  );

  const refR = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const panelW = isMobile ? w : w / 2 - 8;
      const margin = { top: 22, right: 18, bottom: 36, left: 56 };
      const W = panelW - margin.left - margin.right;
      const H = HEIGHT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${panelW} ${HEIGHT}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);

      const trace = data.singleTrace;
      const xScale = d3.scaleBand().domain(['pre-target (ε=0)', `post-target (ε̂=${trace[1].epsilon.toFixed(4)})`])
        .range([0, W]).padding(0.3);
      const psiVals = trace.map((t) => t.psi);
      const yLo = Math.min(PSI_0, ...psiVals) - 0.05;
      const yHi = Math.max(PSI_0, ...psiVals) + 0.05;
      const yScale = d3.scaleLinear().domain([yLo, yHi]).range([H, 0]).nice();
      g.append('g').attr('transform', `translate(0, ${H})`).call(d3.axisBottom(xScale));
      g.append('g').call(d3.axisLeft(yScale).ticks(5));
      g.append('line').attr('x1', 0).attr('x2', W)
        .attr('y1', yScale(PSI_0)).attr('y2', yScale(PSI_0))
        .style('stroke', paletteSemi.theoryLine).style('stroke-width', 1.5)
        .style('stroke-dasharray', '4,3');
      g.append('text').attr('x', W).attr('y', yScale(PSI_0) - 4).attr('text-anchor', 'end')
        .style('fill', paletteSemi.theoryLine)
        .style('font-family', 'var(--font-mono)').style('font-size', 11)
        .text(`ψ_0 ≈ ${PSI_0.toFixed(3)}`);
      // ψ̂ trajectory (line).
      g.append('path').datum(trace).attr('d',
        d3.line<{ psi: number; epsilon: number }>()
          .x((d, i) => (xScale.domain()[i] != null ? (xScale(xScale.domain()[i])! + xScale.bandwidth() / 2) : 0))
          .y((d) => yScale(d.psi)) as unknown as string)
        .style('fill', 'none').style('stroke', paletteSemi.tmle).style('stroke-width', 2);
      trace.forEach((t, i) => {
        g.append('circle')
          .attr('cx', xScale(xScale.domain()[i])! + xScale.bandwidth() / 2)
          .attr('cy', yScale(t.psi))
          .attr('r', 4).style('fill', paletteSemi.tmle);
        g.append('text')
          .attr('x', xScale(xScale.domain()[i])! + xScale.bandwidth() / 2)
          .attr('y', yScale(t.psi) - 10).attr('text-anchor', 'middle')
          .style('fill', 'var(--color-text)')
          .style('font-family', 'var(--font-mono)').style('font-size', 11)
          .text(`ψ̂=${t.psi.toFixed(4)}, EIF mean=${t.eifMean.toExponential(2)}`);
      });
      g.append('text').attr('x', 0).attr('y', -8)
        .style('fill', 'var(--color-text)')
        .style('font-family', 'var(--font-mono)').style('font-size', 12)
        .text('Targeting step: ψ̂ shifts; empirical EIF mean → 0');
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
            min={50}
            max={300}
            step={50}
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
      <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: 8 }}>
        <svg ref={refL} style={{ width: isMobile ? '100%' : '50%', height: HEIGHT, display: 'block' }} />
        <svg ref={refR} style={{ width: isMobile ? '100%' : '50%', height: HEIGHT, display: 'block' }} />
      </div>
    </div>
  );
}
