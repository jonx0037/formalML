import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  linearFit,
  marCorrectOutcomeFeatures,
  marCorrectPropensityFeatures,
  mulberry32,
  oneStepMar,
  paletteSemi,
  plugInMean,
  sampleMar,
  sigmoid,
  logisticFit,
} from './shared/semiparametric-inference';

// =============================================================================
// OneStepCorrection — §5
// Left: a single MC draw decomposed as plug-in (blue bar) + EIF correction
// (red bar); the sum reaches ψ̂_1-step. Black horizontal line marks ψ_0.
// Right: histograms of plug-in vs one-step ψ̂ across B replications, showing
// the variance reduction (in well-specified case both center on ψ_0; SD
// differs).
// =============================================================================

const HEIGHT = 320;
const SM_BREAKPOINT = 640;
const N = 1000;
const PSI_0 = 1 + 0.25 + 2 / 3; // 1.9167

function fitParametricNuisances(
  X: Float64Array,
  R: Int8Array,
  Y: Float64Array,
  n: number,
): { mHat: Float64Array; piHat: Float64Array } {
  // Outcome regression on observed (R=1).
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
  // Logistic propensity on all rows.
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

export default function OneStepCorrection() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [displayB, setDisplayB] = useState(200);
  const [committedB, setCommittedB] = useState(200);

  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;

  const data = useMemo(() => {
    // Single-draw decomposition.
    const rng = mulberry32(20260515);
    const sample = sampleMar(N, rng);
    const { mHat, piHat } = fitParametricNuisances(sample.X, sample.R, sample.Y, N);
    const plug = plugInMean(mHat);
    const { psi, eif } = oneStepMar(sample.R, sample.Y, mHat, piHat);
    const correction = psi - plug;
    // MC: plug-in vs one-step across B replications.
    const plugArr = new Float64Array(committedB);
    const oneStepArr = new Float64Array(committedB);
    for (let b = 0; b < committedB; b++) {
      const r2 = mulberry32(20260515 + b * 23 + 1);
      const s2 = sampleMar(N, r2);
      const { mHat: m2, piHat: p2 } = fitParametricNuisances(s2.X, s2.R, s2.Y, N);
      plugArr[b] = plugInMean(m2);
      oneStepArr[b] = oneStepMar(s2.R, s2.Y, m2, p2).psi;
    }
    return { plug, correction, psi, eif, plugArr, oneStepArr };
  }, [committedB]);

  const refL = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const panelW = isMobile ? w : w / 2 - 8;
      const margin = { top: 22, right: 14, bottom: 36, left: 60 };
      const W = panelW - margin.left - margin.right;
      const H = HEIGHT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${panelW} ${HEIGHT}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);

      const items = [
        { label: 'plug-in', value: data.plug, color: paletteSemi.plugIn, bottom: 0 },
        { label: 'EIF correction', value: data.correction, color: paletteSemi.oneStep, bottom: data.plug },
      ];

      const yLo = Math.min(0, data.plug, data.psi) - 0.05;
      const yHi = Math.max(data.plug, data.psi, PSI_0) + 0.15;
      const xScale = d3.scaleBand().domain(['plug-in', 'EIF correction', 'sum = ψ̂']).range([0, W]).padding(0.2);
      const yScale = d3.scaleLinear().domain([yLo, yHi]).range([H, 0]).nice();
      g.append('g').attr('transform', `translate(0, ${H})`).call(d3.axisBottom(xScale));
      g.append('g').call(d3.axisLeft(yScale).ticks(5));

      items.forEach((it) => {
        g.append('rect')
          .attr('x', xScale(it.label)!)
          .attr('y', yScale(Math.max(it.bottom, it.bottom + it.value)))
          .attr('width', xScale.bandwidth())
          .attr('height', Math.abs(yScale(it.bottom) - yScale(it.bottom + it.value)))
          .style('fill', it.color).style('opacity', 0.75);
        g.append('text')
          .attr('x', xScale(it.label)! + xScale.bandwidth() / 2)
          .attr('y', yScale(it.bottom + it.value) - 4)
          .attr('text-anchor', 'middle')
          .style('fill', 'var(--color-text)').style('font-family', 'var(--font-mono)').style('font-size', 11)
          .text(`${it.value >= 0 ? '+' : ''}${it.value.toFixed(3)}`);
      });

      // Total bar.
      g.append('rect')
        .attr('x', xScale('sum = ψ̂')!).attr('y', yScale(data.psi))
        .attr('width', xScale.bandwidth())
        .attr('height', yScale(0) - yScale(data.psi))
        .style('fill', paletteSemi.tmle).style('opacity', 0.75);
      g.append('text')
        .attr('x', xScale('sum = ψ̂')! + xScale.bandwidth() / 2)
        .attr('y', yScale(data.psi) - 4)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)').style('font-family', 'var(--font-mono)').style('font-size', 11)
        .text(data.psi.toFixed(3));

      // True ψ_0 line.
      g.append('line').attr('x1', 0).attr('x2', W)
        .attr('y1', yScale(PSI_0)).attr('y2', yScale(PSI_0))
        .style('stroke', paletteSemi.theoryLine).style('stroke-width', 1.6)
        .style('stroke-dasharray', '4,3');
      g.append('text').attr('x', W).attr('y', yScale(PSI_0) - 4).attr('text-anchor', 'end')
        .style('fill', paletteSemi.theoryLine)
        .style('font-family', 'var(--font-mono)').style('font-size', 11)
        .text(`ψ_0 ≈ ${PSI_0.toFixed(3)}`);

      g.append('text').attr('x', 0).attr('y', -8)
        .style('fill', 'var(--color-text)')
        .style('font-family', 'var(--font-mono)').style('font-size', 12)
        .text('Single draw: plug-in + correction = one-step ψ̂');
    },
    [data, containerWidth, isMobile],
  );

  const refR = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const panelW = isMobile ? w : w / 2 - 8;
      const margin = { top: 22, right: 14, bottom: 36, left: 50 };
      const W = panelW - margin.left - margin.right;
      const H = HEIGHT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${panelW} ${HEIGHT}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);

      const all = [...Array.from(data.plugArr), ...Array.from(data.oneStepArr)];
      const lo = Math.min(...all) - 0.02;
      const hi = Math.max(...all) + 0.02;
      const xScale = d3.scaleLinear().domain([lo, hi]).range([0, W]).nice();
      const bins = d3.bin().domain(xScale.domain() as [number, number]).thresholds(24);
      const binsPlug = bins(Array.from(data.plugArr));
      const binsOne = bins(Array.from(data.oneStepArr));
      const yMax = Math.max(d3.max(binsPlug, (d) => d.length)!, d3.max(binsOne, (d) => d.length)!) * 1.1 + 1;
      const yScale = d3.scaleLinear().domain([0, yMax]).range([H, 0]).nice();

      g.append('g').attr('transform', `translate(0, ${H})`).call(d3.axisBottom(xScale).ticks(5));
      g.append('g').call(d3.axisLeft(yScale).ticks(4));

      const drawBins = (bs: d3.Bin<number, number>[], color: string) => {
        g.selectAll(null).data(bs).join('rect')
          .attr('x', (d) => xScale(d.x0!) + 1)
          .attr('y', (d) => yScale(d.length))
          .attr('width', (d) => Math.max(xScale(d.x1!) - xScale(d.x0!) - 1, 0))
          .attr('height', (d) => H - yScale(d.length))
          .style('fill', color).style('opacity', 0.55);
      };
      drawBins(binsPlug, paletteSemi.plugIn);
      drawBins(binsOne, paletteSemi.oneStep);

      // True ψ_0.
      g.append('line').attr('x1', xScale(PSI_0)).attr('x2', xScale(PSI_0))
        .attr('y1', 0).attr('y2', H)
        .style('stroke', paletteSemi.theoryLine).style('stroke-width', 1.5)
        .style('stroke-dasharray', '4,3');

      g.append('text').attr('x', 0).attr('y', -8)
        .style('fill', 'var(--color-text)')
        .style('font-family', 'var(--font-mono)').style('font-size', 12)
        .text(`B = ${data.plugArr.length}: plug-in (blue) vs one-step (red)`);
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
            max={500}
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
