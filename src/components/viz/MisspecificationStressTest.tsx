import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  linearFit,
  logisticFit,
  marCorrectOutcomeFeatures,
  marCorrectPropensityFeatures,
  marWrongOutcomeFeatures,
  marWrongPropensityFeatures,
  mulberry32,
  oneStepMar,
  paletteSemi,
  sampleMar,
  sigmoid,
} from './shared/semiparametric-inference';

// =============================================================================
// MisspecificationStressTest — §13.3
// 2×2 grid of AIPW ψ̂ histograms under the four combinations of
//   (correct m̂, correct π̂), (correct m̂, wrong π̂),
//   (wrong m̂, correct π̂), (wrong m̂, wrong π̂).
// Only the (wrong, wrong) cell should be biased — the double-robustness story.
// =============================================================================

const HEIGHT_PANEL = 200;
const SM_BREAKPOINT = 640;
const N = 1000;
const PSI_0 = 1 + 0.25 + 2 / 3;

function fitMHat(
  X: Float64Array,
  R: Int8Array,
  Y: Float64Array,
  n: number,
  correct: boolean,
): Float64Array {
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
  const featuresFn = correct ? marCorrectOutcomeFeatures : marWrongOutcomeFeatures;
  const Fobs = featuresFn(Xobs, obsIdx.length);
  const p = Fobs.length / obsIdx.length;
  const betaM = linearFit(Fobs, Yobs, obsIdx.length, p);
  const Fall = featuresFn(X, n);
  const mHat = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let m = betaM[0];
    for (let j = 0; j < p; j++) m += betaM[j + 1] * Fall[i * p + j];
    mHat[i] = m;
  }
  return mHat;
}

function fitPiHat(
  X: Float64Array,
  R: Int8Array,
  n: number,
  correct: boolean,
): Float64Array {
  const featuresFn = correct ? marCorrectPropensityFeatures : marWrongPropensityFeatures;
  const Fp = featuresFn(X, n);
  const p = Fp.length / n;
  const betaPi = logisticFit(Fp, R, n, p);
  const piHat = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let z = betaPi[0];
    for (let j = 0; j < p; j++) z += betaPi[j + 1] * Fp[i * p + j];
    piHat[i] = Math.max(sigmoid(z), 1e-6);
  }
  return piHat;
}

export default function MisspecificationStressTest() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [displayB, setDisplayB] = useState(80);
  const [committedB, setCommittedB] = useState(80);

  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;

  const data = useMemo(() => {
    const scenarios: { mC: boolean; pC: boolean; label: string }[] = [
      { mC: true, pC: true, label: 'correct m, correct π' },
      { mC: true, pC: false, label: 'correct m, wrong π' },
      { mC: false, pC: true, label: 'wrong m, correct π' },
      { mC: false, pC: false, label: 'wrong m, wrong π' },
    ];
    const results: { label: string; thetas: Float64Array; mean: number; sd: number; biased: boolean }[] = [];
    for (const sc of scenarios) {
      const thetas = new Float64Array(committedB);
      for (let b = 0; b < committedB; b++) {
        const rng = mulberry32(20260515 + b * 13 + (sc.mC ? 1 : 0) * 100 + (sc.pC ? 1 : 0) * 10);
        const sample = sampleMar(N, rng);
        const mHat = fitMHat(sample.X, sample.R, sample.Y, N, sc.mC);
        const piHat = fitPiHat(sample.X, sample.R, N, sc.pC);
        thetas[b] = oneStepMar(sample.R, sample.Y, mHat, piHat).psi;
      }
      const mu = d3.mean(Array.from(thetas))!;
      const sd = d3.deviation(Array.from(thetas)) ?? 0;
      results.push({
        label: sc.label,
        thetas,
        mean: mu,
        sd,
        biased: !sc.mC && !sc.pC,
      });
    }
    return results;
  }, [committedB]);

  const renderPanel = (
    svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
    panelW: number,
    entry: typeof data[number],
  ) => {
    const margin = { top: 30, right: 12, bottom: 28, left: 44 };
    const W = panelW - margin.left - margin.right;
    const H = HEIGHT_PANEL - margin.top - margin.bottom;
    svg.selectAll('*').remove();
    if (W <= 0) return;
    svg.attr('viewBox', `0 0 ${panelW} ${HEIGHT_PANEL}`);
    const g = svg.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);
    const arr = Array.from(entry.thetas);
    const lo = Math.min(...arr, PSI_0 - 0.2);
    const hi = Math.max(...arr, PSI_0 + 0.2);
    const xScale = d3.scaleLinear().domain([lo, hi]).range([0, W]).nice();
    const bins = d3.bin().domain(xScale.domain() as [number, number]).thresholds(18)(arr);
    const yMax = d3.max(bins, (d) => d.length)! * 1.1 + 1;
    const yScale = d3.scaleLinear().domain([0, yMax]).range([H, 0]).nice();
    g.append('g').attr('transform', `translate(0, ${H})`).call(d3.axisBottom(xScale).ticks(4));
    g.append('g').call(d3.axisLeft(yScale).ticks(3));
    const color = entry.biased ? paletteSemi.bias : paletteSemi.oneStep;
    g.selectAll(null).data(bins).join('rect')
      .attr('x', (d) => xScale(d.x0!) + 1)
      .attr('y', (d) => yScale(d.length))
      .attr('width', (d) => Math.max(xScale(d.x1!) - xScale(d.x0!) - 1, 0))
      .attr('height', (d) => H - yScale(d.length))
      .style('fill', color).style('opacity', 0.7);
    g.append('line').attr('x1', xScale(PSI_0)).attr('x2', xScale(PSI_0))
      .attr('y1', 0).attr('y2', H)
      .style('stroke', paletteSemi.theoryLine).style('stroke-width', 1.5)
      .style('stroke-dasharray', '4,3');
    g.append('line').attr('x1', xScale(entry.mean)).attr('x2', xScale(entry.mean))
      .attr('y1', 0).attr('y2', H)
      .style('stroke', color).style('stroke-width', 1.5);
    g.append('text').attr('x', 0).attr('y', -16)
      .style('fill', entry.biased ? paletteSemi.bias : 'var(--color-text)')
      .style('font-family', 'var(--font-mono)').style('font-size', 12)
      .style('font-weight', '600')
      .text(`${entry.label}${entry.biased ? ' — biased' : ''}`);
    g.append('text').attr('x', 0).attr('y', -2)
      .style('fill', 'var(--color-text-secondary)')
      .style('font-family', 'var(--font-mono)').style('font-size', 10)
      .text(`mean=${entry.mean.toFixed(3)} (ψ_0=${PSI_0.toFixed(3)}), SD=${entry.sd.toFixed(3)}`);
  };

  const ref00 = useD3<SVGSVGElement>((svg) => {
    const w = containerWidth || 720;
    const panelW = isMobile ? w : w / 2 - 8;
    renderPanel(svg, panelW, data[0]);
  }, [data, containerWidth, isMobile]);
  const ref01 = useD3<SVGSVGElement>((svg) => {
    const w = containerWidth || 720;
    const panelW = isMobile ? w : w / 2 - 8;
    renderPanel(svg, panelW, data[1]);
  }, [data, containerWidth, isMobile]);
  const ref10 = useD3<SVGSVGElement>((svg) => {
    const w = containerWidth || 720;
    const panelW = isMobile ? w : w / 2 - 8;
    renderPanel(svg, panelW, data[2]);
  }, [data, containerWidth, isMobile]);
  const ref11 = useD3<SVGSVGElement>((svg) => {
    const w = containerWidth || 720;
    const panelW = isMobile ? w : w / 2 - 8;
    renderPanel(svg, panelW, data[3]);
  }, [data, containerWidth, isMobile]);

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
      <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: 8 }}>
        <svg ref={ref00} style={{ width: isMobile ? '100%' : '50%', height: HEIGHT_PANEL, display: 'block' }} />
        <svg ref={ref01} style={{ width: isMobile ? '100%' : '50%', height: HEIGHT_PANEL, display: 'block' }} />
      </div>
      <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: 8, marginTop: 8 }}>
        <svg ref={ref10} style={{ width: isMobile ? '100%' : '50%', height: HEIGHT_PANEL, display: 'block' }} />
        <svg ref={ref11} style={{ width: isMobile ? '100%' : '50%', height: HEIGHT_PANEL, display: 'block' }} />
      </div>
    </div>
  );
}
