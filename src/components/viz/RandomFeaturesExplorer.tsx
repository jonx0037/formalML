import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  activations,
  betaFromSVD,
  fillIsotropicGaussian,
  gaussianRng,
  mulberry32,
  randomFeatureMap,
  sampleSphereBetaStar,
  thinSVD,
} from './shared/double-descent';

// =============================================================================
// RandomFeaturesExplorer — §8.1 (Figure 9)
//
// Sweep p with random-feature design for linear / ReLU / tanh activations.
// Target: y = ⟨x, β*⟩ + ε. Features φ(x) = σ(Wx / √d).
// Static fallback: 09_random_features_three_activations.png
// =============================================================================

const HEIGHT = 460;
const ACTIVATION_NAMES: Array<keyof typeof activations> = ['linear', 'relu', 'tanh'];

function sweepActivation(
  actName: keyof typeof activations,
  d: number,
  n: number,
  pMax: number,
  sigma: number,
  betaStar: Float64Array,
  B: number,
  rngSeed: number,
  pGrid: number[],
): number[] {
  const rng = mulberry32(rngSeed);
  const gauss = gaussianRng(rng);
  const result = new Array(pGrid.length).fill(0);
  const nTest = 400;
  const Xte = new Float64Array(nTest * d);
  fillIsotropicGaussian(Xte, nTest, d, gauss);
  const yTeTrue = new Float64Array(nTest);
  for (let i = 0; i < nTest; i++) {
    let s = 0;
    for (let j = 0; j < d; j++) s += Xte[i * d + j] * betaStar[j];
    yTeTrue[i] = s;
  }
  for (let b = 0; b < B; b++) {
    const X = new Float64Array(n * d);
    fillIsotropicGaussian(X, n, d, gauss);
    const W = new Float64Array(pMax * d);
    fillIsotropicGaussian(W, pMax, d, gauss);
    const PhiAll = randomFeatureMap(X, n, d, W, pMax, activations[actName]);
    const PhiTeAll = randomFeatureMap(Xte, nTest, d, W, pMax, activations[actName]);
    const y = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = 0; j < d; j++) s += X[i * d + j] * betaStar[j];
      y[i] = s + sigma * gauss();
    }
    for (let gi = 0; gi < pGrid.length; gi++) {
      const p = pGrid[gi];
      // Build n×p slice and nTest×p slice (first p columns)
      const Phi = new Float64Array(n * p);
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < p; j++) Phi[i * p + j] = PhiAll[i * pMax + j];
      }
      const PhiTe = new Float64Array(nTest * p);
      for (let i = 0; i < nTest; i++) {
        for (let j = 0; j < p; j++) PhiTe[i * p + j] = PhiTeAll[i * pMax + j];
      }
      const beta = betaFromSVD(thinSVD(Phi, n, p), y);
      let mse = 0;
      for (let i = 0; i < nTest; i++) {
        let pred = 0;
        for (let j = 0; j < p; j++) pred += PhiTe[i * p + j] * beta[j];
        const e = pred - yTeTrue[i];
        mse += e * e;
      }
      result[gi] += mse / nTest / B;
    }
  }
  return result;
}

export default function RandomFeaturesExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [dCommitted, setDCommitted] = useState(20);
  const [dDisplay, setDDisplay] = useState(20);
  const [nCommitted, setNCommitted] = useState(50);
  const [nDisplay, setNDisplay] = useState(50);
  const [sigmaCommitted, setSigmaCommitted] = useState(1);
  const [sigmaDisplay, setSigmaDisplay] = useState(1);
  const [showAll, setShowAll] = useState(true);
  const [selectedAct, setSelectedAct] = useState<keyof typeof activations>('relu');

  const curves = useMemo(() => {
    const d = dCommitted;
    const n = nCommitted;
    const sigma = sigmaCommitted;
    const pMax = 200;
    const pGrid = Array.from({ length: 31 }, (_, i) => Math.max(1, Math.round((i / 30) * pMax)));
    const rng = mulberry32(2024);
    const gauss = gaussianRng(rng);
    const betaStar = sampleSphereBetaStar(d, 1, gauss);
    const acts = showAll ? ACTIVATION_NAMES : [selectedAct];
    return acts.map((actName) => {
      // Compute the entire sweep once per activation; map indexes into the result.
      // Previously this lived inside pGrid.map and reran B replicates × |pGrid| times
      // per activation, yielding 93 sweeps at default settings.
      const result = sweepActivation(actName, d, n, pMax, sigma, betaStar, 4, 7919 + actName.length * 13, pGrid);
      return {
        name: actName,
        data: pGrid.map((p, i) => ({ p, mean: result[i] })),
      };
    });
  }, [dCommitted, nCommitted, sigmaCommitted, showAll, selectedAct]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const margin = { top: 24, right: 32, bottom: 56, left: 64 };
      const W = w - margin.left - margin.right;
      const H = HEIGHT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${w} ${HEIGHT}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const allVals = curves.flatMap((c) => c.data.map((d) => d.mean)).filter((v) => v > 0);
      const yMin = Math.max(1e-2, d3.min(allVals) ?? 1);
      const yMax = d3.max(allVals) ?? 100;
      const xScale = d3.scaleLinear().domain([1, 200]).range([0, W]);
      const yScale = d3.scaleLog().domain([yMin * 0.7, yMax * 1.4]).range([H, 0]);

      g.append('g').attr('transform', `translate(0,${H})`).call(d3.axisBottom(xScale).ticks(8))
        .selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(yScale).ticks(6, '~g'))
        .selectAll('text').style('fill', 'var(--color-text)');
      g.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');
      g.append('text').attr('x', W / 2).attr('y', H + 38).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)').style('font-size', '12px').text('feature dimension p');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -H / 2).attr('y', -48)
        .attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '12px')
        .text('test MSE (log scale)');

      // Threshold markers
      g.append('line').attr('x1', xScale(nCommitted)).attr('x2', xScale(nCommitted)).attr('y1', 0).attr('y2', H)
        .style('stroke', 'var(--color-accent)').attr('stroke-dasharray', '4 4').attr('opacity', 0.5);
      g.append('line').attr('x1', xScale(dCommitted)).attr('x2', xScale(dCommitted)).attr('y1', 0).attr('y2', H)
        .style('stroke', '#534AB7').attr('stroke-dasharray', '4 4').attr('opacity', 0.5);

      const colors: Record<string, string> = { linear: '#888', relu: 'var(--color-accent)', tanh: '#D97706' };
      const lineGen = d3.line<{ p: number; mean: number }>().x((d) => xScale(d.p))
        .y((d) => yScale(Math.max(yMin * 0.7, d.mean))).defined((d) => d.mean > 0);
      curves.forEach((c, idx) => {
        g.append('path').datum(c.data).attr('d', lineGen).attr('fill', 'none')
          .style('stroke', colors[c.name]).attr('stroke-width', 2);
        const ly = 14 + idx * 16;
        g.append('line').attr('x1', W - 100).attr('x2', W - 80).attr('y1', ly).attr('y2', ly)
          .style('stroke', colors[c.name]).attr('stroke-width', 2);
        g.append('text').attr('x', W - 74).attr('y', ly + 4)
          .style('fill', 'var(--color-text)').style('font-size', '11px').text(c.name);
      });
    },
    [containerWidth, curves, nCommitted, dCommitted],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: 'var(--font-sans)' }}>
      <svg ref={svgRef} width="100%" height={HEIGHT} role="img"
        aria-label="Random-features double descent for linear, ReLU, and tanh activations." />
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', marginTop: '0.75rem',
        fontSize: '13px', color: 'var(--color-text)' }}>
        <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', minWidth: 160 }}>
          <span>d (input dim): <strong>{dDisplay}</strong></span>
          <input type="range" min={5} max={50} step={1} value={dDisplay}
            onChange={(e) => setDDisplay(parseInt(e.target.value, 10))}
            onMouseUp={() => setDCommitted(dDisplay)} onTouchEnd={() => setDCommitted(dDisplay)}
            onKeyUp={() => setDCommitted(dDisplay)} aria-label="Input dimension d" />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', minWidth: 160 }}>
          <span>n: <strong>{nDisplay}</strong></span>
          <input type="range" min={20} max={100} step={5} value={nDisplay}
            onChange={(e) => setNDisplay(parseInt(e.target.value, 10))}
            onMouseUp={() => setNCommitted(nDisplay)} onTouchEnd={() => setNCommitted(nDisplay)}
            onKeyUp={() => setNCommitted(nDisplay)} aria-label="Training set size" />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', minWidth: 160 }}>
          <span>σ (noise): <strong>{sigmaDisplay.toFixed(2)}</strong></span>
          <input type="range" min={0.1} max={2} step={0.1} value={sigmaDisplay}
            onChange={(e) => setSigmaDisplay(parseFloat(e.target.value))}
            onMouseUp={() => setSigmaCommitted(sigmaDisplay)} onTouchEnd={() => setSigmaCommitted(sigmaDisplay)}
            onKeyUp={() => setSigmaCommitted(sigmaDisplay)} aria-label="Noise sigma" />
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <input type="checkbox" checked={showAll} onChange={(e) => setShowAll(e.target.checked)} />
          <span>show all three activations</span>
        </label>
        {!showAll && (
          <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
            <span>activation:</span>
            <select value={selectedAct}
              onChange={(e) => setSelectedAct(e.target.value as keyof typeof activations)}>
              <option value="linear">linear</option>
              <option value="relu">ReLU</option>
              <option value="tanh">tanh</option>
            </select>
          </label>
        )}
      </div>
    </div>
  );
}
