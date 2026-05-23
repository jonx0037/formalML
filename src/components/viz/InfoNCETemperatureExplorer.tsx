import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import { infoNCELoss, paletteRL, mulberry32, gaussianPair, logSumExp } from './shared/representation-learning';

// =============================================================================
// InfoNCETemperatureExplorer — §5.2 / §11.3 companion. A scatter of K positive
// pairs on the 2-D unit circle (anchors blue, positives red) — positives are
// slightly rotated copies of each anchor. The reader sweeps the temperature τ
// and batch size K, and the viz reports:
//   • the NT-Xent loss L_NCE on the current similarity matrix
//   • the lower bound on I(X; X+) = log K - L_NCE
//   • the softmax distribution for the first anchor (entropy diagnostic)
//   • the mean max-softmax across anchors (peakedness diagnostic)
// Notebook §11.3 (cell 30) shows the same diagnostics as a 1-D τ sweep; this
// viz lets the reader explore the entire (τ, K) plane interactively.
// =============================================================================

const MAX_K = 32;
const ROT_DEFAULT = 0.18; // positive-pair separation in radians

interface Batch {
  z: Float64Array; // (2K) × 2 row-major: [anchors followed by positives]
  K: number;
}

function buildBatch(K: number, rot: number, seed: number): Batch {
  const rng = mulberry32(seed);
  const z = new Float64Array(2 * K * 2);
  for (let i = 0; i < K; i++) {
    const [a, b] = gaussianPair(rng);
    const theta = Math.atan2(b, a) + (i * 2 * Math.PI) / Math.max(K, 4);
    z[i * 2] = Math.cos(theta);
    z[i * 2 + 1] = Math.sin(theta);
    z[(K + i) * 2] = Math.cos(theta + rot);
    z[(K + i) * 2 + 1] = Math.sin(theta + rot);
  }
  return { z, K };
}

function similarityMatrix(batch: Batch): Float64Array {
  const { z, K } = batch;
  const sim = new Float64Array(K * K);
  for (let i = 0; i < K; i++) {
    for (let j = 0; j < K; j++) {
      let dot = 0;
      for (let d = 0; d < 2; d++) dot += z[i * 2 + d] * z[(K + j) * 2 + d];
      sim[i * K + j] = dot;
    }
  }
  return sim;
}

function softmaxRow(sim: Float64Array, K: number, tau: number, row: number): Float64Array {
  const out = new Float64Array(K);
  const arr = new Array<number>(K);
  for (let j = 0; j < K; j++) arr[j] = sim[row * K + j] / tau;
  const lse = logSumExp(arr);
  for (let j = 0; j < K; j++) out[j] = Math.exp(arr[j] - lse);
  return out;
}

export default function InfoNCETemperatureExplorer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [tauDisplay, setTauDisplay] = useState(0.1);
  const [tauCommitted, setTauCommitted] = useState(0.1);
  const [K, setK] = useState(8);
  const [rot, setRot] = useState(ROT_DEFAULT);

  const W = Math.max(360, Math.min(width || 720, 720));
  const H = 360;
  const margin = { top: 18, right: 18, bottom: 36, left: 42 };
  const splitX = Math.round((W - margin.left - margin.right) * 0.5);
  const innerH = H - margin.top - margin.bottom;

  const batch = useMemo(() => buildBatch(K, rot, 31), [K, rot]);
  const sim = useMemo(() => similarityMatrix(batch), [batch]);
  const loss = useMemo(() => infoNCELoss(sim, K, tauCommitted), [sim, K, tauCommitted]);
  const miBound = Math.log(K) - loss;
  const softmaxAnchor0 = useMemo(() => softmaxRow(sim, K, tauCommitted, 0), [sim, K, tauCommitted]);
  const meanMaxSoftmax = useMemo(() => {
    let s = 0;
    for (let i = 0; i < K; i++) {
      const row = softmaxRow(sim, K, tauCommitted, i);
      s += Math.max(...Array.from(row));
    }
    return s / K;
  }, [sim, K, tauCommitted]);
  const entropy = useMemo(() => {
    let s = 0;
    for (let i = 0; i < K; i++) {
      const row = softmaxRow(sim, K, tauCommitted, i);
      let h = 0;
      for (let j = 0; j < K; j++) if (row[j] > 0) h += -row[j] * Math.log(row[j]);
      s += h;
    }
    return s / K;
  }, [sim, K, tauCommitted]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.attr('viewBox', `0 0 ${W} ${H}`).selectAll('*').remove();
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      // -- Left: unit-circle scatter --
      const xs = d3.scaleLinear().domain([-1.3, 1.3]).range([0, splitX]);
      const ys = d3.scaleLinear().domain([-1.3, 1.3]).range([innerH, 0]);
      g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xs).ticks(4));
      g.append('g').call(d3.axisLeft(ys).ticks(4));
      // Unit circle.
      g.append('circle').attr('cx', xs(0)).attr('cy', ys(0)).attr('r', xs(1) - xs(0)).attr('fill', 'none').attr('stroke-width', 1).attr('opacity', 0.25).style('stroke', paletteRL.data);
      // Anchors.
      g.append('g')
        .selectAll('circle.anchor')
        .data(d3.range(K))
        .join('circle')
        .attr('class', 'anchor')
        .attr('cx', (i) => xs(batch.z[i * 2]))
        .attr('cy', (i) => ys(batch.z[i * 2 + 1]))
        .attr('r', 4.5)
        .style('fill', paletteRL.signal);
      // Positives.
      g.append('g')
        .selectAll('circle.pos')
        .data(d3.range(K))
        .join('circle')
        .attr('class', 'pos')
        .attr('cx', (i) => xs(batch.z[(K + i) * 2]))
        .attr('cy', (i) => ys(batch.z[(K + i) * 2 + 1]))
        .attr('r', 4.5)
        .style('fill', paletteRL.nuisance);
      // Pair lines.
      g.append('g')
        .selectAll('line.pair')
        .data(d3.range(K))
        .join('line')
        .attr('class', 'pair')
        .attr('x1', (i) => xs(batch.z[i * 2]))
        .attr('y1', (i) => ys(batch.z[i * 2 + 1]))
        .attr('x2', (i) => xs(batch.z[(K + i) * 2]))
        .attr('y2', (i) => ys(batch.z[(K + i) * 2 + 1]))
        .attr('stroke-width', 0.8)
        .attr('opacity', 0.45)
        .style('stroke', paletteRL.data);

      // -- Right: softmax bars for anchor 0 --
      const rg = svg.append('g').attr('transform', `translate(${margin.left + splitX + 28},${margin.top})`);
      const rW = W - margin.left - margin.right - splitX - 28;
      const xb = d3.scaleBand<number>().domain(d3.range(K)).range([0, rW]).padding(0.15);
      const yb = d3.scaleLinear().domain([0, 1]).range([innerH, 0]);
      rg.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xb).tickValues(d3.range(K)));
      rg.append('g').call(d3.axisLeft(yb).ticks(5));
      rg.append('text').attr('x', rW / 2).attr('y', -6).attr('text-anchor', 'middle').attr('font-size', 10).style('fill', 'currentColor').text(`softmax row of anchor 0 (τ = ${tauCommitted.toFixed(3)})`);

      rg.selectAll('rect.sm')
        .data(d3.range(K))
        .join('rect')
        .attr('class', 'sm')
        .attr('x', (i) => xb(i)!)
        .attr('y', (i) => yb(softmaxAnchor0[i]))
        .attr('width', xb.bandwidth())
        .attr('height', (i) => innerH - yb(softmaxAnchor0[i]))
        .style('fill', (i) => (i === 0 ? paletteRL.optimum : paletteRL.signal))
        .style('fill-opacity', 0.85);
    },
    [batch, sim, softmaxAnchor0, K, tauCommitted, W, H, innerH, splitX],
  );

  return (
    <div ref={containerRef} className="not-prose my-6">
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-2 text-sm">
        <label className="flex flex-col">
          <span>Temperature τ = {tauDisplay.toFixed(3)}</span>
          <input
            type="range"
            min={Math.log(0.01)}
            max={Math.log(10)}
            step={0.05}
            value={Math.log(tauDisplay)}
            onChange={(e) => setTauDisplay(Math.exp(Number(e.target.value)))}
            onMouseUp={(e) => setTauCommitted(Math.exp(Number((e.target as HTMLInputElement).value)))}
            onTouchEnd={(e) => setTauCommitted(Math.exp(Number((e.target as HTMLInputElement).value)))}
            onKeyUp={(e) => setTauCommitted(Math.exp(Number((e.target as HTMLInputElement).value)))}
            aria-label="InfoNCE temperature"
          />
        </label>
        <label className="flex flex-col">
          <span>Batch size K = {K}</span>
          <input
            type="range"
            min={2}
            max={MAX_K}
            step={1}
            value={K}
            onChange={(e) => setK(Number(e.target.value))}
            aria-label="Batch size"
          />
        </label>
        <label className="flex flex-col">
          <span>Positive-pair rotation = {(rot * 180 / Math.PI).toFixed(0)}°</span>
          <input
            type="range"
            min={0.0}
            max={1.0}
            step={0.02}
            value={rot}
            onChange={(e) => setRot(Number(e.target.value))}
            aria-label="Positive-pair rotation"
          />
        </label>
      </div>
      <svg ref={svgRef} className="w-full" style={{ maxHeight: H }} role="img" aria-label="InfoNCE temperature explorer"/>
      <div className="text-xs opacity-75 mt-1 text-center">
        L<sub>NCE</sub> = <strong>{loss.toFixed(3)}</strong>, log K = <strong>{Math.log(K).toFixed(3)}</strong>, MI bound log K − L<sub>NCE</sub> = <strong>{miBound.toFixed(3)}</strong> nats. Entropy = <strong>{entropy.toFixed(2)}</strong> (uniform ceiling log K = {Math.log(K).toFixed(2)}); mean max-softmax = <strong>{meanMaxSoftmax.toFixed(2)}</strong>. Small τ saturates the softmax; large τ flattens it toward uniform.
      </div>
    </div>
  );
}
