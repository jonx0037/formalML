import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { normalCdf } from './shared/pac-bayes-bounds';
import { mulberry32 } from './shared/normalizing-flows';
import { gaussianFrom } from './shared/generalization-bounds';

// Run a single stream of online-FDR procedures and report wealth trajectories.
const T = 1000;
const PI0 = 0.95;
const MU_ALT = 3;
const ALPHA = 0.05;
const STOREY_LAMBDA = 0.5;
const B0 = 0.5;

type ProcSnapshot = {
  wealth: Float64Array;       // length T+1
  rejections: number[];        // indices of rejected hypotheses
  cumulativeFdp: Float64Array; // length T+1
  rejectionCount: number;
};

type StreamResult = {
  alphaInvesting: ProcSnapshot;
  lord: ProcSnapshot;
  saffron: ProcSnapshot;
  signalCount: number;
};

function lordSchedule(j: number): number {
  // γ_j ∝ log(j ∨ 2) / (j · exp(√log j))  with normalization Σ γ = 1 over first 5000 (approx).
  if (j <= 0) return 0;
  return Math.log(Math.max(j, 2)) / (j * Math.exp(Math.sqrt(Math.log(j))));
}

function makeLordWeights(): Float64Array {
  const w = new Float64Array(T + 1);
  let total = 0;
  for (let j = 1; j <= T; j++) {
    w[j] = lordSchedule(j);
    total += w[j];
  }
  // normalize approximately to 1 (we'll have the residual mass beyond T but it's negligible)
  const norm = total + 0.1; // tail mass beyond T
  for (let j = 1; j <= T; j++) w[j] /= norm;
  return w;
}

function runStream(seed: number): StreamResult {
  const rng = mulberry32(seed);
  const gauss = gaussianFrom(rng);
  // sample stream
  const pvals = new Float64Array(T);
  const isNull = new Uint8Array(T);
  let signals = 0;
  for (let t = 0; t < T; t++) {
    if (rng() < PI0) {
      pvals[t] = rng();
      isNull[t] = 1;
    } else {
      const z = MU_ALT + gauss();
      pvals[t] = 2 * (1 - normalCdf(Math.abs(z)));
      isNull[t] = 0;
      signals++;
    }
  }

  const lordW = makeLordWeights();
  // alpha-investing: α_t = η W_{t-1}, payout = ALPHA on reject
  function runAlphaInvesting(): ProcSnapshot {
    const wealth = new Float64Array(T + 1);
    const cumFdp = new Float64Array(T + 1);
    wealth[0] = ALPHA * B0;
    const rejections: number[] = [];
    let rej = 0;
    let falseRej = 0;
    const eta = 0.05;
    for (let t = 0; t < T; t++) {
      const W = wealth[t];
      const alphaT = Math.min(eta * W, ALPHA * 0.5);
      let isReject = 0;
      if (pvals[t] <= alphaT) {
        isReject = 1;
        rejections.push(t);
        rej++;
        if (isNull[t]) falseRej++;
      }
      wealth[t + 1] = Math.max(W - alphaT + isReject * ALPHA, 0);
      cumFdp[t + 1] = rej > 0 ? falseRej / rej : 0;
    }
    return { wealth, rejections, cumulativeFdp: cumFdp, rejectionCount: rej };
  }

  function runLord(): ProcSnapshot {
    const wealth = new Float64Array(T + 1);
    const cumFdp = new Float64Array(T + 1);
    wealth[0] = ALPHA * B0;
    const rejections: number[] = [];
    let rej = 0;
    let falseRej = 0;
    for (let t = 0; t < T; t++) {
      // α_t = γ_t · W_0 + α · Σ_{τ_j < t} γ_{t - τ_j}
      let alphaT = lordW[t + 1] * ALPHA * B0;
      for (const tau of rejections) {
        if (tau < t) alphaT += ALPHA * (lordW[t - tau] ?? 0);
      }
      alphaT = Math.min(alphaT, ALPHA * 5); // safety cap
      let isReject = 0;
      if (pvals[t] <= alphaT) {
        isReject = 1;
        rejections.push(t);
        rej++;
        if (isNull[t]) falseRej++;
      }
      wealth[t + 1] = Math.max(wealth[t] - alphaT + isReject * ALPHA, 0);
      cumFdp[t + 1] = rej > 0 ? falseRej / rej : 0;
    }
    return { wealth, rejections, cumulativeFdp: cumFdp, rejectionCount: rej };
  }

  function runSaffron(): ProcSnapshot {
    const wealth = new Float64Array(T + 1);
    const cumFdp = new Float64Array(T + 1);
    wealth[0] = ALPHA * B0 * (1 - STOREY_LAMBDA);
    const rejections: number[] = [];
    let rej = 0;
    let falseRej = 0;
    let candidates = 0;
    for (let t = 0; t < T; t++) {
      // α_t = (W_0 + (ψ - α) C_{t-1}) · γ̂_t — simplified version
      let alphaT = lordW[t + 1] * (ALPHA * B0 * (1 - STOREY_LAMBDA));
      for (const tau of rejections) {
        if (tau < t) alphaT += ALPHA * (1 - STOREY_LAMBDA) * (lordW[t - tau] ?? 0);
      }
      alphaT = Math.min(alphaT, ALPHA * 5);
      let isReject = 0;
      if (pvals[t] <= alphaT) {
        isReject = 1;
        rejections.push(t);
        rej++;
        if (isNull[t]) falseRej++;
      }
      if (pvals[t] <= STOREY_LAMBDA) candidates++;
      const drain = pvals[t] <= STOREY_LAMBDA ? alphaT : 0;
      wealth[t + 1] = Math.max(wealth[t] - drain + isReject * ALPHA * (1 - STOREY_LAMBDA), 0);
      cumFdp[t + 1] = rej > 0 ? falseRej / rej : 0;
    }
    return { wealth, rejections, cumulativeFdp: cumFdp, rejectionCount: rej };
  }

  return {
    alphaInvesting: runAlphaInvesting(),
    lord: runLord(),
    saffron: runSaffron(),
    signalCount: signals,
  };
}

const PROCEDURES = [
  { key: 'alphaInvesting' as const, label: 'α-investing', color: '#dc2626' },
  { key: 'lord' as const, label: 'LORD', color: '#16a34a' },
  { key: 'saffron' as const, label: 'SAFFRON', color: '#1d4ed8' },
];

export default function LORDWealthProcess() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [seed, setSeed] = useState(123);

  const result = useMemo(() => runStream(seed), [seed]);

  const W = Math.max(280, Math.min(containerWidth || 720, 720));
  const H_TOP = Math.round(W * 0.34);
  const H_BOT = Math.round(W * 0.22);
  const Mg = { t: 16, r: 18, b: 28, l: 50 };

  const wealthRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const iw = W - Mg.l - Mg.r;
      const ih = H_TOP - Mg.t - Mg.b;
      if (iw <= 0 || ih <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${Mg.l},${Mg.t})`);
      const x = d3.scaleLinear().domain([0, T]).range([0, iw]);

      let yMax = ALPHA * 1.2;
      for (const p of PROCEDURES) {
        const arr = result[p.key].wealth;
        for (let i = 0; i < arr.length; i++) if (arr[i] > yMax) yMax = arr[i];
      }
      const y = d3.scaleLinear().domain([0, yMax]).range([ih, 0]);

      for (const p of PROCEDURES) {
        const snap = result[p.key];
        const line = d3.line<number>().x((i) => x(i)).y((i) => y(snap.wealth[i]));
        g.append('path').datum(d3.range(0, T + 1) as unknown as number[]).attr('d', line).attr('fill', 'none').style('stroke', p.color).attr('stroke-width', 1.2).attr('opacity', 0.9);
        // rejection markers
        g.selectAll(`circle.${p.key}`).data(snap.rejections).enter().append('circle').attr('cx', (d) => x(d)).attr('cy', (d) => y(snap.wealth[d + 1])).attr('r', 2.5).attr('fill', p.color).attr('opacity', 0.8);
      }

      g.append('g').attr('transform', `translate(0,${ih})`).call(d3.axisBottom(x).ticks(6)).selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(y).ticks(4)).selectAll('text').style('fill', 'var(--color-text)');
      g.selectAll('.domain, line').style('stroke', 'var(--color-border)');
      g.append('text').attr('x', 8).attr('y', 10).style('fill', 'var(--color-text-secondary)').style('font-size', '0.72rem').text('wealth W_t (drops by α_t, jumps up on rejection)');
    },
    [W, H_TOP, result],
  );

  const fdpRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const iw = W - Mg.l - Mg.r;
      const ih = H_BOT - Mg.t - Mg.b;
      if (iw <= 0 || ih <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${Mg.l},${Mg.t})`);
      const x = d3.scaleLinear().domain([0, T]).range([0, iw]);
      const y = d3.scaleLinear().domain([0, 0.3]).range([ih, 0]);

      g.append('line').attr('x1', 0).attr('x2', iw).attr('y1', y(ALPHA)).attr('y2', y(ALPHA)).style('stroke', 'var(--color-text)').attr('stroke-dasharray', '4 3').attr('opacity', 0.5);
      g.append('text').attr('x', iw - 4).attr('y', y(ALPHA) - 4).attr('text-anchor', 'end').style('fill', 'var(--color-text-secondary)').style('font-size', '0.68rem').text(`α = ${ALPHA}`);

      for (const p of PROCEDURES) {
        const snap = result[p.key];
        const line = d3.line<number>().x((i) => x(i)).y((i) => y(Math.min(snap.cumulativeFdp[i], 0.3)));
        g.append('path').datum(d3.range(0, T + 1) as unknown as number[]).attr('d', line).attr('fill', 'none').style('stroke', p.color).attr('stroke-width', 1.4).attr('opacity', 0.85);
      }

      g.append('g').attr('transform', `translate(0,${ih})`).call(d3.axisBottom(x).ticks(6)).selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(y).ticks(3)).selectAll('text').style('fill', 'var(--color-text)');
      g.selectAll('.domain, line').style('stroke', 'var(--color-border)');
      g.append('text').attr('x', 8).attr('y', 10).style('fill', 'var(--color-text-secondary)').style('font-size', '0.72rem').text('cumulative FDP');
    },
    [W, H_BOT, result],
  );

  return (
    <div ref={containerRef} className="not-prose my-6 w-full">
      <div className="flex flex-wrap items-end gap-3 text-sm" style={{ color: 'var(--color-text-secondary)' }}>
        <label className="flex flex-col">
          <span>stream seed = <span className="tabular-nums" style={{ color: 'var(--color-text)' }}>{seed}</span></span>
          <input type="range" min={0} max={500} step={1} value={seed} onChange={(e) => setSeed(parseInt(e.target.value, 10))} aria-label="Random seed for stream generation" className="w-56" />
        </label>
        <div className="flex gap-3 text-xs">
          {PROCEDURES.map((p) => (
            <span key={p.key}>
              <span style={{ color: p.color }}>■</span> {p.label}: {result[p.key].rejectionCount} rej
            </span>
          ))}
        </div>
        <span className="text-xs">{result.signalCount} true signals in stream</span>
      </div>
      <svg ref={wealthRef} width={W} height={H_TOP} className="mt-2" role="img" aria-label="Wealth trajectories for three online-FDR procedures" />
      <svg ref={fdpRef} width={W} height={H_BOT} role="img" aria-label="Cumulative FDP over stream position" />
      <p className="mt-1 text-xs" style={{ color: 'var(--color-text-secondary)' }}>
        T = 1000, π₀ = 0.95, μ_alt = 3, α = 0.05. Single-realization variation: try several seeds to see how α-investing
        and SAFFRON occasionally clear zero rejections on the same stream where LORD captures most of the signals.
        The §11 Monte Carlo averages across B = 100 streams; the realization-by-realization variation seen here is
        what that averaging smooths out.
      </p>
    </div>
  );
}
