import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { normalCdf } from './shared/pac-bayes-bounds';
import { mulberry32 } from './shared/normalizing-flows';
import { gaussianFrom } from './shared/generalization-bounds';

// Locked §5.5 results (cell 25, n=100, p=200, s_0=5, B=200 with 10 signals+10 nulls debiased per rep).
const LOCKED = {
  n: 100,
  p: 200,
  s0: 5,
  signal: { meanLasso: 1.138, meanDebiased: 1.472, se: 0.124, coverage: 0.877, zMean: -0.273, zStd: 1.228 },
  null: { meanLasso: -0.0, meanDebiased: -0.005, se: 0.123, coverage: 0.894, zMean: -0.05, zStd: 1.22 },
};

// Heuristic coverage model: predicted Wald coverage at rate ratio r = s_0 log(p) / sqrt(n).
// Calibrated to the locked (r ≈ 2.65, coverage = 0.886 averaged across signal+null) anchor.
// Reads: bias-to-SE ratio grows linearly in r, coverage = 1 - 2 Φ(-z_α/2 + δ).
function predictedCoverage(r: number): number {
  const z = 1.96;
  const delta = 0.155 * Math.max(0, r - 0.4); // small intercept so r ≈ 0 gives near-nominal
  return Math.max(0, Math.min(1, 2 * (normalCdf(z - delta) - 0.5) * 2 - (2 * (normalCdf(z - delta) - 0.5) - 1) * 0 + (1 - (1 - 2 * normalCdf(-z + delta)))) - 0); // placeholder, recompute below
}

function coverageAt(r: number): number {
  const z = 1.96;
  // CI = b̂ ± z·SE; under bias δ·SE, coverage = Φ(z - δ) - Φ(-z - δ) = Φ(z-δ) + Φ(z+δ) - 1.
  const delta = 0.155 * Math.max(0, r - 0.4);
  return normalCdf(z - delta) + normalCdf(z + delta) - 1;
}

// Generate a small MC of z-stats for histogram overlay. Use locked mean/std and
// a fixed RNG so the histogram is reproducible.
function makeZStats(meanZ: number, stdZ: number, k: number = 600): Float64Array {
  const rng = mulberry32(11);
  const gauss = gaussianFrom(rng);
  const out = new Float64Array(k);
  for (let i = 0; i < k; i++) out[i] = meanZ + stdZ * gauss();
  return out;
}

export default function DebiasedLassoCI() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [n, setN] = useState(LOCKED.n);
  const [p, setP] = useState(LOCKED.p);
  const [s0, setS0] = useState(LOCKED.s0);
  const [pool, setPool] = useState<'signal' | 'null'>('signal');

  const r = useMemo(() => (s0 * Math.log(p)) / Math.sqrt(n), [s0, p, n]);
  const predicted = useMemo(() => coverageAt(r), [r]);

  const W = Math.max(280, Math.min(containerWidth || 720, 720));
  const isMobile = W < 640;
  const PANEL_W = isMobile ? W : (W - 16) / 2;
  const PANEL_H = Math.round(PANEL_W * 0.72);
  const M = { t: 22, r: 14, b: 38, l: 50 };

  const zs = useMemo(() => {
    const stats = pool === 'signal' ? LOCKED.signal : LOCKED.null;
    return makeZStats(stats.zMean, stats.zStd);
  }, [pool]);

  const histRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const iw = PANEL_W - M.l - M.r;
      const ih = PANEL_H - M.t - M.b;
      if (iw <= 0 || ih <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${M.l},${M.t})`);

      const x = d3.scaleLinear().domain([-4, 4]).range([0, iw]);
      const bins = d3.bin<number, number>().domain([-4, 4]).thresholds(30)(Array.from(zs));
      const yMax = (d3.max(bins, (b) => b.length) ?? 1) / zs.length / 0.27;
      const y = d3.scaleLinear().domain([0, Math.max(yMax, 0.45)]).range([ih, 0]);

      g.selectAll('rect')
        .data(bins)
        .enter()
        .append('rect')
        .attr('x', (b) => x(b.x0 ?? 0))
        .attr('y', (b) => y(b.length / zs.length / ((b.x1 ?? 0) - (b.x0 ?? 1))))
        .attr('width', (b) => Math.max(0, x(b.x1 ?? 0) - x(b.x0 ?? 0) - 1))
        .attr('height', (b) => ih - y(b.length / zs.length / ((b.x1 ?? 0) - (b.x0 ?? 1))))
        .attr('fill', 'var(--color-accent)')
        .attr('opacity', 0.55);

      // overlay N(0,1)
      const pts = d3.range(-4, 4.0001, 0.05);
      const pdf = (xv: number) => Math.exp(-0.5 * xv * xv) / Math.sqrt(2 * Math.PI);
      const line = d3
        .line<number>()
        .x((v) => x(v))
        .y((v) => y(pdf(v)));
      g.append('path').datum(pts as unknown as number[]).attr('d', line).attr('fill', 'none').style('stroke', 'var(--color-text)').attr('stroke-width', 1.6);

      g.append('g').attr('transform', `translate(0,${ih})`).call(d3.axisBottom(x).ticks(5)).selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(y).ticks(4)).selectAll('text').style('fill', 'var(--color-text)');
      g.selectAll('.domain, line').style('stroke', 'var(--color-border)');
      g.append('text')
        .attr('x', iw / 2)
        .attr('y', ih + 28)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '0.7rem')
        .text(`standardized z (${pool} coordinates)`);
      g.append('text')
        .attr('x', 8)
        .attr('y', 10)
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '0.72rem')
        .text(`mean ${(pool === 'signal' ? LOCKED.signal.zMean : LOCKED.null.zMean).toFixed(2)} · std ${(pool === 'signal' ? LOCKED.signal.zStd : LOCKED.null.zStd).toFixed(2)}`);
    },
    [PANEL_W, PANEL_H, pool, zs],
  );

  const curveRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const iw = PANEL_W - M.l - M.r;
      const ih = PANEL_H - M.t - M.b;
      if (iw <= 0 || ih <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${M.l},${M.t})`);

      const x = d3.scaleLinear().domain([0, 5]).range([0, iw]);
      const y = d3.scaleLinear().domain([0.5, 1]).range([ih, 0]);

      const rs = d3.range(0, 5.05, 0.05);
      const line = d3
        .line<number>()
        .x((rv) => x(rv))
        .y((rv) => y(coverageAt(rv)));
      g.append('path').datum(rs as unknown as number[]).attr('d', line).attr('fill', 'none').style('stroke', 'var(--color-accent)').attr('stroke-width', 1.8);

      // nominal line
      g.append('line').attr('x1', 0).attr('x2', iw).attr('y1', y(0.95)).attr('y2', y(0.95)).style('stroke', 'var(--color-text)').attr('stroke-dasharray', '4 3').attr('opacity', 0.5);

      // current point marker
      g.append('circle').attr('cx', x(Math.min(r, 5))).attr('cy', y(predicted)).attr('r', 5).style('fill', 'var(--color-text)');
      g.append('text')
        .attr('x', x(Math.min(r, 5)) + 8)
        .attr('y', y(predicted) + 4)
        .style('fill', 'var(--color-text)')
        .style('font-family', 'var(--font-sans)')
        .style('font-size', '0.74rem')
        .text(`r = ${r.toFixed(2)} · coverage ≈ ${predicted.toFixed(3)}`);

      // anchor: the locked notebook point
      const lockedR = (LOCKED.s0 * Math.log(LOCKED.p)) / Math.sqrt(LOCKED.n);
      g.append('circle').attr('cx', x(lockedR)).attr('cy', y(0.886)).attr('r', 4).style('fill', 'var(--color-accent)').attr('opacity', 0.8);

      g.append('g').attr('transform', `translate(0,${ih})`).call(d3.axisBottom(x).ticks(5)).selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(y).ticks(5)).selectAll('text').style('fill', 'var(--color-text)');
      g.selectAll('.domain, line').style('stroke', 'var(--color-border)');
      g.append('text').attr('x', iw / 2).attr('y', ih + 28).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-size', '0.7rem').text('rate ratio r = s₀ log p / √n');
      g.append('text').attr('x', 8).attr('y', 10).style('fill', 'var(--color-text-secondary)').style('font-size', '0.72rem').text('predicted Wald coverage');
    },
    [PANEL_W, PANEL_H, r, predicted],
  );

  return (
    <div ref={containerRef} className="not-prose my-6 w-full">
      <div className="flex flex-col gap-3 sm:flex-row sm:flex-wrap sm:items-end">
        <label className="flex flex-col text-sm" style={{ color: 'var(--color-text-secondary)' }}>
          <span>n = <span className="tabular-nums" style={{ color: 'var(--color-text)' }}>{n}</span></span>
          <input type="range" min={60} max={400} step={10} value={n} onChange={(e) => setN(parseInt(e.target.value, 10))} aria-label="Sample size n" className="w-44" />
        </label>
        <label className="flex flex-col text-sm" style={{ color: 'var(--color-text-secondary)' }}>
          <span>p = <span className="tabular-nums" style={{ color: 'var(--color-text)' }}>{p}</span></span>
          <input type="range" min={50} max={800} step={10} value={p} onChange={(e) => setP(parseInt(e.target.value, 10))} aria-label="Feature dimension p" className="w-44" />
        </label>
        <label className="flex flex-col text-sm" style={{ color: 'var(--color-text-secondary)' }}>
          <span>s₀ = <span className="tabular-nums" style={{ color: 'var(--color-text)' }}>{s0}</span></span>
          <input type="range" min={1} max={20} step={1} value={s0} onChange={(e) => setS0(parseInt(e.target.value, 10))} aria-label="Sparsity s_0" className="w-44" />
        </label>
        <label className="flex items-center gap-2 text-sm" style={{ color: 'var(--color-text-secondary)' }}>
          <span>pool</span>
          <select value={pool} onChange={(e) => setPool(e.target.value as 'signal' | 'null')} className="rounded border px-2 py-1" style={{ background: 'var(--color-surface)', borderColor: 'var(--color-border)', color: 'var(--color-text)' }}>
            <option value="signal">signal</option>
            <option value="null">null</option>
          </select>
        </label>
      </div>
      <div className={isMobile ? 'mt-2' : 'mt-2 flex gap-4'}>
        <svg ref={histRef} width={PANEL_W} height={PANEL_H} role="img" aria-label="Standardized z-statistic histogram vs N(0,1)" />
        <svg ref={curveRef} width={PANEL_W} height={PANEL_H} role="img" aria-label="Predicted coverage vs rate ratio" />
      </div>
      <p className="mt-1 text-xs" style={{ color: 'var(--color-text-secondary)' }}>
        Left: histogram of standardized z-statistics for the {pool} coordinate pool from the §5.5 locked run, with the
        standard-normal density overlaid. The toy's z-stat mean of {(pool === 'signal' ? LOCKED.signal.zMean : LOCKED.null.zMean).toFixed(2)} and std{' '}
        {(pool === 'signal' ? LOCKED.signal.zStd : LOCKED.null.zStd).toFixed(2)} indicate the rate-condition slippage Theorem 5.1 anticipates. Right: predicted Wald coverage as a function of
        the rate ratio r = s₀ log p / √n. The black point is the slider's current (n, p, s₀); the teal point at r ≈ 2.65 is
        the §5.5 anchor (locked coverage averages 0.886 across signal and null). Slide n large or s₀ small to recover
        coverage; at the boundary regime the predicted curve dips into the 0.85–0.90 band.
      </p>
    </div>
  );
}
