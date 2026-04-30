import { useEffect, useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  mulberry32,
  synthSinusoidWithWiggle,
  blrFitPosterior,
  blrLogMarginalLikelihood,
  gpLogMarginalLikelihood,
  looLogPredictiveBLR,
  looLogPredictiveBPR,
  looLogPredictiveGP,
  bmaWeights,
  fitStackingWeights,
  polyFeatures,
  paletteStacking,
} from './shared/bayesian-ml-stacking';

// =============================================================================
// BMACollapseAnimator — embedded alongside §2.4.
// Animates the §2.4 BMA-collapse experiment as the reader scrubs n on a timeline.
// Three closed-form candidates (BLR, BPR-2, GP-fixed-hypers); animates BMA
// vs stacking weights side-by-side. The pedagogical contrast: BMA collapses
// exponentially in n; stacking does not.
// =============================================================================

const PANEL_HEIGHT = 320;
const N_GRID_VALUES = [20, 40, 60, 80, 120, 160, 240, 320] as const;
const SIGMA_OPTIONS = [0.1, 0.2, 0.25, 0.35, 0.5] as const;

type WeightMode = 'bma' | 'stacking';

interface WeightRow {
  n: number;
  blr: number;
  bpr2: number;
  gp: number;
}

function fitMlForN(
  n: number,
  sigma: number,
  seed: number,
): { bma: WeightRow; stack: WeightRow } {
  const rng = mulberry32(seed + n);
  const { x, y } = synthSinusoidWithWiggle(n, sigma, rng);

  // Comparable marginal log-likelihoods for all three closed-form candidates.
  // BLR/BPR-2 marginalize over the weight prior under NIG conjugacy
  // (`blrLogMarginalLikelihood`); GP at fixed kernel hypers marginalizes over
  // the latent f via Rasmussen & Williams 5.8 (`gpLogMarginalLikelihood`).
  // Putting all three on the same log p(y | model) scale is what makes the BMA
  // weights mathematically meaningful — earlier versions mixed log-evidences
  // with summed LOO ELPDs, which is pseudo-BMA at best.
  const Xblr = polyFeatures(x, 1);
  const Xbpr2 = polyFeatures(x, 2);
  const blrPost = blrFitPosterior(Xblr, y);
  const bpr2Post = blrFitPosterior(Xbpr2, y);
  const logZblr = blrLogMarginalLikelihood(blrPost);
  const logZbpr2 = blrLogMarginalLikelihood(bpr2Post);
  const logZgp = gpLogMarginalLikelihood(x, y, {
    lengthScale: 0.1,
    outputVar: 1.0,
    noiseVar: sigma * sigma,
  });
  const wBma = bmaWeights([logZblr, logZbpr2, logZgp]);

  // Stacking weights from leave-one-out log predictives. The closed-form GP LOO
  // (Rasmussen & Williams 5.12–5.13) makes this O(n³) per candidate, so the
  // n-grid up to 320 is cheap to recompute on every (sigma, seed) change.
  const Lblr = looLogPredictiveBLR(Xblr, y);
  const Lbpr2 = looLogPredictiveBPR(x, y, 2);
  const Lgp = looLogPredictiveGP(x, y, { lengthScale: 0.1, outputVar: 1.0, noiseVar: sigma * sigma });
  const L = new Float64Array(n * 3);
  for (let i = 0; i < n; i++) {
    L[i * 3 + 0] = Lblr[i];
    L[i * 3 + 1] = Lbpr2[i];
    L[i * 3 + 2] = Lgp[i];
  }
  const stack = fitStackingWeights(L, n, 3);

  return {
    bma: { n, blr: wBma[0], bpr2: wBma[1], gp: wBma[2] },
    stack: { n, blr: stack.weights[0], bpr2: stack.weights[1], gp: stack.weights[2] },
  };
}

export default function BMACollapseAnimator() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();

  const [sigma, setSigma] = useState<number>(0.25);
  const [seed, setSeed] = useState<number>(20260430);
  const [mode, setMode] = useState<WeightMode>('bma');
  const [playing, setPlaying] = useState<boolean>(false);
  const [nIdx, setNIdx] = useState<number>(2);

  const series = useMemo(() => {
    return N_GRID_VALUES.map((nv) => fitMlForN(nv, sigma, seed));
  }, [sigma, seed]);

  // Autoplay timer.
  useEffect(() => {
    if (!playing) return;
    const t = setInterval(() => {
      setNIdx((i) => (i + 1) % N_GRID_VALUES.length);
    }, 850);
    return () => clearInterval(t);
  }, [playing]);

  const ref = useD3(
    (svg) => {
      const w = width || 720;
      const height = PANEL_HEIGHT;
      const margin = { top: 18, right: 18, bottom: 38, left: 44 };
      svg.attr('width', w).attr('height', height);
      svg.selectAll('*').remove();

      const innerW = w - margin.left - margin.right;
      const innerH = height - margin.top - margin.bottom;

      const xScale = d3
        .scaleLog()
        .domain([N_GRID_VALUES[0], N_GRID_VALUES[N_GRID_VALUES.length - 1]])
        .range([0, innerW]);
      const yScale = d3.scaleLinear().domain([0, 1]).range([innerH, 0]);

      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const xAxis = d3
        .axisBottom(xScale)
        .tickValues([...N_GRID_VALUES])
        .tickFormat((d) => String(d));
      g.append('g').attr('transform', `translate(0,${innerH})`).call(xAxis);
      g.append('g').call(d3.axisLeft(yScale).ticks(5));
      g.append('text')
        .attr('transform', `translate(${innerW / 2},${innerH + 30})`)
        .attr('text-anchor', 'middle')
        .attr('font-size', 12)
        .text('n (training size)');
      g.append('text')
        .attr('transform', `translate(-32,${innerH / 2}) rotate(-90)`)
        .attr('text-anchor', 'middle')
        .attr('font-size', 12)
        .text(mode === 'bma' ? 'BMA weight' : 'Stacking weight');

      // Stacked-area data.
      const rows = series.map((s) => (mode === 'bma' ? s.bma : s.stack));
      const stack = d3.stack<WeightRow, keyof Pick<WeightRow, 'blr' | 'bpr2' | 'gp'>>().keys(['blr', 'bpr2', 'gp']);
      const stacked = stack(rows);

      const colorMap: Record<string, string> = {
        blr: paletteStacking.blr,
        bpr2: paletteStacking.bpr,
        gp: paletteStacking.gp,
      };

      const area = d3
        .area<d3.SeriesPoint<WeightRow>>()
        .x((d) => xScale(d.data.n))
        .y0((d) => yScale(d[0]))
        .y1((d) => yScale(d[1]));

      g.selectAll<SVGPathElement, d3.Series<WeightRow, keyof Pick<WeightRow, 'blr' | 'bpr2' | 'gp'>>>('path.layer')
        .data(stacked)
        .join('path')
        .attr('class', 'layer')
        .attr('fill', (d) => colorMap[d.key as string])
        .attr('fill-opacity', 0.85)
        .attr('d', area);

      // Frozen-frame indicator.
      const nFrozen = N_GRID_VALUES[nIdx];
      g.append('line')
        .attr('x1', xScale(nFrozen))
        .attr('x2', xScale(nFrozen))
        .attr('y1', 0)
        .attr('y2', innerH)
        .attr('stroke', '#000')
        .attr('stroke-width', 1.4)
        .attr('stroke-dasharray', '4 3');

      // Frozen-frame weight readout.
      const fr = mode === 'bma' ? series[nIdx].bma : series[nIdx].stack;
      const txt = g.append('text').attr('x', innerW - 4).attr('y', 14).attr('text-anchor', 'end').attr('font-size', 11);
      txt
        .append('tspan')
        .attr('fill', paletteStacking.blr)
        .text(`Linear ${fr.blr.toFixed(2)}  `);
      txt
        .append('tspan')
        .attr('fill', paletteStacking.bpr)
        .text(`Poly-2 ${fr.bpr2.toFixed(2)}  `);
      txt
        .append('tspan')
        .attr('fill', paletteStacking.gp)
        .text(`GP ${fr.gp.toFixed(2)}`);
    },
    [series, mode, nIdx, width],
  );

  return (
    <div className="my-6 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-soft)] p-4">
      <div className="mb-3 flex flex-wrap gap-3 text-sm">
        <label>
          σ:&nbsp;
          <select value={sigma} onChange={(e) => setSigma(Number(e.target.value))} className="rounded border px-1">
            {SIGMA_OPTIONS.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
        </label>
        <label>
          weights:&nbsp;
          <select value={mode} onChange={(e) => setMode(e.target.value as WeightMode)} className="rounded border px-1">
            <option value="bma">BMA</option>
            <option value="stacking">Stacking</option>
          </select>
        </label>
        <label>
          n:&nbsp;
          <input
            type="range"
            min={0}
            max={N_GRID_VALUES.length - 1}
            value={nIdx}
            onChange={(e) => setNIdx(Number(e.target.value))}
          />
          <span className="ml-1 font-mono">{N_GRID_VALUES[nIdx]}</span>
        </label>
        <button
          type="button"
          onClick={() => setPlaying((p) => !p)}
          className="rounded border bg-white px-2 py-0.5 hover:bg-gray-50"
        >
          {playing ? '⏸ pause' : '▶ play'}
        </button>
        <button
          type="button"
          onClick={() => setSeed((s) => s + 1)}
          className="rounded border bg-white px-2 py-0.5 hover:bg-gray-50"
        >
          Reseed
        </button>
      </div>
      <div ref={containerRef} className="w-full">
        <svg ref={ref} />
      </div>
    </div>
  );
}
