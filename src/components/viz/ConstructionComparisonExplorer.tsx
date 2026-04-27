import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  cqrIntervalPI,
  fitPredictRidge,
  hlIntervalPI,
  mulberry32,
  palettePI,
  pureQrIntervalPI,
  scenarioAPI,
  scenarioBPI,
  scenarioCPI,
  scenarioDPI,
  splitConformalIntervalPI,
} from './shared/nonparametric-ml';

// =============================================================================
// ConstructionComparisonExplorer — §6 marquee.
//
// Scenario dropdown (A/B/C/D), n_cal slider [100, 2000], construction
// multi-select. Large scatter with selected bands overlaid; sidebar with four
// readouts per construction (marginal, mean width, cond range, fit-time).
// Static §6.2 4×4 table below with live markers on the active row.
// =============================================================================

const ALPHA = 0.1;
const SM_BREAKPOINT = 900;
const SCATTER_HEIGHT = 380;
const N_TRAIN = 500;
const N_TEST = 1500;
const N_BINS = 8;

type Scenario = 'A' | 'B' | 'C' | 'D';
type Construction = 'split' | 'qr' | 'cqr' | 'hl';

const SCENARIOS: Record<Scenario, { label: string; gen: typeof scenarioAPI; xMin: number; xMax: number; muFn: (x: number) => number }> = {
  A: { label: 'A: Homoscedastic Gaussian', gen: scenarioAPI, xMin: -3, xMax: 3, muFn: Math.sin },
  B: { label: 'B: Heteroscedastic (RE1)',  gen: scenarioBPI, xMin: -3, xMax: 3, muFn: Math.sin },
  C: { label: 'C: Heavy-tailed (RE2)',     gen: scenarioCPI, xMin: -2, xMax: 2, muFn: (x) => 0.4 * Math.cos(Math.PI * x) },
  D: { label: 'D: Contaminated noise',     gen: scenarioDPI, xMin: -3, xMax: 3, muFn: Math.sin },
};

const CONSTRUCTION_META: Record<Construction, { label: string; color: string; light: string }> = {
  split: { label: 'split conformal', color: palettePI.blue,   light: palettePI.lightBlue },
  qr:    { label: 'pure QR',         color: palettePI.green,  light: palettePI.lightGreen },
  cqr:   { label: 'CQR',             color: palettePI.teal,   light: '#CCFBF1' },
  hl:    { label: 'HL',              color: palettePI.purple, light: palettePI.lightPurple },
};

// Notebook-verified reference values for the §6.2 table (averages over n_rep = 300).
// Used to overlay markers on the static table panel.
const TABLE_REF: Record<Scenario, Record<Construction, { marg: number; width: number; cond: number; ms: number }>> = {
  A: {
    split: { marg: 0.899, width: 1.660, cond: 0.056, ms: 0.4 },
    qr:    { marg: 0.896, width: 1.651, cond: 0.071, ms: 45.1 },
    cqr:   { marg: 0.900, width: 1.680, cond: 0.083, ms: 16.6 },
    hl:    { marg: 0.757, width: 1.180, cond: 0.080, ms: 4.3 },
  },
  B: {
    split: { marg: 0.899, width: 1.768, cond: 0.242, ms: 0.4 },
    qr:    { marg: 0.897, width: 1.657, cond: 0.110, ms: 44.3 },
    cqr:   { marg: 0.900, width: 1.686, cond: 0.115, ms: 16.6 },
    hl:    { marg: 0.789, width: 1.257, cond: 0.384, ms: 4.2 },
  },
  C: {
    split: { marg: 0.900, width: 2.978, cond: 0.058, ms: 0.4 },
    qr:    { marg: 0.896, width: 2.953, cond: 0.072, ms: 42.6 },
    cqr:   { marg: 0.901, width: 3.067, cond: 0.085, ms: 16.1 },
    hl:    { marg: 0.817, width: 2.240, cond: 0.084, ms: 4.2 },
  },
  D: {
    split: { marg: 0.899, width: 1.146, cond: 0.064, ms: 0.4 },
    qr:    { marg: 0.895, width: 1.133, cond: 0.072, ms: 44.3 },
    cqr:   { marg: 0.901, width: 1.183, cond: 0.077, ms: 16.6 },
    hl:    { marg: 0.827, width: 0.928, cond: 0.083, ms: 4.2 },
  },
};

const fmt = (x: number, digits = 3) => x.toFixed(digits);

interface ConstructionResult {
  lo: Float64Array;
  hi: Float64Array;
  loGrid: Float64Array;
  hiGrid: Float64Array;
  marg: number;
  width: number;
  cond: number;
  ms: number;
}

export default function ConstructionComparisonExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const isStacked = containerWidth > 0 && containerWidth < SM_BREAKPOINT;

  const [scenario, setScenario] = useState<Scenario>('B');
  const [nCal, setNCal] = useState(500);
  const [enabled, setEnabled] = useState<Record<Construction, boolean>>({
    split: true,
    qr: true,
    cqr: true,
    hl: true,
  });
  const [seed, setSeed] = useState(43);

  const result = useMemo(() => {
    const rng = mulberry32(seed);
    const scen = SCENARIOS[scenario];
    const tr = scen.gen(N_TRAIN, rng);
    const ca = scen.gen(nCal, rng);
    const te = scen.gen(N_TEST, rng);

    const grid = new Float64Array(120);
    for (let i = 0; i < 120; i++) grid[i] = scen.xMin + ((scen.xMax - scen.xMin) * i) / 119;

    const out: Partial<Record<Construction, ConstructionResult>> = {};

    function compute(c: Construction): ConstructionResult {
      const t0 = performance.now();
      let lo: Float64Array;
      let hi: Float64Array;
      let loGrid: Float64Array;
      let hiGrid: Float64Array;
      if (c === 'split') {
        const r = splitConformalIntervalPI(tr.X, tr.Y, ca.X, ca.Y, te.X, ALPHA);
        lo = r.lo; hi = r.hi;
        const muG = fitPredictRidge(Float64Array.from(tr.X), Float64Array.from(tr.Y), grid, 0.1);
        loGrid = new Float64Array(grid.length);
        hiGrid = new Float64Array(grid.length);
        for (let i = 0; i < grid.length; i++) {
          loGrid[i] = muG[i] - r.qHat;
          hiGrid[i] = muG[i] + r.qHat;
        }
      } else if (c === 'qr') {
        const Xfull = new Float64Array(N_TRAIN + nCal);
        const Yfull = new Float64Array(N_TRAIN + nCal);
        for (let i = 0; i < N_TRAIN; i++) { Xfull[i] = tr.X[i]; Yfull[i] = tr.Y[i]; }
        for (let i = 0; i < nCal; i++) { Xfull[N_TRAIN + i] = ca.X[i]; Yfull[N_TRAIN + i] = ca.Y[i]; }
        const r = pureQrIntervalPI(Xfull, Yfull, te.X, ALPHA);
        lo = r.lo; hi = r.hi;
        const r2 = pureQrIntervalPI(Xfull, Yfull, grid, ALPHA);
        loGrid = r2.lo; hiGrid = r2.hi;
      } else if (c === 'cqr') {
        const r = cqrIntervalPI(tr.X, tr.Y, ca.X, ca.Y, te.X, ALPHA);
        lo = r.lo; hi = r.hi;
        const r2 = cqrIntervalPI(tr.X, tr.Y, ca.X, ca.Y, grid, ALPHA);
        loGrid = r2.lo; hiGrid = r2.hi;
      } else {
        const r = hlIntervalPI(tr.X, tr.Y, ca.X, ca.Y, te.X, ALPHA);
        lo = r.lo; hi = r.hi;
        const muG = fitPredictRidge(Float64Array.from(tr.X), Float64Array.from(tr.Y), grid, 0.1);
        loGrid = new Float64Array(grid.length);
        hiGrid = new Float64Array(grid.length);
        for (let i = 0; i < grid.length; i++) {
          loGrid[i] = muG[i] + r.ALo;
          hiGrid[i] = muG[i] + r.AHi;
        }
      }
      const ms = performance.now() - t0;

      let cov = 0;
      let widthSum = 0;
      const inBand = new Uint8Array(N_TEST);
      for (let i = 0; i < N_TEST; i++) {
        const ok = te.Y[i] >= lo[i] && te.Y[i] <= hi[i];
        if (ok) { cov++; inBand[i] = 1; }
        widthSum += hi[i] - lo[i];
      }
      const marg = cov / N_TEST;
      const width = widthSum / N_TEST;

      const binCounts = new Int32Array(N_BINS);
      const binCovered = new Int32Array(N_BINS);
      const binWidth = (scen.xMax - scen.xMin) / N_BINS;
      for (let i = 0; i < N_TEST; i++) {
        let b = Math.floor((te.X[i] - scen.xMin) / binWidth);
        if (b < 0) b = 0;
        if (b >= N_BINS) b = N_BINS - 1;
        binCounts[b]++;
        if (inBand[i]) binCovered[b]++;
      }
      let mn = Infinity, mx = -Infinity;
      for (let b = 0; b < N_BINS; b++) {
        if (binCounts[b] < 5) continue;
        const c = binCovered[b] / binCounts[b];
        if (c < mn) mn = c;
        if (c > mx) mx = c;
      }
      const cond = mx - mn;
      return { lo, hi, loGrid, hiGrid, marg, width, cond, ms };
    }

    for (const c of ['split', 'qr', 'cqr', 'hl'] as Construction[]) {
      out[c] = compute(c);
    }

    // Subsample scatter
    const sub = 350;
    const stride = Math.floor(N_TEST / sub);
    const sX = new Float64Array(sub);
    const sY = new Float64Array(sub);
    for (let i = 0; i < sub; i++) { sX[i] = te.X[i * stride]; sY[i] = te.Y[i * stride]; }

    return { results: out as Record<Construction, ConstructionResult>, grid: Array.from(grid), sX, sY, scen };
  }, [scenario, nCal, seed]);

  const scatterRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const margin = { top: 14, right: 16, bottom: 38, left: 50 };
      const innerW = containerWidth - margin.left - margin.right;
      const innerH = SCATTER_HEIGHT - margin.top - margin.bottom;
      if (innerW <= 0 || innerH <= 0) return;

      const enabledList = (Object.keys(enabled) as Construction[]).filter((c) => enabled[c]);
      const xs = d3.scaleLinear().domain([result.scen.xMin, result.scen.xMax]).range([0, innerW]);
      const allY: number[] = [...Array.from(result.sY)];
      for (const c of enabledList) {
        allY.push(...Array.from(result.results[c].loGrid));
        allY.push(...Array.from(result.results[c].hiGrid));
      }
      const ySorted = result.sY.slice().sort();
      const yLo = Math.min(Math.min(...allY), ySorted[Math.floor(ySorted.length * 0.005)] - 0.3);
      const yHi = Math.max(Math.max(...allY), ySorted[Math.floor(ySorted.length * 0.995)] + 0.3);
      const yScale = d3.scaleLinear().domain([Math.max(yLo, -8), Math.min(yHi, 8)]).range([innerH, 0]);

      const g = svg.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);
      g.append('g').attr('transform', `translate(0, ${innerH})`).call(d3.axisBottom(xs).ticks(7))
        .call((sel) => sel.selectAll('line').style('stroke', 'var(--color-border)').style('opacity', 0.3))
        .call((sel) => sel.selectAll('text').style('fill', 'var(--color-text-secondary)').style('font-size', '10px'))
        .call((sel) => sel.select('.domain').style('stroke', 'var(--color-border)'));
      g.append('g').call(d3.axisLeft(yScale).ticks(6))
        .call((sel) => sel.selectAll('line').style('stroke', 'var(--color-border)').style('opacity', 0.3))
        .call((sel) => sel.selectAll('text').style('fill', 'var(--color-text-secondary)').style('font-size', '10px'))
        .call((sel) => sel.select('.domain').style('stroke', 'var(--color-border)'));

      g.append('text').attr('x', innerW / 2).attr('y', innerH + 30).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('x');
      g.append('text').attr('transform', `translate(-38, ${innerH / 2}) rotate(-90)`).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('y');

      // Bands (back to front so the higher-priority bands sit on top)
      const drawOrder: Construction[] = ['split', 'hl', 'qr', 'cqr'];
      for (const c of drawOrder) {
        if (!enabled[c]) continue;
        const r = result.results[c];
        const meta = CONSTRUCTION_META[c];
        // For QR/CQR, draw filled area; for split/HL (constant-width), only edge curves to reduce visual clutter
        if (c === 'cqr' || c === 'qr') {
          const area = d3.area<number>()
            .x((_d, i) => xs(result.grid[i]))
            .y0((_d, i) => yScale(r.loGrid[i]))
            .y1((_d, i) => yScale(r.hiGrid[i]))
            .curve(d3.curveCatmullRom);
          g.append('path').datum(result.grid).attr('d', area)
            .style('fill', meta.light).style('opacity', 0.4);
        }
        for (const arr of [r.loGrid, r.hiGrid]) {
          g.append('path')
            .datum(result.grid.map((_, i) => arr[i]))
            .attr('d', d3.line<number>().x((_d, i) => xs(result.grid[i])).y((d) => yScale(d)).curve(d3.curveCatmullRom))
            .style('fill', 'none').style('stroke', meta.color).style('stroke-width', 1.4)
            .style('stroke-dasharray', c === 'split' ? '5 3' : c === 'hl' ? '2 3' : 'none');
        }
      }

      // True μ(x) curve
      const muPts = result.grid.map((x) => [x, result.scen.muFn(x)] as [number, number]);
      g.append('path').datum(muPts)
        .attr('d', d3.line<[number, number]>().x((d) => xs(d[0])).y((d) => yScale(d[1])))
        .style('fill', 'none').style('stroke', 'var(--color-text-secondary)').style('stroke-width', 1.2).style('stroke-dasharray', '4 3');

      // Scatter
      for (let i = 0; i < result.sX.length; i++) {
        const cy = yScale(result.sY[i]);
        if (cy < 0 || cy > innerH) continue;
        g.append('circle').attr('cx', xs(result.sX[i])).attr('cy', cy).attr('r', 1.6)
          .style('fill', 'var(--color-text-secondary)').style('opacity', 0.4);
      }

      // Legends
      let yOff = 14;
      for (const c of ['split', 'qr', 'cqr', 'hl'] as Construction[]) {
        if (!enabled[c]) continue;
        const meta = CONSTRUCTION_META[c];
        g.append('text').attr('x', innerW - 6).attr('y', yOff).attr('text-anchor', 'end')
          .style('font-size', '10px').style('fill', meta.color)
          .text(`${meta.label} (cov ${fmt(result.results[c].marg, 3)}, w ${fmt(result.results[c].width, 2)})`);
        yOff += 14;
      }
    },
    [result, containerWidth, enabled, scenario],
  );

  const tableRef = result.results;

  return (
    <div
      ref={containerRef}
      className="my-6 rounded-lg border border-[var(--color-border)] bg-[var(--color-muted-bg)] p-3"
    >
      <svg ref={scatterRef} width={containerWidth} height={SCATTER_HEIGHT} role="img" aria-label="Construction comparison overlay" />

      <div className="mt-3 flex flex-wrap items-center gap-x-6 gap-y-3 text-sm">
        <label className="flex items-center gap-2 text-[var(--color-text-secondary)]">
          <span className="font-mono text-xs">scenario</span>
          <select
            value={scenario}
            onChange={(e) => setScenario(e.target.value as Scenario)}
            className="rounded border border-[var(--color-border)] bg-[var(--color-surface)] px-2 py-1 text-xs"
          >
            {(['A', 'B', 'C', 'D'] as Scenario[]).map((s) => (
              <option key={s} value={s}>{SCENARIOS[s].label}</option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-2 text-[var(--color-text-secondary)]">
          <span className="font-mono text-xs">n_cal</span>
          <input
            type="range"
            min={100}
            max={2000}
            step={100}
            value={nCal}
            onChange={(e) => setNCal(Number(e.target.value))}
            className="accent-[var(--color-accent)]"
          />
          <span className="font-mono w-12 text-right">{nCal}</span>
        </label>
        <div className="flex flex-wrap items-center gap-3 text-xs">
          {(['split', 'qr', 'cqr', 'hl'] as Construction[]).map((c) => (
            <label key={c} className="flex items-center gap-1.5" style={{ color: CONSTRUCTION_META[c].color }}>
              <input
                type="checkbox"
                checked={enabled[c]}
                onChange={(e) => setEnabled((s) => ({ ...s, [c]: e.target.checked }))}
                className="accent-[var(--color-accent)]"
              />
              {CONSTRUCTION_META[c].label}
            </label>
          ))}
        </div>
        <button
          type="button"
          onClick={() => setSeed((s) => s + 1)}
          className="rounded border border-[var(--color-accent)] bg-[var(--color-accent)] px-3 py-1 text-xs font-medium text-white transition hover:opacity-90"
        >
          re-randomize
        </button>
      </div>

      {/* Live readout panel + reference table */}
      <div className="mt-4 overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-[var(--color-border)]">
              <th className="px-2 py-1.5 text-left font-medium">Construction</th>
              <th className="px-2 py-1.5 text-right font-medium">Live marg</th>
              <th className="px-2 py-1.5 text-right font-medium">Live width</th>
              <th className="px-2 py-1.5 text-right font-medium">Live cond Δ</th>
              <th className="px-2 py-1.5 text-right font-medium">Live ms</th>
              <th className="px-2 py-1.5 text-right font-medium">Notebook marg</th>
              <th className="px-2 py-1.5 text-right font-medium">Notebook width</th>
              <th className="px-2 py-1.5 text-right font-medium">Notebook cond Δ</th>
            </tr>
          </thead>
          <tbody>
            {(['split', 'qr', 'cqr', 'hl'] as Construction[]).map((c) => {
              const live = tableRef[c];
              const ref = TABLE_REF[scenario][c];
              const meta = CONSTRUCTION_META[c];
              return (
                <tr
                  key={c}
                  className="border-b border-[var(--color-border)]"
                  style={{ opacity: enabled[c] ? 1 : 0.4 }}
                >
                  <td className="px-2 py-1.5 font-medium" style={{ color: meta.color }}>
                    {meta.label}
                  </td>
                  <td className="px-2 py-1.5 text-right font-mono tabular-nums">
                    <strong>{fmt(live.marg, 3)}</strong>
                  </td>
                  <td className="px-2 py-1.5 text-right font-mono tabular-nums">
                    <strong>{fmt(live.width, 3)}</strong>
                  </td>
                  <td className="px-2 py-1.5 text-right font-mono tabular-nums">
                    <strong>{fmt(live.cond, 3)}</strong>
                  </td>
                  <td className="px-2 py-1.5 text-right font-mono tabular-nums">
                    <strong>{live.ms.toFixed(1)}</strong>
                  </td>
                  <td className="px-2 py-1.5 text-right font-mono tabular-nums text-[var(--color-text-secondary)]">
                    {fmt(ref.marg, 3)}
                  </td>
                  <td className="px-2 py-1.5 text-right font-mono tabular-nums text-[var(--color-text-secondary)]">
                    {fmt(ref.width, 3)}
                  </td>
                  <td className="px-2 py-1.5 text-right font-mono tabular-nums text-[var(--color-text-secondary)]">
                    {fmt(ref.cond, 3)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <p className="mt-3 text-xs text-[var(--color-text-secondary)]">
        Cycle through scenarios A→B→C→D and watch the live readouts converge to the notebook column. On
        Scenario B the CQR row should narrow visibly relative to split conformal (Theorem 5.2 efficiency win).
        Across all four scenarios HL's marginal coverage stays stuck around 0.76–0.83 — the batch under-coverage
        flagged in §6.5. Drag n_cal up to 2000 to confirm HL doesn't recover (the gap is structural, not a
        finite-sample artifact).
      </p>
    </div>
  );
}
