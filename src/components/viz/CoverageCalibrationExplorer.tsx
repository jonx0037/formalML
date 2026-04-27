import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  conformalQuantile,
  fitPredictRidge,
  generateHeteroscedastic,
  mulberry32,
  palettePI,
  pureQrIntervalPI,
  splitConformalIntervalPI,
} from './shared/nonparametric-ml';

// =============================================================================
// CoverageCalibrationExplorer — §§1–3 widget.
//
// Lets the reader interpolate between homoscedastic (σ_max = 0) and strongly
// heteroscedastic (σ_max = 0.6) noise on the running example, vary the data
// budget n, and toggle between four band types:
//
//   - constant-width: |y - sin(x)| ≤ w with w tuned to give 90% marginal
//                     coverage on a held-out sample (the broken baseline)
//   - split-conformal: §2 — μ̂ = polynomial ridge, threshold = conformal quantile
//   - pure-QR:        §3 — two QR fits at τ = α/2 and τ = 1 - α/2, no calib
//   - oracle:         the best you could do — sin(x) ± z_{1-α/2}·σ(x)
//
// Display: scatter with band overlay + marginal coverage readout + 10-bin
// conditional-coverage strip chart with reference line at 1 - α = 0.9.
// =============================================================================

const ALPHA = 0.1;
const Z_HALF_ALPHA = 1.6448536269514722; // standard-normal 0.95-quantile
const SM_BREAKPOINT = 900;
const PANEL_HEIGHT = 320;
const STRIP_HEIGHT = 200;
const N_TEST = 2000;
const N_BINS = 10;
const X_MIN = -3;
const X_MAX = 3;

type BandType = 'constant' | 'split' | 'qr' | 'oracle';

const fmt = (x: number, digits = 3) => x.toFixed(digits);

function trueSigma(x: number, sigmaMax: number): number {
  return 0.2 + (sigmaMax * Math.abs(x)) / 3;
}

export default function CoverageCalibrationExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();

  const [sigmaMax, setSigmaMax] = useState(0.6);
  const [n, setN] = useState(500); // training fold size for split / pure-QR; calibration set scales with this
  const [bandType, setBandType] = useState<BandType>('split');
  const [seed, setSeed] = useState(7);

  const isStacked = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const panelWidth = isStacked ? containerWidth : containerWidth / 2;

  const result = useMemo(() => {
    const rng = mulberry32(seed);
    const tr = generateHeteroscedastic(n, rng, { slope: sigmaMax });
    const ca = generateHeteroscedastic(n, rng, { slope: sigmaMax });
    const te = generateHeteroscedastic(N_TEST, rng, { slope: sigmaMax });

    // x-grid for plotting band as a continuous curve
    const grid = new Float64Array(120);
    for (let i = 0; i < 120; i++) grid[i] = X_MIN + ((X_MAX - X_MIN) * i) / 119;

    let loTest: Float64Array;
    let hiTest: Float64Array;
    let loGrid = new Float64Array(grid.length);
    let hiGrid = new Float64Array(grid.length);

    if (bandType === 'constant') {
      // Broken baseline: |y - sin(x)| <= w, w tuned to 90% marginal on calibration sample.
      const calRes = new Float64Array(n);
      for (let i = 0; i < n; i++) calRes[i] = Math.abs(ca.Y[i] - Math.sin(ca.X[i]));
      const sorted = Array.from(calRes).sort((a, b) => a - b);
      const w = sorted[Math.floor((1 - ALPHA) * (n - 1))];
      loTest = new Float64Array(N_TEST);
      hiTest = new Float64Array(N_TEST);
      for (let i = 0; i < N_TEST; i++) {
        loTest[i] = Math.sin(te.X[i]) - w;
        hiTest[i] = Math.sin(te.X[i]) + w;
      }
      for (let i = 0; i < grid.length; i++) {
        loGrid[i] = Math.sin(grid[i]) - w;
        hiGrid[i] = Math.sin(grid[i]) + w;
      }
    } else if (bandType === 'split') {
      const r = splitConformalIntervalPI(tr.X, tr.Y, ca.X, ca.Y, te.X, ALPHA);
      loTest = r.lo;
      hiTest = r.hi;
      const muGrid = fitPredictRidge(Float64Array.from(tr.X), Float64Array.from(tr.Y), grid, 0.1);
      const calScores = r.calScores;
      const qHat = conformalQuantile(calScores, ALPHA);
      for (let i = 0; i < grid.length; i++) {
        loGrid[i] = muGrid[i] - qHat;
        hiGrid[i] = muGrid[i] + qHat;
      }
    } else if (bandType === 'qr') {
      // Pure QR uses combined train+cal as full training set
      const Xfull = new Float64Array(2 * n);
      const Yfull = new Float64Array(2 * n);
      for (let i = 0; i < n; i++) {
        Xfull[i] = tr.X[i];
        Yfull[i] = tr.Y[i];
        Xfull[n + i] = ca.X[i];
        Yfull[n + i] = ca.Y[i];
      }
      const r = pureQrIntervalPI(Xfull, Yfull, te.X, ALPHA);
      loTest = r.lo;
      hiTest = r.hi;
      const r2 = pureQrIntervalPI(Xfull, Yfull, grid, ALPHA);
      loGrid = r2.lo;
      hiGrid = r2.hi;
    } else {
      // oracle
      loTest = new Float64Array(N_TEST);
      hiTest = new Float64Array(N_TEST);
      for (let i = 0; i < N_TEST; i++) {
        const s = trueSigma(te.X[i], sigmaMax);
        loTest[i] = Math.sin(te.X[i]) - Z_HALF_ALPHA * s;
        hiTest[i] = Math.sin(te.X[i]) + Z_HALF_ALPHA * s;
      }
      for (let i = 0; i < grid.length; i++) {
        const s = trueSigma(grid[i], sigmaMax);
        loGrid[i] = Math.sin(grid[i]) - Z_HALF_ALPHA * s;
        hiGrid[i] = Math.sin(grid[i]) + Z_HALF_ALPHA * s;
      }
    }

    // Coverage stats
    let covered = 0;
    let widthSum = 0;
    const inBand = new Uint8Array(N_TEST);
    for (let i = 0; i < N_TEST; i++) {
      const ok = te.Y[i] >= loTest[i] && te.Y[i] <= hiTest[i];
      if (ok) {
        covered++;
        inBand[i] = 1;
      }
      widthSum += hiTest[i] - loTest[i];
    }
    const marginal = covered / N_TEST;
    const meanWidth = widthSum / N_TEST;

    // Conditional coverage by 10 equal-width bins
    const condCov = new Float64Array(N_BINS);
    const binCounts = new Int32Array(N_BINS);
    const binCovered = new Int32Array(N_BINS);
    const binWidth = (X_MAX - X_MIN) / N_BINS;
    for (let i = 0; i < N_TEST; i++) {
      let b = Math.floor((te.X[i] - X_MIN) / binWidth);
      if (b < 0) b = 0;
      if (b >= N_BINS) b = N_BINS - 1;
      binCounts[b]++;
      if (inBand[i]) binCovered[b]++;
    }
    for (let b = 0; b < N_BINS; b++) {
      condCov[b] = binCounts[b] > 0 ? binCovered[b] / binCounts[b] : NaN;
    }

    // Subsample for scatter (max 600 points to keep DOM manageable)
    const subN = Math.min(600, N_TEST);
    const stride = Math.floor(N_TEST / subN);
    const scatterX = new Float64Array(subN);
    const scatterY = new Float64Array(subN);
    const scatterIn = new Uint8Array(subN);
    for (let i = 0; i < subN; i++) {
      const k = i * stride;
      scatterX[i] = te.X[k];
      scatterY[i] = te.Y[k];
      scatterIn[i] = inBand[k];
    }

    return {
      grid,
      loGrid,
      hiGrid,
      scatterX,
      scatterY,
      scatterIn,
      marginal,
      meanWidth,
      condCov,
    };
  }, [sigmaMax, n, bandType, seed]);

  // ── Scatter + band SVG ─────────────────────────────────────────────
  const scatterRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const margin = { top: 12, right: 12, bottom: 36, left: 44 };
      const w = panelWidth;
      const h = PANEL_HEIGHT;
      const innerW = w - margin.left - margin.right;
      const innerH = h - margin.top - margin.bottom;
      if (innerW <= 0 || innerH <= 0) return;

      const xScale = d3.scaleLinear().domain([X_MIN, X_MAX]).range([0, innerW]);
      const allY = [...result.scatterY, ...result.loGrid, ...result.hiGrid];
      const yMin = Math.min(...allY);
      const yMax = Math.max(...allY);
      const yPad = (yMax - yMin) * 0.05;
      const yScale = d3.scaleLinear().domain([yMin - yPad, yMax + yPad]).range([innerH, 0]);

      const g = svg.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);

      // Axes
      g.append('g')
        .attr('transform', `translate(0, ${innerH})`)
        .call(d3.axisBottom(xScale).ticks(5).tickSize(-innerH))
        .call((sel) => sel.selectAll('line').style('stroke', 'var(--color-border)').style('opacity', 0.3))
        .call((sel) => sel.selectAll('text').style('fill', 'var(--color-text-secondary)').style('font-size', '10px'))
        .call((sel) => sel.select('.domain').style('stroke', 'var(--color-border)'));
      g.append('g')
        .call(d3.axisLeft(yScale).ticks(5).tickSize(-innerW))
        .call((sel) => sel.selectAll('line').style('stroke', 'var(--color-border)').style('opacity', 0.3))
        .call((sel) => sel.selectAll('text').style('fill', 'var(--color-text-secondary)').style('font-size', '10px'))
        .call((sel) => sel.select('.domain').style('stroke', 'var(--color-border)'));

      // Axis labels
      g.append('text')
        .attr('x', innerW / 2)
        .attr('y', innerH + 28)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '11px')
        .text('x');
      g.append('text')
        .attr('transform', `translate(-32, ${innerH / 2}) rotate(-90)`)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '11px')
        .text('y');

      // Band area
      const bandColor =
        bandType === 'constant' ? palettePI.lightAmber :
        bandType === 'split' ? palettePI.lightBlue :
        bandType === 'qr' ? palettePI.lightGreen :
        palettePI.lightPurple;
      const bandStroke =
        bandType === 'constant' ? palettePI.amber :
        bandType === 'split' ? palettePI.blue :
        bandType === 'qr' ? palettePI.green :
        palettePI.purple;

      const area = d3.area<number>()
        .x((_d, i) => xScale(result.grid[i]))
        .y0((_d, i) => yScale(result.loGrid[i]))
        .y1((_d, i) => yScale(result.hiGrid[i]))
        .curve(d3.curveCatmullRom);
      g.append('path')
        .datum(Array.from(result.grid))
        .attr('d', area)
        .style('fill', bandColor)
        .style('opacity', 0.6);
      g.append('path')
        .datum(Array.from(result.grid).map((_, i) => result.loGrid[i]))
        .attr('d', d3.line<number>().x((_d, i) => xScale(result.grid[i])).y((d) => yScale(d)).curve(d3.curveCatmullRom))
        .style('fill', 'none')
        .style('stroke', bandStroke)
        .style('stroke-width', 1.4);
      g.append('path')
        .datum(Array.from(result.grid).map((_, i) => result.hiGrid[i]))
        .attr('d', d3.line<number>().x((_d, i) => xScale(result.grid[i])).y((d) => yScale(d)).curve(d3.curveCatmullRom))
        .style('fill', 'none')
        .style('stroke', bandStroke)
        .style('stroke-width', 1.4);

      // True mean curve sin(x)
      const sinPoints = d3.range(120).map((i) => {
        const x = X_MIN + ((X_MAX - X_MIN) * i) / 119;
        return [x, Math.sin(x)] as [number, number];
      });
      g.append('path')
        .datum(sinPoints)
        .attr(
          'd',
          d3.line<[number, number]>().x((d) => xScale(d[0])).y((d) => yScale(d[1])),
        )
        .style('fill', 'none')
        .style('stroke', 'var(--color-text-secondary)')
        .style('stroke-width', 1.2)
        .style('stroke-dasharray', '4 3');

      // Scatter
      const sub = result.scatterX.length;
      for (let i = 0; i < sub; i++) {
        g.append('circle')
          .attr('cx', xScale(result.scatterX[i]))
          .attr('cy', yScale(result.scatterY[i]))
          .attr('r', 1.8)
          .style('fill', result.scatterIn[i] ? palettePI.blue : palettePI.red)
          .style('opacity', result.scatterIn[i] ? 0.45 : 0.85);
      }
    },
    [result, panelWidth, bandType],
  );

  // ── Conditional coverage strip chart SVG ───────────────────────────
  const stripRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const margin = { top: 12, right: 12, bottom: 36, left: 44 };
      const w = panelWidth;
      const h = STRIP_HEIGHT;
      const innerW = w - margin.left - margin.right;
      const innerH = h - margin.top - margin.bottom;
      if (innerW <= 0 || innerH <= 0) return;

      const binWidth = (X_MAX - X_MIN) / N_BINS;
      const binCenters = d3.range(N_BINS).map((b) => X_MIN + binWidth * (b + 0.5));

      const xScale = d3.scaleBand<number>()
        .domain(binCenters)
        .range([0, innerW])
        .padding(0.15);
      const yScale = d3.scaleLinear().domain([0.5, 1.02]).range([innerH, 0]);

      const g = svg.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);

      g.append('g')
        .attr('transform', `translate(0, ${innerH})`)
        .call(d3.axisBottom(xScale).tickFormat((d) => fmt(d as number, 1)))
        .call((sel) => sel.selectAll('text').style('fill', 'var(--color-text-secondary)').style('font-size', '10px'))
        .call((sel) => sel.select('.domain').style('stroke', 'var(--color-border)'))
        .call((sel) => sel.selectAll('line').style('stroke', 'var(--color-border)'));
      g.append('g')
        .call(d3.axisLeft(yScale).ticks(6).tickSize(-innerW))
        .call((sel) => sel.selectAll('line').style('stroke', 'var(--color-border)').style('opacity', 0.3))
        .call((sel) => sel.selectAll('text').style('fill', 'var(--color-text-secondary)').style('font-size', '10px'))
        .call((sel) => sel.select('.domain').style('stroke', 'var(--color-border)'));

      g.append('text')
        .attr('x', innerW / 2)
        .attr('y', innerH + 28)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '11px')
        .text('x bin');
      g.append('text')
        .attr('transform', `translate(-32, ${innerH / 2}) rotate(-90)`)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '11px')
        .text('cond. coverage');

      // Bars
      for (let b = 0; b < N_BINS; b++) {
        const xc = binCenters[b];
        const bx = xScale(xc) ?? 0;
        const bw = xScale.bandwidth();
        const cov = result.condCov[b];
        if (Number.isNaN(cov)) continue;
        g.append('rect')
          .attr('x', bx)
          .attr('y', yScale(cov))
          .attr('width', bw)
          .attr('height', innerH - yScale(cov))
          .style('fill', cov >= 1 - ALPHA ? palettePI.lightBlue : palettePI.lightRed)
          .style('stroke', cov >= 1 - ALPHA ? palettePI.blue : palettePI.red)
          .style('stroke-width', 0.7);
      }

      // Reference line at 1 - α = 0.9
      g.append('line')
        .attr('x1', 0)
        .attr('x2', innerW)
        .attr('y1', yScale(1 - ALPHA))
        .attr('y2', yScale(1 - ALPHA))
        .style('stroke', palettePI.green)
        .style('stroke-width', 1.5)
        .style('stroke-dasharray', '4 3');
      g.append('text')
        .attr('x', innerW - 4)
        .attr('y', yScale(1 - ALPHA) - 4)
        .attr('text-anchor', 'end')
        .style('fill', palettePI.green)
        .style('font-size', '10px')
        .text(`1 − α = ${fmt(1 - ALPHA, 2)}`);

      // Marginal line
      g.append('line')
        .attr('x1', 0)
        .attr('x2', innerW)
        .attr('y1', yScale(result.marginal))
        .attr('y2', yScale(result.marginal))
        .style('stroke', palettePI.purple)
        .style('stroke-width', 1)
        .style('stroke-dasharray', '2 2');
      g.append('text')
        .attr('x', 4)
        .attr('y', yScale(result.marginal) - 4)
        .style('fill', palettePI.purple)
        .style('font-size', '10px')
        .text(`marginal = ${fmt(result.marginal, 3)}`);
    },
    [result, panelWidth],
  );

  const condRange = useMemo(() => {
    let mn = Infinity;
    let mx = -Infinity;
    for (let i = 0; i < result.condCov.length; i++) {
      const v = result.condCov[i];
      if (Number.isNaN(v)) continue;
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
    return { min: mn, max: mx, range: mx - mn };
  }, [result]);

  return (
    <div
      ref={containerRef}
      className="my-6 rounded-lg border border-[var(--color-border)] bg-[var(--color-muted-bg)] p-3"
    >
      <div className={`flex ${isStacked ? 'flex-col' : 'flex-row'} gap-2`}>
        <svg
          ref={scatterRef}
          width={panelWidth}
          height={PANEL_HEIGHT}
          role="img"
          aria-label="Scatter with band overlay"
        />
        <svg
          ref={stripRef}
          width={panelWidth}
          height={STRIP_HEIGHT}
          role="img"
          aria-label="Conditional coverage by bin"
        />
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-x-6 gap-y-3 text-sm">
        <label className="flex items-center gap-2 text-[var(--color-text-secondary)]">
          <span className="font-mono text-xs">σ_max</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={sigmaMax}
            onChange={(e) => setSigmaMax(Number(e.target.value))}
            className="accent-[var(--color-accent)]"
          />
          <span className="font-mono w-12 text-right">{fmt(sigmaMax, 2)}</span>
        </label>
        <label className="flex items-center gap-2 text-[var(--color-text-secondary)]">
          <span className="font-mono text-xs">n</span>
          <input
            type="range"
            min={50}
            max={1000}
            step={50}
            value={n}
            onChange={(e) => setN(Number(e.target.value))}
            className="accent-[var(--color-accent)]"
          />
          <span className="font-mono w-12 text-right">{n}</span>
        </label>
        <label className="flex items-center gap-2 text-[var(--color-text-secondary)]">
          <span className="font-mono text-xs">band</span>
          <select
            value={bandType}
            onChange={(e) => setBandType(e.target.value as BandType)}
            className="rounded border border-[var(--color-border)] bg-[var(--color-surface)] px-2 py-1 text-xs"
          >
            <option value="constant">constant-width (broken)</option>
            <option value="split">split conformal (§2)</option>
            <option value="qr">pure QR (§3)</option>
            <option value="oracle">oracle</option>
          </select>
        </label>
        <button
          type="button"
          onClick={() => setSeed((s) => s + 1)}
          className="rounded border border-[var(--color-accent)] bg-[var(--color-accent)] px-3 py-1 text-xs font-medium text-white transition hover:opacity-90"
        >
          re-randomize
        </button>
        <span className="text-xs text-[var(--color-text-secondary)]">
          marginal <strong>{fmt(result.marginal, 3)}</strong>
          {' · '}mean width <strong>{fmt(result.meanWidth, 3)}</strong>
          {' · '}cond range Δ = <strong>{fmt(condRange.range, 3)}</strong>
        </span>
      </div>

      <p className="mt-3 text-xs text-[var(--color-text-secondary)]">
        Drag σ_max from 0 (homoscedastic) to 1 (strongly heteroscedastic) with the constant-width band selected:
        the strip chart deforms from flat to U-shaped while the marginal stays near 0.9 — the gap Definition 2
        anticipates. Switch to pure-QR or oracle to flatten the strip chart at heteroscedasticity. Switch to
        split-conformal to see the same constant-width pathology under the proper conformal threshold.
      </p>
    </div>
  );
}
