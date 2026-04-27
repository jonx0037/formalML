import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  cqrIntervalPI,
  fitPredictRidge,
  gaussianSampler,
  hlIntervalPI,
  mulberry32,
  palettePI,
  pureQrIntervalPI,
  splitConformalIntervalPI,
} from './shared/nonparametric-ml';

// =============================================================================
// AssumptionFailureExplorer — §4 widget.
//
// Lets the reader walk through the three failure modes for HL test-inversion's
// Definition 9: residual asymmetry, residual-feature heteroscedasticity, and
// non-iid (the last one isn't selectable here — flagged in §7).
//
// Controls:
//   - residual distribution: gaussian, laplace, t_3, t_1, centered chi-squared,
//     centered lognormal
//   - heteroscedasticity slider σ_max ∈ [0, 1]
//   - n_cal slider ∈ [50, 1000]
//   - band-type toggle (split conformal / pure QR / HL / CQR)
//
// Display: scatter + band, marginal coverage, 8-bin conditional coverage strip,
// "Theorem 3 conditions met?" indicator.
// =============================================================================

type ResidDist = 'gaussian' | 'laplace' | 't3' | 't1' | 'chi3' | 'lognormal';
type Band = 'split' | 'qr' | 'hl' | 'cqr';

const ALPHA = 0.1;
const SM_BREAKPOINT = 900;
const PANEL_HEIGHT = 320;
const STRIP_HEIGHT = 200;
const N_TRAIN = 500;
const N_TEST = 2000;
const N_BINS = 8;
const X_MIN = -2;
const X_MAX = 2;

const fmt = (x: number, digits = 3) => x.toFixed(digits);

const SYMMETRIC: ResidDist[] = ['gaussian', 'laplace', 't3', 't1'];

// Residual sampler — returns iid samples from the chosen centered distribution.
// Each draw has unit-ish scale; we then multiply by σ(x) below for heteroscedasticity.
function residSampler(dist: ResidDist, rng: () => number): () => number {
  const gauss = gaussianSampler(rng);
  switch (dist) {
    case 'gaussian':
      return () => gauss();
    case 'laplace':
      // Laplace via difference of two exponentials. scale = 1/sqrt(2) for unit variance.
      return () => {
        const u = rng() - 0.5;
        const sign = u >= 0 ? 1 : -1;
        return -sign * Math.log(1 - 2 * Math.abs(u)) / Math.SQRT2;
      };
    case 't3':
      return () => {
        const z = gauss();
        const w = gauss() ** 2 + gauss() ** 2 + gauss() ** 2;
        return z / Math.sqrt(w / 3);
      };
    case 't1':
      // Cauchy = ratio of two standard normals
      return () => {
        const a = gauss();
        const b = gauss();
        return Math.abs(b) < 1e-12 ? 0 : a / b;
      };
    case 'chi3': {
      // Centered chi-squared df=3, scaled so std = 1: (X - 3) / sqrt(6)
      return () => {
        const c = gauss() ** 2 + gauss() ** 2 + gauss() ** 2;
        return (c - 3) / Math.sqrt(6);
      };
    }
    case 'lognormal':
      // Centered lognormal: X = exp(N(0, 1)); E[X] = e^0.5, Var[X] = (e-1)e
      // Center and rescale to mean 0, std 1.
      return () => {
        const x = Math.exp(gauss());
        return (x - Math.E ** 0.5) / Math.sqrt((Math.E - 1) * Math.E);
      };
  }
}

function generate(
  n: number,
  rng: () => number,
  dist: ResidDist,
  sigmaMax: number,
  scale: number = 0.6,
): { X: Float64Array; Y: Float64Array } {
  const sampler = residSampler(dist, rng);
  const X = new Float64Array(n);
  const Y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const x = X_MIN + (X_MAX - X_MIN) * rng();
    // σ(x) = 1 + σ_max * |x| / 2  (multiplicative heteroscedasticity factor)
    const sig = 1 + (sigmaMax * Math.abs(x)) / 2;
    X[i] = x;
    Y[i] = 0.4 * Math.cos(Math.PI * x) + scale * sig * sampler();
  }
  return { X, Y };
}

export default function AssumptionFailureExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();

  const [dist, setDist] = useState<ResidDist>('t3');
  const [sigmaMax, setSigmaMax] = useState(0);
  const [nCal, setNCal] = useState(500);
  const [band, setBand] = useState<Band>('hl');
  const [seed, setSeed] = useState(13);

  const isStacked = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const panelWidth = isStacked ? containerWidth : containerWidth / 2;

  const conditionsMet = SYMMETRIC.includes(dist) && sigmaMax === 0;

  const result = useMemo(() => {
    const rng = mulberry32(seed);
    const tr = generate(N_TRAIN, rng, dist, sigmaMax);
    const ca = generate(nCal, rng, dist, sigmaMax);
    const te = generate(N_TEST, rng, dist, sigmaMax);

    // x-grid for plotting
    const grid = new Float64Array(120);
    for (let i = 0; i < 120; i++) grid[i] = X_MIN + ((X_MAX - X_MIN) * i) / 119;

    let lo: Float64Array;
    let hi: Float64Array;
    let loGrid = new Float64Array(grid.length);
    let hiGrid = new Float64Array(grid.length);

    if (band === 'split') {
      const r = splitConformalIntervalPI(tr.X, tr.Y, ca.X, ca.Y, te.X, ALPHA);
      lo = r.lo;
      hi = r.hi;
      const muG = fitPredictRidge(Float64Array.from(tr.X), Float64Array.from(tr.Y), grid, 0.1);
      for (let i = 0; i < grid.length; i++) {
        loGrid[i] = muG[i] - r.qHat;
        hiGrid[i] = muG[i] + r.qHat;
      }
    } else if (band === 'qr') {
      const Xfull = new Float64Array(N_TRAIN + nCal);
      const Yfull = new Float64Array(N_TRAIN + nCal);
      for (let i = 0; i < N_TRAIN; i++) {
        Xfull[i] = tr.X[i];
        Yfull[i] = tr.Y[i];
      }
      for (let i = 0; i < nCal; i++) {
        Xfull[N_TRAIN + i] = ca.X[i];
        Yfull[N_TRAIN + i] = ca.Y[i];
      }
      const r = pureQrIntervalPI(Xfull, Yfull, te.X, ALPHA);
      lo = r.lo;
      hi = r.hi;
      const r2 = pureQrIntervalPI(Xfull, Yfull, grid, ALPHA);
      loGrid = r2.lo;
      hiGrid = r2.hi;
    } else if (band === 'cqr') {
      const r = cqrIntervalPI(tr.X, tr.Y, ca.X, ca.Y, te.X, ALPHA);
      lo = r.lo;
      hi = r.hi;
      const r2 = cqrIntervalPI(tr.X, tr.Y, ca.X, ca.Y, grid, ALPHA);
      loGrid = r2.lo;
      hiGrid = r2.hi;
    } else {
      const r = hlIntervalPI(tr.X, tr.Y, ca.X, ca.Y, te.X, ALPHA);
      lo = r.lo;
      hi = r.hi;
      const muG = fitPredictRidge(Float64Array.from(tr.X), Float64Array.from(tr.Y), grid, 0.1);
      for (let i = 0; i < grid.length; i++) {
        loGrid[i] = muG[i] + r.ALo;
        hiGrid[i] = muG[i] + r.AHi;
      }
    }

    let cov = 0;
    let widthSum = 0;
    const inBand = new Uint8Array(N_TEST);
    for (let i = 0; i < N_TEST; i++) {
      const ok = te.Y[i] >= lo[i] && te.Y[i] <= hi[i];
      if (ok) {
        cov++;
        inBand[i] = 1;
      }
      widthSum += hi[i] - lo[i];
    }
    const marginal = cov / N_TEST;
    const meanWidth = widthSum / N_TEST;

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

    const subN = Math.min(500, N_TEST);
    const stride = Math.floor(N_TEST / subN);
    const sX = new Float64Array(subN);
    const sY = new Float64Array(subN);
    const sIn = new Uint8Array(subN);
    for (let i = 0; i < subN; i++) {
      const k = i * stride;
      sX[i] = te.X[k];
      sY[i] = te.Y[k];
      sIn[i] = inBand[k];
    }

    return {
      grid,
      loGrid,
      hiGrid,
      sX,
      sY,
      sIn,
      marginal,
      meanWidth,
      condCov,
    };
  }, [dist, sigmaMax, nCal, band, seed]);

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

  const bandColor =
    band === 'split' ? palettePI.blue :
    band === 'qr' ? palettePI.green :
    band === 'cqr' ? palettePI.teal :
    palettePI.purple;
  const lightBandColor =
    band === 'split' ? palettePI.lightBlue :
    band === 'qr' ? palettePI.lightGreen :
    band === 'cqr' ? '#CCFBF1' :
    palettePI.lightPurple;

  const scatterRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const margin = { top: 12, right: 12, bottom: 36, left: 44 };
      const innerW = panelWidth - margin.left - margin.right;
      const innerH = PANEL_HEIGHT - margin.top - margin.bottom;
      if (innerW <= 0 || innerH <= 0) return;

      const xScale = d3.scaleLinear().domain([X_MIN, X_MAX]).range([0, innerW]);
      const yVals = [...result.sY, ...result.loGrid, ...result.hiGrid];
      const yMinRaw = Math.min(...yVals);
      const yMaxRaw = Math.max(...yVals);
      // Cap extreme outliers (Cauchy / lognormal explode); use 0.5/99.5 percentile range for axis.
      const ySorted = result.sY.slice().sort();
      const yLo = Math.min(yMinRaw, ySorted[Math.floor(ySorted.length * 0.005)] - 0.3);
      const yHi = Math.max(yMaxRaw, ySorted[Math.floor(ySorted.length * 0.995)] + 0.3);
      const yScale = d3.scaleLinear().domain([Math.max(yLo, -10), Math.min(yHi, 10)]).range([innerH, 0]);

      const g = svg.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);

      g.append('g')
        .attr('transform', `translate(0, ${innerH})`)
        .call(d3.axisBottom(xScale).ticks(5).tickSize(-innerH))
        .call((sel) => sel.selectAll('line').style('stroke', 'var(--color-border)').style('opacity', 0.3))
        .call((sel) => sel.selectAll('text').style('fill', 'var(--color-text-secondary)').style('font-size', '10px'))
        .call((sel) => sel.select('.domain').style('stroke', 'var(--color-border)'));
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
        .text('x');
      g.append('text')
        .attr('transform', `translate(-32, ${innerH / 2}) rotate(-90)`)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '11px')
        .text('y');

      // Band area
      const area = d3.area<number>()
        .x((_d, i) => xScale(result.grid[i]))
        .y0((_d, i) => yScale(result.loGrid[i]))
        .y1((_d, i) => yScale(result.hiGrid[i]))
        .curve(d3.curveCatmullRom);
      g.append('path')
        .datum(Array.from(result.grid))
        .attr('d', area)
        .style('fill', lightBandColor)
        .style('opacity', 0.55);

      // True μ(x) = 0.4 cos(πx)
      const muPoints = d3.range(120).map((i) => {
        const x = X_MIN + ((X_MAX - X_MIN) * i) / 119;
        return [x, 0.4 * Math.cos(Math.PI * x)] as [number, number];
      });
      g.append('path')
        .datum(muPoints)
        .attr('d', d3.line<[number, number]>().x((d) => xScale(d[0])).y((d) => yScale(d[1])))
        .style('fill', 'none')
        .style('stroke', 'var(--color-text-secondary)')
        .style('stroke-width', 1.2)
        .style('stroke-dasharray', '4 3');

      // Band edges
      g.append('path')
        .datum(Array.from(result.grid).map((_, i) => result.loGrid[i]))
        .attr('d', d3.line<number>().x((_d, i) => xScale(result.grid[i])).y((d) => yScale(d)).curve(d3.curveCatmullRom))
        .style('fill', 'none')
        .style('stroke', bandColor)
        .style('stroke-width', 1.4);
      g.append('path')
        .datum(Array.from(result.grid).map((_, i) => result.hiGrid[i]))
        .attr('d', d3.line<number>().x((_d, i) => xScale(result.grid[i])).y((d) => yScale(d)).curve(d3.curveCatmullRom))
        .style('fill', 'none')
        .style('stroke', bandColor)
        .style('stroke-width', 1.4);

      // Scatter
      for (let i = 0; i < result.sX.length; i++) {
        const yV = result.sY[i];
        if (!Number.isFinite(yV)) continue;
        const cy = yScale(yV);
        if (cy < 0 || cy > innerH) continue; // clip to plot area
        g.append('circle')
          .attr('cx', xScale(result.sX[i]))
          .attr('cy', cy)
          .attr('r', 1.8)
          .style('fill', result.sIn[i] ? palettePI.blue : palettePI.red)
          .style('opacity', result.sIn[i] ? 0.45 : 0.85);
      }
    },
    [result, panelWidth, band, lightBandColor, bandColor],
  );

  const stripRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const margin = { top: 12, right: 12, bottom: 36, left: 44 };
      const innerW = panelWidth - margin.left - margin.right;
      const innerH = STRIP_HEIGHT - margin.top - margin.bottom;
      if (innerW <= 0 || innerH <= 0) return;

      const binWidth = (X_MAX - X_MIN) / N_BINS;
      const binCenters = d3.range(N_BINS).map((b) => X_MIN + binWidth * (b + 0.5));
      const xScale = d3.scaleBand<number>().domain(binCenters).range([0, innerW]).padding(0.15);
      const yScale = d3.scaleLinear().domain([0.4, 1.02]).range([innerH, 0]);

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
          .style('fill', cov >= 1 - ALPHA ? lightBandColor : palettePI.lightRed)
          .style('stroke', cov >= 1 - ALPHA ? bandColor : palettePI.red)
          .style('stroke-width', 0.7);
      }

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
    [result, panelWidth, bandColor, lightBandColor],
  );

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
          aria-label="Assumption-failure scatter with band overlay"
        />
        <svg
          ref={stripRef}
          width={panelWidth}
          height={STRIP_HEIGHT}
          role="img"
          aria-label="Assumption-failure conditional coverage strip"
        />
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-x-6 gap-y-3 text-sm">
        <label className="flex items-center gap-2 text-[var(--color-text-secondary)]">
          <span className="font-mono text-xs">residual</span>
          <select
            value={dist}
            onChange={(e) => setDist(e.target.value as ResidDist)}
            className="rounded border border-[var(--color-border)] bg-[var(--color-surface)] px-2 py-1 text-xs"
          >
            <option value="gaussian">Gaussian (sym ✓)</option>
            <option value="laplace">Laplace (sym ✓)</option>
            <option value="t3">t₃ (sym ✓)</option>
            <option value="t1">t₁ Cauchy (sym ✓)</option>
            <option value="chi3">centered χ²₃ (skewed)</option>
            <option value="lognormal">centered lognormal (skewed)</option>
          </select>
        </label>
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
          <span className="font-mono text-xs">n_cal</span>
          <input
            type="range"
            min={50}
            max={1000}
            step={50}
            value={nCal}
            onChange={(e) => setNCal(Number(e.target.value))}
            className="accent-[var(--color-accent)]"
          />
          <span className="font-mono w-12 text-right">{nCal}</span>
        </label>
        <label className="flex items-center gap-2 text-[var(--color-text-secondary)]">
          <span className="font-mono text-xs">band</span>
          <select
            value={band}
            onChange={(e) => setBand(e.target.value as Band)}
            className="rounded border border-[var(--color-border)] bg-[var(--color-surface)] px-2 py-1 text-xs"
          >
            <option value="split">split conformal</option>
            <option value="qr">pure QR</option>
            <option value="cqr">CQR</option>
            <option value="hl">HL test-inversion</option>
          </select>
        </label>
        <button
          type="button"
          onClick={() => setSeed((s) => s + 1)}
          className="rounded border border-[var(--color-accent)] bg-[var(--color-accent)] px-3 py-1 text-xs font-medium text-white transition hover:opacity-90"
        >
          re-randomize
        </button>
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-3 text-xs text-[var(--color-text-secondary)]">
        <span
          className="inline-flex items-center gap-1.5 rounded-md px-2 py-1 font-medium"
          style={{
            background: conditionsMet ? '#D1FAE5' : '#FEE2E2',
            color: conditionsMet ? '#065F46' : '#991B1B',
            border: `1px solid ${conditionsMet ? '#059669' : '#DC2626'}`,
          }}
        >
          {conditionsMet ? '✓' : '✗'} Theorem 3 conditions{' '}
          {conditionsMet ? 'met' : 'broken'}
        </span>
        <span>
          marginal <strong>{fmt(result.marginal, 3)}</strong>
          {' · '}mean width <strong>{fmt(result.meanWidth, 3)}</strong>
          {' · '}cond range Δ = <strong>{fmt(condRange.range, 3)}</strong>
        </span>
      </div>

      <p className="mt-3 text-xs text-[var(--color-text-secondary)]">
        Try walking through the failure modes one at a time. With <code>HL</code> selected:
        <em> symmetric residual + σ_max = 0</em> = green check (Theorem 3 holds), conditional
        coverage flat near 0.9. Switch to centered χ² or lognormal: the indicator flips red and
        marginal coverage drops well below 0.9. Or keep the Gaussian residual but push σ_max up:
        the strip chart re-acquires the U-shape (constant-width can't track heteroscedasticity).
        Switch to CQR to recover conditional adaptivity in the heteroscedastic case.
      </p>
    </div>
  );
}
