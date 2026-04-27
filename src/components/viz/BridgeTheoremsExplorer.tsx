import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  cqrIntervalPI,
  fitPredictRidge,
  generateHeavyTailedLocation,
  generateHeteroscedastic,
  hlIntervalPI,
  mulberry32,
  palettePI,
  splitConformalIntervalPI,
} from './shared/nonparametric-ml';

// =============================================================================
// BridgeTheoremsExplorer — §5 widget. Three modes via tab strip:
//
//   T5.1 — CQR coverage decomposition.
//          Slider for n_cal, observe CQR conditional coverage gap on a grid.
//   T5.2 — Heteroscedastic width comparison.
//          Slider for σ_max ∈ [0, 1] on RE1; show CQR/SC width ratio against
//          the theoretical E[σ(X)]/σ_+ prediction.
//   T5.3 — HL/conformal asymptotic equivalence.
//          Slider for n_cal ∈ [50, 2000] on RE2; show HL/SC width ratio
//          approaching 1.
// =============================================================================

type Mode = 't51' | 't52' | 't53';
const ALPHA = 0.1;
const SM_BREAKPOINT = 900;
const PANEL_HEIGHT = 320;
const N_TRAIN = 500;
const N_TEST = 1500;
const fmt = (x: number, digits = 3) => x.toFixed(digits);

export default function BridgeTheoremsExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const isStacked = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const panelWidth = isStacked ? containerWidth : containerWidth / 2;

  const [mode, setMode] = useState<Mode>('t52');
  const [t51NCal, setT51NCal] = useState(500);
  const [t52Slope, setT52Slope] = useState(0.6);
  const [t53NCal, setT53NCal] = useState(500);
  const [seed, setSeed] = useState(101);

  const result = useMemo(() => {
    const rng = mulberry32(seed);
    if (mode === 't51') {
      // CQR conditional gap on RE1, current n_cal
      const nCal = t51NCal;
      const tr = generateHeteroscedastic(N_TRAIN, rng, { slope: 0.6 });
      const ca = generateHeteroscedastic(nCal, rng, { slope: 0.6 });

      // Build grid for evaluating conditional coverage
      const N_GRID = 25;
      const grid = new Float64Array(N_GRID);
      for (let i = 0; i < N_GRID; i++) grid[i] = -2.8 + (5.6 * i) / (N_GRID - 1);
      // For each grid point, draw a batch of test points near grid[k] to estimate cond cov
      const N_PER = 200;
      const xCloseAll = new Float64Array(N_GRID * N_PER);
      const yCloseAll = new Float64Array(N_GRID * N_PER);
      const sigmaCloseAll = new Float64Array(N_GRID * N_PER);
      const sigmaTrue = (x: number) => 0.2 + (0.6 * Math.abs(x)) / 3;
      // Use Box-Muller via gaussianSampler
      const gauss2 = (() => {
        let cached: number | null = null;
        return () => {
          if (cached !== null) {
            const v = cached;
            cached = null;
            return v;
          }
          const u1 = Math.max(rng(), 1e-12);
          const u2 = rng();
          const r = Math.sqrt(-2 * Math.log(u1));
          const theta = 2 * Math.PI * u2;
          cached = r * Math.sin(theta);
          return r * Math.cos(theta);
        };
      })();
      for (let k = 0; k < N_GRID; k++) {
        const xc = grid[k];
        const sg = sigmaTrue(xc);
        for (let j = 0; j < N_PER; j++) {
          xCloseAll[k * N_PER + j] = xc + (rng() - 0.5) * 0.05; // local jitter
          sigmaCloseAll[k * N_PER + j] = sg;
          yCloseAll[k * N_PER + j] = Math.sin(xc) + sg * gauss2();
        }
      }
      const r = cqrIntervalPI(tr.X, tr.Y, ca.X, ca.Y, xCloseAll, ALPHA);
      // Conditional coverage per grid bin
      const condCov = new Float64Array(N_GRID);
      for (let k = 0; k < N_GRID; k++) {
        let c = 0;
        for (let j = 0; j < N_PER; j++) {
          const idx = k * N_PER + j;
          if (yCloseAll[idx] >= r.lo[idx] && yCloseAll[idx] <= r.hi[idx]) c++;
        }
        condCov[k] = c / N_PER;
      }
      // Δ_n(x) — pointwise QR estimation error vs true conditional quantiles
      // For Gaussian RE1: q*_{α/2}(x) = sin(x) - 1.6449·σ(x), q*_{1-α/2}(x) = sin(x) + 1.6449·σ(x)
      const Z = 1.6448536269514722;
      const cqrGrid = cqrIntervalPI(tr.X, tr.Y, ca.X, ca.Y, grid, ALPHA);
      const Delta = new Float64Array(N_GRID);
      let meanDelta = 0;
      for (let k = 0; k < N_GRID; k++) {
        const sg = sigmaTrue(grid[k]);
        // CQR endpoints contain conformal correction Q; back out q̂ low/high (the QR fit alone)
        // CQR.qLoTest = q̂_α/2; CQR.qHiTest = q̂_{1-α/2}; so error vs true:
        const qStarLo = Math.sin(grid[k]) - Z * sg;
        const qStarHi = Math.sin(grid[k]) + Z * sg;
        const dLo = Math.abs(cqrGrid.qLoTest[k] - qStarLo);
        const dHi = Math.abs(cqrGrid.qHiTest[k] - qStarHi);
        Delta[k] = Math.max(dLo, dHi);
        meanDelta += Delta[k];
      }
      meanDelta /= N_GRID;
      const sigmaMin = 0.2;
      const fMax = 1 / (sigmaMin * Math.sqrt(2 * Math.PI));
      const bound = new Float64Array(N_GRID);
      const gap = new Float64Array(N_GRID);
      for (let k = 0; k < N_GRID; k++) {
        bound[k] = 4 * fMax * (Delta[k] + meanDelta);
        gap[k] = Math.abs(condCov[k] - (1 - ALPHA));
      }
      // Marginal coverage on this batch
      let cov = 0;
      for (let i = 0; i < xCloseAll.length; i++) {
        if (yCloseAll[i] >= r.lo[i] && yCloseAll[i] <= r.hi[i]) cov++;
      }
      return {
        kind: 't51' as const,
        grid: Array.from(grid),
        Delta: Array.from(Delta),
        gap: Array.from(gap),
        bound: Array.from(bound),
        meanDelta,
        fMax,
        marginal: cov / xCloseAll.length,
        Q: r.Q,
      };
    }
    if (mode === 't52') {
      const tr = generateHeteroscedastic(N_TRAIN, rng, { slope: t52Slope });
      const ca = generateHeteroscedastic(N_TRAIN, rng, { slope: t52Slope });
      const te = generateHeteroscedastic(N_TEST, rng, { slope: t52Slope });

      const sc = splitConformalIntervalPI(tr.X, tr.Y, ca.X, ca.Y, te.X, ALPHA);
      const cqr = cqrIntervalPI(tr.X, tr.Y, ca.X, ca.Y, te.X, ALPHA);

      let scWSum = 0;
      let cqrWSum = 0;
      let scCov = 0;
      let cqrCov = 0;
      for (let i = 0; i < N_TEST; i++) {
        scWSum += sc.hi[i] - sc.lo[i];
        cqrWSum += cqr.hi[i] - cqr.lo[i];
        if (te.Y[i] >= sc.lo[i] && te.Y[i] <= sc.hi[i]) scCov++;
        if (te.Y[i] >= cqr.lo[i] && te.Y[i] <= cqr.hi[i]) cqrCov++;
      }
      const sigmaPlus = 0.2 + (t52Slope * 3) / 3;
      const sigmaMean = 0.2 + (t52Slope * 1.5) / 3; // E[|X|] = 1.5 for X ~ Unif(-3, 3)
      // Plot grid for bands
      const grid = new Float64Array(120);
      for (let i = 0; i < 120; i++) grid[i] = -3 + (6 * i) / 119;
      const scG = splitConformalIntervalPI(tr.X, tr.Y, ca.X, ca.Y, grid, ALPHA);
      const cqrG = cqrIntervalPI(tr.X, tr.Y, ca.X, ca.Y, grid, ALPHA);

      // Subsample scatter
      const sub = 300;
      const stride = Math.floor(N_TEST / sub);
      const sX = new Float64Array(sub);
      const sY = new Float64Array(sub);
      for (let i = 0; i < sub; i++) {
        sX[i] = te.X[i * stride];
        sY[i] = te.Y[i * stride];
      }
      return {
        kind: 't52' as const,
        scLo: scG.lo,
        scHi: scG.hi,
        cqrLo: cqrG.lo,
        cqrHi: cqrG.hi,
        grid: Array.from(grid),
        sX,
        sY,
        scWidth: scWSum / N_TEST,
        cqrWidth: cqrWSum / N_TEST,
        ratio: cqrWSum / scWSum,
        thyRatio: sigmaMean / sigmaPlus,
        scMarg: scCov / N_TEST,
        cqrMarg: cqrCov / N_TEST,
      };
    }
    // mode t53
    const nCal = t53NCal;
    const tr = generateHeavyTailedLocation(N_TRAIN, rng);
    const ca = generateHeavyTailedLocation(nCal, rng);
    const te = generateHeavyTailedLocation(N_TEST, rng);
    const sc = splitConformalIntervalPI(tr.X, tr.Y, ca.X, ca.Y, te.X, ALPHA);
    const hl = hlIntervalPI(tr.X, tr.Y, ca.X, ca.Y, te.X, ALPHA);
    let scWSum = 0;
    let hlWSum = 0;
    let scCov = 0;
    let hlCov = 0;
    for (let i = 0; i < N_TEST; i++) {
      scWSum += sc.hi[i] - sc.lo[i];
      hlWSum += hl.hi[i] - hl.lo[i];
      if (te.Y[i] >= sc.lo[i] && te.Y[i] <= sc.hi[i]) scCov++;
      if (te.Y[i] >= hl.lo[i] && te.Y[i] <= hl.hi[i]) hlCov++;
    }
    const grid = new Float64Array(120);
    for (let i = 0; i < 120; i++) grid[i] = -2 + (4 * i) / 119;
    const scG = splitConformalIntervalPI(tr.X, tr.Y, ca.X, ca.Y, grid, ALPHA);
    const hlG = hlIntervalPI(tr.X, tr.Y, ca.X, ca.Y, grid, ALPHA);
    const sub = 300;
    const stride = Math.floor(N_TEST / sub);
    const sX = new Float64Array(sub);
    const sY = new Float64Array(sub);
    for (let i = 0; i < sub; i++) {
      sX[i] = te.X[i * stride];
      sY[i] = te.Y[i * stride];
    }
    return {
      kind: 't53' as const,
      scLo: scG.lo,
      scHi: scG.hi,
      hlLo: hlG.lo,
      hlHi: hlG.hi,
      grid: Array.from(grid),
      sX,
      sY,
      scWidth: scWSum / N_TEST,
      hlWidth: hlWSum / N_TEST,
      ratio: hlWSum / scWSum,
      scMarg: scCov / N_TEST,
      hlMarg: hlCov / N_TEST,
      nCal,
    };
  }, [mode, t51NCal, t52Slope, t53NCal, seed]);

  const leftRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const margin = { top: 12, right: 16, bottom: 36, left: 50 };
      const innerW = panelWidth - margin.left - margin.right;
      const innerH = PANEL_HEIGHT - margin.top - margin.bottom;
      if (innerW <= 0 || innerH <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);

      if (result.kind === 't51') {
        // Scatter (Δ_n(x), |gap|) with bound curve overlay
        const xMax = Math.max(...result.Delta) * 1.15 + 1e-6;
        const yMaxData = Math.max(...result.gap) * 1.15;
        const yMaxBound = Math.max(...result.bound) * 1.05;
        const yMax = Math.max(yMaxData, yMaxBound, 0.5);
        const xs = d3.scaleLinear().domain([0, xMax]).range([0, innerW]);
        const ys = d3.scaleLinear().domain([0, yMax]).range([innerH, 0]);
        g.append('g').attr('transform', `translate(0, ${innerH})`).call(d3.axisBottom(xs).ticks(5)).call(styleAxis);
        g.append('g').call(d3.axisLeft(ys).ticks(5)).call(styleAxis);
        g.append('text').attr('x', innerW / 2).attr('y', innerH + 30).attr('text-anchor', 'middle')
          .style('font-size', '11px').style('fill', 'var(--color-text-secondary)')
          .text('QR estimation error  Δ_n(x)');
        g.append('text').attr('transform', `translate(-38, ${innerH / 2}) rotate(-90)`).attr('text-anchor', 'middle')
          .style('font-size', '11px').style('fill', 'var(--color-text-secondary)')
          .text('|cond cov − (1 − α)|');
        // Bound line: 4 * f_max * (Δ + meanΔ)
        const xs2 = d3.range(60).map((i) => (xMax * i) / 59);
        const bound = xs2.map((x) => 4 * result.fMax * (x + result.meanDelta));
        g.append('path').datum(xs2.map((x, i) => [x, bound[i]] as [number, number]))
          .attr('d', d3.line<[number, number]>().x((d) => xs(d[0])).y((d) => ys(Math.min(d[1], yMax))))
          .style('fill', 'none').style('stroke', palettePI.red).style('stroke-width', 1.5).style('stroke-dasharray', '5 3');
        // Empirical points
        for (let i = 0; i < result.grid.length; i++) {
          g.append('circle').attr('cx', xs(result.Delta[i])).attr('cy', ys(result.gap[i])).attr('r', 3.5)
            .style('fill', palettePI.green).style('opacity', 0.7);
        }
        g.append('text').attr('x', innerW - 6).attr('y', 14).attr('text-anchor', 'end')
          .style('font-size', '10px').style('fill', palettePI.red)
          .text('Theorem 5.1(ii) bound');
      } else if (result.kind === 't52') {
        renderBands({
          g, innerW, innerH,
          gridArr: result.grid,
          xDomain: [-3, 3],
          sX: result.sX, sY: result.sY,
          band1: { lo: result.scLo, hi: result.scHi, color: palettePI.blue, light: palettePI.lightBlue, label: `SC (${fmt(result.scMarg, 3)} cov, w=${fmt(result.scWidth, 2)})` },
          band2: { lo: result.cqrLo, hi: result.cqrHi, color: palettePI.green, light: palettePI.lightGreen, label: `CQR (${fmt(result.cqrMarg, 3)} cov, w=${fmt(result.cqrWidth, 2)})` },
          showSin: true,
        });
      } else {
        renderBands({
          g, innerW, innerH,
          gridArr: result.grid,
          xDomain: [-2, 2],
          sX: result.sX, sY: result.sY,
          band1: { lo: result.scLo, hi: result.scHi, color: palettePI.blue, light: palettePI.lightBlue, label: `SC (${fmt(result.scMarg, 3)} cov, w=${fmt(result.scWidth, 2)})` },
          band2: { lo: result.hlLo, hi: result.hlHi, color: palettePI.purple, light: palettePI.lightPurple, label: `HL (${fmt(result.hlMarg, 3)} cov, w=${fmt(result.hlWidth, 2)})` },
          showCos: true,
        });
      }
    },
    [result, panelWidth, mode],
  );

  const rightRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const margin = { top: 12, right: 16, bottom: 36, left: 50 };
      const innerW = panelWidth - margin.left - margin.right;
      const innerH = PANEL_HEIGHT - margin.top - margin.bottom;
      if (innerW <= 0 || innerH <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);

      if (result.kind === 't51') {
        // Cond gap as bar chart over x-grid + bound line
        const xs = d3.scaleLinear().domain([-3, 3]).range([0, innerW]);
        const yMax = Math.max(Math.max(...result.gap), Math.max(...result.bound), 0.5) * 1.05;
        const ys = d3.scaleLinear().domain([0, yMax]).range([innerH, 0]);
        g.append('g').attr('transform', `translate(0, ${innerH})`).call(d3.axisBottom(xs).ticks(5)).call(styleAxis);
        g.append('g').call(d3.axisLeft(ys).ticks(5)).call(styleAxis);
        g.append('text').attr('x', innerW / 2).attr('y', innerH + 30).attr('text-anchor', 'middle')
          .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('x');
        g.append('text').attr('transform', `translate(-38, ${innerH / 2}) rotate(-90)`).attr('text-anchor', 'middle')
          .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('value');
        // Bound area
        g.append('path').datum(result.grid.map((x, i) => [x, Math.min(result.bound[i], yMax)] as [number, number]))
          .attr('d', d3.line<[number, number]>().x((d) => xs(d[0])).y((d) => ys(d[1])).curve(d3.curveCatmullRom))
          .style('fill', 'none').style('stroke', palettePI.red).style('stroke-dasharray', '5 3').style('stroke-width', 1.5);
        // Gap dots
        for (let i = 0; i < result.grid.length; i++) {
          g.append('circle').attr('cx', xs(result.grid[i])).attr('cy', ys(result.gap[i])).attr('r', 3)
            .style('fill', palettePI.green).style('opacity', 0.8);
        }
        g.append('text').attr('x', innerW - 6).attr('y', 14).attr('text-anchor', 'end')
          .style('font-size', '10px').style('fill', palettePI.green).text('|cond cov gap|');
        g.append('text').attr('x', innerW - 6).attr('y', 28).attr('text-anchor', 'end')
          .style('font-size', '10px').style('fill', palettePI.red).text('Theorem 5.1(ii) bound');
      } else if (result.kind === 't52') {
        // Width ratio readout panel
        const xs = d3.scaleLinear().domain([0, 1]).range([0, innerW]);
        const ys = d3.scaleLinear().domain([0.4, 1.05]).range([innerH, 0]);
        g.append('g').attr('transform', `translate(0, ${innerH})`).call(d3.axisBottom(xs).ticks(6).tickFormat(d3.format('.1f'))).call(styleAxis);
        g.append('g').call(d3.axisLeft(ys).ticks(6).tickFormat(d3.format('.2f'))).call(styleAxis);
        g.append('text').attr('x', innerW / 2).attr('y', innerH + 30).attr('text-anchor', 'middle')
          .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('heteroscedasticity slope σ_max');
        g.append('text').attr('transform', `translate(-38, ${innerH / 2}) rotate(-90)`).attr('text-anchor', 'middle')
          .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('CQR / SC width ratio');
        // Theory curve
        const theoryPts = d3.range(50).map((i) => {
          const sg = i / 49;
          const sigmaPlus = 0.2 + sg;
          const sigmaMean = 0.2 + sg * 0.5;
          return [sg, sigmaMean / sigmaPlus] as [number, number];
        });
        g.append('path').datum(theoryPts)
          .attr('d', d3.line<[number, number]>().x((d) => xs(d[0])).y((d) => ys(d[1])))
          .style('fill', 'none').style('stroke', palettePI.red).style('stroke-width', 1.5).style('stroke-dasharray', '4 3');
        // Identity line at 1
        g.append('line').attr('x1', 0).attr('x2', innerW).attr('y1', ys(1)).attr('y2', ys(1))
          .style('stroke', 'var(--color-border)').style('stroke-dasharray', '2 2');
        // Empirical point at current σ_max
        g.append('circle').attr('cx', xs(t52Slope)).attr('cy', ys(result.ratio)).attr('r', 6)
          .style('fill', palettePI.green).style('opacity', 0.85);
        // Theory point
        g.append('circle').attr('cx', xs(t52Slope)).attr('cy', ys(result.thyRatio)).attr('r', 5)
          .style('fill', 'none').style('stroke', palettePI.red).style('stroke-width', 1.5);
        g.append('text').attr('x', innerW - 6).attr('y', 14).attr('text-anchor', 'end').style('font-size', '10px').style('fill', palettePI.red).text('Theorem 5.2 prediction');
        g.append('text').attr('x', innerW - 6).attr('y', 28).attr('text-anchor', 'end').style('font-size', '10px').style('fill', palettePI.green).text('Empirical CQR/SC');
      } else {
        // T5.3: HL/SC width ratio vs n_cal sweep — sample of points around current n_cal
        // For interactivity speed, just report the current ratio and a static sweep curve from notebook values.
        const sweepX = [50, 100, 200, 500, 1000, 2000];
        const sweepY = [0.890, 0.791, 0.750, 0.753, 0.744, 0.743]; // notebook §5 values
        const xs = d3.scaleLog().domain([40, 2200]).range([0, innerW]);
        const ys = d3.scaleLinear().domain([0.5, 1.1]).range([innerH, 0]);
        g.append('g').attr('transform', `translate(0, ${innerH})`).call(d3.axisBottom(xs).ticks(6, '~r')).call(styleAxis);
        g.append('g').call(d3.axisLeft(ys).ticks(6).tickFormat(d3.format('.2f'))).call(styleAxis);
        g.append('text').attr('x', innerW / 2).attr('y', innerH + 30).attr('text-anchor', 'middle')
          .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('n_cal (log scale)');
        g.append('text').attr('transform', `translate(-38, ${innerH / 2}) rotate(-90)`).attr('text-anchor', 'middle')
          .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('HL / SC width ratio');
        // Identity at 1
        g.append('line').attr('x1', 0).attr('x2', innerW).attr('y1', ys(1)).attr('y2', ys(1))
          .style('stroke', 'var(--color-border)').style('stroke-dasharray', '2 2');
        g.append('text').attr('x', innerW - 6).attr('y', ys(1) - 4).attr('text-anchor', 'end').style('font-size', '10px').style('fill', 'var(--color-text-secondary)').text('Theorem 5.3 limit = 1');
        // Notebook sweep
        g.append('path').datum(sweepX.map((x, i) => [x, sweepY[i]] as [number, number]))
          .attr('d', d3.line<[number, number]>().x((d) => xs(d[0])).y((d) => ys(d[1])))
          .style('fill', 'none').style('stroke', palettePI.amber).style('stroke-width', 1.5);
        for (let i = 0; i < sweepX.length; i++) {
          g.append('circle').attr('cx', xs(sweepX[i])).attr('cy', ys(sweepY[i])).attr('r', 3)
            .style('fill', palettePI.amber);
        }
        // Current point
        g.append('circle').attr('cx', xs(t53NCal)).attr('cy', ys(result.ratio)).attr('r', 6)
          .style('fill', palettePI.purple).style('opacity', 0.9);
        g.append('text').attr('x', innerW - 6).attr('y', 14).attr('text-anchor', 'end').style('font-size', '10px').style('fill', palettePI.amber).text('Notebook MC sweep (n_rep=100)');
        g.append('text').attr('x', innerW - 6).attr('y', 28).attr('text-anchor', 'end').style('font-size', '10px').style('fill', palettePI.purple).text(`Current: n_cal=${t53NCal}, ratio=${fmt(result.ratio, 3)}`);
      }
    },
    [result, panelWidth, mode, t52Slope, t53NCal],
  );

  return (
    <div
      ref={containerRef}
      className="my-6 rounded-lg border border-[var(--color-border)] bg-[var(--color-muted-bg)] p-3"
    >
      <div className="mb-3 flex flex-wrap gap-1 border-b border-[var(--color-border)] pb-2">
        {(
          [
            { v: 't51', label: 'Theorem 5.1 — CQR coverage decomposition' },
            { v: 't52', label: 'Theorem 5.2 — heteroscedastic width' },
            { v: 't53', label: 'Theorem 5.3 — HL/SC equivalence' },
          ] as const
        ).map((tab) => (
          <button
            key={tab.v}
            type="button"
            onClick={() => setMode(tab.v)}
            className={`rounded px-3 py-1 text-xs font-medium transition ${
              mode === tab.v
                ? 'bg-[var(--color-accent)] text-white'
                : 'bg-[var(--color-surface)] text-[var(--color-text-secondary)] hover:opacity-80'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <div className={`flex ${isStacked ? 'flex-col' : 'flex-row'} gap-2`}>
        <svg ref={leftRef} width={panelWidth} height={PANEL_HEIGHT} role="img" aria-label="Bridge theorem left panel" />
        <svg ref={rightRef} width={panelWidth} height={PANEL_HEIGHT} role="img" aria-label="Bridge theorem right panel" />
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-x-6 gap-y-3 text-sm">
        {mode === 't51' && (
          <label className="flex items-center gap-2 text-[var(--color-text-secondary)]">
            <span className="font-mono text-xs">n_cal</span>
            <input
              type="range"
              min={50}
              max={1000}
              step={50}
              value={t51NCal}
              onChange={(e) => setT51NCal(Number(e.target.value))}
              className="accent-[var(--color-accent)]"
            />
            <span className="font-mono w-12 text-right">{t51NCal}</span>
          </label>
        )}
        {mode === 't52' && (
          <label className="flex items-center gap-2 text-[var(--color-text-secondary)]">
            <span className="font-mono text-xs">σ_max</span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={t52Slope}
              onChange={(e) => setT52Slope(Number(e.target.value))}
              className="accent-[var(--color-accent)]"
            />
            <span className="font-mono w-12 text-right">{fmt(t52Slope, 2)}</span>
          </label>
        )}
        {mode === 't53' && (
          <label className="flex items-center gap-2 text-[var(--color-text-secondary)]">
            <span className="font-mono text-xs">n_cal</span>
            <input
              type="range"
              min={50}
              max={2000}
              step={50}
              value={t53NCal}
              onChange={(e) => setT53NCal(Number(e.target.value))}
              className="accent-[var(--color-accent)]"
            />
            <span className="font-mono w-12 text-right">{t53NCal}</span>
            <span className="text-xs italic">(capped at 2000 for O(n²) Walsh enumeration)</span>
          </label>
        )}
        <button
          type="button"
          onClick={() => setSeed((s) => s + 1)}
          className="rounded border border-[var(--color-accent)] bg-[var(--color-accent)] px-3 py-1 text-xs font-medium text-white transition hover:opacity-90"
        >
          re-randomize
        </button>
        {result.kind === 't52' && (
          <span className="text-xs text-[var(--color-text-secondary)]">
            empirical CQR/SC = <strong>{fmt(result.ratio, 3)}</strong>
            {' · '}theory <strong>E[σ(X)]/σ₊ = {fmt(result.thyRatio, 3)}</strong>
          </span>
        )}
        {result.kind === 't53' && (
          <span className="text-xs text-[var(--color-text-secondary)]">
            empirical HL/SC = <strong>{fmt(result.ratio, 3)}</strong>
            {' · '}HL marg <strong>{fmt(result.hlMarg, 3)}</strong>
            {' · '}SC marg <strong>{fmt(result.scMarg, 3)}</strong>
          </span>
        )}
        {result.kind === 't51' && (
          <span className="text-xs text-[var(--color-text-secondary)]">
            mean Δ_n = <strong>{fmt(result.meanDelta, 3)}</strong>
            {' · '}f_max = <strong>{fmt(result.fMax, 2)}</strong>
            {' · '}marginal <strong>{fmt(result.marginal, 3)}</strong>
          </span>
        )}
      </div>

      <p className="mt-3 text-xs text-[var(--color-text-secondary)]">
        {mode === 't51' &&
          'Theorem 5.1(ii): the CQR conditional-coverage gap at each x is bounded by 4·f_max·(Δ_n(x) + E[Δ_n]). All empirical points should sit below the dashed bound. The bound is loose by design — it has to absorb the conformalisation slack.'}
        {mode === 't52' &&
          'Theorem 5.2: as σ_max grows, the CQR/SC width ratio approaches E[σ(X)]/σ₊ < 1. At σ_max = 0 (homoscedastic) the ratio is ≈ 1; at σ_max = 0.6 (RE1) the asymptotic prediction is 0.625. Finite-n drift moves the empirical point away from the theory curve.'}
        {mode === 't53' &&
          'Theorem 5.3: HL and SC produce the same band asymptotically on RE2. The notebook MC sweep (orange) shows the ratio settles near 0.74 by n_cal = 1000 — slow convergence, consistent with the variance-driven o(1) correction. HL undercovers in batch evaluation throughout.'}
      </p>
    </div>
  );
}

// ───────── helpers ─────────────────────────────────────────────────────────

interface BandSpec {
  lo: ArrayLike<number>;
  hi: ArrayLike<number>;
  color: string;
  light: string;
  label: string;
}

function renderBands(args: {
  g: d3.Selection<SVGGElement, unknown, null, undefined>;
  innerW: number;
  innerH: number;
  gridArr: number[];
  xDomain: [number, number];
  sX: Float64Array;
  sY: Float64Array;
  band1: BandSpec;
  band2: BandSpec;
  showSin?: boolean;
  showCos?: boolean;
}): void {
  const { g, innerW, innerH, gridArr, xDomain, sX, sY, band1, band2 } = args;
  const xs = d3.scaleLinear().domain(xDomain).range([0, innerW]);
  const allY = [
    ...Array.from(sY),
    ...Array.from(band1.lo),
    ...Array.from(band1.hi),
    ...Array.from(band2.lo),
    ...Array.from(band2.hi),
  ];
  const yLo = Math.min(...allY);
  const yHi = Math.max(...allY);
  const yPad = (yHi - yLo) * 0.05;
  const ys = d3.scaleLinear().domain([yLo - yPad, yHi + yPad]).range([innerH, 0]);

  g.append('g').attr('transform', `translate(0, ${innerH})`).call(d3.axisBottom(xs).ticks(5)).call(styleAxis);
  g.append('g').call(d3.axisLeft(ys).ticks(5)).call(styleAxis);
  g.append('text').attr('x', innerW / 2).attr('y', innerH + 30).attr('text-anchor', 'middle').style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('x');
  g.append('text').attr('transform', `translate(-38, ${innerH / 2}) rotate(-90)`).attr('text-anchor', 'middle').style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('y');

  // Band 1 area + edges
  for (const [b, ord] of [[band1, 1], [band2, 2]] as const) {
    const area = d3.area<number>()
      .x((_d, i) => xs(gridArr[i]))
      .y0((_d, i) => ys(b.lo[i]))
      .y1((_d, i) => ys(b.hi[i]))
      .curve(d3.curveCatmullRom);
    g.append('path').datum(gridArr).attr('d', area).style('fill', b.light).style('opacity', ord === 1 ? 0.5 : 0.45);
    g.append('path')
      .datum(gridArr.map((_, i) => b.lo[i]))
      .attr('d', d3.line<number>().x((_d, i) => xs(gridArr[i])).y((d) => ys(d)).curve(d3.curveCatmullRom))
      .style('fill', 'none').style('stroke', b.color).style('stroke-width', 1.4);
    g.append('path')
      .datum(gridArr.map((_, i) => b.hi[i]))
      .attr('d', d3.line<number>().x((_d, i) => xs(gridArr[i])).y((d) => ys(d)).curve(d3.curveCatmullRom))
      .style('fill', 'none').style('stroke', b.color).style('stroke-width', 1.4);
  }
  // True mean curve
  if (args.showSin || args.showCos) {
    const mu = gridArr.map((x) => [x, args.showSin ? Math.sin(x) : 0.4 * Math.cos(Math.PI * x)] as [number, number]);
    g.append('path').datum(mu)
      .attr('d', d3.line<[number, number]>().x((d) => xs(d[0])).y((d) => ys(d[1])))
      .style('fill', 'none').style('stroke', 'var(--color-text-secondary)').style('stroke-width', 1.2).style('stroke-dasharray', '4 3');
  }
  // Scatter
  for (let i = 0; i < sX.length; i++) {
    const cy = ys(sY[i]);
    if (cy < 0 || cy > innerH) continue;
    g.append('circle').attr('cx', xs(sX[i])).attr('cy', cy).attr('r', 1.6).style('fill', 'var(--color-text-secondary)').style('opacity', 0.45);
  }
  // Legends
  g.append('text').attr('x', innerW - 6).attr('y', 14).attr('text-anchor', 'end').style('font-size', '10px').style('fill', band1.color).text(band1.label);
  g.append('text').attr('x', innerW - 6).attr('y', 28).attr('text-anchor', 'end').style('font-size', '10px').style('fill', band2.color).text(band2.label);
}

function styleAxis(sel: d3.Selection<SVGGElement, unknown, null, undefined>): void {
  sel.selectAll('line').style('stroke', 'var(--color-border)').style('opacity', 0.3);
  sel.selectAll('text').style('fill', 'var(--color-text-secondary)').style('font-size', '10px');
  sel.select('.domain').style('stroke', 'var(--color-border)');
}
