import { useEffect, useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  bootstrapQuantileCI,
  mulberry32,
  synthHeteroscedastic,
} from './shared/nonparametric-ml';

// =============================================================================
// BootstrapQuantileCI — embedded in §5 after Theorem 3.
//
// Two-panel demonstration of asymptotic normality of QR (Theorem 3 + Remark 8):
//
//   Top/left    : Histogram of B bootstrap β̂₁(τ) draws at the user's current
//                 (n, τ). Overlays a fitted Gaussian density. On hydration runs
//                 a quick B = 50 preview (~1 s); the "Run B = 200" button
//                 replaces it with the precise version (~5 s at n = 200).
//
//   Bottom/right: Convergence plot — empirical std × √n on the y-axis, n on a
//                 log x-axis. Two lines (τ = 0.5 and τ = 0.9). At fixed τ this
//                 should stabilize as n grows (Theorem 3's 1/√n rate); across
//                 τ, the τ = 0.9 line sits above τ = 0.5 (Remark 8 tail
//                 inflation, since D(τ) shrinks in the tail). User's current
//                 (n, τ) shown as a marker on the relevant line.
//
// The convergence-plot grid is precomputed once on mount (deferred 80 ms after
// render so the histogram appears first) — each (n, τ) cell uses B = 30 to
// keep total precompute under ~2 s.
// =============================================================================

const PANEL_HEIGHT = 270;
const SM_BREAKPOINT = 640;
const N_OPTIONS = [50, 100, 200, 500, 1000] as const;
const CONV_N_GRID = [50, 100, 200, 500] as const;
const CONV_TAUS = [0.5, 0.9] as const;
// Per-(n, τ) cell averages B/SEEDS bootstraps over SEEDS data realizations,
// total = B fits. Averaging over data seeds reduces bootstrap-of-bootstrap
// noise — without it, the τ = 0.9 line was visibly noisier than the τ = 0.5
// line and the qualitative "tail inflation" message was muddied.
const CONV_SEEDS = 3;
const CONV_B = 30; // total fits per cell, split across CONV_SEEDS data seeds
const QUICK_B = 50;
const PRECISE_B = 200;

const BLUE = '#2563EB';
const RED = '#DC2626';
const TEAL = '#0F6E56';
const SLATE = '#475569';

// Inline standard-Normal PDF for the Gaussian overlay (fitted to bootstrap
// mean + std per Theorem 3 / Remark 9).
function normalPDF(x: number, mu: number, sigma: number): number {
  const z = (x - mu) / sigma;
  return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI));
}

interface ConvergenceData {
  nGrid: number[];
  taus: number[];
  stdSqrtN: number[][]; // shape (taus.length, nGrid.length)
}

function computeConvergenceData(): ConvergenceData {
  const perSeedB = Math.max(5, Math.round(CONV_B / CONV_SEEDS));
  const stdSqrtN: number[][] = [];
  for (const tau of CONV_TAUS) {
    const row: number[] = [];
    for (const nVal of CONV_N_GRID) {
      let sumStdSqrtN = 0;
      for (let s = 0; s < CONV_SEEDS; s++) {
        const dataRng = mulberry32(2026 + Math.round(tau * 100) + nVal + s * 911);
        const { x, y } = synthHeteroscedastic(nVal, dataRng);
        const bootRng = mulberry32(7777 + Math.round(tau * 100) + nVal + s * 311);
        const result = bootstrapQuantileCI(x, y, tau, perSeedB, 0.1, bootRng);
        sumStdSqrtN += result.empiricalStd * Math.sqrt(nVal);
      }
      row.push(sumStdSqrtN / CONV_SEEDS);
    }
    stdSqrtN.push(row);
  }
  return { nGrid: [...CONV_N_GRID], taus: [...CONV_TAUS], stdSqrtN };
}

const fmt = (x: number, d = 3) => x.toFixed(d);

export default function BootstrapQuantileCI() {
  const { ref: containerRef, width: containerWidth } =
    useResizeObserver<HTMLDivElement>();
  const [nIdx, setNIdx] = useState(2); // default n = 200
  const [tau, setTau] = useState(0.5);
  // precisionB resets to QUICK_B whenever (n, τ) changes; user clicks "Run" to
  // upgrade to PRECISE_B for the current setting.
  const [precisionB, setPrecisionB] = useState<number>(QUICK_B);
  const [running, setRunning] = useState(false);
  // computeKey: increments to force recomputation when sliders change OR
  // when "Run B = 200" is clicked. Lets us avoid a useEffect race.
  const [computeKey, setComputeKey] = useState(0);

  const n = N_OPTIONS[nIdx];

  // Reset to QUICK_B on slider change.
  useEffect(() => {
    setPrecisionB(QUICK_B);
    setComputeKey((k) => k + 1);
  }, [nIdx, tau]);

  // Heavy compute: bootstrap histogram at (n, τ, precisionB).
  const histogram = useMemo(() => {
    const dataRng = mulberry32(2026 + Math.round(tau * 100) + n);
    const { x, y } = synthHeteroscedastic(n, dataRng);
    const bootRng = mulberry32(7777 + Math.round(tau * 100) + n);
    const result = bootstrapQuantileCI(x, y, tau, precisionB, 0.1, bootRng);
    return result;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [n, tau, precisionB, computeKey]);

  // Defer-once convergence grid (precomputed via setTimeout to let the
  // histogram render first).
  const [convergence, setConvergence] = useState<ConvergenceData | null>(null);
  useEffect(() => {
    const id = setTimeout(() => setConvergence(computeConvergenceData()), 80);
    return () => clearTimeout(id);
  }, []);

  // Layout: panels side-by-side ≥ 640 px, stacked below.
  const isStacked = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const panelWidth = isStacked
    ? containerWidth || 0
    : Math.floor((containerWidth || 720) / 2) - 4;

  const onRunPrecise = () => {
    setRunning(true);
    // Defer state update to next tick so the "Computing…" message paints first.
    setTimeout(() => {
      setPrecisionB(PRECISE_B);
      setComputeKey((k) => k + 1);
      setRunning(false);
    }, 30);
  };

  // ── Left/top panel: histogram + Gaussian overlay ────────────────────────
  const histRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelWidth <= 0) return;
      const margin = { top: 36, right: 14, bottom: 36, left: 44 };
      const w = panelWidth - margin.left - margin.right;
      const h = PANEL_HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const draws = Array.from(histogram.coefDraws);
      const sigma = histogram.empiricalStd;
      const mu = histogram.empiricalMean;
      const xLo = mu - 4 * sigma;
      const xHi = mu + 4 * sigma;
      const xScale = d3.scaleLinear().domain([xLo, xHi]).range([0, w]);

      const numBins = Math.max(15, Math.min(40, Math.round(Math.sqrt(draws.length) * 2)));
      const bins = d3
        .bin<number, number>()
        .domain([xLo, xHi])
        .thresholds(numBins)(draws);
      const binW = (xHi - xLo) / numBins;
      // Convert counts to density (counts / (B · binW)) for direct comparison
      // against the Gaussian PDF overlay.
      const densities = bins.map((b) => b.length / (draws.length * binW));
      const gridX = d3.range(120).map((i) => xLo + ((xHi - xLo) * i) / 119);
      const gaussianY = gridX.map((xv) => normalPDF(xv, mu, sigma));
      const yMax = Math.max(d3.max(densities) || 0, d3.max(gaussianY) || 0) * 1.15;
      const yScale = d3.scaleLinear().domain([0, yMax]).range([h, 0]);

      g.append('g')
        .attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(xScale).ticks(5))
        .selectAll('text')
        .style('fill', 'var(--color-text)')
        .style('font-size', '10px');
      g.append('g')
        .call(d3.axisLeft(yScale).ticks(4))
        .selectAll('text')
        .style('fill', 'var(--color-text)')
        .style('font-size', '10px');
      g.selectAll('path.domain, .tick line').style('stroke', 'var(--color-border)');

      // Histogram bars
      g.selectAll('rect.bar')
        .data(bins)
        .enter()
        .append('rect')
        .attr('class', 'bar')
        .attr('x', (b) => xScale(b.x0!))
        .attr('y', (b, i) => yScale(densities[i]))
        .attr('width', (b) => Math.max(0, xScale(b.x1!) - xScale(b.x0!) - 1))
        .attr('height', (b, i) => h - yScale(densities[i]))
        .style('fill', BLUE)
        .style('opacity', 0.55);

      // Gaussian overlay
      const gaussLine = d3
        .line<number>()
        .x((i) => xScale(gridX[i]))
        .y((i) => yScale(gaussianY[i]));
      g.append('path')
        .datum(d3.range(gridX.length))
        .attr('d', gaussLine)
        .style('fill', 'none')
        .style('stroke', RED)
        .style('stroke-width', 2);

      // Mean tick
      g.append('line')
        .attr('x1', xScale(mu))
        .attr('x2', xScale(mu))
        .attr('y1', 0)
        .attr('y2', h)
        .style('stroke', 'var(--color-text)')
        .style('stroke-width', 0.7)
        .style('stroke-dasharray', '3,3')
        .style('opacity', 0.6);

      // Title
      svg
        .append('text')
        .attr('x', margin.left + w / 2)
        .attr('y', 16)
        .attr('text-anchor', 'middle')
        .style('font-family', 'var(--font-sans)')
        .style('font-size', '11.5px')
        .style('fill', 'var(--color-text)')
        .text(`Bootstrap β̂₁(τ),  τ = ${fmt(tau, 2)},  n = ${n}`);
      svg
        .append('text')
        .attr('x', margin.left + w / 2)
        .attr('y', 30)
        .attr('text-anchor', 'middle')
        .style('font-family', 'var(--font-sans)')
        .style('font-size', '10.5px')
        .style('fill', 'var(--color-text-secondary)')
        .text(
          `B = ${precisionB}  ·  μ̂ = ${fmt(mu)},  σ̂ = ${fmt(sigma)},  σ̂·√n = ${fmt(sigma * Math.sqrt(n))}`,
        );

      // x-axis label
      svg
        .append('text')
        .attr('x', margin.left + w / 2)
        .attr('y', PANEL_HEIGHT - 6)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '10px')
        .text('β̂₁(τ)');
    },
    [histogram, n, tau, precisionB, panelWidth],
  );

  // ── Right/bottom panel: convergence plot ────────────────────────────────
  const convRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelWidth <= 0) return;
      const margin = { top: 36, right: 14, bottom: 36, left: 44 };
      const w = panelWidth - margin.left - margin.right;
      const h = PANEL_HEIGHT - margin.top - margin.bottom;

      if (convergence === null) {
        // Placeholder while precomputing
        svg
          .append('text')
          .attr('x', margin.left + w / 2)
          .attr('y', margin.top + h / 2)
          .attr('text-anchor', 'middle')
          .style('fill', 'var(--color-text-secondary)')
          .style('font-family', 'var(--font-sans)')
          .style('font-size', '11.5px')
          .text('Computing convergence plot…');
        svg
          .append('text')
          .attr('x', margin.left + w / 2)
          .attr('y', margin.top + h / 2 + 18)
          .attr('text-anchor', 'middle')
          .style('fill', 'var(--color-text-secondary)')
          .style('font-family', 'var(--font-sans)')
          .style('font-size', '10px')
          .text(`(${CONV_TAUS.length} τ × ${CONV_N_GRID.length} n × B = ${CONV_B})`);
        return;
      }

      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const xScale = d3
        .scaleLog()
        .domain([CONV_N_GRID[0] * 0.85, CONV_N_GRID[CONV_N_GRID.length - 1] * 1.15])
        .range([0, w]);
      let yMax = 0;
      for (const row of convergence.stdSqrtN)
        for (const v of row) if (v > yMax) yMax = v;
      const yScale = d3.scaleLinear().domain([0, yMax * 1.15]).range([h, 0]);

      g.append('g')
        .attr('transform', `translate(0,${h})`)
        .call(
          d3
            .axisBottom(xScale)
            .tickValues([...CONV_N_GRID])
            .tickFormat(d3.format('d')),
        )
        .selectAll('text')
        .style('fill', 'var(--color-text)')
        .style('font-size', '10px');
      g.append('g')
        .call(d3.axisLeft(yScale).ticks(4))
        .selectAll('text')
        .style('fill', 'var(--color-text)')
        .style('font-size', '10px');
      g.selectAll('path.domain, .tick line').style('stroke', 'var(--color-border)');

      // Two lines, color-coded by τ
      const tauColor = (tauIdx: number) => (tauIdx === 0 ? BLUE : RED);
      for (let t = 0; t < convergence.taus.length; t++) {
        const tauVal = convergence.taus[t];
        const row = convergence.stdSqrtN[t];
        const line = d3
          .line<number>()
          .x((i) => xScale(convergence.nGrid[i]))
          .y((i) => yScale(row[i]));
        g.append('path')
          .datum(d3.range(convergence.nGrid.length))
          .attr('d', line)
          .style('fill', 'none')
          .style('stroke', tauColor(t))
          .style('stroke-width', 1.8);
        for (let i = 0; i < convergence.nGrid.length; i++) {
          g.append('circle')
            .attr('cx', xScale(convergence.nGrid[i]))
            .attr('cy', yScale(row[i]))
            .attr('r', 3.4)
            .style('fill', tauColor(t));
        }
        // Per-line legend label at the right end
        const last = convergence.nGrid.length - 1;
        g.append('text')
          .attr('x', xScale(convergence.nGrid[last]) + 4)
          .attr('y', yScale(row[last]))
          .attr('dy', '0.32em')
          .style('fill', tauColor(t))
          .style('font-family', 'var(--font-sans)')
          .style('font-size', '10px')
          .text(`τ = ${tauVal}`);
      }

      // Marker for user's current (n, τ): closest precomputed line + interpolated y
      const closestTauIdx = Math.abs(tau - 0.5) < Math.abs(tau - 0.9) ? 0 : 1;
      const userStdSqrtN = histogram.empiricalStd * Math.sqrt(n);
      g.append('circle')
        .attr('cx', xScale(n))
        .attr('cy', yScale(userStdSqrtN))
        .attr('r', 5)
        .style('fill', 'none')
        .style('stroke', tauColor(closestTauIdx))
        .style('stroke-width', 2);
      g.append('text')
        .attr('x', xScale(n))
        .attr('y', yScale(userStdSqrtN) - 10)
        .attr('text-anchor', 'middle')
        .style('fill', tauColor(closestTauIdx))
        .style('font-family', 'var(--font-sans)')
        .style('font-size', '9.5px')
        .text(`current  σ̂·√n = ${fmt(userStdSqrtN, 2)}`);

      // Title
      svg
        .append('text')
        .attr('x', margin.left + w / 2)
        .attr('y', 16)
        .attr('text-anchor', 'middle')
        .style('font-family', 'var(--font-sans)')
        .style('font-size', '11.5px')
        .style('fill', 'var(--color-text)')
        .text(`σ̂(β̂₁) · √n  vs  n  (log x)`);
      svg
        .append('text')
        .attr('x', margin.left + w / 2)
        .attr('y', 30)
        .attr('text-anchor', 'middle')
        .style('font-family', 'var(--font-sans)')
        .style('font-size', '10.5px')
        .style('fill', 'var(--color-text-secondary)')
        .text(
          `stabilises at fixed τ (Thm 3); τ = 0.9 line sits higher (Rem 8 tail inflation)`,
        );

      // x-axis label
      svg
        .append('text')
        .attr('x', margin.left + w / 2)
        .attr('y', PANEL_HEIGHT - 6)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '10px')
        .text('n  (sample size, log scale)');
    },
    [convergence, histogram, n, tau, panelWidth],
  );

  return (
    <div ref={containerRef} className="not-prose">
      <div
        style={{
          marginBottom: '0.75rem',
          display: 'flex',
          flexWrap: 'wrap',
          gap: '0.6rem 1.2rem',
          alignItems: 'center',
          fontFamily: 'var(--font-sans)',
          fontSize: '12px',
        }}
      >
        <label
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            flex: '1 1 220px',
            minWidth: 200,
          }}
        >
          <span style={{ color: 'var(--color-text-secondary)', minWidth: '1.2em' }}>τ</span>
          <input
            type="range"
            min={0.1}
            max={0.9}
            step={0.05}
            value={tau}
            onChange={(e) => setTau(Number(e.target.value))}
            style={{ flex: 1 }}
            aria-label="Bootstrap τ slider"
          />
          <span
            style={{
              fontFamily: 'var(--font-mono)',
              minWidth: '3em',
              textAlign: 'right',
              color: 'var(--color-text)',
            }}
          >
            {fmt(tau, 2)}
          </span>
        </label>

        <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span style={{ color: 'var(--color-text-secondary)' }}>n</span>
          <input
            type="range"
            min={0}
            max={N_OPTIONS.length - 1}
            step={1}
            value={nIdx}
            onChange={(e) => setNIdx(Number(e.target.value))}
            aria-label="Bootstrap sample-size n slider"
          />
          <span style={{ fontFamily: 'var(--font-mono)', minWidth: '3em', textAlign: 'right' }}>
            {N_OPTIONS[nIdx]}
          </span>
        </label>

        <button
          type="button"
          onClick={onRunPrecise}
          disabled={precisionB === PRECISE_B || running}
          style={{
            padding: '4px 10px',
            background:
              precisionB === PRECISE_B ? 'var(--color-border)' : 'var(--color-surface)',
            color: 'var(--color-text)',
            border: '1px solid var(--color-border)',
            borderRadius: 4,
            fontSize: '11.5px',
            cursor: precisionB === PRECISE_B || running ? 'default' : 'pointer',
            opacity: precisionB === PRECISE_B || running ? 0.6 : 1,
          }}
        >
          {running
            ? 'Running…'
            : precisionB === PRECISE_B
              ? `B = ${PRECISE_B} ✓`
              : `Run B = ${PRECISE_B} bootstraps`}
        </button>

        <span
          style={{
            color: 'var(--color-text-secondary)',
            fontSize: '11px',
            fontStyle: 'italic',
          }}
        >
          (sliders auto-run B = {QUICK_B} for fast feedback)
        </span>
      </div>
      <div
        style={{
          display: 'flex',
          flexDirection: isStacked ? 'column' : 'row',
          gap: isStacked ? '0.75rem' : '0.4rem',
        }}
      >
        <svg
          ref={histRef}
          width={panelWidth}
          height={PANEL_HEIGHT}
          role="img"
          aria-label="Bootstrap distribution of QR slope coefficient with Gaussian overlay"
        />
        <svg
          ref={convRef}
          width={panelWidth}
          height={PANEL_HEIGHT}
          role="img"
          aria-label="Empirical std × √n vs sample size n at two τ levels"
        />
      </div>
    </div>
  );
}
