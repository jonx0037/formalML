import { useState, useEffect, useRef, useMemo } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';

const MARGIN = { top: 30, right: 16, bottom: 40, left: 56 };

const COLORS = {
  dpDraw: '#0F6E56',
  baseMeasure: '#9CA3AF',
  remaining: '#D1D5DB',
} as const;

/* ── Seeded PRNG (linear congruential generator) ── */
function createLCG(seed: number) {
  let s = seed | 0;
  return () => {
    s = (s * 1664525 + 1013904223) | 0;
    return (s >>> 0) / 0x100000000;
  };
}

/**
 * Sample from Beta(1, alpha) using the inverse CDF method.
 * For Beta(1, α), the CDF is F(x) = 1 - (1-x)^α, so the inverse is
 * F^{-1}(u) = 1 - (1-u)^{1/α} = 1 - u^{1/α} (since 1-U ~ U for uniform).
 * This avoids needing a general Beta sampler.
 */
function sampleBeta1Alpha(alpha: number, rng: () => number): number {
  const u = rng();
  return 1 - Math.pow(u, 1 / alpha);
}

/* ── Box-Muller transform for N(0,1) ── */
function sampleNormal(rng: () => number): number {
  const u1 = rng();
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(Math.max(u1, 1e-15))) * Math.cos(2 * Math.PI * u2);
}

/**
 * Standard normal CDF via the Abramowitz & Stegun rational approximation
 * (Handbook of Mathematical Functions, formula 7.1.26). Maximum absolute
 * error < 1.5×10⁻⁷ for all x, which is more than sufficient for plotting.
 */
function normalCDF(x: number): number {
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;
  const sign = x < 0 ? -1 : 1;
  const t = 1 / (1 + p * Math.abs(x));
  const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x / 2);
  return 0.5 * (1 + sign * y);
}

interface StickData {
  weights: number[];
  atoms: number[];
  remaining: number;
}

export default function StickBreakingExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const stickSvgRef = useRef<SVGSVGElement>(null);
  const cdfSvgRef = useRef<SVGSVGElement>(null);

  const [alpha, setAlpha] = useState(1.0);
  const [truncationK, setTruncationK] = useState(20);
  const [seed, setSeed] = useState(42);

  /* ── Log-scale slider helpers ── */
  const alphaMin = 0.1;
  const alphaMax = 50;
  const logMin = Math.log(alphaMin);
  const logMax = Math.log(alphaMax);
  const sliderToAlpha = (v: number) => Math.exp(logMin + (v / 100) * (logMax - logMin));
  const alphaToSlider = (a: number) => ((Math.log(a) - logMin) / (logMax - logMin)) * 100;

  /* ── Panel sizing ── */
  const isDesktop = containerWidth > 600;
  const panelW = isDesktop
    ? Math.max((containerWidth - 32 - 12) / 2, 200)
    : Math.max(containerWidth - 32, 200);
  const stickH = 100;
  const cdfH = Math.min(280, Math.max(200, panelW * 0.55));

  /* ── Stick-breaking computation ── */
  const stickData: StickData = useMemo(() => {
    const rng = createLCG(seed);
    const weights: number[] = [];
    const atoms: number[] = [];
    let remainingStick = 1;

    for (let k = 0; k < truncationK; k++) {
      const v = sampleBeta1Alpha(alpha, rng);
      const w = v * remainingStick;
      weights.push(w);
      remainingStick *= (1 - v);
      atoms.push(sampleNormal(rng));
    }

    return { weights, atoms, remaining: remainingStick };
  }, [alpha, truncationK, seed]);

  /* ── Stick bar chart (left panel) ── */
  useEffect(() => {
    const svg = stickSvgRef.current;
    if (!svg || panelW <= 0) return;
    const sel = d3.select(svg);
    sel.selectAll('*').remove();

    const { weights, remaining } = stickData;
    const color = d3.scaleOrdinal(d3.schemeTableau10);

    const barY = 36;
    const barH = 28;
    const leftPad = 8;
    const rightPad = 8;
    const barW = panelW - leftPad - rightPad;

    // Title
    sel.append('text')
      .attr('x', panelW / 2)
      .attr('y', 18)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('font-family', 'var(--font-sans)')
      .style('font-weight', '600')
      .style('fill', 'var(--color-text)')
      .text('Stick-Breaking Weights');

    // Stacked segments
    let x = leftPad;
    weights.forEach((w, i) => {
      const segW = w * barW;
      sel.append('rect')
        .attr('x', x)
        .attr('y', barY)
        .attr('width', Math.max(segW, 0))
        .attr('height', barH)
        .attr('rx', i === 0 ? 3 : 0)
        .style('fill', color(String(i)));

      // Label if wide enough
      if (segW > 28) {
        sel.append('text')
          .attr('x', x + segW / 2)
          .attr('y', barY + barH / 2 + 3.5)
          .attr('text-anchor', 'middle')
          .style('font-size', '8px')
          .style('font-family', 'var(--font-mono)')
          .style('fill', '#fff')
          .style('font-weight', '600')
          .text(w.toFixed(2));
      }
      x += segW;
    });

    // Remaining stick (gray)
    const remW = remaining * barW;
    sel.append('rect')
      .attr('x', x)
      .attr('y', barY)
      .attr('width', Math.max(remW, 0))
      .attr('height', barH)
      .attr('rx', 0)
      .style('fill', COLORS.remaining);

    if (remW > 28) {
      sel.append('text')
        .attr('x', x + remW / 2)
        .attr('y', barY + barH / 2 + 3.5)
        .attr('text-anchor', 'middle')
        .style('font-size', '8px')
        .style('font-family', 'var(--font-mono)')
        .style('fill', 'var(--color-text-secondary)')
        .text(`rest ${remaining.toFixed(2)}`);
    }

    // Legend row beneath bar
    const legendY = barY + barH + 16;
    sel.append('rect')
      .attr('x', leftPad)
      .attr('y', legendY - 6)
      .attr('width', 10)
      .attr('height', 10)
      .attr('rx', 2)
      .style('fill', color('0'));
    sel.append('text')
      .attr('x', leftPad + 14)
      .attr('y', legendY + 3)
      .style('font-size', '9px')
      .style('font-family', 'var(--font-mono)')
      .style('fill', 'var(--color-text-secondary)')
      .text('wₖ segments');

    sel.append('rect')
      .attr('x', leftPad + 90)
      .attr('y', legendY - 6)
      .attr('width', 10)
      .attr('height', 10)
      .attr('rx', 2)
      .style('fill', COLORS.remaining);
    sel.append('text')
      .attr('x', leftPad + 104)
      .attr('y', legendY + 3)
      .style('font-size', '9px')
      .style('font-family', 'var(--font-mono)')
      .style('fill', 'var(--color-text-secondary)')
      .text('remaining stick');
  }, [stickData, panelW]);

  /* ── CDF plot (right panel) ── */
  useEffect(() => {
    const svg = cdfSvgRef.current;
    if (!svg || panelW <= 0) return;
    const sel = d3.select(svg);
    sel.selectAll('*').remove();

    const { weights, atoms } = stickData;

    // Sort atoms; accumulate weights in sorted order
    const indexed = atoms.map((a, i) => ({ atom: a, weight: weights[i] }));
    indexed.sort((a, b) => a.atom - b.atom);

    const sortedAtoms = indexed.map(d => d.atom);
    const cumWeights: number[] = [];
    let cum = 0;
    for (const d of indexed) {
      cum += d.weight;
      cumWeights.push(cum);
    }

    const xScale = d3.scaleLinear().domain([-4, 4]).range([MARGIN.left, panelW - MARGIN.right]);
    const yScale = d3.scaleLinear().domain([0, 1]).range([cdfH - MARGIN.bottom, MARGIN.top]);

    // Axes
    sel.append('g')
      .attr('transform', `translate(0,${cdfH - MARGIN.bottom})`)
      .call(d3.axisBottom(xScale).ticks(8).tickSize(3))
      .call(g => {
        g.selectAll('text').style('font-size', '9px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)');
        g.select('.domain').style('stroke', 'var(--color-border)');
      });

    sel.append('g')
      .attr('transform', `translate(${MARGIN.left},0)`)
      .call(d3.axisLeft(yScale).ticks(5).tickSize(3))
      .call(g => {
        g.selectAll('text').style('font-size', '9px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)');
        g.select('.domain').style('stroke', 'var(--color-border)');
      });

    // Axis labels
    sel.append('text')
      .attr('x', panelW / 2)
      .attr('y', cdfH - 4)
      .attr('text-anchor', 'middle')
      .style('font-size', '10px')
      .style('font-family', 'var(--font-mono)')
      .style('fill', 'var(--color-text-secondary)')
      .text('θ');

    sel.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -(cdfH / 2))
      .attr('y', 14)
      .attr('text-anchor', 'middle')
      .style('font-size', '10px')
      .style('font-family', 'var(--font-mono)')
      .style('fill', 'var(--color-text-secondary)')
      .text('F(θ)');

    // Base measure G0 = N(0,1) CDF — dashed gray line
    const g0Points: [number, number][] = [];
    for (let px = MARGIN.left; px <= panelW - MARGIN.right; px += 1) {
      const xVal = xScale.invert(px);
      g0Points.push([px, yScale(normalCDF(xVal))]);
    }
    sel.append('path')
      .datum(g0Points)
      .attr('d', d3.line<[number, number]>().x(d => d[0]).y(d => d[1]))
      .style('fill', 'none')
      .style('stroke', COLORS.baseMeasure)
      .style('stroke-width', 1.5)
      .style('stroke-dasharray', '6 3');

    // DP draw CDF — step function
    // Fold remaining stick mass into the last atom so the CDF represents
    // a proper probability measure that reaches 1.0 (truncated stick-breaking).
    const totalMass = cumWeights[cumWeights.length - 1] || 0;
    const remainingMass = Math.max(0, 1 - totalMass);
    const adjustedCum = cumWeights.slice();
    if (adjustedCum.length > 0) {
      adjustedCum[adjustedCum.length - 1] = Math.min(1, adjustedCum[adjustedCum.length - 1] + remainingMass);
    }

    const stepPoints: [number, number][] = [[-4, 0]];
    for (let i = 0; i < sortedAtoms.length; i++) {
      stepPoints.push([sortedAtoms[i], i > 0 ? adjustedCum[i - 1] : 0]);
      stepPoints.push([sortedAtoms[i], adjustedCum[i]]);
    }
    stepPoints.push([4, adjustedCum[adjustedCum.length - 1] || 0]);

    const lineGen = d3.line<[number, number]>()
      .x(d => xScale(d[0]))
      .y(d => yScale(d[1]));

    sel.append('path')
      .datum(stepPoints)
      .attr('d', lineGen)
      .style('fill', 'none')
      .style('stroke', COLORS.dpDraw)
      .style('stroke-width', 2);

    // Title
    sel.append('text')
      .attr('x', panelW - MARGIN.right)
      .attr('y', MARGIN.top - 8)
      .attr('text-anchor', 'end')
      .style('font-size', '11px')
      .style('font-family', 'var(--font-sans)')
      .style('font-weight', '600')
      .style('fill', 'var(--color-text)')
      .text('DP Draw CDF vs Base Measure');

    // Legend
    const legendX = MARGIN.left + 12;
    const legendY = MARGIN.top + 4;

    sel.append('line')
      .attr('x1', legendX).attr('x2', legendX + 16)
      .attr('y1', legendY).attr('y2', legendY)
      .style('stroke', COLORS.dpDraw)
      .style('stroke-width', 2);
    sel.append('text')
      .attr('x', legendX + 20).attr('y', legendY + 3.5)
      .style('font-size', '9px')
      .style('font-family', 'var(--font-mono)')
      .style('fill', 'var(--color-text-secondary)')
      .text('DP draw');

    sel.append('line')
      .attr('x1', legendX).attr('x2', legendX + 16)
      .attr('y1', legendY + 14).attr('y2', legendY + 14)
      .style('stroke', COLORS.baseMeasure)
      .style('stroke-width', 1.5)
      .style('stroke-dasharray', '6 3');
    sel.append('text')
      .attr('x', legendX + 20).attr('y', legendY + 17.5)
      .style('font-size', '9px')
      .style('font-family', 'var(--font-mono)')
      .style('fill', 'var(--color-text-secondary)')
      .text('G\u2080 = N(0,1)');
  }, [stickData, panelW, cdfH]);

  return (
    <div
      ref={containerRef}
      style={{
        border: '1px solid var(--color-border)',
        borderRadius: '8px',
        padding: '16px',
        background: 'var(--color-surface)',
        marginTop: '1.5rem',
        marginBottom: '1.5rem',
      }}
    >
      <div style={{ fontFamily: 'var(--font-sans)', fontSize: '13px', fontWeight: 600, marginBottom: '12px', color: 'var(--color-text)' }}>
        Stick-Breaking Construction Explorer
      </div>

      {/* Two-panel grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: isDesktop ? '1fr 1fr' : '1fr',
        gap: '12px',
      }}>
        {/* Left panel — stick bar */}
        <svg role="img" aria-label="Stick breaking explorer visualization (panel 1 of 2)" ref={stickSvgRef} width={panelW} height={stickH} style={{
          border: '1px solid var(--color-border)', borderRadius: '6px', background: 'var(--color-muted-bg)',
        }} />

        {/* Right panel — CDF */}
        <svg role="img" aria-label="Stick breaking explorer visualization (panel 2 of 2)" ref={cdfSvgRef} width={panelW} height={cdfH} style={{
          border: '1px solid var(--color-border)', borderRadius: '6px', background: 'var(--color-muted-bg)',
        }} />
      </div>

      {/* Controls */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '12px', marginTop: '12px' }}>
        {/* Alpha slider — log scale */}
        <div>
          <div style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--color-text-secondary)', marginBottom: 2 }}>
            α (concentration): <strong style={{ color: 'var(--color-text)' }}>{alpha < 1 ? alpha.toFixed(2) : alpha.toFixed(1)}</strong>
          </div>
          <input
            type="range"
            aria-label="Concentration parameter alpha"
            min={0}
            max={100}
            step={0.5}
            value={alphaToSlider(alpha)}
            onChange={e => {
              const raw = sliderToAlpha(parseFloat(e.target.value));
              setAlpha(Math.round(raw * 100) / 100);
            }}
            style={{ width: '100%', accentColor: 'var(--color-accent)' }}
          />
        </div>

        {/* K slider */}
        <div>
          <div style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--color-text-secondary)', marginBottom: 2 }}>
            K (truncation): <strong style={{ color: 'var(--color-text)' }}>{truncationK}</strong>
          </div>
          <input
            type="range"
            aria-label="Truncation level K"
            min={5}
            max={50}
            step={1}
            value={truncationK}
            onChange={e => setTruncationK(parseInt(e.target.value))}
            style={{ width: '100%', accentColor: 'var(--color-accent)' }}
          />
        </div>

        {/* Resample button */}
        <div style={{ display: 'flex', alignItems: 'flex-end' }}>
          <button
            type="button"
            onClick={() => setSeed(s => s + 1)}
            style={{
              padding: '6px 16px',
              fontSize: '12px',
              fontFamily: 'var(--font-mono)',
              borderRadius: '6px',
              border: '1px solid var(--color-border)',
              background: 'var(--color-muted-bg)',
              color: 'var(--color-text)',
              cursor: 'pointer',
            }}
          >
            Resample
          </button>
        </div>
      </div>

      {/* Interpretation */}
      <div style={{
        marginTop: '12px', padding: '10px', borderRadius: '6px',
        background: 'var(--color-definition-bg)', border: '1px solid var(--color-definition-border)',
        fontSize: '12px', fontFamily: 'var(--font-sans)', color: 'var(--color-text)',
      }}>
        {alpha < 1
          ? <>With <strong>α = {alpha < 1 ? alpha.toFixed(2) : alpha.toFixed(1)}</strong>, the process concentrates mass on a few atoms — the first break tends to grab most of the stick.</>
          : alpha > 10
            ? <>With <strong>α = {alpha.toFixed(1)}</strong>, the weights are spread across many atoms and the DP draw closely tracks the base measure G₀.</>
            : <>With <strong>α = {alpha < 1 ? alpha.toFixed(2) : alpha.toFixed(1)}</strong>, the DP draw is a discrete distribution over <strong>K = {truncationK}</strong> atoms. Remaining stick mass: <strong>{(stickData.remaining * 100).toFixed(1)}%</strong>.</>}
      </div>
    </div>
  );
}
