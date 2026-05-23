import { useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import { betaPdf, betaMean, mulberry32 } from './shared/bayesian-inference';

// =============================================================================
// ConjugateUpdateAnimator — §4 interactive.
//
// Animates the Beta posterior Beta(1+s_k, 1+n_k-s_k) sweeping through
// cumulative sample sizes n_k as Bernoulli(theta_true) draws arrive one at
// a time. Posteriors color-coded along a viridis gradient (light = early,
// dark = late). The true rate is marked as a vertical line.
//
// "Next observation" advances by one. "Play / Pause" runs through n_max
// in real time. "Reset" returns to n=0.
// =============================================================================

const N_MAX = 1000;
const N_GRID = 401;
const theta = Float64Array.from({ length: N_GRID }, (_, i) => (i * 0.2) / (N_GRID - 1));
const TRUE_RATE = 0.047;
const CHECKPOINTS = [0, 5, 10, 25, 50, 100, 250, 500, 1000];

function precomputeData(rngSeed: number): { s: Uint32Array } {
  const rng = mulberry32(rngSeed);
  const cumS = new Uint32Array(N_MAX + 1);
  cumS[0] = 0;
  for (let i = 1; i <= N_MAX; i++) cumS[i] = cumS[i - 1] + (rng() < TRUE_RATE ? 1 : 0);
  return { s: cumS };
}

export default function ConjugateUpdateAnimator() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [n, setN] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [seed, setSeed] = useState(42);
  const rafRef = useRef<number | null>(null);

  const W = Math.max(360, Math.min(width || 720, 720));
  const H = 360;
  const margin = { top: 28, right: 14, bottom: 36, left: 50 };

  const { s: cumS } = useMemo(() => precomputeData(seed), [seed]);
  const sNow = cumS[n];

  // Cumulative posteriors at checkpoints + current n for the curve trace
  const curves = useMemo(() => {
    const out: { n: number; s: number; density: Float64Array }[] = [];
    const checkPlus = Array.from(new Set([...CHECKPOINTS, n])).sort((a, b) => a - b).filter((k) => k <= n);
    for (const k of checkPlus) {
      const alpha = 1 + cumS[k];
      const beta = 1 + k - cumS[k];
      const dens = new Float64Array(N_GRID);
      // Always sample Beta(1, 1) at k=0 — flat density on (0, 1) but unbounded; just clip
      for (let i = 0; i < N_GRID; i++) {
        const t = theta[i];
        if (t <= 0 || t >= 1) dens[i] = 0;
        else dens[i] = betaPdf(t, alpha, beta);
      }
      out.push({ n: k, s: cumS[k], density: dens });
    }
    return out;
  }, [n, cumS]);

  // Animation loop
  useEffect(() => {
    if (!playing) return;
    let last = performance.now();
    let frame = 0;
    const FRAMES_PER_TICK = 1; // advance one observation per RAF frame initially
    const SPEED = 8; // observations per second target (lower means slower)
    function tick(now: number) {
      const dt = now - last;
      last = now;
      frame += dt / (1000 / SPEED);
      const step = Math.floor(frame * FRAMES_PER_TICK);
      if (step > 0) {
        frame -= step / FRAMES_PER_TICK;
        setN((prev) => {
          const next = Math.min(prev + step, N_MAX);
          if (next === N_MAX) setPlaying(false);
          return next;
        });
      }
      rafRef.current = requestAnimationFrame(tick);
    }
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [playing]);

  const ref = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      svg.attr('viewBox', `0 0 ${W} ${H}`).attr('width', W).attr('height', H);

      const innerW = W - margin.left - margin.right;
      const innerH = H - margin.top - margin.bottom;
      const x = d3.scaleLinear().domain([0, 0.2]).range([0, innerW]);

      // Compute ymax across all displayed curves
      let ymax = 1;
      curves.forEach((c) => {
        for (let i = 0; i < c.density.length; i++) if (c.density[i] > ymax) ymax = c.density[i];
      });
      const y = d3.scaleLinear().domain([0, ymax * 1.05]).range([innerH, 0]);
      const g = svg.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);

      // Viridis color for each curve by its n
      const colorScale = d3.scaleSequential(d3.interpolateViridis).domain([0, N_MAX]);

      // Draw curves in order
      curves.forEach((c) => {
        if (c.n === 0) return; // skip flat prior to avoid clutter; mark separately
        g.append('path')
          .attr(
            'd',
            d3
              .line<number>()
              .x((_, i) => x(theta[i]))
              .y((v) => y(v))(Array.from(c.density)) ?? '',
          )
          .attr('fill', 'none')
          .attr('stroke', colorScale(c.n))
          .attr('stroke-width', c.n === n ? 2.4 : 1.2)
          .attr('opacity', c.n === n ? 1 : 0.65);
      });

      // True rate marker
      g.append('line')
        .attr('x1', x(TRUE_RATE))
        .attr('x2', x(TRUE_RATE))
        .attr('y1', 0)
        .attr('y2', innerH)
        .attr('stroke', '#000')
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '4 4');
      g.append('text')
        .attr('x', x(TRUE_RATE) + 4)
        .attr('y', 14)
        .attr('font-size', 10)
        .style('fill', 'var(--color-text)')
        .text('θ_true = 0.047');

      // Axes
      g.append('g')
        .attr('transform', `translate(0, ${innerH})`)
        .call(d3.axisBottom(x).ticks(5).tickSizeOuter(0));
      g.append('g').call(d3.axisLeft(y).ticks(5).tickSizeOuter(0));

      // Labels
      g.append('text')
        .attr('x', innerW / 2)
        .attr('y', innerH + 30)
        .attr('text-anchor', 'middle')
        .attr('font-size', 11)
        .style('fill', 'var(--color-text)')
        .text('θ');
      g.append('text')
        .attr('x', -innerH / 2)
        .attr('y', -36)
        .attr('text-anchor', 'middle')
        .attr('font-size', 11)
        .attr('transform', 'rotate(-90)')
        .style('fill', 'var(--color-text)')
        .text('π(θ | data)');
    },
    [curves, W, H],
  );

  const postMean = n > 0 ? betaMean(1 + sNow, 1 + n - sNow) : 0.5;

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <svg ref={ref} role="img" aria-label="Beta-Binomial conjugate update animator" />
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '0.75rem',
          marginTop: '0.5rem',
          fontSize: '0.85rem',
          alignItems: 'center',
        }}
      >
        <button
          onClick={() => setPlaying((p) => !p)}
          style={{
            padding: '0.3rem 0.75rem',
            border: '1px solid var(--color-text)',
            borderRadius: 4,
            background: 'transparent',
            color: 'var(--color-text)',
            cursor: 'pointer',
          }}
        >
          {playing ? 'Pause' : 'Play'}
        </button>
        <button
          onClick={() => {
            setPlaying(false);
            setN(0);
          }}
          style={{
            padding: '0.3rem 0.75rem',
            border: '1px solid var(--color-text)',
            borderRadius: 4,
            background: 'transparent',
            color: 'var(--color-text)',
            cursor: 'pointer',
          }}
        >
          Reset
        </button>
        <button
          onClick={() => {
            setSeed((s) => s + 1);
            setN(0);
            setPlaying(false);
          }}
          style={{
            padding: '0.3rem 0.75rem',
            border: '1px solid var(--color-text)',
            borderRadius: 4,
            background: 'transparent',
            color: 'var(--color-text)',
            cursor: 'pointer',
          }}
          aria-label="new draw"
        >
          New draws
        </button>
        <span>
          n = <strong>{n}</strong>, s = <strong>{sNow}</strong> → Beta({1 + sNow}, {1 + n - sNow}), mean ={' '}
          <strong>{postMean.toFixed(4)}</strong>
        </span>
      </div>
      <label style={{ display: 'block', marginTop: '0.5rem', fontSize: '0.85rem' }}>
        n (skip ahead): <strong>{n}</strong>
        <input
          type="range"
          min={0}
          max={N_MAX}
          step={1}
          value={n}
          onChange={(e) => {
            setPlaying(false);
            setN(Number(e.target.value));
          }}
          aria-label="cumulative sample size"
          style={{ width: '100%' }}
        />
      </label>
    </div>
  );
}
