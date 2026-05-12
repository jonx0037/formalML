import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { dkwEnvelope, ksDistance, mulberry32, gaussianFrom } from './shared/generalization-bounds';

// =============================================================================
// GlivenkoCantelliExplorer — §4.  Empirical CDF converging to the population CDF
// across three distributions (Uniform, Normal, Cauchy); KS supremum highlighted.
// Live sampling; commit-on-release for the heavy n slider.
// =============================================================================

const HEIGHT = 340;
type DistName = 'uniform' | 'normal' | 'cauchy';

const dists: Record<DistName, {
  label: string;
  cdf: (t: number) => number;
  sample: (rng: () => number) => number;
  domain: [number, number];
}> = {
  uniform: {
    label: 'Uniform[0, 1]',
    cdf: (t) => Math.min(1, Math.max(0, t)),
    sample: (rng) => rng(),
    domain: [-0.1, 1.1],
  },
  normal: {
    label: 'Standard Normal',
    cdf: (t) => 0.5 * (1 + erf(t / Math.SQRT2)),
    sample: (rng) => gaussianFrom(rng)(),
    domain: [-4, 4],
  },
  cauchy: {
    label: 'Standard Cauchy',
    cdf: (t) => 0.5 + Math.atan(t) / Math.PI,
    sample: (rng) => Math.tan(Math.PI * (rng() - 0.5)),
    domain: [-8, 8],
  },
};

// Abramowitz–Stegun approximation of erf
function erf(x: number): number {
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
  const a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1.0 / (1.0 + p * ax);
  const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-ax * ax);
  return sign * y;
}

export default function GlivenkoCantelliExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [displayN, setDisplayN] = useState(100);
  const [committedN, setCommittedN] = useState(100);
  const [distName, setDistName] = useState<DistName>('uniform');
  const [seed, setSeed] = useState(20260511);
  const [showDkw, setShowDkw] = useState(true);

  const data = useMemo(() => {
    const dist = dists[distName];
    const rng = mulberry32(seed);
    const samples = new Float64Array(committedN);
    for (let i = 0; i < committedN; i++) samples[i] = dist.sample(rng);
    samples.sort();
    const ks = ksDistance(samples, dist.cdf);
    // Find the t* achieving the KS supremum (for the visual marker)
    let tStar = samples[0];
    let supSoFar = 0;
    for (let i = 0; i < samples.length; i++) {
      const F = dist.cdf(samples[i]);
      const up = Math.abs((i + 1) / samples.length - F);
      const lo = Math.abs(F - i / samples.length);
      const m = Math.max(up, lo);
      if (m > supSoFar) { supSoFar = m; tStar = samples[i]; }
    }
    const envelope = dkwEnvelope(committedN, 0.05);
    return { samples, ks, tStar, envelope, dist };
  }, [committedN, distName, seed]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (containerWidth <= 0) return;
      const margin = { top: 32, right: 16, bottom: 40, left: 50 };
      const w = containerWidth - margin.left - margin.right;
      const h = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const [tMin, tMax] = data.dist.domain;
      const x = d3.scaleLinear().domain([tMin, tMax]).range([0, w]);
      const y = d3.scaleLinear().domain([0, 1]).range([h, 0]);

      g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(6));
      g.append('g').call(d3.axisLeft(y).ticks(5));
      g.append('text').attr('x', w / 2).attr('y', h + 32).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('t');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -h / 2).attr('y', -36)
        .attr('text-anchor', 'middle').style('font-size', '11px').style('fill', 'var(--color-text-secondary)')
        .text('CDF');
      g.append('text').attr('x', w / 2).attr('y', -12).attr('text-anchor', 'middle')
        .style('font-size', '12px').style('font-weight', '600').style('fill', 'var(--color-text)')
        .text(`${data.dist.label} — KS distance ${data.ks.toFixed(4)} (DKW envelope ${data.envelope.toFixed(4)})`);

      // Population CDF
      const cdfPts = d3.range(tMin, tMax, (tMax - tMin) / 240).map((t) => ({ t, f: data.dist.cdf(t) }));
      const cdfLine = d3.line<{ t: number; f: number }>().x((d) => x(d.t)).y((d) => y(d.f));
      g.append('path').datum(cdfPts).attr('fill', 'none').style('stroke', '#0f172a')
        .style('stroke-width', 2).attr('d', cdfLine);

      // Empirical CDF as a step function (in the sample's [-domain] range)
      const samples = data.samples;
      const n = samples.length;
      let prevX = x(tMin);
      let prevY = y(0);
      for (let i = 0; i < n; i++) {
        const xi = x(samples[i]);
        const yi = y((i + 1) / n);
        g.append('line').attr('x1', prevX).attr('x2', xi).attr('y1', prevY).attr('y2', prevY)
          .style('stroke', '#3b82f6').style('stroke-width', 1.5);
        g.append('line').attr('x1', xi).attr('x2', xi).attr('y1', prevY).attr('y2', yi)
          .style('stroke', '#3b82f6').style('stroke-width', 1.5);
        prevX = xi; prevY = yi;
      }
      g.append('line').attr('x1', prevX).attr('x2', x(tMax)).attr('y1', prevY).attr('y2', prevY)
        .style('stroke', '#3b82f6').style('stroke-width', 1.5);

      // KS supremum marker
      const tStarX = x(data.tStar);
      g.append('line').attr('x1', tStarX).attr('x2', tStarX).attr('y1', 0).attr('y2', h)
        .style('stroke', '#dc2626').style('stroke-dasharray', '4 3').style('stroke-width', 1.5);
      g.append('text').attr('x', tStarX + 6).attr('y', 16).style('font-size', '10px').style('fill', '#dc2626')
        .text(`KS sup at t* = ${data.tStar.toFixed(3)}`);

      // DKW envelope around population CDF (optional)
      if (showDkw) {
        const env = data.envelope;
        const upper = d3.line<{ t: number; f: number }>()
          .x((d) => x(d.t)).y((d) => y(Math.min(1, d.f + env)));
        const lower = d3.line<{ t: number; f: number }>()
          .x((d) => x(d.t)).y((d) => y(Math.max(0, d.f - env)));
        g.append('path').datum(cdfPts).attr('fill', 'none').style('stroke', '#9ca3af')
          .style('stroke-dasharray', '3 3').style('stroke-width', 1).attr('d', upper);
        g.append('path').datum(cdfPts).attr('fill', 'none').style('stroke', '#9ca3af')
          .style('stroke-dasharray', '3 3').style('stroke-width', 1).attr('d', lower);
      }
    },
    [data, containerWidth, showDkw],
  );

  return (
    <figure ref={containerRef} className="my-8 not-prose">
      <svg ref={svgRef} width={containerWidth || 720} height={HEIGHT} role="img" aria-label="Empirical CDF converging to the population CDF; KS supremum highlighted." />
      <figcaption style={{ marginTop: 12, display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'center' }}>
        <label style={{ fontSize: 12, color: 'var(--color-text-secondary)' }}>
          n:
          <input
            type="range" min={5} max={2000} step={5} value={displayN}
            onChange={(e) => setDisplayN(parseInt(e.target.value, 10))}
            onMouseUp={() => setCommittedN(displayN)}
            onTouchEnd={() => setCommittedN(displayN)}
            onKeyUp={() => setCommittedN(displayN)}
            style={{ marginLeft: 8, verticalAlign: 'middle' }}
            aria-label="sample size n"
          />
          <span style={{ marginLeft: 6 }}>{displayN}</span>
        </label>
        <select
          value={distName}
          onChange={(e) => setDistName(e.target.value as DistName)}
          style={{
            fontSize: 12, padding: '2px 6px', borderRadius: 6,
            border: '1px solid var(--color-border)',
            background: 'var(--color-surface)', color: 'var(--color-text)',
          }}
          aria-label="distribution"
        >
          <option value="uniform">Uniform[0, 1]</option>
          <option value="normal">Normal</option>
          <option value="cauchy">Cauchy</option>
        </select>
        <label style={{ fontSize: 12, color: 'var(--color-text-secondary)' }}>
          <input
            type="checkbox" checked={showDkw}
            onChange={(e) => setShowDkw(e.target.checked)}
            style={{ marginRight: 4 }}
          /> show DKW envelope (δ=0.05)
        </label>
        <button
          type="button"
          onClick={() => setSeed((s) => s + 1)}
          style={{
            fontSize: 12, padding: '4px 10px', borderRadius: 6,
            border: '1px solid var(--color-border)',
            background: 'var(--color-surface)', color: 'var(--color-text)',
            cursor: 'pointer',
          }}
        >
          re-sample
        </button>
      </figcaption>
    </figure>
  );
}
