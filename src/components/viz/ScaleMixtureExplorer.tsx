import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  bayesianLassoShrinkageMarginal,
  horseshoeMarginalBound,
  horseshoeShrinkageMarginal,
  mulberry32,
  ridgeShrinkageConstant,
} from './shared/bayesian-ml';

// =============================================================================
// ScaleMixtureExplorer — §2 Polson-Scott marginal-density picker.
//
// Five local-scale priors p(λ): half-Cauchy, half-Normal, half-t_3, exponential
// (on λ²), delta (ridge). The marginal p(β) = ∫ N(β; 0, λ²τ²) p(λ) dλ is shown
// on a linear scale near zero (panel a) and a log scale across the tails
// (panel b). Only the half-Cauchy (horseshoe) satisfies BOTH Polson-Scott
// criteria: mass at zero (logarithmic singularity) AND polynomial tail decay.
//
// Posterior shrinkage E[κ | y] is shown for the closed-form cases
// (ridge, Bayesian LASSO, horseshoe).
// =============================================================================

const PANEL_HEIGHT = 320;
const MARGIN = { top: 24, right: 22, bottom: 50, left: 60 };
const N_GRID = 220;

type ScaleKind = 'ridge' | 'bayesian-lasso' | 'horseshoe' | 'half-normal' | 'half-t3';

const KINDS: { key: ScaleKind; label: string; color: string }[] = [
  { key: 'ridge', label: 'Ridge (λ ≡ 1)', color: '#7f7f7f' },
  { key: 'bayesian-lasso', label: 'Bayesian LASSO (λ² ~ Exp)', color: '#7b3c10' },
  { key: 'horseshoe', label: 'Horseshoe (λ ~ HalfCauchy)', color: '#1f4e79' },
  { key: 'half-normal', label: 'Half-Normal local scale', color: '#2e7baa' },
  { key: 'half-t3', label: 'Half-t_3 local scale', color: '#c0504d' },
];

function gaussianPair01(rng: () => number): [number, number] {
  const u1 = Math.max(rng(), 1e-12);
  const u2 = rng();
  const r = Math.sqrt(-2 * Math.log(u1));
  return [r * Math.cos(2 * Math.PI * u2), r * Math.sin(2 * Math.PI * u2)];
}

function sampleLambda(kind: ScaleKind, rng: () => number): number {
  switch (kind) {
    case 'ridge':
      return 1;
    case 'bayesian-lasso':
      return Math.sqrt(-Math.log(Math.max(rng(), 1e-12)));
    case 'horseshoe':
      return Math.tan((Math.PI / 2) * rng());
    case 'half-normal':
      return Math.abs(gaussianPair01(rng)[0]);
    case 'half-t3': {
      let chi2 = 0;
      for (let k = 0; k < 3; k++) {
        const [z] = gaussianPair01(rng);
        chi2 += z * z;
      }
      const [z] = gaussianPair01(rng);
      return Math.abs(z) / Math.sqrt(chi2 / 3);
    }
  }
}

function kdeFromSamples(samples: number[], grid: number[], bandwidth = 0.04): number[] {
  const n = samples.length;
  const out = new Array(grid.length).fill(0);
  const norm = 1 / (n * bandwidth * Math.sqrt(2 * Math.PI));
  for (let i = 0; i < grid.length; i++) {
    const x = grid[i];
    let acc = 0;
    for (let j = 0; j < n; j++) {
      const u = (x - samples[j]) / bandwidth;
      acc += Math.exp(-0.5 * u * u);
    }
    out[i] = acc * norm;
  }
  return out;
}

export default function ScaleMixtureExplorer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [tau, setTau] = useState(1.0);
  const [kind, setKind] = useState<ScaleKind>('horseshoe');

  const data = useMemo(() => {
    const betaGrid = new Array<number>(N_GRID);
    for (let i = 0; i < N_GRID; i++) {
      betaGrid[i] = -5 + (10 * i) / (N_GRID - 1);
    }

    let priorDensity: number[];
    if (kind === 'ridge') {
      priorDensity = betaGrid.map(
        (b) =>
          Math.exp(-0.5 * (b / tau) * (b / tau)) /
          (tau * Math.sqrt(2 * Math.PI)),
      );
    } else if (kind === 'bayesian-lasso') {
      const scale = tau / Math.sqrt(2);
      priorDensity = betaGrid.map(
        (b) => Math.exp(-Math.abs(b) / scale) / (2 * scale),
      );
    } else if (kind === 'horseshoe') {
      // Geometric mean of Carvalho-Polson-Scott Eq. 7 bounds.
      priorDensity = betaGrid.map((b) => {
        const lower = horseshoeMarginalBound(b, tau, true);
        const upper = horseshoeMarginalBound(b, tau, false);
        return Math.sqrt(Math.max(lower, 1e-12) * Math.max(upper, 1e-12));
      });
    } else {
      // KDE for half-Normal / half-t_3 (no closed-form marginal).
      const rng = mulberry32(20260509);
      const samples: number[] = new Array(8000);
      for (let s = 0; s < samples.length; s++) {
        const lam = sampleLambda(kind, rng);
        const [z] = gaussianPair01(rng);
        samples[s] = lam * tau * z;
      }
      priorDensity = kdeFromSamples(samples, betaGrid, 0.06 + 0.02 * tau);
    }

    const yGrid = new Array<number>(60);
    for (let i = 0; i < yGrid.length; i++) {
      yGrid[i] = 0.05 + (8.0 * i) / (yGrid.length - 1);
    }
    let posteriorShrinkage: number[];
    if (kind === 'ridge') {
      const k = ridgeShrinkageConstant(tau);
      posteriorShrinkage = yGrid.map(() => k);
    } else if (kind === 'bayesian-lasso') {
      posteriorShrinkage = yGrid.map((y) => bayesianLassoShrinkageMarginal(y, tau));
    } else if (kind === 'horseshoe') {
      posteriorShrinkage = yGrid.map((y) => horseshoeShrinkageMarginal(y, tau));
    } else {
      // For half-Normal / half-t_3 we do not expose a closed-form posterior
      // shrinkage; render as a flat NaN trace and skip.
      posteriorShrinkage = yGrid.map(() => Number.NaN);
    }

    return { betaGrid, priorDensity, yGrid, posteriorShrinkage };
  }, [tau, kind]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const w = width || 760;
      const h = PANEL_HEIGHT;
      svg.attr('width', w).attr('height', h);
      const innerW = (w - MARGIN.left - MARGIN.right - 24) / 2;
      const innerH = h - MARGIN.top - MARGIN.bottom;

      // Panel (a): linear-scale mass at zero
      const gA = svg
        .append('g')
        .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);
      const xA = d3.scaleLinear().domain([-2, 2]).range([0, innerW]);
      const yMaxA = Math.max(d3.max(data.priorDensity) ?? 1, 1.5);
      const yA = d3.scaleLinear().domain([0, Math.min(yMaxA, 4)]).range([innerH, 0]).clamp(true);
      gA.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xA).ticks(5));
      gA.append('g').call(d3.axisLeft(yA).ticks(5));
      gA.append('text')
        .attr('x', innerW / 2)
        .attr('y', innerH + 36)
        .attr('text-anchor', 'middle')
        .style('font-size', '11.5px')
        .text('β');
      gA.append('text')
        .attr('x', -innerH / 2)
        .attr('y', -42)
        .attr('text-anchor', 'middle')
        .attr('transform', 'rotate(-90)')
        .style('font-size', '11.5px')
        .text('marginal density p(β)');
      gA.append('text')
        .attr('x', 0)
        .attr('y', -8)
        .style('font-size', '11px')
        .style('font-weight', '600')
        .text('(a) mass-at-zero');

      const lineA = d3
        .line<number>()
        .x((_, i) => xA(data.betaGrid[i]))
        .y((d) => yA(d))
        .curve(d3.curveMonotoneX);
      const color = KINDS.find((k) => k.key === kind)!.color;
      gA.append('path')
        .datum(data.priorDensity)
        .attr('fill', 'none')
        .attr('stroke', color)
        .attr('stroke-width', 2.4)
        .attr('d', lineA);

      // Panel (b): log-scale tails
      const gB = svg
        .append('g')
        .attr('transform', `translate(${MARGIN.left + innerW + 60},${MARGIN.top})`);
      const xB = d3.scaleLinear().domain([-5, 5]).range([0, innerW]);
      const yB = d3
        .scaleLog()
        .domain([1e-5, Math.max(d3.max(data.priorDensity) ?? 1, 1)])
        .range([innerH, 0])
        .clamp(true);
      gB.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xB).ticks(5));
      gB.append('g').call(d3.axisLeft(yB).ticks(5, '~g'));
      gB.append('text')
        .attr('x', innerW / 2)
        .attr('y', innerH + 36)
        .attr('text-anchor', 'middle')
        .style('font-size', '11.5px')
        .text('β');
      gB.append('text')
        .attr('x', -innerH / 2)
        .attr('y', -42)
        .attr('text-anchor', 'middle')
        .attr('transform', 'rotate(-90)')
        .style('font-size', '11.5px')
        .text('p(β) — log scale');
      gB.append('text')
        .attr('x', 0)
        .attr('y', -8)
        .style('font-size', '11px')
        .style('font-weight', '600')
        .text('(b) heavy-tail criterion');
      const lineB = d3
        .line<number>()
        .x((_, i) => xB(data.betaGrid[i]))
        .y((d) => yB(Math.max(d, 1e-12)))
        .curve(d3.curveMonotoneX);
      gB.append('path')
        .datum(data.priorDensity)
        .attr('fill', 'none')
        .attr('stroke', color)
        .attr('stroke-width', 2.4)
        .attr('d', lineB);
    },
    [data, kind, width],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', maxWidth: 760 }}>
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: 18,
          marginBottom: 8,
          alignItems: 'center',
          fontSize: 13,
        }}
      >
        <label>
          Local-scale prior:{' '}
          <select value={kind} onChange={(e) => setKind(e.target.value as ScaleKind)}>
            {KINDS.map((k) => (
              <option key={k.key} value={k.key}>
                {k.label}
              </option>
            ))}
          </select>
        </label>
        <label>
          τ (global scale): <span style={{ fontVariantNumeric: 'tabular-nums' }}>{tau.toFixed(2)}</span>{' '}
          <input
            type="range"
            min={0.2}
            max={3.0}
            step={0.05}
            value={tau}
            onChange={(e) => setTau(Number(e.target.value))}
            style={{ width: 140 }}
          />
        </label>
      </div>
      <svg ref={svgRef} role="img" aria-label="Polson-Scott marginal density panels" />
      <div style={{ fontSize: 11.5, marginTop: 6, color: '#555' }}>
        Polson-Scott (Theorem 1): a sparsity-inducing prior must satisfy <em>both</em> mass-at-zero
        (panel a — density diverges as β → 0) <em>and</em> heavy tails (panel b — slower than
        Gaussian decay). Only the horseshoe meets both criteria simultaneously.
      </div>
    </div>
  );
}
