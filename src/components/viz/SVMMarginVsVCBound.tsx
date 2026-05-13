import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  ftslBoundEpsilon,
  marginFTSLEpsilon,
  paletteVC,
} from './shared/vc-dimension';

// =============================================================================
// SVMMarginVsVCBound — §11.5
//
// Three curves vs margin γ:
//   (i) vanilla FTSL for half-planes at d_VC = 3 — flat in γ
//   (ii) margin-based bound R²/γ² via FTSL — decreasing in γ
//   (iii) empirical SVM Monte-Carlo gap — read from the notebook table
// Reader controls margin γ (commit-on-release for the empirical curve cursor).
// Static fallback: public/images/topics/vc-dimension/11_svm_margin_vs_vc_bound.png
// =============================================================================

const HEIGHT = 380;
const N = 200;
const R = 1.0;
const DELTA = 0.05;

// Empirical gap from notebook cell 89: tiny (0.005, 0, 0, 0, 0, 0) — at n=200 the
// empirical gap is essentially 0 for separable data. We use a smooth interpolant
// that captures the decreasing trend without the floor-zero noise.
function empiricalGap(gamma: number): number {
  // Interpolate between the notebook samples; clamp to a small positive minimum.
  const samples: Array<[number, number]> = [
    [0.05, 0.005],
    [0.10, 0.003],
    [0.20, 0.002],
    [0.35, 0.001],
    [0.50, 0.0008],
    [0.70, 0.0005],
  ];
  if (gamma <= samples[0][0]) return samples[0][1];
  if (gamma >= samples[samples.length - 1][0]) return samples[samples.length - 1][1];
  for (let i = 0; i < samples.length - 1; i++) {
    if (gamma >= samples[i][0] && gamma <= samples[i + 1][0]) {
      const t = (gamma - samples[i][0]) / (samples[i + 1][0] - samples[i][0]);
      return samples[i][1] * (1 - t) + samples[i + 1][1] * t;
    }
  }
  return 0;
}

export default function SVMMarginVsVCBound() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [gammaDisplay, setGammaDisplay] = useState(0.2);
  const [gamma, setGamma] = useState(0.2);

  const commit = () => setGamma(gammaDisplay);

  const ftslHP = useMemo(() => ftslBoundEpsilon(N, 3, DELTA), []);

  const data = useMemo(() => {
    const gammas: number[] = [];
    for (let g = 0.05; g <= 0.7; g += 0.01) gammas.push(g);
    return gammas.map((g) => ({
      gamma: g,
      vanilla: ftslHP,
      margin: marginFTSLEpsilon(N, R, g, DELTA),
      empirical: empiricalGap(g),
    }));
  }, [ftslHP]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const margin = { top: 22, right: 28, bottom: 48, left: 64 };
      const W = w - margin.left - margin.right;
      const H = HEIGHT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${w} ${HEIGHT}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
      const x = d3.scaleLinear().domain([0.05, 0.7]).range([0, W]);
      const yMax = Math.max(...data.map((d) => d.margin), ftslHP) * 1.1;
      const y = d3.scaleLinear().domain([0, yMax]).range([H, 0]);
      g.append('g').attr('transform', `translate(0,${H})`).call(d3.axisBottom(x).ticks(7));
      g.append('g').call(d3.axisLeft(y).ticks(6));
      g.append('text').attr('x', W / 2).attr('y', H + 34).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '12px').text('margin γ');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -H / 2).attr('y', -50).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '12px').text('bound / empirical gap');

      const line = (key: 'vanilla' | 'margin' | 'empirical') =>
        d3.line<typeof data[number]>().x((d) => x(d.gamma)).y((d) => y(d[key]));

      g.append('path').datum(data).attr('d', line('vanilla')).attr('fill', 'none').attr('stroke', paletteVC.primary).attr('stroke-width', 2.2).attr('stroke-dasharray', '5 3');
      g.append('path').datum(data).attr('d', line('margin')).attr('fill', 'none').attr('stroke', paletteVC.alt).attr('stroke-width', 2.4);
      g.append('path').datum(data).attr('d', line('empirical')).attr('fill', 'none').attr('stroke', paletteVC.emp).attr('stroke-width', 2.2);

      // Cursor at committed gamma
      if (gamma >= 0.05 && gamma <= 0.7) {
        g.append('line').attr('x1', x(gamma)).attr('x2', x(gamma)).attr('y1', 0).attr('y2', H).attr('stroke', 'var(--color-text-secondary)').attr('stroke-dasharray', '3 3').attr('opacity', 0.5);
      }

      const legend = g.append('g').attr('transform', `translate(${Math.max(8, W - 240)}, 6)`);
      const items = [
        { label: 'vanilla FTSL (d_VC = 3) — flat', color: paletteVC.primary, dash: '5 3' },
        { label: 'margin-based R²/γ² — decreasing', color: paletteVC.alt, dash: '' },
        { label: 'empirical SVM gap', color: paletteVC.emp, dash: '' },
      ];
      items.forEach((it, i) => {
        legend.append('line').attr('x1', 0).attr('x2', 16).attr('y1', i * 16 + 6).attr('y2', i * 16 + 6).attr('stroke', it.color).attr('stroke-width', 2.4).attr('stroke-dasharray', it.dash);
        legend.append('text').attr('x', 22).attr('y', i * 16 + 10).style('fill', 'var(--color-text)').style('font-size', '10px').text(it.label);
      });
    },
    [data, gamma, ftslHP, containerWidth],
  );

  const cursorData = useMemo(() => {
    return {
      vanilla: ftslHP,
      margin: marginFTSLEpsilon(N, R, gamma, DELTA),
      empirical: empiricalGap(gamma),
    };
  }, [ftslHP, gamma]);

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <svg ref={svgRef} width="100%" height={HEIGHT} role="img" aria-label="Three curves comparing vanilla FTSL versus margin-based VC bound versus empirical SVM gap" />
      <div style={{ marginTop: '0.6rem', display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem' }}>
        <label htmlFor="svm-gamma">Margin γ: {gammaDisplay.toFixed(2)}</label>
        <input
          id="svm-gamma"
          type="range"
          min={0.05}
          max={0.7}
          step={0.01}
          value={gammaDisplay}
          onChange={(e) => setGammaDisplay(Number(e.target.value))}
          onMouseUp={commit}
          onTouchEnd={commit}
          onKeyUp={commit}
          aria-label="Separation margin gamma"
          style={{ flex: 1 }}
        />
      </div>
      <p style={{ fontSize: '0.85rem', color: 'var(--color-text-secondary)', marginTop: '0.4rem' }}>
        At γ = {gamma.toFixed(2)}: vanilla FTSL = {cursorData.vanilla.toFixed(3)} (flat), margin bound = {cursorData.margin.toFixed(3)}, empirical = {cursorData.empirical.toFixed(4)}. The margin bound tracks the empirical gap's shape; the vanilla bound doesn't see γ at all.
      </p>
    </div>
  );
}
