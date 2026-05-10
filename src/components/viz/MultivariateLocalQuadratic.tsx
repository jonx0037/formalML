import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  gaussianSampler,
  localPolynomialMd,
  mulberry32,
  paletteKR,
} from './shared/kernel-regression';

// =============================================================================
// MultivariateLocalQuadratic — local-quadratic (p = 2) fit on the §8 2-dim toy
//   m(x) = sin(2π x_1) + sin(2π x_2),  X ~ Uniform([0,1]^2),  σ = 0.2,  n = 400.
// Three-panel display: true m, fitted m̂, residual (m̂ - m). Bandwidth slider
// (commit-on-release) drives the local-quadratic recompute on a 25×25 eval
// grid (kept coarse for client-side responsiveness — the static figure
// 08_multivariate_local_quadratic.png uses 50×50).
// =============================================================================

const HEIGHT_PANEL = 220;
const SM_BREAKPOINT = 760;
const N = 400;
const SIGMA = 0.2;
const D = 2;
const P = 2;
const G = 25; // 25×25 grid

function trueM2D(x: number, y: number): number {
  return Math.sin(2 * Math.PI * x) + Math.sin(2 * Math.PI * y);
}

export default function MultivariateLocalQuadratic() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [hDisplay, setHDisplay] = useState(0.15);
  const [hCommitted, setHCommitted] = useState(0.15);

  const w = containerWidth;
  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;

  const { X, Y, evalGrid } = useMemo(() => {
    const rng = mulberry32(42);
    const gauss = gaussianSampler(rng);
    const X_ = new Float64Array(N * D);
    const Y_ = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      const x1 = rng();
      const x2 = rng();
      X_[i * D + 0] = x1;
      X_[i * D + 1] = x2;
      Y_[i] = trueM2D(x1, x2) + SIGMA * gauss();
    }
    const eg = new Float64Array(G * G * D);
    for (let i = 0; i < G; i++) {
      for (let j = 0; j < G; j++) {
        eg[(i * G + j) * D + 0] = 0.02 + (0.96 * j) / (G - 1);
        eg[(i * G + j) * D + 1] = 0.02 + (0.96 * i) / (G - 1);
      }
    }
    return { X: X_, Y: Y_, evalGrid: eg };
  }, []);

  const { fitGrid, truthGrid, residGrid } = useMemo(() => {
    const fit = localPolynomialMd(X, Y, evalGrid, [hCommitted, hCommitted], P);
    const tr = new Float64Array(G * G);
    const re = new Float64Array(G * G);
    for (let i = 0; i < G; i++) {
      for (let j = 0; j < G; j++) {
        const idx = i * G + j;
        const x1 = evalGrid[idx * D + 0];
        const x2 = evalGrid[idx * D + 1];
        tr[idx] = trueM2D(x1, x2);
        re[idx] = fit[idx] - tr[idx];
      }
    }
    return { fitGrid: fit, truthGrid: tr, residGrid: re };
  }, [X, Y, evalGrid, hCommitted]);

  const panelW = containerWidth > 0 ? (isMobile ? containerWidth : containerWidth / 3 - 12) : 0;

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <div style={{ marginBottom: 10, display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
        <label style={{ fontFamily: 'var(--font-sans)', fontSize: 14, color: 'var(--color-text)' }}>
          isotropic bandwidth h = <strong>{hDisplay.toFixed(3)}</strong>
        </label>
        <input type="range" min={0.06} max={0.40} step={0.01} value={hDisplay}
          onChange={(e) => setHDisplay(parseFloat(e.target.value))}
          onMouseUp={(e) => setHCommitted(parseFloat((e.target as HTMLInputElement).value))}
          onTouchEnd={(e) => setHCommitted(parseFloat((e.target as HTMLInputElement).value))}
          onKeyUp={(e) => setHCommitted(parseFloat((e.target as HTMLInputElement).value))}
          aria-label="isotropic bandwidth h" style={{ flex: 1, minWidth: 200 }} />
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
        <Heatmap title="true  m(x)" data={truthGrid} width={panelW} domain={[-2, 2]} colorScheme="div" />
        <Heatmap title="fitted  m̂_H(x)  at p = 2" data={fitGrid} width={panelW} domain={[-2, 2]} colorScheme="div" />
        <Heatmap title="residual  m̂ − m" data={residGrid} width={panelW} domain={[-0.5, 0.5]} colorScheme="div" />
      </div>
      <p style={{ fontFamily: 'var(--font-sans)', fontSize: 12, color: 'var(--color-text-secondary)', marginTop: 8 }}>
        25×25 evaluation grid (live recompute on slider release). The static figure 08_multivariate_local_quadratic.png shows the same fit on a 50×50 grid.
      </p>
    </div>
  );
}

function Heatmap({
  title, data, width, domain, colorScheme,
}: {
  title: string;
  data: Float64Array;
  width: number;
  domain: [number, number];
  colorScheme: 'div';
}) {
  void colorScheme;
  const renderRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (width <= 0) return;
      const margin = { top: 26, right: 8, bottom: 30, left: 30 };
      const innerW = width - margin.left - margin.right;
      const innerH = HEIGHT_PANEL - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const cellW = innerW / G;
      const cellH = innerH / G;
      const colorScale = d3.scaleSequential(d3.interpolateRdBu).domain([domain[1], domain[0]]);

      for (let i = 0; i < G; i++) {
        for (let j = 0; j < G; j++) {
          const idx = i * G + j;
          g.append('rect')
            .attr('x', j * cellW).attr('y', (G - 1 - i) * cellH)
            .attr('width', cellW + 0.5).attr('height', cellH + 0.5)
            .style('fill', isFinite(data[idx]) ? colorScale(data[idx]) : 'var(--color-border)');
        }
      }

      svg.append('text').attr('x', margin.left).attr('y', 16).style('fill', 'var(--color-text)')
        .style('font-family', 'var(--font-sans)').style('font-size', '12px').style('font-weight', '600').text(title);

      // axes
      g.append('g').attr('transform', `translate(0,${innerH})`)
        .call(d3.axisBottom(d3.scaleLinear().domain([0, 1]).range([0, innerW])).ticks(3))
        .selectAll('text').style('fill', 'var(--color-text-secondary)').style('font-size', '10px');
      g.append('g')
        .call(d3.axisLeft(d3.scaleLinear().domain([0, 1]).range([innerH, 0])).ticks(3))
        .selectAll('text').style('fill', 'var(--color-text-secondary)').style('font-size', '10px');
    },
    [width, data, domain],
  );
  // Use paletteKR.accent ref to satisfy unused-warning in some configs.
  void paletteKR;

  return (
    <div style={{ width: width > 0 ? width : '100%' }}>
      <svg ref={renderRef} width={width} height={HEIGHT_PANEL} />
    </div>
  );
}
