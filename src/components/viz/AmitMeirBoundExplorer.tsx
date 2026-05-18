// =============================================================================
// AmitMeirBoundExplorer.tsx
//
// §7.5 Numerical visualization of the Amit-Meir meta PAC-Bayes bound (7.2):
//
//   R_meta(Q) ≤ R̂_meta(Q)
//             + √((KL(Q || P_0) + log(4T/δ)) / 2(T-1))      (across-task term)
//             + 𝔼_P [1/T Σ_k √((KL(Q_T^P || P) + log(4Tn/δ)) / 2(n-1))]   (within-task term)
//
// Two panels:
//   (a) total bound vs T (log-x), one curve per slider-selected n
//   (b) across-task vs within-task term decomposition at the slider-selected n
//
// Sliders (all live, since computation is closed-form O(grid size)):
//   - KL_meta ∈ [0.5, 20]
//   - KL_within ∈ [0.1, 10]
//   - n (within-task sample count, log-spaced selector)
//   - empirical meta-risk R̂_meta ∈ [0, 0.5]
//   - δ ∈ [0.01, 0.20]
//
// Static fallback: /images/topics/meta-learning/12_pac_bayes_meta_bound.png
// =============================================================================

import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import { amitMeirBound, META_PALETTE } from './shared/meta-learning';

const SM_BREAKPOINT = 640;
const T_GRID = [5, 10, 25, 50, 100, 250, 500, 1000];
const N_PRESETS = [5, 25, 100];
const N_PRESET_LABELS = ['5', '25', '100'];

export default function AmitMeirBoundExplorer(): React.JSX.Element {
  const [klMeta, setKlMeta] = useState(5.0);
  const [klWithin, setKlWithin] = useState(2.0);
  const [nIdx, setNIdx] = useState(1); // index into N_PRESETS for the decomposition panel
  const [rHat, setRHat] = useState(0.10);
  const [delta, setDelta] = useState(0.05);

  const nFixed = N_PRESETS[nIdx];

  // Closed-form curves: total bound vs T for each n in N_PRESETS;
  // across vs within at nFixed
  const data = useMemo(() => {
    const totalByN = N_PRESETS.map((n) =>
      T_GRID.map((T) => amitMeirBound(T, n, klMeta, klWithin, delta, rHat).total),
    );
    const acrossAtFixed = T_GRID.map(
      (T) => amitMeirBound(T, nFixed, klMeta, klWithin, delta, rHat).across,
    );
    const withinAtFixed = T_GRID.map(
      (T) => amitMeirBound(T, nFixed, klMeta, klWithin, delta, rHat).within,
    );
    return { totalByN, acrossAtFixed, withinAtFixed };
  }, [klMeta, klWithin, nFixed, rHat, delta]);

  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const isMobile = (containerWidth || 800) < SM_BREAKPOINT;
  const panelWidth = isMobile ? containerWidth || 320 : (containerWidth || 800) / 2 - 12;
  const panelHeight = 280;

  // --- Panel A: total bound vs T --------------------------------------------
  const leftRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelWidth <= 0) return;
      const margin = { top: 18, right: 16, bottom: 40, left: 52 };
      const innerW = panelWidth - margin.left - margin.right;
      const innerH = panelHeight - margin.top - margin.bottom;
      if (innerW <= 0 || innerH <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const xScale = d3.scaleLog().domain([5, 1000]).range([0, innerW]);
      const allValues = data.totalByN.flat();
      const yMax = Math.max(rHat * 1.1, Math.max(...allValues) * 1.05);
      const yScale = d3.scaleLinear().domain([0, yMax]).range([innerH, 0]);

      g.append('g')
        .attr('transform', `translate(0, ${innerH})`)
        .call(d3.axisBottom(xScale).tickValues([5, 10, 25, 100, 1000]).tickFormat(d3.format('d')))
        .selectAll('text')
        .style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(yScale).ticks(5)).selectAll('text').style('fill', 'var(--color-text)');

      // axis labels
      g.append('text')
        .attr('x', innerW / 2)
        .attr('y', innerH + 32)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '11px')
        .text('number of tasks T');
      g.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', -innerH / 2)
        .attr('y', -38)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '11px')
        .text('bound on R_meta');

      // r_hat reference line
      g.append('line')
        .attr('x1', 0)
        .attr('x2', innerW)
        .attr('y1', yScale(rHat))
        .attr('y2', yScale(rHat))
        .style('stroke', 'var(--color-text-secondary)')
        .style('stroke-dasharray', '4 4')
        .style('opacity', 0.6);

      const line = d3
        .line<number>()
        .x((_d, i) => xScale(T_GRID[i]))
        .y((d) => yScale(d));

      data.totalByN.forEach((curve, idx) => {
        g.append('path')
          .datum(curve)
          .attr('d', line)
          .style('fill', 'none')
          .style('stroke', META_PALETTE[idx])
          .style('stroke-width', idx === nIdx ? 2.4 : 1.6)
          .style('opacity', idx === nIdx ? 1.0 : 0.55);
        g.selectAll(`.dot-${idx}`)
          .data(curve)
          .enter()
          .append('circle')
          .attr('cx', (_d, i) => xScale(T_GRID[i]))
          .attr('cy', (d) => yScale(d))
          .attr('r', idx === nIdx ? 3.2 : 2.4)
          .style('fill', META_PALETTE[idx]);
      });

      // Legend
      const legendG = g.append('g').attr('transform', `translate(${innerW - 90}, 8)`);
      N_PRESETS.forEach((n, idx) => {
        const row = legendG.append('g').attr('transform', `translate(0, ${idx * 16})`);
        row.append('rect').attr('width', 12).attr('height', 3).attr('y', 5).style('fill', META_PALETTE[idx]);
        row
          .append('text')
          .attr('x', 16)
          .attr('y', 8)
          .style('fill', 'var(--color-text)')
          .style('font-size', '10px')
          .text(`n = ${n}`);
      });
    },
    [data, panelWidth, nIdx, rHat],
  );

  // --- Panel B: across vs within decomposition at nFixed --------------------
  const rightRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelWidth <= 0) return;
      const margin = { top: 18, right: 16, bottom: 40, left: 52 };
      const innerW = panelWidth - margin.left - margin.right;
      const innerH = panelHeight - margin.top - margin.bottom;
      if (innerW <= 0 || innerH <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const xScale = d3.scaleLog().domain([5, 1000]).range([0, innerW]);
      const yMax = Math.max(...data.acrossAtFixed, ...data.withinAtFixed) * 1.1;
      const yScale = d3.scaleLinear().domain([0, yMax]).range([innerH, 0]);

      g.append('g')
        .attr('transform', `translate(0, ${innerH})`)
        .call(d3.axisBottom(xScale).tickValues([5, 10, 25, 100, 1000]).tickFormat(d3.format('d')))
        .selectAll('text')
        .style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(yScale).ticks(5)).selectAll('text').style('fill', 'var(--color-text)');

      g.append('text')
        .attr('x', innerW / 2)
        .attr('y', innerH + 32)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '11px')
        .text('number of tasks T');
      g.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', -innerH / 2)
        .attr('y', -38)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '11px')
        .text('bound term magnitude');

      const line = d3
        .line<number>()
        .x((_d, i) => xScale(T_GRID[i]))
        .y((d) => yScale(d));

      const acrossColor = META_PALETTE[3];
      const withinColor = META_PALETTE[4];

      g.append('path')
        .datum(data.acrossAtFixed)
        .attr('d', line)
        .style('fill', 'none')
        .style('stroke', acrossColor)
        .style('stroke-width', 2);
      g.append('path')
        .datum(data.withinAtFixed)
        .attr('d', line)
        .style('fill', 'none')
        .style('stroke', withinColor)
        .style('stroke-width', 2);

      g.selectAll('.acr')
        .data(data.acrossAtFixed)
        .enter()
        .append('rect')
        .attr('x', (_d, i) => xScale(T_GRID[i]) - 3)
        .attr('y', (d) => yScale(d) - 3)
        .attr('width', 6)
        .attr('height', 6)
        .style('fill', acrossColor);
      g.selectAll('.wth')
        .data(data.withinAtFixed)
        .enter()
        .append('polygon')
        .attr('points', (d, i) => {
          const x = xScale(T_GRID[i]);
          const y = yScale(d);
          return `${x},${y - 4} ${x - 4},${y + 4} ${x + 4},${y + 4}`;
        })
        .style('fill', withinColor);

      // Legend
      const legendG = g.append('g').attr('transform', `translate(${innerW - 130}, 8)`);
      legendG.append('rect').attr('width', 12).attr('height', 3).attr('y', 5).style('fill', acrossColor);
      legendG
        .append('text')
        .attr('x', 16)
        .attr('y', 8)
        .style('fill', 'var(--color-text)')
        .style('font-size', '10px')
        .text('across-task term');
      legendG.append('rect').attr('width', 12).attr('height', 3).attr('y', 21).style('fill', withinColor);
      legendG
        .append('text')
        .attr('x', 16)
        .attr('y', 24)
        .style('fill', 'var(--color-text)')
        .style('font-size', '10px')
        .text('within-task term');
    },
    [data, panelWidth, nFixed],
  );

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr 1fr',
          gap: 12,
          marginBottom: 10,
          fontSize: 11,
        }}
      >
        <label>
          KL(Q || P₀) (meta): {klMeta.toFixed(2)}
          <input
            type="range"
            min={0.5}
            max={20}
            step={0.5}
            value={klMeta}
            onChange={(e) => setKlMeta(Number(e.target.value))}
            aria-label="meta KL"
            style={{ width: '100%' }}
          />
        </label>
        <label>
          KL(Q_T || P) (within): {klWithin.toFixed(2)}
          <input
            type="range"
            min={0.1}
            max={10}
            step={0.1}
            value={klWithin}
            onChange={(e) => setKlWithin(Number(e.target.value))}
            aria-label="within KL"
            style={{ width: '100%' }}
          />
        </label>
        <label>
          R̂_meta: {rHat.toFixed(2)}
          <input
            type="range"
            min={0}
            max={0.5}
            step={0.01}
            value={rHat}
            onChange={(e) => setRHat(Number(e.target.value))}
            aria-label="empirical risk"
            style={{ width: '100%' }}
          />
        </label>
        <label>
          δ (confidence): {delta.toFixed(2)}
          <input
            type="range"
            min={0.01}
            max={0.2}
            step={0.01}
            value={delta}
            onChange={(e) => setDelta(Number(e.target.value))}
            aria-label="delta"
            style={{ width: '100%' }}
          />
        </label>
        <label style={{ gridColumn: 'span 2' }}>
          n (right panel): <strong>{N_PRESET_LABELS[nIdx]}</strong>
          <input
            type="range"
            min={0}
            max={N_PRESETS.length - 1}
            step={1}
            value={nIdx}
            onChange={(e) => setNIdx(Number(e.target.value))}
            aria-label="n preset"
            style={{ width: '100%' }}
          />
        </label>
      </div>
      <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: 16 }}>
        <svg ref={leftRef} width={panelWidth} height={panelHeight} role="img" aria-label="total bound vs T" />
        <svg ref={rightRef} width={panelWidth} height={panelHeight} role="img" aria-label="across vs within decomposition" />
      </div>
    </div>
  );
}
