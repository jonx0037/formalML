// =============================================================================
// TaskFamilyExplorer.tsx
//
// §1.4 Three task families: sinusoidal regression / 1D GP / 2D prototypical.
// Tabbed UI lets the reader switch between the three; each tab re-samples
// four tasks live on a button press.
//
// Closed-form math; uses shared/meta-learning.ts task samplers.
// Combines static Figures 02–04 into a single explorer per the brief's
// "figures 02–04 (task-family panels)" grouping.
//
// Static fallbacks (still rendered alongside in the MDX):
//   /images/topics/meta-learning/{02,03,04}_*.png
// =============================================================================

import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import {
  mulberry32,
  sampleSinusoidTask,
  sampleGpTask,
  sampleProtoNetTask,
  META_PALETTE,
  META_SEED,
} from './shared/meta-learning';

type Tab = 'sinusoid' | 'gp' | 'protonet';

export default function TaskFamilyExplorer(): React.JSX.Element {
  const [tab, setTab] = useState<Tab>('sinusoid');
  const [seedOffset, setSeedOffset] = useState(0);

  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const width = containerWidth || 800;
  const panelW = Math.max(140, (width - 36) / 4);
  const panelH = 140;
  const totalH = panelH + 32;

  // Resample on tab change or seed bump
  const tasks = useMemo(() => {
    const rng = mulberry32(META_SEED + seedOffset * 17 + (tab === 'sinusoid' ? 0 : tab === 'gp' ? 1000 : 2000));
    if (tab === 'sinusoid') {
      return Array.from({ length: 4 }, () => sampleSinusoidTask(rng, { aRange: [0.5, 5.0] }));
    }
    if (tab === 'gp') {
      // For GP we sample three functions at each of four lengthscales
      return [0.3, 0.6, 1.2, 2.5].map((ell) => ({
        ell,
        samples: Array.from({ length: 3 }, () => sampleGpTask(rng, ell)),
      }));
    }
    // protonet: 2 tasks
    return Array.from({ length: 2 }, () => sampleProtoNetTask(rng));
  }, [tab, seedOffset]);

  const drawRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (width <= 0) return;
      const cols = tab === 'protonet' ? 2 : 4;
      const wPerPanel = (width - 36) / cols;

      if (tab === 'sinusoid') {
        const sinTasks = tasks as ReturnType<typeof sampleSinusoidTask>[];
        const yExt = Math.max(...sinTasks.map((t) => Math.max(...t.yDense.map(Math.abs)))) * 1.1;
        sinTasks.forEach((task, i) => {
          const g = svg.append('g').attr('transform', `translate(${i * wPerPanel + 16}, 6)`);
          const xScale = d3.scaleLinear().domain([-5, 5]).range([0, wPerPanel - 16]);
          const yScale = d3.scaleLinear().domain([-yExt, yExt]).range([panelH - 24, 0]);
          // Axes
          g.append('g')
            .attr('transform', `translate(0, ${panelH - 24})`)
            .call(d3.axisBottom(xScale).ticks(3))
            .selectAll('text')
            .style('fill', 'var(--color-text)')
            .style('font-size', '9px');
          // True curve
          const truthLine = d3
            .line<number>()
            .x((_, j) => xScale(task.xDense[j]))
            .y((d) => yScale(d));
          g.append('path').datum(task.yDense).attr('d', truthLine).style('fill', 'none').style('stroke', 'var(--color-text)').style('stroke-width', 1.2).style('opacity', 0.7);
          // Support points
          g.selectAll('.sup')
            .data(task.xSupport)
            .enter()
            .append('circle')
            .attr('cx', (d) => xScale(d))
            .attr('cy', (_d, j) => yScale(task.ySupport[j]))
            .attr('r', 3.5)
            .style('fill', META_PALETTE[0])
            .style('stroke', 'white')
            .style('stroke-width', 0.8);
          g.append('text').attr('x', wPerPanel / 2 - 12).attr('y', -2).attr('text-anchor', 'middle').style('font-size', '9px').style('fill', 'var(--color-text-secondary)').text(`A=${task.A.toFixed(1)}, φ=${task.phi.toFixed(1)}`);
        });
      } else if (tab === 'gp') {
        const gpTasks = tasks as { ell: number; samples: { x: number[]; f: number[] }[] }[];
        const allY = gpTasks.flatMap(({ samples }) => samples.flatMap((s) => s.f));
        const yExt = Math.max(...allY.map(Math.abs)) * 1.05;
        gpTasks.forEach((tt, i) => {
          const g = svg.append('g').attr('transform', `translate(${i * wPerPanel + 16}, 6)`);
          const xScale = d3.scaleLinear().domain([-3, 3]).range([0, wPerPanel - 16]);
          const yScale = d3.scaleLinear().domain([-yExt, yExt]).range([panelH - 24, 0]);
          g.append('g')
            .attr('transform', `translate(0, ${panelH - 24})`)
            .call(d3.axisBottom(xScale).ticks(3))
            .selectAll('text')
            .style('fill', 'var(--color-text)')
            .style('font-size', '9px');
          tt.samples.forEach((s, k) => {
            const lineFn = d3
              .line<number>()
              .x((_, j) => xScale(s.x[j]))
              .y((d) => yScale(d));
            g.append('path').datum(s.f).attr('d', lineFn).style('fill', 'none').style('stroke', META_PALETTE[k]).style('stroke-width', 1.2).style('opacity', 0.8);
          });
          g.append('text').attr('x', wPerPanel / 2 - 12).attr('y', -2).attr('text-anchor', 'middle').style('font-size', '9px').style('fill', 'var(--color-text-secondary)').text(`ℓ = ${tt.ell.toFixed(1)}`);
        });
      } else {
        const protoTasks = tasks as ReturnType<typeof sampleProtoNetTask>[];
        protoTasks.forEach((task, i) => {
          const g = svg.append('g').attr('transform', `translate(${i * wPerPanel + 16}, 6)`);
          const xScale = d3.scaleLinear().domain([-4, 4]).range([0, wPerPanel - 16]);
          const yScale = d3.scaleLinear().domain([-4, 4]).range([panelH - 24, 0]);
          // Box outline
          g.append('rect').attr('width', wPerPanel - 16).attr('height', panelH - 24).style('fill', 'none').style('stroke', 'var(--color-border)');
          // Class points
          for (let pi = 0; pi < task.Xs.length; pi++) {
            const c = task.ys[pi];
            g.append('circle').attr('cx', xScale(task.Xs[pi][0])).attr('cy', yScale(task.Xs[pi][1])).attr('r', 3.5).style('fill', META_PALETTE[c]).style('stroke', 'white').style('stroke-width', 0.6);
          }
          // Class-mean stars
          task.means.forEach((m, ci) => {
            const cx = xScale(m[0]);
            const cy = yScale(m[1]);
            g.append('polygon')
              .attr('points', starPoints(cx, cy, 8, 4))
              .style('fill', META_PALETTE[ci])
              .style('stroke', 'black')
              .style('stroke-width', 0.7);
          });
          g.append('text').attr('x', wPerPanel / 2 - 12).attr('y', -2).attr('text-anchor', 'middle').style('font-size', '9px').style('fill', 'var(--color-text-secondary)').text(`task ${i + 1} (5 support / class)`);
        });
      }
    },
    [tab, tasks, width],
  );

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <div style={{ display: 'flex', gap: 8, marginBottom: 8, alignItems: 'center', fontSize: 11 }}>
        <span style={{ color: 'var(--color-text-secondary)' }}>Task family:</span>
        {(['sinusoid', 'gp', 'protonet'] as Tab[]).map((t) => (
          <button
            key={t}
            type="button"
            onClick={() => setTab(t)}
            style={{
              padding: '4px 10px',
              border: '1px solid var(--color-border)',
              borderRadius: 4,
              background: tab === t ? 'var(--color-accent)' : 'transparent',
              color: tab === t ? 'white' : 'var(--color-text)',
              cursor: 'pointer',
              fontSize: 11,
            }}
            aria-pressed={tab === t}
          >
            {t === 'sinusoid' ? 'Sinusoidal (§2)' : t === 'gp' ? 'GP (§4)' : 'ProtoNet (§5)'}
          </button>
        ))}
        <button
          type="button"
          onClick={() => setSeedOffset((s) => s + 1)}
          style={{
            marginLeft: 'auto',
            padding: '4px 10px',
            border: '1px solid var(--color-border)',
            borderRadius: 4,
            background: 'transparent',
            color: 'var(--color-text)',
            cursor: 'pointer',
            fontSize: 11,
          }}
          aria-label="resample tasks"
        >
          Resample
        </button>
      </div>
      <svg ref={drawRef} width={width} height={totalH} role="img" aria-label="task family explorer" />
    </div>
  );
}

function starPoints(cx: number, cy: number, rOuter: number, rInner: number): string {
  const pts: string[] = [];
  for (let i = 0; i < 10; i++) {
    const r = i % 2 === 0 ? rOuter : rInner;
    const a = (Math.PI * i) / 5 - Math.PI / 2;
    pts.push(`${cx + r * Math.cos(a)},${cy + r * Math.sin(a)}`);
  }
  return pts.join(' ');
}
