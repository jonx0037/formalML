import { useMemo } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  catoniOptimizedCertificate,
  empiricalRiskOnGrid,
  ermOnGrid,
  expectedUnderQ,
  finiteClassUnionBound,
  gaussianOnGridPosterior,
  klDiscrete,
  mcAllesterCertificate,
  mulberry32,
  sampleNormalThresholdProblem,
  seegerCertificate,
  thresholdGrid,
} from './shared/pac-bayes-bounds';

// =============================================================================
// ToyBoundComparison — §10.5
// Panel A: Union vs McAllester vs Catoni-opt vs Seeger as n sweeps log-spaced,
// fixed Q = Q_narrow.
// Panel B: grouped bar chart at n = 200 across three posteriors: δ_{h_ERM},
// Q_narrow, Q_broad.  The union-bound bars do not depend on Q.
// =============================================================================

const N_BASE = 200;
const ETA = 0.05;
const MU_STAR = 0.3;
const DELTA = 0.05;
const SEED = 20260511;
const HEIGHT = 360;
const SM_BREAKPOINT = 640;
const N_GRID = 60;

export default function ToyBoundComparison() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();

  const data = useMemo(() => {
    const rng = mulberry32(SEED >>> 0);
    const { X, Y } = sampleNormalThresholdProblem(N_BASE, ETA, MU_STAR, rng);
    const grid = thresholdGrid();
    const risks = empiricalRiskOnGrid(X, Y, grid);
    const erm = ermOnGrid(grid, risks);
    const P = new Float64Array(grid.length).fill(1 / grid.length);

    const Qpoint = new Float64Array(grid.length);
    let ermIdx = 0;
    for (let i = 0; i < grid.length; i++) if (risks[i] < risks[ermIdx]) ermIdx = i;
    Qpoint[ermIdx] = 1;
    const Qnarrow = gaussianOnGridPosterior(grid, erm.tau, 0.10, P);
    const Qbroad = gaussianOnGridPosterior(grid, erm.tau, 0.50, P);

    const posteriors = [
      { name: 'δ_{h_ERM}', Q: Qpoint, kl: klDiscrete(Qpoint, P), eHat: expectedUnderQ(Qpoint, risks) },
      { name: 'Q_narrow', Q: Qnarrow, kl: klDiscrete(Qnarrow, P), eHat: expectedUnderQ(Qnarrow, risks) },
      { name: 'Q_broad', Q: Qbroad, kl: klDiscrete(Qbroad, P), eHat: expectedUnderQ(Qbroad, risks) },
    ];

    // Sweep over n for Q_narrow (Panel A)
    const nLo = Math.log10(50);
    const nHi = Math.log10(5000);
    const nList = new Float64Array(N_GRID);
    for (let i = 0; i < N_GRID; i++) {
      nList[i] = Math.pow(10, nLo + (i / (N_GRID - 1)) * (nHi - nLo));
    }
    const pNarrow = posteriors[1];
    const sweep = {
      n: nList,
      union: new Float64Array(N_GRID),
      mcAllester: new Float64Array(N_GRID),
      catoni: new Float64Array(N_GRID),
      seeger: new Float64Array(N_GRID),
    };
    for (let i = 0; i < N_GRID; i++) {
      const n = nList[i];
      sweep.union[i] = pNarrow.eHat + finiteClassUnionBound(grid.length, n, DELTA);
      sweep.mcAllester[i] = mcAllesterCertificate(pNarrow.eHat, pNarrow.kl, n, DELTA);
      sweep.catoni[i] = catoniOptimizedCertificate(pNarrow.eHat, pNarrow.kl, n, DELTA);
      sweep.seeger[i] = seegerCertificate(pNarrow.eHat, pNarrow.kl, n, DELTA);
    }

    // Bar chart at n = N_BASE (Panel B)
    const bars = posteriors.map((q) => ({
      name: q.name,
      eHat: q.eHat,
      union: q.eHat + finiteClassUnionBound(grid.length, N_BASE, DELTA),
      mcAllester: mcAllesterCertificate(q.eHat, q.kl, N_BASE, DELTA),
      catoni: catoniOptimizedCertificate(q.eHat, q.kl, N_BASE, DELTA),
      seeger: seegerCertificate(q.eHat, q.kl, N_BASE, DELTA),
    }));

    return { sweep, bars };
  }, []);

  const isMobile = (containerWidth || 800) < SM_BREAKPOINT;
  const panelWidth = isMobile ? (containerWidth || 800) : Math.max(280, (containerWidth || 800) / 2 - 8);

  return (
    <div
      ref={containerRef}
      style={{
        margin: '1.5rem 0',
        padding: '1rem',
        background: 'var(--color-surface)',
        border: '1px solid var(--color-border)',
        borderRadius: '0.5rem',
      }}
    >
      <div style={{ marginBottom: '0.5rem', fontSize: '0.78rem', color: 'var(--color-text-secondary)', fontFamily: 'var(--font-mono)' }}>
        Running example: n = {N_BASE}, δ = {DELTA}, |H| = 101 thresholds
      </div>
      <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: '0.5rem' }}>
        <SweepPanel sweep={data.sweep} width={panelWidth} />
        <BarPanel bars={data.bars} width={panelWidth} />
      </div>
    </div>
  );
}

type SweepT = {
  n: Float64Array;
  union: Float64Array;
  mcAllester: Float64Array;
  catoni: Float64Array;
  seeger: Float64Array;
};

function SweepPanel({ sweep, width }: { sweep: SweepT; width: number }) {
  const ref = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (width <= 0) return;
      const margin = { top: 26, right: 110, bottom: 44, left: 56 };
      const w = width - margin.left - margin.right;
      const h = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const x = d3.scaleLog().domain([sweep.n[0], sweep.n[sweep.n.length - 1]]).range([0, w]);
      const yMax = Math.max(
        d3.max(sweep.union) ?? 0.5,
        d3.max(sweep.mcAllester) ?? 0.5,
      );
      const y = d3.scaleLinear().domain([0, Math.min(0.6, yMax * 1.05)]).range([h, 0]);

      g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(6, '~s'));
      g.append('g').call(d3.axisLeft(y).ticks(6));
      g.append('text').attr('x', w / 2).attr('y', h + 32).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('sample size n (log)');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -h / 2).attr('y', -44).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('upper bound on E_Q[R]');

      const lineGen = (arr: Float64Array) => d3.line<number>()
        .x((_, i) => x(sweep.n[i]))
        .y((v) => y(v))(Array.from(arr));

      const series = [
        { name: 'Union bound', color: 'var(--color-text-secondary)', arr: sweep.union, dash: '4,3' },
        { name: 'McAllester', color: 'var(--color-text)', arr: sweep.mcAllester, dash: '' },
        { name: 'Catoni-opt', color: '#534AB7', arr: sweep.catoni, dash: '' },
        { name: 'Seeger', color: 'var(--color-accent)', arr: sweep.seeger, dash: '' },
      ];
      series.forEach((s) => {
        const path = g.append('path').attr('d', lineGen(s.arr))
          .style('fill', 'none').style('stroke', s.color).style('stroke-width', 2.2);
        if (s.dash) path.style('stroke-dasharray', s.dash);
      });

      const legend = g.append('g').attr('transform', `translate(${w + 6},14)`);
      series.forEach((s, idx) => {
        const yPos = idx * 14;
        const line = legend.append('line').attr('x1', 0).attr('x2', 14)
          .attr('y1', yPos).attr('y2', yPos)
          .style('stroke', s.color).style('stroke-width', 2.2);
        if (s.dash) line.style('stroke-dasharray', s.dash);
        legend.append('text').attr('x', 18).attr('y', yPos + 3)
          .style('font-size', '10px').style('fill', 'var(--color-text-secondary)')
          .text(s.name);
      });

      g.append('text').attr('x', 0).attr('y', -8)
        .style('font-size', '11px').style('font-weight', 600).style('fill', 'var(--color-text)')
        .text('Panel A — bounds vs n at Q_narrow');
    },
    [sweep, width],
  );
  return <svg ref={ref} width={width} height={HEIGHT} style={{ overflow: 'visible' }} />;
}

type BarT = {
  name: string;
  eHat: number;
  union: number;
  mcAllester: number;
  catoni: number;
  seeger: number;
};

function BarPanel({ bars, width }: { bars: BarT[]; width: number }) {
  const ref = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (width <= 0) return;
      const margin = { top: 26, right: 110, bottom: 44, left: 56 };
      const w = width - margin.left - margin.right;
      const h = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const x0 = d3.scaleBand().domain(bars.map((b) => b.name)).range([0, w]).padding(0.2);
      const subKeys = ['Union', 'McAllester', 'Catoni-opt', 'Seeger'] as const;
      const x1 = d3.scaleBand().domain(subKeys as unknown as string[]).range([0, x0.bandwidth()]).padding(0.05);
      const yMax = Math.max(...bars.flatMap((b) => [b.union, b.mcAllester, b.catoni, b.seeger]));
      const y = d3.scaleLinear().domain([0, Math.min(0.6, yMax * 1.05)]).range([h, 0]);

      g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x0));
      g.append('g').call(d3.axisLeft(y).ticks(6));
      g.append('text').attr('x', w / 2).attr('y', h + 32).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('posterior choice');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -h / 2).attr('y', -44).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('upper bound on E_Q[R]');

      const palette: Record<string, string> = {
        Union: 'var(--color-text-secondary)',
        McAllester: 'var(--color-text)',
        'Catoni-opt': '#534AB7',
        Seeger: 'var(--color-accent)',
      };

      bars.forEach((b) => {
        const offset = x0(b.name)!;
        const values: Record<string, number> = {
          Union: b.union,
          McAllester: b.mcAllester,
          'Catoni-opt': b.catoni,
          Seeger: b.seeger,
        };
        subKeys.forEach((k) => {
          g.append('rect')
            .attr('x', offset + (x1(k) ?? 0))
            .attr('y', y(values[k]))
            .attr('width', x1.bandwidth())
            .attr('height', h - y(values[k]))
            .style('fill', palette[k]);
        });
        // E_Q[R̂] floor marker
        g.append('line').attr('x1', offset).attr('x2', offset + x0.bandwidth())
          .attr('y1', y(b.eHat)).attr('y2', y(b.eHat))
          .style('stroke', '#D97706').style('stroke-dasharray', '3,2');
      });

      const legend = g.append('g').attr('transform', `translate(${w + 6},14)`);
      subKeys.forEach((k, idx) => {
        legend.append('rect').attr('x', 0).attr('y', idx * 14 - 6).attr('width', 14).attr('height', 8)
          .style('fill', palette[k]);
        legend.append('text').attr('x', 18).attr('y', idx * 14)
          .style('font-size', '10px').style('fill', 'var(--color-text-secondary)').text(k);
      });
      legend.append('line').attr('x1', 0).attr('x2', 14)
        .attr('y1', subKeys.length * 14 - 2).attr('y2', subKeys.length * 14 - 2)
        .style('stroke', '#D97706').style('stroke-dasharray', '3,2');
      legend.append('text').attr('x', 18).attr('y', subKeys.length * 14 + 1)
        .style('font-size', '10px').style('fill', 'var(--color-text-secondary)').text('E_Q[R̂] floor');

      g.append('text').attr('x', 0).attr('y', -8)
        .style('font-size', '11px').style('font-weight', 600).style('fill', 'var(--color-text)')
        .text('Panel B — four bounds at n = 200 across posteriors');
    },
    [bars, width],
  );
  return <svg ref={ref} width={width} height={HEIGHT} style={{ overflow: 'visible' }} />;
}
