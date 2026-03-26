import { useState, useMemo } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { dimensionColors } from './shared/colorScales';
import {
  klDivergence,
  totalVariation,
  chiSquaredDivergence,
  hellingerDistance,
  jensenShannonDivergence,
  normalize,
} from './shared/informationTheory';

// ── Constants ────────────────────────────────────────────────────────

const PANEL_HEIGHT = 280;
const PINSKER_HEIGHT = 280;
const SM_BREAKPOINT = 640;
const MARGIN = { top: 30, right: 20, bottom: 40, left: 55 };

// ── Divergence definitions ───────────────────────────────────────────

interface DivDef {
  key: string;
  label: string;
  color: string;
  generator: (t: number) => number;
  compute: (p: number[], q: number[]) => number;
}

const DIVERGENCES: DivDef[] = [
  {
    key: 'kl',
    label: 'KL',
    color: dimensionColors[0],
    generator: (t) => (t > 0 ? t * Math.log(t) : 0),
    compute: (p, q) => klDivergence(p, q) * Math.LN2, // convert bits to nats for chart
  },
  {
    key: 'rev_kl',
    label: 'Reverse KL',
    color: '#DC2626',
    generator: (t) => (t > 0 ? -Math.log(t) : Infinity),
    compute: (p, q) => {
      let s = 0;
      for (let i = 0; i < p.length; i++) {
        if (q[i] > 0 && p[i] > 0) s += q[i] * Math.log(q[i] / p[i]);
        else if (q[i] > 0 && p[i] <= 0) return Infinity;
      }
      return s;
    },
  },
  {
    key: 'chi2',
    label: 'χ²',
    color: dimensionColors[1],
    generator: (t) => (t - 1) ** 2,
    compute: chiSquaredDivergence,
  },
  {
    key: 'hellinger',
    label: 'Hellinger²',
    color: '#D97706',
    generator: (t) => (Math.sqrt(Math.max(t, 0)) - 1) ** 2,
    compute: hellingerDistance,
  },
  {
    key: 'tv',
    label: 'TV',
    color: '#059669',
    generator: (t) => Math.abs(t - 1) / 2,
    compute: totalVariation,
  },
  {
    key: 'js',
    label: 'Jensen–Shannon',
    color: '#7C3AED',
    generator: (t) => {
      if (t <= 0) return Math.log(2);
      const m = (t + 1) / 2;
      return t * Math.log(t / m) + Math.log(1 / m);
    },
    compute: (p, q) => jensenShannonDivergence(p, q) * Math.LN2,
  },
];

// ── Seeded random for Pinsker scatter ────────────────────────────────

function mulberry32(seed: number) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function sampleDirichlet(k: number, rng: () => number): number[] {
  // Dirichlet(1,...,1) = uniform on simplex via gamma(1,1)
  const raw = Array.from({ length: k }, () => -Math.log(1 - rng()));
  return normalize(raw);
}

// ── Component ────────────────────────────────────────────────────────

export default function FDivergenceFamilyExplorer() {
  const { ref: containerRef, width: containerWidth } =
    useResizeObserver<HTMLDivElement>();

  const [visible, setVisible] = useState<Record<string, boolean>>(
    Object.fromEntries(DIVERGENCES.map((d) => [d.key, true]))
  );
  const [hovered, setHovered] = useState<string | null>(null);
  const [pinskerK, setPinskerK] = useState(3);

  const isStacked = containerWidth > 0 && containerWidth < SM_BREAKPOINT;

  const toggleDiv = (key: string) =>
    setVisible((prev) => ({ ...prev, [key]: !prev[key] }));

  // ── Generator function panel (left) ──────────────────────────────
  const genWidth = isStacked
    ? containerWidth
    : Math.floor(containerWidth / 2);

  const genRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (genWidth <= 0) return;

      const w = genWidth - MARGIN.left - MARGIN.right;
      const h = PANEL_HEIGHT - MARGIN.top - MARGIN.bottom;
      const g = svg
        .append('g')
        .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const xScale = d3.scaleLinear().domain([0.01, 4]).range([0, w]);
      const yScale = d3.scaleLinear().domain([-1.5, 5]).range([h, 0]);

      g.append('g')
        .attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(xScale).ticks(6))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '11px');

      g.append('g')
        .call(d3.axisLeft(yScale).ticks(6))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '11px');

      g.append('text')
        .attr('x', w / 2)
        .attr('y', h + 30)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '12px')
        .text('t = p(x)/q(x)');

      g.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', -h / 2)
        .attr('y', -40)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '12px')
        .text('f(t)');

      // (1, 0) marker
      g.append('circle')
        .attr('cx', xScale(1))
        .attr('cy', yScale(0))
        .attr('r', 4)
        .attr('fill', 'var(--color-text)')
        .attr('opacity', 0.5);

      const tGrid = d3.range(0.02, 4, 0.01);

      for (const div of DIVERGENCES) {
        if (!visible[div.key]) continue;
        const isHov = hovered === div.key;
        const data = tGrid
          .map((t) => ({ t, v: div.generator(t) }))
          .filter((d) => Number.isFinite(d.v) && d.v <= 5 && d.v >= -1.5);

        g.append('path')
          .datum(data)
          .attr(
            'd',
            d3
              .line<{ t: number; v: number }>()
              .x((d) => xScale(d.t))
              .y((d) => yScale(d.v))
              .curve(d3.curveBasis)
          )
          .attr('fill', 'none')
          .attr('stroke', div.color)
          .attr('stroke-width', isHov ? 3.5 : 2)
          .attr('opacity', hovered && !isHov ? 0.2 : 0.85)
          .style('cursor', 'pointer')
          .on('mouseenter', () => setHovered(div.key))
          .on('mouseleave', () => setHovered(null));
      }

      // Title
      g.append('text')
        .attr('x', w / 2)
        .attr('y', -10)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)')
        .style('font-size', '13px')
        .style('font-weight', '600')
        .text('Generator functions f(t)');

      g.selectAll('.domain').style('stroke', 'var(--color-border)');
      g.selectAll('.tick line').style('stroke', 'var(--color-border)');
    },
    [genWidth, visible, hovered]
  );

  // ── Divergence comparison panel (right) ──────────────────────────
  const compWidth = isStacked
    ? containerWidth
    : containerWidth - genWidth;

  const compRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (compWidth <= 0) return;

      const w = compWidth - MARGIN.left - MARGIN.right;
      const h = PANEL_HEIGHT - MARGIN.top - MARGIN.bottom;
      const g = svg
        .append('g')
        .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const thetaGrid = d3.range(0.01, 0.99, 0.005);
      const q = [0.5, 0.5]; // fixed Bernoulli(0.5)

      // Compute all divergences across theta
      const allVals: { div: DivDef; data: { theta: number; val: number }[] }[] =
        DIVERGENCES.filter((d) => visible[d.key]).map((div) => ({
          div,
          data: thetaGrid.map((theta) => {
            const p = [theta, 1 - theta];
            return { theta, val: div.compute(p, q) };
          }),
        }));

      const maxVal = Math.min(
        d3.max(allVals.flatMap((d) => d.data.map((pt) => pt.val))) ?? 5,
        8
      );

      const xScale = d3.scaleLinear().domain([0, 1]).range([0, w]);
      const yScale = d3
        .scaleLinear()
        .domain([0, maxVal * 1.05])
        .range([h, 0]);

      g.append('g')
        .attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(xScale).ticks(5))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '11px');

      g.append('g')
        .call(d3.axisLeft(yScale).ticks(5))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '11px');

      g.append('text')
        .attr('x', w / 2)
        .attr('y', h + 30)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '12px')
        .text('θ (p = Bernoulli(θ), q = Bernoulli(0.5))');

      for (const { div, data } of allVals) {
        const isHov = hovered === div.key;
        const filtered = data.filter(
          (d) => Number.isFinite(d.val) && d.val <= maxVal * 1.1
        );
        g.append('path')
          .datum(filtered)
          .attr(
            'd',
            d3
              .line<{ theta: number; val: number }>()
              .x((d) => xScale(d.theta))
              .y((d) => yScale(d.val))
              .curve(d3.curveBasis)
          )
          .attr('fill', 'none')
          .attr('stroke', div.color)
          .attr('stroke-width', isHov ? 3.5 : 2)
          .attr('opacity', hovered && !isHov ? 0.2 : 0.85)
          .style('cursor', 'pointer')
          .on('mouseenter', () => setHovered(div.key))
          .on('mouseleave', () => setHovered(null));
      }

      // Title
      g.append('text')
        .attr('x', w / 2)
        .attr('y', -10)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)')
        .style('font-size', '13px')
        .style('font-weight', '600')
        .text('Divergence vs distributional mismatch');

      g.selectAll('.domain').style('stroke', 'var(--color-border)');
      g.selectAll('.tick line').style('stroke', 'var(--color-border)');
    },
    [compWidth, visible, hovered]
  );

  // ── Pinsker scatter panel (bottom) ───────────────────────────────
  const pinskerData = useMemo(() => {
    const rng = mulberry32(42);
    const points: { kl: number; tv: number }[] = [];
    for (let i = 0; i < 500; i++) {
      const p = sampleDirichlet(pinskerK, rng);
      const q = sampleDirichlet(pinskerK, rng);
      const kl = klDivergence(p, q) * Math.LN2; // nats
      const tv = totalVariation(p, q);
      if (Number.isFinite(kl) && Number.isFinite(tv)) {
        points.push({ kl, tv });
      }
    }
    return points;
  }, [pinskerK]);

  const pinskerRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (containerWidth <= 0) return;

      const w = containerWidth - MARGIN.left - MARGIN.right;
      const h = PINSKER_HEIGHT - MARGIN.top - MARGIN.bottom;
      const g = svg
        .append('g')
        .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const maxKL = Math.min(d3.max(pinskerData, (d) => d.kl) ?? 5, 10);
      const xScale = d3.scaleLinear().domain([0, maxKL]).range([0, w]);
      const yScale = d3.scaleLinear().domain([0, 1]).range([h, 0]);

      g.append('g')
        .attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(xScale).ticks(6))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '11px');

      g.append('g')
        .call(d3.axisLeft(yScale).ticks(5))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '11px');

      g.append('text')
        .attr('x', w / 2)
        .attr('y', h + 30)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '12px')
        .text('D_KL(p ‖ q) [nats]');

      g.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', -h / 2)
        .attr('y', -40)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '12px')
        .text('TV(p, q)');

      // Points
      g.selectAll('circle')
        .data(pinskerData)
        .enter()
        .append('circle')
        .attr('cx', (d) => xScale(d.kl))
        .attr('cy', (d) => yScale(d.tv))
        .attr('r', 2.5)
        .attr('fill', dimensionColors[0])
        .attr('opacity', 0.35);

      // Pinsker bound curve: TV = sqrt(KL/2)
      const klGrid = d3.range(0, maxKL, 0.02);
      const boundData = klGrid.map((kl) => ({
        kl,
        tv: Math.sqrt(kl / 2),
      }));

      g.append('path')
        .datum(boundData.filter((d) => d.tv <= 1))
        .attr(
          'd',
          d3
            .line<{ kl: number; tv: number }>()
            .x((d) => xScale(d.kl))
            .y((d) => yScale(d.tv))
        )
        .attr('fill', 'none')
        .attr('stroke', '#DC2626')
        .attr('stroke-width', 2.5)
        .attr('stroke-dasharray', '6,3');

      // Bound label
      g.append('text')
        .attr('x', xScale(Math.min(2, maxKL * 0.7)))
        .attr('y', yScale(Math.sqrt(Math.min(2, maxKL * 0.7) / 2)) - 8)
        .style('fill', '#DC2626')
        .style('font-size', '12px')
        .style('font-weight', '600')
        .text('TV ≤ √(KL/2)');

      // Title
      g.append('text')
        .attr('x', w / 2)
        .attr('y', -10)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)')
        .style('font-size', '13px')
        .style('font-weight', '600')
        .text("Pinsker's inequality — all points below the bound");

      g.selectAll('.domain').style('stroke', 'var(--color-border)');
      g.selectAll('.tick line').style('stroke', 'var(--color-border)');
    },
    [containerWidth, pinskerData]
  );

  return (
    <div
      ref={containerRef}
      style={{
        width: '100%',
        background: 'var(--color-surface, #fff)',
        borderRadius: '8px',
        border: '1px solid var(--color-border, #e5e7eb)',
        padding: '16px',
      }}
    >
      {/* Controls */}
      <div
        style={{
          display: 'flex',
          gap: '8px',
          flexWrap: 'wrap',
          marginBottom: '12px',
          alignItems: 'center',
        }}
      >
        {DIVERGENCES.map((div) => (
          <label
            key={div.key}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
              fontSize: '12px',
              color: visible[div.key]
                ? div.color
                : 'var(--color-text-secondary)',
              fontWeight:
                hovered === div.key ? '700' : visible[div.key] ? '500' : '400',
              cursor: 'pointer',
              opacity: visible[div.key] ? 1 : 0.5,
            }}
            onMouseEnter={() => setHovered(div.key)}
            onMouseLeave={() => setHovered(null)}
          >
            <input
              type="checkbox"
              checked={visible[div.key]}
              onChange={() => toggleDiv(div.key)}
              style={{ accentColor: div.color }}
            />
            {div.label}
          </label>
        ))}
        <label
          style={{
            fontSize: '12px',
            color: 'var(--color-text-secondary)',
            marginLeft: '12px',
          }}
        >
          Pinsker k:
          <select
            value={pinskerK}
            onChange={(e) => setPinskerK(Number(e.target.value))}
            style={{
              marginLeft: '4px',
              padding: '2px 6px',
              borderRadius: '4px',
              border: '1px solid var(--color-border)',
              background: 'var(--color-surface)',
              color: 'var(--color-text)',
              fontSize: '12px',
            }}
          >
            {[3, 5, 10].map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
        </label>
      </div>

      {/* Generator + Comparison panels */}
      <div
        style={{
          display: 'flex',
          flexDirection: isStacked ? 'column' : 'row',
        }}
      >
        <svg
          ref={genRef}
          width={genWidth}
          height={PANEL_HEIGHT}
          style={{ overflow: 'visible' }}
        />
        <svg
          ref={compRef}
          width={compWidth}
          height={PANEL_HEIGHT}
          style={{ overflow: 'visible' }}
        />
      </div>

      {/* Pinsker scatter */}
      <svg
        ref={pinskerRef}
        width={containerWidth}
        height={PINSKER_HEIGHT}
        style={{ overflow: 'visible' }}
      />
    </div>
  );
}
