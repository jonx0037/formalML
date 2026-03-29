import { useState, useMemo } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { dimensionColors } from './shared/colorScales';
import { gaussianPdf, gmmPdf } from './shared/informationTheory';

// ── Constants ────────────────────────────────────────────────────────

const DENSITY_HEIGHT = 260;
const PENALTY_HEIGHT = 200;
const SM_BREAKPOINT = 640;
const MARGIN = { top: 30, right: 15, bottom: 35, left: 50 };

const BLUE = dimensionColors[0];
const RED = '#DC2626';
const PURPLE = dimensionColors[1];
const TARGET_COLOR = '#6B7280';

interface TargetPreset {
  label: string;
  mus: number[];
  sigmas: number[];
  weights: number[];
}

function makePresets(sep: number): Record<string, TargetPreset> {
  return {
    bimodal_sym: {
      label: 'Bimodal symmetric',
      mus: [-sep / 2, sep / 2],
      sigmas: [1, 1],
      weights: [0.5, 0.5],
    },
    bimodal_asym: {
      label: 'Bimodal asymmetric',
      mus: [-sep / 2, sep / 2],
      sigmas: [0.8, 1.2],
      weights: [0.7, 0.3],
    },
    trimodal: {
      label: 'Trimodal',
      mus: [-sep, 0, sep],
      sigmas: [0.8, 0.8, 0.8],
      weights: [0.35, 0.3, 0.35],
    },
  };
}

// ── Fitting logic ────────────────────────────────────────────────────

/** Forward KL optimal single Gaussian: moment matching */
function forwardKLFit(
  xGrid: number[],
  targetPdf: number[]
): { mu: number; sigma: number } {
  const dx = xGrid[1] - xGrid[0];
  let mean = 0;
  let totalMass = 0;
  for (let i = 0; i < xGrid.length; i++) {
    mean += xGrid[i] * targetPdf[i] * dx;
    totalMass += targetPdf[i] * dx;
  }
  mean /= totalMass;
  let variance = 0;
  for (let i = 0; i < xGrid.length; i++) {
    variance += (xGrid[i] - mean) ** 2 * targetPdf[i] * dx;
  }
  variance /= totalMass;
  return { mu: mean, sigma: Math.sqrt(Math.max(variance, 0.01)) };
}

/** Reverse KL optimal single Gaussian: find highest mode */
function reverseKLFit(
  mus: number[],
  sigmas: number[]
): { mu: number; sigma: number } {
  let bestMode = 0;
  let bestDensity = -Infinity;
  for (let j = 0; j < mus.length; j++) {
    const peakDensity = 1 / (sigmas[j] * Math.sqrt(2 * Math.PI));
    if (peakDensity > bestDensity) {
      bestDensity = peakDensity;
      bestMode = j;
    }
  }
  return { mu: mus[bestMode], sigma: sigmas[bestMode] };
}

// ── Component ────────────────────────────────────────────────────────

export default function ForwardReverseKLExplorer() {
  const { ref: containerRef, width: containerWidth } =
    useResizeObserver<HTMLDivElement>();

  const [separation, setSeparation] = useState(4);
  const [presetKey, setPresetKey] = useState<string>('bimodal_sym');

  const isStacked = containerWidth > 0 && containerWidth < SM_BREAKPOINT;

  const presets = useMemo(() => makePresets(separation), [separation]);
  const preset = presets[presetKey] ?? presets.bimodal_sym;

  // Grid for density evaluation
  const xGrid = useMemo(() => {
    const lo = -10;
    const hi = 10;
    const n = 400;
    return Array.from({ length: n }, (_, i) => lo + (i / (n - 1)) * (hi - lo));
  }, []);

  const targetPdf = useMemo(
    () => gmmPdf(xGrid, preset.mus, preset.sigmas, preset.weights),
    [xGrid, preset]
  );

  // Compute single-Gaussian fits
  const forwardFit = useMemo(() => {
    const { mu, sigma } = forwardKLFit(xGrid, targetPdf);
    return gaussianPdf(xGrid, mu, sigma);
  }, [xGrid, targetPdf]);

  const reverseFit = useMemo(() => {
    const { mu, sigma } = reverseKLFit(preset.mus, preset.sigmas);
    return gaussianPdf(xGrid, mu, sigma);
  }, [xGrid, preset]);

  // ── Density panels ─────────────────────────────────────────────────
  const panelWidth = isStacked
    ? containerWidth
    : Math.floor(containerWidth / 3);

  function renderDensityPanel(
    svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
    width: number,
    targetP: number[],
    fitP: number[] | null,
    title: string,
    fitColor: string
  ) {
    svg.selectAll('*').remove();
    if (width <= 0) return;

    const w = width - MARGIN.left - MARGIN.right;
    const h = DENSITY_HEIGHT - MARGIN.top - MARGIN.bottom;
    const g = svg
      .append('g')
      .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    const maxDensity = Math.max(
      d3.max(targetP) ?? 0,
      fitP ? d3.max(fitP) ?? 0 : 0
    );
    const xScale = d3.scaleLinear().domain([-10, 10]).range([0, w]);
    const yScale = d3
      .scaleLinear()
      .domain([0, maxDensity * 1.1])
      .range([h, 0]);

    // Axes
    g.append('g')
      .attr('transform', `translate(0,${h})`)
      .call(d3.axisBottom(xScale).ticks(5))
      .selectAll('text')
      .style('fill', 'var(--color-text-secondary)')
      .style('font-size', '11px');

    g.append('g')
      .call(d3.axisLeft(yScale).ticks(4))
      .selectAll('text')
      .style('fill', 'var(--color-text-secondary)')
      .style('font-size', '11px');

    const line = d3
      .line<number>()
      .x((_, i) => xScale(xGrid[i]))
      .y((d) => yScale(d))
      .curve(d3.curveBasis);

    // Target (shaded)
    const area = d3
      .area<number>()
      .x((_, i) => xScale(xGrid[i]))
      .y0(h)
      .y1((d) => yScale(d))
      .curve(d3.curveBasis);

    g.append('path')
      .datum(targetP)
      .attr('d', area)
      .attr('fill', TARGET_COLOR)
      .attr('opacity', 0.15);

    g.append('path')
      .datum(targetP)
      .attr('d', line)
      .attr('fill', 'none')
      .attr('stroke', TARGET_COLOR)
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '4,3');

    // Fit curve
    if (fitP) {
      g.append('path')
        .datum(fitP)
        .attr('d', line)
        .attr('fill', 'none')
        .attr('stroke', fitColor)
        .attr('stroke-width', 2.5);
    }

    // Title
    g.append('text')
      .attr('x', w / 2)
      .attr('y', -10)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text)')
      .style('font-size', '12px')
      .style('font-weight', '600')
      .text(title);

    g.selectAll('.domain').style('stroke', 'var(--color-border)');
    g.selectAll('.tick line').style('stroke', 'var(--color-border)');
  }

  const targetRef = useD3<SVGSVGElement>(
    (svg) =>
      renderDensityPanel(svg, panelWidth, targetPdf, null, 'Target p(x)', ''),
    [panelWidth, targetPdf]
  );

  const forwardRef = useD3<SVGSVGElement>(
    (svg) =>
      renderDensityPanel(
        svg,
        panelWidth,
        targetPdf,
        forwardFit,
        'Forward KL (mode-covering)',
        BLUE
      ),
    [panelWidth, targetPdf, forwardFit]
  );

  const reverseRef = useD3<SVGSVGElement>(
    (svg) =>
      renderDensityPanel(
        svg,
        panelWidth,
        targetPdf,
        reverseFit,
        'Reverse KL (mode-seeking)',
        RED
      ),
    [panelWidth, targetPdf, reverseFit]
  );

  // ── Penalty landscape ──────────────────────────────────────────────
  const penaltyRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (containerWidth <= 0) return;

      const halfW = Math.floor(containerWidth / 2);
      const w = halfW - MARGIN.left - MARGIN.right;
      const h = PENALTY_HEIGHT - MARGIN.top - MARGIN.bottom;

      const xScale = d3.scaleLinear().domain([-10, 10]).range([0, w]);

      // Forward penalty: p(x) * log(p(x)/q(x))
      const fwdPenalty = xGrid.map((_, i) => {
        const p = targetPdf[i];
        const q = forwardFit[i];
        if (p <= 1e-15) return 0;
        if (q <= 1e-15) return 5;
        return p * Math.log(p / q);
      });

      // Reverse penalty: q(x) * log(q(x)/p(x))
      const revPenalty = xGrid.map((_, i) => {
        const p = targetPdf[i];
        const q = reverseFit[i];
        if (q <= 1e-15) return 0;
        if (p <= 1e-15) return 5;
        return q * Math.log(q / p);
      });

      const maxPen = Math.max(
        d3.max(fwdPenalty) ?? 0.1,
        d3.max(revPenalty) ?? 0.1
      );

      function renderPenaltyPanel(
        gParent: d3.Selection<SVGGElement, unknown, null, undefined>,
        data: number[],
        color: string,
        title: string
      ) {
        const yScale = d3
          .scaleLinear()
          .domain([0, maxPen * 1.1])
          .range([h, 0]);

        gParent
          .append('g')
          .attr('transform', `translate(0,${h})`)
          .call(d3.axisBottom(xScale).ticks(5))
          .selectAll('text')
          .style('fill', 'var(--color-text-secondary)')
          .style('font-size', '11px');

        gParent
          .append('g')
          .call(d3.axisLeft(yScale).ticks(4))
          .selectAll('text')
          .style('fill', 'var(--color-text-secondary)')
          .style('font-size', '11px');

        const area = d3
          .area<number>()
          .x((_, i) => xScale(xGrid[i]))
          .y0(h)
          .y1((d) => yScale(Math.min(d, maxPen * 1.1)))
          .curve(d3.curveBasis);

        gParent
          .append('path')
          .datum(data)
          .attr('d', area)
          .attr('fill', color)
          .attr('opacity', 0.3);

        const line = d3
          .line<number>()
          .x((_, i) => xScale(xGrid[i]))
          .y((d) => yScale(Math.min(d, maxPen * 1.1)))
          .curve(d3.curveBasis);

        gParent
          .append('path')
          .datum(data)
          .attr('d', line)
          .attr('fill', 'none')
          .attr('stroke', color)
          .attr('stroke-width', 2);

        gParent
          .append('text')
          .attr('x', w / 2)
          .attr('y', -10)
          .attr('text-anchor', 'middle')
          .style('fill', color)
          .style('font-size', '12px')
          .style('font-weight', '600')
          .text(title);

        gParent.selectAll('.domain').style('stroke', 'var(--color-border)');
        gParent
          .selectAll('.tick line')
          .style('stroke', 'var(--color-border)');
      }

      const g1 = svg
        .append('g')
        .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);
      renderPenaltyPanel(g1, fwdPenalty, BLUE, 'Forward KL penalty');

      const g2 = svg
        .append('g')
        .attr(
          'transform',
          `translate(${halfW + MARGIN.left},${MARGIN.top})`
        );
      renderPenaltyPanel(g2, revPenalty, RED, 'Reverse KL penalty');
    },
    [containerWidth, xGrid, targetPdf, forwardFit, reverseFit]
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
          gap: '12px',
          flexWrap: 'wrap',
          marginBottom: '12px',
          alignItems: 'center',
        }}
      >
        <label style={{ fontSize: '13px', color: 'var(--color-text-secondary)' }}>
          Mode separation:
          <input
            type="range"
            min={2}
            max={8}
            step={0.5}
            value={separation}
            onChange={(e) => setSeparation(Number(e.target.value))}
            style={{ marginLeft: '8px', width: '100px', verticalAlign: 'middle' }}
          />
          <span style={{ marginLeft: '4px' }}>{separation}σ</span>
        </label>
        <label style={{ fontSize: '13px', color: 'var(--color-text-secondary)' }}>
          Target:
          <select
            value={presetKey}
            onChange={(e) => setPresetKey(e.target.value)}
            style={{
              marginLeft: '6px',
              padding: '4px 8px',
              borderRadius: '4px',
              border: '1px solid var(--color-border)',
              background: 'var(--color-surface)',
              color: 'var(--color-text)',
              fontSize: '13px',
            }}
          >
            {Object.entries(presets).map(([key, p]) => (
              <option key={key} value={key}>
                {p.label}
              </option>
            ))}
          </select>
        </label>
        <span style={{ fontSize: '12px', color: 'var(--color-text-secondary)', fontStyle: 'italic' }}>
          Fit: single Gaussian
        </span>
      </div>

      {/* Three density plots */}
      <div
        style={{
          display: 'flex',
          flexDirection: isStacked ? 'column' : 'row',
        }}
      >
        <svg role="img" aria-label="Forward reverse klexplorer visualization (panel 1 of 4)"
          ref={targetRef}
          width={panelWidth}
          height={DENSITY_HEIGHT}
          style={{ overflow: 'visible' }}
        />
        <svg role="img" aria-label="Forward reverse klexplorer visualization (panel 2 of 4)"
          ref={forwardRef}
          width={panelWidth}
          height={DENSITY_HEIGHT}
          style={{ overflow: 'visible' }}
        />
        <svg role="img" aria-label="Forward reverse klexplorer visualization (panel 3 of 4)"
          ref={reverseRef}
          width={panelWidth}
          height={DENSITY_HEIGHT}
          style={{ overflow: 'visible' }}
        />
      </div>

      {/* Penalty landscape */}
      <svg role="img" aria-label="Forward reverse klexplorer visualization (panel 4 of 4)"
        ref={penaltyRef}
        width={containerWidth}
        height={PENALTY_HEIGHT}
        style={{ overflow: 'visible' }}
      />
    </div>
  );
}
