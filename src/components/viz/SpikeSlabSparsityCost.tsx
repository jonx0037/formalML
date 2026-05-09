import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';

// =============================================================================
// SpikeSlabSparsityCost — §3 three-panel: (a) spike+slab marginal density,
// (b) 2^p model-space blowup vs continuous-shrinkage's constant-1, (c)
// Castillo-van der Vaart minimax sparse-recovery rate vs the dense rate.
//
// Sliders for (π_prior, τ_slab, p, s, n).
// =============================================================================

const PANEL_HEIGHT = 320;
const MARGIN = { top: 24, right: 16, bottom: 50, left: 50 };

const SPIKE_COLOR = '#7b3c10';
const SLAB_COLOR = '#1f4e79';
const DENSE_COLOR = '#7f7f7f';

function gaussianPdf(x: number, mean: number, sigma: number): number {
  return (
    Math.exp(-0.5 * ((x - mean) / sigma) ** 2) /
    (sigma * Math.sqrt(2 * Math.PI))
  );
}

export default function SpikeSlabSparsityCost() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [piPrior, setPiPrior] = useState(0.2);
  const [tauSlab, setTauSlab] = useState(2.0);
  const [pDim, setPDim] = useState(200);
  const [sActive, setSActive] = useState(5);

  const data = useMemo(() => {
    const N = 200;
    const betaGrid = new Array<number>(N);
    const slabDensity = new Array<number>(N);
    for (let i = 0; i < N; i++) {
      const b = -5 + (10 * i) / (N - 1);
      betaGrid[i] = b;
      slabDensity[i] = piPrior * gaussianPdf(b, 0, tauSlab);
    }
    const pPoints = new Array<{ p: number; v: number }>(0);
    for (let p = 2; p <= 30; p++) {
      pPoints.push({ p, v: Math.pow(2, p) });
    }
    const nGrid = new Array<number>(60);
    for (let i = 0; i < 60; i++) {
      nGrid[i] = Math.pow(10, 2 + (4 * i) / 59);
    }
    const sparseRate = nGrid.map((n) =>
      Math.sqrt((sActive * Math.log(pDim / sActive)) / n),
    );
    const denseRate = nGrid.map((n) => Math.sqrt(pDim / n));
    const speedup =
      Math.sqrt(pDim / Math.max(sActive * Math.log(pDim / sActive), 1e-9));
    return { betaGrid, slabDensity, pPoints, nGrid, sparseRate, denseRate, speedup };
  }, [piPrior, tauSlab, pDim, sActive]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const w = width || 760;
      const h = PANEL_HEIGHT;
      svg.attr('width', w).attr('height', h);
      const panelW = (w - MARGIN.left - MARGIN.right - 60) / 3;
      const innerH = h - MARGIN.top - MARGIN.bottom;

      // Panel (a): marginal density with explicit point mass at 0
      const gA = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);
      const xA = d3.scaleLinear().domain([-5, 5]).range([0, panelW]);
      const yMaxA = Math.max(0.5, ...data.slabDensity);
      const yA = d3.scaleLinear().domain([0, yMaxA * 1.05]).range([innerH, 0]);
      gA.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xA).ticks(5));
      gA.append('g').call(d3.axisLeft(yA).ticks(4));
      gA.append('text')
        .attr('x', panelW / 2)
        .attr('y', innerH + 34)
        .attr('text-anchor', 'middle')
        .style('font-size', '11px')
        .text('β');
      gA.append('text')
        .attr('x', 0)
        .attr('y', -8)
        .style('font-size', '11px')
        .style('font-weight', '600')
        .text('(a) spike+slab marginal');

      const slabLine = d3
        .line<number>()
        .x((_, i) => xA(data.betaGrid[i]))
        .y((d) => yA(d));
      gA.append('path')
        .datum(data.slabDensity)
        .attr('fill', 'none')
        .attr('stroke', SLAB_COLOR)
        .attr('stroke-width', 2)
        .attr('d', slabLine);

      // Spike: vertical arrow
      const spikeMass = (1 - piPrior) * yMaxA * 1.6;
      const spikeY = yA(Math.min(spikeMass, yMaxA));
      gA.append('line')
        .attr('x1', xA(0))
        .attr('x2', xA(0))
        .attr('y1', innerH)
        .attr('y2', spikeY)
        .attr('stroke', SPIKE_COLOR)
        .attr('stroke-width', 4);
      gA.append('text')
        .attr('x', xA(0) + 6)
        .attr('y', spikeY + 4)
        .style('font-size', '10.5px')
        .style('fill', SPIKE_COLOR)
        .text(`mass ${(1 - piPrior).toFixed(2)}`);

      // Panel (b): 2^p vs constant
      const gB = svg.append('g').attr('transform', `translate(${MARGIN.left + panelW + 30},${MARGIN.top})`);
      const xB = d3.scaleLinear().domain([2, 30]).range([0, panelW]);
      const yB = d3.scaleLog().domain([1, 1e10]).range([innerH, 0]);
      gB.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xB).ticks(5));
      gB.append('g').call(d3.axisLeft(yB).ticks(5, '~g'));
      gB.append('text')
        .attr('x', panelW / 2)
        .attr('y', innerH + 34)
        .attr('text-anchor', 'middle')
        .style('font-size', '11px')
        .text('p (number of coefficients)');
      gB.append('text')
        .attr('x', 0)
        .attr('y', -8)
        .style('font-size', '11px')
        .style('font-weight', '600')
        .text('(b) 2^p model-space blowup');
      const lineB = d3
        .line<{ p: number; v: number }>()
        .x((d) => xB(d.p))
        .y((d) => yB(d.v));
      gB.append('path')
        .datum(data.pPoints)
        .attr('fill', 'none')
        .attr('stroke', SPIKE_COLOR)
        .attr('stroke-width', 2.4)
        .attr('d', lineB);
      gB.append('line')
        .attr('x1', xB(2))
        .attr('x2', xB(30))
        .attr('y1', yB(1))
        .attr('y2', yB(1))
        .attr('stroke', SLAB_COLOR)
        .attr('stroke-width', 1.4);
      // 2^25 SSVS limit
      gB.append('line')
        .attr('x1', xB(2))
        .attr('x2', xB(30))
        .attr('y1', yB(Math.pow(2, 25)))
        .attr('y2', yB(Math.pow(2, 25)))
        .attr('stroke', '#aa6633')
        .attr('stroke-dasharray', '3 3');
      gB.append('text')
        .attr('x', xB(15))
        .attr('y', yB(Math.pow(2, 25)) - 4)
        .style('font-size', '9.5px')
        .style('fill', '#aa6633')
        .text('SSVS practical limit ≈ 2²⁵');

      // Panel (c): sparse vs dense rate
      const gC = svg.append('g').attr('transform', `translate(${MARGIN.left + 2 * panelW + 60},${MARGIN.top})`);
      const xC = d3.scaleLog().domain([100, 10000]).range([0, panelW]);
      const minRate = Math.min(...data.sparseRate, ...data.denseRate);
      const maxRate = Math.max(...data.sparseRate, ...data.denseRate);
      const yC = d3.scaleLog().domain([Math.max(minRate, 1e-3), maxRate * 1.1]).range([innerH, 0]);
      gC.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xC).ticks(4, '~g'));
      gC.append('g').call(d3.axisLeft(yC).ticks(4, '~g'));
      gC.append('text')
        .attr('x', panelW / 2)
        .attr('y', innerH + 34)
        .attr('text-anchor', 'middle')
        .style('font-size', '11px')
        .text('n (sample size)');
      gC.append('text')
        .attr('x', 0)
        .attr('y', -8)
        .style('font-size', '11px')
        .style('font-weight', '600')
        .text('(c) Castillo-van der Vaart rate');

      const sparseLine = d3
        .line<number>()
        .x((_, i) => xC(data.nGrid[i]))
        .y((d) => yC(d));
      gC.append('path')
        .datum(data.sparseRate)
        .attr('fill', 'none')
        .attr('stroke', SLAB_COLOR)
        .attr('stroke-width', 2.4)
        .attr('d', sparseLine);
      const denseLine = d3
        .line<number>()
        .x((_, i) => xC(data.nGrid[i]))
        .y((d) => yC(d));
      gC.append('path')
        .datum(data.denseRate)
        .attr('fill', 'none')
        .attr('stroke', DENSE_COLOR)
        .attr('stroke-width', 1.8)
        .attr('stroke-dasharray', '5 4')
        .attr('d', denseLine);
      gC.append('text')
        .attr('x', panelW - 8)
        .attr('y', innerH - 6)
        .attr('text-anchor', 'end')
        .style('font-size', '10.5px')
        .style('fill', SLAB_COLOR)
        .text(`sparse/dense speedup = ${data.speedup.toFixed(2)}×`);
    },
    [data, piPrior, width],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', maxWidth: 760 }}>
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: 14,
          marginBottom: 8,
          alignItems: 'center',
          fontSize: 12.5,
        }}
      >
        <label>
          π: {piPrior.toFixed(2)}{' '}
          <input
            type="range"
            min={0.05}
            max={0.5}
            step={0.01}
            value={piPrior}
            onChange={(e) => setPiPrior(Number(e.target.value))}
            style={{ width: 90 }}
          />
        </label>
        <label>
          τ_slab: {tauSlab.toFixed(2)}{' '}
          <input
            type="range"
            min={0.5}
            max={5}
            step={0.05}
            value={tauSlab}
            onChange={(e) => setTauSlab(Number(e.target.value))}
            style={{ width: 90 }}
          />
        </label>
        <label>
          p: {pDim}{' '}
          <input
            type="range"
            min={20}
            max={2000}
            step={10}
            value={pDim}
            onChange={(e) => setPDim(Number(e.target.value))}
            style={{ width: 100 }}
          />
        </label>
        <label>
          s: {sActive}{' '}
          <input
            type="range"
            min={1}
            max={50}
            step={1}
            value={sActive}
            onChange={(e) => setSActive(Number(e.target.value))}
            style={{ width: 90 }}
          />
        </label>
      </div>
      <svg ref={svgRef} role="img" aria-label="Spike-and-slab three-panel sparsity cost" />
    </div>
  );
}
