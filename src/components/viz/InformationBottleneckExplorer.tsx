import { useState, useMemo, useCallback } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { entropy, mutualInformation } from './shared/informationTheory';

const HEIGHT = 360;
const SM_BREAKPOINT = 640;
const MARGIN = { top: 30, right: 20, bottom: 50, left: 55 };

interface JointPreset {
  label: string;
  pxy: number[][];
}

const PRESETS: JointPreset[] = [
  {
    label: 'Noisy identity channel',
    pxy: (() => {
      const px = [0.3, 0.3, 0.2, 0.2];
      const ch = [
        [0.7, 0.1, 0.1, 0.1],
        [0.1, 0.7, 0.1, 0.1],
        [0.1, 0.1, 0.6, 0.2],
        [0.1, 0.1, 0.2, 0.6],
      ];
      const joint = px.map((pi, i) => ch[i].map(c => pi * c));
      const sum = joint.flat().reduce((a, b) => a + b, 0);
      return joint.map(row => row.map(v => v / sum));
    })(),
  },
  {
    label: 'Symmetric channel',
    pxy: (() => {
      const n = 4;
      const joint: number[][] = [];
      for (let i = 0; i < n; i++) {
        joint[i] = [];
        for (let j = 0; j < n; j++) {
          joint[i][j] = i === j ? 0.15 : 0.05 / (n - 1);
        }
      }
      const sum = joint.flat().reduce((a, b) => a + b, 0);
      return joint.map(row => row.map(v => v / sum));
    })(),
  },
  {
    label: 'Asymmetric channel',
    pxy: (() => {
      const joint = [
        [0.30, 0.05, 0.01, 0.01],
        [0.05, 0.20, 0.05, 0.01],
        [0.01, 0.03, 0.10, 0.02],
        [0.01, 0.01, 0.02, 0.12],
      ];
      const sum = joint.flat().reduce((a, b) => a + b, 0);
      return joint.map(row => row.map(v => v / sum));
    })(),
  },
];

function ibSolve(pxy: number[][], beta: number, nT: number, maxIter = 500, tol = 1e-10) {
  const nX = pxy.length;
  const nY = pxy[0].length;
  const px = pxy.map(row => row.reduce((a, b) => a + b, 0));
  const pyGivenX = pxy.map((row, i) => row.map(v => v / Math.max(px[i], 1e-300)));

  // Initialize p(t|x) randomly but deterministically based on beta
  const seed = Math.abs(Math.round(beta * 100)) % 1000;
  let ptGivenX: number[][] = [];
  for (let i = 0; i < nX; i++) {
    ptGivenX[i] = [];
    let rowSum = 0;
    for (let t = 0; t < nT; t++) {
      ptGivenX[i][t] = 1 + 0.5 * Math.sin(seed + i * 7 + t * 13);
      rowSum += ptGivenX[i][t];
    }
    for (let t = 0; t < nT; t++) ptGivenX[i][t] /= rowSum;
  }

  for (let iter = 0; iter < maxIter; iter++) {
    // p(t)
    const pt = new Array(nT).fill(0);
    for (let t = 0; t < nT; t++) {
      for (let i = 0; i < nX; i++) pt[t] += px[i] * ptGivenX[i][t];
      pt[t] = Math.max(pt[t], 1e-300);
    }

    // p(y|t)
    const pyGivenT: number[][] = [];
    for (let t = 0; t < nT; t++) {
      pyGivenT[t] = new Array(nY).fill(0);
      for (let i = 0; i < nX; i++) {
        for (let j = 0; j < nY; j++) {
          pyGivenT[t][j] += ptGivenX[i][t] * px[i] * pyGivenX[i][j] / pt[t];
        }
      }
    }

    // Update p(t|x)
    const newP: number[][] = [];
    let maxDiff = 0;
    for (let i = 0; i < nX; i++) {
      newP[i] = [];
      let rowSum = 0;
      for (let t = 0; t < nT; t++) {
        let dkl = 0;
        for (let j = 0; j < nY; j++) {
          if (pyGivenX[i][j] > 1e-300 && pyGivenT[t][j] > 1e-300) {
            dkl += pyGivenX[i][j] * Math.log(pyGivenX[i][j] / pyGivenT[t][j]);
          }
        }
        newP[i][t] = pt[t] * Math.exp(-beta * dkl);
        rowSum += newP[i][t];
      }
      for (let t = 0; t < nT; t++) {
        newP[i][t] = rowSum > 0 ? newP[i][t] / rowSum : 1 / nT;
        maxDiff = Math.max(maxDiff, Math.abs(newP[i][t] - ptGivenX[i][t]));
      }
    }

    ptGivenX = newP;
    if (maxDiff < tol) break;
  }

  // Compute I(X;T)
  const pxt: number[][] = [];
  for (let i = 0; i < nX; i++) {
    pxt[i] = ptGivenX[i].map(v => px[i] * v);
  }
  const IXT = mutualInformation(pxt);

  // Compute I(T;Y)
  const pt = new Array(nT).fill(0);
  for (let t = 0; t < nT; t++) {
    for (let i = 0; i < nX; i++) pt[t] += px[i] * ptGivenX[i][t];
  }
  const pyGivenT: number[][] = [];
  for (let t = 0; t < nT; t++) {
    pyGivenT[t] = new Array(nY).fill(0);
    for (let i = 0; i < nX; i++) {
      for (let j = 0; j < nY; j++) {
        pyGivenT[t][j] += ptGivenX[i][t] * px[i] * pyGivenX[i][j] / Math.max(pt[t], 1e-300);
      }
    }
  }
  const pty: number[][] = [];
  for (let t = 0; t < nT; t++) {
    pty[t] = pyGivenT[t].map(v => pt[t] * v);
  }
  const ITY = mutualInformation(pty);

  return { IXT, ITY };
}

export default function InformationBottleneckExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();

  const [presetIdx, setPresetIdx] = useState(0);
  const [beta, setBeta] = useState(5);
  const [nT, setNT] = useState(4);

  const preset = PRESETS[presetIdx];

  // Compute I(X;Y) as reference
  const IXY = useMemo(() => mutualInformation(preset.pxy), [preset.pxy]);
  const HX = useMemo(() => {
    const px = preset.pxy.map(row => row.reduce((a, b) => a + b, 0));
    return entropy(px);
  }, [preset.pxy]);

  // Precompute IB curve
  const ibCurve = useMemo(() => {
    const betas = [
      ...Array.from({ length: 15 }, (_, i) => 0.1 + (i / 14) * 1.9),
      ...Array.from({ length: 25 }, (_, i) => 2 + (i / 24) * 18),
      ...Array.from({ length: 10 }, (_, i) => 20 + (i / 9) * 80),
    ];
    return betas.map(b => {
      const { IXT, ITY } = ibSolve(preset.pxy, b, nT);
      return { IXT, ITY, beta: b };
    });
  }, [preset.pxy, nT]);

  // Current beta point
  const currentPoint = useMemo(() => {
    return ibSolve(preset.pxy, beta, nT);
  }, [preset.pxy, beta, nT]);

  const isSmall = containerWidth < SM_BREAKPOINT;
  const leftW = isSmall ? containerWidth : Math.floor(containerWidth * 0.6);
  const rightW = isSmall ? containerWidth : containerWidth - leftW;
  const svgHeight = isSmall ? HEIGHT * 2 : HEIGHT;

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (containerWidth <= 0) return;

      // --- Left: IB curve ---
      const w1 = leftW - MARGIN.left - MARGIN.right;
      const h = HEIGHT - MARGIN.top - MARGIN.bottom;
      const g1 = svg.append('g')
        .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const xMax = HX * 1.1;
      const yMax = IXY * 1.15;
      const xScale = d3.scaleLinear().domain([0, xMax]).range([0, w1]);
      const yScale = d3.scaleLinear().domain([0, yMax]).range([h, 0]);

      // Reference line I(X;Y)
      g1.append('line')
        .attr('x1', 0).attr('y1', yScale(IXY))
        .attr('x2', w1).attr('y2', yScale(IXY))
        .style('stroke', '#D97706').style('stroke-dasharray', '6,3').style('stroke-width', 1.5);
      g1.append('text')
        .attr('x', w1 - 5).attr('y', yScale(IXY) - 6)
        .attr('text-anchor', 'end')
        .style('font-size', '10px').style('fill', '#D97706')
        .text(`I(X;Y) = ${IXY.toFixed(3)}`);

      // IB curve points
      const colorScale = d3.scaleSequential(d3.interpolateViridis)
        .domain([Math.log10(0.1), Math.log10(100)]);

      // Line connecting points
      const sortedCurve = [...ibCurve].sort((a, b) => a.IXT - b.IXT);
      const lineGen = d3.line<typeof ibCurve[0]>()
        .x(d => xScale(d.IXT)).y(d => yScale(d.ITY));
      g1.append('path')
        .datum(sortedCurve)
        .attr('d', lineGen)
        .style('fill', 'none')
        .style('stroke', '#999')
        .style('stroke-width', 1)
        .style('opacity', 0.5);

      // Points
      ibCurve.forEach(pt => {
        g1.append('circle')
          .attr('cx', xScale(pt.IXT)).attr('cy', yScale(pt.ITY))
          .attr('r', 4)
          .style('fill', colorScale(Math.log10(pt.beta)))
          .style('opacity', 0.7);
      });

      // Current point highlighted
      g1.append('circle')
        .attr('cx', xScale(currentPoint.IXT)).attr('cy', yScale(currentPoint.ITY))
        .attr('r', 8)
        .style('fill', '#534AB7').style('stroke', '#fff').style('stroke-width', 2);

      // Axes
      g1.append('g').attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(xScale).ticks(5))
        .selectAll('text').style('fill', 'var(--color-text-secondary, #666)');
      g1.append('g').call(d3.axisLeft(yScale).ticks(5))
        .selectAll('text').style('fill', 'var(--color-text-secondary, #666)');
      g1.selectAll('.domain, .tick line').style('stroke', 'var(--color-text-secondary, #999)');

      g1.append('text').attr('x', w1 / 2).attr('y', h + 40).attr('text-anchor', 'middle')
        .style('font-size', '12px').style('fill', 'var(--color-text-secondary, #666)')
        .text('Complexity: I(X; T) (bits)');
      g1.append('text').attr('transform', 'rotate(-90)').attr('x', -h / 2).attr('y', -42)
        .attr('text-anchor', 'middle')
        .style('font-size', '12px').style('fill', 'var(--color-text-secondary, #666)')
        .text('Relevance: I(T; Y) (bits)');
      g1.append('text').attr('x', w1 / 2).attr('y', -12).attr('text-anchor', 'middle')
        .style('font-size', '12px').style('fill', 'var(--color-text-secondary, #666)').style('font-weight', '600')
        .text('Information Bottleneck Curve');

      // --- Right: Complexity/Relevance bars ---
      const rOffset = isSmall ? 0 : leftW;
      const rYOffset = isSmall ? HEIGHT + 10 : 0;
      const w2 = (isSmall ? containerWidth : rightW) - MARGIN.left - MARGIN.right;
      const g2 = svg.append('g')
        .attr('transform', `translate(${rOffset + MARGIN.left},${rYOffset + MARGIN.top})`);

      const barData = [
        { label: 'Complexity I(X;T)', value: currentPoint.IXT, color: '#534AB7' },
        { label: 'Relevance I(T;Y)', value: currentPoint.ITY, color: '#0F6E56' },
      ];
      const barMax = Math.max(HX, IXY) * 1.1;
      const barScale = d3.scaleLinear().domain([0, barMax]).range([0, w2]);
      const barY = d3.scaleBand<number>().domain([0, 1]).range([h * 0.25, h * 0.75]).padding(0.3);

      barData.forEach((d, i) => {
        g2.append('rect')
          .attr('x', 0).attr('y', barY(i)!)
          .attr('width', barScale(d.value))
          .attr('height', barY.bandwidth())
          .style('fill', d.color).style('rx', 4);

        g2.append('text')
          .attr('x', barScale(d.value) + 5)
          .attr('y', barY(i)! + barY.bandwidth() / 2 + 4)
          .style('font-size', '12px').style('fill', 'var(--color-text-secondary, #666)').style('font-weight', '500')
          .text(`${d.value.toFixed(3)} bits`);

        g2.append('text')
          .attr('x', 0)
          .attr('y', barY(i)! - 6)
          .style('font-size', '11px').style('fill', d.color).style('font-weight', '600')
          .text(d.label);
      });

      // β readout
      g2.append('text')
        .attr('x', w2 / 2).attr('y', h - 10)
        .attr('text-anchor', 'middle')
        .style('font-size', '13px').style('fill', 'var(--color-text-secondary, #666)').style('font-weight', '600')
        .text(`β = ${beta.toFixed(1)}`);

      g2.append('text').attr('x', w2 / 2).attr('y', -12).attr('text-anchor', 'middle')
        .style('font-size', '12px').style('fill', 'var(--color-text-secondary, #666)').style('font-weight', '600')
        .text('Current Operating Point');
    },
    [containerWidth, ibCurve, currentPoint, beta, HX, IXY, isSmall, leftW, rightW]
  );

  const handleBeta = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    // Log scale: slider value 0-1 maps to beta 0.1-100
    const t = parseFloat(e.target.value);
    const b = 0.1 * Math.pow(1000, t);
    setBeta(b);
  }, []);

  const betaSliderVal = Math.log(beta / 0.1) / Math.log(1000);

  return (
    <div ref={containerRef} className="my-6 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 p-4">
      <div className="mb-3 flex flex-wrap items-center gap-4">
        <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
          Joint p(x,y):
          <select
            value={presetIdx}
            onChange={e => setPresetIdx(Number(e.target.value))}
            className="ml-2 rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-2 py-1 text-sm text-gray-900 dark:text-gray-100"
          >
            {PRESETS.map((p, i) => (
              <option key={i} value={i}>{p.label}</option>
            ))}
          </select>
        </label>
        <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
          β:
          <input
            type="range"
            min={0}
            max={1}
            step={0.005}
            value={betaSliderVal}
            onChange={handleBeta}
            className="ml-2 w-28 align-middle"
          />
          <span className="ml-1 text-xs font-mono">{beta.toFixed(1)}</span>
        </label>
        <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
          |T|:
          <select
            value={nT}
            onChange={e => setNT(Number(e.target.value))}
            className="ml-2 rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-2 py-1 text-sm text-gray-900 dark:text-gray-100"
          >
            {[2, 3, 4].map(v => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
        </label>
      </div>
      <svg ref={svgRef} width={containerWidth} height={svgHeight} />
    </div>
  );
}
