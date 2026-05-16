import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  crossFitDmlPlr,
  linearRobinsonNuisanceFitter,
  mulberry32,
  paletteSemi,
  ROBINSON_NOISE_SD,
  ROBINSON_THETA_0,
  sampleRobinson,
  stddev,
} from './shared/semiparametric-inference';

// =============================================================================
// DMLCrossFitting — §7
// Top: K-fold partition visualization. n rows × K columns; cells colored
// "train" (light) vs "evaluate" (dark) for each fold.
// Bottom: estimator behavior (mean, SD) as a function of K ∈ {2, 5, 10}.
// MC budget defaults to B=100; commit-on-release slider supports 50, 100, 200.
// =============================================================================

const HEIGHT_TOP = 220;
const HEIGHT_BOT = 240;
const SM_BREAKPOINT = 640;
const N = 1000;
const K_OPTIONS = [2, 5, 10];
const BKRW_BOUND = ROBINSON_NOISE_SD / (ROBINSON_NOISE_SD * Math.sqrt(N)); // ≈ 0.0316

export default function DMLCrossFitting() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [k, setK] = useState(5);
  const [displayB, setDisplayB] = useState(100);
  const [committedB, setCommittedB] = useState(100);

  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;

  // Top-panel partition: deterministic for k.
  const partition = useMemo(() => {
    const N_VIS = isMobile ? 60 : 120;
    const rng = mulberry32(20260515);
    const idx: number[] = [];
    for (let i = 0; i < N_VIS; i++) idx.push(i);
    for (let i = N_VIS - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      const t = idx[i]; idx[i] = idx[j]; idx[j] = t;
    }
    const foldOf = new Array<number>(N_VIS);
    const sz = Math.floor(N_VIS / k);
    for (let i = 0; i < N_VIS; i++) {
      const pos = idx.indexOf(i);
      const foldIdx = Math.min(Math.floor(pos / sz), k - 1);
      foldOf[i] = foldIdx;
    }
    return { N_VIS, foldOf };
  }, [k, isMobile]);

  // Bottom panel: mean/SD of θ̂ across K options at the committed B.
  const kStats = useMemo(() => {
    const result: { K: number; mean: number; sd: number; thetas: number[] }[] = [];
    for (const Ki of K_OPTIONS) {
      const thetas: number[] = [];
      for (let b = 0; b < committedB; b++) {
        const rng = mulberry32(20260515 + b * 41 + Ki * 1009);
        const sample = sampleRobinson(N, ROBINSON_THETA_0, rng);
        const r = crossFitDmlPlr(sample.X, sample.D, sample.Y, Ki, linearRobinsonNuisanceFitter, rng);
        thetas.push(r.theta);
      }
      const mu = thetas.reduce((a, b) => a + b, 0) / thetas.length;
      const sd = stddev(Float64Array.from(thetas));
      result.push({ K: Ki, mean: mu, sd, thetas });
    }
    return result;
  }, [committedB]);

  const refTop = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const margin = { top: 28, right: 14, bottom: 28, left: 56 };
      const W = (isMobile ? w : Math.min(w, 720)) - margin.left - margin.right;
      const H = HEIGHT_TOP - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${(isMobile ? w : Math.min(w, 720))} ${HEIGHT_TOP}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);
      const { N_VIS, foldOf } = partition;
      const cellW = W / k;
      const cellH = H / N_VIS;
      for (let i = 0; i < N_VIS; i++) {
        for (let fi = 0; fi < k; fi++) {
          const isEval = foldOf[i] === fi;
          g.append('rect')
            .attr('x', fi * cellW)
            .attr('y', i * cellH)
            .attr('width', cellW)
            .attr('height', cellH)
            .style('fill', isEval ? paletteSemi.dml : paletteSemi.tangent)
            .style('opacity', isEval ? 0.8 : 0.2);
        }
      }
      g.append('text').attr('x', W / 2).attr('y', -10).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)')
        .style('font-family', 'var(--font-mono)').style('font-size', 12)
        .text(`K = ${k}: each row is an observation; columns are folds. Dark = evaluate, light = train.`);
      g.append('text').attr('x', -8).attr('y', 0).attr('text-anchor', 'end')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-family', 'var(--font-mono)').style('font-size', 10)
        .text('obs 1');
      g.append('text').attr('x', -8).attr('y', H).attr('text-anchor', 'end')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-family', 'var(--font-mono)').style('font-size', 10)
        .text(`obs ${N_VIS}`);
    },
    [partition, k, containerWidth, isMobile],
  );

  const refBot = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const margin = { top: 24, right: 14, bottom: 40, left: 56 };
      const W = (isMobile ? w : Math.min(w, 720)) - margin.left - margin.right;
      const H = HEIGHT_BOT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${(isMobile ? w : Math.min(w, 720))} ${HEIGHT_BOT}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);

      const allThetas = kStats.flatMap((d) => d.thetas);
      const xScale = d3.scaleBand().domain(K_OPTIONS.map(String)).range([0, W]).padding(0.3);
      const yLo = Math.min(0.85, Math.min(...allThetas));
      const yHi = Math.max(1.15, Math.max(...allThetas));
      const yScale = d3.scaleLinear().domain([yLo, yHi]).range([H, 0]).nice();
      g.append('g').attr('transform', `translate(0, ${H})`).call(d3.axisBottom(xScale));
      g.append('g').call(d3.axisLeft(yScale).ticks(5));
      // True θ_0 = 1.
      g.append('line').attr('x1', 0).attr('x2', W)
        .attr('y1', yScale(ROBINSON_THETA_0)).attr('y2', yScale(ROBINSON_THETA_0))
        .style('stroke', paletteSemi.theoryLine).style('stroke-width', 1.5)
        .style('stroke-dasharray', '4,3');
      g.append('text').attr('x', W).attr('y', yScale(ROBINSON_THETA_0) - 4).attr('text-anchor', 'end')
        .style('fill', paletteSemi.theoryLine).style('font-family', 'var(--font-mono)').style('font-size', 11)
        .text(`θ_0 = 1`);
      // For each K: violin-like strip of θ̂ draws.
      kStats.forEach((entry) => {
        const xBase = xScale(String(entry.K))! + xScale.bandwidth() / 2;
        // Box of mean ± SD.
        g.append('rect')
          .attr('x', xBase - 24)
          .attr('y', yScale(entry.mean + entry.sd))
          .attr('width', 48)
          .attr('height', yScale(entry.mean - entry.sd) - yScale(entry.mean + entry.sd))
          .style('fill', paletteSemi.dml).style('opacity', 0.18)
          .style('stroke', paletteSemi.dml).style('stroke-width', 1);
        // Individual draws (jittered).
        entry.thetas.forEach((t, idx) => {
          const jitter = ((idx % 17) / 17 - 0.5) * 36;
          g.append('circle')
            .attr('cx', xBase + jitter)
            .attr('cy', yScale(t))
            .attr('r', 1.6)
            .style('fill', paletteSemi.dml).style('opacity', 0.5);
        });
        // Mean tick.
        g.append('line')
          .attr('x1', xBase - 24).attr('x2', xBase + 24)
          .attr('y1', yScale(entry.mean)).attr('y2', yScale(entry.mean))
          .style('stroke', paletteSemi.oneStep).style('stroke-width', 2);
        g.append('text').attr('x', xBase).attr('y', yScale(entry.mean - entry.sd) + 16)
          .attr('text-anchor', 'middle')
          .style('fill', 'var(--color-text)')
          .style('font-family', 'var(--font-mono)').style('font-size', 11)
          .text(`mean=${entry.mean.toFixed(3)}, SD=${entry.sd.toFixed(3)}`);
      });
      g.append('text').attr('x', W / 2).attr('y', H + 32).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-family', 'var(--font-mono)').style('font-size', 11)
        .text(`K — BKRW bound at n=${N}: SD ≈ ${BKRW_BOUND.toFixed(4)}`);
    },
    [kStats, containerWidth, isMobile],
  );

  return (
    <div ref={containerRef} className="viz-container" style={{ width: '100%' }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 12 }}>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 13 }}>Folds K:</span>
          {[2, 5, 10].map((Ki) => (
            <label key={Ki} style={{ fontFamily: 'var(--font-mono)', fontSize: 13 }}>
              <input
                type="radio"
                name="kfold"
                value={Ki}
                checked={k === Ki}
                onChange={() => setK(Ki)}
              /> {Ki}
            </label>
          ))}
        </div>
        <label style={{ fontFamily: 'var(--font-mono)', fontSize: 13 }}>
          MC replications B: <strong>{displayB}</strong>
          <input
            type="range"
            min={50}
            max={200}
            step={50}
            value={displayB}
            onChange={(e) => setDisplayB(+e.target.value)}
            onMouseUp={() => setCommittedB(displayB)}
            onTouchEnd={() => setCommittedB(displayB)}
            onKeyUp={() => setCommittedB(displayB)}
            aria-label="Monte Carlo replications"
            style={{ width: '100%', marginTop: 4 }}
          />
        </label>
      </div>
      <svg ref={refTop} style={{ width: '100%', height: HEIGHT_TOP, display: 'block' }} />
      <svg ref={refBot} style={{ width: '100%', height: HEIGHT_BOT, display: 'block' }} />
    </div>
  );
}
