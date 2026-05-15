import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  ece,
  fitIsotonic,
  fitTemperature,
  mulberry32,
  predictIsotonic,
  reliabilityBins,
  temperatureScaleProb,
} from './shared/uncertainty-quantification';

// =============================================================================
// TemperatureScalingDemo — §13
//
// Build a synthetic overconfident classifier (latent score with strong logits,
// 20% label-flip noise). Slider for T ∈ [0.5, 5.0] showing reliability before
// and after temperature scaling. Also shows the auto-fitted T* via golden-section
// minimization on the calibration NLL. Toggle for isotonic-recalibrated baseline.
// Static fallback: public/images/topics/uncertainty-quantification/fig_13_temperature_scaling.png
// =============================================================================

const HEIGHT = 460;

function buildOverconfident(n: number, rng: () => number) {
  const y: number[] = [];
  const logits: number[] = [];
  for (let i = 0; i < n; i++) {
    const latent = rng();
    const yClean = latent > 0.5 ? 1 : 0;
    const z = (yClean === 1 ? 3.0 : -3.0) + 0.3 * (rng() - 0.5);
    const flipped = rng() < 0.2 ? 1 - yClean : yClean;
    y.push(flipped);
    logits.push(z);
  }
  return { y, logits };
}

export default function TemperatureScalingDemo() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [tDisplay, setTDisplay] = useState(1.0);

  // Split-once memo: build train/test, fit T* and isotonic on the calibration
  // (train) split, compute the per-T-invariant test outputs. The T slider only
  // recomputes the per-T probabilities + their ECE in the second memo below.
  const split = useMemo(() => {
    const rng = mulberry32(20260514 + 50);
    const train = buildOverconfident(500, rng);
    const test = buildOverconfident(500, rng);
    const tStar = fitTemperature(train.logits, train.y);
    const pRawTrain = train.logits.map((z) => 1 / (1 + Math.exp(-z)));
    const pRawTest = test.logits.map((z) => 1 / (1 + Math.exp(-z)));
    // Fit isotonic on the calibration (train) probabilities + labels, then
    // predict on the held-out test probabilities. Avoids the in-sample-leak
    // pattern flagged by code review on the first ship.
    const isoFit = fitIsotonic(pRawTrain, train.y);
    const pIso = predictIsotonic(isoFit, pRawTest);
    const pTStar = temperatureScaleProb(test.logits, tStar);
    return {
      yTest: test.y,
      logitsTest: test.logits,
      pRaw: pRawTest,
      pTStar,
      pIso,
      tStar,
      eceRaw: ece(test.y, pRawTest),
      eceTStar: ece(test.y, pTStar),
      eceIso: ece(test.y, pIso),
    };
  }, []);

  // Per-T outputs — recomputed on slider change but the fits above are not.
  const data = useMemo(() => {
    const pTUser = temperatureScaleProb(split.logitsTest, tDisplay);
    return {
      ...split,
      pTUser,
      eceTUser: ece(split.yTest, pTUser),
    };
  }, [split, tDisplay]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const margin = { top: 28, right: 24, bottom: 48, left: 56 };
      const W = w - margin.left - margin.right;
      const H = HEIGHT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${w} ${HEIGHT}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const xScale = d3.scaleLinear().domain([0, 1]).range([0, W]);
      const yScale = d3.scaleLinear().domain([0, 1]).range([H, 0]);

      g.append('line').attr('x1', xScale(0)).attr('y1', yScale(0))
        .attr('x2', xScale(1)).attr('y2', yScale(1))
        .style('stroke', '#94a3b8').attr('stroke-dasharray', '4 3').attr('stroke-width', 1.4);

      const drawCurve = (probs: number[], color: string, label: string,
                         dash: string = '') => {
        const bins = reliabilityBins(data.yTest, probs, 10);
        const valid: { c: number; a: number; w: number }[] = [];
        for (let b = 0; b < 10; b++) {
          if (!Number.isNaN(bins.binConf[b]) && bins.binW[b] > 0) {
            valid.push({ c: bins.binConf[b], a: bins.binAcc[b], w: bins.binW[b] });
          }
        }
        const lineGen = d3.line<typeof valid[number]>()
          .x((d) => xScale(d.c)).y((d) => yScale(d.a));
        g.append('path').datum(valid).attr('d', lineGen).attr('fill', 'none')
          .style('stroke', color).attr('stroke-width', 2).attr('stroke-dasharray', dash);
        for (const d of valid) {
          g.append('circle').attr('cx', xScale(d.c)).attr('cy', yScale(d.a))
            .attr('r', 3 + 8 * d.w).style('fill', color).style('opacity', 0.85);
        }
      };

      drawCurve(data.pRaw, '#dc2626', 'raw');
      drawCurve(data.pTUser, '#06b6d4', 'temp user');
      drawCurve(data.pTStar, '#8b5cf6', 'temp T*', '4 3');
      drawCurve(data.pIso, '#10b981', 'isotonic', '2 4');

      g.append('g').attr('transform', `translate(0,${H})`)
        .call(d3.axisBottom(xScale).ticks(6))
        .selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(yScale).ticks(6))
        .selectAll('text').style('fill', 'var(--color-text)');
      g.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');
      g.append('text').attr('x', W / 2).attr('y', H + 36).attr('text-anchor', 'middle')
        .style('font-size', '12px').style('fill', 'var(--color-text)')
        .text('mean predicted probability  p̂');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -H / 2).attr('y', -40)
        .attr('text-anchor', 'middle').style('font-size', '12px').style('fill', 'var(--color-text)')
        .text('empirical accuracy');

      // Legend / readouts.
      const badge = g.append('g').attr('transform', `translate(${W - 220}, 8)`);
      [
        { c: '#dc2626', t: `raw       ECE=${data.eceRaw.toFixed(3)}` },
        { c: '#06b6d4', t: `T = ${tDisplay.toFixed(2)}    ECE=${data.eceTUser.toFixed(3)}` },
        { c: '#8b5cf6', t: `T* = ${data.tStar.toFixed(2)}   ECE=${data.eceTStar.toFixed(3)}` },
        { c: '#10b981', t: `isotonic  ECE=${data.eceIso.toFixed(3)}` },
      ].forEach((d, i) => {
        badge.append('circle').attr('cx', 0).attr('cy', i * 16).attr('r', 5).style('fill', d.c);
        badge.append('text').attr('x', 10).attr('y', i * 16 + 4)
          .style('font-size', '12px').style('fill', 'var(--color-text)').text(d.t);
      });
    },
    [containerWidth, data, tDisplay],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: 'var(--font-sans)' }}>
      <svg ref={svgRef} width="100%" height={HEIGHT} role="img"
        aria-label="Reliability diagrams of a synthetic overconfident classifier with raw, user-set temperature, fitted T*, and isotonic recalibration overlaid." />
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', marginTop: '0.75rem',
        fontSize: '13px', color: 'var(--color-text)' }}>
        <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', minWidth: 240 }}>
          <span>temperature T: <strong>{tDisplay.toFixed(2)}</strong>  (fitted T* = {data.tStar.toFixed(2)})</span>
          <input type="range" min={0.3} max={5} step={0.05} value={tDisplay}
            onChange={(e) => setTDisplay(parseFloat(e.target.value))}
            aria-label="Temperature T" />
        </label>
      </div>
      <p style={{ fontSize: '12px', color: 'var(--color-text-secondary)', marginTop: '0.4rem' }}>
        Drag T to see the reliability curve respond. T &gt; 1 smooths overconfidence (pulls high-probability
        bins down toward the diagonal); T &lt; 1 sharpens. The fitted T* minimizes the calibration NLL via
        golden-section search; isotonic regression is the more flexible non-parametric alternative.
      </p>
    </div>
  );
}
