import { useEffect, useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';

// =============================================================================
// InferenceDispatchExplorer — embedded after §4.5's pedagogical-takeaway
// paragraph in the probabilistic-programming topic. Three engines fitted
// to the §4.1 Bayesian logistic regression on Iris versicolor-vs-virginica:
//
//   - NUTS:  scatter of 2000 posterior draws (β_1, β_2), tilted by the
//            mild negative correlation between petal-length and petal-
//            width slopes.
//   - ADVI:  the mean-field Gaussian variational approximation rendered
//            as a 95%-credible ellipse — necessarily axis-aligned because
//            the family is mean-field, conspicuously so when overlaid on
//            NUTS's tilted scatter.
//   - MAP:   the L-BFGS optimum as a single star at the mode.
//
// The §4.5 claim "the three methods converge to the same answer at very
// different costs" becomes the visible fact that all three indicators
// land in the same neighborhood. The §4.5 caveat that "ADVI's mean-field
// marginals are slightly tighter than NUTS's" because the diagonal
// covariance ignores the petal-length/width correlation becomes the
// visible orientation gap between the orange ellipse and the blue
// scatter.
//
// A side panel renders the ADVI -ELBO loss trajectory across 20000
// optimization iterations (thinned to ~400 points for transport).
// The §4.3 "stable plateau is the convergence signal" claim becomes the
// observable shape of the loss curve.
//
// Toggles let the reader strip each method to isolate what it contributes:
// just NUTS shows what the gold-standard sampler sees; just ADVI shows
// the variational approximation; just MAP shows the lone optimum.
//
// Data: /sample-data/probabilistic-programming/inference_dispatch.json
// from notebooks/probabilistic-programming/precompute_inference_dispatch.py.
// =============================================================================

const PANEL_HEIGHT = 460;
const PANEL_GAP = 24;
const DATA_URL = '/sample-data/probabilistic-programming/inference_dispatch.json';
const CHI2_95 = 2.447746830680816;

const COLORS = {
  nuts: '#2563eb',     // blue scatter
  advi: '#ea580c',     // orange ellipse
  map: '#9333ea',      // purple star
  loss: '#0891b2',     // cyan loss trajectory
  axis: '#374151',
};

interface DispatchPayload {
  metadata: {
    pymc_version: string;
    n_obs: number;
    n_features: number;
    feature_names: string[];
    n_total: number;
    rhat: { alpha: number; beta_0: number; beta_1: number };
    ess_bulk: { alpha: number; beta_0: number; beta_1: number };
    advi_iterations: number;
  };
  nuts: { alpha: number[]; beta: number[][] };
  advi: {
    alpha_mean: number;
    beta_mean: number[];
    alpha_std: number;
    beta_std: number[];
    loss: number[];
  };
  map: { alpha: number; beta: number[] };
}

export default function InferenceDispatchExplorer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [payload, setPayload] = useState<DispatchPayload | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [showNuts, setShowNuts] = useState<boolean>(true);
  const [showAdvi, setShowAdvi] = useState<boolean>(true);
  const [showMap, setShowMap] = useState<boolean>(true);

  useEffect(() => {
    let cancelled = false;
    fetch(DATA_URL)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((j: DispatchPayload) => {
        if (!cancelled) setPayload(j);
      })
      .catch((e) => {
        if (!cancelled) setLoadError(String(e));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Compute axis ranges to cover all three methods comfortably.
  const betaRanges = useMemo<{
    b1: [number, number];
    b2: [number, number];
  } | null>(() => {
    if (!payload) return null;
    let lo1 = Infinity;
    let hi1 = -Infinity;
    let lo2 = Infinity;
    let hi2 = -Infinity;
    for (const pair of payload.nuts.beta) {
      if (pair[0] < lo1) lo1 = pair[0];
      if (pair[0] > hi1) hi1 = pair[0];
      if (pair[1] < lo2) lo2 = pair[1];
      if (pair[1] > hi2) hi2 = pair[1];
    }
    // Make sure the ADVI ellipse and MAP star are also inside.
    const aMean1 = payload.advi.beta_mean[0];
    const aMean2 = payload.advi.beta_mean[1];
    const aStd1 = payload.advi.beta_std[0];
    const aStd2 = payload.advi.beta_std[1];
    lo1 = Math.min(lo1, aMean1 - 2.5 * aStd1, payload.map.beta[0]);
    hi1 = Math.max(hi1, aMean1 + 2.5 * aStd1, payload.map.beta[0]);
    lo2 = Math.min(lo2, aMean2 - 2.5 * aStd2, payload.map.beta[1]);
    hi2 = Math.max(hi2, aMean2 + 2.5 * aStd2, payload.map.beta[1]);
    const pad1 = (hi1 - lo1) * 0.06;
    const pad2 = (hi2 - lo2) * 0.06;
    return {
      b1: [lo1 - pad1, hi1 + pad1],
      b2: [lo2 - pad2, hi2 + pad2],
    };
  }, [payload]);

  const ref = useD3(
    (svg) => {
      if (!payload || !betaRanges) return;
      const w = width || 720;
      const h = PANEL_HEIGHT;
      const margin = { top: 36, right: 16, bottom: 48, left: 56 };
      svg.attr('width', w).attr('height', h);
      svg.selectAll('*').remove();

      const innerW = w - margin.left - margin.right;
      const innerH = h - margin.top - margin.bottom;
      // Reserve ~38% for the loss trajectory on the right.
      const lossW = Math.max(180, innerW * 0.38);
      const mainW = innerW - lossW - PANEL_GAP;

      const mainG = svg
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);
      const xScale = d3.scaleLinear().domain(betaRanges.b1).range([0, mainW]);
      const yScale = d3.scaleLinear().domain(betaRanges.b2).range([innerH, 0]);

      mainG
        .append('g')
        .attr('transform', `translate(0,${innerH})`)
        .call(d3.axisBottom(xScale).ticks(6).tickSizeOuter(0));
      mainG.append('g').call(d3.axisLeft(yScale).ticks(6).tickSizeOuter(0));
      mainG
        .append('text')
        .attr('transform', `translate(${mainW / 2},${innerH + 36})`)
        .attr('text-anchor', 'middle')
        .attr('font-size', 12)
        .text('β₁ (petal length, standardized)');
      mainG
        .append('text')
        .attr('transform', `translate(-44,${innerH / 2}) rotate(-90)`)
        .attr('text-anchor', 'middle')
        .attr('font-size', 12)
        .text('β₂ (petal width, standardized)');
      mainG
        .append('text')
        .attr('x', mainW / 2)
        .attr('y', -12)
        .attr('text-anchor', 'middle')
        .attr('font-size', 12)
        .attr('font-weight', 600)
        .text('Joint posterior on (β₁, β₂)');

      // NUTS scatter.
      if (showNuts) {
        const nutsG = mainG.append('g');
        for (const pair of payload.nuts.beta) {
          if (pair[0] < betaRanges.b1[0] || pair[0] > betaRanges.b1[1]) continue;
          if (pair[1] < betaRanges.b2[0] || pair[1] > betaRanges.b2[1]) continue;
          nutsG
            .append('circle')
            .attr('cx', xScale(pair[0]))
            .attr('cy', yScale(pair[1]))
            .attr('r', 1.6)
            .attr('fill', COLORS.nuts)
            .attr('opacity', 0.32);
        }
      }

      // ADVI ellipse — mean-field Gaussian, axis-aligned.
      if (showAdvi) {
        const aMean = payload.advi.beta_mean;
        const aStd = payload.advi.beta_std;
        const cx = xScale(aMean[0]);
        const cy = yScale(aMean[1]);
        // 95% confidence: rx = chi2 * sigma_x in pixel space.
        const pxPerUnitX = mainW / (betaRanges.b1[1] - betaRanges.b1[0]);
        const pxPerUnitY = innerH / (betaRanges.b2[1] - betaRanges.b2[0]);
        mainG
          .append('ellipse')
          .attr('cx', cx)
          .attr('cy', cy)
          .attr('rx', CHI2_95 * aStd[0] * pxPerUnitX)
          .attr('ry', CHI2_95 * aStd[1] * pxPerUnitY)
          .attr('fill', 'none')
          .attr('stroke', COLORS.advi)
          .attr('stroke-width', 2.4);
        mainG
          .append('circle')
          .attr('cx', cx)
          .attr('cy', cy)
          .attr('r', 4)
          .attr('fill', COLORS.advi);
      }

      // MAP star.
      if (showMap) {
        const mx = xScale(payload.map.beta[0]);
        const my = yScale(payload.map.beta[1]);
        const r = 9;
        const star = d3
          .symbol()
          .type(d3.symbolStar)
          .size(140);
        mainG
          .append('path')
          .attr('transform', `translate(${mx},${my})`)
          .attr('d', star() as string)
          .attr('fill', COLORS.map)
          .attr('stroke', '#fff')
          .attr('stroke-width', 1.5);
      }

      // Legend.
      const legendItems: { label: string; color: string; marker: 'dot' | 'ring' | 'star' }[] = [];
      if (showNuts) legendItems.push({ label: 'NUTS samples', color: COLORS.nuts, marker: 'dot' });
      if (showAdvi) legendItems.push({ label: 'ADVI 95% ellipse', color: COLORS.advi, marker: 'ring' });
      if (showMap) legendItems.push({ label: 'MAP point', color: COLORS.map, marker: 'star' });
      const legend = mainG.append('g').attr('transform', `translate(${mainW - 160},10)`);
      legendItems.forEach((it, i) => {
        const row = legend.append('g').attr('transform', `translate(0,${i * 16})`);
        if (it.marker === 'dot') {
          row.append('circle').attr('cx', 6).attr('cy', 8).attr('r', 4).attr('fill', it.color).attr('opacity', 0.6);
        } else if (it.marker === 'ring') {
          row
            .append('ellipse')
            .attr('cx', 6)
            .attr('cy', 8)
            .attr('rx', 6)
            .attr('ry', 4)
            .attr('fill', 'none')
            .attr('stroke', it.color)
            .attr('stroke-width', 2);
        } else {
          row
            .append('path')
            .attr('transform', 'translate(6, 8)')
            .attr('d', d3.symbol().type(d3.symbolStar).size(60)() as string)
            .attr('fill', it.color);
        }
        row
          .append('text')
          .attr('x', 16)
          .attr('y', 11)
          .attr('font-size', 10)
          .text(it.label);
      });

      // ── ADVI loss trajectory panel ────────────────────────────────────
      const lossG = svg
        .append('g')
        .attr(
          'transform',
          `translate(${margin.left + mainW + PANEL_GAP},${margin.top})`,
        );
      const loss = payload.advi.loss;
      const lossX = d3
        .scaleLinear()
        .domain([0, loss.length - 1])
        .range([0, lossW]);
      const lossLo = Math.min(...loss);
      const lossHi = Math.max(...loss);
      const lossY = d3
        .scaleLinear()
        .domain([lossLo - (lossHi - lossLo) * 0.04, lossHi + (lossHi - lossLo) * 0.04])
        .range([innerH, 0]);

      lossG
        .append('g')
        .attr('transform', `translate(0,${innerH})`)
        .call(d3.axisBottom(lossX).ticks(4).tickSizeOuter(0).tickFormat(d3.format('~s')));
      lossG.append('g').call(d3.axisLeft(lossY).ticks(5).tickSizeOuter(0));
      lossG
        .append('text')
        .attr('transform', `translate(${lossW / 2},${innerH + 36})`)
        .attr('text-anchor', 'middle')
        .attr('font-size', 12)
        .text('ADVI iteration (thinned)');
      lossG
        .append('text')
        .attr('transform', `translate(-40,${innerH / 2}) rotate(-90)`)
        .attr('text-anchor', 'middle')
        .attr('font-size', 12)
        .text('−ELBO');
      lossG
        .append('text')
        .attr('x', lossW / 2)
        .attr('y', -12)
        .attr('text-anchor', 'middle')
        .attr('font-size', 12)
        .attr('font-weight', 600)
        .text('ADVI loss trajectory');

      const lineGen = d3
        .line<number>()
        .x((_, i) => lossX(i))
        .y((d) => lossY(d))
        .curve(d3.curveMonotoneX);

      lossG
        .append('path')
        .datum(loss)
        .attr('fill', 'none')
        .attr('stroke', COLORS.loss)
        .attr('stroke-width', 1.6)
        .attr('d', lineGen);
    },
    [payload, betaRanges, showNuts, showAdvi, showMap, width],
  );

  const m = payload?.metadata;

  // Compute NUTS posterior summary inline for the readout.
  const nutsSummary = useMemo(() => {
    if (!payload) return null;
    const beta = payload.nuts.beta;
    const N = beta.length;
    let sx = 0,
      sy = 0;
    for (const p of beta) {
      sx += p[0];
      sy += p[1];
    }
    const meanX = sx / N;
    const meanY = sy / N;
    let sdx = 0,
      sdy = 0;
    for (const p of beta) {
      sdx += (p[0] - meanX) ** 2;
      sdy += (p[1] - meanY) ** 2;
    }
    return {
      meanX,
      meanY,
      stdX: Math.sqrt(sdx / (N - 1)),
      stdY: Math.sqrt(sdy / (N - 1)),
    };
  }, [payload]);

  return (
    <div className="my-6 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-soft)] p-4">
      <div className="mb-3 flex flex-wrap items-center gap-4 text-sm">
        <label className="flex items-center gap-1">
          <input
            type="checkbox"
            checked={showNuts}
            onChange={(e) => setShowNuts(e.target.checked)}
            disabled={!payload}
          />
          <span style={{ color: COLORS.nuts }}>NUTS</span>
        </label>
        <label className="flex items-center gap-1">
          <input
            type="checkbox"
            checked={showAdvi}
            onChange={(e) => setShowAdvi(e.target.checked)}
            disabled={!payload}
          />
          <span style={{ color: COLORS.advi }}>ADVI</span>
        </label>
        <label className="flex items-center gap-1">
          <input
            type="checkbox"
            checked={showMap}
            onChange={(e) => setShowMap(e.target.checked)}
            disabled={!payload}
          />
          <span style={{ color: COLORS.map }}>MAP</span>
        </label>
      </div>
      <div
        ref={containerRef}
        className="w-full"
        style={{ minHeight: PANEL_HEIGHT }}
      >
        {loadError ? (
          <div className="text-sm">
            Failed to load dispatch data: {loadError}. Re-run
            <code className="mx-1">
              notebooks/probabilistic-programming/precompute_inference_dispatch.py
            </code>
            .
          </div>
        ) : payload ? (
          <svg ref={ref} />
        ) : (
          <div className="text-sm text-[var(--color-text-muted)]">
            Loading NUTS + ADVI + MAP fits…
          </div>
        )}
      </div>
      {m && nutsSummary && (
        <div className="mt-3 grid grid-cols-1 sm:grid-cols-3 gap-x-4 gap-y-1 text-xs text-[var(--color-text-muted)]">
          <div>
            <strong style={{ color: COLORS.nuts }}>NUTS posterior:</strong>
            <ul className="ml-4 list-disc">
              <li>
                β̄ = ({nutsSummary.meanX.toFixed(2)}, {nutsSummary.meanY.toFixed(2)})
              </li>
              <li>
                σ_β = ({nutsSummary.stdX.toFixed(2)}, {nutsSummary.stdY.toFixed(2)})
              </li>
              <li>
                R̂ ≤ {Math.max(m.rhat.alpha, m.rhat.beta_0, m.rhat.beta_1).toFixed(4)}
              </li>
            </ul>
          </div>
          <div>
            <strong style={{ color: COLORS.advi }}>ADVI mean-field:</strong>
            <ul className="ml-4 list-disc">
              <li>
                μ_q = ({payload.advi.beta_mean[0].toFixed(2)}, {payload.advi.beta_mean[1].toFixed(2)})
              </li>
              <li>
                σ_q = ({payload.advi.beta_std[0].toFixed(2)}, {payload.advi.beta_std[1].toFixed(2)})
              </li>
              <li>final loss: {payload.advi.loss[payload.advi.loss.length - 1].toFixed(2)}</li>
            </ul>
          </div>
          <div>
            <strong style={{ color: COLORS.map }}>MAP optimum:</strong>
            <ul className="ml-4 list-disc">
              <li>β̂ = ({payload.map.beta[0].toFixed(2)}, {payload.map.beta[1].toFixed(2)})</li>
              <li>(no uncertainty)</li>
            </ul>
          </div>
        </div>
      )}
      <div className="mt-2 text-xs text-[var(--color-text-muted)] leading-relaxed">
        Bayesian logistic regression on Iris versicolor-vs-virginica (N=100,
        standardized petal-length and petal-width). Three engines fit from
        the same `pm.Model` declaration: NUTS scatter (the gold-standard
        posterior), ADVI 95% credible ellipse (mean-field Gaussian), MAP
        star (the joint mode). Toggle methods on/off to isolate what each
        contributes — NUTS shows the tilted joint geometry, ADVI's
        axis-aligned ellipse misses the tilt by construction (mean-field
        cannot represent cross-coordinate correlation), and MAP collapses
        the entire posterior to a single point with no width. The right
        panel shows the −ELBO loss trajectory across the 20 000 ADVI
        optimization steps; the stable plateau confirms convergence per
        §4.3. The agreement on the means is the §4.5 takeaway: same model,
        three engines, three matching answers, very different costs.
      </div>
    </div>
  );
}
