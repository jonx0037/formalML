import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  activations,
  effectiveRank,
  fillIsotropicGaussian,
  gaussianRng,
  legendreVandermonde,
  mulberry32,
  randomFeatureMap,
  thinSVD,
} from './shared/double-descent';

// =============================================================================
// EffectiveDimensionExplorer — §11.1 (Figure 12)
//
// Two-panel layout:
//   - left: eigenvalue spectrum on log scale (sorted descending) for three
//     feature distributions (isotropic Gaussian, ReLU random features, Legendre)
//   - right: effective rank as a function of the rcond threshold
// Static fallback: 12_eigenvalue_spectra_two_panel.png
// =============================================================================

const HEIGHT = 460;
const FEATURE_NAMES = ['isotropic', 'relu-rf', 'legendre'] as const;
type FeatureName = (typeof FEATURE_NAMES)[number];
const FEATURE_LABEL: Record<FeatureName, string> = {
  isotropic: 'isotropic Gaussian',
  'relu-rf': 'ReLU random features',
  legendre: 'Legendre polynomial',
};
const FEATURE_COLOR: Record<FeatureName, string> = {
  isotropic: 'var(--color-accent)',
  'relu-rf': '#D97706',
  legendre: '#534AB7',
};
const RCOND_GRID = (() => {
  const out: number[] = [];
  for (let e = -16; e <= 0; e += 0.25) out.push(Math.pow(10, e));
  return out;
})();

function buildDesign(feature: FeatureName, n: number, p: number, rngSeed: number): Float64Array {
  const rng = mulberry32(rngSeed);
  const gauss = gaussianRng(rng);
  if (feature === 'isotropic') {
    const X = new Float64Array(n * p);
    fillIsotropicGaussian(X, n, p, gauss);
    return X;
  }
  if (feature === 'relu-rf') {
    const d = 8;
    const Xin = new Float64Array(n * d);
    fillIsotropicGaussian(Xin, n, d, gauss);
    const W = new Float64Array(p * d);
    fillIsotropicGaussian(W, p, d, gauss);
    return randomFeatureMap(Xin, n, d, W, p, activations.relu);
  }
  // legendre
  const Xpts = new Float64Array(n);
  for (let i = 0; i < n; i++) Xpts[i] = 2 * rng() - 1;
  return legendreVandermonde(Xpts, p);
}

export default function EffectiveDimensionExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [nCommitted, setNCommitted] = useState(50);
  const [nDisplay, setNDisplay] = useState(50);
  const [pCommitted, setPCommitted] = useState(100);
  const [pDisplay, setPDisplay] = useState(100);
  const [rcondLog, setRcondLog] = useState(-10);
  const [selectedFeatures, setSelectedFeatures] = useState<FeatureName[]>([...FEATURE_NAMES]);

  const spectra = useMemo(() => {
    const n = nCommitted;
    const p = pCommitted;
    const r = Math.min(n, p);
    const result: { name: FeatureName; S: Float64Array; sortedDesc: Float64Array }[] = [];
    for (const name of FEATURE_NAMES) {
      const X = buildDesign(name, n, p, 2024 + name.length * 11);
      const svd = thinSVD(X, n, p);
      // Float64Array.sort() defaults to numeric (unlike Array.sort), so .slice().sort()
      // is the fast in-place ascending sort. Reverse it for descending.
      const sortedDesc = svd.S.slice().sort().reverse();
      result.push({ name, S: svd.S, sortedDesc });
    }
    return result;
  }, [nCommitted, pCommitted]);

  const rcondCurves = useMemo(() => {
    return spectra.map((s) => ({
      name: s.name,
      data: RCOND_GRID.map((rc) => ({ rcond: rc, rank: effectiveRank(s.S, rc) })),
    }));
  }, [spectra]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const margin = { top: 28, right: 16, bottom: 56, left: 56 };
      const gap = 32;
      const panelW = (w - margin.left - margin.right - gap) / 2;
      const H = HEIGHT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (panelW <= 0) return;
      svg.attr('viewBox', `0 0 ${w} ${HEIGHT}`);

      // Left panel: spectra
      const gL = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
      const allS = spectra.flatMap((s) => Array.from(s.sortedDesc).filter((v) => v > 0));
      const yMinL = Math.max(1e-12, d3.min(allS) ?? 1e-10);
      const yMaxL = d3.max(allS) ?? 10;
      const xScaleL = d3.scaleLinear().domain([1, Math.min(nCommitted, pCommitted)]).range([0, panelW]);
      const yScaleL = d3.scaleLog().domain([yMinL * 0.7, yMaxL * 1.4]).range([H, 0]);
      gL.append('g').attr('transform', `translate(0,${H})`).call(d3.axisBottom(xScaleL).ticks(6))
        .selectAll('text').style('fill', 'var(--color-text)');
      gL.append('g').call(d3.axisLeft(yScaleL).ticks(6, '~g'))
        .selectAll('text').style('fill', 'var(--color-text)');
      gL.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');
      gL.append('text').attr('x', panelW / 2).attr('y', -8).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)').style('font-size', '12px').style('font-weight', '600')
        .text('singular value spectrum (sorted)');
      gL.append('text').attr('x', panelW / 2).attr('y', H + 36).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)').style('font-size', '12px').text('index k');
      gL.append('text').attr('transform', 'rotate(-90)').attr('x', -H / 2).attr('y', -42)
        .attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '11px')
        .text('s_k (log)');

      const lineGen = d3.line<[number, number]>().x((d) => xScaleL(d[0]))
        .y((d) => yScaleL(Math.max(yMinL * 0.7, d[1]))).defined((d) => d[1] > 0);
      spectra.filter((s) => selectedFeatures.includes(s.name)).forEach((s) => {
        const data: [number, number][] = Array.from(s.sortedDesc).map((v, i) => [i + 1, v]);
        gL.append('path').datum(data).attr('d', lineGen).attr('fill', 'none')
          .style('stroke', FEATURE_COLOR[s.name]).attr('stroke-width', 2);
      });

      // Right panel: rcond → effective rank
      const offR = margin.left + panelW + gap;
      const gR = svg.append('g').attr('transform', `translate(${offR},${margin.top})`);
      const xScaleR = d3.scaleLog().domain([1e-16, 1]).range([0, panelW]);
      const yScaleR = d3.scaleLinear().domain([0, Math.min(nCommitted, pCommitted) * 1.05]).range([H, 0]);
      gR.append('g').attr('transform', `translate(0,${H})`).call(d3.axisBottom(xScaleR).ticks(6, '~g'))
        .selectAll('text').style('fill', 'var(--color-text)');
      gR.append('g').call(d3.axisLeft(yScaleR).ticks(6))
        .selectAll('text').style('fill', 'var(--color-text)');
      gR.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');
      gR.append('text').attr('x', panelW / 2).attr('y', -8).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)').style('font-size', '12px').style('font-weight', '600')
        .text('effective rank vs rcond');
      gR.append('text').attr('x', panelW / 2).attr('y', H + 36).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)').style('font-size', '12px').text('rcond (log)');
      gR.append('text').attr('transform', 'rotate(-90)').attr('x', -H / 2).attr('y', -42)
        .attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '11px')
        .text('rk_eff');

      const lineGen2 = d3.line<{ rcond: number; rank: number }>().x((d) => xScaleR(d.rcond))
        .y((d) => yScaleR(d.rank));
      rcondCurves.filter((c) => selectedFeatures.includes(c.name)).forEach((c) => {
        gR.append('path').datum(c.data).attr('d', lineGen2).attr('fill', 'none')
          .style('stroke', FEATURE_COLOR[c.name]).attr('stroke-width', 2);
      });

      // rcond slider marker
      const rc = Math.pow(10, rcondLog);
      gR.append('line').attr('x1', xScaleR(rc)).attr('x2', xScaleR(rc)).attr('y1', 0).attr('y2', H)
        .style('stroke', 'var(--color-accent)').attr('stroke-dasharray', '4 4').attr('opacity', 0.7);
      gR.append('text').attr('x', xScaleR(rc) + 4).attr('y', 14)
        .style('fill', 'var(--color-accent)').style('font-size', '10px')
        .text(`rcond = 10^${rcondLog.toFixed(1)}`);

      // Legend (left panel)
      let ly = 14;
      for (const name of FEATURE_NAMES) {
        const active = selectedFeatures.includes(name);
        gL.append('line').attr('x1', panelW - 130).attr('x2', panelW - 110).attr('y1', ly).attr('y2', ly)
          .style('stroke', FEATURE_COLOR[name]).attr('stroke-width', 2).attr('opacity', active ? 1 : 0.25);
        gL.append('text').attr('x', panelW - 104).attr('y', ly + 4)
          .style('fill', 'var(--color-text)').style('font-size', '11px').attr('opacity', active ? 1 : 0.4)
          .text(FEATURE_LABEL[name]);
        ly += 16;
      }
    },
    [containerWidth, spectra, rcondCurves, nCommitted, pCommitted, rcondLog, selectedFeatures],
  );

  const toggleFeature = (n: FeatureName) => {
    setSelectedFeatures((prev) => prev.includes(n) ? prev.filter((x) => x !== n) : [...prev, n]);
  };

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: 'var(--font-sans)' }}>
      <svg ref={svgRef} width="100%" height={HEIGHT} role="img"
        aria-label="Eigenvalue spectra and effective rank as a function of rcond for three feature distributions." />
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', marginTop: '0.75rem',
        fontSize: '13px', color: 'var(--color-text)' }}>
        <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', minWidth: 180 }}>
          <span>n: <strong>{nDisplay}</strong></span>
          <input type="range" min={20} max={100} step={5} value={nDisplay}
            onChange={(e) => setNDisplay(parseInt(e.target.value, 10))}
            onMouseUp={() => setNCommitted(nDisplay)} onTouchEnd={() => setNCommitted(nDisplay)}
            onKeyUp={() => setNCommitted(nDisplay)} aria-label="Sample size n" />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', minWidth: 180 }}>
          <span>p: <strong>{pDisplay}</strong></span>
          <input type="range" min={20} max={200} step={5} value={pDisplay}
            onChange={(e) => setPDisplay(parseInt(e.target.value, 10))}
            onMouseUp={() => setPCommitted(pDisplay)} onTouchEnd={() => setPCommitted(pDisplay)}
            onKeyUp={() => setPCommitted(pDisplay)} aria-label="Feature dimension p" />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', minWidth: 220 }}>
          <span>rcond marker: <strong>10^{rcondLog.toFixed(1)}</strong></span>
          <input type="range" min={-16} max={0} step={0.25} value={rcondLog}
            onChange={(e) => setRcondLog(parseFloat(e.target.value))} aria-label="rcond threshold marker" />
        </label>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
          {FEATURE_NAMES.map((n) => (
            <label key={n} style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <input type="checkbox" checked={selectedFeatures.includes(n)} onChange={() => toggleFeature(n)} />
              <span style={{ color: FEATURE_COLOR[n] }}>{FEATURE_LABEL[n]}</span>
            </label>
          ))}
        </div>
      </div>
    </div>
  );
}
