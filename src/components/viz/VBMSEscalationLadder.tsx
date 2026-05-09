import { useEffect, useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';

// =============================================================================
// VBMSEscalationLadder — §11 four-stage escalation ranking on diabetes.
//
// Loads /sample-data/sparse-bayesian-priors/vbms_escalation.json (produced by
// precompute_vbms_escalation.py). The script ran the full hierarchy:
//   Stage 1: mean-field ADVI (5 restarts)            → ELBO + Pareto-k̂
//   Stage 2: full-rank ADVI (k̂ ≥ 0.5 gate)           → ELBO + Pareto-k̂
//   Stage 3: IWELBO with K=64 (k̂ ≥ 0.5 gate again)   → tighter lower bound
//   Stage 4: SMC log-marginal-likelihood (always run) → AIS-class gold ref
//
// Renders a per-stage ELBO/log-evidence trace (one line per prior) plus a
// Pareto-k̂ overlay against the 0.5 / 0.7 thresholds.
//
// Per CLAUDE.md fetch-based-viz rule: container <div> stays mounted.
// =============================================================================

const PANEL_HEIGHT = 360;
const MARGIN = { top: 24, right: 32, bottom: 60, left: 60 };

type StageKey = 'mean_field_advi' | 'full_rank_advi' | 'iwelbo' | 'smc_ais';

interface StageMeanFieldOrFullRank {
  elbo: number;
  all_elbos: number[];
  pareto_k: number;
}
interface StageIWELBO {
  iwelbo_mean: number;
  iwelbo_se: number;
  K: number;
  S: number;
}
interface StageSMC {
  log_evidence_mean: number;
  log_evidence_se: number;
  n_chains: number;
  n_draws: number;
}

interface PriorRecord {
  name: string;
  color: string;
  stages: {
    mean_field_advi: StageMeanFieldOrFullRank;
    full_rank_advi: StageMeanFieldOrFullRank | null;
    iwelbo: StageIWELBO | null;
    smc_ais: StageSMC | { log_evidence_mean: number; log_evidence_se: number; error?: string };
  };
}

interface Payload {
  metadata: {
    dataset: string;
    n_train: number;
    p: number;
    pareto_k_gate_full_rank: number;
    pareto_k_gate_iwelbo: number;
  };
  priors: PriorRecord[];
}

const STAGES: StageKey[] = ['mean_field_advi', 'full_rank_advi', 'iwelbo', 'smc_ais'];
const STAGE_LABELS: Record<StageKey, string> = {
  mean_field_advi: 'MF ADVI',
  full_rank_advi: 'FR ADVI',
  iwelbo: 'IWELBO',
  smc_ais: 'SMC / AIS',
};

function stageValue(rec: PriorRecord, stage: StageKey): number | null {
  const s = rec.stages[stage];
  if (!s) return null;
  if (stage === 'mean_field_advi' || stage === 'full_rank_advi') {
    return (s as StageMeanFieldOrFullRank).elbo;
  }
  if (stage === 'iwelbo') {
    return (s as StageIWELBO).iwelbo_mean;
  }
  return (s as StageSMC).log_evidence_mean;
}

function stageParetoK(rec: PriorRecord, stage: StageKey): number | null {
  if (stage === 'mean_field_advi' || stage === 'full_rank_advi') {
    const s = rec.stages[stage];
    if (!s) return null;
    return (s as StageMeanFieldOrFullRank).pareto_k;
  }
  return null;
}

export default function VBMSEscalationLadder() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [payload, setPayload] = useState<Payload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showPK, setShowPK] = useState(true);

  useEffect(() => {
    let cancelled = false;
    fetch('/sample-data/sparse-bayesian-priors/vbms_escalation.json')
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json() as Promise<Payload>;
      })
      .then((j) => {
        if (!cancelled) setPayload(j);
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const yDomain = useMemo(() => {
    if (!payload) return [0, 1] as [number, number];
    const all: number[] = [];
    payload.priors.forEach((p) =>
      STAGES.forEach((s) => {
        const v = stageValue(p, s);
        if (v !== null && Number.isFinite(v)) all.push(v);
      }),
    );
    if (all.length === 0) return [0, 1] as [number, number];
    const minV = Math.min(...all);
    const maxV = Math.max(...all);
    const pad = (maxV - minV) * 0.12 + 0.5;
    return [minV - pad, maxV + pad] as [number, number];
  }, [payload]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const w = width || 720;
      const h = PANEL_HEIGHT;
      svg.attr('width', w).attr('height', h);

      if (!payload) {
        svg
          .append('text')
          .attr('x', w / 2)
          .attr('y', h / 2)
          .attr('text-anchor', 'middle')
          .style('font-size', '12px')
          .text(error ?? 'Loading vbms_escalation.json…');
        return;
      }

      const innerW = w - MARGIN.left - MARGIN.right;
      const innerH = h - MARGIN.top - MARGIN.bottom;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const xScale = d3
        .scalePoint<StageKey>()
        .domain(STAGES)
        .range([0, innerW])
        .padding(0.4);

      const elboScale = d3.scaleLinear().domain(yDomain).range([innerH, 0]).nice();
      const pkScale = d3.scaleLinear().domain([0, 1.2]).range([innerH, 0]);

      g.append('g')
        .attr('transform', `translate(0,${innerH})`)
        .call(
          d3.axisBottom(xScale).tickFormat((d) => STAGE_LABELS[d as StageKey]),
        );
      g.append('g').call(d3.axisLeft(elboScale).ticks(6));

      g.append('text')
        .attr('x', -innerH / 2)
        .attr('y', -42)
        .attr('text-anchor', 'middle')
        .attr('transform', 'rotate(-90)')
        .style('font-size', '12px')
        .text('ELBO / IWELBO / log p(y | M)');

      g.append('text')
        .attr('x', innerW / 2)
        .attr('y', innerH + 40)
        .attr('text-anchor', 'middle')
        .style('font-size', '12px')
        .text('escalation stage');

      // Per-prior trajectories
      payload.priors.forEach((p) => {
        const points: { stage: StageKey; v: number }[] = [];
        STAGES.forEach((s) => {
          const v = stageValue(p, s);
          if (v !== null && Number.isFinite(v)) points.push({ stage: s, v });
        });
        if (points.length === 0) return;
        const lineGen = d3
          .line<{ stage: StageKey; v: number }>()
          .x((d) => xScale(d.stage)!)
          .y((d) => elboScale(d.v));
        g.append('path')
          .datum(points)
          .attr('fill', 'none')
          .attr('stroke', p.color)
          .attr('stroke-width', 2.2)
          .attr('d', lineGen);
        points.forEach((pt) => {
          g.append('circle')
            .attr('cx', xScale(pt.stage)!)
            .attr('cy', elboScale(pt.v))
            .attr('r', 4)
            .attr('fill', p.color);
        });
      });

      // Pareto-k̂ overlay (right axis-style, scaled separately)
      if (showPK) {
        const axisRight = d3
          .axisRight(pkScale)
          .ticks(5)
          .tickFormat((d) => (d as number).toFixed(2));
        g.append('g').attr('transform', `translate(${innerW}, 0)`).call(axisRight);
        g.append('text')
          .attr('x', innerH / 2)
          .attr('y', -innerW - 40)
          .attr('transform', 'rotate(90)')
          .attr('text-anchor', 'middle')
          .style('font-size', '11px')
          .style('fill', '#666')
          .text('Pareto-k̂');
        // 0.5 / 0.7 thresholds
        [0.5, 0.7].forEach((thr) => {
          g.append('line')
            .attr('x1', 0)
            .attr('x2', innerW)
            .attr('y1', pkScale(thr))
            .attr('y2', pkScale(thr))
            .attr('stroke', '#aa6633')
            .attr('stroke-dasharray', '3 4')
            .attr('stroke-width', 0.8);
          g.append('text')
            .attr('x', innerW - 6)
            .attr('y', pkScale(thr) - 3)
            .attr('text-anchor', 'end')
            .style('font-size', '10px')
            .style('fill', '#aa6633')
            .text(`k̂ = ${thr}`);
        });
        // PK markers per prior
        payload.priors.forEach((p) => {
          STAGES.forEach((s) => {
            const k = stageParetoK(p, s);
            if (k === null || !Number.isFinite(k)) return;
            g.append('rect')
              .attr('x', (xScale(s) ?? 0) - 5)
              .attr('y', pkScale(k) - 5)
              .attr('width', 10)
              .attr('height', 10)
              .attr('fill', 'none')
              .attr('stroke', p.color)
              .attr('stroke-width', 1.4);
          });
        });
      }

      // Legend
      const legend = g.append('g').attr('transform', `translate(${innerW - 200}, 8)`);
      payload.priors.forEach((p, i) => {
        const row = legend.append('g').attr('transform', `translate(0, ${i * 18})`);
        row
          .append('line')
          .attr('x1', 0)
          .attr('x2', 22)
          .attr('y1', 6)
          .attr('y2', 6)
          .attr('stroke', p.color)
          .attr('stroke-width', 2.2);
        row
          .append('text')
          .attr('x', 28)
          .attr('y', 10)
          .style('font-size', '11px')
          .text(p.name);
      });
    },
    [payload, error, yDomain, showPK, width],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', maxWidth: 760 }}>
      <div style={{ display: 'flex', gap: 18, marginBottom: 8, fontSize: 13, alignItems: 'center' }}>
        <label>
          <input
            type="checkbox"
            checked={showPK}
            onChange={(e) => setShowPK(e.target.checked)}
          />{' '}
          Show Pareto-k̂ overlay
        </label>
      </div>
      <svg ref={svgRef} role="img" aria-label="VBMS escalation hierarchy on diabetes data" />
    </div>
  );
}
