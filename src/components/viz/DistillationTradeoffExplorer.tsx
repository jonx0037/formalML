import { useEffect, useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { paletteStacking } from './shared/bayesian-ml-stacking';

// =============================================================================
// DistillationTradeoffExplorer — embedded alongside §5.3.
// Two-panel display: (a) teacher and student posterior predictive bands at the
// §1 dense grid; (b) held-out log-score and inference-cost bars. Loads
// distillation_grid.json with five precomputed student types × four dataset
// sizes. Shows the deployment-time tradeoff: small loss in predictive quality,
// large gain in latency.
// =============================================================================

const PANEL_HEIGHT = 360;
const STUDENT_TYPES = ['linear', 'poly-3', 'poly-6', 'poly-10', 'mlp'] as const;
const DATASET_SIZES = [50, 100, 200, 400] as const;

type StudentType = (typeof STUDENT_TYPES)[number];

interface DistillationData {
  x_eval: number[];
  teacher: { mean: number[]; lower_95: number[]; upper_95: number[]; log_score: number; inference_cost: number };
  best_single: { name: string; log_score: number };
  students: Record<StudentType, Record<string, { mean: number[]; lower_95: number[]; upper_95: number[]; log_score: number; inference_cost: number }>>;
}

export default function DistillationTradeoffExplorer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();

  const [data, setData] = useState<DistillationData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [studentType, setStudentType] = useState<StudentType>('poly-6');
  const [datasetSize, setDatasetSize] = useState<number>(200);

  useEffect(() => {
    let cancelled = false;
    fetch('/sample-data/stacking-and-predictive-ensembles/distillation_grid.json')
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((j: DistillationData) => {
        if (!cancelled) setData(j);
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const student = useMemo(() => {
    if (!data) return null;
    return data.students[studentType]?.[String(datasetSize)] ?? null;
  }, [data, studentType, datasetSize]);

  const ref = useD3(
    (svg) => {
      const w = width || 720;
      const height = PANEL_HEIGHT;
      svg.attr('width', w).attr('height', height);
      svg.selectAll('*').remove();

      if (!data || !student) {
        svg
          .append('text')
          .attr('x', w / 2)
          .attr('y', height / 2)
          .attr('text-anchor', 'middle')
          .attr('font-size', 12)
          .attr('fill', '#666')
          .text(error ? `Awaiting precomputed data — ${error}` : 'Loading distillation grid...');
        return;
      }

      const margin = { top: 24, right: 16, bottom: 40, left: 48 };
      const innerW = (w - margin.left - margin.right - 24) / 2;
      const innerH = height - margin.top - margin.bottom;

      // Left panel: predictive bands.
      const left = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
      const xs = data.x_eval;
      const xS = d3.scaleLinear().domain([0, 1]).range([0, innerW]);
      const yS = d3.scaleLinear().domain([-2, 2]).range([innerH, 0]);
      left.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xS).ticks(5));
      left.append('g').call(d3.axisLeft(yS).ticks(5));

      const band = (mean: number[], lo: number[], hi: number[], color: string, label: string) => {
        const area = d3
          .area<number>()
          .x((_, i) => xS(xs[i]))
          .y0((_, i) => yS(lo[i]))
          .y1((_, i) => yS(hi[i]));
        left
          .append('path')
          .datum(mean)
          .attr('fill', color)
          .attr('fill-opacity', 0.2)
          .attr('d', area);
        const line = d3
          .line<number>()
          .x((_, i) => xS(xs[i]))
          .y((d) => yS(d));
        left
          .append('path')
          .datum(mean)
          .attr('fill', 'none')
          .attr('stroke', color)
          .attr('stroke-width', 2.0)
          .attr('d', line);
        return label;
      };

      band(data.teacher.mean, data.teacher.lower_95, data.teacher.upper_95, paletteStacking.stacking, 'teacher');
      band(student.mean, student.lower_95, student.upper_95, paletteStacking.bma, 'student');

      // Discontinuity marker.
      left.append('line').attr('x1', xS(0.5)).attr('x2', xS(0.5)).attr('y1', 0).attr('y2', innerH).attr('stroke', '#888').attr('stroke-dasharray', '3 3').attr('stroke-width', 0.6);

      // Legend.
      left.append('rect').attr('x', innerW - 110).attr('y', 4).attr('width', 14).attr('height', 3).attr('fill', paletteStacking.stacking);
      left.append('text').attr('x', innerW - 92).attr('y', 8).attr('font-size', 10).text('teacher (stacked)');
      left.append('rect').attr('x', innerW - 110).attr('y', 18).attr('width', 14).attr('height', 3).attr('fill', paletteStacking.bma);
      left.append('text').attr('x', innerW - 92).attr('y', 22).attr('font-size', 10).text(`student (${studentType})`);

      left.append('text').attr('transform', `translate(${innerW / 2},${innerH + 28})`).attr('text-anchor', 'middle').attr('font-size', 11).text('x');
      left.append('text').attr('x', innerW / 2).attr('y', -8).attr('text-anchor', 'middle').attr('font-size', 12).attr('font-weight', 600).text('Predictive bands');

      // Right panel: bars (log-score and inference cost). The legend color for
      // the "best single" candidate is derived from its name so it stays
      // semantically correct if the best model isn't BART (e.g. on a different
      // DGP where GP or polynomial wins). Bar baselines are anchored to the
      // y-scale domain minimum, not the score-array order, so heights stay
      // correct regardless of which candidate is best.
      const candidateColorByName: Record<string, string> = {
        BLR: paletteStacking.blr,
        'BPR-4': paletteStacking.bpr,
        BPR: paletteStacking.bpr,
        GP: paletteStacking.gp,
        BART: paletteStacking.bart,
      };
      const bestColor = candidateColorByName[data.best_single.name] ?? paletteStacking.oracle;

      const right = svg.append('g').attr('transform', `translate(${margin.left + innerW + 24},${margin.top})`);
      const items = [
        { label: 'Teacher', score: data.teacher.log_score, cost: data.teacher.inference_cost, color: paletteStacking.stacking },
        { label: 'Student', score: student.log_score, cost: student.inference_cost, color: paletteStacking.bma },
        { label: `Best (${data.best_single.name})`, score: data.best_single.log_score, cost: 1, color: bestColor },
      ];
      const xB = d3.scaleBand<string>().domain(items.map((d) => d.label)).range([0, innerW]).padding(0.22);
      const scores = items.map((d) => d.score);
      const yMin = Math.min(...scores) - 0.05;
      const yMax = Math.max(...scores) + 0.02;
      const yB = d3.scaleLinear().domain([yMin, yMax]).range([innerH * 0.45, 0]);
      const yBaseline = yB(yMin);
      right.append('g').attr('transform', `translate(0,${innerH * 0.45})`).call(d3.axisBottom(xB));
      right.append('g').call(d3.axisLeft(yB).ticks(4));
      right.append('text').attr('x', innerW / 2).attr('y', -8).attr('text-anchor', 'middle').attr('font-size', 12).attr('font-weight', 600).text('Held-out log-score / obs');

      right
        .selectAll('rect.score')
        .data(items)
        .join('rect')
        .attr('class', 'score')
        .attr('x', (d) => xB(d.label) ?? 0)
        .attr('width', xB.bandwidth())
        .attr('y', (d) => yB(d.score))
        .attr('height', (d) => yBaseline - yB(d.score))
        .attr('fill', (d) => d.color);

      // Inference cost row below.
      const yC = d3.scaleLinear().domain([0, Math.max(...items.map((d) => d.cost))]).range([innerH * 0.4, 0]);
      const costRow = right.append('g').attr('transform', `translate(0,${innerH * 0.55})`);
      costRow.append('g').attr('transform', `translate(0,${innerH * 0.4})`).call(d3.axisBottom(xB));
      costRow.append('g').call(d3.axisLeft(yC).ticks(4));
      costRow.append('text').attr('x', innerW / 2).attr('y', -8).attr('text-anchor', 'middle').attr('font-size', 12).attr('font-weight', 600).text('Inference cost (forward passes)');
      costRow
        .selectAll('rect.cost')
        .data(items)
        .join('rect')
        .attr('class', 'cost')
        .attr('x', (d) => xB(d.label) ?? 0)
        .attr('width', xB.bandwidth())
        .attr('y', (d) => yC(d.cost))
        .attr('height', (d) => innerH * 0.4 - yC(d.cost))
        .attr('fill', (d) => d.color);
    },
    [data, error, student, studentType, width],
  );

  return (
    <div className="my-6 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-soft)] p-4">
      <div className="mb-3 flex flex-wrap gap-3 text-sm">
        <label>
          student type:&nbsp;
          <select value={studentType} onChange={(e) => setStudentType(e.target.value as StudentType)} className="rounded border px-1">
            {STUDENT_TYPES.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </label>
        <label>
          dataset size:&nbsp;
          <select value={datasetSize} onChange={(e) => setDatasetSize(Number(e.target.value))} className="rounded border px-1">
            {DATASET_SIZES.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div ref={containerRef} className="w-full">
        <svg ref={ref} />
      </div>
    </div>
  );
}
