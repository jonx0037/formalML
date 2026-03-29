import { useState, useEffect, useRef, useCallback, useMemo, useId } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';

// ─── Layout constants ───

const SM_BREAKPOINT = 640;
const CHAIN_HEIGHT = 100;
const PANEL_HEIGHT = 220;
const STEP_DELAY = 800;

type CategoryType = 'set' | 'vec' | 'poset';

// ─── Category data ───

const SET_F: Record<string, string> = { '1': 'a', '2': 'b', '3': 'a' };
const SET_G: Record<string, string> = { a: 'x', b: 'z' };
const SET_H: Record<string, string> = { x: '\u2660', y: '\u2665', z: '\u2660' };

const VEC_F = [[1, 1], [0, 1]];
const VEC_G = [[2, 0], [1, 1]];
const VEC_H = [[1, 0], [1, 2]];

function matMul(a: number[][], b: number[][]): number[][] {
  return [
    [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
    [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]],
  ];
}

function matStr(m: number[][]): string {
  return `[${m[0][0]},${m[0][1]}; ${m[1][0]},${m[1][1]}]`;
}

// ─── Colors ───

const OBJ_COLORS = ['#3b82f6', '#8b5cf6', '#f59e0b', '#ef4444'];
const ACTIVE_COLOR = '#f59e0b';
const DONE_COLOR = '#22c55e';
const FUTURE_COLOR = '#94a3b8';

// ─── Component ───

const CHAIN_LABELS: Record<CategoryType, string[]> = {
  set: ['A = {1,2,3}', 'B = {a,b}', 'C = {x,y,z}', 'D = {\u2660,\u2665}'],
  vec: ['A = \u211D\u00B2', 'B = \u211D\u00B2', 'C = \u211D\u00B2', 'D = \u211D\u00B2'],
  poset: ['1', '2', '4', '8'],
};

export default function CompositionExplorer() {
  const { ref: containerRef, width: containerWidth } =
    useResizeObserver<HTMLDivElement>();
  const instanceId = useId().replace(/:/g, '');

  const chainSvgRef = useRef<SVGSVGElement>(null);
  const leftSvgRef = useRef<SVGSVGElement>(null);
  const rightSvgRef = useRef<SVGSVGElement>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const [categoryType, setCategoryType] = useState<CategoryType>('set');
  const [animStep, setAnimStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  const isNarrow = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const panelWidth = useMemo(() => {
    if (!containerWidth) return 300;
    if (isNarrow) return containerWidth - 32;
    return Math.floor((containerWidth - 48) / 2);
  }, [containerWidth, isNarrow]);
  const chainWidth = useMemo(() => {
    if (!containerWidth) return 600;
    return containerWidth - 32;
  }, [containerWidth]);

  // ─── Auto-play ───

  useEffect(() => {
    if (!isPlaying) return;
    if (animStep >= 3) {
      setIsPlaying(false);
      return;
    }
    timerRef.current = setTimeout(() => setAnimStep((s) => s + 1), STEP_DELAY);
    return () => { if (timerRef.current) clearTimeout(timerRef.current); };
  }, [isPlaying, animStep]);

  const handleReset = useCallback(() => {
    setAnimStep(0);
    setIsPlaying(false);
    if (timerRef.current) clearTimeout(timerRef.current);
  }, []);

  const handleStepForward = useCallback(() => {
    if (animStep < 3) setAnimStep((s) => s + 1);
  }, [animStep]);

  const handleCategoryChange = useCallback((cat: CategoryType) => {
    setCategoryType(cat);
    handleReset();
  }, [handleReset]);

  // ─── Chain diagram rendering ───

  useEffect(() => {
    if (!chainSvgRef.current || chainWidth === 0) return;
    const svg = d3.select(chainSvgRef.current);
    svg.selectAll('*').remove();

    const labels = CHAIN_LABELS[categoryType];
    const morphLabels = ['f', 'g', 'h'];
    const spacing = chainWidth / 5;
    const cy = CHAIN_HEIGHT / 2;

    // Arrow marker
    svg.append('defs').append('marker')
      .attr('id', `chain-arrow-${instanceId}`)
      .attr('viewBox', '0 0 10 6').attr('refX', 10).attr('refY', 3)
      .attr('markerWidth', 8).attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path').attr('d', 'M0,0 L10,3 L0,6 Z')
      .style('fill', 'var(--color-text-secondary)');

    for (let i = 0; i < 4; i++) {
      const cx = spacing * (i + 1);
      svg.append('circle').attr('cx', cx).attr('cy', cy).attr('r', 18)
        .style('fill', OBJ_COLORS[i]).style('stroke', 'var(--color-text)').style('stroke-width', '1.5');
      svg.append('text').attr('x', cx).attr('y', cy + 4).attr('text-anchor', 'middle')
        .attr('font-size', 11).attr('font-weight', 600).style('fill', 'white')
        .text(categoryType === 'poset' ? labels[i] : String.fromCharCode(65 + i));
      svg.append('text').attr('x', cx).attr('y', cy + 34).attr('text-anchor', 'middle')
        .attr('font-size', 10).style('fill', 'var(--color-text-secondary)')
        .text(labels[i]);
    }

    for (let i = 0; i < 3; i++) {
      const x1 = spacing * (i + 1) + 22;
      const x2 = spacing * (i + 2) - 22;
      svg.append('line').attr('x1', x1).attr('y1', cy).attr('x2', x2).attr('y2', cy)
        .style('stroke', 'var(--color-text-secondary)').style('stroke-width', '2')
        .attr('marker-end', `url(#chain-arrow-${instanceId})`);
      svg.append('text').attr('x', (x1 + x2) / 2).attr('y', cy - 10)
        .attr('text-anchor', 'middle').attr('font-size', 13).attr('font-style', 'italic')
        .attr('font-weight', 600).style('fill', 'var(--color-text)').text(morphLabels[i]);
    }
  }, [chainWidth, categoryType]);

  // ─── Step panel rendering helper ───

  const renderStepPanel = useCallback(
    (svgEl: SVGSVGElement | null, side: 'left' | 'right') => {
      if (!svgEl || panelWidth === 0) return;
      const svg = d3.select(svgEl);
      svg.selectAll('*').remove();

      const w = panelWidth;
      const title = side === 'left' ? 'h \u2218 (g \u2218 f)' : '(h \u2218 g) \u2218 f';

      // Title
      svg.append('text').attr('x', w / 2).attr('y', 20).attr('text-anchor', 'middle')
        .attr('font-size', 14).attr('font-weight', 700).style('fill', 'var(--color-text)')
        .text(title);

      // Determine which sub-step is active per side
      // Left: step1 = g∘f, step2 = h∘(g∘f)
      // Right: step1 = h∘g, step2 = (h∘g)∘f
      const stepRows: { label: string; detail: string; stepIdx: number }[] =
        side === 'left'
          ? [
              { label: 'Step 1: g \u2218 f', detail: detailText(categoryType, 'gf'), stepIdx: 1 },
              { label: 'Step 2: h \u2218 (g\u2218f)', detail: detailText(categoryType, 'h_gf'), stepIdx: 2 },
            ]
          : [
              { label: 'Step 1: h \u2218 g', detail: detailText(categoryType, 'hg'), stepIdx: 1 },
              { label: 'Step 2: (h\u2218g) \u2218 f', detail: detailText(categoryType, 'hg_f'), stepIdx: 2 },
            ];

      let y = 44;
      for (const row of stepRows) {
        const isActive = animStep === row.stepIdx;
        const isDone = animStep > row.stepIdx;
        const color = isActive ? ACTIVE_COLOR : isDone ? DONE_COLOR : FUTURE_COLOR;

        svg.append('rect').attr('x', 12).attr('y', y - 12).attr('width', w - 24).attr('height', 52)
          .attr('rx', 6)
          .style('fill', isActive ? 'rgba(245,158,11,0.1)' : 'transparent')
          .style('stroke', color).style('stroke-width', isActive ? '2' : '1');
        svg.append('text').attr('x', 22).attr('y', y + 4).attr('font-size', 12)
          .attr('font-weight', 600).style('fill', color).text(row.label);
        svg.append('text').attr('x', 22).attr('y', y + 22).attr('font-size', 10)
          .style('fill', isDone || isActive ? 'var(--color-text)' : FUTURE_COLOR)
          .text(row.detail);
        if (isDone) {
          svg.append('text').attr('x', w - 30).attr('y', y + 10).attr('font-size', 16)
            .attr('text-anchor', 'middle').style('fill', DONE_COLOR).text('\u2713');
        }
        y += 62;
      }

      // Final result row
      if (animStep >= 3) {
        svg.append('rect').attr('x', 12).attr('y', y - 12).attr('width', w - 24).attr('height', 44)
          .attr('rx', 6)
          .style('fill', 'rgba(34,197,94,0.12)')
          .style('stroke', DONE_COLOR).style('stroke-width', '2');
        svg.append('text').attr('x', w / 2).attr('y', y + 6).attr('text-anchor', 'middle')
          .attr('font-size', 13).attr('font-weight', 700).style('fill', DONE_COLOR)
          .text(`Result: ${finalResult(categoryType)}`);
        svg.append('text').attr('x', w / 2).attr('y', y + 22).attr('text-anchor', 'middle')
          .attr('font-size', 10).style('fill', DONE_COLOR).text('Both paths agree!');
      }
    },
    [animStep, categoryType, panelWidth],
  );

  useEffect(() => {
    renderStepPanel(leftSvgRef.current, 'left');
    renderStepPanel(rightSvgRef.current, 'right');
  }, [renderStepPanel]);

  // ─── Render ───

  return (
    <div
      ref={containerRef}
      className="w-full rounded-lg border p-4"
      style={{ borderColor: 'var(--color-border)', backgroundColor: 'var(--color-surface)' }}
    >
      {/* Controls */}
      <div className="mb-4 flex flex-wrap items-center gap-3">
        <span className="text-sm font-medium" style={{ color: 'var(--color-text)' }}>Category:</span>
        {(['set', 'vec', 'poset'] as CategoryType[]).map((cat) => (
          <button
            key={cat}
            className="rounded border px-3 py-1 text-sm font-medium"
            style={{
              borderColor: 'var(--color-border)',
              backgroundColor: categoryType === cat ? 'var(--color-text)' : 'var(--color-bg)',
              color: categoryType === cat ? 'var(--color-bg)' : 'var(--color-text)',
            }}
            onClick={() => handleCategoryChange(cat)}
          >
            {cat === 'set' ? 'Set' : cat === 'vec' ? 'Vec' : 'Poset'}
          </button>
        ))}

        <div style={{ width: 1, height: 20, backgroundColor: 'var(--color-border)' }} />

        <button
          className="rounded border px-3 py-1 text-sm font-medium"
          style={{
            borderColor: 'var(--color-border)',
            backgroundColor: isPlaying ? ACTIVE_COLOR : 'var(--color-bg)',
            color: isPlaying ? 'white' : 'var(--color-text)',
          }}
          onClick={() => {
            if (animStep >= 3) handleReset();
            setIsPlaying((p) => !p);
          }}
        >
          {isPlaying ? 'Pause' : 'Play'}
        </button>
        <button
          className="rounded border px-3 py-1 text-sm font-medium"
          style={{
            borderColor: 'var(--color-border)',
            backgroundColor: 'var(--color-bg)',
            color: 'var(--color-text)',
          }}
          onClick={handleReset}
        >
          Reset
        </button>
        <button
          className="rounded border px-3 py-1 text-sm font-medium"
          style={{
            borderColor: 'var(--color-border)',
            backgroundColor: 'var(--color-bg)',
            color: animStep >= 3 ? FUTURE_COLOR : 'var(--color-text)',
          }}
          onClick={handleStepForward}
          disabled={animStep >= 3}
        >
          Step &rarr;
        </button>
        <span className="text-xs" style={{ color: 'var(--color-text-secondary)' }}>
          Step {animStep} / 3
        </span>
      </div>

      {/* Chain diagram */}
      <svg role="img" aria-label="Composition explorer visualization (panel 1 of 3)"
        ref={chainSvgRef}
        width={chainWidth}
        height={CHAIN_HEIGHT}
        style={{
          border: '1px solid var(--color-border)',
          borderRadius: 8,
          backgroundColor: 'var(--color-bg)',
        }}
      />

      {/* Composition panels */}
      <div
        className={isNarrow ? 'mt-4 flex flex-col gap-4' : 'mt-4 flex gap-4'}
        style={{ alignItems: 'flex-start' }}
      >
        <svg role="img" aria-label="Composition explorer visualization (panel 2 of 3)"
          ref={leftSvgRef}
          width={panelWidth}
          height={PANEL_HEIGHT}
          style={{
            border: '1px solid var(--color-border)',
            borderRadius: 8,
            backgroundColor: 'var(--color-bg)',
          }}
        />
        {animStep >= 3 && (
          <div
            className="flex items-center justify-center self-center text-2xl font-bold"
            style={{ color: DONE_COLOR, minWidth: 24 }}
          >
            =
          </div>
        )}
        <svg role="img" aria-label="Composition explorer visualization (panel 3 of 3)"
          ref={rightSvgRef}
          width={panelWidth}
          height={PANEL_HEIGHT}
          style={{
            border: '1px solid var(--color-border)',
            borderRadius: 8,
            backgroundColor: 'var(--color-bg)',
          }}
        />
      </div>

      {/* Legend */}
      <div className="mt-3 text-xs" style={{ color: 'var(--color-text-secondary)' }}>
        <span className="mr-4"><strong>Associativity:</strong> h \u2218 (g \u2218 f) = (h \u2218 g) \u2218 f</span>
        <span>Step through both parenthesizations to see they produce the same composite morphism.</span>
      </div>
    </div>
  );
}

// ─── Detail text helpers ───

function detailText(cat: CategoryType, key: string): string {
  if (cat === 'set') {
    const gf = composeMap(SET_F, SET_G);
    const hg = composeMap(SET_G, SET_H);
    if (key === 'gf') return `g\u2218f: ${mapStr(gf)}`;
    if (key === 'h_gf') return `h\u2218(g\u2218f): ${mapStr(composeMap(gf, SET_H))}`;
    if (key === 'hg') return `h\u2218g: ${mapStr(hg)}`;
    if (key === 'hg_f') return `(h\u2218g)\u2218f: ${mapStr(composeMap(SET_F, hg))}`;
  }
  if (cat === 'vec') {
    const gf = matMul(VEC_G, VEC_F);
    const hg = matMul(VEC_H, VEC_G);
    if (key === 'gf') return `g\u00B7f = ${matStr(gf)}`;
    if (key === 'h_gf') return `h\u00B7(g\u00B7f) = ${matStr(matMul(VEC_H, gf))}`;
    if (key === 'hg') return `h\u00B7g = ${matStr(hg)}`;
    if (key === 'hg_f') return `(h\u00B7g)\u00B7f = ${matStr(matMul(hg, VEC_F))}`;
  }
  if (cat === 'poset') {
    if (key === 'gf') return '1|2 and 2|4 \u21D2 1|4';
    if (key === 'h_gf') return '1|4 and 4|8 \u21D2 1|8';
    if (key === 'hg') return '2|4 and 4|8 \u21D2 2|8';
    if (key === 'hg_f') return '1|2 and 2|8 \u21D2 1|8';
  }
  return '';
}

function finalResult(cat: CategoryType): string {
  if (cat === 'set') {
    const full = composeMap(composeMap(SET_F, SET_G), SET_H);
    return mapStr(full);
  }
  if (cat === 'vec') return matStr(matMul(VEC_H, matMul(VEC_G, VEC_F)));
  return '1|8';
}

function composeMap(f: Record<string, string>, g: Record<string, string>): Record<string, string> {
  const result: Record<string, string> = {};
  for (const [k, v] of Object.entries(f)) {
    if (v in g) result[k] = g[v];
  }
  return result;
}

function mapStr(m: Record<string, string>): string {
  return Object.entries(m).map(([k, v]) => `${k}\u2192${v}`).join(', ');
}
