import { useState, useMemo, useEffect, useRef, useId } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  presetVec, presetSet, triangleCategory,
  checkFunctorIdentity, checkFunctorComposition,
  type Category, type Functor, type Morphism,
} from './shared/categoryTheory';
import type { CategoryObject } from './shared/types';

// ─── Layout ───

const SM_BREAKPOINT = 640;
const SVG_H = 340;
const PAD = 24;
const R = 18;
const SRC_CLR = '#3b82f6';
const TGT_CLR = '#8b5cf6';
const MAP_CLR = '#9ca3af';

// ─── Types ───

type Obj = CategoryObject & { x: number; y: number };
type Mor = Morphism;
interface Preset {
  name: string; srcLabel: string; tgtLabel: string;
  srcObjs: Obj[]; srcMors: Mor[]; tgtObjs: Obj[]; tgtMors: Mor[];
  objMap: Record<string, string>; morMap: Record<string, string>;
  contra: boolean; build: () => Functor;
}

// ─── Helpers to build morphism arrays compactly ───

const ids = (objs: Obj[]): Mor[] =>
  objs.map((o) => ({ label: `id_${o.label}`, source: o.label, target: o.label, isIdentity: true }));

const mor = (l: string, s: string, t: string): Mor =>
  ({ label: l, source: s, target: t, isIdentity: false });

function makeCat(objs: string[], mors: Mor[], comp: Map<string, string>): Category {
  return {
    objects: objs, morphisms: mors,
    compose: (g, f) => comp.get(`${g},${f}`) ?? null,
    identity: (o) => `id_${o}`,
  };
}

// ─── Preset data ───

const vecObjs: Obj[] = [{ label: 'ℝ', x: .25, y: .25 }, { label: 'ℝ²', x: .75, y: .25 }, { label: 'ℝ³', x: .5, y: .8 }];
const setObjs: Obj[] = [{ label: '{1,2}', x: .25, y: .25 }, { label: '{a,b,c}', x: .75, y: .25 }, { label: '{x}', x: .5, y: .8 }];
const triObjs: Obj[] = [{ label: 'A', x: .2, y: .25 }, { label: 'B', x: .8, y: .25 }, { label: 'C', x: .5, y: .8 }];

const vecMors: Mor[] = [...ids(vecObjs), mor('T', 'ℝ', 'ℝ²'), mor('S', 'ℝ²', 'ℝ³'), mor('S∘T', 'ℝ', 'ℝ³')];
const setMors: Mor[] = [...ids(setObjs), mor('f', '{1,2}', '{a,b,c}'), mor('g', '{a,b,c}', '{x}'), mor('g∘f', '{1,2}', '{x}')];
const triMors: Mor[] = [...ids(triObjs), mor('f', 'A', 'B'), mor('g', 'B', 'C'), mor('g∘f', 'A', 'C')];

const PRESETS: Preset[] = [
  { // Forgetful: Vec → Set
    name: 'Forgetful: Vec → Set', srcLabel: 'Vec', tgtLabel: 'Set',
    srcObjs: vecObjs, srcMors: vecMors, tgtObjs: setObjs, tgtMors: setMors,
    objMap: { 'ℝ': '{1,2}', 'ℝ²': '{a,b,c}', 'ℝ³': '{x}' },
    morMap: { 'id_ℝ': 'id_{1,2}', 'id_ℝ²': 'id_{a,b,c}', 'id_ℝ³': 'id_{x}', 'T': 'f', 'S': 'g', 'S∘T': 'g∘f' },
    contra: false,
    build: () => ({ source: presetVec(), target: presetSet(), contravariant: false,
      onObjects: new Map([['ℝ', '{1,2}'], ['ℝ²', '{a,b,c}'], ['ℝ³', '{x}']]),
      onMorphisms: new Map([['id_ℝ', 'id_{1,2}'], ['id_ℝ²', 'id_{a,b,c}'], ['id_ℝ³', 'id_{x}'], ['T', 'f'], ['S', 'g'], ['S∘T', 'g∘f']]),
    }),
  },
  { // Free: Set → Vec
    name: 'Free: Set → Vec', srcLabel: 'Set', tgtLabel: 'Vec',
    srcObjs: setObjs, srcMors: setMors,
    tgtObjs: [{ label: 'ℝ²', x: .25, y: .25 }, { label: 'ℝ³', x: .75, y: .25 }, { label: 'ℝ', x: .5, y: .8 }],
    tgtMors: [
      { label: 'id_ℝ²', source: 'ℝ²', target: 'ℝ²', isIdentity: true },
      { label: 'id_ℝ³', source: 'ℝ³', target: 'ℝ³', isIdentity: true },
      { label: 'id_ℝ', source: 'ℝ', target: 'ℝ', isIdentity: true },
      mor('T', 'ℝ²', 'ℝ³'), mor('S', 'ℝ³', 'ℝ'), mor('S∘T', 'ℝ²', 'ℝ'),
    ],
    objMap: { '{1,2}': 'ℝ²', '{a,b,c}': 'ℝ³', '{x}': 'ℝ' },
    morMap: { 'id_{1,2}': 'id_ℝ²', 'id_{a,b,c}': 'id_ℝ³', 'id_{x}': 'id_ℝ', 'f': 'T', 'g': 'S', 'g∘f': 'S∘T' },
    contra: false,
    build: () => {
      const comp = new Map([['id_ℝ²,id_ℝ²', 'id_ℝ²'], ['id_ℝ³,id_ℝ³', 'id_ℝ³'], ['id_ℝ,id_ℝ', 'id_ℝ'],
        ['T,id_ℝ²', 'T'], ['id_ℝ³,T', 'T'], ['S,id_ℝ³', 'S'], ['id_ℝ,S', 'S'],
        ['S,T', 'S∘T'], ['S∘T,id_ℝ²', 'S∘T'], ['id_ℝ,S∘T', 'S∘T']]);
      const tgt = makeCat(['ℝ²', 'ℝ³', 'ℝ'], [
        { label: 'id_ℝ²', source: 'ℝ²', target: 'ℝ²', isIdentity: true },
        { label: 'id_ℝ³', source: 'ℝ³', target: 'ℝ³', isIdentity: true },
        { label: 'id_ℝ', source: 'ℝ', target: 'ℝ', isIdentity: true },
        mor('T', 'ℝ²', 'ℝ³'), mor('S', 'ℝ³', 'ℝ'), mor('S∘T', 'ℝ²', 'ℝ'),
      ], comp);
      return { source: presetSet(), target: tgt, contravariant: false,
        onObjects: new Map([['{1,2}', 'ℝ²'], ['{a,b,c}', 'ℝ³'], ['{x}', 'ℝ']]),
        onMorphisms: new Map([['id_{1,2}', 'id_ℝ²'], ['id_{a,b,c}', 'id_ℝ³'], ['id_{x}', 'id_ℝ'], ['f', 'T'], ['g', 'S'], ['g∘f', 'S∘T']]),
      };
    },
  },
  { // Identity: C → C
    name: 'Identity: C → C', srcLabel: 'C', tgtLabel: 'C',
    srcObjs: triObjs, srcMors: triMors, tgtObjs: triObjs, tgtMors: triMors,
    objMap: { A: 'A', B: 'B', C: 'C' },
    morMap: { id_A: 'id_A', id_B: 'id_B', id_C: 'id_C', f: 'f', g: 'g', 'g∘f': 'g∘f' },
    contra: false,
    build: () => {
      const c = triangleCategory();
      return { source: c, target: c, contravariant: false,
        onObjects: new Map([['A', 'A'], ['B', 'B'], ['C', 'C']]),
        onMorphisms: new Map([['id_A', 'id_A'], ['id_B', 'id_B'], ['id_C', 'id_C'], ['f', 'f'], ['g', 'g'], ['g∘f', 'g∘f']]),
      };
    },
  },
  { // Contravariant: Dual V*
    name: 'Contravariant: Dual V*', srcLabel: 'Vec', tgtLabel: 'Vec*',
    srcObjs: [{ label: 'V', x: .3, y: .4 }, { label: 'W', x: .7, y: .4 }],
    srcMors: [{ label: 'id_V', source: 'V', target: 'V', isIdentity: true }, { label: 'id_W', source: 'W', target: 'W', isIdentity: true }, mor('T', 'V', 'W')],
    tgtObjs: [{ label: 'V*', x: .3, y: .4 }, { label: 'W*', x: .7, y: .4 }],
    tgtMors: [{ label: 'id_V*', source: 'V*', target: 'V*', isIdentity: true }, { label: 'id_W*', source: 'W*', target: 'W*', isIdentity: true }, mor('T*', 'W*', 'V*')],
    objMap: { V: 'V*', W: 'W*' }, morMap: { id_V: 'id_V*', id_W: 'id_W*', T: 'T*' },
    contra: true,
    build: () => {
      const sc = new Map([['id_V,id_V', 'id_V'], ['id_W,id_W', 'id_W'], ['T,id_V', 'T'], ['id_W,T', 'T']]);
      const tc = new Map([['id_V*,id_V*', 'id_V*'], ['id_W*,id_W*', 'id_W*'], ['T*,id_W*', 'T*'], ['id_V*,T*', 'T*']]);
      const src = makeCat(['V', 'W'], [
        { label: 'id_V', source: 'V', target: 'V', isIdentity: true },
        { label: 'id_W', source: 'W', target: 'W', isIdentity: true }, mor('T', 'V', 'W')], sc);
      const tgt = makeCat(['V*', 'W*'], [
        { label: 'id_V*', source: 'V*', target: 'V*', isIdentity: true },
        { label: 'id_W*', source: 'W*', target: 'W*', isIdentity: true }, mor('T*', 'W*', 'V*')], tc);
      return { source: src, target: tgt, contravariant: true,
        onObjects: new Map([['V', 'V*'], ['W', 'W*']]),
        onMorphisms: new Map([['id_V', 'id_V*'], ['id_W', 'id_W*'], ['T', 'T*']]),
      };
    },
  },
];

// ─── Component ───

export default function FunctorVisualizer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const svgRef = useRef<SVGSVGElement>(null);
  const instanceId = useId().replace(/:/g, '');

  const [functorIndex, setFunctorIndex] = useState(0);
  const [hoveredSource, setHoveredSource] = useState<string | null>(null);
  const [hoveredTarget, setHoveredTarget] = useState<string | null>(null);

  const preset = PRESETS[functorIndex];
  const isNarrow = containerWidth > 0 && containerWidth < SM_BREAKPOINT;

  const functor = useMemo(() => preset.build(), [preset]);
  const idCheck = useMemo(() => checkFunctorIdentity(functor), [functor]);
  const compCheck = useMemo(() => checkFunctorComposition(functor), [functor]);

  // Highlight partner lookup
  const activeTarget = hoveredSource
    ? (preset.objMap[hoveredSource] ?? preset.morMap[hoveredSource] ?? null)
    : hoveredTarget;
  const activeSource = hoveredSource ?? (hoveredTarget
    ? (Object.entries(preset.objMap).find(([, v]) => v === hoveredTarget)?.[0]
      ?? Object.entries(preset.morMap).find(([, v]) => v === hoveredTarget)?.[0] ?? null)
    : null);

  const svgWidth = containerWidth ? containerWidth - 16 : 600;
  const svgHeight = isNarrow ? SVG_H * 2 + 20 : SVG_H;

  // ─── D3 rendering ───

  useEffect(() => {
    if (!svgRef.current || svgWidth === 0) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const boxW = isNarrow ? svgWidth - PAD * 2 : (svgWidth - PAD * 3) / 2;
    const boxH = isNarrow ? SVG_H - 20 : SVG_H - 40;
    const srcX = PAD, srcY = 20;
    const tgtX = isNarrow ? PAD : boxW + PAD * 2;
    const tgtY = isNarrow ? SVG_H + 10 : 20;

    // Arrow markers
    const defs = svg.append('defs');
    for (const [suffix, c] of [['s', SRC_CLR], ['t', TGT_CLR], ['c', '#ef4444'], ['m', MAP_CLR]] as const) {
      defs.append('marker').attr('id', `a-${suffix}-${instanceId}`).attr('viewBox', '0 0 10 7')
        .attr('refX', 9).attr('refY', 3.5).attr('markerWidth', 8).attr('markerHeight', 6)
        .attr('orient', 'auto').append('path').attr('d', 'M0,0 L10,3.5 L0,7 Z').attr('fill', c);
    }

    const pos = (o: Obj, bx: number, by: number) => ({ x: bx + o.x * boxW, y: by + o.y * boxH });

    // Bounding boxes
    for (const [bx, by, lbl] of [[srcX, srcY, preset.srcLabel], [tgtX, tgtY, preset.tgtLabel]] as const) {
      svg.append('rect').attr('x', bx).attr('y', by).attr('width', boxW).attr('height', boxH)
        .attr('rx', 10).style('fill', 'none').style('stroke', 'var(--color-border)')
        .style('stroke-width', '1.5').style('stroke-dasharray', '6 3');
      svg.append('text').attr('x', bx + boxW / 2).attr('y', by + 16).attr('text-anchor', 'middle')
        .attr('font-size', 13).attr('font-weight', 700).style('fill', 'var(--color-text-secondary)').text(lbl);
    }

    // Draw morphism arrows
    const drawArrow = (m: Mor, objs: Obj[], bx: number, by: number, clr: string, mk: string, dash: boolean) => {
      if (m.isIdentity) return;
      const so = objs.find((o) => o.label === m.source);
      const to = objs.find((o) => o.label === m.target);
      if (!so || !to) return;
      const p1 = pos(so, bx, by), p2 = pos(to, bx, by);
      const dx = p2.x - p1.x, dy = p2.y - p1.y;
      const d = Math.sqrt(dx * dx + dy * dy) || 1;
      const nx = dx / d, ny = dy / d;
      const x1 = p1.x + nx * R, y1 = p1.y + ny * R;
      const x2 = p2.x - nx * (R + 6), y2 = p2.y - ny * (R + 6);
      const mx = (x1 + x2) / 2 - ny * 20, my = (y1 + y2) / 2 + nx * 20;
      const hi = activeSource === m.label || activeTarget === m.label;
      svg.append('path').attr('d', `M${x1},${y1} Q${mx},${my} ${x2},${y2}`)
        .attr('marker-end', `url(#${mk})`).style('fill', 'none').style('stroke', clr)
        .style('stroke-width', hi ? '3' : '1.8').style('stroke-dasharray', dash ? '6 4' : 'none')
        .style('opacity', hi ? '1' : '0.7');
      svg.append('text').attr('x', mx).attr('y', my - 6).attr('text-anchor', 'middle')
        .attr('font-size', 11).attr('font-weight', 600).style('fill', clr)
        .style('opacity', hi ? '1' : '0.8').text(m.label);
    };

    for (const m of preset.srcMors) drawArrow(m, preset.srcObjs, srcX, srcY, SRC_CLR, `a-s-${instanceId}`, false);
    for (const m of preset.tgtMors) {
      const cv = preset.contra && !m.isIdentity;
      drawArrow(m, preset.tgtObjs, tgtX, tgtY, cv ? '#ef4444' : TGT_CLR, cv ? `a-c-${instanceId}` : `a-t-${instanceId}`, cv);
    }

    // Functor mapping arrows (dashed gray between boxes)
    for (const [sl, tl] of Object.entries(preset.objMap)) {
      const so = preset.srcObjs.find((o) => o.label === sl);
      const to = preset.tgtObjs.find((o) => o.label === tl);
      if (!so || !to) continue;
      const p1 = pos(so, srcX, srcY), p2 = pos(to, tgtX, tgtY);
      const hi = activeSource === sl || activeTarget === tl;
      const mx = (p1.x + p2.x) / 2, my = (p1.y + p2.y) / 2 - (isNarrow ? 0 : 15);
      svg.append('path').attr('d', `M${p1.x},${p1.y} Q${mx},${my} ${p2.x},${p2.y}`)
        .attr('marker-end', `url(#a-m-${instanceId})`).style('fill', 'none').style('stroke', MAP_CLR)
        .style('stroke-width', hi ? '2.5' : '1.2').style('stroke-dasharray', '5 4')
        .style('opacity', hi ? '0.9' : '0.4');
    }

    // Object nodes
    const drawNodes = (objs: Obj[], bx: number, by: number, clr: string, side: 'src' | 'tgt') => {
      for (const o of objs) {
        const p = pos(o, bx, by);
        const act = side === 'src' ? activeSource === o.label : activeTarget === o.label;
        svg.append('circle').attr('cx', p.x).attr('cy', p.y).attr('r', R)
          .style('fill', clr).style('stroke', act ? '#f59e0b' : 'var(--color-text)')
          .style('stroke-width', act ? '3' : '1.5').style('opacity', act ? '1' : '0.85')
          .style('cursor', 'pointer')
          .on('mouseenter', () => side === 'src' ? setHoveredSource(o.label) : setHoveredTarget(o.label))
          .on('mouseleave', () => side === 'src' ? setHoveredSource(null) : setHoveredTarget(null));
        svg.append('text').attr('x', p.x).attr('y', p.y + 4.5).attr('text-anchor', 'middle')
          .attr('font-size', 11).attr('font-weight', 700).attr('pointer-events', 'none')
          .style('fill', 'white').text(o.label);
      }
    };
    drawNodes(preset.srcObjs, srcX, srcY, SRC_CLR, 'src');
    drawNodes(preset.tgtObjs, tgtX, tgtY, TGT_CLR, 'tgt');
  }, [svgWidth, svgHeight, preset, isNarrow, activeSource, activeTarget]);

  // ─── Axiom checks ───

  const identityChecks = useMemo(() =>
    preset.srcObjs.map((o) => ({
      label: `F(id_${o.label}) = id_{${preset.objMap[o.label] ?? '?'}}`,
      pass: !idCheck.violations.includes(o.label),
    })), [preset, idCheck]);

  const compositionChecks = useMemo(() => {
    const checks: { label: string; pass: boolean }[] = [];
    const nonId = preset.srcMors.filter((m) => !m.isIdentity);
    for (const g of nonId) for (const f of nonId) {
      if (g.source !== f.target) continue;
      const gf = functor.source.compose(g.label, f.label);
      if (!gf || gf === g.label || gf === f.label) continue;
      const bad = compCheck.violations.some(([a, b]) => a === g.label && b === f.label);
      const Fg = preset.morMap[g.label] ?? '?', Ff = preset.morMap[f.label] ?? '?';
      const Fgf = preset.morMap[gf] ?? '?';
      const rhs = preset.contra ? `${Ff}∘${Fg}` : `${Fg}∘${Ff}`;
      checks.push({ label: `F(${g.label}∘${f.label}) = ${rhs}  [${Fgf}]`, pass: !bad });
    }
    return checks;
  }, [preset, functor, compCheck]);

  // ─── Render ───

  return (
    <div ref={containerRef} className="w-full rounded-lg border p-4"
      style={{ borderColor: 'var(--color-border)', backgroundColor: 'var(--color-surface)' }}>

      <div className="mb-4 flex flex-wrap items-center gap-3">
        <label className="flex items-center gap-1.5 text-sm" style={{ color: 'var(--color-text)' }}>
          <span className="font-medium">Functor:</span>
          <select className="rounded border px-2 py-1 text-sm"
            style={{ borderColor: 'var(--color-border)', backgroundColor: 'var(--color-bg)', color: 'var(--color-text)' }}
            value={functorIndex}
            onChange={(e) => { setFunctorIndex(Number(e.target.value)); setHoveredSource(null); setHoveredTarget(null); }}>
            {PRESETS.map((p, i) => <option key={p.name} value={i}>{p.name}</option>)}
          </select>
        </label>
      </div>

      <svg ref={svgRef} width={svgWidth} height={svgHeight}
        style={{ border: '1px solid var(--color-border)', borderRadius: 8, backgroundColor: 'var(--color-bg)' }} />

      <div className="mt-4 rounded-lg border p-3"
        style={{ borderColor: 'var(--color-border)', backgroundColor: 'var(--color-bg)' }}>
        <div className="mb-2 text-xs font-semibold" style={{ color: 'var(--color-text-secondary)' }}>
          Functor Axiom Verification
        </div>
        <div className="mb-2">
          <div className="mb-1 text-xs font-medium" style={{ color: 'var(--color-text-secondary)' }}>
            Identity preservation
          </div>
          <div className="flex flex-wrap gap-3">
            {identityChecks.map((c) => (
              <span key={c.label} className="flex items-center gap-1 text-xs" style={{ color: 'var(--color-text)' }}>
                <span style={{ color: c.pass ? '#22c55e' : '#ef4444' }}>{c.pass ? '✓' : '✗'}</span>
                {c.label}
              </span>
            ))}
          </div>
        </div>
        {compositionChecks.length > 0 && (
          <div>
            <div className="mb-1 text-xs font-medium" style={{ color: 'var(--color-text-secondary)' }}>
              Composition preservation{preset.contra ? ' (contravariant: reverses order)' : ''}
            </div>
            <div className="flex flex-wrap gap-3">
              {compositionChecks.map((c) => (
                <span key={c.label} className="flex items-center gap-1 text-xs" style={{ color: 'var(--color-text)' }}>
                  <span style={{ color: c.pass ? '#22c55e' : '#ef4444' }}>{c.pass ? '✓' : '✗'}</span>
                  {c.label}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="mt-3 flex flex-wrap gap-4 text-xs" style={{ color: 'var(--color-text-secondary)' }}>
        <span><strong>Hover</strong> an object to highlight its mapping</span>
        <span style={{ color: SRC_CLR }}>● Source</span>
        <span style={{ color: TGT_CLR }}>● Target</span>
        <span style={{ color: MAP_CLR }}>⇢ Functor map</span>
        {preset.contra && <span style={{ color: '#ef4444' }}>⇢ Reversed arrow</span>}
      </div>
    </div>
  );
}
