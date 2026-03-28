import { useState, useMemo, useEffect, useRef, useId, useCallback } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  type NaturalTransformation,
  type Category,
  getNatTransPresets,
  checkNaturality,
} from './shared/categoryTheory';

// ─── Constants ───

const SM_BREAKPOINT = 640;
const SVG_H = 340;
const NODE_R = 22;
const STEP_MS = 500;
const MATH_FONT = 'KaTeX_Math, Georgia, serif';

const COLORS = {
  functorF: '#3b82f6',    // blue
  functorG: '#8b5cf6',    // purple
  natTrans: '#f59e0b',    // amber
  pathA: '#22c55e',       // green  (top-then-right)
  pathB: '#06b6d4',       // cyan   (down-then-across)
  valid: '#22c55e',
  invalid: '#ef4444',
  bg: '#f8fafc',
  text: '#1e293b',
  muted: '#94a3b8',
};

// ─── Types ───

interface PresetData {
  nat: NaturalTransformation;
  sourceCategory: Category;
  targetCategory: Category;
  availableMorphisms: string[];
  description: string;
}

interface SquareLabels {
  topLeft: string; topRight: string;
  botLeft: string; botRight: string;
  top: string; bottom: string;
  left: string; right: string;
}

type Phase = 'idle' | 'pathA1' | 'pathA2' | 'pathB1' | 'pathB2' | 'done';

// ─── Helpers ───

function buildLabels(p: PresetData, morphLabel: string): SquareLabels | null {
  const f = p.sourceCategory.morphisms.find((m) => m.label === morphLabel);
  if (!f) return null;
  const [A, B, F, G] = [f.source, f.target, p.nat.source, p.nat.target];
  const FA = F.onObjects.get(A), FB = F.onObjects.get(B);
  const GA = G.onObjects.get(A), GB = G.onObjects.get(B);
  const aA = p.nat.components.get(A), aB = p.nat.components.get(B);
  const Ff = F.onMorphisms.get(morphLabel), Gf = G.onMorphisms.get(morphLabel);
  if (!FA || !FB || !GA || !GB || !aA || !aB || !Ff || !Gf) return null;
  return { topLeft: FA, topRight: GA, botLeft: FB, botRight: GB, top: aA, bottom: aB, left: Ff, right: Gf };
}

function commCheck(p: PresetData, morphLabel: string) {
  const f = p.sourceCategory.morphisms.find((m) => m.label === morphLabel);
  if (!f) return { topRight: null as string | null, downAcross: null as string | null, commutes: false };
  const [A, B] = [f.source, f.target];
  const aA = p.nat.components.get(A), aB = p.nat.components.get(B);
  const Ff = p.nat.source.onMorphisms.get(morphLabel);
  const Gf = p.nat.target.onMorphisms.get(morphLabel);
  if (!aA || !aB || !Ff || !Gf) return { topRight: null, downAcross: null, commutes: false };
  const tr = p.targetCategory.compose(Gf, aA);
  const da = p.targetCategory.compose(aB, Ff);
  return { topRight: tr, downAcross: da, commutes: tr !== null && da !== null && tr === da };
}

const DEFAULT_LABELS: SquareLabels = {
  topLeft: 'F(A)', topRight: 'G(A)', botLeft: 'F(B)', botRight: 'G(B)',
  top: 'α_A', bottom: 'α_B', left: 'F(f)', right: 'G(f)',
};

// ─── Component ───

export default function NaturalitySquareExplorer() {
  const { ref: containerRef, width: cw } = useResizeObserver<HTMLDivElement>();
  const uid = useId().replace(/:/g, '');
  const svgRef = useRef<SVGSVGElement>(null);

  // State
  const presets = useMemo(() => getNatTransPresets(), []);
  const [presetIdx, setPresetIdx] = useState(0);
  const [isCustom, setIsCustom] = useState(false);
  const [presetData, setPresetData] = useState<PresetData>(() => presets[0].build());
  const [morphIdx, setMorphIdx] = useState(0);
  const [animating, setAnimating] = useState(false);
  const [phase, setPhase] = useState<Phase>('idle');

  // Custom mode
  const [customLabels, setCustomLabels] = useState({
    alphaA: 'α_A', alphaB: 'α_B', Ff: 'F(f)', Gf: 'G(f)',
  });
  const [customResult, setCustomResult] = useState({ checked: false, commutes: false });

  // Derived
  const isNarrow = cw > 0 && cw < SM_BREAKPOINT;
  const currentMorph = isCustom ? 'f' : presetData.availableMorphisms[morphIdx] ?? '';

  const labels: SquareLabels = useMemo(() => {
    if (isCustom) {
      return { ...DEFAULT_LABELS, top: customLabels.alphaA, bottom: customLabels.alphaB, left: customLabels.Ff, right: customLabels.Gf };
    }
    return buildLabels(presetData, currentMorph) ?? DEFAULT_LABELS;
  }, [isCustom, presetData, currentMorph, customLabels]);

  const comm = useMemo(() => {
    if (isCustom) {
      return {
        topRight: `${customLabels.Gf} ∘ ${customLabels.alphaA}`,
        downAcross: `${customLabels.alphaB} ∘ ${customLabels.Ff}`,
        commutes: customResult.checked && customResult.commutes,
      };
    }
    return commCheck(presetData, currentMorph);
  }, [isCustom, presetData, currentMorph, customLabels, customResult]);

  const natCheck = useMemo(
    () => isCustom ? null : checkNaturality(presetData.nat, presetData.sourceCategory, presetData.targetCategory),
    [isCustom, presetData],
  );

  const svgW = useMemo(() => {
    if (!cw) return 320;
    return isNarrow ? Math.max(320, cw - 16) : Math.max(320, Math.floor((cw - 32) * 0.6));
  }, [cw, isNarrow]);

  // ─── Handlers ───

  const onPresetChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    const v = e.target.value;
    if (v === 'custom') { setIsCustom(true); setCustomResult({ checked: false, commutes: false }); setPhase('idle'); return; }
    const idx = parseInt(v, 10);
    setIsCustom(false); setPresetIdx(idx); setPresetData(presets[idx].build());
    setMorphIdx(0); setPhase('idle');
  }, [presets]);

  const onMorphChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    setMorphIdx(parseInt(e.target.value, 10)); setPhase('idle');
  }, []);

  const onCustomCheck = useCallback(() => {
    const a = `${customLabels.Gf}∘${customLabels.alphaA}`;
    const b = `${customLabels.alphaB}∘${customLabels.Ff}`;
    setCustomResult({ checked: true, commutes: a === b });
  }, [customLabels]);

  const timersRef = useRef<ReturnType<typeof setTimeout>[]>([]);

  useEffect(() => {
    return () => { timersRef.current.forEach(clearTimeout); };
  }, []);

  const onAnimate = useCallback(() => {
    if (animating) return;
    timersRef.current.forEach(clearTimeout);
    timersRef.current = [];
    setAnimating(true); setPhase('pathA1');
    timersRef.current.push(setTimeout(() => setPhase('pathA2'), STEP_MS));
    timersRef.current.push(setTimeout(() => setPhase('pathB1'), STEP_MS * 2));
    timersRef.current.push(setTimeout(() => setPhase('pathB2'), STEP_MS * 3));
    timersRef.current.push(setTimeout(() => { setPhase('done'); setAnimating(false); }, STEP_MS * 4));
  }, [animating]);

  // ─── D3 rendering ───

  useEffect(() => {
    if (!svgRef.current || svgW === 0) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    // Background
    svg.append('rect').attr('width', svgW).attr('height', SVG_H).attr('rx', 8).style('fill', COLORS.bg);

    // Node positions
    const [cx, cy] = [svgW / 2, SVG_H / 2];
    const dx = Math.min(200, (svgW - 120) / 2);
    const dy = Math.min(120, (SVG_H - 120) / 2);
    const pts = {
      tl: { x: cx - dx, y: cy - dy }, tr: { x: cx + dx, y: cy - dy },
      bl: { x: cx - dx, y: cy + dy }, br: { x: cx + dx, y: cy + dy },
    };

    // Arrow markers
    const defs = svg.append('defs');
    const addMarker = (id: string, color: string) => {
      defs.append('marker').attr('id', id).attr('viewBox', '0 0 10 10')
        .attr('refX', 8).attr('refY', 5).attr('markerWidth', 8).attr('markerHeight', 8)
        .attr('orient', 'auto-start-reverse')
        .append('path').attr('d', 'M 0 0 L 10 5 L 0 10 Z').style('fill', color);
    };
    const mId = { nat: `${uid}-an`, f: `${uid}-af`, g: `${uid}-ag` };
    addMarker(mId.nat, COLORS.natTrans);
    addMarker(mId.f, COLORS.functorF);
    addMarker(mId.g, COLORS.functorG);

    // Draw arrow between two points
    const drawArrow = (
      p1: { x: number; y: number }, p2: { x: number; y: number },
      markerId: string, color: string, label: string,
      labelOff: { dx: number; dy: number },
    ) => {
      const a = Math.atan2(p2.y - p1.y, p2.x - p1.x);
      const sx = p1.x + Math.cos(a) * (NODE_R + 4);
      const sy = p1.y + Math.sin(a) * (NODE_R + 4);
      const tx = p2.x - Math.cos(a) * (NODE_R + 12);
      const ty = p2.y - Math.sin(a) * (NODE_R + 12);
      svg.append('line').attr('x1', sx).attr('y1', sy).attr('x2', tx).attr('y2', ty)
        .style('stroke', color).style('stroke-width', 2).attr('marker-end', `url(#${markerId})`);
      svg.append('text')
        .attr('x', (sx + tx) / 2 + labelOff.dx).attr('y', (sy + ty) / 2 + labelOff.dy)
        .style('fill', color).style('font-size', '13px').style('font-style', 'italic')
        .style('font-family', MATH_FONT).style('text-anchor', 'middle')
        .style('dominant-baseline', 'middle').text(label);
    };

    // Four edges of the naturality square
    drawArrow(pts.tl, pts.tr, mId.nat, COLORS.natTrans, labels.top, { dx: 0, dy: -16 });
    drawArrow(pts.bl, pts.br, mId.nat, COLORS.natTrans, labels.bottom, { dx: 0, dy: 18 });
    drawArrow(pts.tl, pts.bl, mId.f, COLORS.functorF, labels.left, { dx: -28, dy: 0 });
    drawArrow(pts.tr, pts.br, mId.g, COLORS.functorG, labels.right, { dx: 28, dy: 0 });

    // Animated path overlays
    const drawOverlay = (
      p1: { x: number; y: number }, p2: { x: number; y: number },
      color: string, show: boolean,
    ) => {
      if (!show) return;
      const a = Math.atan2(p2.y - p1.y, p2.x - p1.x);
      svg.append('line')
        .attr('x1', p1.x + Math.cos(a) * (NODE_R + 4))
        .attr('y1', p1.y + Math.sin(a) * (NODE_R + 4))
        .attr('x2', p2.x - Math.cos(a) * (NODE_R + 12))
        .attr('y2', p2.y - Math.sin(a) * (NODE_R + 12))
        .style('stroke', color).style('stroke-width', 6).style('stroke-linecap', 'round')
        .style('opacity', 0).transition().duration(STEP_MS).style('opacity', 0.35);
    };

    const showA1 = phase === 'pathA1' || phase === 'pathA2' || phase === 'done';
    const showA2 = phase === 'pathA2' || phase === 'done';
    const showB1 = phase === 'pathB1' || phase === 'pathB2' || phase === 'done';
    const showB2 = phase === 'pathB2' || phase === 'done';

    drawOverlay(pts.tl, pts.tr, COLORS.pathA, showA1);   // Path A step 1: top
    drawOverlay(pts.tr, pts.br, COLORS.pathA, showA2);   // Path A step 2: right
    drawOverlay(pts.tl, pts.bl, COLORS.pathB, showB1);   // Path B step 1: left
    drawOverlay(pts.bl, pts.br, COLORS.pathB, showB2);   // Path B step 2: bottom

    // Corner nodes (drawn on top)
    const nodeData = [
      { p: pts.tl, l: labels.topLeft, c: COLORS.functorF },
      { p: pts.tr, l: labels.topRight, c: COLORS.functorG },
      { p: pts.bl, l: labels.botLeft, c: COLORS.functorF },
      { p: pts.br, l: labels.botRight, c: COLORS.functorG },
    ];
    for (const nd of nodeData) {
      svg.append('circle').attr('cx', nd.p.x).attr('cy', nd.p.y).attr('r', NODE_R)
        .style('fill', '#fff').style('stroke', nd.c).style('stroke-width', 2.5);
      svg.append('text').attr('x', nd.p.x).attr('y', nd.p.y)
        .style('fill', COLORS.text).style('font-size', '12px').style('font-style', 'italic')
        .style('font-family', MATH_FONT).style('text-anchor', 'middle')
        .style('dominant-baseline', 'central').text(nd.l);
    }

    // "commutes!" center annotation
    if (phase === 'done' && comm.commutes) {
      svg.append('text').attr('x', cx).attr('y', cy)
        .style('fill', COLORS.valid).style('font-size', '14px').style('font-weight', '600')
        .style('text-anchor', 'middle').style('dominant-baseline', 'central')
        .style('opacity', 0).text('commutes!')
        .transition().duration(400).style('opacity', 1);
    }
  }, [svgW, labels, phase, comm.commutes, uid]);

  // ─── Render ───

  const selectStyle: React.CSSProperties = {
    padding: '4px 8px', borderRadius: '6px',
    border: `1px solid ${COLORS.muted}`, fontSize: '13px', background: '#fff',
  };

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif', color: COLORS.text }}>

      {/* ── Controls bar ── */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px', alignItems: 'center', marginBottom: '12px', padding: '10px 12px', background: '#f1f5f9', borderRadius: '8px' }}>
        <label style={{ fontSize: '13px', fontWeight: 600, color: COLORS.muted }}>Preset:</label>
        <select value={isCustom ? 'custom' : String(presetIdx)} onChange={onPresetChange} style={{ ...selectStyle, minWidth: '180px' }}>
          {presets.map((p, i) => <option key={i} value={String(i)}>{p.name}</option>)}
          <option value="custom">Custom</option>
        </select>

        {!isCustom && presetData.availableMorphisms.length > 0 && (<>
          <label style={{ fontSize: '13px', fontWeight: 600, color: COLORS.muted }}>Morphism:</label>
          <select value={String(morphIdx)} onChange={onMorphChange} style={selectStyle}>
            {presetData.availableMorphisms.map((m, i) => <option key={m} value={String(i)}>{m}</option>)}
          </select>
        </>)}

        <button onClick={onAnimate} disabled={animating} style={{
          padding: '5px 14px', borderRadius: '6px', border: 'none',
          background: animating ? COLORS.muted : COLORS.natTrans, color: '#fff',
          fontWeight: 600, fontSize: '13px', cursor: animating ? 'not-allowed' : 'pointer', marginLeft: 'auto',
        }}>
          {animating ? 'Animating\u2026' : 'Animate'}
        </button>
      </div>

      {/* ── Custom mode inputs ── */}
      {isCustom && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', alignItems: 'center', marginBottom: '12px', padding: '10px 12px', background: '#fefce8', borderRadius: '8px', border: '1px solid #fde68a' }}>
          <span style={{ fontSize: '12px', fontWeight: 600, color: '#92400e' }}>Custom labels:</span>
          {([
            ['alphaA', '\u03B1_A', 'F(A)\u2192G(A)'] as const,
            ['alphaB', '\u03B1_B', 'F(B)\u2192G(B)'] as const,
            ['Ff', 'F(f)', 'F(A)\u2192F(B)'] as const,
            ['Gf', 'G(f)', 'G(A)\u2192G(B)'] as const,
          ]).map(([key, placeholder, hint]) => (
            <div key={key} style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
              <span style={{ fontSize: '10px', color: COLORS.muted }}>{hint}</span>
              <input type="text" value={customLabels[key]} placeholder={placeholder}
                onChange={(e) => setCustomLabels((prev) => ({ ...prev, [key]: e.target.value }))}
                style={{ width: '72px', padding: '3px 6px', borderRadius: '4px', border: `1px solid ${COLORS.muted}`, fontSize: '12px', fontStyle: 'italic' }} />
            </div>
          ))}
          <button onClick={onCustomCheck} style={{ padding: '5px 12px', borderRadius: '6px', border: 'none', background: '#d97706', color: '#fff', fontWeight: 600, fontSize: '13px', cursor: 'pointer' }}>
            Check
          </button>
          {customResult.checked && (
            <span style={{ fontSize: '13px', fontWeight: 600, color: customResult.commutes ? COLORS.valid : COLORS.invalid }}>
              {customResult.commutes ? '\u2713 Commutes' : '\u2717 Does not commute'}
            </span>
          )}
        </div>
      )}

      {/* ── Main content: SVG + verification panel ── */}
      <div style={{ display: 'flex', flexDirection: isNarrow ? 'column' : 'row', gap: '16px' }}>

        {/* Diagram */}
        <div style={{ flex: isNarrow ? 'none' : '3 1 0%', minWidth: 0 }}>
          <svg ref={svgRef} width={svgW} height={SVG_H} viewBox={`0 0 ${svgW} ${SVG_H}`} style={{ display: 'block', maxWidth: '100%' }} />
          {/* Legend */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '14px', marginTop: '8px', padding: '6px 10px', fontSize: '11px', color: COLORS.muted }}>
            {[
              { c: COLORS.functorF, l: 'Functor F' },
              { c: COLORS.functorG, l: 'Functor G' },
              { c: COLORS.natTrans, l: 'Components \u03B1' },
              { c: COLORS.pathA, l: 'Path: G(f) \u2218 \u03B1_A' },
              { c: COLORS.pathB, l: 'Path: \u03B1_B \u2218 F(f)' },
            ].map(({ c, l }) => (
              <span key={l} style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: '2px', background: c }} />
                {l}
              </span>
            ))}
          </div>
        </div>

        {/* Verification panel */}
        <div style={{ flex: isNarrow ? 'none' : '2 1 0%', minWidth: '200px', background: '#f1f5f9', borderRadius: '8px', padding: '16px' }}>
          <h4 style={{ margin: '0 0 12px', fontSize: '14px', fontWeight: 700 }}>Naturality Verification</h4>
          {!isCustom && (
            <p style={{ margin: '0 0 12px', fontSize: '12px', color: COLORS.muted, lineHeight: 1.5 }}>
              {presetData.description}
            </p>
          )}

          {/* Path A: top-then-right */}
          <div style={{ marginBottom: '12px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
              <span style={{ width: 8, height: 8, borderRadius: '50%', background: COLORS.pathA, display: 'inline-block' }} />
              <span style={{ fontSize: '12px', fontWeight: 600, color: COLORS.pathA }}>Top-then-right</span>
            </div>
            <div style={{ fontSize: '13px', fontStyle: 'italic', fontFamily: MATH_FONT, padding: '6px 10px', background: '#fff', borderRadius: '4px', border: `1px solid ${COLORS.pathA}40` }}>
              G(f) {'\u2218'} {'\u03B1'}_A = {comm.topRight ?? <span style={{ color: COLORS.muted }}>{'\u2014'}</span>}
            </div>
          </div>

          {/* Path B: down-then-across */}
          <div style={{ marginBottom: '16px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
              <span style={{ width: 8, height: 8, borderRadius: '50%', background: COLORS.pathB, display: 'inline-block' }} />
              <span style={{ fontSize: '12px', fontWeight: 600, color: COLORS.pathB }}>Down-then-across</span>
            </div>
            <div style={{ fontSize: '13px', fontStyle: 'italic', fontFamily: MATH_FONT, padding: '6px 10px', background: '#fff', borderRadius: '4px', border: `1px solid ${COLORS.pathB}40` }}>
              {'\u03B1'}_B {'\u2218'} F(f) = {comm.downAcross ?? <span style={{ color: COLORS.muted }}>{'\u2014'}</span>}
            </div>
          </div>

          {/* Equality result */}
          <div style={{
            display: 'flex', alignItems: 'center', gap: '8px', padding: '10px 12px', borderRadius: '6px',
            background: comm.commutes ? `${COLORS.valid}15` : `${COLORS.invalid}15`,
            border: `1px solid ${comm.commutes ? COLORS.valid : COLORS.invalid}40`,
          }}>
            <span style={{ fontSize: '18px' }}>{comm.commutes ? '\u2713' : '\u2717'}</span>
            <span style={{ fontSize: '13px', fontWeight: 600, color: comm.commutes ? COLORS.valid : COLORS.invalid }}>
              {comm.commutes ? 'Square commutes \u2014 naturality holds' : 'Square does not commute'}
            </span>
          </div>

          {/* Full naturality check */}
          {natCheck && !isCustom && (
            <div style={{ marginTop: '12px', fontSize: '12px', color: COLORS.muted }}>
              <span style={{ fontWeight: 600 }}>Full naturality check: </span>
              {natCheck.valid
                ? <span style={{ color: COLORS.valid, fontWeight: 600 }}>{'\u2713'} Valid for all morphisms</span>
                : <span style={{ color: COLORS.invalid, fontWeight: 600 }}>{'\u2717'} {natCheck.violations.length} violation{natCheck.violations.length > 1 ? 's' : ''}</span>}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
