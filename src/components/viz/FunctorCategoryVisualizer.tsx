import { useState, useId } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  type Category,
  type Functor,
  type NaturalTransformation,
  twoObjectCategory,
  triangleCategory,
  checkNaturality,
  verticalCompose,
} from './shared/categoryTheory';
import type { CategoryMorphism } from './shared/types';

const SM_BREAKPOINT = 640;

const COLORS = {
  functorF: '#3b82f6',
  functorG: '#8b5cf6',
  functorH: '#06b6d4',
  natTrans: '#f59e0b',
  composition: '#10b981',
  valid: '#22c55e',
  invalid: '#ef4444',
  bg: '#f8fafc',
  text: '#1e293b',
  muted: '#94a3b8',
  edge: '#cbd5e1',
};

const FUNCTOR_COLORS = [COLORS.functorF, COLORS.functorG, COLORS.functorH];

// ---------------------------------------------------------------------------
// Target category builder — small explicit categories with composition maps
// ---------------------------------------------------------------------------

function buildTargetForTwo(): Category {
  // Objects: X, Y, X', Y' (and X'', Y'' for 3 functors)
  const objects = ['X', 'Y', "X'", "Y'", "X''", "Y''"];
  const morphisms: CategoryMorphism[] = [
    ...objects.map((o) => ({ label: `id_${o}`, source: o, target: o, isIdentity: true })),
    { label: 'h', source: 'X', target: 'Y', isIdentity: false },
    { label: "h'", source: "X'", target: "Y'", isIdentity: false },
    { label: "h''", source: "X''", target: "Y''", isIdentity: false },
    // Nat trans components α: F⇒G
    { label: 'α_A', source: 'X', target: "X'", isIdentity: false },
    { label: 'α_B', source: 'Y', target: "Y'", isIdentity: false },
    // Nat trans components β: G⇒H
    { label: 'β_A', source: "X'", target: "X''", isIdentity: false },
    { label: 'β_B', source: "Y'", target: "Y''", isIdentity: false },
    // Composed (β∘α)
    { label: '(β∘α)_A', source: 'X', target: "X''", isIdentity: false },
    { label: '(β∘α)_B', source: 'Y', target: "Y''", isIdentity: false },
    // Naturality composites
    { label: "h'∘α_A", source: 'X', target: "Y'", isIdentity: false },
    { label: "α_B∘h", source: 'X', target: "Y'", isIdentity: false },
    { label: "h''∘β_A", source: "X'", target: "Y''", isIdentity: false },
    { label: "β_B∘h'", source: "X'", target: "Y''", isIdentity: false },
    { label: "h''∘(β∘α)_A", source: 'X', target: "Y''", isIdentity: false },
    { label: "(β∘α)_B∘h", source: 'X', target: "Y''", isIdentity: false },
  ];

  const comp = new Map<string, string>();
  // Identity compositions
  for (const o of objects) {
    for (const m of morphisms) {
      if (m.source === o) comp.set(`${m.label},id_${o}`, m.label);
      if (m.target === o) comp.set(`id_${o},${m.label}`, m.label);
    }
  }
  // Naturality squares
  comp.set("h',α_A", "h'∘α_A");
  comp.set("α_B,h", "α_B∘h");
  comp.set("h'',β_A", "h''∘β_A");
  comp.set("β_B,h'", "β_B∘h'");
  comp.set("h'',(β∘α)_A", "h''∘(β∘α)_A");
  comp.set("(β∘α)_B,h", "(β∘α)_B∘h");
  // Make naturality hold: both paths equal
  comp.set("h'∘α_A", "h'∘α_A"); // self
  comp.set("α_B∘h", "h'∘α_A");   // key: naturality of α
  comp.set("h''∘β_A", "h''∘β_A");
  comp.set("β_B∘h'", "h''∘β_A");
  comp.set("h''∘(β∘α)_A", "h''∘(β∘α)_A");
  comp.set("(β∘α)_B∘h", "h''∘(β∘α)_A");
  // Vertical composition
  comp.set('β_A,α_A', '(β∘α)_A');
  comp.set('β_B,α_B', '(β∘α)_B');

  return {
    objects,
    morphisms,
    compose: (g, f) => comp.get(`${g},${f}`) ?? null,
    identity: (obj) => `id_${obj}`,
  };
}

function buildTargetForTriangle(): Category {
  const objects = ['X', 'Y', 'Z', "X'", "Y'", "Z'", "X''", "Y''", "Z''"];
  const morphisms: CategoryMorphism[] = [
    ...objects.map((o) => ({ label: `id_${o}`, source: o, target: o, isIdentity: true })),
    { label: 'h1', source: 'X', target: 'Y', isIdentity: false },
    { label: 'h2', source: 'Y', target: 'Z', isIdentity: false },
    { label: 'h3', source: 'X', target: 'Z', isIdentity: false },
    { label: "h1'", source: "X'", target: "Y'", isIdentity: false },
    { label: "h2'", source: "Y'", target: "Z'", isIdentity: false },
    { label: "h3'", source: "X'", target: "Z'", isIdentity: false },
    { label: "h1''", source: "X''", target: "Y''", isIdentity: false },
    { label: "h2''", source: "Y''", target: "Z''", isIdentity: false },
    { label: "h3''", source: "X''", target: "Z''", isIdentity: false },
    // α components
    { label: 'α_A', source: 'X', target: "X'", isIdentity: false },
    { label: 'α_B', source: 'Y', target: "Y'", isIdentity: false },
    { label: 'α_C', source: 'Z', target: "Z'", isIdentity: false },
    // β components
    { label: 'β_A', source: "X'", target: "X''", isIdentity: false },
    { label: 'β_B', source: "Y'", target: "Y''", isIdentity: false },
    { label: 'β_C', source: "Z'", target: "Z''", isIdentity: false },
    // Composed
    { label: '(β∘α)_A', source: 'X', target: "X''", isIdentity: false },
    { label: '(β∘α)_B', source: 'Y', target: "Y''", isIdentity: false },
    { label: '(β∘α)_C', source: 'Z', target: "Z''", isIdentity: false },
  ];

  const comp = new Map<string, string>();
  for (const o of objects) {
    for (const m of morphisms) {
      if (m.source === o) comp.set(`${m.label},id_${o}`, m.label);
      if (m.target === o) comp.set(`id_${o},${m.label}`, m.label);
    }
  }
  comp.set('h2,h1', 'h3');
  comp.set("h2',h1'", "h3'");
  comp.set("h2'',h1''", "h3''");
  // Naturality for α
  comp.set("h1',α_A", "h1'∘α_A"); comp.set("α_B,h1", "h1'∘α_A");
  comp.set("h2',α_B", "h2'∘α_B"); comp.set("α_C,h2", "h2'∘α_B");
  comp.set("h3',α_A", "h3'∘α_A"); comp.set("α_C,h3", "h3'∘α_A");
  // Naturality for β
  comp.set("h1'',β_A", "h1''∘β_A"); comp.set("β_B,h1'", "h1''∘β_A");
  comp.set("h2'',β_B", "h2''∘β_B"); comp.set("β_C,h2'", "h2''∘β_B");
  comp.set("h3'',β_A", "h3''∘β_A"); comp.set("β_C,h3'", "h3''∘β_A");
  // Vertical composition
  comp.set('β_A,α_A', '(β∘α)_A');
  comp.set('β_B,α_B', '(β∘α)_B');
  comp.set('β_C,α_C', '(β∘α)_C');

  return {
    objects,
    morphisms,
    compose: (g, f) => comp.get(`${g},${f}`) ?? null,
    identity: (obj) => `id_${obj}`,
  };
}

// ---------------------------------------------------------------------------
// Preset construction
// ---------------------------------------------------------------------------

interface Preset {
  sourceCategory: Category;
  targetCategory: Category;
  functors: { label: string; functor: Functor }[];
  natTrans: { label: string; nt: NaturalTransformation }[];
}

function buildTwoObjPreset(numFunctors: number): Preset {
  const source = twoObjectCategory();
  const target = buildTargetForTwo();

  const F: Functor = {
    source, target, contravariant: false,
    onObjects: new Map([['A', 'X'], ['B', 'Y']]),
    onMorphisms: new Map([['f', 'h'], ['id_A', 'id_X'], ['id_B', 'id_Y']]),
  };
  const G: Functor = {
    source, target, contravariant: false,
    onObjects: new Map([['A', "X'"], ['B', "Y'"]]),
    onMorphisms: new Map([['f', "h'"], ['id_A', "id_X'"], ['id_B', "id_Y'"]]),
  };
  const alpha: NaturalTransformation = {
    source: F, target: G,
    components: new Map([['A', 'α_A'], ['B', 'α_B']]),
  };

  const functors: Preset['functors'] = [{ label: 'F', functor: F }, { label: 'G', functor: G }];
  const natTrans: Preset['natTrans'] = [{ label: 'α', nt: alpha }];

  if (numFunctors >= 3) {
    const H: Functor = {
      source, target, contravariant: false,
      onObjects: new Map([['A', "X''"], ['B', "Y''"]]),
      onMorphisms: new Map([['f', "h''"], ['id_A', "id_X''"], ['id_B', "id_Y''"]]),
    };
    const beta: NaturalTransformation = {
      source: G, target: H,
      components: new Map([['A', 'β_A'], ['B', 'β_B']]),
    };
    functors.push({ label: 'H', functor: H });
    natTrans.push({ label: 'β', nt: beta });
  }

  return { sourceCategory: source, targetCategory: target, functors, natTrans };
}

function buildTrianglePreset(numFunctors: number): Preset {
  const source = triangleCategory();
  const target = buildTargetForTriangle();

  const F: Functor = {
    source, target, contravariant: false,
    onObjects: new Map([['A', 'X'], ['B', 'Y'], ['C', 'Z']]),
    onMorphisms: new Map([['f', 'h1'], ['g', 'h2'], ['g∘f', 'h3'], ['id_A', 'id_X'], ['id_B', 'id_Y'], ['id_C', 'id_Z']]),
  };
  const G: Functor = {
    source, target, contravariant: false,
    onObjects: new Map([['A', "X'"], ['B', "Y'"], ['C', "Z'"]]),
    onMorphisms: new Map([['f', "h1'"], ['g', "h2'"], ['g∘f', "h3'"], ['id_A', "id_X'"], ['id_B', "id_Y'"], ['id_C', "id_Z'"]]),
  };
  const alpha: NaturalTransformation = {
    source: F, target: G,
    components: new Map([['A', 'α_A'], ['B', 'α_B'], ['C', 'α_C']]),
  };

  const functors: Preset['functors'] = [{ label: 'F', functor: F }, { label: 'G', functor: G }];
  const natTrans: Preset['natTrans'] = [{ label: 'α', nt: alpha }];

  if (numFunctors >= 3) {
    const H: Functor = {
      source, target, contravariant: false,
      onObjects: new Map([['A', "X''"], ['B', "Y''"], ['C', "Z''"]]),
      onMorphisms: new Map([['f', "h1''"], ['g', "h2''"], ['g∘f', "h3''"], ['id_A', "id_X''"], ['id_B', "id_Y''"], ['id_C', "id_Z''"]]),
    };
    const beta: NaturalTransformation = {
      source: G, target: H,
      components: new Map([['A', 'β_A'], ['B', 'β_B'], ['C', 'β_C']]),
    };
    functors.push({ label: 'H', functor: H });
    natTrans.push({ label: 'β', nt: beta });
  }

  return { sourceCategory: source, targetCategory: target, functors, natTrans };
}

// ---------------------------------------------------------------------------
// Simple force-directed layout (one-shot, not D3-force)
// ---------------------------------------------------------------------------

function circleLayout(labels: string[], cx: number, cy: number, r: number) {
  const positions = new Map<string, { x: number; y: number }>();
  labels.forEach((l, i) => {
    const angle = -Math.PI / 2 + (2 * Math.PI * i) / labels.length;
    positions.set(l, { x: cx + r * Math.cos(angle), y: cy + r * Math.sin(angle) });
  });
  return positions;
}

// ---------------------------------------------------------------------------
// Detail row type
// ---------------------------------------------------------------------------

interface DetailRow {
  object: string;
  fObj: string;
  gObj: string;
  component: string;
  valid: boolean;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function FunctorCategoryVisualizer() {
  const uid = useId();
  const markerId = `arrow-${uid.replace(/:/g, '')}`;
  const markerNtId = `arrow-nt-${uid.replace(/:/g, '')}`;
  const markerCompId = `arrow-comp-${uid.replace(/:/g, '')}`;

  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;

  const [sourceType, setSourceType] = useState<'two' | 'triangle'>('two');
  const [numFunctors, setNumFunctors] = useState<2 | 3>(2);
  const [showComposition, setShowComposition] = useState(false);
  const [selectedNt, setSelectedNt] = useState<string | null>(null);
  const [hoveredNt, setHoveredNt] = useState<string | null>(null);

  // Build preset
  const preset = sourceType === 'two'
    ? buildTwoObjPreset(numFunctors)
    : buildTrianglePreset(numFunctors);

  // Composition
  const composedNt = numFunctors >= 3 && showComposition && preset.natTrans.length >= 2
    ? verticalCompose(preset.natTrans[0].nt, preset.natTrans[1].nt, preset.targetCategory)
    : null;

  // All nat trans for display
  const allNatTrans: { label: string; nt: NaturalTransformation; isComposition: boolean }[] = [
    ...preset.natTrans.map((n) => ({ ...n, isComposition: false })),
    ...(composedNt ? [{ label: 'β∘α', nt: composedNt, isComposition: true }] : []),
  ];

  // Detail panel data
  const detailData: DetailRow[] | null = (() => {
    if (!selectedNt) return null;
    const found = allNatTrans.find((n) => n.label === selectedNt);
    if (!found) return null;
    const { nt } = found;
    const check = checkNaturality(nt, preset.sourceCategory, preset.targetCategory);
    return preset.sourceCategory.objects.map((obj) => {
      const fObj = nt.source.onObjects.get(obj) ?? '?';
      const gObj = nt.target.onObjects.get(obj) ?? '?';
      const comp = nt.components.get(obj) ?? '?';
      const violation = check.violations.find((v) => {
        const m = preset.sourceCategory.morphisms.find((mm) => mm.label === v.morphism);
        return m && (m.source === obj || m.target === obj);
      });
      return { object: obj, fObj, gObj, component: comp, valid: !violation };
    });
  })();

  // Layout dimensions
  const totalW = containerWidth || 700;
  const panelH = isMobile ? 200 : 280;
  const srcW = isMobile ? totalW : totalW * 0.4;
  const fcW = isMobile ? totalW : totalW * 0.6;

  // Source category node positions
  const srcObjs = preset.sourceCategory.objects;
  const srcPos = circleLayout(srcObjs, srcW / 2, panelH / 2, Math.min(srcW, panelH) * 0.3);

  // Functor category node positions
  const fLabels = preset.functors.map((f) => f.label);
  const fcPos = circleLayout(fLabels, fcW / 2, panelH / 2, Math.min(fcW, panelH) * 0.28);

  const nodeR = 20;
  const fcNodeR = 30;

  // Arrowhead for source category
  const renderDefs = (color: string, id: string) => (
    <marker id={id} viewBox="0 0 10 7" refX="10" refY="3.5" markerWidth="8" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 3.5 L 0 7 z" fill={color} />
    </marker>
  );

  // Edge path with offset for arrowhead
  const edgePath = (
    x1: number, y1: number, x2: number, y2: number, r1: number, r2: number,
  ) => {
    const dx = x2 - x1;
    const dy = y2 - y1;
    const dist = Math.sqrt(dx * dx + dy * dy) || 1;
    const ux = dx / dist;
    const uy = dy / dist;
    return {
      x1: x1 + ux * r1,
      y1: y1 + uy * r1,
      x2: x2 - ux * r2,
      y2: y2 - uy * r2,
      mx: (x1 + x2) / 2,
      my: (y1 + y2) / 2,
    };
  };

  // Mini-diagram inside functor nodes
  const renderMiniDiagram = (functor: Functor, cx: number, cy: number) => {
    const objs = functor.source.objects;
    const mini = circleLayout(objs, 0, 0, 14);
    return (
      <g transform={`translate(${cx},${cy})`}>
        {functor.source.morphisms.filter((m) => !m.isIdentity).map((m) => {
          const s = mini.get(m.source);
          const t = mini.get(m.target);
          if (!s || !t) return null;
          return <line key={m.label} x1={s.x} y1={s.y} x2={t.x} y2={t.y} stroke={COLORS.muted} strokeWidth={1} opacity={0.5} />;
        })}
        {objs.map((obj, i) => {
          const p = mini.get(obj);
          if (!p) return null;
          const img = functor.onObjects.get(obj) ?? '?';
          return (
            <g key={obj}>
              <circle cx={p.x} cy={p.y} r={4} fill={COLORS.text} opacity={0.7} />
              <text x={p.x} y={p.y - 6} textAnchor="middle" fontSize={7} fill={COLORS.muted}>{img}</text>
            </g>
          );
        })}
      </g>
    );
  };

  return (
    <div ref={containerRef} style={{ background: COLORS.bg, borderRadius: 8, padding: 12, fontFamily: 'system-ui, sans-serif' }}>
      {/* Controls */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16, marginBottom: 12, alignItems: 'center', fontSize: 13 }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: 6, color: COLORS.text }}>
          Source category:
          <select
            value={sourceType}
            onChange={(e) => { setSourceType(e.target.value as 'two' | 'triangle'); setSelectedNt(null); }}
            style={{ padding: '4px 8px', borderRadius: 4, border: `1px solid ${COLORS.edge}`, fontSize: 13 }}
          >
            <option value="two">Two objects</option>
            <option value="triangle">Three objects (triangle)</option>
          </select>
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: 6, color: COLORS.text }}>
          Functors:
          <select
            value={numFunctors}
            onChange={(e) => { setNumFunctors(Number(e.target.value) as 2 | 3); setSelectedNt(null); }}
            style={{ padding: '4px 8px', borderRadius: 4, border: `1px solid ${COLORS.edge}`, fontSize: 13 }}
          >
            <option value={2}>2</option>
            <option value={3}>3</option>
          </select>
        </label>
        {numFunctors >= 3 && (
          <label style={{ display: 'flex', alignItems: 'center', gap: 6, color: COLORS.text, cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={showComposition}
              onChange={(e) => setShowComposition(e.target.checked)}
            />
            Show composition
          </label>
        )}
      </div>

      {/* Panels */}
      <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: 4 }}>
        {/* Source category */}
        <div style={{ width: isMobile ? '100%' : '40%' }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: COLORS.muted, marginBottom: 4, textTransform: 'uppercase', letterSpacing: 1 }}>
            Source category C
          </div>
          <svg width={srcW} height={panelH} style={{ background: '#fff', borderRadius: 6, border: `1px solid ${COLORS.edge}` }}>
            <defs>{renderDefs(COLORS.edge, markerId)}</defs>
            {/* Edges */}
            {preset.sourceCategory.morphisms.filter((m) => !m.isIdentity).map((m) => {
              const s = srcPos.get(m.source);
              const t = srcPos.get(m.target);
              if (!s || !t) return null;
              const e = edgePath(s.x, s.y, t.x, t.y, nodeR, nodeR);
              return (
                <g key={m.label}>
                  <line x1={e.x1} y1={e.y1} x2={e.x2} y2={e.y2} stroke={COLORS.edge} strokeWidth={2} markerEnd={`url(#${markerId})`} />
                  <text x={e.mx + 8} y={e.my - 6} fontSize={12} fill={COLORS.text} fontStyle="italic">{m.label}</text>
                </g>
              );
            })}
            {/* Nodes */}
            {srcObjs.map((obj) => {
              const p = srcPos.get(obj);
              if (!p) return null;
              return (
                <g key={obj}>
                  <circle cx={p.x} cy={p.y} r={nodeR} fill="#fff" stroke={COLORS.text} strokeWidth={2} />
                  <text x={p.x} y={p.y + 5} textAnchor="middle" fontSize={14} fontWeight={600} fill={COLORS.text}>{obj}</text>
                </g>
              );
            })}
          </svg>
        </div>

        {/* Functor category [C, D] */}
        <div style={{ width: isMobile ? '100%' : '60%' }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: COLORS.muted, marginBottom: 4, textTransform: 'uppercase', letterSpacing: 1 }}>
            Functor category [C, D]
          </div>
          <svg width={fcW} height={panelH} style={{ background: '#fff', borderRadius: 6, border: `1px solid ${COLORS.edge}` }}>
            <defs>
              {renderDefs(COLORS.natTrans, markerNtId)}
              {renderDefs(COLORS.composition, markerCompId)}
            </defs>
            {/* Nat trans edges */}
            {allNatTrans.map(({ label, nt, isComposition }) => {
              const srcLabel = preset.functors.find((f) => f.functor === nt.source)?.label;
              const tgtLabel = preset.functors.find((f) => f.functor === nt.target)?.label;
              if (!srcLabel || !tgtLabel) return null;
              const s = fcPos.get(srcLabel);
              const t = fcPos.get(tgtLabel);
              if (!s || !t) return null;
              const e = edgePath(s.x, s.y, t.x, t.y, fcNodeR + 2, fcNodeR + 2);
              const isHovered = hoveredNt === label;
              const isSelected = selectedNt === label;
              const color = isComposition ? COLORS.composition : COLORS.natTrans;
              // Offset composed edge slightly
              const offset = isComposition ? 12 : 0;
              const dx = t.x - s.x;
              const dy = t.y - s.y;
              const dist = Math.sqrt(dx * dx + dy * dy) || 1;
              const nx = -dy / dist * offset;
              const ny = dx / dist * offset;

              return (
                <g
                  key={label}
                  style={{ cursor: 'pointer' }}
                  onMouseEnter={() => setHoveredNt(label)}
                  onMouseLeave={() => setHoveredNt(null)}
                  onClick={() => setSelectedNt(selectedNt === label ? null : label)}
                >
                  <line
                    x1={e.x1 + nx} y1={e.y1 + ny} x2={e.x2 + nx} y2={e.y2 + ny}
                    stroke={color}
                    strokeWidth={isHovered || isSelected ? 3 : 2}
                    strokeDasharray={isComposition ? '6 3' : undefined}
                    markerEnd={`url(#${isComposition ? markerCompId : markerNtId})`}
                    opacity={isHovered || isSelected ? 1 : 0.8}
                  />
                  {/* Hit area */}
                  <line
                    x1={e.x1 + nx} y1={e.y1 + ny} x2={e.x2 + nx} y2={e.y2 + ny}
                    stroke="transparent" strokeWidth={14}
                  />
                  <text
                    x={e.mx + nx + 10} y={e.my + ny - 8}
                    fontSize={13} fontWeight={600} fill={color} fontStyle="italic"
                  >
                    {label}
                  </text>
                </g>
              );
            })}
            {/* Functor nodes */}
            {preset.functors.map(({ label, functor }, i) => {
              const p = fcPos.get(label);
              if (!p) return null;
              const color = FUNCTOR_COLORS[i] ?? COLORS.text;
              return (
                <g key={label}>
                  <circle cx={p.x} cy={p.y} r={fcNodeR} fill="#fff" stroke={color} strokeWidth={2.5} />
                  {renderMiniDiagram(functor, p.x, p.y)}
                  <text x={p.x} y={p.y - fcNodeR - 6} textAnchor="middle" fontSize={15} fontWeight={700} fill={color}>{label}</text>
                </g>
              );
            })}
          </svg>
        </div>
      </div>

      {/* Detail panel */}
      {detailData && selectedNt && (
        <div style={{ marginTop: 12, background: '#fff', borderRadius: 6, border: `1px solid ${COLORS.edge}`, padding: 12 }}>
          <div style={{ fontSize: 13, fontWeight: 600, color: COLORS.text, marginBottom: 8 }}>
            Natural transformation {selectedNt} — component details
          </div>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: `2px solid ${COLORS.edge}` }}>
                {['Object', 'F(Obj)', 'G(Obj)', 'Component', 'Naturality'].map((h) => (
                  <th key={h} style={{ textAlign: 'left', padding: '4px 8px', color: COLORS.muted, fontWeight: 600, fontSize: 11, textTransform: 'uppercase' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {detailData.map((row) => (
                <tr key={row.object} style={{ borderBottom: `1px solid ${COLORS.edge}` }}>
                  <td style={{ padding: '6px 8px', fontWeight: 600 }}>{row.object}</td>
                  <td style={{ padding: '6px 8px' }}>{row.fObj}</td>
                  <td style={{ padding: '6px 8px' }}>{row.gObj}</td>
                  <td style={{ padding: '6px 8px', fontStyle: 'italic' }}>{row.component}</td>
                  <td style={{ padding: '6px 8px', color: row.valid ? COLORS.valid : COLORS.invalid, fontWeight: 700 }}>
                    {row.valid ? '✓' : '✗'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {!selectedNt && (
        <div style={{ marginTop: 8, fontSize: 12, color: COLORS.muted, textAlign: 'center' }}>
          Click a natural transformation edge to see component details
        </div>
      )}
    </div>
  );
}

