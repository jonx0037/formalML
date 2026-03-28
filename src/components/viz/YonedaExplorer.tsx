import { useState, useRef, useEffect, useId, useMemo, useCallback } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';

// ── Constants ────────────────────────────────────────────────────────

const SM_BREAKPOINT = 640;

const COLORS = {
  homSet: '#3b82f6',
  functorValue: '#8b5cf6',
  yonedaElement: '#f59e0b',
  arrow: '#10b981',
  category: '#06b6d4',
  bg: '#f8fafc',
  text: '#1e293b',
  muted: '#94a3b8',
  highlight: '#fef3c7',
};

// ── Data model ───────────────────────────────────────────────────────

interface Morphism {
  label: string;
  from: string;
  to: string;
}

interface SetValuedFunctor {
  name: string;
  values: Map<string, string[]>;
  action: Map<string, Map<string, string>>;
}

const OBJECTS = ['A', 'B', 'C'];

const MORPHISMS: Morphism[] = [
  { label: 'id_A', from: 'A', to: 'A' },
  { label: 'id_B', from: 'B', to: 'B' },
  { label: 'id_C', from: 'C', to: 'C' },
  { label: 'f', from: 'A', to: 'B' },
  { label: 'g', from: 'B', to: 'C' },
  { label: 'g∘f', from: 'A', to: 'C' },
];

/** Morphisms from A, keyed by target object */
const HOM_FROM_A: Map<string, string[]> = new Map([
  ['A', ['id_A']],
  ['B', ['f']],
  ['C', ['g∘f']],
]);

const FUNCTOR_F: SetValuedFunctor = {
  name: 'F',
  values: new Map([
    ['A', ['a1', 'a2']],
    ['B', ['b1', 'b2', 'b3']],
    ['C', ['c1', 'c2']],
  ]),
  action: new Map([
    ['f', new Map([['a1', 'b1'], ['a2', 'b2']])],
    ['g', new Map([['b1', 'c1'], ['b2', 'c1'], ['b3', 'c2']])],
    ['g∘f', new Map([['a1', 'c1'], ['a2', 'c1']])],
    ['id_A', new Map([['a1', 'a1'], ['a2', 'a2']])],
    ['id_B', new Map([['b1', 'b1'], ['b2', 'b2'], ['b3', 'b3']])],
    ['id_C', new Map([['c1', 'c1'], ['c2', 'c2']])],
  ]),
};

// ── Yoneda logic ─────────────────────────────────────────────────────

function computeNatTrans(
  functor: SetValuedFunctor,
  yonedaElement: string,
): Map<string, Map<string, string>> {
  const result = new Map<string, Map<string, string>>();
  for (const obj of OBJECTS) {
    const homElems = HOM_FROM_A.get(obj) ?? [];
    const componentMap = new Map<string, string>();
    for (const morph of homElems) {
      const actionMap = functor.action.get(morph);
      componentMap.set(morph, actionMap?.get(yonedaElement) ?? '?');
    }
    result.set(obj, componentMap);
  }
  return result;
}

// ── Pill component ───────────────────────────────────────────────────

function Pill({
  label,
  color,
  highlighted,
  onClick,
  small,
}: {
  label: string;
  color: string;
  highlighted?: boolean;
  onClick?: () => void;
  small?: boolean;
}) {
  const [hovered, setHovered] = useState(false);
  const size = small ? '0.7rem' : '0.8rem';
  return (
    <span
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      onClick={onClick}
      style={{
        display: 'inline-block',
        padding: small ? '1px 7px' : '2px 10px',
        margin: '2px',
        borderRadius: '9999px',
        fontSize: size,
        fontFamily: 'monospace',
        fontWeight: 600,
        color: highlighted ? '#1e293b' : '#fff',
        background: highlighted ? COLORS.highlight : color,
        border: highlighted
          ? `2px solid ${COLORS.yonedaElement}`
          : `2px solid ${hovered ? '#fff' : 'transparent'}`,
        cursor: onClick ? 'pointer' : 'default',
        transition: 'all 0.2s ease',
        transform: hovered && onClick ? 'scale(1.08)' : 'scale(1)',
        boxShadow: highlighted ? `0 0 8px ${COLORS.yonedaElement}40` : 'none',
      }}
    >
      {label}
    </span>
  );
}

// ── Category diagram (SVG) ───────────────────────────────────────────

function CategoryDiagram({ width }: { width: number }) {
  const h = 140;
  const cx = width / 2;
  const cy = h / 2 + 5;
  const r = Math.min(width * 0.28, 55);

  const positions: Record<string, [number, number]> = {
    A: [cx - r, cy + r * 0.6],
    B: [cx + r, cy + r * 0.6],
    C: [cx, cy - r * 0.7],
  };

  const nonIdMorphisms = MORPHISMS.filter((m) => !m.label.startsWith('id_'));

  function arrowPath(from: [number, number], to: [number, number], curve: number) {
    const dx = to[0] - from[0];
    const dy = to[1] - from[1];
    const len = Math.sqrt(dx * dx + dy * dy);
    const nx = -dy / len;
    const ny = dx / len;
    const mx = (from[0] + to[0]) / 2 + nx * curve;
    const my = (from[1] + to[1]) / 2 + ny * curve;
    return `M${from[0]},${from[1]} Q${mx},${my} ${to[0]},${to[1]}`;
  }

  const labelId = useId();

  return (
    <svg width={width} height={h} style={{ display: 'block' }}>
      <defs>
        <marker
          id={`${labelId}-arrow`}
          markerWidth="8"
          markerHeight="6"
          refX="8"
          refY="3"
          orient="auto"
        >
          <polygon points="0 0, 8 3, 0 6" fill={COLORS.muted} />
        </marker>
      </defs>
      {nonIdMorphisms.map((m) => {
        const from = positions[m.from];
        const to = positions[m.to];
        const curve = m.label === 'g∘f' ? -20 : 12;
        const nodeR = 16;
        const dx = to[0] - from[0];
        const dy = to[1] - from[1];
        const len = Math.sqrt(dx * dx + dy * dy);
        const ux = dx / len;
        const uy = dy / len;
        const p1: [number, number] = [from[0] + ux * nodeR, from[1] + uy * nodeR];
        const p2: [number, number] = [to[0] - ux * nodeR, to[1] - uy * nodeR];
        const mid = arrowPath(p1, p2, curve);
        const nx = -dy / len;
        const ny = dx / len;
        const lx = (from[0] + to[0]) / 2 + nx * curve * 1.6;
        const ly = (from[1] + to[1]) / 2 + ny * curve * 1.6;
        return (
          <g key={m.label}>
            <path
              d={mid}
              fill="none"
              stroke={COLORS.muted}
              strokeWidth={1.5}
              markerEnd={`url(#${labelId}-arrow)`}
            />
            <text
              x={lx}
              y={ly}
              textAnchor="middle"
              dominantBaseline="middle"
              fill={COLORS.text}
              fontSize={11}
              fontStyle="italic"
            >
              {m.label}
            </text>
          </g>
        );
      })}
      {OBJECTS.map((obj) => {
        const [x, y] = positions[obj];
        return (
          <g key={obj}>
            <circle
              cx={x}
              cy={y}
              r={16}
              fill={obj === 'A' ? COLORS.category : COLORS.bg}
              stroke={COLORS.category}
              strokeWidth={2}
            />
            <text
              x={x}
              y={y}
              textAnchor="middle"
              dominantBaseline="central"
              fill={obj === 'A' ? '#fff' : COLORS.text}
              fontSize={13}
              fontWeight={700}
            >
              {obj}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

// ── Arrow connector SVG overlay ──────────────────────────────────────

interface ArrowSpec {
  fromId: string;
  toId: string;
  label: string;
  delay: number;
}

function ArrowOverlay({
  arrows,
  containerRef,
  animKey,
}: {
  arrows: ArrowSpec[];
  containerRef: React.RefObject<HTMLDivElement | null>;
  animKey: string;
}) {
  const [dims, setDims] = useState({ w: 0, h: 0 });
  const [positions, setPositions] = useState<
    { x1: number; y1: number; x2: number; y2: number; label: string; delay: number }[]
  >([]);
  const svgId = useId();

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    setDims({ w: rect.width, h: rect.height });

    const pos = arrows
      .map((a) => {
        const fromEl = el.querySelector(`[data-pill-id="${a.fromId}"]`);
        const toEl = el.querySelector(`[data-pill-id="${a.toId}"]`);
        if (!fromEl || !toEl) return null;
        const fr = fromEl.getBoundingClientRect();
        const tr = toEl.getBoundingClientRect();
        return {
          x1: fr.right - rect.left,
          y1: fr.top + fr.height / 2 - rect.top,
          x2: tr.left - rect.left,
          y2: tr.top + tr.height / 2 - rect.top,
          label: a.label,
          delay: a.delay,
        };
      })
      .filter(Boolean) as typeof positions;

    setPositions(pos);
  }, [arrows, animKey, dims.w]);

  if (positions.length === 0) return null;

  return (
    <svg
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: dims.w,
        height: dims.h,
        pointerEvents: 'none',
        overflow: 'visible',
      }}
    >
      <defs>
        <marker
          id={`${svgId}-tip`}
          markerWidth="6"
          markerHeight="5"
          refX="6"
          refY="2.5"
          orient="auto"
        >
          <polygon points="0 0, 6 2.5, 0 5" fill={COLORS.arrow} />
        </marker>
      </defs>
      {positions.map((p, i) => {
        const mx = (p.x1 + p.x2) / 2;
        const my = (p.y1 + p.y2) / 2 - 8;
        return (
          <g
            key={i}
            style={{
              opacity: 0,
              animation: `yonedaFadeIn 0.35s ease ${p.delay}s forwards`,
            }}
          >
            <line
              x1={p.x1 + 4}
              y1={p.y1}
              x2={p.x2 - 4}
              y2={p.y2}
              stroke={COLORS.arrow}
              strokeWidth={1.8}
              strokeDasharray="4 2"
              markerEnd={`url(#${svgId}-tip)`}
            />
            <text
              x={mx}
              y={my}
              textAnchor="middle"
              fontSize={9}
              fill={COLORS.arrow}
              fontWeight={600}
            >
              {p.label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

// ── Main component ───────────────────────────────────────────────────

export default function YonedaExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const columnsRef = useRef<HTMLDivElement>(null);

  const [selectedElement, setSelectedElement] = useState<string | null>(null);
  const [reverseMode, setReverseMode] = useState(false);
  const [reverseChoices, setReverseChoices] = useState<Map<string, string>>(new Map());
  const [animKey, setAnimKey] = useState('');

  const functor = FUNCTOR_F;
  const isStacked = containerWidth > 0 && containerWidth < SM_BREAKPOINT;

  // Yoneda-computed natural transformation (forward mode)
  const natTrans = useMemo(() => {
    if (!selectedElement) return null;
    return computeNatTrans(functor, selectedElement);
  }, [selectedElement, functor]);

  // Reverse mode: extract Yoneda element
  const reverseYoneda = useMemo(() => {
    if (!reverseMode) return null;
    return reverseChoices.get('id_A') ?? null;
  }, [reverseMode, reverseChoices]);

  // Trigger animation on element change
  const handleSelectElement = useCallback((el: string) => {
    setSelectedElement(el);
    setAnimKey(`${el}-${Date.now()}`);
  }, []);

  const handleReverseChoice = useCallback(
    (morph: string, value: string) => {
      setReverseChoices((prev) => {
        const next = new Map(prev);
        next.set(morph, value);
        return next;
      });
      setAnimKey(`rev-${morph}-${value}-${Date.now()}`);
    },
    [],
  );

  // Build arrow specs for the connecting overlay
  const arrowSpecs = useMemo((): ArrowSpec[] => {
    if (reverseMode) {
      const specs: ArrowSpec[] = [];
      let delay = 0;
      for (const obj of OBJECTS) {
        const homElems = HOM_FROM_A.get(obj) ?? [];
        for (const morph of homElems) {
          const target = reverseChoices.get(morph);
          if (target) {
            specs.push({
              fromId: `hom-${morph}`,
              toId: `fun-${obj}-${target}`,
              label: `α_${obj}`,
              delay,
            });
            delay += 0.15;
          }
        }
      }
      return specs;
    }
    if (!natTrans) return [];
    const specs: ArrowSpec[] = [];
    let delay = 0;
    for (const obj of OBJECTS) {
      const component = natTrans.get(obj);
      if (!component) continue;
      for (const [morph, target] of component) {
        specs.push({
          fromId: `hom-${morph}`,
          toId: `fun-${obj}-${target}`,
          label: `α_${obj}`,
          delay,
        });
        delay += 0.15;
      }
    }
    return specs;
  }, [natTrans, reverseMode, reverseChoices]);

  // Computation summary lines
  const summaryLines = useMemo(() => {
    if (reverseMode) {
      const x = reverseYoneda;
      if (!x) return ['Select a target for each Hom element to define a natural transformation.'];
      const lines = [`Yoneda element: α_A(id_A) = ${x}`];
      // check consistency
      for (const obj of OBJECTS) {
        const homElems = HOM_FROM_A.get(obj) ?? [];
        for (const morph of homElems) {
          if (morph === 'id_A') continue;
          const chosen = reverseChoices.get(morph);
          const expected = functor.action.get(morph)?.get(x);
          if (chosen && chosen !== expected) {
            lines.push(`Warning: α_${obj}(${morph}) = ${chosen}, but F(${morph})(${x}) = ${expected} — naturality violated!`);
          } else if (chosen) {
            lines.push(`α_${obj}(${morph}) = ${chosen} = F(${morph})(${x}) ✓`);
          }
        }
      }
      return lines;
    }
    if (!selectedElement || !natTrans) return ['Pick an element x ∈ F(A) below to see the Yoneda bijection in action.'];
    const lines: string[] = [];
    lines.push(`α_A(id_A) = ${selectedElement}  (the Yoneda element itself)`);
    const fAction = natTrans.get('B');
    if (fAction) {
      const res = fAction.get('f');
      lines.push(`α_B(f) = F(f)(${selectedElement}) = ${res}`);
    }
    const gfAction = natTrans.get('C');
    if (gfAction) {
      const res = gfAction.get('g∘f');
      lines.push(`α_C(g∘f) = F(g∘f)(${selectedElement}) = ${res}`);
    }
    return lines;
  }, [selectedElement, natTrans, reverseMode, reverseChoices, reverseYoneda, functor]);

  return (
    <div
      ref={containerRef}
      style={{
        background: COLORS.bg,
        borderRadius: 12,
        border: '1px solid #e2e8f0',
        padding: 16,
        fontFamily: 'system-ui, -apple-system, sans-serif',
        color: COLORS.text,
        position: 'relative',
      }}
    >
      {/* Keyframe injection */}
      <style>{`
        @keyframes yonedaFadeIn {
          from { opacity: 0; transform: translateX(-6px); }
          to   { opacity: 1; transform: translateX(0); }
        }
      `}</style>

      {/* Header */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: 8,
          marginBottom: 12,
        }}
      >
        <div style={{ fontSize: '0.95rem', fontWeight: 700 }}>Yoneda Lemma Explorer</div>
        <button
          onClick={() => {
            setReverseMode((v) => !v);
            setSelectedElement(null);
            setReverseChoices(new Map());
            setAnimKey('');
          }}
          style={{
            padding: '4px 12px',
            borderRadius: 6,
            border: `1px solid ${COLORS.muted}`,
            background: reverseMode ? COLORS.arrow : '#fff',
            color: reverseMode ? '#fff' : COLORS.text,
            fontSize: '0.78rem',
            fontWeight: 600,
            cursor: 'pointer',
            transition: 'all 0.2s',
          }}
        >
          {reverseMode ? '← Forward mode' : 'Reverse direction →'}
        </button>
      </div>

      {/* Category diagram */}
      <div style={{ textAlign: 'center', marginBottom: 4 }}>
        <div style={{ fontSize: '0.72rem', color: COLORS.muted, marginBottom: 2 }}>
          Category C
        </div>
        <CategoryDiagram width={Math.min(containerWidth - 32, 320)} />
      </div>

      {/* Hom / Functor columns */}
      <div
        ref={columnsRef}
        style={{
          display: 'flex',
          flexDirection: isStacked ? 'column' : 'row',
          gap: isStacked ? 12 : 20,
          marginBottom: 16,
          position: 'relative',
        }}
      >
        {/* Left: Hom sets */}
        <div style={{ flex: 1 }}>
          <div
            style={{
              fontSize: '0.78rem',
              fontWeight: 700,
              color: COLORS.homSet,
              marginBottom: 6,
              textAlign: 'center',
            }}
          >
            Hom(A, –)
          </div>
          {OBJECTS.map((obj) => {
            const elems = HOM_FROM_A.get(obj) ?? [];
            return (
              <div
                key={obj}
                style={{
                  background: '#eff6ff',
                  borderRadius: 8,
                  padding: '6px 10px',
                  marginBottom: 6,
                  border: `1px solid ${COLORS.homSet}30`,
                }}
              >
                <div
                  style={{
                    fontSize: '0.7rem',
                    color: COLORS.homSet,
                    fontWeight: 600,
                    marginBottom: 3,
                  }}
                >
                  Hom(A, {obj})
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
                  {elems.map((e) => (
                    <span key={e} data-pill-id={`hom-${e}`}>
                      <Pill label={e} color={COLORS.homSet} small />
                    </span>
                  ))}
                </div>
              </div>
            );
          })}
        </div>

        {/* Right: Functor values */}
        <div style={{ flex: 1 }}>
          <div
            style={{
              fontSize: '0.78rem',
              fontWeight: 700,
              color: COLORS.functorValue,
              marginBottom: 6,
              textAlign: 'center',
            }}
          >
            {functor.name}(–)
          </div>
          {OBJECTS.map((obj) => {
            const elems = functor.values.get(obj) ?? [];
            return (
              <div
                key={obj}
                style={{
                  background: '#f5f3ff',
                  borderRadius: 8,
                  padding: '6px 10px',
                  marginBottom: 6,
                  border: `1px solid ${COLORS.functorValue}30`,
                }}
              >
                <div
                  style={{
                    fontSize: '0.7rem',
                    color: COLORS.functorValue,
                    fontWeight: 600,
                    marginBottom: 3,
                  }}
                >
                  {functor.name}({obj})
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
                  {elems.map((e) => {
                    // In reverse mode, allow clicking functor elements as targets
                    const isTarget =
                      !reverseMode &&
                      natTrans &&
                      Array.from(natTrans.get(obj)?.values() ?? []).includes(e);
                    const isYonedaEl =
                      !reverseMode && obj === 'A' && e === selectedElement;
                    const isReverseTarget =
                      reverseMode &&
                      Array.from(reverseChoices.values()).includes(e) &&
                      (() => {
                        // Check if this element is chosen for a morphism to obj
                        const homElems = HOM_FROM_A.get(obj) ?? [];
                        return homElems.some((m) => reverseChoices.get(m) === e);
                      })();

                    return (
                      <span key={e} data-pill-id={`fun-${obj}-${e}`}>
                        <Pill
                          label={e}
                          color={COLORS.functorValue}
                          highlighted={isYonedaEl || isTarget || isReverseTarget}
                          onClick={
                            reverseMode
                              ? () => {
                                  // In reverse mode, clicking a functor element assigns it
                                  // as the target for the corresponding Hom element
                                  const homElems = HOM_FROM_A.get(obj) ?? [];
                                  if (homElems.length === 1) {
                                    handleReverseChoice(homElems[0], e);
                                  }
                                }
                              : undefined
                          }
                          small
                        />
                      </span>
                    );
                  })}
                </div>
                {reverseMode && (
                  <div style={{ marginTop: 4 }}>
                    {(HOM_FROM_A.get(obj) ?? []).map((morph) => {
                      const chosen = reverseChoices.get(morph);
                      return (
                        <div
                          key={morph}
                          style={{
                            fontSize: '0.68rem',
                            color: COLORS.muted,
                            display: 'flex',
                            alignItems: 'center',
                            gap: 4,
                            marginTop: 2,
                          }}
                        >
                          <span style={{ fontFamily: 'monospace' }}>α({morph}) →</span>
                          {(functor.values.get(obj) ?? []).map((e) => (
                            <span
                              key={e}
                              onClick={() => handleReverseChoice(morph, e)}
                              style={{
                                display: 'inline-block',
                                padding: '0 5px',
                                borderRadius: 4,
                                fontSize: '0.66rem',
                                fontFamily: 'monospace',
                                background: chosen === e ? COLORS.highlight : '#f1f5f9',
                                border:
                                  chosen === e
                                    ? `1.5px solid ${COLORS.yonedaElement}`
                                    : '1.5px solid transparent',
                                cursor: 'pointer',
                                fontWeight: chosen === e ? 700 : 400,
                              }}
                            >
                              {e}
                            </span>
                          ))}
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Arrow overlay */}
        {!isStacked && (
          <ArrowOverlay
            arrows={arrowSpecs}
            containerRef={columnsRef}
            animKey={animKey}
          />
        )}
      </div>

      {/* Yoneda element selector (forward mode) */}
      {!reverseMode && (
        <div
          style={{
            background: COLORS.highlight,
            borderRadius: 8,
            padding: '10px 14px',
            marginBottom: 10,
            border: `1px solid ${COLORS.yonedaElement}40`,
          }}
        >
          <div
            style={{
              fontSize: '0.78rem',
              fontWeight: 700,
              marginBottom: 6,
              color: COLORS.text,
            }}
          >
            Pick x ∈ {functor.name}(A):
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {(functor.values.get('A') ?? []).map((el) => (
              <button
                key={el}
                onClick={() => handleSelectElement(el)}
                style={{
                  padding: '4px 14px',
                  borderRadius: 9999,
                  border:
                    selectedElement === el
                      ? `2px solid ${COLORS.yonedaElement}`
                      : '2px solid #e2e8f0',
                  background: selectedElement === el ? COLORS.yonedaElement : '#fff',
                  color: selectedElement === el ? '#fff' : COLORS.text,
                  fontFamily: 'monospace',
                  fontWeight: 700,
                  fontSize: '0.82rem',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                  boxShadow:
                    selectedElement === el
                      ? `0 0 10px ${COLORS.yonedaElement}50`
                      : 'none',
                }}
              >
                {el}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Computation summary */}
      <div
        style={{
          background: '#fff',
          borderRadius: 8,
          padding: '10px 14px',
          border: '1px solid #e2e8f0',
        }}
      >
        <div
          style={{
            fontSize: '0.72rem',
            fontWeight: 700,
            color: COLORS.muted,
            textTransform: 'uppercase',
            letterSpacing: '0.05em',
            marginBottom: 6,
          }}
        >
          {reverseMode ? 'Reverse Yoneda — extract the element' : 'Induced natural transformation'}
        </div>
        {summaryLines.map((line, i) => {
          const isWarning = line.startsWith('Warning');
          const isCheck = line.includes('✓');
          return (
            <div
              key={i}
              style={{
                fontFamily: 'monospace',
                fontSize: '0.78rem',
                padding: '2px 0',
                color: isWarning ? '#dc2626' : isCheck ? COLORS.arrow : COLORS.text,
                fontWeight: i === 0 ? 700 : 400,
                opacity: 0,
                animation: `yonedaFadeIn 0.3s ease ${i * 0.1}s forwards`,
              }}
            >
              {line}
            </div>
          );
        })}
        {!reverseMode && selectedElement && (
          <div
            style={{
              marginTop: 8,
              padding: '6px 10px',
              background: '#f0fdf4',
              borderRadius: 6,
              fontSize: '0.74rem',
              color: COLORS.arrow,
              fontWeight: 600,
              border: `1px solid ${COLORS.arrow}30`,
            }}
          >
            One element x = {selectedElement} determines the entire natural transformation
            — that is the Yoneda bijection.
          </div>
        )}
      </div>
    </div>
  );
}
