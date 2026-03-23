import { useState, useMemo, useCallback } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';

// ─── Sigma-algebra computation ───

type Subset = number; // bitmask representation: subset of {0,1,2,3}

const OMEGA_SIZE = 4;
const FULL_SET = (1 << OMEGA_SIZE) - 1; // 0b1111 = 15

function complement(s: Subset): Subset {
  return (~s) & FULL_SET;
}

function union(a: Subset, b: Subset): Subset {
  return a | b;
}

function subsetToString(s: Subset): string {
  if (s === 0) return '∅';
  if (s === FULL_SET) return 'Ω';
  const elems: number[] = [];
  for (let i = 0; i < OMEGA_SIZE; i++) {
    if (s & (1 << i)) elems.push(i + 1);
  }
  return `{${elems.join(',')}}`;
}

function cardinality(s: Subset): number {
  let c = 0;
  for (let i = 0; i < OMEGA_SIZE; i++) {
    if (s & (1 << i)) c++;
  }
  return c;
}

function generateSigmaAlgebra(generators: Subset[]): Subset[] {
  const sigAlg = new Set<Subset>();
  sigAlg.add(0);          // empty set
  sigAlg.add(FULL_SET);   // Omega
  for (const g of generators) {
    sigAlg.add(g);
  }

  let changed = true;
  while (changed) {
    changed = false;
    const current = Array.from(sigAlg);
    for (const s of current) {
      // closure under complements
      const comp = complement(s);
      if (!sigAlg.has(comp)) { sigAlg.add(comp); changed = true; }
    }
    const updated = Array.from(sigAlg);
    for (let i = 0; i < updated.length; i++) {
      for (let j = i; j < updated.length; j++) {
        // closure under unions
        const u = union(updated[i], updated[j]);
        if (!sigAlg.has(u)) { sigAlg.add(u); changed = true; }
      }
    }
  }

  return Array.from(sigAlg).sort((a, b) => cardinality(a) - cardinality(b) || a - b);
}

function isImmediateSubset(a: Subset, b: Subset, allSets: Subset[]): boolean {
  if ((a & b) !== a || a === b) return false;
  // a ⊂ b: check no c with a ⊂ c ⊂ b in allSets
  for (const c of allSets) {
    if (c !== a && c !== b && (a & c) === a && (c & b) === c) return false;
  }
  return true;
}

// ─── Preset configurations ───

interface Preset {
  label: string;
  generators: Subset[];
}

const PRESETS: Preset[] = [
  { label: 'Trivial {∅, Ω}', generators: [] },
  { label: 'σ({1,2})', generators: [0b0011] },
  { label: 'σ({1}, {2})', generators: [0b0001, 0b0010] },
  { label: 'σ({1,2}, {2,3})', generators: [0b0011, 0b0110] },
  { label: 'Power set 2^Ω', generators: [0b0001, 0b0010, 0b0100, 0b1000] },
];

// ─── Element colors ───

const ELEM_COLORS = ['#0F6E56', '#534AB7', '#D97706', '#DC2626'];
const ELEM_POSITIONS = [
  { x: 0.35, y: 0.35 },
  { x: 0.65, y: 0.35 },
  { x: 0.35, y: 0.65 },
  { x: 0.65, y: 0.65 },
];

// ─── Component ───

export default function SigmaAlgebraExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [presetIdx, setPresetIdx] = useState(1);
  const [selectedElems, setSelectedElems] = useState<Set<number>>(new Set([0, 1])); // {1,2}

  const generators = useMemo(() => {
    if (presetIdx >= 0) return PRESETS[presetIdx].generators;
    // custom: selected elements as singleton generators
    const gens: Subset[] = [];
    const mask = Array.from(selectedElems).reduce((acc, i) => acc | (1 << i), 0);
    if (mask > 0) gens.push(mask);
    return gens;
  }, [presetIdx, selectedElems]);

  const sigmaAlgebra = useMemo(() => generateSigmaAlgebra(generators), [generators]);

  const handlePreset = useCallback((idx: number) => {
    setPresetIdx(idx);
    // Update selected elements to match the first generator visually
    const p = PRESETS[idx];
    if (p.generators.length > 0) {
      const elems = new Set<number>();
      for (const g of p.generators) {
        for (let i = 0; i < OMEGA_SIZE; i++) {
          if (g & (1 << i)) elems.add(i);
        }
      }
      setSelectedElems(elems);
    } else {
      setSelectedElems(new Set());
    }
  }, []);

  const toggleElem = useCallback((i: number) => {
    setSelectedElems(prev => {
      const next = new Set(prev);
      if (next.has(i)) next.delete(i); else next.add(i);
      return next;
    });
    setPresetIdx(-1); // switch to custom mode
  }, []);

  // Layout
  const isDesktop = (containerWidth || 0) > 640;
  const vennSize = isDesktop ? Math.min(220, containerWidth * 0.3) : Math.min(200, containerWidth * 0.45);
  const hassePanelW = isDesktop ? containerWidth - vennSize - 40 : containerWidth;
  const hassePanelH = isDesktop ? 320 : 260;

  // Hasse diagram layout: group by cardinality
  const hasseNodes = useMemo(() => {
    const layers: Map<number, Subset[]> = new Map();
    for (const s of sigmaAlgebra) {
      const c = cardinality(s);
      if (!layers.has(c)) layers.set(c, []);
      layers.get(c)!.push(s);
    }

    const nodes: { subset: Subset; x: number; y: number; label: string }[] = [];
    const layerKeys = Array.from(layers.keys()).sort();
    const numLayers = layerKeys.length;

    for (let li = 0; li < numLayers; li++) {
      const subs = layers.get(layerKeys[li])!;
      const y = hassePanelH - 30 - (li / Math.max(numLayers - 1, 1)) * (hassePanelH - 60);
      for (let si = 0; si < subs.length; si++) {
        const x = 30 + ((si + 0.5) / subs.length) * (hassePanelW - 60);
        nodes.push({ subset: subs[si], x, y, label: subsetToString(subs[si]) });
      }
    }
    return nodes;
  }, [sigmaAlgebra, hassePanelW, hassePanelH]);

  const hasseEdges = useMemo(() => {
    const edges: { from: Subset; to: Subset }[] = [];
    for (const a of sigmaAlgebra) {
      for (const b of sigmaAlgebra) {
        if (isImmediateSubset(a, b, sigmaAlgebra)) {
          edges.push({ from: a, to: b });
        }
      }
    }
    return edges;
  }, [sigmaAlgebra]);

  const nodePos = useMemo(() => {
    const map = new Map<Subset, { x: number; y: number }>();
    for (const n of hasseNodes) map.set(n.subset, { x: n.x, y: n.y });
    return map;
  }, [hasseNodes]);

  // Generator bitmask for highlighting
  const generatorSet = useMemo(() => new Set(generators), [generators]);

  return (
    <div
      ref={containerRef}
      style={{
        border: '1px solid var(--color-border)',
        borderRadius: '8px',
        padding: '16px',
        background: 'var(--color-surface)',
        marginTop: '1.5rem',
        marginBottom: '1.5rem',
      }}
    >
      <div style={{ fontFamily: 'var(--font-sans)', fontSize: '13px', fontWeight: 600, marginBottom: '12px', color: 'var(--color-text)' }}>
        Sigma-Algebra Explorer — Ω = &#123;1, 2, 3, 4&#125;
      </div>

      {/* Controls */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginBottom: '14px' }}>
        {PRESETS.map((p, i) => (
          <button
            key={i}
            onClick={() => handlePreset(i)}
            style={{
              padding: '4px 10px',
              fontSize: '12px',
              fontFamily: 'var(--font-mono)',
              border: `1px solid ${presetIdx === i ? 'var(--color-accent)' : 'var(--color-border)'}`,
              borderRadius: '4px',
              background: presetIdx === i ? 'var(--color-definition-bg)' : 'transparent',
              color: presetIdx === i ? 'var(--color-accent)' : 'var(--color-text-secondary)',
              cursor: 'pointer',
            }}
          >
            {p.label}
          </button>
        ))}
      </div>

      <div style={{ display: 'flex', flexDirection: isDesktop ? 'row' : 'column', gap: '16px' }}>
        {/* Left: Venn diagram */}
        <div style={{ flexShrink: 0 }}>
          <svg width={vennSize} height={vennSize} style={{ border: '1px solid var(--color-border)', borderRadius: '6px', background: 'var(--color-muted-bg)' }}>
            {/* Grid background circle for Omega */}
            <rect x={10} y={10} width={vennSize - 20} height={vennSize - 20} rx={8} fill="none" stroke="var(--color-border)" strokeDasharray="4 4" />
            <text x={vennSize - 16} y={24} fontSize={12} fontFamily="var(--font-serif)" fill="var(--color-text-secondary)" textAnchor="end">Ω</text>
            {/* Elements */}
            {ELEM_POSITIONS.map((pos, i) => {
              const cx = pos.x * (vennSize - 20) + 10;
              const cy = pos.y * (vennSize - 20) + 10;
              const isSelected = selectedElems.has(i);
              return (
                <g key={i} onClick={() => toggleElem(i)} style={{ cursor: 'pointer' }}>
                  <circle
                    cx={cx}
                    cy={cy}
                    r={Math.min(24, vennSize * 0.1)}
                    fill={isSelected ? ELEM_COLORS[i] : 'var(--color-muted-bg)'}
                    stroke={ELEM_COLORS[i]}
                    strokeWidth={2}
                    opacity={isSelected ? 0.85 : 0.4}
                  />
                  <text
                    x={cx}
                    y={cy + 1}
                    textAnchor="middle"
                    dominantBaseline="central"
                    fontSize={14}
                    fontWeight={600}
                    fontFamily="var(--font-mono)"
                    fill={isSelected ? '#fff' : 'var(--color-text-secondary)'}
                  >
                    {i + 1}
                  </text>
                </g>
              );
            })}
          </svg>
          <div style={{ fontSize: '11px', color: 'var(--color-text-secondary)', marginTop: '4px', fontFamily: 'var(--font-sans)' }}>
            Click elements to toggle generators
          </div>
        </div>

        {/* Right: Hasse diagram */}
        <div style={{ flex: 1, minWidth: 0 }}>
          <svg width={hassePanelW} height={hassePanelH} style={{ border: '1px solid var(--color-border)', borderRadius: '6px', background: 'var(--color-muted-bg)' }}>
            {/* Edges */}
            {hasseEdges.map((e, i) => {
              const from = nodePos.get(e.from);
              const to = nodePos.get(e.to);
              if (!from || !to) return null;
              return (
                <line
                  key={`e-${i}`}
                  x1={from.x} y1={from.y} x2={to.x} y2={to.y}
                  stroke="var(--color-border)"
                  strokeWidth={1.2}
                />
              );
            })}
            {/* Nodes */}
            {hasseNodes.map((n, i) => {
              const isGenerator = generatorSet.has(n.subset);
              const isEndpoint = n.subset === 0 || n.subset === FULL_SET;
              const fill = isGenerator
                ? 'var(--color-accent)'
                : isEndpoint
                  ? 'var(--color-theorem-border)'
                  : 'var(--color-text-secondary)';
              const fontSize = Math.max(9, Math.min(11, 140 / sigmaAlgebra.length));
              return (
                <g key={`n-${i}`}>
                  <circle cx={n.x} cy={n.y} r={Math.max(4, 18 - sigmaAlgebra.length * 0.5)} fill={fill} opacity={0.9} />
                  <text
                    x={n.x}
                    y={n.y - Math.max(8, 22 - sigmaAlgebra.length * 0.5)}
                    textAnchor="middle"
                    fontSize={fontSize}
                    fontFamily="var(--font-mono)"
                    fill="var(--color-text)"
                  >
                    {n.label}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>
      </div>

      {/* Properties readout */}
      <div style={{
        marginTop: '12px',
        display: 'flex',
        flexWrap: 'wrap',
        gap: '16px',
        fontSize: '12px',
        fontFamily: 'var(--font-mono)',
        color: 'var(--color-text-secondary)',
      }}>
        <span>|𝓕| = <strong style={{ color: 'var(--color-text)' }}>{sigmaAlgebra.length}</strong></span>
        <span>Generators: <strong style={{ color: 'var(--color-accent)' }}>{generators.length === 0 ? '∅ (trivial)' : generators.map(subsetToString).join(', ')}</strong></span>
        <span>{sigmaAlgebra.length === (1 << (1 << OMEGA_SIZE))
          ? '= 2^Ω (power set)'
          : sigmaAlgebra.length === 2
            ? '= {∅, Ω} (trivial)'
            : `⊂ 2^Ω (${sigmaAlgebra.length} of ${1 << OMEGA_SIZE} subsets)`}</span>
      </div>
    </div>
  );
}
