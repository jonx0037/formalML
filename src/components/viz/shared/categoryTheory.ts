// ============================================================================
// Category Theory utilities for the Category Theory track.
// Covers category construction, axiom verification, functor verification,
// preset categories/functors, composition helpers, and universal properties.
// ============================================================================

import type { CategoryMorphism } from './types';

// === Types ===

/** Re-export CategoryMorphism as Morphism for convenience within this module. */
export type Morphism = CategoryMorphism;

export interface Category {
  objects: string[];
  morphisms: Morphism[];
  compose: (g: string, f: string) => string | null;
  identity: (obj: string) => string;
}

export interface Functor {
  source: Category;
  target: Category;
  onObjects: Map<string, string>;
  onMorphisms: Map<string, string>;
  contravariant: boolean;
}

// === Category Construction ===

/** Create a discrete category (only identity morphisms). */
export function discreteCategory(objects: string[]): Category {
  const morphisms: Morphism[] = objects.map((o) => ({
    label: `id_${o}`,
    source: o,
    target: o,
    isIdentity: true,
  }));
  return {
    objects,
    morphisms,
    compose: (g, f) => {
      const gm = morphisms.find((m) => m.label === g);
      const fm = morphisms.find((m) => m.label === f);
      if (!gm || !fm || gm.source !== fm.target) return null;
      if (gm.isIdentity) return f;
      if (fm.isIdentity) return g;
      return null;
    },
    identity: (obj) => `id_${obj}`,
  };
}

/** Create a category from a poset (partial order as morphisms). */
export function posetCategory(
  elements: string[],
  leq: [string, string][],
): Category {
  // Compute transitive closure
  const pairs = new Set<string>();
  elements.forEach((e) => pairs.add(`${e},${e}`));
  leq.forEach(([a, b]) => pairs.add(`${a},${b}`));

  // Transitive closure via Floyd-Warshall style
  let changed = true;
  while (changed) {
    changed = false;
    for (const ab of pairs) {
      const [a, b] = ab.split(',');
      for (const bc of pairs) {
        const [b2, c] = bc.split(',');
        if (b === b2 && !pairs.has(`${a},${c}`)) {
          pairs.add(`${a},${c}`);
          changed = true;
        }
      }
    }
  }

  const morphisms: Morphism[] = [];
  for (const pair of pairs) {
    const [a, b] = pair.split(',');
    const isId = a === b;
    morphisms.push({
      label: isId ? `id_${a}` : `${a}≤${b}`,
      source: a,
      target: b,
      isIdentity: isId,
    });
  }

  return {
    objects: elements,
    morphisms,
    compose: (g, f) => {
      const gm = morphisms.find((m) => m.label === g);
      const fm = morphisms.find((m) => m.label === f);
      if (!gm || !fm || gm.source !== fm.target) return null;
      // In a poset, the composition a≤b then b≤c is a≤c
      const result = morphisms.find(
        (m) => m.source === fm.source && m.target === gm.target,
      );
      return result ? result.label : null;
    },
    identity: (obj) => `id_${obj}`,
  };
}

/** Create a monoid category (one object, elements as morphisms). */
export function monoidCategory(
  elements: string[],
  multiply: (a: string, b: string) => string,
  identityEl: string,
): Category {
  const obj = '*';
  const morphisms: Morphism[] = elements.map((e) => ({
    label: e,
    source: obj,
    target: obj,
    isIdentity: e === identityEl,
  }));

  return {
    objects: [obj],
    morphisms,
    compose: (g, f) => {
      if (!elements.includes(g) || !elements.includes(f)) return null;
      const result = multiply(g, f);
      return elements.includes(result) ? result : null;
    },
    identity: () => identityEl,
  };
}

/** Create the category with objects A, B, C and morphisms f: A→B, g: B→C, g∘f: A→C. */
export function triangleCategory(): Category {
  const objects = ['A', 'B', 'C'];
  const morphisms: Morphism[] = [
    { label: 'id_A', source: 'A', target: 'A', isIdentity: true },
    { label: 'id_B', source: 'B', target: 'B', isIdentity: true },
    { label: 'id_C', source: 'C', target: 'C', isIdentity: true },
    { label: 'f', source: 'A', target: 'B', isIdentity: false },
    { label: 'g', source: 'B', target: 'C', isIdentity: false },
    { label: 'g∘f', source: 'A', target: 'C', isIdentity: false },
  ];

  const compMap = new Map<string, string>([
    ['id_A,id_A', 'id_A'],
    ['id_B,id_B', 'id_B'],
    ['id_C,id_C', 'id_C'],
    ['f,id_A', 'f'],
    ['id_B,f', 'f'],
    ['g,id_B', 'g'],
    ['id_C,g', 'g'],
    ['g∘f,id_A', 'g∘f'],
    ['id_C,g∘f', 'g∘f'],
    ['g,f', 'g∘f'],
  ]);

  return {
    objects,
    morphisms,
    compose: (g, f) => compMap.get(`${g},${f}`) ?? null,
    identity: (obj) => `id_${obj}`,
  };
}

/** Create a two-object category with a single morphism A → B. */
export function twoObjectCategory(): Category {
  const objects = ['A', 'B'];
  const morphisms: Morphism[] = [
    { label: 'id_A', source: 'A', target: 'A', isIdentity: true },
    { label: 'id_B', source: 'B', target: 'B', isIdentity: true },
    { label: 'f', source: 'A', target: 'B', isIdentity: false },
  ];

  const compMap = new Map<string, string>([
    ['id_A,id_A', 'id_A'],
    ['id_B,id_B', 'id_B'],
    ['f,id_A', 'f'],
    ['id_B,f', 'f'],
  ]);

  return {
    objects,
    morphisms,
    compose: (g, f) => compMap.get(`${g},${f}`) ?? null,
    identity: (obj) => `id_${obj}`,
  };
}

/** Create a category with n parallel morphisms A → B. */
export function parallelCategory(n: number): Category {
  const objects = ['A', 'B'];
  const morphisms: Morphism[] = [
    { label: 'id_A', source: 'A', target: 'A', isIdentity: true },
    { label: 'id_B', source: 'B', target: 'B', isIdentity: true },
  ];

  for (let i = 1; i <= n; i++) {
    morphisms.push({
      label: `f${i}`,
      source: 'A',
      target: 'B',
      isIdentity: false,
    });
  }

  return {
    objects,
    morphisms,
    compose: (g, f) => {
      const gm = morphisms.find((m) => m.label === g);
      const fm = morphisms.find((m) => m.label === f);
      if (!gm || !fm || gm.source !== fm.target) return null;
      if (gm.isIdentity) return f;
      if (fm.isIdentity) return g;
      return null;
    },
    identity: (obj) => `id_${obj}`,
  };
}

// === Preset Categories ===

/** Z/3 as a one-object category (addition mod 3). */
export function presetZ3(): Category {
  return monoidCategory(
    ['0', '1', '2'],
    (a, b) => String((parseInt(a) + parseInt(b)) % 3),
    '0',
  );
}

/** A preset small "Set" category: three sets with a couple of functions. */
export function presetSet(): Category {
  const objects = ['{1,2}', '{a,b,c}', '{x}'];
  const morphisms: Morphism[] = [
    {
      label: 'id_{1,2}',
      source: '{1,2}',
      target: '{1,2}',
      isIdentity: true,
    },
    {
      label: 'id_{a,b,c}',
      source: '{a,b,c}',
      target: '{a,b,c}',
      isIdentity: true,
    },
    { label: 'id_{x}', source: '{x}', target: '{x}', isIdentity: true },
    {
      label: 'f',
      source: '{1,2}',
      target: '{a,b,c}',
      isIdentity: false,
    },
    {
      label: 'g',
      source: '{a,b,c}',
      target: '{x}',
      isIdentity: false,
    },
    {
      label: 'g∘f',
      source: '{1,2}',
      target: '{x}',
      isIdentity: false,
    },
  ];

  const compMap = new Map<string, string>([
    ['id_{1,2},id_{1,2}', 'id_{1,2}'],
    ['id_{a,b,c},id_{a,b,c}', 'id_{a,b,c}'],
    ['id_{x},id_{x}', 'id_{x}'],
    ['f,id_{1,2}', 'f'],
    ['id_{a,b,c},f', 'f'],
    ['g,id_{a,b,c}', 'g'],
    ['id_{x},g', 'g'],
    ['g,f', 'g∘f'],
    ['g∘f,id_{1,2}', 'g∘f'],
    ['id_{x},g∘f', 'g∘f'],
  ]);

  return {
    objects,
    morphisms,
    compose: (g, f) => compMap.get(`${g},${f}`) ?? null,
    identity: (obj) => `id_${obj}`,
  };
}

/** A preset small "Vec" category: R, R², R³ with linear maps. */
export function presetVec(): Category {
  const objects = ['ℝ', 'ℝ²', 'ℝ³'];
  const morphisms: Morphism[] = [
    { label: 'id_ℝ', source: 'ℝ', target: 'ℝ', isIdentity: true },
    { label: 'id_ℝ²', source: 'ℝ²', target: 'ℝ²', isIdentity: true },
    { label: 'id_ℝ³', source: 'ℝ³', target: 'ℝ³', isIdentity: true },
    { label: 'T', source: 'ℝ', target: 'ℝ²', isIdentity: false },
    { label: 'S', source: 'ℝ²', target: 'ℝ³', isIdentity: false },
    { label: 'S∘T', source: 'ℝ', target: 'ℝ³', isIdentity: false },
  ];

  const compMap = new Map<string, string>([
    ['id_ℝ,id_ℝ', 'id_ℝ'],
    ['id_ℝ²,id_ℝ²', 'id_ℝ²'],
    ['id_ℝ³,id_ℝ³', 'id_ℝ³'],
    ['T,id_ℝ', 'T'],
    ['id_ℝ²,T', 'T'],
    ['S,id_ℝ²', 'S'],
    ['id_ℝ³,S', 'S'],
    ['S,T', 'S∘T'],
    ['S∘T,id_ℝ', 'S∘T'],
    ['id_ℝ³,S∘T', 'S∘T'],
  ]);

  return {
    objects,
    morphisms,
    compose: (g, f) => compMap.get(`${g},${f}`) ?? null,
    identity: (obj) => `id_${obj}`,
  };
}

// === Axiom Verification ===

/** Check associativity for all composable triples. */
export function checkAssociativity(cat: Category): {
  valid: boolean;
  violations: [string, string, string][];
} {
  const violations: [string, string, string][] = [];
  const allMorphisms = cat.morphisms;

  for (const h of allMorphisms) {
    for (const g of allMorphisms) {
      for (const f of allMorphisms) {
        // Check if h ∘ (g ∘ f) and (h ∘ g) ∘ f are both defined
        const gf = cat.compose(g.label, f.label);
        const hg = cat.compose(h.label, g.label);

        if (gf !== null && hg !== null) {
          const h_gf = cat.compose(h.label, gf);
          const hg_f = cat.compose(hg, f.label);

          // Flag any failure: either side undefined, or both defined but unequal
          if (h_gf === null || hg_f === null || h_gf !== hg_f) {
            violations.push([h.label, g.label, f.label]);
          }
        }
      }
    }
  }

  return { valid: violations.length === 0, violations };
}

/** Check that identities satisfy the identity law. */
export function checkIdentity(cat: Category): {
  valid: boolean;
  violations: string[];
} {
  const violations: string[] = [];

  for (const obj of cat.objects) {
    const idLabel = cat.identity(obj);
    const idMorphism = cat.morphisms.find((m) => m.label === idLabel);
    if (!idMorphism) {
      violations.push(obj);
      continue;
    }

    // Check f ∘ id_A = f for all f with source = obj
    for (const f of cat.morphisms) {
      if (f.source === obj) {
        const result = cat.compose(f.label, idLabel);
        if (result !== f.label) {
          violations.push(obj);
          break;
        }
      }
      // Check id_B ∘ f = f for all f with target = obj
      if (f.target === obj) {
        const result = cat.compose(idLabel, f.label);
        if (result !== f.label) {
          violations.push(obj);
          break;
        }
      }
    }
  }

  return { valid: violations.length === 0, violations };
}

// === Functor Verification ===

/** Check that a functor preserves identities. */
export function checkFunctorIdentity(func: Functor): {
  valid: boolean;
  violations: string[];
} {
  const violations: string[] = [];

  for (const obj of func.source.objects) {
    const srcId = func.source.identity(obj);
    const mappedObj = func.onObjects.get(obj);
    if (!mappedObj) {
      violations.push(obj);
      continue;
    }
    const tgtId = func.target.identity(mappedObj);
    const mappedId = func.onMorphisms.get(srcId);

    if (mappedId !== tgtId) {
      violations.push(obj);
    }
  }

  return { valid: violations.length === 0, violations };
}

/** Check that a functor preserves composition. */
export function checkFunctorComposition(func: Functor): {
  valid: boolean;
  violations: [string, string][];
} {
  const violations: [string, string][] = [];

  for (const g of func.source.morphisms) {
    for (const f of func.source.morphisms) {
      const gf = func.source.compose(g.label, f.label);
      if (gf === null) continue;

      const Fg = func.onMorphisms.get(g.label);
      const Ff = func.onMorphisms.get(f.label);
      const Fgf = func.onMorphisms.get(gf);

      if (!Fg || !Ff || !Fgf) {
        violations.push([g.label, f.label]);
        continue;
      }

      const composed = func.contravariant
        ? func.target.compose(Ff, Fg)
        : func.target.compose(Fg, Ff);

      if (composed !== Fgf) {
        violations.push([g.label, f.label]);
      }
    }
  }

  return { valid: violations.length === 0, violations };
}

// === Preset Functors ===

/** Forgetful functor Vec → Set (forget linear structure). */
export function forgetfulVecToSet(): Functor {
  const source = presetVec();
  const target = presetSet();

  return {
    source,
    target,
    onObjects: new Map([
      ['ℝ', '{1,2}'],
      ['ℝ²', '{a,b,c}'],
      ['ℝ³', '{x}'],
    ]),
    onMorphisms: new Map([
      ['id_ℝ', 'id_{1,2}'],
      ['id_ℝ²', 'id_{a,b,c}'],
      ['id_ℝ³', 'id_{x}'],
      ['T', 'f'],
      ['S', 'g'],
      ['S∘T', 'g∘f'],
    ]),
    contravariant: false,
  };
}

// === Composition Helpers ===

/** Compute the composition table for a category. Returns { labels, table }. */
export function compositionTable(cat: Category): {
  labels: string[];
  table: (string | null)[][];
} {
  const labels = cat.morphisms.map((m) => m.label);
  const table: (string | null)[][] = labels.map((g) =>
    labels.map((f) => cat.compose(g, f)),
  );
  return { labels, table };
}

/** Get all composable pairs in a category. */
export function composablePairs(cat: Category): [Morphism, Morphism][] {
  const pairs: [Morphism, Morphism][] = [];
  for (const g of cat.morphisms) {
    for (const f of cat.morphisms) {
      if (g.source === f.target) {
        pairs.push([g, f]);
      }
    }
  }
  return pairs;
}

// === Universal Property Helpers ===

/** Compute the product mediating morphism in Set.
 *  Given f: Z → A and g: Z → B, returns h: Z → A × B where h(z) = (f(z), g(z)).
 */
export function productMediatingSet(
  zElements: string[],
  f: Map<string, string>,
  g: Map<string, string>,
): Map<string, [string, string]> {
  const h = new Map<string, [string, string]>();
  for (const z of zElements) {
    const a = f.get(z);
    const b = g.get(z);
    if (a !== undefined && b !== undefined) {
      h.set(z, [a, b]);
    }
  }
  return h;
}

/** Compute the coproduct mediating morphism in Set.
 *  Given f: A → Z and g: B → Z, returns h: A ⊔ B → Z.
 */
export function coproductMediatingSet(
  f: Map<string, string>,
  g: Map<string, string>,
): Map<string, string> {
  const h = new Map<string, string>();
  for (const [a, z] of f) {
    h.set(`ι₁(${a})`, z);
  }
  for (const [b, z] of g) {
    h.set(`ι₂(${b})`, z);
  }
  return h;
}

// === Category Presets for Viz ===

/** Get all preset categories for the CategoryExplorer. */
export function getCategoryPresets(): {
  name: string;
  build: () => Category;
}[] {
  return [
    { name: 'Two Objects (one arrow)', build: twoObjectCategory },
    { name: 'Three Objects (triangle)', build: triangleCategory },
    {
      name: 'Poset {a ≤ b ≤ c}',
      build: () => posetCategory(['a', 'b', 'c'], [['a', 'b'], ['b', 'c']]),
    },
    { name: 'Monoid (ℤ/3, +)', build: presetZ3 },
    {
      name: 'Discrete (3 objects)',
      build: () => discreteCategory(['A', 'B', 'C']),
    },
  ];
}
