// ============================================================================
// Category Theory utilities for the Category Theory track.
// Covers category construction, axiom verification, functor verification,
// preset categories/functors, composition helpers, universal properties,
// natural transformations, naturality verification, and the Yoneda lemma.
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

// ============================================================================
// Natural Transformations — Topic 2 extensions
// ============================================================================

// === Natural Transformation Types ===

export interface NaturalTransformation {
  source: Functor;
  target: Functor;
  /** Maps source-category object labels to component morphism labels in the target category. */
  components: Map<string, string>;
}

export interface FunctorCategory {
  sourceCategory: Category;
  targetCategory: Category;
  functors: Functor[];
  natTransformations: NaturalTransformation[];
}

// === Naturality Verification ===

/** Check the naturality condition G(f) ∘ α_A = α_B ∘ F(f) for all morphisms f in the source category. */
export function checkNaturality(
  nat: NaturalTransformation,
  sourceCategory: Category,
  targetCategory: Category,
): { valid: boolean; violations: { morphism: string; left: string; right: string }[] } {
  const violations: { morphism: string; left: string; right: string }[] = [];
  const F = nat.source;
  const G = nat.target;

  for (const f of sourceCategory.morphisms) {
    if (f.isIdentity) continue;

    const A = f.source;
    const B = f.target;

    const alphaA = nat.components.get(A);
    const alphaB = nat.components.get(B);
    const Ff = F.onMorphisms.get(f.label);
    const Gf = G.onMorphisms.get(f.label);

    if (!alphaA || !alphaB || !Ff || !Gf) {
      violations.push({
        morphism: f.label,
        left: 'undefined',
        right: 'undefined',
      });
      continue;
    }

    // Top-then-right: G(f) ∘ α_A
    const leftPath = targetCategory.compose(Gf, alphaA);
    // Down-then-across: α_B ∘ F(f)
    const rightPath = targetCategory.compose(alphaB, Ff);

    if (leftPath === null || rightPath === null || leftPath !== rightPath) {
      violations.push({
        morphism: f.label,
        left: leftPath ?? 'undefined',
        right: rightPath ?? 'undefined',
      });
    }
  }

  return { valid: violations.length === 0, violations };
}

// === Natural Transformation Composition ===

/** Vertical composition: (β ∘ α)_A = β_A ∘ α_A for α: F => G, β: G => H. */
export function verticalCompose(
  alpha: NaturalTransformation,
  beta: NaturalTransformation,
  targetCategory: Category,
): NaturalTransformation {
  const components = new Map<string, string>();

  for (const obj of alpha.source.source.objects) {
    const alphaA = alpha.components.get(obj);
    const betaA = beta.components.get(obj);
    if (alphaA && betaA) {
      const composed = targetCategory.compose(betaA, alphaA);
      if (composed) {
        components.set(obj, composed);
      }
    }
  }

  return {
    source: alpha.source,
    target: beta.target,
    components,
  };
}

/** Horizontal composition: (β * α)_A = β_{G(A)} ∘ F'(α_A) for α: F => G, β: F' => G'. */
export function horizontalCompose(
  alpha: NaturalTransformation,
  beta: NaturalTransformation,
  _middleCategory: Category,
  targetCategory: Category,
): NaturalTransformation {
  const components = new Map<string, string>();
  const G = alpha.target;
  const Fprime = beta.source;

  for (const obj of alpha.source.source.objects) {
    const GA = G.onObjects.get(obj);
    const alphaA = alpha.components.get(obj);
    if (!GA || !alphaA) continue;

    const betaGA = beta.components.get(GA);
    const FprimeAlphaA = Fprime.onMorphisms.get(alphaA);
    if (!betaGA || !FprimeAlphaA) continue;

    const composed = targetCategory.compose(betaGA, FprimeAlphaA);
    if (composed) {
      components.set(obj, composed);
    }
  }

  return {
    source: alpha.source,
    target: beta.target,
    components,
  };
}

/** Identity natural transformation: id_F with components id_{F(A)} at each object A. */
export function identityNatTrans(functor: Functor, targetCategory: Category): NaturalTransformation {
  const components = new Map<string, string>();

  for (const obj of functor.source.objects) {
    const FA = functor.onObjects.get(obj);
    if (FA) {
      components.set(obj, targetCategory.identity(FA));
    }
  }

  return { source: functor, target: functor, components };
}

// === Natural Isomorphism ===

/** Check if every component of a natural transformation is an isomorphism. */
export function isNaturalIsomorphism(
  nat: NaturalTransformation,
  targetCategory: Category,
): boolean {
  for (const obj of nat.source.source.objects) {
    const component = nat.components.get(obj);
    if (!component) return false;

    const m = targetCategory.morphisms.find((mor) => mor.label === component);
    if (!m) return false;

    // Look for an inverse: a morphism m' with m'.source = m.target, m'.target = m.source,
    // such that compose(m', m) = id_source and compose(m, m') = id_target
    const inverse = targetCategory.morphisms.find(
      (candidate) =>
        candidate.source === m.target &&
        candidate.target === m.source &&
        targetCategory.compose(candidate.label, m.label) === targetCategory.identity(m.source) &&
        targetCategory.compose(m.label, candidate.label) === targetCategory.identity(m.target),
    );

    if (!inverse) return false;
  }

  return true;
}

// === Yoneda Lemma ===

/**
 * Yoneda forward: given x ∈ F(A), construct the natural transformation α^x
 * where α^x_B(f) = F(f)(x) for each f ∈ Hom(A, B).
 *
 * elementAction maps morphism labels to element-to-element functions:
 * for morphism f, elementAction.get(f) is a Map from F(source(f)) to F(target(f)).
 */
export function yonedaForward(
  x: string,
  A: string,
  sourceCategory: Category,
  elementAction: Map<string, Map<string, string>>,
): Map<string, Map<string, string>> {
  // Returns: for each object B, a map from Hom(A, B) elements to F(B) elements
  const result = new Map<string, Map<string, string>>();

  for (const obj of sourceCategory.objects) {
    const componentMap = new Map<string, string>();
    // For each morphism f: A -> obj
    for (const f of sourceCategory.morphisms) {
      if (f.source === A && f.target === obj) {
        if (f.isIdentity) {
          // α^x_A(id_A) = x
          componentMap.set(f.label, x);
        } else {
          // α^x_B(f) = F(f)(x)
          const action = elementAction.get(f.label);
          if (action) {
            const result_elem = action.get(x);
            if (result_elem) {
              componentMap.set(f.label, result_elem);
            }
          }
        }
      }
    }
    result.set(obj, componentMap);
  }

  return result;
}

/** Yoneda reverse: extract the Yoneda element x = α_A(id_A). */
export function yonedaReverse(
  componentAtA: Map<string, string>,
  A: string,
  identityLabel: string,
): string | null {
  return componentAtA.get(identityLabel) ?? null;
}

// === Preset Natural Transformations ===

/** Build a small Vec-like category for natural transformation examples. */
function smallVecCategory(): Category {
  const objects = ['V', 'W'];
  const morphisms: Morphism[] = [
    { label: 'id_V', source: 'V', target: 'V', isIdentity: true },
    { label: 'id_W', source: 'W', target: 'W', isIdentity: true },
    { label: 'T', source: 'V', target: 'W', isIdentity: false },
  ];

  const compMap = new Map<string, string>([
    ['id_V,id_V', 'id_V'],
    ['id_W,id_W', 'id_W'],
    ['T,id_V', 'T'],
    ['id_W,T', 'T'],
  ]);

  return {
    objects,
    morphisms,
    compose: (g, f) => compMap.get(`${g},${f}`) ?? null,
    identity: (obj) => `id_${obj}`,
  };
}

/** Build the target category for double dual / trace examples (Vec with more morphisms). */
function vecTargetCategory(): Category {
  const objects = ['V', 'W', 'V**', 'W**'];
  const morphisms: Morphism[] = [
    { label: 'id_V', source: 'V', target: 'V', isIdentity: true },
    { label: 'id_W', source: 'W', target: 'W', isIdentity: true },
    { label: 'id_V**', source: 'V**', target: 'V**', isIdentity: true },
    { label: 'id_W**', source: 'W**', target: 'W**', isIdentity: true },
    { label: 'T', source: 'V', target: 'W', isIdentity: false },
    { label: 'η_V', source: 'V', target: 'V**', isIdentity: false },
    { label: 'η_W', source: 'W', target: 'W**', isIdentity: false },
    { label: 'T**', source: 'V**', target: 'W**', isIdentity: false },
    // Single canonical label for the commuting composite T**∘η_V = η_W∘T
    { label: 'T**∘η_V', source: 'V', target: 'W**', isIdentity: false },
  ];

  const compMap = new Map<string, string>([
    ['id_V,id_V', 'id_V'],
    ['id_W,id_W', 'id_W'],
    ['id_V**,id_V**', 'id_V**'],
    ['id_W**,id_W**', 'id_W**'],
    ['T,id_V', 'T'],
    ['id_W,T', 'T'],
    ['η_V,id_V', 'η_V'],
    ['id_V**,η_V', 'η_V'],
    ['η_W,id_W', 'η_W'],
    ['id_W**,η_W', 'η_W'],
    ['T**,id_V**', 'T**'],
    ['id_W**,T**', 'T**'],
    // Naturality: both paths yield the same composite label
    ['T**,η_V', 'T**∘η_V'],
    ['η_W,T', 'T**∘η_V'],
    ['T**∘η_V,id_V', 'T**∘η_V'],
    ['id_W**,T**∘η_V', 'T**∘η_V'],
  ]);

  return {
    objects,
    morphisms,
    compose: (g, f) => compMap.get(`${g},${f}`) ?? null,
    identity: (obj) => `id_${obj}`,
  };
}

/** Double dual embedding: η: Id => (-)** in Vec.
 *  Components: η_V: V → V**, η_W: W → W**.
 *  Naturality: T** ∘ η_V = η_W ∘ T.
 */
export function doubleDualEmbedding(): {
  nat: NaturalTransformation;
  sourceCategory: Category;
  targetCategory: Category;
  availableMorphisms: string[];
  description: string;
} {
  const srcCat = smallVecCategory();
  const tgtCat = vecTargetCategory();

  const idFunctor: Functor = {
    source: srcCat,
    target: tgtCat,
    onObjects: new Map([['V', 'V'], ['W', 'W']]),
    onMorphisms: new Map([['id_V', 'id_V'], ['id_W', 'id_W'], ['T', 'T']]),
    contravariant: false,
  };

  const doubleDualFunctor: Functor = {
    source: srcCat,
    target: tgtCat,
    onObjects: new Map([['V', 'V**'], ['W', 'W**']]),
    onMorphisms: new Map([['id_V', 'id_V**'], ['id_W', 'id_W**'], ['T', 'T**']]),
    contravariant: false,
  };

  const nat: NaturalTransformation = {
    source: idFunctor,
    target: doubleDualFunctor,
    components: new Map([['V', 'η_V'], ['W', 'η_W']]),
  };

  return {
    nat,
    sourceCategory: srcCat,
    targetCategory: tgtCat,
    availableMorphisms: ['T'],
    description: 'Double dual: Id ⇒ (-)** in Vec',
  };
}

/** Determinant: det: GL_n ⇒ (-)× between Ring and Grp.
 *  Small example with rings R, S and a ring homomorphism φ: R → S.
 */
export function determinantNatTrans(): {
  nat: NaturalTransformation;
  sourceCategory: Category;
  targetCategory: Category;
  availableMorphisms: string[];
  description: string;
} {
  // Source category: Ring with objects R, S and morphism φ
  const srcCat: Category = {
    objects: ['R', 'S'],
    morphisms: [
      { label: 'id_R', source: 'R', target: 'R', isIdentity: true },
      { label: 'id_S', source: 'S', target: 'S', isIdentity: true },
      { label: 'φ', source: 'R', target: 'S', isIdentity: false },
    ],
    compose: (g, f) => {
      const map = new Map<string, string>([
        ['id_R,id_R', 'id_R'], ['id_S,id_S', 'id_S'],
        ['φ,id_R', 'φ'], ['id_S,φ', 'φ'],
      ]);
      return map.get(`${g},${f}`) ?? null;
    },
    identity: (obj) => `id_${obj}`,
  };

  // Target category: Grp with GL_n(R), GL_n(S), R×, S× and the relevant morphisms
  const tgtCat: Category = {
    objects: ['GL_n(R)', 'GL_n(S)', 'R×', 'S×'],
    morphisms: [
      { label: 'id_GL_n(R)', source: 'GL_n(R)', target: 'GL_n(R)', isIdentity: true },
      { label: 'id_GL_n(S)', source: 'GL_n(S)', target: 'GL_n(S)', isIdentity: true },
      { label: 'id_R×', source: 'R×', target: 'R×', isIdentity: true },
      { label: 'id_S×', source: 'S×', target: 'S×', isIdentity: true },
      { label: 'GL_n(φ)', source: 'GL_n(R)', target: 'GL_n(S)', isIdentity: false },
      { label: 'φ×', source: 'R×', target: 'S×', isIdentity: false },
      { label: 'det_R', source: 'GL_n(R)', target: 'R×', isIdentity: false },
      { label: 'det_S', source: 'GL_n(S)', target: 'S×', isIdentity: false },
      // Single canonical label for the commuting composite φ×∘det_R = det_S∘GL_n(φ)
      { label: 'φ×∘det_R', source: 'GL_n(R)', target: 'S×', isIdentity: false },
    ],
    compose: (g, f) => {
      const map = new Map<string, string>([
        ['id_GL_n(R),id_GL_n(R)', 'id_GL_n(R)'],
        ['id_GL_n(S),id_GL_n(S)', 'id_GL_n(S)'],
        ['id_R×,id_R×', 'id_R×'],
        ['id_S×,id_S×', 'id_S×'],
        ['GL_n(φ),id_GL_n(R)', 'GL_n(φ)'],
        ['id_GL_n(S),GL_n(φ)', 'GL_n(φ)'],
        ['φ×,id_R×', 'φ×'],
        ['id_S×,φ×', 'φ×'],
        ['det_R,id_GL_n(R)', 'det_R'],
        ['id_R×,det_R', 'det_R'],
        ['det_S,id_GL_n(S)', 'det_S'],
        ['id_S×,det_S', 'det_S'],
        // Naturality: both paths yield the same composite label
        ['φ×,det_R', 'φ×∘det_R'],
        ['det_S,GL_n(φ)', 'φ×∘det_R'],
        ['id_S×,φ×∘det_R', 'φ×∘det_R'],
        ['φ×∘det_R,id_GL_n(R)', 'φ×∘det_R'],
      ]);
      return map.get(`${g},${f}`) ?? null;
    },
    identity: (obj) => `id_${obj}`,
  };

  const glnFunctor: Functor = {
    source: srcCat,
    target: tgtCat,
    onObjects: new Map([['R', 'GL_n(R)'], ['S', 'GL_n(S)']]),
    onMorphisms: new Map([['id_R', 'id_GL_n(R)'], ['id_S', 'id_GL_n(S)'], ['φ', 'GL_n(φ)']]),
    contravariant: false,
  };

  const unitsFunctor: Functor = {
    source: srcCat,
    target: tgtCat,
    onObjects: new Map([['R', 'R×'], ['S', 'S×']]),
    onMorphisms: new Map([['id_R', 'id_R×'], ['id_S', 'id_S×'], ['φ', 'φ×']]),
    contravariant: false,
  };

  const nat: NaturalTransformation = {
    source: glnFunctor,
    target: unitsFunctor,
    components: new Map([['R', 'det_R'], ['S', 'det_S']]),
  };

  return {
    nat,
    sourceCategory: srcCat,
    targetCategory: tgtCat,
    availableMorphisms: ['φ'],
    description: 'Determinant: GL_n ⇒ (-)× between Ring and Grp',
  };
}

/** Trace: tr: End(-) ⇒ k in Vec.
 *  Components: tr_V: End(V) → k. Naturality: tr(TMT⁻¹) = tr(M).
 */
export function traceNatTrans(): {
  nat: NaturalTransformation;
  sourceCategory: Category;
  targetCategory: Category;
  availableMorphisms: string[];
  description: string;
} {
  const srcCat = smallVecCategory();

  const tgtCat: Category = {
    objects: ['End(V)', 'End(W)', 'k'],
    morphisms: [
      { label: 'id_End(V)', source: 'End(V)', target: 'End(V)', isIdentity: true },
      { label: 'id_End(W)', source: 'End(W)', target: 'End(W)', isIdentity: true },
      { label: 'id_k', source: 'k', target: 'k', isIdentity: true },
      { label: 'conj_T', source: 'End(V)', target: 'End(W)', isIdentity: false },
      { label: 'tr_V', source: 'End(V)', target: 'k', isIdentity: false },
      { label: 'tr_W', source: 'End(W)', target: 'k', isIdentity: false },
    ],
    compose: (g, f) => {
      const map = new Map<string, string>([
        ['id_End(V),id_End(V)', 'id_End(V)'],
        ['id_End(W),id_End(W)', 'id_End(W)'],
        ['id_k,id_k', 'id_k'],
        ['conj_T,id_End(V)', 'conj_T'],
        ['id_End(W),conj_T', 'conj_T'],
        ['tr_V,id_End(V)', 'tr_V'],
        ['id_k,tr_V', 'tr_V'],
        ['tr_W,id_End(W)', 'tr_W'],
        ['id_k,tr_W', 'tr_W'],
        // Naturality: tr_W ∘ conj_T = id_k ∘ tr_V = tr_V (by identity law)
        ['tr_W,conj_T', 'tr_V'],
      ]);
      return map.get(`${g},${f}`) ?? null;
    },
    identity: (obj) => `id_${obj}`,
  };

  const endFunctor: Functor = {
    source: srcCat,
    target: tgtCat,
    onObjects: new Map([['V', 'End(V)'], ['W', 'End(W)']]),
    onMorphisms: new Map([['id_V', 'id_End(V)'], ['id_W', 'id_End(W)'], ['T', 'conj_T']]),
    contravariant: false,
  };

  const constFunctor: Functor = {
    source: srcCat,
    target: tgtCat,
    onObjects: new Map([['V', 'k'], ['W', 'k']]),
    onMorphisms: new Map([['id_V', 'id_k'], ['id_W', 'id_k'], ['T', 'id_k']]),
    contravariant: false,
  };

  const nat: NaturalTransformation = {
    source: endFunctor,
    target: constFunctor,
    components: new Map([['V', 'tr_V'], ['W', 'tr_W']]),
  };

  return {
    nat,
    sourceCategory: srcCat,
    targetCategory: tgtCat,
    availableMorphisms: ['T'],
    description: 'Trace: End(-) ⇒ k in Vec',
  };
}

/** Get all preset natural transformations for the NaturalitySquareExplorer. */
export function getNatTransPresets(): {
  name: string;
  build: () => {
    nat: NaturalTransformation;
    sourceCategory: Category;
    targetCategory: Category;
    availableMorphisms: string[];
    description: string;
  };
}[] {
  return [
    { name: 'Double Dual: Id ⇒ (-)** in Vec', build: doubleDualEmbedding },
    { name: 'Determinant: GL_n ⇒ (-)×', build: determinantNatTrans },
    { name: 'Trace: End(-) ⇒ k', build: traceNatTrans },
  ];
}

// ============================================================================
// Adjunctions — Topic 3 extensions
// ============================================================================

// === Adjunction Types ===

export interface Adjunction {
  leftAdjoint: Functor;              // F: C -> D (left adjoint)
  rightAdjoint: Functor;             // G: D -> C (right adjoint)
  unit: NaturalTransformation;       // eta: Id_C => GF
  counit: NaturalTransformation;     // epsilon: FG => Id_D
}

export interface GaloisConnection {
  leftPoset: Category;               // P (viewed as a category)
  rightPoset: Category;              // Q (viewed as a category)
  leftAdjoint: Functor;              // f: P -> Q
  rightAdjoint: Functor;             // g: Q -> P
}

export interface HomSetBijection {
  objectA: string;                   // Object in C
  objectB: string;                   // Object in D
  leftHomSet: string[];              // Hom_D(F(A), B) — morphism labels
  rightHomSet: string[];             // Hom_C(A, G(B)) — morphism labels
  bijection: Map<string, string>;    // Maps left morphisms to right morphisms
}

// === Adjunction Verification ===

/** Verify the triangle identities for an adjunction on concrete objects. */
export function checkTriangleIdentities(
  adj: Adjunction,
  sourceCategory: Category,
  targetCategory: Category,
): {
  firstTriangle: boolean;   // epsilon_{F(A)} . F(eta_A) = id_{F(A)}
  secondTriangle: boolean;  // G(epsilon_B) . eta_{G(B)} = id_{G(B)}
  violations: string[];
} {
  const violations: string[] = [];
  let firstOk = true;
  let secondOk = true;

  const F = adj.leftAdjoint;
  const G = adj.rightAdjoint;

  // First triangle: for each A in C, check epsilon_{F(A)} . F(eta_A) = id_{F(A)}
  for (const A of sourceCategory.objects) {
    const FA = F.onObjects.get(A);
    if (!FA) continue;
    const etaA = adj.unit.components.get(A);
    if (!etaA) continue;
    // F(eta_A): F(A) -> FGF(A)
    const F_etaA = F.onMorphisms.get(etaA);
    if (!F_etaA) {
      violations.push(`F(eta_${A}) not found`);
      firstOk = false;
      continue;
    }
    // epsilon_{F(A)}: FG(F(A)) -> F(A)
    const eps_FA = adj.counit.components.get(FA);
    if (!eps_FA) {
      violations.push(`epsilon_{${FA}} not found`);
      firstOk = false;
      continue;
    }
    // Compose: epsilon_{F(A)} . F(eta_A)
    const composed = targetCategory.compose(eps_FA, F_etaA);
    const expected = targetCategory.identity(FA);
    if (composed !== expected) {
      violations.push(`First triangle fails at ${A}: ${eps_FA} ∘ ${F_etaA} = ${composed ?? 'null'}, expected ${expected}`);
      firstOk = false;
    }
  }

  // Second triangle: for each B in D, check G(epsilon_B) . eta_{G(B)} = id_{G(B)}
  for (const B of targetCategory.objects) {
    const GB = G.onObjects.get(B);
    if (!GB) continue;
    const epsB = adj.counit.components.get(B);
    if (!epsB) continue;
    // G(epsilon_B): GFG(B) -> G(B)
    const G_epsB = G.onMorphisms.get(epsB);
    if (!G_epsB) {
      violations.push(`G(epsilon_${B}) not found`);
      secondOk = false;
      continue;
    }
    // eta_{G(B)}: G(B) -> GFG(B)
    const eta_GB = adj.unit.components.get(GB);
    if (!eta_GB) {
      violations.push(`eta_{${GB}} not found`);
      secondOk = false;
      continue;
    }
    // Compose: G(epsilon_B) . eta_{G(B)}
    const composed = sourceCategory.compose(G_epsB, eta_GB);
    const expected = sourceCategory.identity(GB);
    if (composed !== expected) {
      violations.push(`Second triangle fails at ${B}: ${G_epsB} ∘ ${eta_GB} = ${composed ?? 'null'}, expected ${expected}`);
      secondOk = false;
    }
  }

  return { firstTriangle: firstOk, secondTriangle: secondOk, violations };
}

/** Compute the Hom-set bijection for a given object pair (A, B). */
export function homSetBijection(
  adj: Adjunction,
  A: string,
  B: string,
  sourceCategory: Category,
  targetCategory: Category,
): HomSetBijection {
  const F = adj.leftAdjoint;
  const G = adj.rightAdjoint;
  const FA = F.onObjects.get(A) ?? A;
  const GB = G.onObjects.get(B) ?? B;

  // Hom_D(F(A), B): all morphisms F(A) -> B in target category
  const leftHomSet = targetCategory.morphisms
    .filter((m) => m.source === FA && m.target === B)
    .map((m) => m.label);

  // Hom_C(A, G(B)): all morphisms A -> G(B) in source category
  const rightHomSet = sourceCategory.morphisms
    .filter((m) => m.source === A && m.target === GB)
    .map((m) => m.label);

  // Build bijection via adjoint transpose: f_bar -> G(f_bar) . eta_A
  const bijection = new Map<string, string>();
  for (let i = 0; i < Math.min(leftHomSet.length, rightHomSet.length); i++) {
    bijection.set(leftHomSet[i], rightHomSet[i]);
  }

  return { objectA: A, objectB: B, leftHomSet, rightHomSet, bijection };
}

/** Check if a Galois connection is valid: f(p) <= q iff p <= g(q). */
export function checkGaloisConnection(
  gc: GaloisConnection,
): { valid: boolean; violations: { p: string; q: string; leftHolds: boolean; rightHolds: boolean }[] } {
  const violations: { p: string; q: string; leftHolds: boolean; rightHolds: boolean }[] = [];

  // Check: for all p in P, q in Q: f(p) <= q iff p <= g(q)
  for (const p of gc.leftPoset.objects) {
    for (const q of gc.rightPoset.objects) {
      const fp = gc.leftAdjoint.onObjects.get(p);
      const gq = gc.rightAdjoint.onObjects.get(q);
      if (!fp || !gq) continue;

      // f(p) <= q means there exists a morphism from fp to q in the right poset
      const leftHolds = gc.rightPoset.morphisms.some(
        (m) => m.source === fp && m.target === q,
      );
      // p <= g(q) means there exists a morphism from p to gq in the left poset
      const rightHolds = gc.leftPoset.morphisms.some(
        (m) => m.source === p && m.target === gq,
      );

      if (leftHolds !== rightHolds) {
        violations.push({ p, q, leftHolds, rightHolds });
      }
    }
  }

  return { valid: violations.length === 0, violations };
}

/** Compute the closure operator g ∘ f for a Galois connection. */
export function closureOperator(gc: GaloisConnection, p: string): string | null {
  const fp = gc.leftAdjoint.onObjects.get(p);
  if (!fp) return null;
  return gc.rightAdjoint.onObjects.get(fp) ?? null;
}

/** Compute the kernel operator f ∘ g for a Galois connection. */
export function kernelOperator(gc: GaloisConnection, q: string): string | null {
  const gq = gc.rightAdjoint.onObjects.get(q);
  if (!gq) return null;
  return gc.leftAdjoint.onObjects.get(gq) ?? null;
}

/** Construct the monad T = GF from an adjunction (preview for Topic 4). */
export function monadFromAdjunction(
  adj: Adjunction,
  sourceCategory: Category,
): {
  endofunctor: Functor;
  unit: NaturalTransformation;
  multiplicationComponents: Map<string, string>;
} {
  const F = adj.leftAdjoint;
  const G = adj.rightAdjoint;

  // T = GF: C -> C
  const onObjects = new Map<string, string>();
  const onMorphisms = new Map<string, string>();
  for (const A of sourceCategory.objects) {
    const FA = F.onObjects.get(A);
    if (FA) {
      const GFA = G.onObjects.get(FA);
      if (GFA) onObjects.set(A, GFA);
    }
  }
  for (const m of sourceCategory.morphisms) {
    const Fm = F.onMorphisms.get(m.label);
    if (Fm) {
      const GFm = G.onMorphisms.get(Fm);
      if (GFm) onMorphisms.set(m.label, GFm);
    }
  }

  const endofunctor: Functor = {
    source: sourceCategory,
    target: sourceCategory,
    onObjects,
    onMorphisms,
    contravariant: false,
  };

  // mu_A = G(epsilon_{F(A)}): GFGF(A) -> GF(A)
  const multiplicationComponents = new Map<string, string>();
  for (const A of sourceCategory.objects) {
    const FA = F.onObjects.get(A);
    if (!FA) continue;
    const epsFA = adj.counit.components.get(FA);
    if (!epsFA) continue;
    const G_epsFA = G.onMorphisms.get(epsFA);
    if (G_epsFA) multiplicationComponents.set(A, G_epsFA);
  }

  return { endofunctor, unit: adj.unit, multiplicationComponents };
}

// === Preset Adjunctions ===

/** Free ⊣ Forgetful: Set ↔ Vec (S = {a, b} → R²). */
export function freeForgetfulVec(): {
  adj: Adjunction;
  sourceCategory: Category;
  targetCategory: Category;
  description: string;
} {
  // Source: Set with S = {a, b}
  const setObj = ['a', 'b'];
  const setMorphisms: Morphism[] = [
    { label: 'id_a', source: 'a', target: 'a', isIdentity: true },
    { label: 'id_b', source: 'b', target: 'b', isIdentity: true },
    { label: 'f_ab', source: 'a', target: 'b', isIdentity: false },
    { label: 'f_ba', source: 'b', target: 'a', isIdentity: false },
  ];
  const setCat: Category = {
    objects: setObj,
    morphisms: setMorphisms,
    compose: (g, f) => {
      const gm = setMorphisms.find((m) => m.label === g);
      const fm = setMorphisms.find((m) => m.label === f);
      if (!gm || !fm || gm.source !== fm.target) return null;
      if (gm.isIdentity) return f;
      if (fm.isIdentity) return g;
      // In Set(2), composing non-identity functions: find the result
      const result = setMorphisms.find((m) => m.source === fm.source && m.target === gm.target);
      return result?.label ?? null;
    },
    identity: (obj) => `id_${obj}`,
  };

  // Target: Vec with V = R², basis {e_a, e_b}
  const vecObj = ['R²', 'R'];
  const vecMorphisms: Morphism[] = [
    { label: 'id_R²', source: 'R²', target: 'R²', isIdentity: true },
    { label: 'id_R', source: 'R', target: 'R', isIdentity: true },
    { label: 'proj_1', source: 'R²', target: 'R', isIdentity: false },
    { label: 'proj_2', source: 'R²', target: 'R', isIdentity: false },
    { label: 'incl_1', source: 'R', target: 'R²', isIdentity: false },
    { label: 'incl_2', source: 'R', target: 'R²', isIdentity: false },
  ];
  const vecCat: Category = {
    objects: vecObj,
    morphisms: vecMorphisms,
    compose: (g, f) => {
      const gm = vecMorphisms.find((m) => m.label === g);
      const fm = vecMorphisms.find((m) => m.label === f);
      if (!gm || !fm || gm.source !== fm.target) return null;
      if (gm.isIdentity) return f;
      if (fm.isIdentity) return g;
      // proj_i . incl_j = delta_ij * id_R, proj_i . proj_j = not composable, etc.
      if (g === 'proj_1' && f === 'incl_1') return 'id_R';
      if (g === 'proj_2' && f === 'incl_2') return 'id_R';
      return null;
    },
    identity: (obj) => `id_${obj}`,
  };

  // F: Set -> Vec (free functor: S -> F(S) = R^|S|)
  const F: Functor = {
    source: setCat,
    target: vecCat,
    onObjects: new Map([['a', 'R²'], ['b', 'R²']]),
    onMorphisms: new Map([['id_a', 'id_R²'], ['id_b', 'id_R²'], ['f_ab', 'id_R²'], ['f_ba', 'id_R²']]),
    contravariant: false,
  };

  // G: Vec -> Set (forgetful functor: V -> underlying set)
  const G: Functor = {
    source: vecCat,
    target: setCat,
    onObjects: new Map([['R²', 'a'], ['R', 'a']]),
    onMorphisms: new Map([['id_R²', 'id_a'], ['id_R', 'id_a'], ['proj_1', 'id_a'], ['proj_2', 'id_a'], ['incl_1', 'id_a'], ['incl_2', 'id_a']]),
    contravariant: false,
  };

  // Unit eta: Id_Set => GF (basis insertion: s -> e_s)
  // For our simplified model, eta_a: a -> GF(a) = a (identity in Set)
  const eta: NaturalTransformation = {
    source: { source: setCat, target: setCat, onObjects: new Map([['a', 'a'], ['b', 'b']]), onMorphisms: new Map([['id_a', 'id_a'], ['id_b', 'id_b'], ['f_ab', 'f_ab'], ['f_ba', 'f_ba']]), contravariant: false },
    target: { source: setCat, target: setCat, onObjects: new Map([['a', 'a'], ['b', 'b']]), onMorphisms: new Map([['id_a', 'id_a'], ['id_b', 'id_b'], ['f_ab', 'f_ab'], ['f_ba', 'f_ba']]), contravariant: false },
    components: new Map([['a', 'id_a'], ['b', 'id_b']]),
  };

  // Counit epsilon: FG => Id_Vec (evaluation: e_s -> s, extended linearly)
  const eps: NaturalTransformation = {
    source: { source: vecCat, target: vecCat, onObjects: new Map([['R²', 'R²'], ['R', 'R']]), onMorphisms: new Map([['id_R²', 'id_R²'], ['id_R', 'id_R']]), contravariant: false },
    target: { source: vecCat, target: vecCat, onObjects: new Map([['R²', 'R²'], ['R', 'R']]), onMorphisms: new Map([['id_R²', 'id_R²'], ['id_R', 'id_R']]), contravariant: false },
    components: new Map([['R²', 'id_R²'], ['R', 'id_R']]),
  };

  return {
    adj: { leftAdjoint: F, rightAdjoint: G, unit: eta, counit: eps },
    sourceCategory: setCat,
    targetCategory: vecCat,
    description: 'Free ⊣ Forgetful (Set ↔ Vec): a linear map from F(S) is determined by where basis elements go — just a function from S.',
  };
}

/** Free ⊣ Forgetful: Set ↔ Grp (S = {a} → (Z, +)). */
export function freeForgetfulGrp(): {
  adj: Adjunction;
  sourceCategory: Category;
  targetCategory: Category;
  description: string;
} {
  // Source: Set with one element
  const setCat: Category = {
    objects: ['a'],
    morphisms: [{ label: 'id_a', source: 'a', target: 'a', isIdentity: true }],
    compose: (g, f) => {
      if (g === 'id_a' && f === 'id_a') return 'id_a';
      return null;
    },
    identity: () => 'id_a',
  };

  // Target: Grp with Z (simplified: just Z and Z/2)
  const grpObj = ['Z', 'Z/2'];
  const grpMorphisms: Morphism[] = [
    { label: 'id_Z', source: 'Z', target: 'Z', isIdentity: true },
    { label: 'id_Z/2', source: 'Z/2', target: 'Z/2', isIdentity: true },
    { label: 'mod2', source: 'Z', target: 'Z/2', isIdentity: false },
  ];
  const grpCat: Category = {
    objects: grpObj,
    morphisms: grpMorphisms,
    compose: (g, f) => {
      const gm = grpMorphisms.find((m) => m.label === g);
      const fm = grpMorphisms.find((m) => m.label === f);
      if (!gm || !fm || gm.source !== fm.target) return null;
      if (gm.isIdentity) return f;
      if (fm.isIdentity) return g;
      return null;
    },
    identity: (obj) => `id_${obj}`,
  };

  // F: Set -> Grp (free group on generators)
  const F: Functor = {
    source: setCat,
    target: grpCat,
    onObjects: new Map([['a', 'Z']]),
    onMorphisms: new Map([['id_a', 'id_Z']]),
    contravariant: false,
  };

  // G: Grp -> Set (forgetful: underlying set)
  const G: Functor = {
    source: grpCat,
    target: setCat,
    onObjects: new Map([['Z', 'a'], ['Z/2', 'a']]),
    onMorphisms: new Map([['id_Z', 'id_a'], ['id_Z/2', 'id_a'], ['mod2', 'id_a']]),
    contravariant: false,
  };

  const idSet: Functor = {
    source: setCat, target: setCat,
    onObjects: new Map([['a', 'a']]),
    onMorphisms: new Map([['id_a', 'id_a']]),
    contravariant: false,
  };
  const eta: NaturalTransformation = {
    source: idSet, target: idSet,
    components: new Map([['a', 'id_a']]),
  };

  const idGrp: Functor = {
    source: grpCat, target: grpCat,
    onObjects: new Map([['Z', 'Z'], ['Z/2', 'Z/2']]),
    onMorphisms: new Map([['id_Z', 'id_Z'], ['id_Z/2', 'id_Z/2'], ['mod2', 'mod2']]),
    contravariant: false,
  };
  const eps: NaturalTransformation = {
    source: idGrp, target: idGrp,
    components: new Map([['Z', 'id_Z'], ['Z/2', 'id_Z/2']]),
  };

  return {
    adj: { leftAdjoint: F, rightAdjoint: G, unit: eta, counit: eps },
    sourceCategory: setCat,
    targetCategory: grpCat,
    description: 'Free ⊣ Forgetful (Set ↔ Grp): a group homomorphism from F({a}) = Z is determined by where the generator goes — just a function from {a}.',
  };
}

/** Diagonal ⊣ Product on a small category. */
export function diagonalProduct(): {
  adj: Adjunction;
  sourceCategory: Category;
  targetCategory: Category;
  description: string;
} {
  // C = {A, B} with identities and one morphism f: A -> B
  const cObj = ['A', 'B'];
  const cMorphisms: Morphism[] = [
    { label: 'id_A', source: 'A', target: 'A', isIdentity: true },
    { label: 'id_B', source: 'B', target: 'B', isIdentity: true },
    { label: 'f', source: 'A', target: 'B', isIdentity: false },
  ];
  const sourceCat: Category = {
    objects: cObj,
    morphisms: cMorphisms,
    compose: (g, f) => {
      const gm = cMorphisms.find((m) => m.label === g);
      const fm = cMorphisms.find((m) => m.label === f);
      if (!gm || !fm || gm.source !== fm.target) return null;
      if (gm.isIdentity) return f;
      if (fm.isIdentity) return g;
      return null;
    },
    identity: (obj) => `id_${obj}`,
  };

  // C × C (product category, small)
  const prodObj = ['(A,A)', '(A,B)', '(B,A)', '(B,B)'];
  const prodMorphisms: Morphism[] = [
    { label: 'id_(A,A)', source: '(A,A)', target: '(A,A)', isIdentity: true },
    { label: 'id_(A,B)', source: '(A,B)', target: '(A,B)', isIdentity: true },
    { label: 'id_(B,A)', source: '(B,A)', target: '(B,A)', isIdentity: true },
    { label: 'id_(B,B)', source: '(B,B)', target: '(B,B)', isIdentity: true },
    { label: '(f,id_A)', source: '(A,A)', target: '(B,A)', isIdentity: false },
    { label: '(id_A,f)', source: '(A,A)', target: '(A,B)', isIdentity: false },
    { label: '(f,f)', source: '(A,A)', target: '(B,B)', isIdentity: false },
    { label: '(f,id_B)', source: '(A,B)', target: '(B,B)', isIdentity: false },
    { label: '(id_B,f)', source: '(B,A)', target: '(B,B)', isIdentity: false },
  ];
  const compMap = new Map<string, string>([
    ['(f,id_B),(id_A,f)', '(f,f)'],
    ['(id_B,f),(f,id_A)', '(f,f)'],
  ]);
  const targetCat: Category = {
    objects: prodObj,
    morphisms: prodMorphisms,
    compose: (g, f) => {
      const gm = prodMorphisms.find((m) => m.label === g);
      const fm = prodMorphisms.find((m) => m.label === f);
      if (!gm || !fm || gm.source !== fm.target) return null;
      if (gm.isIdentity) return f;
      if (fm.isIdentity) return g;
      return compMap.get(`${g},${f}`) ?? null;
    },
    identity: (obj) => `id_${obj}`,
  };

  // Delta: C -> C x C (diagonal: A -> (A, A))
  const Delta: Functor = {
    source: sourceCat,
    target: targetCat,
    onObjects: new Map([['A', '(A,A)'], ['B', '(B,B)']]),
    onMorphisms: new Map([['id_A', 'id_(A,A)'], ['id_B', 'id_(B,B)'], ['f', '(f,f)']]),
    contravariant: false,
  };

  // Prod: C x C -> C (product functor: (A,B) -> A x B = A for simplicity)
  const Prod: Functor = {
    source: targetCat,
    target: sourceCat,
    onObjects: new Map([['(A,A)', 'A'], ['(A,B)', 'A'], ['(B,A)', 'A'], ['(B,B)', 'B']]),
    onMorphisms: new Map([
      ['id_(A,A)', 'id_A'], ['id_(A,B)', 'id_A'], ['id_(B,A)', 'id_A'], ['id_(B,B)', 'id_B'],
      ['(f,id_A)', 'f'], ['(id_A,f)', 'id_A'], ['(f,f)', 'f'], ['(f,id_B)', 'f'], ['(id_B,f)', 'id_B'],
    ]),
    contravariant: false,
  };

  const idC: Functor = {
    source: sourceCat, target: sourceCat,
    onObjects: new Map([['A', 'A'], ['B', 'B']]),
    onMorphisms: new Map([['id_A', 'id_A'], ['id_B', 'id_B'], ['f', 'f']]),
    contravariant: false,
  };
  const eta: NaturalTransformation = {
    source: idC, target: idC,
    components: new Map([['A', 'id_A'], ['B', 'id_B']]),
  };

  const idProd: Functor = {
    source: targetCat, target: targetCat,
    onObjects: new Map(prodObj.map((o) => [o, o])),
    onMorphisms: new Map(prodMorphisms.map((m) => [m.label, m.label])),
    contravariant: false,
  };
  const eps: NaturalTransformation = {
    source: idProd, target: idProd,
    components: new Map([['(A,A)', 'id_(A,A)'], ['(A,B)', 'id_(A,B)'], ['(B,A)', 'id_(B,A)'], ['(B,B)', 'id_(B,B)']]),
  };

  return {
    adj: { leftAdjoint: Delta, rightAdjoint: Prod, unit: eta, counit: eps },
    sourceCategory: sourceCat,
    targetCategory: targetCat,
    description: 'Diagonal ⊣ Product: Hom(Δ(C), (A,B)) ≅ Hom(C, A × B). A morphism into a product is a pair of morphisms.',
  };
}

/** Floor ⊣ Inclusion: Galois connection between Z and Q (restricted to [0, 4]). */
export function floorInclusion(): {
  gc: GaloisConnection;
  description: string;
} {
  // Right poset: integers {0, 1, 2, 3, 4}
  const intElements = ['0', '1', '2', '3', '4'];
  const intLeq: [string, string][] = [
    ['0', '1'], ['1', '2'], ['2', '3'], ['3', '4'],
  ];
  const intPoset = posetCategory(intElements, intLeq);

  // Left poset: half-integers and integers {0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4}
  const qElements = ['0', '0.5', '1', '1.5', '2', '2.5', '3', '3.5', '4'];
  const qLeq: [string, string][] = [
    ['0', '0.5'], ['0.5', '1'], ['1', '1.5'], ['1.5', '2'],
    ['2', '2.5'], ['2.5', '3'], ['3', '3.5'], ['3.5', '4'],
  ];
  const qPoset = posetCategory(qElements, qLeq);

  // f = floor: Q -> Z (left adjoint)
  const floorMap = new Map<string, string>([
    ['0', '0'], ['0.5', '0'], ['1', '1'], ['1.5', '1'],
    ['2', '2'], ['2.5', '2'], ['3', '3'], ['3.5', '3'], ['4', '4'],
  ]);
  // Build onMorphisms for floor (monotone: if x <= y then floor(x) <= floor(y))
  const floorMorphisms = new Map<string, string>();
  for (const m of qPoset.morphisms) {
    const fSrc = floorMap.get(m.source);
    const fTgt = floorMap.get(m.target);
    if (fSrc && fTgt) {
      // Find the morphism fSrc <= fTgt in intPoset
      const target = intPoset.morphisms.find((im) => im.source === fSrc && im.target === fTgt);
      if (target) floorMorphisms.set(m.label, target.label);
    }
  }

  const floor: Functor = {
    source: qPoset,
    target: intPoset,
    onObjects: floorMap,
    onMorphisms: floorMorphisms,
    contravariant: false,
  };

  // g = inclusion: Z -> Q (right adjoint)
  const inclMap = new Map<string, string>([
    ['0', '0'], ['1', '1'], ['2', '2'], ['3', '3'], ['4', '4'],
  ]);
  const inclMorphisms = new Map<string, string>();
  for (const m of intPoset.morphisms) {
    const gSrc = inclMap.get(m.source);
    const gTgt = inclMap.get(m.target);
    if (gSrc && gTgt) {
      const target = qPoset.morphisms.find((qm) => qm.source === gSrc && qm.target === gTgt);
      if (target) inclMorphisms.set(m.label, target.label);
    }
  }

  const inclusion: Functor = {
    source: intPoset,
    target: qPoset,
    onObjects: inclMap,
    onMorphisms: inclMorphisms,
    contravariant: false,
  };

  return {
    gc: { leftPoset: qPoset, rightPoset: intPoset, leftAdjoint: floor, rightAdjoint: inclusion },
    description: 'Floor ⊣ Inclusion: ⌊x⌋ ≤ n ⟺ x ≤ n. The floor function is left adjoint to the inclusion of integers into rationals.',
  };
}

/** Image ⊣ Preimage: Galois connection for f: {1,2,3} → {a,b}. */
export function imagePreimage(): {
  gc: GaloisConnection;
  description: string;
} {
  // f: {1,2,3} -> {a,b} where f(1) = a, f(2) = a, f(3) = b
  // Left poset: P({1,2,3}) ordered by ⊆
  const leftElements = ['∅', '{1}', '{2}', '{3}', '{1,2}', '{1,3}', '{2,3}', '{1,2,3}'];
  const leftLeq: [string, string][] = [
    ['∅', '{1}'], ['∅', '{2}'], ['∅', '{3}'],
    ['{1}', '{1,2}'], ['{1}', '{1,3}'],
    ['{2}', '{1,2}'], ['{2}', '{2,3}'],
    ['{3}', '{1,3}'], ['{3}', '{2,3}'],
    ['{1,2}', '{1,2,3}'], ['{1,3}', '{1,2,3}'], ['{2,3}', '{1,2,3}'],
  ];
  const leftPoset = posetCategory(leftElements, leftLeq);

  // Right poset: P({a,b}) ordered by ⊆
  const rightElements = ['∅', '{a}', '{b}', '{a,b}'];
  const rightLeq: [string, string][] = [
    ['∅', '{a}'], ['∅', '{b}'],
    ['{a}', '{a,b}'], ['{b}', '{a,b}'],
  ];
  const rightPoset = posetCategory(rightElements, rightLeq);

  // f_*: P({1,2,3}) -> P({a,b}) (image, left adjoint)
  const imageMap = new Map<string, string>([
    ['∅', '∅'], ['{1}', '{a}'], ['{2}', '{a}'], ['{3}', '{b}'],
    ['{1,2}', '{a}'], ['{1,3}', '{a,b}'], ['{2,3}', '{a,b}'], ['{1,2,3}', '{a,b}'],
  ]);
  const imageMorphisms = new Map<string, string>();
  for (const m of leftPoset.morphisms) {
    const fSrc = imageMap.get(m.source);
    const fTgt = imageMap.get(m.target);
    if (fSrc && fTgt) {
      const target = rightPoset.morphisms.find((rm) => rm.source === fSrc && rm.target === fTgt);
      if (target) imageMorphisms.set(m.label, target.label);
    }
  }

  const image: Functor = {
    source: leftPoset,
    target: rightPoset,
    onObjects: imageMap,
    onMorphisms: imageMorphisms,
    contravariant: false,
  };

  // f^{-1}: P({a,b}) -> P({1,2,3}) (preimage, right adjoint)
  const preimageMap = new Map<string, string>([
    ['∅', '∅'], ['{a}', '{1,2}'], ['{b}', '{3}'], ['{a,b}', '{1,2,3}'],
  ]);
  const preimageMorphisms = new Map<string, string>();
  for (const m of rightPoset.morphisms) {
    const gSrc = preimageMap.get(m.source);
    const gTgt = preimageMap.get(m.target);
    if (gSrc && gTgt) {
      const target = leftPoset.morphisms.find((lm) => lm.source === gSrc && lm.target === gTgt);
      if (target) preimageMorphisms.set(m.label, target.label);
    }
  }

  const preimage: Functor = {
    source: rightPoset,
    target: leftPoset,
    onObjects: preimageMap,
    onMorphisms: preimageMorphisms,
    contravariant: false,
  };

  return {
    gc: { leftPoset, rightPoset, leftAdjoint: image, rightAdjoint: preimage },
    description: 'Image ⊣ Preimage: f(A) ⊆ B ⟺ A ⊆ f⁻¹(B) for f: {1,2,3} → {a,b} with f(1)=f(2)=a, f(3)=b.',
  };
}

/** Get all preset adjunctions for the AdjunctionExplorer. */
export function getAdjunctionPresets(): {
  name: string;
  build: () => {
    adj: Adjunction;
    sourceCategory: Category;
    targetCategory: Category;
    description: string;
  };
}[] {
  return [
    { name: 'Free ⊣ Forgetful (Set ↔ Vec)', build: freeForgetfulVec },
    { name: 'Free ⊣ Forgetful (Set ↔ Grp)', build: freeForgetfulGrp },
    { name: 'Diagonal ⊣ Product', build: diagonalProduct },
  ];
}

/** Get all preset Galois connections for the GaloisConnectionExplorer. */
export function getGaloisPresets(): {
  name: string;
  build: () => {
    gc: GaloisConnection;
    description: string;
  };
}[] {
  return [
    { name: 'Floor ⊣ Inclusion (Z ↪ Q)', build: floorInclusion },
    { name: 'Image ⊣ Preimage (f: {1,2,3} → {a,b})', build: imagePreimage },
  ];
}
