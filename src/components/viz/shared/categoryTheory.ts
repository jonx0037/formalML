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
