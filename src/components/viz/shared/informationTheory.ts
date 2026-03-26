import type { HuffmanNode } from './types';

/**
 * Safe log2: returns 0 for p = 0 (convention 0 log 0 = 0).
 */
function safeLog2(p: number): number {
  return p > 0 ? Math.log2(p) : 0;
}

/** Shannon entropy in bits. Convention: 0 log 0 = 0. */
export function entropy(probs: number[]): number {
  return -probs.reduce((sum, p) => sum + p * safeLog2(p), 0);
}

/** Binary entropy function H_b(p) in bits. */
export function binaryEntropy(p: number): number {
  if (p <= 0 || p >= 1) return 0;
  return -p * Math.log2(p) - (1 - p) * Math.log2(1 - p);
}

/** Joint entropy H(X,Y) from a 2D probability table. */
export function jointEntropy(joint: number[][]): number {
  let sum = 0;
  for (const row of joint) {
    for (const p of row) {
      sum += p * safeLog2(p);
    }
  }
  return -sum;
}

/** Marginal distribution from a joint distribution (sum over rows or columns). */
export function marginal(joint: number[][], axis: 'x' | 'y'): number[] {
  const rows = joint.length;
  const cols = joint[0]?.length ?? 0;

  if (axis === 'y') {
    // Sum over columns → marginal p(x_i)
    return joint.map((row) => row.reduce((s, v) => s + v, 0));
  }
  // Sum over rows → marginal p(y_j)
  const result = new Array(cols).fill(0);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      result[j] += joint[i][j];
    }
  }
  return result;
}

/** Conditional entropy H(Y|X) from a joint distribution. */
export function conditionalEntropy(joint: number[][]): number {
  // H(Y|X) = H(X,Y) - H(X)
  const pX = marginal(joint, 'y');
  return jointEntropy(joint) - entropy(pX);
}

/** Mutual information I(X;Y) from a joint distribution. */
export function mutualInformation(joint: number[][]): number {
  const pX = marginal(joint, 'y');
  const pY = marginal(joint, 'x');
  return entropy(pX) + entropy(pY) - jointEntropy(joint);
}

/** Differential entropy of a Gaussian N(mu, sigma^2) in bits. */
export function gaussianDifferentialEntropy(sigma: number): number {
  // h(X) = 0.5 * log2(2 * pi * e * sigma^2)
  return 0.5 * Math.log2(2 * Math.PI * Math.E * sigma * sigma);
}

/** Build a Huffman tree from symbols and probabilities.
 *  Uses repeated sorting (O(k² log k)) rather than a heap — acceptable
 *  for the small k values (≤8) used in the visualizations. */
export function buildHuffmanTree(
  symbols: string[],
  probs: number[]
): HuffmanNode {
  // Create leaf nodes
  let nodes: HuffmanNode[] = symbols.map((s, i) => ({
    symbol: s,
    probability: probs[i],
  }));

  // Iteratively merge the two lowest-probability nodes
  while (nodes.length > 1) {
    nodes.sort((a, b) => a.probability - b.probability);
    const left = nodes.shift()!;
    const right = nodes.shift()!;
    nodes.push({
      probability: left.probability + right.probability,
      left,
      right,
    });
  }

  return nodes[0];
}

/** Extract code assignments from a Huffman tree. */
export function huffmanCodes(tree: HuffmanNode): Map<string, string> {
  const codes = new Map<string, string>();

  function traverse(node: HuffmanNode, prefix: string) {
    if (node.symbol !== undefined) {
      codes.set(node.symbol, prefix || '0'); // single-symbol edge case
      return;
    }
    if (node.left) traverse(node.left, prefix + '0');
    if (node.right) traverse(node.right, prefix + '1');
  }

  traverse(tree, '');
  return codes;
}

/** Expected code length from a Huffman tree. */
export function expectedCodeLength(
  symbols: string[],
  probs: number[],
  codes: Map<string, string>
): number {
  return symbols.reduce((sum, s, i) => {
    const len = codes.get(s)?.length ?? 0;
    return sum + probs[i] * len;
  }, 0);
}

/** Normalize a probability array to sum to 1. */
export function normalize(probs: number[]): number[] {
  const sum = probs.reduce((s, p) => s + p, 0);
  if (sum === 0) return probs.map(() => 1 / probs.length);
  return probs.map((p) => p / sum);
}

// ---------------------------------------------------------------------------
// Divergence functions (added for KL Divergence & f-Divergences topic)
// ---------------------------------------------------------------------------

/** Cross-entropy H(p, q) = -sum p(x) log2 q(x). Returns Infinity if q(x) = 0 where p(x) > 0. */
export function crossEntropy(p: number[], q: number[]): number {
  let sum = 0;
  for (let i = 0; i < p.length; i++) {
    if (p[i] > 0 && q[i] <= 0) return Infinity;
    if (p[i] > 0) sum += p[i] * Math.log2(q[i]);
  }
  return -sum;
}

/** KL divergence D_KL(p || q) in bits. Returns Infinity if q(x) = 0 where p(x) > 0. */
export function klDivergence(p: number[], q: number[]): number {
  let sum = 0;
  for (let i = 0; i < p.length; i++) {
    if (p[i] > 0 && q[i] <= 0) return Infinity;
    if (p[i] > 0) sum += p[i] * Math.log2(p[i] / q[i]);
  }
  return sum;
}

/** General f-divergence D_f(p || q) = sum q(x) f(p(x)/q(x)). */
export function fDivergence(
  p: number[],
  q: number[],
  f: (t: number) => number
): number {
  let sum = 0;
  for (let i = 0; i < p.length; i++) {
    if (q[i] <= 0) {
      if (p[i] > 0) return Infinity;
      continue;
    }
    sum += q[i] * f(p[i] / q[i]);
  }
  return sum;
}

/** Jensen-Shannon divergence JS(p || q) = (D_KL(p||m) + D_KL(q||m))/2 where m = (p+q)/2. In bits. */
export function jensenShannonDivergence(p: number[], q: number[]): number {
  const m = p.map((pi, i) => (pi + q[i]) / 2);
  return (klDivergence(p, m) + klDivergence(q, m)) / 2;
}

/** Total variation distance TV(p, q) = (1/2) sum |p(x) - q(x)|. */
export function totalVariation(p: number[], q: number[]): number {
  let sum = 0;
  for (let i = 0; i < p.length; i++) {
    sum += Math.abs(p[i] - q[i]);
  }
  return sum / 2;
}

/** Chi-squared divergence sum (p(x) - q(x))^2 / q(x). */
export function chiSquaredDivergence(p: number[], q: number[]): number {
  let sum = 0;
  for (let i = 0; i < p.length; i++) {
    if (q[i] <= 0) {
      if (p[i] > 0) return Infinity;
      continue;
    }
    const diff = p[i] - q[i];
    sum += (diff * diff) / q[i];
  }
  return sum;
}

/** Squared Hellinger distance sum (sqrt(p(x)) - sqrt(q(x)))^2. */
export function hellingerDistance(p: number[], q: number[]): number {
  let sum = 0;
  for (let i = 0; i < p.length; i++) {
    const diff = Math.sqrt(p[i]) - Math.sqrt(q[i]);
    sum += diff * diff;
  }
  return sum;
}

/** Rényi divergence of order α in nats. α → 1 gives KL (in nats). */
export function renyiDivergence(
  p: number[],
  q: number[],
  alpha: number
): number {
  if (Math.abs(alpha - 1) < 1e-10) {
    // Limit α → 1: KL divergence in nats
    let sum = 0;
    for (let i = 0; i < p.length; i++) {
      if (p[i] > 0 && q[i] <= 0) return Infinity;
      if (p[i] > 0) sum += p[i] * Math.log(p[i] / q[i]);
    }
    return sum;
  }
  let sum = 0;
  for (let i = 0; i < p.length; i++) {
    if (p[i] > 0 && q[i] <= 0) return Infinity;
    if (p[i] > 0) sum += Math.pow(p[i], alpha) * Math.pow(q[i], 1 - alpha);
  }
  if (sum <= 0) return Infinity;
  return Math.log(sum) / (alpha - 1);
}

/** Gaussian PDF evaluated at each point in x. */
export function gaussianPdf(
  x: number[],
  mu: number,
  sigma: number
): number[] {
  const coeff = 1 / (sigma * Math.sqrt(2 * Math.PI));
  const denom = 2 * sigma * sigma;
  return x.map((xi) => coeff * Math.exp(-((xi - mu) ** 2) / denom));
}

/** Gaussian mixture PDF evaluated at each point in x. */
export function gmmPdf(
  x: number[],
  mus: number[],
  sigmas: number[],
  weights: number[]
): number[] {
  const k = mus.length;
  return x.map((xi) => {
    let density = 0;
    for (let j = 0; j < k; j++) {
      const coeff = 1 / (sigmas[j] * Math.sqrt(2 * Math.PI));
      density +=
        weights[j] * coeff * Math.exp(-((xi - mus[j]) ** 2) / (2 * sigmas[j] * sigmas[j]));
    }
    return density;
  });
}
