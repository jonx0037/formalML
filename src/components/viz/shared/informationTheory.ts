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
