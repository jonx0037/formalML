/**
 * Typewriter sequence intervals for the ConvergenceModesDemo.
 *
 * The n-th term is the indicator of [k/m, (k+1)/m] where n maps to (m, k):
 *   n=1 → [0,1], n=2 → [0,1/2], n=3 → [1/2,1], n=4 → [0,1/3], ...
 * This sequence converges in probability to 0 but NOT almost surely.
 */
export interface TypewriterInterval {
  start: number;
  end: number;
}

let cache: TypewriterInterval[] | null = null;

function computeTypewriterSequence(nTerms: number): TypewriterInterval[] {
  const intervals: TypewriterInterval[] = [];
  let idx = 0;
  let m = 1;
  while (idx < nTerms) {
    for (let k = 0; k < m && idx < nTerms; k++) {
      intervals.push({ start: k / m, end: (k + 1) / m });
      idx++;
    }
    m++;
  }
  return intervals;
}

export function getTypewriterSequence(nTerms: number = 200): TypewriterInterval[] {
  if (cache === null || cache.length < nTerms) {
    cache = computeTypewriterSequence(nTerms);
  }
  return cache.slice(0, nTerms);
}
