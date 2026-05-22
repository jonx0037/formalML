import { useState, useMemo } from 'react';
import { paletteNB, multinomialNBFit, multinomialNBPredictProba } from './shared/naive-bayes';

// =============================================================================
// MultinomialNBTextExplorer — §5.2 companion. A toy bag-of-words classifier
// with two classes ("baseball", "medicine") and ~10 tokens. The reader edits
// the per-class token counts (left grid), the query document's token counts
// (right vector), and a Laplace α; the panel reports the model's class
// log-posteriors and the top-contributing tokens for each class.
//
// The intent is the demystify Multinomial NB at the count-table level — the
// "training" is just normalizing the count matrix per row, and prediction is
// dot-product + log-prior.
// =============================================================================

const TOKENS = ['pitcher', 'inning', 'hit', 'team', 'season', 'patient', 'doctor', 'drug', 'health', 'clinic'];
const CLASS_LABELS = ['baseball', 'medicine'];

// Initial counts: token counts per class. Each row is one class.
const INITIAL_COUNTS: number[][] = [
  // baseball
  [42, 28, 31, 35, 22, 1, 2, 0, 3, 1],
  // medicine
  [0, 1, 0, 4, 6, 38, 27, 32, 25, 18],
];

const INITIAL_DOC = [3, 2, 4, 1, 0, 0, 0, 0, 1, 0];

export default function MultinomialNBTextExplorer() {
  const [counts, setCounts] = useState<number[][]>(INITIAL_COUNTS.map((row) => row.slice()));
  const [doc, setDoc] = useState<number[]>(INITIAL_DOC.slice());
  const [alpha, setAlpha] = useState(1.0);

  const model = useMemo(() => {
    const n = counts.length;
    const d = TOKENS.length;
    // Each "document" is one row of the counts matrix; labels are the row indices.
    const X = new Float64Array(n * d);
    const y = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      y[i] = i;
      for (let j = 0; j < d; j++) X[i * d + j] = counts[i][j];
    }
    return multinomialNBFit(X, y, d, alpha);
  }, [counts, alpha]);

  const result = useMemo(() => {
    const d = TOKENS.length;
    const Xq = new Float64Array(d);
    for (let j = 0; j < d; j++) Xq[j] = doc[j];
    const proba = multinomialNBPredictProba(Xq, model);
    return Array.from(proba);
  }, [doc, model]);

  // Top-contributing tokens per class (using featureLogProbs - mean over classes as importance).
  const topTokensPerClass = useMemo(() => {
    const d = TOKENS.length;
    const K = CLASS_LABELS.length;
    const meanLog: number[] = new Array(d).fill(0);
    for (let j = 0; j < d; j++) {
      let s = 0;
      for (let k = 0; k < K; k++) s += model.featureLogProbs[k * d + j];
      meanLog[j] = s / K;
    }
    return CLASS_LABELS.map((_, k) => {
      const scored = TOKENS.map((tok, j) => ({
        tok,
        score: model.featureLogProbs[k * d + j] - meanLog[j],
      }));
      return scored.sort((a, b) => b.score - a.score).slice(0, 5);
    });
  }, [model]);

  const setCount = (i: number, j: number, val: number) => {
    setCounts((prev) => {
      const next = prev.map((r) => r.slice());
      next[i][j] = Math.max(0, val);
      return next;
    });
  };

  const setDocCount = (j: number, val: number) => {
    setDoc((prev) => {
      const next = prev.slice();
      next[j] = Math.max(0, val);
      return next;
    });
  };

  return (
    <div className="not-prose my-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div>
          <div className="flex items-center gap-3 mb-2 text-sm">
            <label className="flex items-center gap-2">
              Laplace α = {alpha.toFixed(2)}
              <input
                type="range"
                min={0.01}
                max={5}
                step={0.01}
                value={alpha}
                onChange={(e) => setAlpha(Number(e.target.value))}
                aria-label="Laplace smoothing alpha"
              />
            </label>
          </div>
          <div className="text-sm font-semibold mb-1">Per-class token counts</div>
          <div className="overflow-x-auto">
            <table className="w-full text-xs border-collapse" role="grid">
              <thead>
                <tr>
                  <th className="text-left py-1 pr-2"></th>
                  {TOKENS.map((tok) => (
                    <th key={tok} className="px-1 py-1 font-normal" scope="col">
                      {tok}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {counts.map((row, i) => (
                  <tr key={CLASS_LABELS[i]}>
                    <th
                      className="text-left pr-2 font-medium"
                      scope="row"
                      style={{ color: i === 0 ? paletteNB.pos : paletteNB.neg }}
                    >
                      {CLASS_LABELS[i]}
                    </th>
                    {row.map((v, j) => (
                      <td key={j} className="px-1 py-0.5">
                        <input
                          type="number"
                          min={0}
                          step={1}
                          value={v}
                          onChange={(e) => setCount(i, j, Number(e.target.value))}
                          className="w-12 text-center bg-transparent border rounded text-xs px-1"
                          aria-label={`${CLASS_LABELS[i]} count for ${TOKENS[j]}`}
                        />
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="text-xs opacity-75 mt-2">
            Top distinctive tokens per class (highest log-likelihood-ratio against the mean):
            <ul className="list-none pl-2 mt-1">
              {CLASS_LABELS.map((cls, k) => (
                <li key={cls}>
                  <strong style={{ color: k === 0 ? paletteNB.pos : paletteNB.neg }}>{cls}:</strong>{' '}
                  {topTokensPerClass[k].map((t) => t.tok).join(', ')}
                </li>
              ))}
            </ul>
          </div>
        </div>
        <div>
          <div className="text-sm font-semibold mb-1">Query document token counts</div>
          <div className="grid grid-cols-2 gap-1 mb-3">
            {TOKENS.map((tok, j) => (
              <label key={tok} className="flex items-center gap-1 text-xs">
                <span className="w-16">{tok}</span>
                <input
                  type="number"
                  min={0}
                  step={1}
                  value={doc[j]}
                  onChange={(e) => setDocCount(j, Number(e.target.value))}
                  className="w-12 text-center bg-transparent border rounded px-1"
                  aria-label={`Query count for ${tok}`}
                />
              </label>
            ))}
          </div>
          <div className="text-sm font-semibold mb-1">Posterior P(class | doc)</div>
          <div className="space-y-1">
            {CLASS_LABELS.map((cls, k) => (
              <div key={cls} className="flex items-center gap-2 text-sm">
                <span className="w-20" style={{ color: k === 0 ? paletteNB.pos : paletteNB.neg }}>
                  {cls}
                </span>
                <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded h-3 overflow-hidden">
                  <div
                    className="h-full"
                    style={{
                      width: `${result[k] * 100}%`,
                      background: k === 0 ? paletteNB.pos : paletteNB.neg,
                    }}
                  />
                </div>
                <span className="w-14 text-right tabular-nums">{result[k].toFixed(3)}</span>
              </div>
            ))}
          </div>
          <div className="text-xs opacity-75 mt-3">
            The classifier predicts <strong>{CLASS_LABELS[result[0] > result[1] ? 0 : 1]}</strong>.
            Try adding patient/doctor counts to flip the verdict.
          </div>
        </div>
      </div>
    </div>
  );
}
