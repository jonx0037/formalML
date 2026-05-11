// =============================================================================
// DebiasedLassoCoverage — §10.5 static figure + numerical coverage table.
//
// The Monte Carlo (B = 200, n = 200, p = 100, s = 5, three coord types) takes
// ~30-60 s in Python; replicating it client-side would freeze the browser.
// Instead, this component renders:
//   • the precomputed figure (fig_10_01_debiased_coverage.png) inline, and
//   • a numerical table of the printed values from notebook §10.5 Cell 75.
//
// The table is the headline: at "borderline" signals (β* = 0.15), the naive
// post-selection CI undercovers at 24.0% while OLS and the debiased lasso
// both stay near 95%. The undercoverage is the §10.5 thesis.
//
// Values copied verbatim from the executed notebook output (B = 200 MC reps,
// nominal 95% CIs):
//
//   Method                    Strong (b*=1.0)    Borderline (b*=0.15)    Noise (b*=0)
//   OLS                                 0.950                  0.943           0.951
//   Naive post-sel                      0.945                  0.240           1.000
//   Debiased lasso                      0.982                  0.988           0.984
//
// No solver computation, no slider — pure presentation component.
// =============================================================================

interface CoverageRow {
  method: string;
  strong: number;
  borderline: number;
  noise: number;
  isHighlight?: boolean; // true for the row whose borderline-coverage demos the §10.5 thesis
}

const COVERAGE_ROWS: CoverageRow[] = [
  { method: 'OLS (gold standard)', strong: 0.950, borderline: 0.943, noise: 0.951 },
  { method: 'Naive post-selection', strong: 0.945, borderline: 0.240, noise: 1.000, isHighlight: true },
  { method: 'Debiased lasso', strong: 0.982, borderline: 0.988, noise: 0.984 },
];

const NOMINAL = 0.95;

function CoverageCell({ value, undercover }: { value: number; undercover: boolean }) {
  const pct = (value * 100).toFixed(1);
  return (
    <td
      style={{
        padding: '6px 10px',
        textAlign: 'right',
        fontFamily: 'var(--font-mono)',
        fontSize: '13px',
        color: undercover ? '#B91C1C' : 'var(--color-text)',
        fontWeight: undercover ? 600 : 400,
        borderBottom: '1px solid var(--color-border)',
      }}
    >
      {pct}%
    </td>
  );
}

export default function DebiasedLassoCoverage() {
  return (
    <div style={{ width: '100%' }}>
      <img
        src="/images/topics/high-dimensional-regression/fig_10_01_debiased_coverage.png"
        alt="Empirical 95% CI coverage rates for OLS, naive post-selection, and debiased lasso at three signal strengths (strong, borderline, noise)."
        style={{ width: '100%', height: 'auto', display: 'block', marginBottom: '12px' }}
      />
      <div style={{ overflowX: 'auto' }}>
        <table
          style={{
            width: '100%',
            borderCollapse: 'collapse',
            fontFamily: 'var(--font-sans)',
            fontSize: '13px',
            color: 'var(--color-text)',
          }}
        >
          <caption
            style={{
              captionSide: 'top',
              textAlign: 'left',
              fontFamily: 'var(--font-sans)',
              fontSize: '12px',
              color: 'var(--color-text-secondary)',
              padding: '0 0 8px 0',
            }}
          >
            Empirical 95% CI coverage rates (B = 200 MC replicates) on DGP-1-style data at n = 200, p = 100, s = 5.
            Three coord types: <strong>strong</strong> (β* = 1.0), <strong>borderline</strong> (β* = 0.15), <strong>noise</strong> (β* = 0).
          </caption>
          <thead>
            <tr style={{ borderBottom: '2px solid var(--color-border)' }}>
              <th style={{ padding: '8px 10px', textAlign: 'left', fontWeight: 600 }}>Method</th>
              <th style={{ padding: '8px 10px', textAlign: 'right', fontWeight: 600 }}>
                Strong (β* = 1.0)
              </th>
              <th style={{ padding: '8px 10px', textAlign: 'right', fontWeight: 600 }}>
                Borderline (β* = 0.15)
              </th>
              <th style={{ padding: '8px 10px', textAlign: 'right', fontWeight: 600 }}>
                Noise (β* = 0)
              </th>
            </tr>
          </thead>
          <tbody>
            {COVERAGE_ROWS.map((row) => (
              <tr key={row.method}>
                <td
                  style={{
                    padding: '6px 10px',
                    fontFamily: 'var(--font-sans)',
                    fontSize: '13px',
                    fontWeight: row.isHighlight ? 600 : 400,
                    borderBottom: '1px solid var(--color-border)',
                  }}
                >
                  {row.method}
                </td>
                <CoverageCell value={row.strong} undercover={row.strong < NOMINAL - 0.1} />
                <CoverageCell value={row.borderline} undercover={row.borderline < NOMINAL - 0.1} />
                <CoverageCell value={row.noise} undercover={row.noise < NOMINAL - 0.1} />
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p
        style={{
          fontFamily: 'var(--font-sans)',
          fontSize: '13px',
          color: 'var(--color-text-secondary)',
          marginTop: '12px',
        }}
      >
        The headline is the <strong>24.0%</strong> entry: at borderline-strength signals (β* = 0.15), naive post-selection CIs undercover by ~70 percentage points. Strong signals (β* = 1.0) are always selected and stably estimated, so the naive procedure happens to cover near nominal; noise coords are typically unselected, making the naive CI degenerate at {'{0}'} which trivially covers β* = 0. The OLS baseline (available here because p &lt; n) and the debiased lasso both recover ~95% coverage uniformly across coord types. At p &gt; n where OLS is infeasible, the debiased lasso with nodewise-M̂ is the only valid CI procedure of the three.
      </p>
    </div>
  );
}
