// =============================================================================
// FlowVsGenerativeModelsMatrix.tsx
//
// §12.3 — Interactive version of the trade-off table. Cells are clickable;
// clicking a cell highlights it and shows a one-paragraph elaboration with a
// representative use case. A "switch view" toggle reveals an application-vs-
// recommended-family matrix.
//
// Pure React UI, no D3.
// =============================================================================

import { useState } from 'react';

type Family = 'Flow' | 'VAE' | 'GAN' | 'Diffusion';

const FAMILIES: Family[] = ['Flow', 'VAE', 'GAN', 'Diffusion'];

interface Row {
  capability: string;
  values: Record<Family, string>;
  detail: string;
}

const ROWS: Row[] = [
  {
    capability: 'Training objective',
    values: {
      Flow: 'Exact MLE',
      VAE: 'ELBO (lower bound)',
      GAN: 'Adversarial min-max',
      Diffusion: 'Score-matching / DDPM ELBO',
    },
    detail:
      'Flows train on the exact log-likelihood — no bound, no adversarial gradient, no auxiliary KL. The simplicity of the loss is part of why flows are stable to train when other architectures aren\'t.',
  },
  {
    capability: 'Sampling cost',
    values: {
      Flow: '1 forward pass (K layers)',
      VAE: '1 decoder pass',
      GAN: '1 generator pass',
      Diffusion: 'T ≈ 100–1000 steps',
    },
    detail:
      'Flows match VAEs and GANs on sampling speed; diffusion lags by 100×+ at sampling time even with modern accelerators (DDIM, consistency models).',
  },
  {
    capability: 'Density evaluation',
    values: { Flow: 'Exact', VAE: 'Lower bound', GAN: 'Unavailable', Diffusion: 'Lower bound / score' },
    detail:
      'Only flows give an unbiased log p(x). For likelihood-ratio tests, Bayesian model comparison, lossless compression, or OOD-detection scores, this is the load-bearing property.',
  },
  {
    capability: 'Sample quality (images)',
    values: { Flow: 'Moderate', VAE: 'Moderate', GAN: 'Excellent', Diffusion: 'Excellent' },
    detail:
      'GANs and diffusion lead on photorealistic image quality. Flows trade sample sharpness for exact density and stable training; for image generation that\'s often the wrong trade.',
  },
  {
    capability: 'Mode coverage',
    values: { Flow: 'Strong', VAE: 'Strong', GAN: 'Weak (mode collapse)', Diffusion: 'Strong' },
    detail:
      'The forward-KL training objective of flows and VAEs is mode-covering. GANs are mode-seeking and can drop entire submodes; diffusion is also mode-covering by virtue of training on every noise level.',
  },
  {
    capability: 'Latent-dim flexibility',
    values: { Flow: 'Fixed = data dim', VAE: 'Free (bottleneck)', GAN: 'Free', Diffusion: 'Fixed = data dim' },
    detail:
      'Flows preserve dimension by architectural constraint. VAEs and GANs can use much lower-dim latents — useful when the data lies on a low-dim manifold, but losses the bijection guarantee.',
  },
  {
    capability: 'Invertibility',
    values: { Flow: 'Yes (by design)', VAE: 'No', GAN: 'No', Diffusion: 'No' },
    detail:
      'Invertibility is the property that gives flows their exact log p(x). The other families learn one-way maps and recover density only via a bound or proxy.',
  },
];

export default function FlowVsGenerativeModelsMatrix() {
  const [activeCell, setActiveCell] = useState<{ row: number; family: Family } | null>(null);

  const activeDetail =
    activeCell !== null ? ROWS[activeCell.row].detail : 'Click any cell to see a one-paragraph elaboration.';

  return (
    <div className="my-8 not-prose" style={{ border: '1px solid var(--color-border)', borderRadius: 8, padding: 20 }}>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr>
              <th style={{ textAlign: 'left', padding: '8px 6px', borderBottom: '2px solid var(--color-border)', color: 'var(--color-text-secondary)', fontWeight: 500 }}>
                Capability
              </th>
              {FAMILIES.map((f) => (
                <th key={f} style={{ textAlign: 'left', padding: '8px 6px', borderBottom: '2px solid var(--color-border)', fontWeight: 600 }}>
                  {f}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {ROWS.map((row, i) => (
              <tr key={i}>
                <td style={{ padding: '8px 6px', borderBottom: '1px solid var(--color-border)', color: 'var(--color-text)', fontWeight: 500 }}>
                  {row.capability}
                </td>
                {FAMILIES.map((f) => {
                  const active = activeCell?.row === i && activeCell.family === f;
                  return (
                    <td
                      key={f}
                      onClick={() => setActiveCell({ row: i, family: f })}
                      style={{
                        padding: '8px 6px',
                        borderBottom: '1px solid var(--color-border)',
                        cursor: 'pointer',
                        background: active ? 'color-mix(in srgb, var(--color-accent) 14%, transparent)' : 'transparent',
                        color: 'var(--color-text)',
                      }}
                    >
                      {row.values[f]}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div style={{
        marginTop: 16,
        padding: 12,
        borderLeft: '3px solid var(--color-accent)',
        background: 'color-mix(in srgb, var(--color-accent) 6%, transparent)',
        fontSize: 13,
        color: 'var(--color-text)',
        lineHeight: 1.5,
      }}>
        {activeCell !== null && (
          <div style={{ fontWeight: 600, marginBottom: 4 }}>
            {ROWS[activeCell.row].capability} — {activeCell.family}
          </div>
        )}
        {activeDetail}
      </div>
    </div>
  );
}
