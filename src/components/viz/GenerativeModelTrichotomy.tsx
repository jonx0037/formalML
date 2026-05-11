// =============================================================================
// GenerativeModelTrichotomy.tsx
//
// §1.2 — Three-panel interactive that visualizes the VAE / GAN / flow
// trichotomy. The reader selects one of the three families via a tab control;
// for each, the panel shows the forward sampling path, which boxes are
// tractable (sample? density? exact-density?), and the training-objective
// box.
//
// Pure React UI, no D3. Tab control + SVG schematic per family. Each panel
// uses the same layout so the differences are visible by inspection.
// =============================================================================

import { useState } from 'react';

type Family = 'flow' | 'vae' | 'gan';

interface FamilySpec {
  label: string;
  forwardLabel: string;
  forwardSub: string;
  tractable: {
    sample: 'yes' | 'bound' | 'no';
    density: 'exact' | 'bound' | 'no';
  };
  objective: string;
  description: string;
}

const FAMILIES: Record<Family, FamilySpec> = {
  flow: {
    label: 'Flow',
    forwardLabel: 'z ~ N(0, I_d)  →  T_φ(z)  =  x',
    forwardSub: 'invertible map T_φ; both directions cheap',
    tractable: { sample: 'yes', density: 'exact' },
    objective: 'Exact MLE on log p_φ(x)',
    description:
      'Flows parameterize an invertible map T_φ between a Gaussian base and the data distribution. The change-of-variables formula gives the exact density. Sampling and density evaluation are both cheap.',
  },
  vae: {
    label: 'VAE',
    forwardLabel: 'z ~ p(z)  →  Decoder(z)  =  x',
    forwardSub: 'Encoder for posterior q_φ(z|x); decoder for p_θ(x|z)',
    tractable: { sample: 'yes', density: 'bound' },
    objective: 'Maximize ELBO (lower bound on log p_θ(x))',
    description:
      'VAEs maximize a lower bound on the log-likelihood because marginalizing z is intractable. Sampling is cheap, but density evaluation only gives the ELBO — not the truth.',
  },
  gan: {
    label: 'GAN',
    forwardLabel: 'z ~ p(z)  →  G_φ(z)  =  x',
    forwardSub: 'Discriminator D_ψ(x) trained against G_φ',
    tractable: { sample: 'yes', density: 'no' },
    objective: 'Adversarial min-max (no explicit density)',
    description:
      'GANs train a generator implicitly — push noise through, get a sample. No density to query. Often the sharpest samples, but useless when you need log p(x).',
  },
};

const PIP_COLOR = {
  yes: 'var(--color-accent)',
  exact: 'var(--color-accent)',
  bound: '#D97706',
  no: '#9CA3AF',
};

function Pip({ state, label, sub }: { state: 'yes' | 'exact' | 'bound' | 'no'; label: string; sub?: string }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
      <span
        style={{
          width: 14,
          height: 14,
          borderRadius: 999,
          backgroundColor: PIP_COLOR[state],
          flexShrink: 0,
        }}
      />
      <span style={{ fontSize: 13, color: 'var(--color-text)' }}>{label}</span>
      {sub && (
        <span style={{ fontSize: 11, color: 'var(--color-text-secondary)', fontStyle: 'italic' }}>{sub}</span>
      )}
    </div>
  );
}

export default function GenerativeModelTrichotomy() {
  const [family, setFamily] = useState<Family>('flow');
  const spec = FAMILIES[family];

  return (
    <div className="my-8 not-prose" style={{ border: '1px solid var(--color-border)', borderRadius: 8, padding: 20 }}>
      {/* Tab control */}
      <div style={{ display: 'flex', gap: 0, borderBottom: '1px solid var(--color-border)', marginBottom: 18 }}>
        {(Object.keys(FAMILIES) as Family[]).map((f) => {
          const active = family === f;
          return (
            <button
              key={f}
              onClick={() => setFamily(f)}
              aria-pressed={active}
              style={{
                background: 'none',
                border: 'none',
                padding: '8px 18px',
                fontSize: 14,
                cursor: 'pointer',
                color: active ? 'var(--color-accent)' : 'var(--color-text-secondary)',
                borderBottom: active ? '2px solid var(--color-accent)' : '2px solid transparent',
                fontWeight: active ? 600 : 400,
                marginBottom: -1,
              }}
            >
              {FAMILIES[f].label}
            </button>
          );
        })}
      </div>

      {/* Forward path schematic */}
      <div style={{ marginBottom: 18 }}>
        <div style={{ fontSize: 12, color: 'var(--color-text-secondary)', marginBottom: 4 }}>Forward path</div>
        <div
          style={{
            fontFamily: 'var(--font-mono, monospace)',
            fontSize: 14,
            color: 'var(--color-text)',
            background: 'color-mix(in srgb, var(--color-accent) 8%, transparent)',
            padding: '10px 14px',
            borderRadius: 6,
          }}
        >
          {spec.forwardLabel}
        </div>
        <div style={{ fontSize: 12, color: 'var(--color-text-secondary)', marginTop: 4, fontStyle: 'italic' }}>
          {spec.forwardSub}
        </div>
      </div>

      {/* Tractability pips */}
      <div style={{ marginBottom: 18 }}>
        <div style={{ fontSize: 12, color: 'var(--color-text-secondary)', marginBottom: 8 }}>What's tractable</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <Pip state={spec.tractable.sample} label="Sampling" sub="forward through the model" />
          <Pip
            state={spec.tractable.density}
            label="Density evaluation"
            sub={spec.tractable.density === 'exact' ? 'closed-form log p(x)' : spec.tractable.density === 'bound' ? 'ELBO only' : 'unavailable'}
          />
        </div>
      </div>

      {/* Training objective */}
      <div style={{ marginBottom: 14 }}>
        <div style={{ fontSize: 12, color: 'var(--color-text-secondary)', marginBottom: 4 }}>Training objective</div>
        <div style={{ fontSize: 13, color: 'var(--color-text)' }}>{spec.objective}</div>
      </div>

      {/* Description */}
      <div style={{ fontSize: 13, color: 'var(--color-text-secondary)', lineHeight: 1.5 }}>{spec.description}</div>
    </div>
  );
}
