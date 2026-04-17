# Topic assets (optimized path)

Images placed under `src/assets/topics/<topic>/` are processed by Astro's
`astro:assets` service at build time. They receive:

- Automatic **srcset** generation (multiple widths, browser picks the right one)
- Automatic **WebP** conversion (falls back to the original format)
- Automatic **width/height** injection (prevents CLS)
- **Hash-based filenames** for aggressive caching

## When to put images here vs. in `public/`

| Location | When to use | Optimization |
|----------|-------------|--------------|
| `src/assets/topics/<topic>/` | New images, or any image worth optimizing for mobile | Full srcset + WebP |
| `public/images/topics/<topic>/` | Legacy images not yet migrated; images referenced by URL outside MDX | Plain `<img>` only |

## Usage pattern

Topic MDX files use YAML frontmatter (between `---` lines) for content-collection metadata. JS imports go **after** the YAML block, at the top of the MDX body. `Figure` must be imported explicitly — there is no global MDX components mapping configured.

```mdx
---
title: "Adjunctions"
# ... other YAML frontmatter ...
---

import Figure from '../../components/ui/Figure.astro';
import galoisConnections from '../../assets/topics/adjunctions/galois-connections.png';

<Figure
  src={galoisConnections}
  alt="Galois connection between ordered sets, showing f ⊣ g"
  caption="A Galois connection is an adjunction in the 2-category of posets."
/>
```

For legacy `public/` images, pass the string path — `<Figure>` falls back to a plain `<img>`:

```mdx
---
title: "..."
---

import Figure from '../../components/ui/Figure.astro';

<Figure
  src="/images/topics/adjunctions/galois-connections.png"
  alt="..."
  caption="..."
/>
```

## Migration

Existing `public/images/topics/*` images can be moved here incrementally; no
bulk migration is required. When a topic is revisited for content work, move
its images into `src/assets/topics/<topic>/` and update the MDX imports.
