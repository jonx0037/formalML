# formalml.com Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build formalml.com — an Astro 5 + React + MDX site for deep-dive ML mathematics explainers with interactive D3 visualizations, KaTeX math rendering, and a three-pillar content model (rigorous math, visual intuition, working code).

**Architecture:** Static-first Astro 5 site using content collections (glob loader) for MDX topics. React components hydrate as islands (`client:visible`) for interactive D3 visualizations. Tailwind v4 via Vite plugin for styling. KaTeX server-renders math. Pagefind provides static search. Deploys to Vercel.

**Tech Stack:** Astro 5.x, React 18, Tailwind CSS 4 (@tailwindcss/vite), D3.js v7, KaTeX (remark-math + rehype-katex), Shiki, Pagefind, pnpm

**Brief:** Full spec at `/formalml-handoff-brief-v1.md` — Sections 6-8 are the densest references.

**Key corrections from brief:** (1) Config file is `src/content.config.ts` not `src/content/config.ts` (Astro 5 change). (2) Collections use `glob()` loader, not `type: 'content'`. (3) Tailwind v4 uses `@tailwindcss/vite` plugin, not deprecated `@astrojs/tailwind`. (4) Collection entries use `id` not `slug` for routing.

---

## Phase 1: Project Scaffold

### Task 1: Initialize Astro Project

**Files:**
- Create: `package.json`, `astro.config.mjs`, `tsconfig.json`
- Create: `src/styles/global.css`
- Create: `public/favicon.svg`

**Step 1: Create Astro project with pnpm**

```bash
cd /Users/jonathanrocha/Developer/Sites/formalML
pnpm create astro@latest . --template minimal --install --no-git --typescript strict
```

If the directory isn't empty, accept the prompt to overwrite (the only existing files are the brief and firebase log).

**Step 2: Add integrations — React and MDX**

```bash
pnpm astro add react mdx --yes
```

**Step 3: Install Tailwind v4 via Vite plugin**

```bash
pnpm add tailwindcss @tailwindcss/vite
```

Then update `astro.config.mjs`:

```javascript
import { defineConfig } from 'astro/config';
import react from '@astrojs/react';
import mdx from '@astrojs/mdx';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  site: 'https://formalml.com',
  integrations: [react(), mdx()],
  vite: {
    plugins: [tailwindcss()],
  },
});
```

**Step 4: Install KaTeX + math remark/rehype plugins**

```bash
pnpm add remark-math rehype-katex katex
```

Update `astro.config.mjs` to add markdown plugins:

```javascript
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

export default defineConfig({
  site: 'https://formalml.com',
  integrations: [react(), mdx()],
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
  },
  vite: {
    plugins: [tailwindcss()],
  },
});
```

**Step 5: Install D3**

```bash
pnpm add d3 @types/d3
```

**Step 6: Create global CSS**

Create `src/styles/global.css`:

```css
@import "tailwindcss";

/* === Design System Tokens === */
@theme {
  --color-bg: #FAFAF8;
  --color-bg-dark: #1A1A18;
  --color-text: #1A1A1A;
  --color-text-dark: #E8E8E4;
  --color-text-secondary: #6B6B6B;
  --color-accent: #0F6E56;
  --color-accent-secondary: #534AB7;
  --color-code-bg: #F5F5F0;
  --color-definition-bg: #E1F5EE;
  --color-definition-border: #0F6E56;
  --color-theorem-bg: #EEEDFE;
  --color-theorem-border: #534AB7;

  --font-sans: 'Inter', ui-sans-serif, system-ui, sans-serif;
  --font-serif: 'Source Serif 4', 'Charter', 'Georgia', ui-serif, serif;
  --font-mono: 'JetBrains Mono', 'Fira Code', ui-monospace, monospace;
}

/* === Base typography === */
html {
  background-color: var(--color-bg);
  color: var(--color-text);
}

.dark {
  background-color: var(--color-bg-dark);
  color: var(--color-text-dark);
}

/* === Prose styling === */
.prose {
  font-family: var(--font-serif);
  font-size: 1.125rem;
  line-height: 1.7;
  max-width: 45rem; /* 720px */
}

.prose h1, .prose h2, .prose h3, .prose h4 {
  font-family: var(--font-sans);
  font-weight: 600;
}

/* === KaTeX overrides === */
.katex-display {
  overflow-x: auto;
  overflow-y: hidden;
  padding: 0.5rem 0;
}

/* === Code block styling === */
pre {
  background-color: var(--color-code-bg) !important;
  border-radius: 0.5rem;
  padding: 1rem;
  overflow-x: auto;
  font-size: 0.875rem;
}

code {
  font-family: var(--font-mono);
}
```

**Step 7: Create favicon**

Create `public/favicon.svg` — a simple math-themed SVG:

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
  <rect width="32" height="32" rx="4" fill="#0F6E56"/>
  <text x="16" y="23" text-anchor="middle" fill="white" font-family="serif" font-size="20" font-weight="bold">∂</text>
</svg>
```

**Step 8: Verify build**

```bash
pnpm build
```

Expected: Build succeeds with no errors.

**Step 9: Commit**

```bash
git init
echo "node_modules\ndist\n.astro\n.vercel\nfirebase-debug.log" > .gitignore
git add .
git commit -m "feat: initialize Astro 5 project with React, MDX, Tailwind v4, KaTeX, D3"
```

---

### Task 2: Content Collection Schema

**Files:**
- Create: `src/content.config.ts`
- Create: `src/content/topics/` directory (empty, for future MDX files)

**Step 1: Create the content collection config**

Create `src/content.config.ts`:

```typescript
import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

const topics = defineCollection({
  loader: glob({ pattern: '**/*.mdx', base: './src/content/topics' }),
  schema: z.object({
    title: z.string(),
    subtitle: z.string().optional(),
    status: z.enum(['draft', 'review', 'published']),
    difficulty: z.enum(['foundational', 'intermediate', 'advanced']),
    prerequisites: z.array(z.string()).default([]),
    tags: z.array(z.string()),
    domain: z.enum([
      'topology',
      'geometry',
      'probability',
      'optimization',
      'linear-algebra',
      'information-theory',
      'graph-theory',
      'category-theory',
    ]),
    videoId: z.string().nullable().default(null),
    notebookPath: z.string().nullable().default(null),
    githubUrl: z.string().url().nullable().default(null),
    datePublished: z.coerce.date().optional(),
    dateUpdated: z.coerce.date().optional(),
    estimatedReadTime: z.number().optional(),
    abstract: z.string(),
    connections: z
      .array(
        z.object({
          topic: z.string(),
          relationship: z.string(),
        }),
      )
      .default([]),
    references: z
      .array(
        z.object({
          type: z.enum(['paper', 'book', 'course', 'blog', 'video']),
          title: z.string(),
          authors: z.string().optional(),
          year: z.number().optional(),
          url: z.string().url().optional(),
          note: z.string().optional(),
        }),
      )
      .default([]),
  }),
});

export const collections = { topics };
```

**Step 2: Create a minimal test MDX to validate schema**

Create `src/content/topics/test-topic.mdx`:

```mdx
---
title: "Test Topic"
status: "draft"
difficulty: "foundational"
tags: ["test"]
domain: "topology"
abstract: "A test topic to validate the content collection schema."
---

# Test Topic

This is a test.
```

**Step 3: Verify build compiles with schema**

```bash
pnpm build
```

Expected: Build succeeds. Then delete the test topic file.

**Step 4: Commit**

```bash
git add src/content.config.ts
git commit -m "feat: add content collection schema for topics"
```

---

## Phase 2: Layouts and Core UI

### Task 3: BaseLayout

**Files:**
- Create: `src/layouts/BaseLayout.astro`

**Step 1: Build the HTML shell layout**

Create `src/layouts/BaseLayout.astro`:

```astro
---
interface Props {
  title: string;
  description?: string;
  ogImage?: string;
}

const { title, description = 'The mathematical machinery behind modern machine learning', ogImage } = Astro.props;
const canonicalURL = new URL(Astro.url.pathname, Astro.site);
---

<!doctype html>
<html lang="en" class="scroll-smooth">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <link rel="icon" type="image/svg+xml" href="/favicon.svg" />
    <link rel="canonical" href={canonicalURL} />

    <title>{title} | formalML</title>
    <meta name="description" content={description} />

    <!-- OG -->
    <meta property="og:title" content={`${title} | formalML`} />
    <meta property="og:description" content={description} />
    <meta property="og:type" content="article" />
    <meta property="og:url" content={canonicalURL} />
    {ogImage && <meta property="og:image" content={ogImage} />}

    <!-- KaTeX CSS -->
    <link
      rel="stylesheet"
      href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css"
      crossorigin="anonymous"
    />

    <!-- Global styles -->
    <style>
      @import '../styles/global.css';
    </style>
  </head>
  <body class="min-h-screen bg-[var(--color-bg)] text-[var(--color-text)] font-[family-name:var(--font-serif)]">
    <slot name="nav" />
    <main>
      <slot />
    </main>
    <slot name="footer" />
  </body>
</html>
```

**Step 2: Verify it renders**

Update `src/pages/index.astro` to use the layout:

```astro
---
import BaseLayout from '../layouts/BaseLayout.astro';
---

<BaseLayout title="Home">
  <h1 class="text-4xl font-semibold font-[family-name:var(--font-sans)] text-center mt-24">
    formalML
  </h1>
  <p class="text-center text-[var(--color-text-secondary)] mt-4 text-lg">
    The mathematical machinery behind modern machine learning
  </p>
</BaseLayout>
```

```bash
pnpm dev
# Visit http://localhost:4321 — should show title + tagline on warm white background
```

**Step 3: Commit**

```bash
git add src/layouts/BaseLayout.astro src/pages/index.astro src/styles/global.css
git commit -m "feat: add BaseLayout with KaTeX CSS, global styles, design system tokens"
```

---

### Task 4: Nav and Footer

**Files:**
- Create: `src/components/ui/Nav.astro`
- Create: `src/components/ui/Footer.astro`
- Modify: `src/layouts/BaseLayout.astro`

**Step 1: Build Nav component**

Create `src/components/ui/Nav.astro`:

```astro
---
const currentPath = Astro.url.pathname;

const links = [
  { href: '/topics', label: 'Topics' },
  { href: '/paths', label: 'Paths' },
  { href: '/about', label: 'About' },
];
---

<nav class="sticky top-0 z-50 border-b border-gray-200/60 bg-[var(--color-bg)]/95 backdrop-blur-sm">
  <div class="mx-auto flex max-w-5xl items-center justify-between px-6 py-4">
    <a href="/" class="font-[family-name:var(--font-sans)] text-xl font-semibold tracking-tight">
      formal<span class="text-[var(--color-accent)]">ML</span>
    </a>
    <div class="flex items-center gap-6">
      {links.map(({ href, label }) => (
        <a
          href={href}
          class:list={[
            'font-[family-name:var(--font-sans)] text-sm transition-colors',
            currentPath.startsWith(href)
              ? 'text-[var(--color-accent)] font-medium'
              : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text)]',
          ]}
        >
          {label}
        </a>
      ))}
      <button id="search-trigger" class="text-[var(--color-text-secondary)] hover:text-[var(--color-text)]" aria-label="Search">
        <svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
        </svg>
      </button>
    </div>
  </div>
</nav>
```

**Step 2: Build Footer component**

Create `src/components/ui/Footer.astro`:

```astro
---
const year = new Date().getFullYear();
---

<footer class="mt-24 border-t border-gray-200/60 py-12">
  <div class="mx-auto max-w-5xl px-6">
    <div class="flex flex-col items-center gap-4 text-center text-sm text-[var(--color-text-secondary)]">
      <p>
        A project by <a href="https://github.com/jonx0037" class="text-[var(--color-accent)] hover:underline">Jonathan Rocha</a>
      </p>
      <p>
        <a href="https://datasalt.ai" class="hover:underline">DataSalt LLC</a>
        &middot;
        <a href="https://github.com/jonx0037/formalml" class="hover:underline">GitHub</a>
      </p>
      <p class="text-xs">&copy; {year} formalML. All rights reserved.</p>
    </div>
  </div>
</footer>
```

**Step 3: Wire Nav and Footer into BaseLayout**

Update `BaseLayout.astro` to import and render Nav and Footer directly (not via named slots):

```astro
---
import Nav from '../components/ui/Nav.astro';
import Footer from '../components/ui/Footer.astro';

interface Props {
  title: string;
  description?: string;
  ogImage?: string;
}

const { title, description = 'The mathematical machinery behind modern machine learning', ogImage } = Astro.props;
const canonicalURL = new URL(Astro.url.pathname, Astro.site);
---

<!doctype html>
<html lang="en" class="scroll-smooth">
  <head>
    <!-- same head as before -->
  </head>
  <body class="min-h-screen bg-[var(--color-bg)] text-[var(--color-text)] font-[family-name:var(--font-serif)]">
    <Nav />
    <main>
      <slot />
    </main>
    <Footer />
  </body>
</html>
```

**Step 4: Verify dev server shows nav + footer**

```bash
pnpm dev
```

**Step 5: Commit**

```bash
git add src/components/ui/Nav.astro src/components/ui/Footer.astro src/layouts/BaseLayout.astro
git commit -m "feat: add Nav and Footer components, wire into BaseLayout"
```

---

### Task 5: TheoremBlock Component

**Files:**
- Create: `src/components/ui/TheoremBlock.astro`

This is a critical editorial component — it makes the site feel like a mathematical text.

**Step 1: Build the TheoremBlock**

Create `src/components/ui/TheoremBlock.astro`:

```astro
---
interface Props {
  type: 'definition' | 'theorem' | 'lemma' | 'proposition' | 'corollary' | 'proof' | 'remark' | 'example';
  number?: number;
  title?: string;
}

const { type, number, title } = Astro.props;

const config: Record<string, { label: string; borderColor: string; bgColor: string; numbered: boolean; isProof: boolean }> = {
  definition: { label: 'Definition', borderColor: 'border-l-[var(--color-definition-border)]', bgColor: 'bg-[var(--color-definition-bg)]', numbered: true, isProof: false },
  theorem: { label: 'Theorem', borderColor: 'border-l-[var(--color-theorem-border)]', bgColor: 'bg-[var(--color-theorem-bg)]', numbered: true, isProof: false },
  lemma: { label: 'Lemma', borderColor: 'border-l-[var(--color-theorem-border)]', bgColor: 'bg-[var(--color-theorem-bg)]', numbered: true, isProof: false },
  proposition: { label: 'Proposition', borderColor: 'border-l-[var(--color-theorem-border)]', bgColor: 'bg-[var(--color-theorem-bg)]', numbered: true, isProof: false },
  corollary: { label: 'Corollary', borderColor: 'border-l-[var(--color-theorem-border)]', bgColor: 'bg-[var(--color-theorem-bg)]', numbered: true, isProof: false },
  proof: { label: 'Proof', borderColor: 'border-l-gray-300', bgColor: 'bg-transparent', numbered: false, isProof: true },
  remark: { label: 'Remark', borderColor: 'border-l-gray-300', bgColor: 'bg-gray-50', numbered: false, isProof: false },
  example: { label: 'Example', borderColor: 'border-l-gray-300', bgColor: 'bg-gray-50', numbered: true, isProof: false },
};

const c = config[type];
const displayLabel = c.numbered && number ? `${c.label} ${number}` : c.label;
const fullLabel = title ? `${displayLabel} (${title})` : displayLabel;
---

<div class:list={['my-6 border-l-4 rounded-r-lg px-6 py-4', c.borderColor, c.bgColor]}>
  {c.isProof ? (
    <details open>
      <summary class="cursor-pointer font-[family-name:var(--font-sans)] font-medium italic select-none">
        {fullLabel}.
      </summary>
      <div class="prose mt-2">
        <slot />
        <span class="float-right text-lg">∎</span>
      </div>
    </details>
  ) : (
    <>
      <p class="mb-2 font-[family-name:var(--font-sans)] font-semibold">
        {fullLabel}.
      </p>
      <div class="prose">
        <slot />
      </div>
    </>
  )}
</div>
```

**Step 2: Test with a scratch page**

Create a temporary `src/pages/test-theorem.astro`:

```astro
---
import BaseLayout from '../layouts/BaseLayout.astro';
import TheoremBlock from '../components/ui/TheoremBlock.astro';
---

<BaseLayout title="TheoremBlock Test">
  <div class="mx-auto max-w-3xl px-6 py-12">
    <TheoremBlock type="definition" number={1}>
      A <strong>metric space</strong> is a pair (X, d) where X is a set and d is a distance function.
    </TheoremBlock>

    <TheoremBlock type="theorem" number={1} title="Triangle Inequality">
      For all x, y, z in X: d(x, z) ≤ d(x, y) + d(y, z).
    </TheoremBlock>

    <TheoremBlock type="proof">
      Follows directly from the definition of a metric. The non-negativity and symmetry axioms, combined with the triangle inequality, ensure...
    </TheoremBlock>

    <TheoremBlock type="remark">
      This is a foundational result used throughout topology and analysis.
    </TheoremBlock>
  </div>
</BaseLayout>
```

```bash
pnpm dev
# Visit http://localhost:4321/test-theorem
# Verify: teal-bordered definition, purple-bordered theorem, collapsible proof with ∎, gray remark
```

**Step 3: Delete test page, commit**

```bash
rm src/pages/test-theorem.astro
git add src/components/ui/TheoremBlock.astro
git commit -m "feat: add TheoremBlock component with definition/theorem/proof/remark styles"
```

---

### Task 6: TopicLayout, DifficultyBadge, PrerequisiteChip

**Files:**
- Create: `src/layouts/TopicLayout.astro`
- Create: `src/components/ui/DifficultyBadge.astro`
- Create: `src/components/ui/PrerequisiteChip.astro`

**Step 1: Build DifficultyBadge**

Create `src/components/ui/DifficultyBadge.astro`:

```astro
---
interface Props {
  difficulty: 'foundational' | 'intermediate' | 'advanced';
}

const { difficulty } = Astro.props;

const styles: Record<string, string> = {
  foundational: 'bg-green-100 text-green-800',
  intermediate: 'bg-amber-100 text-amber-800',
  advanced: 'bg-red-100 text-red-800',
};
---

<span class:list={['inline-block rounded-full px-3 py-0.5 text-xs font-medium font-[family-name:var(--font-sans)] capitalize', styles[difficulty]]}>
  {difficulty}
</span>
```

**Step 2: Build PrerequisiteChip**

Create `src/components/ui/PrerequisiteChip.astro`:

```astro
---
interface Props {
  slug: string;
  title: string;
  available: boolean;
}

const { slug, title, available } = Astro.props;
---

{available ? (
  <a
    href={`/topics/${slug}`}
    class="inline-flex items-center gap-1 rounded-full border border-gray-200 px-3 py-1 text-xs font-[family-name:var(--font-sans)] text-[var(--color-accent)] hover:bg-gray-50 transition-colors"
  >
    {title}
  </a>
) : (
  <span class="inline-flex items-center gap-1 rounded-full border border-gray-200 px-3 py-1 text-xs font-[family-name:var(--font-sans)] text-gray-400">
    {title} <span class="text-[0.65rem]">(coming soon)</span>
  </span>
)}
```

**Step 3: Build TopicLayout**

Create `src/layouts/TopicLayout.astro`:

```astro
---
import BaseLayout from './BaseLayout.astro';
import DifficultyBadge from '../components/ui/DifficultyBadge.astro';
import PrerequisiteChip from '../components/ui/PrerequisiteChip.astro';
import { getCollection } from 'astro:content';

interface Props {
  frontmatter: {
    title: string;
    subtitle?: string;
    difficulty: 'foundational' | 'intermediate' | 'advanced';
    domain: string;
    prerequisites: string[];
    estimatedReadTime?: number;
    videoId?: string | null;
    abstract: string;
    connections: { topic: string; relationship: string }[];
    references: { type: string; title: string; authors?: string; year?: number; url?: string; note?: string }[];
  };
}

const { frontmatter } = Astro.props;

// Resolve prerequisites
const allTopics = await getCollection('topics');
const prereqs = frontmatter.prerequisites.map((slug) => {
  const topic = allTopics.find((t) => t.id === slug);
  return {
    slug,
    title: topic?.data.title ?? slug,
    available: topic?.data.status === 'published',
  };
});

// Find "where to go next" — topics that list this one as a prerequisite
const currentSlug = Astro.url.pathname.split('/').filter(Boolean).pop() ?? '';
const nextTopics = allTopics.filter(
  (t) => t.data.prerequisites.includes(currentSlug) && t.data.status === 'published'
);
---

<BaseLayout title={frontmatter.title} description={frontmatter.abstract}>
  <article class="mx-auto max-w-3xl px-6 py-12">
    <!-- Header -->
    <header class="mb-12">
      <div class="flex flex-wrap items-center gap-3 mb-4">
        <DifficultyBadge difficulty={frontmatter.difficulty} />
        <span class="text-xs font-[family-name:var(--font-sans)] text-[var(--color-text-secondary)] uppercase tracking-wider">
          {frontmatter.domain}
        </span>
        {frontmatter.estimatedReadTime && (
          <span class="text-xs text-[var(--color-text-secondary)]">
            {frontmatter.estimatedReadTime} min read
          </span>
        )}
      </div>
      <h1 class="text-4xl font-semibold font-[family-name:var(--font-sans)] leading-tight">
        {frontmatter.title}
      </h1>
      {frontmatter.subtitle && (
        <p class="mt-3 text-xl text-[var(--color-text-secondary)]">
          {frontmatter.subtitle}
        </p>
      )}
    </header>

    <!-- Prerequisites bar -->
    {prereqs.length > 0 && (
      <div class="mb-10 flex flex-wrap items-center gap-2">
        <span class="text-xs font-[family-name:var(--font-sans)] text-[var(--color-text-secondary)] uppercase tracking-wider mr-2">
          Prerequisites:
        </span>
        {prereqs.map((p) => (
          <PrerequisiteChip slug={p.slug} title={p.title} available={p.available} />
        ))}
      </div>
    )}

    <!-- Video embed (conditional) -->
    {frontmatter.videoId && (
      <div class="mb-12 aspect-video overflow-hidden rounded-lg">
        <iframe
          src={`https://www.youtube.com/embed/${frontmatter.videoId}`}
          class="h-full w-full"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          allowfullscreen
        />
      </div>
    )}

    <!-- MDX content (the three pillars) -->
    <div class="prose">
      <slot />
    </div>

    <!-- Connections -->
    {frontmatter.connections.length > 0 && (
      <section class="mt-16 border-t border-gray-200/60 pt-8">
        <h2 class="mb-4 text-lg font-semibold font-[family-name:var(--font-sans)]">Connections</h2>
        <ul class="space-y-2">
          {frontmatter.connections.map((c) => (
            <li class="text-sm">
              <span class="text-[var(--color-text-secondary)] italic">{c.relationship}</span>{' '}
              <a href={`/topics/${c.topic}`} class="text-[var(--color-accent)] hover:underline">{c.topic}</a>
            </li>
          ))}
        </ul>
      </section>
    )}

    <!-- References -->
    {frontmatter.references.length > 0 && (
      <section class="mt-12 border-t border-gray-200/60 pt-8">
        <h2 class="mb-4 text-lg font-semibold font-[family-name:var(--font-sans)]">References &amp; Further Reading</h2>
        <ul class="space-y-3">
          {frontmatter.references.map((r) => (
            <li class="text-sm">
              <span class="rounded bg-gray-100 px-1.5 py-0.5 text-xs uppercase">{r.type}</span>{' '}
              {r.url ? (
                <a href={r.url} class="text-[var(--color-accent)] hover:underline">{r.title}</a>
              ) : (
                <span>{r.title}</span>
              )}
              {r.authors && <span class="text-[var(--color-text-secondary)]"> — {r.authors}</span>}
              {r.year && <span class="text-[var(--color-text-secondary)]"> ({r.year})</span>}
              {r.note && <span class="block text-xs text-[var(--color-text-secondary)] mt-0.5">{r.note}</span>}
            </li>
          ))}
        </ul>
      </section>
    )}

    <!-- Navigation footer -->
    <nav class="mt-16 flex justify-between border-t border-gray-200/60 pt-8 text-sm">
      {prereqs.length > 0 && (
        <div>
          <span class="block text-xs text-[var(--color-text-secondary)] uppercase tracking-wider mb-2">Prerequisites</span>
          <div class="flex flex-wrap gap-2">
            {prereqs.filter((p) => p.available).map((p) => (
              <a href={`/topics/${p.slug}`} class="text-[var(--color-accent)] hover:underline">{p.title}</a>
            ))}
          </div>
        </div>
      )}
      {nextTopics.length > 0 && (
        <div class="text-right">
          <span class="block text-xs text-[var(--color-text-secondary)] uppercase tracking-wider mb-2">Where to go next</span>
          <div class="flex flex-wrap justify-end gap-2">
            {nextTopics.map((t) => (
              <a href={`/topics/${t.id}`} class="text-[var(--color-accent)] hover:underline">{t.data.title}</a>
            ))}
          </div>
        </div>
      )}
    </nav>
  </article>
</BaseLayout>
```

**Step 4: Commit**

```bash
git add src/layouts/TopicLayout.astro src/components/ui/DifficultyBadge.astro src/components/ui/PrerequisiteChip.astro
git commit -m "feat: add TopicLayout with prerequisite chips, difficulty badges, navigation footer"
```

---

## Phase 3: Flagship Visualization Pair

These two components are the heart of the site. Build and test them in isolation before wiring into MDX.

### Task 7: Shared Viz Infrastructure

**Files:**
- Create: `src/components/viz/shared/types.ts`
- Create: `src/components/viz/shared/useResizeObserver.ts`
- Create: `src/components/viz/shared/useD3.ts`
- Create: `src/components/viz/shared/colorScales.ts`

**Step 1: Create shared types**

Create `src/components/viz/shared/types.ts`:

```typescript
export interface Point2D {
  x: number;
  y: number;
  id: string;
}

export interface PersistenceInterval {
  birth: number;
  death: number;
  dimension: number;
}

export interface Simplex {
  vertices: string[];
  dimension: number;
  birthTime: number;
}

export interface DAGNode {
  id: string;
  label: string;
  status: string;
  domain: string;
}

export interface DAGEdge {
  source: string;
  target: string;
}
```

**Step 2: Create useResizeObserver hook**

Create `src/components/viz/shared/useResizeObserver.ts`:

```typescript
import { useRef, useState, useEffect } from 'react';

export function useResizeObserver<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) {
        setDimensions({
          width: entry.contentRect.width,
          height: entry.contentRect.height,
        });
      }
    });

    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  return { ref, ...dimensions };
}
```

**Step 3: Create useD3 hook**

Create `src/components/viz/shared/useD3.ts`:

```typescript
import { useRef, useEffect } from 'react';
import * as d3 from 'd3';

export function useD3<T extends SVGSVGElement>(
  renderFn: (svg: d3.Selection<T, unknown, null, undefined>) => void,
  deps: React.DependencyList,
) {
  const ref = useRef<T>(null);

  useEffect(() => {
    if (ref.current) {
      renderFn(d3.select(ref.current));
    }
  }, deps);

  return ref;
}
```

**Step 4: Create color scales**

Create `src/components/viz/shared/colorScales.ts`:

```typescript
import * as d3 from 'd3';

// Homological dimension colors: H0 = teal, H1 = purple, H2 = amber
export const dimensionColors = ['#0F6E56', '#534AB7', '#D97706'];

export const dimensionColorScale = d3
  .scaleOrdinal<number, string>()
  .domain([0, 1, 2])
  .range(dimensionColors);

export const domainColorScale = d3
  .scaleOrdinal<string, string>()
  .domain([
    'topology',
    'geometry',
    'probability',
    'optimization',
    'linear-algebra',
    'information-theory',
    'graph-theory',
    'category-theory',
  ])
  .range(d3.schemeTableau10);
```

**Step 5: Commit**

```bash
git add src/components/viz/shared/
git commit -m "feat: add shared viz infrastructure — types, hooks, color scales"
```

---

### Task 8: SimplicialComplex Component

**Files:**
- Create: `src/components/viz/SimplicialComplex.tsx`

This is the flagship component. A reader watches topology emerge from a point cloud as they drag a slider.

**Step 1: Build the component**

Create `src/components/viz/SimplicialComplex.tsx`:

```tsx
import { useState, useMemo, useCallback } from 'react';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { dimensionColors } from './shared/colorScales';
import type { Point2D, Simplex } from './shared/types';
import * as d3 from 'd3';

interface SimplicialComplexProps {
  points: Point2D[];
  epsilon: number;
  maxEpsilon?: number;
  showSlider?: boolean;
  highlightSimplices?: number[];
  colorScheme?: 'dimension' | 'birth-time';
  onEpsilonChange?: (eps: number) => void;
}

/** Compute pairwise distances between points. */
function pairwiseDistances(points: Point2D[]): Map<string, number> {
  const dists = new Map<string, number>();
  for (let i = 0; i < points.length; i++) {
    for (let j = i + 1; j < points.length; j++) {
      const dx = points[i].x - points[j].x;
      const dy = points[i].y - points[j].y;
      dists.set(`${points[i].id}-${points[j].id}`, Math.sqrt(dx * dx + dy * dy));
    }
  }
  return dists;
}

/** Compute the Vietoris-Rips complex at a given epsilon. */
function vietorisRips(points: Point2D[], epsilon: number, distances: Map<string, number>): Simplex[] {
  const simplices: Simplex[] = [];

  // 0-simplices (vertices) — always present
  for (const p of points) {
    simplices.push({ vertices: [p.id], dimension: 0, birthTime: 0 });
  }

  // 1-simplices (edges)
  for (let i = 0; i < points.length; i++) {
    for (let j = i + 1; j < points.length; j++) {
      const dist = distances.get(`${points[i].id}-${points[j].id}`) ?? Infinity;
      if (dist <= epsilon) {
        simplices.push({ vertices: [points[i].id, points[j].id], dimension: 1, birthTime: dist });
      }
    }
  }

  // 2-simplices (triangles) — three vertices all pairwise within epsilon
  for (let i = 0; i < points.length; i++) {
    for (let j = i + 1; j < points.length; j++) {
      for (let k = j + 1; k < points.length; k++) {
        const dij = distances.get(`${points[i].id}-${points[j].id}`) ?? Infinity;
        const dik = distances.get(`${points[i].id}-${points[k].id}`) ?? Infinity;
        const djk = distances.get(`${points[j].id}-${points[k].id}`) ?? Infinity;
        const maxDist = Math.max(dij, dik, djk);
        if (maxDist <= epsilon) {
          simplices.push({
            vertices: [points[i].id, points[j].id, points[k].id],
            dimension: 2,
            birthTime: maxDist,
          });
        }
      }
    }
  }

  return simplices;
}

export default function SimplicialComplex({
  points,
  epsilon: initialEpsilon,
  maxEpsilon = 2,
  showSlider = true,
  highlightSimplices,
  colorScheme = 'dimension',
  onEpsilonChange,
}: SimplicialComplexProps) {
  const [epsilon, setEpsilon] = useState(initialEpsilon);
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();

  const width = containerWidth || 600;
  const height = 400;
  const margin = { top: 20, right: 20, bottom: 20, left: 20 };

  const distances = useMemo(() => pairwiseDistances(points), [points]);
  const simplices = useMemo(() => vietorisRips(points, epsilon, distances), [points, epsilon, distances]);

  const handleEpsilonChange = useCallback(
    (value: number) => {
      setEpsilon(value);
      onEpsilonChange?.(value);
    },
    [onEpsilonChange],
  );

  const pointMap = useMemo(() => new Map(points.map((p) => [p.id, p])), [points]);

  const xExtent = d3.extent(points, (p) => p.x) as [number, number];
  const yExtent = d3.extent(points, (p) => p.y) as [number, number];
  const pad = maxEpsilon * 0.3;

  const xScale = d3
    .scaleLinear()
    .domain([xExtent[0] - pad, xExtent[1] + pad])
    .range([margin.left, width - margin.right]);

  const yScale = d3
    .scaleLinear()
    .domain([yExtent[0] - pad, yExtent[1] + pad])
    .range([height - margin.bottom, margin.top]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();

      const shouldHighlight = (dim: number) => !highlightSimplices || highlightSimplices.includes(dim);

      // Draw 2-simplices (triangles)
      const triangles = simplices.filter((s) => s.dimension === 2);
      svg
        .selectAll('.triangle')
        .data(triangles)
        .join('polygon')
        .attr('class', 'triangle')
        .attr('points', (s) =>
          s.vertices
            .map((id) => {
              const p = pointMap.get(id)!;
              return `${xScale(p.x)},${yScale(p.y)}`;
            })
            .join(' '),
        )
        .attr('fill', dimensionColors[2])
        .attr('fill-opacity', shouldHighlight(2) ? 0.15 : 0.03)
        .attr('stroke', dimensionColors[2])
        .attr('stroke-opacity', shouldHighlight(2) ? 0.3 : 0.05)
        .attr('stroke-width', 1);

      // Draw 1-simplices (edges)
      const edges = simplices.filter((s) => s.dimension === 1);
      svg
        .selectAll('.edge')
        .data(edges)
        .join('line')
        .attr('class', 'edge')
        .attr('x1', (s) => xScale(pointMap.get(s.vertices[0])!.x))
        .attr('y1', (s) => yScale(pointMap.get(s.vertices[0])!.y))
        .attr('x2', (s) => xScale(pointMap.get(s.vertices[1])!.x))
        .attr('y2', (s) => yScale(pointMap.get(s.vertices[1])!.y))
        .attr('stroke', dimensionColors[1])
        .attr('stroke-opacity', shouldHighlight(1) ? 0.6 : 0.1)
        .attr('stroke-width', 1.5);

      // Draw 0-simplices (vertices)
      svg
        .selectAll('.vertex')
        .data(points)
        .join('circle')
        .attr('class', 'vertex')
        .attr('cx', (p) => xScale(p.x))
        .attr('cy', (p) => yScale(p.y))
        .attr('r', 4)
        .attr('fill', dimensionColors[0])
        .attr('fill-opacity', shouldHighlight(0) ? 1 : 0.2);

      // Draw epsilon-balls (translucent circles showing radius)
      svg
        .selectAll('.eps-ball')
        .data(points)
        .join('circle')
        .attr('class', 'eps-ball')
        .attr('cx', (p) => xScale(p.x))
        .attr('cy', (p) => yScale(p.y))
        .attr('r', xScale(xExtent[0] + epsilon / 2) - xScale(xExtent[0]))
        .attr('fill', dimensionColors[0])
        .attr('fill-opacity', 0.04)
        .attr('stroke', dimensionColors[0])
        .attr('stroke-opacity', 0.1)
        .attr('stroke-width', 0.5);
    },
    [simplices, points, epsilon, width, height, highlightSimplices],
  );

  return (
    <div ref={containerRef} className="w-full">
      <svg ref={svgRef} width={width} height={height} className="rounded-lg border border-gray-200" />
      {showSlider && (
        <div className="mt-3 flex items-center gap-4">
          <label className="text-sm font-medium whitespace-nowrap" style={{ fontFamily: 'var(--font-sans)' }}>
            ε = {epsilon.toFixed(2)}
          </label>
          <input
            type="range"
            min={0}
            max={maxEpsilon}
            step={0.01}
            value={epsilon}
            onChange={(e) => handleEpsilonChange(parseFloat(e.target.value))}
            className="w-full accent-[var(--color-accent)]"
          />
        </div>
      )}
    </div>
  );
}
```

**Step 2: Create a test page to verify in isolation**

Create `src/pages/test-viz.astro`:

```astro
---
import BaseLayout from '../layouts/BaseLayout.astro';
import SimplicialComplex from '../components/viz/SimplicialComplex.tsx';

// Sample point cloud — a circle with some noise
const points = Array.from({ length: 12 }, (_, i) => {
  const angle = (2 * Math.PI * i) / 12;
  return {
    x: Math.cos(angle) + (Math.random() - 0.5) * 0.2,
    y: Math.sin(angle) + (Math.random() - 0.5) * 0.2,
    id: `p${i}`,
  };
});
---

<BaseLayout title="Viz Test">
  <div class="mx-auto max-w-3xl px-6 py-12">
    <h2 class="mb-6 text-2xl font-semibold font-[family-name:var(--font-sans)]">Simplicial Complex</h2>
    <SimplicialComplex client:visible points={points} epsilon={0.5} maxEpsilon={2.5} />
  </div>
</BaseLayout>
```

```bash
pnpm dev
# Visit http://localhost:4321/test-viz
# Drag the epsilon slider — edges and triangles should appear as epsilon grows
```

**Step 3: Commit**

```bash
git add src/components/viz/SimplicialComplex.tsx
git commit -m "feat: add SimplicialComplex component with Vietoris-Rips computation and epsilon slider"
```

---

### Task 9: PersistenceDiagram Component

**Files:**
- Create: `src/components/viz/PersistenceDiagram.tsx`

**Step 1: Build the component**

Create `src/components/viz/PersistenceDiagram.tsx`:

```tsx
import { useMemo } from 'react';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { dimensionColors } from './shared/colorScales';
import type { PersistenceInterval } from './shared/types';
import * as d3 from 'd3';

interface PersistenceDiagramProps {
  intervals: PersistenceInterval[];
  currentEpsilon?: number;
  showDiagonal?: boolean;
  highlightDimension?: number | null;
  mode?: 'diagram' | 'barcode';
}

export default function PersistenceDiagram({
  intervals,
  currentEpsilon,
  showDiagonal = true,
  highlightDimension = null,
  mode = 'diagram',
}: PersistenceDiagramProps) {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const width = containerWidth || 400;
  const height = mode === 'barcode' ? Math.max(200, intervals.length * 16 + 60) : 400;
  const margin = { top: 30, right: 30, bottom: 50, left: 50 };

  const maxVal = useMemo(() => {
    const deaths = intervals.map((i) => (i.death === Infinity ? 0 : i.death));
    const births = intervals.map((i) => i.birth);
    return Math.max(...deaths, ...births) * 1.15 || 1;
  }, [intervals]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();

      if (mode === 'diagram') {
        renderDiagram(svg);
      } else {
        renderBarcode(svg);
      }
    },
    [intervals, currentEpsilon, highlightDimension, mode, width, height, maxVal],
  );

  function renderDiagram(svg: d3.Selection<SVGSVGElement, unknown, null, undefined>) {
    const xScale = d3.scaleLinear().domain([0, maxVal]).range([margin.left, width - margin.right]);
    const yScale = d3.scaleLinear().domain([0, maxVal]).range([height - margin.bottom, margin.top]);

    // Diagonal line
    if (showDiagonal) {
      svg
        .append('line')
        .attr('x1', xScale(0))
        .attr('y1', yScale(0))
        .attr('x2', xScale(maxVal))
        .attr('y2', yScale(maxVal))
        .attr('stroke', '#ccc')
        .attr('stroke-dasharray', '4,4');
    }

    // Axes
    svg
      .append('g')
      .attr('transform', `translate(0,${height - margin.bottom})`)
      .call(d3.axisBottom(xScale).ticks(5))
      .selectAll('text')
      .style('font-size', '11px');

    svg
      .append('g')
      .attr('transform', `translate(${margin.left},0)`)
      .call(d3.axisLeft(yScale).ticks(5))
      .selectAll('text')
      .style('font-size', '11px');

    // Axis labels
    svg
      .append('text')
      .attr('x', width / 2)
      .attr('y', height - 8)
      .attr('text-anchor', 'middle')
      .style('font-size', '12px')
      .style('font-family', 'var(--font-sans)')
      .text('Birth');

    svg
      .append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -height / 2)
      .attr('y', 14)
      .attr('text-anchor', 'middle')
      .style('font-size', '12px')
      .style('font-family', 'var(--font-sans)')
      .text('Death');

    // Current epsilon sweep line
    if (currentEpsilon !== undefined) {
      svg
        .append('line')
        .attr('x1', xScale(currentEpsilon))
        .attr('y1', yScale(0))
        .attr('x2', xScale(currentEpsilon))
        .attr('y2', yScale(maxVal))
        .attr('stroke', '#0F6E56')
        .attr('stroke-width', 1.5)
        .attr('stroke-opacity', 0.5)
        .attr('stroke-dasharray', '6,3');

      svg
        .append('text')
        .attr('x', xScale(currentEpsilon))
        .attr('y', margin.top - 8)
        .attr('text-anchor', 'middle')
        .style('font-size', '11px')
        .style('font-family', 'var(--font-sans)')
        .style('fill', '#0F6E56')
        .text(`ε = ${currentEpsilon.toFixed(2)}`);
    }

    // Points
    const finiteIntervals = intervals.filter((i) => i.death !== Infinity);
    svg
      .selectAll('.pd-point')
      .data(finiteIntervals)
      .join('circle')
      .attr('class', 'pd-point')
      .attr('cx', (d) => xScale(d.birth))
      .attr('cy', (d) => yScale(d.death))
      .attr('r', 5)
      .attr('fill', (d) => dimensionColors[d.dimension] ?? dimensionColors[0])
      .attr('fill-opacity', (d) =>
        highlightDimension === null || highlightDimension === d.dimension ? 0.8 : 0.15,
      )
      .attr('stroke', (d) => dimensionColors[d.dimension] ?? dimensionColors[0])
      .attr('stroke-width', 1);

    // Infinite intervals — draw as triangles at top
    const infiniteIntervals = intervals.filter((i) => i.death === Infinity);
    svg
      .selectAll('.pd-inf')
      .data(infiniteIntervals)
      .join('path')
      .attr('class', 'pd-inf')
      .attr('d', (d) => {
        const x = xScale(d.birth);
        const y = margin.top + 5;
        return `M${x},${y - 5}L${x + 5},${y + 5}L${x - 5},${y + 5}Z`;
      })
      .attr('fill', (d) => dimensionColors[d.dimension] ?? dimensionColors[0])
      .attr('fill-opacity', 0.8);

    // Legend
    const dims = [...new Set(intervals.map((i) => i.dimension))].sort();
    const legend = svg
      .append('g')
      .attr('transform', `translate(${width - margin.right - 80}, ${margin.top})`);

    dims.forEach((dim, i) => {
      const g = legend.append('g').attr('transform', `translate(0, ${i * 20})`);
      g.append('circle').attr('r', 4).attr('cx', 0).attr('cy', 0).attr('fill', dimensionColors[dim]);
      g.append('text')
        .attr('x', 10)
        .attr('y', 4)
        .style('font-size', '11px')
        .style('font-family', 'var(--font-sans)')
        .text(`H${dim}`);
    });
  }

  function renderBarcode(svg: d3.Selection<SVGSVGElement, unknown, null, undefined>) {
    const xScale = d3.scaleLinear().domain([0, maxVal]).range([margin.left, width - margin.right]);
    const barHeight = 10;
    const gap = 4;

    // Sort by dimension, then by birth
    const sorted = [...intervals].sort((a, b) => a.dimension - b.dimension || a.birth - b.birth);

    svg
      .append('g')
      .attr('transform', `translate(0,${height - margin.bottom})`)
      .call(d3.axisBottom(xScale).ticks(5));

    // Bars
    sorted.forEach((interval, i) => {
      const y = margin.top + i * (barHeight + gap);
      const deathVal = interval.death === Infinity ? maxVal : interval.death;

      svg
        .append('rect')
        .attr('x', xScale(interval.birth))
        .attr('y', y)
        .attr('width', xScale(deathVal) - xScale(interval.birth))
        .attr('height', barHeight)
        .attr('fill', dimensionColors[interval.dimension] ?? dimensionColors[0])
        .attr('fill-opacity', highlightDimension === null || highlightDimension === interval.dimension ? 0.7 : 0.15)
        .attr('rx', 2);
    });

    // Sweep line
    if (currentEpsilon !== undefined) {
      svg
        .append('line')
        .attr('x1', xScale(currentEpsilon))
        .attr('y1', margin.top - 5)
        .attr('x2', xScale(currentEpsilon))
        .attr('y2', height - margin.bottom)
        .attr('stroke', '#0F6E56')
        .attr('stroke-width', 1.5)
        .attr('stroke-dasharray', '6,3');
    }
  }

  return (
    <div ref={containerRef} className="w-full">
      <svg ref={svgRef} width={width} height={height} className="rounded-lg border border-gray-200" />
    </div>
  );
}
```

**Step 2: Test linked pair on the test page**

Update `src/pages/test-viz.astro` to show both components linked:

```astro
---
import BaseLayout from '../layouts/BaseLayout.astro';
import SimplicialComplex from '../components/viz/SimplicialComplex.tsx';
import PersistenceDiagram from '../components/viz/PersistenceDiagram.tsx';
import LinkedVizDemo from '../components/viz/LinkedVizDemo.tsx';
---

<BaseLayout title="Viz Test">
  <div class="mx-auto max-w-4xl px-6 py-12">
    <h2 class="mb-6 text-2xl font-semibold font-[family-name:var(--font-sans)]">Linked Filtration Demo</h2>
    <LinkedVizDemo client:visible />
  </div>
</BaseLayout>
```

For this to work, create a `LinkedVizDemo.tsx` wrapper that manages shared state:

Create `src/components/viz/LinkedVizDemo.tsx`:

```tsx
import { useState, useMemo } from 'react';
import SimplicialComplex from './SimplicialComplex';
import PersistenceDiagram from './PersistenceDiagram';
import type { Point2D, PersistenceInterval } from './shared/types';

/** Compute persistence intervals from a point set via incremental union-find for H0. */
function computePersistence(points: Point2D[]): PersistenceInterval[] {
  // Compute all pairwise edges sorted by distance
  const edges: { i: number; j: number; dist: number }[] = [];
  for (let i = 0; i < points.length; i++) {
    for (let j = i + 1; j < points.length; j++) {
      const dx = points[i].x - points[j].x;
      const dy = points[i].y - points[j].y;
      edges.push({ i, j, dist: Math.sqrt(dx * dx + dy * dy) });
    }
  }
  edges.sort((a, b) => a.dist - b.dist);

  // Union-Find for H0
  const parent = points.map((_, i) => i);
  const rank = points.map(() => 0);

  function find(x: number): number {
    if (parent[x] !== x) parent[x] = find(parent[x]);
    return parent[x];
  }

  function union(x: number, y: number): boolean {
    const px = find(x);
    const py = find(y);
    if (px === py) return false;
    if (rank[px] < rank[py]) parent[px] = py;
    else if (rank[px] > rank[py]) parent[py] = px;
    else {
      parent[py] = px;
      rank[px]++;
    }
    return true;
  }

  const intervals: PersistenceInterval[] = [];

  // All components born at 0
  const birthTimes = points.map(() => 0);

  for (const edge of edges) {
    const pi = find(edge.i);
    const pj = find(edge.j);
    if (pi !== pj) {
      // Merge: the younger component dies
      const younger = birthTimes[pi] >= birthTimes[pj] ? pi : pj;
      if (edge.dist > 0) {
        intervals.push({ birth: birthTimes[younger], death: edge.dist, dimension: 0 });
      }
      union(edge.i, edge.j);
    }
  }

  // One component survives to infinity
  intervals.push({ birth: 0, death: Infinity, dimension: 0 });

  // H1 detection (approximate): find "short" cycles by looking for edges
  // that DON'T merge components (they close a loop)
  const parent2 = points.map((_, i) => i);
  const rank2 = points.map(() => 0);

  function find2(x: number): number {
    if (parent2[x] !== x) parent2[x] = find2(parent2[x]);
    return parent2[x];
  }

  function union2(x: number, y: number): boolean {
    const px = find2(x);
    const py = find2(y);
    if (px === py) return false;
    if (rank2[px] < rank2[py]) parent2[px] = py;
    else if (rank2[px] > rank2[py]) parent2[py] = px;
    else { parent2[py] = px; rank2[px]++; }
    return true;
  }

  // Simple H1 heuristic: track cycle-creating edges
  const cycleEdges: number[] = [];
  for (const edge of edges) {
    if (!union2(edge.i, edge.j)) {
      cycleEdges.push(edge.dist);
    }
  }

  // Report the first few cycles as H1 features (heuristic death = next edge distance * 1.5)
  for (let i = 0; i < Math.min(cycleEdges.length, 3); i++) {
    intervals.push({
      birth: cycleEdges[i],
      death: cycleEdges[i] * 1.8 + 0.2,
      dimension: 1,
    });
  }

  return intervals;
}

// Generate a noisy circle
const defaultPoints: Point2D[] = Array.from({ length: 15 }, (_, i) => {
  const angle = (2 * Math.PI * i) / 15;
  const r = 1 + (Math.sin(i * 7) * 0.15); // deterministic "noise"
  return { x: r * Math.cos(angle), y: r * Math.sin(angle), id: `p${i}` };
});

export default function LinkedVizDemo() {
  const [epsilon, setEpsilon] = useState(0.3);
  const intervals = useMemo(() => computePersistence(defaultPoints), []);

  return (
    <div className="space-y-8">
      <SimplicialComplex
        points={defaultPoints}
        epsilon={epsilon}
        maxEpsilon={3}
        onEpsilonChange={setEpsilon}
      />
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h3 className="mb-2 text-sm font-medium" style={{ fontFamily: 'var(--font-sans)' }}>
            Persistence Diagram
          </h3>
          <PersistenceDiagram intervals={intervals} currentEpsilon={epsilon} mode="diagram" />
        </div>
        <div>
          <h3 className="mb-2 text-sm font-medium" style={{ fontFamily: 'var(--font-sans)' }}>
            Persistence Barcode
          </h3>
          <PersistenceDiagram intervals={intervals} currentEpsilon={epsilon} mode="barcode" />
        </div>
      </div>
    </div>
  );
}
```

```bash
pnpm dev
# Visit http://localhost:4321/test-viz
# Drag the epsilon slider on the simplicial complex — the sweep line on both
# the persistence diagram and barcode should move in sync
```

**Step 3: Commit**

```bash
git add src/components/viz/PersistenceDiagram.tsx src/components/viz/LinkedVizDemo.tsx
git commit -m "feat: add PersistenceDiagram + LinkedVizDemo — flagship linked filtration pair"
```

---

### Task 10: DAGGraph Component

**Files:**
- Create: `src/components/viz/DAGGraph.tsx`
- Create: `src/lib/prerequisiteGraph.ts`

**Step 1: Build the DAGGraph**

Create `src/components/viz/DAGGraph.tsx`:

```tsx
import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { domainColorScale } from './shared/colorScales';
import type { DAGNode, DAGEdge } from './shared/types';

interface DAGGraphProps {
  nodes: DAGNode[];
  edges: DAGEdge[];
  highlightNode?: string;
  onNodeClick?: (id: string) => void;
  layout?: 'force' | 'layered';
}

export default function DAGGraph({
  nodes,
  edges,
  highlightNode,
  onNodeClick,
  layout = 'force',
}: DAGGraphProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || nodes.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const width = svgRef.current.clientWidth || 600;
    const height = 400;

    svg.attr('viewBox', `0 0 ${width} ${height}`);

    // Create simulation
    const simNodes = nodes.map((n) => ({ ...n })) as (DAGNode & d3.SimulationNodeDatum)[];
    const simEdges = edges.map((e) => ({ ...e })) as (DAGEdge & d3.SimulationLinkDatum<DAGNode & d3.SimulationNodeDatum>)[];

    const simulation = d3
      .forceSimulation(simNodes)
      .force('link', d3.forceLink(simEdges).id((d: any) => d.id).distance(100))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('y', d3.forceY(height / 2).strength(0.05));

    // Arrow markers
    svg
      .append('defs')
      .append('marker')
      .attr('id', 'arrowhead')
      .attr('viewBox', '-0 -5 10 10')
      .attr('refX', 20)
      .attr('refY', 0)
      .attr('orient', 'auto')
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .append('path')
      .attr('d', 'M 0,-5 L 10,0 L 0,5')
      .attr('fill', '#999');

    // Edges
    const link = svg
      .append('g')
      .selectAll('line')
      .data(simEdges)
      .join('line')
      .attr('stroke', '#ccc')
      .attr('stroke-width', 1.5)
      .attr('marker-end', 'url(#arrowhead)');

    // Nodes
    const node = svg
      .append('g')
      .selectAll('g')
      .data(simNodes)
      .join('g')
      .style('cursor', onNodeClick ? 'pointer' : 'default')
      .on('click', (_, d) => onNodeClick?.(d.id));

    node
      .append('circle')
      .attr('r', (d) => (d.id === highlightNode ? 12 : 8))
      .attr('fill', (d) => domainColorScale(d.domain))
      .attr('stroke', (d) => (d.id === highlightNode ? '#1A1A1A' : 'white'))
      .attr('stroke-width', (d) => (d.id === highlightNode ? 3 : 2))
      .attr('opacity', (d) => (d.status === 'draft' ? 0.4 : 1));

    node
      .append('text')
      .text((d) => d.label)
      .attr('dy', -14)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('font-family', 'var(--font-sans)')
      .style('fill', 'var(--color-text)')
      .style('pointer-events', 'none');

    // Drag behavior
    const drag = d3
      .drag<SVGGElement, DAGNode & d3.SimulationNodeDatum>()
      .on('start', (event, d) => {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      })
      .on('drag', (event, d) => {
        d.fx = event.x;
        d.fy = event.y;
      })
      .on('end', (event, d) => {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
      });

    node.call(drag);

    simulation.on('tick', () => {
      link
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y);

      node.attr('transform', (d: any) => `translate(${d.x},${d.y})`);
    });

    return () => simulation.stop();
  }, [nodes, edges, highlightNode, onNodeClick, layout]);

  return <svg ref={svgRef} className="w-full rounded-lg" style={{ minHeight: 400 }} />;
}
```

**Step 2: Create prerequisiteGraph utility**

Create `src/lib/prerequisiteGraph.ts`:

```typescript
import type { DAGNode, DAGEdge } from '../components/viz/shared/types';

interface TopicEntry {
  id: string;
  data: {
    title: string;
    status: string;
    domain: string;
    prerequisites: string[];
  };
}

export function buildPrerequisiteGraph(topics: TopicEntry[]): { nodes: DAGNode[]; edges: DAGEdge[] } {
  const nodes: DAGNode[] = topics.map((t) => ({
    id: t.id,
    label: t.data.title,
    status: t.data.status,
    domain: t.data.domain,
  }));

  const edges: DAGEdge[] = [];
  for (const topic of topics) {
    for (const prereq of topic.data.prerequisites) {
      edges.push({ source: prereq, target: topic.id });
    }
  }

  return { nodes, edges };
}
```

**Step 3: Commit**

```bash
git add src/components/viz/DAGGraph.tsx src/lib/prerequisiteGraph.ts
git commit -m "feat: add DAGGraph component and prerequisite graph builder"
```

---

## Phase 4: Pages

### Task 11: Landing Page

**Files:**
- Modify: `src/pages/index.astro`
- Create: `src/components/ui/TopicCard.astro`

**Step 1: Build TopicCard**

Create `src/components/ui/TopicCard.astro`:

```astro
---
import DifficultyBadge from './DifficultyBadge.astro';

interface Props {
  id: string;
  title: string;
  subtitle?: string;
  abstract: string;
  difficulty: 'foundational' | 'intermediate' | 'advanced';
  domain: string;
  prerequisiteCount: number;
}

const { id, title, subtitle, abstract, difficulty, domain, prerequisiteCount } = Astro.props;
---

<a
  href={`/topics/${id}`}
  class="group block rounded-xl border border-gray-200/80 bg-white p-6 transition-all hover:border-[var(--color-accent)]/30 hover:shadow-sm"
>
  <div class="mb-3 flex items-center gap-2">
    <DifficultyBadge difficulty={difficulty} />
    <span class="text-xs uppercase tracking-wider text-[var(--color-text-secondary)]">{domain}</span>
  </div>
  <h3 class="text-lg font-semibold font-[family-name:var(--font-sans)] group-hover:text-[var(--color-accent)] transition-colors">
    {title}
  </h3>
  {subtitle && <p class="mt-1 text-sm text-[var(--color-text-secondary)]">{subtitle}</p>}
  <p class="mt-3 text-sm leading-relaxed text-[var(--color-text-secondary)] line-clamp-3">
    {abstract}
  </p>
  {prerequisiteCount > 0 && (
    <p class="mt-3 text-xs text-[var(--color-text-secondary)]">
      {prerequisiteCount} prerequisite{prerequisiteCount > 1 ? 's' : ''}
    </p>
  )}
</a>
```

**Step 2: Build landing page**

Update `src/pages/index.astro`:

```astro
---
import BaseLayout from '../layouts/BaseLayout.astro';
import TopicCard from '../components/ui/TopicCard.astro';
import DAGGraph from '../components/viz/DAGGraph.tsx';
import { getCollection } from 'astro:content';
import { buildPrerequisiteGraph } from '../lib/prerequisiteGraph';

const allTopics = await getCollection('topics');
const publishedTopics = allTopics.filter((t) => t.data.status === 'published');
const latestTopics = publishedTopics
  .sort((a, b) => (b.data.datePublished?.getTime() ?? 0) - (a.data.datePublished?.getTime() ?? 0))
  .slice(0, 3);

const { nodes, edges } = buildPrerequisiteGraph(allTopics);
---

<BaseLayout title="Home">
  <!-- Hero -->
  <section class="mx-auto max-w-3xl px-6 pt-24 pb-16 text-center">
    <h1 class="text-5xl font-semibold font-[family-name:var(--font-sans)] tracking-tight">
      formal<span class="text-[var(--color-accent)]">ML</span>
    </h1>
    <p class="mt-6 text-xl text-[var(--color-text-secondary)] leading-relaxed">
      The mathematical machinery behind modern machine learning
    </p>
    <p class="mt-4 text-base text-[var(--color-text-secondary)] max-w-xl mx-auto leading-relaxed">
      Deep-dive explainers combining rigorous mathematics, interactive visualizations,
      and working code. Built for practitioners, graduate students, and researchers.
    </p>
  </section>

  <!-- Prerequisite Graph -->
  {nodes.length > 0 && (
    <section class="mx-auto max-w-4xl px-6 pb-16">
      <DAGGraph
        client:visible
        nodes={nodes}
        edges={edges}
        onNodeClick={(id) => window.location.href = `/topics/${id}`}
      />
    </section>
  )}

  <!-- Latest Topics -->
  {latestTopics.length > 0 && (
    <section class="mx-auto max-w-4xl px-6 pb-24">
      <h2 class="mb-8 text-sm font-medium font-[family-name:var(--font-sans)] uppercase tracking-wider text-[var(--color-text-secondary)]">
        Latest Topics
      </h2>
      <div class="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
        {latestTopics.map((topic) => (
          <TopicCard
            id={topic.id}
            title={topic.data.title}
            subtitle={topic.data.subtitle}
            abstract={topic.data.abstract}
            difficulty={topic.data.difficulty}
            domain={topic.data.domain}
            prerequisiteCount={topic.data.prerequisites.length}
          />
        ))}
      </div>
    </section>
  )}
</BaseLayout>
```

**Step 3: Commit**

```bash
git add src/pages/index.astro src/components/ui/TopicCard.astro
git commit -m "feat: build landing page with hero, DAG graph, latest topics grid"
```

---

### Task 12: Topic Index Page

**Files:**
- Create: `src/pages/topics/index.astro`

**Step 1: Build topic index**

Create `src/pages/topics/index.astro`:

```astro
---
import BaseLayout from '../../layouts/BaseLayout.astro';
import TopicCard from '../../components/ui/TopicCard.astro';
import { getCollection } from 'astro:content';

const allTopics = await getCollection('topics');
const publishedTopics = allTopics
  .filter((t) => t.data.status === 'published')
  .sort((a, b) => a.data.title.localeCompare(b.data.title));

const domains = [...new Set(publishedTopics.map((t) => t.data.domain))].sort();
---

<BaseLayout title="Topics" description="Browse all ML mathematics topics">
  <section class="mx-auto max-w-4xl px-6 py-12">
    <h1 class="text-3xl font-semibold font-[family-name:var(--font-sans)] mb-2">Topics</h1>
    <p class="text-[var(--color-text-secondary)] mb-10">
      Browse all published topics in advanced ML mathematics.
    </p>

    <!-- Domain filters (static, no JS needed) -->
    <div class="mb-8 flex flex-wrap gap-2">
      <a href="/topics" class="rounded-full border px-3 py-1 text-xs font-[family-name:var(--font-sans)] hover:bg-gray-50">
        All
      </a>
      {domains.map((domain) => (
        <span class="rounded-full border px-3 py-1 text-xs font-[family-name:var(--font-sans)] capitalize text-[var(--color-text-secondary)]">
          {domain}
        </span>
      ))}
    </div>

    {publishedTopics.length === 0 ? (
      <p class="text-center text-[var(--color-text-secondary)] py-12">
        Topics coming soon. Check back shortly.
      </p>
    ) : (
      <div class="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
        {publishedTopics.map((topic) => (
          <TopicCard
            id={topic.id}
            title={topic.data.title}
            subtitle={topic.data.subtitle}
            abstract={topic.data.abstract}
            difficulty={topic.data.difficulty}
            domain={topic.data.domain}
            prerequisiteCount={topic.data.prerequisites.length}
          />
        ))}
      </div>
    )}
  </section>
</BaseLayout>
```

**Step 2: Commit**

```bash
git add src/pages/topics/index.astro
git commit -m "feat: add topic index page with domain filters and topic card grid"
```

---

### Task 13: Dynamic Topic Pages + About Page

**Files:**
- Create: `src/pages/topics/[...slug].astro`
- Create: `src/pages/about.astro`

**Step 1: Create dynamic topic route**

Create `src/pages/topics/[...slug].astro`:

```astro
---
import TopicLayout from '../../layouts/TopicLayout.astro';
import { getCollection, render } from 'astro:content';

export async function getStaticPaths() {
  const topics = await getCollection('topics');
  return topics.map((topic) => ({
    params: { slug: topic.id },
    props: { topic },
  }));
}

const { topic } = Astro.props;
const { Content } = await render(topic);
---

<TopicLayout frontmatter={topic.data}>
  <Content />
</TopicLayout>
```

**Step 2: Create about page**

Create `src/pages/about.astro`:

```astro
---
import BaseLayout from '../layouts/BaseLayout.astro';
---

<BaseLayout title="About" description="About formalML and its author">
  <article class="mx-auto max-w-3xl px-6 py-12">
    <h1 class="text-3xl font-semibold font-[family-name:var(--font-sans)] mb-8">About</h1>

    <div class="prose">
      <h2>Mission</h2>
      <p>
        formalML is a curated collection of deep-dive explainers on the mathematical machinery
        behind modern machine learning. Every topic receives a three-pillar treatment: rigorous
        mathematical exposition, interactive visual intuition, and working code you can run
        immediately.
      </p>
      <p>
        The site exists because the gap between textbook formalism and practical ML understanding
        is wider than it needs to be. We believe that interactive visualization — watching a
        filtration sweep across a point cloud, seeing eigenvalues shift as you perturb a matrix —
        builds the kind of geometric intuition that no amount of static notation can provide.
      </p>

      <h2>Author</h2>
      <p>
        <strong>Jonathan Rocha</strong> is a data scientist and researcher with a background spanning
        mathematics, data science, and the humanities. He holds an MS in Data Science from SMU, an MA
        in English from Texas A&M University, and a BA in History from Texas A&M. His research
        interests include time-series data mining and topology-aware deep learning, with a PhD
        trajectory at UTRGV.
      </p>

      <h2>DataSalt</h2>
      <p>
        formalml.com is an independent educational project by the founder of
        <a href="https://datasalt.ai" class="text-[var(--color-accent)] hover:underline">DataSalt LLC</a>.
      </p>

      <h2>Links</h2>
      <ul>
        <li><a href="https://github.com/jonx0037" class="text-[var(--color-accent)]">GitHub (@jonx0037)</a></li>
        <li><a href="https://datasalt.ai" class="text-[var(--color-accent)]">DataSalt</a></li>
        <li><a href="https://github.com/jonx0037/formalml" class="text-[var(--color-accent)]">formalML Repository</a></li>
      </ul>
    </div>
  </article>
</BaseLayout>
```

**Step 3: Commit**

```bash
git add src/pages/topics/\\[...slug\\].astro src/pages/about.astro
git commit -m "feat: add dynamic topic pages route and about page"
```

---

## Phase 5: Day-One Content

### Task 14: Simplicial Complexes Topic (Prerequisite Entry)

**Files:**
- Create: `src/content/topics/simplicial-complexes.mdx`

**Step 1: Write the lighter prerequisite topic**

Create `src/content/topics/simplicial-complexes.mdx` with full three-pillar content (lighter treatment). This is a substantial MDX file — see brief Section 9 item 4 for scope. The content should include:

- Frontmatter with `status: 'published'`, `difficulty: 'foundational'`, `domain: 'topology'`
- Overview section: what simplicial complexes are and why ML practitioners should care
- Formal Framework: definitions for simplex, simplicial complex, abstract vs geometric, Vietoris-Rips construction (using TheoremBlock components)
- Visual Intuition: embedded `<SimplicialComplex client:visible ... />` with a static point set
- Working Code: Python snippet showing `gudhi` or manual construction
- References

**USER CONTRIBUTION POINT:** The mathematical content for this topic (formal definitions, theorem statements, proof sketches) is domain-expert work. The implementer should write the structural MDX with TheoremBlock placeholders, and Jonathan should review/fill in the precise mathematical statements.

**Step 2: Verify the topic renders**

```bash
pnpm dev
# Visit http://localhost:4321/topics/simplicial-complexes
# Verify: TheoremBlock renders with teal/purple borders, viz component hydrates, KaTeX math renders
```

**Step 3: Commit**

```bash
git add src/content/topics/simplicial-complexes.mdx
git commit -m "feat: add simplicial complexes topic — foundational prerequisite entry"
```

---

### Task 15: Persistent Homology Topic (Full Three-Pillar)

**Files:**
- Create: `src/content/topics/persistent-homology.mdx`

**Step 1: Write the flagship topic**

Create `src/content/topics/persistent-homology.mdx` — the complete three-pillar treatment per brief Section 9 item 3:

- Frontmatter: `status: 'published'`, `difficulty: 'intermediate'`, `domain: 'topology'`, `prerequisites: ['simplicial-complexes']`
- Overview + Motivation: why TDA matters for ML (no formulas)
- Formal Framework: filtrations, chain complexes, boundary operators, homology groups, persistence modules, stability theorem — all using TheoremBlock
- Visual Intuition: `<LinkedVizDemo client:visible />` (or inline SimplicialComplex + PersistenceDiagram)
- Working Code: Python with `ripser`, `persim`, inline code blocks
- Connections: links to simplicial complexes, metric spaces
- References: Edelsbrunner & Harer, Carlsson survey, Ghrist

**USER CONTRIBUTION POINT:** Same as Task 14 — mathematical content is domain-expert territory. Build the structural MDX, mark sections for Jonathan's review.

**Step 2: Verify the topic renders with linked viz**

```bash
pnpm dev
# Visit http://localhost:4321/topics/persistent-homology
# Verify: all three pillars render, linked viz works, prereq chip links to simplicial-complexes
```

**Step 3: Commit**

```bash
git add src/content/topics/persistent-homology.mdx
git commit -m "feat: add persistent homology topic — full three-pillar treatment with linked viz"
```

---

### Task 16: Sample Data Files

**Files:**
- Create: `src/data/sampleData/circle-filtration.json`

**Step 1: Generate deterministic sample data**

Create `src/data/sampleData/circle-filtration.json`:

```json
{
  "points": [
    { "x": 1.0, "y": 0.0, "id": "p0" },
    { "x": 0.866, "y": 0.5, "id": "p1" },
    { "x": 0.5, "y": 0.866, "id": "p2" },
    { "x": 0.0, "y": 1.0, "id": "p3" },
    { "x": -0.5, "y": 0.866, "id": "p4" },
    { "x": -0.866, "y": 0.5, "id": "p5" },
    { "x": -1.0, "y": 0.0, "id": "p6" },
    { "x": -0.866, "y": -0.5, "id": "p7" },
    { "x": -0.5, "y": -0.866, "id": "p8" },
    { "x": 0.0, "y": -1.0, "id": "p9" },
    { "x": 0.5, "y": -0.866, "id": "p10" },
    { "x": 0.866, "y": -0.5, "id": "p11" }
  ],
  "description": "12 points sampled uniformly from the unit circle — demonstrates H0 merging and H1 loop detection"
}
```

**Step 2: Commit**

```bash
git add src/data/sampleData/circle-filtration.json
git commit -m "feat: add circle filtration sample data for viz demos"
```

---

## Phase 6: Search, Polish, Deploy

### Task 17: Pagefind Integration

**Files:**
- Modify: `astro.config.mjs`
- Modify: `src/components/ui/Nav.astro`

**Step 1: Install Pagefind**

```bash
pnpm add -D pagefind
```

Add a postbuild script in `package.json`:

```json
"scripts": {
  "postbuild": "pagefind --site dist"
}
```

**Step 2: Add Pagefind UI to the site**

Update the search button in Nav to load Pagefind's built-in UI, or create a minimal search dialog. The simplest approach: add a `<div id="search"></div>` and Pagefind's JS in BaseLayout.

Add to `BaseLayout.astro` before closing `</body>`:

```html
<link href="/pagefind/pagefind-ui.css" rel="stylesheet" />
<script src="/pagefind/pagefind-ui.js" is:inline></script>
<script is:inline>
  document.getElementById('search-trigger')?.addEventListener('click', () => {
    document.getElementById('search-dialog')?.classList.toggle('hidden');
  });
</script>
```

Add search dialog markup after Nav in BaseLayout.

**Step 3: Build and verify search index is generated**

```bash
pnpm build
# Should see Pagefind indexing output
```

**Step 4: Commit**

```bash
git add package.json astro.config.mjs src/layouts/BaseLayout.astro src/components/ui/Nav.astro
git commit -m "feat: integrate Pagefind for static search"
```

---

### Task 18: Vercel Deployment Config

**Files:**
- Create: `vercel.json`

**Step 1: Create Vercel config**

Create `vercel.json`:

```json
{
  "framework": "astro",
  "buildCommand": "pnpm build",
  "outputDirectory": "dist",
  "installCommand": "pnpm install"
}
```

**Step 2: Commit**

```bash
git add vercel.json
git commit -m "feat: add Vercel deployment config"
```

---

### Task 19: Clean Up Test Pages and Final Build

**Step 1: Remove test pages**

```bash
rm -f src/pages/test-viz.astro src/pages/test-theorem.astro
rm -f src/content/topics/test-topic.mdx
```

**Step 2: Full build validation**

```bash
pnpm build
```

Expected: Clean build with zero errors. Pagefind indexes all published topics.

**Step 3: Final commit**

```bash
git add -A
git commit -m "chore: clean up test pages, verify production build"
```

---

## Summary

| Phase | Tasks | What ships |
|-------|-------|-----------|
| 1. Scaffold | 1-2 | Astro + React + Tailwind + KaTeX + D3 + content schema |
| 2. Layouts & UI | 3-6 | BaseLayout, Nav, Footer, TheoremBlock, TopicLayout |
| 3. Flagship Viz | 7-10 | SimplicialComplex, PersistenceDiagram, LinkedVizDemo, DAGGraph |
| 4. Pages | 11-13 | Landing, topic index, dynamic topic route, about |
| 5. Content | 14-16 | Simplicial complexes + persistent homology topics + sample data |
| 6. Deploy | 17-19 | Pagefind search, Vercel config, clean build |

**User contribution points:** Tasks 14-15 mark mathematical content (theorem statements, proof sketches, formal definitions) as domain-expert contributions where Jonathan's input shapes the educational quality.
