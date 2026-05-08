// @ts-check
import { defineConfig } from 'astro/config';
import react from '@astrojs/react';
import mdx from '@astrojs/mdx';
import tailwindcss from '@tailwindcss/vite';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeExternalLinks from 'rehype-external-links';

// https://astro.build/config
export default defineConfig({
  site: 'https://formalml.com',
  integrations: [react(), mdx()],
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [
      rehypeKatex,
      [rehypeExternalLinks, { target: '_blank', rel: ['noopener', 'noreferrer'] }],
    ],
    shikiConfig: {
      themes: { light: 'github-light', dark: 'github-dark' },
    },
  },
  vite: {
    plugins: [tailwindcss()],
    build: {
      // Disable minification on Vercel to reduce build-time memory pressure.
      // The ~30 MB extra HTML/JS is acceptable for a static math-heavy site;
      // gzip/brotli compression still applies at the CDN. See PR #73 for
      // why this matters: rehype-katex on the BNN page consumes >6 GB of
      // V8 heap during build, leaving no room for esbuild/terser minification
      // workers in the 8 GB Vercel container.
      minify: process.env.VERCEL ? false : true,
    },
  },
});
