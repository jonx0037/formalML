// @ts-check
import { defineConfig } from 'astro/config';
import react from '@astrojs/react';
import mdx from '@astrojs/mdx';
import tailwindcss from '@tailwindcss/vite';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeExternalLinks from 'rehype-external-links';

// MDX's JSX path turns a GFM table cell's `align` into `style="text-align:…"`, but `optimize`
// serializes static subtrees as plain HTML and would keep the obsolete `align` attribute.
// Converting it up front keeps the optimized output identical to the JSX output.
function rehypeTableCellAlignToStyle() {
  /** @param {any} node */
  const walk = (node) => {
    const props = node.properties;
    if ((node.tagName === 'td' || node.tagName === 'th') && props?.align) {
      props.style = `text-align:${props.align}` + (props.style ? `;${props.style}` : '');
      delete props.align;
    }
    node.children?.forEach(walk);
  };
  return walk;
}

// https://astro.build/config
export default defineConfig({
  site: 'https://formalml.com',
  // optimize: collapse static MDX subtrees (KaTeX output above all) into HTML strings.
  // Without it every KaTeX span becomes a JSX call and the SSR bundle peaked at ~19.0 GB RSS;
  // with it the build peaks at ~4.1 GB and the rendered pages are unchanged.
  integrations: [react(), mdx({ optimize: true })],
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [
      rehypeKatex,
      [rehypeExternalLinks, { target: '_blank', rel: ['noopener', 'noreferrer'] }],
      // Must stay here: an `mdx({ rehypePlugins })` list replaces this one instead of extending it.
      rehypeTableCellAlignToStyle,
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
