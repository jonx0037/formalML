// Run: pnpm exec tsx --test src/lib/__tests__/renderInlineMath.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { renderInlineMath } from '../renderInlineMath';

const katexCount = (html: string) => (html.match(/class="katex"/g) ?? []).length;

test('renders inline $…$ math with KaTeX and leaves no raw delimiters', () => {
  const html = renderInlineMath('Prior at $\\theta_0$ and $\\mathbb{E}[Y \\mid X]$.');
  assert.equal(katexCount(html), 2);
  assert.match(html, /<annotation encoding="application\/x-tex">\\theta_0<\/annotation>/);
  assert.ok(html.startsWith('Prior at <span class="katex">'));
  assert.ok(html.endsWith('</span>.'));
});

test('plain captions pass through unchanged apart from escaping', () => {
  assert.equal(renderInlineMath('Figure 1. No math here.'), 'Figure 1. No math here.');
  assert.equal(renderInlineMath('a < b & "c"'), 'a &lt; b &amp; &quot;c&quot;');
});

test('text segments are HTML-escaped (no injection through captions)', () => {
  const html = renderInlineMath('<script>x</script> $x$');
  assert.ok(!html.includes('<script>'));
  assert.ok(html.startsWith('&lt;script&gt;x&lt;/script&gt; '));
});

test('escaped \\$ is a literal dollar sign', () => {
  assert.equal(renderInlineMath('costs \\$5 total'), 'costs $5 total');
});

test('currency dollars are not treated as math', () => {
  assert.equal(renderInlineMath('between $5 and $10'), 'between $5 and $10');
  assert.equal(renderInlineMath('a lone $ sign'), 'a lone $ sign');
  assert.equal(renderInlineMath('ends with $'), 'ends with $');
});

test('opening $ followed by space / closing $ preceded by space is not math', () => {
  assert.equal(renderInlineMath('$ x$ and $y $'), '$ x$ and $y $');
});

test('escaped dollar inside math stays in the TeX', () => {
  const html = renderInlineMath('price $\\$5 + x$ here');
  assert.equal(katexCount(html), 1);
  assert.match(html, /<annotation encoding="application\/x-tex">\\\$5 \+ x<\/annotation>/);
});

test('$$…$$ renders display math', () => {
  const html = renderInlineMath('see $$\\int_0^1 f$$ below');
  assert.match(html, /class="katex-display"/);
});

test('invalid TeX renders a katex-error span instead of throwing', () => {
  const html = renderInlineMath('bad $\\frac{1}{$ math');
  assert.match(html, /katex-error/);
});

test('JS-style doubled backslashes in caption math are read as single', () => {
  // MDX passes `caption="$\\hat f_h$"` through literally (no JS escape processing).
  const doubled = renderInlineMath('$\\\\hat f_h$ at $\\\\sim 50\\\\%$ and $[0.2,\\\\, 4.7]$');
  const single = renderInlineMath('$\\hat f_h$ at $\\sim 50\\%$ and $[0.2,\\, 4.7]$');
  assert.equal(doubled, single);
  assert.ok(!doubled.includes('\\\\'));
});
