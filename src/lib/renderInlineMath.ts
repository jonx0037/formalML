import katex from 'katex';

/**
 * Render a plain-text string that may contain `$…$` (inline) or `$$…$$`
 * (display) TeX into an HTML string: math segments go through KaTeX, text
 * segments are HTML-escaped. Used for props such as `<Figure caption="…">`,
 * which MDX passes through as raw strings, so remark-math/rehype-katex never
 * see them.
 *
 * Delimiter rules (Pandoc's, so prose dollars survive):
 * - `\$` outside math is a literal dollar sign.
 * - An opening `$` must not be followed by whitespace; a closing `$` must not
 *   be preceded by whitespace or followed by a digit. So "costs $5 and $10"
 *   stays text.
 * - A `$` with no valid closer is kept as literal text.
 *
 * MDX/JSX string attributes do not process escapes, yet many captions were
 * written JS-style (`$\\hat f_h$`), so `\\` reaches us literally. Inside
 * caption math a TeX line break is meaningless, so `\\` is read as an
 * over-escaped `\` (`\\hat` → `\hat`, `\\%` → `\%`, `\\,` → `\,`).
 *
 * KaTeX runs at build time (Astro components render on the server) with the
 * same options rehype-katex uses for MDX math, so bad TeX renders as a red
 * `katex-error` span instead of failing the build.
 */
export function renderInlineMath(input: string): string {
  let out = '';
  let text = '';
  let i = 0;

  const flushText = () => {
    out += escapeHtml(text);
    text = '';
  };

  while (i < input.length) {
    const ch = input[i];

    if (ch === '\\' && input[i + 1] === '$') {
      text += '$';
      i += 2;
      continue;
    }

    if (ch === '$') {
      const display = input[i + 1] === '$';
      const delim = display ? '$$' : '$';
      const start = i + delim.length;
      const end = findCloser(input, start, delim);
      if (end !== -1) {
        flushText();
        const tex = input.slice(start, end).replace(/\\\\/g, '\\');
        out += katex.renderToString(tex, {
          displayMode: display,
          throwOnError: false,
        });
        i = end + delim.length;
        continue;
      }
      text += delim;
      i += delim.length;
      continue;
    }

    text += ch;
    i += 1;
  }

  flushText();
  return out;
}

/** Index of the closing delimiter for math opened just before `start`, or -1. */
function findCloser(s: string, start: number, delim: '$' | '$$'): number {
  if (start >= s.length || /\s/.test(s[start])) return -1;
  for (let j = start; j < s.length; j++) {
    if (s[j] === '\\') {
      j += 1; // skip the escaped character (e.g. `\$` or `\\` inside TeX)
      continue;
    }
    if (!s.startsWith(delim, j)) continue;
    if (j === start || /\s/.test(s[j - 1])) continue;
    if (delim === '$') {
      if (s[j + 1] === '$') continue; // part of a `$$`, not a closer
      if (/[0-9]/.test(s[j + 1] ?? '')) continue;
    }
    return j;
  }
  return -1;
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}
