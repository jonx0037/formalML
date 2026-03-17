# Claude Code Handoff Brief: Dark/Light Mode Toggle
**Project:** formalML — [formalml.com](https://www.formalml.com)  
**Repo:** `github.com/jonx0037/formalml`  
**Stack:** Astro · MDX · TypeScript · CSS · Vercel  
**Package Manager:** pnpm  
**Status:** Ready for implementation

---

## Objective

Add a persistent dark/light mode toggle to the site header. The toggle should respect the user's system preference on first visit, allow manual override, and persist the choice across sessions.

---

## Current State

- The site header lives in the Astro layout/component layer and contains the `formalML` wordmark plus nav links (Topics, Paths, About).
- The site currently renders in a single (light) color scheme with no theme switching logic.
- There is no existing CSS custom property system for theming — colors are applied directly or via Astro/CSS scoping.
- Content pages use MDX with math rendering and code blocks, both of which need to respect the active theme.

---

## Implementation Plan

### 1. CSS Custom Properties (Design Tokens)

Create a centralized set of CSS custom properties on `:root` and `[data-theme="dark"]`. At minimum, define tokens for:

| Token | Light Value | Dark Value | Usage |
|---|---|---|---|
| `--color-bg` | `#FFFFFF` | `#0f1117` | Page background |
| `--color-bg-surface` | `#F8F9FA` | `#1a1d27` | Card/surface backgrounds |
| `--color-text-primary` | `#1a1a2e` | `#e4e4e7` | Body text |
| `--color-text-secondary` | `#64748b` | `#94a3b8` | Muted/secondary text |
| `--color-text-heading` | `#0f172a` | `#f1f5f9` | Headings |
| `--color-accent` | `#6366f1` | `#818cf8` | Links, interactive elements |
| `--color-accent-hover` | `#4f46e5` | `#a5b4fc` | Hover states |
| `--color-border` | `#e2e8f0` | `#2d3348` | Borders, dividers |
| `--color-code-bg` | `#f1f5f9` | `#1e2130` | Inline code background |
| `--color-tag-bg` | varies | varies | Topic difficulty tags |

**Note:** Sample the existing site's actual color values and map them into the light set. The dark values above are starting points — adjust to taste.

### 2. Theme Toggle Component

Create `src/components/ThemeToggle.astro` (or `.tsx` if using a client-side island):

- **Render:** A button in the header with sun/moon icons (use inline SVG, not an icon library dependency).
- **Behavior (client-side script):**
  1. On page load, check `localStorage.getItem('theme')`.
  2. If no stored preference, read `prefers-color-scheme` media query.
  3. Set `data-theme` attribute on `<html>` element.
  4. On click, toggle between `light` and `dark`, update `data-theme`, and persist to `localStorage`.
- **Flash prevention:** Add an inline `<script is:inline>` in the `<head>` of the base layout that reads the stored theme and sets `data-theme` *before* the page paints. This avoids a light→dark flash on dark-mode pages.

### 3. Header Integration

- Place the toggle button in the header, right-aligned after the nav links.
- On mobile/narrow viewports, the toggle should remain visible and accessible (do not hide it inside a hamburger menu if one exists).

### 4. Content Compatibility

- **Code blocks:** If using a syntax highlighting library (Shiki, Prism, etc.), configure it to support both a light and dark theme, switching based on the `data-theme` attribute. Astro's built-in Shiki integration supports `css-variables` theme mode — prefer this approach.
- **Math rendering:** KaTeX or MathJax text color should inherit from `--color-text-primary` via CSS.
- **MDX content:** Ensure any hardcoded colors in MDX component styles use CSS custom properties or inherit correctly.
- **Images/SVGs:** Any diagrammatic SVGs embedded in content should either use `currentColor` or have explicit light/dark variants.

### 5. Transition Polish

Add a brief CSS transition on theme switch to avoid jarring color jumps:

```css
html {
  transition: background-color 0.2s ease, color 0.2s ease;
}
```

Keep the transition short — long transitions feel sluggish.

---

## File Touchpoints (Expected)

| File | Action |
|---|---|
| `src/styles/global.css` (or equivalent) | Add CSS custom property definitions for `:root` and `[data-theme="dark"]` |
| `src/components/ThemeToggle.astro` | **New file** — toggle button component |
| `src/layouts/Layout.astro` (or base layout) | Add flash-prevention script in `<head>`, import and place `<ThemeToggle />` in header |
| `astro.config.mjs` | Update Shiki theme config to `css-variables` if not already set |
| Any component with hardcoded colors | Migrate to CSS custom properties |

---

## Constraints

- **No new runtime dependencies.** This should be pure CSS + vanilla JS in an Astro inline script. Do not add `next-themes`, `astro-color-scheme`, or similar packages — the implementation is simple enough to own directly.
- **No FOUC.** The inline head script is mandatory. Users on dark system preferences must never see a white flash.
- **Accessible.** The toggle button needs `aria-label` (e.g., "Switch to dark mode" / "Switch to light mode") that updates with the current state.
- **Respect existing design language.** The dark palette should feel like a natural inversion of the current site, not a separate design. Keep the same typography, spacing, and hierarchy.

---

## Acceptance Criteria

1. The toggle button is visible in the header on all viewport sizes.
2. Clicking the toggle switches between light and dark mode immediately.
3. Preference persists across page navigations and browser sessions via `localStorage`.
4. First visit defaults to system preference (`prefers-color-scheme`).
5. No flash of wrong theme on any page load.
6. Code blocks render with appropriate syntax highlighting in both modes.
7. Math expressions are legible in both modes.
8. All interactive elements (links, buttons, tags) have visible focus/hover states in both modes.

---

*Handoff prepared by Claude · formalML · March 2026*
