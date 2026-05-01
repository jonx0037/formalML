#!/bin/bash
# preview-before-commit.sh
#
# Blocks `git commit` when src/content/topics/*.mdx is staged unless a
# dev server (`pnpm dev` / `astro dev`) is currently running. Enforces the
# project policy: "always preview a topic in a dev server before publishing
# it live."
#
# Why: KaTeX is configured non-strict in this project, so math parse errors
# render as inline `<span class="katex-error">` rather than failing
# `pnpm build`. Build success is therefore insufficient verification — the
# only way to catch rendering issues like `\begin{aligned}` MDX/JSX
# collisions is to view the page in a dev server.
#
# Wired in .claude/settings.json as a PreToolUse hook on Bash.
# Exit code 0 = allow; exit code 2 = block with the message on stderr.

set -euo pipefail

input=$(cat)

# Extract the Bash command from the tool input JSON.
if command -v jq >/dev/null 2>&1; then
  cmd=$(echo "$input" | jq -r '.tool_input.command // empty')
else
  cmd=$(echo "$input" | python3 -c "import json,sys; print(json.load(sys.stdin).get('tool_input',{}).get('command',''))")
fi

# Only fire on commands that contain `git commit`. Skip everything else.
# The pattern matches `git commit`, `git commit -m ...`, `git commit -a`, etc.,
# but not `git commit-tree` (which is a plumbing command for low-level use).
if ! echo "$cmd" | grep -qE '\bgit commit(\s|$|\")'; then
  exit 0
fi

# Move into the repo root so `git diff --cached` works regardless of cwd.
repo_root=$(git rev-parse --show-toplevel 2>/dev/null || echo .)
cd "$repo_root"

# List staged topic MDX files. If none are staged, the policy doesn't apply.
staged_topics=$(git diff --cached --name-only 2>/dev/null | grep -E '^src/content/topics/.*\.mdx$' || true)
if [ -z "$staged_topics" ]; then
  exit 0
fi

# Topic MDX changes are staged. Require a dev server process to be running.
# `pgrep -f` matches the full command line as an extended regex, so the
# `(\.mjs)?` suffix catches both `astro dev` (older invocations) and
# `node .../astro.mjs dev` (current pnpm/Astro 6 spawn shape).
if pgrep -f 'astro(\.mjs)? dev' >/dev/null 2>&1; then
  exit 0
fi

# No dev server detected — block the commit with a directive message.
cat >&2 <<EOF
🛑 BLOCKED: git commit on topic MDX files requires a running dev server.

Staged topic files:
$staged_topics

Project policy: always preview a topic in a dev server before publishing it
live. KaTeX errors and MDX rendering issues render as inline error spans
rather than failing 'pnpm build', so build success is not sufficient
verification.

To proceed:
  1. In another shell: pnpm dev
  2. Open: http://localhost:4321/topics/<topic-slug>/  (port may be 4322 if 4321 is taken)
  3. Visually verify the page renders correctly — math, figures, layout
  4. Keep the dev server running, then re-run 'git commit'

If you genuinely need to bypass (e.g., a non-rendering YAML-only change),
either unstage the MDX file or temporarily disable this hook in
.claude/settings.json with explicit user authorization.
EOF
exit 2
