#!/usr/bin/env bash
# Project hook: inject stupid-cat MVP workflow context at session start.
set -euo pipefail

PLAN="docs/superpowers/plans/2026-06-02-stupid-cat-mvp.md"
SPEC="docs/superpowers/specs/2026-06-02-stupid-cat-litter-vision-design.md"

python3 - <<'PY'
import json

plan = "docs/superpowers/plans/2026-06-02-stupid-cat-mvp.md"
spec = "docs/superpowers/specs/2026-06-02-stupid-cat-litter-vision-design.md"

ctx = f"""## stupid-cat agent workflow (project hooks)

1. Implement the **next unchecked** task in `{plan}` using **/tdd** (vertical slices).
2. Run **`/code-review`** on changed code; fix only **confirmed** issues.
3. Verify: `pytest tests/ -q` (must pass before claiming done).
4. Repeat until Task 16 complete.

Spec: `{spec}` (v0.3). Skills: `.agents/skills/tdd`, Matt Pocock engineering skills, `docs/agents/*`.

Do not commit unless the user asks."""

print(json.dumps({"additional_context": ctx}))
PY
