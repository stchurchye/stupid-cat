#!/usr/bin/env bash
# After agent stop: run pytest and suggest next MVP task + workflow.
set -euo pipefail

ROOT="${CURSOR_PROJECT_DIR:-$(pwd)}"
cd "$ROOT"

if [[ ! -f pyproject.toml ]] || ! grep -q 'name = "stupid-cat"' pyproject.toml 2>/dev/null; then
  exit 0
fi

PYTEST_STATUS="not_run"
if [[ -d .venv ]] && [[ -f .venv/bin/pytest ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  if pytest tests/ -q --tb=no 2>/tmp/stupid-cat-pytest.log; then
    PYTEST_STATUS="passed"
  else
    PYTEST_STATUS="failed"
  fi
fi

NEXT_TASK="(read plan)"
if [[ -f docs/superpowers/plans/2026-06-02-stupid-cat-mvp.md ]]; then
  NEXT_TASK=$(python3 - <<'PY'
import re
from pathlib import Path
text = Path("docs/superpowers/plans/2026-06-02-stupid-cat-mvp.md").read_text(encoding="utf-8")
for m in re.finditer(r"^## (Task \d+:[^\n]+)\n\n", text, re.M):
    block_start = m.end()
    next_heading = text.find("\n## ", block_start)
    block = text[block_start : next_heading if next_heading != -1 else len(text)]
    if "- [ ]" in block:
        print(m.group(1))
        break
else:
    print("Phase 1 complete — run Task 16 checklist")
PY
)
fi

python3 - <<PY
import json
status = "$PYTEST_STATUS"
next_task = """$NEXT_TASK"""

if status == "failed":
    msg = (
        "pytest failed. Run \`pytest tests/ -v\`, fix failures, then /code-review "
        "recent changes before continuing the MVP plan."
    )
elif status == "passed":
    msg = (
        f"pytest passed. Next: **{next_task}**. Workflow: /code-review (confirm issues) "
        "→ fix → /tdd continue. Update plan checkboxes when a task is truly done."
    )
else:
    msg = (
        f"Run \`source .venv/bin/activate && pytest tests/ -q\`. Then /tdd for **{next_task}**."
    )

print(json.dumps({"followup_message": msg}))
PY
