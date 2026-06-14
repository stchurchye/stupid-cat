# stupid-cat — agent instructions

Vision-based monitor for five cats sharing a large lidded litter box (who / when / duration). Phase 1 MVP is specified in `docs/superpowers/`.

## Agent skills

### Issue tracker

Issues live in **GitHub Issues** (via `gh` CLI). See `docs/agents/issue-tracker.md`. Initialize git + GitHub remote before publishing issues.

### Triage labels

Default five-label vocabulary (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

No root `CONTEXT.md` yet. Read project domain from:

- **Spec (architecture & requirements, v0.3):** `docs/superpowers/specs/2026-06-02-stupid-cat-litter-vision-design.md`
- **MVP implementation plan:** `docs/superpowers/plans/2026-06-02-stupid-cat-mvp.md`

Use terminology from the spec. ADRs under `docs/adr/` may be added later; proceed without them if missing.

## Cursor hooks (MVP loop)

Project hooks in `.cursor/hooks.json` inject workflow on session start and after agent `stop`:

1. `/tdd` — next unchecked task in `docs/superpowers/plans/2026-06-02-stupid-cat-mvp.md`
2. `/code-review` — confirm issues before fixing
3. `pytest tests/ -q` — must pass before done
