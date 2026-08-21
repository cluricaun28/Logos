# Pinned Project Brief — Hide Conversation‑State Header + Batch Completion Bars (TUI)

_STATUS: active · pinned 2026‑08‑20 · protected head, survives pruning · check items off;
prune stale (drop to Perpetual Memory) · unpin on user approval when complete/abandoned._
_Supersedes the completed Crenshaw description/dims‑audit briefs._

## Scope
IN: (1) Hide the per‑turn "conversation state" header from the USER's view — the blocks:
     "## Active Tasks (with retrieval pointers)", "## Files Currently Being Edited",
     "## Known Errors/Issues", "## Historical Context Retrieval", "[Your active task list…]".
     (2) Hide the delegate/batch **completion bars** from the TUI instances.
OUT: Removing the functionality (keep generating it internally + for debug).
     These stay useful for the agent/debug — just not user‑visible.

## Goal
User never sees the state header or the batch completion bars; they still exist for the
agent + debugging. DONE = both hidden by default, **config‑gated to re‑enable**, verified
(syntax + targeted pytest + a visible check), shared with the user.

## Relevant details (add as relevant, prune as stale)
- **User believes the conversation‑state hide was ALREADY done "in a different commit."**
  CHECK git first (`git log -S "with retrieval pointers"`, `--grep` for hide/state/bar)
  before re‑implementing — it may exist but not be on/active in THIS sandbox build, or be
  gated by a config flag that's currently off. Don't duplicate it.
- Code root (sandbox): /data1/logos‑sandbox/logos. This is an unmerged prod‑candidate build.
- Already in this sandbox, UNCOMMITTED: `tools/delegate_tool.py` (+117/−4) — per‑task
  `timeout`, "returned‑the‑intro" fix (`_looks_like_intermediate_summary` + `output_tail`),
  resume‑from‑partial on timeout. Verified (syntax + 135 delegate tests). Don't lose it.
- The state header is likely built in `prompt_builder.py` or a context/bridge module.
  Search the distinctive strings first.
- Batch completion bars: delegate_task batch progress + terminal/cron progress bars —
  find the TUI render path (progress callback / display module). Gate behind a
  config/verbosity flag, don't delete.
- Edit discipline (same as the delegate work): additive + config‑gated + verified
  (py_compile + targeted `pytest tests/tools/test_delegate*.py` etc.).

## Checklist
- [x] Pin written (this file)
- [ ] Find state‑header builder + check git for the prior "hide" commit
- [ ] Find the batch completion bar TUI render path
- [ ] Implement hide (config‑gated, default off) for the state header
- [ ] Implement hide (config‑gated, default off) for the batch completion bars
- [ ] Verify: syntax + targeted pytest + visible check
- [ ] Report what changed + how to re‑enable for debugging
