# Work Order — 2026-08-26 Review Follow-up (Patrick-approved: A, B, C, D, E, F)

Repo: /data1/logos-sandbox/logos (sandbox main = prod tip 606a1458). Prod = /home/exx/logos.
Prod venv for gates: /home/exx/logos/venv. Flat imports (bare module names). pytest: never `-p`.
GitHub push = ops-only, allowed (PAT configured). NEVER restart any gateway process.

**Discipline (Patrick, standing):** coding best practices optimized for AGENTS, not people —
self-describing config, explicit loud warnings over silent drops, machine-readable outputs,
tests as specification (RL system/design-principles.md; pivot 2026-07-16).
Update this file's "Phase state" section after EVERY phase (resume = read here).

## Phase P1 — Delegation completion contract (A + F-resume) [tools/delegate_tool.py]
1. Structured completion contract: child final summary must yield {completed, total,
   output_paths, failures}. Add a lightweight parser on the orchestrator side + a line in the
   subagent system prompt requiring the summary to end with a fenced `json completion_report`
   block. Orchestrator compares completed==total; on mismatch, return an explicit
   `contract_mismatch` warning block in the tool result (do NOT hard-fail — model output is
   best-effort evidence, artifacts are ground truth).
2. `fan_in(expected: list, output_dir) -> {merged, missing, dupes}` helper exposed in
   tools/delegate_* + one test proving it catches the 144-vs-129 class (expected set vs glob).
3. Resumable children: on timeout/partial, persist {done_items, output_dir, task_context}
   to ~/.hermes/state/delegations/<child_id>.json; accept `resume_from: <child_id>` per task
   that injects the done-set into the child prompt ("these items are already complete —
   verify and continue with the remainder"). Test the resume payload round-trip.

## Phase P2 — context-engine.jsonl session tagging (C) [plugins/context_engine/semantic_vector/__init__.py]
Calibration events lack a `session` field (archive events have it). Add session id (same value
archive events use) to every calibration/other engine event emitted; regression test asserting
every emitted event line contains non-empty session.

## Phase P3 — Config honesty: context tree with dual-read (B)
As-found (measured 8/26): top-level `archiving:` (fallback `compression:`) is LIVE
(run_agent.py:1866-1875 + gateway/run.py:4661 hygiene); `context.archiving.threshold: 0.9` is
DEAD (no reader); `context.rolling_window` live via C9-A. Target:
```
context:
  engine: semantic_vector
  semantic_vector: {...}   # primary engine knobs (unchanged)
  fallback: {...}          # canonical home for compressor/rolling fallback knobs
```
- Loader prefers `context.fallback.*`, falls back to legacy top-level `archiving/compression`
  (keeps every fleet user's existing config working verbatim).
- ANY recognized-but-unread key (e.g. context.archiving) → ONE explicit startup WARNING line
  naming the key + its live replacement (agents get told; silence is the bug).
- Update live exx config ONLY after merge+restart guidance (do not edit ~/.hermes/config.yaml).
- Tests: legacy parity, new-path precedence, dead-key warning emission.

## Phase P4 — Hygiene (D)
- rm -rf '%s' (sandbox AND /home/exx/logos — untracked artifacts), add gitignore pattern
  `/%s` + verify egg-info/temp_vision_images ignored (rm on-disk egg-infos both trees;
  they regenerate).
- Delete .github/workflows referencing removed subsystems: deploy-site, docs-site-checks,
  nix-lockfile-check, docker-publish (verify each references only deleted trees first;
  keep contributor-check.yml unless it references nix/website).
- Bring prod's uncommitted rl_viewer.py (166+/48−) INTO sandbox:
  `git -C /home/exx/logos show HEAD:rl_viewer.py > /tmp/base; diff` → copy working-tree
  file into sandbox, review diff for sanity, commit as separate logical commit.

## Phase P5 — Whitepaper v3.3, single-user-first (E)
WHITEPAPER.md last_updated 2026-08-20 — 66 commits stale (rebrand landed, green skin,
usage metering, fleet keys, cleanup R1-R13). Rewrite affected sections. FRAMING RULE
(Patrick 8/26): the default reader is a SINGLE-USER open-source deployer — core system
must stand alone on one box (vLLM or llama.cpp + one gateway + SQLite; LiteLLM fleet
keys, per-user units, A2A are documented OPTIONAL fleet extensions, never prerequisites).
Add/verify a Quickstart section matching that. Do not touch ~/.hermes/reference-library
(Patrick's agent owns RL updates).

## Phase P6 — Gate + ff-merge + push
- Full canonical gate on final tree: scripts/run_gate_final.py layout, prod venv
  (/home/exx/logos/venv), detached via `systemd-run --user` if >10min; 0 failures required.
- ff-merge sandbox→prod: `git -C /home/exx/logos fetch /data1/logos-sandbox/logos main`
  → `merge --ff-only FETCH_HEAD`. Push origin from prod. NO gateway restart (Patrick's
  action). Note in commit messages which changes are restart-pending.

## Definition of done
Each phase = its own logical commit; phase-state section below updated after each;
final report: commits list, gate result, merge SHA, pushed?, restart-pending inventory.

## Phase state
P1 DONE commit=c4648f38 — delegate_tool: completion contract (parse_completion_report + child-prompt requirement + contract_mismatch warning, no hard fail), fan_in(expected, output_dir) {merged,missing,dupes}, resumable children (persist_delegation_state/load_delegation_state/build_resume_note under <HERMES_HOME>/state/delegations/; resume_from per task; timeout+max_iterations persist). Tests: tests/tools/test_delegate_completion_contract.py 18 passed; existing delegate suites 135 passed. Handler now also forwards persona/capability_mode/timeout (previously dead schema params).
P2 DONE commit=d05e43ef — session tagging: SV calibration+task_aware_prune, RW calibration; on_session_start stash both engines; tests/agent/test_context_engine_session_tagging.py 5 new (29 passed incl. adjacent). Restart-pending.
P3 DONE commit=04d3c2a7 — resolve_context_fallback_config (context.fallback canonical, legacy archiving/compression verbatim, fallback>archiving>compression precedence), once-per-process WARNINGs for dead context.archiving/compression + shadowed legacy; gateway hygiene read + cache-bust signature wired. Tests: tests/run_agent/test_context_config_honesty.py 10 new + test_semantic_vector_rolling_config + 6 closest compression/config files + gateway agent_cache/session_hygiene = 87 passed. Restart-pending.
P4 DONE commits=acc07a60 (hygiene: removed 7 dead workflows website/nix/docker, gitignore /%s; strays + temp jpgs + prod egg-infos rm-ed both trees), 5bd32da2 (rl_viewer adopted from prod working tree; no secrets, no /home/exx paths). Note: .github/actions/nix-setup left in place (orphan action, flagged to parent). No tests reference deleted workflow names.
P5 DONE commit=07f1e467 — WHITEPAPER v3.3: header/date, measured scale, rebrand 5.0, context.fallback in 4.7, delegation contract in 4.9, 7.3 single-user Quickstart + 7.4 optional fleet metering, version-history line. Docs-only, no tests reference WHITEPAPER.
