# Logos — Divergence from Upstream

## Origin

Logos was built on NousResearch/hermes-agent as a foundation. It has since pursued a fundamentally different direction: sovereign knowledge management with persistent memory, epistemic filtering, and user-defined worldview alignment.

Logos is now a fully detached, independent project — no longer in any fork network (unlinked May 2026).

## Relationship to Upstream

The `upstream` remote is retained for cherry-picking useful improvements. We do not track upstream linearly, nor do we attempt to merge.

**Cherry-pick strategy:**
1. `git fetch upstream main`
2. `git log upstream/main --oneline -20` — review what's new
3. `git cherry-pick <hash>` — apply only what adds value
4. Test immediately after each pick

We do not merge blindly. Custom plugins and modified core files will conflict. Each pick is evaluated on its own merit.

## What We Don't Track

We deliberately do not follow upstream changes that add:
- Cloud-dependent features (third-party API integrations we don't use)
- Chat platform integrations irrelevant to our deployment (Chinese platforms, etc.)
- Features that undermine data sovereignty
- Complexity that doesn't serve the Logos use case

## Other Sources

We may also cherry-pick or adapt useful patterns from other projects (OpenClaw, Claude Code, Codex, etc.) when they offer functionality worth integrating. Same rules apply: review, pick selectively, test.

## The Divergence Is Intentional

Logos optimizes for one use case: a single user building a private, sovereign knowledge system with persistent memory and epistemic rigor. Upstream Hermes optimizes for general-purpose AI assistant deployment. Those goals diverge. We don't regret the divergence — it's the point.
