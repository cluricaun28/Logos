# Upstream Tracking

Logos originated from [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) and has diverged substantially. This file tracks which upstream commits have been cherry-picked and why, and which are deliberately ignored.

## Philosophy

We cherry-pick selectively. Being "behind" upstream by commit count is intentional — many upstream changes add features we don't want (cloud integrations, third-party chat services, paid API dependencies). We take only what improves our sovereign local-first architecture.

## Cherry-Picked Commits

| Date | Commit | Description | Why |
|------|--------|-------------|-----|
| 2026-04-30 | Batch #1 (20 commits) | StreamingContextScrubber, compress→archive rename, Docker fixes | Core infra improvements we needed |
| 2026-05-XX | [TBD] | [TBD] | [TBD] |

## Deliberately Ignored

- Cloud API integrations (OpenRouter fallback, Anthropic direct, etc.) — all inference is local
- Third-party chat service integrations — we use Telegram gateway
- Cloud storage sync — all data stays local
- Features that depend on external APIs for core functionality

## How to Cherry-Pick

```bash
# Fetch latest from upstream
git fetch upstream main

# Review changes
git log upstream/main --oneline -20

# Cherry-pick specific commits
git cherry-pick <commit-hash>

# Test after each pick
hermes gateway restart
hermes logs --follow
```

## Upstream Status

- **Upstream:** NousResearch/hermes-agent
- **Our remote:** `upstream` → github.com/NousResearch/hermes-agent
- **Current divergence:** ~2,000 commits behind, ~350 ahead (as of May 2026)

This is intentional and sustainable. We're building a different system now.
