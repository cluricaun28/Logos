# Skills Template — Example Skill

This directory shows how skills are structured. Each skill is a self-contained markdown file with YAML frontmatter that documents a reusable procedure or workflow.

## Directory Structure

```
skills/
├── category-name/           ← Group related skills by domain
│   └── skill-name/          ← One directory per skill
│       ├── SKILL.md         ← The skill document (required)
│       ├── references/      ← Optional: supporting documentation
│       ├── templates/       ← Optional: config files, scripts
│       └── scripts/         ← Optional: executable helpers
├── another-category/
│   └── another-skill/
│       └── SKILL.md
└── README.md                ← This file — explains the system
```

## Skill Frontmatter Format

```yaml
---
name: skill-name
description: One-line description of what this skill does.
trigger_keywords: [keyword1, keyword2, related_term]
category: devops|data-science|mlops|web|other
created: YYYY-MM-DD
last_updated: YYYY-MM-DD
confidence: high|medium|low
---
```

## Example Skill Document

See `devops/codebase-backup/SKILL.md` for a complete example of a well-structured skill.

## When to Create Skills

Create a skill when:
1. You solve a complex problem (5+ tool calls) that you'll likely face again
2. A workflow involves multiple steps with dependencies and edge cases
3. You discover a non-obvious procedure worth remembering
4. The user asks you to remember how to do something

## When NOT to Create Skills

Skip skills for:
1. Simple one-off tasks with no reusable pattern
2. Tasks easily re-discovered from documentation
3. Raw data dumps or session logs — those belong in Perpetual Memory, not skills

## Loading Skills On-Demand

Skills are loaded via `skill_view(skill_name)` when needed. The system prompt includes a scan of available skill names and descriptions — the agent reads only relevant skills into context, keeping the window lean.

```markdown
## Before replying, scan the skills below and load only those DIRECTLY relevant to your task
- '''codebase-backup''' — Versioned backup system for Logos codebase
- '''logos''' — Logos configuration, setup, troubleshooting
[... more skills ...]
```
