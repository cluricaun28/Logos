---
type: master_index
updated: YYYY-MM-DD
description: Master index for your personal knowledge base and system documentation.
---

# Reference Library — Master Index

## System Documentation

### Logos Setup
- [logos-customizations](topics/logos-customizations.md) — Document your local modifications here
- [perpetual-memory-system](topics/perpetual-memory-system.md) — Architecture, configuration, operational status
- [rolling-window-context-engine](topics/rolling-window-context-engine.md) — Context archiving strategy

### Tools & Infrastructure
- [tool-system](tools/tool-system.md) — Dynamic schema fetching, tool categories, usage patterns
- [skills-index](topics/skills-index.md) — Compact index of all available skills with trigger keywords

## Entity Registry (People, Organizations, Publications)

*Tracks historical behavior, reaction patterns, funding ties, and topic-specific credibility.*

### Key Entities
<!-- Add your own entities here as you research topics -->

## Research Topics

<!-- Add research topics as you build them -->

---

## How to Use This Index

1. **Before answering any question:** Check this index for relevant entries about the topic
2. **When learning something new:** Create a new entry in the appropriate directory, then add it here
3. **Keep it organized:** Group related topics under headers. Link between related entries using `[name](path/to/file.md)` format

## Directory Structure

```
reference-library/
├── index.md          ← You are here — master index linking to everything
├── topics/           ← System docs, workflows, procedures, research
│   ├── logos-customizations.md
│   ├── perpetual-memory-system.md
│   └── skills-index.md
├── tools/            ← Tool schemas and usage guides
│   └── tool-system.md
└── entities/         ← People, organizations, publications
    └── (your entity pages)
```

## Creating New Entries

Use this frontmatter format for all entries:

```yaml
---
type: topic|entity|tool
name: "Entry Name"
category: relevant_category
created: YYYY-MM-DD
last_updated: YYYY-MM-DD
confidence: high|medium|low
related_entries: ["Other Entry", "Another Entry"]
---
```
