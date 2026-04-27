# Hermes Agent Fork — Perpetual Context Memory & Plugin Extensions

**Fork of [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) with custom extensions for persistent cross-session memory, context archiving, and plugin-based retrieval.**

---

## What This Fork Adds

This fork transforms Hermes Agent from a stateless chat interface into an agent with **persistent memory**, **curated knowledge**, and **proactive retrieval**. The key additions:

### 1. Perpetual Context Database (`agent/perpetual_context_db.py`)
SQLite-backed conversation archive with FTS5 full-text search and optional semantic embeddings (ONNX/SentenceTransformers). Every turn is stored locally — no cloud sync, no external API calls. Hybrid retrieval: keyword + vector similarity in a single query.

### 2. Perpetual Context Plugin (`plugins/memory/perpetual_context/`)
Full plugin suite providing perpetual memory tools to the agent:
- **Hybrid search** — `perpetual_search` (FTS5 + semantic), `query_messages` (SQL-style filtering), `get_messages` (exact pattern matching)
- **Smart retrieval** — auto-routes queries to optimal strategy (recent, topic-specific, decision trace, file history)
- **Context bridge builder** — extracts active tasks, errors, and decisions for injection at archival boundaries
- **Deep research engine** — web research client (SearXNG/Firecrawl), source credibility scrutiny gate, multi-pass synthesis

### 3. Rolling Window Context Archiving (`plugins/context_engine/rolling_window/`)
Replaces passive context retention with active archiving: compress completed conversation turns to permanent storage while keeping the working window lean for deep present-moment reasoning. State continuity through retrieval, not retention.

### 4. Dataset Export Utilities (`agent/export_dataset.py`)
Export conversation trajectories for DPO training data extraction and analysis.

### Modified Core Files (8 files)
| File | Change |
|------|--------|
| `run_agent.py` | Context Bridge injection at archival boundary, rolling window integration |
| `agent/prompt_builder.py` | System prompt modifications for perpetual memory context |
| `plugins/context_engine/__init__.py` | Architecture changes for rolling window archiving |
| `acp_adapter/server.py` | ACP server customizations |
| `cli.py` | CLI commands for perpetual memory operations |
| `model_tools.py` | Local inference priority routing |
| `tools/skill_manager_tool.py` | Fork-aware skill paths |
| `.gitignore` | Patterns for custom plugin artifacts |

---

## Quick Start — Full Setup Guide

Follow these steps to set up the fork with perpetual memory, reference library, and skills. Each step builds on the previous one.

### Step 1: Install the Fork

```bash
# Clone the fork
git clone https://github.com/cluricaun28/hermes-agent.git
cd hermes-agent

# Install in development mode
pip install -e ".[dev]"

# Run Hermes setup to configure model, gateway, and plugins
hermes setup
```

### Step 2: Enable Perpetual Memory Plugin

Add to your `~/.hermes/config.yaml`:

```yaml
plugins:
  enabled:
    - perpetual_context
```

The plugin initializes the SQLite database (`~/.hermes/perpetual_context.db`) on first run. No manual DB creation needed.

### Step 3: Set Up Your Persona (SOUL.md)

**This is the most critical step.** The code provides infrastructure — your SOUL.md tells the agent *how to use it*. Without these prompt sections, the plugins load but the agent never calls them proactively.

Copy the template and customize it:

```bash
cp extras/soul-template.md ~/.hermes/SOUL.md
```

Then edit `~/.hermes/SOUL.md`:
1. Replace all `[YOUR NAME]` with your actual name
2. Customize the **Worldview Baseline** section with your values and beliefs
3. Customize the **Tone & Style** section with your preferred communication style
4. Keep all **Knowledge Architecture**, **Operational Discipline**, and **Active Retrieval** sections — these are what make perpetual memory work

**For detailed explanations of each system prompt section, see [`extras/system-prompt-guide.md`](extras/system-prompt-guide.md).**

### Step 4: Initialize Your Reference Library

The Reference Library is your agent's curated knowledge base. Start with the template structure:

```bash
# Copy the template to create your reference library skeleton
cp -r extras/reference-library-template ~/.hermes/reference-library
```

This creates:
```
~/.hermes/reference-library/
├── index.md              ← Master index (update as you add entries)
├── topics/               ← System docs, workflows, research
│   └── context-window-management.md  ← Example entry
├── tools/                ← Tool schemas and usage guides
│   └── tool-system.md    ← Explains how to document tools here
└── entities/             ← People, organizations, publications
    └── README.md         ← Instructions for building entity pages
```

**How it grows:** Your agent will automatically create new RL entries as you work. When researching a topic, the agent documents findings in `topics/`. When encountering tools, it schemas them in `tools/`. The index stays current because the agent updates it.

### Step 5: Understand Skills On-Demand Loading

Skills are reusable procedures stored in `~/.hermes/skills/`. They load on demand — only when relevant to your current task. This keeps the context window lean.

The template shows the structure:
```
extras/skills-template/
├── README.md                    ← How skills work, frontmatter format
└── devops/
    └── codebase-backup/         ← Example skill
        └── SKILL.md             ← Complete example of a well-structured skill
```

**How it works:** The system prompt includes a list of available skill names and descriptions. Before replying, the agent scans this list and loads only skills directly relevant to your task via `skill_view(skill_name)`.

### Step 6 (Optional): Set Up Deep Research

For web search with source extraction (SearXNG + Firecrawl), see [`extras/deep-research-setup.md`](extras/deep-research-setup.md). This enables the agent to:
- Search the web via a local meta-search engine
- Extract full content from URLs using Firecrawl
- Store research results in Perpetual Memory with source tracking

---

## Architecture Overview

```
┌─────────────────────────────────────────────┐
│  Hermes Agent (upstream core)               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │   CLI    │  │ Gateway  │  │  Tools   │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  │
│       └──────────────┼─────────────┘        │
│                      ▼                      │
│  ┌──────────────────────────────┐           │
│  │  Context Engine (modified)   │           │
│  │  - Rolling window archiving  │           │
│  │  - Context Bridge injection  │           │
│  └──────────────┬───────────────┘           │
│                 ▼                            │
│  ┌──────────────────────────────┐           │
│  │  Perpetual Context Plugin    │           │
│  │  - Hybrid search (FTS5+vec)  │           │
│  │  - Smart retrieval routing   │           │
│  │  - Deep research engine      │           │
│  └──────────────┬───────────────┘           │
│                 ▼                            │
│  ┌──────────────────────────────┐           │
│  │  SQLite Database (local)     │           │
│  │  - messages table + FTS5 idx │           │
│  │  - embeddings BLOB column    │           │
│  │  - topic relationships       │           │
│  └──────────────────────────────┘           │
└─────────────────────────────────────────────┘
```

### How the Dual-Memory System Works

```
User asks question → Agent calls recent_messages (last 10 turns)
→ Agent reads current prompt + recent context
→ Agent checks Reference Library for curated knowledge on topic
→ Agent calls perpetual_search for historical discussion on topic
→ Agent formulates response using retrieved info + present context
→ Completed turns are archived to permanent storage
→ Context window stays lean for next turn
```

Key design decisions:
- **`recent_messages` is mandatory first step** — prevents loops, maintains continuity
- **Search during thinking, not after** — tool use IS the reasoning process
- **Archive aggressively** — if a task is done, it belongs in permanent storage, not working memory
- **Print task status every turn** — `[Tasks: 3/5 complete]` becomes searchable PM data

---

## Semantic Embeddings (Optional)

For hybrid search with vector similarity alongside FTS5 keyword search:

```bash
pip install onnxruntime sentence-transformers
```

The default model is `all-MiniLM-L6-v2` — lightweight, runs locally. Embeddings are stored as BLOB in SQLite alongside FTS5 indexes. No configuration needed beyond installing the packages; the plugin auto-detects and enables them.

---

## How Things Grow Over Time

### Reference Library Growth
- **Topics:** Created when you research something or solve a complex problem. The agent documents findings in `~/.hermes/reference-library/topics/`.
- **Tools:** Created when the agent encounters deferred tools. Schemas are documented in `~/.hermes/reference-library/tools/` for future lookup.
- **Entities:** Created when researching people, organizations, or publications. Tracks credibility and behavior patterns in `~/.hermes/reference-library/entities/`.
- **Index:** Updated automatically by the agent as new entries are created.

### Skills Growth
- Skills are created when you solve a complex problem (5+ tool calls) that you'll likely face again.
- Stored in `~/.hermes/skills/CATEGORY/skill-name/SKILL.md`.
- The system prompt's skills list grows as you add more skills.
- See [`extras/skills-template/`](extras/skills-template/) for format examples.

### Perpetual Memory Growth
- Every conversation turn is stored automatically — no action needed.
- Grows continuously across sessions. Search it with `perpetual_search`, `query_messages`, or `recent_messages`.
- Old turns are pruned from the context window but remain fully searchable in the database.

---

## Daily Usage

### Starting a Session
```bash
hermes gateway start    # Start the agent gateway
hermes                  # Open interactive session
```

The agent will:
1. Load your SOUL.md (persona + behavioral rules)
2. Check `recent_messages` for session continuity
3. Scan available skills list (on-demand loading, not pre-loaded)
4. Wait for your input

### What the Agent Does Automatically
- **Checks recent conversation history** before taking action (anti-loop discipline)
- **Consults Reference Library** before answering factual questions
- **Searches Perpetual Memory** when topics reference past work
- **Loads skills on demand** only when relevant to your task
- **Creates new RL entries** when learning something new
- **Archives completed turns** to keep context window lean

### What You Should Do
- **Customize SOUL.md** with your values, preferences, and communication style
- **Review saved memories/skills periodically** — the agent will ask if anything should be saved
- **Keep config.yaml updated** as you add tools or services (SearXNG, Firecrawl, etc.)

---

## Maintenance

### Updating from Upstream

This fork tracks upstream via the `upstream` remote:

```bash
# Fetch latest from NousResearch
git fetch upstream main

# Review changes before cherry-picking
git log upstream/main --oneline -10

# Selectively apply commits
git cherry-pick <commit-hash>
```

**Do NOT merge blindly** — custom plugin files may conflict with upstream context engine changes. Cherry-pick selectively and test after each change.

### Backup Your Data

Your data lives in three places:
```bash
# Perpetual Memory database (conversation history)
~/.hermes/perpetual_context.db

# Reference Library (curated knowledge)
~/.hermes/reference-library/

# Skills (reusable procedures)
~/.hermes/skills/
```

Back them up regularly:
```bash
# Quick backup script
tar czf hermes-data-$(date +%Y%m%d).tar.gz \
  ~/.hermes/perpetual_context.db \
  ~/.hermes/reference-library/ \
  ~/.hermes/skills/ \
  ~/.hermes/SOUL.md \
  ~/.hermes/config.yaml
```

### Troubleshooting

| Problem | Solution |
|---------|----------|
| Agent doesn't use perpetual memory tools | Check SOUL.md has "Knowledge Architecture" and "Active Retrieval" sections |
| Plugin not loading | Verify `perpetual_context` is in `config.yaml` plugins.enabled list |
| Semantic search not working | Install `onnxruntime sentence-transformers` |
| Context window too full | Check rolling window config, verify archiving is enabled |
| Agent loops on same task | SOUL.md should have "Anti-Loop Discipline" section; agent checks recent_messages first |

---

## What's Not Here (By Design)

This fork does **not** include:
- Personal data, tokens, or credentials — those live in your local `~/.hermes/` directory
- Reference library content — starts as a template, grows through use
- Skill definitions — starts with examples, grows as you solve problems
- SQLite database files — created on first run, persists locally

**The philosophy:** Ship the system that builds knowledge, not the knowledge itself. Your agent should grow its own reference library and skills through your usage patterns.

---

## Extras Directory

| File | Purpose |
|------|---------|
| [`system-prompt-guide.md`](extras/system-prompt-guide.md) | Exact SOUL.md sections needed for perpetual memory to work |
| [`soul-template.md`](extras/soul-template.md) | Ready-to-use SOUL.md template with all system prompt additions |
| [`deep-research-setup.md`](extras/deep-research-setup.md) | SearXNG + Firecrawl setup guide for web research |
| [`reference-library-template/`](extras/reference-library-template/) | Empty RL structure to copy as your starting point |
| [`skills-template/`](extras/skills-template/) | Example skill showing proper format and structure |

---

## License

Same as upstream: MIT. Custom additions are also MIT licensed.

---

**Upstream:** [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent)  
**Fork:** [cluricaun28/hermes-agent](https://github.com/cluricaun28/hermes-agent)
