# Logos — Sovereign Knowledge Management System

**A sovereign agentic intelligence system built on local hardware with persistent memory, curated knowledge, and user-defined worldview alignment.**

Most AI agents are stateless — they forget who you are and what you've decided once a conversation ends. Logos is a sovereign knowledge system that builds a permanent, local library of truth, anchored in your worldview.

### The Core Philosophy
- **Sovereignty:** All memory and knowledge live on your hardware. No cloud sync, no external API dependencies for recall.
- **Truth Hierarchy:** The system prioritizes curated, verified facts (Reference Library) over the volatile noise of the open web.
- **Continuity:** It remembers not just what was said, but the *decisions* made and the *tasks* deferred, ensuring work continues exactly where it left off across sessions.

---

### What Makes Logos Different

This isn't a chatbot. It's a research tool with memory.

The distinguishing features:

- **Frame-Stripping Skill** — 10 rules for separating facts from framing. Strips loaded language, extracts verifiable claims, cross-references independent sources, and presents findings through your stated worldview.
- **Narrative-Control-Detection** — Identifies six-phase information warfare patterns (initial break → narrative shift → article removal → flood the zone → entrenchment) when they appear in research results.
- **SourceAnalyzer** (`agent/source_analysis.py`) — Phase 3.5 in the research pipeline. Builds and updates source dossiers automatically, flagging ideological markers and consistent omission patterns.
- **Nightly Learning Loop** — Scheduled jobs run deep research, apply frame-stripping, and distill findings into the Reference Library. The knowledge base grows through use.

The system runs locally. No cloud APIs for memory or retrieval. No moral relativism baked in.

---

### 📄 White Paper
For a comprehensive architectural deep-dive covering all subsystems, the epistemic framework, and the design philosophy in detail, see [WHITEPAPER.md](WHITEPAPER.md).

---

## What Logos Is

Logos is an agent with **persistent memory**, **curated knowledge**, and **proactive retrieval**. The key subsystems:

### 1. Perpetual Context Database (`agent/perpetual_context_db.py`)
SQLite-backed conversation archive with FTS5 full-text search and optional semantic embeddings (ONNX/SentenceTransformers). Every turn is stored locally — no cloud sync, no external API calls. Hybrid retrieval: keyword + vector similarity in a single query.

### 2. Perpetual Context Plugin (`plugins/memory/perpetual_context/`)
Full plugin suite providing perpetual memory tools to the agent:
- **Hybrid search** — `perpetual_search` (FTS5 + semantic), `query_messages` (SQL-style filtering), `get_messages` (exact pattern matching)
- **Smart retrieval** — auto-routes queries to optimal strategy (recent, topic-specific, decision trace, file history)
- **Context bridge builder** — extracts active tasks, errors, and decisions for injection at archival boundaries
- **Source analysis** — `source_analyze` examines web search results for ideological alignment, omissions, and deviations. `deep=true` mode extracts full article content via Firecrawl before analyzing. Auto-creates source dossiers in `sources/` for new domains.
- **Logos Deep Research & Continuity** — Sovereign knowledge acquisition pipeline:
    - **Three-Tier Web Stack:** SearXNG (Discovery) $\rightarrow$ Firecrawl (Extraction) $\rightarrow$ Camofox (Anti-detection Browser).
    - **Epistemic Filtering:** Integrated scrutiny gate that filters raw web data through a user-defined worldview baseline before RL ingestion.
    - **Adaptive Retrieval Cascade:** A reasoning-driven flow (Immediate Context $\rightarrow$ PM Recall $\rightarrow$ RL Authority $\rightarrow$ Deep Research) to ensure the most accurate source is used for every query.

### 3. Rolling Window Context Archiving (`plugins/context_engine/rolling_window/`)
Replaces passive context retention with active archiving: compress completed conversation turns to permanent storage while keeping the working window lean for deep present-moment reasoning. State continuity through retrieval, not retention.

Key design:
- **Incremental tail-off** — When the context window approaches its threshold (~75%), the engine strips tool calls, truncates verbose tool results, then drops the oldest unprotected messages until token count falls below the archive target (65%). No LLM calls, no semantic clustering, no task tracking — simple and deterministic.
- **Hard ceiling at 85%** — Emergency safety net that performs a drastic split if the incremental pruning isn't enough, preventing OOM mid-turn.
- **Single-file implementation** — The rolling window directory contains only `__init__.py`. All task-aware and semantic-vector code has been removed as unnecessary overhead.

### Modified Core Files (7 files)

These core files were modified from the Hermes Agent base. The diffs are committed in this repo — you can see exactly what changed with `git log -- <file>`. Key changes:

|| File | What Changed | Why It Matters |
|------|-------------|----------------|
| `run_agent.py` | Renamed "compression" $\\rightarrow$ "archiving", selective tool loading | Enables rolling window + perpetual memory integration. Config key is now `archiving:` instead of `compression:` |
| `agent/prompt_builder.py` | Skills section changed from mandatory to on-demand loading with validation | Prevents context bloat — only loads skills actually relevant to the task |
| `agent/context_engine.py` | Rolling window engine with incremental tail-off, removed semantic vector engine | Deterministic pruning: no LLM calls, no task tracking, no overhead |
| `plugins/context_engine/__init__.py` | Rolling window engine loader with config passthrough | Pluggable context archiving strategy |
| `model_tools.py` | Added `get_selective_tool_definitions()` and deferred tools index | Essential tools loaded inline, deferred tools listed for RL lookup — saves context tokens |
| `cli.py` | Perpetual memory CLI commands (`hermes pm search`, etc.) | Query your conversation history from the terminal |
| `tools/skill_manager_tool.py` | Fork-aware skill path resolution | Skills find custom categories correctly |
| `.gitignore` | Patterns for plugin artifacts and cache files | Keeps git clean |

**Config note:** The config key changed from `compression:` to `archiving:`. Your `config.yaml` should use:
```yaml
archiving:
  enabled: true
  threshold: 0.50
  target_ratio: 0.20
  protect_last_n: 20
```
(Old `compression:` key still works for backward compatibility.)

---

## 🧠 Research Priority Matrix (Epistemic Routing)
The agent uses a tiered priority system to determine where to find truth:
- **Bedrock Topics (Worldview, Theology, Core Logic):** RL $\rightarrow$ PM $\rightarrow$ Web. The Reference Library is the ultimate authority; web search is only for filling gaps.
- **Volatile Topics (News, Tech Specs, Pricing):** Web $\rightarrow$ RL. Fresh data is prioritized over archived knowledge.

---

## 🛡️ Sovereignty & System Requirements

This repository provides the **engine**, not the **knowledge**. To duplicate this system, you must provide your own local data and worldview.

### Technical Requirements
- **Local Embeddings:** For hybrid search to function without cloud APIs, you must install `onnxruntime` and `sentence-transformers`. The system defaults to `all-MiniLM-L6-v2` for local, private vectorization.
- **Hardware:** A machine capable of running LLMs locally (via LM Studio or vLLM) is strongly recommended to maintain total data sovereignty.
- **Storage:** All memory and the Reference Library are stored in your local home directory (`~/.hermes/`), ensuring that your personal knowledge base never leaves your hardware.
- **Local Services:** The deep research pipeline requires three local services (all self-hosted, no cloud API calls):
    - **SearXNG** — Local meta-search engine (Docker container)
    - **Firecrawl** — Local web content extraction (Docker container)
    - **Camofox** — Anti-detection browser for sites that block scrapers

---

## For Agent Operators — Critical Setup Details

**Before following the Quick Start, read this if you are an AI agent tasked with setting up Logos.** The infrastructure alone is not enough — the *prompt structure* is what makes the system work.

### SOUL.md — The Most Critical File

The code provides the plumbing. SOUL.md tells the agent *how to use it*. Without these specific sections in SOUL.md, the plugins load but the agent never calls them:

1. **Knowledge Architecture section** — Must instruct the agent to:
   - Check `reference_library_search` BEFORE `web_search` for any knowledge question
   - Use `recent_messages` for immediate context before acting
   - Use `perpetual_search` / `smart_retrieve` for cross-session recall
   - Treat the Reference Library as *authoritative truth*, training data as *suggestion*

2. **Worldview Baseline section** — Must define the user's epistemic framework:
   - What truth is (not relativism)
   - How to handle contradictory worldviews (state the difference, don't false-balance)
   - Tone and style preferences

3. **Operational Discipline section** — Must include:
   - Anti-loop rules (check recent context before repeating work)
   - Memory save rules (what to persist vs. what to discard)
   - Skill loading protocol (scan available skills, load only relevant ones)
   - Three-tier web escalation (SearXNG → Firecrawl → Camofox)

4. **Active Retrieval section** — Must instruct the agent to:
   - Call `source_analyze` after `web_search` for substantive topics (politics, religion, economics, culture)
   - Skip `source_analyze` for utility queries (weather, code docs, recipes)
   - Use `deep=true` when source analysis is critical

**See [`extras/system-prompt-guide.md`](extras/system-prompt-guide.md) for the exact SOUL.md template with all required sections.**

### Epistemic Hierarchy — The Truth Chain

The agent must follow this order for every query:

1. **Immediate context** — What the user just said
2. **Perpetual Memory** — Past conversations on this topic
3. **Reference Library** — Curated, verified knowledge (authoritative)
4. **Web Research** — Only when local knowledge is insufficient
5. **Training data** — Last resort, always suspect

This is not optional. If the agent skips ahead to web search before checking the Reference Library, the system fails.

### Nightly Automation

The system runs autonomous jobs that maintain and improve itself. These are configured as cron jobs and should be set up after initial install:

- **PM Signal Scanner** (2:00 AM) — Scans for high-signal conversation clusters
- **Nightly Distillation** (3:00 AM) — Processes clusters through Synthesis → Audit → Commit
- **RL Growth** (3:00 AM) — Expands Reference Library based on gaps
- **Logos Intelligence Scout** (4:00 AM) — Builds source intelligence dossiers
- **System Backup** (4:00 AM) — Backs up data to Windows host

See [`WHITEPAPER.md`](WHITEPAPER.md) Section 4.6 and Section 8 for full details.

### Read the White Paper

The README gives you the structure. [`WHITEPAPER.md`](WHITEPAPER.md) gives you the *why*. Read it before configuring SOUL.md — it explains the epistemic framework, the Sovereign Sieve, and the design philosophy that makes everything work together.

---

## Quick Start — Full Setup Guide

Follow these steps to set up Logos with perpetual memory, reference library, and skills. Each step builds on the previous one.

### Step 1: Install

```bash
# Clone
git clone https://github.com/cluricaun28/logos.git
cd logos

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
├── entities/             ← People, organizations, publications
│   └── README.md         ← Instructions for building entity pages
└── sources/              ← Source intelligence dossiers (auto-created by source_analyze)
    └── state.gov.md      ← Example: domain, alignment, truthful_on, omits
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
│  Logos Core                                 │
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
│  │  - Deep Research Pipeline    │           │
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
- **Sources:** Auto-created by `source_analyze` when new domains are encountered. Each dossier tracks alignment, truthful_on, omits in `~/.hermes/reference-library/sources/`. Compounds over time — each analysis enriches the dossier.
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
- **Analyzes web sources** for bias and omissions via `source_analyze` (mandatory for substantive topics)
- **Auto-creates source dossiers** for new domains encountered during research
- **Creates new RL entries** when learning something new
- **Archives completed turns** to keep context window lean

### What You Should Do
- **Customize SOUL.md** with your values, preferences, and communication style
- **Review saved memories/skills periodically** — the agent will ask if anything should be saved
- **Keep config.yaml updated** as you add tools or services (SearXNG, Firecrawl, etc.)

---

## Maintenance

### Upstream Changes

Logos is a fully independent project, detached from upstream Hermes Agent on 2026-05-11. The upstream remote is retained for selective cherry-picking of useful improvements:

```bash
# Fetch latest from upstream
git fetch upstream main

# Review changes before selectively applying
git log upstream/main --oneline -10

# Apply specific commits that add value
git cherry-pick <commit-hash>
```

**Do NOT merge blindly** — custom plugin files may conflict with upstream changes. Cherry-pick selectively and test after each change. For a documented record of cherry-picked commits and the rationale, see [`docs/upstream_tracking.md`](docs/upstream_tracking.md).

Other projects (OpenClaw, Claude Code, Codex) may also yield useful patterns. The same approach applies: review, cherry-pick what's useful, test, commit.

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

|| Problem | Solution |
|---------|----------|
| Agent doesn't use perpetual memory tools | Check SOUL.md has "Knowledge Architecture" and "Active Retrieval" sections |
| Plugin not loading | Verify `perpetual_context` is in `config.yaml` plugins.enabled list |
| Semantic search not working | Install `onnxruntime sentence-transformers` |
| Context window too full | Check rolling window config, verify archiving is enabled |
| Agent loops on same task | SOUL.md should have "Anti-Loop Discipline" section; agent checks recent_messages first |
| `source_analyze` returns "unknown" alignment | Normal for new domains — dossiers compound over time. Use `deep=true` for substantive topics. |
| `source_analyze` deep mode fails | Check Firecrawl is running (`curl -s localhost:3002/health`). If Camofox `/tabs/create` 404, skip to `browser_navigate` + `browser_console`. |

---

## What's Not Here (By Design)

This project does **not** include:
- Personal data, tokens, or credentials — those live in your local `~/.hermes/` directory
- Reference library content — starts as a template, grows through use (including `sources/` dossiers created by `source_analyze`)
- Skill definitions — starts with examples, grows as you solve problems
- SQLite database files — created on first run, persists locally

**The philosophy:** Ship the system that builds knowledge, not the knowledge itself. Your agent should grow its own reference library and skills through your usage patterns.

---

## Extras Directory

|| File | Purpose |
|------|---------|
| [`system-prompt-guide.md`](extras/system-prompt-guide.md) | Exact SOUL.md sections needed for perpetual memory to work |
| [`soul-template.md`](extras/soul-template.md) | Ready-to-use SOUL.md template with all system prompt additions |
| [`deep-research-setup.md`](extras/deep-research-setup.md) | SearXNG + Firecrawl setup guide for web research |
| [`reference-library-template/`](extras/reference-library-template/) | Empty RL structure to copy as your starting point |
| [`skills-template/`](extras/skills-template/) | Example skill showing proper format and structure |

---

## License

MIT. All custom additions are MIT licensed.

---

**Provenance:** Detached from [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) on 2026-05-11.  \
**Project:** [cluricaun28/Logos](https://github.com/cluricaun28/Logos) — fully independent, selective cherry-picking only.
