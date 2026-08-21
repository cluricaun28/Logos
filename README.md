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
- **Discernment Workflow** — Skill-driven claim evaluation with explicit step-by-step reasoning chain (replaced sovereign sieve from Phase 1).
- **Skill-Driven Personas** — Pre-configured subagent roles (discernment-researcher, behavioral-tester, institutional-analyst, etc.) that auto-load the right skills and constraints.
- **sqlite-vec Single-DB Storage** — All 20k+ PM vectors and 33k+ RL vectors live inside their respective SQLite databases via vec0 virtual tables. Atomic storage, no index drift. FAISS removed July 2026.
- **Schema Versioning** — Both databases use `PRAGMA user_version` for migration tracking (PM: v3 — tool-role rows persisted since 2026-08, RL: v1)
- **Daily Maintenance** — Scheduled VACUUM + REINDEX + integrity checks run at 2:00 AM to prevent degradation
- **Recency Weighting** — Recent messages (7 days) get 1.5x score boost, 30 days get 1.2x. No decay — old messages don't lose score, recent ones gain it.
- **Nightly Learning Loop** — Scheduled jobs run deep research, apply frame-stripping, and distill findings into the Reference Library. The knowledge base grows through use.
- **Pinned Project Briefs** — Ask your agent to "pin" a long-running project. Its objective, status, and next steps are held in the system prompt across context-window archiving and session resets — and toggle on/off mid-session without a restart. One active pin per agent; briefs stay short (details live in a state file on disk).
- **Agent-to-Agent Messaging** — Each gateway exposes a keyed A2A HTTP endpoint, so a fleet of agents can message and coordinate with each other directly (no human relay).
- **Hardened Subagent Delegation** — Per-task timeout overrides, detection of children that "answer" with a stale intent line, and resume hints with partial output on timeout, so delegated work is never silently lost.

The system runs locally. No cloud APIs for memory or retrieval.

---

### 📄 White Paper
For a comprehensive architectural deep-dive covering all subsystems, the epistemic framework, and the design philosophy in detail, see [WHITEPAPER.md](WHITEPAPER.md).

---

## 🧠 Research Priority Matrix (Epistemic Routing)
The agent uses a tiered priority system to determine where to find truth:
- **Bedrock Topics (Worldview, Theology, Core Logic):** RL $\rightarrow$ PM $\rightarrow$ Web. The Reference Library is the ultimate authority; web search is only for filling gaps.
- **Volatile Topics (News, Tech Specs, Pricing):** Web $\rightarrow$ RL. Fresh data is prioritized over archived knowledge.

---

## For Agent Operators — Critical Setup Details

**Before following the Quick Start, read this if you are an AI agent tasked with setting up Logos.**

### SOUL.md — The Most Critical File

The code provides the plumbing. SOUL.md tells the agent *how to use it*. Without these specific sections, the plugins load but the agent never calls them:

1. **Knowledge Architecture** — Instruct the agent to check RL before web search, treat RL as authoritative
2. **Worldview Baseline** — Define the user's epistemic framework and tone preferences
3. **Operational Discipline** — Anti-loop rules, memory save rules, skill loading protocol
4. **Active Retrieval** — When to call `source_analyze`, when to use `deep=true`

**See [`extras/system-prompt-guide.md`](extras/system-prompt-guide.md) for the exact template.**

### Epistemic Hierarchy — The Truth Chain

The agent must follow this order for every query:

1. **Immediate context** → 2. **Perpetual Memory** → 3. **Reference Library** → 4. **Web Research** → 5. **Training data** (last resort)

### Nightly Automation

Scheduled jobs maintain and grow the system. See WHITEPAPER.md Section 4.6 for details.

**For the epistemic framework, discernment workflow, and design philosophy, see [WHITEPAPER.md](WHITEPAPER.md).**

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
