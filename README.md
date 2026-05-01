# Hermes Agent Fork — The Logos Engine

**The Logos Engine is a modification of the Hermes Agent designed to move from 'chatting' to 'knowledge management.'**

Most AI agents are stateless—they forget who you are and what you've decided once a conversation ends. This fork transforms the agent into a sovereign knowledge system that builds a permanent, local library of truth.

### The Core Philosophy
- **Sovereignty:** All memory and knowledge live on your hardware. No cloud sync, no external API dependencies for recall.
- **Truth Hierarchy:** The system prioritizes curated, verified facts (Reference Library) over the volatile noise of the open web.
- **Continuity:** It remembers not just what was said, but the *decisions* made and the *tasks* deferred, ensuring work continues exactly where it left off across sessions.

---

## What This Fork Adds
     8|
     9|This fork transforms Hermes Agent from a stateless chat interface into an agent with **persistent memory**, **curated knowledge**, and **proactive retrieval**. The key additions:
    10|
    11|### 1. Perpetual Context Database (`agent/perpetual_context_db.py`)
    12|SQLite-backed conversation archive with FTS5 full-text search and optional semantic embeddings (ONNX/SentenceTransformers). Every turn is stored locally — no cloud sync, no external API calls. Hybrid retrieval: keyword + vector similarity in a single query.
    13|
    14|### 2. Perpetual Context Plugin (`plugins/memory/perpetual_context/`)
    15|Full plugin suite providing perpetual memory tools to the agent:
    16|- **Hybrid search** — `perpetual_search` (FTS5 + semantic), `query_messages` (SQL-style filtering), `get_messages` (exact pattern matching)
    17|- **Smart retrieval** — auto-routes queries to optimal strategy (recent, topic-specific, decision trace, file history)
    18|- **Context bridge builder** — extracts active tasks, errors, and decisions for injection at archival boundaries
    19|- **Logos Engine (Deep Research & Continuity)** — Sovereign knowledge acquisition pipeline:
    20|    - **Three-Tier Web Stack:** SearXNG (Discovery) $ightarrow$ Firecrawl (Extraction) $ightarrow$ Camofox (Anti-detection Browser).
    21|    - **Epistemic Filtering:** Integrated scrutiny gate that filters raw web data through a user-defined worldview baseline before RL ingestion.
    22|    - **Adaptive Retrieval Cascade:** A reasoning-driven flow (Immediate Context $ightarrow$ PM Recall $ightarrow$ RL Authority $ightarrow$ Deep Research) to ensure the most accurate source is used for every query.
    23|
    24|### 3. Rolling Window Context Archiving (`plugins/context_engine/rolling_window/`)
    25|Replaces passive context retention with active archiving: compress completed conversation turns to permanent storage while keeping the working window lean for deep present-moment reasoning. State continuity through retrieval, not retention.
    26|
    27|Key components and optimizations:
    28|- **`task_tagger.py`** — Task detection engine with precompiled regex patterns (module-level constants), 10-message detection window, and role-aware outcome extraction
    29|- **`task_pruner.py`** — O(n) pruning via index-based lookups instead of list slicing; deduped summary logic through shared `build_task_summary()`
    30|- **`test_task_pruning.py`** — Full test suite covering task tagging (10 tests), pruning strategies (8 tests), and edge cases (4 tests) = 22 total
    31|
    32|### Modified Core Files (7 files)
    33|
    34|These are the upstream Hermes Agent files we changed. The diffs are committed in this repo — you can see exactly what changed with `git diff upstream/main -- <file>`. Key changes:
    35|
    36|| File | What Changed | Why It Matters |
    37||------|-------------|----------------|
    38|| `run_agent.py` | Renamed "compression" $ightarrow$ "archiving", added Context Bridge injection at archival boundary, selective tool loading | Enables rolling window + perpetual memory integration. Config key is now `archiving:` instead of `compression:` |
    39|| `agent/prompt_builder.py` | Skills section changed from mandatory to on-demand loading with validation | Prevents context bloat — only loads skills actually relevant to the task |
    40|| `agent/context_engine.py` | Integrated rolling window engine, context bridge injection at archival boundary | Core archiving logic that preserves active tasks across compression boundaries |
    41|| `plugins/context_engine/__init__.py` | Added rolling window engine loader with config passthrough | Pluggable context archiving strategy |
    42|| `model_tools.py` | Added `get_selective_tool_definitions()` and deferred tools index | Essential tools loaded inline, deferred tools listed for RL lookup — saves context tokens |
    43|| `cli.py` | Perpetual memory CLI commands (`hermes pm search`, etc.) | Query your conversation history from the terminal |
    44|| `tools/skill_manager_tool.py` | Fork-aware skill path resolution | Skills find custom categories correctly |
    45|| `.gitignore` | Patterns for plugin artifacts and cache files | Keeps git clean |
    46|
    47|**Config note:** The config key changed from `compression:` to `archiving:`. Your `config.yaml` should use:
    48|```yaml
    49|archiving:
    50|  enabled: true
    51|  threshold: 0.50
    52|  target_ratio: 0.20
    53|  protect_last_n: 20
    54|```
    55|(Old `compression:` key still works for backward compatibility.)
    56|
    57|---
    58|
    59|## 🧠 Research Priority Matrix (Epistemic Routing)
    60|The agent uses a tiered priority system to determine where to find truth:
    61|- **Bedrock Topics (Worldview, Theology, Core Logic):** RL $ightarrow$ PM $ightarrow$ Web. The Reference Library is the ultimate authority; web search is only for filling gaps.
    62|- **Volatile Topics (News, Tech Specs, Pricing):** Web $ightarrow$ RL. Fresh data is prioritized over archived knowledge.
    63|
    64|---
    65|
    66|## Quick Start — Full Setup Guide
    67|
    68|Follow these steps to set up the fork with perpetual memory, reference library, and skills. Each step builds on the previous one.
    69|
    70|### Step 1: Install the Fork
    71|
    72|```bash
    73|# Clone the fork
    74|git clone https://github.com/cluricaun28/hermes-agent.git
    75|cd hermes-agent
    76|
    77|# Install in development mode
    78|pip install -e ".[dev]"
    79|
    80|# Run Hermes setup to configure model, gateway, and plugins
    81|hermes setup
    82|```
    83|
    84|### Step 2: Enable Perpetual Memory Plugin
    85|
    86|Add to your `~/.hermes/config.yaml`:
    87|
    88|```yaml
    89|plugins:
    90|  enabled:
    91|    - perpetual_context
    92|```
    93|
    94|The plugin initializes the SQLite database (`~/.hermes/perpetual_context.db`) on first run. No manual DB creation needed.
    95|
    96|### Step 3: Set Up Your Persona (SOUL.md)
    97|
    98|**This is the most critical step.** The code provides infrastructure — your SOUL.md tells the agent *how to use it*. Without these prompt sections, the plugins load but the agent never calls them proactively.
    99|
   100|Copy the template and customize it:
   101|
   102|```bash
   103|cp extras/soul-template.md ~/.hermes/SOUL.md
   104|```
   105|
   106|Then edit `~/.hermes/SOUL.md`:
   107|1. Replace all `[YOUR NAME]` with your actual name
   108|2. Customize the **Worldview Baseline** section with your values and beliefs
   109|3. Customize the **Tone & Style** section with your preferred communication style
   110|4. Keep all **Knowledge Architecture**, **Operational Discipline**, and **Active Retrieval** sections — these are what make perpetual memory work
   111|
   112|**For detailed explanations of each system prompt section, see [`extras/system-prompt-guide.md`](extras/system-prompt-guide.md).**
   113|
   114|### Step 4: Initialize Your Reference Library
   115|
   116|The Reference Library is your agent's curated knowledge base. Start with the template structure:
   117|
   118|```bash
   119|# Copy the template to create your reference library skeleton
   120|cp -r extras/reference-library-template ~/.hermes/reference-library
   121|```
   122|
   123|This creates:
   124|```
   125|~/.hermes/reference-library/
   126|├── index.md              ← Master index (update as you add entries)
   127|├── topics/               ← System docs, workflows, research
   128|│   └── context-window-management.md  ← Example entry
   129|├── tools/                ← Tool schemas and usage guides
   130|│   └── tool-system.md    ← Explains how to document tools here
   131|└── entities/             ← People, organizations, publications
   132|    └── README.md         ← Instructions for building entity pages
   133|```
   134|
   135|**How it grows:** Your agent will automatically create new RL entries as you work. When researching a topic, the agent documents findings in `topics/`. When encountering tools, it schemas them in `tools/`. The index stays current because the agent updates it.
   136|
   137|### Step 5: Understand Skills On-Demand Loading
   138|
   139|Skills are reusable procedures stored in `~/.hermes/skills/`. They load on demand — only when relevant to your current task. This keeps the context window lean.
   140|
   141|The template shows the structure:
   142|```
   143|extras/skills-template/
   144|├── README.md                    ← How skills work, frontmatter format
   145|└── devops/
   146|    └── codebase-backup/         ← Example skill
   147|        └── SKILL.md             ← Complete example of a well-structured skill
   148|```
   149|
   150|**How it works:** The system prompt includes a list of available skill names and descriptions. Before replying, the agent scans this list and loads only skills directly relevant to your task via `skill_view(skill_name)`.
   151|
   152|### Step 6 (Optional): Set Up Deep Research
   153|
   154|For web search with source extraction (SearXNG + Firecrawl), see [`extras/deep-research-setup.md`](extras/deep-research-setup.md). This enables the agent to:
   155|- Search the web via a local meta-search engine
   156|- Extract full content from URLs using Firecrawl
   157|- Store research results in Perpetual Memory with source tracking
   158|
   159|---
   160|
   161|## Architecture Overview
   162|
   163|```
   164|┌─────────────────────────────────────────────┐
   165|│  Hermes Agent (upstream core)               │
   166|│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
   167|│  │   CLI    │  │ Gateway  │  │  Tools   │  │
   168|│  └────┬─────┘  └────┬─────┘  └────┬─────┘  │
   169|│       └──────────────┼─────────────┘        │
   170|│                      ▼                      │
   171|│  ┌──────────────────────────────┐           │
   172|│  │  Context Engine (modified)   │           │
   173|│  │  - Rolling window archiving  │           │
   174|│  │  - Context Bridge injection  │           │
   175|│  └──────────────┬───────────────┘           │
   176|│                 ▼                            │
   177|│  ┌──────────────────────────────┐           │
   178|│  │  Perpetual Context Plugin    │           │
   179|│  │  - Hybrid search (FTS5+vec)  │           │
   180|│  │  - Smart retrieval routing   │           │
   181|│  │  - Logos Engine              │           │
   182|│  └──────────────┬───────────────┘           │
   183|│                 ▼                            │
   184|│  ┌──────────────────────────────┐           │
   185|│  │  SQLite Database (local)     │           │
   186|│  │  - messages table + FTS5 idx │           │
   187|│  │  - embeddings BLOB column    │           │
   188|│  │  - topic relationships       │           │
   189|│  └──────────────────────────────┘           │
   190|└─────────────────────────────────────────────┘
   191|```
   192|
   193|### How the Dual-Memory System Works
   194|
   195|```
   196|User asks question → Agent calls recent_messages (last 10 turns)
   197|→ Agent reads current prompt + recent context
   198|→ Agent checks Reference Library for curated knowledge on topic
   199|→ Agent calls perpetual_search for historical discussion on topic
   200|→ Agent formulates response using retrieved info + present context
   201|→ Completed turns are archived to permanent storage
   202|→ Context window stays lean for next turn
   203|```
   204|
   205|Key design decisions:
   206|- **`recent_messages` is mandatory first step** — prevents loops, maintains continuity
   207|- **Search during thinking, not after** — tool use IS the reasoning process
   208|- **Archive aggressively** — if a task is done, it belongs in permanent storage, not working memory
   209|- **Print task status every turn** — `[Tasks: 3/5 complete]` becomes searchable PM data
   210|
   211|---
   212|
   213|## Semantic Embeddings (Optional)
   214|
   215|For hybrid search with vector similarity alongside FTS5 keyword search:
   216|
   217|```bash
   218|pip install onnxruntime sentence-transformers
   219|```
   220|
   221|The default model is `all-MiniLM-L6-v2` — lightweight, runs locally. Embeddings are stored as BLOB in SQLite alongside FTS5 indexes. No configuration needed beyond installing the packages; the plugin auto-detects and enables them.
   222|
   223|---
   224|
   225|## How Things Grow Over Time
   226|
   227|### Reference Library Growth
   228|- **Topics:** Created when you research something or solve a complex problem. The agent documents findings in `~/.hermes/reference-library/topics/`.
   229|- **Tools:** Created when the agent encounters deferred tools. Schemas are documented in `~/.hermes/reference-library/tools/` for future lookup.
   230|- **Entities:** Created when researching people, organizations, or publications. Tracks credibility and behavior patterns in `~/.hermes/reference-library/entities/`.
   231|- **Index:** Updated automatically by the agent as new entries are created.
   232|
   233|### Skills Growth
   234|- Skills are created when you solve a complex problem (5+ tool calls) that you'll likely face again.
   235|- Stored in `~/.hermes/skills/CATEGORY/skill-name/SKILL.md`.
   236|- The system prompt's skills list grows as you add more skills.
   237|- See [`extras/skills-template/`](extras/skills-template/) for format examples.
   238|
   239|### Perpetual Memory Growth
   240|- Every conversation turn is stored automatically — no action needed.
   241|- Grows continuously across sessions. Search it with `perpetual_search`, `query_messages`, or `recent_messages`.
   242|- Old turns are pruned from the context window but remain fully searchable in the database.
   243|
   244|---
   245|
   246|## Daily Usage
   247|
   248|### Starting a Session
   249|```bash
   250|hermes gateway start    # Start the agent gateway
   251|hermes                  # Open interactive session
   252|```
   253|
   254|The agent will:
   255|1. Load your SOUL.md (persona + behavioral rules)
   256|2. Check `recent_messages` for session continuity
   257|3. Scan available skills list (on-demand loading, not pre-loaded)
   258|4. Wait for your input
   259|
   260|### What the Agent Does Automatically
   261|- **Checks recent conversation history** before taking action (anti-loop discipline)
   262|- **Consults Reference Library** before answering factual questions
   263|- **Searches Perpetual Memory** when topics reference past work
   264|- **Loads skills on demand** only when relevant to your task
   265|- **Creates new RL entries** when learning something new
   266|- **Archives completed turns** to keep context window lean
   267|
   268|### What You Should Do
   269|- **Customize SOUL.md** with your values, preferences, and communication style
   270|- **Review saved memories/skills periodically** — the agent will ask if anything should be saved
   271|- **Keep config.yaml updated** as you add tools or services (SearXNG, Firecrawl, etc.)
   272|
   273|---
   274|
   275|## Maintenance
   276|
   277|### Updating from Upstream
   278|
   279|This fork tracks upstream via the `upstream` remote:
   280|
   281|```bash
   282|# Fetch latest from NousResearch
   283|git fetch upstream main
   284|
   285|# Review changes before cherry-picking
   286|git log upstream/main --oneline -10
   287|
   288|# Selectively apply commits
   289|git cherry-pick <commit-hash>
   290|```
   291|
   292|**Do NOT merge blindly** — custom plugin files may conflict with upstream context engine changes. Cherry-pick selectively and test after each change.
   293|
   294|### Backup Your Data
   295|
   296|Your data lives in three places:
   297|```bash
   298|# Perpetual Memory database (conversation history)
   299|~/.hermes/perpetual_context.db
   300|
   301|# Reference Library (curated knowledge)
   302|~/.hermes/reference-library/
   303|
   304|# Skills (reusable procedures)
   305|~/.hermes/skills/
   306|```
   307|
   308|Back them up regularly:
   309|```bash
   310|# Quick backup script
   311|tar czf hermes-data-$(date +%Y%m%d).tar.gz \
   312|  ~/.hermes/perpetual_context.db \
   313|  ~/.hermes/reference-library/ \
   314|  ~/.hermes/skills/ \
   315|  ~/.hermes/SOUL.md \
   316|  ~/.hermes/config.yaml
   317|```
   318|
   319|### Troubleshooting
   320|
   321|| Problem | Solution |
   322||---------|----------|
   323|| Agent doesn't use perpetual memory tools | Check SOUL.md has "Knowledge Architecture" and "Active Retrieval" sections |
   324|| Plugin not loading | Verify `perpetual_context` is in `config.yaml` plugins.enabled list |
   325|| Semantic search not working | Install `onnxruntime sentence-transformers` |
   326|| Context window too full | Check rolling window config, verify archiving is enabled |
   327|| Agent loops on same task | SOUL.md should have "Anti-Loop Discipline" section; agent checks recent_messages first |
   328|
   329|---
   330|
   331|## What's Not Here (By Design)
   332|
   333|This fork does **not** include:
   334|- Personal data, tokens, or credentials — those live in your local `~/.hermes/` directory
   335|- Reference library content — starts as a template, grows through use
   336|- Skill definitions — starts with examples, grows as you solve problems
   337|- SQLite database files — created on first run, persists locally
   338|
   339|**The philosophy:** Ship the system that builds knowledge, not the knowledge itself. Your agent should grow its own reference library and skills through your usage patterns.
   340|
   341|---
   342|
   343|## Extras Directory
   344|
   345|| File | Purpose |
   346||------|---------|
   347|| [`system-prompt-guide.md`](extras/system-prompt-guide.md) | Exact SOUL.md sections needed for perpetual memory to work |
   348|| [`soul-template.md`](extras/soul-template.md) | Ready-to-use SOUL.md template with all system prompt additions |
   349|| [`deep-research-setup.md`](extras/deep-research-setup.md) | SearXNG + Firecrawl setup guide for web research |
   350|| [`reference-library-template/`](extras/reference-library-template/) | Empty RL structure to copy as your starting point |
   351|| [`skills-template/`](extras/skills-template/) | Example skill showing proper format and structure |
   352|
   353|---
   354|
   355|## License
   356|
   357|Same as upstream: MIT. Custom additions are also MIT licensed.
   358|
   359|---
   360|
   361|**Upstream:** [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent)  
   362|**Fork:** [cluricaun28/hermes-agent](https://github.com/cluricaun28/hermes-agent)
   363|