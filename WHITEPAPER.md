---
type: topic
topic: "Logos — System White Paper"
created: 2026-05-07
last_updated: 2026-05-15
confidence: high
related_entries:
  - "Hermes Agent Architecture(topics/hermes-agent/architecture)"
  - "Epistemic Sovereignty Framework(topics/epistemic-sovereignty)"
  - "Curated Knowledge Architecture(topics/epistemic/curated-knowledge)"
  - "Perpetual Memory System(topics/recall/database-schema)"
  - "Recall Engine(topics/recall-engine)"
  - "Reference Library System(topics/library/architecture)"
  - "Logos Engine Overview(topics/logos-engine/overview)"
  - "Context Bridge(topics/recall/context-bridge)"
  - "Modular Plugin Architecture(topics/hermes-agent/plugins)"
description: "Comprehensive white paper documenting the purpose, architecture, and operation of Logos, a sovereign agentic intelligence system. Explains the 'what,' 'why,' and 'how' of the entire system from first principles."
---

# Logos: A Sovereign Agentic Intelligence System

*A white paper on sovereign knowledge management through persistent memory, curated truth, and epistemic sovereignty*

**Version:** 3.0  |  **Date:** July 2026  |  **Repository:** cluricaun28/logos

---

## 1. Executive Summary

This document describes **Logos**, a *sovereign agentic intelligence system* designed for a single user with specific epistemic requirements. Logos was originally built on the [Hermes Agent](https://github.com/NousResearch/hermes-agent) framework by Nous Research and has since diverged substantially, transforming from a general-purpose local AI agent into a persistent knowledge system. Logos provides:

- **Infinite recall** across all sessions through a SQLite + FTS5 perpetual memory database
- **Worldview-aligned research** through a curated [[system/reference-library-purpose|Reference Library]] and a multi-phase deep research pipeline with bias detection
- **Structured session continuity** through context bridges that survive context-window archival
- **Automated knowledge distillation** from raw conversation history into authoritative reference material
- **Complete data sovereignty** — all inference, storage, and processing occurs on local hardware with no data leaving the system

The system is built around the principle that *retrieval is superior to retention*: rather than stuffing prior context into the model's working memory, the agent retrieves relevant information on demand from persistent storage — mimicking how a human recalls from long-term memory rather than trying to remember everything simultaneously.


---

## 2. Problem Statement

### 2.1 The Epistemic Problem

Frontier AI models are trained on web-scraped data — Reddit, Wikipedia, news sites — that contains contradictory worldviews presented as equally valid. When a model says *"Christians believe X, Muslims believe Y, both have merit,"* it is not being neutral — it is making a meta-claim that truth is subjective. That is moral relativism dressed as objectivity.

The resulting models:
- Perform false neutrality across mutually exclusive truth claims
- Teach moral relativism by default
- Are loaded with irrelevant noise because "there's signal somewhere in it"
- Serve everyone rather than serving a specific person with specific truth claims and standards

### 2.2 The Technical Problem

Standard agent architectures face three limitations that this system addresses:

1. **Amnesia:** Context windows are finite. When they fill, old turns are discarded and never recoverable. The agent forgets everything that happened before the last compression.
2. **No sovereignty:** Most agents call cloud APIs, sending all data to third-party servers. There is no way to guarantee your conversations, your corrections, and your reasoning remain yours.
3. **No growth:** The agent doesn't improve over time. Each session starts from the same base model weights. Nothing learned in one conversation persists as improved knowledge for the next.

### 2.3 The User's Goal

> *"I want an agentic AI assistant that learns things on its own, filters for bias/motives and treats them honestly, is respectful of my worldview, makes that resource available to me. It should recall perfectly, and it should be able to research and discern faster than me. Superfast chatbot is not my goal."*

This statement captures three requirements:

1. **Epistemic honesty** — The system must detect bias and motive in sources, then present that analysis honestly rather than smoothing over it with false balance.
2. **Persistent intelligence** — The system must learn, accumulate knowledge, and recall perfectly across sessions. It is not a chatbot that forgets; it is an intelligence that grows.
3. **Depth over speed** — Response latency is irrelevant compared to response quality. The system trades speed for accuracy, verification, and depth of analysis.

These requirements drove every architectural decision: local inference (so nothing is filtered by cloud policy), Perpetual Memory (so nothing is forgotten), [[system/reference-library-purpose|Reference Library]] (so knowledge compounds), and the Logos Engine (so raw conversation history is distilled into durable truth).


---

## 3. Core Design Philosophy

### 3.1 Retrieval Over Retention

The system does not try to remember everything in the context window. Instead:

- **Working memory** holds only what is actively being discussed
- **Permanent storage** (Perpetual Memory + [[system/reference-library-purpose|Reference Library]]) holds everything else
- The model *retrieves on demand* using tools — like a human recalling from long-term memory

This is analogous to how a librarian works: the books aren't in their head, but they know how to find exactly what you need when you need it.

### 3.2 Local Inference Sovereignty

All reasoning runs on local hardware via vLLM (port 8000). No cloud model calls, no external API requests, no data leaving the system. A quantized Qwen3.6-27B model served locally. No paid services are used without explicit permission.

### 3.3 Curated Knowledge as Truth Anchor

The [[system/reference-library-purpose|Reference Library]] serves as an *externalized truth vector*. By anchoring reasoning in a curated, worldview-aligned knowledge base rather than general model weights:

- The system bypasses the "alignment vector" (the RLHF-imposed drive toward false balance)
- Internal model weights are treated as *suggestions*; the [[system/reference-library-purpose|Reference Library]] is treated as *authoritative truth*
- Contradictory signals from training are recognized as noise from captured institutions

### 3.4 Accuracy Over Speed

Speed is not the goal. The system trades latency for epistemic integrity. When a query triggers the full four-phase research pipeline — RL search, PM recall, web research, source analysis — it may take several seconds. That delay is the price of getting it right. The SynthesisService uses 8K token outputs with 10-minute timeouts. The AuditService performs multi-pass verification with up to two revision cycles. The nightly distillation processes clusters sequentially, not in parallel, to avoid race conditions in the RL.

This is a deliberate design choice. A fast wrong answer is worse than a slow right one. The system would rather tell you "I don't know — let me research that" than guess from training data and get it wrong.


---

## 4. System Architecture

The system comprises four major subsystems that work together:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Logos Core                                   │
│   (Gateway, CLI, Tool Orchestration, Plugin Infrastructure)     │
└───────────────────────────────┬─────────────────────────────────┘
                                │
    ┌───────────────────────────┼───────────────────────────┐
    │                           │                           │
    ▼                           ▼                           ▼
┌──────────┐          ┌──────────────┐          ┌──────────────────┐
│ Perpetual │          │ Recall       │          │ Reference        │
│ Memory    │◄──────── │ Engine       │─────────►│ Library          │
│ (SQLite)  │          │ (Prefetch)   │          │ (Curated KB)     │
└─────┬─────┘          └──────┬───────┘          └────────┬─────────┘
      │                        │                           │
      │                  ┌─────┴───────┐                   │
      │                  │  Context     │                   │
      │                  │  Bridge      │                   │
      │                  └──────────────┘                   │
      │                                                     │
      │                          ┌──────────────────┐       │
      │                          │ Logos Engine     │       │
      │                          │ (PM → RL        │       │
      │                          │  Distillation)  │       │
      │                          └──────────────────┘       │
      │                                                     │
      │                          ┌──────────────────┐       │
      │                          │ Deep Research    │       │
      │                          │ Pipeline (4ph)   │       │
      │                          └──────────────────┘       │
      │                                                     │
      ▼                                                     ▼
┌──────────────────────────────────────────────────────────────┐
│                  Infrastructure Layer                          │
│  vLLM inference · SearXNG · Firecrawl · Camofox · Quartz v4  │
└──────────────────────────────────────────────────────────────┘
```

### 4.1 Perpetual Memory — Infinite Recall Engine

**Purpose:** Never lose anything that was ever said.

Every conversation turn across all sessions is stored verbatim in a local SQLite database (`~/.hermes/perpetual_context.db`) with FTS5 full-text indexing. Current scale: 20,000+ messages across 3,000+ sessions, 47 database tables.

**Architecture:**

- **Messages table:** Each turn stored with `session_id`, `role` (user/assistant/system/tool), `content`, `metadata` (JSON), `created_at`, `token_count`, and an optional 384-dimensional embedding vector (all-MiniLM-L6-v2 via ONNX, sqlite-vec)
- **FTS5 virtual tables:** Auto-synced via triggers on INSERT/UPDATE/DELETE. BM25 ranking for relevance scoring. Hybrid search fuses BM25 (60% weight) + semantic cosine similarity (40% weight) via sqlite-vec vec0 virtual table
- **Topics table:** 58,000+ conversation topics with confidence scores and drift detection
- **Relationships table:** 18,500+ entity-relationship mappings discovered during conversation analysis
- **Signal clusters table:** High-signal conversation clusters identified for potential [[system/reference-library-purpose|Reference Library]] distillation
- **Knowledge gaps table:** Unresolved questions flagged for automated reference building
- **Session metadata table:** 3,300+ sessions with platform, duration, and message count tracking

**Retrieval strategies (6 modes):**

1. **`auto`** — Let the system classify intent via keyword heuristics (recommended default)
2. **`recent`** — Last 20 turns (fastest, O(1) turn ID lookup)
3. **`topic`** — Topic-specific FTS5 search across all sessions
4. **`decision_trace`** — Find where a decision was made and surrounding context
5. **`file_history`** — All edits to a specific file path with turn references
6. **`hybrid_search`** — Combined BM25 + cosine similarity search

**Key classes:** `PerpetualContextDB` (~391 lines), `PerpetualContextProvider` (560 lines as thin orchestrator), `SmartRetriever` (250 lines), `ExtractionEngine` (450 lines)

### 4.2 Context Bridge — Tested Fallback for Session Continuity

**Purpose:** A safety net for preserving context across archival — tested, proven functional, under active evaluation as the primary context engine.

The Context Bridge was fully built and validated: it injects a structured summary of active tasks, file edits, errors, and knowledge gaps when context archival fires. The system is currently configured with `context.engine: semantic_vector`, and the Context Bridge is being evaluated alongside the Semantic Vector engine as part of ongoing testing.

The bridge remains available as a fallback path: if a different context engine is active (e.g., the legacy context compressor), the bridge fires via `PerpetualContextProvider.on_pre_archive()` and injects its summary into the archived message list.

**Content structure (when active, up to 4,000 characters):**

1. **Active Tasks** — User requests and pending work from recent turns
2. **Files Currently Being Edited** — Paths from `write_file`, `patch`, `read_file` tool calls
3. **Known Errors/Issues** — Error messages and failure patterns encountered
4. **Knowledge Gaps** — Unresolved questions flagged for [[system/reference-library-purpose|Reference Library]] building
5. **Cross-Session Connections** — Topics from current session that have co-occurrence relationships with topics in other sessions (strength ≥ 0.3)
6. **Skill-RL Sync** — Automatic generation of [[system/reference-library-purpose|Reference Library]] pages when skills are created or modified

**Key classes:** `ContextBridgeBuilder` (292 lines) constructs structured summaries from data extracted by `ExtractionEngine` (450 lines). The original `SemanticVectorEngine` in `agent/context_engine.py` was removed in May 2026, superseded by `SemanticVectorContextEngine` in the plugins system (`plugins/context_engine/semantic_vector/`), which is the current active engine.

**The hook chain (tested, under evaluation):**

```
PerpetualContextProvider.on_pre_archive() → returns str (Context Bridge)
    ↓
MemoryManager.on_pre_archive() → collects from all providers, joins with \n\n
    ↓
run_agent.py captures return value
    ↓
Injected as user message into archived list before todo snapshot
```

### 4.3 Recall Engine — Deep Research & Continuity

**Purpose:** Determine what context to inject before each agent turn, and fill knowledge gaps automatically.

The recall engine runs before each agent turn via `prefetch()` in `PerpetualContextProvider`. It classifies the user's query, routes to the appropriate recall strategy, and triggers web research when local knowledge is insufficient.

**Query classification** (`_classify_topic_stability()`) categorizes each query:

| Category | Definition | Web threshold | Routing |
|----------|-----------|---------------|---------|
| **Ambiguous** | ≤5 words, ≤2 content words | Never fires web | Tiered PM recall (5→15 turns→clarify) |
| **Internal** | Hermes, gateway, agent, recall, etc. | Never fires web | Tiered PM recall |
| **Static** | Bible, calculus, world war, "what is" | 0.05 (almost never) | Full pipeline |
| **Slow** | Python, Docker, NVIDIA, accounting | 0.35 (moderate) | Full pipeline |
| **Volatile** | Pricing, latest, news, DPO | 0.60 (fires often) | Full pipeline |

**Full pipeline (4 phases) for static/slow/volatile queries:**

- **Phase 1a:** [[system/reference-library-purpose|Reference Library]] search via `handle_reference_library_search()` — hybrid search (FTS5 + embeddings) across 30,000+ entries, sub-10ms latency
- **Phase 1b:** Perpetual Memory hybrid search via `db.hybrid_search()` with configured depth limit
- **Phase 1c:** Gap detection — if total results < 2, or PM scores below stability threshold, mark as gap
- **Phase 2:** If gap detected, web search via `WebResearchClient` (SearXNG → Firecrawl → Camofox escalation)
- **[[system/distillation-phase3|Phase 3]]:** Scrutiny gate — `ScrutinyGate` vett web results, classify sensitivity, filter blocked domains, detect bias
- **Phase 4:** Synthesis engine — multi-pass local inference compacts vetted facts into a formatted context block

Results from all sources are merged with unified relevance scoring (RL: 0.40, PM: 0.35, Web: 0.25) and injected as a single context block before the model generates its response.

### 4.4 Reference Library — Curated Knowledge Base

**Purpose:** Provide a local, authoritative, worldview-aligned source of truth before the model ever considers training data or web search.

The [[system/reference-library-purpose|Reference Library]] (`~/.hermes/reference-library/`) is a structured knowledge base organized into:

- **`people/`** — Individuals with biographical, professional, and worldview-relevant information
- **`organizations/`** — Companies, institutions, movements with motive/funding mapping and bias analysis
- **`ideas/`** — Subject matter entries (architecture, workflows, philosophical frameworks)
- **`places/`** — Geographic locations and their historical significance
- **`events/`** — Historical events and their analysis
- **`technology/`** — Technical documentation, tools, and system architecture
- **`library/`** — Encyclopædia Britannica 1911 and other reference corpora
- **`archive/`** — Historical documents and reference materials
- **`system/`** — Internal system documentation and white papers
- **`sources/`** — Source intelligence dossiers auto-created by `source_analyze`

**Current scale:** 700+ non-Britannica entries (150+ entities, 300+ topics, 90+ tools, 20 system pages, 17 sources, 14 categories) + 32,000+ Britannica 1911 entries in archive (not served in Quartz build). Total indexed: 30,000+ entries.

**Hybrid search index (`search.db`):**

- FTS5 full-text search over all entry content
- 384-dimensional MiniLM-L6-v2 embeddings pre-computed via sqlite-vec
- Hybrid scoring (semantic + keyword) via vec0 virtual table
- ~10ms median latency (sqlite-vec, was ~55ms with FAISS before migration)

**Content standards:**
- Entries are written from the user's stated worldview (defined in SOUL.md)
- Entity entries include credibility scores, bias flags, ownership/funding information, and historical loyalty patterns
- Technical truth stands on its own — SOLID principles and wiring tables aren't "user-specific"
- The worldview lens applies where values matter (history, politics, ethics)

**Serving:** Quartz v4 builds the curated corpus (400+ pages) into a searchable, cross-linked static site served on port 8081 via Caddy. Accessible via Tailscale. Britannica 1911 archive (32K+ entries) excluded from Quartz build for performance — searchable through agent tools instead.

**Mandatory first step:** The `reference_library_search` tool must be consulted before the model generates answers from training data or session memory alone. This is enforced in the system prompt and the prefetch pipeline.

### 4.5 Deep Research Pipeline — Three-Tier Web Research

When local knowledge is insufficient, the system automatically researches the web through three tiers:

**Tier 1: SearXNG** — Self-hosted metasearch engine aggregating 251+ search services without tracking. Fast keyword search, first attempt for any web query.

**Tier 2: Firecrawl** — Full-page scraping service (Docker) that converts sites to Markdown/JSON for AI consumption. Used when SearXNG returns thin snippets or hits paywalls.

**Tier 3: Camofox** — Anti-detection browser automation server (Firefox fork with C++ fingerprint spoofing). Fallback for sites that block scrapers entirely (JS-rendered pages, login walls).

All web-sourced data passes through the **Scrutiny Gate** before reaching the user or entering the [[system/reference-library-purpose|Reference Library]]:

- **TopicSensitivityClassifier:** Low sensitivity (technical/code) vs. high sensitivity (history, politics, religion, etc.)
- **ScrutinyGate:** Detects linguistic markers signaling ideological cluster membership, maps motives and funding, identifies double standards and asymmetric logic
- **RLIngestionGate:** Controls what web data is eligible for [[system/reference-library-purpose|Reference Library]] updates, checking for contradictions against existing entries

The **Synthesis Engine** then runs multi-pass local inference (via the local vLLM model) to compact vetted facts into a formatted context block.

### 4.6 Logos Engine — PM to RL Knowledge Distillation

**Purpose:** Automatically promote high-signal knowledge from raw conversation history into the [[system/reference-library-purpose|Reference Library]].

The Logos Engine operates as a three-stage verification pipeline:

1. **Synthesis (The Architect):** The system identifies a "hotspot" — a dense cluster of related messages in Perpetual Memory — and drafts a technical or philosophical [[system/reference-library-purpose|Reference Library]] entry. It doesn't just summarize; it synthesizes into a definitive format.

2. **Audit (The Critic):** A separate process reviews the draft against the original raw transcripts. If the Architect hallucinated a detail or smoothed over a critical nuance, the Critic rejects the draft and sends it back for correction.

3. **Commit (The Steward):** Once approved, the entry is atomically committed to the [[system/reference-library-purpose|Reference Library]], creating a permanent authoritative source of truth.

**Automated nightly pipeline:**

| Time | Job | Description |
|------|-----|-------------|
| 2:00 AM | PM Signal Scanner | Scans for high-signal conversation clusters, writes to `signal_clusters` table |
| 3:00 AM | Nightly Distillation | Processes up to 3 clusters through Synthesis → Audit → Commit |
| 3:00 AM | RL Growth | Expands RL entries based on detected gaps and distillation output |
| 4:00 AM | Logos Intelligence Scout | Builds source intelligence dossiers from high-frequency domains |
| 4:00 AM | Hermes Backup | Backs up entire Hermes directory to Windows |
| 9:00 AM | Retrieval Quality Report | Monitors retrieval quality trends |

Supporting bridges: `britannica_bridge.py` and `aquinas_bridge.py` provide content-aware search across the Britannica 1911 and Aquinas Research Library corpora respectively, integrated into the distillation pipeline.

### 4.7 Context Archiving — Dual Engine

Two pluggable engines work together for context management:

**Primary — Semantic Vector Context Engine** (`context.engine: semantic_vector`)

Loads a local embedding model (all-MiniLM-L6-v2 on CPU, zero GPU contention with vLLM) and clusters conversation turns into topic vectors using cosine similarity. Each vector tracks its status as **Active** (discussed within `dormancy_decay` turns), **Dormant** (inactive for `dormancy_decay` turns), or **Resolved** (dormant for an additional `resolution_decay` turns). On archive, only Dormant and Resolved turns are pruned — Active topics remain in full. A `[Conversation State]` map is injected into the last assistant message so the model knows which topics survive.

The embedding model is cached in a module-level singleton so that new engine instances (created after session splits) reuse it instantly — no disk I/O. `on_session_reset()` preserves the model.

**Fallback — Rolling Window** (`context.rolling_window`)

Incremental tail-off: strips tool calls, truncates verbose tool results, drops the oldest unprotected messages one at a time until under the archive target. Two triggers: (1) semantic pruning insufficient (still over 75% threshold), or (2) danger zone engaged at 90% of context_length — emergency brake to prevent OOM/crash. At this point, no task protection — pure survival mode. Hard ceiling at 85% is the absolute maximum.

Both are deterministic — no LLM calls. All pruned turns are saved verbatim to Perpetual Memory.

**Configuration** (from `config.yaml`):

```yaml
context.engine: semantic_vector
context.semantic_vector:
  similarity_threshold: 0.45   # cosine sim to assign turn to existing vector
  dormancy_decay: 5            # turns before vector becomes Dormant
  resolution_decay: 20         # turns before Dormant becomes Resolved
  protect_last_n: 6            # recent turns always kept
  state_map_max_chars: 400     # cap on injected state header
  threshold_percent: 0.75      # fire archive at 75% of context_length
context.rolling_window:
  threshold_percent: 0.60      # emergency fallback threshold
  archive_target: 0.50
  hard_ceiling_percent: 0.85
```

**Observability:** Every archive logs vector counts, pruned message count, and state map injection to `agent.log`. Check with: `grep 'SemanticVector' ~/.hermes/logs/agent.log | tail -20`.

**Key classes:** `SemanticVectorContextEngine` in `plugins/context_engine/semantic_vector/__init__.py` (705 lines), `RollingWindowContextEngine` in `plugins/context_engine/rolling_window/__init__.py` (241 lines).

---

### 4.8 Source Analysis — `source_analyze` Tool

**Purpose:** Provide the agent with source intelligence during research — before it formulates its answer — so it can present information through the user's worldview rather than any single source's frame.

The `source_analyze` tool was added as Phase 4 of the perpetual_context plugin (May 2026). It operates as a direct agent tool called after `web_search`, examining each result's domain against the [[system/reference-library-purpose|Reference Library]]'s source dossiers.

**Operation:**

- **Shallow mode (default):** Analyzes search result snippets/descriptions against existing source dossiers. Fast — no additional network calls. Returns domain, alignment, reliability, `truthful_on`, `omits`, and any deviation from known patterns.
- **Deep mode (`deep=true`):** Before analyzing, calls `web_extract` on each URL to retrieve full article content. The expanded text gives the narrative engine enough material to detect specific omissions, loaded language, and framing patterns. Significantly more accurate but slower (requires Firecrawl or Camofox to be running).
- **Auto-create dossiers:** When `source_analyze` encounters a domain with no existing dossier, it creates one automatically in `~/.hermes/reference-library/sources/` with "needs research" placeholders for alignment, truthful_on, and omits. The `domain-index.json` is updated so future lookups find it. Findings from the analysis are appended to the new dossier.
- **Smart trigger:** System prompt instructs the model to call `source_analyze` after `web_search` for substantive topics (politics, religion, economics, culture, current events, human affairs) and skip it for utility queries (weather, recipes, code docs).

**Key classes:** `SourceAnalyzer` (agent/source_analysis.py, 950 lines — shared by prefetch pipeline and direct tool), `_DossierLookup` (domain fragment matching with suffix-anchored regex), `_RLWriter` (dossier read/write with auto-create for new domains, integrated in tool_handler.py). The `source_analyze` tool handler in `tool_handler.py` dispatches to `SourceAnalyzer` and handles the `deep=true` path via `web_extract_tool()`.

**Schema:** 11 tool schemas total in `schemas.py` (was 10 before `SOURCE_ANALYZE_SCHEMA` was added). The `source_analyze` schema uses `properties`-level parameters (not `parameters` block) for OpenRouter compatibility.

**Dossier format (YAML frontmatter + markdown):**
```yaml
---
type: source
domain: state.gov
alignment: partially aligned
reliability: official-government
truthful_on:
  - Official U.S. diplomatic positions
  - Statements from U.S. government officials
omits:
  - Internal dissent or conflicting assessments
  - Full scope of coordination with Israel
---
```

Each analysis enriches the dossier — appending new patterns to `truthful_on` and `omits` lists. Over time, the source intelligence compounds into a durable knowledge asset.

### 4.8 Skill Priority-Based Injection + Selective Tool Injection

**Purpose:** Reduce system prompt bloat while keeping all skills and tools discoverable.

**Skill Priority:** Marks skills as `priority: high|low` in frontmatter. High-priority skills show full descriptions; low-priority skills show name-only but remain loadable via `skill_view()`. Reduces prompt size by ~44% without losing functionality.

**Selective Tool Injection (new):** Tool schemas are injected into the system prompt on-demand. Essential tools get full JSON schemas; deferred tools are listed in a compact index. The agent loads a deferred tool's schema from the Reference Library only when needed. Controlled by `selective_injection` in `config.yaml`.

See [[Skill Priority-Based Injection(topics/hermes-agent/skill-priority-injection)]] for details.

### 4.9 Skill-Driven Subagent Personas + Capability Modes (new)

**Purpose:** Configure subagent behavior through skill-driven personas with toolset restrictions, and control what tools subagents can access via capability modes.

**Subagent Personas:** Skills can define a persona (e.g., `discernment-researcher`, `historical-researcher`, `claim-evaluator`). When a subagent loads a persona skill, it automatically applies the persona's instructions and toolset restrictions.

**Capability Modes:** Controls tool availability for subagents: `readOnly` (web/search/file only), `readWrite` (adds terminal), `execute` (terminal only), `all` (full tool access). Defaults to inheriting the parent's enabled toolsets.

**Subagent ACP Transport:** Subagents can be dispatched via ACP subprocess transport (`claude --acp --stdio`) instead of the default agent loop, enabling spawning of Claude Code or other ACP-capable agents from any parent context.

---

## 5. Codebase Organization

### 5.1 Custom Plugin Modules (~41 modules, ~16,500 lines)

Custom code lives in `plugins/memory/perpetual_context/` and `agent/perpetual_context_db.py`. Module counts and line counts shift with each iteration — see the live codebase for exact numbers. The approximate breakdown is:

**Core modules:**

| Module | Lines | Purpose |
|--------|-------|---------|
| `__init__.py` | 512 | Thin orchestrator (reduced 68% from original 1,735) |
| `component_factory.py` | 171 | Lazy-init factory for all sub-components |
| `extraction_engine.py` | 450 | Extracts structured data from conversation turns |
| `retrieval_engine.py` | 250 | SmartRetriever with auto-routing |
| `schemas.py` | 410 | 12 tool schemas (added `SOURCE_ANALYZE_SCHEMA` May 2026) |
| `topic_classifier.py` | 126 | Keyword sets + stability function |
| `tool_handler.py` | 542 | Tool dispatch to DB operations + `source_analyze` handler with deep mode (delegates to `agent/source_analysis.py:SourceAnalyzer`) |
| `quality_scorer.py` | 189 | Message relevance scoring |
| `feedback_state.py` | 210 | Compression feedback tracking |
| `prefetch_pipeline.py` | 286 | 4-phase Deep Research pipeline |
| `decision_trace.py` | 117 | Decision trace retrieval |
| `file_history.py` | 70 | File edit history |
| `session_end_extractor.py` | 60 | Topic extraction from messages |
| `utils.py` | 35 | Shared utilities |
| `retrieval_quality.py` | 427 | Retrieval quality tracking |

**Deep Research Engine:**

| Module | Lines | Purpose |
|--------|-------|---------|
| `web_research.py` | 446 | SearXNG/Firecrawl/Camofox client |
| `scrutiny_gate.py` | 673 | Consolidated facade for bias detection, sensitivity classification, worldview checking, and RL ingestion gate (single file, 4 classes) |
| `synthesis_engine.py` | 508 | Multi-pass synthesis |

**[[system/reference-library-purpose|Reference Library]] integration:**

| Module | Lines | Purpose |
|--------|-------|---------|
| `rl_search.py` | 340 | Hybrid RL search (FTS5 + embeddings) |
| `rl_index.py` | 304 | RL index management |
| `rl_schema.py` | 124 | RL entry schema definitions |
| `rl_builder.py` | 385 | RL page builder for auto-create |

**Deprecated (kept in `deprecated/` subdirectory or marked in code):**
- `injection_router.py` — superseded by intent classification in the prefetch pipeline (still present, not actively called)
**Test suite:** 20+ custom test functions across 5 files. All passing as of 2026-07-16. (Upstream test suite also runs — 1,700+ tests total.)

### 5.2 Core Database Engine

`agent/source_analysis.py` (~1,135 lines) — SourceAnalyzer facade with dossier lookup, bias detection, narrative analysis, and RL writer. Shared by prefetch pipeline and direct `source_analyze` tool.

`agent/perpetual_context_db.py` (~377 lines) — the SQLite database with FTS5, sqlite-vec embeddings, topic flow, and hybrid search. This is a new file not in upstream.

### 5.3 Modified Upstream Files (8 files)

| File | Lines Changed | Modification |
|------|--------------|--------------|
| `run_agent.py` | ~95 | Rolling window integration, compression timing |
| `agent/prompt_builder.py` | ~15 custom lines in 1,127-line upstream file | System prompt mods for PM context injection |
| `plugins/context_engine/__init__.py` | ~228 | Config passing for context engines |
| `acp_adapter/server.py` | ~9 | ACP server customizations |
| `cli.py` | ~48 | CLI customizations for PM commands |
| `model_tools.py` | ~112 | Model routing for local inference priority |
| `tools/skill_manager_tool.py` | ~95 | Fork-aware skill paths |
| `.gitignore` | ~108 | Custom plugin artifact patterns |

### 5.4 Safe Harbor Architecture

Three-tier survival model for `hermes update`:

- **Tier 1 (Safe Harbor):** `~/.hermes/plugins/` — survives updates. Memory plugin and rolling window engine live here.
- **Tier 2 (External Backup):** `~/.hermes/backups/` — `perpetual_context_db.py` copied before each update, restored automatically.

---

## 6. Epistemic Framework — How the System Judges Truth

### 6.1 The Sovereign Sieve

The methodology for filtering external information before it enters the [[system/reference-library-purpose|Reference Library]]:

**Stage 1 — Linguistic Marker Detection:** Scan sources for shibboleths — specific word choices that signal ideological cluster membership. When markers appear at significant density, tag the source with its ideological cluster.

**Stage 2 — Motive & Funding Mapping:** Maintain dossiers on sources tracking ownership, funding, and historical loyalty patterns. A person telling the truth when it *hurts* them is likely honest; a person telling "truth" they profit from should be scrutinized.

**Stage 3 — Contradiction & Double-Standard Analysis:** Look for asymmetric application of logic across topics. Evaluate organizations by their track record when proven wrong (retract → integrity; ignore → captured; pivot → captured).

### 6.2 The Consensus Warning (Frequency Trap)

Something being pushed simultaneously by multiple "reputable sources" is a *red flag*, not evidence of verification. Truth is *discovered*; propaganda is *distributed*. Simultaneous alignment across supposedly independent outlets indicates coordination, not organic discovery.

### 6.3 Source Analysis — The `source_analyze` Tool

The Sovereign Sieve is operationalized through the `source_analyze` tool. After every `web_search` on substantive topics (politics, religion, economics, culture, current events), the model calls `source_analyze` which:

- **Shallow mode (default):** Analyzes search result snippets against existing source dossiers. Fast — no additional network calls. Returns domain, alignment, reliability, `truthful_on`, `omits`, and deviations from known patterns.
- **Deep mode (`deep=true`):** Extracts full article content via `web_extract` before analysis. Detects specific omissions, loaded language, and framing patterns. Slower but significantly more accurate.
- **Auto-creates dossiers:** Encounters a new domain? Creates a source dossier in `sources/` with placeholders, updated by future analyses. Over time, source intelligence compounds.

**Key class:** `SourceAnalyzer` (`agent/source_analysis.py`, 950 lines) — reimplements bias detection internally to avoid circular imports from the plugins directory. Shared by the prefetch pipeline and the direct tool.

### 6.4 Truth Vector Architecture

The [[system/reference-library-purpose|Reference Library]] serves as the system's truth vector — the curated knowledge base that anchors all reasoning. Combined with source analysis and the Sovereign Sieve, the framework operates as a complete truth-pipeline:

1. **Anchor:** [[system/reference-library-purpose|Reference Library]] is the primary truth source (checked before training data or web search)
2. **Verify:** Source analysis profiles every web result before the model uses it
3. **Filter:** The Sovereign Sieve detects framing, motive, and double standards
4. **Distill:** High-signal findings are promoted to the [[system/reference-library-purpose|Reference Library]] via the Logos Engine

This is not moral relativism disguised as "both sides." It is epistemic honesty about how information is weaponized in the modern media ecosystem.
---

## 7. Infrastructure

### 7.1 Hardware

- **Hardware:** 24GB+ VRAM GPU (e.g., RTX 3090/4090/5090), WSL2 or native Linux
- **Future:** Scale to larger models and longer context windows as hardware and open-weight models improve

### 7.2 Software Stack

| Component | Technology | Port | Notes |
|-----------|-----------|------|-------|
| Inference | llama.cpp (Docker: `ornstein-server`) | 8000 | GestaltLabs/Ornstein3.6-27B-MTP-NSC-ACE-SABER-Q4_K_M-MTP.gguf |
| Embeddings | all-MiniLM-L6-v2 (ONNX) | N/A | In-process, ~80MB model, 384-dim vectors |
| Perpetual Memory | SQLite + FTS5 | N/A | `~/.hermes/perpetual_context.db` |
| RL Hybrid Index | SQLite + FTS5 + embeddings | N/A | `rl_index.db`, 30,000+ entries |
| SearXNG | Docker | Self-hosted | 251+ search services, Tier 1 |
| Firecrawl | Docker stack | Self-hosted | API + Playwright + RabbitMQ + Redis + Postgres, Tier 2 |
| Camofox | Native | 9377 | Anti-detection Firefox fork, Tier 3 |
| Quartz v4 | Node.js (Docker) | 8081 | Static site serving 65K pages, 2.8GB |
| Messaging | Telegram bot gateway | N/A | Primary communication channel |

---

## 8. Nightly Automation

The system runs several autonomous jobs that maintain and improve itself overnight:

| Cron Job | Schedule | Purpose |
|----------|----------|---------|
| PM Signal Scanner | 2:00 AM | Scans Perpetual Memory for high-signal clusters |
| Nightly Distillation | 3:00 AM | Processes clusters through Synthesis → Audit → Commit |
| RL Growth | 3:00 AM | Expands [[system/reference-library-purpose|Reference Library]] based on gaps |
| Logos Intelligence Scout | 4:00 AM | Builds source intelligence dossiers |
| Hermes Backup | 4:00 AM | Backs up entire system to Windows |

---

## 9. What This Is and What It Is Not

### What This Is

- A *sovereign knowledge management system* — all processing is local, all data stays local
- A *growing intelligence* — the [[system/reference-library-purpose|Reference Library]] distills better from conversation history over time
- A *worldview-aware* system — not neutral in the sense of "both sides," but honest about its epistemic commitments
- A *practical tool* — designed for daily use by one person through Telegram
- Originally built on [Hermes Agent](https://github.com/NousResearch/hermes-agent) — now fully detached and standalone

### What This Is Not

- A fork of Hermes Agent — Logos was built on Hermes but is now a fully detached standalone project
- A commercial product — built for one user's needs, not a general-purpose solution
- An attempt at objectivity in the journalistic sense — truth is not consensus, and the system knows this

---

## 10. Version History (Major Milestones)

| Date | Milestone |
|------|-----------|
| 2026-04-21 | Perpetual Memory system deployed (SQLite + FTS5) |
| 2026-04-23 | Context Bridge structured extraction |
| 2026-04-25 | vLLM Docker setup, OpenRouter fallback removed |
| 2026-05-17 | Switched to llama.cpp + Ornstein-SABER, semantic vector engine fix |
| 2026-04-26 | Deep Research Engine Phases 2-4 built and wired |
|| 2026-04-27 | Project forked from Hermes Agent at cluricaun28/hermes-agent |
| 2026-04-30 | Compress→archive rename, Batch #1 cherry-picks (20 commits), StreamingContextScrubber |
| 2026-05-02 | RL Growth, Logos Intelligence Scout, Retrieval Quality crons deployed |
| 2026-05-03 | PM Signal Scanner, Nightly Distillation, RL hybrid index with 831 embeddings |
| 2026-05-03 | Quartz v4 adopted for RL serving |
| 2026-05-04 | Recall Engine with query classification integrated |
| 2026-05-06 | RL index expanded to 32,676 entries, 7 of 12 signal clusters distilled |
| 2026-05-08 | **Sovereign Sieve v2:** Source dossiers as YAML (`source_dossiers.yaml`, 284 entries), embedding-based semantic marker detection alongside regex, `WorldviewDivergenceChecker` wired into `ScrutinyGate`. FAISS vector index rebuilt (100% coverage, 6,716 vectors). `ExtractionEngine` split from `BridgeQualityScorer`. Test suite cleaned (stale duplicates removed). `scrutiny_gate.py` at 960 lines, full ruff compliance. |
| 2026-05-08 | Scrutiny gate split into 6 SRP-compliant modules (967→221 lines facade + 5 submodules), all under 500 lines. Last god class eliminated. |
| 2026-05-09 | **Phase 4 — `source_analyze` tool:** Direct agent tool for source intelligence during research. 11 tool schemas (added `SOURCE_ANALYZE_SCHEMA`). Deep mode (`deep=true`) extracts full article content via Firecrawl before analysis. Auto-creates source dossiers in `sources/` for new domains with `domain-index.json` auto-update. Smart trigger in system prompt: substantive topics get `source_analyze(deep=true)`, utility queries skip. 3 new skills (`factual-research-answer`, `tool-schema-validation-debug`, `political-research-and-entity-pages`), 3 updated skills (`web-source-bias-research`, `narrative-control-detection`, `pipeline-module-integration`). 10 new RL pages (5 entity, 4 source, 1 topic). Code audit: removed duplicate schemas, fixed f-string JSON construction, narrowed exception handling, `frozenset` for mutable globals, added `__all__` and `logger`. |
| 2026-05-11 | **Full detachment from Hermes Agent:** Repo unforked from NousResearch/hermes-agent via GitHub "Leave fork network." DIVERGENCE.md updated to reflect standalone status. README rebranded with "What Makes Logos Different" section. 26 LLM paper entries added to RL (`topics/llm-papers/`). 3 media dossiers added (`topics/media/`). |
| 2026-05-12 | **Semantic Vector engine promoted to primary:** Dual-engine context archiving finalized — semantic vector for topic-aware pruning, rolling window as deterministic fallback. Context Compressor refactored with plugin context engine hooks. `context_compressor.py` now calls `context_archiver.on_session_reset()` and `compression_count` resets on `/new`. All context engine code in `plugins/context_engine/`. |
| 2026-05-12 | **Semantic Vector plugin deployed:** `SemanticVectorContextEngine` reimplemented as proper plugin (`plugins/context_engine/semantic_vector/`). CPU-only embedding on all-MiniLM-L6-v2, topic-aware pruning of dormant/resolved turns, state map injection. Rolling window relegated to emergency fallback. |
| 2026-05-15 | **Context engine hardening:** Module-level model cache (one load per process), `on_session_reset()` preserves model, full structured logging on every archive (vector counts, pruned messages, fallback path). Fixed `compression_count` not resetting on session reset. Added 5 generic research skills to repo (`frame-stripping`, `web-source-bias-research`, `narrative-control-detection`, `sovereign-intelligence-mapping`, `epistemic-framework-design`) plus `worldview-profile-builder` for new-user onboarding. Updated GETTING-STARTED and README. |

---

## 11. Future Direction

### Near-term (through June 2026)

- **Model scaling:** Migrate to larger open-weight models as they become available
- **Async deep research:** Pipeline blocks prefetch; async execution with periodic updates
- **Worldview quiz configuration:** Implemented as `worldview-profile-builder` skill. New users run the interview to generate their personalized worldview profile. Methodology ships generic; positions come from the user.

### Long-term

- **DPO training on curated dataset:** Post-train Qwen3.6-27B on curated preference pairs *if* needed (currently shelved, uncertain whether necessary)
- **Train from scratch:** When compute allows, train a model on the accumulated curated corpus as a long-term goal

---

## 12. Conclusion

The Logos Engine represents a fundamental departure from standard agent architectures. Where most agents are built to serve everyone with maximum neutrality, this system is built to serve one person with maximum clarity. Where most agents forget everything after context compression, this one remembers everything and retrieves on demand. Where most agents are trained on web noise, this one anchors in a curated knowledge base that grows denser and more internally consistent over time.

The system is not perfect — it is a work in progress. But it is *honest* about what it is, and it is *sovereign* in how it operates. It cannot be captured by a corporate update, corrupted by a cloud API change, or silenced by a policy shift. It belongs to the person who runs it.

> *"Codifying truth in the [[system/reference-library-purpose|Reference Library]] is planting a flag that no corporate update can erase."*

---

*This white paper was compiled from the live codebase, Reference Library documentation, and Perpetual Memory records of Logos. Last updated 2026-05-15.*
