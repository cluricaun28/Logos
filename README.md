# Hermes Agent Fork — Perpetual Context Memory & Plugin Extensions

**Fork of [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) with custom extensions for persistent cross-session memory, context archiving, and plugin-based retrieval.**

## What This Fork Adds

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

## Setup

### Prerequisites
- Python 3.10+ with pip
- SQLite (bundled with Python)
- Optional: ONNX Runtime + SentenceTransformers for semantic embeddings

### Install from Fork
```bash
# Clone the fork
git clone https://github.com/cluricaun28/hermes-agent.git
cd hermes-agent

# Install in development mode
pip install -e ".[dev]"

# Run Hermes setup to configure model, gateway, and plugins
hermes setup
```

### Enable Perpetual Memory Plugin
Add to your `~/.hermes/config.yaml`:
```yaml
plugins:
  enabled:
    - perpetual_context
```

The plugin initializes the SQLite database on first run. No manual DB creation needed.

### Semantic Embeddings (Optional)
For hybrid search with vector similarity:
```bash
pip install onnxruntime sentence-transformers
```
The default model is `all-MiniLM-L6-v2` — lightweight, runs locally. Embeddings are stored as BLOB in SQLite alongside FTS5 indexes.

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

## Cherry-Picking Upstream Changes

This fork tracks upstream via the `upstream` remote:
```bash
# Fetch latest from NousResearch
git fetch upstream main

# Review changes before cherry-picking
git log upstream/main --oneline -10

# Selectively apply commits
git cherry-pick <commit-hash>
```

Do NOT merge blindly — custom plugin files may conflict with upstream context engine changes.

## What's Not Here

This fork does **not** include:
- Personal data, tokens, or credentials
- Reference library content (lives in `~/.hermes/reference-library/`, not tracked)
- Skill definitions (lives in `~/.hermes/skills/`, not tracked)
- SQLite database files (data persists locally, code is here)

## License

Same as upstream: MIT. Custom additions are also MIT licensed.

---

**Upstream:** [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent)  
**Fork:** [cluricaun28/hermes-agent](https://github.com/cluricaun28/hermes-agent)
