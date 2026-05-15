# Getting Started with Logos

This assumes you've cloned the repository and can run Python. It covers what the system does and how to make it yours.

## What You're Working With

Logos is a research agent with permanent memory. It doesn't forget. Every conversation is stored locally and searchable. It has a curated Reference Library you build over time, and skills for stripping bias from web content before it enters your knowledge base.

The three things that set it apart:

1. **Frame-Stripping** — When the agent researches something, it doesn't just accept web content at face value. It separates facts from framing, cross-references sources, and presents findings aligned with your worldview. The methodology is generic — *your* positions come from the worldview profile you build in step 4.

2. **Source Analysis** — Every domain the agent researches gets a dossier: what the source is truthful about, what it consistently omits, its ideological alignment, and its primary motive. These compound over time. The more you research, the sharper the analysis becomes.

3. **Permanent Memory** — SQLite with full-text search and optional semantic embeddings. Every turn is stored. The agent can search across all past sessions when you ask a question. Completed work is archived from the active window but never deleted.

## Setup

### Hardware Requirements

Logos is designed for local inference. The original setup runs a 27B-parameter int4 model on an RTX 5090 (32GB VRAM) with 64GB RAM.

- **Minimum:** Single GPU with 24GB+ VRAM (RTX 4090, RTX 5090)
- **RAM:** 64GB recommended (the agent, Docker services, and embedding model need headroom)
- **OS:** WSL2 on Windows 11, or native Linux

If you have less VRAM, use a smaller model and reduce `--max-model-len` accordingly.

### Local Services and Ports

Logos depends on several local services. Their default ports:

| Service | Port | Purpose |
|---------|------|---------|
| vLLM | 8000 | Local model inference |
| SearXNG | 8080 | Metasearch engine |
| Firecrawl | 3002 | Web content extraction |
| Camofox | 9377 | Anti-detection browser |
| Quartz v4 | 8081 | Reference Library static site |

These are referenced throughout the codebase and config files.

### Key Configuration Values

In `~/.hermes/config.yaml`, the most important values to understand:

- **`context.engine: rolling_window`** — The context management strategy
- **`context.archiving.threshold_percent`** — When to archive (default: 0.75)
- **`context.archiving.archive_target`** — Prune down to this level (default: 0.65)
- **`context.archiving.hard_ceiling_percent`** — Safety net (default: 0.85)
- **Custom provider `context_length`** — Must match your vLLM `--max-model-len`

All file paths are relative to `~/.hermes/`. If you install elsewhere, update these accordingly.

### 1. Install

```bash
git clone https://github.com/cluricaun28/Logos.git
cd Logos
pip install -e ".[dev]"
```

### 2. Run Initial Setup

```bash
hermes setup
```

This configures your model provider, gateway, and plugins.

### 3. Enable Perpetual Memory

Add to your `~/.hermes/config.yaml`:

```yaml
plugins:
  enabled:
    - perpetual_context
```

The database (`~/.hermes/perpetual_context.db`) is created automatically on first run.

### 4. Set Up Your Worldview Profile

This is where the research pipeline gets personalized. The frame-stripping and source analysis skills ship with generic methodology — they need *your* positions to work.

Ask the agent: **"Build my worldview profile"**

It will walk you through a short interview — your foundational beliefs, political views, source preferences, and reasoning style. The results are saved as `~/.hermes/skills/worldview-profile/SKILL.md` and automatically used for all future research.

If you skip this step, the agent defaults to neutral fact presentation with source attribution. It will still work, just less aligned.

### 5. Create Your SOUL.md

The code provides infrastructure — SOUL.md tells the agent *how* to use it.

```bash
cp extras/soul-template.md ~/.hermes/SOUL.md
```

Then edit `~/.hermes/SOUL.md`:

- Replace `[YOUR NAME]` with your name
- Write your **Worldview Baseline** — what you consider truth, how to handle contradictory claims, your communication style
- Keep the **Knowledge Architecture** and **Active Retrieval** sections — these are what make the system work

For detailed guidance, see [`extras/system-prompt-guide.md`](extras/system-prompt-guide.md).

### 6. Initialize the Reference Library

```bash
cp -r extras/reference-library-template ~/.hermes/reference-library
```

This creates the directory structure. It starts empty and grows as you work.

### 7. (Optional) Set Up Deep Research

For web search, content extraction, and anti-detection browsing, see [`extras/deep-research-setup.md`](extras/deep-research-setup.md). Requires three local services:

- **SearXNG** — Meta-search engine (Docker)
- **Firecrawl** — Content extraction (Docker)
- **Camofox** — Anti-detection browser for sites that block scrapers

## Daily Use

```bash
hermes gateway start
hermes
```

The agent loads your SOUL.md, checks recent conversation history, scans available skills, and waits for your input.

What happens automatically:
- Recent context is checked before each action (prevents loops)
- The Reference Library is consulted before web search
- Perpetual Memory is searched when topics reference past work
- Skills load on demand — only when relevant
- Web sources are analyzed for bias via `source_analyze`
- If a worldview profile exists, research results are reframed through your lens
- Completed turns are archived to keep the context window lean

## What Grows Over Time

- **Reference Library** — New topics, entities, and source dossiers are created as you research
- **Skills** — Complex problems you solve repeatedly become reusable procedures
- **Perpetual Memory** — Every conversation is stored and searchable
- **Source Dossiers** — Each domain gets richer analysis with every research session

## More Information

- [`WHITEPAPER.md`](WHITEPAPER.md) — Full architectural deep-dive
- [`README.md`](README.md) — System overview and configuration
- [`DIVERGENCE.md`](DIVERGENCE.md) — Relationship to upstream Hermes Agent
