# Getting Started with Logos

This assumes you've cloned the repository and can run Python. It covers what the system does and how to make it yours.

## What You're Working With

Logos is a research agent with permanent memory. It doesn't forget. Every conversation is stored locally and searchable. It has a curated Reference Library you build over time, and skills for stripping bias from web content before it enters your knowledge base.

The three things that set it apart:

1. **Frame-Stripping** — When the agent researches something, it doesn't just accept web content at face value. It applies a 10-rule filter that separates facts from framing, replaces loaded terminology, and cross-references sources. The result is presented through your stated worldview, not through whatever bias the source carries.

2. **Source Analysis** — Every domain the agent researches gets a dossier: what the source is truthful about, what it consistently omits, its ideological alignment, and its primary motive. These compound over time. The more you research, the sharper the analysis becomes.

3. **Permanent Memory** — SQLite with full-text search and optional semantic embeddings. Every turn is stored. The agent can search across all past sessions when you ask a question. Completed work is archived from the active window but never deleted.

## Setup

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

### 4. Create Your SOUL.md

This is the most important step. The code provides infrastructure — SOUL.md tells the agent *how* to use it. Without the right sections, the plugins load but the agent never calls them.

```bash
cp extras/soul-template.md ~/.hermes/SOUL.md
```

Then edit `~/.hermes/SOUL.md`:

- Replace `[YOUR NAME]` with your name
- Write your **Worldview Baseline** — what you consider truth, how to handle contradictory claims, your communication style
- Keep the **Knowledge Architecture** and **Active Retrieval** sections — these are what make the system work

For detailed guidance, see [`extras/system-prompt-guide.md`](extras/system-prompt-guide.md).

### 5. Initialize the Reference Library

```bash
cp -r extras/reference-library-template ~/.hermes/reference-library
```

This creates the directory structure. It starts empty and grows as you work.

### 6. (Optional) Set Up Deep Research

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
