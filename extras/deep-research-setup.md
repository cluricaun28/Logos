# Deep Research Setup — SearXNG + Firecrawl

This document explains how to set up the web search infrastructure needed for the perpetual memory plugin's deep research engine. Without these services, the agent falls back to basic web search or training data only.

## Architecture Overview

```
Logos (perpetual_context plugin)
  ├── Tier 1: SearXNG (local meta-search)
  │   └── Aggregates results from Google, Bing, DuckDuckGo, etc.
  ├── Tier 2: Firecrawl (web scraping + content extraction)
  │   └── Fetches full page content from URLs found by SearXNG
  └── Tier 3: Camofox (optional, for JS-heavy sites)
```

## Prerequisites

- Docker and Docker Compose installed
- At least 2GB free RAM (SearXNG ~100MB, Firecrawl ~500MB+)
- Network access to search engines (no corporate firewall blocking common search APIs)

---

## Step 1: SearXNG (Local Meta-Search Engine)

### Quick Setup with Docker

```bash
# Create config directory
mkdir -p ~/.docker/searxng

# docker-compose.yml
cat > ~/.docker/searxng/docker-compose.yml << 'EOF'
version: "3"
services:
  searxng:
    image: searxng/searxng:latest
    container_name: searxng
    ports:
      - "8080:8080"
    volumes:
      - ./settings.yml:/etc/searxng/settings.yml:ro
    environment:
      - SEARXNG_BASE_URL=http://localhost:8080/
    restart: unless-stopped
EOF

# settings.yml (minimal config)
cat > ~/.docker/searxng/settings.yml << 'EOF'
use_default_settings: true

general:
  instance_name: "Local SearXNG"
  debug: false

search:
  safe_search: 0
  autocomplete: "google"
  default_lang: "en-US"

engines:
  - name: google
    disabled: false
  - name: bing
    disabled: false
  - name: duckduckgo
    disabled: false
  - name: wikipedia
    disabled: false

server:
  secret_key: "change-this-to-a-random-string"
  bind_address: "0.0.0.0"
  port: 8080
  method: "POST"
  http_protocol_headers:
    - (HTTP_HOST, localhost:8080)

ui:
  static_path: ""
  templates_path: ""
EOF

# Start the service
cd ~/.docker/searxng
docker compose up -d
```

### Verify SearXNG is Running

```bash
curl -s "http://localhost:8080/search?q=test&format=json" | head -c 200
# Should return JSON with results
```

### Configure Logos to Use SearXNG

Add to `~/.logos/config.yaml`:

```yaml
web_search:
  searxng_url: "http://localhost:8080/search"
```

Or set the environment variable:

```bash
export SEARXNG_BASE_URL="http://localhost:8080/search"
```

---

## Step 2: Firecrawl (Web Content Extraction)

Firecrawl fetches and extracts clean text content from URLs. This is essential for deep research — SearXNG gives you links, Firecrawl gives you the actual content behind those links.

### Option A: Self-Hosted (Free, Requires More Resources)

```bash
# docker-compose.yml for Firecrawl
cat > ~/.docker/firecrawl/docker-compose.yml << 'EOF'
version: "3"
services:
  firecrawl:
    image: arxiv/firecrawl:v1.0.0
    # Note: Check latest version on Docker Hub
    ports:
      - "3000:3000"
    environment:
      - REDIS_URL=redis://redis:6379
      - RATE_LIMIT_QUEUE_SIZE=20
    depends_on:
      - redis
    restart: unless-stopped

  redis:
    image: redis:alpine
    container_name: firecrawl-redis
    restart: unless-stopped
EOF
```

**Note:** Self-hosted Firecrawl can be complex. See [Firecrawl docs](https://docs.firecrawl.dev) for latest setup instructions.

### Option B: Firecrawl Cloud API (Easier, Free Tier Available)

1. Sign up at [firecrawl.dev](https://www.firecrawl.dev)
2. Get your API key from the dashboard
3. Set the environment variable:

```bash
export FIRECRAWL_API_KEY="fc-you...here"
export FIRECRAWL_API_URL="https://api.firecrawl.dev"
```

Or add to `~/.logos/config.yaml`:

```yaml
web_search:
  firecrawl_api_key: "fc-your-api-key-here"
  firecrawl_api_url: "https://api.firecrawl.dev"
```

### Verify Firecrawl is Working

```bash
# Test with cloud API
curl -s "https://api.firecrawl.dev/v1/scrape?url=https://example.com&format=markdown" \
  -H "Authorization: Bearer $FIREC...KEY" | head -c 200

# Should return markdown content from example.com
```

---

## Step 3: Configure the Perpetual Context Plugin

The deep research engine is part of the `perpetual_context` plugin. It's automatically available once the plugin is enabled and environment variables are set.

In `~/.logos/config.yaml`:

```yaml
plugins:
  enabled:
    - perpetual_context

web_search:
  searxng_url: "http://localhost:8080/search"
  firecrawl_api_key: "fc-your-api-key-here"  # Optional, for deep scraping
  firecrawl_api_url: "https://api.firecrawl.dev"  # Optional
```

---

## Step 4: Verify the Full Pipeline

Start a Logos session and test:

1. **Basic search:** Ask a factual question — the agent should use SearXNG via `web_search_tool`
2. **Deep research:** Ask for detailed analysis on a topic — the agent should call Firecrawl to extract content from top results
3. **Source tracking:** Results should be stored in Perpetual Memory with source URLs

---

## Troubleshooting

### SearXNG Returns No Results

- Check that search engines aren't blocked by your network: `curl -s "http://localhost:8080/search?q=test&format=json"`
- Verify `settings.yml` has at least one engine enabled
- Try changing the secret_key in settings and restarting

### Firecrawl Timeout / Connection Refused

- For self-hosted: check Docker logs with `docker compose -f ~/.docker/firecrawl/docker-compose.yml logs`
- For cloud API: verify your API key is correct and you haven't exceeded rate limits
- Check that `FIRECRAWL_API_URL` points to the correct endpoint

### Agent Not Using Web Search Tools

Make sure your SOUL.md includes the "Proactive Web Tool Usage" section from `extras/system-prompt-guide.md`. Without it, the agent may not call search tools proactively.

---

## Resource Requirements

| Service | RAM (approx) | CPU | Notes |
|---------|-------------|-----|-------|
| SearXNG | ~100MB | Minimal | Very lightweight |
| Firecrawl (self-hosted) | ~500MB-2GB | Moderate | Depends on scraping volume |
| Firecrawl (cloud API) | 0 | 0 | Free tier: limited requests/month |

## Alternative: Search Without Docker

If you can't run Docker, you can still use:

1. **Public SearXNG instances** — find one at [searx.space](https://searx.space) and point `SEARXNG_BASE_URL` there
2. **DuckDuckGo HTML scraping** — the web_search_tool has a fallback mode that works without any external services (slower, less reliable)

---

## Security Notes

- SearXNG should not be exposed to the internet without authentication
- Firecrawl API keys are secrets — don't commit them to git
- Consider running both behind a reverse proxy if exposing beyond localhost
