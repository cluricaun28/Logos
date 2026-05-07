"""Topic stability classification for Perpetual Context.

Classifies queries into stability levels (static, slow, volatile) using
keyword frozensets. Each level has an associated half-life (days) and
a web search trigger threshold.

This drives the stability-aware gap detection in the retrieval pipeline:
  - static:  timeless knowledge (biology, math, scripture)
  - slow:    evolves over months (software docs, legal, established tech)
  - volatile: changes weekly (current events, active dev, pricing)
"""

from __future__ import annotations


# STATIC — timeless knowledge. RL is authoritative; no web search needed.
STATIC_TOPIC_KEYWORDS: frozenset = frozenset([
    # Biology / nature
    "dog", "cat", "bird", "fish", "plant", "tree", "animal", "species",
    "photosynthesis", "evolution", "DNA", "cell", "ecosystem",
    # Mathematics / science fundamentals
    "calculus", "algebra", "geometry", "theorem", "proof", "prime number",
    "gravity", "thermodynamics", "quantum", "atom", "molecule",
    # Scripture / theology
    "genesis", "exodus", "psalms", "gospel", "epistle", "revelation",
    "bible", "scripture", "covenant", "redemption", "sanctification",
    "justification", "election", "predestination", "trinity",
    # Established history
    "world war", "american revolution", "french revolution",
    "roman empire", "byzantine", "reformation", "dark ages",
    # General definitions
    "what is", "define", "definition of", "meaning of",
    "how does", "how does a",
])

# SLOW — evolves over months. Software docs, frameworks, legal, established tech.
# Half-life: 90 days. Web search fires if local data is older than that with low score.
SLOW_TOPIC_KEYWORDS: frozenset = frozenset([
    # Software / frameworks (well-established)
    "python", "sql", "docker", "linux", "git", "github", "api",
    "postgresql", "sqlite", "nginx", "redis", "kubernetes",
    "flask", "django", "fastapi", "react", "typescript",
    # AI / ML (established concepts)
    "transformer", "attention", "backpropagation", "gradient",
    "neural network", "convolution", "gan", "vae",
    # Legal / regulation
    "regulation", "compliance", "tax", "irs", "llc", "corporation",
    "contract", "liability", "warranty", "indemnification",
    # Hardware (established)
    "nvidia", "amd", "intel", "gpu", "cpu", "ram", "ssd",
    "cuda", "opencl", "vulkan",
    # Business / finance (stable concepts)
    "accounting", "bookkeeping", "invoice", "receivable", "payable",
    "p&l", "balance sheet", "cash flow", "depreciation",
    "retail", "wholesale", "margin", "overhead",
])

# VOLATILE — changes weekly or daily. Current events, active development, pricing.
# Half-life: 7 days. Web search fires aggressively.
VOLATILE_TOPIC_KEYWORDS: frozenset = frozenset([
    # AI / ML (active development)
    "vllm", "llama", "qwen", "gpt", "claude", "gemini",
    "model release", "new model", "model update", "fine-tuning",
    "dpo", "sft", "rlhf", "training", "inference",
    "unsloth", "axolotl", "trl", "peft", "qlora",
    "xai", "grok", "openai", "anthropic", "google ai", "deepmind",
    "llm", "large language model", "llm benchmark", "llm leader",
    # Pricing / availability
    "price", "pricing", "cost", "how much", "pricing change",
    "sale", "discount", "deal", "buy", "purchase",
    "available", "unavailable", "out of stock", "released",
    # Current events / time-sensitive
    "latest", "new", "recent", "current", "today", "this week",
    "this month", "breaking", "news", "announcement", "update",
    "what's happening", "what is going on", "status", "latest version",
    "latest news", "recent development",
    # Software versions / releases
    "version", "release", "changelog", "roadmap", "beta", "alpha",
    "stable release", "nightly", "latest release",
    # Active projects / ongoing development
    "progress", "development", "work in progress", "under construction",
    "upcoming", "planned", "scheduled",
])

# Half-life in days per stability level
STATIC_HALF_LIFE_DAYS = 3650      # ~10 years — essentially permanent
SLOW_HALF_LIFE_DAYS = 90          # ~3 months
VOLATILE_HALF_LIFE_DAYS = 7       # ~1 week

# Score thresholds per stability level — web search fires if best local score is below this
STATIC_WEB_THRESHOLD = 0.05       # Almost never fire web for static topics
SLOW_WEB_THRESHOLD = 0.35         # Moderate — fire if local results are weak
VOLATILE_WEB_THRESHOLD = 0.60     # Aggressive — fire unless local results are strong


def _classify_topic_stability(query: str) -> tuple[str, int, float]:
    """Classify a query's topic stability.

    Returns (stability: str, half_life_days: int, web_threshold: float).

    Stability levels:
      - 'static': timeless knowledge (what is a dog, math, scripture)
      - 'slow': evolves over months (software docs, legal, established tech)
      - 'volatile': changes weekly (current events, active dev, pricing)

    When no keywords match, defaults to 'slow' — conservative fallback.

    Extend the keyword frozensets above to add or remove topics.
    """
    q = query.lower()
    q_words = set(q.split())

    # Check volatile first (most specific, highest priority)
    if q_words & VOLATILE_TOPIC_KEYWORDS:
        return ("volatile", VOLATILE_HALF_LIFE_DAYS, VOLATILE_WEB_THRESHOLD)

    # Check static
    if q_words & STATIC_TOPIC_KEYWORDS:
        return ("static", STATIC_HALF_LIFE_DAYS, STATIC_WEB_THRESHOLD)

    # Check slow
    if q_words & SLOW_TOPIC_KEYWORDS:
        return ("slow", SLOW_HALF_LIFE_DAYS, SLOW_WEB_THRESHOLD)

    # Default: treat as slow — better to slightly over-search than serve stale data
    return ("slow", SLOW_HALF_LIFE_DAYS, SLOW_WEB_THRESHOLD)
