"""Tool schemas for Perpetual Context memory provider.

All OpenAI-format tool schema definitions used by the perpetual context
memory provider. Each dict contains 'name', 'description', and 'parameters'
keys following the standard tool schema format.

Aggregated in TOOL_SCHEMAS for convenient bulk access.
"""

from __future__ import annotations

from typing import Any

PERPETUAL_SEARCH_SCHEMA: dict[str, Any] = {
    "name": "perpetual_search",
    "description": (
        "Search across all perpetual memory storage backends using hybrid semantic + keyword search. "
        "Combines FTS5 full-text keyword matching with cosine similarity against stored embedding vectors "
        "(weighted fusion: 60% keyword, 40% semantic). Returns the most relevant historical messages.\n\n"
        "PARAMETERS:\\n"
        "• query: Search text (required)\\n"
        "• session_id: Optional session filter\\n"
        "• top_k: Number of results (default 5, max 20)\\n"
        "\\n"
        "EXAMPLES:\\n"
        "• 'Hermes configuration' — Find messages about setup\\n"
        "• 'GPU training' — Find messages mentioning GPU training\\n"
        "• session_id='20260421_023037' — Limit to one session"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query text"},
            "session_id": {"type": "string", "description": "Optional session ID filter"},
            "top_k": {"type": "integer", "description": "Number of results (default 5)", "default": 5},
        },
        "required": ["query"],
    },
}

TOPIC_FLOW_SCHEMA: dict[str, Any] = {
    "name": "topic_flow",
    "description": (
        "View and manage topic clusters for a session. Shows the conversation's "
        "topic evolution, message counts per topic, and allows adding new topics.\n\n"
        "PARAMETERS:\n"
        "• action: 'list', 'add', or 'drift_check' (default 'list')\n"
        "• session_id: Session to analyze (defaults to current)\n"
        "• topic_name: Topic name for 'add' action\n"
        "• confidence: Confidence score 0.0-1.0 for new topics\n"
        "\n"
        "EXAMPLES:\n"
        "• action='list' — Show all topics for current session\n"
        "• action='drift_check' — Detect if conversation has drifted\n"
        "• action='add', topic_name='GPU optimization' — Register new topic"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "description": "Action: list, add, or drift_check", "default": "list"},
            "session_id": {"type": "string", "description": "Session ID (defaults to current)"},
            "topic_name": {"type": "string", "description": "Topic name for 'add' action"},
            "confidence": {"type": "number", "description": "Confidence score 0.0-1.0", "default": 0.5},
        },
    },
}

CONTEXT_DEPTH_SCHEMA: dict[str, Any] = {
    "name": "context_depth",
    "description": (
        "Control how much historical context is surfaced from perpetual memory. "
        "Adjusts the depth of recall based on conversation needs.\n\n"
        "DEPTH LEVELS:\n"
        "• broad_overview: Only main topics and high-level summaries\n"
        "• moderate: Topics + key messages (default)\n"
        "• deep: All topics with detailed message content\n"
        "• expert: Full history with relationships and metadata\n\n"
        "PARAMETERS:\n"
        "• action: 'get', 'set', or 'status'\n"
        "• level: Depth level (for 'set' action)\n"
        "\n"
        "EXAMPLES:\n"
        "• action='get' — Show current depth setting\n"
        "• action='set', level='deep' — Increase recall depth\n"
        "• action='status' — Full memory system status report"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "description": "Action: get, set, or status", "default": "get"},
            "level": {"type": "string", "description": "Depth level: broad_overview, moderate, deep, expert"},
        },
    },
}

GET_MESSAGES_SCHEMA: dict[str, Any] = {
    "name": "get_messages",
    "description": (
        "Search messages using SQL LIKE-style pattern matching on content. "
        "Returns full raw message content — no summarization, no truncation.\n\n"
        "USE THIS WHEN:\n"
        "• You know what you're looking for and need exact matches\n"
        "• Searching for tokens, keys, or specific strings (e.g., 'ghp_%')\n"
        "• You need the complete content of a message without search indexing abstraction\n\n"
        "PARAMETERS:\n"
        "• pattern: SQL LIKE pattern (use % as wildcard, _ as single char)\n"
        "• session_id: Optional session filter\n"
        "• role: Filter by role (user, assistant, system, tool)\n"
        "• limit: Maximum results to return (default 50)\n\n"
        "EXAMPLES:\n"
        "• pattern='ghp_%' — Find all GitHub tokens\n"
        "• pattern='%github token%' — Find messages mentioning github tokens\n"
        "• pattern='%', role='user', limit=10 — Last 10 user messages"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "SQL LIKE pattern (use % as wildcard)"},
            "session_id": {"type": "string", "description": "Optional session ID filter"},
            "role": {"type": "string", "description": "Filter by role: user, assistant, system, tool"},
            "limit": {"type": "integer", "description": "Maximum results (default 50)", "default": 50},
        },
        "required": ["pattern"],
    },
}

RECENT_MESSAGES_SCHEMA: dict[str, Any] = {
    "name": "recent_messages",
    "description": (
        "Get the N most recent messages from the database. Returns raw content "
        "in chronological order — no summarization, no search indexing.\n\n"
        "USE THIS WHEN:\n"
        "• You need to see what was discussed recently without searching\n"
        "• Reviewing the last few turns of a conversation\n"
        "• Getting raw message content for verification (e.g., checking token length)\n\n"
        "PARAMETERS:\n"
        "• n: Number of recent messages to retrieve (default 10, max 50)\n"
        "• session_id: Optional session filter (None for all sessions)\n"
        "• role: Optional role filter\n\n"
        "EXAMPLES:\n"
        "• n=5 — Last 5 messages across all sessions\n"
        "• n=10, session_id='20260421_124052' — Last 10 in specific session\n"
        "• n=3, role='user' — Last 3 user messages"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "n": {"type": "integer", "description": "Number of recent messages (default 10)", "default": 10},
            "session_id": {"type": "string", "description": "Optional session ID filter"},
            "role": {"type": "string", "description": "Filter by role: user, assistant, system, tool"},
        },
    },
}

QUERY_MESSAGES_SCHEMA: dict[str, Any] = {
    "name": "query_messages",
    "description": (
        "Master query tool — comprehensive message filtering with time ranges,\n"
        "token counts, direct ID lookup, metadata filters, and statistics.\n\n"
        "USE THIS WHEN:\n"
        "• You need precise control over what messages to retrieve\n"
        "• Filtering by time range (e.g., 'messages from April 21st')\n"
        "• Looking up specific message IDs directly\n"
        "• Getting statistics about your conversation history\n\n"
        "PARAMETERS:\n"
        "• pattern: SQL LIKE pattern for content (use % as wildcard)\n"
        "• session_id: Filter by session ID\n"
        "• role: Filter by role (user, assistant, system, tool)\n"
        "• ids: List of specific message IDs to retrieve\n"
        "• time_start: Unix timestamp filter (messages >= this time)\n"
        "• time_end: Unix timestamp filter (messages <= this time)\n"
        "• min_tokens: Minimum token count filter\n"
        "• max_tokens: Maximum token count filter\n"
        "• metadata_key: Filter by metadata key name\n"
        "• metadata_value: Value to match for the metadata key\n"
        "• stats: True to return statistics instead of messages\n"
        "• limit: Maximum results (default 100, max 500)\n"
        "• offset: Pagination offset (default 0)\n\n"
        "EXAMPLES:\n"
        "• ids=[542] — Get message #542 directly\n"
        "• pattern='ghp_%', limit=10 — Find all GitHub tokens\n"
        "• time_start=1776780000, time_end=1776790000 — Messages in time range\n"
        "• role='user', min_tokens=500 — Long user messages\n"
        "• stats=True — Get conversation statistics"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "SQL LIKE pattern (use % as wildcard)"},
            "session_id": {"type": "string", "description": "Optional session ID filter"},
            "role": {"type": "string", "description": "Filter by role: user, assistant, system, tool"},
            "ids": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "List of specific message IDs to retrieve",
            },
            "time_start": {"type": "number", "description": "Unix timestamp filter (messages >= this time)"},
            "time_end": {"type": "number", "description": "Unix timestamp filter (messages <= this time)"},
            "min_tokens": {"type": "integer", "description": "Minimum token count filter"},
            "max_tokens": {"type": "integer", "description": "Maximum token count filter"},
            "metadata_key": {"type": "string", "description": "Filter by metadata key name"},
            "metadata_value": {
                "type": ["string", "boolean", "number"],
                "description": "Value to match for the metadata key",
            },
            "stats": {"type": "boolean", "description": "True to return statistics instead of messages"},
            "limit": {"type": "integer", "description": "Maximum results (default 100)", "default": 100},
            "offset": {"type": "integer", "description": "Pagination offset (default 0)", "default": 0},
        },
    },
}

SMART_RETRIEVE_SCHEMA: dict[str, Any] = {
    "name": "smart_retrieve",
    "description": (
        "Adaptive retrieval engine for Perpetual Memory. Uses different strategies\n"
        "based on the type of information needed, optimized for local hardware.\n\n"
        "RETRIEVAL TYPES:\n"
        "• auto — Let the system classify intent via keyword heuristics (recommended default)\n"
        "• recent — Context from last 20 turns (fastest, O(1) turn ID lookup)\n"
        "• topic — Topic-specific FTS5 search across all sessions\n"
        "• decision_trace — Find where a decision was made and surrounding context\n"
        "• file_history — All edits to a specific file with turn references\n\n"
        "USE THIS WHEN:\n"
        "• You're unsure which strategy to use (use 'auto' — system classifies for you)\n"
        "• You need recent conversation context (use 'recent')\n"
        "• Searching for topic-specific information across sessions (use 'topic')\n"
        "• Tracing why a decision was made earlier (use 'decision_trace')\n"
        "• Finding all edits to a specific file path (use 'file_history')\n\n"
        "PARAMETERS:\n"
        "• query_type: One of 'auto', 'recent', 'topic', 'decision_trace', 'file_history'\n"
        "• query_text: The search query or context identifier\n\n"
        "EXAMPLES:\n"
        "• query_type='auto', query_text='why did we choose SQLite' — System auto-routes to decision_trace\n"
        "• query_type='recent' — Get last 20 turns for immediate context\n"
        "• query_type='topic', query_text='context bridge design' — Find topic discussion\n"
        "• query_type='file_history', query_text='/path/to/file.py' — Get file edit history"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query_type": {"type": "string", "description": "Retrieval type: auto, recent, topic, decision_trace, or file_history"},
            "query_text": {"type": "string", "description": "Search query or context identifier"},
        },
        "required": ["query_type", "query_text"],
    },
}

QUERY_MESSAGES_SCHEMA: dict[str, Any] = {
    "name": "query_messages",
    "description": (
        "Master query tool — comprehensive message filtering with time ranges,\n"
        "token counts, direct ID lookup, metadata filters, and statistics.\n\n"
        "USE THIS WHEN:\n"
        "• You need precise control over what messages to retrieve\n"
        "• Filtering by time range (e.g., 'messages from April 21st')\n"
        "• Looking up specific message IDs directly\n"
        "• Getting statistics about your conversation history\n\n"
        "PARAMETERS:\n"
        "• pattern: SQL LIKE pattern for content (use % as wildcard)\n"
        "• session_id: Filter by session ID\n"
        "• role: Filter by role (user, assistant, system, tool)\n"
        "• ids: List of specific message IDs to retrieve\n"
        "• time_start: Unix timestamp filter (messages >= this time)\n"
        "• time_end: Unix timestamp filter (messages <= this time)\n"
        "• min_tokens: Minimum token count filter\n"
        "• max_tokens: Maximum token count filter\n"
        "• metadata_key: Filter by metadata key name\n"
        "• metadata_value: Value to match for the metadata key\n"
        "• stats: True to return statistics instead of messages\n"
        "• limit: Maximum results (default 100, max 500)\n"
        "• offset: Pagination offset (default 0)\n\n"
        "EXAMPLES:\n"
        "• ids=[542] — Get message #542 directly\n"
        "• pattern='ghp_%', limit=10 — Find all GitHub tokens\n"
        "• time_start=1776780000, time_end=1776790000 — Messages in time range\n"
        "• role='user', min_tokens=500 — Long user messages\n"
        "• stats=True — Get conversation statistics"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "SQL LIKE pattern (use % as wildcard)"},
            "session_id": {"type": "string", "description": "Optional session ID filter"},
            "role": {"type": "string", "description": "Filter by role: user, assistant, system, tool"},
            "ids": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "List of specific message IDs to retrieve",
            },
            "time_start": {"type": "number", "description": "Unix timestamp filter (messages >= this time)"},
            "time_end": {"type": "number", "description": "Unix timestamp filter (messages <= this time)"},
            "min_tokens": {"type": "integer", "description": "Minimum token count filter"},
            "max_tokens": {"type": "integer", "description": "Maximum token count filter"},
            "metadata_key": {"type": "string", "description": "Filter by metadata key name"},
            "metadata_value": {
                "type": ["string", "boolean", "number"],
                "description": "Value to match for the metadata key",
            },
            "stats": {"type": "boolean", "description": "True to return statistics instead of messages"},
            "limit": {"type": "integer", "description": "Maximum results (default 100)", "default": 100},
            "offset": {"type": "integer", "description": "Pagination offset (default 0)", "default": 0},
        },
    },
}

SMART_RETRIEVE_SCHEMA: dict[str, Any] = {
    "name": "smart_retrieve",
    "description": (
        "Adaptive retrieval engine for Perpetual Memory. Uses different strategies\n"
        "based on the type of information needed, optimized for local hardware.\n\n"
        "RETRIEVAL TYPES:\n"
        "• auto — Let the system classify intent via keyword heuristics (recommended default)\n"
        "• recent — Context from last 20 turns (fastest, O(1) turn ID lookup)\n"
        "• topic — Topic-specific FTS5 search across all sessions\n"
        "• decision_trace — Find where a decision was made and surrounding context\n"
        "• file_history — All edits to a specific file with turn references\n\n"
        "USE THIS WHEN:\n"
        "• You're unsure which strategy to use (use 'auto' — system classifies for you)\n"
        "• You need recent conversation context (use 'recent')\n"
        "• Searching for topic-specific information across sessions (use 'topic')\n"
        "• Tracing why a decision was made earlier (use 'decision_trace')\n"
        "• Finding all edits to a specific file path (use 'file_history')\n\n"
        "PARAMETERS:\n"
        "• query_type: One of 'auto', 'recent', 'topic', 'decision_trace', 'file_history'\n"
        "• query_text: The search query or context identifier\n\n"
        "EXAMPLES:\n"
        "• query_type='auto', query_text='why did we choose SQLite' — System auto-routes to decision_trace\n"
        "• query_type='recent' — Get last 20 turns for immediate context\n"
        "• query_type='topic', query_text='context bridge design' — Find topic discussion\n"
        "• query_type='file_history', query_text='/path/to/file.py' — Get file edit history"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query_type": {"type": "string", "description": "Retrieval type: auto, recent, topic, decision_trace, or file_history"},
            "query_text": {"type": "string", "description": "Search query or context identifier"},
        },
        "required": ["query_type", "query_text"],
    },
}

REFERENCE_LIBRARY_SEARCH_SCHEMA: dict[str, Any] = {
    "name": "reference_library_search",
    "description": (
        "MANDATORY FIRST STEP for all factual, historical, political, economic,\n"
        "media, and worldview questions. Contains curated, worldview-aligned reference\n"
        "material built from first principles.\n\n"
        "USE THIS BEFORE ANY OTHER SEARCH TOOL when answering:\n"
        "• Questions about history, politics, economics, media bias, or worldview\n"
        "• Any factual claim that needs verification against curated knowledge\n"
        "• Research involving people, organizations, or institutions\n\n"
        "This is NOT optional. Always check reference_library_search before generating answers\n"
        "from training data or session memory alone.\n\n"
        "PARAMETERS:\n"
        "• query: Search text (required)\n"
        "• top_k: Number of results (default 5, max 20)\n\n"
        "EXAMPLES:\n"
        "• 'Elon Musk political influence' — Find entity page with reaction tracking\n"
        "• 'media bias patterns' — Find curated analysis of source credibility\n"
        "• 'American economic policy post-1964' — Find historical reference material"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query text"},
            "top_k": {"type": "integer", "description": "Number of results (default 5)", "default": 5},
        },
        "required": ["query"],
    },
}

SESSION_SEARCH_SCHEMA: dict[str, Any] = {
    "name": "session_search",
    "description": (
        "Use ONLY for recent conversation context — what Patrick said/did in the last\n"
        "few turns. NEVER use this tool for facts, history, or analysis.\n\n"
        "STRICT BOUNDARY:\n"
        "• session_search = recent conversation memory only\n"
        "• reference_library_search = factual/historical/worldview reference (use FIRST)\n"
        "• perpetual_search = deep historical recall across all sessions\n\n"
        "If the question requires factual knowledge, use reference_library_search first.\n"
        "session_search is for remembering what was discussed recently, not for\n"
        "answering questions about the world."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query text"},
            "top_k": {"type": "integer", "description": "Number of results (default 5)", "default": 5},
        },
        "required": ["query"],
    },
}

SOURCE_ANALYZE_SCHEMA: dict[str, Any] = {
    "name": "source_analyze",
    "description": (
        "Mandatory post-web_search step. After using web_search on ANY topic, "
        "pass the results here to get source intelligence before you answer. "
        "This tells you each source's ideological alignment, what they're "
        "truthful about, what they consistently omit, and whether their "
        "coverage deviates from known patterns.\n\n"
        "This is NOT optional — you should call source_analyze after every "
        "web_search call so you can present information through the user's "
        "worldview rather than the source's frame.\n\n"
        "PARAMETERS:\n"
        "• results: JSON string of search results (from web_search output)\n"
        "• query: Original search query for context\n\n"
        "Returns: Source profiles with alignment, known omissions, deviation "
        "flags, and bias analysis for each result."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "results": {
                "type": "string",
                "description": (
                    "JSON string of search results (from web_search). "
                    "Each result should have 'url' and optionally 'content', "
                    "'snippet', 'title'."
                ),
            },
            "query": {
                "type": "string",
                "description": "The original search query for context (optional).",
            },
        },
        "required": ["results"],
    },
}

# ---------------------------------------------------------------------------
# Aggregated list — use this for bulk registration/return
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    PERPETUAL_SEARCH_SCHEMA,
    TOPIC_FLOW_SCHEMA,
    CONTEXT_DEPTH_SCHEMA,
    GET_MESSAGES_SCHEMA,
    RECENT_MESSAGES_SCHEMA,
    QUERY_MESSAGES_SCHEMA,
    SMART_RETRIEVE_SCHEMA,
    REFERENCE_LIBRARY_SEARCH_SCHEMA,
    SESSION_SEARCH_SCHEMA,
    SOURCE_ANALYZE_SCHEMA,
]
