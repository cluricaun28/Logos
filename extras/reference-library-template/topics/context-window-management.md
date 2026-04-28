# Context Window Management for LLM Agents

**Last updated:** YYYY-MM-DD
**Status:** Research synthesis + practical recommendations

---

## TL;DR

The current paradigm of stuffing as much context as possible into the window and then compressing what doesn't fit is fundamentally at odds with how LLMs actually reason. A better approach: keep the working window lean for deep present-moment reasoning, offload completed work to permanent storage (SQLite + FTS5), and retrieve on demand during the thinking phase.

## The Problem with "Stuff It All In"

Most agent frameworks treat context windows like a bucket — fill it up, then compress when full. This creates three problems:

1. **Stale history pollutes reasoning.** Old turns about completed tasks compete with present-moment tokens for attention.
2. **Compression loses signal.** Summarizing 50 turns of debugging into "we fixed the GPU issue" throws away the *how* that you'll need next time.
3. **No retrieval strategy.** The model can't selectively recall — it sees everything equally, which means nothing stands out.

## The Alternative: Active Retrieval Architecture

Instead of passive retention, use a dual-memory system:

### Working Memory (Context Window)
- Holds only what's actively being worked on
- Aggressively prunes completed tasks
- Leaves clear "hooks" pointing to permanent storage (e.g., "see PM for full GPU debugging session")

### Permanent Storage (SQLite + FTS5)
- Every turn stored verbatim — no lossy compression
- Full-text search via FTS5 indexes
- Optional semantic embeddings for hybrid retrieval
- Query during the thinking phase, not as a separate step

## How It Works in Practice

```
User asks question → Agent calls recent_messages (last 10 turns)
→ Agent reads current prompt + recent context
→ Agent calls perpetual_search for topic-specific history
→ Agent formulates response using retrieved info + present context
→ Completed turns are archived to permanent storage
→ Context window stays lean for next turn
```

## Key Design Decisions

- **recent_messages is mandatory first step** — prevents loops, maintains continuity
- **Search during thinking, not after** — tool use IS the reasoning process
- **Archive aggressively** — if a task is done, it belongs in permanent storage, not working memory
- **Print task status every turn** — `[Tasks: 3/5 complete]` becomes searchable PM data

## Model Recommendations

For local inference with this architecture:
- **27B dense models** outperform MoE (e.g., 34B-A4H) for reasoning-heavy retrieval tasks
- Context window of 32K+ is sufficient when you're not stuffing stale history
- Qwen 3.6 27B, Hermes 3 27B, or similar are good starting points

## References

- Meta-Harness paper (Lee et al.) — turn-level retrieval as index
- Rolling window vs. compression debate in agent frameworks
- FTS5 full-text search performance benchmarks
