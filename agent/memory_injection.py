"""Bounded, relevance-ranked MEMORY block injection (WS1 — A/B 2026-10-07).

Ships the TWEAK verdict of /data1/logos-sandbox/ws1-memory-injection/REPORT.md
(§7 "Specific changes before ship"). Pattern: bound what's injected, archive
verbatim, retrieve on demand.

    OLD: the per-user MEMORY block is injected in FULL every turn.
    NEW: inject a deterministic standing CORE (rule entries, no LLM) plus the
         top-N remaining entries by cosine similarity to the current turn
         (user message + last 3 prior messages). The full store stays
         verbatim in MEMORY.md — nothing is lost; only the injection is
         bounded and relevance-ranked.

Ship config (the five TWEAK items):
  1. N = 5 (N=3 cost a measured recall miss for only +9% savings).
  2. Core-trim rule: when the core alone exceeds char_limit, trim core
     entries by last-added/position order (last position first) until the
     block is <= char_limit. Before this rule the overflow was silent.
     Trimmed entries rejoin the releasable pool (a trimmed standing entry
     that is highly topical to the turn may still be worth injecting).
  3. Per-user validation ran against all 16 fleet memory blocks (no core
     drops at N=5; see report of the ship workstream).
  4. Optional "strong match" gate (top-1 cosine >= threshold -> include
     top-N; else core-only) as a config knob. Shipped default: OFF —
     measured on the frozen 200-turn corpus: 94.0% strong / 6.0% no-match,
     0 required-entry recall lost on the 2 judged no-match samples, 525
     chars saved per no-match turn (~1.4% fleet-mean). Recall-first rule:
     10 of 12 no-match turns have no human judgment, so the gate's recall
     risk in 83% of its activation zone is unproven — the small saving
     does not buy it a default-ON place.
  5. Byte-identical full-block fallback for embed-failure/cold-start:
     embed_fn returning None renders the exact OLD full block.

No services, no network, no LLM in the loop. Deterministic given inputs.
"""
from __future__ import annotations

import re
from typing import Callable, List, Optional, Sequence

ENTRY_DELIMITER = "\n§\n"
DEFAULT_CHAR_LIMIT = 2200          # prod memory_char_limit (memory_tool.py)
DEFAULT_TOP_N = 5                  # TWEAK item 1: ship N=5
DEFAULT_STRONG_MATCH_THRESHOLD = 0.25  # natural "strong topical match" gate (A/B §5i)
EMBED_TRUNC = 8000                 # match EMBED_MAX_CONTENT_LEN
_SEPARATOR = "═" * 46
DEFAULT_TITLE = "MEMORY (your personal notes)"

# ---------------------------------------------------------------------------
# Standing-core classifier — candidate A (keyword markers). Deterministic,
# no LLM, O(entries) per turn with pure string ops (A/B §3: F1 0.696 vs
# candidate B's 0.500; standing-pair coverage 16/18 vs 4/18).
# ---------------------------------------------------------------------------
_STANDING_MARKERS = (
    re.compile(r"\bowner\b", re.IGNORECASE),      # "Owner: ...", "(10/5, owner)"
    re.compile(r"\bNEVER\b"),                     # "NEVER raise Teresa's retirement"
    re.compile(r"\bProd rule\b"),                 # "Prod rule: impact-manage; ..."
    re.compile(r"\bduty\b", re.IGNORECASE),       # "Errlog duty (10/5, owner)"
    re.compile(r"need-to-know"),                  # data-boundary entry
    re.compile(r"Delegate ops"),                  # delegation/effort-cap entry
    re.compile(r"\bredact\b", re.IGNORECASE),     # secrets-handling practice
    re.compile(r"no reset"),                      # "NC web pw: no reset w/o the user"
)


def is_standing_rule(entry: str) -> bool:
    """True if the entry carries an explicit owner-ruling/standing-rule marker."""
    return any(m.search(entry) for m in _STANDING_MARKERS)


def split_entries(block: str) -> List[str]:
    """Split a raw MEMORY block into its entries (prod ENTRY_DELIMITER)."""
    if not block or not block.strip():
        return []
    return [e.strip() for e in block.split(ENTRY_DELIMITER) if e.strip()]


def render_block(entries: Sequence[str], limit: int = DEFAULT_CHAR_LIMIT,
                 title: str = DEFAULT_TITLE) -> str:
    """Render exactly like prod memory_tool.MemoryStore._render_block."""
    if not entries:
        return ""
    content = ENTRY_DELIMITER.join(entries)
    current = len(content)
    pct = min(100, int((current / limit) * 100)) if limit > 0 else 0
    header = f"{title} [{pct}% — {current:,}/{limit:,} chars]"
    return f"{_SEPARATOR}\n{header}\n{_SEPARATOR}\n{content}"


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def _content_chars(entries: Sequence[str]) -> int:
    """Char count of the joined content (same accounting as render_block)."""
    if not entries:
        return 0
    return len(ENTRY_DELIMITER.join(entries))


def _trim_core(core: List[int], entries: Sequence[str], limit: int) -> List[int]:
    """TWEAK item 2 — deterministic core trim for the over-budget case.

    Trim core entries by last-added/position order (last position first)
    until the core content is <= limit. Stops at one core entry: a single
    entry longer than the limit is kept (the fallback block would be no
    better), so the function always terminates.
    """
    core = list(core)
    while len(core) > 1 and _content_chars([entries[i] for i in core]) > limit:
        core.pop()  # last position == last added
    return core


def build_memory_injection(memory_block: str, turn_text: str,
                           embed_fn: Optional[Callable[[str], Optional[List[float]]]],
                           top_n: int = DEFAULT_TOP_N,
                           char_limit: int = DEFAULT_CHAR_LIMIT,
                           strong_match_gate: bool = False,
                           strong_match_threshold: float = DEFAULT_STRONG_MATCH_THRESHOLD,
                           ) -> str:
    """Return the system-prompt MEMORY block for this turn.

    Steps:
      1. Parse entries. If empty -> "".
      2. If total entries <= top_n: inject all (bounding is a no-op).
      3. Core = deterministic standing-rule entries (classifier A).
      4. Turn vector = embed_fn(turn_text[:8000]); if None (cold start or
         embedder down): fall back to the FULL block (OLD behavior,
         byte-identical).
      5. Optional strong-match gate: top-1 cosine (vs any entry) below the
         threshold -> core-only, no top-N (config knob, default off).
      6. Core-trim (TWEAK 2): if the core alone exceeds char_limit, trim
         by last-added/position order until it fits; trimmed entries rejoin
         the releasable pool.
      7. Rank the releasable entries by cosine to the turn; take top-N that
         fit under char_limit after the core (deterministic: position
         order on score ties).
      8. Render in file order so the block stays stable/cache-friendly.
    """
    entries = split_entries(memory_block)
    if not entries:
        return ""
    if len(entries) <= top_n:
        return render_block(entries, char_limit)

    n = len(entries)
    core = [i for i, e in enumerate(entries) if is_standing_rule(e)]
    core_set = set(core)

    qvec = embed_fn(turn_text[:EMBED_TRUNC]) if embed_fn else None
    if qvec is None:
        # Cold start / embedder down -> preserve current behavior exactly.
        return render_block(entries, char_limit)

    # Gate decision uses top-1 across ALL entries (matches the A/B threshold
    # analysis in REPORT.md §5i: 94% of turns have top-1 >= 0.25).
    use_topn = True
    if strong_match_gate:
        top1 = max(cosine(qvec, _embed_entry(entries[i], embed_fn)) for i in range(n))
        use_topn = top1 >= strong_match_threshold

    # TWEAK 2: deterministic core trim for the over-budget case.
    core_kept = _trim_core(core, entries, char_limit)
    trimmed = [i for i in core if i not in set(core_kept)]

    rest = [i for i in range(n) if i not in core_set] + trimmed

    scored = sorted(
        ((cosine(qvec, _embed_entry(entries[i], embed_fn)), i) for i in rest),
        key=lambda t: (-t[0], t[1]),
    )

    chosen = list(core_kept)
    used = _content_chars([entries[i] for i in chosen])
    if use_topn:
        for _score, i in scored:
            if len(chosen) >= len(core_kept) + top_n:
                break
            delta = len(entries[i]) + len(ENTRY_DELIMITER)
            if used + delta > char_limit:
                continue  # keep skipping; a later (shorter) entry may still fit
            chosen.append(i)
            used += delta
    chosen.sort()  # file order so the block stays stable/cache-friendly

    return render_block([entries[i] for i in chosen], char_limit)


_ENTRY_VEC_CACHE: dict[str, Optional[List[float]]] = {}


def _embed_entry(entry: str, embed_fn: Callable[[str], Optional[List[float]]]) -> List[float]:
    """Embed an entry with a tiny per-process cache (entries rarely change)."""
    v = _ENTRY_VEC_CACHE.get(entry)
    if v is None:
        v = embed_fn(entry[:EMBED_TRUNC]) or [0.0]
        _ENTRY_VEC_CACHE[entry] = v
    return v


def clear_entry_cache() -> None:
    """Drop the per-process entry-vector cache (entries changed)."""
    _ENTRY_VEC_CACHE.clear()
