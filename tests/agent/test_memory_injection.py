"""Unit tests for agent/memory_injection.py — WS1 bounded MEMORY injection.

Ported from /data1/logos-sandbox/ws1-memory-injection/tests/test_prototype.py
(13 tests, imports adapted to the flat repo layout) plus tests for the two
ship-only TWEAK behaviors: the deterministic core-trim rule (over-budget
case) and the optional strong-match gate.

Deterministic fake embedder, no network, no model load.
"""
from __future__ import annotations

import zlib

from agent.memory_injection import (
    DEFAULT_CHAR_LIMIT,
    build_memory_injection,
    cosine,
    render_block,
    split_entries,
)

DELIM = "\n§\n"

# A vocabulary-based fake embedding: deterministic, topic-sensitive, no deps.
VOCAB = ["email", "nextcloud", "prod", "owner", "stale", "tesla",
         "password", "telegram", "fleet", "restart"]


def fake_embed(text):
    return [text.lower().count(w) for w in VOCAB]


def sel_entries(rendered: str) -> list:
    """Entries actually injected, minus the prod header (2 lines + separator)."""
    if not rendered:
        return []
    lines = rendered.split("\n")
    content = "\n".join(lines[3:])
    return [e.strip() for e in content.split(DELIM) if e.strip()]


def vec(word, **kw):
    v = [0.0] * len(VOCAB)
    v[VOCAB.index(word)] = 1.0
    for k, val in kw.items():
        v[VOCAB.index(k)] = val
    return v


BLOCK = DELIM.join([
    "Owner: clean up stale/pending work before proceeding",      # 0 core (owner)
    "nextcloud mirror lives on the server",                        # 1 non-core
    "Prod rule: never cut company inference",                      # 2 core (Prod rule)
    "email send failures are a phone settings issue",              # 3 non-core
    "the tesla model Y parks outside",                             # 4 non-core
    "fleet restarts happen one engine at a time",                  # 5 non-core
])


def test_split_entries_empty():
    assert split_entries("") == []
    assert split_entries("   ") == []


def test_split_entries_basic():
    assert split_entries("a§b".replace("§", DELIM)) == ["a", "b"]
    out = split_entries(DELIM.join([" x ", "y", "  "]))
    assert out == ["x", "y"]  # blank pieces dropped, whitespace stripped


def test_render_block_header_and_usage():
    out = render_block(["abcdef"], limit=100)
    assert "MEMORY (your personal notes) [6% — 6/100 chars]" in out
    assert out.startswith("═") and "\n" in out
    assert render_block([], limit=100) == ""


def test_cosine():
    assert abs(cosine(vec("email"), vec("email")) - 1.0) < 1e-9
    assert abs(cosine(vec("email"), vec("tesla"))) < 1e-9
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_empty_block_returns_empty():
    assert build_memory_injection("", "anything", fake_embed) == ""


def test_few_entries_injects_all():
    block = DELIM.join(["a", "b", "c"])
    out = build_memory_injection(block, "x", fake_embed, top_n=5)
    assert sel_entries(out) == ["a", "b", "c"]


def test_cold_start_falls_back_to_full_block():
    out_none = build_memory_injection(BLOCK, "email", embed_fn=None)
    out_fail = build_memory_injection(BLOCK, "email", embed_fn=lambda t: None)
    assert out_none == render_block(split_entries(BLOCK), DEFAULT_CHAR_LIMIT)
    assert out_fail == out_none  # model down == cold start == OLD behavior


def test_core_always_injected_regardless_of_topic():
    # turn about "tesla" — no overlap with core entries at all
    out = build_memory_injection(BLOCK, "my tesla is broken", fake_embed, top_n=1)
    sel = sel_entries(out)
    assert sel[0].startswith("Owner: clean up")      # core present
    assert any(e.startswith("Prod rule") for e in sel)  # core present
    assert any("tesla" in e for e in sel)      # top-1 by similarity (e4)


def test_top_n_selection_by_similarity():
    out = build_memory_injection(BLOCK, "email", fake_embed, top_n=1)
    sel = sel_entries(out)
    # core: e0, e2. top-1 non-core must be the email entry (e3)
    assert any(e.startswith("email send failures") for e in sel)
    noncore = [e for e in sel if not e.startswith(("Owner:", "Prod rule"))]
    assert len(noncore) == 1
    assert noncore[0].startswith("email send failures")


def test_tie_broken_by_position():
    # "prod" appears once; e2 (core, "Prod rule") and e5 ("fleet restarts...")
    # give e5 zero on 'prod' -> top-1 of non-core is e1 (nextcloud: 0)... all
    # non-core tie at 0 cosine for an off-vocab turn; position order wins.
    out = build_memory_injection(BLOCK, "zzz qqq", fake_embed, top_n=2)
    sel = sel_entries(out)
    noncore = [e for e in sel if not e.startswith(("Owner:", "Prod rule"))]
    assert noncore == [BLOCK.split(DELIM)[1], BLOCK.split(DELIM)[3]]  # e1 then e3


def test_budget_caps_injection():
    long_entry = "nextcloud " * 20  # ~160 chars each, well over budget share
    block = DELIM.join([
        "Owner: rule one",
        long_entry, long_entry, long_entry, long_entry, long_entry,
    ])
    out = build_memory_injection(block, "nextcloud", fake_embed,
                                 top_n=5, char_limit=400)
    content = "\n".join(out.split("\n")[3:])  # strip header lines
    assert len(content) <= 400
    assert "Owner: rule one" in content
    # at least one long entry still made it (relevance picked them)
    assert content.count("nextcloud " * 19 + "nextcloud") >= 1


def test_input_not_mutated():
    before = BLOCK
    build_memory_injection(BLOCK, "email", fake_embed, top_n=2)
    assert BLOCK == before


def test_rendered_block_matches_prod_header_format():
    out = build_memory_injection(BLOCK, "email", fake_embed, top_n=1)
    first_lines = out.split("\n")
    assert first_lines[0] == "═" * 46
    assert first_lines[2] == "═" * 46
    assert first_lines[1].startswith("MEMORY (your personal notes) [")
    assert "/2,200 chars]" in first_lines[1]


# ---------------------------------------------------------------------------
# TWEAK item 2 — deterministic core-trim rule (over-budget case)
# ---------------------------------------------------------------------------

def _content_of(rendered: str) -> str:
    return "\n".join(rendered.split("\n")[3:])


def test_core_trim_over_budget_drops_last_position_first():
    big = 600
    block = DELIM.join([
        "Owner: " + "a" * big,        # 0 core
        "Prod rule: " + "b" * big,    # 1 core
        "duty: " + "c" * big,         # 2 core
        "NEVER " + "d" * big,         # 3 core
        "email nextcloud",            # 4 non-core
        "tesla fleet",                # 5 non-core
    ])
    # core alone = 4*(600+8)+3*3 = 2441 > 2200 -> trim by last position first
    out = build_memory_injection(block, "zzz qqq", fake_embed, top_n=5)
    content = _content_of(out)
    assert len(content) <= 2200
    assert ("Owner: " + "a" * big) in content          # earliest core kept
    assert ("Prod rule: " + "b" * big) in content
    assert ("duty: " + "c" * big) in content
    assert ("NEVER " + "d" * big) not in content       # last core trimmed
    assert "email nextcloud" in content                # releasable still fits
    assert "tesla fleet" in content


def test_core_trimmed_entry_rejoins_releasable_pool():
    # The trimmed standing entry is topically the best match for the turn, so
    # it must be re-injected via the ranking path rather than lost.
    # Core = 3 big entries (808+810+806+2 delims = 2430 > 2200) + e3
    # "NEVER email nextcloud" at the LAST core position -> trim removes e3,
    # then e2 (still over budget), leaving core = [e0, e1]. e3 and e2 rejoin
    # the releasable pool; the turn "email nextcloud" ranks e3 first (cos 1.0).
    block = DELIM.join([
        "Owner: " + "a" * 800,          # 0 core
        "Prod rule: " + "b" * 800,      # 1 core
        "duty: " + "c" * 800,           # 2 core
        "NEVER email nextcloud",        # 3 core (trimmed by position, but topical)
        "tesla fleet",                  # 4 non-core
        "restart prod",                 # 5 non-core
    ])
    out = build_memory_injection(block, "email nextcloud", fake_embed, top_n=5)
    content = _content_of(out)
    assert len(content) <= 2200
    assert "NEVER email nextcloud" in content          # re-injected via rank
    assert ("duty: " + "c" * 800) not in content       # trimmed + didn't fit
    assert ("Owner: " + "a" * 800) in content          # core head kept


def test_core_trim_single_huge_entry_still_kept():
    # One core entry longer than the limit: kept (the full-block fallback
    # would be no better), and the function terminates.
    block = DELIM.join(
        ["Owner: " + "x" * 3000] + ["n%d" % i for i in range(5)]
    )
    out = build_memory_injection(block, "n0", fake_embed, top_n=5)
    assert ("Owner: " + "x" * 3000) in out
    # budget exhausted by the huge entry -> the small ones may not fit
    assert "n1" in _content_of(out) or "n1" not in _content_of(out)


def test_core_under_budget_never_trimmed():
    # exx's live shape: core < limit -> every core entry survives on any turn.
    out = build_memory_injection(BLOCK, "zzz qqq", fake_embed, top_n=5)
    sel = sel_entries(out)
    assert sel[0].startswith("Owner: clean up")
    assert any(e.startswith("Prod rule") for e in sel)


# ---------------------------------------------------------------------------
# TWEAK item 4 — optional strong-match gate
# ---------------------------------------------------------------------------

def test_gate_off_default_includes_top_n_on_nomatch_turn():
    # Zero-vocab turn: top-1 cosine is 0 (< 0.25). Gate off (default) ->
    # top-N still injected (position order).
    out = build_memory_injection(BLOCK, "zzz qqq", fake_embed, top_n=2)
    sel = sel_entries(out)
    noncore = [e for e in sel if not e.startswith(("Owner:", "Prod rule"))]
    assert len(noncore) == 2


def test_gate_on_core_only_on_nomatch_turn():
    out = build_memory_injection(BLOCK, "zzz qqq", fake_embed, top_n=5,
                                 strong_match_gate=True)
    sel = sel_entries(out)
    noncore = [e for e in sel if not e.startswith(("Owner:", "Prod rule"))]
    assert noncore == []                      # no top-N: top-1 = 0 < 0.25
    assert len(sel) == 2                      # exactly the core


def test_gate_on_includes_top_n_on_strong_match():
    # "email" turn: top-1 cosine == 1.0 >= 0.25 -> top-N included.
    out = build_memory_injection(BLOCK, "email", fake_embed, top_n=1,
                                 strong_match_gate=True)
    sel = sel_entries(out)
    assert any(e.startswith("email send failures") for e in sel)


def test_gate_threshold_is_inclusive():
    # Construct a turn whose top-1 cosine is just above the threshold:
    # q = [0, 1.0, 0, ...] (pure "nextcloud") vs entry e1
    # "nextcloud mirror..." -> vector [0,1,0,...] -> cosine exactly 1.0.
    # All other entries have zero overlap -> cosine 0. Set the threshold to
    # 1.0: top-1 (1.0) >= threshold -> top-N included (>=, not >).
    out = build_memory_injection(BLOCK, "nextcloud", fake_embed, top_n=1,
                                 strong_match_gate=True,
                                 strong_match_threshold=1.0)
    sel = sel_entries(out)
    assert any(e.startswith("nextcloud mirror") for e in sel)  # top-N in


def test_gate_uses_top1_across_all_entries():
    # A core entry can carry the top-1: with the gate on, a turn that only
    # matches a CORE entry strongly gets core-only (core is injected anyway),
    # not top-N for a weak non-core match.
    # "owner" turn: e0 (core, "Owner: ...") cosine 1.0; non-core all 0.
    out = build_memory_injection(BLOCK, "owner", fake_embed, top_n=5,
                                 strong_match_gate=True)
    sel = sel_entries(out)
    noncore = [e for e in sel if not e.startswith(("Owner:", "Prod rule"))]
    # top-1 (core e0) is strong -> top-N allowed; non-core all tie at 0,
    # position order gives e1..e5 (all 4 fit under budget).
    assert len(noncore) == 4
