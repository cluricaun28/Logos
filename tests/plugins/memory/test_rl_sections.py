"""Pytest suite for the WS3 section-injection module (rl_sections.py).

Ported from /data1/logos-sandbox/ws3-rl-section-injection/test_prototype.py
(23 tests, imports adapted to the repo layout) plus tests for the shipped
envelope builder (build_page_envelope) implementing the REPORT.md §8 TWEAK:
5K cap, hit-density ordering before capping, full path pointer, and
britannica/heading-less whole-page treatment.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from plugins.memory.perpetual_context.rl_sections import (  # noqa: E402
    Section,
    build_page_envelope,
    injection_envelope,
    page_summary,
    select_sections,
    split_page_by_heading,
    term_hit_offsets,
)

PAGE = """---
title: Example Page
summary: A test page about lasers and optics.
---
Intro text before any heading.

## Lasers

Laser physics goes here.
Some more laser detail.

### Diode Lasers

Sub-section content about diodes.

## Optics

Optics content, including the word laser again.

## Appendix

Final section.
"""


def test_partition_exact():
    secs = split_page_by_heading(PAGE)
    assert secs[0].start == 0
    assert secs[-1].end == len(PAGE)
    for a, b in zip(secs, secs[1:]):
        assert a.end == b.start, "sections must partition the page exactly"


def test_no_headings_section_equals_page():
    page = "no headings at all\njust text\n"
    secs = split_page_by_heading(page)
    assert len(secs) == 1
    assert secs[0].title == ""
    assert secs[0].text == page


def test_empty_page():
    secs = split_page_by_heading("")
    assert len(secs) == 1 and secs[0].text == ""


def test_heading_structure():
    secs = split_page_by_heading(PAGE)
    titles = [s.title for s in secs]
    assert titles == ["", "Lasers", "Diode Lasers", "Optics", "Appendix"]
    assert [s.level for s in secs] == [0, 2, 3, 2, 2]
    assert secs[0].text.startswith("---")
    assert secs[1].text.startswith("## Lasers")
    assert secs[1].text.endswith("Some more laser detail.\n\n")


def test_fenced_heading_not_a_heading():
    page = "# Top\n\n```\n# not a heading\n```\n\ntail\n"
    secs = split_page_by_heading(page)
    assert [s.title for s in secs] == ["", "Top"]
    assert "# not a heading" in secs[1].text


def test_select_hit_in_one_section():
    secs = select_sections([PAGE.index("Laser physics")], PAGE)
    assert [s.title for s in secs] == ["Lasers"]


def test_select_dedupes_hits_same_section():
    hits = [PAGE.index("Laser physics"), PAGE.index("more laser detail")]
    secs = select_sections(hits, PAGE)
    assert [s.title for s in secs] == ["Lasers"]


def test_select_multiple_sections_sorted_by_hits():
    # 'laser' appears in Lasers (2x) and Optics (1x)
    offs = term_hit_offsets(PAGE, ["laser"])
    secs = select_sections(offs, PAGE)
    assert [s.title for s in secs] == ["Lasers", "Optics"]


def test_select_k_limits_and_prefers_density():
    offs = term_hit_offsets(PAGE, ["laser"])
    secs = select_sections(offs, PAGE, k=1)
    assert [s.title for s in secs] == ["Lasers"]  # 2 hits beats 1


def test_select_k_tie_prefers_earlier():
    page = "## A\nword x\n## B\nword y\n"
    offs = term_hit_offsets(page, ["word", "x", "y"])
    # 'word' hits A; x hits A; y hits B  -> A:2, B:1, k=1 -> A
    assert [s.title for s in select_sections(offs, page, k=1)] == ["A"]


def test_select_adjacent_adds_next_section():
    offs = term_hit_offsets(PAGE, ["diodes"])
    base = [s.title for s in select_sections(offs, PAGE)]
    adj = [s.title for s in select_sections(offs, PAGE, adjacent=1)]
    assert base == ["Diode Lasers"]
    assert adj == ["Diode Lasers", "Optics"]


def test_select_adjacent_at_last_section_no_overflow():
    offs = term_hit_offsets(PAGE, ["Final"])
    assert [s.title for s in select_sections(offs, PAGE, adjacent=1)] == ["Appendix"]


def test_select_adjacent_collapses_contiguous_run():
    # hits in two adjacent sections -> one adjacent (after the run), not two
    offs = term_hit_offsets(PAGE, ["detail", "diodes"])
    adj = [s.title for s in select_sections(offs, PAGE, adjacent=1)]
    assert adj == ["Lasers", "Diode Lasers", "Optics"]


def test_select_hit_in_preamble():
    page = "intro mentions target\n## H\nbody\n"
    offs = term_hit_offsets(page, ["target"])
    assert [s.title for s in select_sections(offs, page)] == [""]


def test_select_no_hits_returns_empty():
    assert select_sections([], PAGE) == []


def test_term_offsets_unicode61_style():
    offs = term_hit_offsets("self-hosted SelfHosted model models", ["self", "hosted", "model"])
    # 'SelfHosted' is a single alphanumeric token -> matches neither 'self'
    # nor 'hosted'; 'models' != 'model'.  Hits: self, hosted, model.
    assert len(offs) == 3


def test_term_offsets_case_insensitive():
    assert len(term_hit_offsets("Alpha BETA", ["beta"])) == 1


def test_tuple_hits():
    pos = PAGE.index("Laser physics")
    secs = select_sections([(pos, pos + 10)], PAGE)
    assert [s.title for s in secs] == ["Lasers"]


def test_page_summary_prefers_frontmatter():
    assert page_summary(PAGE) == "A test page about lasers and optics."


def test_page_summary_falls_back_to_h1():
    assert page_summary("# Hello World\n\ntext\n") == "Hello World"


def test_page_summary_truncates_long_lines():
    s = page_summary("x " * 500)
    assert len(s) <= 201 and "\n" not in s


def test_injection_envelope_contains_pointer_and_sections():
    secs = select_sections(term_hit_offsets(PAGE, ["diodes"]), PAGE)
    env = injection_envelope(PAGE, secs, "/rl/people/example.md")
    assert "/rl/people/example.md" in env
    assert "## Diode Lasers" in env
    assert "Laser physics goes here" not in env  # unselected section excluded


def test_section_dataclass_chars():
    s = Section(0, "", 0, 0, 5, "abcde")
    assert s.chars == 5


# ---------------------------------------------------------------------------
# Shipped envelope builder (build_page_envelope) — TWEAK items 1-4
# ---------------------------------------------------------------------------

def test_envelope_under_cap():
    env = build_page_envelope(PAGE, ["laser"], file_path="technology/example.md",
                              abs_path="/rl/technology/example.md")
    assert len(env) <= 5000
    assert "/rl/technology/example.md" in env  # TWEAK 3: full path pointer
    assert "A test page about lasers and optics." in env  # summary line
    assert "## Lasers" in env
    assert "## Optics" in env


def test_envelope_cap_is_mandatory_on_huge_page():
    # 3 hit sections of 6KB each -> uncapped would be 18KB; the 5K cap binds.
    page = "## A\n" + ("a word " * 1000) + "\n## B\n" + ("b word " * 1000) + "\n## C\n" + ("c word " * 1000)
    env = build_page_envelope(page, ["word"], file_path="tech/big.md",
                              abs_path="/rl/tech/big.md")
    assert len(env) <= 5000


def test_envelope_density_order_before_capping():
    # k=2. Section A (3 hits) is the densest; section B (2 hits) next;
    # section C (1 hit) is dropped by k. Within the cap, A must come before B
    # (density desc) even though C < B in page order is irrelevant here —
    # verify A precedes B in the envelope.
    page = (
        "## A\nalpha alpha alpha word\n"
        "## B\nbeta beta word\n"
        "## C\ngamma word\n"
    )
    env = build_page_envelope(page, ["alpha", "beta", "gamma", "word"],
                              file_path="tech/order.md", abs_path="/rl/tech/order.md",
                              k=2)
    assert env.index("## A") < env.index("## B")
    assert "## C" not in env  # k=2 drops the least dense hit section


def test_envelope_cap_eats_least_relevant_tail():
    # Two hit sections; only one fits under a tiny cap. The denser one must
    # be the one that survives (cap eats the least-relevant tail).
    a_body = "alpha " * 200   # section A: dense (term appears 200x + header)
    b_body = "quiet text\n" * 50
    page = (
        "## A\n" + a_body + "\n"
        "## B\n" + b_body + "alpha"  # B has exactly 1 hit
    )
    env = build_page_envelope(page, ["alpha"], file_path="tech/cap.md",
                              abs_path="/rl/tech/cap.md", cap=1200)
    assert "## A" in env
    assert len(env) <= 1200
    assert "quiet text" not in env  # least-relevant tail eaten by the cap


def test_envelope_britannica_whole_page_no_split():
    # TWEAK item 4: archive/britannica/* -> whole page under the cap,
    # no section selection (even though headings exist).
    page = "## One\n" + ("x " * 3000) + "\n## Two\n" + ("y " * 3000)
    env = build_page_envelope(page, ["x"], file_path="archive/britannica/vol-23/roman-law.md",
                              abs_path="/rl/archive/britannica/vol-23/roman-law.md")
    assert len(env) <= 5000
    assert "## One" in env          # page top injected whole...
    assert "## Two" not in env      # ...until the cap cuts (not by selection)
    assert env.index("## One") < len(env)


def test_envelope_headingless_whole_page():
    page = "just text, no headings at all\n" + ("filler " * 3000)
    env = build_page_envelope(page, ["filler"], file_path="religion/bible/lxx-greek/psalm-1.md",
                              abs_path="/rl/religion/bible/lxx-greek/psalm-1.md")
    assert len(env) <= 5000
    assert "just text, no headings at all" in env  # page top, not a section pick


def test_envelope_pointer_uses_absolute_path():
    env = build_page_envelope(PAGE, ["laser"], file_path="technology/example.md",
                              abs_path="/home/exx/.hermes/reference-library/technology/example.md")
    assert "Full page (verbatim archive): /home/exx/.hermes/reference-library/technology/example.md" in env


def test_envelope_semantic_fallback_to_page_top():
    # No term hits at all -> section 0 (preamble/page top) under the cap.
    page = "intro paragraph here\n## Top\ntop body\n## Mid\nmiddle paragraph\n"
    env = build_page_envelope(page, ["zebra"], file_path="people/z.md",
                              abs_path="/rl/people/z.md")
    assert "intro paragraph here" in env  # preamble selected as fallback


def test_envelope_empty_page():
    env = build_page_envelope("", ["x"], file_path="a/b.md", abs_path="/rl/a/b.md")
    assert "/rl/a/b.md" in env
