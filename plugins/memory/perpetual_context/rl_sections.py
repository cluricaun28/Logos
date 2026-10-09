"""Section-level RL injection (WS3 — A/B 2026-10-07, shipped TWEAK).

Port of /data1/logos-sandbox/ws3-rl-section-injection/prototype.py plus the
envelope builder that implements the four TWEAK items from REPORT.md §8:

  1. Keep the 5K per-page envelope cap on summary + path-pointer + sections
     (uncapped measured 105% bloat — the cap is mandatory).
  2. Order selected sections by hit density (desc) BEFORE capping, so the
     cap eats the least-relevant tail instead of the most-relevant head.
  3. Emit the full page path pointer (makes every miss recoverable — it
     carries the 10/10 precision in the A/B).
  4. Treat archive/britannica/* and heading-less pages as whole-page (5K)
     — no splitting attempt (7.8% of the corpus, zero cut possible, and
     the 105% bloat source under the uncapped design).

Pure functions, no I/O beyond what the caller passes in, no deps beyond
stdlib.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

__all__ = [
    "Section",
    "split_page_by_heading",
    "select_sections",
    "term_hit_offsets",
    "page_summary",
    "injection_envelope",
    "build_page_envelope",
    "RL_ENVELOPE_CAP",
    "RL_MAX_SECTIONS",
]

RL_ENVELOPE_CAP = 5000       # TWEAK item 1: mandatory per-page envelope cap
RL_MAX_SECTIONS = 3          # k=3 sections/page max (the "bound" bound)

_ATX = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_TOKEN = re.compile(r"[A-Za-z0-9_]+")
_SUMMARY_MAX = 200


@dataclass
class Section:
    """One heading-delimited block of a markdown page.

    index 0 is the *preamble*: frontmatter + any content before the first
    heading (title '').  A page with no headings at all yields exactly one
    section whose text is the whole page — callers must measure that case
    honestly (section == page).
    """
    index: int
    title: str
    level: int          # 0 for preamble, else heading level (1-6)
    start: int          # char offset in the page (inclusive)
    end: int            # char offset in the page (exclusive)
    text: str = ""

    @property
    def chars(self) -> int:
        return len(self.text)


def split_page_by_heading(page: str) -> list[Section]:
    """Split raw markdown into heading-delimited sections.

    A section starts at a heading line and runs up to the NEXT heading line
    of ANY level.  Heading lines inside fenced code blocks (``` / ~~~) do
    not split.  Offsets are character offsets into `page`; the sections
    partition the page exactly (start=0 .. end=len(page), no gaps).
    """
    if not page:
        return [Section(0, "", 0, 0, 0, "")]

    line_offsets: list[int] = []
    off = 0
    for line in page.split("\n"):
        line_offsets.append(off)
        off += len(line) + 1

    heading_lines: list[tuple[int, int, str]] = []  # (line_idx, level, title)
    in_fence = False
    for i, line in enumerate(page.split("\n")):
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = _ATX.match(line)
        if m:
            heading_lines.append((i, len(m.group(1)), m.group(2)))

    sections: list[Section] = []
    boundaries = [0] + [line_offsets[i] for i, _, _ in heading_lines]
    ends = boundaries[1:] + [len(page)]
    n_head = len(heading_lines)
    for idx, (s, e) in enumerate(zip(boundaries, ends)):
        if n_head == 0:
            level, title = 0, ""            # no headings: section == page
        elif idx == 0:
            level, title = 0, ""            # preamble (frontmatter + pre-heading text)
        else:
            li, level, title = heading_lines[idx - 1]
        sections.append(Section(idx, title, level, s, e, page[s:e]))
    return sections


def _section_at(sections: list[Section], pos: int) -> int:
    """Index of the section containing `pos` (clamped to page bounds)."""
    if not sections:
        return 0
    if pos <= sections[0].start:
        return 0
    for sec in sections:
        if sec.start <= pos < sec.end:
            return sec.index
    last = sections[-1]
    return last.index if pos < last.end or last.end <= pos <= last.end + 1 else last.index


def select_sections(hits: Iterable, page: str, k: Optional[int] = None,
                    adjacent: int = 0) -> list[Section]:
    """Pick the heading-delimited section(s) containing `hits`.

    hits: iterable of int char-offsets (or (start, end) pairs) into `page`.
    k: max number of sections to return (None = all).  When more sections
        contain hits than k allows, keep the ones with the most hits
        (ties: earliest section wins).
    adjacent: 0 or 1.  With 1, each maximal contiguous run of selected
        sections gains the section immediately AFTER it (the classic
        cross-section continuation case).  Returns sections in page order.
    """
    sections = split_page_by_heading(page)
    counts: dict[int, int] = {}
    for h in hits:
        pos = h[0] if isinstance(h, (tuple, list)) else h
        idx = _section_at(sections, int(pos))
        counts[idx] = counts.get(idx, 0) + 1
    if not counts:
        return []

    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    chosen = {idx for idx, _ in ranked[:k]} if k else set(counts)

    if adjacent:
        idxs = sorted(chosen)
        runs: list[list[int]] = []
        for i in idxs:
            if runs and i == runs[-1][-1] + 1:
                runs[-1].append(i)
            else:
                runs.append([i])
        for run in runs:
            nxt = run[-1] + 1
            if nxt < len(sections):
                chosen.add(nxt)

    return [sections[i] for i in sorted(chosen)]


def term_hit_offsets(page: str, terms: Iterable[str]) -> list[int]:
    """Char offsets of every query-term occurrence in `page`.

    Mirrors FTS5/unicode61 semantics closely enough for selection: a term
    hits when it appears as a whole alphanumeric token (hyphens and other
    punctuation act as separators, case-insensitive).  Returns offsets
    sorted, one per occurrence.
    """
    wanted = {t.lower() for t in terms if t and _TOKEN.fullmatch(t)}
    if not wanted:
        return []
    offsets: list[int] = []
    for m in _TOKEN.finditer(page):
        if m.group(0).lower() in wanted:
            offsets.append(m.start())
    return offsets


def page_summary(page: str) -> str:
    """One-line page summary for the bounded injection envelope.

    Preference: frontmatter `summary` (first sentence, truncated), else
    frontmatter `title`, else the first H1, else the first non-empty
    non-frontmatter line.  Always a single line, <= _SUMMARY_MAX chars.
    """
    text = page
    fm = {}
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", page, re.DOTALL)
    if m:
        for line in m.group(1).split("\n"):
            if ":" in line:
                k, _, v = line.partition(":")
                fm[k.strip().lower()] = v.strip().strip("\"'")
        text = page[m.end():]
    cand = fm.get("summary") or fm.get("title") or ""
    if not cand:
        h1 = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        cand = h1.group(1).strip() if h1 else ""
    if not cand:
        for line in text.split("\n"):
            if line.strip() and not line.startswith("#"):
                cand = line.strip()
                break
    cand = re.sub(r"\s+", " ", cand).strip()
    if len(cand) > _SUMMARY_MAX:
        cand = cand[:_SUMMARY_MAX].rsplit(" ", 1)[0] + "…"
    return cand


def injection_envelope(page: str, sections: Sequence[Section],
                       page_path: str) -> str:
    """The exact bounded payload: summary line + path pointer + sections."""
    parts = [page_summary(page), f"Full page (verbatim archive): {page_path}"]
    parts.extend(sec.text for sec in sections)
    return "\n\n".join(p for p in parts if p)


def _is_britannica(file_path: str) -> bool:
    """TWEAK item 4: archive/britannica/* is whole-page (no splitting)."""
    return file_path.startswith("archive/britannica/")


def _section_hit_counts(sections: list[Section], offsets: Sequence[int]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for pos in offsets:
        idx = _section_at(sections, pos)
        counts[idx] = counts.get(idx, 0) + 1
    return counts


def build_page_envelope(page: str, terms: Iterable[str], *, file_path: str,
                        abs_path: Optional[str] = None,
                        cap: int = RL_ENVELOPE_CAP,
                        k: int = RL_MAX_SECTIONS) -> str:
    """The shipped per-page injection payload (REPORT.md §8 TWEAK).

    summary line + full page path pointer + selected sections, total <= cap.

    Behavior:
      - archive/britannica/* or a heading-less page: whole page under the
        cap (TWEAK item 4 — no splitting attempt).
      - otherwise: sections containing term hits, k max, ordered by hit
        density (desc; ties: page order) BEFORE capping so the cap eats
        the least-relevant tail (TWEAK item 2).  No term hits: fall back
        to the first query-phrase occurrence, else the page top
        (semantic-only page — measured 0/100 in the A/B corpus).
      - the pointer always carries the absolute verbatim-archive path
        (TWEAK item 3 — every miss is recoverable via read_file).
    """
    pointer_path = abs_path or file_path
    summary = page_summary(page)
    header = f"{summary}\n\nFull page (verbatim archive): {pointer_path}"
    if not page:
        return header

    if _is_britannica(file_path):
        room = max(0, cap - len(header) - 2)
        body = page[:room]
        return header + ("\n\n" + body if body else "")

    sections = split_page_by_heading(page)
    if len(sections) == 1 and sections[0].title == "":
        # heading-less page: section == page (TWEAK item 4)
        room = max(0, cap - len(header) - 2)
        body = page[:room]
        return header + ("\n\n" + body if body else "")

    wanted = [t for t in terms if t and _TOKEN.fullmatch(t)]
    offsets = term_hit_offsets(page, wanted)
    if not offsets and wanted:
        # semantic-only fallback: first occurrence of the leading phrase
        phrase = " ".join(wanted[:3]).strip()
        m = re.search(re.escape(phrase), page, re.IGNORECASE) if phrase else None
        if m:
            offsets = [m.start()]
        else:
            offsets = [0]

    counts = _section_hit_counts(sections, offsets)
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    chosen = [idx for idx, _ in ranked[:k]] if k else [idx for idx, _ in ranked]

    # TWEAK item 2: density order (desc) BEFORE capping.
    chosen_by_density = sorted(chosen, key=lambda i: (-counts[i], i))

    budget = max(0, cap - len(header) - 2)
    parts: list[str] = []
    used = 0
    for idx in chosen_by_density:
        text = sections[idx].text
        add = len(text) + (2 if parts else 0)
        if used + add <= budget:
            parts.append(text)
            used += add
        else:
            room = budget - used - (2 if parts else 0)
            if room > 0:
                parts.append(text[:room])
                used = budget
            break

    if not parts:
        return header
    return header + "\n\n" + "\n\n".join(parts)
