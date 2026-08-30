"""Pinned project briefs — project context that lives in the system prompt.

The system prompt is protected from every pruning path (semantic_vector
keeps all system messages; the rolling-window fallback keeps the head),
so a brief pinned here persists for the life of the project — long after
the conversation turns that defined the project were pruned. This is the
4/30 "system prompt must be protected" idea: summarize a project's goals,
guidance, and relevant data pointers, and preserve them from pruning by
adding them to the system prompt.

Storage: ``$HERMES_HOME/state/pinned/<name>.md`` with a tiny frontmatter
block::

    ---
    project: example-website     # display name (defaults to file stem)
    expires: 2026-09-30           # optional; expired briefs are archived
    priority: 1                   # optional; lower = injected first
    max_chars: 3000               # optional per-brief cap
    ---
    ## Goal
    ...
    ## Guidance
    ...
    ## Data pointers
    RL: reference-library/...     # keep knowledge in the library, not here

Design rules (deliberately minimal):
- A brief is a *pointer + intent*, not a data dump. Hard caps: per-brief
  and total characters.
- Deterministic ordering (priority, then name) so the prompt prefix is
  stable across turns — prefix-cache friendly on vLLM.
- Fail-open: any error yields no briefs; a brief can never break the
  system prompt.
"""

from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from logos_constants import logos_env

logger = logging.getLogger(__name__)

DEFAULT_PER_BRIEF_CHARS = 3000
DEFAULT_TOTAL_CHARS = 8000

_frontmatter_re = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split a brief into (metadata, body). Tolerant of missing blocks."""
    meta: dict = {}
    body = text
    m = _frontmatter_re.match(text)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                key, _, val = line.partition(":")
                meta[key.strip().lower()] = val.strip()
        body = text[m.end():]
    return meta, body


def load_briefs(pinned_dir: Path, now: datetime | None = None) -> list[dict]:
    """Return active briefs, archiving expired ones. Never raises."""
    now = now or datetime.now().astimezone()
    out: list[dict] = []
    try:
        if not pinned_dir.is_dir():
            return out
        archive = pinned_dir / "archive"
        for path in sorted(pinned_dir.glob("*.md")):
            try:
                text = path.read_text(encoding="utf-8")
                meta, body = _parse_frontmatter(text)
                # active: false → toggled OFF without deleting/archiving the
                # file. Re-add `active: true` (or remove the line) to re-enable.
                # Fail-open: any missing/other value counts as active.
                if str(meta.get("active", "true")).strip().lower() in (
                    "false", "0", "no", "off",
                ):
                    continue
                expires = meta.get("expires", "")
                if expires:
                    try:
                        exp_dt = datetime.fromisoformat(expires)
                        if exp_dt.tzinfo is None:
                            exp_dt = exp_dt.replace(tzinfo=now.tzinfo)
                        if exp_dt < now:
                            archive.mkdir(parents=True, exist_ok=True)
                            dest = archive / path.name
                            if dest.exists():
                                dest.unlink()
                            path.rename(dest)
                            logger.info("pinned brief %s expired — archived", path.name)
                            continue
                    except ValueError:
                        pass  # unparseable date: keep the brief
                try:
                    priority = int(meta.get("priority", "10"))
                except ValueError:
                    priority = 10
                out.append(
                    {
                        "name": path.stem,
                        "project": meta.get("project", path.stem),
                        "priority": priority,
                        "expires": expires,
                        "path": str(path),
                        "body": body.strip(),
                        "max_chars": int(meta.get("max_chars", DEFAULT_PER_BRIEF_CHARS))
                        if str(meta.get("max_chars", "")).isdigit()
                        else DEFAULT_PER_BRIEF_CHARS,
                    }
                )
            except Exception as e:  # noqa: BLE001 — one bad file can't sink the rest
                logger.debug("pinned brief %s unreadable: %s", path.name, e)
    except Exception as e:  # noqa: BLE001 — fail-open
        logger.debug("pinned briefs unavailable: %s", e)
    return out


def render_briefs(
    briefs: list[dict],
    max_total_chars: int = DEFAULT_TOTAL_CHARS,
) -> str:
    """Render briefs into a system-prompt section. Deterministic order."""
    if not briefs:
        return ""
    briefs = sorted(briefs, key=lambda b: (b["priority"], b["name"]))
    parts: list[str] = []
    used = 0
    for b in briefs:
        body = b["body"]
        cap = min(b["max_chars"], max(0, max_total_chars - used))
        if cap <= 0:
            break
        if len(body) > cap:
            body = body[:cap] + f"\n[…truncated — full brief at {b['path']}]"
        until = f"until {b['expires']}" if b.get("expires") else "until unpinned"
        parts.append(f"### {b['project']} ({until})\n{body}")
        used += len(body)
    return (
        "\n## Pinned Project Briefs\n"
        "Protected context: the project's goals and guidance below persist for "
        "the life of the project. They are a summary — details live in the "
        "Reference Library and Perpetual Memory, follow the pointers.\n" + "\n\n".join(parts)
    )


# -- module-level cache (fingerprint-keyed; the block is built every turn) --

_cache: tuple[float, str] | None = None


def _dir_fingerprint(d: Path) -> float:
    """Max mtime across the dir and its briefs.

    Dir mtime alone is not enough: overwriting an existing brief file
    (the normal update path) doesn't touch the dir mtime.
    """
    mt = d.stat().st_mtime
    for p in d.glob("*.md"):
        try:
            mt = max(mt, p.stat().st_mtime)
        except OSError:
            pass
    return mt


def get_pinned_block(pinned_dir: Path | None = None) -> str:
    """Rendered briefs section, or "". Cached on dir+file mtime."""
    global _cache
    try:
        d = pinned_dir or _default_pinned_dir()
        if not d.is_dir():
            return ""
        fp = _dir_fingerprint(d)
        if _cache and _cache[0] >= fp:
            return _cache[1]
        block = render_briefs(load_briefs(d))
        _cache = (fp, block)
        return block
    except Exception as e:  # noqa: BLE001 — fail-open
        logger.debug("pinned block failed: %s", e)
        return ""


def _default_pinned_dir() -> Path:
    hermes_home = logos_env("HOME") or os.path.expanduser("~/.hermes")
    return Path(hermes_home) / "state" / "pinned"


def invalidate_cache() -> None:
    """Test helper."""
    global _cache
    _cache = None
