"""Display-layer filtering for engine-injected context scaffolding.

The context engines inject model-facing scaffolding into message content:

- ``[Conversation State]`` topic maps (semantic_vector engine, prepended to
  the last assistant message at archive time).
- Context Bridge blocks ("## Active Tasks (with retrieval pointers)",
  "## Context Bridge") and todo snapshots ("[Your active task list was
  preserved across context compression]"), appended as synthetic user
  messages at archive time (run_agent.py ``_archive_context``).

These blocks are for the model, not human readers, and they persist in the
session store.  Every human-facing render/delivery surface (CLI display,
gateway platform delivery, cron delivery) must strip them before showing or
sending.  The stored data is never modified — stripping happens only on the
copy being displayed or delivered.

Sentinels mirror the producers:

- ``plugins/context_engine/semantic_vector/__init__.py``
  (``_build_state_map`` / ``_inject_state_map``)
- ``plugins/memory/perpetual_context/context_bridge_builder.py``
- ``tools/todo_tool.py`` (``format_for_injection``)
"""

import re

__all__ = ["strip_engine_scaffolding", "strip_state_map", "is_scaffolding_only"]

# A state map looks like:
#   [Conversation State]\n  #0 name: status (...)\n  #1 name: status (...)\n\n
# The engine always injects ``state_map + "\n\n" + original_content``, so the
# block ends at the first blank line after the header — or at end-of-string
# when the original content was empty.
_STATE_MAP_RE = re.compile(r"^\[Conversation State\]\n(?:[^\n]*\n)*?(?:\n|$)")

# Whole-message scaffolding sentinels (synthetic user messages appended at
# archive time).  A message whose entire content is one of these is hidden
# from humans entirely.
_SCAFFOLDING_ONLY_PREFIXES = (
    "[Conversation State]",
    "## Active Tasks (with retrieval pointers)",
    "## Context Bridge",
    "[Your active task list was preserved",
)


def strip_state_map(text: str) -> str:
    """Remove all leading ``[Conversation State]`` blocks.

    Repeats to handle legacy stacked maps (pre-fix archives injected a new
    map on top of the previous one).  Content that does not start with a
    state map is returned unchanged.
    """
    if not text:
        return text
    s = text
    while True:
        m = _STATE_MAP_RE.match(s)
        if not m:
            return s
        s = s[m.end():]


def is_scaffolding_only(text: str) -> bool:
    """True if the entire message is engine scaffolding (no human content)."""
    if not text:
        return False
    s = text.strip()
    if not s.startswith(_SCAFFOLDING_ONLY_PREFIXES):
        return False
    # A state-map-only message: map lines, nothing after the block.  Use the
    # RAW text for the strip — the trailing blank line is the map's
    # terminator, and strip() would remove it.
    if s.startswith("[Conversation State]"):
        return strip_state_map(text).strip() == ""
    # Bridge / todo-snapshot messages are wholly scaffolding by construction.
    return True


def strip_engine_scaffolding(text: str) -> str:
    """Return ``text`` with engine scaffolding removed, for human display.

    - A leading ``[Conversation State]`` block is dropped (through the first
      blank line after it), including stacked legacy maps.
    - A message that is entirely scaffolding (context bridge, todo snapshot,
      state map only) yields ``""`` so callers can skip rendering/sending.
    - Anything else is returned with only edge newlines normalized.
    """
    if not text:
        return text
    if is_scaffolding_only(text):
        return ""
    return strip_state_map(text).strip("\n")
