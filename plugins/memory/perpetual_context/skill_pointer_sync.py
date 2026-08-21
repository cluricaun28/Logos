"""Skill pointer page sync — L2 skill-push retrieval surface.

Generates one RL pointer page per skill under ``<rl_root>/skills/`` from the
skill frontmatter, so the hybrid RL index (FTS5 + semantic) can retrieve
skills the same way it retrieves Reference Library pages.

Design (Phase 2a, W2):
  - SKILL.md remains the single source of truth; pointer pages are derived.
  - Pointer pages carry ``skill_path`` so the prefetch push can read the full
    SKILL.md body at injection time.
  - Idempotent: pages are only rewritten when content changes (stable mtime).
  - Stale pointer pages (skill deleted/renamed) are pruned.

The pointer page is the *embedding surface* for skill retrieval — its body
is the name + description (+ triggers), exactly what the recall probe gates.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from logos_constants import logos_env

logger = logging.getLogger(__name__)

FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
_POINTERS_MARKER = "type: skill"


def _parse_pointer_frontmatter(path: Path) -> dict[str, Any]:
    """Parse the frontmatter of a generated pointer page (best effort)."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}
    try:
        import yaml  # noqa: PLC0415

        parsed = yaml.safe_load(m.group(1))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:  # noqa: BLE001 — degradation, best effort
        return {}


def _yaml_scalar(value: str) -> str:
    """Render a scalar for YAML frontmatter, quoting only when needed.

    Quartz's frontmatter parser (and PyYAML) choke on unquoted values
    containing `: `, `#`, or an unclosed `"` — e.g. a skill named
    `Shopify: Admin` or a description cut mid-quote. yaml.safe_dump quotes
    only when the plain form is ambiguous, keeping simple names readable.

    PyYAML quirk (verified 2026-08-19): for plain scalars safe_dump appends
    a document-end marker on line 2 (`plain\\n...\\n`); drop that line.
    Values with embedded newlines fall back to JSON (a YAML subset).
    """
    import json  # noqa: PLC0415
    import yaml  # noqa: PLC0415

    if "\n" in value:
        return json.dumps(value)
    dumped = yaml.safe_dump(value, default_flow_style=True, width=4096).strip()
    return dumped.split("\n", 1)[0].strip() if "\n" in dumped else dumped


def _render_pointer(name: str, frontmatter: dict[str, Any], description: str,
                    skill_path: Path) -> str:
    """Render the pointer page content for one skill."""
    category = str(frontmatter.get("category") or "general").strip() or "general"
    priority = str(frontmatter.get("priority") or "high").strip() or "high"
    triggers = frontmatter.get("trigger_keywords") or frontmatter.get("triggers") \
        or frontmatter.get("trigger") or ""
    if isinstance(triggers, (list, tuple)):
        trigger_line = "\n".join(f"- {str(t).strip()}" for t in triggers if str(t).strip())
    else:
        trigger_line = str(triggers).strip()

    lines = [
        "---",
        "type: skill",
        f"name: {_yaml_scalar(name)}",
        f"title: {_yaml_scalar(name)}",
        f"category: {_yaml_scalar(category)}",
        f"priority: {_yaml_scalar(priority)}",
        f"skill_path: {_yaml_scalar(str(skill_path))}",
        f"synced: {datetime.now(UTC).strftime('%Y-%m-%d')}",
        "---",
        f"# {name}",
        "",
        description.strip(),
        "",
    ]
    if trigger_line:
        lines += ["**Triggers:**", trigger_line, ""]
    return "\n".join(lines)


def collect_skill_metadata(skills_dirs: list[Path]) -> dict[str, dict[str, Any]]:
    """Scan skill directories and return {frontmatter_name: metadata}.

    Mirrors production discovery (agent.prompt_builder.build_skills_system_prompt):
    local dir first, external dirs after, local wins on name collision.
    """
    from agent.prompt_builder import _parse_skill_file  # noqa: PLC0415
    from agent.skill_utils import iter_skill_index_files  # noqa: PLC0415

    found: dict[str, dict[str, Any]] = {}
    for skills_dir in skills_dirs:
        if not skills_dir.exists():
            continue
        for skill_file in iter_skill_index_files(skills_dir, "SKILL.md"):
            try:
                is_compatible, frontmatter, _desc = _parse_skill_file(skill_file)
            except Exception as e:  # noqa: BLE001 — one bad file must not kill the sync
                logger.debug("skill_pointer_sync: skipping %s: %s", skill_file, e)
                continue
            if not is_compatible:
                continue
            name = str(frontmatter.get("name") or skill_file.parent.name).strip()
            if not name:
                continue
            # Local (earlier) dirs win on collision.
            if name in found:
                continue
            # Use the FULL frontmatter description — it is the retrieval
            # surface. extract_skill_description() truncates to 57 chars for
            # prompt display and must NOT be used for pointer pages.
            found[name] = {
                "frontmatter": frontmatter,
                "description": str(frontmatter.get("description") or "").strip(),
                "skill_path": skill_file.resolve(),
            }
    return found


def sync_skill_pointers(skills_dirs: list[Path], rl_root: Path) -> dict[str, int]:
    """Generate/prune pointer pages under ``<rl_root>/skills/``.

    Returns stats: {written, unchanged, pruned, total_skills}.
    """
    out_dir = rl_root / "skills"
    found = collect_skill_metadata(skills_dirs)
    stats = {"written": 0, "unchanged": 0, "pruned": 0, "total_skills": len(found)}

    # Render desired state in memory first.
    desired: dict[str, str] = {}
    for name, meta in found.items():
        desc = meta["description"]
        if not desc:
            # No description — pointer page would be a name-only stub; skip
            # (nothing to retrieve on) but keep it discoverable via L3 backstop.
            continue
        desired[name] = _render_pointer(name, meta["frontmatter"], desc, meta["skill_path"])

    if desired:
        out_dir.mkdir(parents=True, exist_ok=True)

    for name, content in desired.items():
        target = out_dir / f"{name}.md"
        try:
            existing = target.read_text(encoding="utf-8") if target.exists() else None
        except OSError:
            existing = None
        if existing == content:
            stats["unchanged"] += 1
            continue
        try:
            target.write_text(content, encoding="utf-8")
            stats["written"] += 1
        except OSError as e:
            logger.warning("skill_pointer_sync: failed writing %s: %s", target, e)

    # Prune: pointer pages whose skill no longer exists (or lost its description).
    if out_dir.exists():
        for stale in sorted(out_dir.glob("*.md")):
            if stale.stem in desired:
                continue
            # Only prune pages WE generated (guard against foreign files).
            try:
                head = stale.read_text(encoding="utf-8", errors="replace")[:400]
            except OSError:
                head = ""
            if _POINTERS_MARKER in head:
                try:
                    stale.unlink()
                    stats["pruned"] += 1
                    logger.info("skill_pointer_sync: pruned stale pointer %s", stale.name)
                except OSError as e:
                    logger.warning("skill_pointer_sync: failed pruning %s: %s", stale, e)

    return stats


def default_skills_dirs() -> list[Path]:
    """The same skill directories production prompt injection scans."""
    from agent.skill_utils import get_all_skills_dirs  # noqa: PLC0415
    from logos_constants import get_skills_dir  # noqa: PLC0415

    local = get_skills_dir()
    external = get_all_skills_dirs()[1:]
    return [local] + list(external)


def default_rl_root() -> Path:
    hermes_home = logos_env("HOME") or os.path.expanduser("~/.hermes")
    return Path(hermes_home) / "reference-library"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sync skill pointer pages into the RL tree.")
    parser.add_argument("--skills-dirs", nargs="*", default=None,
                        help="Skill directories (default: production skill dirs)")
    parser.add_argument("--rl-root", default=None,
                        help="RL root (default: $HERMES_HOME/reference-library)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    skills_dirs = [Path(d) for d in args.skills_dirs] if args.skills_dirs else default_skills_dirs()
    rl_root = Path(args.rl_root) if args.rl_root else default_rl_root()

    stats = sync_skill_pointers(skills_dirs, rl_root)
    print(
        f"skill_pointer_sync: {stats['total_skills']} skills → "
        f"{stats['written']} written, {stats['unchanged']} unchanged, {stats['pruned']} pruned "
        f"(rl_root={rl_root})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
