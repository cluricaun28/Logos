"""Tests for skill pointer page sync (L2 skill-push retrieval surface)."""
from __future__ import annotations

import textwrap
from pathlib import Path

from plugins.memory.perpetual_context.skill_pointer_sync import sync_skill_pointers


def _write_skill(base: Path, rel: str, name: str, desc: str,
                 extra_frontmatter: str = "") -> Path:
    d = base / rel
    d.mkdir(parents=True, exist_ok=True)
    skill_md = d / "SKILL.md"
    skill_md.write_text(
        textwrap.dedent(f"""\
        ---
        name: {name}
        {extra_frontmatter}
        description: {desc}
        ---

        # {name}

        Body content for {name}.
        """),
        encoding="utf-8",
    )
    return skill_md


def test_sync_generates_pointer_pages(tmp_path: Path) -> None:
    skills = tmp_path / "skills"
    rl_root = tmp_path / "rl"
    _write_skill(skills, "alpha", "alpha-skill", "Do alpha things.",
                 "priority: high\n")
    _write_skill(skills, "beta/beta-skill", "beta-skill", "Do beta things.",
                 "priority: low\n")

    stats = sync_skill_pointers([skills], rl_root)

    assert stats["total_skills"] == 2
    assert stats["written"] == 2
    alpha = (rl_root / "skills" / "alpha-skill.md").read_text()
    assert "type: skill" in alpha
    assert "name: alpha-skill" in alpha
    assert "Do alpha things." in alpha
    assert "priority: high" in alpha
    # skill_path points at the real SKILL.md
    skill_path_line = [ln for ln in alpha.splitlines() if ln.startswith("skill_path:")][0]
    assert skill_path_line.split(":", 1)[1].strip() == str((skills / "alpha" / "SKILL.md").resolve())
    assert (rl_root / "skills" / "beta-skill.md").exists()


def test_sync_is_idempotent(tmp_path: Path) -> None:
    skills = tmp_path / "skills"
    rl_root = tmp_path / "rl"
    _write_skill(skills, "alpha", "alpha-skill", "Do alpha things.")

    first = sync_skill_pointers([skills], rl_root)
    assert first["written"] == 1
    second = sync_skill_pointers([skills], rl_root)
    assert second["written"] == 0
    assert second["unchanged"] == 1


def test_sync_prunes_stale_pointers(tmp_path: Path) -> None:
    skills = tmp_path / "skills"
    rl_root = tmp_path / "rl"
    a = _write_skill(skills, "alpha", "alpha-skill", "Do alpha things.")
    _write_skill(skills, "beta", "beta-skill", "Do beta things.")
    sync_skill_pointers([skills], rl_root)

    (skills / "alpha" / "SKILL.md").unlink()
    stats = sync_skill_pointers([skills], rl_root)

    assert stats["pruned"] == 1
    assert not (rl_root / "skills" / "alpha-skill.md").exists()
    assert (rl_root / "skills" / "beta-skill.md").exists()
    del a


def test_sync_keeps_foreign_files(tmp_path: Path) -> None:
    skills = tmp_path / "skills"
    rl_root = tmp_path / "rl"
    _write_skill(skills, "alpha", "alpha-skill", "Do alpha things.")
    sync_skill_pointers([skills], rl_root)

    foreign = rl_root / "skills" / "not-a-pointer.md"
    foreign.write_text("---\ntype: note\n---\nkeep me\n", encoding="utf-8")
    # Remove the backing skill so the alpha pointer becomes stale.
    (skills / "alpha" / "SKILL.md").unlink()
    (skills / "alpha").rmdir()

    stats = sync_skill_pointers([skills], rl_root)

    assert stats["pruned"] == 1  # alpha pointer pruned
    assert foreign.exists()      # foreign file untouched


def test_sync_handles_missing_dirs(tmp_path: Path) -> None:
    rl_root = tmp_path / "rl"
    stats = sync_skill_pointers([tmp_path / "nope"], rl_root)
    assert stats["total_skills"] == 0
    assert stats["written"] == 0
