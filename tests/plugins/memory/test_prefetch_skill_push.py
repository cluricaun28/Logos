"""Tests for the L2 skill push in the prefetch pipeline."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

from plugins.memory.perpetual_context import prefetch_pipeline as pp


def _make_rl_root(tmp_path: Path) -> Path:
    """Build a fake RL root with one skill pointer + backing SKILL.md."""
    rl_root = tmp_path / "rl"
    skills = rl_root / "skills"
    skills.mkdir(parents=True)
    skill_md = tmp_path / "SKILL.md"
    skill_md.write_text(
        textwrap.dedent("""\
        ---
        name: test-skill
        description: A test skill.
        ---

        # Test Skill

        Step 1: do the thing.
        Step 2: verify it.
        """),
        encoding="utf-8",
    )
    (skills / "test-skill.md").write_text(
        f'---\ntype: skill\nname: test-skill\nskill_path: {skill_md}\n---\n'
        "# test-skill\n\nA test skill.\n",
        encoding="utf-8",
    )
    return rl_root


def _skill_results(scores: dict[str, float]) -> list[dict]:
    out = []
    for name, score in scores.items():
        out.append({
            "file": f"{name}.md",
            "directory": "skills",
            "name": name,  # production passes the pointer title verbatim
            "score": score,
            "snippet": f"Snippet for {name}...",
        })
    return out


class TestBuildSkillPushBlock:
    def test_pushes_full_body_when_separated(self, tmp_path: Path) -> None:
        rl_root = _make_rl_root(tmp_path)
        results = _skill_results({"test-skill": 0.66, "other-skill": 0.40})
        block = pp._build_skill_push_block(
            results, rl_root, gap=0.15, min_score=0.5,
            max_chars=2400, max_candidates=2,
        )
        assert "[SKILL: test-skill]" in block
        assert "Step 1: do the thing." in block  # full body pushed
        assert "other-skill" in block  # candidate listed

    def test_no_push_when_candidates_close(self, tmp_path: Path) -> None:
        rl_root = _make_rl_root(tmp_path)
        results = _skill_results({"test-skill": 0.66, "other-skill": 0.55})
        block = pp._build_skill_push_block(
            results, rl_root, gap=0.15, min_score=0.5,
            max_chars=2400, max_candidates=2,
        )
        assert "[SKILL: test-skill]" not in block  # no confident push
        assert "other-skill" in block  # but candidate surfaced

    def test_nothing_below_min_score(self, tmp_path: Path) -> None:
        rl_root = _make_rl_root(tmp_path)
        results = _skill_results({"test-skill": 0.40})
        block = pp._build_skill_push_block(
            results, rl_root, gap=0.15, min_score=0.5,
            max_chars=2400, max_candidates=2,
        )
        assert block == ""

    def test_empty_results(self, tmp_path: Path) -> None:
        rl_root = _make_rl_root(tmp_path)
        assert pp._build_skill_push_block(
            [], rl_root, gap=0.15, min_score=0.5,
            max_chars=2400, max_candidates=2,
        ) == ""

    def test_unreadable_body_degrades_to_candidate(self, tmp_path: Path) -> None:
        rl_root = _make_rl_root(tmp_path)
        (rl_root / "skills" / "test-skill.md").unlink()  # pointer missing
        results = _skill_results({"test-skill": 0.66, "other-skill": 0.50})
        block = pp._build_skill_push_block(
            results, rl_root, gap=0.15, min_score=0.5,
            max_chars=2400, max_candidates=2,
        )
        assert "Step 1" not in block  # no body
        assert "test-skill" in block  # still named

    def test_body_truncated_with_pointer(self, tmp_path: Path) -> None:
        rl_root = _make_rl_root(tmp_path)
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text(
            "---\nname: test-skill\ndescription: A test skill.\n---\n"
            + "x" * 5000,
            encoding="utf-8",
        )
        results = _skill_results({"test-skill": 0.66})
        block = pp._build_skill_push_block(
            results, rl_root, gap=0.15, min_score=0.5,
            max_chars=100, max_candidates=2,
        )
        assert "truncated" in block
        assert "skill_view('test-skill')" in block


class _FakeTools:
    def __init__(self, results: list[dict]) -> None:
        self.results = results

    def handle_reference_library_search(self, args: dict[str, Any]) -> str:
        return json.dumps({"results": self.results[: args.get("top_k", 5)],
                           "count": len(self.results)})


class TestPipelineSkillPush:
    def _run(self, tmp_path: Path, results: list[dict], **overrides) -> str:
        kwargs = dict(
            query="do the test thing",
            routing={"fire_prefetch": True, "fire_recall": False,
                     "fire_web": False, "needs_recent_context": False},
            db=None,
            tools=_FakeTools(results),
            web_research=None,
            scrutiny_gate=None,
            source_analyzer=None,
            synthesis_engine=None,
            session_id="test-session",
            depth_limit=5,
            prefetch_enabled=True,
            recall_past_enabled=False,
            deep_research_enabled=False,
            prefetch_trunc_chars=1500,
            recall_output_max_chars=4000,
            rl_search_top_k=5,
            gap_detection_min_results=3,
            web_search_top_k=5,
            worldview_blocked_domains=frozenset(),
            deep_research_master=False,
        )
        kwargs.update(overrides)
        return pp.run_prefetch_pipeline(**kwargs)

    def test_skill_block_precedes_rl_pages(self, tmp_path: Path) -> None:
        rl_root = _make_rl_root(tmp_path)
        results = (
            _skill_results({"test-skill": 0.66, "other-skill": 0.40})
            + [{"file": "page.md", "directory": "topics",
                "name": "Some Page", "score": 0.66, "snippet": "page snippet"}]
        )
        out = self._run(tmp_path, results, rl_root=rl_root, skill_push_enabled=True)
        assert out.index("[SKILL: test-skill]") < out.index("[RL: Some Page (score")
        # skill pointers must NOT appear as generic RL snippets
        assert "[RL: Test Skill]" not in out

    def test_push_disabled_is_noop(self, tmp_path: Path) -> None:
        rl_root = _make_rl_root(tmp_path)
        results = _skill_results({"test-skill": 0.66}) + [
            {"file": "page.md", "directory": "topics",
             "name": "Some Page", "score": 0.66, "snippet": "page snippet"},
        ]
        out = self._run(tmp_path, results, rl_root=rl_root, skill_push_enabled=False)
        assert "[SKILL: test-skill]" not in out
        assert "[RL: Some Page (score: 0.66)]" in out

    def test_skill_only_results_still_produce_output(self, tmp_path: Path) -> None:
        rl_root = _make_rl_root(tmp_path)
        results = _skill_results({"test-skill": 0.66, "other-skill": 0.40})
        out = self._run(tmp_path, results, rl_root=rl_root, skill_push_enabled=True)
        assert "[SKILL: test-skill]" in out
        assert "Step 1: do the thing." in out
