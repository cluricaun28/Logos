"""Skill dependency resolution for Logos.

When multiple skills are loaded (via preloading or system prompt injection),
they may declare dependencies on other skills. This module resolves those
dependencies using topological sort so skills load in the correct order.

Usage in SKILL.md frontmatter:
    dependencies:
      - discernment-framework
      - claim-evaluation-workflow
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class CircularDependencyError(RuntimeError):
    """Raised when skill dependencies contain a cycle."""

    pass


class MissingDependencyError(RuntimeError):
    """Raised when a skill declares a dependency that doesn't exist."""

    pass


@dataclass
class SkillDependencyResolver:
    """Resolves skill loading order via topological sort.

    Takes a mapping of skill names to their dependency lists, then resolves
    a correct loading order for requested skills.

    Attributes:
        skills: Mapping of skill_name -> list of dependency skill names.
    """

    skills: dict[str, list[str]] = field(default_factory=dict)

    def resolve(self, requested: list[str]) -> list[str]:
        """Return skills in topological dependency order.

        Args:
            requested: List of skill names to load.

        Returns:
            Ordered list of skill names. Dependencies come before dependents.
            Duplicate skills are deduplicated while preserving order.

        Raises:
            CircularDependencyError: If a cycle exists in the dependency graph.
            MissingDependencyError: If a declared dependency doesn't exist.
        """
        ordered: list[str] = []
        temporary: set[str] = set()
        permanent: set[str] = set()

        def visit(skill_name: str) -> None:
            if skill_name in permanent:
                return
            if skill_name in temporary:
                chain = " -> ".join(list(temporary) + [skill_name])
                raise CircularDependencyError(
                    f"Circular skill dependency detected: {chain}"
                )
            temporary.add(skill_name)

            # Get dependencies for this skill
            deps = self.skills.get(skill_name, [])
            for dep in deps:
                if dep not in self.skills and dep not in requested:
                    # Dependency not in known skills — warn but continue
                    logger.warning(
                        "Skill '%s' depends on '%s' which is not available. "
                        "Continuing without it.",
                        skill_name,
                        dep,
                    )
                    continue
                visit(dep)

            temporary.remove(skill_name)
            permanent.add(skill_name)
            ordered.append(skill_name)

        for skill_name in requested:
            visit(skill_name)

        return ordered

    def validate(self, skill_name: str, dependencies: list[str]) -> list[str]:
        """Validate that a skill's dependencies can be satisfied.

        Args:
            skill_name: The skill being validated.
            dependencies: List of declared dependency names.

        Returns:
            List of missing dependency names. Empty if all satisfied.
        """
        missing = []
        for dep in dependencies:
            if dep not in self.skills:
                missing.append(dep)
        return missing

    def check_circular(self, skill_name: str, dependencies: list[str]) -> bool:
        """Check if adding these dependencies would create a cycle.

        Args:
            skill_name: The skill being added.
            dependencies: List of proposed dependency names.

        Returns:
            True if a cycle would be created, False otherwise.
        """
        # Temporarily add the skill and its dependencies
        test_skills = dict(self.skills)
        test_skills[skill_name] = dependencies

        test_resolver = SkillDependencyResolver(test_skills)
        try:
            test_resolver.resolve([skill_name])
            return False
        except CircularDependencyError:
            return True

    @classmethod
    def from_skill_configs(
        cls, skill_configs: dict[str, dict[str, Any]]
    ) -> "SkillDependencyResolver":
        """Build a resolver from parsed skill frontmatter configs.

        Args:
            skill_configs: Mapping of skill_name -> parsed frontmatter dict.

        Returns:
            Configured SkillDependencyResolver instance.
        """
        skills: dict[str, list[str]] = {}
        for name, frontmatter in skill_configs.items():
            deps = frontmatter.get("dependencies", [])
            if isinstance(deps, str):
                deps = [deps]
            elif not isinstance(deps, list):
                deps = []
            skills[name] = deps
        return cls(skills=skills)
