"""Tiered skill injection with priority-based tiering.

Uses the existing priority field (high/low) from skill frontmatter to determine
injection tiers:

  - Tier 0 (high priority): Always injected with full descriptions into context
  - Tier 1+ (low priority): Listed but not injected, available on-demand

Task-aware filtering further narrows the available set based on detected task
type from the user message.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

TASK_KEYWORDS = {
    "research": [
        "research", "analyze", "evaluate", "investigate", "discern", "bias",
        "source", "political", "media", "claim", "framing",
    ],
    "software-development": [
        "code", "debug", "implement", "build", "test", "refactor", "deploy",
        "programming", "python", "git",
    ],
    "browser": ["navigate", "browser", "website", "page", "click", "form"],
    "data-science": ["data", "model", "train", "dataset", "ml", "ai"],
    "communication": ["send", "message", "telegram", "discord", "email"],
    "mlops": [
        "dpo", "train", "fine-tune", "inference", "gpu", "cuda", "unsloth",
        "quantize", "model",
    ],
    "devops": [
        "docker", "cron", "config", "hermes", "gateway", "update", "install",
        "setup", "restart", "status", "log",
    ],
    "creative": [
        "draw", "create", "design", "image", "art", "generate", "diagram",
        "visual", "comic",
    ],
    "worldview": [
        "worldview", "faith", "theology", "reference library", "rl",
        "christian", "spiritual",
    ],
}


@dataclass
class SkillNode:
    """Represents a skill in the injection graph."""

    name: str
    category: str
    priority: str  # "high" or "low"
    description: str = ""
    dependencies: list[str] = field(default_factory=list)
    tier: int = 0
    platforms: list[str] = field(default_factory=list)
    conditions: dict = field(default_factory=dict)


class SkillInjectionManager:
    """Manages skill injection based on priority-based tiers and task type."""

    def __init__(self) -> None:
        self.graph: dict[str, SkillNode] = {}
        self.categories: dict[str, set[str]] = {}

    def build_graph(self, configs: dict[str, dict[str, Any]]) -> None:
        """Build the skill injection graph from skill configurations.

        Args:
            configs: Dict mapping skill names to their configs including
                     category, dependencies, description, and priority.
        """
        for name, config in configs.items():
            node = SkillNode(
                name=name,
                category=config.get("category", "general"),
                priority=config.get("priority", "high"),
                description=config.get("description", ""),
                dependencies=config.get("dependencies", []),
            )
            # Map priority to tier: high = 0, low = 1
            node.tier = 0 if node.priority == "high" else 1

            self.graph[name] = node
            self.categories.setdefault(node.category, set()).add(name)

    @staticmethod
    def detect_task_type(user_message: str) -> str | None:
        """Detect the task type from a user message.

        Returns the matching category name or None if no match found.
        """
        if not user_message:
            return None

        message_lower = user_message.lower()
        best_match = None
        best_count = 0

        for category, keywords in TASK_KEYWORDS.items():
            count = sum(1 for keyword in keywords if keyword in message_lower)
            if count > best_count:
                best_count = count
                best_match = category

        return best_match if best_count > 0 else None

    def get_injection_set(
        self,
        task_type: str | None = None,
        max_tier: int = 1,
    ) -> list[str]:
        """Get the set of skills to inject based on task type and tier.

        Tier 0 (high priority) skills are always included. Tier 1+ (low priority)
        skills are included only if within max_tier and, when task_type is
        specified, belong to a relevant category.

        Args:
            task_type: Detected task type for category filtering.
            max_tier: Maximum tier to include in the injection set.

        Returns:
            List of skill names to inject, ordered by tier then alphabetically.
        """
        if not self.graph:
            return []

        # Determine target categories if task_type is specified
        target_categories = None
        if task_type and task_type in self.categories:
            target_categories = {task_type}

        injection_set = []

        # Sort by tier, then alphabetically within tiers
        sorted_skills = sorted(
            self.graph.values(),
            key=lambda node: (node.tier, node.name),
        )

        for node in sorted_skills:
            if node.tier > max_tier:
                continue

            # If task_type filtering is active, skip unrelated categories for
            # Tier 1+ skills only (Tier 0 is always included)
            if node.tier > 0 and target_categories and node.category not in target_categories:
                continue

            injection_set.append(node.name)

        return injection_set

    def get_skill_category(self, skill_name: str) -> str:
        """Get the category for a skill name."""
        node = self.graph.get(skill_name)
        return node.category if node else "general"

    def format_injection_block(self, skills: list[str]) -> str:
        """Format the injection block for the system prompt.

        Args:
            skills: List of skill names to include.

        Returns:
            Formatted markdown block for the system prompt.
        """
        if not skills:
            return ""

        lines = [
            "## Skills (on-demand)",
            "Before replying, scan the skills below and load only those DIRECTLY relevant",
            "to your task — validate relevance before loading. Do NOT load a skill just",
            "because it shares keywords; ensure its instructions actually apply to what",
            "you're doing. Load with skill_view(name) when genuinely needed, not as a",
            "reflexive step.",
            "",
            "**Core (always active):**",
        ]

        # Separate core (Tier 0 / high priority) from available (Tier 1+ / low priority)
        core_skills = [name for name in skills if self.graph[name].tier == 0]
        other_skills = [name for name in skills if self.graph[name].tier > 0]

        # Core skills get full descriptions
        for name in sorted(core_skills):
            node = self.graph[name]
            if node.description:
                lines.append(f"- ``{name}`` — {node.description}")
            else:
                lines.append(f"- ``{name}``")

        # Other skills are listed without descriptions (available on-demand)
        if other_skills:
            lines.append("")
            lines.append("**Available (load on-demand):**")
            for name in sorted(other_skills):
                lines.append(f"- ``{name}``")

        lines.append("")
        lines.append("If a skill has issues, fix it with skill_manage(action='patch').")
        lines.append("After difficult/iterative tasks, offer to save as a skill.")
        lines.append("If a skill you loaded was missing steps, had wrong commands, or needed")
        lines.append("pitfalls you discovered, update it before finishing.")

        return "\n".join(lines)
