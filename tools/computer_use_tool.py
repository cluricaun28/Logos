"""Computer use tool wrapper — bridges the subpackage to the flat registry scanner.

The actual implementation lives in tools/computer_use/ (9 files). This wrapper
imports and registers it so the auto-discovery scanner (tools/registry.py)
finds it.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

from tools.registry import registry
from tools.computer_use.tool import (
    handle_computer_use,
    check_computer_use_requirements,
)
from tools.computer_use.schema import COMPUTER_USE_SCHEMA

registry.register(
    name="computer_use",
    toolset="computer_use",
    schema=COMPUTER_USE_SCHEMA,
    handler=handle_computer_use,
    check_fn=check_computer_use_requirements,
    emoji="🖥️",
    max_result_size_chars=200_000,  # screenshots can be large
)
