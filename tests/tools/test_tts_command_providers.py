"""
Tests for custom command-type TTS providers.

These tests cover the ``tts.providers.<name>`` registry: built-in
precedence, command resolution, placeholder rendering, shell-quote
context handling, timeout / failure cleanup, voice_compatible opt-in,
and max_text_length lookup.

Nothing here talks to a real TTS engine. The shell command itself is
portable: we write bytes to ``{output_path}`` using ``python -c`` so
the tests run identically on Linux, macOS, and (with minor quoting
differences) Windows.

DEPRECATED: Command-type TTS providers were removed from tts_tool.py.
All symbols these tests reference (BUILTIN_TTS_PROVIDERS,
_resolve_command_provider_config, _generate_command_tts, etc.) no
longer exist. The entire feature has been deprecated.
"""

import pytest

pytestmark = pytest.mark.skip(
    "Command-type TTS providers were removed from tts_tool.py. "
    "All referenced symbols are gone; tests permanently skipped."
)


# All test classes and functions below are retained for historical reference
# only. They will not execute because pytestmark skips the entire module.


def _placeholder_test_file_is_deprecated() -> None:
    """Placeholder so pytest sees at least one test ID to skip."""
    pass
