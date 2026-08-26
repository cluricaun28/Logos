"""P3 config honesty — context.fallback canonical section + dead-key warnings.

Specification (work order 2026-08-26-review-followup, phase P3):

* Legacy configs (top-level ``archiving:`` / ``compression:``) resolve to
  EXACTLY the same knob dict as before — fleet configs keep working verbatim.
* The new canonical ``context.fallback`` section takes precedence over the
  legacy top-level sections, and shadowing the legacy section produces an
  explicit warning naming it.
* Recognized-but-unread keys (``context.archiving``) produce ONE loud
  WARNING at startup naming the key and its live replacement — silent
  drops are the bug.
"""

import logging

import pytest

import run_agent


@pytest.fixture(autouse=True)
def _reset_warning_once_guard():
    """Per-test reset of the once-per-process warning dedupe guard."""
    run_agent._EMITTED_CONTEXT_CONFIG_WARNINGS.clear()
    yield
    run_agent._EMITTED_CONTEXT_CONFIG_WARNINGS.clear()


# --------------------------------------------------------------------------
# Legacy parity
# --------------------------------------------------------------------------

def test_legacy_archiving_section_resolves_verbatim():
    cfg = {"archiving": {"threshold": 0.6, "enabled": False,
                         "target_ratio": 0.25, "protect_last_n": 10}}
    resolved, warnings = run_agent.resolve_context_fallback_config(cfg)
    assert resolved == cfg["archiving"]
    assert warnings == []


def test_legacy_compression_fallback_when_archiving_absent():
    cfg = {"compression": {"threshold": 0.7}}
    resolved, warnings = run_agent.resolve_context_fallback_config(cfg)
    assert resolved == {"threshold": 0.7}
    assert warnings == []


def test_empty_archiving_falls_through_to_compression():
    """Matches the pre-P3 behaviour: falsy archiving → compression."""
    cfg = {"archiving": {}, "compression": {"threshold": 0.8}}
    resolved, _ = run_agent.resolve_context_fallback_config(cfg)
    assert resolved == {"threshold": 0.8}


def test_non_dict_sections_resolve_to_empty():
    cfg = {"archiving": "yes please"}
    resolved, _ = run_agent.resolve_context_fallback_config(cfg)
    assert resolved == {}


def test_empty_config_resolves_empty_without_warnings():
    resolved, warnings = run_agent.resolve_context_fallback_config({})
    assert resolved == {}
    assert warnings == []


# --------------------------------------------------------------------------
# New canonical path: context.fallback precedence
# --------------------------------------------------------------------------

def test_context_fallback_alone_is_used():
    cfg = {"context": {"engine": "semantic_vector",
                       "fallback": {"threshold": 0.42, "protect_last_n": 5}}}
    resolved, warnings = run_agent.resolve_context_fallback_config(cfg)
    assert resolved == {"threshold": 0.42, "protect_last_n": 5}
    assert warnings == []


def test_context_fallback_precedence_over_legacy_archiving():
    cfg = {
        "archiving": {"threshold": 0.5},
        "compression": {"threshold": 0.9},
        "context": {"fallback": {"threshold": 0.42}},
    }
    resolved, warnings = run_agent.resolve_context_fallback_config(cfg)
    assert resolved == {"threshold": 0.42}
    # shadowing the legacy section must be LOUD, not silent
    assert len(warnings) == 1
    assert "context.fallback" in warnings[0]
    assert "shadowed" in warnings[0]


def test_empty_fallback_defers_to_legacy():
    cfg = {"archiving": {"threshold": 0.5}, "context": {"fallback": {}}}
    resolved, warnings = run_agent.resolve_context_fallback_config(cfg)
    assert resolved == {"threshold": 0.5}
    assert warnings == []


# --------------------------------------------------------------------------
# Dead-key warnings
# --------------------------------------------------------------------------

def test_dead_context_archiving_key_flagged():
    cfg = {"context": {"engine": "semantic_vector",
                       "archiving": {"threshold": 0.9}}}
    _resolved, warnings = run_agent.resolve_context_fallback_config(cfg)
    assert any("context.archiving" in w and "context.fallback" in w
               for w in warnings)


def test_dead_key_warning_emitted_once_via_caplog(caplog):
    cfg = {"context": {"archiving": {"threshold": 0.9}}}
    _resolved, warnings = run_agent.resolve_context_fallback_config(cfg)
    with caplog.at_level(logging.WARNING, logger="run_agent"):
        run_agent.emit_context_config_warnings(warnings)
        run_agent.emit_context_config_warnings(warnings)  # dedupe: still one
    dead = [r for r in caplog.records
            if "context.archiving" in r.getMessage()
            and r.levelno == logging.WARNING]
    assert len(dead) == 1
    # names the live replacement
    assert "context.fallback" in dead[0].getMessage()


def test_shadow_warning_emitted_via_caplog(caplog):
    cfg = {"archiving": {"threshold": 0.5},
           "context": {"fallback": {"threshold": 0.4}}}
    _resolved, warnings = run_agent.resolve_context_fallback_config(cfg)
    with caplog.at_level(logging.WARNING, logger="run_agent"):
        run_agent.emit_context_config_warnings(warnings)
    assert any("context.fallback" in r.getMessage()
               and "shadowed" in r.getMessage()
               for r in caplog.records if r.levelno == logging.WARNING)
