"""2026-09-21 fleet embedfix: default embedding-model path is home-relative.

The hard-coded ``~/.hermes/models/embeddings/all-MiniLM-L6-v2`` defaults in
the semantic_vector context engine and the perpetual-context embedding
engine were dead paths for fleet homes (``HERMES_HOME=/data1/agents/<u>/hermes``,
``HOME=/home/<u>``), silently degrading the engine to tail-off-only pruning
and RL search to FTS-only (aaron's 9/4 + 9/11 A2A notes; fleet-wide
config patch 9/21). The default now resolves through ``get_logos_home()``
— the single source of truth for the home directory.

Pure logic — no model load, no GPU, no network.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from logos_constants import default_embed_model_path
from plugins.context_engine.semantic_vector import SemanticVectorContextEngine


def test_helper_prefers_logos_home(monkeypatch):
    monkeypatch.setenv("LOGOS_HOME", "/fake/logos")
    monkeypatch.setenv("HERMES_HOME", "/fake/hermes")
    assert default_embed_model_path() == Path(
        "/fake/logos/models/embeddings/all-MiniLM-L6-v2"
    )


def test_helper_legacy_hermes_home(monkeypatch):
    monkeypatch.delenv("LOGOS_HOME", raising=False)
    monkeypatch.setenv("HERMES_HOME", "/fake/hermes")
    assert default_embed_model_path() == Path(
        "/fake/hermes/models/embeddings/all-MiniLM-L6-v2"
    )


def test_helper_no_env_uses_default_home(monkeypatch):
    monkeypatch.delenv("LOGOS_HOME", raising=False)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setenv("HOME", "/fakeuser")
    p = default_embed_model_path()
    assert p.parent.name == "embeddings"
    assert p.name == "all-MiniLM-L6-v2"
    # get_logos_home() no-env resolution: ~/.logos or (existing) ~/.hermes
    assert p.parent.parent.parent in (
        Path("/fakeuser") / ".logos",
        Path("/fakeuser") / ".hermes",
    )


def test_engine_default_is_home_relative(monkeypatch):
    monkeypatch.setenv("LOGOS_HOME", "/fake/logos")
    e = SemanticVectorContextEngine()
    assert e._model_path == "/fake/logos/models/embeddings/all-MiniLM-L6-v2"


def test_engine_explicit_path_still_wins(monkeypatch):
    monkeypatch.setenv("LOGOS_HOME", "/fake/logos")
    e = SemanticVectorContextEngine(model_path="/explicit/model")
    assert e._model_path == "/explicit/model"


def test_pcd_module_import_smoke():
    """perpetual_context_db's call site compiles and imports clean."""
    import agent.perpetual_context_db  # noqa: F401
