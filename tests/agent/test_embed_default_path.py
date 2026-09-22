"""2026-09-21/22 fleet embedfix: default embedding-model path is centralized
and $HOME-based.

Both embedding consumers (semantic_vector context engine, perpetual-context
EmbeddingEngine) hardcoded ``~/.hermes/models/embeddings/all-MiniLM-L6-v2``.
The 9/21-22 fleet-wide silent degradation traced to dead model paths:
explicit config keys pointing at the pre-migration home (``/data1/.hermes/...``,
absent) in all 13 fleet configs, plus 3 users missing the ``$HOME/.hermes``
model symlink. The default is now one tested helper.

Resolution is $HOME-based on purpose — NOT get_logos_home(): the fleet sets
``HOME=/data1/agents/<u>`` (models under ``$HOME/.hermes/``) while
``HERMES_HOME`` points at the no-dot tree ``/data1/agents/<u>/hermes`` (no
models there).

Pure logic — no model load, no GPU, no network.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from logos_constants import default_embed_model_path
from plugins.context_engine.semantic_vector import SemanticVectorContextEngine

EXPECTED = Path(".hermes") / "models" / "embeddings" / "all-MiniLM-L6-v2"


def test_default_is_home_based(monkeypatch):
    monkeypatch.setenv("HOME", "/fakeuser")
    monkeypatch.delenv("LOGOS_HOME", raising=False)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    assert default_embed_model_path() == Path("/fakeuser") / EXPECTED


def test_default_ignores_hermes_home(monkeypatch):
    """Fleet: HERMES_HOME is a no-dot tree that does not carry models."""
    monkeypatch.setenv("HOME", "/fakeuser")
    monkeypatch.setenv("HERMES_HOME", "/data1/agents/u/hermes")
    monkeypatch.delenv("LOGOS_HOME", raising=False)
    assert default_embed_model_path() == Path("/fakeuser") / EXPECTED


def test_engine_default_is_home_based(monkeypatch):
    monkeypatch.setenv("HOME", "/fakeuser")
    monkeypatch.delenv("LOGOS_HOME", raising=False)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    e = SemanticVectorContextEngine()
    assert e._model_path == str(Path("/fakeuser") / EXPECTED)


def test_engine_explicit_path_still_wins(monkeypatch):
    monkeypatch.setenv("HOME", "/fakeuser")
    e = SemanticVectorContextEngine(model_path="/explicit/model")
    assert e._model_path == "/explicit/model"


def test_pcd_module_import_smoke():
    """perpetual_context_db's call site compiles and imports clean."""
    import agent.perpetual_context_db  # noqa: F401
