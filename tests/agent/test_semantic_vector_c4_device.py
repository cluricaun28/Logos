"""c4-backbone — device-aware embedding for SemanticVectorContextEngine.

The engine used to hardcode ``device="cpu"`` (May 2026, conservative choice
to avoid vLLM GPU contention). On a 32 GB GPU, MiniLM (~90 MB) adds
negligible contention, so c4 mirrors the proven
``EmbeddingEngine._select_device_candidates`` pattern from
``agent/perpetual_context_db.py``:

  explicit ``device`` config or ``HERMES_EMBED_DEVICE`` > free-GPU ranked
  (most-free first, packed GPUs last) > CPU fallback.

The module-level model cache is keyed by (path, device) so a CPU-loaded
cache is never served to an instance that explicitly wants CUDA.
"""

import sys
import types

import pytest

import plugins.context_engine.semantic_vector as sv
from plugins.context_engine.semantic_vector import SemanticVectorContextEngine


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """Reset the module-level model cache and env per test (hermetic)."""
    monkeypatch.delenv("HERMES_EMBED_DEVICE", raising=False)
    monkeypatch.setattr(sv, "_model_cache", None)
    monkeypatch.setattr(sv, "_model_path_cache", "")
    monkeypatch.setattr(sv, "_model_device_cache", "")
    yield
    monkeypatch.setattr(sv, "_model_cache", None)
    monkeypatch.setattr(sv, "_model_path_cache", "")
    monkeypatch.setattr(sv, "_model_device_cache", "")


def _model_dir(tmp_path):
    d = tmp_path / "all-MiniLM-L6-v2"
    d.mkdir(exist_ok=True)
    return str(d)


def _fake_torch(monkeypatch, cuda_available=True, gpus=None):
    """Install a fake torch module: gpus = {index: free_bytes}."""
    gpus = gpus or {0: 8 * 1024 ** 3}
    torch = types.ModuleType("torch")

    class _Cuda:
        @staticmethod
        def is_available():
            return cuda_available

        @staticmethod
        def device_count():
            return len(gpus)

        @staticmethod
        def mem_get_info(i):
            free = gpus.get(i, 0)
            return free, 32 * 1024 ** 3

    torch.cuda = _Cuda()
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def _fake_st(monkeypatch, fail_devices=()):
    """Install a fake sentence_transformers; returns (calls, fake_cls)."""
    calls = []

    class FakeST:
        def __init__(self, path, device=None):
            calls.append(device)  # record the attempt BEFORE the fail check
            if device in fail_devices:
                raise RuntimeError(f"fake OOM on {device}")
            self.device = device
            self.path = path

    mod = types.ModuleType("sentence_transformers")
    mod.SentenceTransformer = FakeST
    monkeypatch.setitem(sys.modules, "sentence_transformers", mod)
    return calls, FakeST


def _engine(tmp_path, **kw):
    kw.setdefault("model_path", _model_dir(tmp_path))
    return SemanticVectorContextEngine(**kw)


class TestDeviceCandidates:
    def test_forced_device_config(self, tmp_path, monkeypatch):
        _fake_torch(monkeypatch, gpus={0: 8 * 1024 ** 3, 1: 4 * 1024 ** 3})
        e = _engine(tmp_path, device="cuda:3")
        assert e._device_candidates() == ["cuda:3"]

    def test_config_wins_over_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_EMBED_DEVICE", "cuda:2")
        e = _engine(tmp_path, device="cuda:3")
        assert e._device_candidates() == ["cuda:3"]

    def test_env_override_only(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_EMBED_DEVICE", "cuda:2")
        e = _engine(tmp_path)
        assert e._device_candidates() == ["cuda:2"]

    def test_no_cuda_returns_cpu(self, tmp_path, monkeypatch):
        _fake_torch(monkeypatch, cuda_available=False)
        e = _engine(tmp_path)
        assert e._device_candidates() == ["cpu"]

    def test_cuda_ranked_most_free_first(self, tmp_path, monkeypatch):
        _fake_torch(monkeypatch, gpus={0: 1 * 1024 ** 3, 1: 8 * 1024 ** 3,
                                       2: 20 * 1024 ** 3})
        e = _engine(tmp_path)
        # free: cuda:2 (20G), cuda:1 (8G); packed: cuda:0 (1G); then cpu
        assert e._device_candidates() == ["cuda:2", "cuda:1", "cuda:0", "cpu"]

    def test_all_packed_gpus_still_tried_before_cpu(self, tmp_path, monkeypatch):
        _fake_torch(monkeypatch, gpus={0: 512 * 1024 ** 2, 1: 1 * 1024 ** 3})
        e = _engine(tmp_path)
        assert e._device_candidates() == ["cuda:1", "cuda:0", "cpu"]

    def test_torch_import_failure_degrades_to_cpu(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)  # forces ImportError
        e = _engine(tmp_path)
        assert e._device_candidates() == ["cpu"]


class TestLoadModelDeviceAware:
    def test_loads_on_free_gpu_and_caches_device(self, tmp_path, monkeypatch):
        _fake_torch(monkeypatch, gpus={0: 8 * 1024 ** 3})
        calls, _ = _fake_st(monkeypatch)
        e = _engine(tmp_path)
        assert e._load_model() is True
        assert e.model is not None
        assert e.model.device == "cuda:0"
        assert calls == ["cuda:0"]
        m, p, d = sv._get_model_cache()
        assert m is e.model and d == "cuda:0"

    def test_gpu_failure_falls_back_to_cpu(self, tmp_path, monkeypatch):
        _fake_torch(monkeypatch, gpus={0: 8 * 1024 ** 3})
        calls, _ = _fake_st(monkeypatch, fail_devices={"cuda:0"})
        e = _engine(tmp_path)
        assert e._load_model() is True
        assert e.model.device == "cpu"
        assert calls == ["cuda:0", "cpu"]

    def test_all_candidates_fail_returns_false(self, tmp_path, monkeypatch):
        _fake_torch(monkeypatch, cuda_available=False)
        calls, _ = _fake_st(monkeypatch, fail_devices={"cpu"})
        e = _engine(tmp_path)
        assert e._load_model() is False
        assert e.model is None
        assert calls == ["cpu"]

    def test_missing_model_dir_returns_false(self, tmp_path, monkeypatch):
        _fake_torch(monkeypatch, cuda_available=False)
        _fake_st(monkeypatch)
        e = SemanticVectorContextEngine(model_path=str(tmp_path / "nope"))
        assert e._load_model() is False

    def test_missing_sentence_transformers_returns_false(self, tmp_path, monkeypatch):
        _fake_torch(monkeypatch, cuda_available=False)
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
        e = _engine(tmp_path)
        assert e._load_model() is False


class TestCacheDeviceKeying:
    def test_cpu_cache_not_served_to_explicit_cuda(self, tmp_path, monkeypatch):
        _fake_torch(monkeypatch, cuda_available=False)
        calls, _ = _fake_st(monkeypatch, fail_devices={"cuda:0"})
        # First engine: auto → cpu (only device available), loads + caches.
        e1 = _engine(tmp_path)
        assert e1._load_model() is True
        assert sv._get_model_cache()[2] == "cpu"

        # Second engine explicitly wants cuda:0 — must NOT reuse the cpu
        # cache; it should attempt a fresh load. The fake load fails on
        # cuda:0 (and there is no further candidate), leaving model None —
        # the point is the cache was not served.
        e2 = _engine(tmp_path, device="cuda:0")
        model = e2._get_embedding_model()
        assert model is None  # fresh load attempted, cpu cache not served
        assert calls[-1] == "cuda:0"

    def test_auto_reuses_prior_device(self, tmp_path, monkeypatch):
        _fake_torch(monkeypatch, cuda_available=False)
        calls, _ = _fake_st(monkeypatch)
        e1 = _engine(tmp_path)
        assert e1._load_model() is True
        assert calls == ["cpu"]

        # Auto (no explicit device) reuses whatever device was cached —
        # no second load.
        e2 = _engine(tmp_path)
        model = e2._get_embedding_model()
        assert model is e1.model
        assert calls == ["cpu"]
