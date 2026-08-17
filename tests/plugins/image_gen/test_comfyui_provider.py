#!/usr/bin/env python3
"""Tests for the local ComfyUI image generation provider.

Regression focus: the tool surface speaks the core vocabulary
(landscape/square/portrait — see VALID_ASPECT_RATIOS in
agent/image_gen_provider.py). The ComfyUI dimension map once contained
only ratio-string keys ("16:9", ...), so every tool request silently
fell back to 1:1. These tests guard the vocabulary mapping and the
16-grid dimension contract.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.image_gen_provider import VALID_ASPECT_RATIOS, resolve_aspect_ratio
from plugins.image_gen.comfyui import ComfyUIProvider, _ASPECT_DIMS, _MODELS

# Expected dimensions for the core vocabulary (16:9 wide / 9:16 tall / 1:1).
EXPECTED_CORE_DIMS = {
    "landscape": (1536, 864),
    "square": (1344, 1344),
    "portrait": (864, 1536),
}


class TestAspectMapping:
    def test_core_vocabulary_keys_present(self):
        """Every tool-surface ratio must resolve to a real dimension.

        This is the invariant that broke on 2026-08-17: 'landscape' was
        absent from _ASPECT_DIMS and every request fell back to 1:1.
        """
        for ratio in VALID_ASPECT_RATIOS:
            assert ratio in _ASPECT_DIMS, (
                f"core ratio {ratio!r} missing from _ASPECT_DIMS — tool requests "
                f"would silently fall back to 1:1"
            )

    def test_core_vocabulary_dimensions(self):
        for ratio, dims in EXPECTED_CORE_DIMS.items():
            assert _ASPECT_DIMS.get(ratio) == dims

    def test_all_dimensions_on_16_grid(self):
        for ratio, (w, h) in _ASPECT_DIMS.items():
            assert w % 16 == 0 and h % 16 == 0, f"{ratio}: {w}x{h} not on 16-grid"

    def test_unknown_ratio_coerced_to_default(self):
        assert resolve_aspect_ratio("banana") in VALID_ASPECT_RATIOS
        assert resolve_aspect_ratio(None) in VALID_ASPECT_RATIOS


class TestGenerate:
    def _provider_with_fakes(self, monkeypatch, captured: dict):
        provider = ComfyUIProvider()
        monkeypatch.setattr(provider, "is_available", lambda: True)
        monkeypatch.setattr(provider, "_submit", lambda url, wf: captured.update(wf=wf) or "pid-1")

        def _fake_workflow(prompt, model, width, height):
            captured["dims"] = (width, height)
            return {"prompt": {}}

        monkeypatch.setattr(provider, "_workflow", _fake_workflow)
        monkeypatch.setattr(provider, "_wait", lambda url, pid: {"fn": "SaveImage", "images": [{}]})
        monkeypatch.setattr(provider, "_fetch_image", lambda url, meta: b"\x89PNG fake-bytes")
        monkeypatch.setattr(
            "plugins.image_gen.comfyui.save_b64_image",
            lambda b64, prefix: f"/tmp/{prefix}_test.png",
        )
        return provider

    @pytest.mark.parametrize("ratio,expected", sorted(EXPECTED_CORE_DIMS.items()))
    def test_generate_uses_requested_aspect(self, monkeypatch, ratio, expected):
        captured: dict = {}
        provider = self._provider_with_fakes(monkeypatch, captured)
        result = provider.generate(prompt="a test subject", aspect_ratio=ratio)
        assert result["success"] is True
        assert captured["dims"] == expected
        assert result["resolution"] == f"{expected[0]}x{expected[1]}"
        assert result["aspect_ratio"] == ratio

    def test_generate_invalid_aspect_falls_back_to_default(self, monkeypatch):
        captured: dict = {}
        provider = self._provider_with_fakes(monkeypatch, captured)
        result = provider.generate(prompt="a test subject", aspect_ratio="definitely-not-a-ratio")
        assert result["success"] is True
        # resolve_aspect_ratio coerces to the default (landscape).
        assert captured["dims"] == EXPECTED_CORE_DIMS["landscape"]

    def test_generate_honors_explicit_model(self, monkeypatch):
        captured: dict = {}
        provider = self._provider_with_fakes(monkeypatch, captured)
        result = provider.generate(prompt="a product", aspect_ratio="square", model="flux2-klein-4b")
        assert result["success"] is True
        assert result["model"] == "flux2-klein-4b"
        assert captured["dims"] == EXPECTED_CORE_DIMS["square"]

    def test_unknown_explicit_model_falls_back_to_default(self, monkeypatch):
        # Forgiving contract: a bad per-request model is ignored, not an error.
        captured: dict = {}
        provider = self._provider_with_fakes(monkeypatch, captured)
        result = provider.generate(prompt="x", model="no-such-model")
        assert result["success"] is True
        assert result["model"] == "qwen-image"

    def test_config_model_unknown_errors(self, monkeypatch):
        captured: dict = {}
        provider = self._provider_with_fakes(monkeypatch, captured)
        monkeypatch.setattr(
            "plugins.image_gen.comfyui._read_comfyui_config", lambda: {"model": "weird-model"}
        )
        result = provider.generate(prompt="x")
        assert result["success"] is False
        assert result["error_type"] == "bad_model"

    def test_generate_server_down_returns_unavailable(self, monkeypatch):
        provider = ComfyUIProvider()
        monkeypatch.setattr(provider, "is_available", lambda: False)
        result = provider.generate(prompt="x")
        assert result["success"] is False
        assert result["error_type"] == "unavailable"

    def test_registered_models_are_apache_licensed(self):
        # Fleet licensing rule: commercially usable (Apache-2.0) models only.
        assert set(_MODELS) == {"qwen-image", "flux2-klein-4b"}
