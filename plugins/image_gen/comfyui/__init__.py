"""Local ComfyUI image generation backend.

Exposes the on-box ComfyUI server (127.0.0.1:8188, card 5) as an
:class:`ImageGenProvider`. Fully local — no cloud, no paid services.

Models (open-licensed, commercial use):
- ``qwen-image`` — Qwen/Qwen-Image, Apache 2.0. Concept work, dense prompts,
  in-image text. Default.
- ``flux2-klein-4b`` — black-forest-labs/FLUX.2-klein-4B, Apache 2.0.
  Photoreal product renders.

Model selection:
1. ``image_gen.comfyui.model`` in config.yaml
2. default: ``qwen-image``

Endpoints used:
- ``POST /prompt``      — submit workflow, returns prompt_id
- ``GET  /history/{id}`` — poll for completion + output filenames
- ``GET  /view``        — fetch rendered image bytes
"""

from __future__ import annotations

import base64
import json
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional

import requests

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    save_b64_image,
    success_response,
)

logger = logging.getLogger(__name__)

DEFAULT_URL = "http://127.0.0.1:8188"
POLL_INTERVAL_S = 2.0
POLL_TIMEOUT_S = 600.0

# ComfyUI wants dimensions on a 16-grid.
# Core tool surface uses the vocabulary landscape/square/portrait
# (see VALID_ASPECT_RATIOs in agent/image_gen_provider.py); ratio
# strings are kept as forward-compat aliases.
_ASPECT_DIMS: Dict[str, tuple] = {
    "landscape": (1536, 864),
    "square": (1344, 1344),
    "portrait": (864, 1536),
    "16:9": (1536, 864),
    "9:16": (864, 1536),
    "1:1": (1344, 1344),
    "4:3": (1472, 1104),
    "3:4": (1104, 1472),
    "2:3": (1024, 1536),
    "3:2": (1536, 1024),
}

# ComfyUI split-file layout (Comfy-Org/Qwen-Image_ComfyUI + BFL FLUX.2 klein 4B),
# files symlinked into ComfyUI's models/ folders.
_MODELS: Dict[str, Dict[str, Any]] = {
    "qwen-image": {
        "display": "Qwen-Image (Apache 2.0)",
        "strengths": "Concept work, dense prompts, in-image text/labels",
        "unet": "qwen_image_bf16.safetensors",
        "clip": "qwen_2.5_vl_7b.safetensors",
        "clip_type": "qwen_image",
        "vae": "qwen_image_vae.safetensors",
        "cfg": 1.0,
        "steps": 20,
        "negative": "blurry, low quality, watermark, jpeg artifacts, deformed, extra limbs",
    },
    "flux2-klein-4b": {
        "display": "FLUX.2 [klein] 4B (Apache 2.0)",
        "strengths": "Photoreal product renders, materials, lighting",
        "unet": "flux-2-klein-4b.safetensors",
        "clip": "flux2-klein-4b-text_encoder.safetensors",
        "clip_type": "flux2",
        "vae": "flux2-klein-4b-vae.safetensors",
        "cfg": 1.0,
        "steps": 20,
        "negative": "blurry, low quality, watermark, jpeg artifacts, deformed",
    },
}

DEFAULT_MODEL = "qwen-image"


def _read_comfyui_config() -> Dict[str, Any]:
    """Read ``image_gen.comfyui`` section from config.yaml (tolerant)."""
    try:
        import yaml
        from logos_constants import get_hermes_home

        cfg_file = get_hermes_home() / "config.yaml"
        if not cfg_file.exists():
            return {}
        with open(cfg_file) as f:
            data = yaml.safe_load(f) or {}
        return (data.get("image_gen") or {}).get("comfyui") or {}
    except Exception:  # noqa: BLE001 — config is optional sugar
        return {}


class ComfyUIProvider(ImageGenProvider):
    @property
    def name(self) -> str:
        return "comfyui"

    @property
    def display_name(self) -> str:
        return "Local ComfyUI (on-box)"

    def is_available(self) -> bool:
        url = _read_comfyui_config().get("url") or os.environ.get("COMFYUI_URL") or DEFAULT_URL
        try:
            r = requests.get(f"{url}/system_stats", timeout=3)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> List[Dict[str, Any]]:
        out = []
        for mid, meta in _MODELS.items():
            out.append({
                "id": mid,
                "display": meta["display"],
                "strengths": meta["strengths"],
            })
        return out

    def default_model(self) -> Optional[str]:
        return _read_comfyui_config().get("model") or DEFAULT_MODEL

    # ------------------------------------------------------------------
    # Workflow
    # ------------------------------------------------------------------

    def _workflow(self, prompt: str, model: str, width: int, height: int) -> Dict[str, Any]:
        meta = _MODELS[model]
        seed = random.randint(1, 2**31 - 1)
        return {
            # Loaders
            "1": {
                "class_type": "UNETLoader",
                "inputs": {"unet_name": meta["unet"], "weight_dtype": "default"},
            },
            "2": {
                "class_type": "CLIPLoader",
                "inputs": {"clip_name": meta["clip"], "type": meta["clip_type"]},
            },
            "3": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": meta["vae"]},
            },
            # Conditioning
            "4": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": meta["negative"], "clip": ["2", 0]},
            },
            "5": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": prompt, "clip": ["2", 0]},
            },
            # Sampling
            "6": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": width, "height": height, "batch_size": 1},
            },
            "7": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed,
                    "steps": meta["steps"],
                    "cfg": meta["cfg"],
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "denoise": 1.0,
                    "model": ["1", 0],
                    "positive": ["5", 0],
                    "negative": ["4", 0],
                    "latent_image": ["6", 0],
                },
            },
            # Decode + save
            "8": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["7", 0], "vae": ["3", 0]},
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {"filename_prefix": "hermes", "images": ["8", 0]},
            },
        }

    def _submit(self, url: str, workflow: Dict[str, Any]) -> str:
        r = requests.post(f"{url}/prompt", json={"prompt": workflow}, timeout=30)
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise RuntimeError(f"ComfyUI rejected workflow: {data.get('error_message') or data['error']}")
        return data["prompt_id"]

    def _wait(self, url: str, prompt_id: str) -> Dict[str, Any]:
        deadline = time.monotonic() + POLL_TIMEOUT_S
        while time.monotonic() < deadline:
            r = requests.get(f"{url}/history/{prompt_id}", timeout=15)
            r.raise_for_status()
            entry = r.json().get(prompt_id)
            if entry is not None:
                status = (entry.get("status") or {}).get("status_str", "")
                if status in ("in_queue", "queued", "pending", ""):
                    time.sleep(POLL_INTERVAL_S)
                    continue
                if status == "error":
                    for msg in (entry.get("status") or {}).get("messages", []):
                        if msg[0] == "execution_error":
                            raise RuntimeError(f"ComfyUI node error: {msg[1].get('exception_message', 'unknown')}")
                    raise RuntimeError(f"ComfyUI prompt {prompt_id} errored (no details)")
                out = entry.get("outputs", {})
                for node_out in out.values():
                    for img in node_out.get("images", []):
                        return img
                raise RuntimeError(f"ComfyUI finished prompt {prompt_id} with no image output")
            time.sleep(POLL_INTERVAL_S)
        raise TimeoutError(f"ComfyUI prompt {prompt_id} timed out after {POLL_TIMEOUT_S:.0f}s")

    def _fetch_image(self, url: str, img: Dict[str, Any]) -> bytes:
        params = {
            "filename": img["filename"],
            "subfolder": img.get("subfolder", ""),
            "type": img.get("type", "output"),
        }
        r = requests.get(f"{url}/view", params=params, timeout=60)
        r.raise_for_status()
        return r.content

    # ------------------------------------------------------------------
    # Provider API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        cfg = _read_comfyui_config()
        url = cfg.get("url") or os.environ.get("COMFYUI_URL") or DEFAULT_URL
        # Model precedence: explicit request > config default > qwen-image.
        requested = kwargs.get("model")
        if requested and str(requested).strip().lower() in _MODELS:
            model = str(requested).strip().lower()
        else:
            model = cfg.get("model") or DEFAULT_MODEL
        aspect = resolve_aspect_ratio(aspect_ratio)

        if model not in _MODELS:
            return error_response(
                error=f"Unknown ComfyUI model '{model}'. Valid: {', '.join(_MODELS)}",
                error_type="bad_model",
                provider="comfyui",
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        if not self.is_available():
            return error_response(
                error=f"ComfyUI server not reachable at {url}. Is the comfyui service running?",
                error_type="unavailable",
                provider="comfyui",
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        width, height = _ASPECT_DIMS.get(aspect, _ASPECT_DIMS["1:1"])
        workflow = self._workflow(prompt, model, width, height)

        try:
            prompt_id = self._submit(url, workflow)
            img_meta = self._wait(url, prompt_id)
            raw = self._fetch_image(url, img_meta)
        except requests.RequestException as exc:
            return error_response(
                error=f"ComfyUI request failed: {exc}",
                error_type="http_error",
                provider="comfyui",
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except (RuntimeError, TimeoutError) as exc:
            return error_response(
                error=str(exc),
                error_type="generation_failed",
                provider="comfyui",
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        path = save_b64_image(base64.b64encode(raw).decode(), prefix=f"comfyui_{model}")
        return success_response(
            image=str(path),
            model=model,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="comfyui",
            extra={"resolution": f"{width}x{height}"},
        )


def register(ctx: Any) -> None:
    """Register this provider with the image gen registry."""
    ctx.register_image_gen_provider(ComfyUIProvider())
