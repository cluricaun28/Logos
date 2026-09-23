"""Streaming STT client for a vLLM Voxtral-Realtime endpoint.

Live-turn transcription over the vLLM Realtime WebSocket protocol
(``ws://host/v1/realtime``) — sub-second first token, incremental deltas,
multi-turn. Used by the live voice loop as an OPT-IN fast path; the
batch faster-whisper path (``tools.transcription_tools.transcribe_audio``)
remains the default and the non-fatal fallback.

Design rules (match the voice stack's non-fatal philosophy):
  * Every public function RETURNS a dict — never raises.
  * ``success: False`` + ``error`` means "caller should fall back".
  * No global state except a short-TTL health cache.

Protocol (vLLM 0.26.x, measured 9/23):
  server: session.created
  client: session.update {model}            -> validates model
  client: input_audio_buffer.commit final=False   -> STARTS generation
  client: input_audio_buffer.append {audio}  -> PCM16 16k mono base64
  client: input_audio_buffer.commit final=True    -> stop sentinel
  server: transcription.delta {delta}*      -> incremental text
  server: transcription.done {text, usage}
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
import urllib.request
import wave
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# 16 kHz mono PCM16 is what the vLLM Voxtral Realtime endpoint expects, and
# what voice_mode.SAMPLE_RATE captures at. Anything else -> batch fallback.
REQUIRED_SAMPLE_RATE = 16000
REQUIRED_CHANNELS = 1

_STREAMING_DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    "endpoint": "ws://127.0.0.1:8123/v1/realtime",
    "health_url": "http://127.0.0.1:8123/health",
    "model": "voxtral-realtime",
    "chunk_ms": 160,
    "min_stream_seconds": 2.0,
    "connect_timeout_s": 5.0,
    "turn_timeout_s": 120.0,
}

# Health result cache (avoid a per-turn HTTP probe when the endpoint is up).
_health_cache: Dict[str, Any] = {"ok": None, "checked_at": 0.0}
_HEALTH_OK_TTL_S = 10.0
_HEALTH_FAIL_TTL_S = 15.0


def get_streaming_config() -> Dict[str, Any]:
    """Return the effective streaming-STT config (defaults + user config)."""
    cfg = dict(_STREAMING_DEFAULTS)
    try:
        from logos_cli.config import load_config
        stt = load_config().get("stt") or {}
        user = stt.get("streaming") or {}
        if isinstance(user, dict):
            for key in _STREAMING_DEFAULTS:
                if key in user:
                    cfg[key] = user[key]
    except Exception:
        pass  # config unavailable -> pure defaults (streaming stays off)
    return cfg


def is_enabled(cfg: Optional[Dict[str, Any]] = None) -> bool:
    cfg = cfg or get_streaming_config()
    return bool(cfg.get("enabled"))


def endpoint_healthy(cfg: Optional[Dict[str, Any]] = None, timeout: float = 3.0) -> bool:
    """Cheap health probe with a short-TTL cache. Non-fatal (False on error)."""
    cfg = cfg or get_streaming_config()
    now = time.monotonic()
    ttl = _HEALTH_OK_TTL_S if _health_cache["ok"] else _HEALTH_FAIL_TTL_S
    if _health_cache["ok"] is not None and now - _health_cache["checked_at"] < ttl:
        return bool(_health_cache["ok"])
    url = cfg.get("health_url") or ""
    ok = False
    if url:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                ok = 200 <= resp.status < 300
        except Exception:
            ok = False
    _health_cache["ok"] = ok
    _health_cache["checked_at"] = now
    return ok


def wav_duration_seconds(wav_path: str) -> Optional[float]:
    """Duration in seconds, or None if unreadable."""
    try:
        with wave.open(wav_path, "rb") as w:
            frames = w.getnframes()
            rate = w.getframerate() or 1
            return frames / float(rate)
    except Exception:
        return None


def should_stream(wav_path: str, cfg: Optional[Dict[str, Any]] = None) -> bool:
    """Gate for the live-turn streaming path.

    True only when: enabled, endpoint healthy, file is 16k mono, and at least
    ``min_stream_seconds`` long (short barge-in clips are faster on batch).
    """
    cfg = cfg or get_streaming_config()
    if not cfg.get("enabled"):
        return False
    if not endpoint_healthy(cfg):
        return False
    dur = wav_duration_seconds(wav_path)
    if dur is None:
        return False
    return dur >= float(cfg.get("min_stream_seconds", 2.0))


def _read_wav_pcm16(wav_path: str):
    """Return (pcm16_bytes, sample_rate, channels) or raise."""
    with wave.open(wav_path, "rb") as w:
        rate = w.getframerate()
        ch = w.getnchannels()
        sampwidth = w.getsampwidth()
        if sampwidth != 2:
            raise ValueError(f"unsupported sample width {sampwidth} (need 16-bit PCM)")
        return w.readframes(w.getnframes()), rate, ch


def split_pcm_chunks(pcm: bytes, chunk_ms: int, sample_rate: int = REQUIRED_SAMPLE_RATE):
    """Split 16-bit mono PCM into byte chunks of ``chunk_ms`` milliseconds."""
    step = max(1, int(sample_rate * chunk_ms / 1000) * 2)  # 2 bytes per sample
    return [pcm[i : i + step] for i in range(0, len(pcm), step)]


class RealtimeSTTClient:
    """One WebSocket turn: start -> feed* -> commit. Synchronous, non-fatal."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        chunk_ms: int = 160,
        connect_timeout_s: float = 5.0,
        turn_timeout_s: float = 120.0,
    ):
        self.endpoint = endpoint
        self.model = model
        self.chunk_ms = chunk_ms
        self.connect_timeout_s = connect_timeout_s
        self.turn_timeout_s = turn_timeout_s
        self._ws = None
        self._t_start = 0.0
        self._t_first_delta: Optional[float] = None
        self._delta_count = 0

    # -- lifecycle ---------------------------------------------------------
    def start_turn(self) -> None:
        """Connect, wait for session.created, validate model, start generation."""
        import websocket  # websocket-client (sync)

        t0 = time.monotonic()
        self._ws = websocket.create_connection(
            self.endpoint, timeout=self.connect_timeout_s
        )
        # server: session.created
        first = self._recv()
        if first.get("type") != "session.created":
            self.close()
            raise RuntimeError(f"expected session.created, got {first}")
        self._ws.send(json.dumps({"type": "session.update", "model": self.model}))
        # Generation STARTS on commit(final=False) (measured 9/23 — sending
        # only final=True yields zero deltas); appends are consumed as they arrive.
        self._ws.send(
            json.dumps({"type": "input_audio_buffer.commit", "final": False})
        )
        self._t_start = time.monotonic()

    def feed(self, pcm16_chunk: bytes) -> None:
        if self._ws is None:
            raise RuntimeError("start_turn() not called")
        self._ws.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm16_chunk).decode(),
                }
            )
        )

    def commit(self) -> Dict[str, Any]:
        """End the turn; read until transcription.done; return result dict."""
        if self._ws is None:
            return {"success": False, "error": "start_turn() not called"}
        self._ws.settimeout(self.turn_timeout_s)
        self._ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))
        t_commit = time.monotonic()
        text = ""
        try:
            while True:
                msg = self._recv()
                mtype = msg.get("type")
                if mtype == "transcription.delta":
                    if self._t_first_delta is None:
                        self._t_first_delta = time.monotonic()
                    self._delta_count += 1
                    text += msg.get("delta", "")
                elif mtype == "transcription.done":
                    text = msg.get("text", text)
                    return {
                        "success": True,
                        "transcript": text,
                        "first_token_ms": (
                            int((self._t_first_delta - self._t_start) * 1000)
                            if self._t_first_delta
                            else None
                        ),
                        "finalize_ms": int((time.monotonic() - t_commit) * 1000),
                        "deltas": self._delta_count,
                    }
                elif mtype == "error":
                    return {"success": False, "error": str(msg.get("error"))}
                # ignore session.update echoes etc.
        finally:
            self.close()

    # -- internals ---------------------------------------------------------
    def _recv(self) -> Dict[str, Any]:
        assert self._ws is not None
        return json.loads(self._ws.recv())

    def close(self) -> None:
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass


def transcribe_wav_streaming(
    wav_path: str, cfg: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Stream a 16k mono WAV through the realtime endpoint. Never raises.

    Returns:
      {success, transcript, method, first_token_ms, finalize_ms, duration_s}
      or {success: False, error: str}
    """
    cfg = cfg or get_streaming_config()
    try:
        pcm, rate, ch = _read_wav_pcm16(wav_path)
        if rate != REQUIRED_SAMPLE_RATE or ch != REQUIRED_CHANNELS:
            return {
                "success": False,
                "error": f"need {REQUIRED_SAMPLE_RATE} Hz mono, got {rate} Hz {ch} ch",
            }
    except Exception as e:
        return {"success": False, "error": f"read wav: {e}"}

    duration = len(pcm) / 2 / float(REQUIRED_SAMPLE_RATE)
    client = RealtimeSTTClient(
        endpoint=cfg.get("endpoint", _STREAMING_DEFAULTS["endpoint"]),
        model=cfg.get("model", _STREAMING_DEFAULTS["model"]),
        chunk_ms=int(cfg.get("chunk_ms", 160)),
        connect_timeout_s=float(cfg.get("connect_timeout_s", 5.0)),
        turn_timeout_s=float(cfg.get("turn_timeout_s", 120.0)),
    )
    try:
        client.start_turn()
        for chunk in split_pcm_chunks(pcm, client.chunk_ms):
            client.feed(chunk)
        result = client.commit()
    except Exception as e:
        client.close()
        return {"success": False, "error": f"{type(e).__name__}: {e}"}

    if result.get("success"):
        result["method"] = "streaming-vllm-realtime"
        result["duration_s"] = round(duration, 2)
    return result
