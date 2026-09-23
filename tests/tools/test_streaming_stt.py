"""Tests for tools.streaming_stt — all mocked, no live endpoint required.

One optional end-to-end test (live 9/23 endpoint) is skipped unless
STREAME2E=1 is set and the endpoint answers on 127.0.0.1:8123.
"""

import base64
import json
import struct
import wave
from unittest.mock import MagicMock, patch

import pytest


# ============================================================================
# Fixtures
# ============================================================================

def _make_wav(path, seconds, rate=16000, channels=1):
    n_frames = int(seconds * rate)
    # gentle ramp so it's not pure silence (content-agnostic, 16-bit PCM)
    frames = b"".join(
        struct.pack("<h", int(3000 * (i / n_frames))) for i in range(n_frames)
    )
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(frames)
    return str(path)


@pytest.fixture
def wav_3s(tmp_path):
    return _make_wav(tmp_path / "turn3s.wav", 3.0)


@pytest.fixture
def wav_05s(tmp_path):
    return _make_wav(tmp_path / "turn05s.wav", 0.5)


@pytest.fixture
def wav_48k(tmp_path):
    return _make_wav(tmp_path / "turn48k.wav", 2.0, rate=48000)


@pytest.fixture
def stream_cfg():
    return {
        "enabled": True,
        "endpoint": "ws://127.0.0.1:8123/v1/realtime",
        "health_url": "http://127.0.0.1:8123/health",
        "model": "voxtral-realtime",
        "chunk_ms": 160,
        "min_stream_seconds": 2.0,
        "connect_timeout_s": 5.0,
        "turn_timeout_s": 120.0,
    }


# ============================================================================
# Config
# ============================================================================

class TestConfig:
    def test_defaults_disabled(self):
        from tools import streaming_stt

        assert streaming_stt._STREAMING_DEFAULTS["enabled"] is False
        assert streaming_stt._STREAMING_DEFAULTS["endpoint"].startswith("ws://")

    def test_get_streaming_config_merges_user(self):
        from tools import streaming_stt

        with patch(
            "logos_cli.config.load_config",
            return_value={"stt": {"streaming": {"model": "custom"}}},
        ):
            cfg = streaming_stt.get_streaming_config()
        assert cfg["model"] == "custom"
        # untouched keys keep defaults
        assert cfg["chunk_ms"] == 160

    def test_get_streaming_config_survives_config_error(self):
        from tools import streaming_stt

        def boom():
            raise RuntimeError("no config")

        with patch("logos_cli.config.load_config", side_effect=boom):
            cfg = streaming_stt.get_streaming_config()
        assert cfg["enabled"] is False


# ============================================================================
# Helpers
# ============================================================================

class TestHelpers:
    def test_split_pcm_chunks_exact(self):
        from tools.streaming_stt import split_pcm_chunks

        pcm = b"\x00\x01" * 1600  # 0.1 s @16k
        chunks = split_pcm_chunks(pcm, chunk_ms=100)
        assert sum(len(c) for c in chunks) == len(pcm)
        assert all(len(c) <= 3200 for c in chunks)  # 100ms * 16k * 2 bytes

    def test_split_pcm_chunks_tail(self):
        from tools.streaming_stt import split_pcm_chunks

        pcm = b"\x00\x01" * 1601  # odd sample count
        chunks = split_pcm_chunks(pcm, chunk_ms=100)
        assert sum(len(c) for c in chunks) == len(pcm)

    def test_wav_duration(self, wav_3s):
        from tools.streaming_stt import wav_duration_seconds

        assert abs(wav_duration_seconds(wav_3s) - 3.0) < 0.05

    def test_wav_duration_unreadable(self):
        from tools.streaming_stt import wav_duration_seconds

        assert wav_duration_seconds("/nonexistent/nope.wav") is None


# ============================================================================
# should_stream gate
# ============================================================================

class TestShouldStream:
    def test_disabled(self, wav_3s, stream_cfg):
        from tools import streaming_stt

        cfg = dict(stream_cfg, enabled=False)
        with patch.object(streaming_stt, "endpoint_healthy", return_value=True):
            assert streaming_stt.should_stream(wav_3s, cfg) is False

    def test_endpoint_unhealthy(self, wav_3s, stream_cfg):
        from tools import streaming_stt

        with patch.object(streaming_stt, "endpoint_healthy", return_value=False):
            assert streaming_stt.should_stream(wav_3s, stream_cfg) is False

    def test_short_clip_stays_on_batch(self, wav_05s, stream_cfg):
        from tools import streaming_stt

        with patch.object(streaming_stt, "endpoint_healthy", return_value=True):
            assert streaming_stt.should_stream(wav_05s, stream_cfg) is False

    def test_long_healthy_clip_streams(self, wav_3s, stream_cfg):
        from tools import streaming_stt

        with patch.object(streaming_stt, "endpoint_healthy", return_value=True):
            assert streaming_stt.should_stream(wav_3s, stream_cfg) is True

    def test_wrong_rate_not_streamable(self, wav_48k, stream_cfg):
        # 48k clip: should_stream only gates on duration; the rate check
        # happens in transcribe_wav_streaming (returns success=False there,
        # caller falls back to batch).
        from tools import streaming_stt

        assert abs(streaming_stt.wav_duration_seconds(wav_48k) - 2.0) < 0.05


# ============================================================================
# transcribe_wav_streaming (mocked WS)
# ============================================================================

class FakeWS:
    """Minimal fake of websocket-client's WebSocket for protocol tests."""

    def __init__(self, script):
        self._script = list(script)  # list of server messages (dicts)
        self.sent = []
        self.timeout = None

    def send(self, data):
        self.sent.append(json.loads(data))

    def recv(self):
        return json.dumps(self._script.pop(0))

    def settimeout(self, t):
        self.timeout = t

    def close(self):
        pass


def _make_client_with_fake(fakews):
    from tools.streaming_stt import RealtimeSTTClient

    client = RealtimeSTTClient(
        endpoint="ws://fake", model="voxtral-realtime", chunk_ms=160
    )
    client._ws = fakews
    client._t_start = 1.0
    return client


class TestTranscribeWavStreaming:
    def test_wrong_sample_rate_falls_through(self, wav_48k, stream_cfg):
        from tools import streaming_stt

        res = streaming_stt.transcribe_wav_streaming(wav_48k, stream_cfg)
        assert res["success"] is False
        assert "16000" in res["error"]

    def test_unreadable_wav(self, tmp_path, stream_cfg):
        from tools import streaming_stt

        (tmp_path / "junk.wav").write_bytes(b"not a wav")
        res = streaming_stt.transcribe_wav_streaming(str(tmp_path / "junk.wav"), stream_cfg)
        assert res["success"] is False
        assert "read wav" in res["error"]

    def test_protocol_flow_and_result(self, wav_3s, stream_cfg):
        from tools import streaming_stt

        fake = FakeWS(
            [
                {"type": "session.created", "id": "sess-1"},
                {"type": "transcription.delta", "delta": "One ber"},
                {"type": "transcription.delta", "delta": "berine"},
                {"type": "transcription.done", "text": " One berberine", "usage": None},
            ]
        )
        with patch("websocket.create_connection", return_value=fake):
            res = streaming_stt.transcribe_wav_streaming(wav_3s, stream_cfg)
        assert res["success"] is True
        assert res["transcript"] == " One berberine"
        assert res["method"] == "streaming-vllm-realtime"
        assert res["first_token_ms"] is not None and res["first_token_ms"] >= 0
        assert res["finalize_ms"] is not None and res["finalize_ms"] >= 0
        assert res["deltas"] == 2
        # client -> server event order: session.update, commit(False) [start],
        # appends..., commit(True) [stop]
        types = [m["type"] for m in fake.sent]
        assert types[0] == "session.update"
        assert types[1] == "input_audio_buffer.commit"
        assert fake.sent[1]["final"] is False
        assert "input_audio_buffer.append" in types
        assert types[-1] == "input_audio_buffer.commit"
        assert fake.sent[-1]["final"] is True
        # audio payload is valid base64
        appends = [m for m in fake.sent if m["type"] == "input_audio_buffer.append"]
        assert len(appends) >= 1
        base64.b64decode(appends[0]["audio"], validate=True)

    def test_error_event(self, wav_3s, stream_cfg):
        from tools import streaming_stt

        fake = FakeWS(
            [
                {"type": "session.created", "id": "sess-1"},
                {"type": "error", "error": "boom", "code": "processing_error"},
            ]
        )
        with patch("websocket.create_connection", return_value=fake):
            res = streaming_stt.transcribe_wav_streaming(wav_3s, stream_cfg)
        assert res["success"] is False
        assert res["error"] == "boom"

    def test_connect_failure_returns_error_dict(self, wav_3s, stream_cfg):
        from tools import streaming_stt

        with patch(
            "websocket.create_connection",
            side_effect=ConnectionRefusedError("refused"),
        ):
            res = streaming_stt.transcribe_wav_streaming(wav_3s, stream_cfg)
        assert res["success"] is False
        assert "ConnectionRefusedError" in res["error"]

    def test_bad_first_event(self, wav_3s, stream_cfg):
        from tools import streaming_stt

        fake = FakeWS([{"type": "weird", "x": 1}])
        with patch("websocket.create_connection", return_value=fake):
            res = streaming_stt.transcribe_wav_streaming(wav_3s, stream_cfg)
        assert res["success"] is False


# ============================================================================
# endpoint_healthy
# ============================================================================

class TestEndpointHealthy:
    def test_ok(self, stream_cfg):
        from tools import streaming_stt

        streaming_stt._health_cache["ok"] = None
        resp = MagicMock()
        resp.status = 200
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=resp) as u:
            assert streaming_stt.endpoint_healthy(stream_cfg) is True
        u.assert_called_once()

    def test_error_is_false_and_cached(self, stream_cfg):
        from tools import streaming_stt

        streaming_stt._health_cache["ok"] = None
        with patch("urllib.request.urlopen", side_effect=OSError("down")):
            assert streaming_stt.endpoint_healthy(stream_cfg) is False
        # second call within fail TTL: no new probe
        with patch("urllib.request.urlopen", side_effect=AssertionError("re-probed")):
            assert streaming_stt.endpoint_healthy(stream_cfg) is False


# ============================================================================
# transcribe_recording routing (voice_mode)
# ============================================================================

class TestTranscribeRecordingRouting:
    def test_batch_when_streaming_disabled(self, wav_3s):
        from tools.voice_mode import transcribe_recording

        batch = {"success": True, "transcript": "batch text"}
        with patch(
            "tools.streaming_stt.should_stream", return_value=False
        ), patch(
            "tools.transcription_tools.transcribe_audio", return_value=batch
        ) as b:
            res = transcribe_recording(wav_3s)
        assert res["transcript"] == "batch text"
        b.assert_called_once()

    def test_streaming_preferred_when_healthy(self, wav_3s):
        from tools.voice_mode import transcribe_recording

        sres = {
            "success": True,
            "transcript": "streamed text",
            "method": "streaming-vllm-realtime",
            "first_token_ms": 512,
            "finalize_ms": 280,
            "deltas": 40,
            "duration_s": 3.0,
        }
        with patch(
            "tools.streaming_stt.should_stream", return_value=True
        ), patch(
            "tools.streaming_stt.transcribe_wav_streaming", return_value=sres
        ), patch(
            "tools.transcription_tools.transcribe_audio",
            side_effect=AssertionError("batch should not run"),
        ):
            res = transcribe_recording(wav_3s)
        assert res["transcript"] == "streamed text"
        assert res["method"] == "streaming-vllm-realtime"

    def test_fallback_on_streaming_error(self, wav_3s):
        from tools.voice_mode import transcribe_recording

        batch = {"success": True, "transcript": "batch rescue"}
        with patch(
            "tools.streaming_stt.should_stream", return_value=True
        ), patch(
            "tools.streaming_stt.transcribe_wav_streaming",
            return_value={"success": False, "error": "boom"},
        ), patch(
            "tools.transcription_tools.transcribe_audio", return_value=batch
        ):
            res = transcribe_recording(wav_3s)
        assert res["transcript"] == "batch rescue"

    def test_fallback_on_empty_transcript(self, wav_3s):
        from tools.voice_mode import transcribe_recording

        batch = {"success": True, "transcript": "batch rescue"}
        with patch(
            "tools.streaming_stt.should_stream", return_value=True
        ), patch(
            "tools.streaming_stt.transcribe_wav_streaming",
            return_value={"success": True, "transcript": "   "},
        ), patch(
            "tools.transcription_tools.transcribe_audio", return_value=batch
        ):
            res = transcribe_recording(wav_3s)
        assert res["transcript"] == "batch rescue"

    def test_fallback_when_streaming_module_raises(self, wav_3s):
        from tools.voice_mode import transcribe_recording

        batch = {"success": True, "transcript": "batch rescue"}
        with patch(
            "tools.streaming_stt.should_stream",
            side_effect=RuntimeError("config broken"),
        ), patch(
            "tools.transcription_tools.transcribe_audio", return_value=batch
        ):
            res = transcribe_recording(wav_3s)
        assert res["transcript"] == "batch rescue"

    def test_hallucination_filter_applies_to_streaming(self, wav_3s):
        from tools.voice_mode import transcribe_recording

        sres = {"success": True, "transcript": "Thank you for watching.",
                "method": "streaming-vllm-realtime"}
        with patch(
            "tools.streaming_stt.should_stream", return_value=True
        ), patch(
            "tools.streaming_stt.transcribe_wav_streaming", return_value=sres
        ), patch(
            "tools.transcription_tools.transcribe_audio",
            side_effect=AssertionError("batch should not run"),
        ):
            res = transcribe_recording(wav_3s)
        assert res["success"] is True
        assert res["transcript"] == ""
        assert res.get("filtered") is True


# ============================================================================
# Optional live end-to-end (STREAME2E=1 + endpoint up on 8123)
# ============================================================================

@pytest.mark.skipif(
    __import__("os").environ.get("STREAME2E") != "1",
    reason="live endpoint e2e (set STREAME2E=1)",
)
class TestLiveE2E:
    def test_live_transcribe(self, wav_3s):
        from tools import streaming_stt

        cfg = streaming_stt.get_streaming_config()
        cfg["enabled"] = True
        res = streaming_stt.transcribe_wav_streaming(wav_3s, cfg)
        assert res["success"] is True, res
        assert res["transcript"].strip()
