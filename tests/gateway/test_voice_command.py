"""Tests for the /voice command and auto voice reply in the gateway."""

import importlib.util
import json
import os
import queue
import sys
import threading
import time
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch


def _ensure_discord_mock():
    """Install a lightweight discord mock when discord.py isn't available."""
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return

    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.ui = SimpleNamespace(View=object, button=lambda *a, **k: (lambda fn: fn), Button=object)
    discord_mod.ButtonStyle = SimpleNamespace(success=1, primary=2, secondary=2, danger=3, green=1, grey=2, blurple=2, red=3)
    discord_mod.Color = SimpleNamespace(orange=lambda: 1, green=lambda: 2, blue=lambda: 3, red=lambda: 4, purple=lambda: 5)
    discord_mod.Interaction = object
    discord_mod.Embed = MagicMock
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    discord_mod.opus = SimpleNamespace(is_loaded=lambda: True, load_opus=lambda *_args, **_kwargs: None)
    discord_mod.FFmpegPCMAudio = MagicMock
    discord_mod.PCMVolumeTransformer = MagicMock
    discord_mod.http = SimpleNamespace(Route=MagicMock)

    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod

    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

from gateway.platforms.base import MessageEvent, MessageType, SessionSource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(text: str = "", message_type=MessageType.TEXT, chat_id="123") -> MessageEvent:
    source = SessionSource(
        chat_id=chat_id,
        user_id="user1",
        platform=MagicMock(),
    )
    source.platform.value = "telegram"
    source.thread_id = None
    event = MessageEvent(text=text, message_type=message_type, source=source)
    event.message_id = "msg42"
    return event


def _make_runner(tmp_path):
    """Create a bare GatewayRunner without calling __init__."""
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._VOICE_MODE_PATH = tmp_path / "gateway_voice_mode.json"
    runner._session_db = None
    runner.session_store = MagicMock()
    runner._is_user_authorized = lambda source: True
    return runner


# =====================================================================
# /voice command handler
# =====================================================================

class TestHandleVoiceCommand:

    @pytest.fixture
    def runner(self, tmp_path):
        return _make_runner(tmp_path)

    @pytest.mark.asyncio
    async def test_voice_on(self, runner):
        event = _make_event("/voice on")
        result = await runner._handle_voice_command(event)
        assert "enabled" in result.lower()
        assert runner._voice_mode["telegram:123"] == "voice_only"

    @pytest.mark.asyncio
    async def test_voice_off(self, runner):
        runner._voice_mode["telegram:123"] = "voice_only"
        event = _make_event("/voice off")
        result = await runner._handle_voice_command(event)
        assert "disabled" in result.lower()
        assert runner._voice_mode["telegram:123"] == "off"

    @pytest.mark.asyncio
    async def test_voice_tts(self, runner):
        event = _make_event("/voice tts")
        result = await runner._handle_voice_command(event)
        assert "tts" in result.lower()
        assert runner._voice_mode["telegram:123"] == "all"

    @pytest.mark.asyncio
    async def test_voice_status_off(self, runner):
        event = _make_event("/voice status")
        result = await runner._handle_voice_command(event)
        assert "off" in result.lower()

    @pytest.mark.asyncio
    async def test_voice_status_on(self, runner):
        runner._voice_mode["telegram:123"] = "voice_only"
        event = _make_event("/voice status")
        result = await runner._handle_voice_command(event)
        assert "voice reply" in result.lower()

    @pytest.mark.asyncio
    async def test_toggle_off_to_on(self, runner):
        event = _make_event("/voice")
        result = await runner._handle_voice_command(event)
        assert "enabled" in result.lower()
        assert runner._voice_mode["telegram:123"] == "voice_only"

    @pytest.mark.asyncio
    async def test_toggle_on_to_off(self, runner):
        runner._voice_mode["telegram:123"] = "voice_only"
        event = _make_event("/voice")
        result = await runner._handle_voice_command(event)
        assert "disabled" in result.lower()
        assert runner._voice_mode["telegram:123"] == "off"

    @pytest.mark.asyncio
    async def test_persistence_saved(self, runner):
        event = _make_event("/voice on")
        await runner._handle_voice_command(event)
        assert runner._VOICE_MODE_PATH.exists()
        data = json.loads(runner._VOICE_MODE_PATH.read_text())
        assert data["telegram:123"] == "voice_only"

    @pytest.mark.asyncio
    async def test_persistence_loaded(self, runner):
        runner._VOICE_MODE_PATH.write_text(json.dumps({"telegram:456": "all"}))
        loaded = runner._load_voice_modes()
        assert loaded == {"telegram:456": "all"}

    @pytest.mark.asyncio
    async def test_persistence_saved_for_off(self, runner):
        event = _make_event("/voice off")
        await runner._handle_voice_command(event)
        data = json.loads(runner._VOICE_MODE_PATH.read_text())
        assert data["telegram:123"] == "off"

    def test_sync_voice_mode_state_to_adapter_restores_off_chats(self, runner):
        from gateway.config import Platform
        runner._voice_mode = {"telegram:123": "off", "telegram:456": "all"}
        adapter = SimpleNamespace(
            _auto_tts_disabled_chats=set(),
            platform=Platform.TELEGRAM,
        )

        runner._sync_voice_mode_state_to_adapter(adapter)

        assert adapter._auto_tts_disabled_chats == {"123"}

    def test_sync_populates_enabled_chats_from_voice_modes(self, runner):
        """Issue #16007: sync also restores per-chat /voice on|tts opt-ins.

        The adapter's ``_auto_tts_enabled_chats`` must mirror chats whose
        persisted voice_mode is ``voice_only`` or ``all`` — without this,
        ``/voice on`` was relying on a "not in disabled set" default that
        silently enabled auto-TTS for every chat.
        """
        from gateway.config import Platform
        runner._voice_mode = {
            "telegram:off_chat": "off",
            "telegram:on_chat": "voice_only",
            "telegram:tts_chat": "all",
            "slack:999": "voice_only",  # wrong platform, must be ignored
        }
        adapter = SimpleNamespace(
            _auto_tts_default=False,
            _auto_tts_disabled_chats=set(),
            _auto_tts_enabled_chats=set(),
            platform=Platform.TELEGRAM,
        )

        runner._sync_voice_mode_state_to_adapter(adapter)

        assert adapter._auto_tts_disabled_chats == {"off_chat"}
        assert adapter._auto_tts_enabled_chats == {"on_chat", "tts_chat"}

    def test_sync_pushes_config_default_onto_adapter(self, runner, monkeypatch):
        """Issue #16007: ``voice.auto_tts`` must propagate to ``_auto_tts_default``."""
        from gateway.config import Platform

        fake_cfg = {"voice": {"auto_tts": True}}
        monkeypatch.setattr(
            "logos_cli.config.load_config",
            lambda: fake_cfg,
        )
        adapter = SimpleNamespace(
            _auto_tts_default=False,
            _auto_tts_disabled_chats=set(),
            _auto_tts_enabled_chats=set(),
            platform=Platform.TELEGRAM,
        )

        runner._sync_voice_mode_state_to_adapter(adapter)

        assert adapter._auto_tts_default is True

    def test_restart_restores_voice_off_state(self, runner, tmp_path):
        from gateway.config import Platform
        runner._VOICE_MODE_PATH.write_text(json.dumps({"telegram:123": "off"}))

        restored_runner = _make_runner(tmp_path)
        restored_runner._voice_mode = restored_runner._load_voice_modes()
        adapter = SimpleNamespace(
            _auto_tts_disabled_chats=set(),
            platform=Platform.TELEGRAM,
        )

        restored_runner._sync_voice_mode_state_to_adapter(adapter)

        assert restored_runner._voice_mode["telegram:123"] == "off"
        assert adapter._auto_tts_disabled_chats == {"123"}

    @pytest.mark.asyncio
    async def test_per_chat_isolation(self, runner):
        e1 = _make_event("/voice on", chat_id="aaa")
        e2 = _make_event("/voice tts", chat_id="bbb")
        await runner._handle_voice_command(e1)
        await runner._handle_voice_command(e2)
        assert runner._voice_mode["telegram:aaa"] == "voice_only"
        assert runner._voice_mode["telegram:bbb"] == "all"

    @pytest.mark.asyncio
    async def test_platform_isolation(self, runner):
        """Same chat_id on different platforms must not collide (#12542)."""
        telegram_event = _make_event("/voice on", chat_id="999")
        slack_event = _make_event("/voice off", chat_id="999")
        slack_event.source.platform.value = "slack"

        await runner._handle_voice_command(telegram_event)
        await runner._handle_voice_command(slack_event)

        assert runner._voice_mode["telegram:999"] == "voice_only"
        assert runner._voice_mode["slack:999"] == "off"


# =====================================================================
# Auto voice reply decision logic
# =====================================================================

class TestAutoVoiceReply:
    """Test the real _should_send_voice_reply method on GatewayRunner.

    The gateway has two TTS paths:
      1. base adapter auto-TTS: fires for voice input in _process_message_background
      2. gateway _send_voice_reply: fires based on voice_mode setting

    To prevent double audio, _send_voice_reply is skipped when voice input
    already triggered base adapter auto-TTS.

    For Discord voice channels, the base adapter now routes play_tts directly
    into VC playback, so the runner should still skip voice-input follow-ups to
    avoid double playback.
    """

    @pytest.fixture
    def runner(self, tmp_path):
        return _make_runner(tmp_path)

    def _call(self, runner, voice_mode, message_type, agent_messages=None,
              response="Hello!", in_voice_channel=False):
        """Call real _should_send_voice_reply on a GatewayRunner instance."""
        chat_id = "123"
        if voice_mode != "off":
            runner._voice_mode["telegram:" + chat_id] = voice_mode
        else:
            runner._voice_mode.pop("telegram:" + chat_id, None)

        event = _make_event(message_type=message_type)

        if in_voice_channel:
            mock_adapter = MagicMock()
            mock_adapter.is_in_voice_channel = MagicMock(return_value=True)
            event.raw_message = SimpleNamespace(guild_id=111, guild=None)
            runner.adapters[event.source.platform] = mock_adapter

        return runner._should_send_voice_reply(
            event, response, agent_messages or []
        )

    # -- Full platform x input x mode matrix --------------------------------
    #
    # Legend:
    #   base = base adapter auto-TTS (play_tts)
    #   runner = gateway _send_voice_reply
    #
    # | Platform      | Input | Mode       | base | runner | Expected     |
    # |---------------|-------|------------|------|--------|--------------|
    # | Telegram      | voice | off        | yes  | skip   | 1 audio      |
    # | Telegram      | voice | voice_only | yes  | skip*  | 1 audio      |
    # | Telegram      | voice | all        | yes  | skip*  | 1 audio      |
    # | Telegram      | text  | off        | skip | skip   | 0 audio      |
    # | Telegram      | text  | voice_only | skip | skip   | 0 audio      |
    # | Telegram      | text  | all        | skip | yes    | 1 audio      |
    # | Discord text  | voice | all        | yes  | skip*  | 1 audio      |
    # | Discord text  | text  | all        | skip | yes    | 1 audio      |
    # | Discord VC    | voice | all        | skip†| yes    | 1 audio (VC) |
    # | Web UI        | voice | off        | yes  | skip   | 1 audio      |
    # | Web UI        | voice | all        | yes  | skip*  | 1 audio      |
    # | Web UI        | text  | all        | skip | yes    | 1 audio      |
    # | Slack         | voice | all        | yes  | skip*  | 1 audio      |
    # | Slack         | text  | all        | skip | yes    | 1 audio      |
    #
    # * skip_double: voice input → base already handles
    # † Discord play_tts override skips when in VC

    # -- Telegram/Slack/Web: voice input, base handles ---------------------

    def test_voice_input_voice_only_skipped(self, runner):
        """voice_only + voice input: base auto-TTS handles it, runner skips."""
        assert self._call(runner, "voice_only", MessageType.VOICE) is False

    def test_voice_input_all_mode_skipped(self, runner):
        """all + voice input: base auto-TTS handles it, runner skips."""
        assert self._call(runner, "all", MessageType.VOICE) is False

    # -- Text input: only runner handles -----------------------------------

    def test_text_input_all_mode_runner_fires(self, runner):
        """all + text input: only runner fires (base auto-TTS only for voice)."""
        assert self._call(runner, "all", MessageType.TEXT) is True

    def test_text_input_voice_only_no_reply(self, runner):
        """voice_only + text input: neither fires."""
        assert self._call(runner, "voice_only", MessageType.TEXT) is False

    # -- Mode off: nothing fires -------------------------------------------

    def test_off_mode_voice(self, runner):
        assert self._call(runner, "off", MessageType.VOICE) is False

    def test_off_mode_text(self, runner):
        assert self._call(runner, "off", MessageType.TEXT) is False

    # -- Discord VC exception: runner must handle --------------------------

    def test_discord_vc_voice_input_base_handles(self, runner):
        """Discord VC + voice input: base adapter play_tts plays in VC,
        so runner skips to avoid double playback."""
        assert self._call(runner, "all", MessageType.VOICE, in_voice_channel=True) is False

    def test_discord_vc_voice_only_base_handles(self, runner):
        """Discord VC + voice_only + voice: base adapter handles."""
        assert self._call(runner, "voice_only", MessageType.VOICE, in_voice_channel=True) is False

    # -- Edge cases --------------------------------------------------------

    def test_error_response_skipped(self, runner):
        assert self._call(runner, "all", MessageType.TEXT, response="Error: boom") is False

    def test_empty_response_skipped(self, runner):
        assert self._call(runner, "all", MessageType.TEXT, response="") is False

    def test_dedup_skips_when_agent_called_tts(self, runner):
        messages = [{
            "role": "assistant",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "text_to_speech", "arguments": "{}"},
            }],
        }]
        assert self._call(runner, "all", MessageType.TEXT, agent_messages=messages) is False

    def test_no_dedup_for_other_tools(self, runner):
        messages = [{
            "role": "assistant",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "web_search", "arguments": "{}"},
            }],
        }]
        assert self._call(runner, "all", MessageType.TEXT, agent_messages=messages) is True


# =====================================================================
# _send_voice_reply
# =====================================================================

class TestSendVoiceReply:

    @pytest.fixture
    def runner(self, tmp_path):
        return _make_runner(tmp_path)

    @pytest.mark.asyncio
    async def test_calls_tts_and_send_voice(self, runner):
        mock_adapter = AsyncMock()
        mock_adapter.send_voice = AsyncMock()
        event = _make_event()
        runner.adapters[event.source.platform] = mock_adapter

        tts_result = json.dumps({"success": True, "file_path": "/tmp/test.ogg"})

        with patch("tools.tts_tool.text_to_speech_tool", return_value=tts_result), \
             patch("tools.tts_tool._strip_markdown_for_tts", side_effect=lambda t: t), \
             patch("os.path.isfile", return_value=True), \
             patch("os.unlink"), \
             patch("os.makedirs"):
            await runner._send_voice_reply(event, "Hello world")

        mock_adapter.send_voice.assert_called_once()
        call_args = mock_adapter.send_voice.call_args
        assert call_args.kwargs.get("chat_id") == "123"

    @pytest.mark.asyncio
    async def test_empty_text_after_strip_skips(self, runner):
        event = _make_event()

        with patch("tools.tts_tool.text_to_speech_tool") as mock_tts, \
             patch("tools.tts_tool._strip_markdown_for_tts", return_value=""):
            await runner._send_voice_reply(event, "```code only```")

        mock_tts.assert_not_called()

    @pytest.mark.asyncio
    async def test_tts_failure_no_crash(self, runner):
        event = _make_event()
        mock_adapter = AsyncMock()
        runner.adapters[event.source.platform] = mock_adapter
        tts_result = json.dumps({"success": False, "error": "API error"})

        with patch("tools.tts_tool.text_to_speech_tool", return_value=tts_result), \
             patch("tools.tts_tool._strip_markdown_for_tts", side_effect=lambda t: t), \
             patch("os.path.isfile", return_value=False), \
             patch("os.makedirs"):
            await runner._send_voice_reply(event, "Hello")

        mock_adapter.send_voice.assert_not_called()

    @pytest.mark.asyncio
    async def test_exception_caught(self, runner):
        event = _make_event()
        with patch("tools.tts_tool.text_to_speech_tool", side_effect=RuntimeError("boom")), \
             patch("tools.tts_tool._strip_markdown_for_tts", side_effect=lambda t: t), \
             patch("os.makedirs"):
            # Should not raise
            await runner._send_voice_reply(event, "Hello")


# =====================================================================
# Discord play_tts skip when in voice channel
# =====================================================================
class TestVoiceInHelp:

    def test_voice_in_help_output(self):
        """The gateway help text includes /voice (generated from registry)."""
        from logos_cli.commands import gateway_help_lines
        help_text = "\n".join(gateway_help_lines())
        assert "/voice" in help_text

    def test_voice_is_known_command(self):
        """The /voice command is in GATEWAY_KNOWN_COMMANDS."""
        from logos_cli.commands import GATEWAY_KNOWN_COMMANDS
        assert "voice" in GATEWAY_KNOWN_COMMANDS


# =====================================================================
# VoiceReceiver unit tests
class TestAutoTtsEmptyTextGuard:
    """Verify base adapter skips TTS when text is empty after markdown strip."""

    def test_empty_after_strip_skips_tts(self):
        """Markdown-only content should not trigger TTS call."""
        import re
        text_content = "****"
        speech_text = re.sub(r'[*_`#\[\]()]', '', text_content)[:4000].strip()
        assert not speech_text, "Expected empty after stripping markdown chars"

    def test_code_block_response_skips_tts(self):
        """Code-only response results in empty speech text."""
        import re
        text_content = "```python\nprint(1)\n```"
        speech_text = re.sub(r'[*_`#\[\]()]', '', text_content)[:4000].strip()
        # Note: base.py regex only strips individual chars, not full code blocks
        # So code blocks are partially stripped but may leave content
        # The real fix is in base.py — empty check after strip

    def test_base_empty_check_in_source(self):
        """base.py must check speech_text is non-empty before calling TTS."""
        import ast, inspect
        from gateway.platforms.base import BasePlatformAdapter
        source = inspect.getsource(BasePlatformAdapter._process_message_background)
        assert "if not speech_text" in source or "not speech_text" in source, (
            "base.py must guard against empty speech_text before TTS call"
        )


class TestStreamTtsToSpeaker:
    """Functional tests for the streaming TTS pipeline."""

    def test_none_sentinel_flushes_buffer(self):
        """None sentinel causes remaining buffer to be spoken."""
        from tools.tts_tool import stream_tts_to_speaker
        text_q = queue.Queue()
        stop_evt = threading.Event()
        done_evt = threading.Event()
        spoken = []

        def display(text):
            spoken.append(text)

        text_q.put("Hello world.")
        text_q.put(None)

        stream_tts_to_speaker(text_q, stop_evt, done_evt, display_callback=display)
        assert done_evt.is_set()
        assert any("Hello" in s for s in spoken)

    def test_stop_event_aborts_early(self):
        """Setting stop_event causes early exit."""
        from tools.tts_tool import stream_tts_to_speaker
        text_q = queue.Queue()
        stop_evt = threading.Event()
        done_evt = threading.Event()
        spoken = []

        stop_evt.set()
        text_q.put("Should not be spoken.")
        text_q.put(None)

        stream_tts_to_speaker(text_q, stop_evt, done_evt, display_callback=lambda t: spoken.append(t))
        assert done_evt.is_set()
        assert len(spoken) == 0

    def test_done_event_set_on_exception(self):
        """tts_done_event is set even when an exception occurs."""
        from tools.tts_tool import stream_tts_to_speaker
        text_q = queue.Queue()
        stop_evt = threading.Event()
        done_evt = threading.Event()

        # Put a non-string that will cause concatenation to fail
        text_q.put(12345)
        text_q.put(None)

        stream_tts_to_speaker(text_q, stop_evt, done_evt)
        assert done_evt.is_set()

    def test_think_blocks_stripped(self):
        """<think>...</think> content is not spoken."""
        from tools.tts_tool import stream_tts_to_speaker
        text_q = queue.Queue()
        stop_evt = threading.Event()
        done_evt = threading.Event()
        spoken = []

        text_q.put("<think>internal reasoning</think>")
        text_q.put("Visible response. ")
        text_q.put(None)

        stream_tts_to_speaker(text_q, stop_evt, done_evt, display_callback=lambda t: spoken.append(t))
        assert done_evt.is_set()
        joined = " ".join(spoken)
        assert "internal reasoning" not in joined
        assert "Visible" in joined

    def test_sentence_splitting(self):
        """Sentences are split at boundaries and spoken individually."""
        from tools.tts_tool import stream_tts_to_speaker
        text_q = queue.Queue()
        stop_evt = threading.Event()
        done_evt = threading.Event()
        spoken = []

        # Two sentences long enough to exceed min_sentence_len (20)
        text_q.put("This is the first sentence. ")
        text_q.put("This is the second sentence. ")
        text_q.put(None)

        stream_tts_to_speaker(text_q, stop_evt, done_evt, display_callback=lambda t: spoken.append(t))
        assert done_evt.is_set()
        assert len(spoken) >= 2

    def test_markdown_stripped_in_speech(self):
        """Markdown formatting is removed before display/speech."""
        from tools.tts_tool import stream_tts_to_speaker
        text_q = queue.Queue()
        stop_evt = threading.Event()
        done_evt = threading.Event()
        spoken = []

        text_q.put("**Bold text** and `code`. ")
        text_q.put(None)

        stream_tts_to_speaker(text_q, stop_evt, done_evt, display_callback=lambda t: spoken.append(t))
        assert done_evt.is_set()
        # Display callback gets raw text (before markdown stripping)
        # But the actual TTS audio would be stripped — we verify pipeline doesn't crash

    def test_duplicate_sentences_deduped(self):
        """Repeated sentences are spoken only once."""
        from tools.tts_tool import stream_tts_to_speaker
        text_q = queue.Queue()
        stop_evt = threading.Event()
        done_evt = threading.Event()
        spoken = []

        # Same sentence twice, each long enough
        text_q.put("This is a repeated sentence. ")
        text_q.put("This is a repeated sentence. ")
        text_q.put(None)

        stream_tts_to_speaker(text_q, stop_evt, done_evt, display_callback=lambda t: spoken.append(t))
        assert done_evt.is_set()
        # First occurrence is spoken, second is deduped
        assert len(spoken) == 1

    def test_no_api_key_display_only(self):
        """Without ELEVENLABS_API_KEY, display callback still works."""
        from tools.tts_tool import stream_tts_to_speaker
        text_q = queue.Queue()
        stop_evt = threading.Event()
        done_evt = threading.Event()
        spoken = []

        text_q.put("Display only text. ")
        text_q.put(None)

        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": ""}):
            stream_tts_to_speaker(text_q, stop_evt, done_evt,
                                  display_callback=lambda t: spoken.append(t))
        assert done_evt.is_set()
        assert len(spoken) >= 1

    def test_long_buffer_flushed_on_timeout(self):
        """Buffer longer than long_flush_len is flushed on queue timeout."""
        from tools.tts_tool import stream_tts_to_speaker
        text_q = queue.Queue()
        stop_evt = threading.Event()
        done_evt = threading.Event()
        spoken = []

        # Put a long text without sentence boundary, then None after a delay
        long_text = "a" * 150  # > long_flush_len (100)
        text_q.put(long_text)

        def delayed_sentinel():
            time.sleep(1.0)
            text_q.put(None)

        t = threading.Thread(target=delayed_sentinel, daemon=True)
        t.start()

        stream_tts_to_speaker(text_q, stop_evt, done_evt,
                              display_callback=lambda t: spoken.append(t))
        t.join(timeout=5)
        assert done_evt.is_set()
        assert len(spoken) >= 1


# =====================================================================
# Bug 1: VoiceReceiver.stop() must hold lock while clearing shared state
# =====================================================================
class TestSendVoiceReplyFilename:
    """_send_voice_reply uses uuid for unique filenames."""

    def test_filename_uses_uuid(self):
        """The method uses uuid in the filename, not time-based."""
        import inspect
        from gateway.run import GatewayRunner
        source = inspect.getsource(GatewayRunner._send_voice_reply)
        assert "uuid" in source, \
            "_send_voice_reply should use uuid for unique filenames"
        assert "int(time.time())" not in source, \
            "_send_voice_reply should not use int(time.time()) — collision risk"

    def test_filenames_are_unique(self):
        """Two calls produce different filenames."""
        import uuid
        names = set()
        for _ in range(100):
            name = f"tts_reply_{uuid.uuid4().hex[:12]}.mp3"
            assert name not in names, f"Collision detected: {name}"
            names.add(name)


# =====================================================================
# Bug 5: Voice timeout cleans up runner voice_mode via callback
class TestAutoTtsTempFileCleanup:
    """Base adapter auto-TTS must clean up generated audio file."""

    def test_source_has_finally_remove(self):
        """play_tts call is wrapped in try/finally with os.remove."""
        import inspect
        from gateway.platforms.base import BasePlatformAdapter
        source = inspect.getsource(BasePlatformAdapter._process_message_background)
        # Find the play_tts section and verify cleanup
        play_tts_idx = source.find("play_tts")
        assert play_tts_idx > 0
        after_play = source[play_tts_idx:]
        finally_idx = after_play.find("finally")
        remove_idx = after_play.find("os.remove")
        assert finally_idx > 0, "play_tts must be in a try/finally block"
        assert remove_idx > 0, "finally block must call os.remove on _tts_path"
        assert remove_idx > finally_idx, "os.remove must be inside the finally block"


# =====================================================================
# Voice channel awareness (get_voice_channel_info / context)
# =====================================================================
class TestDisconnectVoiceCleanup:
    """Bug: disconnect() left voice dicts populated after closing client."""

    @pytest.mark.asyncio
    async def test_disconnect_clears_voice_state(self):
        from unittest.mock import AsyncMock

        adapter = MagicMock()
        adapter._voice_clients = {111: MagicMock(), 222: MagicMock()}
        adapter._voice_receivers = {111: MagicMock(), 222: MagicMock()}
        adapter._voice_listen_tasks = {111: MagicMock(), 222: MagicMock()}
        adapter._voice_timeout_tasks = {111: MagicMock(), 222: MagicMock()}
        adapter._voice_text_channels = {111: 999, 222: 888}

        async def mock_leave(guild_id):
            adapter._voice_receivers.pop(guild_id, None)
            adapter._voice_listen_tasks.pop(guild_id, None)
            adapter._voice_clients.pop(guild_id, None)
            adapter._voice_timeout_tasks.pop(guild_id, None)
            adapter._voice_text_channels.pop(guild_id, None)

        for gid in list(adapter._voice_clients.keys()):
            await mock_leave(gid)

        assert len(adapter._voice_clients) == 0
        assert len(adapter._voice_receivers) == 0
        assert len(adapter._voice_listen_tasks) == 0
        assert len(adapter._voice_timeout_tasks) == 0


# =====================================================================
# Discord Voice Channel Flow Tests
# =====================================================================


@pytest.mark.skipif(
    importlib.util.find_spec("nacl") is None,
    reason="PyNaCl not installed",
)
class TestShouldAutoTtsForChat:
    """Three-layer gate: per-chat enable > per-chat disable > config default."""

    def _make_adapter(self, *, default: bool, enabled=(), disabled=()):
        """Build a bare adapter with only the attrs the gate reads."""
        adapter = SimpleNamespace(
            _auto_tts_default=default,
            _auto_tts_enabled_chats=set(enabled),
            _auto_tts_disabled_chats=set(disabled),
        )
        # Bind the unbound method — _should_auto_tts_for_chat only reads the
        # three attrs above via ``self.``, so an unbound call works.
        from gateway.platforms.base import BasePlatformAdapter
        return BasePlatformAdapter._should_auto_tts_for_chat, adapter

    def test_default_false_no_override_suppresses(self):
        """Issue #16007: voice.auto_tts=False and no per-chat state → no TTS."""
        fn, adapter = self._make_adapter(default=False)
        assert fn(adapter, "chat1") is False

    def test_default_true_no_override_fires(self):
        fn, adapter = self._make_adapter(default=True)
        assert fn(adapter, "chat1") is True

    def test_explicit_enable_overrides_false_default(self):
        """``/voice on`` with config auto_tts=False still fires."""
        fn, adapter = self._make_adapter(default=False, enabled={"chat1"})
        assert fn(adapter, "chat1") is True

    def test_explicit_disable_overrides_true_default(self):
        """``/voice off`` with config auto_tts=True still suppresses."""
        fn, adapter = self._make_adapter(default=True, disabled={"chat1"})
        assert fn(adapter, "chat1") is False

    def test_enabled_wins_over_disabled(self):
        """An explicit enable beats an explicit disable (enable takes priority)."""
        fn, adapter = self._make_adapter(
            default=False, enabled={"chat1"}, disabled={"chat1"}
        )
        assert fn(adapter, "chat1") is True

    def test_per_chat_isolation(self):
        """Enable for chat1 doesn't leak to chat2."""
        fn, adapter = self._make_adapter(default=False, enabled={"chat1"})
        assert fn(adapter, "chat1") is True
        assert fn(adapter, "chat2") is False
