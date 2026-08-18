"""Shared fixtures for gateway e2e tests (Telegram).

These tests exercise the full async message flow:
    adapter.handle_message(event)
        → background task
        → GatewayRunner._handle_message (command dispatch)
        → adapter.send() (captured by mock)

No LLM, no real platform connections.
"""

import asyncio
import sys
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, SendResult
from gateway.session import SessionEntry, SessionSource, build_session_key

E2E_MESSAGE_SETTLE_DELAY = 0.3

# Platform library mocks

# Ensure telegram module is available (mock it if not installed)
def _ensure_telegram_mock():
    """Install mock telegram modules so TelegramAdapter can be imported."""
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return # Real library installed

    telegram_mod = MagicMock()
    telegram_mod.Update = MagicMock()
    telegram_mod.Update.ALL_TYPES = []
    telegram_mod.Bot = MagicMock
    telegram_mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    telegram_mod.ext.Application = MagicMock()
    telegram_mod.ext.Application.builder = MagicMock
    telegram_mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    telegram_mod.ext.MessageHandler = MagicMock
    telegram_mod.ext.CommandHandler = MagicMock
    telegram_mod.ext.filters = MagicMock()
    telegram_mod.request.HTTPXRequest = MagicMock

    for name in (
        "telegram",
        "telegram.constants",
        "telegram.ext",
        "telegram.ext.filters",
        "telegram.request",
    ):
        sys.modules.setdefault(name, telegram_mod)


_ensure_telegram_mock()

from gateway.platforms.telegram import TelegramAdapter  # noqa: E402


# Platform-generic factories

def make_source(platform: Platform, chat_id: str = "e2e-chat-1", user_id: str = "e2e-user-1", chat_type: str = "dm") -> SessionSource:
    return SessionSource(
        platform=platform,
        chat_id=chat_id,
        user_id=user_id,
        user_name="e2e_tester",
        chat_type=chat_type,
    )


def make_session_entry(platform: Platform, source: SessionSource = None) -> SessionEntry:
    source = source or make_source(platform)
    return SessionEntry(
        session_key=build_session_key(source),
        session_id=f"sess-{uuid.uuid4().hex[:8]}",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=platform,
        chat_type="dm",
    )


def make_event(
    platform: Platform,
    text: str = "/help",
    chat_id: str = "e2e-chat-1",
    user_id: str = "e2e-user-1",
    chat_type: str = "dm",
) -> MessageEvent:
    return MessageEvent(
        text=text,
        source=make_source(platform, chat_id, user_id, chat_type),
        message_id=f"msg-{uuid.uuid4().hex[:8]}",
    )


def make_runner(platform: Platform, session_entry: SessionEntry = None) -> "GatewayRunner":
    """Create a GatewayRunner with mocked internals for e2e testing.

    Skips __init__ to avoid filesystem/network side effects.
    """
    from gateway.run import GatewayRunner

    if session_entry is None:
        session_entry = make_session_entry(platform)

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={platform: PlatformConfig(enabled=True, token="e2e-test-token")}
    )
    runner.adapters = {}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), emit_collect=AsyncMock(return_value=[]), loaded_hooks=False)

    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner.session_store.reset_session = MagicMock()

    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._shutdown_event = asyncio.Event()
    runner._exit_reason = None
    runner._exit_code = None
    runner._background_tasks = set()
    runner._draining = False
    runner._restart_requested = False
    runner._restart_task_started = False
    runner._restart_detached = False
    runner._restart_via_service = False
    from gateway.restart import DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
    runner._restart_drain_timeout = DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
    runner._stop_task = None
    runner._busy_input_mode = "interrupt"
    runner._running_agents_ts = {}
    runner._pending_model_notes = {}
    runner._update_prompt_pending = {}
    runner._voice_mode = {}
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False

    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._handle_message_with_agent = AsyncMock(return_value="agent-handled-default")
    runner._should_send_voice_reply = lambda *_a, **_kw: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *a, **kw: None
    runner._emit_gateway_run_progress = AsyncMock()

    runner.pairing_store = MagicMock()
    runner.pairing_store._is_rate_limited = MagicMock(return_value=False)
    runner.pairing_store.generate_code = MagicMock(return_value="ABC123")

    return runner


def make_adapter(platform: Platform, runner=None):
    """Create a platform adapter wired to *runner*, with send methods mocked."""
    if runner is None:
        runner = make_runner(platform)

    config = PlatformConfig(enabled=True, token="e2e-test-token")

    adapter = TelegramAdapter(config)
    platform_key = Platform.TELEGRAM

    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="e2e-resp-1"))
    adapter.send_typing = AsyncMock()

    adapter.set_message_handler(runner._handle_message)
    runner.adapters[platform_key] = adapter

    return adapter


async def send_and_capture(adapter, text: str, platform: Platform, **event_kwargs) -> AsyncMock:
    """Send a message through the full e2e flow and return the send mock."""
    event = make_event(platform, text, **event_kwargs)
    adapter.send.reset_mock()
    await adapter.handle_message(event)
    await asyncio.sleep(0.3)
    return adapter.send


# Parametrized fixtures for platform-generic tests
@pytest.fixture(params=[Platform.TELEGRAM], ids=["telegram"])
def platform(request):
    return request.param


@pytest.fixture()
def source(platform):
    return make_source(platform)


@pytest.fixture()
def session_entry(platform, source):
    return make_session_entry(platform, source)


@pytest.fixture()
def runner(platform, session_entry):
    return make_runner(platform, session_entry)


@pytest.fixture()
def adapter(platform, runner):
    return make_adapter(platform, runner)
