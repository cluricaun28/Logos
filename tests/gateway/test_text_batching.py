"""Tests for text message batching across all gateway adapters.

When a user sends a long message, the messaging client splits it at the
platform's character limit.  Each adapter should buffer rapid successive
text messages from the same session and aggregate them before dispatching.

Covers: Discord, Matrix, WeCom, and the adaptive delay logic for
Telegram and Feishu.
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, SessionSource


# =====================================================================
# Helpers
# =====================================================================

def _make_event(
    text: str,
    platform: Platform,
    chat_id: str = "12345",
    msg_type: MessageType = MessageType.TEXT,
) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=msg_type,
        source=SessionSource(platform=platform, chat_id=chat_id, chat_type="dm"),
    )


# =====================================================================
# Discord text batching
# =====================================================================

def _make_telegram_adapter():
    """Create a minimal TelegramAdapter for testing adaptive delay."""
    from gateway.platforms.telegram import TelegramAdapter

    config = PlatformConfig(enabled=True, token="test-token")
    adapter = object.__new__(TelegramAdapter)
    adapter._platform = Platform.TELEGRAM
    adapter.config = config
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._text_batch_delay_seconds = 0.1
    adapter._text_batch_split_delay_seconds = 0.3
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    adapter._message_handler = AsyncMock()
    adapter.handle_message = AsyncMock()
    return adapter


class TestTelegramAdaptiveDelay:
    @pytest.mark.asyncio
    async def test_short_chunk_uses_normal_delay(self):
        adapter = _make_telegram_adapter()
        adapter._enqueue_text_event(_make_event("short msg", Platform.TELEGRAM))

        # Should flush after the normal 0.1s delay
        await asyncio.sleep(0.15)
        adapter.handle_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_near_limit_chunk_uses_split_delay(self):
        """A chunk near the 4096-char limit should trigger longer delay."""
        adapter = _make_telegram_adapter()
        long_text = "x" * 4050  # near the 4096 limit
        adapter._enqueue_text_event(_make_event(long_text, Platform.TELEGRAM))

        # After the short delay, should NOT have flushed yet
        await asyncio.sleep(0.15)
        adapter.handle_message.assert_not_called()

        # After the split delay, should be flushed
        await asyncio.sleep(0.25)
        adapter.handle_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_split_continuation_merged(self):
        """Two near-limit chunks should both be merged."""
        adapter = _make_telegram_adapter()

        adapter._enqueue_text_event(_make_event("x" * 4050, Platform.TELEGRAM))
        await asyncio.sleep(0.05)
        adapter._enqueue_text_event(_make_event("continuation text", Platform.TELEGRAM))

        # Short chunk arrived → should use normal delay now
        await asyncio.sleep(0.15)
        adapter.handle_message.assert_called_once()
        text = adapter.handle_message.call_args[0][0].text
        assert "continuation text" in text
