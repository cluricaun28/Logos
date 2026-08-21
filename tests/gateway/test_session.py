"""Tests for gateway session management."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from gateway.config import Platform, HomeChannel, GatewayConfig, PlatformConfig
from gateway.session import (
    SessionSource,
    SessionStore,
    build_session_context,
    build_session_context_prompt,
    build_session_key,
)


class TestSessionSourceRoundtrip:
    def test_full_roundtrip(self):
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="12345",
            chat_name="My Group",
            chat_type="group",
            user_id="99",
            user_name="alice",
            thread_id="t1",
        )
        d = source.to_dict()
        restored = SessionSource.from_dict(d)

        assert restored.platform == Platform.TELEGRAM
        assert restored.chat_id == "12345"
        assert restored.chat_name == "My Group"
        assert restored.chat_type == "group"
        assert restored.user_id == "99"
        assert restored.user_name == "alice"
        assert restored.thread_id == "t1"
    def test_minimal_roundtrip(self):
        source = SessionSource(platform=Platform.LOCAL, chat_id="cli")
        d = source.to_dict()
        restored = SessionSource.from_dict(d)
        assert restored.platform == Platform.LOCAL
        assert restored.chat_id == "cli"
        assert restored.chat_type == "dm"  # default value preserved

    def test_chat_id_coerced_to_string(self):
        """from_dict should handle numeric chat_id (common from Telegram)."""
        restored = SessionSource.from_dict({
            "platform": "telegram",
            "chat_id": 12345,
        })
        assert restored.chat_id == "12345"
        assert isinstance(restored.chat_id, str)
    def test_invalid_platform_raises(self):
        with pytest.raises((ValueError, KeyError)):
            SessionSource.from_dict({"platform": "nonexistent", "chat_id": "1"})


class TestSessionSourceDescription:
    def test_local_cli(self):
        source = SessionSource(
            platform=Platform.LOCAL, chat_id="cli",
            chat_name="CLI terminal", chat_type="dm",
        )
        assert source.description == "CLI terminal"

    def test_dm_with_username(self):
        source = SessionSource(
            platform=Platform.TELEGRAM, chat_id="123",
            chat_type="dm", user_name="bob",
        )
        assert "DM" in source.description
        assert "bob" in source.description

    def test_dm_without_username_falls_back_to_user_id(self):
        source = SessionSource(
            platform=Platform.TELEGRAM, chat_id="123",
            chat_type="dm", user_id="456",
        )
        assert "456" in source.description
    def test_channel_type(self):
        source = SessionSource(
            platform=Platform.TELEGRAM, chat_id="100",
            chat_type="channel", chat_name="Announcements",
        )
        assert "channel" in source.description
        assert "Announcements" in source.description
class TestLocalCliFactory:
    def test_local_cli_defaults(self):
        source = SessionSource(
            platform=Platform.LOCAL, chat_id="cli",
            chat_name="CLI terminal", chat_type="dm",
        )
        assert source.platform == Platform.LOCAL
        assert source.chat_id == "cli"
        assert source.chat_type == "dm"
        assert source.chat_name == "CLI terminal"
class TestSessionStoreRewriteTranscript:
    """Regression: /retry and /undo must persist truncated history to disk."""

    @pytest.fixture()
    def store(self, tmp_path):
        config = GatewayConfig()
        with patch("gateway.session.SessionStore._ensure_loaded"):
            s = SessionStore(sessions_dir=tmp_path, config=config)
        s._db = None  # no SQLite for these tests
        s._loaded = True
        return s

    def test_rewrite_replaces_jsonl(self, store, tmp_path):
        session_id = "test_session_1"
        # Write initial transcript
        for msg in [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "undo this"},
            {"role": "assistant", "content": "ok"},
        ]:
            store.append_to_transcript(session_id, msg)

        # Rewrite with truncated history
        store.rewrite_transcript(session_id, [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ])

        reloaded = store.load_transcript(session_id)
        assert len(reloaded) == 2
        assert reloaded[0]["content"] == "hello"
        assert reloaded[1]["content"] == "hi"

    def test_rewrite_with_empty_list(self, store):
        session_id = "test_session_2"
        store.append_to_transcript(session_id, {"role": "user", "content": "hi"})

        store.rewrite_transcript(session_id, [])

        reloaded = store.load_transcript(session_id)
        assert reloaded == []


class TestLoadTranscriptCorruptLines:
    """Regression: corrupt JSONL lines (e.g. from mid-write crash) must be
    skipped instead of crashing the entire transcript load.  GH-1193."""

    @pytest.fixture()
    def store(self, tmp_path):
        config = GatewayConfig()
        with patch("gateway.session.SessionStore._ensure_loaded"):
            s = SessionStore(sessions_dir=tmp_path, config=config)
        s._db = None
        s._loaded = True
        return s

    def test_corrupt_line_skipped(self, store, tmp_path):
        session_id = "corrupt_test"
        transcript_path = store.get_transcript_path(session_id)
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        with open(transcript_path, "w") as f:
            f.write('{"role": "user", "content": "hello"}\n')
            f.write('{"role": "assistant", "content": "hi th')  # truncated
            f.write("\n")
            f.write('{"role": "user", "content": "goodbye"}\n')

        messages = store.load_transcript(session_id)
        assert len(messages) == 2
        assert messages[0]["content"] == "hello"
        assert messages[1]["content"] == "goodbye"

    def test_all_lines_corrupt_returns_empty(self, store, tmp_path):
        session_id = "all_corrupt"
        transcript_path = store.get_transcript_path(session_id)
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        with open(transcript_path, "w") as f:
            f.write("not json at all\n")
            f.write("{truncated\n")

        messages = store.load_transcript(session_id)
        assert messages == []

    def test_valid_transcript_unaffected(self, store, tmp_path):
        session_id = "valid_test"
        store.append_to_transcript(session_id, {"role": "user", "content": "a"})
        store.append_to_transcript(session_id, {"role": "assistant", "content": "b"})

        messages = store.load_transcript(session_id)
        assert len(messages) == 2
        assert messages[0]["content"] == "a"
        assert messages[1]["content"] == "b"


class TestLoadTranscriptPreferLongerSource:
    """Regression: load_transcript must return whichever source (SQLite or JSONL)
    has more messages to prevent silent truncation.  GH-3212."""

    @pytest.fixture()
    def store_with_db(self, tmp_path):
        """SessionStore with both SQLite and JSONL active."""
        from logos_state import SessionDB

        config = GatewayConfig()
        with patch("gateway.session.SessionStore._ensure_loaded"):
            s = SessionStore(sessions_dir=tmp_path, config=config)
        s._db = SessionDB(db_path=tmp_path / "state.db")
        s._loaded = True
        return s

    def test_jsonl_longer_than_sqlite_returns_jsonl(self, store_with_db):
        """Legacy session: JSONL has full history, SQLite has only recent turn."""
        sid = "legacy_session"
        store_with_db._db.create_session(session_id=sid, source="gateway", model="m")
        # JSONL has 10 messages (legacy history — written before SQLite existed)
        for i in range(10):
            role = "user" if i % 2 == 0 else "assistant"
            store_with_db.append_to_transcript(
                sid, {"role": role, "content": f"msg-{i}"}, skip_db=True,
            )
        # SQLite has only 2 messages (recent turn after migration)
        store_with_db._db.append_message(session_id=sid, role="user", content="new-q")
        store_with_db._db.append_message(session_id=sid, role="assistant", content="new-a")

        result = store_with_db.load_transcript(sid)
        assert len(result) == 10
        assert result[0]["content"] == "msg-0"

    def test_sqlite_longer_than_jsonl_returns_sqlite(self, store_with_db):
        """Fully migrated session: SQLite has more (JSONL stopped growing)."""
        sid = "migrated_session"
        store_with_db._db.create_session(session_id=sid, source="gateway", model="m")
        # JSONL has 2 old messages
        store_with_db.append_to_transcript(
            sid, {"role": "user", "content": "old-q"}, skip_db=True,
        )
        store_with_db.append_to_transcript(
            sid, {"role": "assistant", "content": "old-a"}, skip_db=True,
        )
        # SQLite has 4 messages (superset after migration)
        for i in range(4):
            role = "user" if i % 2 == 0 else "assistant"
            store_with_db._db.append_message(session_id=sid, role=role, content=f"db-{i}")

        result = store_with_db.load_transcript(sid)
        assert len(result) == 4
        assert result[0]["content"] == "db-0"

    def test_sqlite_empty_falls_back_to_jsonl(self, store_with_db):
        """No SQLite rows — falls back to JSONL (original behavior preserved)."""
        sid = "no_db_rows"
        store_with_db.append_to_transcript(
            sid, {"role": "user", "content": "hello"}, skip_db=True,
        )
        store_with_db.append_to_transcript(
            sid, {"role": "assistant", "content": "hi"}, skip_db=True,
        )

        result = store_with_db.load_transcript(sid)
        assert len(result) == 2
        assert result[0]["content"] == "hello"

    def test_both_empty_returns_empty(self, store_with_db):
        """Neither source has data — returns empty list."""
        result = store_with_db.load_transcript("nonexistent")
        assert result == []

    def test_equal_length_prefers_sqlite(self, store_with_db):
        """When both have same count, SQLite wins (has richer fields like reasoning)."""
        sid = "equal_session"
        store_with_db._db.create_session(session_id=sid, source="gateway", model="m")
        # Write 2 messages to JSONL only
        store_with_db.append_to_transcript(
            sid, {"role": "user", "content": "jsonl-q"}, skip_db=True,
        )
        store_with_db.append_to_transcript(
            sid, {"role": "assistant", "content": "jsonl-a"}, skip_db=True,
        )
        # Write 2 different messages to SQLite only
        store_with_db._db.append_message(session_id=sid, role="user", content="db-q")
        store_with_db._db.append_message(session_id=sid, role="assistant", content="db-a")

        result = store_with_db.load_transcript(sid)
        assert len(result) == 2
        # Should be the SQLite version (equal count → prefers SQLite)
        assert result[0]["content"] == "db-q"


class TestSessionStoreSwitchSession:
    """Regression coverage for gateway /resume session switching semantics."""
class TestSessionStoreEntriesAttribute:
    """Regression: /reset must access _entries, not _sessions."""

    def test_entries_attribute_exists(self):
        config = GatewayConfig()
        with patch("gateway.session.SessionStore._ensure_loaded"):
            store = SessionStore(sessions_dir=Path("/tmp"), config=config)
        store._loaded = True
        assert hasattr(store, "_entries")
        assert not hasattr(store, "_sessions")


class TestHasAnySessions:
    """Tests for has_any_sessions() fix (issue #351)."""

    @pytest.fixture
    def store_with_mock_db(self, tmp_path):
        """SessionStore with a mocked database."""
        config = GatewayConfig()
        with patch("gateway.session.SessionStore._ensure_loaded"):
            s = SessionStore(sessions_dir=tmp_path, config=config)
        s._loaded = True
        s._entries = {}
        s._db = MagicMock()
        return s

    def test_uses_database_count_when_available(self, store_with_mock_db):
        """has_any_sessions should use database session_count, not len(_entries)."""
        store = store_with_mock_db
        # Simulate single-platform user with only 1 entry in memory
        store._entries = {"telegram:12345": MagicMock()}
        # But database has 3 sessions (current + 2 previous resets)
        store._db.session_count.return_value = 3

        assert store.has_any_sessions() is True
        store._db.session_count.assert_called_once()

    def test_first_session_ever_returns_false(self, store_with_mock_db):
        """First session ever should return False (only current session in DB)."""
        store = store_with_mock_db
        store._entries = {"telegram:12345": MagicMock()}
        # Database has exactly 1 session (the current one just created)
        store._db.session_count.return_value = 1

        assert store.has_any_sessions() is False

    def test_fallback_without_database(self, tmp_path):
        """Should fall back to len(_entries) when DB is not available."""
        config = GatewayConfig()
        with patch("gateway.session.SessionStore._ensure_loaded"):
            store = SessionStore(sessions_dir=tmp_path, config=config)
        store._loaded = True
        store._db = None
        store._entries = {"key1": MagicMock(), "key2": MagicMock()}

        # > 1 entries means has sessions
        assert store.has_any_sessions() is True

        store._entries = {"key1": MagicMock()}
        assert store.has_any_sessions() is False


class TestLastPromptTokens:
    """Tests for the last_prompt_tokens field — actual API token tracking."""

    def test_session_entry_default(self):
        """New sessions should have last_prompt_tokens=0."""
        from gateway.session import SessionEntry
        from datetime import datetime
        entry = SessionEntry(
            session_key="test",
            session_id="s1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        assert entry.last_prompt_tokens == 0

    def test_session_entry_roundtrip(self):
        """last_prompt_tokens should survive serialization/deserialization."""
        from gateway.session import SessionEntry
        from datetime import datetime
        entry = SessionEntry(
            session_key="test",
            session_id="s1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            last_prompt_tokens=42000,
        )
        d = entry.to_dict()
        assert d["last_prompt_tokens"] == 42000
        restored = SessionEntry.from_dict(d)
        assert restored.last_prompt_tokens == 42000

    def test_session_entry_from_old_data(self):
        """Old session data without last_prompt_tokens should default to 0."""
        from gateway.session import SessionEntry
        data = {
            "session_key": "test",
            "session_id": "s1",
            "created_at": "2025-01-01T00:00:00",
            "updated_at": "2025-01-01T00:00:00",
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            # No last_prompt_tokens — old format
        }
        entry = SessionEntry.from_dict(data)
        assert entry.last_prompt_tokens == 0

    def test_update_session_sets_last_prompt_tokens(self, tmp_path):
        """update_session should store the actual prompt token count."""
        config = GatewayConfig()
        with patch("gateway.session.SessionStore._ensure_loaded"):
            store = SessionStore(sessions_dir=tmp_path, config=config)
        store._loaded = True
        store._db = None
        store._save = MagicMock()

        from gateway.session import SessionEntry
        from datetime import datetime
        entry = SessionEntry(
            session_key="k1",
            session_id="s1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        store._entries = {"k1": entry}

        store.update_session("k1", last_prompt_tokens=85000)
        assert entry.last_prompt_tokens == 85000

    def test_update_session_none_does_not_change(self, tmp_path):
        """update_session with default (None) should not change last_prompt_tokens."""
        config = GatewayConfig()
        with patch("gateway.session.SessionStore._ensure_loaded"):
            store = SessionStore(sessions_dir=tmp_path, config=config)
        store._loaded = True
        store._db = None
        store._save = MagicMock()

        from gateway.session import SessionEntry
        from datetime import datetime
        entry = SessionEntry(
            session_key="k1",
            session_id="s1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            last_prompt_tokens=50000,
        )
        store._entries = {"k1": entry}

        store.update_session("k1")  # No last_prompt_tokens arg
        assert entry.last_prompt_tokens == 50000  # unchanged

    def test_update_session_zero_resets(self, tmp_path):
        """update_session with last_prompt_tokens=0 should reset the field."""
        config = GatewayConfig()
        with patch("gateway.session.SessionStore._ensure_loaded"):
            store = SessionStore(sessions_dir=tmp_path, config=config)
        store._loaded = True
        store._db = None
        store._save = MagicMock()

        from gateway.session import SessionEntry
        from datetime import datetime
        entry = SessionEntry(
            session_key="k1",
            session_id="s1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            last_prompt_tokens=85000,
        )
        store._entries = {"k1": entry}

        store.update_session("k1", last_prompt_tokens=0)
        assert entry.last_prompt_tokens == 0

class TestRewriteTranscriptPreservesReasoning:
    """rewrite_transcript must not drop reasoning fields from SQLite."""

    def test_reasoning_survives_rewrite(self, tmp_path):
        from logos_state import SessionDB

        db = SessionDB(db_path=tmp_path / "test.db")
        session_id = "reasoning-test"
        db.create_session(session_id=session_id, source="cli")

        # Insert a message WITH all three reasoning fields
        db.append_message(
            session_id=session_id,
            role="assistant",
            content="The answer is 42.",
            reasoning="I need to think step by step.",
            reasoning_content="provider scratchpad",
            reasoning_details=[{"type": "summary", "text": "step by step"}],
            codex_reasoning_items=[{"id": "r1", "type": "reasoning"}],
        )

        # Verify all three were stored
        before = db.get_messages_as_conversation(session_id)
        assert before[0].get("reasoning") == "I need to think step by step."
        assert before[0].get("reasoning_content") == "provider scratchpad"
        assert before[0].get("reasoning_details") == [{"type": "summary", "text": "step by step"}]
        assert before[0].get("codex_reasoning_items") == [{"id": "r1", "type": "reasoning"}]

        # Now simulate /retry: build the SessionStore and call rewrite_transcript
        config = GatewayConfig()
        with patch("gateway.session.SessionStore._ensure_loaded"):
            store = SessionStore(sessions_dir=tmp_path, config=config)
        store._db = db
        store._loaded = True

        # rewrite_transcript receives the messages that load_transcript returned
        store.rewrite_transcript(session_id, before)

        # Load again — all three reasoning fields must survive
        after = db.get_messages_as_conversation(session_id)
        assert after[0].get("reasoning") == "I need to think step by step."
        assert after[0].get("reasoning_content") == "provider scratchpad"
        assert after[0].get("reasoning_details") == [{"type": "summary", "text": "step by step"}]
        assert after[0].get("codex_reasoning_items") == [{"id": "r1", "type": "reasoning"}]
