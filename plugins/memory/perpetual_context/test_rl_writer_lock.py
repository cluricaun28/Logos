"""Cross-process writer lock for the RL index (rl_index.RLIndex).

Root cause covered: external-content FTS5 shadow tables drift (and later
read as "database disk image is malformed") when a second process commits
content-table changes during build_index's trigger-drop window. Recurred
10/4, 10/10, 10/11. See RL
technology/rl-fts-malformed-root-cause-2026-10-11.md.
"""

import fcntl
import os
import sqlite3
import subprocess
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from plugins.memory.perpetual_context import rl_index  # noqa: E402
from plugins.memory.perpetual_context.rl_index import RLIndex  # noqa: E402


@pytest.fixture
def idx(tmp_path):
    r = RLIndex(db_path=str(tmp_path / "rl_test.db"))
    yield r
    # Ensure clean process-level state between tests.
    rl_index._PROCESS_LOCK_TOKEN = None
    rl_index._PROCESS_LOCK_DEPTH = 0
    if rl_index._PROCESS_LOCK_FD is not None:
        try:
            rl_index._PROCESS_LOCK_FD.close()
        except OSError:
            pass
        rl_index._PROCESS_LOCK_FD = None


def _lock_path(idx: RLIndex) -> str:
    return idx._db_path + ".writer.lock"


def _try_flock(path: str):
    """Open a FRESH fd and attempt a non-blocking exclusive flock."""
    f = open(path, "a+")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return f
    except (BlockingIOError, InterruptedError):
        f.close()
        return None


def _release(f):
    try:
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()
    except (OSError, ValueError):
        pass


def test_lock_acquire_releases(idx):
    idx._writer_lock()
    # A fresh fd cannot grab the lock while held.
    assert _try_flock(_lock_path(idx)) is None
    idx._writer_unlock()
    # Now a fresh fd can.
    f = _try_flock(_lock_path(idx))
    assert f is not None
    _release(f)


def test_lock_reentrant_per_process(idx):
    idx._writer_lock()
    idx._writer_lock()  # nested (reindex_stale -> update_file pattern)
    assert rl_index._PROCESS_LOCK_DEPTH == 2
    idx._writer_unlock()  # inner release keeps the lock held
    assert _try_flock(_lock_path(idx)) is None
    idx._writer_unlock()  # outer release frees it
    assert rl_index._PROCESS_LOCK_DEPTH == 0
    f = _try_flock(_lock_path(idx))
    assert f is not None
    _release(f)


def test_unlock_without_lock_is_noop(idx):
    idx._writer_unlock()  # must not raise


def test_adopt_process_writer_lock(idx):
    """A wrapper script's flock + adopt => RLIndex re-enters, no
    self-contention, and the wrapper's fd keeps the OS lock until it
    closes (coexistence with pre-repo-lock ops scripts)."""
    wrapper_fd = open(_lock_path(idx), "a+")
    fcntl.flock(wrapper_fd, fcntl.LOCK_EX)
    try:
        rl_index.adopt_process_writer_lock()
        # RLIndex must re-enter without blocking on the wrapper's fd.
        idx._writer_lock()
        assert rl_index._PROCESS_LOCK_DEPTH == 2
        idx._writer_unlock()
        # The OS lock is still held — by the wrapper's fd.
        assert _try_flock(_lock_path(idx)) is None
        # Wrapper releases at process end: lock is free.
        _release(wrapper_fd)
        assert _try_flock(_lock_path(idx)) is not None
    finally:
        try:
            _release(wrapper_fd)
        except OSError:
            pass


def test_cross_process_timeout(monkeypatch, idx):
    monkeypatch.setattr(rl_index, "RL_WRITER_LOCK_TIMEOUT", 0.5)
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import fcntl,time,sys;"
                f"f=open({str(_lock_path(idx))!r},'a+');"
                "fcntl.flock(f,fcntl.LOCK_EX);"
                "print('held',flush=True);"
                "time.sleep(5)"
            ),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        t0 = time.monotonic()
        with pytest.raises(sqlite3.OperationalError):
            idx._writer_lock()
        assert time.monotonic() - t0 < 3.0
    finally:
        holder.kill()
        holder.wait()
    # After the holder is gone, acquisition succeeds promptly.
    idx._writer_lock()
    idx._writer_unlock()


def test_cross_process_wait_then_acquire(monkeypatch, idx):
    monkeypatch.setattr(rl_index, "RL_WRITER_LOCK_TIMEOUT", 10.0)
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import fcntl,time,sys;"
                f"f=open({str(_lock_path(idx))!r},'a+');"
                "fcntl.flock(f,fcntl.LOCK_EX);"
                "print('held',flush=True);"
                "time.sleep(1.2);"
                "fcntl.flock(f,fcntl.LOCK_UN)"
            ),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        t0 = time.monotonic()
        idx._writer_lock()  # should wait ~1.2s, then acquire
        waited = time.monotonic() - t0
        assert 0.9 <= waited <= 6.0
        idx._writer_unlock()
    finally:
        holder.kill()
        holder.wait()


def test_write_path_holds_lock_during_work(idx, monkeypatch):
    """reindex_stale must hold the writer lock while the builder runs."""
    from plugins.memory.perpetual_context import rl_builder

    observed = {}

    def fake_reindex_stale(**kwargs):
        observed["held"] = _try_flock(_lock_path(idx)) is None
        return 3

    monkeypatch.setattr(rl_builder, "reindex_stale", fake_reindex_stale)
    n = idx.reindex_stale()
    assert n == 3
    assert observed["held"] is True  # lock held during the builder call
    assert _try_flock(_lock_path(idx)) is not None  # released after


def test_build_index_holds_lock_during_work(idx, monkeypatch):
    from plugins.memory.perpetual_context import rl_builder

    observed = {}

    def fake_build(**kwargs):
        observed["held"] = _try_flock(_lock_path(idx)) is None
        return {"files_indexed": 0}

    monkeypatch.setattr(rl_builder, "build_index", fake_build)
    stats = idx.build_index()
    assert stats == {"files_indexed": 0}
    assert observed["held"] is True
