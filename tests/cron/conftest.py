"""Shared fixtures for tests/cron.

Isolate the scheduler tick lock so tests that call the real ``tick()`` do not
contend with the live gateway's in-process cron ticker.  ``tick()`` takes an
exclusive non-blocking lock on ``_logos_home/cron/.tick.lock`` and, if it
cannot acquire it, logs "Tick skipped — another instance holds the lock" and
returns 0 without processing any job.  On a host running a live gateway the
lock is held, so an unisolated ``tick()`` call silently processes nothing and
delivery/output assertions fail.  Redirecting the lock to a per-test temp file
removes that dependency deterministically.
"""
import pytest


@pytest.fixture(autouse=True)
def _isolate_tick_lock(tmp_path, monkeypatch):
    """Point cron.scheduler's tick lock at a per-test temp file (no shared lock)."""
    import cron.scheduler as _sched
    lock_dir = tmp_path / "cron-lock"
    lock_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(_sched, "_LOCK_DIR", lock_dir)
    monkeypatch.setattr(_sched, "_LOCK_FILE", lock_dir / ".tick.lock")
    yield
