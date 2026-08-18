"""Tests for backend-specific bulk download implementations and cleanup() wiring."""

import asyncio
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from tools.environments import ssh as ssh_env
from tools.environments.ssh import SSHEnvironment


# ── SSH helpers ──────────────────────────────────────────────────────


@pytest.fixture
def ssh_mock_env(monkeypatch):
    """Create an SSHEnvironment with mocked connection/sync."""
    monkeypatch.setattr(ssh_env.shutil, "which", lambda _name: "/usr/bin/ssh")
    monkeypatch.setattr(ssh_env.SSHEnvironment, "_establish_connection", lambda self: None)
    monkeypatch.setattr(ssh_env.SSHEnvironment, "_detect_remote_home", lambda self: "/home/testuser")
    monkeypatch.setattr(ssh_env.SSHEnvironment, "_ensure_remote_dirs", lambda self: None)
    monkeypatch.setattr(ssh_env.SSHEnvironment, "init_session", lambda self: None)
    monkeypatch.setattr(
        ssh_env, "FileSyncManager",
        lambda **kw: type("M", (), {
            "sync": lambda self, **k: None,
            "sync_back": lambda self: None,
        })(),
    )
    return SSHEnvironment(host="example.com", user="testuser")


# =====================================================================
# SSH bulk download
# =====================================================================


class TestSSHBulkDownload:
    """Unit tests for _ssh_bulk_download."""

    def test_ssh_bulk_download_runs_tar_over_ssh(self, ssh_mock_env, tmp_path):
        """subprocess.run command should include tar cf - over SSH."""
        dest = tmp_path / "backup.tar"

        with patch.object(subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as mock_run:
            # open() will be called to write stdout; mock it to avoid actual file I/O
            ssh_mock_env._ssh_bulk_download(dest)

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(cmd)
        assert "tar cf -" in cmd_str
        assert "-C /" in cmd_str
        assert "home/testuser/.hermes" in cmd_str
        assert "ssh" in cmd_str
        assert "testuser@example.com" in cmd_str

    def test_ssh_bulk_download_writes_to_dest(self, ssh_mock_env, tmp_path):
        """subprocess.run should receive stdout=open(dest, 'wb')."""
        dest = tmp_path / "backup.tar"

        with patch.object(subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as mock_run:
            ssh_mock_env._ssh_bulk_download(dest)

        # The stdout kwarg should be a file object opened for writing
        call_kwargs = mock_run.call_args
        # stdout is passed as a keyword arg
        stdout_val = call_kwargs.kwargs.get("stdout") or call_kwargs[1].get("stdout")
        # The file was opened via `with open(dest, "wb") as f` and passed as stdout=f.
        # After the context manager exits, the file is closed, but we can verify
        # the dest path was used by checking if the file was created.
        assert dest.exists()

    def test_ssh_bulk_download_raises_on_failure(self, ssh_mock_env, tmp_path):
        """Non-zero returncode should raise RuntimeError."""
        dest = tmp_path / "backup.tar"

        failed = subprocess.CompletedProcess([], 1, stderr=b"Permission denied")
        with patch.object(subprocess, "run", return_value=failed):
            with pytest.raises(RuntimeError, match="SSH bulk download failed"):
                ssh_mock_env._ssh_bulk_download(dest)

    def test_ssh_bulk_download_uses_120s_timeout(self, ssh_mock_env, tmp_path):
        """The subprocess.run call should use a 120s timeout."""
        dest = tmp_path / "backup.tar"

        with patch.object(subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as mock_run:
            ssh_mock_env._ssh_bulk_download(dest)

        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs.get("timeout") == 120 or call_kwargs[1].get("timeout") == 120


class TestSSHCleanup:
    """Verify SSH cleanup() calls sync_back() before closing ControlMaster."""

    def test_ssh_cleanup_calls_sync_back(self, monkeypatch):
        """cleanup() should call sync_back() before SSH control socket teardown."""
        monkeypatch.setattr(ssh_env.shutil, "which", lambda _name: "/usr/bin/ssh")
        monkeypatch.setattr(ssh_env.SSHEnvironment, "_establish_connection", lambda self: None)
        monkeypatch.setattr(ssh_env.SSHEnvironment, "_detect_remote_home", lambda self: "/home/u")
        monkeypatch.setattr(ssh_env.SSHEnvironment, "_ensure_remote_dirs", lambda self: None)
        monkeypatch.setattr(ssh_env.SSHEnvironment, "init_session", lambda self: None)

        call_order = []

        class TrackingSyncManager:
            def __init__(self, **kwargs):
                pass

            def sync(self, **kw):
                pass

            def sync_back(self):
                call_order.append("sync_back")

        monkeypatch.setattr(ssh_env, "FileSyncManager", TrackingSyncManager)

        env = SSHEnvironment(host="h", user="u")
        # Ensure control_socket does not exist so cleanup skips the SSH exit call
        env.control_socket = Path("/nonexistent/socket")

        env.cleanup()

        assert "sync_back" in call_order

    def test_ssh_cleanup_calls_sync_back_before_control_exit(self, monkeypatch):
        """sync_back() must run before the ControlMaster exit command."""
        monkeypatch.setattr(ssh_env.shutil, "which", lambda _name: "/usr/bin/ssh")
        monkeypatch.setattr(ssh_env.SSHEnvironment, "_establish_connection", lambda self: None)
        monkeypatch.setattr(ssh_env.SSHEnvironment, "_detect_remote_home", lambda self: "/home/u")
        monkeypatch.setattr(ssh_env.SSHEnvironment, "_ensure_remote_dirs", lambda self: None)
        monkeypatch.setattr(ssh_env.SSHEnvironment, "init_session", lambda self: None)

        call_order = []

        class TrackingSyncManager:
            def __init__(self, **kwargs):
                pass

            def sync(self, **kw):
                pass

            def sync_back(self):
                call_order.append("sync_back")

        monkeypatch.setattr(ssh_env, "FileSyncManager", TrackingSyncManager)

        env = SSHEnvironment(host="h", user="u")

        # Create a fake control socket so cleanup tries the SSH exit
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".sock") as tmp:
            env.control_socket = Path(tmp.name)

        def mock_run(cmd, **kwargs):
            cmd_str = " ".join(cmd)
            if "-O" in cmd and "exit" in cmd_str:
                call_order.append("control_exit")
            return subprocess.CompletedProcess([], 0)

        with patch.object(subprocess, "run", side_effect=mock_run):
            env.cleanup()

        assert call_order.index("sync_back") < call_order.index("control_exit")


# =====================================================================
# FileSyncManager wiring: bulk_download_fn passed by each backend
# =====================================================================


class TestBulkDownloadWiring:
    """Verify each backend passes bulk_download_fn to FileSyncManager."""

    def test_ssh_passes_bulk_download_fn(self, monkeypatch):
        """SSHEnvironment should pass _ssh_bulk_download to FileSyncManager."""
        monkeypatch.setattr(ssh_env.shutil, "which", lambda _name: "/usr/bin/ssh")
        monkeypatch.setattr(ssh_env.SSHEnvironment, "_establish_connection", lambda self: None)
        monkeypatch.setattr(ssh_env.SSHEnvironment, "_detect_remote_home", lambda self: "/root")
        monkeypatch.setattr(ssh_env.SSHEnvironment, "_ensure_remote_dirs", lambda self: None)
        monkeypatch.setattr(ssh_env.SSHEnvironment, "init_session", lambda self: None)

        captured_kwargs = {}

        class CaptureSyncManager:
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)

            def sync(self, **kw):
                pass

        monkeypatch.setattr(ssh_env, "FileSyncManager", CaptureSyncManager)

        SSHEnvironment(host="h", user="u")

        assert "bulk_download_fn" in captured_kwargs
        assert callable(captured_kwargs["bulk_download_fn"])
