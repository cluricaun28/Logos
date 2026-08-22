"""Tests for Phase C class F (2026-08-22): hermes_cli.main launch-path shim.

Prod incident 2026-08-21 19:00: the main gateway crash-looped 12x
("No module named hermes_cli.main") when units invoked the pre-rebrand
module path against rebranded code. The hermes_cli shim package restored
by class F must keep the legacy invocation working.
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestHermesCliMainShim:
    def test_legacy_module_path_runs_version(self):
        """`python -m hermes_cli.main --version` exits 0 and prints the
        Logos banner — identical behavior to `python -m logos_cli.main`."""
        p = subprocess.run(
            [sys.executable, "-m", "hermes_cli.main", "--version"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert p.returncode == 0, f"stderr: {p.stderr[-800:]}"
        assert "Logos" in p.stdout
        # same banner the primary path prints
        q = subprocess.run(
            [sys.executable, "-m", "logos_cli.main", "--version"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert q.returncode == 0
        assert p.stdout.splitlines()[0] == q.stdout.splitlines()[0]
