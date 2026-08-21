"""Regression: terminal env cwd must not be corrupted by command output.

2026-08-20 incident: a command whose stdout contained the in-band CWD marker
text (printed wrapper/source code) made _extract_cwd_from_output parse the
literal "%s" between a stray marker pair and store it as self.cwd. Every
subsequent Popen(cwd="%s") then failed with FileNotFoundError, breaking the
entire terminal tool until process restart.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tools.environments.base import _cwd_marker  # noqa: E402
from tools.environments.local import LocalEnvironment  # noqa: E402


def _make_env():
    env = LocalEnvironment(cwd="/tmp")
    env._session_id = "test123"
    env._cwd_marker = _cwd_marker("test123")
    return env


def test_marker_pair_with_format_token_does_not_corrupt_cwd():
    env = _make_env()
    m = env._cwd_marker
    fake_output = (
        "printf \'\n" + m + "%s" + m + "\n\" \"$(pwd -P)\"\n"
        "\n" + m + "/real/path\n" + m + "\n"
    )
    env._extract_cwd_from_output({"output": fake_output})
    assert env.cwd == "/real/path"


def test_garbage_only_marker_pair_leaves_cwd_unchanged():
    env = _make_env()
    m = env._cwd_marker
    env._extract_cwd_from_output({"output": "trap: " + m + "%s" + m})
    assert env.cwd == "/tmp"


def test_bogus_local_cwd_file_ignored():
    env = _make_env()
    env._cwd_file = "/tmp/hermes-test-cwd-guard.txt"
    try:
        with open(env._cwd_file, "w") as f:
            f.write("%s")
        env._update_cwd({"output": ""})
        assert env.cwd == "/tmp"
    finally:
        if os.path.exists(env._cwd_file):
            os.unlink(env._cwd_file)
