"""Windows subprocess compatibility helpers — shim for Logos fork.

On WSL/Linux these are no-ops. The real implementation lives in upstream
logos_cli/_subprocess_compat.py with full Win32 creationflags support.

We only need windows_hide_flags() for computer_use — it returns 0 on non-Windows.
"""
from __future__ import annotations

import sys

IS_WINDOWS = sys.platform == "win32"

def windows_hide_flags() -> int:
    """Return Win32 creationflags to hide console window. 0 on non-Windows."""
    if not IS_WINDOWS:
        return 0
    # On actual Windows, use CREATE_NO_WINDOW (0x08000000)
    try:
        import subprocess
        return getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    except Exception:
        return 0

def windows_detach_flags() -> int:
    """Return Win32 creationflags to detach child. 0 on non-Windows."""
    if not IS_WINDOWS:
        return 0
    try:
        import subprocess
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        flags |= getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        return flags
    except Exception:
        return 0

def windows_detach_flags_without_breakaway() -> int:
    return windows_detach_flags()

def windows_detach_popen_kwargs() -> dict:
    """Return Popen kwargs for detaching a child process."""
    if IS_WINDOWS:
        return {"creationflags": windows_detach_flags()}
    return {"start_new_session": True}
