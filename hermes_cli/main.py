#!/usr/bin/env python3
"""
Legacy `python -m hermes_cli.main` entry point — rebrand shim.

Phase C class F (2026-08-22): prod incident 2026-08-21 19:00 — the main
gateway crash-looped 12x ("No module named hermes_cli.main") when units
invoked the pre-rebrand module path against rebranded code. This shim
restores that path as a thin delegate so old units/scripts/docs keep
working. The real entry point is logos_cli.main; new invocations should
use `python -m logos_cli.main` (or the `logos`/`hermes` launcher
scripts).
"""

from logos_cli.main import main


if __name__ == "__main__":
    main()
