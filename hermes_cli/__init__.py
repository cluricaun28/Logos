#!/usr/bin/env python3
"""
Legacy hermes_cli package — rebrand shim (Phase C class F, 2026-08-22).

The 2026-08 rebrand renamed hermes_cli/ -> logos_cli/ structurally
(commit 2f0b1243). Fleet units that still invoke the old module path
(`python -m hermes_cli.main`) crash-looped against rebranded code with
"No module named hermes_cli.main" (prod incident 2026-08-21 19:00 —
the main gateway crash-looped 12x). Units were patched live to
logos_cli.main, but any future unit/script/doc using the old
invocation would crash identically.

This package restores the old entry point as a thin delegate so the
legacy invocation keeps working. New code should use logos_cli.main.
"""
