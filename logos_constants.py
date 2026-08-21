"""Shared constants for Logos.

Import-safe module with no dependencies — can be imported from anywhere
without risk of circular imports.
"""

import os
from pathlib import Path


def _default_home() -> Path:
    """Return the default Logos home directory (no env vars).

    New installs: ``~/.logos`` (preferred when it exists). Legacy
    installs: ``~/.hermes`` if it already exists on disk and ``~/.logos``
    does not (keeps pre-rebrand homes working with zero migration).

    Existence checks are OSError-guarded: under a monkeypatched or
    permission-restricted ``Path.home()`` (e.g. tests simulating sudo as
    root), a stat failure is treated as "does not exist" instead of
    propagating.
    """
    home = Path.home()
    new_home = home / ".logos"
    legacy = home / ".hermes"
    try:
        if new_home.exists():
            return new_home
    except OSError:
        pass
    try:
        if legacy.exists():
            return legacy
    except OSError:
        pass
    return new_home


def get_logos_home() -> Path:
    """Return the Logos home directory.

    Resolution order:
    1. ``$LOGOS_HOME`` env var
    2. ``$HERMES_HOME`` env var (legacy, still honored)
    3. ``~/.logos`` if it already exists on disk
    4. ``~/.hermes`` if it already exists on disk (legacy home — kept
       working with zero migration)
    5. ``~/.logos`` (new-install default)

    This is the single source of truth — all other copies should import this.
    """
    val = os.environ.get("LOGOS_HOME", "").strip() or os.environ.get("HERMES_HOME", "").strip()
    return Path(val) if val else _default_home()


# Legacy alias (pre-rebrand name) — keep for plugin/third-party compatibility.
get_hermes_home = get_logos_home


def get_logos_root() -> Path:
    """Return the root Logos directory for profile-level operations.

    In standard deployments this is the default home (``~/.logos`` for new
    installs, ``~/.hermes`` for legacy installs).

    In Docker or custom deployments where ``HERMES_HOME``/``LOGOS_HOME``
    points outside the default home (e.g. ``/opt/data``), returns that
    value directly — that IS the root.

    In profile mode where the home env var is ``<root>/profiles/<name>``,
    returns ``<root>`` so that ``profile list`` can see all profiles.
    Works both for standard (``~/.hermes/profiles/coder``) and Docker
    (``/opt/data/profiles/coder``) layouts.

    Import-safe — no dependencies beyond stdlib.
    """
    native_home = _default_home()
    env_home = os.environ.get("LOGOS_HOME", "") or os.environ.get("HERMES_HOME", "")
    if not env_home:
        return native_home
    env_path = Path(env_home)
    try:
        env_path.resolve().relative_to(native_home.resolve())
        # Home env var is under the default home (normal or profile mode)
        return native_home
    except ValueError:
        pass

    # Docker / custom deployment.
    # Check if this is a profile path: <root>/profiles/<name>
    # If the immediate parent dir is named "profiles", the root is
    # the grandparent — this covers Docker profiles correctly.
    if env_path.parent.name == "profiles":
        return env_path.parent.parent

    # Not a profile path — the home env var itself is the root
    return env_path


# Legacy alias (pre-rebrand name).
get_default_hermes_root = get_logos_root


def get_optional_skills_dir(default: Path | None = None) -> Path:
    """Return the optional-skills directory, honoring package-manager wrappers.

    Packaged installs may ship ``optional-skills`` outside the Python package
    tree and expose it via ``LOGOS_OPTIONAL_SKILLS`` (legacy
    ``HERMES_OPTIONAL_SKILLS`` still honored).
    """
    override = logos_env("OPTIONAL_SKILLS", "").strip()
    if override:
        return Path(override)
    if default is not None:
        return default
    return get_logos_home() / "optional-skills"


def get_logos_dir(new_subpath: str, old_name: str) -> Path:
    """Resolve a Logos subdirectory with backward compatibility.

    New installs get the consolidated layout (e.g. ``cache/images``).
    Existing installs that already have the old path (e.g. ``image_cache``)
    keep using it — no migration required.

    Args:
        new_subpath: Preferred path relative to the Logos home (e.g. ``"cache/images"``).
        old_name: Legacy path relative to the Logos home (e.g. ``"image_cache"``).

    Returns:
        Absolute ``Path`` — old location if it exists on disk, otherwise the new one.
    """
    home = get_logos_home()
    old_path = home / old_name
    if old_path.exists():
        return old_path
    return home / new_subpath


# Legacy alias (pre-rebrand name).
get_hermes_dir = get_logos_dir


def display_logos_home() -> str:
    """Return a user-friendly display string for the current Logos home.

    Uses ``~/`` shorthand for readability::

        default:  ``~/.hermes`` (legacy) or ``~/.logos`` (new)
        profile:  ``~/.hermes/profiles/coder``
        custom:   ``/opt/logos-custom``

    Use this in **user-facing** print/log messages instead of hardcoding
    the home path.  For code that needs a real ``Path``, use
    :func:`get_logos_home` instead.
    """
    home = get_logos_home()
    try:
        return "~/" + str(home.relative_to(Path.home()))
    except ValueError:
        return str(home)


# Legacy alias (pre-rebrand name).
display_hermes_home = display_logos_home


# ─── Legacy env-var compatibility shims ──────────────────────────────────────
# The Hermes→Logos rebrand renamed every ``HERMES_*`` environment variable to
# ``LOGOS_*``.  Fleet ``.env`` files still carry the old names until the deploy
# wave, so every env-var read in the codebase goes through these shims:
# ``LOGOS_*`` wins when set, otherwise the legacy ``HERMES_*`` value is honored.
# This is the single documented legacy-compatibility point for env vars.


def logos_env(name: str, default: str | None = None) -> str | None:
    """Read ``LOGOS_<name>``, falling back to legacy ``HERMES_<name>``.

    If both are set, ``LOGOS_<name>`` wins.  Returns *default* when neither is
    set.  Drop-in replacement for the pre-rebrand ``os.getenv("HERMES_<name>")``
    reads — behavior is a strict superset (the legacy value is still honored).
    """
    val = os.environ.get("LOGOS_" + name)
    if val is None:
        val = os.environ.get("HERMES_" + name)
    if val is None:
        return default
    return val


def logos_env_set(name: str) -> bool:
    """True when either ``LOGOS_<name>`` or legacy ``HERMES_<name>`` is set."""
    return ("LOGOS_" + name) in os.environ or ("HERMES_" + name) in os.environ


def logos_env_raw(name: str) -> str:
    """``LOGOS_<name>``-first read that preserves ``KeyError`` when unset.

    Drop-in for pre-rebrand ``os.environ["HERMES_<name>"]`` reads.
    """
    if ("LOGOS_" + name) in os.environ:
        return os.environ["LOGOS_" + name]
    return os.environ["HERMES_" + name]  # raises KeyError if absent


def logos_env_delete(name: str) -> None:
    """Clear both ``LOGOS_<name>`` and legacy ``HERMES_<name>`` if present."""
    os.environ.pop("LOGOS_" + name, None)
    os.environ.pop("HERMES_" + name, None)


def get_subprocess_home() -> str | None:
    """Return a per-profile HOME directory for subprocesses, or None.

    When the Logos home has a ``home/`` subdirectory on disk, subprocesses
    should use it as ``HOME`` so system tools (git, ssh, gh, npm …) write
    their configs inside the Logos data directory instead of the OS-level
    ``/root`` or ``~/``.  This provides:

    * **Docker persistence** — tool configs land inside the persistent volume.
    * **Profile isolation** — each profile gets its own git identity, SSH
      keys, gh tokens, etc.

    The Python process's own ``os.environ["HOME"]`` and ``Path.home()`` are
    **never** modified — only subprocess environments should inject this value.
    Activation is directory-based: if the ``home/`` subdirectory doesn't
    exist, returns ``None`` and behavior is unchanged.
    """
    home = os.getenv("LOGOS_HOME") or os.getenv("HERMES_HOME")
    if not home:
        return None
    profile_home = os.path.join(home, "home")
    if os.path.isdir(profile_home):
        return profile_home
    return None


VALID_REASONING_EFFORTS = ("minimal", "low", "medium", "high", "xhigh")


def parse_reasoning_effort(effort: str) -> dict | None:
    """Parse a reasoning effort level into a config dict.

    Valid levels: "none", "minimal", "low", "medium", "high", "xhigh".
    Returns None when the input is empty or unrecognized (caller uses default).
    Returns {"enabled": False} for "none".
    Returns {"enabled": True, "effort": <level>} for valid effort levels.
    """
    if not effort or not effort.strip():
        return None
    effort = effort.strip().lower()
    if effort == "none":
        return {"enabled": False}
    if effort in VALID_REASONING_EFFORTS:
        return {"enabled": True, "effort": effort}
    return None


def is_termux() -> bool:
    """Return True when running inside a Termux (Android) environment.

    Checks ``TERMUX_VERSION`` (set by Termux) or the Termux-specific
    ``PREFIX`` path.  Import-safe — no heavy deps.
    """
    prefix = os.getenv("PREFIX", "")
    return bool(os.getenv("TERMUX_VERSION") or "com.termux/files/usr" in prefix)


_wsl_detected: bool | None = None


def is_wsl() -> bool:
    """Return True when running inside WSL (Windows Subsystem for Linux).

    Checks ``/proc/version`` for the ``microsoft`` marker that both WSL1
    and WSL2 inject.  Result is cached for the process lifetime.
    Import-safe — no heavy deps.
    """
    global _wsl_detected
    if _wsl_detected is not None:
        return _wsl_detected
    try:
        with open("/proc/version", "r") as f:
            _wsl_detected = "microsoft" in f.read().lower()
    except Exception:
        _wsl_detected = False
    return _wsl_detected


_container_detected: bool | None = None


def is_container() -> bool:
    """Return True when running inside a Docker/Podman container.

    Checks ``/.dockerenv`` (Docker), ``/run/.containerenv`` (Podman),
    and ``/proc/1/cgroup`` for container runtime markers.  Result is
    cached for the process lifetime.  Import-safe — no heavy deps.
    """
    global _container_detected
    if _container_detected is not None:
        return _container_detected
    if os.path.exists("/.dockerenv"):
        _container_detected = True
        return True
    if os.path.exists("/run/.containerenv"):
        _container_detected = True
        return True
    try:
        with open("/proc/1/cgroup", "r") as f:
            cgroup = f.read()
            if "docker" in cgroup or "podman" in cgroup or "/lxc/" in cgroup:
                _container_detected = True
                return True
    except OSError:
        pass
    _container_detected = False
    return False


# ─── Well-Known Paths ─────────────────────────────────────────────────────────


def get_config_path() -> Path:
    """Return the path to ``config.yaml`` under HERMES_HOME.

    Replaces the ``get_logos_home() / "config.yaml"`` pattern repeated
    in 7+ files (skill_utils.py, logos_logging.py, logos_time.py, etc.).
    """
    return get_logos_home() / "config.yaml"


def get_skills_dir() -> Path:
    """Return the path to the skills directory under HERMES_HOME."""
    return get_logos_home() / "skills"



def get_env_path() -> Path:
    """Return the path to the ``.env`` file under HERMES_HOME."""
    return get_logos_home() / ".env"


# ─── Network Preferences ─────────────────────────────────────────────────────


def apply_ipv4_preference(force: bool = False) -> None:
    """Monkey-patch ``socket.getaddrinfo`` to prefer IPv4 connections.

    On servers with broken or unreachable IPv6, Python tries AAAA records
    first and hangs for the full TCP timeout before falling back to IPv4.
    This affects httpx, requests, urllib, the OpenAI SDK — everything that
    uses ``socket.getaddrinfo``.

    When *force* is True, patches ``getaddrinfo`` so that calls with
    ``family=AF_UNSPEC`` (the default) resolve as ``AF_INET`` instead,
    skipping IPv6 entirely.  If no A record exists, falls back to the
    original unfiltered resolution so pure-IPv6 hosts still work.

    Safe to call multiple times — only patches once.
    Set ``network.force_ipv4: true`` in ``config.yaml`` to enable.
    """
    if not force:
        return

    import socket

    # Guard against double-patching
    if getattr(socket.getaddrinfo, "_logos_ipv4_patched", False):
        return

    _original_getaddrinfo = socket.getaddrinfo

    def _ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        if family == 0:  # AF_UNSPEC — caller didn't request a specific family
            try:
                return _original_getaddrinfo(
                    host, port, socket.AF_INET, type, proto, flags
                )
            except socket.gaierror:
                # No A record — fall back to full resolution (pure-IPv6 hosts)
                return _original_getaddrinfo(host, port, family, type, proto, flags)
        return _original_getaddrinfo(host, port, family, type, proto, flags)

    _ipv4_getaddrinfo._logos_ipv4_patched = True  # type: ignore[attr-defined]
    socket.getaddrinfo = _ipv4_getaddrinfo  # type: ignore[assignment]


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODELS_URL = f"{OPENROUTER_BASE_URL}/models"

AI_GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh/v1"
