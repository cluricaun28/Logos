"""
Status command for hermes CLI.

Shows the status of all Logos components.
"""
from __future__ import annotations

import os
import sys
import subprocess  # noqa: F401 — re-exported for tests that monkeypatch status.subprocess to guard against regressions
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

from logos_cli.auth import AuthError, resolve_provider
from logos_cli.colors import Colors, color
from logos_cli.config import get_env_path, get_env_value, get_logos_home, load_config
from logos_cli.models import provider_label
from logos_cli.nous_subscription import get_nous_subscription_features
from logos_cli.runtime_provider import resolve_requested_provider
from logos_constants import OPENROUTER_MODELS_URL
from tools.tool_backend_helpers import managed_nous_tools_enabled

def check_mark(ok: bool) -> str:
    if ok:
        return color("✓", Colors.GREEN)
    return color("✗", Colors.RED)

def redact_key(key: str) -> str:
    """Redact an API key for display."""
    if not key:
        return "(not set)"
    if len(key) < 12:
        return "***"
    return key[:4] + "..." + key[-4:]


def _format_iso_timestamp(value) -> str:
    """Format ISO timestamps for status output, converting to local timezone."""
    if not value or not isinstance(value, str):
        return "(unknown)"
    from datetime import datetime, timezone
    text = value.strip()
    if not text:
        return "(unknown)"
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
    except (AttributeError, KeyError, OSError, RuntimeError, TypeError):
        return value
    return parsed.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def _configured_model_label(config: dict) -> str:
    """Return the configured default model from config.yaml."""
    model_cfg = config.get("model")
    if isinstance(model_cfg, dict):
        model = (model_cfg.get("default") or model_cfg.get("name") or "").strip()
    elif isinstance(model_cfg, str):
        model = model_cfg.strip()
    else:
        model = ""
    return model or "(not set)"


def _effective_provider_label() -> str:
    """Return the provider label matching current CLI runtime resolution."""
    requested = resolve_requested_provider()
    try:
        effective = resolve_provider(requested)
    except AuthError:
        effective = requested or "auto"

    if effective == "openrouter" and get_env_value("OPENAI_BASE_URL"):
        effective = "custom"

    return provider_label(effective)


from logos_constants import is_termux as _is_termux


def show_status(args):
    from logos_constants import get_logos_home
    """Show status of all Logos components."""
    show_all = getattr(args, 'all', False)
    deep = getattr(args, 'deep', False)
    
    print()
    print(color("┌─────────────────────────────────────────────────────────┐", Colors.CYAN))
    print(color("│                 ⚕ Logos Status                  │", Colors.CYAN))
    print(color("└─────────────────────────────────────────────────────────┘", Colors.CYAN))
    
    # =========================================================================
    # Environment
    # =========================================================================
    print()
    print(color("◆ Environment", Colors.CYAN, Colors.BOLD))
    print(f"  Project:      {PROJECT_ROOT}")
    print(f"  Python:       {sys.version.split()[0]}")
    
    env_path = get_env_path()
    print(f"  .env file:    {check_mark(env_path.exists())} {'exists' if env_path.exists() else 'not found'}")

    try:
        config = load_config()
    except (AttributeError, KeyError, OSError, RuntimeError, TypeError):
        config = {}

    print(f"  Model:        {_configured_model_label(config)}")
    print(f"  Provider:     {_effective_provider_label()}")
    
    # =========================================================================
    # API Keys
    # =========================================================================
    print()
    print(color("◆ API Keys", Colors.CYAN, Colors.BOLD))
    
    keys = {
        "OpenRouter": "OPENROUTER_API_KEY",
        "OpenAI": "OPENAI_API_KEY",
        "Z.AI/GLM": "GLM_API_KEY",
        "Kimi": "KIMI_API_KEY",
        "StepFun Step Plan": "STEPFUN_API_KEY",
        "MiniMax": "MINIMAX_API_KEY",
        "MiniMax-CN": "MINIMAX_CN_API_KEY",
        "Firecrawl": "FIRECRAWL_API_KEY",
        "Tavily": "TAVILY_API_KEY",
        "Browser Use": "BROWSER_USE_API_KEY",  # Optional — local browser works without this
        "Browserbase": "BROWSERBASE_API_KEY",  # Optional — direct credentials only
        "FAL": "FAL_KEY",
        "Tinker": "TINKER_API_KEY",
        "WandB": "WANDB_API_KEY",
        "ElevenLabs": "ELEVENLABS_API_KEY",
        "GitHub": "GITHUB_TOKEN",
    }
    
    for name, env_var in keys.items():
        value = get_env_value(env_var) or ""
        has_key = bool(value)
        display = redact_key(value) if not show_all else value
        print(f"  {name:<12}  {check_mark(has_key)} {display}")

    from logos_cli.auth import get_anthropic_key
    anthropic_value = get_anthropic_key()
    anthropic_display = redact_key(anthropic_value) if not show_all else anthropic_value
    print(f"  {'Anthropic':<12}  {check_mark(bool(anthropic_value))} {anthropic_display}")

    # =========================================================================
    # Auth Providers (OAuth)
    # =========================================================================
    print()
    print(color("◆ Auth Providers", Colors.CYAN, Colors.BOLD))

    try:
        from logos_cli.auth import get_nous_auth_status, get_codex_auth_status, get_qwen_auth_status
        nous_status = get_nous_auth_status()
        codex_status = get_codex_auth_status()
        qwen_status = get_qwen_auth_status()
    except (ImportError, ModuleNotFoundError):
        nous_status = {}
        codex_status = {}
        qwen_status = {}

    nous_logged_in = bool(nous_status.get("logged_in"))
    nous_error = nous_status.get("error")
    nous_label = "logged in" if nous_logged_in else "not logged in (run: logos auth add nous --type oauth)"
    print(
        f"  {'Nous Portal':<12}  {check_mark(nous_logged_in)} "
        f"{nous_label}"
    )
    portal_url = nous_status.get("portal_base_url") or "(unknown)"
    access_exp = _format_iso_timestamp(nous_status.get("access_expires_at"))
    key_exp = _format_iso_timestamp(nous_status.get("agent_key_expires_at"))
    refresh_label = "yes" if nous_status.get("has_refresh_token") else "no"
    if nous_logged_in or portal_url != "(unknown)" or nous_error:
        print(f"    Portal URL: {portal_url}")
    if nous_logged_in or nous_status.get("access_expires_at"):
        print(f"    Access exp: {access_exp}")
    if nous_logged_in or nous_status.get("agent_key_expires_at"):
        print(f"    Key exp:    {key_exp}")
    if nous_logged_in or nous_status.get("has_refresh_token"):
        print(f"    Refresh:    {refresh_label}")
    if nous_error and not nous_logged_in:
        print(f"    Error:      {nous_error}")

    codex_logged_in = bool(codex_status.get("logged_in"))
    print(
        f"  {'OpenAI Codex':<12}  {check_mark(codex_logged_in)} "
        f"{'logged in' if codex_logged_in else 'not logged in (run: logos model)'}"
    )
    codex_auth_file = codex_status.get("auth_store")
    if codex_auth_file:
        print(f"    Auth file:  {codex_auth_file}")
    codex_last_refresh = _format_iso_timestamp(codex_status.get("last_refresh"))
    if codex_status.get("last_refresh"):
        print(f"    Refreshed:  {codex_last_refresh}")
    if codex_status.get("error") and not codex_logged_in:
        print(f"    Error:      {codex_status.get('error')}")

    qwen_logged_in = bool(qwen_status.get("logged_in"))
    print(
        f"  {'Qwen OAuth':<12}  {check_mark(qwen_logged_in)} "
        f"{'logged in' if qwen_logged_in else 'not logged in (run: qwen auth qwen-oauth)'}"
    )
    qwen_auth_file = qwen_status.get("auth_file")
    if qwen_auth_file:
        print(f"    Auth file:  {qwen_auth_file}")
    qwen_exp = qwen_status.get("expires_at_ms")
    if qwen_exp:
        from datetime import datetime, timezone
        print(f"    Access exp: {datetime.fromtimestamp(int(qwen_exp) / 1000, tz=timezone.utc).isoformat()}")
    if qwen_status.get("error") and not qwen_logged_in:
        print(f"    Error:      {qwen_status.get('error')}")

    # =========================================================================
    # Nous Subscription Features
    # =========================================================================
    if managed_nous_tools_enabled():
        features = get_nous_subscription_features(config)
        print()
        print(color("◆ Nous Tool Gateway", Colors.CYAN, Colors.BOLD))
        if not features.nous_auth_present:
            print("  Nous Portal   ✗ not logged in")
        else:
            print("  Nous Portal   ✓ managed tools available")
        for feature in features.items():
            if feature.managed_by_nous:
                state = "active via Nous subscription"
            elif feature.active:
                current = feature.current_provider or "configured provider"
                state = f"active via {current}"
            elif feature.included_by_default and features.nous_auth_present:
                state = "included by subscription, not currently selected"
            else:
                state = "not configured"
            print(f"  {feature.label:<15} {check_mark(feature.available or feature.active or feature.managed_by_nous)} {state}")
    elif nous_logged_in:
        # Logged into Nous but on the free tier — show upgrade nudge
        print()
        print(color("◆ Nous Tool Gateway", Colors.CYAN, Colors.BOLD))
        print("  Your free-tier Nous account does not include Tool Gateway access.")
        print("  Upgrade your subscription to unlock managed web, image, TTS, and browser tools.")
        try:
            portal_url = nous_status.get("portal_base_url", "").rstrip("/")
            if portal_url:
                print(f"  Upgrade: {portal_url}")
        except (AttributeError, KeyError, TypeError):
            pass

    # =========================================================================
    # API-Key Providers
    # =========================================================================
    print()
    print(color("◆ API-Key Providers", Colors.CYAN, Colors.BOLD))

    apikey_providers = {
        "Z.AI / GLM":       ("GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY"),
        "Kimi / Moonshot":  ("KIMI_API_KEY",),
        "StepFun Step Plan": ("STEPFUN_API_KEY",),
        "MiniMax":          ("MINIMAX_API_KEY",),
        "MiniMax (China)":  ("MINIMAX_CN_API_KEY",),
    }
    for pname, env_vars in apikey_providers.items():
        key_val = ""
        for ev in env_vars:
            key_val = get_env_value(ev) or ""
            if key_val:
                break
        configured = bool(key_val)
        label = "configured" if configured else "not configured (run: logos model)"
        print(f"  {pname:<16} {check_mark(configured)} {label}")

    # LM Studio reachability — only probe when it's the active provider so
    # users with foreign configs don't see noise. Auth rejection vs. silent
    # empty list is the most common LM Studio support case.
    if _effective_provider_label() == "LM Studio":
        from logos_cli.models import probe_lmstudio_models
        model_cfg = config.get("model")
        base = (model_cfg.get("base_url") if isinstance(model_cfg, dict) else None) or get_env_value("LM_BASE_URL") or "http://127.0.0.1:1234/v1"
        try:
            models = probe_lmstudio_models(api_key=get_env_value("LM_API_KEY") or "", base_url=base, timeout=1.5)
            if models is None:
                ok, msg = False, f"unreachable at {base}"
            else:
                ok, msg = True, f"reachable ({len(models)} model(s)) at {base}"
        except AuthError:
            ok, msg = False, "auth rejected — set LM_API_KEY"
        print(f"  {'LM Studio':<16} {check_mark(ok)} {msg}")

    # =========================================================================
    # Terminal Configuration
    # =========================================================================
    print()
    print(color("◆ Terminal Backend", Colors.CYAN, Colors.BOLD))
    
    terminal_env = os.getenv("TERMINAL_ENV", "")
    if not terminal_env:
        # Fall back to config file value when env var isn't set
        # (hermes status doesn't go through cli.py's config loading)
        try:
            _cfg = load_config()
            terminal_env = _cfg.get("terminal", {}).get("backend", "local")
        except (AttributeError, KeyError, TypeError):
            terminal_env = "local"
    print(f"  Backend:      {terminal_env}")
    
    if terminal_env == "ssh":
        ssh_host = os.getenv("TERMINAL_SSH_HOST", "")
        ssh_user = os.getenv("TERMINAL_SSH_USER", "")
        print(f"  SSH Host:     {ssh_host or '(not set)'}")
        print(f"  SSH User:     {ssh_user or '(not set)'}")
    elif terminal_env == "docker":
        docker_image = os.getenv("TERMINAL_DOCKER_IMAGE", "python:3.11-slim")
        print(f"  Docker Image: {docker_image}")
    
    sudo_password = os.getenv("SUDO_PASSWORD", "")
    print(f"  Sudo:         {check_mark(bool(sudo_password))} {'enabled' if sudo_password else 'disabled'}")
    
    # =========================================================================
    # Messaging Platforms
    # =========================================================================
    print()
    print(color("◆ Messaging Platforms", Colors.CYAN, Colors.BOLD))
    
    platforms = {
        "Telegram": ("TELEGRAM_BOT_TOKEN", "TELEGRAM_HOME_CHANNEL"),
    }
    
    for name, (token_var, home_var) in platforms.items():
        token = os.getenv(token_var, "")
        has_token = bool(token)
        
        home_channel = ""
        if home_var:
            home_channel = os.getenv(home_var, "")
        
        status = "configured" if has_token else "not configured"
        if home_channel:
            status += f" (home: {home_channel})"
        
        print(f"  {name:<12}  {check_mark(has_token)} {status}")
    
    # =========================================================================
    # Gateway Status
    # =========================================================================
    print()
    print(color("◆ Gateway Service", Colors.CYAN, Colors.BOLD))

    try:
        from logos_cli.gateway import get_gateway_runtime_snapshot, _format_gateway_pids

        snapshot = get_gateway_runtime_snapshot()
        is_running = snapshot.running
        print(f"  Status:       {check_mark(is_running)} {'running' if is_running else 'stopped'}")
        print(f"  Manager:      {snapshot.manager}")
        if snapshot.gateway_pids:
            print(f"  PID(s):       {_format_gateway_pids(snapshot.gateway_pids)}")
        if snapshot.has_process_service_mismatch:
            print("  Service:      installed but not managing the current running gateway")
        elif _is_termux() and not snapshot.gateway_pids:
            print("  Start with:   logos gateway")
            print("  Note:         Android may stop background jobs when Termux is suspended")
        elif snapshot.service_installed and not snapshot.service_running:
            print("  Service:      installed but stopped")
    except (ImportError, ModuleNotFoundError):
        if _is_termux():
            print(f"  Status:       {color('unknown', Colors.DIM)}")
            print("  Manager:      Termux / manual process")
        elif sys.platform.startswith('linux'):
            print(f"  Status:       {color('unknown', Colors.DIM)}")
            print("  Manager:      systemd/manual")
        elif sys.platform == 'darwin':
            print(f"  Status:       {color('unknown', Colors.DIM)}")
            print("  Manager:      launchd")
        else:
            print(f"  Status:       {color('N/A', Colors.DIM)}")
            print("  Manager:      (not supported on this platform)")
    
    # =========================================================================
    # Cron Jobs
    # =========================================================================
    print()
    print(color("◆ Scheduled Jobs", Colors.CYAN, Colors.BOLD))
    
    jobs_file = get_logos_home() / "cron" / "jobs.json"
    if jobs_file.exists():
        import json
        try:
            with open(jobs_file, encoding="utf-8") as f:
                data = json.load(f)
                jobs = data.get("jobs", [])
                enabled_jobs = [j for j in jobs if j.get("enabled", True)]
                print(f"  Jobs:         {len(enabled_jobs)} active, {len(jobs)} total")
        except (AttributeError, json.JSONDecodeError, KeyError, OSError, PermissionError, TypeError, ValueError):
            print("  Jobs:         (error reading jobs file)")
    else:
        print("  Jobs:         0")
    
    # =========================================================================
    # Sessions
    # =========================================================================
    print()
    print(color("◆ Sessions", Colors.CYAN, Colors.BOLD))
    
    sessions_file = get_logos_home() / "sessions" / "sessions.json"
    if sessions_file.exists():
        import json
        try:
            with open(sessions_file, encoding="utf-8") as f:
                data = json.load(f)
                print(f"  Active:       {len(data)} session(s)")
        except (json.JSONDecodeError, OSError, PermissionError, ValueError):
            print("  Active:       (error reading sessions file)")
    else:
        print("  Active:       0")
    
    # =========================================================================
    # Deep checks
    # =========================================================================
    if deep:
        print()
        print(color("◆ Deep Checks", Colors.CYAN, Colors.BOLD))
        
        # Check OpenRouter connectivity
        openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
        if openrouter_key:
            try:
                import httpx
                response = httpx.get(
                    OPENROUTER_MODELS_URL,
                    headers={"Authorization": f"Bearer {openrouter_key}"},
                    timeout=10
                )
                ok = response.status_code == 200
                print(f"  OpenRouter:   {check_mark(ok)} {'reachable' if ok else f'error ({response.status_code})'}")
            except (AttributeError, ConnectionError, ImportError, KeyError, ModuleNotFoundError, OSError, TimeoutError, TypeError) as e:
                print(f"  OpenRouter:   {check_mark(False)} error: {e}")
        
        # Check gateway port
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('127.0.0.1', 18789))
            sock.close()
            # Port in use = gateway likely running
            port_in_use = result == 0
            # This is informational, not necessarily bad
            print(f"  Port 18789:   {'in use' if port_in_use else 'available'}")
        except OSError:
            pass
    
    print()
    print(color("─" * 60, Colors.DIM))
    print(color("  Run 'logos doctor' for detailed diagnostics", Colors.DIM))
    print(color("  Run 'logos setup' to configure", Colors.DIM))
    print()
