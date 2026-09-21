#!/usr/bin/env python3
"""A/B benchmark: vLLM prefix-cache stability under deferred-tool churn.

WHY
---
With the Qwen chat template the tools JSON renders as the FIRST block of the
prompt, ahead of the system message and the conversation. Any change to the
tools array mid-session (deferred-tool promotion/demotion) therefore busts
the vLLM prefix cache for the ENTIRE conversation: the next request re-prefills
system + history from scratch.

The old default (demote_after_turns=5) demotes an idle promoted tool after 5
unused API rounds and re-promotes it on next use — every cycle re-prefills the
whole conversation. The T1 fix (demote_after_turns=0) makes the tools array
append-only per session: each deferred tool promotes exactly once, then the
prefix is stable.

ARMS
----
A = current behavior: demote after 5 idle rounds (churn pattern below)
B = T1: no demotion (append-only)

Both arms use the REAL exx system prompt (latest long one from state.db) and
the REAL essential tool schemas (get_selective_tool_definitions), plus a
synthetic multi-turn agent conversation grown to ~90K input tokens so the
re-prefill cost on a bust is realistic.

METRICS per API call
--------------------
- TTFT: time to first streamed chunk (proxy for prefill cost)
- prefix-cache hit rate from vLLM /metrics deltas
  (vllm:prefix_cache_hits_total / vllm:prefix_cache_queries_total)
- prompt_tokens from the streamed usage chunk

CHURN PATTERN (16 API calls, 3 deferred tools T1/T2/T3 promoted on call 1)
--------------------------------------------------------------------------
call  1: promote T1,T2,T3 (use all three)
calls 2-9: only T1 used; T2,T3 idle
call  6: arm A demotes T2,T3 (api_call_count - 1 >= 5)  -> bust on call 6
call 10: T2 used again; arm A re-promotes it             -> bust on call 10
call 15: T3 used again; arm A re-promotes it             -> bust on call 15
arm B: tools change only on call 1 (promotion), stable after.

Usage: /home/exx/logos/venv/bin/python scripts/bench_prefix_cache.py
Output: table to stdout + JSON at ../bench-prefix-cache/report-<ts>.json
"""

import json
import os
import re
import sqlite3
import sys
import time
import urllib.request

# ── config ───────────────────────────────────────────────────────────────────
VLLM = os.environ.get("BENCH_VLLM", "http://127.0.0.1:8000")
MODEL = os.environ.get("BENCH_MODEL", "Qwen3.8-27B")
STATE_DB = os.environ.get("BENCH_STATE_DB", "/home/exx/.hermes/state.db")
N_CALLS = 16
TARGET_TOKENS = 90_000  # conversation size at the end (rough, chars//4)
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "bench-prefix-cache")
os.makedirs(OUT_DIR, exist_ok=True)

# ── real inputs ──────────────────────────────────────────────────────────────

def load_real_system_prompt() -> str:
    db = sqlite3.connect(STATE_DB)
    row = db.execute(
        "SELECT system_prompt FROM sessions "
        "WHERE system_prompt IS NOT NULL AND length(system_prompt) > 10000 "
        "ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    if not row:
        raise SystemExit("no usable system prompt found in " + STATE_DB)
    return row[0]


def load_essential_tools():
    import tools  # noqa: F401  (registration side effects)
    from model_tools import get_selective_tool_definitions
    return get_selective_tool_definitions(
        enabled_toolsets=None, disabled_toolsets=None, quiet_mode=True)


def load_deferred_tools(essential_names, n=3):
    """Pick n real deferred tool schemas (non-essential)."""
    import tools  # noqa: F401
    from model_tools import get_tool_definitions
    all_tools = get_tool_definitions(
        enabled_toolsets=None, disabled_toolsets=None, quiet_mode=True)
    out = []
    for t in all_tools:
        name = t["function"]["name"]
        if name not in essential_names and name not in [x["function"]["name"] for x in out]:
            out.append(t)
        if len(out) >= n:
            break
    return out


# ── conversation synthesis ───────────────────────────────────────────────────

def make_tool_result(i: int) -> str:
    """A realistic-ish chunk of tool output (~4.5K tokens of filler)."""
    base = (
        f"[tool result {i}] Retrieved file section. Contents: "
        "line 1 of 500: def process_batch(items): # iterate over records\n"
        "    for idx, item in enumerate(items):\n"
        "        if item.status == 'pending':\n"
        "            log.debug('processing %d', idx)\n"
        "        results.append(transform(item))\n"
    )
    # repeat to ~18K chars (~4.5K tokens)
    return (base * 45)[:18_000]


def synthesize_turns(n_calls: int, target_tokens: int):
    """Build the incremental conversation: list of (user_msg, [tool rounds]).

    Each API call k receives messages[0..k]. Call k appends one user message
    plus one assistant tool-call + tool result round (except the final call,
    which gets a plain user message and the assistant answers directly).
    """
    user_msgs = []
    # total chars budget for conversation (system+tools excluded)
    budget_chars = target_tokens * 4
    per_call = int(budget_chars / n_calls)
    for k in range(n_calls):
        if k < n_calls - 1:
            # one user msg + one tool round, split the budget
            user_msgs.append((
                f"Turn {k+1}: please investigate the logs and check the "
                f"deployment state. (context filler token {k}) " + "ctx " * (per_call // 400),
                True,
            ))
        else:
            user_msgs.append(
                (f"Turn {k+1}: summarize what you found. " + "ctx " * (per_call // 400),
                 False),
            )
    return user_msgs


# ── vLLM client ──────────────────────────────────────────────────────────────

def metrics_counters():
    with urllib.request.urlopen(VLLM + "/metrics", timeout=15) as r:
        text = r.read().decode()
    q = re.search(r"vllm:prefix_cache_queries_total\{[^}]*\} ([0-9.e+]+)", text)
    h = re.search(r"vllm:prefix_cache_hits_total\{[^}]*\} ([0-9.e+]+)", text)
    return (float(q.group(1)) if q else None, float(h.group(1)) if h else None)


def chat_stream(messages, tools, max_tokens=16):
    body = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(
        VLLM + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    ttft = None
    usage = None
    with urllib.request.urlopen(req, timeout=300) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data:"):
                continue
            d = line[5:].strip()
            if d == "[DONE]":
                break
            if ttft is None:
                ttft = time.time() - t0
            try:
                j = json.loads(d)
            except json.JSONDecodeError:
                continue
            if j.get("usage"):
                usage = j["usage"]
    return ttft, usage


# ── arm state machine ────────────────────────────────────────────────────────

def tools_for_call(arm: str, call: int, essential, deferred):
    """Return the tools list the agent would send on `call` (1-based).

    call 1: promote all three deferred tools (appended after essential).
    arm A: demote deferred tools unused for >=5 rounds (removed from list);
           re-promote on use (appended again).
    arm B: never demote.
    """
    t1, t2, t3 = deferred
    USE = {
        1: (1, 1, 1), 2: (1, 0, 0), 3: (1, 0, 0), 4: (1, 0, 0),
        5: (1, 0, 0), 6: (1, 0, 0), 7: (1, 0, 0), 8: (1, 0, 0),
        9: (1, 0, 0), 10: (1, 1, 0), 11: (1, 0, 0), 12: (1, 0, 0),
        13: (1, 0, 0), 14: (1, 0, 0), 15: (1, 0, 1), 16: (1, 0, 0),
    }
    last_used = {1: 1, 2: 1, 3: 1}  # promotion round
    tools = list(essential)
    # replay rounds 1..call to derive arm-A list state
    present = {1: False, 2: False, 3: False}
    for r in range(1, call + 1):
        u = USE[r]
        if r == 1:
            for i, t in ((1, t1), (2, t2), (3, t3)):
                tools = [x for x in tools if x["function"]["name"] != t["function"]["name"]]
            for i, t in ((1, t1), (2, t2), (3, t3)):
                tools.append(t)
                present[i] = True
        else:
            if arm == "A":
                # demote at top of round r if unused for >= 5 rounds
                # (matches run_agent._maybe_demote_tools: api_call_count - last >= n)
                for i in (2, 3):
                    if present[i] and r - last_used[i] >= 5:
                        t = (t1, t2, t3)[i - 1]
                        tools = [x for x in tools if x["function"]["name"] != t["function"]["name"]]
                        present[i] = False
            # promote on use
            for i in (1, 2, 3):
                if u[i - 1] and not present[i]:
                    t = (t1, t2, t3)[i - 1]
                    tools.append(t)
                    present[i] = True
                    last_used[i] = r
                elif u[i - 1] and present[i]:
                    last_used[i] = r
    return tools


# ── main ─────────────────────────────────────────────────────────────────────

def run_arm(arm, system_prompt, essential, deferred, user_msgs):
    print(f"\n=== ARM {arm} ({'demote=5 churn' if arm=='A' else 'append-only (T1)'}) ===")
    # Per-arm salt: isolates the two arms' KV prefixes so each arm's call 1
    # is a clean cold start (and the arms can't accidentally share a hit).
    system_prompt = system_prompt + f" [bench-arm-{arm}]"
    rows = []
    history = []  # messages after the system msg
    for call in range(1, N_CALLS + 1):
        tools = tools_for_call(arm, call, essential, deferred)
        user_msg, has_tool = user_msgs[call - 1]
        # append the new user message
        history.append({"role": "user", "content": user_msg})
        q0, h0 = metrics_counters()
        t0 = time.time()
        ttft, usage = chat_stream(
            [{"role": "system", "content": system_prompt}] + history, tools)
        wall = time.time() - t0
        q1, h1 = metrics_counters()
        prompt_tokens = usage.get("prompt_tokens") if usage else None
        hit_pct = None
        if q1 is not None and h1 is not None and q0 is not None and h0 is not None:
            dq, dh = q1 - q0, h1 - h0
            if dq > 0:
                hit_pct = 100.0 * dh / dq
        rows.append({
            "call": call,
            "tools_n": len(tools),
            "ttft_s": round(ttft, 3) if ttft else None,
            "wall_s": round(wall, 2),
            "prompt_tokens": prompt_tokens,
            "hit_pct": round(hit_pct, 1) if hit_pct is not None else None,
        })
        print(f"  call {call:2d}  tools={len(tools):2d}  TTFT={ttft:6.2f}s  "
              f"prompt={prompt_tokens}  hit={hit_pct if hit_pct is None else round(hit_pct,1)}%", flush=True)
        # append assistant tool-call + tool result so the next call grows
        if has_tool:
            history.append({
                "role": "assistant", "content": None,
                "tool_calls": [{
                    "id": f"call_{arm}_{call}", "type": "function",
                    "function": {"name": "terminal",
                                 "arguments": json.dumps({"command": f"inspect {call}"})},
                }],
            })
            history.append({
                "role": "tool", "tool_call_id": f"call_{arm}_{call}",
                "content": make_tool_result(call),
            })
    return rows


def main():
    print("Loading real system prompt + tool schemas ...")
    system_prompt = load_real_system_prompt()
    essential = load_essential_tools()
    ess_names = {t["function"]["name"] for t in essential}
    deferred = load_deferred_tools(ess_names, n=3)
    print(f"  system prompt: {len(system_prompt)} chars (~{len(system_prompt)//4} tok)")
    print(f"  essential tools: {len(essential)}  deferred picked: "
          f"{[t['function']['name'] for t in deferred]}")
    user_msgs = synthesize_turns(N_CALLS, TARGET_TOKENS)

    results = {}
    for arm in ("A", "B"):
        results[arm] = run_arm(arm, system_prompt, essential, deferred, user_msgs)

    # summary
    print("\n=== SUMMARY ===")
    for arm in ("A", "B"):
        rows = results[arm]
        busts = [r for r in rows if r["hit_pct"] is not None and r["hit_pct"] < 50 and r["call"] > 1]
        warm = [r for r in rows if r["hit_pct"] is not None and r["hit_pct"] >= 50]
        ttfts = [r["ttft_s"] for r in rows if r["ttft_s"]]
        print(f"arm {arm}: avg TTFT {sum(ttfts)/len(ttfts):6.2f}s  "
              f"max TTFT {max(ttfts):6.2f}s  "
              f"bust calls (hit<50%): {[r['call'] for r in busts]}  "
              f"avg hit (warm) {sum(r['hit_pct'] for r in warm)/max(1,len(warm)):5.1f}%")

    ts = time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join(OUT_DIR, f"report-{ts}.json")
    with open(out, "w") as f:
        json.dump({"vllm": VLLM, "model": MODEL, "n_calls": N_CALLS,
                   "system_prompt_chars": len(system_prompt),
                   "essential_tools": len(essential),
                   "deferred_tools": [t["function"]["name"] for t in deferred],
                   "results": results}, f, indent=2)
    print(f"\nreport: {out}")


if __name__ == "__main__":
    main()
