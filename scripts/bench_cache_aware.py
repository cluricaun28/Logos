#!/usr/bin/env python3
"""A/B benchmark: vLLM prefix-cache stability across ARCHIVE events (P1b).

WHY
---
Archive events (context-engine compaction) rewrite the request prefix:
  1. run_agent._archive_context rebuilds the system prompt — the
     [Current Time] minute header re-stamp changes the FIRST block of the
     prompt (tools JSON renders first in the Qwen template) → the ENTIRE
     conversation re-prefills on the next call.
  2. The [Conversation State] map is strip+prepended onto the last
     assistant message on every archive → mid-tail rewrite on anchor
     moves / map drift.
  3. The semantic prune rewrites mid-history.

P1b (cache_aware=True, 2026-09-21): system-prompt pin (a re-stamp-only
rebuild keeps the old bytes unless the date rolls over), sticky state
map (an identical map anchors in place → zero rewrite), savings gate
(low-savings archives skipped entirely + cooldown band).

ARMS
----
A = legacy (cache_aware=False): exact pre-9/21 archive behavior.
B = cache-aware (cache_aware=True).
Both arms run the REAL SemanticVectorContextEngine (real MiniLM
embeddings) on the SAME synthetic conversation. The only differences are
the P1b mechanisms + the system-prompt pin (reproduced here with
AIAgent._pin_system_prompt_bytes, exactly as _archive_context applies it).

SCENARIO (19 calls/arm)
-----------------------
calls 1-11: build conversation = 9 active-topic turns + 2 dormant-topic
            turns (~25.6K conversation tokens; engine view).
E0 (after call 11): archive, active_tail_turns=0 — prunes the 2 dormant
            turns (~2.2K savings, clears B's 1.5K savings gate).
call 12: measure post-E0.  A: system re-stamp → full bust.
                      B: pinned system + identical prune/map → no bust.
call 13: 1 dormant-topic turn (~1.2K).
E1 (after call 13): archive, active_tail_turns=26 — prunes ~0.9K.
                      A: legacy (prune + map re-anchor + system re-stamp).
                      B: savings-gate SKIP (0.9K < 1.5K) + pinned system.
calls 14-17: 4 active turns (15, 18 are plain measurement turns).
E2 (after call 17): archive, active_tail_turns=6 — big roll-off (~30K).
                      Both arms prune (savings clears the gate);
                      A additionally re-stamps the system prompt.
call 18: measure.  A: hit ≈ tools block only. B: hit from first pruned byte.
call 19: stability check (both arms should be warm).

METRICS: per-call TTFT + prefix-cache hit % from vLLM /metrics deltas
(same method as scripts/bench_prefix_cache.py), plus the system-prompt
sha per call (proves the pin: B's hash is constant across events, A's
changes at every event).

Usage: /home/exx/logos/venv/bin/python scripts/bench_cache_aware.py
Env:   BENCH_VLLM, BENCH_MODEL, BENCH_STATE_DB, BENCH_EMBED
Output: table to stdout + bench-prefix-cache/report-cache-aware-<ts>.json
"""

import json
import os
import re
import sqlite3
import sys
import time
import urllib.request
from datetime import datetime

# ── config ───────────────────────────────────────────────────────────────────
VLLM = os.environ.get("BENCH_VLLM", "http://127.0.0.1:8000")
MODEL = os.environ.get("BENCH_MODEL", "Qwen3.8-27B")
STATE_DB = os.environ.get("BENCH_STATE_DB", "/home/exx/.hermes/state.db")
EMBED_PATH = os.environ.get(
    "BENCH_EMBED", "/home/exx/.hermes/models/embeddings/all-MiniLM-L6-v2")
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "bench-prefix-cache")
os.makedirs(OUT_DIR, exist_ok=True)

# Engine params (engine view = conversation-content tokens only).
ENG = dict(
    context_length=100000,
    threshold_percent=0.24,      # 24,000 tokens
    similarity_threshold=0.45,
    dormancy_decay=10,
    resolution_decay=40,
    protect_first_n=3,
    protect_last_n=6,
    task_aware=False,
    model_path=EMBED_PATH,
)

# call plan: call -> (topic, tool_result_chars, event_after)
# ~2.5K chars/token budgeting: 10400 chars ≈ 2.6K tokens per active turn,
# 4400 ≈ 1.1K / 4800 ≈ 1.2K per dormant turn.
PLAN = {
    1: ("active", 10400, None),
    2: ("dormant", 4400, None),
    3: ("active", 10400, None),
    4: ("active", 10400, None),
    5: ("dormant", 4400, None),
    6: ("active", 10400, None),
    7: ("active", 10400, None),
    8: ("active", 10400, None),
    9: ("active", 10400, None),
    10: ("active", 10400, None),
    11: ("active", 10400, "E0"),   # E0: tail=0  (dormant prune ~2.2K)
    13: ("dormant", 4800, "E1"),   # E1: tail=26 (prune ~0.9K → B gate skip)
    14: ("active", 10400, None),
    16: ("active", 10400, None),
    17: ("active", 10400, "E2"),   # E2: tail=6  (big roll-off ~30K)
}
EVENT_TAIL = {"E0": 0, "E1": 26, "E2": 6}
MEASURE_CALLS = {12, 15, 18, 19}   # plain user-message turns (no tool round)

BRIDGE = ("## Active Tasks (with retrieval pointers)\n"
          "- [bench: investigate deployment] (turns [1..n])")
TODO = "[Your active task list was preserved]\n- [x] bench turn"


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


def re_stamp(prompt: str) -> str:
    """Simulate _build_system_prompt's volatile-line re-stamp at archive
    time: fresh [Current Time] minute header + grown PM count."""
    now = datetime.now().astimezone().strftime(
        "%A, %B %d, %Y %-I:%M %p (%Z)")
    out = re.sub(r"^\[Current Time: .*?\]$",
                 f"[Current Time: {now}]", prompt, flags=re.M)
    out = re.sub(
        r"^\[Perpetual Context Memory: (\d+) messages across (\d+) sessions",
        lambda m: ("[Perpetual Context Memory: %d messages across %d sessions"
                   % (int(m.group(1)) + 137, int(m.group(2)) + 1)),
        out, flags=re.M)
    return out


# ── conversation synthesis ───────────────────────────────────────────────────

def make_turn(n: int, topic: str, target_chars: int):
    """One agent turn: user msg + assistant tool-call + tool result."""
    if topic == "active":
        user = (f"Turn {n}: check deployment state for service-{n} and "
                f"verify latency budgets. ")
        result_head = (f"deployment state check: service-{n} healthy, "
                       f"latency p50 38ms p99 172ms, replicas 3/3, "
                       f"error rate 0.02%, autoscaler steady, config "
                       f"revision r{n}. ")
    else:
        user = (f"Turn {n}: investigate the old parser memory leak "
                f"(ticket 4411) — heap snapshot follow-up. ")
        result_head = ("heap snapshot analysis: parser module leak "
                       "confirmed, gc roots traced to ring buffer, "
                       "dominator tree stable since fix 4411-b, "
                       "monitoring retained. ")
    body = (result_head + " " * 0) * 1
    reps = max(1, target_chars // len(body))
    result = (body * reps)[:target_chars]
    call_id = f"call_{n}"
    return (
        {"role": "user", "content": user},
        {"role": "assistant", "content": None,
         "tool_calls": [{
             "id": call_id, "type": "function",
             "function": {"name": "terminal",
                          "arguments": json.dumps(
                              {"command": f"inspect {topic} {n}"})},
         }]},
        {"role": "tool", "tool_call_id": call_id, "content": result},
    )


# ── vLLM client ──────────────────────────────────────────────────────────────

def metrics_counters():
    with urllib.request.urlopen(VLLM + "/metrics", timeout=15) as r:
        text = r.read().decode()
    q = re.search(r"vllm:prefix_cache_queries_total\{[^}]*\} ([0-9.e+]+)", text)
    h = re.search(r"vllm:prefix_cache_hits_total\{[^}]*\} ([0-9.e+]+)", text)
    return (float(q.group(1)) if q else None,
            float(h.group(1)) if h else None)


def chat_stream(messages, tools, max_tokens=16):
    body = {
        "model": MODEL, "messages": messages, "max_tokens": max_tokens,
        "temperature": 0, "stream": True,
        "stream_options": {"include_usage": True},
    }
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(
        VLLM + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
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


# ── engine factory ───────────────────────────────────────────────────────────

def make_engine(tail: int, cache_aware: bool, embed_model):
    from plugins.context_engine.semantic_vector import (
        SemanticVectorContextEngine)
    e = SemanticVectorContextEngine(
        **ENG, active_tail_turns=tail, cache_aware=cache_aware)
    e.update_model("bench", ENG["context_length"])
    e._embedding_engine = embed_model  # documented test-mock path
    return e


# ── arm runner ───────────────────────────────────────────────────────────────

def run_arm(arm, system_prompt, tools, embed_model):
    from run_agent import AIAgent
    print(f"\n=== ARM {arm} "
          f"({'legacy archive' if arm == 'A' else 'cache-aware P1b'}) ===",
          flush=True)
    salted = system_prompt + f" [bench-cache-aware-{arm}]"
    system_now = salted
    history = []
    rows = []
    last_prompt_tokens = 0
    events = {}

    for call in range(1, 20):
        if call in MEASURE_CALLS:
            history.append({
                "role": "user",
                "content": f"Turn {call}: acknowledge and continue. "
                           + "ctx " * 60,
            })
        else:
            topic, chars, _event = PLAN[call]
            user, asst, tool = make_turn(call, topic, chars)
            history.append(user)

        q0, h0 = metrics_counters()
        ttft, usage = chat_stream(
            [{"role": "system", "content": system_now}] + history, tools)
        q1, h1 = metrics_counters()
        prompt_tokens = usage.get("prompt_tokens") if usage else None
        if prompt_tokens:
            last_prompt_tokens = prompt_tokens
        hit_pct = None
        if None not in (q0, h0, q1, h1):
            dq, dh = q1 - q0, h1 - h0
            if dq > 0:
                hit_pct = 100.0 * dh / dq
        import hashlib
        sp_hash = hashlib.sha256(system_now.encode()).hexdigest()[:12]
        rows.append({
            "call": call,
            "ttft_s": round(ttft, 3) if ttft else None,
            "prompt_tokens": prompt_tokens,
            "hit_pct": round(hit_pct, 1) if hit_pct is not None else None,
            "system_sha": sp_hash,
            "history_msgs": len(history),
        })
        print(f"  call {call:2d}  TTFT={ttft:6.2f}s  "
              f"prompt={prompt_tokens}  "
              f"hit={hit_pct if hit_pct is None else round(hit_pct, 1)}%  "
              f"sys={sp_hash}", flush=True)

        if call in MEASURE_CALLS:
            history.append({"role": "assistant", "content": "ack"})
            continue
        topic, chars, event = PLAN[call]
        # complete the tool round
        _, asst, tool = make_turn(call, topic, chars)
        history.append(asst)
        history.append(tool)

        if event:
            tail = EVENT_TAIL[event]
            eng = make_engine(tail, arm == "B", embed_model)
            archived = eng.archive(history, current_tokens=last_prompt_tokens)
            # _archive_context: bridge + todo snapshots appended fresh
            archived.append({"role": "user", "content": BRIDGE})
            archived.append({"role": "user", "content": TODO})
            # system-prompt handling: A re-stamps; B pins via the exact
            # run_agent logic.
            stamped = re_stamp(system_now)
            if arm == "B":
                pinned = AIAgent._pin_system_prompt_bytes(system_now, stamped)
                system_now = pinned if pinned is not None else stamped
            else:
                system_now = stamped
            history = archived
            events[event] = {
                "path": eng._last_archive_path,
                "pre_msgs": rows[-1]["history_msgs"],
                "post_msgs": len(archived),
                "system_sha": hashlib.sha256(system_now.encode()).hexdigest()[:12],
            }
            print(f"  -- {event}: path={events[event]['path']}  "
                  f"msgs {events[event]['pre_msgs']} -> "
                  f"{events[event]['post_msgs']}  "
                  f"sys={events[event]['system_sha']}", flush=True)

    return rows, events


def main():
    print("Loading real system prompt + tool schemas + embedding model ...")
    system_prompt = load_real_system_prompt()
    tools = load_essential_tools()
    print(f"  system prompt: {len(system_prompt)} chars "
          f"(~{len(system_prompt)//4} tok)  tools: {len(tools)}")
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer(EMBED_PATH, device="cpu")
    print(f"  embeddings: {EMBED_PATH} (cpu)")

    results, events = {}, {}
    for arm in ("A", "B"):
        results[arm], events[arm] = run_arm(arm, system_prompt, tools,
                                            embed_model)

    print("\n=== SUMMARY ===")
    for arm in ("A", "B"):
        rows = results[arm]
        ev = events[arm]
        sys_h = {r["system_sha"] for r in rows}
        busts = [r["call"] for r in rows
                 if r["hit_pct"] is not None and r["hit_pct"] < 50
                 and r["call"] > 1]
        warm = [r["hit_pct"] for r in rows if r["hit_pct"] is not None
                and r["hit_pct"] >= 50]
        post = {c: next((r["hit_pct"] for r in rows if r["call"] == c), None)
                for c in (12, 15, 18)}
        print(f"arm {arm}: distinct system prompts used: {len(sys_h)}  "
              f"bust calls (hit<50%): {busts}")
        print(f"        post-E0={post[12]}%  post-E1={post[15]}%  "
              f"post-E2={post[18]}%  "
              f"avg hit (warm) {sum(warm)/max(1,len(warm)):5.1f}%")
        for e, d in ev.items():
            print(f"        {e}: path={d['path']}  msgs {d['pre_msgs']} -> "
                  f"{d['post_msgs']}")

    ts = time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join(OUT_DIR, f"report-cache-aware-{ts}.json")
    with open(out, "w") as f:
        json.dump({"vllm": VLLM, "model": MODEL, "engine": ENG,
                   "system_prompt_chars": len(system_prompt),
                   "tools": len(tools),
                   "events": events, "results": results}, f, indent=2)
    print(f"\nreport: {out}")


if __name__ == "__main__":
    main()
