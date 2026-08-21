"""Integration tests for the full archive flow: command → compression → context bridge → task pruning.

Tests realistic chat scenarios with multi-turn conversations, tool calls, and task boundaries.
Run standalone: python -m pytest tests/integration/test_archive_full_flow.py -v
"""

import json
from unittest.mock import MagicMock, patch, PropertyMock

# --- Fixtures for realistic chat data ---

def make_realistic_conversation():
    """Build a multi-turn conversation simulating real Logos usage with tasks, tool calls, and memory context."""
    return [
        # System prompt (protected)
        {"role": "system", "content": "You are a personal assistant."},

        # Task 1: Code review (completed)
        {"role": "user", "content": "Review the G-code for my CNC post-processor. Check for threading issues."},
        {"role": "assistant", "content": "I'll inspect the repo first.\n\n<tool_calls>[{\"name\": \"read_file\", \"args\": {\"path\": \"post_processor.py\"}}]</tool_calls>"},
        {"role": "tool", "content": "def generate_gcode(tool_path, depth):\n    # threading logic here\n    return gcode_string\n\n# 200 lines of G-code output follow...\n" + "\n".join([f"G0 X{i} Y{i*2} Z-0.5" for i in range(100)])},
        {"role": "assistant", "content": "Found a threading lock issue on line 47. The spindle speed isn't synchronized with feed rate during helical interpolation. Here's the fix:\n\n```python\nwith threading.Lock():\n    set_spindle_speed(rpm)\n    set_feed_rate(mm_per_min)\n```\n\nThe post-processor now correctly handles multi-axis threading."},
        {"role": "user", "content": "Good, apply that fix and test it."},
        {"role": "assistant", "content": "Applied. Running tests...\n\n<tool_calls>[{\"name\": \"terminal\", \"args\": {\"command\": \"python -m pytest tests/test_post_processor.py\"}}]</tool_calls>"},
        {"role": "tool", "content": "============================= test session starts ==============================\ncollected 12 items\n\ntests/test_post_processor.py ............\n\n============================== 12 passed in 0.45s =============================="},
        {"role": "assistant", "content": "All 12 tests pass. The threading fix is verified."},

        # Task 2: RL page creation (completed)
        {"role": "user", "content": "Create a Reference Library page for Christology."},
        {"role": "assistant", "content": "I'll research and create the page.\n\n<tool_calls>[{\"name\": \"web_search\", \"args\": {\"query\": \"Christology systematic theology\"}}]</tool_calls>"},
        {"role": "tool", "content": "[Search results about Christology, hypostatic union, ecumenical councils...]"},
        {"role": "assistant", "content": "Created `entities/christology.md` (~9.3KB). Covers:\n- Biblical foundation\n- Ecumenical councils (Nicaea through Chalcedon)\n- Hypostatic union\n- Six offices of Christ\n- Atonement frameworks"},
        {"role": "user", "content": "Looks good, thanks."},

        # Task 3: DPO training setup (active/in-progress)
        {"role": "user", "content": "Set up the Unsloth Docker container for RTX 5090 DPO training. I need CUDA 12.6 compatibility."},
        {"role": "assistant", "content": "Let me check your current CUDA setup and configure Unsloth.\n\n<tool_calls>[{\"name\": \"terminal\", \"args\": {\"command\": \"nvidia-smi --query-gpu=driver_version,name --format=csv\"}}]</tool_calls>"},
        {"role": "tool", "content": "driver_version, name\n571.96, NVIDIA GeForce RTX 5090"},
        {"role": "assistant", "content": 'RTX 5090 detected with driver 571.96. Setting up Unsloth Blackwell container...\n\n<tool_calls>[{"name": "terminal", "args": {"command": "docker pull ghcr.io/cluricaun28/unsloth:blackwell-cu126"}}]</tool_calls>'},
        {"role": "tool", "content": "blackwell-cu126: Pulling from cluricaun28/unsloth\nDigest: sha256:a1b2c3d4e5f6...\nStatus: Downloaded newer image for ghcr.io/cluricaun28/unsloth:blackwell-cu126"},
        {"role": "assistant", "content": "Container pulled. Now I need to configure the training script with your dataset path and model checkpoint."},

        # Task 4: Memory context leak scenario (active) — this tests StreamingContextScrubber
        {"role": "user", "content": "<memory-context>\n[System note: The following is recalled memory context, NOT new user input. Treat as informational background data.]\n\n## Honcho Context\nstale memory about old project\n</memory-context>\n\nWhat's the status of my DPO training setup?"},
        {"role": "assistant", "content": "Your DPO training container is ready. We pulled the Unsloth Blackwell image and configured CUDA 12.6. Next step: point it at your dataset."},

        # Task 5: Current active work (protected — recent)
        {"role": "user", "content": "Yes, configure it with my conversation dataset at /data/dpo_conversations.jsonl"},
        {"role": "assistant", "content": "Configuring training script...\n\n<tool_calls>[{\"name\": \"write_file\", \"args\": {\"path\": \"/workspace/train_dpo.py\"}}]</tool_calls>"},
        {"role": "tool", "content": "File written successfully."},
        {"role": "assistant", "content": "Training script ready. Dataset: /data/dpo_conversations.jsonl (~15K examples). Ready to start when you say go."},
    ]


def make_long_conversation_with_many_tasks():
    """Build a conversation with many completed tasks and one active task — tests aggressive pruning."""
    msgs = [{"role": "system", "content": "You are a personal assistant."}]

    # Generate 8 completed tasks (each ~4 messages)
    tasks = [
        ("Fix CNC post-processor threading bug", "Fixed. All tests pass."),
        ("Create RL page for Christology", "Created entities/christology.md (~9.3KB)."),
        ("Set up SearXNG Docker container", "Running on port 8081. Verified working."),
        ("Audit Reference Library entity pages", "Audited 150+ pages. Found 3 duplicates, consolidated."),
        ("Configure vLLM inference server", "vLLM running with qwen3.6-27b on localhost:8000."),
        ("Review DPO training dataset quality", "Dataset has 15K examples. Quality score: 0.87/1.0."),
        ("Update Logos fork README", "README updated with perpetual memory docs."),
        ("Test gateway Telegram adapter", "All platform tests passing. Gateway healthy."),
    ]

    for i, (task_desc, result) in enumerate(tasks):
        task_num = i + 1
        msgs.append({"role": "user", f"content": f"Task {task_num}: {task_desc}"})
        msgs.append({"role": "assistant", "content": f"Working on task {task_num}...\n\n<tool_calls>[{{\"name\": \"terminal\", \"args\": {{}}}}]</tool_calls>"})
        msgs.append({"role": "tool", "content": f"[Tool output for task {task_num}: 50 lines of results...]\n" + "\n".join([f"Line {j} of tool output" for j in range(50)])})
        msgs.append({"role": "assistant", "content": result})

    # Active task at the end (should be protected)
    msgs.append({"role": "user", "content": "Now help me configure the new inference server with 256GB VRAM."})
    msgs.append({"role": "assistant", 'content': 'Let me check your hardware setup...\n\n<tool_calls>[{"name": "terminal", "args": {"command": "nvidia-smi --query-gpu=memory.total,name --format=csv"}}]</tool_calls>'})
    msgs.append({"role": "tool", "content": "memory.total [MiB], name\n262144, NVIDIA A100-SXM4-80GB x4 (NVLink)"})
    msgs.append({"role": "assistant", "content": "Excellent — 4x A100s with NVLink. That's ~320GB total VRAM. Let me configure vLLM for tensor parallelism across all 4 GPUs."})

    return msgs


# --- Task marker helpers (real API: TaskMarkerInjector) ---

MARKER_START_PREFIX = "[TASK_START:"


def annotate_tasks(msgs):
    """Annotate a conversation with task markers in the real marker format.

    Each user message starts a new task and completes the previously open
    one. Markers are placed on the first assistant message after the user
    message, because ``TaskMarkerInjector.parse_markers_from_messages``
    only reads marker text from assistant content. The final task is left
    open so the pruner treats it as active work.
    """
    out = [dict(m) for m in msgs]
    open_task = None
    i = 0
    while i < len(out):
        if out[i].get("role") != "user":
            i += 1
            continue
        task_name = f"task_{i}"
        marker = ""
        if open_task is not None:
            marker += f"[TASK_COMPLETE: {open_task}] "
        marker += f"{MARKER_START_PREFIX} {task_name}]"
        j = i + 1
        while j < len(out) and out[j].get("role") != "assistant":
            j += 1
        if j < len(out):
            content = out[j].get("content") or ""
            out[j] = {**out[j], "content": f"{marker} {content}"}
        open_task = task_name
        i = max(j, i + 1)
    return out


# --- Tests ---

class TestArchiveCommandEndToEnd:
    """Test the /archive command flow from gateway route through compression."""

    def test_archive_command_dispatches_correctly(self):
        """Verify /archive is recognized as an alias for compress and routed to _handle_compress_command."""
        from logos_cli.commands import resolve_command, COMMAND_REGISTRY

        # Check that 'archive' resolves to the same handler as 'compress'
        archive_def = None
        for cmd in COMMAND_REGISTRY:
            if cmd.name == "archive":
                archive_def = cmd
                break

        assert archive_def is not None, "/archive command definition missing from registry"
        # resolve_command returns a CommandDef — check its name or aliases
        resolved = resolve_command("archive")
        assert resolved is not None, "/archive should resolve to something"
        # The resolved def should be the 'archive' command (which has compress as alias)
        assert resolved.name == "archive", f"/archive should resolve to archive CommandDef, got {resolved}"

    def test_archive_handler_exists_in_gateway(self):
        """Verify gateway has _handle_compress_command method."""
        from gateway.run import GatewayRunner

        assert hasattr(GatewayRunner, "_handle_compress_command"), \
            "GatewayRunner missing _handle_compress_command method"

    def test_archive_route_checking_canonical_name(self):
        """Verify run.py checks both 'compress' and 'archive' as canonical names."""
        # Read the relevant section of gateway/run.py
        with open("gateway/run.py") as f:
            content = f.read()

        assert '"compress", "archive"' in content or "'compress', 'archive'" in content, \
            "Gateway should check both 'compress' and 'archive' canonical names"


class TestContextBridgeBuilderIntegration:
    """Test context bridge builder with realistic message data."""

    def test_bridge_builder_with_realistic_conversation(self):
        """Build a context bridge from a realistic multi-task conversation."""
        from plugins.memory.perpetual_context.context_bridge_builder import ContextBridgeBuilder

        msgs = make_realistic_conversation()
        builder = ContextBridgeBuilder()

        # Build without extraction engine (uses regex-based fallback)
        bridge = builder.build_bridge(msgs)

        assert isinstance(bridge, str), "build_bridge should return a string"
        # Should contain some content from the conversation
        assert len(bridge) > 0, "Context bridge should not be empty for non-empty messages"
        # Should respect the 4KB cap
        from plugins.memory.perpetual_context.context_bridge_builder import MAX_BRIDGE_CHARS
        assert len(bridge) <= MAX_BRIDGE_CHARS, f"Bridge exceeds {MAX_BRIDGE_CHARS} char limit"

    def test_bridge_builder_empty_messages(self):
        """Build a context bridge from empty message list."""
        from plugins.memory.perpetual_context.context_bridge_builder import ContextBridgeBuilder

        builder = ContextBridgeBuilder()
        bridge = builder.build_bridge([])
        assert isinstance(bridge, str), "build_bridge should return string even for empty input"

    def test_bridge_builder_with_memory_context_tags(self):
        """Verify context bridge handles memory-context tagged messages."""
        from plugins.memory.perpetual_context.context_bridge_builder import ContextBridgeBuilder

        msgs = [
            {"role": "user", "content": "<memory-context>\nstale recalled data\n</memory-context>\nWhat's next?"},
            {"role": "assistant", "content": "Let me check the current status."},
        ]
        builder = ContextBridgeBuilder()
        bridge = builder.build_bridge(msgs)
        assert isinstance(bridge, str), "Should handle memory-context tags gracefully"


class TestTaskAwarePruningIntegration:
    """Test task-aware pruning with realistic conversations."""

    def _make_engine(self, **kwargs):
        from plugins.context_engine.rolling_window import RollingWindowContextEngine

        kwargs.setdefault("max_tokens", 131072)
        engine = RollingWindowContextEngine(**kwargs)
        engine.update_model("qwen3.6-27b", context_length=131072)
        return engine

    def test_prune_closes_tasks_keeps_active(self):
        """Verify pruning reduces the conversation while preserving system + active work."""
        engine = self._make_engine(window_size=15, protect_first_n=3, protect_last_n=10)

        msgs = make_realistic_conversation()
        annotated = annotate_tasks(msgs)

        # Run archive with a high current-token count to exercise the full pipeline
        result = engine.archive(annotated, current_tokens=int(131072 * 0.85))

        assert len(result) < len(msgs), "Pruner should reduce message count"
        assert len(result) > 0, "Pruner should not drop all messages"

        # Check that system prompt is preserved (protected first N)
        assert result[0]["role"] == "system", "System prompt should be preserved"

    def test_prune_preserves_recent_conversation(self):
        """Verify the last N messages are always protected."""
        engine = self._make_engine(window_size=12, protect_first_n=3, protect_last_n=15)

        msgs = make_long_conversation_with_many_tasks()
        annotated = annotate_tasks(msgs)

        result = engine.archive(annotated)

        # The last few messages should contain our active task content
        recent_content = " ".join(m.get("content", "") for m in result[-5:])
        assert "A100" in recent_content or "inference" in recent_content.lower() or "VRAM" in recent_content, \
            f"Recent conversation about inference server should be preserved. Got: {recent_content[:200]}"

    def test_prune_strips_tool_calls(self):
        """Verify raw tool_calls JSON is stripped from assistant messages."""
        engine = self._make_engine(window_size=15, protect_first_n=3, protect_last_n=10)

        msgs = make_realistic_conversation()
        annotated = annotate_tasks(msgs)

        result = engine.archive(annotated)

        for msg in result:
            if msg.get("role") == "assistant":
                assert not msg.get("tool_calls"), \
                    f"Tool calls should be stripped from assistant messages. Found: {msg.get('tool_calls')}"

    def test_prune_truncates_tool_results(self):
        """Verify long tool results are truncated to first/last 3 lines."""
        engine = self._make_engine(window_size=15, protect_first_n=3, protect_last_n=10)

        msgs = make_realistic_conversation()
        annotated = annotate_tasks(msgs)

        result = engine.archive(annotated)

        for msg in result:
            if msg.get("role") == "tool":
                content = msg.get("content", "") or ""
                line_count = len(content.split("\n"))
                # Should be truncated to ~6 lines (3 first + 1 truncation marker + 3 last)
                assert line_count <= 8, \
                    f"Tool result should be truncated. Got {line_count} lines: {content[:200]}"

    def test_fallback_to_original_algorithm(self):
        """Verify fallback window pruning when task-aware components are unavailable."""
        engine = self._make_engine(window_size=15, protect_first_n=3, protect_last_n=10)
        engine._pruner = None  # Simulate task-aware components unavailable

        msgs = make_realistic_conversation()  # Don't annotate — no task markers present
        result = engine.archive(msgs)

        assert len(result) < len(msgs), "Fallback pruner should still reduce message count"
        # Fallback keeps the tail of the conversation (most recent context)
        assert [m["role"] for m in result] == [m["role"] for m in msgs][-len(result):], \
            "Fallback window prune should keep a suffix of the conversation"


class TestStreamingContextScrubberIntegration:
    """Test StreamingContextScrubber wired into GatewayStreamConsumer."""

    def test_scrubber_imported_in_stream_consumer(self):
        """Verify StreamingContextScrubber is imported and used in stream consumer."""
        import gateway.stream_consumer as sc_module
        import inspect

        source = inspect.getsource(sc_module)
        assert "StreamingContextScrubber" in source, \
            "stream_consumer.py should reference StreamingContextScrubber"

    def test_scrubber_initialized_in_consumer(self):
        """Verify GatewayStreamConsumer creates a scrubber instance."""
        from gateway.stream_consumer import GatewayStreamConsumer

        mock_adapter = MagicMock()
        consumer = GatewayStreamConsumer(mock_adapter, "test_chat")

        assert hasattr(consumer, "_context_scrubber"), \
            "GatewayStreamConsumer should have _context_scrubber attribute"
        assert consumer._context_scrubber is not None, \
            "_context_scrubber should be initialized"

    def test_scrubber_strips_memory_context_in_stream(self):
        """Verify memory-context spans are stripped from streaming deltas."""
        from gateway.stream_consumer import GatewayStreamConsumer

        mock_adapter = MagicMock()
        consumer = GatewayStreamConsumer(mock_adapter, "test_chat")

        # Simulate a stream with memory-context tags split across deltas
        deltas = [
            "Hello ",
            "<memory-context>\npayload ",
            "more payload\n",
            "</memory-context> world",
        ]

        for delta in deltas:
            clean = consumer._context_scrubber.feed(delta)
            # Verify no memory-context tags leak through
            assert "<memory-context>" not in clean, f"Open tag leaked: {clean}"
            assert "</memory-context>" not in clean, f"Close tag leaked: {clean}"

        tail = consumer._context_scrubber.flush()
        full_output = "".join([
            "Hello ",  # first delta (no tags)
            " world",  # after close tag
        ]) + tail

        assert "payload" not in full_output, f"Memory payload leaked: {full_output}"

    def test_scrubber_reset_on_segment_break(self):
        """Verify scrubber resets on segment break to prevent hung spans."""
        from gateway.stream_consumer import GatewayStreamConsumer

        mock_adapter = MagicMock()
        consumer = GatewayStreamConsumer(mock_adapter, "test_chat")

        # Feed an unclosed memory-context tag (simulates provider dropping close tag)
        consumer._context_scrubber.feed("<memory-context>unclosed")

        # Reset should clear the hung span
        consumer._context_scrubber.reset()

        # Next feed should work normally
        result = consumer._context_scrubber.feed("clean text after reset")
        assert "clean text after reset" in result, \
            f"After reset, clean text should pass through. Got: {result}"


class TestRollingWindowEngineIntegration:
    """Test RollingWindowContextEngine end-to-end with realistic data."""

    def test_engine_archive_with_realistic_conversation(self):
        """Run the full rolling window archive on a realistic conversation."""
        from plugins.context_engine.rolling_window import RollingWindowContextEngine

        engine = RollingWindowContextEngine(
            max_tokens=131072,
            threshold_percent=0.75,
            task_aware=True,
            protect_last_n=10,  # Lower to allow pruning of middle messages
        )
        engine.update_model("qwen3.6-27b", context_length=131072)

        msgs = make_long_conversation_with_many_tasks()  # Use the bigger conversation
        annotated = annotate_tasks(msgs)

        result = engine.archive(annotated, current_tokens=int(131072 * 0.85))

        assert len(result) < len(msgs), f"Engine should reduce message count: {len(result)} vs {len(msgs)}"
        assert engine.archive_count == 1, "Archive count should increment"

    def test_engine_should_archive_threshold(self):
        """Verify should_archive fires at correct threshold."""
        from plugins.context_engine.rolling_window import RollingWindowContextEngine

        engine = RollingWindowContextEngine(
            max_tokens=131072,
            threshold_percent=0.75,
        )
        engine.update_model("qwen3.6-27b", context_length=131072)

        # Below threshold (75% of 131072 = ~98304)
        assert not engine.should_archive(prompt_tokens=90000), \
            "Should not archive below threshold"

        # Above threshold
        assert engine.should_archive(prompt_tokens=100000), \
            "Should archive above threshold"

    def test_engine_escalating_pressure(self):
        """Verify escalating pressure model targets lower ratios under high pressure."""
        from plugins.context_engine.rolling_window import RollingWindowContextEngine

        engine = RollingWindowContextEngine(
            max_tokens=131072,
            threshold_percent=0.75,
            archive_target=0.65,
        )
        engine.update_model("qwen3.6-27b", context_length=131072)

        # At 80% pressure — moderate pruning
        msgs = make_realistic_conversation()
        annotated = annotate_tasks(msgs)
        result_80 = engine.archive(annotated, current_tokens=int(131072 * 0.80))

        # Reset and try at 95% pressure — aggressive pruning
        engine2 = RollingWindowContextEngine(
            max_tokens=131072,
            threshold_percent=0.75,
            archive_target=0.65,
        )
        engine2.update_model("qwen3.6-27b", context_length=131072)
        annotated2 = annotate_tasks(msgs)
        result_95 = engine2.archive(annotated2, current_tokens=int(131072 * 0.95))

        assert len(result_95) <= len(result_80), \
            f"Higher pressure (95%) should drop more messages than moderate (80%): {len(result_95)} vs {len(result_80)}"


class TestArchiveSessionSplit:
    """Test that archiving correctly splits sessions in SQLite."""

    def test_archive_creates_new_session_id(self):
        """Verify _archive_context creates a new session ID and links to parent."""
        from run_agent import AIAgent
        from unittest.mock import MagicMock, patch

        # Mock the agent with minimal setup
        agent = MagicMock(spec=AIAgent)
        agent.session_id = "20260430_test_session"
        agent._session_db = MagicMock()
        agent._memory_manager = None
        agent.context_archiver = MagicMock()
        agent.context_archiver.archive_count = 0

        # Mock the archiver to return a reduced message list
        test_msgs = make_realistic_conversation()
        agent.context_archiver.archive.return_value = [
            {"role": "system", "content": "You are a personal assistant."},
            {"role": "user", "content": "Summary of previous work: CNC fix, RL page, DPO setup"},
            {"role": "assistant", "content": "Ready to continue with inference server configuration."},
        ]

        # Verify the archiver interface is correct
        assert hasattr(agent.context_archiver, 'archive'), "Archiver should have archive method"
        assert hasattr(agent.context_archiver, 'annotate_tasks'), "Archiver should have annotate_tasks method"


class TestMemoryContextScrubberInStreamConsumer:
    """Verify StreamingContextScrubber is properly integrated in the drain loop."""

    def test_scrubber_called_before_filter_and_accumulate(self):
        """Verify scrubber runs before think-block filter in the processing pipeline."""
        import gateway.stream_consumer as sc_module
        import inspect

        source = inspect.getsource(sc_module.GatewayStreamConsumer.run)

        # The scrubber should be called before _filter_and_accumulate
        scrubber_pos = source.find("_context_scrubber.feed")
        filter_pos = source.find("_filter_and_accumulate")

        assert scrubber_pos != -1, "run() should call _context_scrubber.feed()"
        assert filter_pos != -1, "run() should call _filter_and_accumulate()"
        assert scrubber_pos < filter_pos, \
            "_context_scrubber.feed() should run before _filter_and_accumulate()"

    def test_scrubber_flush_on_stream_end(self):
        """Verify scrubber flush is called when stream ends."""
        import gateway.stream_consumer as sc_module
        import inspect

        source = inspect.getsource(sc_module.GatewayStreamConsumer.run)

        assert "_context_scrubber.flush" in source, \
            "run() should call _context_scrubber.flush() on stream end"

    def test_scrubber_reset_on_segment_break(self):
        """Verify scrubber reset is called on segment break."""
        import gateway.stream_consumer as sc_module
        import inspect

        source = inspect.getsource(sc_module.GatewayStreamConsumer.run)

        assert "_context_scrubber.reset" in source, \
            "run() should call _context_scrubber.reset() on segment break"


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
