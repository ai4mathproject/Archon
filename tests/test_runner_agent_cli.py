import json

from archon.runner import codex_event_to_archon_events, opencode_event_to_archon_events


def test_codex_agent_message_maps_to_text_event():
    raw = {
        "type": "item.completed",
        "item": {"id": "item_0", "type": "agent_message", "text": "Proof complete."},
    }

    events = codex_event_to_archon_events(raw)

    assert events == [{"event": "text", "content": "Proof complete."}]


def test_codex_turn_completed_maps_to_session_end():
    raw = {
        "type": "turn.completed",
        "usage": {
            "input_tokens": 100,
            "cached_input_tokens": 40,
            "output_tokens": 25,
            "reasoning_output_tokens": 7,
        },
    }

    events = codex_event_to_archon_events(raw, last_result="Proof complete.", session_id="thread-1")

    assert len(events) == 1
    event = events[0]
    assert event["event"] == "session_end"
    assert event["session_id"] == "thread-1"
    assert event["total_cost_usd"] == 0
    assert event["duration_ms"] == 0
    assert event["num_turns"] == 1
    assert event["input_tokens"] == 100
    assert event["cache_read_input_tokens"] == 40
    assert event["output_tokens"] == 25
    assert event["summary"] == "Proof complete."
    assert event["model_usage"]["codex"]["inputTokens"] == 100
    assert event["model_usage"]["codex"]["outputTokens"] == 25


def test_opencode_text_maps_to_text_event():
    raw = {
        "type": "text",
        "sessionID": "ses_1",
        "part": {"type": "text", "text": "\n\nOK"},
    }

    events = opencode_event_to_archon_events(raw)

    assert events == [{"event": "text", "content": "OK"}]


def test_opencode_tool_use_maps_to_tool_call_and_result():
    raw = {
        "type": "tool_use",
        "sessionID": "ses_1",
        "part": {
            "type": "tool",
            "tool": "bash",
            "state": {
                "status": "completed",
                "input": {"command": "pwd", "description": "Print working directory"},
                "output": "/repo\n",
            },
        },
    }

    events = opencode_event_to_archon_events(raw)

    assert events == [
        {
            "event": "tool_call",
            "tool": "Bash",
            "input": {"command": "pwd", "description": "Print working directory"},
        },
        {"event": "tool_result", "content": "/repo\n"},
    ]


def test_opencode_step_finish_maps_to_session_end():
    raw = {
        "type": "step_finish",
        "sessionID": "ses_1",
        "part": {
            "type": "step-finish",
            "tokens": {
                "input": 47,
                "output": 39,
                "reasoning": 0,
                "cache": {"write": 82, "read": 32699},
            },
            "cost": 0,
        },
    }

    events = opencode_event_to_archon_events(raw, last_result="done")

    assert len(events) == 1
    event = events[0]
    assert event["event"] == "session_end"
    assert event["session_id"] == "ses_1"
    assert event["input_tokens"] == 47
    assert event["output_tokens"] == 39
    assert event["cache_read_input_tokens"] == 32699
    assert event["cache_creation_input_tokens"] == 82
    assert event["summary"] == "done"
    assert event["model_usage"]["opencode"]["inputTokens"] == 47
    assert event["model_usage"]["opencode"]["outputTokens"] == 39


def test_opencode_tool_call_step_finish_is_not_session_end():
    raw = {
        "type": "step_finish",
        "sessionID": "ses_1",
        "part": {
            "type": "step-finish",
            "reason": "tool-calls",
            "tokens": {"input": 10, "output": 5, "cache": {"write": 0, "read": 0}},
            "cost": 0,
        },
    }

    assert opencode_event_to_archon_events(raw, last_result="") == []


def test_parse_agent_jsonl_stream_ignores_non_json_lines(tmp_path):
    from archon.runner import _parse_agent_jsonl_stream

    log_path = tmp_path / "agent.jsonl"
    lines = [
        "2026-05-09T00:00:00Z WARN plugin warning",
        json.dumps({"type": "thread.started", "thread_id": "thread-1"}),
        json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "OK"}}),
        json.dumps({"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 2}}),
    ]

    _parse_agent_jsonl_stream(lines, log_path, converter=codex_event_to_archon_events)

    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["text", "session_end"]
    assert rows[0]["content"] == "OK"
    assert rows[1]["session_id"] == "thread-1"


def test_run_agent_dispatches_to_codex(monkeypatch, tmp_path):
    from archon import runner
    from archon.types import AgentBackend

    calls = []

    def fake_run_codex(prompt, *, cwd, log_base=None, verbose_logs=False, extra_args=None):
        calls.append((prompt, cwd, log_base, verbose_logs, extra_args))
        return True

    monkeypatch.setattr(runner, "run_codex", fake_run_codex)

    ok = runner.run_agent(
        "hello",
        backend=AgentBackend.codex,
        cwd=tmp_path,
        log_base=tmp_path / "agent",
        verbose_logs=True,
    )

    assert ok is True
    assert calls == [("hello", tmp_path, tmp_path / "agent", True, None)]


def test_run_agent_dispatches_to_opencode(monkeypatch, tmp_path):
    from archon import runner
    from archon.types import AgentBackend

    calls = []

    def fake_run_opencode(prompt, *, cwd, log_base=None, verbose_logs=False, extra_args=None):
        calls.append((prompt, cwd, log_base, verbose_logs, extra_args))
        return True

    monkeypatch.setattr(runner, "run_opencode", fake_run_opencode)

    ok = runner.run_agent(
        "hello",
        backend=AgentBackend.opencode,
        cwd=tmp_path,
        log_base=tmp_path / "agent",
        verbose_logs=False,
    )

    assert ok is True
    assert calls == [("hello", tmp_path, tmp_path / "agent", False, None)]


def test_run_agent_default_claude_path_is_unchanged(monkeypatch, tmp_path):
    from archon import runner
    from archon.types import AgentBackend

    calls = []

    def fake_run_claude(prompt, *, cwd, log_base=None, verbose_logs=False, extra_args=None):
        calls.append((prompt, cwd, log_base, verbose_logs, extra_args))
        return True

    monkeypatch.setattr(runner, "run_claude", fake_run_claude)

    ok = runner.run_agent(
        "hello",
        backend=AgentBackend.claude,
        cwd=tmp_path,
        log_base=tmp_path / "agent",
        verbose_logs=False,
    )

    assert ok is True
    assert calls == [("hello", tmp_path, tmp_path / "agent", False, None)]
