# Archon Agent CLI Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Archon run through Codex CLI and OpenCode CLI while keeping Archon's existing plan/prover/review workflow, Lean LSP MCP access, and dashboard log format.

**Architecture:** Add one small execution-mode selector with three values: `claude`, `codex`, and `opencode`. Keep Claude as the default only to avoid breaking the current workflow; the product goal is compatibility with Codex and OpenCode. Codex and OpenCode JSON streams are translated into Archon's existing JSONL rows (`text`, `tool_call`, `tool_result`, `session_end`) so the dashboard and `extract-attempts.py` continue to work.

**Tech Stack:** Python 3.10+, Typer, pytest, Codex CLI (`codex exec --json`), OpenCode CLI (`opencode run --format json`), existing Archon dashboard JSONL schema, bundled `lean-lsp-mcp`.

---

## Purpose Lock

The only product goal is adapting Archon to run with Codex and OpenCode. Do not turn this into a general agent framework, prompt rewrite, dashboard redesign, proof-strategy refactor, or native plugin migration.

Acceptance means:

- `archon init --agent codex <lean-project>` configures Codex to access the bundled `archon-lean-lsp` MCP server.
- `archon init --agent opencode <lean-project>` configures OpenCode to access the bundled `archon-lean-lsp` MCP server.
- `archon loop --agent codex <lean-project>` runs plan/prover/review phases through Codex.
- `archon loop --agent opencode <lean-project>` runs plan/prover/review phases through OpenCode.
- Codex and OpenCode logs are written in Archon's existing dashboard-compatible JSONL format.
- Existing Claude behavior remains the default path as a non-regression guardrail.

## File Structure

- Modify `pyproject.toml`: add pytest as an optional test dependency.
- Modify `src/archon/types.py`: add `AgentBackend` enum with `claude`, `codex`, `opencode`.
- Modify `src/archon/runner.py`: add Codex/OpenCode event converters, process runners, and a minimal `run_agent()` dispatcher.
- Modify `src/archon/commands/loop.py`: add `--agent` and pass the selected backend through plan/prover/review.
- Modify `src/archon/commands/init.py`: configure `archon-lean-lsp` for Codex and OpenCode.
- Modify `src/archon/commands/doctor.py`: report Codex/OpenCode MCP readiness.
- Create `tests/test_runner_agent_cli.py`: JSON conversion and runner dispatch tests.
- Create `tests/test_loop_agent_cli.py`: loop backend plumbing tests.
- Create `tests/test_init_doctor_agent_cli.py`: Codex/OpenCode MCP setup and doctor checks.

## Implementation Tasks

### Task 1: Add Focused Test Infrastructure

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/test_runner_agent_cli.py`

- [ ] **Step 1: Add pytest optional dependency**

Modify `pyproject.toml`:

```toml
[project.optional-dependencies]
test = [
    "pytest>=8",
]
```

- [ ] **Step 2: Write failing conversion tests**

Create `tests/test_runner_agent_cli.py`:

```python
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
```

- [ ] **Step 3: Run tests and confirm expected failure**

Run:

```bash
python3 -m pytest tests/test_runner_agent_cli.py -q
```

Expected: FAIL with import errors for the new conversion functions.

- [ ] **Step 4: Commit the test scaffold**

```bash
git add pyproject.toml tests/test_runner_agent_cli.py
git commit -m "test: add agent cli compatibility scaffold"
```

### Task 2: Add Agent Backend Type and JSON Event Conversion

**Files:**
- Modify: `src/archon/types.py`
- Modify: `src/archon/runner.py`
- Test: `tests/test_runner_agent_cli.py`

- [ ] **Step 1: Add `AgentBackend`**

Modify `src/archon/types.py`:

```python
"""Shared types and constants."""

from enum import Enum


class Stage(str, Enum):
    autoformalize = "autoformalize"
    prover = "prover"
    polish = "polish"


class AgentBackend(str, Enum):
    claude = "claude"
    codex = "codex"
    opencode = "opencode"
```

- [ ] **Step 2: Add conversion helpers**

Add to `src/archon/runner.py` near the stream parsing code:

```python
def _normalize_opencode_tool_name(tool: str) -> str:
    mapping = {
        "bash": "Bash",
        "read": "Read",
        "write": "Write",
        "edit": "Edit",
        "grep": "Grep",
        "glob": "Glob",
    }
    return mapping.get(tool, tool)


def codex_event_to_archon_events(
    raw: dict,
    *,
    last_result: str = "",
    session_id: str = "",
) -> list[dict]:
    """Convert one `codex exec --json` event into Archon's dashboard JSONL events."""
    event_type = raw.get("type")

    if event_type == "item.completed":
        item = raw.get("item") or {}
        item_type = item.get("type")
        if item_type == "agent_message":
            text = (item.get("text") or "").strip()
            return [{"event": "text", "content": text}] if text else []
        if item_type in {"tool_call", "function_call"}:
            return [{
                "event": "tool_call",
                "tool": item.get("name") or item.get("tool") or item_type,
                "input": item.get("arguments") or item.get("input") or {},
            }]
        if item_type in {"tool_result", "function_call_output"}:
            content = item.get("text") or item.get("output") or item.get("content") or ""
            return [{"event": "tool_result", "content": str(content)}]
        return []

    if event_type == "turn.completed":
        usage = raw.get("usage") or {}
        input_tokens = usage.get("input_tokens", 0) or 0
        output_tokens = usage.get("output_tokens", 0) or 0
        cached_input_tokens = usage.get("cached_input_tokens", 0) or 0
        return [{
            "event": "session_end",
            "session_id": session_id,
            "total_cost_usd": 0,
            "duration_ms": 0,
            "duration_api_ms": 0,
            "num_turns": 1,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_input_tokens": cached_input_tokens,
            "cache_creation_input_tokens": 0,
            "model_usage": {
                "codex": {
                    "inputTokens": input_tokens,
                    "outputTokens": output_tokens,
                    "costUSD": 0,
                }
            },
            "summary": last_result,
        }]

    return []


def opencode_event_to_archon_events(
    raw: dict,
    *,
    last_result: str = "",
    session_id: str = "",
) -> list[dict]:
    """Convert one `opencode run --format json` event into Archon's dashboard JSONL events."""
    event_type = raw.get("type")
    effective_session_id = raw.get("sessionID") or session_id

    if event_type == "text":
        part = raw.get("part") or {}
        text = (part.get("text") or "").strip()
        return [{"event": "text", "content": text}] if text else []

    if event_type == "tool_use":
        part = raw.get("part") or {}
        state = part.get("state") or {}
        tool = _normalize_opencode_tool_name(part.get("tool") or "tool")
        tool_input = state.get("input") or {}
        output = state.get("output")
        if output is None:
            metadata = state.get("metadata") or {}
            output = metadata.get("output") or ""
        return [
            {"event": "tool_call", "tool": tool, "input": tool_input},
            {"event": "tool_result", "content": str(output)},
        ]

    if event_type == "step_finish":
        part = raw.get("part") or {}
        tokens = part.get("tokens") or {}
        cache = tokens.get("cache") or {}
        input_tokens = tokens.get("input", 0) or 0
        output_tokens = tokens.get("output", 0) or 0
        cost = part.get("cost", 0) or 0
        return [{
            "event": "session_end",
            "session_id": effective_session_id,
            "total_cost_usd": cost,
            "duration_ms": 0,
            "duration_api_ms": 0,
            "num_turns": 1,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_input_tokens": cache.get("read", 0) or 0,
            "cache_creation_input_tokens": cache.get("write", 0) or 0,
            "model_usage": {
                "opencode": {
                    "inputTokens": input_tokens,
                    "outputTokens": output_tokens,
                    "costUSD": cost,
                }
            },
            "summary": last_result,
        }]

    return []
```

- [ ] **Step 3: Add shared parser**

Add to `src/archon/runner.py`:

```python
def _parse_agent_jsonl_stream(lines, jsonl_path: Path, *, converter) -> str:
    """Write Archon JSONL rows parsed from an agent JSON stream and return final text."""
    session_id = ""
    last_result = ""

    with jsonl_path.open("a") as out:
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue

            if raw.get("type") == "thread.started":
                session_id = raw.get("thread_id") or session_id
                continue
            if raw.get("sessionID"):
                session_id = raw.get("sessionID") or session_id

            events = converter(raw, last_result=last_result, session_id=session_id)
            for event in events:
                if event.get("event") == "text":
                    last_result = event.get("content", "")
                out.write(json.dumps(event) + "\n")
                out.flush()

    return last_result
```

- [ ] **Step 4: Run conversion tests**

Run:

```bash
python3 -m pytest tests/test_runner_agent_cli.py -q
```

Expected: PASS for conversion tests.

- [ ] **Step 5: Commit conversion code**

```bash
git add src/archon/types.py src/archon/runner.py tests/test_runner_agent_cli.py
git commit -m "feat: translate codex and opencode logs"
```

### Task 3: Add Codex and OpenCode Process Runners

**Files:**
- Modify: `src/archon/runner.py`
- Modify: `tests/test_runner_agent_cli.py`

- [ ] **Step 1: Add dispatch tests**

Append to `tests/test_runner_agent_cli.py`:

```python
from archon.types import AgentBackend


def test_run_agent_dispatches_to_codex(monkeypatch, tmp_path):
    from archon import runner

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
```

- [ ] **Step 2: Add `run_codex()`, `run_opencode()`, and `run_agent()`**

Import the enum in `src/archon/runner.py`:

```python
from archon.types import AgentBackend
```

Add below `run_claude()`:

```python
def _run_json_agent_process(
    cmd: list[str],
    *,
    cwd: Path,
    log_base: Path | None,
    verbose_logs: bool,
    converter,
) -> bool:
    if log_base is None:
        r = subprocess.run(cmd, cwd=cwd, env=claude_env(), stdin=subprocess.DEVNULL)
        return r.returncode == 0

    log_base.parent.mkdir(parents=True, exist_ok=True)
    jsonl = Path(f"{log_base}.jsonl")
    raw_log = Path(f"{log_base}.raw.jsonl")
    stderr_dest = raw_log if verbose_logs else Path(os.devnull)

    with stderr_dest.open("a") as stderr_file:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=stderr_file,
            stdin=subprocess.DEVNULL,
            cwd=cwd,
            env=claude_env(),
            text=True,
        )
        assert proc.stdout is not None
        final_text = _parse_agent_jsonl_stream(proc.stdout, jsonl, converter=converter)
        proc.wait()

    if final_text:
        print(final_text, flush=True)
    return proc.returncode == 0


def run_codex(
    prompt: str,
    *,
    cwd: Path,
    log_base: Path | None = None,
    verbose_logs: bool = False,
    extra_args: list[str] | None = None,
) -> bool:
    cmd = [
        "codex",
        "exec",
        "--json",
        "--dangerously-bypass-approvals-and-sandbox",
        "-C",
        str(cwd),
        prompt,
    ]
    if extra_args:
        cmd.extend(extra_args)
    return _run_json_agent_process(
        cmd,
        cwd=cwd,
        log_base=log_base,
        verbose_logs=verbose_logs,
        converter=codex_event_to_archon_events,
    )


def run_opencode(
    prompt: str,
    *,
    cwd: Path,
    log_base: Path | None = None,
    verbose_logs: bool = False,
    extra_args: list[str] | None = None,
) -> bool:
    cmd = [
        "opencode",
        "run",
        "--format",
        "json",
        "--dangerously-skip-permissions",
        "--dir",
        str(cwd),
        prompt,
    ]
    if extra_args:
        cmd.extend(extra_args)
    return _run_json_agent_process(
        cmd,
        cwd=cwd,
        log_base=log_base,
        verbose_logs=verbose_logs,
        converter=opencode_event_to_archon_events,
    )


def run_agent(
    prompt: str,
    *,
    backend: AgentBackend,
    cwd: Path,
    log_base: Path | None = None,
    verbose_logs: bool = False,
    extra_args: list[str] | None = None,
) -> bool:
    if backend == AgentBackend.codex:
        return run_codex(prompt, cwd=cwd, log_base=log_base, verbose_logs=verbose_logs, extra_args=extra_args)
    if backend == AgentBackend.opencode:
        return run_opencode(prompt, cwd=cwd, log_base=log_base, verbose_logs=verbose_logs, extra_args=extra_args)
    return run_claude(prompt, cwd=cwd, log_base=log_base, verbose_logs=verbose_logs, extra_args=extra_args)
```

- [ ] **Step 3: Run runner tests**

Run:

```bash
python3 -m pytest tests/test_runner_agent_cli.py -q
```

Expected: PASS.

- [ ] **Step 4: Commit runners**

```bash
git add src/archon/runner.py tests/test_runner_agent_cli.py
git commit -m "feat: add codex and opencode runners"
```

### Task 4: Wire `--agent` Through `archon loop`

**Files:**
- Modify: `src/archon/commands/loop.py`
- Create: `tests/test_loop_agent_cli.py`

- [ ] **Step 1: Write loop plumbing tests**

Create `tests/test_loop_agent_cli.py`:

```python
from archon.types import AgentBackend


def test_run_single_prover_passes_opencode_backend_to_run_agent(monkeypatch, tmp_path):
    from archon.commands import loop

    calls = []

    def fake_run_agent(prompt, *, backend, cwd, log_base=None, verbose_logs=False):
        calls.append({
            "prompt": prompt,
            "backend": backend,
            "cwd": cwd,
            "log_base": log_base,
            "verbose_logs": verbose_logs,
        })
        return True

    monkeypatch.setattr(loop, "run_agent", fake_run_agent)

    ok = loop._run_single_prover(
        "prove",
        tmp_path,
        tmp_path / "logs" / "prover",
        False,
        backend=AgentBackend.opencode,
    )

    assert ok is True
    assert calls == [{
        "prompt": "prove",
        "backend": AgentBackend.opencode,
        "cwd": tmp_path,
        "log_base": tmp_path / "logs" / "prover",
        "verbose_logs": False,
    }]


def test_preflight_checks_selected_non_claude_binary(monkeypatch, tmp_path):
    from archon.commands import loop

    state_dir = tmp_path / ".archon"
    state_dir.mkdir()
    (state_dir / "PROGRESS.md").write_text("## Current Stage\nprover\n")

    seen = []

    def fake_which(binary):
        seen.append(binary)
        return f"/usr/bin/{binary}"

    monkeypatch.setattr(loop.shutil, "which", fake_which)

    loop._preflight(tmp_path, state_dir, dry_run=False, backend=AgentBackend.codex)
    loop._preflight(tmp_path, state_dir, dry_run=False, backend=AgentBackend.opencode)

    assert "codex" in seen
    assert "opencode" in seen
```

- [ ] **Step 2: Update loop imports**

Modify `src/archon/commands/loop.py` imports:

```python
from archon.runner import (
    build_parallel_prover_prompt,
    build_plan_prompt,
    build_prover_prompt,
    build_review_prompt,
    claude_env,
    run_agent,
)
from archon.types import AgentBackend, Stage
```

- [ ] **Step 3: Update `_preflight()`**

Change signature:

```python
def _preflight(project_path: Path, state_dir: Path, dry_run: bool, backend: AgentBackend) -> None:
```

Replace the binary/auth block with:

```python
    if not dry_run:
        if backend == AgentBackend.codex:
            if not shutil.which("codex"):
                log.error("Codex CLI is not installed. Install it or run with --agent claude.")
                raise typer.Exit(1)
            log.success("Codex CLI is available")
        elif backend == AgentBackend.opencode:
            if not shutil.which("opencode"):
                log.error("OpenCode CLI is not installed. Install it or run with --agent claude.")
                raise typer.Exit(1)
            log.success("OpenCode CLI is available")
        else:
            if not shutil.which("claude"):
                log.error("Claude Code is not installed. Run: archon setup")
                raise typer.Exit(1)

            r = subprocess.run(
                ["claude", "-p", "reply with OK", "--no-session-persistence"],
                capture_output=True, text=True,
                env=claude_env(),
            )
            if r.returncode != 0:
                log.error("Claude Code cannot run. Check: claude auth and network.")
                raise typer.Exit(1)
            log.success("Claude Code is authenticated and ready")
```

- [ ] **Step 4: Add `--agent` to `loop()`**

Add parameter:

```python
agent: AgentBackend = typer.Option(
    AgentBackend.claude,
    "--agent",
    case_sensitive=False,
    help="Agent CLI used for plan/prover/review phases.",
),
```

Update preflight:

```python
_preflight(resolved, state_dir, dry_run, agent)
```

Add to config:

```python
"Agent": agent.value,
```

- [ ] **Step 5: Pass backend through phase calls**

Replace `run_claude(...)` in `loop.py` with `run_agent(..., backend=agent, ...)`.

Add `backend: AgentBackend = AgentBackend.claude` to `_run_single_prover()`, `backend: AgentBackend` to `_run_parallel_provers()`, and `backend: AgentBackend` to `_run_review_phase()`. Pass the selected backend into every call and process pool submission.

- [ ] **Step 6: Run loop tests**

Run:

```bash
python3 -m pytest tests/test_loop_agent_cli.py tests/test_runner_agent_cli.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit loop wiring**

```bash
git add src/archon/commands/loop.py tests/test_loop_agent_cli.py
git commit -m "feat: select agent cli in loop"
```

### Task 5: Configure Lean LSP MCP for Codex and OpenCode

**Files:**
- Modify: `src/archon/commands/init.py`
- Modify: `src/archon/commands/doctor.py`
- Create: `tests/test_init_doctor_agent_cli.py`

- [ ] **Step 1: Write MCP setup tests**

Create `tests/test_init_doctor_agent_cli.py`:

```python
import json

from archon.types import AgentBackend


def test_init_registers_codex_mcp_when_codex_backend(monkeypatch, tmp_path):
    from archon.commands import init

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr(init, "_run", fake_run)
    monkeypatch.setattr(init, "_data_path", lambda sub="": tmp_path / sub)

    init._step3_lean_lsp_mcp(tmp_path, fresh=True, backend=AgentBackend.codex)

    assert calls == [[
        "codex",
        "mcp",
        "add",
        "archon-lean-lsp",
        "--",
        "uv",
        "run",
        "--directory",
        str(tmp_path / "tools/lean-lsp-mcp"),
        "lean-lsp-mcp",
    ]]


def test_init_writes_opencode_mcp_config(monkeypatch, tmp_path):
    from archon.commands import init

    config_path = tmp_path / "opencode.json"
    monkeypatch.setattr(init, "_opencode_config_path", lambda: config_path)

    lean_lsp_dir = tmp_path / "tools/lean-lsp-mcp"
    init._write_opencode_mcp_config(lean_lsp_dir)

    data = json.loads(config_path.read_text())
    assert data["$schema"] == "https://opencode.ai/config.json"
    assert data["mcp"]["archon-lean-lsp"] == {
        "type": "local",
        "command": ["uv", "run", "--directory", str(lean_lsp_dir), "lean-lsp-mcp"],
        "enabled": True,
        "timeout": 30000,
    }


def test_doctor_reports_opencode_mcp_from_config(monkeypatch, tmp_path):
    from archon.commands import doctor

    config_path = tmp_path / "opencode.json"
    config_path.write_text(json.dumps({"mcp": {"archon-lean-lsp": {"enabled": True}}}))
    monkeypatch.setattr(doctor, "_opencode_config_path", lambda: config_path)
    monkeypatch.setattr(doctor, "_has", lambda binary: binary == "opencode")

    rows = doctor._check_opencode_mcp()

    assert rows == [("opencode archon-lean-lsp", "ok", "configured")]
```

- [ ] **Step 2: Add OpenCode config helper to init**

In `src/archon/commands/init.py`, import `os` if needed and add:

```python
def _opencode_config_path() -> Path:
    return Path.home() / ".config" / "opencode" / "opencode.json"


def _write_opencode_mcp_config(lean_lsp_dir: Path) -> None:
    config_path = _opencode_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    data: dict = {}
    if config_path.exists():
        data = _read_json(config_path)

    data.setdefault("$schema", "https://opencode.ai/config.json")
    mcp = data.setdefault("mcp", {})
    mcp["archon-lean-lsp"] = {
        "type": "local",
        "command": ["uv", "run", "--directory", str(lean_lsp_dir), "lean-lsp-mcp"],
        "enabled": True,
        "timeout": 30000,
    }

    config_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
```

- [ ] **Step 3: Extend `_step3_lean_lsp_mcp()`**

Modify signature:

```python
def _step3_lean_lsp_mcp(
    project_path: Path,
    fresh: bool,
    backend: AgentBackend = AgentBackend.claude,
) -> None:
```

Import the enum:

```python
from archon.types import AgentBackend
```

At the start of `_step3_lean_lsp_mcp()`, after `lean_lsp_dir = _data_path("tools/lean-lsp-mcp")`, add:

```python
    if backend == AgentBackend.codex:
        r = _run(
            ["codex", "mcp", "add", "archon-lean-lsp", "--",
             "uv", "run", "--directory", str(lean_lsp_dir), "lean-lsp-mcp"],
            cwd=project_path,
        )
        output = r.stdout + r.stderr
        if r.returncode == 0 or "already" in output.lower():
            log.success("codex archon-lean-lsp configured")
        else:
            log.error(f"Failed to add codex archon-lean-lsp: {output.strip()}")
        return

    if backend == AgentBackend.opencode:
        _write_opencode_mcp_config(lean_lsp_dir)
        log.success(f"opencode archon-lean-lsp configured in {_opencode_config_path()}")
        return
```

- [ ] **Step 4: Add `--agent` to `init()`**

Add parameter:

```python
agent: AgentBackend = typer.Option(
    AgentBackend.claude,
    "--agent",
    case_sensitive=False,
    help="Agent CLI setup to configure for this project.",
),
```

Pass `backend=agent` to every `_step3_lean_lsp_mcp(...)` call.

- [ ] **Step 5: Add doctor checks**

In `src/archon/commands/doctor.py`, add:

```python
def _opencode_config_path() -> Path:
    return Path.home() / ".config" / "opencode" / "opencode.json"


def _check_codex_mcp() -> list[tuple[str, str, str]]:
    if not _has("codex"):
        return [("codex archon-lean-lsp", "skipped", "codex not installed")]

    r = _run(["codex", "mcp", "list"])
    output = (r.stdout or "") + (r.stderr or "")
    if "archon-lean-lsp" in output:
        return [("codex archon-lean-lsp", "ok", "configured")]
    return [("codex archon-lean-lsp", "warning", "not found")]


def _check_opencode_mcp() -> list[tuple[str, str, str]]:
    if not _has("opencode"):
        return [("opencode archon-lean-lsp", "skipped", "opencode not installed")]

    data = _read_json(_opencode_config_path())
    server = (data.get("mcp") or {}).get("archon-lean-lsp")
    if isinstance(server, dict) and server.get("enabled", True):
        return [("opencode archon-lean-lsp", "ok", "configured")]
    return [("opencode archon-lean-lsp", "warning", "not found")]
```

In `doctor()`, after the Claude project config section, add:

```python
    section = _check_codex_mcp() + _check_opencode_mcp()
    log.results_table(section, title="Agent CLI Config")
    all_rows.extend(section)
```

- [ ] **Step 6: Run MCP tests**

Run:

```bash
python3 -m pytest tests/test_init_doctor_agent_cli.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit MCP support**

```bash
git add src/archon/commands/init.py src/archon/commands/doctor.py tests/test_init_doctor_agent_cli.py
git commit -m "feat: configure agent cli mcp"
```

### Task 6: Acceptance Smoke Tests

**Files:**
- No new files expected.

- [ ] **Step 1: Run full unit suite**

Run:

```bash
python3 -m pytest -q
```

Expected: PASS.

- [ ] **Step 2: Run default-path non-regression dry-run**

Run:

```bash
python3 -m archon.cli loop . --dry-run --no-dashboard --max-iterations 1
```

Expected: existing default path still prints plan/prover prompts. This is a guardrail only; the feature under test is Codex/OpenCode mode.

- [ ] **Step 3: Run Codex dry-run**

Run:

```bash
python3 -m archon.cli loop . --agent codex --dry-run --no-dashboard --max-iterations 1
```

Expected: prints prompts and displays `Agent: codex` in the loop configuration.

- [ ] **Step 4: Run OpenCode dry-run**

Run:

```bash
python3 -m archon.cli loop . --agent opencode --dry-run --no-dashboard --max-iterations 1
```

Expected: prints prompts and displays `Agent: opencode` in the loop configuration.

- [ ] **Step 5: Run real Codex runner smoke**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
from archon.runner import run_agent
from archon.types import AgentBackend

log_base = Path("/tmp/archon-codex-real")
ok = run_agent("reply exactly OK", backend=AgentBackend.codex, cwd=Path.cwd(), log_base=log_base)
print("ok=", ok)
print(Path(str(log_base) + ".jsonl").read_text())
PY
```

Expected: output includes `OK`, `ok= True`, and JSONL rows with `event` values `text` and `session_end`.

- [ ] **Step 6: Run real OpenCode runner smoke**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
from archon.runner import run_agent
from archon.types import AgentBackend

log_base = Path("/tmp/archon-opencode-real")
ok = run_agent("reply exactly OK", backend=AgentBackend.opencode, cwd=Path.cwd(), log_base=log_base)
print("ok=", ok)
print(Path(str(log_base) + ".jsonl").read_text())
PY
```

Expected: output includes `OK`, `ok= True`, and JSONL rows with `event` values `text` and `session_end`.

- [ ] **Step 7: Run OpenCode tool-event smoke**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
from archon.runner import run_agent
from archon.types import AgentBackend

log_base = Path("/tmp/archon-opencode-tool")
ok = run_agent("Run pwd with the shell tool, then reply done.", backend=AgentBackend.opencode, cwd=Path.cwd(), log_base=log_base)
print("ok=", ok)
print(Path(str(log_base) + ".jsonl").read_text())
PY
```

Expected: JSONL includes `tool_call`, `tool_result`, `text`, and `session_end`.

- [ ] **Step 8: Commit smoke-test fixes if needed**

If smoke tests reveal small fixes, apply them and commit:

```bash
git add src/archon tests
git commit -m "fix: stabilize agent cli compatibility"
```

## Test Strategy

Use three layers of tests.

1. **Pure compatibility tests**
   - Codex JSON maps into Archon's `text` and `session_end` rows.
   - OpenCode JSON maps into Archon's `text`, `tool_call`, `tool_result`, and `session_end` rows.
   - Non-JSON warning lines from either CLI are ignored.

2. **CLI/plumbing tests**
   - `loop._preflight(..., backend=AgentBackend.codex)` checks `codex`, not `claude`.
   - `loop._preflight(..., backend=AgentBackend.opencode)` checks `opencode`, not `claude`.
   - `_run_single_prover(..., backend=...)` passes the selected backend to `run_agent()`.
   - `init._step3_lean_lsp_mcp(..., backend=AgentBackend.codex)` constructs the expected `codex mcp add` command.
   - `init._write_opencode_mcp_config()` writes the OpenCode MCP config shape validated by OpenCode's local schema.
   - Doctor reports Codex and OpenCode MCP readiness without requiring a real Lean project.

3. **Manual smoke tests**
   - `archon loop --agent codex --dry-run`.
   - `archon loop --agent opencode --dry-run`.
   - Real `run_agent(..., backend=codex)` with `reply exactly OK`.
   - Real `run_agent(..., backend=opencode)` with `reply exactly OK`.
   - Real OpenCode tool-use smoke so `extract-attempts.py` will still receive tool rows.

Do not run a real multi-hour Lean proving loop as part of this change. The first real Lean project trial should be a separate acceptance run after the compatibility layer passes the smoke tests.

## Self-Review

- Spec coverage: The plan now covers both Codex and OpenCode execution, MCP setup, and dashboard-compatible logs.
- Placeholder scan: No unresolved placeholder markers remain.
- Type consistency: `AgentBackend` is defined once in `src/archon/types.py` and used consistently in runner, loop, and init.
- Scope check: The plan intentionally avoids native Codex/OpenCode plugin migration, prompt rewrites, proof strategy changes, and dashboard API changes.
