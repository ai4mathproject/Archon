"""Claude Code runner with structured JSONL logging.

Wraps `claude -p` with stream-json parsing, cost tracking, and log output.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
import os

from archon import log
from archon.types import AgentBackend


def claude_env() -> dict[str, str]:
    """Environment for Claude Code calls that prefers subscription auth."""
    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)
    return env


# ── prompt building ───────────────────────────────────────────────────


def build_plan_prompt(
    project_name: str, project_path: Path, state_dir: Path, stage: str,
) -> str:
    return dedent(f"""\
        You are the plan agent for project '{project_name}'. Current stage: {stage}.
        Project directory: {project_path}
        Project state directory: {state_dir}
        Read {state_dir}/CLAUDE.md for your role, then read {state_dir}/prompts/plan.md and {state_dir}/PROGRESS.md.
        All state files (PROGRESS.md, task_pending.md, task_done.md, USER_HINTS.md, task_results/) are in {state_dir}/.
        The .lean files are in {project_path}/.""")


def build_prover_prompt(
    project_name: str, project_path: Path, state_dir: Path, stage: str,
) -> str:
    return dedent(f"""\
        You are the prover agent for project '{project_name}'. Current stage: {stage}.
        Project directory: {project_path}
        Project state directory: {state_dir}
        Read {state_dir}/CLAUDE.md for your role, then read {state_dir}/prompts/prover-{stage}.md and {state_dir}/PROGRESS.md.
        All state files are in {state_dir}/. The .lean files are in {project_path}/.""")


def build_parallel_prover_prompt(
    project_name: str, project_path: Path, state_dir: Path, stage: str,
) -> str:
    return dedent(f"""\
        You are a prover agent for project '{project_name}'. Current stage: {stage}.
        Project directory: {project_path}
        Project state directory: {state_dir}
        Read {state_dir}/CLAUDE.md for your role, then read {state_dir}/prompts/prover-{stage}.md and {state_dir}/PROGRESS.md.
        Check your .lean file for /- USER: ... -/ comments for file-specific hints.

        IMPORTANT:
        - You own ONLY the file assigned below. Do NOT edit any other .lean file.
        - Write your results to {state_dir}/task_results/<your_file>.md when done.
        - Do NOT edit PROGRESS.md, task_pending.md, or task_done.md.
        - Missing Mathlib infrastructure is NEVER a valid reason to leave a sorry.
        - NEVER revert to a bare sorry. Always leave your partial proof attempt in the code.""")


def build_review_prompt(
    project_name: str, project_path: Path, state_dir: Path, stage: str,
    session_num: int, session_dir: Path, attempts_file: Path,
    combined_prover_log: Path,
) -> str:
    return dedent(f"""\
        You are the review agent for project '{project_name}'. Current stage: {stage}.
        Project directory: {project_path}
        Project state directory: {state_dir}
        Read {state_dir}/CLAUDE.md for your role, then read {state_dir}/prompts/review.md.
        Session number: {session_num}.
        Pre-processed attempt data: {attempts_file} (READ THIS FIRST).
        Prover log: {combined_prover_log}

        CRITICAL — Write your output files to EXACTLY these paths:
          {session_dir}/milestones.jsonl
          {session_dir}/summary.md
          {session_dir}/recommendations.md
          {state_dir}/PROJECT_STATUS.md""")


# ── JSONL log stream parser (embedded Python script) ──────────────────

# This is the Python script that gets piped Claude's stream-json output.
# It parses events, writes structured JSONL, and prints cost summaries.
_STREAM_PARSER = r'''
import sys, json, datetime

VERBOSE = '{verbose}' == 'True'
RAW = open('{raw_log}', 'a') if VERBOSE else None
JSONL = open('{jsonl}', 'a')

def emit(event_type, **fields):
    row = {{'ts': datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00', 'Z'), 'event': event_type, **fields}}
    JSONL.write(json.dumps(row) + '\n')
    JSONL.flush()

def terminal(s):
    print(s, flush=True)

last_result = ''

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    if RAW:
        RAW.write(line + '\n')
        RAW.flush()

    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        continue

    t = obj.get('type', '')

    if t == 'assistant' and 'message' in obj:
        msg = obj['message']
        if not isinstance(msg, dict):
            continue
        for block in msg.get('content', []):
            bt = block.get('type', '')
            if bt == 'thinking':
                thinking = block.get('thinking', '').strip()
                if thinking:
                    emit('thinking', content=thinking)
            elif bt == 'text':
                text = block.get('text', '').strip()
                if text:
                    emit('text', content=text)
                    last_result = text
            elif bt == 'tool_use':
                name = block.get('name', '?')
                inp = block.get('input', {{}})
                emit('tool_call', tool=name, input=inp)

    elif t == 'user' and 'message' in obj:
        msg = obj['message']
        if not isinstance(msg, dict):
            continue
        for block in msg.get('content', []):
            if block.get('type') == 'tool_result':
                content = block.get('content', '')
                if isinstance(content, str):
                    emit('tool_result', content=content)
                elif isinstance(content, list):
                    texts = [p.get('text','') for p in content if isinstance(p,dict) and p.get('type')=='text']
                    emit('tool_result', content='\n'.join(texts))

    elif t == 'result':
        cost = obj.get('total_cost_usd', 0) or obj.get('cost_usd', 0) or 0
        duration = obj.get('duration_ms', 0) or 0
        turns = obj.get('num_turns', 0) or 0
        session_id = obj.get('session_id', '') or ''
        result = obj.get('result', '')
        usage = obj.get('usage', {{}}) or {{}}
        model_usage = obj.get('modelUsage', {{}}) or {{}}
        summary = result if isinstance(result, str) and result else last_result

        emit('session_end',
            session_id=session_id,
            total_cost_usd=cost,
            duration_ms=duration,
            duration_api_ms=usage.get('duration_api_ms', 0) or 0,
            num_turns=turns,
            input_tokens=usage.get('input_tokens', 0) or 0,
            output_tokens=usage.get('output_tokens', 0) or 0,
            cache_read_input_tokens=usage.get('cache_read_input_tokens', 0) or 0,
            cache_creation_input_tokens=usage.get('cache_creation_input_tokens', 0) or 0,
            model_usage=model_usage,
            summary=summary,
        )

        if summary:
            terminal(summary)
        parts = []
        if duration:  parts.append(f'{{duration/60000:.1f}}min')
        if cost:      parts.append(f'${{cost:.4f}}')
        if usage.get('input_tokens') or usage.get('output_tokens'):
            parts.append(f'in={{usage.get("input_tokens",0)}} out={{usage.get("output_tokens",0)}}')
        if turns:     parts.append(f'turns={{turns}}')
        if parts:
            terminal(f'[COST] {{" | ".join(parts)}}')

JSONL.close()
if RAW: RAW.close()
'''


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
        if part.get("reason") not in (None, "stop"):
            return []
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


# ── run_claude ────────────────────────────────────────────────────────


def run_claude(
    prompt: str,
    *,
    cwd: Path,
    log_base: Path | None = None,
    verbose_logs: bool = False,
    extra_args: list[str] | None = None,
) -> bool:
    """Run `claude -p` with optional JSONL logging.

    Args:
        prompt: The prompt string to pass to Claude.
        cwd: Working directory (project path).
        log_base: If provided, enables JSONL logging to {log_base}.jsonl
                  (and optionally {log_base}.raw.jsonl).
        verbose_logs: If True, also write raw stream events.
        extra_args: Additional arguments to pass to claude.

    Returns:
        True if claude exited successfully, False otherwise.
    """
    claude_cmd = [
        "claude", "-p", prompt,
        "--dangerously-skip-permissions", "--permission-mode", "bypassPermissions",
    ]
    if extra_args:
        claude_cmd.extend(extra_args)

    if log_base is not None:
        log_base.parent.mkdir(parents=True, exist_ok=True)
        jsonl = f"{log_base}.jsonl"
        raw_log = f"{log_base}.raw.jsonl"

        claude_cmd.extend(["--verbose", "--output-format", "stream-json"])

        stderr_dest = raw_log if verbose_logs else os.devnull

        parser_script = _STREAM_PARSER.format(
            verbose=str(verbose_logs),
            raw_log=raw_log,
            jsonl=jsonl,
        )

        with open(stderr_dest, "a") as stderr_file:
            claude_proc = subprocess.Popen(
                claude_cmd,
                stdout=subprocess.PIPE,
                stderr=stderr_file,
                cwd=cwd,
                env=claude_env(),
            )
            parser_proc = subprocess.Popen(
                [sys.executable, "-u", "-c", parser_script],
                stdin=claude_proc.stdout,
                cwd=cwd,
            )
            claude_proc.stdout.close()  # allow SIGPIPE
            parser_proc.wait()
            claude_proc.wait()

        return claude_proc.returncode == 0
    else:
        r = subprocess.run(claude_cmd, cwd=cwd, env=claude_env())
        return r.returncode == 0


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
        return run_codex(
            prompt,
            cwd=cwd,
            log_base=log_base,
            verbose_logs=verbose_logs,
            extra_args=extra_args,
        )
    if backend == AgentBackend.opencode:
        return run_opencode(
            prompt,
            cwd=cwd,
            log_base=log_base,
            verbose_logs=verbose_logs,
            extra_args=extra_args,
        )
    return run_claude(
        prompt,
        cwd=cwd,
        log_base=log_base,
        verbose_logs=verbose_logs,
        extra_args=extra_args,
    )
