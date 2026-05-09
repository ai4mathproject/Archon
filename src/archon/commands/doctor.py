"""Verify Archon setup for a project."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from importlib import resources
from pathlib import Path

import typer

from archon import log
from archon.runner import claude_env


def _has(binary: str) -> bool:
    return shutil.which(binary) is not None


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    if cmd and cmd[0] == "claude" and "env" not in kwargs:
        kwargs["env"] = claude_env()
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def _version(cmd: list[str]) -> str:
    try:
        r = _run(cmd)
        return (r.stdout or r.stderr).strip().splitlines()[0]
    except Exception:
        return "unknown"


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _data_path(sub_path: str = "") -> Path:
    root = resources.files("archon").joinpath(".archon-src")
    if sub_path:
        return Path(str(root.joinpath(sub_path)))
    return Path(str(root))


# ── individual checks ─────────────────────────────────────────────────


def _check_lean_toolchain() -> list[tuple[str, str, str]]:
    """Check elan, lean, lake. Returns list of (name, status, detail)."""
    rows: list[tuple[str, str, str]] = []
    for tool in ("elan", "lean", "lake"):
        if _has(tool):
            rows.append((tool, "ok", _version([tool, "--version"])))
        else:
            rows.append((tool, "error", "not found in PATH"))
    return rows


def _check_python_tools() -> list[tuple[str, str, str]]:
    """Check Python, uv, tmux."""
    rows: list[tuple[str, str, str]] = []

    v = sys.version_info
    if v >= (3, 10):
        rows.append(("python", "ok", f"{v.major}.{v.minor}.{v.micro}"))
    else:
        rows.append(("python", "error", f"{v.major}.{v.minor} (need 3.10+)"))

    if _has("uv"):
        rows.append(("uv", "ok", _version(["uv", "--version"])))
    else:
        rows.append(("uv", "error", "not found"))

    return rows


def _check_claude_code() -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    if _has("claude"):
        rows.append(("claude", "ok", _version(["claude", "--version"])))
        # Test auth
        r = _run(["claude", "-p", "reply with OK", "--no-session-persistence"])
        if r.returncode == 0:
            rows.append(("claude auth", "ok", "authenticated"))
        else:
            rows.append(("claude auth", "error", "not authenticated — check claude auth/network"))
    else:
        rows.append(("claude", "error", "not installed — run: archon setup"))
    return rows


def _check_api_keys() -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    keys = {
        "OPENAI_API_KEY": "OpenAI",
        "GEMINI_API_KEY": "Gemini",
        "OPENROUTER_API_KEY": "OpenRouter",
    }
    found_any = False
    for var, label in keys.items():
        if os.environ.get(var):
            rows.append((f"{label} key", "ok", f"${var} is set"))
            found_any = True
        else:
            rows.append((f"{label} key", "skipped", "not set"))
    if not found_any:
        rows.append(("informal agent", "warning", "no API keys — informal agent won't work"))
    return rows


def _check_package_data() -> list[tuple[str, str, str]]:
    """Check that bundled package data is accessible."""
    rows: list[tuple[str, str, str]] = []

    checks = {
        "templates": "archon-template/PROGRESS.md",
        "prompts": "prompts",
        "skills": "skills/lean4/.claude-plugin/plugin.json",
        "scripts": "scripts",
        "tools": "tools",
    }
    for name, sub in checks.items():
        p = _data_path(sub)
        if p.exists():
            rows.append((f"data: {name}", "ok", str(p.parent if p.is_file() else p)))
        else:
            rows.append((f"data: {name}", "error", f"not found at {p}"))
    return rows


def _check_project_state(project_path: Path) -> list[tuple[str, str, str]]:
    """Check .archon/ state directory and contents."""
    rows: list[tuple[str, str, str]] = []
    state_dir = project_path / ".archon"

    if not state_dir.is_dir():
        rows.append((".archon/", "error", f"not found — run: archon init {project_path}"))
        return rows
    rows.append((".archon/", "ok", str(state_dir)))

    # Required state files
    for name in ("PROGRESS.md", "CLAUDE.md"):
        f = state_dir / name
        if f.exists():
            rows.append((name, "ok", f"{f.stat().st_size:,} bytes"))
        else:
            rows.append((name, "error", "missing"))

    # Prompts directory
    prompts_dir = state_dir / "prompts"
    if prompts_dir.is_dir():
        prompt_count = len(list(prompts_dir.glob("*.md")))
        rows.append(("prompts/", "ok", f"{prompt_count} prompt file(s)"))
    else:
        rows.append(("prompts/", "error", "missing"))

    # Stage
    progress = state_dir / "PROGRESS.md"
    if progress.exists():
        lines = progress.read_text().splitlines()
        stage = "unknown"
        for i, line in enumerate(lines):
            if line.startswith("## Current Stage"):
                if i + 1 < len(lines):
                    stage = lines[i + 1].strip()
                break
        rows.append(("current stage", "ok", stage))

    # Proof journal
    journal = state_dir / "proof-journal" / "sessions"
    if journal.is_dir():
        sessions = len(list(journal.glob("session_*")))
        rows.append(("proof journal", "ok", f"{sessions} session(s)"))
    else:
        rows.append(("proof journal", "skipped", "no sessions yet"))

    # Logs
    log_dir = state_dir / "logs"
    if log_dir.is_dir():
        iters = len(list(log_dir.glob("iter-*")))
        rows.append(("logs/", "ok", f"{iters} iteration(s)"))
    else:
        rows.append(("logs/", "skipped", "no iterations yet"))

    return rows


def _check_project_claude(project_path: Path) -> list[tuple[str, str, str]]:
    """Check .claude/ directory (skills, tools, MCP)."""
    rows: list[tuple[str, str, str]] = []
    claude_dir = project_path / ".claude"

    if not claude_dir.is_dir():
        rows.append((".claude/", "warning", "not found — skills may not be installed"))
        return rows

    # Skills
    skills_dir = claude_dir / "skills"
    if skills_dir.is_dir():
        skill_count = len([d for d in skills_dir.iterdir() if d.is_dir()])
        rows.append(("user skills", "ok", f"{skill_count} skill(s)"))
    else:
        rows.append(("user skills", "skipped", "none"))

    # Informal agent tool
    agent = claude_dir / "tools" / "archon-informal-agent.py"
    if agent.exists():
        if agent.is_symlink():
            rows.append(("informal agent", "ok", f"symlink → {agent.resolve()}"))
        else:
            rows.append(("informal agent", "ok", "copied"))
    else:
        rows.append(("informal agent", "warning", "not found"))

    # MCP config
    mcp_json = project_path / ".mcp.json"
    if mcp_json.exists():
        data = _read_json(mcp_json)
        servers = list(data.get("mcpServers", {}).keys())
        archon_lsp = [s for s in servers if "archon" in s.lower()]
        if archon_lsp:
            rows.append(("archon-lean-lsp", "ok", ", ".join(archon_lsp)))
        else:
            rows.append(("archon-lean-lsp", "warning", f"not found (servers: {', '.join(servers) or 'none'})"))
    else:
        rows.append(("MCP config", "warning", ".mcp.json not found"))

    return rows


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


def _check_sorry_count(project_path: Path) -> list[tuple[str, str, str]]:
    """Run sorry_analyzer if available."""
    rows: list[tuple[str, str, str]] = []
    analyzer = _data_path("skills/lean4/lib/scripts/sorry_analyzer.py")
    if not analyzer.exists():
        rows.append(("sorry count", "skipped", "sorry_analyzer.py not found"))
        return rows

    r = _run([sys.executable, str(analyzer), str(project_path), "--format=summary"])
    if r.returncode == 0 and r.stdout.strip():
        # Extract total from summary output
        rows.append(("sorry count", "ok", r.stdout.strip().splitlines()[-1]))
    else:
        rows.append(("sorry count", "skipped", "could not run analyzer"))

    return rows


# ── main command ──────────────────────────────────────────────────────


def doctor(
    project_path: str = typer.Argument(".", help="Path to Lean project"),
    skip_auth: bool = typer.Option(
        False, "--skip-auth",
        help="Skip Claude Code authentication test (faster).",
    ),
) -> None:
    """Verify the full Archon setup.

    Checks system tools, package data, project state, skills, MCP,
    and reports any issues.
    """
    resolved = Path(project_path).resolve()

    all_rows: list[tuple[str, str, str]] = []
    has_error = False

    # System tools
    log.header("System Tools")
    section = _check_lean_toolchain() + _check_python_tools()
    if not skip_auth:
        section += _check_claude_code()
    else:
        if _has("claude"):
            section.append(("claude", "ok", f"{_version(['claude', '--version'])} (auth skipped)"))
        else:
            section.append(("claude", "error", "not installed"))
    log.results_table(section, title="System")
    all_rows.extend(section)

    # API keys
    log.header("API Keys (optional)")
    section = _check_api_keys()
    log.results_table(section, title="External Models")
    all_rows.extend(section)

    # Package data
    log.header("Package Data")
    section = _check_package_data()
    log.results_table(section, title="Bundled Data")
    all_rows.extend(section)

    # Project state
    log.header(f"Project: {resolved.name}")
    section = _check_project_state(resolved)
    log.results_table(section, title="State (.archon/)")
    all_rows.extend(section)

    section = _check_project_claude(resolved)
    log.results_table(section, title="Claude Config (.claude/)")
    all_rows.extend(section)

    section = _check_codex_mcp() + _check_opencode_mcp()
    log.results_table(section, title="Agent CLI Config")
    all_rows.extend(section)

    # Sorry count
    if (resolved / ".archon").is_dir():
        section = _check_sorry_count(resolved)
        if section:
            log.results_table(section, title="Lean Project")
            all_rows.extend(section)

    # Summary
    errors = sum(1 for _, s, _ in all_rows if s == "error")
    warnings = sum(1 for _, s, _ in all_rows if s == "warning")

    log.rule()
    if errors:
        log.error(f"{errors} error(s), {warnings} warning(s) — fix errors before running.")
        raise typer.Exit(1)
    elif warnings:
        log.warn(f"All clear with {warnings} warning(s).")
    else:
        log.success("All checks passed.")
