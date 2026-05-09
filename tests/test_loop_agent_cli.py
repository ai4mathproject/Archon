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
