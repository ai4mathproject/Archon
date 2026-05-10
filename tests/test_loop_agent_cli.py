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


def test_serial_prover_records_error_when_agent_fails(monkeypatch, tmp_path):
    from archon.commands import loop

    state_dir = tmp_path / ".archon"
    state_dir.mkdir()
    (state_dir / "PROGRESS.md").write_text(
        "## Current Stage\nprover\n\n## Current Objectives\n\n1. **Base.lean** — fill sorry\n"
    )
    (tmp_path / "Base.lean").write_text("theorem t : True := by\n  trivial\n")

    iter_dir = state_dir / "logs" / "iter-001"
    iter_dir.mkdir(parents=True)
    iter_meta = iter_dir / "meta.json"

    monkeypatch.setattr(loop, "run_agent", lambda *args, **kwargs: False)

    ok = loop._run_serial_prover(
        "proj",
        tmp_path,
        state_dir,
        "prover",
        iter_dir,
        iter_meta,
        verbose_logs=False,
        dry_run=False,
        backend=AgentBackend.opencode,
    )

    assert ok is False
    assert '"status": "error"' in iter_meta.read_text()


def test_plan_agent_records_error_when_agent_fails(monkeypatch, tmp_path):
    from archon.commands import loop

    state_dir = tmp_path / ".archon"
    state_dir.mkdir()
    (state_dir / "PROGRESS.md").write_text("## Current Stage\nprover\n")

    iter_dir = state_dir / "logs" / "iter-001"
    iter_dir.mkdir(parents=True)
    iter_meta = iter_dir / "meta.json"

    monkeypatch.setattr(loop, "run_agent", lambda *args, **kwargs: False)

    ok = loop._run_plan_agent(
        "proj",
        tmp_path,
        state_dir,
        "prover",
        iter_dir,
        iter_meta,
        verbose_logs=False,
        dry_run=False,
        backend=AgentBackend.opencode,
    )

    assert ok is False
    assert '"plan": {' in iter_meta.read_text()
    assert '"status": "error"' in iter_meta.read_text()
