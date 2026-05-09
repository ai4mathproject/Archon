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


def test_doctor_reports_codex_mcp(monkeypatch):
    from archon.commands import doctor

    def fake_run(cmd, **kwargs):
        class Result:
            returncode = 0
            stdout = "archon-lean-lsp  uv run --directory /x lean-lsp-mcp\n"
            stderr = ""

        return Result()

    monkeypatch.setattr(doctor, "_run", fake_run)
    monkeypatch.setattr(doctor, "_has", lambda binary: binary == "codex")

    rows = doctor._check_codex_mcp()

    assert rows == [("codex archon-lean-lsp", "ok", "configured")]


def test_doctor_reports_opencode_mcp_from_config(monkeypatch, tmp_path):
    from archon.commands import doctor

    config_path = tmp_path / "opencode.json"
    config_path.write_text(json.dumps({"mcp": {"archon-lean-lsp": {"enabled": True}}}))
    monkeypatch.setattr(doctor, "_opencode_config_path", lambda: config_path)
    monkeypatch.setattr(doctor, "_has", lambda binary: binary == "opencode")

    rows = doctor._check_opencode_mcp()

    assert rows == [("opencode archon-lean-lsp", "ok", "configured")]


def test_init_required_cli_uses_selected_backend(monkeypatch):
    from archon.commands import init

    seen = []

    def fake_has(binary):
        seen.append(binary)
        return True

    monkeypatch.setattr(init, "_has", fake_has)

    init._require_agent_cli(AgentBackend.codex)
    init._require_agent_cli(AgentBackend.opencode)

    assert seen == ["codex", "opencode"]


def test_init_agent_specific_setup_skips_claude_plugins_for_opencode(monkeypatch, tmp_path):
    from archon.commands import init

    calls = []
    monkeypatch.setattr(init, "_step4_skills", lambda *args, **kwargs: calls.append("skills"))
    monkeypatch.setattr(init, "_step5_disable_conflicting_plugins", lambda *args, **kwargs: calls.append("disable"))

    init._agent_specific_setup(tmp_path, fresh=True, backend=AgentBackend.opencode)

    assert calls == []


def test_initial_setup_uses_selected_non_claude_agent(monkeypatch, tmp_path):
    from archon.commands import init

    state_dir = tmp_path / ".archon"
    state_dir.mkdir()
    (state_dir / "PROGRESS.md").write_text("## Current Stage\ninit\n")

    calls = []

    def fake_run_agent(prompt, *, backend, cwd, **kwargs):
        calls.append((prompt, backend, cwd))
        (state_dir / "PROGRESS.md").write_text("## Current Stage\nprover\n")
        return True

    monkeypatch.setattr(init, "run_agent", fake_run_agent)

    init._step6_initial_agent(tmp_path, state_dir, backend=AgentBackend.codex)

    assert len(calls) == 1
    assert calls[0][1] == AgentBackend.codex
    assert calls[0][2] == tmp_path
