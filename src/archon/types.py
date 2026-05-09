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
