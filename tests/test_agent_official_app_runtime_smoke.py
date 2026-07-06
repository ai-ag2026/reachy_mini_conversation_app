"""No-side-effect official-app runtime smoke for the AGENT v0.1 adapter."""

from __future__ import annotations
import sys
import importlib
from types import ModuleType
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

import reachy_mini_conversation_app.config as config_mod
from reachy_mini_conversation_app.config import LOCAL_BACKEND
from reachy_mini_conversation_app.handler_factory import build_conversation_handler
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.agent_voice_handler import AgentVoiceHandler, FakeAudioTtsClient, FakeTextAgentClient


def _reload_core_tools() -> ModuleType:
    """Reload tool registry after test config changes."""
    for module_name in list(sys.modules):
        if module_name.startswith("reachy_mini_conversation_app.tools."):
            sys.modules.pop(module_name, None)
    sys.modules.pop("reachy_mini_conversation_app.tools.core_tools", None)
    core_tools_mod = importlib.import_module("reachy_mini_conversation_app.tools.core_tools")
    core_tools_mod.initialize_tools(force=True)
    return core_tools_mod


def test_agent_backend_builds_handler_with_injected_clients() -> None:
    """The official app factory should build a AGENT handler without live HTTP when clients are injected."""
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())

    handler = build_conversation_handler(
        deps,
        backend_provider=LOCAL_BACKEND,
        agent_client=FakeTextAgentClient("Runtime smoke ok."),
        tts_client=FakeAudioTtsClient(sample_rate=24_000, audio=np.zeros(16, dtype=np.int16)),
    )

    assert isinstance(handler, AgentVoiceHandler)


def test_agent_profile_loads_only_agent_v01_tools(tmp_path: Path, monkeypatch) -> None:
    """A AGENT profile should load the curated v0.1 tool surface through the official registry."""
    profile_name = "agent_v01_runtime_smoke"
    profiles_root = tmp_path / "profiles"
    profile_dir = profiles_root / profile_name
    profile_dir.mkdir(parents=True)
    (profile_dir / "instructions.txt").write_text("AGENT remains the brain.\n", encoding="utf-8")
    (profile_dir / "tools.txt").write_text("agent_vision\nagent_safe_movement\n", encoding="utf-8")

    monkeypatch.setattr(config_mod.config, "REACHY_MINI_CUSTOM_PROFILE", profile_name)
    monkeypatch.setattr(config_mod.config, "PROFILES_DIRECTORY", profiles_root)
    monkeypatch.setattr(config_mod.config, "TOOLS_DIRECTORY", None)
    monkeypatch.setattr(config_mod.config, "AUTOLOAD_EXTERNAL_TOOLS", False)

    core_tools_mod = _reload_core_tools()

    assert "agent_vision" in core_tools_mod.ALL_TOOLS
    assert "agent_safe_movement" in core_tools_mod.ALL_TOOLS
    assert "camera" not in core_tools_mod.ALL_TOOLS
    assert "move_head" not in core_tools_mod.ALL_TOOLS
    assert core_tools_mod.ALL_TOOLS["agent_vision"].name == "agent_vision"
    assert core_tools_mod.ALL_TOOLS["agent_safe_movement"].name == "agent_safe_movement"
