# ruff: noqa: D103
from __future__ import annotations

import numpy as np
import pytest

from reachy_mini_conversation_app.agent_clients import HermesVoiceClient, QwenVoiceTtsClient
from reachy_mini_conversation_app.handler_factory import build_conversation_handler
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.agent_voice_handler import AgentVoiceHandler, FakeAudioTtsClient, FakeTextAgentClient


class _FakeMovementManager:
    pass


def test_factory_builds_agent_handler_only_with_injected_clients() -> None:
    audio = np.zeros(4, dtype=np.int16)

    handler = build_conversation_handler(
        ToolDependencies(reachy_mini=object(), movement_manager=_FakeMovementManager()),
        backend_provider="local",
        agent_client=FakeTextAgentClient(reply="ok"),
        tts_client=FakeAudioTtsClient(sample_rate=24000, audio=audio),
    )

    assert isinstance(handler, AgentVoiceHandler)
    assert handler.second_assistant_detected is False


def test_factory_refuses_agent_handler_without_injected_clients() -> None:
    with pytest.raises(RuntimeError, match="requires injected AGENT and TTS clients"):
        build_conversation_handler(
            ToolDependencies(reachy_mini=object(), movement_manager=_FakeMovementManager()),
            backend_provider="local",
        )


def test_factory_builds_agent_live_clients_only_with_explicit_flag() -> None:
    handler = build_conversation_handler(
        ToolDependencies(reachy_mini=object(), movement_manager=_FakeMovementManager()),
        backend_provider="local",
        allow_live_agent_clients=True,
    )

    assert isinstance(handler, AgentVoiceHandler)
    assert isinstance(handler.agent_client, HermesVoiceClient)
    assert isinstance(handler.tts_client, QwenVoiceTtsClient)
