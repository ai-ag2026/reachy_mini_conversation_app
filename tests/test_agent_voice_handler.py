# ruff: noqa: D103
from __future__ import annotations
import asyncio
from typing import Any, cast

import numpy as np
import pytest
from fastrtc import AdditionalOutputs

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.agent_voice_handler import (
    AgentVoiceHandler,
    FakeAudioTtsClient,
    FakeTextAgentClient,
    _text_for_speech,
)


class _FakeMovementManager:
    pass


def test_text_for_speech_removes_emojis() -> None:
    assert _text_for_speech("Great news 😊 — all done ✅") == "Great news — all done"
    assert _text_for_speech("👨‍💻") == ""
    assert _text_for_speech("The temperature is 72°F.") == "The temperature is 72°F."


class _AsyncTextClient:
    def __init__(self, reply: str, *, delay: float = 0.0) -> None:
        self.reply = reply
        self.delay = delay
        self.calls: list[str] = []

    async def ask(self, transcript: str) -> str:
        self.calls.append(transcript)
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.reply


class _AsyncTtsClient:
    def __init__(self, sample_rate: int, audio: np.ndarray, *, delay: float = 0.0) -> None:
        self.sample_rate = sample_rate
        self.audio = audio.astype(np.int16)
        self.delay = delay
        self.calls: list[str] = []

    async def synthesize(self, text: str) -> tuple[int, np.ndarray]:
        self.calls.append(text)
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.sample_rate, self.audio


class _FailingTextClient:
    async def ask(self, transcript: str) -> str:
        _ = transcript
        raise RuntimeError("secret backend url http://internal.example")


class _FailingTtsClient:
    async def synthesize(self, text: str) -> tuple[int, np.ndarray]:
        _ = text
        raise RuntimeError("secret tts url http://tts.internal")


@pytest.mark.asyncio
async def test_agent_voice_handler_final_transcript_enqueues_metadata_and_audio() -> None:
    audio = np.array([1, -1, 2, -2], dtype=np.int16)
    handler = AgentVoiceHandler(
        ToolDependencies(reachy_mini=object(), movement_manager=_FakeMovementManager()),
        agent_client=FakeTextAgentClient(reply="AGENT Antwort."),
        tts_client=FakeAudioTtsClient(sample_rate=24000, audio=audio),
    )
    handler._output_gain = 1.0  # this test asserts exact PCM; loudness gain is tested elsewhere

    await handler.handle_final_transcript("Hallo AGENT")

    outputs = _drain(handler)
    messages = _messages(outputs)
    audio_frames = [output for output in outputs if isinstance(output, tuple)]

    assert messages == [
        {"role": "user", "content": "Hallo AGENT"},
        {"role": "assistant", "content": "AGENT Antwort."},
    ]
    assert len(audio_frames) == 1
    sample_rate, queued_audio = audio_frames[0]
    assert sample_rate == 24000
    np.testing.assert_array_equal(queued_audio, audio)
    assert handler.second_assistant_detected is False
    assert handler.agent_client.calls == ["Hallo AGENT"]
    assert handler.tts_client.calls == ["AGENT Antwort."]


@pytest.mark.asyncio
async def test_agent_voice_handler_accepts_async_agent_and_tts_clients() -> None:
    audio = np.array([3, 4, 5], dtype=np.int16)
    agent_client = _AsyncTextClient(reply="Async AGENT Antwort.")
    tts_client = _AsyncTtsClient(sample_rate=22050, audio=audio)
    handler = AgentVoiceHandler(
        ToolDependencies(reachy_mini=object(), movement_manager=_FakeMovementManager()),
        agent_client=agent_client,
        tts_client=tts_client,
    )
    handler._output_gain = 1.0  # this test asserts exact PCM

    await handler.handle_final_transcript("Async hallo")

    outputs = _drain(handler)
    audio_frames = [output for output in outputs if isinstance(output, tuple)]

    assert agent_client.calls == ["Async hallo"]
    assert tts_client.calls == ["Async AGENT Antwort."]
    assert len(audio_frames) == 1
    sample_rate, queued_audio = audio_frames[0]
    assert sample_rate == 22050
    np.testing.assert_array_equal(queued_audio, audio)


@pytest.mark.asyncio
async def test_agent_voice_handler_normalizes_display_text_before_tts() -> None:
    tts_client = FakeAudioTtsClient(sample_rate=24000, audio=np.array([1], dtype=np.int16))
    handler = AgentVoiceHandler(
        ToolDependencies(reachy_mini=object(), movement_manager=_FakeMovementManager()),
        agent_client=FakeTextAgentClient(reply="There is a 20–35% chance. See https://example.com."),
        tts_client=tts_client,
    )

    await handler.handle_final_transcript("Will it rain?")

    assert tts_client.calls == ["There is a 20 to 35 percent chance. See the link."]


@pytest.mark.asyncio
async def test_agent_voice_handler_apply_personality_keeps_agent_identity() -> None:
    handler = AgentVoiceHandler(
        ToolDependencies(reachy_mini=object(), movement_manager=_FakeMovementManager()),
        agent_client=FakeTextAgentClient(reply="ok"),
        tts_client=FakeAudioTtsClient(sample_rate=24000, audio=np.zeros(1, dtype=np.int16)),
    )

    result = await handler.apply_personality("pirate")

    # AGENT stays the brain; a non-work profile keeps full tools (no gateway-toolset restriction).
    assert "Agent" in result and "full tools active" in result
    assert handler.second_assistant_detected is False


@pytest.mark.asyncio
async def test_agent_voice_handler_serializes_final_transcript_turns() -> None:
    handler = AgentVoiceHandler(
        ToolDependencies(reachy_mini=object(), movement_manager=_FakeMovementManager()),
        agent_client=_AsyncTextClient(reply="Antwort.", delay=0.01),
        tts_client=_AsyncTtsClient(24000, np.array([1], dtype=np.int16), delay=0.01),
    )

    await asyncio.gather(
        handler.handle_final_transcript("erste Frage"),
        handler.handle_final_transcript("zweite Frage"),
    )

    messages = _messages(_drain(handler))
    assert messages == [
        {"role": "user", "content": "erste Frage"},
        {"role": "assistant", "content": "Antwort."},
        {"role": "user", "content": "zweite Frage"},
        {"role": "assistant", "content": "Antwort."},
    ]


@pytest.mark.asyncio
async def test_agent_voice_handler_reports_sanitized_agent_failure() -> None:
    handler = AgentVoiceHandler(
        ToolDependencies(reachy_mini=object(), movement_manager=_FakeMovementManager()),
        agent_client=_FailingTextClient(),
        tts_client=FakeAudioTtsClient(sample_rate=24000, audio=np.array([1], dtype=np.int16)),
    )

    await handler.handle_final_transcript("Hallo")

    messages = _messages(_drain(handler))
    assert messages == [
        {"role": "user", "content": "Hallo"},
        {"role": "assistant", "content": "I'm having trouble connecting to the agent right now."},
    ]
    assert "internal" not in str(messages)


@pytest.mark.asyncio
async def test_agent_voice_handler_reports_sanitized_tts_failure_without_audio() -> None:
    handler = AgentVoiceHandler(
        ToolDependencies(reachy_mini=object(), movement_manager=_FakeMovementManager()),
        agent_client=FakeTextAgentClient(reply="Antwort."),
        tts_client=_FailingTtsClient(),
    )

    await handler.handle_final_transcript("Hallo")

    outputs = _drain(handler)
    messages = _messages(outputs)
    assert messages == [
        {"role": "user", "content": "Hallo"},
        {"role": "assistant", "content": "Antwort."},
        {"role": "assistant", "content": "I generated the answer, but speech output is having trouble."},
    ]
    assert not [output for output in outputs if isinstance(output, tuple)]
    assert "internal" not in str(messages)


def _drain(handler: AgentVoiceHandler) -> list[object]:
    outputs = []
    while not handler.output_queue.empty():
        outputs.append(handler.output_queue.get_nowait())
    return outputs


def _messages(outputs: list[object]) -> list[object]:
    messages: list[object] = []
    for output in outputs:
        if isinstance(output, AdditionalOutputs):
            messages.extend(cast(tuple[Any, ...], getattr(output, "args")))
    return messages


def test_handler_toggles(monkeypatch):
    import os
    from unittest.mock import MagicMock

    from reachy_mini_conversation_app.agent_voice_handler import AgentVoiceHandler

    h = AgentVoiceHandler(MagicMock(), agent_client=MagicMock(), tts_client=MagicMock())
    monkeypatch.delenv("AGENT_VOICE_TOOLS", raising=False)
    monkeypatch.delenv("AGENT_VISION_ENABLED", raising=False)
    monkeypatch.delenv("AGENT_VISION_BLOCK_PERSON_ID", raising=False)
    monkeypatch.delenv("AGENT_COMPANION", raising=False)
    monkeypatch.delenv("AGENT_IDLE_ACTIONS", raising=False)
    monkeypatch.delenv("AGENT_SPEECH_SWAY", raising=False)
    t = h.get_toggles()
    assert t == {
        "tools": True,
        "vision": True,
        "person_id": True,
        "mic": True,
        "companion": False,
        "idle_actions": True,
        "speech_sway": True,
    }
    h.set_toggle("tools", False)
    assert os.environ["AGENT_VOICE_TOOLS"] == "0"
    h.set_toggle("person_id", False)
    assert os.environ["AGENT_VISION_BLOCK_PERSON_ID"] == "1"
    h.set_toggle("mic", False)
    assert h._mic_muted is True
    h.set_toggle("companion", True)
    assert os.environ["AGENT_COMPANION"] == "1"
    assert h.get_toggles() == {
        "tools": False,
        "vision": True,
        "person_id": False,
        "mic": False,
        "companion": True,
        "idle_actions": True,
        "speech_sway": True,
    }
    import pytest

    with pytest.raises(KeyError):
        h.set_toggle("nope", True)


def test_handler_settings(monkeypatch):
    import os
    from unittest.mock import MagicMock

    import pytest

    from reachy_mini_conversation_app.agent_voice_handler import AgentVoiceHandler

    h = AgentVoiceHandler(MagicMock(), agent_client=MagicMock(), tts_client=MagicMock())
    monkeypatch.setenv("AGENT_VOICE_REASONING_EFFORT", "minimal")
    # select validated + applied to env
    s = h.set_setting("reasoning_effort", "low")
    assert os.environ["AGENT_VOICE_REASONING_EFFORT"] == "low" and s["reasoning_effort"] == "low"
    # numerics applied + clamped
    h.set_setting("quicktake_delay_s", 1.2)
    assert os.environ["AGENT_QUICKTAKE_DELAY_S"] == "1.2"
    h.set_setting("first_audio_budget_s", 999)
    assert os.environ["AGENT_VOICE_FIRST_AUDIO_BUDGET_S"] == "120.0"
    h.set_setting("output_gain", 5.0)
    assert h._output_gain == 3.0  # clamped to 3.0
    # bools
    h.set_setting("quicktake", False)
    assert os.environ["AGENT_QUICKTAKE_ENABLED"] == "0"
    h.set_setting("tool_status", True)
    assert os.environ["AGENT_TOOL_STATUS_ENABLED"] == "1"
    # validation errors -> ValueError / KeyError (console maps to 4xx)
    with pytest.raises(ValueError):
        h.set_setting("reasoning_effort", "ludicrous")
    with pytest.raises(ValueError):
        h.set_setting("quicktake_delay_s", "abc")
    with pytest.raises(KeyError):
        h.set_setting("nope", 1)
    # SETTING_ENV maps every settable name (console relies on it to persist)
    assert set(h.SETTING_ENV) >= {
        "reasoning_effort",
        "quicktake_delay_s",
        "first_audio_budget_s",
        "output_gain",
        "quicktake",
        "tool_status",
    }


@pytest.mark.asyncio
async def test_apply_personality_work_mode_restricts_gateway_tools(monkeypatch):
    import os
    from unittest.mock import MagicMock

    from reachy_mini_conversation_app.agent_voice_handler import AgentVoiceHandler

    h = AgentVoiceHandler(MagicMock(), agent_client=MagicMock(), tts_client=MagicMock())
    monkeypatch.delenv("AGENT_VOICE_TOOLSETS", raising=False)
    await h.apply_personality("agent-workmodus")
    ts = os.environ["AGENT_VOICE_TOOLSETS"]
    assert "terminal" not in ts and "code_execution" not in ts and "file" not in ts.split(",")
    assert "web" in ts and "memory" in ts and "vision" in ts
    await h.apply_personality("local")
    assert "AGENT_VOICE_TOOLSETS" not in os.environ  # full tools restored


class _FakeLeadInClient:
    """9B quick-take stand-in: returns ``text`` after ``delay`` s."""

    def __init__(self, text: str = "", delay: float = 0.0) -> None:
        self._text = text
        self._delay = delay

    async def lead_in(self, transcript: str) -> str:
        if self._delay:
            await asyncio.sleep(self._delay)
        return self._text


def _mk_handler(lead_client: Any):
    from unittest.mock import MagicMock

    return AgentVoiceHandler(MagicMock(), agent_client=MagicMock(), tts_client=MagicMock(), lead_in_client=lead_client)


@pytest.mark.asyncio
async def test_quicktake_uses_9b_bridge_when_gateway_slow(monkeypatch):
    monkeypatch.setenv("AGENT_QUICKTAKE_DELAY_S", "0.05")
    monkeypatch.setenv("AGENT_QUICKTAKE_ENABLED", "1")
    h = _mk_handler(_FakeLeadInClient(text="Also.", delay=0.0))

    async def slow_stream(_t):
        await asyncio.sleep(0.25)
        yield "Echte Antwort."

    out = [s async for s in h._stream_with_lead_in("Frag was", slow_stream)]
    assert out[0] == "Also." and out[-1] == "Echte Antwort."


@pytest.mark.asyncio
async def test_quicktake_static_fallback_when_9b_not_ready(monkeypatch):
    from reachy_mini_conversation_app.agent_clients import _STATIC_QUICKTAKES

    monkeypatch.setenv("AGENT_QUICKTAKE_DELAY_S", "0.05")
    monkeypatch.setenv("AGENT_QUICKTAKE_ENABLED", "1")
    h = _mk_handler(_FakeLeadInClient(text="Also.", delay=5.0))  # 9B too slow -> static

    async def slow_stream(_t):
        await asyncio.sleep(0.25)
        yield "Antwort."

    out = [s async for s in h._stream_with_lead_in("Frag was", slow_stream)]
    assert out[0] in _STATIC_QUICKTAKES and out[-1] == "Antwort."


@pytest.mark.asyncio
async def test_quicktake_guards_against_moment(monkeypatch):
    from reachy_mini_conversation_app.agent_clients import _STATIC_QUICKTAKES

    monkeypatch.setenv("AGENT_QUICKTAKE_DELAY_S", "0.05")
    monkeypatch.setenv("AGENT_QUICKTAKE_ENABLED", "1")
    h = _mk_handler(_FakeLeadInClient(text="Moment, gleich.", delay=0.0))  # collides w/ Phase A

    async def slow_stream(_t):
        await asyncio.sleep(0.25)
        yield "Antwort."

    out = [s async for s in h._stream_with_lead_in("Frag was", slow_stream)]
    assert out[0] in _STATIC_QUICKTAKES and "moment" not in out[0].lower()


@pytest.mark.asyncio
async def test_quicktake_silent_on_fast_turn(monkeypatch):
    monkeypatch.setenv("AGENT_QUICKTAKE_DELAY_S", "0.5")
    monkeypatch.setenv("AGENT_QUICKTAKE_ENABLED", "1")
    h = _mk_handler(_FakeLeadInClient(text="Also.", delay=0.0))

    async def fast_stream(_t):
        yield "Sofort."
        yield "Mehr."

    out = [s async for s in h._stream_with_lead_in("Frag was", fast_stream)]
    assert out == ["Sofort.", "Mehr."]  # gateway fast -> no opener


@pytest.mark.asyncio
async def test_quicktake_disabled_emits_no_opener(monkeypatch):
    monkeypatch.setenv("AGENT_QUICKTAKE_DELAY_S", "0.05")
    monkeypatch.setenv("AGENT_QUICKTAKE_ENABLED", "0")
    h = _mk_handler(_FakeLeadInClient(text="Also.", delay=0.0))

    async def slow_stream(_t):
        await asyncio.sleep(0.20)
        yield "Antwort."

    out = [s async for s in h._stream_with_lead_in("Frag was", slow_stream)]
    assert out == ["Antwort."]


@pytest.mark.asyncio
async def test_start_up_blocks_until_shutdown(monkeypatch) -> None:
    """Regression: start_up() must block for the session lifetime (upstream base_realtime contract).

    If it returns after setup instead, the console startup loop treats that as "session ended" and re-runs it
    on a retry loop — spawning a fresh idle/IMU/sway/companion watcher set each pass without stopping the old
    ones (task+CPU leak).
    """
    handler = AgentVoiceHandler(
        ToolDependencies(reachy_mini=object(), movement_manager=_FakeMovementManager()),
        agent_client=FakeTextAgentClient(reply="ok"),
        tts_client=FakeAudioTtsClient(sample_rate=24000, audio=np.zeros(4, dtype=np.int16)),
    )
    monkeypatch.setattr(handler, "_ensure_stt", lambda: None)
    monkeypatch.setattr(handler, "_ensure_barge_stt", lambda: None)

    task = asyncio.create_task(handler.start_up())
    await asyncio.sleep(0.2)
    assert not task.done(), "start_up must block for the session lifetime"

    await handler.shutdown()
    await asyncio.wait_for(task, timeout=2.0)


@pytest.mark.parametrize("language,expected", [(None, "en"), ("", "en"), ("auto", "auto"), ("fr", "fr")])
def test_stt_language_defaults_to_english(monkeypatch, language, expected):
    """Default to English while preserving explicit language and auto-detection choices."""
    from reachy_mini_conversation_app.agent_voice_handler import _stt_language

    if language is None:
        monkeypatch.delenv("AGENT_STT_LANGUAGE", raising=False)
    else:
        monkeypatch.setenv("AGENT_STT_LANGUAGE", language)
    assert _stt_language() == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reply,expected",
    [
        ("- item one\n- item two", "item one item two"),
        ("1. First\n2. Second", "First Second"),
        ("# First\n## Second", "First Second"),
        ("😊✅", None),
    ],
)
async def test_handler_normalizes_multiline_and_empty_replies(reply, expected):
    tts = FakeAudioTtsClient(sample_rate=24000, audio=np.array([1], dtype=np.int16))
    handler = AgentVoiceHandler(
        ToolDependencies(reachy_mini=object(), movement_manager=_FakeMovementManager()),
        agent_client=FakeTextAgentClient(reply=reply),
        tts_client=tts,
    )
    await handler.handle_final_transcript("Question")
    assert tts.calls == ([expected] if expected else [])
    outputs = _drain(handler)
    assert "speech output is having trouble" not in repr(outputs)
    if expected is None:
        assert not any(isinstance(output, tuple) for output in outputs)
        assert await handler._speak_sentence(reply) is True


@pytest.mark.asyncio
@pytest.mark.parametrize("log_content", [False, True])
async def test_handler_pipeline_events_and_log_privacy(monkeypatch, caplog, log_content):
    import logging
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from reachy_mini_conversation_app.pipeline_monitor import PipelineMonitor

    monkeypatch.setenv("AGENT_PIPELINE_MONITOR_LOG_CONTENT", "1" if log_content else "0")
    monkeypatch.setenv("AGENT_QUICKTAKE_ENABLED", "0")
    monkeypatch.setenv("AGENT_STT_LANGUAGE", "en")
    caplog.set_level(logging.INFO)
    tts = FakeAudioTtsClient(sample_rate=24000, audio=np.array([1], dtype=np.int16))
    tts.config = SimpleNamespace(model="test-model", voice="test-voice", speed=0.95)
    client = FakeTextAgentClient(reply="unused")

    async def stream(text):
        yield "Private answer."

    monkeypatch.setattr(client, "ask_stream", stream, raising=False)
    handler = AgentVoiceHandler(
        ToolDependencies(reachy_mini=object(), movement_manager=_FakeMovementManager()),
        agent_client=client,
        tts_client=tts,
    )
    # Record real handler calls and exercise logging without opening a server.
    monitor = MagicMock(wraps=PipelineMonitor(port=0))
    handler._pipeline_monitor = monitor
    await handler.handle_final_transcript("Private question")
    calls = monitor.emit.call_args_list
    stt = next(call for call in calls if call.args[0] == "stt")
    llm = next(call for call in calls if call.args[0] == "llm")
    spoken = next(call for call in calls if call.args[0] == "tts")
    assert stt.args == ("stt", "Private question")
    assert stt.kwargs == {"language": "en"}
    assert llm.args == ("llm", "Private answer.")
    assert llm.kwargs["first_chunk"] is True
    assert llm.kwargs["elapsed_ms"] >= 0
    assert spoken.args == ("tts", "Private answer.")
    assert spoken.kwargs == {"model": "test-model", "voice": "test-voice", "speed": 0.95}
    assert ("Private question" in caplog.text) is log_content
    assert ("Private answer." in caplog.text) is log_content
    assert "test-model" in caplog.text
