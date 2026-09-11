"""Stage 4 handler wiring: proactive speech in a half-duplex gap.

The platform transport calls ``handler._speak_proactive(text)`` when a background
result / cron / send_message arrives with no active turn. It must speak it, and it
must be mutually exclusive with interactive turns via ``_turn_lock``.
"""

# ruff: noqa: D103
from __future__ import annotations
import time

import numpy as np
import pytest
from fastrtc import AdditionalOutputs

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.agent_voice_handler import AgentVoiceHandler, FakeAudioTtsClient


class _FakeMovementManager:
    pass


class _ProactiveClient:
    """A agent_client whose transport can push proactive messages."""

    def __init__(self) -> None:
        self.proactive_handler = None

    def set_proactive_handler(self, handler) -> None:
        self.proactive_handler = handler

    async def ask(self, transcript: str, *a, **k) -> str:
        return "ok"


def _handler() -> tuple[AgentVoiceHandler, _ProactiveClient]:
    client = _ProactiveClient()
    handler = AgentVoiceHandler(
        ToolDependencies(reachy_mini=object(), movement_manager=_FakeMovementManager()),
        agent_client=client,
        tts_client=FakeAudioTtsClient(sample_rate=24000, audio=np.ones(24000, dtype=np.int16)),
    )
    return handler, client


def _drain(q):
    items = []
    while not q.empty():
        items.append(q.get_nowait())
    return items


def test_proactive_handler_is_wired():
    handler, client = _handler()
    assert client.proactive_handler is not None
    # it is the handler's bound method
    assert client.proactive_handler == handler._speak_proactive


@pytest.mark.asyncio
async def test_speak_proactive_speaks_in_a_gap(monkeypatch):
    monkeypatch.setenv("AGENT_PROACTIVE_LEADIN", "")  # keep the transcript assertion simple
    handler, _ = _handler()
    await handler._speak_proactive("Ergebnis: Die Hauptstadt ist Wellington.")
    items = _drain(handler.output_queue)
    # spoke at least one audio frame (sr, ndarray)
    audio = [x for x in items if isinstance(x, tuple)]
    assert audio, "expected audio frames queued"
    # emitted the assistant transcript
    texts = [x for x in items if isinstance(x, AdditionalOutputs)]
    assert any("Wellington" in str(getattr(t, "args", t)) for t in texts)


class _InterruptClient(_ProactiveClient):
    """A platform-transport client that can interrupt the gateway turn."""

    def __init__(self) -> None:
        super().__init__()
        self.interrupts: list = []

    async def interrupt(self, text=None) -> None:
        self.interrupts.append(text)


@pytest.mark.asyncio
async def test_platform_barge_commit_continues():
    client = _InterruptClient()
    handler = AgentVoiceHandler(
        ToolDependencies(reachy_mini=object(), movement_manager=_FakeMovementManager()),
        agent_client=client,
        tts_client=FakeAudioTtsClient(sample_rate=24000, audio=np.ones(8, dtype=np.int16)),
    )
    handler._turn_active = True
    handler._pending_barge = "mach stattdessen etwas anderes"
    handler._barge_event.set()
    cont = await handler._maybe_platform_barge()
    assert cont is True  # committed command -> keep streaming the new turn
    assert client.interrupts == ["mach stattdessen etwas anderes"]
    assert not handler._barge_event.is_set()
    assert handler._pending_barge is None


@pytest.mark.asyncio
async def test_platform_barge_bare_stop_breaks():
    client = _InterruptClient()
    handler = AgentVoiceHandler(
        ToolDependencies(reachy_mini=object(), movement_manager=_FakeMovementManager()),
        agent_client=client,
        tts_client=FakeAudioTtsClient(sample_rate=24000, audio=np.ones(8, dtype=np.int16)),
    )
    handler._turn_active = True
    handler._pending_barge = None  # a bare stop
    handler._barge_event.set()
    cont = await handler._maybe_platform_barge()
    assert cont is False  # stop -> caller breaks
    assert client.interrupts == [None]


@pytest.mark.asyncio
async def test_http_barge_falls_through_untouched():
    client = _ProactiveClient()  # no interrupt() -> HTTP transport
    handler = AgentVoiceHandler(
        ToolDependencies(reachy_mini=object(), movement_manager=_FakeMovementManager()),
        agent_client=client,
        tts_client=FakeAudioTtsClient(sample_rate=24000, audio=np.ones(8, dtype=np.int16)),
    )
    handler._pending_barge = "X"
    handler._barge_event.set()
    cont = await handler._maybe_platform_barge()
    assert cont is False  # falls through to the caller's existing break+respawn
    assert handler._pending_barge == "X"  # untouched
    assert handler._barge_event.is_set()  # untouched


@pytest.mark.asyncio
async def test_speak_proactive_drops_when_no_gap(monkeypatch):
    monkeypatch.setenv("AGENT_PROACTIVE_ACQUIRE_S", "0.2")
    handler, _ = _handler()
    await handler._turn_lock.acquire()  # simulate an in-flight turn
    try:
        t0 = time.monotonic()
        await handler._speak_proactive("späte Nachricht")
        waited = time.monotonic() - t0
    finally:
        handler._turn_lock.release()
    assert waited >= 0.2  # it waited for a gap
    assert handler.output_queue.empty()  # ...found none -> dropped, spoke nothing
