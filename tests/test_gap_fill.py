"""Latency #1: full-window quick-take / gap-filling.

When the brain (the agent) stays silent past the opener, _stream_with_lead_in emits
a few varied content-free tail fillers so the 4-10 s cloud hole isn't dead air —
and stops the instant the first real chunk arrives (never delays the answer).
"""

# ruff: noqa: D103
from __future__ import annotations

import asyncio

import numpy as np
import pytest

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.agent_voice_handler import AgentVoiceHandler, FakeAudioTtsClient
from reachy_mini_conversation_app.agent_clients import _STATIC_QUICKTAKES, _GAP_FILLERS


class _FakeMovementManager:
    pass


class _LeadIn:
    async def lead_in(self, transcript: str) -> str:
        return ""  # force the instant static opener path


class _Client:
    async def ask(self, transcript: str, *a, **k) -> str:
        return "ok"


def _handler():
    return AgentVoiceHandler(
        ToolDependencies(reachy_mini=object(), movement_manager=_FakeMovementManager()),
        agent_client=_Client(),
        tts_client=FakeAudioTtsClient(sample_rate=24000, audio=np.ones(4, dtype=np.int16)),
        lead_in_client=_LeadIn(),
    )


@pytest.mark.asyncio
async def test_full_window_gap_fill_masks_long_hole(monkeypatch):
    monkeypatch.setenv("AGENT_QUICKTAKE_ENABLED", "1")
    monkeypatch.setenv("AGENT_QUICKTAKE_DELAY_S", "0.05")
    monkeypatch.setenv("AGENT_GAP_FILL_INTERVAL_S", "0.05")
    monkeypatch.setenv("AGENT_GAP_FILL_MAX", "2")
    h = _handler()

    async def slow_ask_stream(transcript):
        await asyncio.sleep(0.5)  # brain silent well past opener + 2 fillers
        yield "Die echte Antwort."

    out = [c async for c in h._stream_with_lead_in("frage", slow_ask_stream)]

    assert out[-1] == "Die echte Antwort."          # real answer always last, never delayed away
    assert len(out) >= 3                             # opener + >=1 filler + real
    fillers = out[:-1]
    assert len(set(fillers)) == len(fillers)         # no repeated line
    assert all(f in _STATIC_QUICKTAKES or f in _GAP_FILLERS for f in fillers)  # content-free only


@pytest.mark.asyncio
async def test_fast_turn_emits_no_filler(monkeypatch):
    monkeypatch.setenv("AGENT_QUICKTAKE_DELAY_S", "0.3")
    monkeypatch.setenv("AGENT_GAP_FILL_MAX", "2")
    h = _handler()

    async def fast_ask_stream(transcript):
        yield "Sofort da."  # first chunk arrives before the opener delay

    out = [c async for c in h._stream_with_lead_in("frage", fast_ask_stream)]
    assert out == ["Sofort da."]  # no opener, no fillers on a fast turn


@pytest.mark.asyncio
async def test_prewarm_is_best_effort(monkeypatch):
    # unreachable gateway -> pre-warm must swallow the error, never raise
    monkeypatch.setenv("AGENT_BASE_URL", "http://127.0.0.1:1/v1")
    h = _handler()
    await h._prewarm()
