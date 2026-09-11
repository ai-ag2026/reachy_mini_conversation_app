# ruff: noqa: D103
"""Non-pausing semantic barge-in in AgentVoiceHandler.

AGENT keeps talking while the user speaks; only a COMPLETED user utterance is classified (off-thread)
and acted on: ignore (backchannel -> keep talking), stop (bare stop -> stop, no forward), commit
(real interrupt -> stop + forward the whole transcript as the next turn).

The tests drive handle_final_transcript with fake streaming clients and simulate the user utterance
endpointing by calling _classify_and_act from inside the fake TTS. The gate runs for real via its
pure heuristic ("ja" -> ignore, "stopp" -> stop); the commit case monkeypatches the classifier.
"""

from __future__ import annotations
import asyncio

import numpy as np
import pytest
import reachy_agent.voice.semantic_gate as gate

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.agent_voice_handler import AgentVoiceHandler


class _FakeMovementManager:
    pass


def _make_handler(agent_client, tts_client) -> AgentVoiceHandler:
    h = AgentVoiceHandler(
        ToolDependencies(reachy_mini=object(), movement_manager=_FakeMovementManager()),
        agent_client=agent_client,
        tts_client=tts_client,
    )
    return h


class _StreamAgentClient:
    def __init__(self, sentences: list[str]) -> None:
        self.sentences = sentences
        self.calls: list[str] = []

    async def ask_stream(self, transcript: str):
        self.calls.append(transcript)
        for s in self.sentences:
            yield s


class _ActingTtsClient:
    """Streams a chunk per sentence; on the configured sentence index, simulates a completed user utterance by invoking the handler's _classify_and_act (what _feed_barge does on an endpoint)."""

    def __init__(self, box: dict, *, act_on_call: int, transcript: str) -> None:
        self.box = box
        self.act_on_call = act_on_call
        self.transcript = transcript
        self.calls: list[str] = []
        self.fired = False

    def set_voice(self, voice: str) -> None:  # pragma: no cover
        pass

    async def stream_pcm(self, text: str):
        self.calls.append(text)
        yield 24000, np.zeros(12000, dtype=np.int16)
        if len(self.calls) - 1 == self.act_on_call and not self.fired:
            self.fired = True
            await self.box["h"]._classify_and_act(self.transcript)
        yield 24000, np.zeros(12000, dtype=np.int16)


@pytest.mark.asyncio
async def test_backchannel_is_ignored_agent_keeps_talking() -> None:
    box: dict = {}
    agent = _StreamAgentClient(["Satz eins.", "Satz zwei.", "Satz drei."])
    tts = _ActingTtsClient(box, act_on_call=0, transcript="ja")  # heuristic -> ignore
    handler = _make_handler(agent, tts)
    box["h"] = handler
    handler._turn_active = True  # receive() sets this before scheduling the turn

    await handler.handle_final_transcript("Erzähl mir was")
    await asyncio.sleep(0)

    assert tts.calls == ["Satz eins.", "Satz zwei.", "Satz drei."]  # never interrupted
    assert handler._pending_barge is None
    assert handler._barge_event.is_set() is False


@pytest.mark.asyncio
async def test_bare_stop_stops_without_forward() -> None:
    box: dict = {}
    agent = _StreamAgentClient(["Satz eins.", "Satz zwei.", "Satz drei."])
    tts = _ActingTtsClient(box, act_on_call=0, transcript="stopp")  # heuristic -> stop
    handler = _make_handler(agent, tts)
    box["h"] = handler
    handler._turn_active = True  # receive() sets this before scheduling the turn

    await handler.handle_final_transcript("Erzähl mir was")
    for _ in range(4):
        await asyncio.sleep(0)

    assert tts.calls == ["Satz eins."]  # stopped after the first sentence
    assert handler._pending_barge is None  # bare stop forwards nothing
    assert handler._turn_active is False  # no follow-up turn
    assert agent.calls == ["Erzähl mir was"]


@pytest.mark.asyncio
async def test_commit_stops_and_forwards_whole_transcript(monkeypatch) -> None:
    monkeypatch.setattr(gate, "classify_interrupt", lambda *_a, **_k: "commit")
    box: dict = {}
    agent = _StreamAgentClient(["Satz eins.", "Satz zwei.", "Satz drei."])
    tts = _ActingTtsClient(box, act_on_call=0, transcript="hör auf, erzähl mir über Katzen")
    handler = _make_handler(agent, tts)
    box["h"] = handler
    handler._turn_active = True  # receive() sets this before scheduling the turn

    await handler.handle_final_transcript("Erzähl mir was")
    for _ in range(6):
        await asyncio.sleep(0)

    assert tts.calls[0] == "Satz eins."  # first reply started
    # the committed interrupt becomes a SECOND turn, forwarded whole:
    assert agent.calls == ["Erzähl mir was", "hör auf, erzähl mir über Katzen"]


def test_barge_can_be_disabled_by_env(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_BARGE_IN", "0")
    handler = _make_handler(_StreamAgentClient([]), _ActingTtsClient({}, act_on_call=-1, transcript=""))
    assert handler._barge_enabled is False


class _PlatformishClient(_StreamAgentClient):
    """Stream client WITH an interrupt() coroutine — the platform-transport shape."""

    def __init__(self, sentences: list[str]) -> None:
        super().__init__(sentences)
        self.interrupts: list[str | None] = []

    async def interrupt(self, text: str | None = None) -> None:
        self.interrupts.append(text)


@pytest.mark.asyncio
async def test_classify_dispatches_platform_interrupt_immediately(monkeypatch) -> None:
    """P1-5 (review 2026-07-02 round 2): a stop/commit decided during the gateway's silent tool/think phase must reach the gateway NOW — not at the next spoken sentence."""
    monkeypatch.setattr(gate, "classify_interrupt", lambda *_a, **_k: "commit")
    client = _PlatformishClient([])
    handler = _make_handler(client, _ActingTtsClient({}, act_on_call=-1, transcript=""))
    handler._turn_active = True  # a turn is running, gateway silent (no sentences flowing)

    await handler._classify_and_act("stopp, erzähl lieber über Katzen")

    assert client.interrupts == ["stopp, erzähl lieber über Katzen"]  # dispatched immediately
    assert handler._pending_barge is None  # consumed by the dispatch (no duplicate turn)
    assert handler._barge_event.is_set() is False  # state reset -> speak loop keeps streaming
    assert handler._classify_inflight is False


@pytest.mark.asyncio
async def test_classify_bare_stop_dispatches_slash_stop(monkeypatch) -> None:
    client = _PlatformishClient([])
    handler = _make_handler(client, _ActingTtsClient({}, act_on_call=-1, transcript=""))
    handler._turn_active = True

    await handler._classify_and_act("stopp")  # heuristic bare stop

    assert client.interrupts == [None]  # forwarded as bare /stop by the client
    assert handler._barge_event.is_set() is False


@pytest.mark.asyncio
async def test_classify_http_transport_keeps_loop_side_path(monkeypatch) -> None:
    """No interrupt() on the client (HTTP): the event must STAY set so the speak loop handles the barge at the next sentence, exactly as before."""
    monkeypatch.setattr(gate, "classify_interrupt", lambda *_a, **_k: "commit")
    client = _StreamAgentClient([])  # no interrupt attr
    handler = _make_handler(client, _ActingTtsClient({}, act_on_call=-1, transcript=""))
    handler._turn_active = True

    await handler._classify_and_act("mach lieber etwas anderes")

    assert handler._barge_event.is_set() is True
    assert handler._pending_barge == "mach lieber etwas anderes"


@pytest.mark.asyncio
async def test_silent_phase_commit_is_deferred_and_fires_at_first_audio(monkeypatch) -> None:
    """Review 2026-07-02 round 2, P2: a commit during the silent think phase must not vanish — it defers via _pending_barge and executes at the first queued audio."""
    monkeypatch.setattr(gate, "classify_interrupt", lambda *_a, **_k: "commit")
    client = _PlatformishClient([])
    handler = _make_handler(client, _ActingTtsClient({}, act_on_call=-1, transcript=""))
    handler._turn_active = True
    handler._turn_spoke = False  # silent think phase

    await handler._classify_and_act("nein warte, nimm die andere Datei", my_seq=handler._turn_seq, silent_phase=True)

    # deferred: nothing stopped/dispatched yet, command parked
    assert handler._pending_barge == "nein warte, nimm die andere Datei"
    assert handler._barge_event.is_set() is False
    assert client.interrupts == []

    # first audio arrives -> deferred commit executes (event + async dispatch)
    handler._advance_playback_clock(0.5)
    assert handler._barge_event.is_set() is True
    await asyncio.sleep(0)  # let the dispatch task run
    await asyncio.sleep(0)
    assert client.interrupts == ["nein warte, nimm die andere Datei"]
    assert handler._barge_event.is_set() is False  # _maybe_platform_barge reset the state


@pytest.mark.asyncio
async def test_silent_phase_bare_stop_lets_turn_deliver(monkeypatch) -> None:
    client = _PlatformishClient([])
    handler = _make_handler(client, _ActingTtsClient({}, act_on_call=-1, transcript=""))
    handler._turn_active = True
    handler._turn_spoke = False

    await handler._classify_and_act("stopp", my_seq=handler._turn_seq, silent_phase=True)

    # nothing audible to stop -> no cancel (no-answer-loop protection), nothing forwarded
    assert handler._pending_barge is None
    assert handler._barge_event.is_set() is False
    assert client.interrupts == []


@pytest.mark.asyncio
async def test_stale_classify_after_turn_end_is_dropped(monkeypatch) -> None:
    """Review 2026-07-02 round 2, P2: a gate decision arriving after its turn ended (turn_seq rolled) must neither stop the NEXT turn nor linger as a ghost _pending_barge."""
    monkeypatch.setattr(gate, "classify_interrupt", lambda *_a, **_k: "commit")
    client = _PlatformishClient([])
    handler = _make_handler(client, _ActingTtsClient({}, act_on_call=-1, transcript=""))
    handler._turn_active = True
    handler._turn_spoke = True
    stale_seq = handler._turn_seq
    handler._turn_seq += 1  # the candidate's turn ended; a NEWER turn owns the state

    await handler._classify_and_act("alte anweisung", my_seq=stale_seq)

    assert handler._pending_barge is None
    assert handler._barge_event.is_set() is False
    assert client.interrupts == []


@pytest.mark.asyncio
async def test_chirp_books_playback_clock_without_turn_spoke(monkeypatch) -> None:
    """Review 2026-07-02 round 2, P3: the chirp bumped only _speaking_until; the next speech segment restarted the cursor at `now` -> tail mute ended ~a chirp too early.

    Booking it via the clock must NOT set _turn_spoke or fire a deferred commit.
    """
    monkeypatch.setenv("AGENT_CHIRPS", "1")
    client = _PlatformishClient([])
    handler = _make_handler(client, _ActingTtsClient({}, act_on_call=-1, transcript=""))
    handler._turn_active = True
    handler._turn_spoke = False
    handler._pending_barge = "deferred kommando"  # must NOT fire on a chirp

    import time as _t

    before = _t.monotonic()
    handler._status_chirp("acknowledge")
    assert handler._playback_cursor > before  # cue is on the playback clock
    assert handler._turn_spoke is False  # a cue is not spoken content
    assert handler._barge_event.is_set() is False  # deferred commit NOT fired
    assert client.interrupts == []
