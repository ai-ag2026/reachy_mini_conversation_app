# ruff: noqa: D103
"""Stage-2 preemptive turn-start: the speculative turn holds its audio behind a gate until the real
endpoint confirms, and adopts only when the final transcript equals the partial it started on."""
from __future__ import annotations

import asyncio

from unittest.mock import MagicMock

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.agent_voice_handler import AgentVoiceHandler


class _StreamAgentClient:
    def __init__(self, sentences):
        self.sentences = sentences
        self.calls = []

    async def ask_stream(self, transcript, *a, **k):
        self.calls.append(transcript)
        for s in self.sentences:
            yield s


def _make_handler(sentences):
    return AgentVoiceHandler(
        ToolDependencies(reachy_mini=object(), movement_manager=MagicMock()),
        agent_client=_StreamAgentClient(sentences),
        tts_client=MagicMock(),
    )


def test_spec_matches_normalized_equality():
    h = _make_handler([])
    h._spec_partial = "Erzähl mir etwas über schwarze Löcher"
    assert h._spec_matches("Erzähl mir etwas über schwarze Löcher.")  # trailing punct ignored
    assert h._spec_matches("erzähl  mir etwas über schwarze löcher")  # case/spacing ignored
    assert not h._spec_matches("Erzähl mir etwas über schwarze Löcher und Sterne")  # user added words
    assert not h._spec_matches("")


def test_speculative_holds_until_gate_then_speaks():
    async def run():
        h = _make_handler(["Ein schwarzes Loch.", "Es krümmt die Raumzeit."])
        spoken = []

        async def fake_speak(s):
            spoken.append(s)
            return True

        h._speak_sentence = fake_speak  # isolate from real TTS/pacing
        gate = asyncio.Event()
        adopted = {"v": False}
        task = asyncio.create_task(h._speculative_turn("frage", my_seq=1, gate=gate, adopted=adopted))
        await asyncio.sleep(0.05)
        held = list(spoken)          # nothing spoken while gate closed
        adopted["v"] = True          # adopt marker (set before the gate, like receive() does)
        gate.set()                   # adopt -> release
        h._turn_seq = 1              # this task owns seq 1
        await asyncio.wait_for(task, timeout=2)
        return held, spoken, h

    held, spoken, h = asyncio.run(run())
    assert held == []                                        # audio was held pre-confirm
    assert spoken == ["Ein schwarzes Loch.", "Es krümmt die Raumzeit."]  # released after gate
    assert h._turn_active is False                           # turn cleanup ran (adopted)


def test_speculative_discarded_before_gate_never_speaks():
    async def run():
        h = _make_handler(["Falsche Antwort."])
        spoken = []

        async def fake_speak(s):
            spoken.append(s)
            return True

        h._speak_sentence = fake_speak
        gate = asyncio.Event()
        task = asyncio.create_task(h._speculative_turn("halbe frage", my_seq=1, gate=gate, adopted={"v": False}))
        await asyncio.sleep(0.05)    # holding at the gate
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return spoken

    spoken = asyncio.run(run())
    assert spoken == []              # a discarded speculation never speaks a wrong word


def test_speculative_discarded_after_gate_open_runs_no_adopted_cleanup():
    """Review 2026-07-02 round 2, P2: _discard_speculative cancels THEN opens the gate — the
    cancelled task's finally used to see gate.is_set() and ran the ADOPTED cleanup (clearing
    _turn_active mid fresh turn, dropping _pending_barge). The adopted marker fixes that."""

    async def run():
        h = _make_handler(["Falsche Antwort."])

        async def fake_speak(s):
            return True

        h._speak_sentence = fake_speak
        gate = asyncio.Event()
        adopted = {"v": False}
        task = asyncio.create_task(h._speculative_turn("halbe frage", my_seq=1, gate=gate, adopted=adopted))
        await asyncio.sleep(0.05)  # holding at the gate
        # discard order like _discard_speculative: cancel, THEN open the gate
        task.cancel()
        gate.set()
        # meanwhile a fresh turn owns the state
        h._turn_active = True
        h._turn_seq = 1  # matches my_seq -> the OLD buggy finally would clear _turn_active
        h._pending_barge = "wichtiges kommando"
        try:
            await task
        except asyncio.CancelledError:
            pass
        return h

    h = asyncio.run(run())
    assert h._turn_active is True                      # fresh turn NOT clobbered
    assert h._pending_barge == "wichtiges kommando"    # pending command survives the discard
