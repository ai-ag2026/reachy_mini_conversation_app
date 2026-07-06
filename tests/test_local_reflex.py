"""Latency #3: local 9B reflex for clearly-trivial turns.

Fully answers a trivial short turn locally (removing the 3-6s cloud brain), or
returns None to escalate. Conservative + opt-in; ESCALATEs on any doubt.
"""

# ruff: noqa: D101, D102, D103
from __future__ import annotations

import asyncio

from reachy_mini_conversation_app.agent_clients import LocalReflexClient, LocalReflexConfig


class _FakeResp:
    def __init__(self, content: str) -> None:
        self._content = content

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {"choices": [{"message": {"content": self._content}}]}


class _FakePost:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = 0

    async def post(self, url: str, **kwargs) -> _FakeResp:
        self.calls += 1
        return _FakeResp(self.content)


def _client(content: str, *, enabled: bool = True, max_words: int = 6):
    http = _FakePost(content)
    cfg = LocalReflexConfig(enabled=enabled, max_words=max_words)
    return LocalReflexClient(config=cfg, http_client=http), http


def test_disabled_by_default_no_network():
    # default config is disabled ("voller AGENT" is the default)
    assert LocalReflexConfig().enabled is False
    c, http = _client("Hallo.", enabled=False)
    assert asyncio.run(c.answer("hallo")) is None
    assert http.calls == 0


def test_trivial_turn_answered_locally():
    c, http = _client("Hallo Operator.")
    assert asyncio.run(c.answer("hallo agent")) == "Hallo Operator."
    assert http.calls == 1


def test_escalate_falls_through():
    c, http = _client("ESCALATE")
    assert asyncio.run(c.answer("wie spät ist es")) is None
    assert http.calls == 1


def test_long_turn_prefiltered_no_network():
    c, http = _client("Egal.")
    long = "erklär mir bitte ausführlich wie ein schwarzes loch entsteht und was danach passiert"
    assert asyncio.run(c.answer(long)) is None
    assert http.calls == 0  # pre-filtered before any 9B call


def test_empty_transcript_is_none():
    c, http = _client("x")
    assert asyncio.run(c.answer("   ")) is None
    assert http.calls == 0


def test_confirmations_never_answered_locally():
    """A context-free reflex must never eat a confirmation the stateful session is waiting for
    (review 2026-07-02 round 2, P1-6): hard filter, no network call."""
    c, http = _client("Gerne.")
    for utterance in ("Ja, bitte.", "Nein.", "Okay, mach das.", "Ja", "Passt, genau so.",
                      "Stopp.", "Weiter bitte."):
        assert asyncio.run(c.answer(utterance)) is None, utterance
    assert http.calls == 0


def test_greetings_still_pass_the_confirmation_filter():
    c, http = _client("Hallo Operator.")
    assert asyncio.run(c.answer("hallo agent, alles fit?")) == "Hallo Operator."
    assert http.calls == 1
