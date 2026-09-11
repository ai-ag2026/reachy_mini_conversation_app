"""Offline tests for the Reachy platform ws transport client (Stage 2 + 4).

A FakeWS replays a recorded frame sequence so we test the frame -> sentence
logic (cursor strip, notice skip, position-based dedup), the turn/proactive
routing, and the persistent-reader proactive path — without a gateway.
"""

import json
import asyncio

from reachy_mini_conversation_app.reachy_platform_client import (
    ReachyPlatformClient,
    _AnswerAccumulator,
    _complete_sentences,
)


def _f(**kw) -> str:
    return json.dumps(kw)


def test_platform_config_reads_api_key(monkeypatch):
    """Load the shared platform credential from the environment."""
    monkeypatch.setenv("AGENT_PLATFORM_API_KEY", "shared-secret")

    config = ReachyPlatformClient().config

    assert config.api_key == "shared-secret"


class FakeWS:
    """Async-iterable stand-in for a websocket connection."""

    def __init__(self, outbound):
        self._out = list(outbound)
        self.sent = []

    async def send(self, m):
        self.sent.append(m)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._out:
            raise StopAsyncIteration
        await asyncio.sleep(0)  # let the awaiting turn run
        return self._out.pop(0)

    async def close(self):
        pass


def _interactive(frames, transcript="hallo"):
    c = ReachyPlatformClient()
    c._ws = FakeWS(frames)  # inject: _ensure_session skips connect, starts the reader

    async def go():
        return [chunk async for chunk in c.ask_stream(transcript)]

    return asyncio.run(go())


def _proactive(frames):
    captured = []
    c = ReachyPlatformClient()

    async def on_p(text):
        captured.append(text)

    c.set_proactive_handler(on_p)
    c._ws = FakeWS(frames)

    async def go():
        await c._ensure_session()
        await c._reader_task  # drain the FakeWS

    asyncio.run(go())
    return captured


# ── pure ─────────────────────────────────────────────────────────────────────
def test_complete_sentences():
    comp, tail = _complete_sentences("Satz eins. Satz zwei! Rest ohne")
    assert comp == ["Satz eins.", "Satz zwei!"]
    assert tail.strip() == "Rest ohne"


# ── interactive turn ──────────────────────────────────────────────────────────
def test_cursor_strip_and_single_sentence():
    out = _interactive([
        _f(type="say", kind="message", message_id="A", content="Eins, zwei ▉", final=True),
        _f(type="say", kind="stream", message_id="A", content="Eins, zwei, drei.", final=True),
        _f(type="turn_end", outcome="success"),
    ])
    assert out == ["Eins, zwei, drei."]


def test_notice_is_skipped():
    out = _interactive([
        _f(type="say", kind="message", message_id="N", content="ℹ Kontext-Hinweis vom System.", final=True),
        _f(type="say", kind="stream", message_id="A", content="Hallo Operator.", final=True),
        _f(type="turn_end"),
    ])
    assert out == ["Hallo Operator."]


def test_computer_use_tool_status_is_skipped():
    """Hermes gear-prefixed tool activity is monitor-only and must never reach TTS."""
    out = _interactive([
        _f(type="say", kind="message", message_id="N", content="⚙️ computer_use...", final=True),
        _f(type="say", kind="stream", message_id="A", content="Your Things list is ready.", final=True),
        _f(type="turn_end", outcome="success"),
    ])
    assert out == ["Your Things list is ready."]


def test_bear_running_status_and_repeat_counter_do_not_capture_answer_stream():
    """Tool UI notices must not lock the accumulator away from the later real answer ID."""
    answer = (
        "Here's your note called Reading List. Start here with Marcus Aurelius, Meditations, "
        "and Plato, Apology."
    )
    out = _interactive([
        _f(
            type="say",
            kind="message",
            message_id="TOOL-1",
            content="💻 Running /Applications/Bear.app/Contents/MacOS...",
            final=True,
        ),
        _f(type="say", kind="message", message_id="TOOL-2", content="(×3)", final=True),
        _f(type="say", kind="stream", message_id="ANSWER", content=answer, final=True),
        _f(type="turn_end", outcome="success"),
    ])
    assert out == ["Here's your note called Reading List.", "Start here with Marcus Aurelius, Meditations, and Plato, Apology."]


def test_web_search_status_does_not_capture_weather_answer_stream():
    """A search notice must stay silent while the later weather answer remains speakable."""
    answer = (
        "Possibly, but it isn't certain. Denver has roughly a 20 to 35 percent chance of "
        "scattered showers this evening. I'd bring a light rain jacket just in case."
    )
    out = _interactive([
        _f(
            type="say",
            kind="message",
            message_id="SEARCH",
            content="🔍 Searching the web for Denver hourly weather...",
            final=True,
        ),
        _f(type="say", kind="stream", message_id="ANSWER", content=answer, final=True),
        _f(type="turn_end", outcome="success"),
    ])
    assert " ".join(out) == answer


def test_streaming_dedup_across_resend():
    out = _interactive([
        _f(type="say", kind="message", message_id="A", content="Satz eins. ▉", final=True),
        _f(type="say", kind="stream", message_id="A", content="Satz eins. Satz zwei. ▉", final=False),
        _f(type="say", kind="stream", message_id="A", content="Satz eins. Satz zwei. Satz drei.", final=True),
        _f(type="say", kind="message", message_id="B", content="Satz eins. Satz zwei. Satz drei.", final=True),
        _f(type="turn_end"),
    ])
    assert out == ["Satz eins.", "Satz zwei.", "Satz drei."]


def test_long_first_sentence_streams_first_clause_early():
    # A long first sentence should start speaking at the first clause boundary
    # (lower time-to-first-audio) rather than waiting for the whole sentence.
    out = _interactive([
        _f(type="say", kind="stream", message_id="A",
           content="Ein schwarzes Loch ist eine Region im All, ▉", final=False),
        _f(type="say", kind="stream", message_id="A",
           content="Ein schwarzes Loch ist eine Region im All, deren Gravitation so stark ist, dass nichts entkommt.",
           final=True),
        _f(type="turn_end"),
    ])
    assert out[0] == "Ein schwarzes Loch ist eine Region im All,"  # early clause
    assert len(out) >= 2
    assert "entkommt" in out[-1]


def test_trailing_sentence_without_terminator_flushed_on_turn_end():
    out = _interactive([
        _f(type="say", kind="stream", message_id="A", content="Kein Punkt hier", final=True),
        _f(type="turn_end"),
    ])
    assert out == ["Kein Punkt hier"]


def test_typing_frames_ignored():
    out = _interactive([
        _f(type="typing", robot_id="reachy"),
        _f(type="say", kind="stream", message_id="A", content="Alles gut.", final=True),
        _f(type="turn_end"),
    ])
    assert out == ["Alles gut."]


def test_empty_transcript_short_circuits():
    out = _interactive([], transcript="   ")
    assert out == ["I didn't quite catch that."]


# ── proactive (Stage 4) ───────────────────────────────────────────────────────
def test_proactive_delivery_between_turns():
    # frames arriving with NO active turn -> spoken via on_proactive
    captured = _proactive([
        _f(type="say", kind="message", message_id="P", content="Ergebnis: ▉", final=True),
        _f(type="say", kind="stream", message_id="P", content="Ergebnis: Die Hauptstadt ist Canberra.", final=True),
        _f(type="turn_end", outcome="success"),
    ])
    assert captured == ["Ergebnis: Die Hauptstadt ist Canberra."]


def test_proactive_skips_pure_notice():
    captured = _proactive([
        _f(type="say", kind="message", message_id="N", content="ℹ Nur ein Hinweis.", final=True),
        _f(type="turn_end"),
    ])
    assert captured == []


# ── barge-in / interrupt (Stage 3) ────────────────────────────────────────────
def test_accumulator_restart_ignores_straggler_and_relocks():
    acc = _AnswerAccumulator()
    # trailing space confirms the final sentence boundary (buffer-end punctuation is held otherwise)
    assert acc.feed({"type": "say", "kind": "stream", "message_id": "T1", "content": "Eins. Zwei. "}) == ["Eins.", "Zwei."]
    acc.restart()  # barge: cancel T1, re-lock on the next new id
    # a straggler from the cancelled turn is ignored
    assert acc.feed({"type": "say", "kind": "stream", "message_id": "T1", "content": "Eins. Zwei. Drei. "}) == []
    # the new (interrupt) turn is emitted from scratch
    assert acc.feed({"type": "say", "kind": "stream", "message_id": "T2", "content": "Verstanden. "}) == ["Verstanden."]


def test_interrupt_sends_frame_and_injects_barge_reset():
    import asyncio as _a

    class _WS:
        def __init__(self):
            self.sent = []

        async def send(self, m):
            self.sent.append(m)

    async def go():
        c = ReachyPlatformClient()
        c._ws = _WS()
        c._turn_q = _a.Queue()
        await c.interrupt("mach stattdessen etwas anderes")
        sent = json.loads(c._ws.sent[-1])
        ctrl = c._turn_q.get_nowait()
        return sent, ctrl

    sent, ctrl = asyncio.run(go())
    assert sent["type"] == "interrupt" and sent["text"] == "mach stattdessen etwas anderes"
    assert ctrl["type"] == "_barge_reset"


def test_interrupt_bare_stop_defaults_to_slash_stop():
    class _WS:
        def __init__(self):
            self.sent = []

        async def send(self, m):
            self.sent.append(m)

    async def go():
        c = ReachyPlatformClient()
        c._ws = _WS()
        await c.interrupt(None)
        return json.loads(c._ws.sent[-1])

    sent = asyncio.run(go())
    assert sent["type"] == "interrupt" and sent["text"] == "/stop"


# ── turn_id correlation routing (audit 2026-07-02, V5c) ────────────────────────
def test_route_by_turn_id_interactive_proactive_and_straggler():
    c = ReachyPlatformClient()
    c._turn_q = object()  # a turn is active (truthy sentinel; _route only checks None-ness)
    c._active_turn_id = "t-1"
    # matching interactive turn frame -> turn
    assert c._route({"type": "say", "turn_id": "t-1", "origin": "turn"}) == "turn"
    # proactive delivery arriving mid-turn -> proactive (NOT hijacking the turn)
    assert c._route({"type": "say", "turn_id": None, "origin": "proactive"}) == "proactive"
    # straggler of a different/cancelled turn -> dropped (not spoken as the answer)
    assert c._route({"type": "say", "turn_id": "t-OLD", "origin": "turn"}) == "drop"
    # frame without turn_id (older gateway) -> temporal fallback (turn active)
    assert c._route({"type": "say"}) == "turn"


def test_route_without_active_turn():
    c = ReachyPlatformClient()
    c._turn_q = None
    c._active_turn_id = None
    # proactive tagged -> proactive
    assert c._route({"type": "say", "turn_id": None, "origin": "proactive"}) == "proactive"
    # an interactive-tagged frame with no active turn -> drop (a straggler)
    assert c._route({"type": "say", "turn_id": "t-1", "origin": "turn"}) == "drop"
    # untagged frame, no active turn -> proactive (temporal fallback)
    assert c._route({"type": "say"}) == "proactive"


def test_ask_stream_sends_turn_id_and_reader_drops_mid_turn_straggler():
    # A proactive delivery interleaved into an active turn must NOT be spoken as part of the
    # reply, and a foreign-turn straggler must be dropped — only the active turn's frames stream.
    frames = [
        _f(type="say", kind="stream", message_id="A", content="Ein schwarzes Loch,", final=False, turn_id="__TID__", origin="turn"),
        _f(type="say", kind="message", message_id="P", content="Proaktiv dazwischen.", final=True, turn_id=None, origin="proactive"),
        _f(type="say", kind="stream", message_id="A", content="Ein schwarzes Loch, ist eine Region.", final=True, turn_id="__TID__", origin="turn"),
        _f(type="turn_end", outcome="success", turn_id="__TID__", origin="turn"),
        _f(type="turn_end", outcome="success", turn_id=None, origin="proactive"),
    ]

    captured_proactive = []
    c = ReachyPlatformClient()

    async def on_p(text):
        captured_proactive.append(text)

    c.set_proactive_handler(on_p)

    # FakeWS that stamps the real client turn_id into the queued frames once ask_stream sets it.
    class _TidWS(FakeWS):
        async def send(self, m):
            self.sent.append(m)
            tid = json.loads(m).get("turn_id")
            if tid:
                self._out = [fr.replace("__TID__", tid) for fr in self._out]

    ws = _TidWS(frames)
    c._ws = ws

    async def go():
        out = [chunk async for chunk in c.ask_stream("was ist ein schwarzes loch")]
        # drain the reader (proactive turn_end) + let the fire-and-forget proactive task run
        if c._reader_task is not None:
            await c._reader_task
        for _ in range(5):
            await asyncio.sleep(0)
        if c._proactive_tasks:
            await asyncio.gather(*list(c._proactive_tasks))
        return out

    out = asyncio.run(go())
    # only the active turn's content streamed
    assert out == ["Ein schwarzes Loch,", "ist eine Region."]
    # the interleaved proactive delivery was routed to the proactive handler, not the turn
    assert captured_proactive == ["Proaktiv dazwischen."]
    # and the stt frame carried a turn_id
    assert json.loads(ws.sent[0]).get("turn_id")


# ── review 2026-07-02 round 2: P1-8 (proactive without turn_end) + P1-9 (interrupt drain) ──
def test_proactive_without_turn_end_flushes_after_settle(monkeypatch):
    """Cron/send_message deliveries call adapter.send() directly and never emit a turn_end —
    the settle timer must flush them instead of waiting forever (P1-8)."""
    monkeypatch.setenv("AGENT_PROACTIVE_SETTLE_S", "0.05")
    captured = []
    c = ReachyPlatformClient()

    async def on_p(text):
        captured.append(text)

    c.set_proactive_handler(on_p)
    c._ws = FakeWS([
        _f(type="say", kind="message", message_id="C1", content="Cron: Backup fertig.",
           final=True, turn_id=None, origin="proactive"),
        # NO turn_end
    ])

    async def go():
        await c._ensure_session()
        await c._reader_task
        await asyncio.sleep(0.2)  # let the settle timer fire

    asyncio.run(go())
    assert captured == ["Cron: Backup fertig."]


def test_proactive_second_delivery_after_settle_not_swallowed(monkeypatch):
    """A delivery following a flushed (turn_end-less) one must not be swallowed by a jammed
    accumulator (P1-8 second half)."""
    monkeypatch.setenv("AGENT_PROACTIVE_SETTLE_S", "0.05")

    class DelayedWS(FakeWS):
        async def __anext__(self):
            if not self._out:
                raise StopAsyncIteration
            delay, frame = self._out.pop(0)
            await asyncio.sleep(delay)
            return frame

    captured = []
    c = ReachyPlatformClient()

    async def on_p(text):
        captured.append(text)

    c.set_proactive_handler(on_p)
    c._ws = DelayedWS([
        (0.0, _f(type="say", kind="message", message_id="C1", content="Erste Lieferung.",
                 final=True, turn_id=None, origin="proactive")),
        (0.15, _f(type="say", kind="message", message_id="C2", content="Zweite Lieferung.",
                  final=True, turn_id=None, origin="proactive")),
    ])

    async def go():
        await c._ensure_session()
        await c._reader_task
        await asyncio.sleep(0.2)

    asyncio.run(go())
    assert captured == ["Erste Lieferung.", "Zweite Lieferung."]


def test_proactive_streamed_with_turn_end_not_double_spoken(monkeypatch):
    """Streamed proactive answers (delegation watcher) end with turn_end: the settle timer must
    not produce a second delivery."""
    monkeypatch.setenv("AGENT_PROACTIVE_SETTLE_S", "0.05")
    captured = _proactive([
        _f(type="say", kind="message", message_id="P", content="Ergebnis: ▉", final=True,
           turn_id=None, origin="proactive"),
        _f(type="say", kind="stream", message_id="P", content="Ergebnis: Alles erledigt.",
           final=True, turn_id=None, origin="proactive"),
        _f(type="turn_end", outcome="success", turn_id=None, origin="proactive"),
    ])
    assert captured == ["Ergebnis: Alles erledigt."]


def test_interrupt_drains_stale_turn_queue():
    """interrupt() must drop the superseded turn's queued backlog (old full-resends and its
    turn_end) so the interrupt turn isn't preceded by stale speech or ended early (P1-9);
    a reader-disconnect poison (None) must survive the drain."""

    class _WS:
        def __init__(self):
            self.sent = []

        async def send(self, m):
            self.sent.append(m)

    async def go():
        c = ReachyPlatformClient()
        c._ws = _WS()
        q = asyncio.Queue()
        c._turn_q = q
        q.put_nowait(json.loads(_f(type="say", kind="stream", message_id="A",
                                   content="Alte Antwort, Teil eins.", final=False)))
        q.put_nowait(None)  # reader poison — must survive
        q.put_nowait(json.loads(_f(type="turn_end", outcome="success")))
        await c.interrupt("stopp, mach was anderes")
        items = []
        while True:
            try:
                items.append(q.get_nowait())
            except asyncio.QueueEmpty:
                break
        return items

    items = asyncio.run(go())
    assert items[0] == {"type": "_barge_reset"}
    assert items[1] is None
    assert len(items) == 2  # stale say + stale turn_end are gone


def test_concurrent_ensure_session_connects_once(monkeypatch):
    """Review 2026-07-02 round 2, P2: supervisor + ask_stream racing _ensure_session created two
    sockets/two hellos; the unread one could win the gateway's robot map -> permanent wedge."""
    connects = {"n": 0}

    class _OneWS(FakeWS):
        def __init__(self):
            super().__init__([])

        async def __anext__(self):
            await asyncio.sleep(3600)  # stay open: an exhausted fake lets the reader null _ws

    async def go():
        c = ReachyPlatformClient()

        async def fake_connect(url):
            connects["n"] += 1
            await asyncio.sleep(0.05)  # window in which the second caller would race in
            return _OneWS()

        import websockets.asyncio.client as wac
        monkeypatch.setattr(wac, "connect", fake_connect)
        await asyncio.gather(c._ensure_session(), c._ensure_session())
        return c._ws is not None  # check before asyncio.run tears the reader down

    had_ws = asyncio.run(go())
    assert connects["n"] == 1  # second caller waited on the lock and reused the socket
    assert had_ws


def test_successful_turn_with_no_speakable_text_stays_silent():
    """Review 2026-07-02 round 2, P3: a notice-only/empty SUCCESS turn falsely announced a
    connection problem."""
    out = _interactive([
        _f(type="say", kind="message", message_id="N", content="ℹ Nur ein Hinweis.", final=True),
        _f(type="turn_end", outcome="success"),
    ])
    assert out == []  # no false "Verbindung hakt" line

    # a failure outcome keeps the honest line
    out2 = _interactive([
        _f(type="turn_end", outcome="failure"),
    ])
    assert out2 == ["I'm having trouble connecting to the agent right now."]
