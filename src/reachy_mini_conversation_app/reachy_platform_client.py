"""Reachy *platform* transport client (Stage 2 + Stage 4).

An alternative to :class:`HermesVoiceClient` that talks to the gateway's **reachy
platform adapter** over a persistent WebSocket instead of the stateless HTTP
``/v1/chat/completions`` endpoint. Selected with ``AGENT_TRANSPORT=platform``.

Why: as a platform (like Telegram), the gateway session has
``supports_async_delivery=True`` — so background delegation, cron and
send_message can reach Reachy proactively (the HTTP path hardwires that off).
See reachy-agent-app/docs/plans/2026-07-01_reachy-as-hermes-platform.md.

Architecture: a single persistent ws with ONE background reader task. Frames are
routed by whether an interactive turn is in flight:
- ``ask_stream`` (a user turn) registers a queue; the reader routes the turn's
  frames to it and it yields voice-sized sentence chunks (same TTS path as HTTP).
- Frames arriving with NO active turn are **proactive** (async-delegation results,
  cron, send_message); the reader accumulates them and hands the finished text to
  ``on_proactive`` so the app can speak it in a safe half-duplex gap.

Streaming detail: the gateway resends the FULL accumulated answer each edit, marks
the in-progress tail with a ``▉`` cursor, and may re-send the finished answer as a
fresh message — so we strip the cursor, skip emoji-prefixed system/tool-status
notices, emit voice chunks by CHAR POSITION at stable clause/sentence boundaries
(first chunk clause-early for low latency), and end each turn at the explicit
``turn_end`` frame.
"""

from __future__ import annotations
import os
import re
import json
import uuid
import asyncio
import logging
from pathlib import Path
from typing import Any
from collections.abc import Callable, Awaitable

from reachy_mini_conversation_app.agent_clients import (
    _env_int,
    _env_float,
    _compose_user,
    _trim_for_voice,
    _drain_voice_chunks,
)


logger = logging.getLogger(__name__)

# A run of text ending in sentence punctuation or a newline = one complete sentence.
_SENTENCE_RE = re.compile(r"[^.!?…\n]*(?:[.!?…]+|\n+)", re.S)

# Block-drawing streaming cursor the gateway appends to in-progress edits
# (default ▉ = U+2589); strip the whole block range to be robust to config.
_CURSORS = "█▉▊▋▌▍▎▏"
_CURSOR_STRIP = {ord(c): None for c in _CURSORS}
# System/tool-status notices the gateway pushes as standalone messages; not
# conversational answer text, so not spoken.
_NOTICE_PREFIXES = ("ℹ", "\U0001f4ec", "✅", "\U0001f40d", "⚡", "\U0001f4e1",
                    "\U0001f514", "\U0001f916", "♻", "⏳", "❌", "⚠",
                    "⚙", "\U0001f6e0", "\U0001f527", "\U0001f4ce", "\U0001f4f7",
                    "\U0001f3a4", "\U0001f500")
_EMOJI_PREFIX_RE = re.compile(
    r"^[\U0001F1E6-\U0001F1FF\U0001F300-\U0001FAFF\u2600-\u27BF]"
)
_REPEAT_NOTICE_RE = re.compile(r"^\(\s*[×x]\s*\d+\s*\)$", re.IGNORECASE)


def _complete_sentences(text: str) -> tuple[list[str], str]:
    """Split into (complete sentences, trailing incomplete remainder)."""
    out: list[str] = []
    pos = 0
    for m in _SENTENCE_RE.finditer(text):
        s = m.group().strip()
        if s:
            out.append(s)
        pos = m.end()
    return out, text[pos:]


def _looks_like_notice(text: str) -> bool:
    """A gateway system/tool-status line, not user-facing answer text."""
    t = (text or "").lstrip()
    return bool(t) and (
        t.startswith(_NOTICE_PREFIXES)
        or _EMOJI_PREFIX_RE.match(t) is not None
        or _REPEAT_NOTICE_RE.fullmatch(t) is not None
    )


class _AnswerAccumulator:
    """Turns a stream of gateway ``say`` frames into ordered, deduped, voice-sized
    chunks.

    Locks onto the first non-notice message id, strips the streaming cursor, and
    emits via the shared ``_drain_voice_chunks``: the FIRST chunk may end at a clause
    boundary (comma/dash/colon) once it is long enough — so long first sentences start
    speaking sooner — and later chunks end at sentence boundaries for natural prosody.
    Robust to the gateway's full-resend + tail revision (only stable, already-passed
    boundaries are emitted; ``_emitted_len`` tracks the char position) and to
    fresh-final resends under a NEW id (ignored via the answer-id lock).
    """

    def __init__(self) -> None:
        self.answer_id: str | None = None
        self.last_full = ""
        self._ignored: set = set()  # message ids from interrupted (barged) turns
        self._emitted_len = 0       # chars of the answer already emitted (a stable prefix)
        self._produced = False      # first chunk emitted -> later chunks are sentence-bounded
        self._min = _env_int("AGENT_FIRST_CHUNK_MIN_CHARS", 15)

    def restart(self) -> None:
        """After a barge/interrupt: forget the current (now-cancelled) answer and
        re-lock onto the NEXT new message, ignoring stragglers of the old one.
        """
        if self.answer_id is not None:
            self._ignored.add(self.answer_id)
        self.answer_id = None
        self.last_full = ""
        self._emitted_len = 0
        self._produced = False

    def feed(self, frame: dict) -> list[str]:
        """Consume one ``say`` frame; return newly-complete voice chunks to emit."""
        content = (frame.get("content") or "").translate(_CURSOR_STRIP)
        mid = frame.get("message_id")
        if mid in self._ignored:
            return []  # a straggler from an interrupted turn
        if self.answer_id is None:
            if _looks_like_notice(content) and frame.get("kind") == "message":
                return []  # standalone notice / tool-status — not spoken
            self.answer_id = mid  # first non-notice message = the answer
        if mid != self.answer_id:
            return []  # a concurrent notice while the answer streams — skip
        self.last_full = content
        candidate = content[self._emitted_len:]  # unemitted suffix (emitted prefix is stable)
        chunks, remaining, self._produced = _drain_voice_chunks(candidate, self._produced, self._min)
        self._emitted_len += len(candidate) - len(remaining)
        return chunks

    def finish(self) -> list[str]:
        """Flush the final answer incl. a trailing chunk with no terminator."""
        tail = self.last_full[self._emitted_len:].strip()
        self._emitted_len = len(self.last_full)
        return [tail] if tail else []


class ReachyPlatformConfig:
    """Config for the platform ws transport (env-driven)."""

    def __init__(self) -> None:
        self.ws_url = os.getenv("AGENT_PLATFORM_WS_URL", "ws://127.0.0.1:8770/robot/reachy").strip()
        self.robot_id = os.getenv("AGENT_PLATFORM_ROBOT_ID", "reachy").strip() or "reachy"
        self.api_key = os.getenv("AGENT_PLATFORM_API_KEY", "").strip()
        key_file = os.getenv("AGENT_PLATFORM_API_KEY_FILE", "").strip()
        if not self.api_key and key_file:
            try:
                self.api_key = Path(key_file).expanduser().read_text(encoding="utf-8").strip()
            except OSError as exc:
                logger.error("[reachy-platform] could not read API key file %s: %s", key_file, exc)
        self.turn_timeout_s = _env_float("AGENT_PLATFORM_TURN_TIMEOUT_S", 90.0)
        self.connect_timeout_s = _env_float("AGENT_PLATFORM_CONNECT_TIMEOUT_S", 10.0)
        self.max_response_chars = _env_int("AGENT_MAX_RESPONSE_CHARS", 2000)


ProactiveHandler = Callable[[str], Awaitable[None]]


class ReachyPlatformClient:
    """Drop-in for HermesVoiceClient over the gateway reachy platform (ws).

    Holds one persistent connection with a background reader that routes turn
    frames to the active ``ask_stream`` and proactive frames to ``on_proactive``.
    """

    def __init__(
        self,
        config: ReachyPlatformConfig | None = None,
        on_proactive: ProactiveHandler | None = None,
    ) -> None:
        self.config = config or ReachyPlatformConfig()
        self._on_proactive = on_proactive
        self._ws = None
        self._reader_task: asyncio.Task | None = None
        self._turn_q: asyncio.Queue | None = None  # set while an interactive turn is in flight
        self._active_turn_id: str | None = None  # client turn id the reader routes frames by
        self._turn_lock = asyncio.Lock()  # single-flight interactive turns
        self._closed = False  # final shutdown flag: stops the reconnect supervisor
        self._supervisor_task: asyncio.Task | None = None
        self._proactive_tasks: set[asyncio.Task] = set()  # keep refs (loop holds tasks weakly)
        self.on_turn_progress: Callable[[], None] | None = None  # stall-watchdog liveness hook
        # Body-tool surface (gap-map Stufe 3): async callback(action, params) -> result dict.
        self.on_tool_call: Callable[[str, dict], Any] | None = None
        self._tool_tasks: set[asyncio.Task] = set()
        # Serializes session creation: supervisor and ask_stream both call _ensure_session; two
        # concurrent connects raced to two sockets/two hellos, and if the UNREAD one won the
        # gateway's robot map, every reply ran into the void until app restart (review
        # 2026-07-02 round 2, P2).
        self._connect_lock = asyncio.Lock()

    async def start(self) -> None:
        """Start maintaining the session in the background (reconnect supervisor).

        Without this the ws only ever connected inside ask_stream — the proactive channel
        (the whole point of the platform transport) was dead from app start until the first
        user turn and after every gateway restart (audit 2026-07-02).
        """
        self._closed = False
        if self._supervisor_task is None or self._supervisor_task.done():
            self._supervisor_task = asyncio.create_task(self._supervise())

    async def _supervise(self) -> None:
        backoff = 1.0
        while not self._closed:
            if self._ws is None:
                try:
                    await self._ensure_session()
                    backoff = 1.0
                except Exception as e:
                    logger.info("[reachy-platform] reconnect failed (%s) — retry in %.0fs", e, backoff)
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2.0, 30.0)
                    continue
            await asyncio.sleep(1.0)

    def set_proactive_handler(self, handler: ProactiveHandler | None) -> None:
        """Register the async callback that speaks proactively-delivered text."""
        self._on_proactive = handler

    # ── session / reader ────────────────────────────────────────────────────
    async def _ensure_session(self) -> None:
        async with self._connect_lock:
            if self._ws is None:
                from websockets.asyncio.client import connect

                ws = await asyncio.wait_for(connect(self.config.ws_url), timeout=self.config.connect_timeout_s)
                try:
                    await ws.send(
                        json.dumps(
                            {
                                "type": "hello",
                                "robot_id": self.config.robot_id,
                                "api_key": self.config.api_key,
                            }
                        )
                    )
                except Exception:
                    # gateway reset between connect and hello: close instead of leaking the socket
                    try:
                        await ws.close()
                    except Exception:
                        pass
                    raise
                self._ws = ws
                logger.info("[reachy-platform] connected to %s", self.config.ws_url)
            if self._reader_task is None or self._reader_task.done():
                self._reader_task = asyncio.create_task(self._reader())

    def _route(self, frame: dict) -> str:
        """Decide where an inbound frame goes: 'turn' (active interactive turn), 'proactive'
        (unsolicited delivery), or 'drop' (straggler of a cancelled/older turn).

        Uses the gateway's turn_id/origin correlation (audit 2026-07-02, V5c). Falls back to the
        old purely-temporal rule when the frame carries no turn_id (older gateway): active turn -> turn."""
        has_tid = "turn_id" in frame
        if not has_tid:
            return "turn" if self._turn_q is not None else "proactive"
        ftid = frame.get("turn_id")
        origin = frame.get("origin")
        if origin == "proactive" or ftid is None:
            return "proactive"
        # origin == "turn"
        if self._turn_q is not None and ftid == self._active_turn_id:
            return "turn"
        # a real reply, but for a turn we're no longer consuming (cancelled/superseded) -> drop
        return "drop"

    async def _reader(self) -> None:
        """Single consumer of the ws: route each frame by turn_id to the active turn queue,
        the proactive handler, or drop (straggler of a cancelled turn).
        """
        ws = self._ws
        prot = _AnswerAccumulator()
        prot_chunks: list[str] = []
        settle_task: asyncio.Task | None = None
        settle_s = float(os.getenv("AGENT_PROACTIVE_SETTLE_S", "3.0"))

        def _flush_proactive() -> None:
            # Shared by turn_end (normal end of a proactive handle_message turn) and the
            # settle timer: cron-delivery and the send_message tool call adapter.send()
            # directly and never emit a turn_end — waiting for one silenced those deliveries
            # forever and jammed the accumulator for the NEXT one (review 2026-07-02, P1-8).
            nonlocal prot, prot_chunks
            prot_chunks.extend(prot.finish())
            text = " ".join(c.strip() for c in prot_chunks if c.strip()).strip()
            prot, prot_chunks = _AnswerAccumulator(), []
            if text and self._on_proactive is not None:
                # Fire-and-forget: awaiting the handler here blocked the reader for the
                # whole spoken delivery (and up to 25s of lock-wait), starving an active
                # turn of its frames (audit 2026-07-02). The handler serializes itself
                # via the turn lock; we just keep a reference against loop GC.
                t = asyncio.create_task(self._speak_proactive_safe(text))
                self._proactive_tasks.add(t)
                t.add_done_callback(self._proactive_tasks.discard)

        def _cancel_settle() -> None:
            nonlocal settle_task
            if settle_task is not None and not settle_task.done():
                settle_task.cancel()
            settle_task = None

        def _reschedule_settle() -> None:
            nonlocal settle_task
            _cancel_settle()

            async def _settle() -> None:
                try:
                    await asyncio.sleep(settle_s)
                except asyncio.CancelledError:
                    return
                _flush_proactive()

            settle_task = asyncio.create_task(_settle())

        try:
            async for raw in ws:
                try:
                    frame = json.loads(raw)
                except Exception:
                    continue
                if frame.get("type") == "tool_call":
                    # gateway-requested body action — independent of turn routing
                    t = asyncio.create_task(self._run_tool_call(frame))
                    self._tool_tasks.add(t)
                    t.add_done_callback(self._tool_tasks.discard)
                    continue
                dest = self._route(frame)
                if dest == "drop":
                    continue
                if dest == "turn":
                    # Any routed turn frame (say/typing/turn_end) is proof the gateway is alive —
                    # the handler's stall watchdog hooks in here so a long silent tool phase
                    # (typing every ~2s, no audio) doesn't count as a stall (review 2026-07-02, P1-4).
                    cb = self.on_turn_progress
                    if cb is not None:
                        try:
                            cb()
                        except Exception:
                            pass
                    self._turn_q.put_nowait(frame)
                    continue
                # proactive delivery
                ty = frame.get("type")
                if ty == "say":
                    prot_chunks.extend(prot.feed(frame))
                    _reschedule_settle()
                elif ty == "turn_end":
                    _cancel_settle()
                    _flush_proactive()
        except Exception as e:
            logger.info("[reachy-platform] reader ended: %s", e)
        finally:
            # unblock any waiting turn and drop the session so the supervisor/next turn
            # reconnects — but only OUR session (a stale reader must not clobber a newer ws).
            if self._turn_q is not None:
                self._turn_q.put_nowait(None)
            if self._ws is ws:
                self._ws = None

    async def _run_tool_call(self, frame: dict) -> None:
        """Execute one gateway body-tool request and send the tool_result back."""
        tcid = str(frame.get("tool_call_id") or "")
        from reachy_mini_conversation_app.pipeline_monitor import get_pipeline_monitor

        monitor = get_pipeline_monitor()
        action = str(frame.get("action") or "")
        if monitor is not None:
            monitor.emit("tool", action, status="started", tool_call_id=tcid)
        handler = self.on_tool_call
        if handler is None:
            result: dict = {"error": "no tool handler registered"}
        else:
            try:
                result = await handler(action, frame.get("params") or {})
                if not isinstance(result, dict):
                    result = {"result": result}
            except Exception as exc:
                logger.warning("[reachy-platform] tool call failed: %s", exc)
                result = {"error": f"{type(exc).__name__}: {exc}"}
        ws = self._ws
        if ws is None:
            return
        try:
            await ws.send(json.dumps(
                {"type": "tool_result", "tool_call_id": tcid, "result": result,
                 "robot_id": self.config.robot_id},
                ensure_ascii=False,
            ))
            if monitor is not None:
                monitor.emit("tool", action, status="completed", tool_call_id=tcid, ok="error" not in result)
        except Exception as exc:
            logger.warning("[reachy-platform] tool_result send failed: %s", exc)

    async def _speak_proactive_safe(self, text: str) -> None:
        try:
            handler = self._on_proactive
            if handler is not None:
                await handler(text)
        except Exception as e:  # never let a handler kill anything
            logger.warning("[reachy-platform] proactive handler failed: %s", e)

    async def _reset_session(self, only_if=None) -> None:
        """Drop the current ws/reader after a turn error so the supervisor (or the next turn)
        reconnects. Does NOT stop the supervisor — that's aclose()'s job.

        ``only_if``: the ws the failing turn was using. A turn generator can wake seconds after
        the reader died (pacing) — by then the supervisor may have reconnected, and resetting
        unconditionally tore down the fresh healthy session (review 2026-07-02 round 2, P3)."""
        if only_if is not None and self._ws is not None and self._ws is not only_if:
            return  # a newer session exists — leave it alone
        ws, self._ws = self._ws, None
        task, self._reader_task = self._reader_task, None
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass
        if task is not None:
            task.cancel()

    async def aclose(self) -> None:
        """Final shutdown: stop the reconnect supervisor and close the session."""
        self._closed = True
        sup, self._supervisor_task = self._supervisor_task, None
        if sup is not None:
            sup.cancel()
        await self._reset_session()

    async def interrupt(self, text: str | None = None) -> None:
        """Barge-in: cancel the gateway's in-flight turn. ``text`` = a committed
        interrupt command (the gateway's ``interrupt`` mode cancels the running
        turn and answers this instead); ``None``/empty = a bare stop (``/stop``).

        Also rolls the active turn_id to a NEW id and injects a local ``_barge_reset``
        control frame so the active ``ask_stream`` re-locks onto the interrupt turn; the
        gateway tags the new turn's frames with this id and the cancelled turn's stragglers
        with the old one, so the reader routes/drops them correctly (audit 2026-07-02, V5c).
        """
        new_turn_id = uuid.uuid4().hex
        self._active_turn_id = new_turn_id
        q = self._turn_q
        if q is not None:
            # Drop the superseded turn's queued backlog FIRST: when the gateway produces faster
            # than we speak, the queue holds full-resends (and possibly the old turn_end). Left
            # in place, they were spoken before the reset — and an old turn_end ended ask_stream
            # before the interrupt turn's answer arrived, losing the committed question
            # (review 2026-07-02 round 2, P1-9). The reader routes anything arriving from now on
            # by the NEW turn_id, so only pre-roll frames need draining.
            saw_poison = False
            while True:
                try:
                    stale = q.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if stale is None:
                    saw_poison = True  # reader-disconnect signal — must survive the drain
            q.put_nowait({"type": "_barge_reset"})
            if saw_poison:
                q.put_nowait(None)
        ws = self._ws
        if ws is None:
            return
        payload = (text or "").strip() or "/stop"
        try:
            await ws.send(json.dumps(
                {"type": "interrupt", "text": payload, "robot_id": self.config.robot_id, "turn_id": new_turn_id}
            ))
        except Exception as e:
            logger.warning("[reachy-platform] interrupt send failed: %s", e)

    # ── interactive turn ────────────────────────────────────────────────────
    async def ask_stream(self, transcript: str, context: str | None = None, image_url: str | None = None):
        """Yield voice-sized sentence chunks of AGENT's reply over the ws transport.

        ``image_url`` is not carried by the platform text channel in v1 (native
        vision stays on the HTTP path); ``context`` is folded into the user turn.
        """
        cleaned = (transcript or "").strip()
        if not cleaned:
            yield "I didn't quite catch that."
            return
        composed = _compose_user(cleaned, context)

        async with self._turn_lock:
            q: asyncio.Queue = asyncio.Queue()
            self._turn_q = q  # register BEFORE the reader starts so no frame is missed
            turn_id = uuid.uuid4().hex
            self._active_turn_id = turn_id  # reader routes only this turn's frames here
            turn_ws = None
            try:
                await self._ensure_session()
                turn_ws = self._ws
                await self._ws.send(json.dumps(
                    {"type": "stt", "text": composed, "robot_id": self.config.robot_id, "turn_id": turn_id}
                ))
            except Exception as e:
                self._turn_q = None
                self._active_turn_id = None  # invariant: no active turn -> frames route proactive
                logger.warning("[reachy-platform] send failed: %s", e)
                await self._reset_session(only_if=turn_ws)  # keep the reconnect supervisor alive
                yield "I'm having trouble connecting to the agent right now."
                return

            acc = _AnswerAccumulator()
            produced = False
            turn_outcome = None
            try:
                while True:
                    frame = await asyncio.wait_for(q.get(), timeout=self.config.turn_timeout_s)
                    if frame is None:  # reader signalled disconnect
                        raise ConnectionError("reader closed")
                    if frame.get("type") == "_barge_reset":
                        # local control frame from interrupt(): the gateway is cancelling
                        # this turn; re-lock the accumulator onto the new (interrupt) turn.
                        acc.restart()
                        continue
                    if frame.get("type") == "turn_end":
                        turn_outcome = frame.get("outcome")
                        break
                    if frame.get("type") != "say":
                        continue
                    for chunk in acc.feed(frame):
                        produced = True
                        yield chunk
                for chunk in acc.finish():
                    produced = True
                    yield chunk
                if not produced:
                    # A SUCCESSFUL turn that just had nothing speakable (notice-only / empty
                    # answer) is not a connection problem — announcing one was misleading
                    # (review 2026-07-02 round 2, P3). Failure outcomes keep the honest line.
                    if turn_outcome == "success":
                        logger.info("[reachy-platform] turn ended successfully with no speakable text")
                    else:
                        yield "I'm having trouble connecting to the agent right now."
            except asyncio.TimeoutError:
                logger.warning("[reachy-platform] turn timed out")
                # Stop the gateway's generation: without the /stop it keeps producing, and the
                # follow-up full-resend frames (arriving after _turn_q=None) would be spoken as
                # a proactive DUPLICATE of the whole answer (audit 2026-07-02).
                try:
                    await self.interrupt(None)
                except Exception:
                    pass
                if not produced:
                    yield "This is taking unusually long. Please ask me again in a moment."
            except Exception as e:
                logger.warning("[reachy-platform] turn failed: %s", e)
                await self._reset_session(only_if=turn_ws)  # keep the reconnect supervisor alive
                if not produced:
                    yield "I'm having trouble connecting to the agent right now."
            finally:
                self._turn_q = None
                self._active_turn_id = None  # no active turn -> later frames route as proactive

    async def ask(self, transcript: str, context: str | None = None, image_url: str | None = None) -> str:
        parts = [chunk async for chunk in self.ask_stream(transcript, context, image_url)]
        text = " ".join(p.strip() for p in parts if p and p.strip())
        return _trim_for_voice(
            text or "I'm having trouble connecting to the agent right now.", self.config.max_response_chars
        )
