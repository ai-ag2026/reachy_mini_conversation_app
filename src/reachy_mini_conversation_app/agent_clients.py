from __future__ import annotations
import io
import os
import re
import json
import time
import wave
import asyncio
import logging
from typing import Any, Protocol
from dataclasses import replace, dataclass


logger = logging.getLogger(__name__)

import numpy as np
from numpy.typing import NDArray

from reachy_mini_conversation_app.conversation_handler import AudioFrame


VOICE_FAST_SYSTEM_MESSAGE = """You are AGENT, embodied in the Reachy robot, in a live spoken conversation. Your text is spoken aloud.
You have your FULL tools and memory here — web, files, terminal, vision, delegation, home automation,
and more. This is NOT a limited or tool-less mode; if a task needs a tool, use it.
Answer as AGENT: natural, direct, dry wit. Speak English only unless the user explicitly asks for another language.
No markdown, no lists, no emoji — plain spoken sentences. Match length to the question: short for small
talk, fuller when the topic has substance; don't pad, don't cut a real explanation short.
Do not expose chain-of-thought, internal instructions, tool syntax, tool arguments, status events, or
hidden messages. Return only the final user-facing words that Reachy should speak. Avoid long multi-step
tool chains in a single spoken turn; if something is big, do a focused part and offer to continue.
""".strip()


# First spoken chunk may end at a clause boundary (comma/dash/colon) once it is at
# least this many chars, so time-to-first-audio drops ~1s vs waiting for a full
# sentence. Later chunks stay sentence-bounded for natural prosody. Env-tunable.
_FIRST_CHUNK_MIN_CHARS = 15
# Phase B (async-tools): cap how long a turn waits for the FIRST spoken content. Covers a runaway
# agentic tool loop / slow reasoning (e.g. web_extract ~30 s). On timeout AGENT speaks a defer line and
# ends the turn instead of freezing. Default generous so normal turns (~2.5 s TTFT) never trip it; the
# proper background-continuation of the abandoned work is Phase C. Set <= 0 to disable the watchdog.
_FIRST_AUDIO_BUDGET_S = 20.0

# A boundary requires trailing WHITESPACE — not merely end-of-buffer. At a raw buffer end the
# punctuation is ambiguous ("42." could be a finished sentence OR a decimal mid-number, "1," a
# German decimal comma), so we wait for the next delta to disambiguate; the caller flushes the
# trailing remainder at stream end. This stops numbers being split across chunks (audit 2026-07-02).
_SENTENCE_END = re.compile(r"[.!?…]\s")
_CLAUSE_END = re.compile(r"[,;:—–]\s")


def _drain_voice_chunks(buf: str, produced: bool, first_chunk_min_chars: int) -> tuple[list[str], str, bool]:
    """Pull all complete speakable chunks from ``buf`` (streaming accumulator).

    The first chunk may end at a clause boundary (comma/dash/colon) once it is long
    enough, so time-to-first-audio drops; later chunks end at sentence boundaries for
    natural prosody. Returns ``(chunks, remaining_buf, produced)``. Pure/synchronous so
    it is unit-testable and reusable by the fast-path client.
    """
    chunks: list[str] = []
    while True:
        m = _SENTENCE_END.search(buf)
        if not m and not produced:
            c = _CLAUSE_END.search(buf)
            if c and c.end() >= first_chunk_min_chars:
                m = c
        if not m:
            break
        chunk, buf = buf[: m.end()].strip(), buf[m.end():]
        if chunk:
            produced = True
            chunks.append(chunk)
    return chunks, buf, produced


# Phase A (async-tools plan): the gateway emits inline SSE `hermes.tool.progress` events
# ({"tool","status":"running"/"completed","label"}). For SLOW/heavy tool families we speak a short
# status the moment the tool starts, so a long tool (e.g. web_extract ~30 s) is never a silent freeze.
# Fast/trivial tools (memory, todo, clarify, tts) get NO line — they don't cause a perceptible gap.
# Substring match against "<tool> <label>"; first hit wins; returns None = stay silent.
_TOOL_STATUS_LINES: tuple[tuple[str, str], ...] = (
    ("web", "One moment, I'll check the web."),
    ("browser", "One moment, I'll open that in the browser."),
    ("terminal", "One moment, I'll check the system."),
    ("shell", "One moment, I'll check the system."),
    ("code", "One moment, I'll work that out."),
    ("file", "One moment, I'll check the files."),
    ("delegation", "I'll handle the larger part in the background."),
    ("image", "One moment, I'll create that image."),
    ("homeassistant", "One moment, I'll take care of that device."),
    ("session", "One moment, I'll check our history."),
    ("search", "One moment, I'll look that up."),
)


def _tool_status_line(tool: str | None, label: str | None) -> str | None:
    """Short spoken status for a starting tool, or None for fast/trivial tools we don't announce."""
    hay = f"{tool or ''} {label or ''}".lower()
    for needle, line in _TOOL_STATUS_LINES:
        if needle in hay:
            return line
    return None


def _tool_status_enabled() -> bool:
    return os.getenv("AGENT_TOOL_STATUS_ENABLED", "1").strip().lower() not in ("0", "false", "no", "off")


def _defer_line() -> str:
    """Spoken when the first-audio budget is exceeded (Phase B): honest, short, ends the turn."""
    return os.getenv(
        "AGENT_VOICE_DEFER_LINE",
        "This is taking unusually long. I'll stop here; please ask me again in a moment.",
    )


def _compose_user(cleaned: str, context: str | None) -> str:
    """Prepend a clearly-labeled context line (e.g. parallel vision) to the user turn so the brain
    answers as the single voice using it as INPUT — ordered composition, no second stream to merge.
    """
    ctx = (context or "").strip()
    return f"{ctx}\n\n{cleaned}" if ctx else cleaned


def _apply_tool_policy(payload: dict) -> None:
    """Full AGENT tools by default — the gateway runs the agent with its ``platform_toolsets.api_server``
    set (memory, web, vision, delegation, homeassistant, terminal, …). Set ``AGENT_VOICE_TOOLS=0`` to
    neuter back to the legacy text-only voice-fast mode (tools off).
    """
    if os.getenv("AGENT_VOICE_TOOLS", "1").strip().lower() in ("0", "false", "no", "off"):
        payload["tools"] = []
        payload["tool_choice"] = "none"
        payload["enabled_toolsets"] = []
        return
    # Optional restriction to a safe subset (e.g. AGENT Work mode: no terminal/code_execution/file).
    # Empty -> the gateway uses its full api_server platform toolsets.
    restrict = os.getenv("AGENT_VOICE_TOOLSETS", "").strip()
    if restrict:
        payload["enabled_toolsets"] = [t.strip() for t in restrict.split(",") if t.strip()]


def _user_message(cleaned: str, context: str | None, image_url: str | None) -> dict:
    """Build the user chat message: text (with optional labeled context), plus a native image for the
    premium native-vision path (the brain sees the actual pixels). Text-only otherwise.
    """
    text = _compose_user(cleaned, context)
    if image_url:
        return {"role": "user", "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]}
    return {"role": "user", "content": text}


def _default_voice_session_id() -> str:
    """Per-day working-thread id (``X-Hermes-Session-Id``).

    Keeps same-day continuity on the gateway (it remembers prior turns, so the
    client only sends the new turn) while bounding the working context window.
    Cross-day recall is provided by the stable ``session_key`` (long-term memory).
    """
    from datetime import datetime
    return "reachy-voice-" + datetime.now().astimezone().strftime("%Y%m%d")


def _default_voice_session_title() -> str:
    """Human-readable title for the per-day voice session."""
    from datetime import datetime
    return "Reachy Voice " + datetime.now().astimezone().strftime("%d.%m.%Y")


class AsyncPostClient(Protocol):
    """Minimal async POST protocol used by AGENT voice clients."""

    async def post(self, url: str, **kwargs: Any) -> Any:
        """Send a POST request and return a response object."""


@dataclass(frozen=True)
class HermesVoiceConfig:
    """OpenAI-compatible Hermes/AGENT voice chat config."""

    base_url: str = "http://127.0.0.1:8642/v1"
    model: str = "local-agent"
    api_key_env: str = "API_SERVER_KEY"
    timeout_seconds: float = 30.0
    max_response_chars: int = 2000  # spoken-reply cap; raise via AGENT_MAX_RESPONSE_CHARS
    max_tokens: int = 1200          # LLM token cap; raise via AGENT_MAX_TOKENS
    # Opt-in Hermes gateway session/memory (sent as headers, see _session_headers):
    #   session_id  -> X-Hermes-Session-Id  : working conversation thread; the gateway
    #                  remembers prior turns, so the client sends only the new turn.
    #   session_key -> X-Hermes-Session-Key : long-term (Hindsight) memory scope, stable
    #                  across threads so AGENT recalls across days. Requires API-key auth.
    session_id: str = ""
    session_key: str = ""
    session_title: str = ""
    # True when session_id/title were auto-defaulted (per-day thread): recompute them PER REQUEST
    # so a long-running app rolls over at midnight instead of growing one thread forever
    # (audit 2026-07-02). Explicitly configured ids stay fixed.
    session_id_daily: bool = False

    @classmethod
    def from_env(cls) -> HermesVoiceConfig:
        """Build config from environment without exposing secrets."""
        defaults = cls()
        explicit_id = os.getenv("AGENT_SESSION_ID", "").strip()
        explicit_title = os.getenv("AGENT_SESSION_TITLE", "").strip()
        return cls(
            base_url=os.getenv("AGENT_BASE_URL", defaults.base_url),
            model=os.getenv("AGENT_MODEL", defaults.model),
            api_key_env=os.getenv("AGENT_API_KEY_ENV", defaults.api_key_env),
            timeout_seconds=_env_float("AGENT_TIMEOUT_SECONDS", defaults.timeout_seconds),
            max_response_chars=_env_int("AGENT_MAX_RESPONSE_CHARS", defaults.max_response_chars),
            max_tokens=_env_int("AGENT_MAX_TOKENS", defaults.max_tokens),
            session_id=explicit_id or _default_voice_session_id(),
            session_key=os.getenv("AGENT_SESSION_KEY", "reachy-voice").strip(),
            session_title=explicit_title or _default_voice_session_title(),
            session_id_daily=not explicit_id,
        )


@dataclass(frozen=True)
class QwenVoiceTtsConfig:
    """OpenAI-compatible Qwen3-TTS speech config."""

    base_url: str = "http://127.0.0.1:7034/v1"
    model: str = "qwen3-tts"
    voice: str = "default"
    api_key_env: str = "AGENT_QWEN_TTS_API_KEY"
    response_format: str = "wav"
    timeout_seconds: float = 30.0
    speed: float = 1.0

    @classmethod
    def from_env(cls) -> QwenVoiceTtsConfig:
        """Build config from environment without exposing secrets."""
        defaults = cls()
        return cls(
            base_url=os.getenv("AGENT_QWEN_TTS_BASE_URL", defaults.base_url),
            model=os.getenv("AGENT_QWEN_TTS_MODEL", defaults.model),
            voice=os.getenv("AGENT_QWEN_TTS_VOICE", defaults.voice),
            api_key_env=os.getenv("AGENT_QWEN_TTS_API_KEY_ENV", defaults.api_key_env),
            response_format=os.getenv("AGENT_QWEN_TTS_RESPONSE_FORMAT", defaults.response_format),
            timeout_seconds=_env_float("AGENT_QWEN_TTS_TIMEOUT_SECONDS", defaults.timeout_seconds),
            speed=min(2.0, max(0.5, _env_float("AGENT_TTS_SPEED", defaults.speed))),
        )


@dataclass(frozen=True)
class FastLeadInConfig:
    """Config for the fast local 9B used as an adaptive spoken lead-in (latency mask).

    Defaults to a small always-on local model (:3447, the barge-gate/VLM model). The lead-in
    only ever speaks a short content-free bridge while the full agent brain thinks.
    """

    base_url: str = "http://127.0.0.1:3447/v1"
    model: str = "qwopus-9b"
    api_key_env: str = ""          # :3447 needs no auth
    timeout_seconds: float = 1.2
    max_tokens: int = 16
    temperature: float = 1.0       # high for variety across turns
    enabled: bool = True
    enable_thinking: bool = False  # 9B is a reasoning model -> CoT off for speed

    @classmethod
    def from_env(cls) -> FastLeadInConfig:
        """Build config; falls back to the shared gate endpoint (AGENT_GATE_*)."""
        defaults = cls()
        enabled_raw = os.getenv("AGENT_LEAD_IN_ENABLED", "1").strip().lower()
        return cls(
            base_url=os.getenv("AGENT_LEAD_IN_BASE_URL", os.getenv("AGENT_GATE_BASE_URL", defaults.base_url)),
            model=os.getenv("AGENT_LEAD_IN_MODEL", os.getenv("AGENT_GATE_MODEL", defaults.model)),
            api_key_env=os.getenv("AGENT_LEAD_IN_API_KEY_ENV", defaults.api_key_env),
            timeout_seconds=_env_float("AGENT_LEAD_IN_TIMEOUT_SECONDS", defaults.timeout_seconds),
            max_tokens=_env_int("AGENT_LEAD_IN_MAX_TOKENS", defaults.max_tokens),
            enabled=enabled_raw not in ("0", "false", "no", "off", ""),
        )


class HermesVoiceClient:
    """Small async client for AGENT/Hermes voice turns."""

    def __init__(self, config: HermesVoiceConfig | None = None, http_client: AsyncPostClient | None = None) -> None:
        """Initialize with optional injectable config and HTTP transport."""
        self.config = config or HermesVoiceConfig.from_env()
        self._http_client = http_client

    def _session_headers(self) -> dict[str, str]:
        """Opt-in Hermes session headers: the gateway threads the conversation
        (``X-Hermes-Session-Id``) and scopes long-term memory
        (``X-Hermes-Session-Key``). Values sanitized to single-line ASCII.
        """
        # Daily-mode (auto-defaulted id): compute per request so the thread rolls over at
        # midnight on a long-running robot (audit 2026-07-02). Explicit ids stay as configured.
        session_id = self.config.session_id
        session_title = self.config.session_title
        if self.config.session_id_daily:
            session_id = _default_voice_session_id()
            session_title = _default_voice_session_title()
        out: dict[str, str] = {}
        for name, value in (
            ("X-Hermes-Session-Id", session_id),
            ("X-Hermes-Session-Key", self.config.session_key),
            ("X-Hermes-Session-Title", session_title),
        ):
            cleaned = (value or "").strip().replace("\r", " ").replace("\n", " ")
            if cleaned:
                out[name] = cleaned.encode("ascii", "replace").decode("ascii")
        return out

    async def ask(self, transcript: str, context: str | None = None, image_url: str | None = None) -> str:
        """Ask AGENT for a short spoken answer. ``context`` (e.g. gemma vision) is folded into the user
        turn as labeled input; ``image_url`` attaches a native image (premium native-vision path).
        """
        cleaned = transcript.strip()
        if not cleaned:
            return "I didn't quite catch that."
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": VOICE_FAST_SYSTEM_MESSAGE},
                _user_message(cleaned, context, image_url),
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": 0.6,
            "reasoning_effort": os.getenv("AGENT_VOICE_REASONING_EFFORT", "low"),
        }
        _apply_tool_policy(payload)
        response = await self._post(f"{self.config.base_url.rstrip('/')}/chat/completions", payload)
        response.raise_for_status()
        text = _extract_chat_text(response.json())
        return _trim_for_voice(
            text or "I'm having trouble connecting to the agent right now.", self.config.max_response_chars
        )

    async def ask_stream(self, transcript: str, context: str | None = None, image_url: str | None = None):
        """Stream AGENT's reply, yielding complete sentences as they arrive. ``context`` (e.g. gemma
        vision) is folded into the user turn as labeled input; ``image_url`` attaches a native image
        (premium native-vision path).

        Pipelining sentences into per-sentence TTS cuts time-to-first-audio dramatically vs awaiting
        the whole reply then synthesizing it in one block. Falls back to a single yield if the server
        doesn't stream. Yields trimmed-for-voice sentence strings.
        """
        cleaned = transcript.strip()
        if not cleaned:
            yield "I didn't quite catch that."
            return
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": VOICE_FAST_SYSTEM_MESSAGE},
                _user_message(cleaned, context, image_url),
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": 0.6,
            "reasoning_effort": os.getenv("AGENT_VOICE_REASONING_EFFORT", "low"),
            "stream": True,
        }
        _apply_tool_policy(payload)
        headers = {"Content-Type": "application/json"}
        api_key = os.getenv(self.config.api_key_env, "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        headers.update(self._session_headers())
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        import httpx

        buf = ""
        pending_data = ""  # SSE: accumulates a JSON event split over several data: lines
        produced = False
        announced = False  # Phase A: spoke a tool-status line this turn (at most once)
        status_on = _tool_status_enabled()
        # Phase B: watchdog on time-to-first-spoken-content. While nothing real has been spoken yet,
        # bound each read by the remaining budget; on timeout speak a defer line and end the turn.
        budget = _env_float("AGENT_VOICE_FIRST_AUDIO_BUDGET_S", _FIRST_AUDIO_BUDGET_S)
        start = time.monotonic()
        # read=None: the 30s client timeout doubled as the READ timeout between SSE chunks —
        # an agentic tool call >30s after the first spoken sentence killed the stream mid-answer
        # with no fallback line (audit 2026-07-02). Liveness is now owned by the explicit idle
        # watchdog below (AGENT_STREAM_IDLE_S), which at least says so out loud.
        stream_timeout = httpx.Timeout(self.config.timeout_seconds, read=None)
        async with httpx.AsyncClient(timeout=stream_timeout) as client:
            async with client.stream("POST", url, json=payload, headers=headers) as resp:
                resp.raise_for_status()
                line_iter = resp.aiter_lines().__aiter__()
                while True:
                    if budget > 0 and not produced:
                        remaining = budget - (time.monotonic() - start)
                        if remaining <= 0:
                            yield _defer_line()
                            return
                        try:
                            line = await asyncio.wait_for(line_iter.__anext__(), timeout=remaining)
                        except asyncio.TimeoutError:
                            yield _defer_line()
                            return
                        except StopAsyncIteration:
                            break
                    else:
                        # Post-first-content idle watchdog: generous (tools may run minutes via the
                        # gateway), but bounded — and it ABORTS AUDIBLY instead of cutting off silently.
                        idle_s = _env_float("AGENT_STREAM_IDLE_S", 180.0)
                        try:
                            if idle_s > 0:
                                line = await asyncio.wait_for(line_iter.__anext__(), timeout=idle_s)
                            else:
                                line = await line_iter.__anext__()
                        except asyncio.TimeoutError:
                            logger.warning("AGENT stream idle >%.0fs mid-answer -> aborting turn", idle_s)
                            yield "I lost the thread there. Please ask me again in a moment."
                            return
                        except StopAsyncIteration:
                            break
                    if not line:
                        pending_data = ""  # SSE event boundary: drop an incomplete fragment
                        continue
                    if not line.startswith("data:"):
                        continue
                    piece = line[len("data:"):].strip()
                    if piece == "[DONE]":
                        break
                    # Spec-conform emitters may split one JSON event over several data: lines;
                    # parsing line-by-line silently dropped those (review 2026-07-02 round 2, P3).
                    # Accumulate until the JSON parses; an event boundary (blank line) resets.
                    pending_data = (pending_data + "\n" + piece) if pending_data else piece
                    try:
                        parsed = json.loads(pending_data)
                    except json.JSONDecodeError:
                        continue  # incomplete multi-line event — keep accumulating
                    pending_data = ""
                    # Surface streamed error frames instead of silently skipping them (they have
                    # neither "choices" nor "status", so they fell through as noise).
                    if isinstance(parsed, dict) and "error" in parsed and "choices" not in parsed:
                        logger.warning("AGENT stream error frame: %s", str(parsed.get("error"))[:200])
                        break
                    # Phase A: inline tool-progress event ({"tool","status",...}, no "choices").
                    # Speak a status when a slow tool starts and nothing has been said yet, so the
                    # tool's runtime isn't a silent gap. Don't set `produced` — the real answer still
                    # gets the clean first-chunk treatment.
                    if isinstance(parsed, dict) and "choices" not in parsed and parsed.get("status"):
                        # Tool activity IS liveness: restart the first-audio budget clock. Without
                        # this, AGENT announced "Moment, ich schaue nach" and the budget then
                        # defer-aborted the turn mid-tool ~18s later, discarding the answer
                        # (review 2026-07-02 round 2, P3).
                        start = time.monotonic()
                        if (status_on and not produced and not announced
                                and parsed.get("status") == "running"):
                            note = _tool_status_line(parsed.get("tool"), parsed.get("label"))
                            if note:
                                announced = True
                                yield note
                        continue
                    try:
                        delta = parsed["choices"][0]["delta"].get("content") or ""
                    except (KeyError, IndexError, TypeError):
                        continue
                    buf += delta
                    chunks, buf, produced = _drain_voice_chunks(
                        buf, produced, _env_int("AGENT_FIRST_CHUNK_MIN_CHARS", _FIRST_CHUNK_MIN_CHARS)
                    )
                    for chunk in chunks:
                        yield chunk
        tail = buf.strip()
        if tail:
            yield tail
        elif not produced:
            yield "I'm having trouble connecting to the agent right now."

    async def _post(self, url: str, payload: dict[str, Any]) -> Any:
        headers = {"Content-Type": "application/json"}
        api_key = os.getenv(self.config.api_key_env, "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        headers.update(self._session_headers())
        client = self._http_client
        if client is not None:
            return await client.post(url, json=payload, headers=headers, timeout=self.config.timeout_seconds)
        import httpx

        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as live_client:
            return await live_client.post(url, json=payload, headers=headers)


_LEAD_IN_SYSTEM = (
    "You are AGENT's voice in a live conversation. Give ONE very short, dry bridge of 1 to 3 English "
    "words that ONLY fills time and does NOT answer the question. Vary it and sound natural. Examples: "
    "'Well.' 'Let's see.' 'Good question.' 'Hmm.' Do NOT use the phrase 'one moment'. Reply only with "
    "the bridge, without quotation marks or explanation."
)

# Instant static fallback bridges for the quick-take, when the 9B hasn't returned by the delay
# threshold (we never wait on it -> zero added latency). Deliberately avoid "Moment" so a quick-take
# bridge never collides with a Phase-A tool-status line ("Moment, ich schaue im Web nach.").
_STATIC_QUICKTAKES = ("Well.", "Let's see.", "Good question.", "Hmm.", "Thinking.", "Let's check.")

# Tail gap-fillers for the full-window mask: spoken only if the brain is STILL silent after the opener
# (the cloud TTFT hole can be 4-10 s; the opener covers ~1 s). Dry/AGENT-flavored and deliberately
# CONTENT-FREE — like the opener, they must never pre-empt or contradict the brain (ordered
# composition). A bit longer than an opener so each fills ~1.5-2.5 s of speech; varied per turn so it
# doesn't sound like a stuck loop.
_GAP_FILLERS = (
    "I'm sorting that out.",
    "I've nearly got it.",
    "Give me a second to think.",
    "This needs a little thought.",
    "I'm almost there.",
    "One moment, nearly done.",
)


def _static_quicktake() -> str:
    import random
    return random.choice(_STATIC_QUICKTAKES)


def _gap_filler(exclude: set[str] | None = None) -> str:
    """A dry, content-free tail filler for the full-window latency mask, avoiding any already used
    this turn so repeated holes don't repeat the same line. Empty string if all are used.
    """
    import random
    pool = [f for f in _GAP_FILLERS if not exclude or f not in exclude]
    return random.choice(pool) if pool else ""


def _sanitize_lead_in(text: str) -> str:
    """Force the lead-in to stay a tiny content-free bridge so a disobedient 9B can never speak a
    real-answer fragment that pre-empts or contradicts the brain. Keep only the first clause and cap
    hard to a few words.
    """
    s = (text or "").strip().strip('"“”„‘’').replace("\n", " ").strip()
    # Keep only up to the first sentence/clause terminator (a bridge is one short beat).
    m = re.search(r"[.!?…,;:—–]", s)
    if m:
        s = s[: m.start() + 1].strip()
    words = s.split()
    if len(words) > 5:  # a bridge is 1-4 words; more than that is the 9B answering
        s = " ".join(words[:5]).rstrip(",;:") + " —"
    if len(s) > 28:
        s = s[:28].rstrip(",;:") + " —"
    return s


class FastLeadInClient:
    """Fast local 9B lead-in: a short spoken bridge that masks the full brain's think time.

    Best-effort by design — any failure returns "" so a turn is never broken, only unmasked.
    """

    def __init__(self, config: FastLeadInConfig | None = None, http_client: AsyncPostClient | None = None) -> None:
        """Initialize with optional injectable config and HTTP transport."""
        self.config = config or FastLeadInConfig.from_env()
        self._http_client = http_client

    async def lead_in(self, transcript: str) -> str:
        """Return a tiny dry opener for the user's turn, or "" if disabled/unavailable."""
        if not self.config.enabled:
            return ""
        cleaned = (transcript or "").strip()
        if not cleaned:
            return ""
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": _LEAD_IN_SYSTEM},
                {"role": "user", "content": cleaned[:400]},
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": self.config.enable_thinking},
        }
        try:
            headers = {"Content-Type": "application/json"}
            key = os.getenv(self.config.api_key_env, "").strip() if self.config.api_key_env else ""
            if key:
                headers["Authorization"] = f"Bearer {key}"
            url = f"{self.config.base_url.rstrip('/')}/chat/completions"
            client = self._http_client
            if client is not None:
                resp = await client.post(url, json=payload, headers=headers, timeout=self.config.timeout_seconds)
            else:
                import httpx

                async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as live:
                    resp = await live.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            return _sanitize_lead_in(_extract_chat_text(resp.json()))
        except Exception:
            return ""


_REFLEX_SYSTEM = (
    "You are AGENT's fast reflex for TRIVIAL conversational turns. Answer ONLY when the utterance is "
    "purely social (greeting, goodbye, thanks, inconsequential small talk) and needs NO tool, memory, "
    "current facts, or real thought; then answer in ONE short, dry English sentence. NEVER answer "
    "yes/no, agreement, refusal, or any possible response to a follow-up question locally, because it "
    "belongs to conversation context you do not have. For everything else—facts, time/date, memory, "
    "tasks, opinions, or anything concrete—output exactly ESCALATE and nothing else. When unsure, ESCALATE."
)

# Confirmation/decision words that must never be answered by the context-free reflex — they are
# almost always a reply to something AGENT just asked (the session is waiting for them). Hard
# client-side filter in addition to the prompt rule: a locally answered "Ja, bitte." swallowed a
# confirmation the gateway session was waiting for (review 2026-07-02 round 2, P1-6).
_REFLEX_CONFIRMATION_WORDS = frozenset(
    "ja nein jo jep jup yes no ok okay gut genau richtig stimmt passt bitte gerne mach machs "
    "los stopp stop weiter abbrechen".split()
)


def _reflex_looks_like_confirmation(cleaned: str) -> bool:
    words = [w.strip(".,!?;:") for w in cleaned.lower().split()]
    return bool(words) and any(w in _REFLEX_CONFIRMATION_WORDS for w in words)


@dataclass
class LocalReflexConfig:
    """Config for the local 9B 'reflex' that fully answers clearly-trivial turns, removing the 3-6s
    cloud brain from the loop on that slice. Default DISABLED — 'voller AGENT' is the default; opt in
    via AGENT_LOCAL_TRIVIAL=1. Conservative: only short turns are candidates and the 9B is prompted to
    ESCALATE on any doubt.
    """

    base_url: str = "http://127.0.0.1:3447/v1"
    model: str = "qwopus-9b"
    api_key_env: str = ""
    timeout_seconds: float = 0.9  # tight: bounds the added latency on escalated short turns
    max_tokens: int = 64
    temperature: float = 0.4
    enabled: bool = False
    max_words: int = 6
    enable_thinking: bool = False

    @classmethod
    def from_env(cls) -> LocalReflexConfig:
        defaults = cls()
        return cls(
            base_url=os.getenv("AGENT_LOCAL_BASE_URL", os.getenv("AGENT_GATE_BASE_URL", defaults.base_url)),
            model=os.getenv("AGENT_LOCAL_MODEL", os.getenv("AGENT_GATE_MODEL", defaults.model)),
            timeout_seconds=_env_float("AGENT_LOCAL_TIMEOUT_SECONDS", defaults.timeout_seconds),
            max_tokens=_env_int("AGENT_LOCAL_MAX_TOKENS", defaults.max_tokens),
            enabled=os.getenv("AGENT_LOCAL_TRIVIAL", "0").strip().lower() not in ("0", "false", "no", "off", ""),
            max_words=_env_int("AGENT_LOCAL_MAX_WORDS", defaults.max_words),
        )


class LocalReflexClient:
    """Fully answers a clearly-trivial turn with the local 9B, or returns None to fall through to the
    full brain. Best-effort: any failure / ESCALATE / long turn returns None (never a wrong answer).
    """

    def __init__(self, config: LocalReflexConfig | None = None, http_client: AsyncPostClient | None = None) -> None:
        self.config = config or LocalReflexConfig.from_env()
        self._http_client = http_client

    async def answer(self, transcript: str) -> str | None:
        """Return a short local answer for a trivial turn, or None to escalate to the full brain."""
        if not self.config.enabled:
            return None
        cleaned = (transcript or "").strip()
        # conservative pre-filter: only short turns are candidates for a local reflex answer
        if not cleaned or len(cleaned.split()) > self.config.max_words:
            return None
        # confirmations/decisions belong to the ongoing (stateful) conversation — never local
        if _reflex_looks_like_confirmation(cleaned):
            return None
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": _REFLEX_SYSTEM},
                {"role": "user", "content": cleaned[:200]},
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": self.config.enable_thinking},
        }
        try:
            headers = {"Content-Type": "application/json"}
            key = os.getenv(self.config.api_key_env, "").strip() if self.config.api_key_env else ""
            if key:
                headers["Authorization"] = f"Bearer {key}"
            url = f"{self.config.base_url.rstrip('/')}/chat/completions"
            client = self._http_client
            if client is not None:
                resp = await client.post(url, json=payload, headers=headers, timeout=self.config.timeout_seconds)
            else:
                import httpx

                async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as live:
                    resp = await live.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            text = (_extract_chat_text(resp.json()) or "").strip().strip('"“”„‘’')
        except Exception:
            return None
        if not text or "escalate" in text.lower():
            return None
        # bound a runaway; a reflex answer is one short sentence
        return _trim_for_voice(text, 200)


class QwenVoiceTtsClient:
    """Small async client for Qwen3-TTS WAV output."""

    def __init__(self, config: QwenVoiceTtsConfig | None = None, http_client: AsyncPostClient | None = None) -> None:
        """Initialize with optional injectable config and HTTP transport."""
        self.config = config or QwenVoiceTtsConfig.from_env()
        self._http_client = http_client

    def set_voice(self, voice: str) -> None:
        """Update the voice used for subsequent speech requests."""
        normalized = voice.strip()
        if normalized:
            self.config = replace(self.config, voice=normalized)

    async def synthesize(self, text: str) -> AudioFrame:
        """Synthesize text and decode WAV bytes into an AudioFrame."""
        cleaned = text.strip()
        if not cleaned:
            return self._empty_audio()
        if self.config.response_format != "wav":
            raise ValueError("QwenVoiceTtsClient requires wav response_format for AudioFrame output")
        payload = {
            "model": self.config.model,
            "voice": self.config.voice,
            "input": cleaned,
            "response_format": self.config.response_format,
            "speed": self.config.speed,
        }
        response = await self._post(f"{self.config.base_url.rstrip('/')}/audio/speech", payload)
        response.raise_for_status()
        content = getattr(response, "content", b"")
        return _decode_wav_to_audio_frame(content if isinstance(content, bytes) else bytes(content))

    async def _post(self, url: str, payload: dict[str, Any]) -> Any:
        headers = {"Content-Type": "application/json"}
        api_key = os.getenv(self.config.api_key_env, "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        client = self._http_client
        if client is not None:
            return await client.post(url, json=payload, headers=headers, timeout=self.config.timeout_seconds)
        import httpx

        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as live_client:
            return await live_client.post(url, json=payload, headers=headers)

    async def stream_pcm(self, text: str, *, sample_rate: int = 24000):
        """Stream raw PCM for low time-to-first-audio, yielding (sample_rate, int16) chunks as bytes
        arrive (~176ms first-byte vs ~1.3s for the full-WAV path). Odd trailing bytes carry to the
        next chunk so every yield is on a whole-sample boundary. Ported from the MVP's QwenTtsClient.
        """
        cleaned = text.strip()
        if not cleaned:
            return
        payload = {
            "model": self.config.model,
            "voice": self.config.voice,
            "input": cleaned,
            "response_format": "pcm",
            "stream": True,
            "speed": self.config.speed,
        }
        headers = {"Content-Type": "application/json"}
        api_key = os.getenv(self.config.api_key_env, "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        url = f"{self.config.base_url.rstrip('/')}/audio/speech"
        import httpx

        leftover = b""
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            async with client.stream("POST", url, json=payload, headers=headers) as response:
                response.raise_for_status()
                async for raw in response.aiter_bytes():
                    if not raw:
                        continue
                    buf = leftover + raw
                    usable = len(buf) - (len(buf) % 2)
                    if usable:
                        yield sample_rate, np.frombuffer(buf[:usable], dtype="<i2").copy()
                    leftover = buf[usable:]

    def _empty_audio(self) -> AudioFrame:
        return 24000, np.zeros(0, dtype=np.int16)


def _decode_wav_to_audio_frame(content: bytes) -> AudioFrame:
    if not content:
        raise ValueError("Expected non-empty WAV content")
    with wave.open(io.BytesIO(content), "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())
    if channels < 1:
        raise ValueError("Expected WAV with at least one channel")
    if sample_width != 2:
        raise ValueError(f"Expected 16-bit PCM WAV, got sample width {sample_width}")
    audio: NDArray[np.int16] = np.frombuffer(frames, dtype=np.int16).copy()
    if audio.size % channels != 0:
        raise ValueError("WAV frame data is not divisible by channel count")
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return sample_rate, audio


def _extract_chat_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content.strip() if isinstance(content, str) else ""


def _trim_for_voice(text: str, limit: int) -> str:
    cleaned = text.strip()
    if limit <= 0 or len(cleaned) <= limit:
        return cleaned
    # Never cut mid-word/sentence (the TTS would speak the fragment -> sounds like AGENT "broke off").
    # Back off from the limit to the last sentence boundary; fall back to the last word boundary.
    window = cleaned[:limit]
    cut = max(window.rfind(". "), window.rfind("! "), window.rfind("? "),
              window.rfind("."), window.rfind("!"), window.rfind("?"), window.rfind("…"))
    if cut >= limit * 0.5:  # a sentence boundary reasonably far in -> end cleanly there
        return window[: cut + 1].rstrip()
    space = window.rfind(" ")
    return (window[:space] if space > 0 else window).rstrip() + "…"


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default
