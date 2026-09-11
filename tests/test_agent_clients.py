# ruff: noqa: D103
from __future__ import annotations
import io
import wave
from typing import Any

import numpy as np
import pytest

from reachy_mini_conversation_app.agent_clients import (
    HermesVoiceClient,
    HermesVoiceConfig,
    QwenVoiceTtsClient,
    QwenVoiceTtsConfig,
)


class _FakeJsonResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeBytesResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None


class _FakeHttpClient:
    def __init__(self, response: object) -> None:
        self.response = response
        self.requests: list[dict[str, Any]] = []

    async def post(self, url: str, **kwargs: Any) -> object:
        self.requests.append({"url": url, **kwargs})
        return self.response


@pytest.mark.asyncio
async def test_hermes_voice_client_sends_no_tools_chat_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_AGENT_KEY", "test-secret")
    monkeypatch.setenv("AGENT_VOICE_TOOLS", "0")  # legacy neutered mode: tools explicitly disabled
    http = _FakeHttpClient(_FakeJsonResponse({"choices": [{"message": {"content": "Kurz und agentig."}}]}))
    client = HermesVoiceClient(
        HermesVoiceConfig(
            base_url="http://agent.local:8642/v1",
            model="AGENT",
            api_key_env="TEST_AGENT_KEY",
            max_response_chars=80,
        ),
        http_client=http,
    )

    response = await client.ask("Was bist du?")

    assert response == "Kurz und agentig."
    assert len(http.requests) == 1
    request = http.requests[0]
    assert request["url"] == "http://agent.local:8642/v1/chat/completions"
    assert request["headers"]["Authorization"] == "Bearer test-secret"
    assert request["json"]["model"] == "AGENT"
    assert request["json"]["tools"] == []
    assert request["json"]["tool_choice"] == "none"
    assert request["json"]["enabled_toolsets"] == []
    assert request["json"]["messages"][-1] == {"role": "user", "content": "Was bist du?"}


@pytest.mark.asyncio
async def test_hermes_voice_client_enables_tools_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """By default (P1b) the voice client does NOT disable tools -> the gateway runs the full agent  with its api_server toolsets."""
    monkeypatch.setenv("TEST_AGENT_KEY", "test-secret")
    monkeypatch.delenv("AGENT_VOICE_TOOLS", raising=False)
    http = _FakeHttpClient(_FakeJsonResponse({"choices": [{"message": {"content": "ok"}}]}))
    client = HermesVoiceClient(HermesVoiceConfig(api_key_env="TEST_AGENT_KEY"), http_client=http)
    await client.ask("Was bist du?")
    body = http.requests[0]["json"]
    assert "tools" not in body and "tool_choice" not in body and "enabled_toolsets" not in body


@pytest.mark.asyncio
async def test_hermes_voice_client_sends_session_headers_and_only_new_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    """Session continuity + long-term memory are opt-in via headers; the gateway  threads the conversation so the client posts only the new user turn."""
    monkeypatch.setenv("TEST_AGENT_KEY", "test-secret")
    http = _FakeHttpClient(_FakeJsonResponse({"choices": [{"message": {"content": "Lampenschirm."}}]}))
    client = HermesVoiceClient(
        HermesVoiceConfig(
            base_url="http://agent.local:8642/v1",
            api_key_env="TEST_AGENT_KEY",
            session_id="reachy-voice-20260626",
            session_key="reachy-voice",
            session_title="Reachy Voice 26.06.2026",
        ),
        http_client=http,
    )

    await client.ask("Wie lautet mein Codewort?")

    headers = http.requests[0]["headers"]
    assert headers["X-Hermes-Session-Id"] == "reachy-voice-20260626"
    assert headers["X-Hermes-Session-Key"] == "reachy-voice"
    assert headers["X-Hermes-Session-Title"] == "Reachy Voice 26.06.2026"
    # Client carries no local history — the gateway remembers prior turns.
    messages = http.requests[0]["json"]["messages"]
    assert [m["role"] for m in messages] == ["system", "user"]


@pytest.mark.asyncio
async def test_hermes_voice_client_omits_blank_session_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty session values produce no header (stateless fallback)."""
    monkeypatch.setenv("TEST_AGENT_KEY", "test-secret")
    http = _FakeHttpClient(_FakeJsonResponse({"choices": [{"message": {"content": "ok"}}]}))
    client = HermesVoiceClient(
        HermesVoiceConfig(api_key_env="TEST_AGENT_KEY", session_id="", session_key="", session_title=""),
        http_client=http,
    )

    await client.ask("hallo")

    headers = http.requests[0]["headers"]
    assert "X-Hermes-Session-Id" not in headers
    assert "X-Hermes-Session-Key" not in headers


@pytest.mark.asyncio
async def test_qwen_voice_tts_client_posts_speech_and_decodes_wav() -> None:
    wav_bytes = _wav_bytes(sample_rate=16000, samples=np.array([0, 100, -100, 200], dtype=np.int16))
    http = _FakeHttpClient(_FakeBytesResponse(wav_bytes))
    client = QwenVoiceTtsClient(
        QwenVoiceTtsConfig(
            base_url="http://qwen.local:7034/v1",
            model="qwen3-tts",
            voice="default",
            response_format="wav",
        ),
        http_client=http,
    )

    sample_rate, audio = await client.synthesize("Hallo Reachy.")

    assert sample_rate == 16000
    np.testing.assert_array_equal(audio, np.array([0, 100, -100, 200], dtype=np.int16))
    assert len(http.requests) == 1
    request = http.requests[0]
    assert request["url"] == "http://qwen.local:7034/v1/audio/speech"
    assert request["json"] == {
        "model": "qwen3-tts",
        "voice": "default",
        "input": "Hallo Reachy.",
        "response_format": "wav",
        "speed": 1.0,
    }


@pytest.mark.asyncio
async def test_qwen_voice_tts_client_set_voice_affects_next_payload() -> None:
    wav_bytes = _wav_bytes(sample_rate=16000, samples=np.array([0], dtype=np.int16))
    http = _FakeHttpClient(_FakeBytesResponse(wav_bytes))
    client = QwenVoiceTtsClient(QwenVoiceTtsConfig(voice="default"), http_client=http)

    client.set_voice("nova_de")
    await client.synthesize("Hallo.")

    assert http.requests[0]["json"]["voice"] == "nova_de"


def test_qwen_voice_tts_config_reads_bounded_speed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_TTS_SPEED", "0.95")
    assert QwenVoiceTtsConfig.from_env().speed == 0.95

    monkeypatch.setenv("AGENT_TTS_SPEED", "9")
    assert QwenVoiceTtsConfig.from_env().speed == 2.0


@pytest.mark.asyncio
async def test_qwen_voice_tts_client_requires_wav_response_format() -> None:
    client = QwenVoiceTtsClient(
        QwenVoiceTtsConfig(response_format="mp3"), http_client=_FakeHttpClient(_FakeBytesResponse(b""))
    )

    with pytest.raises(ValueError, match="requires wav response_format"):
        await client.synthesize("Hallo.")


@pytest.mark.asyncio
async def test_qwen_voice_tts_client_rejects_empty_audio_body() -> None:
    client = QwenVoiceTtsClient(QwenVoiceTtsConfig(), http_client=_FakeHttpClient(_FakeBytesResponse(b"")))

    with pytest.raises(ValueError, match="non-empty WAV"):
        await client.synthesize("Hallo.")


def _wav_bytes(*, sample_rate: int, samples: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples.astype(np.int16).tobytes())
    return buf.getvalue()


def test_drain_voice_chunks_first_chunk_breaks_at_clause() -> None:
    """First chunk ends at a clause boundary once long enough (faster first audio);  later chunks only at sentence end."""
    from reachy_mini_conversation_app.agent_clients import _drain_voice_chunks

    # First clause is long enough -> emit at the comma.
    chunks, buf, produced = _drain_voice_chunks("Schwarze Löcher sind Regionen, in denen", False, 15)
    assert chunks == ["Schwarze Löcher sind Regionen,"]
    assert produced is True
    # Already produced: a later comma does NOT split; waits for sentence end. A sentence end is
    # only a boundary when trailing whitespace confirms it (a raw buffer-end "." is held back).
    chunks2, buf2, produced2 = _drain_voice_chunks(buf + " nichts entkommt, nicht mal Licht. ", produced, 15)
    assert chunks2 == ["in denen nichts entkommt, nicht mal Licht."]


def test_drain_voice_chunks_short_leading_clause_not_split() -> None:
    """A short leading clause below the min stays buffered (no tiny first fragment)."""
    from reachy_mini_conversation_app.agent_clients import _drain_voice_chunks

    chunks, buf, produced = _drain_voice_chunks("Ja, aber warte. ", False, 15)
    # "Ja," is below 15 chars -> not split there; the sentence end (confirmed by trailing space) wins.
    assert chunks == ["Ja, aber warte."]
    assert produced is True


def test_drain_voice_chunks_keeps_decimals_together() -> None:
    """A period/comma between digits (42.5 / 1,5) must NOT split the number across chunks;  the ambiguous buffer-end punctuation is held until the next delta disambiguates."""
    from reachy_mini_conversation_app.agent_clients import _drain_voice_chunks

    # Buffer ends right after "42." — could be a decimal in progress -> hold, don't emit.
    chunks, buf, produced = _drain_voice_chunks("Der Preis liegt bei 42.", False, 15)
    assert chunks == [] and buf == "Der Preis liegt bei 42." and produced is False
    # Next delta reveals the decimal; still no false split, whole number stays intact.
    chunks2, buf2, produced2 = _drain_voice_chunks("Der Preis liegt bei 42.5 Prozent. ", produced, 15)
    assert chunks2 == ["Der Preis liegt bei 42.5 Prozent."]
    # German decimal comma likewise not treated as a clause boundary.
    chunks3, buf3, produced3 = _drain_voice_chunks("Etwa 1,5 Meter groß. ", False, 15)
    assert chunks3 == ["Etwa 1,5 Meter groß."]


def test_tool_status_line_announces_slow_tools_only() -> None:
    """Phase A: slow/heavy tools get a spoken status; fast/trivial ones (memory, todo) stay silent."""
    from reachy_mini_conversation_app.agent_clients import _tool_status_line

    # Slow families announce (match by tool name or label).
    assert _tool_status_line("web", '+web: "bitcoin price"') == "One moment, I'll check the web."
    assert _tool_status_line("web_extract", None) == "One moment, I'll check the web."
    assert _tool_status_line("terminal", None) == "One moment, I'll check the system."
    assert _tool_status_line("delegation", None) == "I'll handle the larger part in the background."
    # Fast / trivial tools: no announcement (would be noise, not a freeze).
    assert _tool_status_line("memory", '+memory: "note"') is None
    assert _tool_status_line("todo", None) is None
    assert _tool_status_line(None, None) is None


class _FakeStreamResp:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def raise_for_status(self) -> None:
        return None

    async def aiter_lines(self):  # type: ignore[no-untyped-def]
        for ln in self._lines:
            yield ln

    async def __aenter__(self) -> "_FakeStreamResp":
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False


class _FakeStreamClient:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def stream(self, method: str, url: str, **kw: Any) -> _FakeStreamResp:
        return _FakeStreamResp(self._lines)

    async def __aenter__(self) -> "_FakeStreamClient":
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False


@pytest.mark.asyncio
async def test_ask_stream_announces_slow_tool_then_streams_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Phase A end-to-end: an inline `hermes.tool.progress` running event for a slow tool yields one  spoken status, the completed event is silent, and the real answer streams after."""
    import httpx

    lines = [
        'data: {"choices":[{"delta":{}}]}',  # initial empty chunk
        "event: hermes.tool.progress",  # skipped (not a data: line)
        'data: {"tool":"web","status":"running","label":"+web: q"}',
        'data: {"tool":"web","status":"completed"}',  # completed -> no status spoken
        'data: {"choices":[{"delta":{"content":"Das Ergebnis ist da."}}]}',
        "data: [DONE]",
    ]
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _FakeStreamClient(lines))
    monkeypatch.setenv("TEST_AGENT_KEY", "test-secret")
    client = HermesVoiceClient(HermesVoiceConfig(api_key_env="TEST_AGENT_KEY"))

    out = [chunk async for chunk in client.ask_stream("Was kostet Bitcoin?")]

    assert out == ["One moment, I'll check the web.", "Das Ergebnis ist da."]


@pytest.mark.asyncio
async def test_ask_stream_tool_status_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """`AGENT_TOOL_STATUS_ENABLED=0` suppresses the spoken tool status (only the answer streams)."""
    import httpx

    lines = [
        'data: {"tool":"web","status":"running","label":"+web: q"}',
        'data: {"choices":[{"delta":{"content":"Fertig."}}]}',
        "data: [DONE]",
    ]
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _FakeStreamClient(lines))
    monkeypatch.setenv("TEST_AGENT_KEY", "test-secret")
    monkeypatch.setenv("AGENT_TOOL_STATUS_ENABLED", "0")
    client = HermesVoiceClient(HermesVoiceConfig(api_key_env="TEST_AGENT_KEY"))

    out = [chunk async for chunk in client.ask_stream("Was kostet Bitcoin?")]

    assert out == ["Fertig."]


class _SlowStreamResp(_FakeStreamResp):
    """Stream whose first line is delayed past the budget (simulates a runaway tool/reasoning)."""

    def __init__(self, lines: list[str], delay_s: float) -> None:
        super().__init__(lines)
        self._delay_s = delay_s

    async def aiter_lines(self):  # type: ignore[no-untyped-def]
        import asyncio as _a

        await _a.sleep(self._delay_s)
        for ln in self._lines:
            yield ln


@pytest.mark.asyncio
async def test_ask_stream_defers_when_first_audio_budget_exceeded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Phase B: if no spoken content arrives within the budget, AGENT speaks the defer line and ends."""
    import httpx

    lines = ['data: {"choices":[{"delta":{"content":"zu spät"}}]}', "data: [DONE]"]

    class _Client(_FakeStreamClient):
        def stream(self, method: str, url: str, **kw: Any) -> _FakeStreamResp:
            return _SlowStreamResp(self._lines, delay_s=0.30)

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _Client(lines))
    monkeypatch.setenv("TEST_AGENT_KEY", "test-secret")
    monkeypatch.setenv("AGENT_VOICE_FIRST_AUDIO_BUDGET_S", "0.05")  # tiny budget -> trips before content
    client = HermesVoiceClient(HermesVoiceConfig(api_key_env="TEST_AGENT_KEY"))

    out = [chunk async for chunk in client.ask_stream("Mach was Langsames.")]

    assert len(out) == 1 and "ask me again in a moment" in out[0]


@pytest.mark.asyncio
async def test_ask_stream_no_defer_when_budget_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Budget <= 0 disables the watchdog: even a slow stream streams its real content."""
    import httpx

    lines = ['data: {"choices":[{"delta":{"content":"Fertig."}}]}', "data: [DONE]"]

    class _Client(_FakeStreamClient):
        def stream(self, method: str, url: str, **kw: Any) -> _FakeStreamResp:
            return _SlowStreamResp(self._lines, delay_s=0.10)

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _Client(lines))
    monkeypatch.setenv("TEST_AGENT_KEY", "test-secret")
    monkeypatch.setenv("AGENT_VOICE_FIRST_AUDIO_BUDGET_S", "0")  # disabled
    client = HermesVoiceClient(HermesVoiceConfig(api_key_env="TEST_AGENT_KEY"))

    out = [chunk async for chunk in client.ask_stream("Mach was.")]

    assert out == ["Fertig."]


@pytest.mark.asyncio
async def test_fast_lead_in_client_sanitizes_and_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lead-in is reduced to a tiny content-free bridge; never breaks on errors."""
    from reachy_mini_conversation_app.agent_clients import FastLeadInClient, FastLeadInConfig

    http = _FakeHttpClient(_FakeJsonResponse({"choices": [{"message": {"content": '  "Also —"\n'}}]}))
    client = FastLeadInClient(FastLeadInConfig(api_key_env=""), http_client=http)
    assert await client.lead_in("Erzähl mir was über schwarze Löcher") == "Also —"
    # request shape: thinking off, tiny budget
    body = http.requests[0]["json"]
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["max_tokens"] == 16


@pytest.mark.asyncio
async def test_fast_lead_in_client_disabled_returns_empty() -> None:
    from reachy_mini_conversation_app.agent_clients import FastLeadInClient, FastLeadInConfig

    client = FastLeadInClient(FastLeadInConfig(enabled=False), http_client=_FakeHttpClient(_FakeJsonResponse({})))
    assert await client.lead_in("hallo") == ""


def test_sanitize_lead_in_reduces_answer_fragment_to_bridge() -> None:
    """A disobedient 9B that starts answering must be cut to a content-free bridge."""
    from reachy_mini_conversation_app.agent_clients import _sanitize_lead_in

    out = _sanitize_lead_in("Der Himmel ist blau wegen Rayleigh-Streuung des Sonnenlichts")
    assert out == "Der Himmel ist blau wegen —"
    assert "Rayleigh" not in out
    # a proper short bridge is preserved
    assert _sanitize_lead_in("Moment.") == "Moment."


@pytest.mark.asyncio
async def test_ask_attaches_native_image(monkeypatch: pytest.MonkeyPatch) -> None:
    """image_url builds a multimodal user message (native vision path)."""
    monkeypatch.setenv("TEST_AGENT_KEY", "test-secret")
    http = _FakeHttpClient(_FakeJsonResponse({"choices": [{"message": {"content": "ok"}}]}))
    client = HermesVoiceClient(HermesVoiceConfig(api_key_env="TEST_AGENT_KEY"), http_client=http)
    await client.ask("Wie viele?", context=None, image_url="data:image/png;base64,AAAA")
    content = http.requests[0]["json"]["messages"][-1]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[1]["image_url"]["url"].startswith("data:image/png")


@pytest.mark.asyncio
async def test_ask_text_only_when_no_image(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_AGENT_KEY", "test-secret")
    http = _FakeHttpClient(_FakeJsonResponse({"choices": [{"message": {"content": "ok"}}]}))
    client = HermesVoiceClient(HermesVoiceConfig(api_key_env="TEST_AGENT_KEY"), http_client=http)
    await client.ask("Hallo", context="[Szene: ein Tisch]")
    content = http.requests[0]["json"]["messages"][-1]["content"]
    assert isinstance(content, str) and "[Szene: ein Tisch]" in content
