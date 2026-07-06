# ruff: noqa: D103
"""Gap-map Stufe 3: body-tool surface (gateway->app) + companion mode."""
from __future__ import annotations

import asyncio
import json

import pytest

from reachy_mini_conversation_app.body_surface import ALLOWED_ACTIONS, run_body_action
from reachy_mini_conversation_app.companion import CompanionWatcher, event_transcript
from reachy_mini_conversation_app.reachy_platform_client import ReachyPlatformClient


@pytest.mark.asyncio
async def test_body_surface_allowlist_and_mapping(monkeypatch):
    calls = []

    async def fake_dispatch(deps, name, args):
        calls.append((name, args))
        return {"status": "queued"}

    monkeypatch.setattr(
        "reachy_mini_conversation_app.tools.core_tools.dispatch_tool_call", fake_dispatch
    )

    class _MM:
        cleared = False

        def clear_move_queue(self):
            self.cleared = True

    class _Cam:
        tracked = None

        def set_head_tracking_enabled(self, on):
            self.tracked = on

    class _Deps:
        movement_manager = _MM()
        camera_worker = _Cam()

    deps = _Deps()
    chirped = []

    assert (await run_body_action(deps, "emote", {"emotion": "happy"}))["status"] == "queued"
    assert calls[-1] == ("play_emotion", {"emotion": "happy"})
    assert (await run_body_action(deps, "look", {"direction": "left"}))["status"] == "queued"
    assert calls[-1] == ("move_head", {"direction": "left"})
    assert "error" in await run_body_action(deps, "look", {"direction": "backflip"})
    r = await run_body_action(deps, "stop", {})
    assert r["status"] == "stopped" and deps.movement_manager.cleared
    r = await run_body_action(deps, "head_tracking", {"enabled": False})
    assert r["head_tracking"] is False and deps.camera_worker.tracked is False
    r = await run_body_action(deps, "chirp", {"name": "affirm"}, chirp=lambda n: chirped.append(n))
    assert r["status"] == "chirped" and chirped == ["affirm"]
    # outside the allowlist -> rejected, nothing dispatched
    r = await run_body_action(deps, "set_target", {"head": "raw"})
    assert "error" in r and r["allowed"] == list(ALLOWED_ACTIONS)


def test_client_tool_call_roundtrip():
    """A gateway tool_call frame must be executed and answered with tool_result — independent
    of turn routing (the frame carries no turn_id)."""

    class _WS:
        def __init__(self, frames):
            self._out = list(frames)
            self.sent = []

        async def send(self, m):
            self.sent.append(json.loads(m))

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self._out:
                await asyncio.sleep(0.05)  # keep the reader alive briefly
                raise StopAsyncIteration
            await asyncio.sleep(0)
            return json.dumps(self._out.pop(0))

    async def go():
        c = ReachyPlatformClient()
        seen = []

        async def on_tool(action, params):
            seen.append((action, params))
            return {"status": "queued", "emotion": params.get("emotion")}

        c.on_tool_call = on_tool
        ws = _WS([{"type": "tool_call", "tool_call_id": "t1",
                   "action": "emote", "params": {"emotion": "happy"}}])
        c._ws = ws
        await c._ensure_session()
        await asyncio.sleep(0.03)  # let the tool task run + send while the reader is alive
        await c._reader_task
        return seen, ws.sent

    seen, sent = asyncio.run(go())
    assert seen == [("emote", {"emotion": "happy"})]
    results = [m for m in sent if m.get("type") == "tool_result"]
    assert results and results[0]["tool_call_id"] == "t1"
    assert results[0]["result"]["status"] == "queued"


def test_companion_event_transcript_licenses_silence():
    t = event_transcript("Testereignis")
    assert "KEIN Nutzer-Turn" in t and "leerem Text" in t


@pytest.mark.asyncio
async def test_companion_watcher_only_fires_when_enabled(monkeypatch):
    fired = []
    monkeypatch.setattr(
        "reachy_mini_conversation_app.liveliness.read_local_doa",
        lambda: {"angle": 1.0, "speech_detected": True},
    )

    class _Deps:
        movement_manager = None

    w = CompanionWatcher(_Deps(), on_event=fired.append, is_busy=lambda: False, poll_s=0.02)
    monkeypatch.setenv("AGENT_COMPANION", "0")
    w.start()
    await asyncio.sleep(0.12)
    assert fired == []  # mode off -> watcher stays passive

    monkeypatch.setenv("AGENT_COMPANION", "1")
    await asyncio.sleep(0.15)  # two sustained speech polls -> one event (global cooldown after)
    w.stop()
    assert len(fired) == 1 and "gesprochen" in fired[0]
