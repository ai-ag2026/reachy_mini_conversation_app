# ruff: noqa: D103
"""Gap-map Stufe 1 (2026-07-02): idle actions, listening sync, emotion sounds, chirp library."""

from __future__ import annotations
import io
import wave
import asyncio

import numpy as np
import pytest

from reachy_mini_conversation_app.liveliness import (
    IdleActionRunner,
    chirp_wav_bytes,
    wav_file_to_pcm,
)


def test_chirp_wav_bytes_is_valid_wav():
    data = chirp_wav_bytes("acknowledge")
    with wave.open(io.BytesIO(data), "rb") as w:
        assert w.getframerate() == 24000
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getnframes() > 1000


def test_wav_file_to_pcm_roundtrip(tmp_path):
    p = tmp_path / "t.wav"
    pcm_in = (np.sin(np.linspace(0, 100, 4800)) * 20000).astype(np.int16)
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(pcm_in.tobytes())
    sr, pcm = wav_file_to_pcm(str(p))
    assert sr == 24000 and len(pcm) == 4800
    assert wav_file_to_pcm(str(tmp_path / "missing.wav")) is None


@pytest.mark.asyncio
async def test_idle_runner_fires_only_when_idle(monkeypatch):
    """Busy handler / recent activity must suppress actions; a quiet stretch fires exactly one (cooldown suppresses the rest)."""
    import reachy_mini_conversation_app.liveliness as lv

    dispatched = []

    async def fake_dispatch(name, args, deps):
        dispatched.append(name)
        return {"status": "queued"}

    monkeypatch.setattr("reachy_mini_conversation_app.tools.core_tools.dispatch_tool_call_obj", fake_dispatch)
    monkeypatch.setattr(
        "reachy_mini_conversation_app.idle_policy.choose_idle_tool_call",
        lambda names, **k: ("play_emotion", {}),
    )
    monkeypatch.setenv("AGENT_IDLE_ACTIONS", "1")

    class _Deps:
        movement_manager = None

    busy = {"v": True}
    runner = IdleActionRunner(
        _Deps(), is_busy=lambda: busy["v"], idle_after_s=0.05, cooldown_s=10.0, check_interval_s=0.02
    )
    runner.start()
    await asyncio.sleep(0.1)
    assert dispatched == []  # busy the whole time

    busy["v"] = False
    await asyncio.sleep(0.3)  # idle_after (0.05) passes -> one action; cooldown blocks more
    runner.stop()
    assert dispatched == ["play_emotion"]
    assert lv is not None


@pytest.mark.asyncio
async def test_idle_runner_respects_disable_env(monkeypatch):
    monkeypatch.setenv("AGENT_IDLE_ACTIONS", "0")

    class _Deps:
        movement_manager = None

    runner = IdleActionRunner(_Deps(), is_busy=lambda: False, idle_after_s=0.01, check_interval_s=0.01)
    runner.start()
    assert runner._task is None  # disabled -> no task


def test_goto_cartoon_easing_overshoots():
    from reachy_mini_conversation_app.dance_emotion_moves import GotoQueueMove

    start = np.eye(4, dtype=np.float32)
    target = np.eye(4, dtype=np.float32)
    target[0, 3] = 0.02
    goto = GotoQueueMove(
        target_head_pose=target,
        start_head_pose=start,
        target_antennas=(0.0, 0.0),
        start_antennas=(0.0, 0.0),
        target_body_yaw=0.0,
        start_body_yaw=0.0,
        duration=1.0,
        interpolation="cartoon",
    )
    xs = [goto.evaluate(t)[0][0, 3] for t in np.linspace(0.05, 0.98, 30)]
    assert max(xs) > 0.02 + 1e-4  # cartoon overshoots past the target, then settles
    # linear default stays monotonic (behavior-neutral)
    goto_lin = GotoQueueMove(
        target_head_pose=target,
        start_head_pose=start,
        target_antennas=(0.0, 0.0),
        start_antennas=(0.0, 0.0),
        target_body_yaw=0.0,
        start_body_yaw=0.0,
        duration=1.0,
        interpolation="linear",
    )
    xs_lin = [goto_lin.evaluate(t)[0][0, 3] for t in np.linspace(0.05, 0.98, 30)]
    assert max(xs_lin) <= 0.02 + 1e-9


@pytest.mark.asyncio
async def test_play_emotion_routes_bundled_sound(monkeypatch, tmp_path):
    """The tool must hand the move's .wav to the deps seam (AGENT_EMOTION_SOUNDS on)."""
    monkeypatch.setenv("AGENT_EMOTION_SOUNDS", "1")
    import reachy_mini_conversation_app.tools.play_emotion as pe

    wav = tmp_path / "happy1.wav"
    wav.write_bytes(chirp_wav_bytes("affirm"))

    class _Rec:
        description = "happy"
        sound_path = str(wav)

        def evaluate(self, t):
            return np.eye(4), (0.0, 0.0), 0.0

        duration = 1.0

    class _Lib:
        def list_moves(self):
            return ["happy1"]

        def get(self, name):
            return _Rec()

    monkeypatch.setattr(pe, "_get_recorded_moves", lambda: _Lib())

    played = []

    class _MM:
        def queue_move(self, m):
            played.append("move")

    class _Deps:
        movement_manager = _MM()
        play_sound_path = lambda self, p: played.append(("sound", p))  # noqa: E731

    deps = _Deps()
    deps.play_sound_path = lambda p: played.append(("sound", p))
    tool = pe.PlayEmotion()
    res = await tool(deps, emotion="happy")
    assert res.get("status") == "queued"
    assert "move" in played
    assert ("sound", str(wav)) in played


# ── Stufe 2 ────────────────────────────────────────────────────────────────────


def test_doa_mapping():
    import math

    from reachy_mini_conversation_app.liveliness import map_doa_angle_to_direction

    assert map_doa_angle_to_direction(0.1) == "left"
    assert map_doa_angle_to_direction(math.pi - 0.1) == "right"
    assert map_doa_angle_to_direction(math.pi / 2) == "front"
    assert map_doa_angle_to_direction(math.pi / 2 + 0.2) == "front"  # inside deadzone


def test_emotion_cues_sparse_and_matching():
    from reachy_mini_conversation_app.emotion_cues import emotion_for_turn

    assert emotion_for_turn("hallo agent", "Hallo Operator.") == "greeting"
    assert emotion_for_turn("wie lief der test", "Perfekt, alles erledigt.") == "success"
    assert emotion_for_turn("was ist 2+2", "Vier.") is None  # normal turns: NO emote
    assert emotion_for_turn("", "Leider ist der Deploy fehlgeschlagen.") == "downcast"


@pytest.mark.asyncio
async def test_speech_sway_applies_and_clears(monkeypatch):
    monkeypatch.setenv("AGENT_SPEECH_SWAY", "1")
    from reachy_mini_conversation_app.liveliness import SpeechSway

    calls = []

    class _MM:
        def set_external_offsets(self, offsets, antennas=(0.0, 0.0)):
            calls.append(antennas)

    sway = SpeechSway(_MM())
    sway.start()
    try:
        loud = (np.sin(np.linspace(0, 300, 24000)) * 20000).astype(np.int16)  # 1s loud tone
        sway.feed(24000, loud, play_at=asyncio.get_event_loop().time() * 0 + __import__("time").monotonic())
        await asyncio.sleep(0.3)
        moving = [a for a in calls if abs(a[0]) > 0.001]
        assert moving, "sway should drive the antennas for a loud segment"
        assert all(abs(a[0] + a[1]) < 1e-9 for a in moving)  # opposite directions
        calls.clear()
        sway.clear()
        assert calls and calls[-1] == (0.0, 0.0)  # flush releases immediately
    finally:
        sway.stop()


@pytest.mark.asyncio
async def test_thinking_cue_is_small_slow_and_releases(monkeypatch):
    monkeypatch.setenv("AGENT_THINKING_CUE", "1")
    monkeypatch.setenv("AGENT_THINKING_CUE_DELAY_S", "0")
    monkeypatch.setenv("AGENT_THINKING_CUE_MAX_DEG", "2")
    monkeypatch.setenv("AGENT_THINKING_CUE_HZ", "1")
    from reachy_mini_conversation_app.liveliness import ThinkingAntennaCue

    calls = []

    class _MM:
        def set_external_offsets(self, offsets, antennas=(0.0, 0.0)):
            calls.append(antennas)

    cue = ThinkingAntennaCue(_MM())
    cue.start()
    await asyncio.sleep(0.3)
    cue.stop()

    moving = [a for a in calls if abs(a[0]) > 0.0001]
    assert moving
    assert all(abs(a[0]) <= np.deg2rad(2.0) + 1e-9 for a in moving)
    assert all(abs(a[0] + a[1]) < 1e-9 for a in moving)
    assert calls[-1] == (0.0, 0.0)


def test_imu_magnitude_extraction():
    from reachy_mini_conversation_app.liveliness import ImuWatcher

    class _V:
        x, y, z = 0.0, 0.0, 9.81

    class _D:
        accel = _V()

    assert abs(ImuWatcher._accel_magnitude(_D()) - 9.81) < 1e-6
    assert ImuWatcher._accel_magnitude(object()) is None


@pytest.mark.asyncio
async def test_play_wav_path_non_wav_falls_back_to_daemon(monkeypatch):
    """A non-WAV emotion sound (e.g. .ogg) must route through the daemon sound library instead of being dropped when wav_file_to_pcm can't read it."""
    from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
    from reachy_mini_conversation_app.agent_voice_handler import (
        AgentVoiceHandler,
        FakeAudioTtsClient,
        FakeTextAgentClient,
    )

    handler = AgentVoiceHandler(
        ToolDependencies(reachy_mini=object(), movement_manager=None),
        agent_client=FakeTextAgentClient(reply="ok"),
        tts_client=FakeAudioTtsClient(sample_rate=24000, audio=np.zeros(4, dtype=np.int16)),
    )

    calls = []
    monkeypatch.setattr(
        "reachy_mini_conversation_app.liveliness.ensure_daemon_sound",
        lambda p: calls.append(("ensure", p)) or "laughing2.ogg",
    )
    monkeypatch.setattr(
        "reachy_mini_conversation_app.liveliness.play_daemon_sound",
        lambda name: calls.append(("play", name)) or True,
    )

    handler._play_wav_path("/nonexistent/laughing2.ogg")  # wav load fails -> daemon fallback
    for _ in range(20):
        await asyncio.sleep(0.02)
        if len(calls) == 2:
            break

    assert ("ensure", "/nonexistent/laughing2.ogg") in calls
    assert ("play", "laughing2.ogg") in calls


@pytest.mark.asyncio
async def test_idle_runner_reaches_real_dispatcher(monkeypatch):
    """Regression: IdleActionRunner must call the dispatcher with the (name, args, deps) shape that the real core_tools dispatcher expects — a stub with the wrong argument order can stay green while every idle action dies with a TypeError at runtime.

    This one routes a sentinel tool through the REAL core_tools dispatcher.
    """
    import reachy_mini_conversation_app.tools.core_tools as core_tools

    core_tools.initialize_tools()
    seen = []

    async def probe_tool(deps, **kwargs):
        seen.append((deps, kwargs))
        return {"status": "queued"}

    monkeypatch.setitem(core_tools.ALL_TOOLS, "idle_probe", probe_tool)
    monkeypatch.setattr(
        "reachy_mini_conversation_app.idle_policy.choose_idle_tool_call",
        lambda names, **k: ("idle_probe", {"direction": "front"}),
    )
    monkeypatch.setenv("AGENT_IDLE_ACTIONS", "1")

    class _Deps:
        movement_manager = None

    deps = _Deps()
    runner = IdleActionRunner(deps, is_busy=lambda: False, idle_after_s=0.01, cooldown_s=10.0, check_interval_s=0.02)
    runner.start()
    await asyncio.sleep(0.15)
    runner.stop()

    assert seen, "idle action never reached the real dispatcher"
    assert seen[0][0] is deps
    assert seen[0][1] == {"direction": "front"}


@pytest.mark.asyncio
async def test_dispatch_tool_call_obj_unknown_tool():
    """dispatch_tool_call_obj must resolve names against the registry (not explode on deps)."""
    from reachy_mini_conversation_app.tools.core_tools import dispatch_tool_call_obj

    class _Deps:
        movement_manager = None

    result = await dispatch_tool_call_obj("definitely_not_a_tool", {"x": 1}, _Deps())
    assert result == {"error": "unknown tool: definitely_not_a_tool"}


@pytest.mark.asyncio
async def test_thinking_stop_does_not_overwrite_sway(monkeypatch):
    """An idle or delayed cue must never clear another writer's antenna offset."""
    from unittest.mock import MagicMock

    from reachy_mini_conversation_app.liveliness import ThinkingAntennaCue

    monkeypatch.setenv("AGENT_THINKING_CUE_DELAY_S", "0.2")
    monkeypatch.setenv("AGENT_THINKING_CUE", "1")
    mm = MagicMock()
    cue = ThinkingAntennaCue(mm)
    cue.stop()
    mm.set_external_offsets.assert_not_called()
    cue.start()
    await asyncio.sleep(0.02)
    mm.set_external_offsets.assert_not_called()
    cue.stop()
    await asyncio.sleep(0)
    mm.set_external_offsets.assert_not_called()
    cue._apply(0.03)
    cue.stop()
    assert mm.set_external_offsets.call_args.kwargs["antennas"] == (0.0, 0.0)
    assert not cue._applied
    mm.set_external_offsets.reset_mock()
    cue.stop()
    mm.set_external_offsets.assert_not_called()


@pytest.mark.asyncio
async def test_thinking_cancellation_releases_applied_offset(monkeypatch):
    """Cancellation clears an active cue exactly once, before another writer takes over."""
    from unittest.mock import MagicMock

    from reachy_mini_conversation_app.liveliness import ThinkingAntennaCue

    monkeypatch.setenv("AGENT_THINKING_CUE_DELAY_S", "0")
    monkeypatch.setenv("AGENT_THINKING_CUE", "1")
    cue = ThinkingAntennaCue(MagicMock())
    cue.start()
    await asyncio.sleep(0.08)
    assert cue._applied
    task = cue._task
    task.cancel()
    await task
    assert not cue._applied
    assert cue.mm.set_external_offsets.call_args.kwargs["antennas"] == (0.0, 0.0)


@pytest.mark.parametrize("value,thinking,sway", [("999", 5, 14), ("-9", 0, 0), ("2", 2, 2)])
def test_antenna_amplitude_bounds(monkeypatch, value, thinking, sway):
    """Clamp cue and sway amplitudes without increasing their existing defaults."""
    from reachy_mini_conversation_app.liveliness import SpeechSway, ThinkingAntennaCue

    monkeypatch.setenv("AGENT_THINKING_CUE_MAX_DEG", value)
    monkeypatch.setenv("AGENT_SWAY_MAX_DEG", value)
    assert ThinkingAntennaCue(None).max_rad == pytest.approx(np.deg2rad(thinking))
    assert SpeechSway(None).max_rad == pytest.approx(np.deg2rad(sway))


@pytest.mark.parametrize(
    "text,intent",
    [
        ("Hello", "greeting"),
        ("Goodbye", "goodbye"),
        ("Bye", "goodbye"),
        ("Thanks", "grateful"),
        ("Thank you", "grateful"),
        ("Sorry", "downcast"),
        ("Unfortunately", "downcast"),
        ("Great", "success"),
        ("Done", "success"),
        ("Awesome", "success"),
        ("Careful", "anxious"),
        ("Watch out", "anxious"),
        ("That's hilarious", "laughing"),
        ("Incredible", "amazed"),
        ("Unclear", "confused"),
        ("Yes.", "yes"),
        ("No.", "no"),
        ("Danke", "grateful"),
        ("Vorsicht", "anxious"),
    ],
)
def test_emotion_cues_support_english_and_german(text, intent):
    """English cues mirror the existing German intents without removing German support."""
    from reachy_mini_conversation_app.emotion_cues import emotion_for_turn

    assert emotion_for_turn("", text) == intent
