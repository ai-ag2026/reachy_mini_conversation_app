"""Liveliness helpers for the AGENT path (gap-map Stufe 1, 2026-07-02).

The app ships a complete expressiveness layer (idle policy, emotions with sounds, the daemon
sound library) that the AGENT voice handler never reached — this module wires the small pieces:

- ``IdleActionRunner``: background task that fires the app's weighted ``idle_policy`` (do
  nothing / dance / emotion / look around) after a period of idleness, exactly like the
  base_realtime/gemini handlers do — the trigger simply never existed in the AGENT path.
- ``ensure_chirps_uploaded``: pushes the procedural astromech chirps into the daemon's sound
  library (``/tmp/reachy_mini_sounds`` — wiped on reboot, so this runs at every app start).
  Daemon-played sounds cost ~0 latency and drive the head wobbler for free.
- ``wav_file_to_pcm``: loads an emotion's bundled .wav for playback through the handler's
  normal audio path (half-duplex mute + playback clock respected).
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import time
import wave
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

DAEMON_BASE_URL = os.getenv("AGENT_DAEMON_BASE_URL", "http://localhost:8000")
CHIRP_FILE_PREFIX = "agent_chirp_"

# Idle actions available to the runner — the app's idle_policy weights within this set
# (60% stillness / 16% dance / 16% emotion / 8% look around).
IDLE_TOOL_NAMES = ("idle_do_nothing", "dance", "play_emotion", "move_head")


def _env_on(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() not in ("0", "false", "no", "off", "")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def wav_file_to_pcm(path: str) -> tuple[int, np.ndarray] | None:
    """Load a wav file as (sample_rate, int16 mono PCM); None on any failure."""
    try:
        with wave.open(path, "rb") as w:
            sr = w.getframerate()
            n = w.getnframes()
            ch = w.getnchannels()
            width = w.getsampwidth()
            raw = w.readframes(n)
        if width != 2:
            return None  # only 16-bit PCM supported (the emotion library is 16-bit)
        pcm = np.frombuffer(raw, dtype=np.int16)
        if ch > 1:
            pcm = pcm.reshape(-1, ch).mean(axis=1).astype(np.int16)
        return sr, pcm
    except Exception as exc:
        logger.warning("wav load failed (%s): %r", path, exc)
        return None


class IdleActionRunner:
    """Fires the app's idle policy while nothing else is going on.

    ``is_busy`` must return True while an idle action would disturb (active turn, AGENT
    speaking, user speaking). Actions are queue-ops into the MovementManager, so they are
    cheap; a cooldown keeps the robot from performing constantly.
    """

    def __init__(
        self,
        deps: Any,
        *,
        is_busy: Callable[[], bool],
        idle_after_s: float | None = None,
        cooldown_s: float | None = None,
        check_interval_s: float = 5.0,
    ) -> None:
        self.deps = deps
        self.is_busy = is_busy
        self.idle_after_s = idle_after_s if idle_after_s is not None else _env_float("AGENT_IDLE_AFTER_S", 120.0)
        self.cooldown_s = cooldown_s if cooldown_s is not None else _env_float("AGENT_IDLE_COOLDOWN_S", 90.0)
        self.check_interval_s = check_interval_s
        self._task: asyncio.Task | None = None
        self._last_action = 0.0
        self._last_busy = time.monotonic()

    def note_activity(self) -> None:
        """Call on any interaction (turn start/end, speech) — resets the idle clock."""
        self._last_busy = time.monotonic()

    def start(self) -> None:
        if not _env_on("AGENT_IDLE_ACTIONS", "1"):
            logger.info("idle actions disabled (AGENT_IDLE_ACTIONS=0)")
            return
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())

    def stop(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = None

    async def _loop(self) -> None:
        from reachy_mini_conversation_app.idle_policy import choose_idle_tool_call
        from reachy_mini_conversation_app.tools.core_tools import dispatch_tool_call_obj

        logger.info(
            "idle-action runner up (after %.0fs idle, cooldown %.0fs)", self.idle_after_s, self.cooldown_s
        )
        while True:
            try:
                await asyncio.sleep(self.check_interval_s)
                now = time.monotonic()
                if self.is_busy():
                    self._last_busy = now
                    continue
                mm = getattr(self.deps, "movement_manager", None)
                if mm is not None and hasattr(mm, "is_idle") and not mm.is_idle():
                    continue  # a queued move (emotion/dance) is still running
                if (now - self._last_busy) < self.idle_after_s:
                    continue
                if (now - self._last_action) < self.cooldown_s:
                    continue
                selected = choose_idle_tool_call(IDLE_TOOL_NAMES)
                if selected is None:
                    continue
                name, args = selected
                self._last_action = now
                if name == "idle_do_nothing":
                    continue  # stillness — chosen 60% of the time on purpose
                logger.info("idle action: %s %s", name, args)
                try:
                    result = await dispatch_tool_call_obj(name, args, self.deps)
                    logger.info("idle action result: %s", result)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.warning("idle action %s failed", name, exc_info=True)
            except asyncio.CancelledError:
                return
            except Exception:
                logger.warning("idle-action loop error — continuing", exc_info=True)


def chirp_wav_bytes(name: str, sample_rate: int = 24000, gain: float = 0.9) -> bytes:
    """Render one astromech chirp as a wav file (full gain — the daemon volume governs)."""
    from reachy_agent.voice.astromech import chirp

    pcm = chirp(name, sample_rate, gain=gain)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def ensure_chirps_uploaded(base_url: str = DAEMON_BASE_URL) -> int:
    """Upload all astromech chirps missing from the daemon sound library (idempotent, sync —
    run off-thread). Returns the number uploaded. The library lives in /tmp on the robot, so
    every app start re-checks."""
    import httpx

    from reachy_agent.voice.astromech import CHIRPS

    uploaded = 0
    try:
        with httpx.Client(base_url=base_url, timeout=10.0) as client:
            try:
                existing = set(client.get("/api/media/sounds").json())
            except Exception:
                existing = set()
            for name in CHIRPS:
                fname = f"{CHIRP_FILE_PREFIX}{name}.wav"
                if any(fname in str(e) for e in existing):
                    continue
                data = chirp_wav_bytes(name)
                r = client.post(
                    "/api/media/sounds/upload",
                    files={"file": (fname, data, "audio/wav")},
                )
                r.raise_for_status()
                uploaded += 1
        if uploaded:
            logger.info("uploaded %d chirps to the daemon sound library", uploaded)
    except Exception as exc:
        logger.warning("chirp upload to daemon failed (queue path stays the fallback): %r", exc)
    return uploaded


def ensure_daemon_sound(path: str, base_url: str = DAEMON_BASE_URL) -> str | None:
    """Make a local sound file (any GStreamer-decodable format, e.g. the emotion library's
    .ogg) available in the daemon sound library; returns the library file name or None.
    Idempotent + sync — run off-thread. Library lives in /tmp (wiped on reboot)."""
    import httpx

    fname = os.path.basename(path)
    try:
        with httpx.Client(base_url=base_url, timeout=10.0) as client:
            try:
                existing = set(client.get("/api/media/sounds").json())
            except Exception:
                existing = set()
            if not any(fname in str(e) for e in existing):
                with open(path, "rb") as fh:
                    r = client.post(
                        "/api/media/sounds/upload",
                        files={"file": (fname, fh.read(), "application/octet-stream")},
                    )
                r.raise_for_status()
                logger.info("uploaded sound %s to the daemon library", fname)
        return fname
    except Exception as exc:
        logger.warning("daemon sound upload %s failed: %r", fname, exc)
        return None


def play_daemon_sound(file_name: str, base_url: str = DAEMON_BASE_URL) -> bool:
    """Fire-and-forget daemon-side sound playback (drives the head wobbler for free).
    Sync + fast (local REST); returns False on failure so callers fall back."""
    import httpx

    try:
        r = httpx.post(f"{base_url}/api/media/play_sound", json={"file": file_name}, timeout=3.0)
        r.raise_for_status()
        return True
    except Exception as exc:
        logger.warning("daemon play_sound %s failed: %r", file_name, exc)
        return False


# ── Stufe 2 (gap-map): speech-sway antennas, DOA orienting, IMU reactivity ─────────────────────


def read_local_doa(base_url: str = DAEMON_BASE_URL) -> dict | None:
    """Read the mic-array Direction-of-Arrival from the local daemon (ported from
    conversation-app-agent-bridge/look_toward_sound)."""
    import httpx

    try:
        r = httpx.get(f"{base_url}/api/state/doa", timeout=3.0)
        r.raise_for_status()
        parsed = r.json()
        return parsed if isinstance(parsed, dict) else None
    except Exception as exc:
        logger.debug("DoA read failed: %r", exc)
        return None


def map_doa_angle_to_direction(angle_radians: float, *, front_deadzone_radians: float = 0.35) -> str:
    """Map ReSpeaker DoA radians to a coarse head direction (0=left, pi/2=front/back, pi=right).
    Ported from conversation-app-agent-bridge/audio_orientation.py."""
    import math

    angle = max(0.0, min(math.pi, float(angle_radians)))
    front = math.pi / 2
    if angle < front - front_deadzone_radians:
        return "left"
    if angle > front + front_deadzone_radians:
        return "right"
    return "front"


async def orient_to_speaker(deps: Any) -> str | None:
    """One-shot: read DoA and, if the sound clearly comes from the side, queue a head turn
    toward it (through the app's move_head tool = MovementManager seam). Returns the direction
    moved, or None."""
    doa = await asyncio.to_thread(read_local_doa)
    if not doa or "angle" not in doa:
        return None
    direction = map_doa_angle_to_direction(float(doa["angle"]))
    if direction == "front":
        return None
    try:
        from reachy_mini_conversation_app.tools.core_tools import dispatch_tool_call_obj

        result = await dispatch_tool_call_obj("move_head", {"direction": direction}, deps)
        if "error" in result:
            return None
        logger.info("oriented toward speaker: %s (doa %.2f rad)", direction, float(doa["angle"]))
        return direction
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.debug("orient_to_speaker failed", exc_info=True)
        return None


class SpeechSway:
    """Antenna sway synced to AGENT's own speech (gap-map Stufe 2 / Operators Prio-1-Feature).

    The daemon HeadWobbler already nods the HEAD to played audio; the antennas stayed dead.
    ``feed()`` receives every queued TTS segment together with the monotonic time it will
    actually START playing (the handler's playback cursor), converts it to an RMS envelope, and
    a 25 Hz driver applies the amplitude as ADDITIVE antenna offsets through the
    MovementManager's external-offsets channel — so it composes with breathing/face-tracking
    instead of fighting them, and a barge flush zeroes it instantly.
    """

    def __init__(self, movement_manager: Any, *, hop_s: float = 0.04) -> None:
        self.mm = movement_manager
        self.hop_s = hop_s
        self.max_rad = float(np.deg2rad(_env_float("AGENT_SWAY_MAX_DEG", 14.0)))
        self.gain = _env_float("AGENT_SWAY_GAIN", 5.0)  # rms (0..1) -> amplitude scale
        self._points: list[tuple[float, float]] = []  # (play_at_monotonic, amplitude 0..1)
        self._task: asyncio.Task | None = None
        self._active = False  # last applied state (avoid redundant zero writes)
        self._phase = 0.0

    def feed(self, sr: int, pcm: np.ndarray, play_at: float) -> None:
        if self._task is None:
            return
        try:
            x = np.asarray(pcm, dtype=np.float32) / 32768.0
            hop = max(1, int(sr * self.hop_s))
            for i in range(0, len(x), hop):
                w = x[i : i + hop]
                if not len(w):
                    continue
                rms = float(np.sqrt(np.mean(w * w)))
                amp = min(1.0, rms * self.gain)
                self._points.append((play_at + i / sr, amp))
            # bound the schedule (a runaway producer must not grow unbounded)
            if len(self._points) > 4000:
                self._points = self._points[-2000:]
        except Exception:
            logger.debug("sway feed failed", exc_info=True)

    def clear(self) -> None:
        """Barge/flush: drop the schedule and release the antennas immediately."""
        self._points.clear()
        self._apply(0.0, force=True)

    def start(self) -> None:
        if not _env_on("AGENT_SPEECH_SWAY", "1"):
            logger.info("speech sway disabled (AGENT_SPEECH_SWAY=0)")
            return
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())

    def stop(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = None
        self._apply(0.0, force=True)

    def _apply(self, amp: float, force: bool = False) -> None:
        setter = getattr(self.mm, "set_external_offsets", None)
        if not callable(setter):
            return
        if amp <= 0.001:
            if self._active or force:
                self._active = False
                try:
                    setter((0.0,) * 6, antennas=(0.0, 0.0))
                except Exception:
                    pass
            return
        self._active = True
        # opposite-direction sway with a slow oscillation so it reads as lively, not jittery
        a = amp * self.max_rad * float(np.sin(self._phase))
        try:
            setter((0.0,) * 6, antennas=(a, -a))
        except Exception:
            pass

    async def _loop(self) -> None:
        decay = 0.0
        while True:
            try:
                await asyncio.sleep(0.04)  # 25 Hz driver
                now = time.monotonic()
                # consume all points due; keep the loudest of the due window
                due_amp = None
                while self._points and self._points[0][0] <= now:
                    _, a = self._points.pop(0)
                    due_amp = a if due_amp is None else max(due_amp, a)
                if due_amp is not None:
                    decay = max(decay * 0.85, due_amp)
                else:
                    decay *= 0.85  # ~150 ms release
                self._phase += 2.0 * np.pi * 1.8 * 0.04  # 1.8 Hz sway
                self._apply(decay if decay > 0.02 else 0.0)
            except asyncio.CancelledError:
                self._apply(0.0, force=True)
                return
            except Exception:
                logger.debug("sway loop error", exc_info=True)


class ThinkingAntennaCue:
    """Very subtle antenna motion while the backend is working before speech begins."""

    def __init__(self, movement_manager: Any) -> None:
        self.mm = movement_manager
        self.delay_s = _env_float("AGENT_THINKING_CUE_DELAY_S", 0.8)
        self.max_rad = float(np.deg2rad(_env_float("AGENT_THINKING_CUE_MAX_DEG", 2.0)))
        self.frequency_hz = _env_float("AGENT_THINKING_CUE_HZ", 0.18)
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        """Begin a delayed cue, replacing any cue left by a previous turn."""
        self.stop()
        if not _env_on("AGENT_THINKING_CUE", "1"):
            return
        self._task = asyncio.create_task(self._loop())

    def stop(self) -> None:
        """Stop the cue and release its additive antenna offset."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = None
        self._apply(0.0)

    def _apply(self, offset: float) -> None:
        setter = getattr(self.mm, "set_external_offsets", None)
        if callable(setter):
            try:
                setter((0.0,) * 6, antennas=(offset, -offset))
            except Exception:
                logger.debug("thinking antenna cue write failed", exc_info=True)

    async def _loop(self) -> None:
        try:
            await asyncio.sleep(max(0.0, self.delay_s))
            started = time.monotonic()
            while True:
                elapsed = time.monotonic() - started
                # Slow sine with a smooth onset so the cue reads as waiting, not twitching.
                envelope = min(1.0, elapsed / 1.5)
                offset = self.max_rad * envelope * float(np.sin(2.0 * np.pi * self.frequency_hz * elapsed))
                self._apply(offset)
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            self._apply(0.0)
            return


class ImuWatcher:
    """React when the robot is physically bumped/lifted (gap-map Stufe 2).

    Polls the Wireless IMU (50 Hz daemon cache, cheap read) and fires ``on_event`` when the
    acceleration magnitude deviates from its running baseline by more than the threshold.
    Silent no-op when no IMU is available (Lite/sim)."""

    def __init__(self, robot: Any, on_event: Callable[[str], None]) -> None:
        self.robot = robot
        self.on_event = on_event
        self.threshold = _env_float("AGENT_IMU_THRESHOLD", 2.5)  # m/s^2 deviation
        self.cooldown_s = _env_float("AGENT_IMU_COOLDOWN_S", 20.0)
        self._task: asyncio.Task | None = None
        self._last_fire = 0.0
        self._baseline: float | None = None

    @staticmethod
    def _accel_magnitude(data: Any) -> float | None:
        for attr in ("accel", "acceleration", "linear_acceleration"):
            v = getattr(data, attr, None)
            if v is None and isinstance(data, dict):
                v = data.get(attr)
            if v is None:
                continue
            try:
                def _axis(obj: Any, name: str, idx: int) -> float:
                    val = getattr(obj, name, None)  # NOT `or obj[idx]`: 0.0 is a valid reading
                    if val is None:
                        val = obj[idx]
                    return float(val)

                arr = np.asarray([_axis(v, "x", 0), _axis(v, "y", 1), _axis(v, "z", 2)], dtype=np.float64)
                return float(np.linalg.norm(arr))
            except Exception:
                continue
        return None

    def start(self) -> None:
        if not _env_on("AGENT_IMU_REACT", "1"):
            logger.info("IMU reactivity disabled (AGENT_IMU_REACT=0)")
            return
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())

    def stop(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = None

    async def _loop(self) -> None:
        misses = 0
        while True:
            try:
                await asyncio.sleep(0.1)  # 10 Hz is plenty for bump detection
                data = getattr(self.robot, "imu", None)
                if data is None:
                    misses += 1
                    if misses > 50:
                        logger.info("no IMU data (Lite/sim?) — IMU watcher stops")
                        return
                    continue
                mag = self._accel_magnitude(data)
                if mag is None:
                    misses += 1
                    if misses > 50:
                        logger.info("IMU data has no readable acceleration — watcher stops")
                        return
                    continue
                misses = 0
                if self._baseline is None:
                    self._baseline = mag
                    continue
                dev = abs(mag - self._baseline)
                self._baseline = 0.98 * self._baseline + 0.02 * mag  # slow EMA tracks gravity
                now = time.monotonic()
                if dev > self.threshold and (now - self._last_fire) > self.cooldown_s:
                    self._last_fire = now
                    logger.info("IMU event: |accel| deviation %.2f m/s^2", dev)
                    try:
                        self.on_event("bump")
                    except Exception:
                        logger.debug("IMU event handler failed", exc_info=True)
            except asyncio.CancelledError:
                return
            except Exception:
                logger.debug("IMU loop error", exc_info=True)
