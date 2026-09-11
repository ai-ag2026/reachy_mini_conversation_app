"""Companion mode (gap-map Stufe 3, 2026-07-02) — AGENT may act on its OWN perception.

An explicit, visibly toggleable mode (dashboard toggle "companion" + env AGENT_COMPANION,
default OFF): local event watchers turn perception into short event notices that are handed to
the BRAIN as a turn — AGENT decides whether anything is worth saying (an empty answer stays
silent). This is deliberately a separate mode: continuously acting on mic/camera events is a
conscious exposure choice, so it ships opt-in with one obvious off-switch.

Events (each with its own cooldown, plus a global one):
- sustained nearby speech while idle (mic-array hardware speech flag via DoA endpoint)
- a face (re)appearing after a long absence (requires head tracking)
- physical bump/lift (IMU) — the local startle reflex fires regardless; companion additionally
  tells the brain so it can react in character.
"""

from __future__ import annotations
import os
import time
import asyncio
import logging
from typing import Any, Callable


logger = logging.getLogger(__name__)


def _env_on(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() not in ("0", "false", "no", "off", "")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


class CompanionWatcher:
    """Polls local perception and emits event notices while companion mode is ON."""

    def __init__(
        self,
        deps: Any,
        *,
        on_event: Callable[[str], None],
        is_busy: Callable[[], bool],
        camera_worker: Any | None = None,
        poll_s: float = 2.5,
    ) -> None:
        """Initialize the configured state."""
        self.deps = deps
        self.on_event = on_event
        self.is_busy = is_busy
        self.camera_worker = camera_worker
        self.poll_s = poll_s
        self.global_cooldown_s = _env_float("AGENT_COMPANION_COOLDOWN_S", 120.0)
        self._task: asyncio.Task[Any] | None = None
        self._last_event = 0.0
        self._speech_streak = 0
        self._face_last_seen: float | None = None
        self._face_absent_since = time.monotonic()

    @staticmethod
    def enabled() -> bool:
        """Handle enabled."""
        return _env_on("AGENT_COMPANION", "0")

    def start(self) -> None:
        """Start the background worker."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())

    def stop(self) -> None:
        """Stop the background worker."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = None

    def _fire(self, desc: str) -> None:
        now = time.monotonic()
        if now - self._last_event < self.global_cooldown_s:
            return
        self._last_event = now
        logger.info("companion event: %s", desc)
        try:
            self.on_event(desc)
        except Exception:
            logger.debug("companion event handler failed", exc_info=True)

    async def _loop(self) -> None:
        from reachy_mini_conversation_app.liveliness import read_local_doa

        logger.info("companion watcher up (mode currently %s)", "ON" if self.enabled() else "off")
        while True:
            try:
                await asyncio.sleep(self.poll_s)
                if not self.enabled():
                    self._speech_streak = 0
                    continue
                if self.is_busy():
                    self._speech_streak = 0
                    continue

                # nearby speech (hardware flag; sustained across polls to avoid one-off noise)
                doa = await asyncio.to_thread(read_local_doa)
                if doa and doa.get("speech_detected"):
                    self._speech_streak += 1
                    if self._speech_streak >= 2:
                        self._speech_streak = 0
                        self._fire("Speech is happening nearby (microphone array), but nobody addressed you directly.")
                else:
                    self._speech_streak = 0

                # face (re)appearance — only meaningful with head tracking running
                cam = self.camera_worker
                if cam is not None:
                    seen = getattr(cam, "last_face_detected_time", None)
                    now = time.monotonic()
                    if seen is None:
                        if self._face_last_seen is not None:
                            self._face_absent_since = now
                        self._face_last_seen = None
                    else:
                        if self._face_last_seen is None and (now - self._face_absent_since) > 120.0:
                            self._fire("A face has just appeared, or reappeared, in front of your camera.")
                        self._face_last_seen = seen
            except asyncio.CancelledError:
                return
            except Exception:
                logger.debug("companion loop error", exc_info=True)


def event_transcript(desc: str) -> str:
    """Build the event notice handed to the brain as a turn.

    Explicitly frames it as a non-user event and licenses silence — an empty answer ends the turn without
    speech.
    """
    return (
        f"[Event, companion mode — NOT a user turn: {desc}] "
        "Respond only if one short, natural English reaction genuinely fits (one sentence, in character). "
        "Otherwise return completely empty text and do nothing."
    )
