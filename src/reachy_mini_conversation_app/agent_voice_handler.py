# ruff: noqa: D101,D102,D103,D105,D107
from __future__ import annotations
import os
import time
import asyncio
import inspect
import logging
from typing import Any, Protocol, TypeAlias
from dataclasses import dataclass

import numpy as np


logger = logging.getLogger(__name__)
from fastrtc import AdditionalOutputs, wait_for_item
from numpy.typing import NDArray

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.conversation_handler import AudioFrame, HandlerOutput, ConversationHandler


MaybeText: TypeAlias = str | object
MaybeAudioFrame: TypeAlias = AudioFrame | object


class TextAgentClient(Protocol):
    def ask(self, transcript: str) -> MaybeText: ...


class AudioTtsClient(Protocol):
    def synthesize(self, text: str) -> MaybeAudioFrame: ...

    def set_voice(self, voice: str) -> None: ...


class FakeTextAgentClient:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[str] = []

    def ask(self, transcript: str) -> str:
        self.calls.append(transcript)
        return self.reply


@dataclass
class FakeAudioTtsClient:
    sample_rate: int
    audio: NDArray[np.int16]

    def __post_init__(self) -> None:
        self.calls: list[str] = []

    def synthesize(self, text: str) -> AudioFrame:
        self.calls.append(text)
        return self.sample_rate, self.audio


class AgentVoiceHandler(ConversationHandler):
    def __init__(
        self,
        deps: ToolDependencies,
        *,
        agent_client: TextAgentClient,
        tts_client: AudioTtsClient,
        lead_in_client: Any = None,
        output_sample_rate: int = 24000,
        input_sample_rate: int = 16000,
    ) -> None:
        super().__init__(
            expected_layout="mono",
            output_sample_rate=output_sample_rate,
            input_sample_rate=input_sample_rate,
        )
        self.deps = deps
        self.agent_client = agent_client
        self.tts_client = tts_client
        # Optional fast 9B lead-in (adaptive latency mask). None -> plain full-brain path.
        self._lead_in_client = lead_in_client
        self.output_queue: asyncio.Queue[AudioFrame | AdditionalOutputs] = asyncio.Queue()
        self.second_assistant_detected = False
        self._closed = False
        self._mic_muted = False  # dashboard mic toggle (drops mic frames in receive when True)
        self._prewarmed = False  # latency #2: fire the one-shot prefix-cache pre-warm on first receive
        self._current_voice = "default"
        self._turn_lock = asyncio.Lock()
        self._stt = None  # lazy in-receive STT front-end (Silero VAD + Parakeet) — see _ensure_stt
        # Speaking gate: monotonic deadline until which AGENT is emitting audio. When barge-in is OFF
        # this is a hard half-duplex mute (drop mic frames so the daemon-captured playback isn't
        # transcribed as a new turn). When barge-in is ON it instead routes frames to the barge
        # monitor. Measured (docs/findings/2026-06-24_daemon-aec-and-semantic-gate.md): Silero does
        # NOT self-trigger on AGENT's own daemon playback, so an open mic during playback is safe.
        self._speaking_until = 0.0
        # Single-flight: True from the moment a transcript is accepted until AGENT finishes the turn,
        # so one spoken question yields exactly one reply (the VAD may split a question into several
        # utterances; without this each becomes its own turn -> multiple/duplicate answers).
        self._turn_active = False
        # Barge-in (full-duplex): AGENT_BARGE_IN=0 falls back to the proven hard half-duplex mute.
        self._barge_enabled = os.getenv("AGENT_BARGE_IN", "1").strip().lower() not in ("0", "false", "no", "off")
        self._barge_stt = None          # lazy second Silero front-end, fed only during playback
        self._barge_event = asyncio.Event()  # set on speech ONSET over playback -> pause AGENT now
        self._barge_transcript: str | None = None  # the full interrupt utterance once it endpoints
        self._pending_barge: str | None = None  # a committed interrupt -> becomes the next turn
        self._turn_task: asyncio.Task | None = None  # current turn task — the watchdog cancels it on stall
        self._turn_seq = 0              # generation counter: a stale turn's finally must not clobber a newer turn
        # Stage 2 — preemptive turn-start (AGENT_PREEMPT_TURN=1, stream STT only): when the STT partial
        # is stable mid-speech, start generating on it early so the cloud TTFT overlaps the speech tail
        # + endpoint pause. The speculative turn HOLDS its audio behind _spec_gate until the real
        # endpoint confirms the transcript (confirm ~450ms << TTFT ~3-6s, so no wrong word is ever
        # spoken); on mismatch it is cancelled + a fresh turn runs. OFF by default.
        self._spec_task: asyncio.Task | None = None
        self._spec_partial: str | None = None
        self._spec_gate: asyncio.Event | None = None  # set = confirmed, release held audio
        self._spec_seq = 0             # tentative turn seq the speculative task adopts if confirmed
        self._classify_inflight = False  # single-flight: at most one barge classify at a time
        # Playback clock: monotonic cursor of when the last queued audio segment will finish playing.
        # _speaking_until derives from it (audit 2026-07-02: the old per-segment "now + const" estimate
        # drifted 15% of the reply length because production paces 0.85x real-time).
        self._playback_cursor = 0.0
        # Output loudness: AGENT plays at _output_gain (applied to all queued PCM, > 1 = louder than the
        # ALSA max). Non-pausing barge-in keeps AGENT at full volume while the user speaks (Operator's
        # preferred behavior — no ducking); the awake HW AEC keeps the user's parallel transcript clean.
        self._output_gain = float(os.getenv("AGENT_OUTPUT_GAIN", "1.2"))
        self._last_progress = 0.0       # monotonic of the last queued audio / barge decision (stall watchdog)
        # Barge only once AGENT is actually SPEAKING: during the silent 3-6s think phase there is nothing
        # to interrupt, and a user re-prompt ("hörst du mich?") would otherwise commit-barge and cancel
        # the turn before any audio — an endless no-answer loop when the quick-take ack is off
        # (live 2026-07-02). Set True on the first queued audio; reset per turn.
        self._turn_spoke = False
        # Sustained-voice gate: Silero's in_speech flickers True for 1-3 chunks on AGENT's OWN playback
        # (verified live — it never endpoints into a transcript) which would falsely pause AGENT the
        # instant it starts speaking. A real interrupt sustains, so only declare a barge onset after N
        # consecutive voiced chunks (~190ms) and, optionally, mic energy above the echo floor.
        self._barge_min_chunks = int(os.getenv("AGENT_BARGE_MIN_CHUNKS", "6"))
        self._barge_min_rms = float(os.getenv("AGENT_BARGE_MIN_RMS", "0.0"))  # 0 = energy gate off
        # Stage 4: platform transport can deliver proactive/background messages
        # (async-delegation results, cron, send_message). Speak them in a safe gap.
        _set_proactive = getattr(self.agent_client, "set_proactive_handler", None)
        if callable(_set_proactive):
            _set_proactive(self._speak_proactive)

    async def _speak_proactive(self, text: str) -> None:
        """Speak a proactively-delivered message (background result / cron / send_message)
        in a safe half-duplex gap. Mutually exclusive with interactive turns via
        ``_turn_lock``; keeps ``_speaking_until`` ahead so receive() hard-mutes the mic
        while speaking (no barge target). Drops the message if no gap opens in time.
        """
        text = (text or "").strip()
        if not text or self._closed:
            return
        acquire_s = float(os.getenv("AGENT_PROACTIVE_ACQUIRE_S", "25"))
        try:
            await asyncio.wait_for(self._turn_lock.acquire(), timeout=acquire_s)
        except asyncio.TimeoutError:
            logger.info("[proactive] no half-duplex gap within %.0fs — dropping: %s", acquire_s, text[:60])
            return
        try:
            if self._closed:
                return
            # let any tail playback drain so we don't clip the previous reply
            while time.monotonic() < self._speaking_until and not self._closed:
                await asyncio.sleep(0.1)
            self._barge_event.clear()
            self._status_chirp("notify")  # non-verbal cue that an unsolicited (background) result is coming
            self.output_queue.put_nowait(AdditionalOutputs({"role": "assistant", "content": text}))
            lead = os.getenv("AGENT_PROACTIVE_LEADIN", "Kurzer Nachtrag:").strip()
            for part in ([lead] if lead else []) + [text]:
                if self._closed:
                    break
                await self._speak_sentence(part)
            self._speaking_until = max(self._speaking_until, time.monotonic()) + 0.4
        except Exception:
            logger.warning("[proactive] speak failed", exc_info=True)
        finally:
            self._turn_lock.release()

    def _ensure_stt(self):
        """Lazily build the Silero+Parakeet front-end used to transcribe the live mic in receive().
        Firmware AEC keeps AGENT's own playback out of the mic, so this is the in-receive STT that the
        no-op receive() lacked (the app's agent backend now actually hears the user).
        """
        if self._stt is None:
            from reachy_agent.voice.stt_frontend import AgentSttFrontend
            self._stt = AgentSttFrontend(
                stt_remote_url=os.getenv("AGENT_STT_BASE_URL"),
                stt_model=os.getenv("AGENT_STT_MODEL", "parakeet"),
                pause_ms=int(os.getenv("AGENT_PAUSE_MS", "550")),
                vad_threshold=float(os.getenv("AGENT_VAD_THRESHOLD", "0.4")),
            )
        return self._stt

    def _ensure_barge_stt(self):
        """Second Silero+Parakeet front-end used ONLY while AGENT is speaking, to detect a real human
        interrupt over the playback. Tuned for fast onset (short pause) — the goal is to react the
        instant the user starts, not to wait for a full sentence; the clean tail (after AGENT pauses)
        carries the actual instruction.
        """
        if self._barge_stt is None:
            from reachy_agent.voice.stt_frontend import AgentSttFrontend
            self._barge_stt = AgentSttFrontend(
                stt_remote_url=os.getenv("AGENT_STT_BASE_URL"),
                stt_model=os.getenv("AGENT_STT_MODEL", "parakeet"),
                pause_ms=int(os.getenv("AGENT_BARGE_PAUSE_MS", "350")),
                vad_threshold=float(os.getenv("AGENT_BARGE_THRESHOLD", "0.5")),
            )
        return self._barge_stt

    def _gain(self, arr: np.ndarray) -> np.ndarray:
        """Apply output loudness to a PCM segment (clipped to int16)."""
        if self._output_gain == 1.0:
            return arr
        return np.clip(arr.astype(np.float32) * self._output_gain, -32768, 32767).astype(np.int16)

    def _status_chirp(self, name: str) -> None:
        """Queue a short non-verbal astromech status cue (AGENT_CHIRPS=1). Best-effort — never breaks
        a turn. Mutes the mic for the cue's duration (half-duplex) but does NOT set _turn_spoke: a cue
        is not spoken content, so barge stays suppressed until the real answer starts."""
        if not self._env_on("AGENT_CHIRPS", "0"):
            return
        try:
            from reachy_agent.voice.astromech import chirp as _chirp

            sr = 24000
            pcm = _chirp(name, sr, gain=self._float_env("AGENT_CHIRP_GAIN", 0.5))
            # Daemon-library path (gap-map Stufe 1): pre-uploaded chirps play daemon-side with
            # ~0 latency AND drive the head wobbler (the robot visibly reacts to its own beep).
            # Off-thread fire-and-forget; the queue path is the fallback.
            if self._env_on("AGENT_CHIRP_VIA_DAEMON", "1"):
                from reachy_mini_conversation_app.liveliness import CHIRP_FILE_PREFIX, play_daemon_sound

                t = asyncio.create_task(
                    asyncio.to_thread(play_daemon_sound, f"{CHIRP_FILE_PREFIX}{name}.wav")
                )
                self._misc_tasks = getattr(self, "_misc_tasks", set())
                self._misc_tasks.add(t)
                t.add_done_callback(self._misc_tasks.discard)

                def _fallback(task: asyncio.Task, _sr=sr, _pcm=pcm) -> None:
                    try:
                        if task.cancelled() or task.result():
                            return
                    except Exception:
                        pass
                    try:
                        self.output_queue.put_nowait((_sr, self._gain(_pcm)))
                    except Exception:
                        pass

                t.add_done_callback(_fallback)
            else:
                self.output_queue.put_nowait((sr, self._gain(pcm)))
            # Book the cue into the playback clock (speech=False): bumping only _speaking_until let
            # the next speech segment restart the cursor at `now`, so the tail mute ended ~a chirp
            # too early and the pacing lead overfilled (review 2026-07-02 round 2, P3).
            self._advance_playback_clock(len(pcm) / sr + 0.15, speech=False)
        except Exception:
            logger.debug("status chirp %r failed", name, exc_info=True)

    def _reset_barge(self, keep_pending: bool = False) -> None:
        # keep_pending: a turn that was WAITING on the lock while a proactive message spoke must not
        # discard a command the user committed against that proactive speech (audit 2026-07-02) —
        # the turn's finally chains it as the next turn instead.
        self._barge_event.clear()
        self._barge_transcript = None
        if not keep_pending:
            self._pending_barge = None
        if self._barge_stt is not None:
            self._barge_stt.reset()

    def _flush_downstream(self) -> None:
        sway = getattr(self, "_speech_sway", None)
        if sway is not None:
            try:
                sway.clear()
            except Exception:
                pass
        """Drop audio already handed to the daemon player. On a stop, the handler's output_queue is
        nearly empty (play_loop drains it eagerly into the GStreamer appsrc) — the audible backlog
        lives DOWNSTREAM, so a stop must also flush the SDK player (audit 2026-07-02: without this,
        AGENT keeps talking ~15% of the reply-so-far after a stop). console wires _clear_queue to
        LocalStream.clear_audio_queue -> SDK clear_player().
        """
        clear = getattr(self, "_clear_queue", None)
        if callable(clear):
            try:
                clear()
            except Exception:
                logger.warning("AGENT player flush failed", exc_info=True)
        self._playback_cursor = 0.0
        self._speaking_until = time.monotonic() + 0.3  # short guard while the player drains the flush

    @staticmethod
    def _env_on(name: str, default: str = "1") -> bool:
        return os.getenv(name, default).strip().lower() not in ("0", "false", "no", "off")

    def get_toggles(self) -> dict:
        """Current state of the dashboard live switches. Most are env-backed and read per turn, so a
        change takes effect on the next turn with no restart; mic is a handler-level mute.
        """
        return {
            "tools": self._env_on("AGENT_VOICE_TOOLS"),          # full AGENT agent tools
            "vision": self._env_on("AGENT_VISION_ENABLED"),       # camera vision (gemma/native)
            "person_id": not self._env_on("AGENT_VISION_BLOCK_PERSON_ID", "0"),  # person recognition
            "mic": not self._mic_muted,                          # microphone listening
            "companion": self._env_on("AGENT_COMPANION", "0"),    # proactive perception mode
            "idle_actions": self._env_on("AGENT_IDLE_ACTIONS"),   # idle emotes/dances/looks
            "speech_sway": self._env_on("AGENT_SPEECH_SWAY"),     # antenna sway while speaking
        }

    def set_toggle(self, name: str, on: bool) -> dict:
        """Flip a live switch and return the new full toggle state."""
        on = bool(on)
        if name == "tools":
            os.environ["AGENT_VOICE_TOOLS"] = "1" if on else "0"
        elif name == "vision":
            os.environ["AGENT_VISION_ENABLED"] = "1" if on else "0"
        elif name == "person_id":
            os.environ["AGENT_VISION_BLOCK_PERSON_ID"] = "0" if on else "1"  # env BLOCKS -> inverted
        elif name == "companion":
            os.environ["AGENT_COMPANION"] = "1" if on else "0"
        elif name == "idle_actions":
            os.environ["AGENT_IDLE_ACTIONS"] = "1" if on else "0"
            runner = getattr(self, "_idle_runner", None)
            if runner is not None:
                (runner.start if on else runner.stop)()
        elif name == "speech_sway":
            os.environ["AGENT_SPEECH_SWAY"] = "1" if on else "0"
            sway = getattr(self, "_speech_sway", None)
            if sway is not None:
                (sway.start if on else sway.stop)()
        elif name == "mic":
            # Just flip the flag — receive() drops frames while muted. Do NOT touch _stt here: feed()
            # may be running in a worker thread (offloaded), so mutating its state from this loop
            # callback would race; the VAD re-endpoints cleanly after unmute anyway.
            self._mic_muted = not on
        else:
            raise KeyError(name)
        logger.info("AGENT toggle: %s -> %s", name, on)
        return self.get_toggles()

    # Live runtime knobs (env-backed, read per turn -> next-turn effect, no restart). The console
    # persists them to .env too, so they survive a restart. Maps dashboard name -> env var.
    SETTING_ENV: dict[str, str] = {
        "reasoning_effort": "AGENT_VOICE_REASONING_EFFORT",
        "quicktake_delay_s": "AGENT_QUICKTAKE_DELAY_S",
        "first_audio_budget_s": "AGENT_VOICE_FIRST_AUDIO_BUDGET_S",
        "quicktake": "AGENT_QUICKTAKE_ENABLED",
        "tool_status": "AGENT_TOOL_STATUS_ENABLED",
        "output_gain": "AGENT_OUTPUT_GAIN",
    }

    @staticmethod
    def _truthy(v: object) -> bool:
        if isinstance(v, bool):
            return v
        return str(v).strip().lower() in ("1", "true", "yes", "on")

    @staticmethod
    def _float_env(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)) or default)
        except ValueError:
            return default

    def get_settings(self) -> dict:
        """Current values of the live runtime knobs (latency / volume), for the dashboard panel."""
        return {
            "reasoning_effort": os.getenv("AGENT_VOICE_REASONING_EFFORT", "minimal"),
            "quicktake_delay_s": self._float_env("AGENT_QUICKTAKE_DELAY_S", 0.8),
            "first_audio_budget_s": self._float_env("AGENT_VOICE_FIRST_AUDIO_BUDGET_S", 20.0),
            "output_gain": round(self._output_gain, 2),
            "quicktake": self._env_on("AGENT_QUICKTAKE_ENABLED"),
            "tool_status": self._env_on("AGENT_TOOL_STATUS_ENABLED"),
        }

    def set_setting(self, name: str, value: object) -> dict:
        """Set one live runtime knob (validated); returns the new full settings state. Raises KeyError
        for an unknown name and ValueError for a bad value (the console maps these to 4xx).
        """
        if name == "reasoning_effort":
            v = str(value).strip().lower()
            if v not in ("minimal", "low", "medium", "high"):
                raise ValueError("reasoning_effort must be one of minimal/low/medium/high")
            os.environ["AGENT_VOICE_REASONING_EFFORT"] = v
        elif name == "quicktake_delay_s":
            os.environ["AGENT_QUICKTAKE_DELAY_S"] = str(max(0.0, min(5.0, float(value))))
        elif name == "first_audio_budget_s":
            os.environ["AGENT_VOICE_FIRST_AUDIO_BUDGET_S"] = str(max(0.0, min(120.0, float(value))))
        elif name == "output_gain":
            gain = max(0.1, min(3.0, float(value)))
            self._output_gain = gain               # live: applied to all queued PCM
            os.environ["AGENT_OUTPUT_GAIN"] = str(gain)
        elif name in ("quicktake", "tool_status"):
            os.environ[self.SETTING_ENV[name]] = "1" if self._truthy(value) else "0"
        else:
            raise KeyError(name)
        logger.info("AGENT setting: %s -> %s", name, value)
        return self.get_settings()

    def copy(self) -> AgentVoiceHandler:
        # fastrtc calls copy() per connection (gradio mode). Carry the constructor + runtime
        # settings so the session copy behaves identically — notably lead_in_client, without
        # which quicktake was silently dead in the copy (audit 2026-07-02).
        copied = type(self)(
            self.deps,
            agent_client=self.agent_client,
            tts_client=self.tts_client,
            lead_in_client=self._lead_in_client,
        )
        copied._current_voice = self._current_voice
        copied._output_gain = self._output_gain
        copied._mic_muted = self._mic_muted
        return copied

    async def start_up(self) -> None:
        self._closed = False
        # Platform transport: start the client's reconnect supervisor so the proactive channel
        # is alive from app start and survives gateway restarts — previously the ws only ever
        # connected inside the first user turn (audit 2026-07-02).
        start = getattr(self.agent_client, "start", None)
        if callable(start):
            try:
                await start()
            except Exception:
                logger.warning("AGENT client start failed", exc_info=True)
        # Pre-warm the Silero+Parakeet front-ends off the event loop so the first mic frame (and
        # the first barge frame while AGENT speaks) doesn't stall the loop on the ONNX model load
        # (audit 2026-07-02). Best-effort — receive() still lazily builds them if this is skipped.
        async def _prewarm_stt() -> None:
            try:
                await asyncio.to_thread(self._ensure_stt)
                await asyncio.to_thread(self._ensure_barge_stt)
            except Exception:
                logger.debug("STT pre-warm skipped", exc_info=True)
        self._stt_prewarm_task = asyncio.create_task(_prewarm_stt())
        # Liveliness (gap-map Stufe 1): idle actions + daemon chirp library + emotion sounds.
        try:
            from reachy_mini_conversation_app.liveliness import IdleActionRunner, ensure_chirps_uploaded

            def _busy() -> bool:
                if self._turn_active or self._closed:
                    return True
                if time.monotonic() < self._speaking_until:
                    return True
                stt = self._stt
                return bool(stt is not None and stt.in_speech)

            self._idle_runner = IdleActionRunner(self.deps, is_busy=_busy)
            self._idle_runner.start()
            from reachy_mini_conversation_app.liveliness import ImuWatcher, SpeechSway

            self._speech_sway = SpeechSway(self.deps.movement_manager)
            self._speech_sway.start()
            self._imu_watcher = ImuWatcher(self.deps.reachy_mini, self._on_imu_event)
            self._imu_watcher.start()
            # Body-tool surface (Stufe 3): the gateway's reachy_body tool reaches the body here.
            if hasattr(self.agent_client, "on_tool_call"):
                self.agent_client.on_tool_call = self._on_body_tool
            # Companion mode (Stufe 3): watcher always runs, acts only while the toggle is ON.
            from reachy_mini_conversation_app.companion import CompanionWatcher

            self._companion = CompanionWatcher(
                self.deps,
                on_event=self._on_companion_event,
                is_busy=_busy,
                camera_worker=getattr(self.deps, "camera_worker", None),
            )
            self._companion.start()
            # The daemon sound library lives in /tmp (wiped on reboot) — re-check every start.
            self._chirp_upload_task = asyncio.create_task(asyncio.to_thread(ensure_chirps_uploaded))
            # Emotion .wav playback through the normal audio path (half-duplex + playback clock).
            self.deps.play_sound_path = self._play_wav_path
        except Exception:
            logger.warning("liveliness setup failed (continuing without)", exc_info=True)

    def _play_wav_path(self, path: str) -> None:
        """Queue a wav file (e.g. an emotion's bundled sound) on the spoken-audio path —
        respects the half-duplex mute and books the playback clock (speech=False)."""
        try:
            from reachy_mini_conversation_app.liveliness import wav_file_to_pcm

            loaded = wav_file_to_pcm(str(path))
            if loaded is None:
                return
            sr, pcm = loaded
            self.output_queue.put_nowait((sr, self._gain(pcm)))
            self._advance_playback_clock(len(pcm) / sr + 0.1, speech=False)
        except Exception:
            logger.debug("emotion sound playback failed", exc_info=True)

    async def _on_body_tool(self, action: str, params: dict) -> dict:
        """Gateway-requested body action (reachy_body tool) — bounded via body_surface allowlist."""
        from reachy_mini_conversation_app.body_surface import run_body_action

        return await run_body_action(self.deps, action, params, chirp=self._status_chirp)

    def _begin_event_turn(self, text: str) -> bool:
        """Start a turn from a LOCAL event (companion mode) — same single-flight discipline as a
        user transcript; refused while any turn is active."""
        if self._turn_active or self._closed:
            return False
        self._turn_active = True
        self._turn_seq += 1
        self._last_progress = time.monotonic()
        self._turn_task = asyncio.create_task(self.handle_final_transcript(text))
        return True

    def _on_companion_event(self, desc: str) -> None:
        from reachy_mini_conversation_app.companion import event_transcript

        if not self._begin_event_turn(event_transcript(desc)):
            logger.info("companion event dropped (busy): %s", desc)

    def _on_imu_event(self, kind: str) -> None:
        """Physical bump/lift detected: non-verbal startle (chirp + surprised emote)."""
        try:
            self._status_chirp("curious")
            from reachy_mini_conversation_app.tools.core_tools import dispatch_tool_call

            t = asyncio.create_task(dispatch_tool_call(self.deps, "play_emotion", {"emotion": "surprised"}))
            self._misc_tasks = getattr(self, "_misc_tasks", set())
            self._misc_tasks.add(t)
            t.add_done_callback(self._misc_tasks.discard)
            from reachy_mini_conversation_app.companion import CompanionWatcher, event_transcript

            if CompanionWatcher.enabled():
                self._begin_event_turn(event_transcript(
                    "Du wurdest gerade angestossen oder hochgehoben (IMU)."
                ))
        except Exception:
            logger.debug("IMU reaction failed", exc_info=True)

    def _sync_listening(self, listening: bool) -> None:
        """Mirror 'user is speaking' onto the body: antennas freeze + breathing pauses while
        listening (MovementManager.set_listening, debounced there). The info always existed in
        the handler (Silero in_speech) — it was just never forwarded (gap-map Stufe 1)."""
        if listening == getattr(self, "_listening_state", False):
            return
        self._listening_state = listening
        mm = getattr(self.deps, "movement_manager", None)
        setl = getattr(mm, "set_listening", None)
        if callable(setl):
            try:
                setl(listening)
            except Exception:
                logger.debug("set_listening failed", exc_info=True)
        # Orient toward the speaker on speech onset (gap-map Stufe 2): the mic array's DoA is
        # read once and, when the voice clearly comes from the side, a bounded head turn is
        # queued through the move_head tool. Cooldown keeps it from ping-ponging.
        if listening and self._env_on("AGENT_ORIENT_TO_SPEAKER", "1"):
            now = time.monotonic()
            if now - getattr(self, "_last_orient", 0.0) >= self._float_env("AGENT_ORIENT_COOLDOWN_S", 8.0):
                self._last_orient = now
                from reachy_mini_conversation_app.liveliness import orient_to_speaker

                t = asyncio.create_task(orient_to_speaker(self.deps))
                self._misc_tasks = getattr(self, "_misc_tasks", set())
                self._misc_tasks.add(t)
                t.add_done_callback(self._misc_tasks.discard)

    async def shutdown(self) -> None:
        self._closed = True
        for attr in ("_idle_runner", "_speech_sway", "_imu_watcher", "_companion"):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.stop()
                except Exception:
                    pass
        self._sync_listening(False)
        self._clear_output_queue()
        if self._spec_gate is not None:
            self._spec_gate.set()  # unblock any held speculative task so it can exit
        for task in (self._turn_task, self._spec_task):
            if task is not None and not task.done():
                task.cancel()
        # Close the transport client — without this a backend/persona restart leaks the platform
        # ws + reader task as a zombie session that silently swallows proactive deliveries
        # (audit 2026-07-02). Best-effort: HTTP client has no aclose and needs none.
        aclose = getattr(self.agent_client, "aclose", None)
        if callable(aclose):
            try:
                await aclose()
            except Exception:
                logger.warning("AGENT client aclose failed", exc_info=True)

    async def receive(self, frame: AudioFrame) -> None:
        # In-receive STT: feed the live mic (mono@16k, firmware-AEC'd) to Silero+Parakeet; on a
        # completed utterance, run a AGENT turn. handle_final_transcript serializes via _turn_lock.
        if self._closed:
            # Diagnostic: if the handler is closed but the record loop keeps feeding us, surface it
            # (a closed handler that never reopens = a deaf mic). Rate-limited.
            self._closed_drops = getattr(self, "_closed_drops", 0) + 1
            if self._closed_drops == 1 or self._closed_drops % 300 == 0:
                logger.warning("AGENT receive: _closed=True, dropping frames (n=%d)", self._closed_drops)
            return
        self._closed_drops = 0
        if not self._prewarmed:
            # Latency #2: warm OpenAI's shared 79k-prefix cache once, on the FIRST mic activity, so the
            # user's first real turn is less likely to hit a cold ~9-10s TTFT (diagnostic 2026-07-01).
            # Fire-and-forget, non-blocking, best-effort — never touches the turn path. Overlaps the
            # user's first utterance. AGENT_PREWARM=0 disables.
            self._prewarmed = True
            if self._env_on("AGENT_PREWARM"):
                t = asyncio.create_task(self._prewarm())
                self._misc_tasks = getattr(self, "_misc_tasks", set())
                self._misc_tasks.add(t)
                t.add_done_callback(self._misc_tasks.discard)
        if self._mic_muted:
            return  # dashboard mic toggle is OFF -> ignore the microphone
        now = time.monotonic()
        # Stall watchdog: measure time since the turn last made PROGRESS (queued audio / barge decision),
        # NOT since it started — a legitimate long reply is real-time-paced and can run for minutes, so a
        # start-based timeout would falsely kill it mid-answer. A genuinely wedged turn queues nothing.
        # Threshold: never below the dashboard-settable first-audio budget (+grace) — a 120s budget
        # with a fixed 30s stall would false-fire on every silent long think (audit 2026-07-02).
        stall_s = max(self._float_env("AGENT_TURN_STALL_S", 30.0),
                      self._float_env("AGENT_VOICE_FIRST_AUDIO_BUDGET_S", 20.0) + 15.0)
        if self._turn_active and self._last_progress and (now - self._last_progress) > stall_s:
            logger.warning("AGENT stall watchdog: no turn progress for %.0fs (barge_event=%s, pending=%r) -> cancel turn",
                           now - self._last_progress, self._barge_event.is_set(), self._pending_barge)
            # Actually CANCEL the wedged turn task — just resetting flags left it running and speaking,
            # while its finally later clobbered the next turn's _turn_active (audit 2026-07-02).
            task = self._turn_task
            if task is not None and not task.done():
                task.cancel()
            self._turn_seq += 1  # invalidate the cancelled turn's finally (generation guard)
            self._turn_active = False
            # keep_pending: a commit that raced the watchdog must survive — the cancelled turn's
            # finally picks it up as the next turn instead of dropping the user's command
            # (review 2026-07-02 round 2, P1-5c).
            self._reset_barge(keep_pending=True)
            self._speaking_until = 0.0
            # Tell the gateway to stop generating: the cancelled task can't (it never wakes from
            # its stream read) — without the /stop the gateway kept producing a whole answer that
            # was then dropped frame by frame (review 2026-07-02 round 2, P1-4b).
            itr = getattr(self.agent_client, "interrupt", None)
            if callable(itr):
                t = asyncio.create_task(self._safe_interrupt(itr))
                self._misc_tasks = getattr(self, "_misc_tasks", set())
                self._misc_tasks.add(t)
                t.add_done_callback(self._misc_tasks.discard)
        speaking = time.monotonic() < self._speaking_until
        if self._turn_active:
            # Non-pausing barge-in (Operator's design): throughout the turn, route mic frames to the
            # second VAD. AGENT keeps talking; awake the XVF3800 HW AEC keeps AGENT's own voice near the
            # noise floor, so the (louder) user transcribes cleanly in parallel and AGENT's echo neither
            # self-triggers nor endpoints. The decision (ignore / stop / commit) happens only once a
            # FULL user utterance endpoints — see _feed_barge/_classify_and_act.
            if self._barge_enabled:
                # The main front-end is not fed during a turn — if it was mid-utterance when the
                # turn started (proactive turn while the user spoke, VAD-triggered noise), close
                # its state/stream session. Left open, a stream-mode session held the Nemotron
                # server for the whole turn while the barge front-end opened a second one
                # (review 2026-07-02 round 2, P1-2). reset() is cheap since it aborts (no flush).
                if self._stt is not None and self._stt.in_speech:
                    self._stt.reset()
                await self._feed_barge(frame)
                return
            if self._stt is not None and self._stt.in_speech:
                self._stt.reset()
            self._muted = True
            return
        if speaking:
            # not in a turn but tail audio still draining -> hard half-duplex (no barge target)
            if self._stt is not None and self._stt.in_speech:
                self._stt.reset()
            self._muted = True
            return
        if getattr(self, "_muted", False):
            self._muted = False
            if self._stt is not None:
                self._stt.reset()  # discard any tail captured as the gate lifted
        frontend = None  # guard: _ensure_stt can throw before assignment (preempt path read it)
        try:
            sr, samples = frame
            raw0 = np.asarray(samples)
            self._rx_frames = getattr(self, "_rx_frames", 0) + 1
            if self._rx_frames == 1 or (os.getenv("AGENT_VOICE_DEBUG") and self._rx_frames % 30 == 0):
                fr = raw0.reshape(-1).astype(np.float64)
                logger.info(
                    "AGENT raw frame#%d sr=%s shape=%s dtype=%s min=%.4f max=%.4f std=%.4f",
                    self._rx_frames, sr, raw0.shape, raw0.dtype,
                    float(fr.min()) if fr.size else 0, float(fr.max()) if fr.size else 0,
                    float(fr.std()) if fr.size else 0,
                )
            # Daemon mic frames are stereo float32 (shape (N, 2)); collapse to mono by averaging
            # channels BEFORE flattening — a bare reshape(-1) interleaves L,R,L,R into a scrambled
            # "mono" stream that Silero scores at conf~0 (the bug that kept the VAD from triggering).
            if raw0.ndim == 2:
                raw = raw0.mean(axis=1)
            else:
                raw = raw0.reshape(-1)
            # Daemon mic frames are float32 in [-1, 1]; the STT front-end expects int16 PCM at the
            # ORIGINAL level (gain is applied inside the front-end for the VAD only, so Parakeet gets
            # clean audio). A bare .astype(int16) on floats truncates 0.0x -> 0 — must scale by 32768.
            if np.issubdtype(raw.dtype, np.floating):
                arr = np.clip(raw * 32768.0, -32768, 32767).astype(np.int16)
            else:
                arr = raw.astype(np.int16)
            # Offload to a thread: feed() may run a blocking remote Parakeet round-trip on endpoint,
            # which would otherwise freeze playback/mic/pacing on the single event-loop thread.
            # receive() is serialized per frame (record_loop awaits it), so the frontend stays single-threaded.
            frontend = self._ensure_stt()
            transcript = await asyncio.to_thread(frontend.feed, int(sr), arr)
            self._sync_listening(frontend.in_speech)  # antennas freeze while the user speaks
        except Exception as exc:
            self._rx_err = getattr(self, "_rx_err", 0) + 1
            if self._rx_err <= 3 or self._rx_err % 100 == 0:
                logger.warning("AGENT receive error #%d: %r", self._rx_err, exc)
            transcript = None
        # A speculation whose utterance endpointed to NOTHING (noise discard / empty STT) is never
        # adopted or discarded by the transcript path — it held _turn_lock at the gate and proactive
        # deliveries ran into their timeout until the next real transcript (review 2026-07-02, P3).
        if (self._spec_task is not None and not transcript and not self._turn_active
                and frontend is not None and not frontend.in_speech):
            logger.info("AGENT preempt: utterance endpointed empty -> discarding speculation")
            await self._discard_speculative()
        # Preemptive turn-start (Stage 2): while still speaking, if the STT partial is stable, begin
        # generating on it so the cloud TTFT overlaps the endpoint pause. Held behind _spec_gate.
        if (not transcript and not self._turn_active and self._spec_task is None
                and frontend is not None and self._env_on("AGENT_PREEMPT_TURN", "0")):
            sp = frontend.stable_partial(
                min_chars=int(self._float_env("AGENT_PREEMPT_MIN_CHARS", 15)),
                stable_frames=int(self._float_env("AGENT_PREEMPT_STABLE_FRAMES", 6)),
            )
            if sp:
                self._launch_speculative(sp)
        if transcript:
            logger.info("AGENT in-receive transcript: %r", transcript)
            if (self._spec_task is not None and not self._spec_task.done()
                    and self._spec_matches(transcript)):
                # ADOPT: the speculative turn's input matches the final -> it IS this turn; release
                # its held audio and don't re-generate (the TTFT head start is the win).
                logger.info("AGENT preempt: ADOPT speculative (final==partial %r)", transcript)
                self._turn_active = True
                self._turn_seq = self._spec_seq
                self._last_progress = time.monotonic()
                self._turn_task = self._spec_task
                self._spec_task = None
                self._spec_partial = None
                if self._spec_adopted is not None:
                    self._spec_adopted["v"] = True  # before the gate: the task IS the live turn
                if self._spec_gate is not None:
                    self._spec_gate.set()
            else:
                # No/mismatched speculation -> discard it and run a fresh turn on the real final.
                await self._discard_speculative()
                self._turn_active = True  # gate the mic for the whole turn (cleared in handle_final_transcript)
                self._turn_seq += 1
                self._last_progress = time.monotonic()
                # Keep the task reference: the stall watchdog cancels it, and a held reference
                # protects against event-loop GC of a running fire-and-forget task.
                self._turn_task = asyncio.create_task(self.handle_final_transcript(transcript))

    @staticmethod
    def _norm_text(s: str) -> str:
        import re

        return re.sub(r"[^\wäöüß]+", " ", (s or "").lower()).strip()

    def _spec_matches(self, final: str) -> bool:
        """Adopt the speculative turn ONLY if the final transcript equals the partial it started on
        (normalized) — i.e. the stable partial WAS the whole utterance and the trailing was just the
        endpoint pause. If the user added words, the speculation was on incomplete input -> discard."""
        return bool(self._spec_partial) and self._norm_text(final) == self._norm_text(self._spec_partial)

    def _launch_speculative(self, partial: str) -> None:
        self._spec_partial = partial
        self._spec_gate = asyncio.Event()
        # Per-task adopted marker: _discard_speculative cancels THEN opens the gate (to unblock a
        # held await), so gate.is_set() in the task's finally could not distinguish adopt from
        # discard — a discarded speculation ran the adopted-cleanup and cleared _turn_active mid
        # fresh turn (review 2026-07-02 round 2, P2).
        self._spec_adopted = {"v": False}
        self._spec_seq = self._turn_seq + 1
        logger.info("AGENT preempt: speculating on stable partial %r", partial)
        self._spec_task = asyncio.create_task(
            self._speculative_turn(partial, self._spec_seq, self._spec_gate, self._spec_adopted)
        )

    async def _discard_speculative(self) -> None:
        task, gate = self._spec_task, self._spec_gate
        self._spec_task = None
        self._spec_partial = None
        self._spec_gate = None
        if task is not None and not task.done():
            # Stop the gateway generating the wrong (partial-based) answer, then cancel the held task.
            interrupt = getattr(self.agent_client, "interrupt", None)
            if callable(interrupt):
                try:
                    await interrupt(None)  # /stop
                except Exception:
                    pass
            task.cancel()
        # Never leave a held task blocked on the gate.
        if gate is not None:
            gate.set()

    async def _speculative_turn(
        self, partial: str, my_seq: int, gate: asyncio.Event, adopted: dict
    ) -> None:
        """Generate the reply for a stable partial early (TTFT overlaps the endpoint pause), holding
        every spoken sentence behind ``gate`` until the real endpoint confirms (adopt) — confirm
        (~450ms) arrives well before the first token (~3-6s), so no wrong word is ever spoken. On
        adopt this task becomes the turn; on discard it is cancelled before the gate opens."""
        ask_stream = getattr(self.agent_client, "ask_stream", None)
        if ask_stream is None:
            return
        parts: list[str] = []
        spoke_any = False
        try:
            async with self._turn_lock:
                if self._closed:
                    return
                self._reset_barge(keep_pending=True)
                async for sentence in ask_stream(partial):
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    if not gate.is_set():
                        await gate.wait()  # hold audio until this turn is confirmed (adopted)
                    if self._closed or self._barge_event.is_set():
                        break
                    parts.append(sentence)
                    if await self._speak_sentence(sentence):
                        spoke_any = True
                    if self._barge_event.is_set():
                        if await self._maybe_platform_barge():
                            continue
                        break
                if parts:
                    self.output_queue.put_nowait(AdditionalOutputs({"role": "user", "content": partial}))
                    self.output_queue.put_nowait(AdditionalOutputs({"role": "assistant", "content": " ".join(parts)}))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("AGENT speculative turn failed", exc_info=True)
        finally:
            # Only if ADOPTED is this task the live turn -> do turn cleanup. (Not gate.is_set():
            # discard also opens the gate, purely to unblock a held await.)
            if adopted["v"]:
                pending = self._pending_barge
                self._pending_barge = None
                self._barge_event.clear()
                if self._barge_stt is not None:
                    self._barge_stt.reset()
                if pending and not self._closed:
                    self._turn_active = True
                    self._turn_seq += 1
                    self._last_progress = time.monotonic()
                    self._turn_task = asyncio.create_task(self.handle_final_transcript(pending))
                elif self._turn_seq == my_seq:
                    self._turn_active = False

    async def _feed_barge(self, frame: AudioFrame) -> None:
        """Feed one frame to the barge VAD while AGENT is talking. NON-PAUSING: AGENT keeps speaking; we
        do NOT react to onset. Only when a FULL user utterance endpoints do we classify it (off-thread)
        and decide ignore / stop / commit — so a backchannel never interrupts and a real interrupt
        carries its whole instruction. Awake the HW AEC keeps AGENT's own voice near the floor, so it
        neither self-triggers Silero nor endpoints as an utterance.
        """
        try:
            sr, samples = frame
            raw0 = np.asarray(samples)
            raw = raw0.mean(axis=1) if raw0.ndim == 2 else raw0.reshape(-1)
            if np.issubdtype(raw.dtype, np.floating):
                arr = np.clip(raw * 32768.0, -32768, 32767).astype(np.int16)
            else:
                arr = raw.astype(np.int16)
            fe = self._ensure_barge_stt()
            res = await asyncio.to_thread(fe.feed, int(sr), arr)  # blocking transcribe off the loop
            self._sync_listening(fe.in_speech)
        except Exception as exc:
            self._barge_err = getattr(self, "_barge_err", 0) + 1
            if self._barge_err <= 3:
                logger.warning("AGENT barge feed error #%d: %r", self._barge_err, exc)
            return
        if not (res and res.strip()) or self._barge_event.is_set():
            return
        txt = res.strip()
        # Silent think phase (no audio yet): don't DROP the utterance (a real correction like
        # "nein, nimm die andere Datei" vanished without a trace, review 2026-07-02 round 2, P2) —
        # classify it, and a commit is DEFERRED via _pending_barge: executed at first audio (see
        # _advance_playback_clock) or, if the turn errors out silently, as the next turn in the
        # finally. Cancelling before any audio stays forbidden (the no-answer loop), so stop in
        # silence still resolves to ignore.
        silent_phase = not self._turn_spoke
        if silent_phase and self._classify_inflight:
            return  # silent phase: no audible output to stop; wait for the running classify
        if self._classify_inflight:
            # A classify is already running — it may hang on a cold gate LLM for seconds. Never let a
            # bare STOP wait on it: decide it heuristically right here (audit 2026-07-02: repeated
            # "STOPP" was dropped by single-flight while the gate call hung).
            try:
                from reachy_agent.voice.semantic_gate import is_bare_stop
                if is_bare_stop(txt):
                    logger.info("AGENT barge: bare STOP during in-flight classify -> stop now")
                    self._barge_event.set()
                    self._clear_output_queue()
                    self._flush_downstream()
            except Exception:
                pass
            return
        logger.info("AGENT barge candidate (full utterance%s): %r", " — silent phase" if silent_phase else "", txt)
        self._classify_inflight = True  # single-flight: don't pile up classify threads/LLM calls
        asyncio.create_task(self._classify_and_act(txt, my_seq=self._turn_seq, silent_phase=silent_phase))

    async def _classify_and_act(self, transcript: str, my_seq: int | None = None, silent_phase: bool = False) -> None:
        """Classify a completed user utterance heard while AGENT was talking, then act:
        ignore -> AGENT keeps talking; stop -> stop AGENT, forward nothing; commit -> stop AGENT and
        forward the WHOLE transcript as the next turn. Runs concurrently with the speak loop; the LLM
        gate runs off the event loop. Stopping = set _barge_event (the speak loop bails) + drain queue.
        Always clears the single-flight flag (any path) so candidates can fire again.
        """
        try:
            if self._closed or self._barge_event.is_set() or not self._turn_active:
                return
            try:
                from reachy_agent.voice.semantic_gate import classify_interrupt
                # Short gate timeout: the library default of 20s makes AGENT un-interruptible for that
                # long when the 9B gate hangs (cold model load). On timeout the safe default (commit)
                # applies anyway, so waiting longer buys nothing (audit 2026-07-02).
                gate_timeout = self._float_env("AGENT_GATE_TIMEOUT_S", 5.0)
                decision = await asyncio.to_thread(classify_interrupt, transcript, timeout=gate_timeout)
            except Exception as exc:
                logger.warning("AGENT barge classify failed -> commit: %r", exc)
                decision = "commit"
            self._last_progress = time.monotonic()  # a decision is progress (don't let the watchdog fire)
            logger.info("AGENT barge %r -> %s%s", transcript, decision, " (silent phase)" if silent_phase else "")
            if decision == "ignore":
                return  # backchannel / not directed -> AGENT keeps talking at full volume
            # Generation guard: the gate can take seconds (5s timeout => default commit). If the
            # turn the candidate belonged to has ended meanwhile, a stale stop/commit would hit
            # the NEXT turn — or linger as a ghost _pending_barge (review 2026-07-02 round 2, P2).
            if not self._turn_active or (my_seq is not None and self._turn_seq != my_seq):
                logger.info("AGENT barge: decision %r arrived after its turn ended -> dropped", decision)
                return
            if self._barge_event.is_set():
                return  # another interrupt already stopped AGENT
            if silent_phase and not self._turn_spoke:
                # Still silent: a commit is deferred (first audio / turn-finally executes it);
                # a bare stop has nothing audible to stop — cancelling now would re-open the
                # no-answer loop, so let the turn deliver.
                if decision == "commit":
                    self._pending_barge = transcript
                    logger.info("AGENT barge: commit deferred until first audio")
                return
            # stop or commit: stop AGENT now — flush BOTH the handler queue and the daemon player
            # (play_loop drains the queue eagerly, so the audible backlog lives downstream; without
            # the player flush AGENT kept talking ~15% of the reply-so-far, audit 2026-07-02).
            self._barge_event.set()
            self._clear_output_queue()
            self._flush_downstream()
            if decision == "commit":
                self._pending_barge = transcript  # forward the whole instruction as the next turn
            # Propagate the interrupt to the gateway NOW (platform transport): during a silent
            # tool/think phase the speak loop is parked in its stream read and only notices the
            # event at the NEXT sentence — meanwhile the gateway kept generating, no /stop was
            # sent, and every further user utterance was dropped by the barge-event guard
            # (review 2026-07-02 round 2, P1-5). _maybe_platform_barge resets the barge state,
            # so the woken speak loop simply keeps streaming the interrupt turn (no double
            # interrupt); on the HTTP transport it is a no-op and the loop-side path stands.
            # (is_set guard: if the speak loop woke first and already dispatched, the event is
            # cleared and we must not fire a second interrupt — that would /stop the new turn.)
            if self._turn_active and not self._closed and self._barge_event.is_set():
                try:
                    await self._maybe_platform_barge()
                except Exception:
                    logger.warning("AGENT barge: immediate interrupt dispatch failed", exc_info=True)
        finally:
            self._classify_inflight = False

    async def _dispatch_barge_if_set(self) -> None:
        """Platform-interrupt dispatch, guarded: if the speak loop already handled the barge
        (event cleared), do nothing — a second interrupt would /stop the new turn."""
        if self._barge_event.is_set() and self._turn_active and not self._closed:
            try:
                await self._maybe_platform_barge()
            except Exception:
                logger.warning("AGENT barge: deferred interrupt dispatch failed", exc_info=True)

    async def _safe_interrupt(self, itr) -> None:
        try:
            await itr(None)
        except Exception:
            logger.warning("AGENT watchdog: interrupt dispatch failed", exc_info=True)

    def _note_turn_progress(self) -> None:
        """Inbound-frame progress hook (set on the platform client): ANY routed turn frame —
        say, typing, turn_end — proves the gateway is alive and working. Without it, a >35s
        silent tool phase produced no 'progress' and the stall watchdog killed a legitimate
        turn mid-answer (review 2026-07-02 round 2, P1-4)."""
        if self._turn_active:
            self._last_progress = time.monotonic()

    async def _maybe_platform_barge(self) -> bool:
        """Platform transport barge-in: interrupt the gateway's turn IN PLACE instead
        of tearing down the ask_stream and respawning (which, over a multiplexed ws,
        would let the cancelled turn's stragglers be spoken proactively).

        Returns True to CONTINUE the same ask_stream — a committed command makes the
        gateway cancel the current turn and answer the new one, and the client re-locks
        its accumulator onto it. Returns False to fall through to the caller's break:
        the HTTP transport (no ``interrupt``), or a bare stop (no command).
        """
        interrupt = getattr(self.agent_client, "interrupt", None)
        if not callable(interrupt):
            return False  # HTTP transport — caller breaks + respawns as before
        command = self._pending_barge
        self._reset_barge()  # clears _barge_event + _pending_barge, resets the barge STT
        try:
            await interrupt(command)  # command (commit) or None -> "/stop"
        except Exception:
            logger.warning("[platform barge] interrupt failed", exc_info=True)
        return bool(command)  # commit -> keep streaming the new turn; stop -> break

    async def emit(self) -> HandlerOutput:
        return await wait_for_item(self.output_queue)  # type: ignore[no-any-return]

    # "AGENT Work mode" safe toolset: api_server minus the tools that can run or schedule arbitrary
    # code/commands/file ops, directly OR indirectly. Dropped: terminal, code_execution, file
    # (direct); delegation, cronjob (indirect — a spawned agent / scheduled job could use the above).
    _WORK_MODE_TOOLSETS = ("browser,clarify,homeassistant,image_gen,memory,"
                           "messaging,moa,rl,session_search,skills,todo,tts,vision,web,workflow")

    async def apply_personality(self, profile: str | None) -> str:
        """AGENT stays the brain, but a profile can switch the GATEWAY toolset policy. Work mode drops
        the dangerous tools (terminal/code_execution/file); any other profile = full AGENT tools. Takes
        effect on the next turn (env read per turn), no restart.
        """
        self.second_assistant_detected = False
        name = (profile or "").strip().lower()
        if name in ("local-agent-work", "agent-workmodus", "agent_workmodus", "work-mode", "workmodus", "work", "work mode"):
            os.environ["AGENT_VOICE_TOOLSETS"] = self._WORK_MODE_TOOLSETS
            logger.info("Agent profile -> Work mode (no terminal/code/file)")
            return "Work mode aktiv: sichere Tools (kein Terminal, kein Code, kein Dateizugriff)."
        os.environ.pop("AGENT_VOICE_TOOLSETS", None)  # full tools
        logger.info("Agent profile -> %s (full tools)", name or "default")
        return f"Agent ({name or 'default'}): volle Tools aktiv."

    async def get_available_voices(self) -> list[str]:
        return [self._current_voice]

    def get_current_voice(self) -> str:
        return self._current_voice

    async def change_voice(self, voice: str) -> str:
        normalized = voice.strip() or self._current_voice
        self._current_voice = normalized
        set_voice = getattr(self.tts_client, "set_voice", None)
        if callable(set_voice):
            set_voice(normalized)
        return f"AGENT voice set to {self._current_voice}."

    async def _maybe_local_answer(self, transcript: str) -> str | None:
        """Latency #3: fully answer a clearly-trivial short turn with the local 9B (opt-in, conservative),
        or None to fall through to the full cloud brain. Best-effort — any failure returns None.
        """
        try:
            from reachy_mini_conversation_app.agent_clients import LocalReflexClient

            return await LocalReflexClient().answer(transcript)  # re-reads env (picks up dashboard toggle)
        except Exception:
            return None

    async def _prewarm(self) -> None:
        """One-shot, best-effort warm of the gateway/OpenAI shared 79k-prefix cache so the first real
        turn is less likely to be cold (~9-10s). Raw stateless HTTP with a trivial 1-token request —
        does NOT go through the client/turn path (no lock, no ws, no side effects). Any failure is
        swallowed; this can only help or no-op.
        """
        try:
            import httpx

            base = os.getenv("AGENT_BASE_URL", "http://127.0.0.1:8642/v1").rstrip("/")
            headers = {"Content-Type": "application/json"}
            key = os.getenv(os.getenv("AGENT_API_KEY_ENV", "API_SERVER_KEY"), "").strip()
            if key:
                headers["Authorization"] = f"Bearer {key}"
            payload = {
                "model": os.getenv("AGENT_MODEL", "local-agent"),
                "messages": [{"role": "user", "content": "Bereit?"}],
                "max_tokens": 1,
                "reasoning_effort": "minimal",
                "stream": False,
            }
            async with httpx.AsyncClient(timeout=30) as client:
                await client.post(f"{base}/chat/completions", json=payload, headers=headers)
            logger.info("AGENT prefix-cache pre-warm done")
        except Exception as e:
            logger.debug("AGENT pre-warm skipped: %s", e)

    async def _stream_with_lead_in(self, transcript: str, ask_stream):
        """Wrap the slow full-brain ask_stream with the conditional 9B *quick-take*.

        The 9B opener is fired in PARALLEL at t=0 so its latency overlaps the gateway. If the gateway's
        first real chunk has not arrived within AGENT_QUICKTAKE_DELAY_S, we speak an opener to mask the
        think-time: the 9B's bridge IF it already returned, else an instant static bridge — we never
        wait on the 9B, so the opener adds no latency of its own. Openers avoid "Moment" so they don't
        collide with a Phase-A tool-status line. On fast turns nothing extra is emitted. The underlying
        ask_stream task is never cancelled before its first chunk (cancelling would abort generation).
        """
        def _f(name: str, default: float) -> float:
            v = os.getenv(name)
            try:
                return float(v) if v else default
            except ValueError:
                return default

        from reachy_mini_conversation_app.agent_clients import _gap_filler, _static_quicktake

        quicktake_on = (
            self._lead_in_client is not None
            and os.getenv("AGENT_QUICKTAKE_ENABLED", "1").strip().lower() not in ("0", "false", "no", "off")
        )
        agen = ask_stream(transcript)
        lead_task = None
        first_task = None
        if quicktake_on:
            lead_task = asyncio.create_task(self._lead_in_client.lead_in(transcript))
        try:
            first_task = asyncio.ensure_future(agen.__anext__())
            delay = _f("AGENT_QUICKTAKE_DELAY_S", _f("AGENT_LEAD_IN_DELAY_S", 0.8))
            done, _pending = await asyncio.wait({first_task}, timeout=delay)
            if not done and quicktake_on and not self._barge_event.is_set():
                # Gateway still silent: take the 9B bridge only if it's ALREADY back; otherwise speak a
                # static bridge instantly. Guard against a disobedient 9B emitting "Moment" (Phase-A's word).
                bridge = ""
                if lead_task is not None and lead_task.done() and not lead_task.cancelled():
                    try:
                        bridge = (lead_task.result() or "").strip()
                    except Exception:
                        bridge = ""
                if not bridge or "moment" in bridge.lower():
                    bridge = _static_quicktake()
                if bridge and not self._barge_event.is_set():
                    yield bridge
                # Full-window mask: the cloud TTFT hole can be 4-10 s; the opener covers ~1 s. While the
                # brain is STILL silent, emit up to N varied content-free tail fillers at intervals so the
                # rest of the hole isn't dead air. Breaks the instant the first real chunk arrives (never
                # delays the answer). Content-free by design (ordered composition — never contradicts AGENT).
                gap_max = int(_f("AGENT_GAP_FILL_MAX", 2))
                gap_interval = _f("AGENT_GAP_FILL_INTERVAL_S", 2.2)
                used_fills = {bridge}
                fills = 0
                while (quicktake_on and fills < gap_max and not self._barge_event.is_set()
                       and not first_task.done()):
                    done2, _p2 = await asyncio.wait({first_task}, timeout=gap_interval)
                    if done2 or self._barge_event.is_set():
                        break  # first real chunk arrived (or barge) -> stop filling
                    filler = _gap_filler(exclude=used_fills)
                    if not filler:
                        break
                    used_fills.add(filler)
                    fills += 1
                    yield filler
            try:
                first = await first_task
            except StopAsyncIteration:
                first = None
            if first is not None and str(first).strip():
                yield first
            async for sentence in agen:
                yield sentence
        finally:
            # On an early break (barge) the async generator must be closed, else the inner ask_stream
            # httpx streaming connection leaks and first_task is orphaned ("Task destroyed but pending").
            if lead_task is not None and not lead_task.done():
                lead_task.cancel()
            if first_task is not None and not first_task.done():
                first_task.cancel()
                try:
                    await first_task
                except BaseException:
                    pass
            try:
                await agen.aclose()
            except BaseException:
                pass

    async def _maybe_scene_context(self, transcript: str) -> str | None:
        """Parallel vision: if the turn plausibly needs sight, fetch a 9B-VLM scene description
        (offloaded, time-boxed) and return it as labeled context for the brain (ordered composition —
        AGENT answers as the single voice using it). Best-effort: any failure returns None and the turn
        proceeds without sight. v1 fetches at transcript time; v2 will fire at speech onset to hide the
        VLM latency under the user's speech.
        """
        try:
            from reachy_agent.voice.vision_context import scene_context, is_visual_query
        except Exception:
            return None
        if not is_visual_query(transcript):
            return None
        frame = self._grab_frame()  # single chokepoint (BGR->RGB conversion happens there)
        if frame is None:
            return None
        try:
            from reachy_agent.body.vlm_client import smolvlm_describe
        except Exception:
            return None
        budget = self._float_env("AGENT_VISION_BUDGET_S", 2.0)

        def _describe(f: Any, p: str) -> str:
            return smolvlm_describe(f, p, timeout=budget)

        try:
            ctx = await asyncio.wait_for(
                asyncio.to_thread(scene_context, transcript, frame, _describe, force=True),
                timeout=budget + 0.3,
            )
            if ctx:
                logger.info("AGENT vision: scene context attached (%d chars)", len(ctx))
            return ctx
        except Exception as exc:
            logger.info("AGENT vision: scene context skipped (%s)", type(exc).__name__)
            return None

    def _grab_frame(self):
        """Latest camera frame from the worker as RGB, or None (best-effort).

        The SDK camera pipeline delivers BGR (caps video/x-raw,format=BGR) but the VLM PNG
        encoder assumes RGB — without the swap every colour answer was systematically wrong
        (red<->blue, audit 2026-07-02). Single chokepoint for all three vision paths.
        """
        cam = getattr(self.deps, "camera_worker", None)
        get_frame = getattr(cam, "get_latest_frame", None) if cam is not None else None
        if not callable(get_frame):
            return None
        try:
            frame = get_frame()
        except Exception:
            return None
        if frame is not None and getattr(frame, "ndim", 0) == 3 and frame.shape[2] == 3:
            frame = np.ascontiguousarray(frame[:, :, ::-1])  # BGR (SDK) -> RGB (encoder)
        return frame

    async def _frame_data_url(self) -> str | None:
        """Latest frame as a downscaled PNG data-URL for the native native-vision path (premium,
        complex visual reasoning). Best-effort -> None.
        """
        frame = self._grab_frame()
        if frame is None:
            return None
        try:
            from reachy_agent.body.vlm_client import _png_data_url
            return await asyncio.to_thread(_png_data_url, frame)
        except Exception:
            return None

    async def _video_scene_context(self, transcript: str) -> str | None:
        """Grab a short multi-frame clip and let GEMMA (only) describe what happens over time (Gemma
        3n does video); fold the description into the brain's prompt as labeled input. gemma-only —
        uses AGENT_VLM_BASE_URL/MODEL (the dedicated gemma vision endpoint).
        """
        n = max(1, int(self._float_env("AGENT_VIDEO_FRAMES", 6)))
        interval = self._float_env("AGENT_VIDEO_INTERVAL_S", 0.15)
        frames = []
        for i in range(n):
            fr = self._grab_frame()
            if fr is not None:
                frames.append(fr)
            if i < n - 1:
                await asyncio.sleep(interval)
        if not frames:
            return None
        try:
            from reachy_agent.body.vlm_client import describe_frames
        except Exception:
            return None
        budget = self._float_env("AGENT_VIDEO_BUDGET_S", 6.0)

        def _desc() -> str:
            return describe_frames(frames, transcript, timeout=budget)

        try:
            desc = await asyncio.wait_for(asyncio.to_thread(_desc), timeout=budget + 0.5)
        except Exception as exc:
            logger.info("AGENT vision: video context skipped (%s)", type(exc).__name__)
            return None
        desc = (desc or "").strip()
        if not desc:
            return None
        if len(desc) > 900:
            desc = desc[:900].rstrip() + "…"
        logger.info("AGENT vision: VIDEO context attached (%d frames, %d chars)", len(frames), len(desc))
        return f"[Was Reachys Kamera über die letzten Sekunden sieht (Video): {desc}]"

    async def handle_final_transcript(self, transcript: str) -> None:
        # Stream AGENT sentence-by-sentence and speak each as it arrives — this cuts time-to-first-audio
        # from "whole LLM reply + whole TTS" (~18s) to "first sentence + its TTS". Always release the
        # single-flight gate in finally; _speak_sentence keeps _speaking_until ahead of playback so the
        # mic stays muted continuously through the whole reply.
        my_seq = self._turn_seq  # generation guard: a stale/cancelled turn must not clobber a newer one
        try:
            clean_transcript = transcript.strip()
            if not clean_transcript or self._closed:
                return
            async with self._turn_lock:
                if self._closed:
                    return
                # keep_pending: don't discard a command the user committed against a proactive
                # message that spoke while this turn waited on the lock (audit 2026-07-02).
                self._reset_barge(keep_pending=True)  # fresh barge state per turn (no stale onset/transcript)
                self._turn_spoke = False  # barge suppressed until this turn produces audio
                self._sync_listening(False)  # utterance done — unfreeze antennas
                runner = getattr(self, "_idle_runner", None)
                if runner is not None:
                    runner.note_activity()
                self.output_queue.put_nowait(AdditionalOutputs({"role": "user", "content": clean_transcript}))
                # Latency #3: a clearly-trivial short turn can be fully answered by the local 9B in
                # ~300ms, removing the 3-6s cloud brain from the loop. Opt-in (AGENT_LOCAL_TRIVIAL=1) and
                # conservative (short turns only; the 9B ESCALATEs on any doubt) so it never dumbs down
                # the full brain. On escalate/miss it returns None fast (tight timeout) and we proceed.
                local = await self._maybe_local_answer(clean_transcript)
                if local and not self._barge_event.is_set():
                    await self._speak_sentence(local)
                    self.output_queue.put_nowait(AdditionalOutputs({"role": "assistant", "content": local}))
                    logger.info("AGENT local-reflex answered (skipped cloud brain)")
                    return
                # Substantive turn -> the cloud brain will be silent for a few seconds. Play a short
                # astromech "heard you, thinking" cue so the user knows it landed (replaces the
                # word-filler quick-take with a non-verbal signal).
                self._status_chirp("acknowledge")
                # Liveness hook for the stall watchdog: the platform client bumps this on every
                # routed turn frame (incl. typing during tool phases). Idempotent per turn.
                if hasattr(self.agent_client, "on_turn_progress"):
                    self.agent_client.on_turn_progress = self._note_turn_progress
                ask_stream = getattr(self.agent_client, "ask_stream", None)
                spoke_any = False
                parts: list[str] = []
                committed = False
                # Vision routing (Operator): VIDEO -> short gemma multi-frame clip; complex still ->
                # NATIVE vision (sees the pixels); simple still -> fast gemma DESCRIBE. Folded into the
                # brain as labeled INPUT / a native image — never a second spoken stream.
                scene = None
                image_url = None
                try:
                    from reachy_agent.voice.vision_context import vision_mode
                    mode = vision_mode(clean_transcript) if self._env_on("AGENT_VISION_ENABLED") else "none"
                except Exception:
                    mode = "none"
                if mode == "native":
                    image_url = await self._frame_data_url()
                    if image_url:
                        logger.info("AGENT vision: NATIVE path (the agent sees the frame)")
                elif mode == "video":
                    scene = await self._video_scene_context(clean_transcript)
                elif mode == "describe":
                    scene = await self._maybe_scene_context(clean_transcript)
                try:
                    if ask_stream is not None:
                        def _stream_fn(t: str, _ctx=scene, _img=image_url):
                            return ask_stream(t, _ctx, _img) if (_ctx or _img) else ask_stream(t)
                        async for sentence in self._stream_with_lead_in(clean_transcript, _stream_fn):
                            sentence = sentence.strip()
                            if not sentence:
                                continue
                            # A barge (stop/commit) may have fired from _classify_and_act -> stop here.
                            if self._barge_event.is_set():
                                if await self._maybe_platform_barge():
                                    continue  # platform: interrupted in-place, keep streaming the new turn
                                committed = self._pending_barge is not None
                                break
                            parts.append(sentence)
                            if await self._speak_sentence(sentence):
                                spoke_any = True
                            if self._closed:
                                break
                            if self._barge_event.is_set():
                                if await self._maybe_platform_barge():
                                    continue  # platform: interrupted in-place, keep streaming the new turn
                                committed = self._pending_barge is not None
                                break  # stopped mid/after a sentence; commit forwards the interrupt
                    else:  # non-streaming client (e.g. tests) — single shot
                        reply = await _resolve_text(
                            self.agent_client.ask(clean_transcript, scene, image_url)
                            if (scene or image_url) else self.agent_client.ask(clean_transcript)
                        )
                        parts.append(reply)
                        spoke_any = await self._speak_sentence(reply)
                except Exception:
                    logger.warning("AGENT ask/stream failed", exc_info=True)
                    if not parts:
                        self._status_chirp("error")  # non-verbal "uh-oh" before the spoken fallback
                        fallback = "Da hakt gerade die Verbindung zu AGENT."
                        try:
                            # actually SPEAK it — play_loop only logs AdditionalOutputs, so the
                            # user got total silence on a gateway outage (review 2026-07-02, P2)
                            await self._speak_sentence(fallback)
                        except Exception:
                            logger.warning("AGENT fallback TTS failed too", exc_info=True)
                        self.output_queue.put_nowait(
                            AdditionalOutputs({"role": "assistant", "content": fallback})
                        )
                        return
                if parts:
                    self.output_queue.put_nowait(AdditionalOutputs({"role": "assistant", "content": " ".join(parts)}))
                    self._maybe_turn_emote(clean_transcript, " ".join(parts))
                if parts and not spoke_any and not committed:
                    self.output_queue.put_nowait(
                        AdditionalOutputs(
                            {"role": "assistant", "content": "Ich habe die Antwort erzeugt, aber die Sprachausgabe hakt gerade."}
                        )
                    )
        finally:
            # A committed barge becomes the next turn directly (no new LISTEN). Hand off the
            # single-flight gate to that turn so the mic stays serialized. Always clear the barge
            # event/state here so a set event can never persist past a turn and deafen _feed_barge.
            pending = self._pending_barge
            self._pending_barge = None
            self._barge_event.clear()
            if self._barge_stt is not None:
                self._barge_stt.reset()
            if pending and not self._closed:
                self._turn_active = True
                self._turn_seq += 1
                self._last_progress = time.monotonic()
                self._turn_task = asyncio.create_task(self.handle_final_transcript(pending))
            elif self._turn_seq == my_seq:
                # Only clear the gate if no NEWER turn has been accepted meanwhile (the watchdog may
                # have cancelled us and a fresh turn already owns _turn_active, audit 2026-07-02).
                self._turn_active = False

    def _maybe_turn_emote(self, user_text: str, answer_text: str) -> None:
        """Fire a curated emotion matching the finished turn's affect (gap-map Stufe 2).
        Conservative: sparse keyword cues + cooldown — most turns end without an emote."""
        if not self._env_on("AGENT_TURN_EMOTES", "1"):
            return
        now = time.monotonic()
        if now - getattr(self, "_last_turn_emote", 0.0) < self._float_env("AGENT_TURN_EMOTE_COOLDOWN_S", 60.0):
            return
        try:
            from reachy_mini_conversation_app.emotion_cues import emotion_for_turn
            from reachy_mini_conversation_app.tools.core_tools import dispatch_tool_call

            intent = emotion_for_turn(user_text, answer_text)
            if not intent:
                return
            self._last_turn_emote = now
            logger.info("turn emote: %s", intent)
            t = asyncio.create_task(dispatch_tool_call(self.deps, "play_emotion", {"emotion": intent}))
            self._misc_tasks = getattr(self, "_misc_tasks", set())
            self._misc_tasks.add(t)
            t.add_done_callback(self._misc_tasks.discard)
        except Exception:
            logger.debug("turn emote failed", exc_info=True)

    async def _speak_sentence(self, text: str) -> bool:
        """Speak one sentence with low latency: stream PCM from the TTS as it is generated and queue
        ~0.5s real-time-paced segments to the speaker. Streaming (response_format=pcm) yields the first
        audio ~176ms after the request vs waiting for the whole-sentence WAV; pacing + segmenting keeps
        the GStreamer appsrc queue small (a whole blob overflows max-bytes). Falls back to the full-WAV
        synthesize() if the client can't stream. Returns True if any audio was queued.
        """
        seg = max(1, int(24000 * 0.5))
        stream_pcm = getattr(self.tts_client, "stream_pcm", None)
        spoke = False
        if self._barge_event.is_set():  # a barge fired before this sentence started -> don't speak it
            return False
        try:
            if stream_pcm is not None:
                buf = np.zeros(0, dtype=np.int16)
                sr = 24000
                async for csr, chunk in stream_pcm(text):
                    if self._barge_event.is_set():
                        return spoke  # interrupted mid-sentence: stop pulling/queuing immediately
                    sr = int(csr or 24000)
                    seg = max(1, int(sr * 0.5))
                    buf = np.concatenate([buf, np.asarray(chunk, dtype=np.int16)])
                    while len(buf) >= seg and not self._closed:
                        if self._barge_event.is_set():
                            return spoke
                        out, buf = buf[:seg], buf[seg:]
                        self.output_queue.put_nowait((sr, self._gain(out)))
                        spoke = True
                        self._sway_feed(sr, out)
                        self._advance_playback_clock(len(out) / float(sr))
                        await self._pace_playback()
                if len(buf) and not self._closed and not self._barge_event.is_set():
                    self.output_queue.put_nowait((sr, self._gain(buf)))
                    spoke = True
                    self._sway_feed(sr, buf)
                    self._advance_playback_clock(len(buf) / float(sr))
                    await self._pace_playback()
            else:
                out_sr, out_audio = await _resolve_audio_frame(self.tts_client.synthesize(text))
                out_sr = int(out_sr or 24000)
                seg = max(1, int(out_sr * 0.5))
                for start in range(0, len(out_audio), seg):
                    if self._closed or self._barge_event.is_set():
                        return spoke
                    s = out_audio[start : start + seg]
                    self.output_queue.put_nowait((out_sr, self._gain(s)))
                    spoke = True
                    self._sway_feed(out_sr, s)
                    self._advance_playback_clock(len(s) / float(out_sr))
                    await self._pace_playback()
        except Exception as exc:
            logger.warning("AGENT TTS failed (len=%d): %r", len(text), exc)
            return spoke
        # tail guard from the playback clock, not "now" (the old estimate drifted 15% of the reply)
        self._speaking_until = max(self._playback_cursor, time.monotonic()) + 0.8
        return spoke

    def _sway_feed(self, sr: int, pcm) -> None:
        """Hand a queued TTS segment to the antenna sway, stamped with the monotonic time it
        will actually start playing (the playback cursor before this segment is booked)."""
        sway = getattr(self, "_speech_sway", None)
        if sway is not None:
            try:
                sway.feed(int(sr), pcm, max(self._playback_cursor, time.monotonic()))
            except Exception:
                pass

    def _advance_playback_clock(self, seg_seconds: float, speech: bool = True) -> None:
        """Advance the monotonic playback cursor by one queued segment and derive _speaking_until
        from it. The cursor tracks when the LAST queued audio actually finishes playing (real-time),
        so the half-duplex gate stays closed through the whole audible reply — the old per-segment
        "now + const" estimate expired while several seconds of audio were still downstream.
        """
        now = time.monotonic()
        self._playback_cursor = max(self._playback_cursor, now) + seg_seconds
        self._speaking_until = self._playback_cursor + 0.8
        self._last_progress = now
        if not speech:
            return  # a status chirp is not spoken content: no _turn_spoke, no deferred commit
        first_audio = not self._turn_spoke
        self._turn_spoke = True  # AGENT is now audibly speaking -> barge is meaningful from here
        if first_audio and self._pending_barge and self._turn_active and not self._barge_event.is_set():
            # A commit was deferred during the silent think phase (review 2026-07-02 round 2, P2):
            # the user corrected themself before any audio. Execute it now — stop the just-started
            # audio and let the correction take over instead of speaking the superseded answer.
            logger.info("AGENT barge: executing deferred commit at first audio")
            self._barge_event.set()
            self._clear_output_queue()
            self._flush_downstream()
            t = asyncio.create_task(self._dispatch_barge_if_set())
            self._misc_tasks = getattr(self, "_misc_tasks", set())
            self._misc_tasks.add(t)
            t.add_done_callback(self._misc_tasks.discard)

    async def _pace_playback(self) -> None:
        """Keep at most ~AGENT_PLAYBACK_LEAD_S of audio queued downstream. Bounding the lead also
        bounds how much audio a barge-stop has to flush (it can only ever be ~the lead).
        """
        lead_target = self._float_env("AGENT_PLAYBACK_LEAD_S", 1.0)
        lead = self._playback_cursor - time.monotonic()
        if lead > lead_target:
            await asyncio.sleep(lead - lead_target)

    def _clear_output_queue(self) -> None:
        while not self.output_queue.empty():
            self.output_queue.get_nowait()


async def _resolve_text(value: MaybeText) -> str:
    resolved = await value if inspect.isawaitable(value) else value
    if not isinstance(resolved, str):
        raise TypeError(f"Expected AGENT client to return str, got {type(resolved).__name__}")
    return resolved


async def _resolve_audio_frame(value: MaybeAudioFrame) -> AudioFrame:
    resolved = await value if inspect.isawaitable(value) else value
    if not isinstance(resolved, tuple) or len(resolved) != 2:
        raise TypeError("Expected TTS client to return an AudioFrame tuple")
    sample_rate, audio = resolved
    if not isinstance(sample_rate, int):
        raise TypeError("Expected AudioFrame sample rate to be int")
    if not isinstance(audio, np.ndarray):
        raise TypeError("Expected AudioFrame audio to be a numpy ndarray")
    return sample_rate, audio.astype(np.int16, copy=False)
