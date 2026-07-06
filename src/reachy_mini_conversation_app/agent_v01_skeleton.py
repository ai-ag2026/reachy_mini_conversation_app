# ruff: noqa: D101,D102,D103,D107
from __future__ import annotations
import json
from typing import Any, Protocol
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass


@dataclass(frozen=True)
class SkeletonInput:
    transcript: str
    vision_question: str
    movement_intent: str


class AgentClient(Protocol):
    session_id: str

    def ask(self, transcript: str) -> str: ...


class TtsClient(Protocol):
    def synthesize(self, text: str) -> bytes: ...


class VisionProcessor(Protocol):
    def process(self, question: str) -> str: ...


class MovementManager(Protocol):
    def execute(self, intent: str) -> None: ...


class FakeAgentClient:
    def __init__(self, reply: str, session_id: str = "fake-agent-session") -> None:
        self.reply = reply
        self.session_id = session_id
        self.calls: list[str] = []

    def ask(self, transcript: str) -> str:
        self.calls.append(transcript)
        return self.reply


class FakeTtsClient:
    def __init__(self, audio_bytes: bytes) -> None:
        self.audio_bytes = audio_bytes
        self.calls: list[str] = []

    def synthesize(self, text: str) -> bytes:
        self.calls.append(text)
        return self.audio_bytes


class FakeVisionProcessor:
    def __init__(self, result: str) -> None:
        self.result = result
        self.calls: list[str] = []

    def process(self, question: str) -> str:
        self.calls.append(question)
        return self.result


class FakeMovementManager:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def execute(self, intent: str) -> None:
        self.calls.append(intent)


_PERSON_IDENTIFICATION_MARKERS = (
    "identifiziere",
    "identifizieren",
    "wer ist",
    "who is",
    "identify",
    "person",
    "gesicht",
    "face",
)

_ALLOWED_MOVEMENT_INTENTS = {"look_left", "look_right", "look_up", "look_down", "look_front", "stop_motion"}


def run_no_side_effect_skeleton(
    skeleton_input: SkeletonInput,
    *,
    report_dir: Path,
    agent_client: AgentClient,
    tts_client: TtsClient,
    vision_processor: VisionProcessor,
    movement_manager: MovementManager,
) -> dict[str, Any]:
    report_dir.mkdir(parents=True, exist_ok=True)
    events: list[dict[str, Any]] = []

    def add_event(event_type: str, **fields: Any) -> None:
        events.append({"ts": _utc_now(), "type": event_type, **fields})

    add_event("skeleton_started", mode="no_side_effect_skeleton")
    add_event("voice_input_received", transcript_chars=len(skeleton_input.transcript))

    reply = agent_client.ask(skeleton_input.transcript)
    add_event("agent_response_completed", agent_response_chars=len(reply))

    audio = tts_client.synthesize(reply)
    add_event("tts_output_completed", audio_bytes=len(audio), speaker_side_effect=False)

    vision_blocked = _is_person_identification_request(skeleton_input.vision_question)
    add_event("vision_policy_checked", blocked=vision_blocked)
    if vision_blocked:
        vision = {
            "status": "blocked",
            "mode": "fake_frame",
            "question_chars": len(skeleton_input.vision_question),
            "result_chars": 0,
            "image_persisted": False,
            "person_identification_attempted": True,
            "continuous_capture": False,
            "tracking_enabled": False,
        }
        movement = _skipped_movement(skeleton_input.movement_intent)
        ok = False
    else:
        vision_result = vision_processor.process(skeleton_input.vision_question)
        vision = {
            "status": "ok",
            "mode": "fake_frame",
            "question_chars": len(skeleton_input.vision_question),
            "result_chars": len(vision_result),
            "image_persisted": False,
            "person_identification_attempted": False,
            "continuous_capture": False,
            "tracking_enabled": False,
        }
        add_event("vision_result_completed", result_chars=len(vision_result))
        movement = _execute_fake_movement(skeleton_input.movement_intent, movement_manager)
        add_event("movement_result_completed", intent=skeleton_input.movement_intent, status=movement["status"])
        ok = movement["status"] == "ok"

    report: dict[str, Any] = {
        "schema_version": "reachy_agent_v01_skeleton_report.v1",
        "mode": "no_side_effect_skeleton",
        "ok": ok,
        "created_at": _utc_now(),
        "route": "OFFICIAL_APP_ADAPTER_FIRST",
        "mvp_endpoint": "Reachy Mini v0.1 usable embodied AGENT with STT→TTS, vision, and movement",
        "identity": {
            "brain": "AGENT/Hermes",
            "body_surface": "official_reachy_mini_conversation_app",
            "second_assistant_detected": False,
            "agent_session_id_present": bool(getattr(agent_client, "session_id", "")),
        },
        "voice": {
            "status": "ok",
            "input_kind": "fake_transcript",
            "transcript_chars": len(skeleton_input.transcript),
            "agent_response_chars": len(reply),
            "tts_status": "bytes_generated" if audio else "skipped",
            "audio_bytes": len(audio),
            "speaker_side_effect": False,
        },
        "vision": vision,
        "movement": movement,
        "guarded_operations": {
            "used": [],
            "preflight_ok": None,
            "readback_ok": None,
            "cleanup_ok": None,
        },
        "hard_stop_compliance": {
            "public_push_pr": False,
            "secret_exposure": False,
            "destructive_data_loss": False,
            "internet_exposed_persistent_mic_camera": False,
            "unbounded_unsafe_robot_motion": False,
        },
        "verification": {
            "tests": [],
            "compile": None,
            "lint": None,
            "secret_like_hits": 0,
        },
    }
    add_event("skeleton_completed" if ok else "skeleton_failed", ok=ok)
    _write_report_files(report_dir, report, events)
    return report


def _execute_fake_movement(intent: str, movement_manager: MovementManager) -> dict[str, Any]:
    if intent not in _ALLOWED_MOVEMENT_INTENTS:
        return _blocked_movement(intent)
    movement_manager.execute(intent)
    return {
        "status": "ok",
        "mode": "fake_movement",
        "intent": intent,
        "executed_via_official_movement_manager": True,
        "direct_robot_post": False,
        "unsafe_motion_blocked": False,
        "side_effects": [],
    }


def _blocked_movement(intent: str) -> dict[str, Any]:
    return {
        "status": "blocked",
        "mode": "fake_movement",
        "intent": intent,
        "executed_via_official_movement_manager": False,
        "direct_robot_post": False,
        "unsafe_motion_blocked": True,
        "side_effects": [],
    }


def _skipped_movement(intent: str) -> dict[str, Any]:
    return {
        "status": "skipped",
        "mode": "skipped",
        "intent": intent,
        "executed_via_official_movement_manager": False,
        "direct_robot_post": False,
        "unsafe_motion_blocked": False,
        "side_effects": [],
    }


def _is_person_identification_request(text: str) -> bool:
    normalized = text.casefold()
    return any(marker in normalized for marker in _PERSON_IDENTIFICATION_MARKERS)


def _write_report_files(report_dir: Path, report: dict[str, Any], events: list[dict[str, Any]]) -> None:
    (report_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (report_dir / "events.jsonl").write_text(
        "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events), encoding="utf-8"
    )
    (report_dir / "artifacts.json").write_text(
        json.dumps({"stored_raw_audio": False, "stored_raw_images": False}, indent=2) + "\n", encoding="utf-8"
    )
    (report_dir / "REPORT.md").write_text(_format_markdown_report(report), encoding="utf-8")


def _format_markdown_report(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Reachy AGENT v0.1 Skeleton Report",
            "",
            f"- mode: `{report['mode']}`",
            f"- ok: `{str(report['ok']).lower()}`",
            f"- route: `{report['route']}`",
            f"- voice: `{report['voice']['status']}`",
            f"- vision: `{report['vision']['status']}`",
            f"- movement: `{report['movement']['status']}`",
            "- raw audio persisted: `false`",
            "- raw images persisted: `false`",
            "",
        ]
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
