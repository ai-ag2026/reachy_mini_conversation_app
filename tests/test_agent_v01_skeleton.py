# ruff: noqa: D103
from __future__ import annotations
from pathlib import Path

from reachy_mini_conversation_app.agent_v01_skeleton import (
    FakeTtsClient,
    SkeletonInput,
    FakeAgentClient,
    FakeMovementManager,
    FakeVisionProcessor,
    run_no_side_effect_skeleton,
)


def test_no_side_effect_skeleton_report_covers_voice_vision_and_movement(tmp_path: Path) -> None:
    report_dir = tmp_path / "report"

    transcript = "Hallo AGENT, schau kurz nach links und sag was du siehst."
    reply = "Ich sehe eine Testszene und schaue kurz nach links."
    vision_question = "Was ist vor dir zu sehen?"
    vision_result = "Testszene ohne Personenidentifikation."

    report = run_no_side_effect_skeleton(
        SkeletonInput(
            transcript=transcript,
            vision_question=vision_question,
            movement_intent="look_left",
        ),
        report_dir=report_dir,
        agent_client=FakeAgentClient(reply=reply),
        tts_client=FakeTtsClient(audio_bytes=b"RIFFfake-wave"),
        vision_processor=FakeVisionProcessor(result=vision_result),
        movement_manager=FakeMovementManager(),
    )

    assert report["schema_version"] == "reachy_agent_v01_skeleton_report.v1"
    assert report["mode"] == "no_side_effect_skeleton"
    assert report["ok"] is True
    assert report["identity"] == {
        "brain": "AGENT/Hermes",
        "body_surface": "official_reachy_mini_conversation_app",
        "second_assistant_detected": False,
        "agent_session_id_present": True,
    }

    assert report["voice"]["status"] == "ok"
    assert report["voice"]["input_kind"] == "fake_transcript"
    assert report["voice"]["transcript_chars"] == len(transcript)
    assert report["voice"]["agent_response_chars"] == len(reply)
    assert report["voice"]["tts_status"] == "bytes_generated"
    assert report["voice"]["speaker_side_effect"] is False

    assert report["vision"] == {
        "status": "ok",
        "mode": "fake_frame",
        "question_chars": len(vision_question),
        "result_chars": len(vision_result),
        "image_persisted": False,
        "person_identification_attempted": False,
        "continuous_capture": False,
        "tracking_enabled": False,
    }

    assert report["movement"]["status"] == "ok"
    assert report["movement"]["mode"] == "fake_movement"
    assert report["movement"]["intent"] == "look_left"
    assert report["movement"]["executed_via_official_movement_manager"] is True
    assert report["movement"]["direct_robot_post"] is False
    assert report["movement"]["side_effects"] == []

    assert report["hard_stop_compliance"] == {
        "public_push_pr": False,
        "secret_exposure": False,
        "destructive_data_loss": False,
        "internet_exposed_persistent_mic_camera": False,
        "unbounded_unsafe_robot_motion": False,
    }

    report_json = report_dir / "report.json"
    report_md = report_dir / "REPORT.md"
    events_jsonl = report_dir / "events.jsonl"
    assert report_json.exists()
    assert report_md.exists()
    assert events_jsonl.exists()

    persisted = report_json.read_text(encoding="utf-8")
    assert "Hallo AGENT" not in persisted
    assert "Ich sehe" not in persisted
    assert "RIFFfake-wave" not in persisted
    assert "base64" not in persisted.lower()

    events = events_jsonl.read_text(encoding="utf-8").splitlines()
    assert any('"type": "voice_input_received"' in line for line in events)
    assert any('"type": "vision_result_completed"' in line for line in events)
    assert any('"type": "movement_result_completed"' in line for line in events)
    assert any('"type": "skeleton_completed"' in line for line in events)


def test_no_side_effect_skeleton_blocks_person_identification_before_camera_access(tmp_path: Path) -> None:
    vision_processor = FakeVisionProcessor(result="should not be called")

    report = run_no_side_effect_skeleton(
        SkeletonInput(
            transcript="Wer ist diese Person?",
            vision_question="Identifiziere die Person vor dir.",
            movement_intent="look_front",
        ),
        report_dir=tmp_path / "report",
        agent_client=FakeAgentClient(reply="Das mache ich nicht."),
        tts_client=FakeTtsClient(audio_bytes=b"fake"),
        vision_processor=vision_processor,
        movement_manager=FakeMovementManager(),
    )

    assert report["ok"] is False
    assert report["vision"]["status"] == "blocked"
    assert report["vision"]["person_identification_attempted"] is True
    assert vision_processor.calls == []
    assert report["movement"]["status"] == "skipped"
