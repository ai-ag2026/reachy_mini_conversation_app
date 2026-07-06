"""Tests for the AGENT one-shot vision wrapper tool."""

from __future__ import annotations
from unittest.mock import MagicMock

import numpy as np
import pytest

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.agent_vision import AgentVision


@pytest.mark.asyncio
async def test_agent_vision_uses_one_shot_frame_and_local_vision_processor() -> None:
    """AGENT vision should use one current frame and return sanitized metadata plus description."""
    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    camera_worker = MagicMock()
    camera_worker.get_latest_frame.return_value = frame
    vision_processor = MagicMock()
    vision_processor.process_image.return_value = "A red cube is on the table."
    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        camera_worker=camera_worker,
        vision_processor=vision_processor,
    )

    result = await AgentVision()(deps, question="What is visible?")

    assert result == {
        "status": "ok",
        "image_description": "A red cube is on the table.",
        "question_chars": len("What is visible?"),
        "description_chars": len("A red cube is on the table."),
        "frame_shape": [24, 32, 3],
        "image_persisted": False,
        "raw_image_returned": False,
        "policy": {"blocked": False, "reason": None},
    }
    camera_worker.get_latest_frame.assert_called_once_with()
    vision_processor.process_image.assert_called_once_with(frame, "What is visible?")


@pytest.mark.asyncio
async def test_agent_vision_blocks_person_identification_when_env_set(monkeypatch) -> None:
    """Person-ID is ALLOWED by default (local model, personal robot — Operator); the legacy denylist
    can be re-enabled via AGENT_VISION_BLOCK_PERSON_ID=1, which blocks before reading a frame."""
    monkeypatch.setenv("AGENT_VISION_BLOCK_PERSON_ID", "1")
    camera_worker = MagicMock()
    vision_processor = MagicMock()
    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        camera_worker=camera_worker,
        vision_processor=vision_processor,
    )

    result = await AgentVision()(deps, question="Who is this person?")

    assert result == {
        "status": "blocked",
        "error": "person identification is not allowed",
        "question_chars": len("Who is this person?"),
        "image_persisted": False,
        "raw_image_returned": False,
        "policy": {"blocked": True, "reason": "person_identification"},
    }
    camera_worker.get_latest_frame.assert_not_called()
    vision_processor.process_image.assert_not_called()


@pytest.mark.asyncio
async def test_agent_vision_requires_camera_worker_and_local_vision_processor() -> None:
    """AGENT wrapper should fail closed instead of returning raw base64 images."""
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock(), camera_worker=None, vision_processor=None)

    result = await AgentVision()(deps, question="What is visible?")

    assert result == {
        "status": "error",
        "error": "camera worker not available",
        "question_chars": len("What is visible?"),
        "image_persisted": False,
        "raw_image_returned": False,
    }


@pytest.mark.asyncio
async def test_agent_vision_returns_sanitized_error_when_processor_fails() -> None:
    """Vision processor exceptions should not leak raw internals or persist images."""
    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    camera_worker = MagicMock()
    camera_worker.get_latest_frame.return_value = frame
    vision_processor = MagicMock()
    vision_processor.process_image.side_effect = RuntimeError("secret local vision path")
    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        camera_worker=camera_worker,
        vision_processor=vision_processor,
    )

    result = await AgentVision()(deps, question="What is visible?")

    assert result == {
        "status": "error",
        "error": "vision processing failed",
        "question_chars": len("What is visible?"),
        "image_persisted": False,
        "raw_image_returned": False,
    }
    assert "secret" not in str(result)
