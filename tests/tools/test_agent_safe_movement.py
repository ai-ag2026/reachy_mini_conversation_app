"""Tests for AGENT safe movement planning and tool delegation."""

from __future__ import annotations
import asyncio
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pytest

from reachy_mini.utils import create_head_pose
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.agent_movement_policy import plan_agent_movement
from reachy_mini_conversation_app.tools.agent_safe_movement import AgentSafeMovement


class RecordingMovementManager:  # noqa: D101
    """Minimal movement-manager seam recorder for AGENT movement tests."""

    def __init__(self) -> None:  # noqa: D107
        self.queued: list[object] = []
        self.moving_state: list[float] = []
        self.cleared = 0

    def queue_move(self, move: object) -> None:  # noqa: D102
        self.queued.append(move)

    def set_moving_state(self, duration: float) -> None:  # noqa: D102
        self.moving_state.append(duration)

    def clear_move_queue(self) -> None:  # noqa: D102
        self.cleared += 1


def _fake_robot() -> MagicMock:
    robot = MagicMock()
    robot.get_current_head_pose.return_value = np.eye(4, dtype=np.float32)
    robot.get_current_joint_positions.return_value = (np.array([0.2]), np.array([0.1, -0.1]))
    return robot


def test_plan_agent_movement_allows_only_curated_v01_intents() -> None:
    """Movement planning should expose only small curated v0.1 intents."""
    plan = plan_agent_movement("look_left", reason="acknowledge user")

    assert plan.status == "planned"
    assert plan.intent == "look_left"
    assert plan.selected_tool == "agent_safe_head_motion"
    assert plan.tool_args == {"direction": "left"}
    assert plan.reason == "acknowledge user"
    assert plan.side_effects == []
    assert plan.requires_live_execute is True


def test_plan_agent_movement_blocks_unknown_or_freeform_intents() -> None:
    """Unknown/freeform movement requests must fail closed."""
    plan = plan_agent_movement("dance wildly for ten seconds")

    assert plan.status == "blocked"
    assert plan.intent == "dance wildly for ten seconds"
    assert plan.selected_tool is None
    assert plan.tool_args == {}
    assert plan.side_effects == []
    assert plan.requires_live_execute is False


@pytest.mark.asyncio
async def test_agent_safe_movement_queues_bounded_head_motion_through_movement_manager() -> None:
    """Allowed look intents should queue small bounded movement through MovementManager."""
    movement_manager = RecordingMovementManager()
    deps = ToolDependencies(reachy_mini=_fake_robot(), movement_manager=movement_manager, motion_duration_s=0.35)

    result = await AgentSafeMovement()(deps, intent="look_left", reason="user asked")

    assert result == {
        "status": "executed",
        "intent": "look_left",
        "selected_tool": "agent_safe_head_motion",
        "tool_args": {"direction": "left"},
        "official_result": {
            "status": "queued safe left",
            "bounded_degrees": {"x": 0, "y": 0, "z": 0, "roll": 0, "pitch": 0, "yaw": 5},
            "duration_s": 0.35,
        },
        "side_effects": ["movement_queued"],
        "requires_live_execute": True,
    }
    assert len(movement_manager.queued) == 1
    queued = movement_manager.queued[0]
    assert type(queued).__name__ == "GotoQueueMove"
    assert getattr(queued, "duration") == 0.35
    assert movement_manager.moving_state == [0.35]
    assert movement_manager.cleared == 0


@pytest.mark.asyncio
async def test_agent_safe_movement_applies_delta_relative_to_current_pose_and_preserves_other_joints() -> None:
    """Safe movement should be bounded relative motion, not an absolute reset-to-neutral target."""
    movement_manager = RecordingMovementManager()
    robot = _fake_robot()
    current_pose = create_head_pose(0, 0, 0, 0, 0, 20, degrees=True).astype("float32")
    robot.get_current_head_pose.return_value = current_pose
    robot.get_current_joint_positions.return_value = (np.array([0.25]), np.array([0.3, -0.4]))
    deps = ToolDependencies(reachy_mini=robot, movement_manager=movement_manager, motion_duration_s=0.35)

    await AgentSafeMovement()(deps, intent="look_left", reason="user asked")

    queued = cast(Any, movement_manager.queued[0])
    expected_target = np.matmul(current_pose, create_head_pose(0, 0, 0, 0, 0, 5, degrees=True).astype("float32"))
    np.testing.assert_allclose(queued.target_head_pose, expected_target, rtol=1e-6, atol=1e-6)
    assert queued.start_antennas == (0.3, -0.4)
    assert queued.target_antennas == (0.3, -0.4)
    assert queued.start_body_yaw == 0.25
    assert queued.target_body_yaw == 0.25


@pytest.mark.asyncio
async def test_agent_safe_movement_blocks_unknown_without_touching_movement_manager() -> None:
    """Blocked movement requests must not touch movement manager or robot."""
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())

    result = await AgentSafeMovement()(deps, intent="follow that person", reason="unsafe")

    assert result == {
        "status": "blocked",
        "intent": "follow that person",
        "selected_tool": None,
        "tool_args": {},
        "side_effects": [],
        "requires_live_execute": False,
        "reason": "unsafe",
    }
    deps.movement_manager.queue_move.assert_not_called()
    deps.movement_manager.clear_move_queue.assert_not_called()


@pytest.mark.asyncio
async def test_agent_safe_movement_stop_motion_clears_official_queue() -> None:
    """Stop intent should use the official movement-manager queue clear seam."""
    movement_manager = MagicMock()
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=movement_manager)

    result = await AgentSafeMovement()(deps, intent="stop_motion", reason="user stop")

    assert result == {
        "status": "executed",
        "intent": "stop_motion",
        "selected_tool": "clear_move_queue",
        "tool_args": {},
        "official_result": {"status": "movement queue cleared"},
        "side_effects": ["movement_queue_cleared"],
        "requires_live_execute": True,
    }
    movement_manager.clear_move_queue.assert_called_once_with()


@pytest.mark.asyncio
async def test_agent_safe_movement_reports_safe_motion_error_with_sanitized_possible_side_effect() -> None:
    """Movement errors should be sanitized and report possible partial movement."""
    movement_manager = MagicMock()
    movement_manager.queue_move.side_effect = RuntimeError("secret motor path /tmp/private")
    deps = ToolDependencies(reachy_mini=_fake_robot(), movement_manager=movement_manager)

    result = await AgentSafeMovement()(deps, intent="look_left", reason="user asked")

    assert result == {
        "status": "error",
        "intent": "look_left",
        "selected_tool": "agent_safe_head_motion",
        "tool_args": {"direction": "left"},
        "official_result": {"error": "safe movement tool failed"},
        "side_effects": ["possible_movement_queued"],
        "requires_live_execute": True,
    }
    assert "secret" not in str(result)
    assert "/tmp/private" not in str(result)


@pytest.mark.asyncio
async def test_agent_safe_movement_stop_motion_supports_async_clear_queue() -> None:
    """Stop intent should support async movement manager seams."""
    calls = 0

    class AsyncMovementManager:
        async def clear_move_queue(self) -> None:
            nonlocal calls
            await asyncio.sleep(0)
            calls += 1

    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=AsyncMovementManager())

    result = await AgentSafeMovement()(deps, intent="stop_motion", reason="user stop")

    assert result["status"] == "executed"
    assert result["side_effects"] == ["movement_queue_cleared"]
    assert calls == 1
