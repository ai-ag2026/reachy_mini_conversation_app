from __future__ import annotations
import math
import inspect
import logging
from typing import Any, Dict

import numpy as np

from reachy_mini.utils import create_head_pose
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.dance_emotion_moves import GotoQueueMove
from reachy_mini_conversation_app.agent_movement_policy import plan_agent_movement


logger = logging.getLogger(__name__)

_SAFE_DELTAS = {
    "left": (0, 0, 0, 0, 0, 5),
    "right": (0, 0, 0, 0, 0, -5),
    "up": (0, 0, 0, 0, -4, 0),
    "down": (0, 0, 0, 0, 4, 0),
    "front": (0, 0, 0, 0, 0, 0),
}


class AgentSafeMovement(Tool):
    """Safely execute one curated v0.1 AGENT movement via official app seams."""

    name = "agent_safe_movement"
    description = "Safely perform one simple Reachy movement: look_left/right/up/down/front or stop_motion."
    needs_response = False
    parameters_schema = {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "enum": ["look_left", "look_right", "look_up", "look_down", "look_front", "stop_motion"],
                "description": "Curated v0.1 movement intent.",
            },
            "reason": {"type": "string", "description": "Short reason for the movement."},
        },
        "required": ["intent"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Plan and execute only curated movement through official primitives."""
        intent = (kwargs.get("intent") or "").strip().lower()
        reason = (kwargs.get("reason") or "").strip()
        plan = plan_agent_movement(intent, reason=reason)
        if plan.status != "planned":
            return {
                "status": "blocked",
                "intent": plan.intent,
                "selected_tool": plan.selected_tool,
                "tool_args": plan.tool_args,
                "side_effects": plan.side_effects,
                "requires_live_execute": plan.requires_live_execute,
                "reason": plan.reason,
            }

        logger.info("Tool call: agent_safe_movement intent=%s selected_tool=%s", plan.intent, plan.selected_tool)
        if plan.selected_tool == "clear_move_queue":
            clear_queue = getattr(deps.movement_manager, "clear_move_queue", None)
            if not callable(clear_queue):
                return {
                    "status": "error",
                    "intent": plan.intent,
                    "selected_tool": plan.selected_tool,
                    "tool_args": plan.tool_args,
                    "official_result": {"error": "movement manager cannot clear queue"},
                    "side_effects": [],
                    "requires_live_execute": False,
                }
            result = clear_queue()
            if inspect.isawaitable(result):
                await result
            return {
                "status": "executed",
                "intent": plan.intent,
                "selected_tool": plan.selected_tool,
                "tool_args": plan.tool_args,
                "official_result": {"status": "movement queue cleared"},
                "side_effects": ["movement_queue_cleared"],
                "requires_live_execute": plan.requires_live_execute,
            }

        if plan.selected_tool == "agent_safe_head_motion":
            try:
                official_result = _queue_safe_head_motion(deps, direction=plan.tool_args["direction"])
            except Exception:
                logger.warning("AGENT-safe head motion failed for intent=%s", plan.intent)
                return {
                    "status": "error",
                    "intent": plan.intent,
                    "selected_tool": plan.selected_tool,
                    "tool_args": plan.tool_args,
                    "official_result": {"error": "safe movement tool failed"},
                    "side_effects": ["possible_movement_queued"],
                    "requires_live_execute": plan.requires_live_execute,
                }
            return {
                "status": "executed",
                "intent": plan.intent,
                "selected_tool": plan.selected_tool,
                "tool_args": plan.tool_args,
                "official_result": official_result,
                "side_effects": ["movement_queued"],
                "requires_live_execute": plan.requires_live_execute,
            }

        return {
            "status": "blocked",
            "intent": plan.intent,
            "selected_tool": plan.selected_tool,
            "tool_args": plan.tool_args,
            "side_effects": [],
            "requires_live_execute": False,
            "reason": "unsupported movement tool",
        }


def _queue_safe_head_motion(deps: ToolDependencies, *, direction: str) -> Dict[str, Any]:
    """Queue a small curated head motion through the official MovementManager seam."""
    if direction not in _SAFE_DELTAS:
        raise ValueError("unsupported safe movement direction")
    current_head_pose = deps.reachy_mini.get_current_head_pose().astype("float32")
    current_body_yaw, current_antennas = deps.reachy_mini.get_current_joint_positions()
    start_body_yaw = _first_float(current_body_yaw)
    start_antennas = (float(current_antennas[0]), float(current_antennas[1]))
    # SSoT parity (body/safe_movement, review 2026-07-02 round 2, P3): a daemon glitch can hand
    # back a non-finite pose — queued unchecked it drove NaN through the interpolation into
    # set_target (IK ValueError spam for the whole move duration).
    if (
        not np.all(np.isfinite(current_head_pose))
        or not math.isfinite(start_body_yaw)
        or not all(math.isfinite(a) for a in start_antennas)
    ):
        logger.warning("agent_safe_movement: non-finite current pose -> abort queue")
        return {"status": "error", "error": "non-finite current pose", "side_effects": []}
    duration = deps.motion_duration_s
    if not math.isfinite(duration):
        duration = 0.3
    duration = min(max(float(duration), 0.15), 2.0)
    # `front` recenters to the absolute neutral head pose; a relative zero-delta would be a no-op
    # (it would re-target the current pose and never recenter). Other directions compose a bounded
    # relative delta onto the current pose. (AGENT P0.4 fix, 2026-06-24 audits.)
    if direction == "front":
        target = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True).astype("float32")
    else:
        delta = create_head_pose(*_SAFE_DELTAS[direction], degrees=True).astype("float32")
        target = np.matmul(current_head_pose, delta).astype("float32")
    if not np.all(np.isfinite(target)):
        logger.warning("agent_safe_movement: non-finite target pose -> abort queue")
        return {"status": "error", "error": "non-finite target pose", "side_effects": []}
    goto_move = GotoQueueMove(
        target_head_pose=target,
        start_head_pose=current_head_pose,
        target_antennas=start_antennas,
        start_antennas=start_antennas,
        target_body_yaw=start_body_yaw,
        start_body_yaw=start_body_yaw,
        duration=duration,
    )
    deps.movement_manager.queue_move(goto_move)
    deps.movement_manager.set_moving_state(duration)
    return {
        "status": f"queued safe {direction}",
        "bounded_degrees": {
            "x": _SAFE_DELTAS[direction][0],
            "y": _SAFE_DELTAS[direction][1],
            "z": _SAFE_DELTAS[direction][2],
            "roll": _SAFE_DELTAS[direction][3],
            "pitch": _SAFE_DELTAS[direction][4],
            "yaw": _SAFE_DELTAS[direction][5],
        },
        "duration_s": deps.motion_duration_s,
    }


def _first_float(value: Any) -> float:
    """Return the first scalar from SDK joint-position values."""
    try:
        return float(value[0])
    except (TypeError, IndexError):
        return float(value)
