from __future__ import annotations
from enum import StrEnum
from typing import Any
from dataclasses import dataclass


class AgentMovementIntent(StrEnum):
    """Curated v0.1 movement intents exposed to AGENT."""

    LOOK_LEFT = "look_left"
    LOOK_RIGHT = "look_right"
    LOOK_UP = "look_up"
    LOOK_DOWN = "look_down"
    LOOK_FRONT = "look_front"
    STOP_MOTION = "stop_motion"


@dataclass(frozen=True)
class AgentMovementPlan:
    """Side-effect-free movement plan before official tool execution."""

    status: str
    intent: str
    selected_tool: str | None
    tool_args: dict[str, Any]
    reason: str
    side_effects: list[str]
    requires_live_execute: bool


_INTENT_TO_DIRECTION = {
    AgentMovementIntent.LOOK_LEFT: "left",
    AgentMovementIntent.LOOK_RIGHT: "right",
    AgentMovementIntent.LOOK_UP: "up",
    AgentMovementIntent.LOOK_DOWN: "down",
    AgentMovementIntent.LOOK_FRONT: "front",
}


def plan_agent_movement(intent: str, *, source: str = "local", reason: str = "") -> AgentMovementPlan:
    """Normalize a requested movement intent into the curated official-app execution seam."""
    _ = source
    normalized = (intent or "").strip().lower()
    try:
        parsed_intent = AgentMovementIntent(normalized)
    except ValueError:
        return AgentMovementPlan(
            status="blocked",
            intent=normalized,
            selected_tool=None,
            tool_args={},
            reason=reason,
            side_effects=[],
            requires_live_execute=False,
        )

    if parsed_intent == AgentMovementIntent.STOP_MOTION:
        return AgentMovementPlan(
            status="planned",
            intent=parsed_intent.value,
            selected_tool="clear_move_queue",
            tool_args={},
            reason=reason,
            side_effects=[],
            requires_live_execute=True,
        )

    return AgentMovementPlan(
        status="planned",
        intent=parsed_intent.value,
        selected_tool="agent_safe_head_motion",
        tool_args={"direction": _INTENT_TO_DIRECTION[parsed_intent]},
        reason=reason,
        side_effects=[],
        requires_live_execute=True,
    )
