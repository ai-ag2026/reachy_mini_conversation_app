import logging
from typing import Any, Dict, Tuple, Literal

from reachy_mini.utils import create_head_pose
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.dance_emotion_moves import GotoQueueMove


logger = logging.getLogger(__name__)

Direction = Literal["left", "right", "up", "down", "front"]


class MoveHead(Tool):
    """Move head in a given direction."""

    name = "move_head"
    description = "Move your head in a given direction: left, right, up, down or front."
    needs_response = False
    parameters_schema = {
        "type": "object",
        "properties": {
            "direction": {
                "type": "string",
                "enum": ["left", "right", "up", "down", "front"],
            },
        },
        "required": ["direction"],
    }

    # mapping: direction -> args for create_head_pose
    DELTAS: Dict[str, Tuple[int, int, int, int, int, int]] = {
        "left": (0, 0, 0, 0, 0, 40),
        "right": (0, 0, 0, 0, 0, -40),
        "up": (0, 0, 0, 0, -30, 0),
        "down": (0, 0, 0, 0, 30, 0),
        "front": (0, 0, 0, 0, 0, 0),
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Move head in a given direction."""
        direction_raw = kwargs.get("direction")
        if not isinstance(direction_raw, str):
            return {"error": "direction must be a string"}
        direction: Direction = direction_raw  # type: ignore[assignment]
        logger.info("Tool call: move_head direction=%s", direction)

        deltas = self.DELTAS.get(direction, self.DELTAS["front"])
        target = create_head_pose(*deltas, degrees=True)

        # Use new movement manager
        try:
            movement_manager = deps.movement_manager

            # Get current state for interpolation. get_current_joint_positions() returns
            # (body_yaw, antennas): the body_yaw is the FIRST value. The old code discarded it and
            # fed the LEFT-ANTENNA angle (~-0.17 rad) as start_body_yaw while targeting 0, snapping
            # the body ~10° on every call (and this tool fires autonomously via the idle policy).
            current_head_pose = deps.reachy_mini.get_current_head_pose()
            current_body_yaw, current_antennas = deps.reachy_mini.get_current_joint_positions()
            body_yaw = float(current_body_yaw[0]) if hasattr(current_body_yaw, "__len__") else float(current_body_yaw)
            antennas = (float(current_antennas[0]), float(current_antennas[1]))

            # Move only the head to the (absolute) direction pose; PRESERVE antennas and body_yaw —
            # a "move head" tool must not reset the body or antennas.
            goto_move = GotoQueueMove(
                target_head_pose=target,
                start_head_pose=current_head_pose,
                target_antennas=antennas,
                start_antennas=antennas,
                target_body_yaw=body_yaw,
                start_body_yaw=body_yaw,
                duration=deps.motion_duration_s,
            )

            movement_manager.queue_move(goto_move)
            movement_manager.set_moving_state(deps.motion_duration_s)

            return {"status": f"looking {direction}"}

        except Exception as e:
            logger.error("move_head failed")
            return {"error": f"move_head failed: {type(e).__name__}: {e}"}
