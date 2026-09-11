"""Client side of the body-tool surface (gap-map Stufe 3, 2026-07-02).

The gateway's ``reachy_body`` tool pushes ``tool_call`` frames over the platform WebSocket;
this module maps them onto the app's OWN expressiveness layer — MovementManager queue moves,
the curated emotion/dance tools, the camera worker toggle, the chirp palette. Everything runs
through the same bounded seams the app itself uses (the MovementManager stays the single
set_target caller); anything outside the allowlist is rejected.
"""

from __future__ import annotations
import logging
from typing import Any, Callable


logger = logging.getLogger(__name__)

ALLOWED_ACTIONS = ("emote", "dance", "look", "stop", "head_tracking", "chirp")
_ALLOWED_DIRECTIONS = ("left", "right", "up", "down", "front")


async def run_body_action(
    deps: Any,
    action: str,
    params: dict[str, Any],
    *,
    chirp: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Execute one gateway-requested body action; always returns a result dict."""
    action = (action or "").strip().lower()
    if action not in ALLOWED_ACTIONS:
        return {"error": f"action {action!r} not allowed", "allowed": list(ALLOWED_ACTIONS)}
    try:
        from reachy_mini_conversation_app.tools.core_tools import dispatch_tool_call_obj

        if action == "emote":
            emotion = str(params.get("emotion") or "random")
            return await dispatch_tool_call_obj("play_emotion", {"emotion": emotion}, deps)

        if action == "dance":
            return await dispatch_tool_call_obj("dance", {}, deps)

        if action == "look":
            direction = str(params.get("direction") or "front").strip().lower()
            if direction not in _ALLOWED_DIRECTIONS:
                return {"error": f"direction {direction!r} not allowed"}
            return await dispatch_tool_call_obj("move_head", {"direction": direction}, deps)

        if action == "stop":
            mm = getattr(deps, "movement_manager", None)
            if mm is None:
                return {"error": "no movement manager"}
            mm.clear_move_queue()
            return {"status": "stopped", "side_effects": ["move_queue_cleared"]}

        if action == "head_tracking":
            cam = getattr(deps, "camera_worker", None)
            setter = getattr(cam, "set_head_tracking_enabled", None)
            if not callable(setter):
                return {"error": "head tracking unavailable (no camera worker / tracker)"}
            enabled = bool(params.get("enabled", True))
            setter(enabled)
            return {"status": "ok", "head_tracking": enabled}

        if action == "chirp":
            name = str(params.get("name") or "notify")
            if chirp is None:
                return {"error": "chirp seam unavailable"}
            chirp(name)
            return {"status": "chirped", "name": name}
    except Exception as exc:
        logger.warning("body action %s failed", action, exc_info=True)
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"error": "unreachable"}
