from __future__ import annotations
import os
import re
import asyncio
import logging
from typing import Any, Dict

import numpy as np

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

_PERSON_IDENTIFICATION_RE = re.compile(
    # OPT-IN legacy denylist (OFF by default — person ID is allowed; AGENT_VISION_BLOCK_PERSON_ID=1
    # re-blocks). Best-effort only, NOT a hard guarantee. Uses German compound-safe STEMS (no
    # trailing \b) so "Gesichtserkennung", "Erkennung", "Identifikation", "wiedererkennen" are
    # caught, plus strong identity phrasings. Keep identical to reachy_agent.body.vision.
    r"(\bwho\b|\bwer\b|person|persona|\bname\b|identit|gesicht|\bface\b|"
    r"recogni[sz]e|erkenn|identifi|"
    r"\bwie (heißt|heisst|alt)\b|\bhow old\b|\b(his|her) name\b)",
    re.IGNORECASE,
)


class AgentVision(Tool):
    """One-shot AGENT vision wrapper using the official camera/vision seams."""

    name = "agent_vision"
    description = (
        "Answer a safe one-shot question about the current camera frame without persisting or returning raw images."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "Safe question about visible non-identifying scene details.",
            },
        },
        "required": ["question"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Process one current frame with local vision and sanitized output."""
        question = (kwargs.get("question") or "").strip()
        question_chars = len(question)
        if not question:
            return _error("question must be a non-empty string", question_chars=question_chars)

        policy = _vision_policy(question)
        if policy["blocked"]:
            return {
                "status": "blocked",
                "error": "person identification is not allowed",
                "question_chars": question_chars,
                "image_persisted": False,
                "raw_image_returned": False,
                "policy": policy,
            }

        if deps.camera_worker is None:
            return _error("camera worker not available", question_chars=question_chars)
        if deps.vision_processor is None:
            return _error("local vision processor not available", question_chars=question_chars)

        frame = deps.camera_worker.get_latest_frame()
        if frame is None:
            return _error("no frame available", question_chars=question_chars)
        if not isinstance(frame, np.ndarray):
            return _error("camera frame is not a numpy array", question_chars=question_chars)

        logger.info("Tool call: agent_vision question_chars=%s frame_shape=%s", question_chars, list(frame.shape))
        try:
            vision_result = await asyncio.to_thread(deps.vision_processor.process_image, frame, question)
        except Exception:
            return _error("vision processing failed", question_chars=question_chars)
        if not isinstance(vision_result, str):
            return _error("vision returned non-string", question_chars=question_chars)

        return {
            "status": "ok",
            "image_description": vision_result,
            "question_chars": question_chars,
            "description_chars": len(vision_result),
            "frame_shape": list(frame.shape),
            "image_persisted": False,
            "raw_image_returned": False,
            "policy": policy,
        }


def _vision_policy(question: str) -> dict[str, Any]:
    # Person identification ALLOWED by default (local model, personal robot — Operator). Set
    # AGENT_VISION_BLOCK_PERSON_ID=1 to re-enable the legacy denylist.
    if os.getenv("AGENT_VISION_BLOCK_PERSON_ID", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ) and _PERSON_IDENTIFICATION_RE.search(question):
        return {"blocked": True, "reason": "person_identification"}
    return {"blocked": False, "reason": None}


def _error(message: str, *, question_chars: int) -> Dict[str, Any]:
    return {
        "status": "error",
        "error": message,
        "question_chars": question_chars,
        "image_persisted": False,
        "raw_image_returned": False,
    }
