# ruff: noqa: D103
from __future__ import annotations

from reachy_mini_conversation_app.config import (
    LOCAL_BACKEND,
    get_backend_label,
    get_model_name_for_backend,
    get_default_voice_for_backend,
    get_available_voices_for_backend,
)


def test_local_backend_is_available_for_official_app_adapter() -> None:
    assert LOCAL_BACKEND == "local"
    assert get_model_name_for_backend("local") == "local-agent"
    assert get_backend_label("local") == "Local Agent"
    assert get_default_voice_for_backend("local") == "default"
    assert get_available_voices_for_backend("local") == ["default"]
