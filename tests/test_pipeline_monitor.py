from __future__ import annotations

import pytest

from reachy_mini_conversation_app.pipeline_monitor import PipelineMonitor


def test_monitor_rejects_non_loopback_bind() -> None:
    """The monitor must never expose conversation data beyond loopback."""
    with pytest.raises(ValueError, match="loopback"):
        PipelineMonitor(host="0.0.0.0")


def test_monitor_records_sanitized_event() -> None:
    """An emitted observation is retained with its safe metadata."""
    monitor = PipelineMonitor(port=0)
    monitor.emit("stt", "Hello Reachy", language="en")
    assert len(monitor._history) == 1
    event = monitor._history[0]
    assert event.stage == "stt"
    assert event.text == "Hello Reachy"
    assert event.metadata == {"language": "en"}
