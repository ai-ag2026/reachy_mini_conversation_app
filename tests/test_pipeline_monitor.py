from __future__ import annotations
import logging
import urllib.request

import pytest

from reachy_mini_conversation_app import pipeline_monitor
from reachy_mini_conversation_app.pipeline_monitor import PipelineMonitor


_LOGGER = "reachy_mini_conversation_app.pipeline_monitor"


def test_monitor_rejects_non_loopback_bind() -> None:
    """The monitor must never expose conversation data beyond loopback."""
    with pytest.raises(ValueError, match="loopback"):
        PipelineMonitor(host="0.0.0.0")


def test_monitor_records_event_with_metadata() -> None:
    """An emitted observation is retained verbatim, with None metadata dropped."""
    monitor = PipelineMonitor(port=0)
    monitor.emit("stt", "  Hello Reachy  ", language="en", unused=None)
    monitor.emit("llm", "   ")  # whitespace-only text is not an event
    assert len(monitor._history) == 1
    event = monitor._history[0]
    assert event.stage == "stt"
    assert event.text == "Hello Reachy"
    assert event.metadata == {"language": "en"}


def test_monitor_logs_metadata_not_content_by_default(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Transcripts and assistant text stay out of the logs unless explicitly enabled."""
    monkeypatch.delenv("AGENT_PIPELINE_MONITOR_LOG_CONTENT", raising=False)
    monitor = PipelineMonitor(port=0)
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        monitor.emit("stt", "my private plans", language="en")
    assert "my private plans" not in caplog.text
    assert "16 chars" in caplog.text
    assert "'language': 'en'" in caplog.text


def test_monitor_logs_content_when_opted_in(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """AGENT_PIPELINE_MONITOR_LOG_CONTENT=1 restores full-text log lines for debugging."""
    monkeypatch.setenv("AGENT_PIPELINE_MONITOR_LOG_CONTENT", "1")
    monitor = PipelineMonitor(port=0)
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        monitor.emit("llm", "It will rain later.")
    assert "It will rain later." in caplog.text


def test_monitor_serves_dashboard_on_loopback() -> None:
    """The dashboard page is reachable on the loopback port it was bound to."""
    monitor = PipelineMonitor(port=0)
    monitor.start()
    try:
        assert monitor.url.startswith("http://127.0.0.1:")
        with urllib.request.urlopen(monitor.url + "/", timeout=5) as response:
            assert response.status == 200
            assert "Reachy Live Pipeline" in response.read().decode("utf-8")
    finally:
        monitor.stop()


def test_get_pipeline_monitor_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """AGENT_PIPELINE_MONITOR=0 turns the process-wide monitor off entirely."""
    monkeypatch.setenv("AGENT_PIPELINE_MONITOR", "0")
    monkeypatch.setattr(pipeline_monitor, "_monitor", None)
    assert pipeline_monitor.get_pipeline_monitor() is None
