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


@pytest.mark.parametrize("path", ["/", "/events"])
@pytest.mark.parametrize("host", ["attacker.example", "127.0.0.1.attacker.example", "localhost:1", ""])
def test_monitor_rejects_untrusted_host(path, host):
    """Reject rebinding requests before returning either HTML or transcript events."""
    import http.client

    monitor = PipelineMonitor(port=0)
    monitor.emit("stt", "private transcript")
    monitor.start()
    connection = http.client.HTTPConnection("127.0.0.1", monitor.port, timeout=3)
    try:
        connection.request("GET", path, headers={"Host": host})
        response = connection.getresponse()
        assert response.status == 403
        assert b"private transcript" not in response.read()
    finally:
        connection.close()
        monitor.stop()


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "[::1]"])
@pytest.mark.parametrize("with_port", [False, True])
def test_monitor_accepts_loopback_host(host, with_port):
    """Allow only loopback hostnames and the actual configured port."""
    import http.client

    monitor = PipelineMonitor(port=0)
    monitor.start()
    connection = http.client.HTTPConnection("127.0.0.1", monitor.port, timeout=3)
    try:
        connection.request("GET", "/", headers={"Host": host + (f":{monitor.port}" if with_port else "")})
        response = connection.getresponse()
        assert response.status == 200
        response.read()
    finally:
        connection.close()
        monitor.stop()


def test_monitor_serves_ipv6_loopback():
    """An explicit IPv6 bind serves the dashboard and produces a bracketed URL."""
    import socket
    import http.client

    if not socket.has_ipv6:
        pytest.skip("IPv6 is unavailable on this host")
    monitor = PipelineMonitor(host="::1", port=0)
    monitor.start()
    connection = http.client.HTTPConnection("::1", monitor.port, timeout=3)
    try:
        assert monitor.url == f"http://[::1]:{monitor.port}"
        connection.request("GET", "/")
        response = connection.getresponse()
        assert response.status == 200
        response.read()
    finally:
        connection.close()
        monitor.stop()
