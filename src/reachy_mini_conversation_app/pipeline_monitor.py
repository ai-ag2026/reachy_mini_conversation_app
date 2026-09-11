"""Loopback-only live monitor for the local speech/agent pipeline."""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from dataclasses import asdict, dataclass
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineEvent:
    """One safe, user-visible pipeline observation."""

    sequence: int
    timestamp: float
    stage: str
    text: str
    metadata: dict[str, Any]


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Reachy Live Pipeline</title><style>
:root{color-scheme:dark;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:#101217;color:#eef1f7}
body{max-width:1050px;margin:0 auto;padding:24px}.top{display:flex;justify-content:space-between;gap:16px;align-items:center}
h1{font:600 22px system-ui;margin:0}.status{color:#8de1a8}.muted{color:#9299a8;font:13px system-ui}
#events{margin-top:20px;display:flex;flex-direction:column;gap:9px}.event{display:grid;grid-template-columns:92px 86px 1fr;
gap:12px;padding:12px 14px;background:#181b22;border:1px solid #292e39;border-radius:9px}.time{color:#9299a8}
.stage{font-weight:700}.stt .stage{color:#72c7ff}.llm .stage{color:#cda8ff}.tts .stage{color:#ffbe72}
.tool .stage{color:#80e5be}.system .stage,.latency .stage{color:#aab2c3}.text{white-space:pre-wrap;overflow-wrap:anywhere}
.meta{grid-column:3;color:#9299a8;font-size:12px;margin-top:3px}button{background:#292e39;color:#fff;border:0;border-radius:6px;padding:8px 12px}
</style></head><body><div class="top"><div><h1>Reachy Live Pipeline</h1><div class="muted">Final STT, streamed LLM output, TTS input, tools, and timing. Raw audio and private reasoning are not recorded.</div></div>
<div><span id="status" class="status">connecting</span> <button id="clear">Clear</button></div></div><div id="events"></div>
<script>
const events=document.getElementById('events'),status=document.getElementById('status');
document.getElementById('clear').onclick=()=>events.replaceChildren();
function add(e){const row=document.createElement('div');row.className='event '+e.stage;
 const t=document.createElement('div');t.className='time';t.textContent=new Date(e.timestamp*1000).toLocaleTimeString();
 const s=document.createElement('div');s.className='stage';s.textContent=e.stage.toUpperCase();
 const x=document.createElement('div');x.className='text';x.textContent=e.text;row.append(t,s,x);
 if(e.metadata&&Object.keys(e.metadata).length){const m=document.createElement('div');m.className='meta';m.textContent=JSON.stringify(e.metadata);row.append(m)}
 events.append(row);while(events.children.length>250)events.firstChild.remove();row.scrollIntoView({block:'end'});}
const es=new EventSource('/events');es.onopen=()=>status.textContent='live';es.onerror=()=>status.textContent='reconnecting';
es.onmessage=x=>add(JSON.parse(x.data));
</script></body></html>"""


class PipelineMonitor:
    """Fan out sanitized events to a small in-memory SSE dashboard."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8766, history_size: int = 200) -> None:
        """Configure a loopback dashboard and bounded event history."""
        if host not in {"127.0.0.1", "::1", "localhost"}:
            raise ValueError("Pipeline monitor must bind to a loopback address")
        self.host = "127.0.0.1" if host == "localhost" else host
        self.port = port
        self.history_size = history_size
        self._lock = threading.Lock()
        self._history: list[PipelineEvent] = []
        self._subscribers: set[queue.Queue[PipelineEvent]] = set()
        self._sequence = 0
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        """Return the local dashboard URL."""
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        """Start the dashboard server once in a daemon thread."""
        if self._server is not None:
            return
        monitor = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/":
                    body = _PAGE.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path == "/events":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("X-Accel-Buffering", "no")
                    self.end_headers()
                    subscriber: queue.Queue[PipelineEvent] = queue.Queue(maxsize=256)
                    with monitor._lock:
                        history = list(monitor._history)
                        monitor._subscribers.add(subscriber)
                    try:
                        for event in history:
                            self._write_event(event)
                        while True:
                            try:
                                event = subscriber.get(timeout=15)
                                self._write_event(event)
                            except queue.Empty:
                                self.wfile.write(b": keepalive\n\n")
                                self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    finally:
                        with monitor._lock:
                            monitor._subscribers.discard(subscriber)
                    return
                self.send_error(404)

            def _write_event(self, event: PipelineEvent) -> None:
                payload = json.dumps(asdict(event), ensure_ascii=False, separators=(",", ":"))
                self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()

            def log_message(self, _format: str, *_args: Any) -> None:
                return

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(target=self._server.serve_forever, name="pipeline-monitor", daemon=True)
        self._thread.start()
        logger.info("Reachy live pipeline monitor: %s", self.url)

    def emit(self, stage: str, text: str, **metadata: Any) -> None:
        """Publish one sanitized event and log the same structured observation."""
        clean_stage = stage.strip().lower() or "system"
        clean_text = str(text).strip()
        if not clean_text:
            return
        safe_metadata = {str(k): v for k, v in metadata.items() if v is not None}
        with self._lock:
            self._sequence += 1
            event = PipelineEvent(self._sequence, time.time(), clean_stage, clean_text, safe_metadata)
            self._history.append(event)
            del self._history[:-self.history_size]
            subscribers = list(self._subscribers)
        logger.info("PIPELINE %-7s | %s", clean_stage.upper(), escape(clean_text, quote=False))
        for subscriber in subscribers:
            try:
                subscriber.put_nowait(event)
            except queue.Full:
                try:
                    subscriber.get_nowait()
                    subscriber.put_nowait(event)
                except (queue.Empty, queue.Full):
                    pass

    def stop(self) -> None:
        """Stop the local HTTP server."""
        server, self._server = self._server, None
        if server is not None:
            server.shutdown()
            server.server_close()


_monitor: PipelineMonitor | None = None
_monitor_lock = threading.Lock()


def get_pipeline_monitor() -> PipelineMonitor | None:
    """Return the enabled process-wide monitor, starting it on first use."""
    enabled = os.getenv("AGENT_PIPELINE_MONITOR", "1").strip().lower() not in {"0", "false", "no", "off"}
    if not enabled:
        return None
    global _monitor
    with _monitor_lock:
        if _monitor is None:
            host = os.getenv("AGENT_PIPELINE_MONITOR_HOST", "127.0.0.1").strip()
            port = int(os.getenv("AGENT_PIPELINE_MONITOR_PORT", "8766"))
            _monitor = PipelineMonitor(host=host, port=port)
            try:
                _monitor.start()
            except OSError as exc:
                logger.warning("Could not start pipeline monitor on %s: %s", _monitor.url, exc)
                _monitor = None
    return _monitor
