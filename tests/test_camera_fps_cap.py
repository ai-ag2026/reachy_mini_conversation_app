"""Camera framerate cap (AGENT_CAMERA_MAX_FPS) — real GStreamer pipeline surgery."""

from __future__ import annotations

import pytest

gi = pytest.importorskip("gi")
gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

from reachy_mini_conversation_app.camera_worker import CAMERA_FPS_FLOOR, cap_camera_framerate  # noqa: E402

Gst.init([])


class _FakeCam:
    """Duck-typed stand-in for the SDK's GStreamerCamera (source->convert->appsink)."""

    def __init__(self, native_fps: int = 30) -> None:
        self._fps = native_fps
        self.pipeline = Gst.Pipeline.new("test_cam")
        self._src = Gst.ElementFactory.make("videotestsrc")
        self._convert = Gst.ElementFactory.make("videoconvert")
        self._appsink_video = Gst.ElementFactory.make("appsink")
        self._appsink_video.set_property(
            "caps",
            Gst.Caps.from_string(f"video/x-raw,format=BGR,width=64,height=48,framerate={native_fps}/1"),
        )
        for el in (self._src, self._convert, self._appsink_video):
            self.pipeline.add(el)
        self._src.link(self._convert)
        self._convert.link(self._appsink_video)
        self.closed = False
        self.opened = False

    @property
    def framerate(self) -> int:
        return self._fps

    @property
    def resolution(self) -> tuple[int, int]:
        return (64, 48)

    def close(self) -> None:
        self.closed = True

    def open(self) -> None:
        self.opened = True


class _FakeMedia:
    def __init__(self, cam) -> None:
        self.camera = cam


class _FakeMini:
    def __init__(self, cam) -> None:
        self.media = _FakeMedia(cam)


def _pipeline_element_names(pipeline) -> list[str]:
    it = pipeline.iterate_elements()
    names = []
    while True:
        ok, el = it.next()
        if ok != Gst.IteratorResult.OK:
            break
        names.append(el.get_factory().get_name())
    return names


def _iterate(pipeline):
    it = pipeline.iterate_elements()
    while True:
        ok, el = it.next()
        if ok != Gst.IteratorResult.OK:
            return
        yield el


def _appsink_fps(cam) -> int:
    caps = cam._appsink_video.get_property("caps")
    fps = caps.get_structure(0).get_fraction("framerate")
    return int(fps.value_numerator / fps.value_denominator)


def test_cap_inserts_droponly_videorate_and_recaps_appsink():
    cam = _FakeCam(native_fps=30)
    assert cap_camera_framerate(_FakeMini(cam), 15) is True
    assert "videorate" in _pipeline_element_names(cam.pipeline)
    assert _appsink_fps(cam) == 15
    vr = [el for el in _iterate(cam.pipeline) if el.get_factory().get_name() == "videorate"][0]
    assert vr.get_property("drop-only") is True
    assert vr.get_property("max-rate") == 15


def test_cap_enforces_15fps_floor():
    """Never throttle below CAMERA_FPS_FLOOR, even for a lower requested cap."""
    cam = _FakeCam(native_fps=30)
    assert cap_camera_framerate(_FakeMini(cam), 5) is True
    assert _appsink_fps(cam) == CAMERA_FPS_FLOOR == 15


def test_cap_noop_when_native_at_or_below_cap():
    cam = _FakeCam(native_fps=15)
    assert cap_camera_framerate(_FakeMini(cam), 15) is True
    assert "videorate" not in _pipeline_element_names(cam.pipeline)


def test_cap_fails_soft_without_camera():
    class _NoCamMini:
        media = None

    assert cap_camera_framerate(_NoCamMini(), 15) is False
