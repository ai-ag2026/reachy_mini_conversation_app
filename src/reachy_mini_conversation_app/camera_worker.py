"""Camera worker thread with frame buffering and optional head tracking."""

import time
import logging
import threading
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from reachy_mini import ReachyMini
from reachy_mini.utils.interpolation import linear_pose_interpolation
from reachy_mini_conversation_app.vision.head_tracking import HeadTracker


logger = logging.getLogger(__name__)

# Hard floor for the camera-framerate cap: never throttle below this rate.
CAMERA_FPS_FLOOR = 15


def cap_camera_framerate(reachy_mini: ReachyMini, max_fps: int) -> bool:
    """Cap the SDK's client-side camera pipeline to ``max_fps`` (never below ``CAMERA_FPS_FLOOR``).

    The daemon streams at its native rate; the SDK client pipeline (unixfdsrc -> queue ->
    convert -> appsink) converts EVERY frame and get_frame() paces the CameraWorker (and thus
    any downstream tracker) at that rate. When there is no enum resolution between the native
    rate and the desired cap and no daemon-side API for it, we insert a drop-only ``videorate``
    before the appsink and re-cap its caps — frames are dropped before conversion/tracking, the
    daemon side stays untouched. Uses the same close->modify->open cycle as the SDK's own
    ``_apply_resolution``. Best-effort: returns False and leaves the native rate on any failure.
    Run ONCE at app start (repeated media surgery is what crashes the producer)."""
    max_fps = max(CAMERA_FPS_FLOOR, int(max_fps))
    try:
        from gi.repository import Gst

        cam = getattr(getattr(reachy_mini, "media", None), "camera", None)
        appsink = getattr(cam, "_appsink_video", None)
        pipeline = getattr(cam, "pipeline", None)
        if cam is None or appsink is None or pipeline is None:
            logger.info("camera fps cap skipped (no gstreamer client camera)")
            return False
        native = int(cam.framerate)
        if native <= max_fps:
            logger.info("camera fps cap skipped (native %d <= cap %d)", native, max_fps)
            return True

        was_playing = pipeline.get_state(0).state == Gst.State.PLAYING
        if was_playing:
            cam.close()

        upstream = appsink.get_static_pad("sink").get_peer().get_parent_element()
        videorate = Gst.ElementFactory.make("videorate")
        if videorate is None:
            raise RuntimeError("videorate element unavailable")
        videorate.set_property("drop-only", True)
        videorate.set_property("max-rate", max_fps)

        upstream.unlink(appsink)
        pipeline.add(videorate)
        upstream.link(videorate)
        width, height = cam.resolution
        appsink.set_property(
            "caps",
            Gst.Caps.from_string(
                f"video/x-raw,format=BGR,width={width},height={height},framerate={max_fps}/1"
            ),
        )
        videorate.link(appsink)

        if was_playing:
            cam.open()
        logger.info("camera fps capped: %d -> %d (drop-only videorate)", native, max_fps)
        return True
    except Exception:
        logger.warning("camera fps cap failed — staying at native rate", exc_info=True)
        return False


class CameraWorker:
    """Thread-safe camera worker with frame buffering and optional head tracking."""

    def __init__(self, reachy_mini: ReachyMini, head_tracker: HeadTracker | None = None) -> None:
        """Initialize."""
        self.reachy_mini = reachy_mini
        self.head_tracker = head_tracker

        self.latest_frame: NDArray[np.uint8] | None = None
        self.latest_frame_ts: float = 0.0  # monotonic time of the last fresh frame
        self.frame_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self.is_head_tracking_enabled = True
        self.face_tracking_offsets: List[float] = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]  # x, y, z, roll, pitch, yaw
        self.face_tracking_lock = threading.Lock()

        self.last_face_detected_time: float | None = None
        self.interpolation_start_time: float | None = None
        self.interpolation_start_pose: NDArray[np.float32] | None = None
        self.face_lost_delay = 2.0
        self.interpolation_duration = 1.0

        self.previous_head_tracking_state = self.is_head_tracking_enabled

    def get_latest_frame(self, max_age_s: float = 5.0) -> NDArray[np.uint8] | None:
        """Get the latest frame (thread-safe), or None if it is older than ``max_age_s``.

        Staleness gate (audit 2026-07-02): after a camera/producer stall the last frame froze
        forever and vision answered "what I see right now" from an arbitrarily old scene.
        ``max_age_s=0`` disables the check.
        """
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            if max_age_s > 0 and self.latest_frame_ts and (time.monotonic() - self.latest_frame_ts) > max_age_s:
                return None
            return self.latest_frame.copy()

    def get_face_tracking_offsets(
        self,
    ) -> Tuple[float, float, float, float, float, float]:
        """Get current face tracking offsets (thread-safe)."""
        with self.face_tracking_lock:
            offsets = self.face_tracking_offsets
            return (offsets[0], offsets[1], offsets[2], offsets[3], offsets[4], offsets[5])

    def set_head_tracking_enabled(self, enabled: bool) -> None:
        """Enable/disable head tracking."""
        self.is_head_tracking_enabled = enabled
        logger.info(f"Head tracking {'enabled' if enabled else 'disabled'}")

    def start(self) -> None:
        """Start the camera worker loop in a thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.working_loop, daemon=True)
        self._thread.start()
        logger.debug("Camera worker started")

    def stop(self) -> None:
        """Stop the camera worker loop."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        head_tracker_close = getattr(self.head_tracker, "close", None)
        if callable(head_tracker_close):
            head_tracker_close()

        logger.debug("Camera worker stopped")

    def working_loop(self) -> None:
        """Run the camera worker loop."""
        logger.debug("Starting camera working loop")

        neutral_pose = np.eye(4)
        self.previous_head_tracking_state = self.is_head_tracking_enabled

        while not self._stop_event.is_set():
            try:
                current_time = time.monotonic()  # wall clock jumped with NTP -> frozen/snapping tracking (P3)
                frame = self.reachy_mini.media.get_frame()

                if frame is not None:
                    # Keep the latest frame available for tools and UI consumers
                    with self.frame_lock:
                        self.latest_frame = frame
                        self.latest_frame_ts = time.monotonic()

                    if self.previous_head_tracking_state and not self.is_head_tracking_enabled:
                        # Reuse the face-lost interpolation path to return smoothly to neutral
                        self.last_face_detected_time = current_time
                        self.interpolation_start_time = None
                        self.interpolation_start_pose = None

                    self.previous_head_tracking_state = self.is_head_tracking_enabled

                    if self.is_head_tracking_enabled and self.head_tracker is not None:
                        eye_center, _ = self.head_tracker.get_head_position(frame)

                        if eye_center is not None:
                            self.last_face_detected_time = current_time
                            self.interpolation_start_time = None

                            # The tracker returns normalized coordinates in [-1, 1]
                            h, w, _ = frame.shape
                            eye_center_norm = (eye_center + 1) / 2
                            eye_center_pixels = [
                                min(max(eye_center_norm[0] * w, 1.0), w - 1.0),
                                min(max(eye_center_norm[1] * h, 1.0), h - 1.0),
                            ]  # clamp: SDK look_at_image asserts 0 < u < w (edge face crashed it, P3)

                            target_pose = self.reachy_mini.look_at_image(
                                eye_center_pixels[0],
                                eye_center_pixels[1],
                                duration=0.0,
                                perform_movement=False,
                            )

                            translation = target_pose[:3, 3]
                            rotation = R.from_matrix(target_pose[:3, :3]).as_euler("xyz", degrees=False)

                            # The camera FOV is tighter than the motion model expects
                            translation *= 0.6
                            rotation *= 0.6

                            with self.face_tracking_lock:
                                self.face_tracking_offsets = [
                                    translation[0],
                                    translation[1],
                                    translation[2],
                                    rotation[0],
                                    rotation[1],
                                    rotation[2],
                                ]

                        elif self.last_face_detected_time is None or self.last_face_detected_time == current_time:
                            pass

                    if self.last_face_detected_time is not None:
                        time_since_face_lost = current_time - self.last_face_detected_time

                        if time_since_face_lost >= self.face_lost_delay:
                            if self.interpolation_start_time is None:
                                self.interpolation_start_time = current_time
                                with self.face_tracking_lock:
                                    current_translation = self.face_tracking_offsets[:3]
                                    current_rotation_euler = self.face_tracking_offsets[3:]
                                    # Interpolate from the current tracking pose back to neutral
                                    pose_matrix = np.eye(4, dtype=np.float32)
                                    pose_matrix[:3, 3] = current_translation
                                    pose_matrix[:3, :3] = R.from_euler(
                                        "xyz",
                                        current_rotation_euler,
                                    ).as_matrix()
                                    self.interpolation_start_pose = pose_matrix

                            elapsed_interpolation = current_time - self.interpolation_start_time
                            t = min(1.0, elapsed_interpolation / self.interpolation_duration)

                            interpolated_pose = linear_pose_interpolation(
                                self.interpolation_start_pose,
                                neutral_pose,
                                t,
                            )

                            translation = interpolated_pose[:3, 3]
                            rotation = R.from_matrix(interpolated_pose[:3, :3]).as_euler("xyz", degrees=False)

                            with self.face_tracking_lock:
                                self.face_tracking_offsets = [
                                    translation[0],
                                    translation[1],
                                    translation[2],
                                    rotation[0],
                                    rotation[1],
                                    rotation[2],
                                ]

                            if t >= 1.0:
                                self.last_face_detected_time = None
                                self.interpolation_start_time = None
                                self.interpolation_start_pose = None

                time.sleep(0.04)

            except Exception as e:
                logger.error(f"Camera worker error: {e}")
                time.sleep(0.1)

        logger.debug("Camera worker thread exited")
