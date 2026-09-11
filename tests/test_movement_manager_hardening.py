# ruff: noqa: D103
"""MovementManager hardening (review 2026-07-02 round 2, P1-10 + seam).

1. A throwing move / degenerate pose must not kill the 60 Hz worker thread (guarded tick).
2. BreathingMove.evaluate falls back to neutral instead of propagating (e.g. scipy
   R.from_matrix ValueError on a degenerate interpolation_start_pose).
3. The external-offsets seam is ADDITIVE with face tracking and actually reaches the
   composed pose (the old pending seam was clobbered every tick by _update_face_tracking).
"""

from __future__ import annotations

import numpy as np
import pytest


pytest.importorskip("reachy_mini")

from reachy_mini_conversation_app.moves import (  # noqa: E402
    BreathingMove,
    MovementManager,
)


class _FakeRobot:
    def __init__(self) -> None:
        self.targets = []

    def set_target(self, *a, **kw) -> None:
        self.targets.append((a, kw))

    def get_current_head_pose(self):
        return np.eye(4, dtype=np.float32)


def test_breathing_evaluate_degenerate_start_pose_falls_back_to_neutral():
    move = BreathingMove(
        interpolation_start_pose=np.zeros((4, 4), dtype=np.float32),  # degenerate (not a pose)
        interpolation_start_antennas=(0.0, 0.0),
    )
    head, antennas, body_yaw = move.evaluate(0.1)  # inside the interpolation phase
    assert head is not None and np.isfinite(head).all()
    assert antennas is not None and np.isfinite(antennas).all()
    assert body_yaw == 0.0


def test_worker_tick_survives_throwing_stage():
    mgr = MovementManager(_FakeRobot())
    ticks = {"n": 0}

    def _boom(_t):
        ticks["n"] += 1
        if ticks["n"] >= 3:
            mgr._stop_event.set()  # end the loop after a few iterations
        raise ValueError("degenerate pose")

    mgr._update_primary_motion = _boom  # stage 2 throws every tick
    mgr.working_loop()  # must return (stop_event), NOT raise
    assert ticks["n"] >= 3
    assert getattr(mgr, "_tick_errors", 0) >= 3


def test_external_offsets_are_additive_with_face_tracking():
    class _Cam:
        def get_face_tracking_offsets(self):
            return (0.01, 0.0, 0.0, 0.0, 0.0, 0.02)

    mgr = MovementManager(_FakeRobot(), camera_worker=_Cam())
    mgr.set_external_offsets((0.0, 0.0, 0.005, 0.0, 0.0, 0.03), antennas=(0.1, -0.1))
    mgr._apply_pending_offsets()
    mgr._update_face_tracking(0.0)  # camera write must NOT clobber the external channel

    assert mgr.state.external_offsets == (0.0, 0.0, 0.005, 0.0, 0.0, 0.03)
    assert mgr.state.face_tracking_offsets == (0.01, 0.0, 0.0, 0.0, 0.0, 0.02)

    head, antennas, body_yaw = mgr._get_secondary_pose()
    assert antennas == (0.1, -0.1)  # external antennas reach the secondary pose
    assert body_yaw == 0.0
    # the composed secondary head pose reflects the SUMMED offsets (z and yaw both non-zero)
    assert head[2, 3] == pytest.approx(0.005, abs=1e-6)  # z translation
    yaw = float(np.arctan2(head[1, 0], head[0, 0]))
    assert yaw == pytest.approx(0.05, abs=1e-3)  # 0.02 + 0.03 rad


def test_external_offsets_zero_is_behavior_neutral():
    mgr = MovementManager(_FakeRobot())
    mgr._update_face_tracking(0.0)
    head, antennas, body_yaw = mgr._get_secondary_pose()
    assert antennas == (0.0, 0.0) and body_yaw == 0.0
    assert np.allclose(head, np.eye(4), atol=1e-9)


def test_quiet_idle_uses_stable_antenna_rest_offset(monkeypatch):
    """Stationary antennas keep the small outward bias that avoids zero-position servo hunting."""
    monkeypatch.setenv("AGENT_ANTENNA_REST_DEG", "10")
    mgr = MovementManager(_FakeRobot())

    _, antennas, _ = mgr.state.last_primary_pose

    assert antennas[0] == pytest.approx(-np.deg2rad(10), abs=1e-6)
    assert antennas[1] == pytest.approx(np.deg2rad(10), abs=1e-6)


def test_breathing_interpolates_body_yaw_to_zero():
    """Review 2026-07-02 round 2, P2: breathing hard-returned body_yaw=0.0 from the first tick — a body SNAP after any emotion ending with body_yaw != 0.

    Phase 1 now blends it to 0.
    """
    move = BreathingMove(
        interpolation_start_pose=np.eye(4, dtype=np.float32),
        interpolation_start_antennas=(0.0, 0.0),
        interpolation_duration=1.0,
        interpolation_start_body_yaw=0.4,
    )
    _, _, yaw_start = move.evaluate(0.0)
    _, _, yaw_mid = move.evaluate(0.5)
    _, _, yaw_late = move.evaluate(2.0)  # phase 2
    assert yaw_start == pytest.approx(0.4, abs=1e-6)
    assert yaw_mid == pytest.approx(0.2, abs=1e-6)
    assert yaw_late == 0.0


def test_dequeued_goto_is_rebased_onto_current_pose():
    """Review 2026-07-02 round 2, P2: a goto queued behind a running move froze its start pose at ENQUEUE time -> one-tick jump back on dequeue.

    The manager now re-bases it at dequeue.
    """
    from reachy_mini_conversation_app.moves import clone_full_body_pose  # noqa: F401 (import check)
    from reachy_mini_conversation_app.dance_emotion_moves import GotoQueueMove

    mgr = MovementManager(_FakeRobot())
    stale_start = np.eye(4, dtype=np.float32)
    target = np.eye(4, dtype=np.float32)
    target[0, 3] = 0.02
    goto = GotoQueueMove(
        target_head_pose=target,
        start_head_pose=stale_start,
        target_antennas=(0.0, 0.0),
        start_antennas=(0.5, -0.5),  # stale
        target_body_yaw=0.0,
        start_body_yaw=0.3,  # stale
        duration=1.0,
    )
    mgr.move_queue.append(goto)
    # the head is ACTUALLY somewhere else by now (end pose of the previous move)
    actual = np.eye(4, dtype=np.float32)
    actual[1, 3] = 0.01
    mgr.state.last_primary_pose = (actual, (0.1, -0.1), 0.05)

    mgr._manage_move_queue(current_time=100.0)

    assert mgr.state.current_move is goto
    head0, ant0, yaw0 = goto.evaluate(0.0)
    assert head0[1, 3] == pytest.approx(0.01, abs=1e-6)  # starts where the head IS
    assert tuple(np.round(ant0, 6)) == (0.1, -0.1)
    assert yaw0 == pytest.approx(0.05, abs=1e-6)


def test_breathing_starts_from_last_primary_pose_not_measured():
    """Review 2026-07-02 round 2, P3: breathing started from the MEASURED pose (which contains the face-tracking offset) — the composition added the offset again = one-tick lurch toward 2x offset.

    It must start from the last COMMANDED primary pose.
    """

    class _OffsetRobot(_FakeRobot):
        def get_current_joint_positions(self):
            return 0.3, (0.5, -0.5)  # measured (contains secondary) — must NOT be used

        def get_current_head_pose(self):
            m = np.eye(4, dtype=np.float32)
            m[0, 3] = 0.05  # measured pose with tracking offset baked in
            return m

    mgr = MovementManager(_OffsetRobot())
    primary = np.eye(4, dtype=np.float32)
    primary[0, 3] = 0.01
    mgr.state.last_primary_pose = (primary, (0.1, -0.1), 0.02)
    mgr.state.last_activity_time = 0.0
    mgr._manage_breathing(current_time=mgr.idle_inactivity_delay + 1.0)

    assert len(mgr.move_queue) == 1
    breathing = mgr.move_queue[0]
    assert breathing.interpolation_start_pose[0, 3] == pytest.approx(0.01, abs=1e-6)  # primary, not measured
    assert breathing.interpolation_start_body_yaw == pytest.approx(0.02, abs=1e-6)


def test_quiet_profile_disables_idle_breathing(monkeypatch):
    """Quiet mode holds position instead of continuously exercising head and antenna motors."""
    monkeypatch.setenv("AGENT_IDLE_BREATHING", "0")
    mgr = MovementManager(_FakeRobot())
    mgr.state.last_activity_time = 0.0

    mgr._manage_breathing(current_time=mgr.idle_inactivity_delay + 1.0)

    assert mgr.state.current_move is None
    assert list(mgr.move_queue) == []
    assert mgr._breathing_active is False


def test_listening_stops_active_idle_breathing():
    """Microphone listening takes priority and immediately stops motor-noisy breathing."""
    mgr = MovementManager(_FakeRobot())
    breathing = BreathingMove(np.eye(4, dtype=np.float32), (0.0, 0.0))
    mgr.state.current_move = breathing
    mgr.state.move_start_time = 1.0
    mgr.move_queue.append(BreathingMove(np.eye(4, dtype=np.float32), (0.0, 0.0)))
    mgr._breathing_active = True
    mgr._last_listening_toggle_time = mgr._now() - 1.0

    mgr._handle_command("set_listening", True, mgr._now())

    assert mgr.state.current_move is None
    assert list(mgr.move_queue) == []
    assert mgr._breathing_active is False


def test_dance_emotion_error_fallback_holds_last_pose():
    from reachy_mini_conversation_app.dance_emotion_moves import EmotionQueueMove

    class _Rec:
        description = "x"

        def evaluate(self, t):
            if t > 1.0:
                raise ValueError("t beyond timestamps")  # upstream boundary raise
            m = np.eye(4, dtype=np.float64)
            m[0, 3] = 0.02
            return m, (0.3, -0.3), 0.1

        duration = 1.0

    class _Lib:
        def get(self, name):
            return _Rec()

    move = EmotionQueueMove("happy1", _Lib())
    ok = move.evaluate(0.5)
    bad = move.evaluate(1.5)  # raises upstream -> must hold the last valid pose, not neutral
    assert bad[0][0, 3] == pytest.approx(0.02)
    assert bad == ok
