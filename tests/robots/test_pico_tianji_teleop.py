from unittest.mock import Mock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lerobot.processor.factory import make_default_teleop_action_processor
from lerobot.robots.tianji.tianji import JOINT_NAMES
from lerobot.teleoperators.bi_pico4.tianji_processor import (
    CHEST_FROM_XR,
    POSE_FIELDS,
    PicoToTianjiJoints,
    WristTarget,
)


def test_wrist_target_axes_and_rotation():
    for index, sign in enumerate((-1, 1)):
        wrist = WristTarget(np.eye(4), np.eye(4), np.zeros(7), 0.0)
        controller = np.eye(4)
        controller[1, 3] = 0.1
        controller[:3, :3] = Rotation.from_euler("y", 30, degrees=True).as_matrix()
        for tick in range(1, 901):
            pose = wrist.update(controller, CHEST_FROM_XR[index], tick / 90)
        assert pose[:3, 3] == pytest.approx([0, sign * 0.1, 0], abs=1e-8)
        expected = Rotation.from_euler("y", sign * 30, degrees=True).as_matrix()
        assert pose[:3, :3] == pytest.approx(expected, abs=1e-8)


def test_clutch_rebases_and_bounds_with_fallback():
    solver = Mock()
    solver.forward.return_value = np.eye(4)
    solver.inverse.side_effect = lambda side, target, seed, mode: seed + (40 if mode else 4)
    from lerobot.teleoperators.bi_pico4.wrist_calibration import WristCalibration

    step = PicoToTianjiJoints(solver=solver, calibration=WristCalibration(np.zeros((2, 3)), [0, 0, -1]))
    pipeline = make_default_teleop_action_processor()
    pipeline.steps = [step]
    observation = dict.fromkeys(JOINT_NAMES, 0.0)

    def tick(stamp, left, x=0.0, age=0.0):
        step.clock = lambda: stamp + age
        action = {"sample_time": stamp, "left.grip": left, "right.grip": 0}
        for side in ("head", "left", "right"):
            action.update(
                {
                    f"{side}.{field}": value
                    for field, value in zip(POSE_FIELDS, (x, 0, 0, 0, 0, 0, 1), strict=True)
                }
            )
        return pipeline((action, observation))

    assert tick(0, 1) == observation
    assert tick(0.1, 0) == observation
    assert tick(0.2, 1) == observation
    action = tick(0.3, 1, x=0.1)
    assert list(action) == list(JOINT_NAMES)
    assert list(action.values()) == pytest.approx([1.8] * 7 + [0] * 7)
    assert step.diagnostics["left"]["ik_mode"] == 0
    # Stationary feedback in preview must not make a continuous IK trajectory
    # look like a joint-branch jump when the command reaches its measured bound.
    for stamp in np.arange(0.31, 0.4, 0.01):
        assert list(tick(stamp, 1, x=0.1).values()) == pytest.approx([1.8] * 7 + [0] * 7)
    assert step.diagnostics["left"]["ik_deg"][0] > 15
    observation[JOINT_NAMES[0]] = 0.6
    assert tick(0.4, 0) == observation
    assert tick(0.5, 1, x=0.3) == observation
    assert solver.forward.call_count == 2
    solver.inverse.side_effect = lambda side, target, seed, mode: seed + 40
    assert tick(0.55, 1, x=0.35) == observation
    assert step.diagnostics["left"]["status"] == "release_to_reengage"
    assert tick(0.56, 1, x=0.4) == observation
    assert solver.forward.call_count == 3
    tick(0.57, 0)
    tick(0.58, 1, x=0.4)
    assert solver.forward.call_count == 4
    with pytest.raises(TimeoutError, match="expired"):
        tick(0.6, 1, age=1)
    pipeline.reset()
    assert tick(0.7, 1) == observation
    assert step.recording_frame["teleop.clutch"].tolist() == [0, 0]


def test_forward_controller_motion_uses_g1_head_relative_mapping():
    solver = Mock()
    solver.forward.return_value = np.eye(4)
    solver.solve_local.side_effect = lambda side, pose, seed, cap, *, measured: seed
    step = PicoToTianjiJoints(solver=solver)
    pipeline = make_default_teleop_action_processor()
    pipeline.steps = [step]
    observation = dict.fromkeys(JOINT_NAMES, 0.0)
    for tick, grip in enumerate((0, 1, 1)):
        step.clock = lambda tick=tick: float(tick)
        action = {"sample_time": tick, "left.grip": grip, "right.grip": grip}
        for side in ("left", "right", "head"):
            # Looking to XR -X at engagement. Keep that heading while moving
            # both controllers along that original forward direction.
            heading = 90 if side == "head" else 0
            quat = Rotation.from_euler("y", heading, degrees=True).as_quat()
            x = -0.05 if side != "head" and tick == 2 else 0
            action.update(zip((f"{side}.{f}" for f in POSE_FIELDS), (x, 0, 0, *quat), strict=True))
        pipeline((action, observation))
    for side in ("left", "right"):
        target = np.array(step.diagnostics[side]["target"])
        assert target[0, 3] > 0.04
        np.testing.assert_allclose(target[1:3, 3], 0, atol=1e-9)


def test_handheld_pivot_and_forward_calibration():
    from lerobot.teleoperators.bi_pico4.wrist_calibration import WristCalibration, fit_wrist_offset

    wrist = np.array([0.1, 0.5, -0.3])
    offset = np.array([-0.03, 0.02, 0.12])
    poses = np.repeat(np.eye(4)[None], 60, axis=0)
    poses[:, :3, :3] = Rotation.random(60, random_state=24).as_matrix()
    poses[:, :3, 3] = wrist - poses[:, :3, :3] @ offset
    fitted, rms = fit_wrist_offset(poses)
    np.testing.assert_allclose(fitted, offset)
    assert rms < 1e-12
    with pytest.raises(ValueError, match="two different axes"):
        fit_wrist_offset(np.repeat(poses[:1], 60, axis=0))
    calibration = WristCalibration([fitted, fitted], [0.15, 0.003, -0.15])
    np.testing.assert_allclose(calibration.world_from_xr @ calibration.forward, [1, 0, 0], atol=1e-12)
    solver = Mock()
    solver.forward.return_value = np.eye(4)
    solver.inverse.side_effect = lambda side, pose, seed, mode: seed
    step = PicoToTianjiJoints(solver=solver, calibration=calibration)
    pipeline = make_default_teleop_action_processor()
    pipeline.steps = [step]
    for i, pose in enumerate(poses):
        step.clock = lambda i=i: float(i)
        action = {"sample_time": i}
        for side in ("left", "right"):
            action[f"{side}.grip"] = float(i > 0)
            values = (*pose[:3, 3], *Rotation.from_matrix(pose[:3, :3]).as_quat())
            action.update(zip((f"{side}.{f}" for f in POSE_FIELDS), values, strict=True))
        pipeline((action, dict.fromkeys(JOINT_NAMES, 0.0)))
        if i > 1:
            for diagnostic in step.diagnostics.values():
                np.testing.assert_allclose(np.array(diagnostic["target"])[:3, 3], 0, atol=1e-12)


@pytest.mark.parametrize("side", [0, 1])
def test_local_posture_bias_only_while_tracking(side):
    from lerobot.robots.tianji.kinematics import TianjiKinematics
    from lerobot.robots.tianji.tianji import WUJI_START_JOINTS

    # A free seventh joint makes the posture preference directly observable.
    solver = object.__new__(TianjiKinematics)
    solver._home_deg = np.array(WUJI_START_JOINTS).reshape(2, 7)
    solver._joint_bounds = np.tile([-180, 180], (2, 7, 1))
    solver._wrist_limits = np.tile([[0, 0, 90], [0, 0, 90], [0, 0, -90], [0, 0, -90]], (2, 1, 1))

    def forward(side, joints):
        pose = np.eye(4)
        pose[:3, 3] = np.deg2rad(joints[:3])
        return pose

    solver.forward = forward
    measured = solver._home_deg[side].astype(float)
    measured[-1] = 20
    measured[4] = 0
    target = forward(side, measured)
    solver._posture_scale = np.sqrt(0.001)
    np.testing.assert_allclose(solver.solve_local(side, target, measured, 2), measured)
    target[0, 3] += 0.05
    biased = solver.solve_local(side, target, measured, 2)
    assert 0 < biased[-1] < 20
    assert np.sign(biased[4]) == np.sign(solver._home_deg[side, 4])
    solver._posture_scale = 0
    unbiased = solver.solve_local(side, target, measured, 2)
    assert unbiased[-1] == pytest.approx(20)

    reference = measured.copy()
    reference[-1] += 5
    bounded = solver.solve_local(side, target, reference, 2, measured=measured)
    assert np.max(np.abs(bounded - measured)) <= 2 + 1e-9
    assert bounded[-1] == pytest.approx(measured[-1] + 2, abs=1e-4)


def test_clutch_uses_measurement_and_tracking_uses_last_sent_action():
    solver = Mock()
    solver.forward.return_value = np.eye(4)
    solver.solve_local.side_effect = lambda side, pose, seed, cap, *, measured: seed
    sent = dict.fromkeys(JOINT_NAMES, 1.0)
    step = PicoToTianjiJoints(solver=solver, last_sent_action=lambda: sent)
    pipeline = make_default_teleop_action_processor()
    pipeline.steps = [step]

    def tick(stamp, grip, x=0):
        step.clock = lambda: stamp
        action = {"sample_time": stamp, "left.grip": grip, "right.grip": 0}
        for side in ("head", "left", "right"):
            xyzquat = (x if side == "left" else 0, 0, 0, 0, 0, 0, 1)
            action.update(zip((f"{side}.{field}" for field in POSE_FIELDS), xyzquat, strict=True))
        return pipeline((action, dict.fromkeys(JOINT_NAMES, 0.0)))

    tick(1, 0)
    engaged = tick(2, 1)
    np.testing.assert_array_equal(solver.forward.call_args.args[1], np.zeros(7))
    assert all(engaged[key] == 0 for key in JOINT_NAMES[:7])
    solver.solve_local.assert_not_called()
    # Model the driver acknowledging a command that differs from fresh feedback.
    sent = {**sent, **dict.fromkeys(JOINT_NAMES[:7], 0.5)}
    motion = tick(3, 1, x=0.01)
    np.testing.assert_array_equal(solver.solve_local.call_args.args[2], np.full(7, 0.5))
    np.testing.assert_array_equal(solver.solve_local.call_args.kwargs["measured"], np.zeros(7))
    assert all(motion[key] == 0.5 for key in JOINT_NAMES[:7])
