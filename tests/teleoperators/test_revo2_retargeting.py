"""Retargeting frame, channel order, and real BrainCo kinematic model checks."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lerobot.teleoperators.pico4.revo2_retargeting import (
    ANGLE_RANGES,
    JOINT_SUFFIXES,
    MOTOR_NAMES,
    PicoRevo2Retargeter,
    hand_points,
)


def skeleton():
    joints = np.tile([0, 0, 0, 0, 0, 0, 1.0], (26, 1))
    joints[1:, :3] = np.arange(75).reshape(25, 3) * 0.001
    return joints


def test_frame_matches_explicit_upstream_transform_and_is_world_invariant():
    joints = skeleton()
    expected = hand_points(joints)
    # Palm is not part of the WebXR 25-joint hand.
    joints[0, :3] = 999
    rotation = Rotation.from_euler("xyz", [0.4, -0.7, 1.2])
    joints[:, :3] = rotation.apply(joints[:, :3]) + [1, 2, 3]
    joints[:, 3:] = rotation.as_quat()
    basis = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
    correction = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    robot_wrist = basis @ rotation.as_matrix() @ basis.T
    robot_points = basis @ (joints[1:, :3] - joints[1, :3]).T
    upstream = (correction @ robot_wrist.T @ robot_points).T
    np.testing.assert_allclose(hand_points(joints), upstream, atol=1e-14)
    np.testing.assert_allclose(hand_points(joints), expected, atol=1e-14)


@pytest.mark.parametrize("fault", ["nan", "shape", "quaternion", "tip"])
def test_invalid_skeletons_rejected(fault):
    joints = skeleton()
    if fault == "nan":
        joints[10, 0] = np.nan
    elif fault == "shape":
        joints = joints[1:]
    elif fault == "quaternion":
        joints[1, 3:] = 0
    else:
        joints[5, :3] = joints[1, :3]
    with pytest.raises(ValueError):
        hand_points(joints)


@pytest.mark.parametrize("inverted", [False, True])
def test_driver_order_and_angle_normalization(inverted):
    retargeter = PicoRevo2Retargeter.__new__(PicoRevo2Retargeter)
    retargeter.sides = ("left",)
    retargeter.invert_thumb_base = frozenset(("left",)) if inverted else frozenset()
    retargeter.motor_indices = {"left": [5, 2, 0, 4, 1, 3]}
    q = np.zeros(6)
    fractions = np.array([0.1, 0.2, 0.3, 0.4, -0.1, 1.1])
    q[retargeter.motor_indices["left"]] = ANGLE_RANGES * fractions
    retargeter.solvers = {"left": SimpleNamespace(retarget=lambda ref: q)}
    action = retargeter.retarget({"left": skeleton()})
    assert list(action) == [f"left_hand.{name}.pos" for name in MOTOR_NAMES]
    expected = np.clip(fractions, 0, 1)
    if inverted:
        expected[0] = 1 - expected[0]
    np.testing.assert_allclose(list(action.values()), expected)
    q[0] = np.nan
    with pytest.raises(ValueError, match="output"):
        retargeter.retarget({"left": skeleton()})


def test_thumb_base_inversion_is_per_side():
    retargeter = PicoRevo2Retargeter.__new__(PicoRevo2Retargeter)
    retargeter.sides = ("left", "right")
    retargeter.invert_thumb_base = frozenset(("left",))
    retargeter.motor_indices = {side: list(range(6)) for side in retargeter.sides}
    retargeter.solvers = {
        side: SimpleNamespace(retarget=lambda ref: ANGLE_RANGES * 0.2) for side in retargeter.sides
    }
    action = retargeter.retarget({side: skeleton() for side in retargeter.sides})
    assert action.pop("left_hand.motor_0.pos") == pytest.approx(0.8)
    assert list(action.values()) == pytest.approx([0.2] * 11)


@pytest.mark.parametrize("side", ["left", "right"])
def test_real_dexpilot_recovers_finger_closure_from_robot_fingertips(side):
    pytest.importorskip("dex_retargeting")
    retargeter = PicoRevo2Retargeter((side,))
    solver = retargeter.solvers[side]
    robot = solver.optimizer.robot
    outputs = []
    for fraction in (0.05, 0.7):
        q = np.zeros(robot.dof)
        q[retargeter.motor_indices[side]] = ANGLE_RANGES * fraction
        q = solver.optimizer.adaptor.forward_qpos(q)
        robot.compute_forward_kinematics(q)
        joints = skeleton()
        joints[1, :3] = 0
        for index, finger in zip(
            (5, 10, 15, 20, 25), ("thumb", "index", "middle", "ring", "pinky"), strict=True
        ):
            point = robot.get_link_pose(robot.get_link_index(f"{side}_{finger}_tip"))[:3, 3]
            joints[index, :3] = point[[2, 0, 1]]
        for _ in range(40):
            action = retargeter.retarget({side: joints})
        outputs.append(np.array(list(action.values())))
        assert np.isfinite(outputs[-1]).all()
        assert np.all((outputs[-1] >= 0) & (outputs[-1] <= 1))
    assert np.all(outputs[1][2:] - outputs[0][2:] > 0.3)
    assert [solver.joint_names[i] for i in retargeter.motor_indices[side]] == [
        f"{side}_{name}" for name in JOINT_SUFFIXES
    ]
