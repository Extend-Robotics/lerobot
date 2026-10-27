"""Upstream XR mapping and motor-control behavior, without headset or robot writes."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lerobot.robots.unitree_g1.g1_kinematics import G1_29_ArmIK, WeightedMovingFilter
from lerobot.teleoperators.bi_pico4.xr_mapping import (
    ROBOT_FROM_XR,
    map_xr_wrist_targets,
)


def sample_poses():
    return {
        "head": np.array([0, 1.5, 0, 0, 0, 0, 1.0]),
        "left": np.array([-0.2, 1.2, -0.4, 0, 0, 0, 1.0]),
        "right": np.array([0.2, 1.2, -0.4, 0, 0, 0, 1.0]),
    }


def test_absolute_targets_have_upstream_head_to_waist_offsets():
    left, right = map_xr_wrist_targets(sample_poses())
    np.testing.assert_allclose(left[:3, 3], [0.55, 0.2, 0.15])
    np.testing.assert_allclose(right[:3, 3], [0.55, -0.2, 0.15])
    np.testing.assert_allclose(left[:3, :3], np.eye(3))


def test_common_head_yaw_and_translation_do_not_change_targets():
    poses = sample_poses()
    expected = map_xr_wrist_targets(poses)
    # Rotate the entire XR scene about vertical and translate it.
    yaw = Rotation.from_euler("y", 70, degrees=True)
    for pose in poses.values():
        pose[:3] = yaw.apply(pose[:3]) + [1, 0.3, -2]
        pose[3:] = yaw.as_quat()
    for actual, target in zip(map_xr_wrist_targets(poses), expected, strict=True):
        np.testing.assert_allclose(actual, target, atol=1e-12)


def test_full_absolute_rotation_does_not_depend_on_clutch_or_initial_tool_pose():
    poses = sample_poses()
    rotation = Rotation.from_euler("xyz", [25, -35, 70], degrees=True)
    poses["left"][3:] = rotation.as_quat()
    left, _ = map_xr_wrist_targets(poses)
    change_basis = ROBOT_FROM_XR[:3, :3]
    np.testing.assert_allclose(left[:3, :3], change_basis @ rotation.as_matrix() @ change_basis.T)


def test_head_pitch_and_roll_do_not_rotate_wrist_targets():
    poses = sample_poses()
    expected = map_xr_wrist_targets(poses)
    # Robot-frame pitch/roll with zero yaw, converted back to the XR convention.
    basis = ROBOT_FROM_XR[:3, :3]
    rotation = Rotation.from_euler("xyz", [20, 30, 0], degrees=True).as_matrix()
    poses["head"][3:] = Rotation.from_matrix(basis.T @ rotation @ basis).as_quat()
    for actual, target in zip(map_xr_wrist_targets(poses), expected, strict=True):
        np.testing.assert_allclose(actual, target, atol=1e-12)


def test_upstream_joint_filter_puts_largest_weight_on_newest_sample():
    filtered = WeightedMovingFilter([0.4, 0.3, 0.2, 0.1], 14, newest_first=True)
    for value in [0, 1, 2, 3]:
        filtered.add_data(np.full(14, value))
    np.testing.assert_allclose(filtered.filtered_data, np.full(14, 2.0))
    filtered.add_data(np.full(14, 3))  # Upstream ignores duplicates.
    np.testing.assert_allclose(filtered.filtered_data, np.full(14, 2.0))


def test_native_ik_filter_and_gravity_match_pinned_upstream_sequence():
    pytest.importorskip("pinocchio.casadi")
    pytest.importorskip("casadi")
    reference = json.loads(
        (Path(__file__).parents[1] / "fixtures/unitree_g1/xr_ik_reference.json").read_text()
    )
    ik = G1_29_ArmIK(unitree_xr=True)
    measured = np.array(reference["measured_q"])
    for frame in reference["sequence"]:
        q, tau = ik.solve_ik(np.array(frame["left"]), np.array(frame["right"]), measured, np.zeros(14))
        assert ik.last_solve_success
        np.testing.assert_allclose(q, frame["q"], atol=1e-5, rtol=0)
        np.testing.assert_allclose(tau, frame["tau"], atol=5e-5, rtol=0)
