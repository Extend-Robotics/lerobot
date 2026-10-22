"""Check G1/Pinocchio joint ordering without loading the optional IK dependencies."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from lerobot.robots.unitree_g1.g1_kinematics import G1_29_ArmIK, WeightedMovingFilter


@pytest.mark.parametrize("converged", [True, False])
def test_solve_ik_motor_order(converged):
    ik = G1_29_ArmIK.__new__(G1_29_ArmIK)
    ik._arm_reorder_g1_to_pin = np.roll(np.arange(14), 3)
    ik._arm_reorder_pin_to_g1 = np.argsort(ik._arm_reorder_g1_to_pin)
    ik.opti = MagicMock()
    ik.var_q, ik.var_q_last, ik.param_tf_l, ik.param_tf_r = range(4)
    ik.smooth_filter = WeightedMovingFilter([0.4, 0.3, 0.2, 0.1], 14)
    ik.reduced_robot = SimpleNamespace(model=SimpleNamespace(nv=14), data=object())
    ik._pin = MagicMock()
    measured_q = np.arange(14, dtype=float)
    solution_pin = np.arange(14, dtype=float) + 0.5
    tau_pin = np.arange(14, dtype=float) + 1.5
    ik.opti.value.return_value = solution_pin
    ik.opti.debug.value.return_value = solution_pin
    ik._pin.rnea.return_value = tau_pin
    if not converged:
        ik.opti.solve.side_effect = RuntimeError("Unreachable target")

    q, tau = ik.solve_ik(np.eye(4), np.eye(4), measured_q)

    np.testing.assert_array_equal(
        ik.opti.set_initial.call_args.args[1], measured_q[ik._arm_reorder_g1_to_pin]
    )
    if converged:
        np.testing.assert_array_equal(q, solution_pin[ik._arm_reorder_pin_to_g1])
        np.testing.assert_array_equal(tau, tau_pin[ik._arm_reorder_pin_to_g1])
    else:
        np.testing.assert_array_equal(q, measured_q)
        np.testing.assert_array_equal(tau, np.zeros(14))


def test_forward_kinematics_motor_order():
    ik = G1_29_ArmIK.__new__(G1_29_ArmIK)
    ik._arm_reorder_g1_to_pin = np.roll(np.arange(14), 3)
    left, right = np.eye(4), np.eye(4)
    left[:3, 3] = [0.3, 0.2, 0.5]
    right[:3, 3] = [0.3, -0.2, 0.5]
    data = SimpleNamespace(oMf=[SimpleNamespace(homogeneous=left), SimpleNamespace(homogeneous=right)])
    model = MagicMock()
    model.createData.return_value = data
    ik.reduced_robot = SimpleNamespace(model=model)
    ik.L_hand_id, ik.R_hand_id = 0, 1
    ik._pin = MagicMock()
    measured_q = np.arange(14, dtype=float)

    actual_left, actual_right = ik.forward_kinematics(measured_q)

    np.testing.assert_array_equal(
        ik._pin.framesForwardKinematics.call_args.args[2], measured_q[ik._arm_reorder_g1_to_pin]
    )
    np.testing.assert_array_equal(actual_left, left)
    np.testing.assert_array_equal(actual_right, right)
