#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
# Upstream XR profile: Copyright 2025 Unitree Robotics, Apache-2.0.
# See examples/unitree_g1d/UNITREE_XR_NOTICES.md for source revisions and notices.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


class WeightedMovingFilter:
    def __init__(self, weights, data_size=14, *, newest_first: bool = False):
        self._window_size = len(weights)
        self._weights = np.array(weights)
        if newest_first:
            self._weights = self._weights[::-1]
        self._data_size = data_size
        self._filtered_data = np.zeros(self._data_size)
        self._data_queue = deque(maxlen=self._window_size)

    def _apply_filter(self):
        if len(self._data_queue) < self._window_size:
            return self._data_queue[-1]

        data_array = np.array(self._data_queue)
        return data_array.T @ self._weights

    def add_data(self, new_data):
        assert len(new_data) == self._data_size

        if len(self._data_queue) > 0 and np.array_equal(
            new_data, self._data_queue[-1]
        ):  # skip duplicate data
            return

        self._data_queue.append(new_data)
        self._filtered_data = self._apply_filter()

    def reset(self) -> None:
        self._data_queue.clear()
        self._filtered_data = np.zeros(self._data_size)

    @property
    def filtered_data(self):
        return self._filtered_data


class G1_29_ArmIK:  # noqa: N801
    def __init__(self, unit_test=False, *, unitree_xr: bool = False):
        import casadi
        import pinocchio as pin
        from huggingface_hub import snapshot_download
        from pinocchio import casadi as cpin

        self._pin = pin
        self.unit_test = unit_test

        self.repo_path = snapshot_download("lerobot/unitree-g1-mujoco")
        urdf_path = os.path.join(self.repo_path, "assets", "g1_body29_hand14.urdf")
        mesh_dir = os.path.join(self.repo_path, "assets")

        self.robot = self._pin.RobotWrapper.BuildFromURDF(urdf_path, mesh_dir)

        self.mixed_jointsToLockIDs = [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_hand_thumb_0_joint",
            "left_hand_thumb_1_joint",
            "left_hand_thumb_2_joint",
            "left_hand_middle_0_joint",
            "left_hand_middle_1_joint",
            "left_hand_index_0_joint",
            "left_hand_index_1_joint",
            "right_hand_thumb_0_joint",
            "right_hand_thumb_1_joint",
            "right_hand_thumb_2_joint",
            "right_hand_index_0_joint",
            "right_hand_index_1_joint",
            "right_hand_middle_0_joint",
            "right_hand_middle_1_joint",
        ]

        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.array([0.0] * self.robot.model.nq),
        )

        # Arm joint names in G1 motor order (G1_29_JointArmIndex)
        self._arm_joint_names_g1 = [
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]
        # Pinocchio uses its own joint order in q; build index mapping.
        self._arm_joint_names_pin = sorted(
            self._arm_joint_names_g1,
            key=lambda name: self.reduced_robot.model.idx_qs[self.reduced_robot.model.getJointId(name)],
        )
        logger.info(f"Pinocchio arm joint order: {self._arm_joint_names_pin}")
        self._arm_reorder_g1_to_pin = [
            self._arm_joint_names_g1.index(name) for name in self._arm_joint_names_pin
        ]
        # Inverse mapping to return tau in G1 motor order.
        self._arm_reorder_pin_to_g1 = np.argsort(self._arm_reorder_g1_to_pin)

        self.reduced_robot.model.addFrame(
            self._pin.Frame(
                "L_ee",
                self.reduced_robot.model.getJointId("left_wrist_yaw_joint"),
                self._pin.SE3(np.eye(3), np.array([0.05, 0, 0]).T),
                self._pin.FrameType.OP_FRAME,
            )
        )

        self.reduced_robot.model.addFrame(
            self._pin.Frame(
                "R_ee",
                self.reduced_robot.model.getJointId("right_wrist_yaw_joint"),
                self._pin.SE3(np.eye(3), np.array([0.05, 0, 0]).T),
                self._pin.FrameType.OP_FRAME,
            )
        )

        # Creating Casadi models and data for symbolic computing
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        # Creating symbolic variables
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)
        self.cTf_l = casadi.SX.sym("tf_l", 4, 4)
        self.cTf_r = casadi.SX.sym("tf_r", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # Get the hand joint ID and define the error function
        self.L_hand_id = self.reduced_robot.model.getFrameId("L_ee")
        self.R_hand_id = self.reduced_robot.model.getFrameId("R_ee")

        self.translational_error = casadi.Function(
            "translational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    self.cdata.oMf[self.L_hand_id].translation - self.cTf_l[:3, 3],
                    self.cdata.oMf[self.R_hand_id].translation - self.cTf_r[:3, 3],
                )
            ],
        )
        self.rotational_error = casadi.Function(
            "rotational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    cpin.log3(self.cdata.oMf[self.L_hand_id].rotation @ self.cTf_l[:3, :3].T),
                    cpin.log3(self.cdata.oMf[self.R_hand_id].rotation @ self.cTf_r[:3, :3].T),
                )
            ],
        )

        # Defining the optimization problem
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)  # for smooth
        self.param_tf_l = self.opti.parameter(4, 4)
        self.param_tf_r = self.opti.parameter(4, 4)
        self.translational_cost = casadi.sumsqr(
            self.translational_error(self.var_q, self.param_tf_l, self.param_tf_r)
        )
        self.rotation_cost = casadi.sumsqr(
            self.rotational_error(self.var_q, self.param_tf_l, self.param_tf_r)
        )
        # Redundant arm joints should stay near the measured pose, not drift
        # toward zero even when the requested wrists have not moved.
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)

        # Setting optimization constraints and goals
        self.opti.subject_to(
            self.opti.bounded(
                self.reduced_robot.model.lowerPositionLimit,
                self.var_q,
                self.reduced_robot.model.upperPositionLimit,
            )
        )
        objective = 50 * self.translational_cost + self.rotation_cost + 0.1 * self.smooth_cost
        if unitree_xr:
            # xr_teleoperate G1_29_ArmIK: bias redundant joints toward zero.
            objective += 0.02 * casadi.sumsqr(self.var_q)
        self.opti.minimize(objective)

        opts = {
            "ipopt": {"print_level": 0, "max_iter": 50, "tol": 1e-6},
            "print_time": False,  # print or not
            "calc_lam_p": False,  # https://github.com/casadi/casadi/wiki/FAQ:-Why-am-I-getting-%22NaN-detected%22in-my-optimization%3F
        }
        if unitree_xr:
            opts = {
                "expand": True,
                "detect_simple_bounds": True,
                "calc_lam_p": False,
                "print_time": False,
                "ipopt.sb": "yes",
                "ipopt.print_level": 0,
                "ipopt.max_iter": 30,
                "ipopt.tol": 1e-4,
                "ipopt.acceptable_tol": 5e-4,
                "ipopt.acceptable_iter": 5,
                "ipopt.warm_start_init_point": "yes",
                "ipopt.derivative_test": "none",
                "ipopt.jacobian_approximation": "exact",
            }
        self.opti.solver("ipopt", opts)

        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.smooth_filter = WeightedMovingFilter(np.array([0.4, 0.3, 0.2, 0.1]), 14, newest_first=unitree_xr)
        self.last_solve_success = False

    def forward_kinematics(self, arm_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return left/right end-effector transforms in the reduced model's base frame.

        Arm positions use G1 motor order. Legs and waist are fixed at zero in this model.
        """
        q_pin = np.asarray(arm_q)[self._arm_reorder_g1_to_pin]
        data = self.reduced_robot.model.createData()
        self._pin.framesForwardKinematics(self.reduced_robot.model, data, q_pin)
        return data.oMf[self.L_hand_id].homogeneous.copy(), data.oMf[self.R_hand_id].homogeneous.copy()

    def solve_ik(
        self,
        left_wrist,
        right_wrist,
        current_lr_arm_motor_q=None,
        current_lr_arm_motor_dq=None,
        *,
        smooth: bool = True,
    ):
        """Solve in G1 motor order; disable smoothing when the caller limits joint speed."""
        self.last_solve_success = False
        if current_lr_arm_motor_q is not None:
            self.init_data = np.asarray(current_lr_arm_motor_q)[self._arm_reorder_g1_to_pin]
        self.opti.set_initial(self.var_q, self.init_data)

        self.opti.set_value(self.param_tf_l, left_wrist)
        self.opti.set_value(self.param_tf_r, right_wrist)
        self.opti.set_value(self.var_q_last, self.init_data)  # for smooth

        try:
            self.opti.solve()
            sol_q = np.asarray(self.opti.value(self.var_q), dtype=float)
            if sol_q.shape != (14,) or not np.isfinite(sol_q).all():
                raise ValueError("Nonfinite or malformed IK solution")
        except Exception as e:
            logger.error(f"IK convergence error: {e}")
            # Failed iterates must not enter the moving filter or warm-start history.
            return (
                self.init_data[self._arm_reorder_pin_to_g1].copy(),
                np.zeros(self.reduced_robot.model.nv),
            )

        if smooth:
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data
        self.init_data = sol_q

        sol_tauff = self._pin.rnea(
            self.reduced_robot.model,
            self.reduced_robot.data,
            sol_q,
            np.zeros(self.reduced_robot.model.nv),
            np.zeros(self.reduced_robot.model.nv),
        )

        self.last_solve_success = True
        return sol_q[self._arm_reorder_pin_to_g1], sol_tauff[self._arm_reorder_pin_to_g1]

    def solve_tau(self, current_lr_arm_motor_q=None, current_lr_arm_motor_dq=None):
        try:
            q_g1 = np.array(current_lr_arm_motor_q, dtype=float)
            if q_g1.shape[0] != len(self._arm_joint_names_g1):
                raise ValueError(f"Expected {len(self._arm_joint_names_g1)} arm joints, got {q_g1.shape[0]}")
            q_pin = q_g1[self._arm_reorder_g1_to_pin]
            sol_tauff = self._pin.rnea(
                self.reduced_robot.model,
                self.reduced_robot.data,
                q_pin,
                np.zeros(self.reduced_robot.model.nv),
                np.zeros(self.reduced_robot.model.nv),
            )
            return sol_tauff[self._arm_reorder_pin_to_g1]

        except Exception as e:
            logger.error(f"ERROR in convergence, plotting debug info.{e}")
            return np.zeros(self.reduced_robot.model.nv)
