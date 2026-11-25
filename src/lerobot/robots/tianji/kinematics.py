"""Marvin flange kinematics using the Wuji Tianji PICO model.

The ctypes ABI follows Wuji Technology's MIT-licensed fx_kine.py and
robot_structures.py at 647801345a6a27dec5cbf56280ce63bb8b2f6a32.
See _sdk/LICENSE for the upstream license.
"""

import ctypes as ct
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .tianji import WUJI_START_JOINTS

_Joints = ct.c_double * 7
_Pose = (ct.c_double * 4) * 4
_DH = (ct.c_double * 4) * 8
_PNVA = (ct.c_double * 4) * 7
_BD = (ct.c_double * 3) * 4
_ZSP = ct.c_double * 6
_DEFAULT_NORMALS = ((0, -0.5, -0.3), (0, 0.5, -0.3))


class _IKParameters(ct.Structure):
    _fields_ = [
        ("target", _Pose),
        ("reference", _Joints),
        ("mode", ct.c_long),
        ("normal", _ZSP),
        ("angle", ct.c_double),
        ("dgr", ct.c_double * 3),
        ("joints", _Joints),
        ("out_of_range", ct.c_bool),
        ("singular", ct.c_bool * 7),
        ("joint_exceeded_flags", ct.c_bool * 7),
        ("joint_exceeded", ct.c_bool),
        ("positive_limits", _Joints),
        ("negative_limits", _Joints),
    ]


class TianjiKinematics:
    """Work in metres and joint degrees. The native library does not open a robot connection."""

    def __init__(self, library_path: Path, model_path: Path, posture_weight: float = 0.0):
        self._posture_scale = np.sqrt(posture_weight)
        self._home_deg = np.asarray(WUJI_START_JOINTS).reshape(2, 7)
        # The native configuration loader uses a mutable char pointer in its ABI.
        model_path = model_path.expanduser().resolve(strict=True)
        self._library = ct.CDLL(str(library_path.expanduser().resolve(strict=True)))
        types = (ct.c_long * 2)()
        gravity = ((ct.c_double * 3) * 2)()
        dh, pnva, bd = (_DH * 2)(), (_PNVA * 2)(), (_BD * 2)()
        mass = (_Joints * 2)()
        centers = (((ct.c_double * 3) * 7) * 2)()
        inertia = (((ct.c_double * 6) * 7) * 2)()
        arrays = (types, gravity, dh, pnva, bd, mass, centers, inertia)
        load = self._bind("LOADMvCfg", ct.c_char_p, *(ct.POINTER(type(a)) for a in arrays))
        if not load(bytes(model_path), *(ct.byref(a) for a in arrays)):
            raise OSError(f"Marvin could not load kinematics model {model_path}")
        init_type = self._bind("FX_Robot_Init_Type", ct.c_long, ct.c_long)
        init_kine = self._bind("FX_Robot_Init_Kine", ct.c_long, _DH)
        init_limits = self._bind("FX_Robot_Init_Lmt", ct.c_long, _PNVA, _BD)
        for side in range(2):
            if not (
                init_type(side, types[side])
                and init_kine(side, dh[side])
                and init_limits(side, pnva[side], bd[side])
            ):
                raise OSError(f"Marvin could not initialize kinematics for arm {side}")
        self._joint_bounds = np.sort(np.array(pnva)[:, :, :2], axis=2)
        self._wrist_limits = np.array(bd)
        self._fk = self._bind("FX_Robot_Kine_FK", ct.c_long, ct.POINTER(_Joints), ct.POINTER(_Pose))
        self._ik = self._bind("FX_Robot_Kine_IK", ct.c_long, ct.POINTER(_IKParameters))

    def _bind(self, name, *argtypes):
        function = getattr(self._library, name)
        function.argtypes = argtypes
        function.restype = ct.c_bool
        return function

    def forward(self, side: int, joints: np.ndarray) -> np.ndarray:
        pose = _Pose()
        if not self._fk(side, ct.byref(_Joints(*joints)), ct.byref(pose)):
            raise ValueError(f"Marvin FK failed for arm {side}")
        result = np.array(pose)
        result[:3, 3] *= 0.001
        if not np.isfinite(result).all():
            raise ValueError(f"Marvin FK returned invalid pose for arm {side}")
        return result

    def solve_local(self, side, target, reference, max_step_deg, *, measured=None):
        """Fit a small joint step to G1D wrist position, rotation and joint continuity costs."""
        if measured is None:
            measured = reference
        bounds = self._joint_bounds[side]
        lower = np.maximum(bounds[:, 0], measured - max_step_deg) - reference
        upper = np.minimum(bounds[:, 1], measured + max_step_deg) - reference
        if np.any(lower >= upper):
            return None

        home_error = np.deg2rad(reference - self._home_deg[side])

        posture_scale = self._posture_scale
        if posture_scale:
            current = self.forward(side, reference)
            distance = np.linalg.norm(current[:3, 3] - target[:3, 3])
            angle = Rotation.from_matrix(current[:3, :3] @ target[:3, :3].T).magnitude()
            # Taper the posture cost inside 1 cm / 5 degrees so a satisfied
            # wrist target cannot drive null-space motion over successive ticks.
            posture_scale *= min(1.0, max(distance / 0.01, angle / np.deg2rad(5)))

        def residual(step):
            pose = self.forward(side, reference + np.rad2deg(step))
            angular = Rotation.from_matrix(pose[:3, :3] @ target[:3, :3].T).as_rotvec()
            tracking = np.r_[np.sqrt(50) * (pose[:3, 3] - target[:3, 3]), angular, np.sqrt(0.1) * step]
            if posture_scale:
                return np.r_[tracking, posture_scale * (home_error + step)]
            return tracking

        rad_bounds = np.deg2rad(lower), np.deg2rad(upper)
        initial = np.clip(np.zeros(7), *rad_bounds)
        solution = least_squares(residual, initial, bounds=rad_bounds, max_nfev=16, gtol=1e-7)
        if not solution.success or not np.isfinite(solution.x).all():
            return None
        # Respect both polynomials on each side of the M6 wrist envelope.
        for fraction in (1, 0.5, 0.25, 0.125):
            step_deg = fraction * np.rad2deg(solution.x)
            if np.any(step_deg < lower - 1e-9) or np.any(step_deg > upper + 1e-9):
                continue
            candidate = reference + step_deg
            j6, j7 = candidate[-2:]
            limits = self._wrist_limits[side] @ np.array([j6 * j6, j6, 1.0])
            if max(limits[2:]) <= j7 <= min(limits[:2]):
                return candidate
        return None

    def inverse(self, side: int, pose: np.ndarray, seed: np.ndarray, mode: int) -> np.ndarray | None:
        target = pose.copy()
        target[:3, 3] *= 1000
        params = _IKParameters()
        params.target = _Pose(*(tuple(row) for row in target))
        params.reference = _Joints(*seed)
        params.mode = mode
        normal = np.array(_DEFAULT_NORMALS[side])
        params.normal = _ZSP(*(normal / np.linalg.norm(normal)), 0, 0, 0)
        params.angle = 0
        params.dgr[:] = (5, 5, 5)
        if self._ik(side, ct.byref(params)) and not (params.out_of_range or params.joint_exceeded):
            joints = np.array(params.joints)
            if np.isfinite(joints).all():
                return joints
        return None
