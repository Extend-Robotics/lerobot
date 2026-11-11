# Copyright 2025 The HuggingFace & XenseRobotics Inc. team. All rights reserved.
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

import numpy as np


def slerp_quaternion(q1: np.ndarray, q2: np.ndarray, t: float, input_format: str = "wxyz") -> np.ndarray:
    """Spherical Linear Interpolation (SLERP) between two quaternions.

    Args:
        q1: First quaternion [qw, qx, qy, qz]
        q2: Second quaternion [qw, qx, qy, qz]
        t: Interpolation factor [0, 1], where 0 returns q1 and 1 returns q2
        input_format: Input quaternion format:
            - "wxyz": [qw, qx, qy, qz] format (Flexiv, scipy)
            - "xyzw": [qx, qy, qz, qw] format (Pico4, ROS, OpenGL)
            Default is "wxyz".

    Returns:
        Interpolated quaternion [qw, qx, qy, qz]
    """
    q1 = normalize_quaternion(q1, input_format=input_format)
    q2 = normalize_quaternion(q2, input_format=input_format)

    dot = np.dot(q1, q2)

    if dot < 0.0:
        q2 = -q2
        dot = -dot

    dot = np.clip(dot, -1.0, 1.0)

    if abs(dot) > 0.9995:
        result = q1 + t * (q2 - q1)
        return normalize_quaternion(result, input_format=input_format)

    theta = np.arccos(abs(dot))
    sin_theta = np.sin(theta)

    w1 = np.sin((1 - t) * theta) / sin_theta
    w2 = np.sin(t * theta) / sin_theta

    result = w1 * q1 + w2 * q2

    return normalize_quaternion(result, input_format=input_format)


def normalize_quaternion(q: np.ndarray, input_format: str = "wxyz") -> np.ndarray:
    """Normalize quaternion and convert to [qw, qx, qy, qz] format (Flexiv convention).

    Args:
        q: Quaternion as numpy array with 4 elements
        input_format: Input quaternion format:
            - "wxyz": [qw, qx, qy, qz] format (Flexiv, scipy)
            - "xyzw": [qx, qy, qz, qw] format (Pico4, ROS, OpenGL)

    Returns:
        Normalized quaternion in [qw, qx, qy, qz] format (Flexiv convention)
    """
    q = np.asarray(q, dtype=np.float32)
    if q.ndim > 1:
        q = q.flatten()
    if len(q) != 4:
        raise ValueError(f"Quaternion must have 4 components, got {len(q)}")

    # Check norm and normalize if needed
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        # Invalid quaternion, return identity in input_format
        if input_format == "wxyz":
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        elif input_format == "xyzw":
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        else:
            raise ValueError(f"Unknown input_format: {input_format}. Use 'wxyz' or 'xyzw'.")

    # Skip normalization if already unit quaternion (|norm - 1| < tolerance)
    if abs(norm - 1.0) > 1e-6:
        q = q / norm

    # Convert to [qw, qx, qy, qz] format
    if input_format == "wxyz":
        # Already in [qw, qx, qy, qz] format
        return q.astype(np.float32)
    elif input_format == "xyzw":
        # Convert from [qx, qy, qz, qw] to [qw, qx, qy, qz]
        return np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)
    else:
        raise ValueError(f"Unknown input_format: {input_format}. Use 'wxyz' or 'xyzw'.")


def quaternion_to_rotation_6d(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Convert quaternion to 6D rotation representation.

    The 6D representation consists of the first two columns of the rotation matrix,
    which can be used to uniquely reconstruct the full rotation matrix via
    Gram-Schmidt orthogonalization.

    Args:
        qw: Quaternion scalar component
        qx: Quaternion x component
        qy: Quaternion y component
        qz: Quaternion z component

    Returns:
        np.ndarray of shape (6,) containing [r1, r2, r3, r4, r5, r6] where:
        - [r1, r2, r3] is the first column of the rotation matrix
        - [r4, r5, r6] is the second column of the rotation matrix
    """
    # Rotation matrix from quaternion
    # First column
    r1 = 1.0 - 2.0 * (qy * qy + qz * qz)
    r2 = 2.0 * (qx * qy + qz * qw)
    r3 = 2.0 * (qx * qz - qy * qw)

    # Second column
    r4 = 2.0 * (qx * qy - qz * qw)
    r5 = 1.0 - 2.0 * (qx * qx + qz * qz)
    r6 = 2.0 * (qy * qz + qx * qw)

    return np.array([r1, r2, r3, r4, r5, r6], dtype=np.float32)
