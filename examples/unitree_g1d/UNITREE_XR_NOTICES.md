# Unitree XR source notices

Ported sources:

- `xr_teleoperate` revision `817fb00c63cde15e5f24a0f8fa08e1e33ed89d3b`: `teleop/robot_control/robot_arm_ik.py`, `teleop/robot_control/robot_arm.py`, `teleop/utils/weighted_moving_filter.py`.
- Its pinned `televuer` submodule revision `766de45e74373ae0ea66321d942ce538385655a5`: controller/head-yaw mapping in `src/televuer/tv_wrapper.py`.

Modified to use LeRobot's model loading, explicit joint-order conversion, native PICO SDK poses and the existing socket bridge. Preview, arm-only G1-D body handling and fault cleanup are adapters; upstream mapping/IK/control coefficients are preserved. Numerical parity was checked against the pinned upstream IK class, filter and gravity feedforward on an eight-target sequence.

## xr_teleoperate license

Copyright [2025] [HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics")]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

------------------

This code builds upon following open-source code-bases. Please visit the URLs to see the respective LICENSES:

1) https://github.com/OpenTeleVision/TeleVision
2) https://github.com/dexsuite/dex-retargeting
3) https://github.com/vuer-ai/vuer
4) https://github.com/stack-of-tasks/pinocchio
5) https://github.com/casadi/casadi
6) https://github.com/meshcat-dev/meshcat-python
7) https://github.com/zeromq/pyzmq
8) https://github.com/Dingry/BunnyVisionPro
9) https://github.com/unitreerobotics/unitree_sdk2_python

------------------

## televuer license

MIT License

Copyright (c) 2025, Unitree Robotics

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

