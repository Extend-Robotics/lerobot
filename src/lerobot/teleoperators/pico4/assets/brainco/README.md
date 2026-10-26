BrainCo hand kinematic models derived from
https://github.com/unitreerobotics/xr_teleoperate/tree/817fb00c63cde15e5f24a0f8fa08e1e33ed89d3b/assets/brainco_hand
under the accompanying MIT LICENSE.

Visual and collision elements were removed, and meshes are not bundled.
Joint/link kinematics, inertias, limits and mimic joints are preserved.
The mapping in revo2_retargeting.py follows that revision's brainco.yml and
robot_hand_brainco.py. It uses dex-retargeting 0.5.0's unified
target_link_human_indices field instead of the older fork's suffixed field.
