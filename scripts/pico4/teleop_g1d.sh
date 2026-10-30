#!/usr/bin/env bash
# Use the existing Unitree environment, shared with the PICO SDK.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
PYTHON=${PICO_G1_PYTHON:-$ROOT/../envs/unitree-g1/bin/python}
[[ -x "$PYTHON" ]] || {
    echo 'Set PICO_G1_PYTHON to a Python environment containing LeRobot, Unitree SDK, Pinocchio/CasADi and PICO SDK.' >&2
    exit 1
}
cd "$ROOT"
exec uv run --no-project --python "$PYTHON" python examples/unitree_g1d/teleop_pico4.py "$@"
