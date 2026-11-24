#!/usr/bin/env bash
# Install only PICO runtime/build requirements into the selected environment.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
PYTHON=${PICO_PYTHON:-$ROOT/.venv/bin/python}
RUNTIME="$HERE/.runtime"
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    echo 'This installer supports Linux x86_64 and XenseVR 0.2.1.' >&2
    exit 1
fi
[[ -x "$PYTHON" ]] || { echo "Missing environment: $PYTHON. Run uv sync first." >&2; exit 1; }
mkdir -p "$RUNTIME"
if [[ ! -f "$RUNTIME/xensevr.deb" ]]; then
    curl -fL https://github.com/XenseRobotics-AI/XenseVR-PC-Service/releases/download/v0.2.1/XenseVR-PC-Service_0.2.1_amd64.deb -o "$RUNTIME/xensevr.deb.tmp"
    mv "$RUNTIME/xensevr.deb.tmp" "$RUNTIME/xensevr.deb"
fi
dpkg-deb -x "$RUNTIME/xensevr.deb" "$RUNTIME/service"
SDK="$RUNTIME/service/opt/apps/roboticsservice/SDK"
mkdir -p "$HERE/sdk/include" "$HERE/sdk/lib"
cp "$SDK/include/PXREARobotSDK.h" "$HERE/sdk/include/"
cp "$SDK/x64/libPXREARobotSDK.so" "$HERE/sdk/lib/"
uv pip install --python "$PYTHON" 'setuptools>=71,<81' wheel pybind11 'cmake>=3.29,<4' ninja
export PATH="$(dirname "$PYTHON"):$PATH"
if [[ ! -f "$RUNTIME/json.tar.xz" ]]; then
    curl -fL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz -o "$RUNTIME/json.tar.xz.tmp"
    mv "$RUNTIME/json.tar.xz.tmp" "$RUNTIME/json.tar.xz"
fi
tar -xf "$RUNTIME/json.tar.xz" -C "$RUNTIME"
cmake -S "$RUNTIME/json" -B "$RUNTIME/json-build" -DJSON_BuildTests=OFF -DCMAKE_INSTALL_PREFIX="$RUNTIME/json-install"
cmake --install "$RUNTIME/json-build"
export CMAKE_PREFIX_PATH="$RUNTIME/json-install:$($PYTHON -m pybind11 --cmakedir)${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
uv pip install --python "$PYTHON" "$HERE/sdk" --no-build-isolation
"$PYTHON" -c 'import xensevr_pc_service_sdk; print("PICO SDK import OK")'
