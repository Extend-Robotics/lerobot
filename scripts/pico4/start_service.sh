#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SERVICE="$HERE/.runtime/service/opt/apps/roboticsservice"
[[ -x "$SERVICE/RoboticsServiceProcess" ]] || { echo 'Run scripts/pico4/install_sdk.sh first.' >&2; exit 1; }
export LD_LIBRARY_PATH="$SERVICE:$SERVICE/lib:$SERVICE/SDK/x64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export QT_PLUGIN_PATH="$SERVICE/plugins"
export QT_QML_PATH="$SERVICE/qml"
cd "$SERVICE"
exec ./RoboticsServiceProcess
