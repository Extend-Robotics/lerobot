"""Compile and run the actual native hand parser with a stubbed SDK transport."""

import shutil
import subprocess
import sysconfig
from pathlib import Path

import pytest


def test_native_hand_packet_invalidation(tmp_path):
    pybind11 = pytest.importorskip("pybind11")
    compiler = shutil.which("c++")
    root = Path(__file__).parents[2]
    json_include = root / "scripts/pico4/.runtime/json-install/include"
    if (
        not compiler
        or not (json_include / "nlohmann/json.hpp").exists()
        or not (root / "scripts/pico4/sdk/include/PXREARobotSDK.h").exists()
    ):
        pytest.skip("Native PICO build dependencies are not installed")
    libdir = sysconfig.get_config_var("LIBDIR")
    library = sysconfig.get_config_var("LDVERSION")
    executable = tmp_path / "pico_hand_snapshot"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O0",
            "-pthread",
            f"-I{pybind11.get_include()}",
            f"-I{sysconfig.get_path('include')}",
            f"-I{json_include}",
            f"-I{root / 'scripts/pico4/sdk/include'}",
            str(Path(__file__).parent / "native/pico_hand_snapshot.cpp"),
            f"-L{libdir}",
            f"-lpython{library}",
            f"-Wl,-rpath,{libdir}",
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    subprocess.run([str(executable)], check=True, capture_output=True, text=True, timeout=10)
