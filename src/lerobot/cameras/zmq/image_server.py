#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""
Streams camera images over ZMQ.
Uses lerobot's OpenCVCamera for capture, publishes binary JPEG frames on camera-specific topics.
"""

import argparse
import contextlib
import logging
import threading
import time
from uuid import uuid4

import cv2
import numpy as np
import zmq

from ..configs import ColorMode
from ..opencv import OpenCVCamera, OpenCVCameraConfig
from .protocol import camera_topic, pack_frame

logger = logging.getLogger(__name__)


def encode_image(image: np.ndarray, quality: int = 80, color_mode: ColorMode = ColorMode.RGB) -> bytes:
    """Encode RGB or native OpenCV BGR pixels to binary JPEG."""
    if color_mode == ColorMode.RGB:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif color_mode == ColorMode.BGR:
        bgr = image
    else:
        raise ValueError(f"Unsupported color mode: {color_mode}")
    success, buffer = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        raise RuntimeError("Failed to encode camera image")
    return buffer.tobytes()


class CameraCaptureThread:
    """Background thread that continuously captures and encodes frames from a camera."""

    def __init__(self, camera: OpenCVCamera, name: str, frame_ready: threading.Event) -> None:
        self.camera = camera
        self.name = name
        self.frame_ready = frame_ready
        self.stop_event = threading.Event()
        self.latest_encoded: bytes | None = None
        self.latest_sequence = 0
        self.latest_timestamp: float = 0.0
        self.frame_lock = threading.Lock()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the capture thread."""
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Stop the capture thread."""
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=1.0)

    def _capture_loop(self) -> None:
        """Encode the newest unprocessed frame and wake the publisher immediately."""
        last_capture_timestamp = float("-inf")
        while not self.stop_event.is_set():
            try:
                # Unlike read(), async_read() preserves a frame notification that
                # arrived while encoding. Keep waits short so shutdown is bounded.
                self.camera.async_read(timeout_ms=100)
                # Hardware may have advanced again since async_read returned. Take
                # the newest image and its capture timestamp as one atomic snapshot.
                with self.camera.frame_lock:
                    frame = self.camera.latest_frame
                    captured_at = self.camera.latest_timestamp
                if frame is None or captured_at is None or captured_at <= last_capture_timestamp:
                    continue
                # OpenCVCamera timestamps after hardware capture/postprocessing using
                # perf_counter. Translate to wall time without hiding buffer age.
                timestamp = time.time() - (time.perf_counter() - captured_at)
                encoded = encode_image(frame, color_mode=ColorMode.BGR)
                with self.frame_lock:
                    self.latest_encoded = encoded
                    self.latest_timestamp = timestamp
                    self.latest_sequence += 1
                last_capture_timestamp = captured_at
                self.frame_ready.set()
            except TimeoutError:
                continue
            except Exception as e:
                logger.warning(f"Camera {self.name} capture error: {e}")
                self.stop_event.wait(0.1)

    def get_latest(self) -> tuple[bytes | None, float, int]:
        """Get the latest encoded frame and its timestamp."""
        with self.frame_lock:
            return self.latest_encoded, self.latest_timestamp, self.latest_sequence


class ImageServer:
    def __init__(self, config: dict, port: int = 5555):
        # Configure capture FPS; encoded frames are published as soon as they are ready.
        self.fps = config.get("fps", 30)
        self.session = uuid4().bytes
        self.frame_ready = threading.Event()
        for name in config.get("cameras", {}):
            camera_topic(name)  # Validate topics before opening devices.
        self.cameras: dict[str, OpenCVCamera] = {}
        self.capture_threads: dict[str, CameraCaptureThread] = {}

        for name, cfg in config.get("cameras", {}).items():
            shape = cfg.get("shape", [480, 640])
            cam_config = OpenCVCameraConfig(
                index_or_path=cfg.get("device_id", 0),
                fps=self.fps,
                width=shape[1],
                height=shape[0],
                fourcc=cfg.get("fourcc", "MJPG"),
                color_mode=ColorMode.BGR,
            )
            camera = OpenCVCamera(cam_config)
            camera.connect()
            self.cameras[name] = camera
            logger.info(f"Camera {name}: {shape[1]}x{shape[0]}")

            # Create capture thread for this camera
            capture_thread = CameraCaptureThread(camera, name, self.frame_ready)
            self.capture_threads[name] = capture_thread

        # ZMQ PUB socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 20)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(f"tcp://*:{port}")

        logger.info(f"ImageServer running on port {port}")

    def run(self) -> None:
        last_published_sequence: dict[str, int] = {}
        waiting_for_first_frame = set(self.capture_threads)
        deadline = time.monotonic() + 10

        try:
            for capture_thread in self.capture_threads.values():
                capture_thread.start()
            logger.info("Waiting for cameras to start capturing...")

            while True:
                # A shared event wakes the single socket owner for any camera.
                # Clear BEFORE inspecting slots: a capture during/after the scan
                # then leaves the event set for the next iteration (no lost wake).
                self.frame_ready.wait(timeout=0.1)
                self.frame_ready.clear()
                for name, capture_thread in self.capture_threads.items():
                    encoded, timestamp, sequence = capture_thread.get_latest()
                    if encoded is None or sequence <= last_published_sequence.get(name, 0):
                        continue
                    if name in waiting_for_first_frame:
                        waiting_for_first_frame.remove(name)
                        logger.info(f"Camera {name} ready (capture + encode in background)")
                    message = pack_frame(name, self.session, sequence, timestamp, encoded)
                    with contextlib.suppress(zmq.Again):
                        self.socket.send(message, zmq.NOBLOCK)
                        last_published_sequence[name] = sequence

                # Healthy cameras publish immediately even if another camera is
                # slow to start. A camera that never starts still fails startup.
                if waiting_for_first_frame and time.monotonic() >= deadline:
                    missing = ", ".join(sorted(waiting_for_first_frame))
                    raise TimeoutError(f"Cameras {missing} did not produce a frame within 10 seconds")

        except KeyboardInterrupt:
            pass
        finally:
            for capture_thread in self.capture_threads.values():
                capture_thread.stop()
            for cam in self.cameras.values():
                cam.disconnect()
            self.socket.close()
            self.context.term()


def main() -> None:
    """Publish named cameras from the robot without starting the motor bridge."""
    parser = argparse.ArgumentParser(description="Stream robot cameras to remote LeRobot clients")
    parser.add_argument("--device", default="4", help="Camera index or stable /dev/v4l/by-id/... device path")
    parser.add_argument(
        "--camera",
        action="append",
        metavar="NAME=DEVICE",
        help="Named camera index/path; repeat for multiple cameras. Replaces the default --device head camera.",
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Camera capture FPS; publish immediately after encoding"
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--fourcc", default="MJPG", help="Capture format, e.g. MJPG or YUYV")
    args = parser.parse_args()
    if min(args.fps, args.width, args.height) <= 0 or not 1 <= args.port <= 65535:
        parser.error("FPS and dimensions must be positive; port must be between 1 and 65535")
    cameras = {}
    for spec in args.camera or [f"head_camera={args.device}"]:
        name, separator, device = spec.partition("=")
        if not separator or not name.strip() or not device.strip():
            parser.error("Each --camera must be NAME=DEVICE with a nonempty name and device")
        if name in cameras:
            parser.error(f"Duplicate camera name: {name}")
        cameras[name] = {
            "device_id": int(device) if device.isdecimal() else device,
            "shape": [args.height, args.width],
            "fourcc": args.fourcc,
        }
    logging.basicConfig(level=logging.INFO)
    config = {"fps": args.fps, "cameras": cameras}
    ImageServer(config, port=args.port).run()


if __name__ == "__main__":
    main()
