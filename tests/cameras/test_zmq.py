"""Remote camera transport tests using synthetic images and a local ZMQ publisher."""

import time
from queue import Queue
from threading import Event, Thread

import numpy as np
import pytest

zmq = pytest.importorskip("zmq")

from lerobot.cameras.configs import ColorMode  # noqa: E402
from lerobot.cameras.zmq import ZMQCamera, ZMQCameraConfig  # noqa: E402
from lerobot.cameras.zmq.image_server import encode_image  # noqa: E402


@pytest.fixture
def publisher():
    """Keep the PUB socket in its owner thread; allow changing the published cameras."""
    endpoints = Queue()
    stopped = Event()
    red = np.zeros((24, 32, 3), dtype=np.uint8)
    red[:, :, 0] = 255
    message = [{"images": {"head_camera": encode_image(red)}, "timestamps": {}}]

    def publish():
        with zmq.Context() as context, context.socket(zmq.PUB) as socket:
            port = socket.bind_to_random_port("tcp://127.0.0.1")
            endpoints.put(port)
            while not stopped.wait(0.01):
                socket.send_json(message[0])

    thread = Thread(target=publish)
    thread.start()
    port = endpoints.get(timeout=5)
    yield port, message
    stopped.set()
    thread.join(timeout=2)
    assert not thread.is_alive()


def make_camera(port, **kwargs):
    return ZMQCamera(
        ZMQCameraConfig(
            server_address="127.0.0.1",
            port=port,
            camera_name="head_camera",
            timeout_ms=250,
            warmup_s=0,
            **kwargs,
        )
    )


@pytest.mark.parametrize("color_mode,red_channel", [(ColorMode.RGB, 0), (ColorMode.BGR, 2)])
def test_remote_head_camera_preserves_colours(publisher, color_mode, red_channel):
    port, _ = publisher
    camera = make_camera(port, color_mode=color_mode)
    try:
        camera.connect()
        frame = camera.async_read(timeout_ms=1000)
        assert frame.shape == (24, 32, 3)
        assert (camera.width, camera.height) == (32, 24)
        assert frame[:, :, red_channel].mean() > 240
        assert frame[:, :, 1].mean() < 10
        assert frame[:, :, 2 - red_channel].mean() < 10
    finally:
        camera.disconnect()
    assert camera.thread is None and not camera.is_connected


def test_missing_head_camera_is_not_replaced(publisher):
    port, message = publisher
    message[0] = {"images": {"left_wrist": message[0]["images"]["head_camera"]}}
    camera = make_camera(port)
    with pytest.raises(RuntimeError, match="no frame for 'head_camera'"):
        camera.connect()
    assert camera.thread is None and not camera.is_connected


def test_remote_head_camera_rejects_wrong_resolution(publisher):
    port, _ = publisher
    camera = make_camera(port, width=640)
    with pytest.raises(RuntimeError, match="received 32x24, expected 640xNone"):
        camera.connect()
    assert not camera.is_connected


def test_missing_frames_do_not_refresh_latest_timestamp(publisher):
    port, message = publisher
    camera = make_camera(port)
    try:
        camera.connect()
        camera.async_read(timeout_ms=1000)
        message[0] = {"images": {}}
        time.sleep(0.15)
        with pytest.raises(TimeoutError, match="latest frame is too old"):
            camera.read_latest(max_age_ms=50)
        # Partial messages are normal; receiving the named camera again must recover.
        blue = np.zeros((24, 32, 3), dtype=np.uint8)
        blue[:, :, 2] = 255
        message[0] = {"images": {"head_camera": encode_image(blue)}}
        camera.new_frame_event.clear()
        frame = camera.async_read(timeout_ms=1000)
        assert frame[:, :, 2].mean() > 240
    finally:
        camera.disconnect()
