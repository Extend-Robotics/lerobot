"""Remote camera transport tests using synthetic images and a local ZMQ publisher."""

import time
from contextlib import ExitStack
from queue import Queue
from threading import Event, Lock, Thread

import numpy as np
import pytest

zmq = pytest.importorskip("zmq")

from lerobot.cameras.configs import ColorMode  # noqa: E402
from lerobot.cameras.zmq import ZMQCamera, ZMQCameraConfig  # noqa: E402
from lerobot.cameras.zmq.image_server import encode_image  # noqa: E402
from lerobot.cameras.zmq.protocol import camera_topic, pack_frame, unpack_frame  # noqa: E402


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
            sequence = 0
            while not stopped.wait(0.01):
                sequence += 1
                update = message[0]
                for name, jpeg in update["images"].items():
                    socket.send(
                        pack_frame(
                            name,
                            update.get("session", b"a" * 16),
                            update.get("sequence", sequence),
                            time.time(),
                            jpeg,
                        )
                    )

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


def test_three_named_views_remain_distinct(publisher):
    port, message = publisher
    names = ("head_camera", "left_wrist_camera", "right_wrist_camera")
    images = {}
    for channel, name in enumerate(names):
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        frame[:, :, channel] = 255
        images[name] = encode_image(frame)
    message[0] = {"images": images}
    with ExitStack() as stack:
        cameras = [
            stack.enter_context(
                ZMQCamera(ZMQCameraConfig(server_address="127.0.0.1", port=port, camera_name=name))
            )
            for name in names
        ]
        for channel, camera in enumerate(cameras):
            frame = camera.async_read(timeout_ms=1000)
            assert frame.shape == (24, 32, 3)
            assert frame[:, :, channel].mean() > 240
            assert np.delete(frame, channel, axis=2).mean() < 10


def test_publisher_cli_accepts_named_devices(monkeypatch):
    from unittest.mock import Mock

    from lerobot.cameras.zmq import image_server

    server = Mock()
    monkeypatch.setattr(image_server, "ImageServer", server)
    monkeypatch.setattr(
        "sys.argv",
        [
            "image_server",
            "--camera",
            "head_camera=4",
            "--camera",
            "left_wrist_camera=/dev/v4l/by-path/left",
            "--camera",
            "right_wrist_camera=/dev/v4l/by-id/right",
        ],
    )
    image_server.main()
    cameras = server.call_args.args[0]["cameras"]
    assert cameras["head_camera"]["device_id"] == 4
    assert cameras["left_wrist_camera"]["device_id"] == "/dev/v4l/by-path/left"
    assert cameras["right_wrist_camera"]["device_id"] == "/dev/v4l/by-id/right"
    server.return_value.run.assert_called_once()


def test_publisher_cli_preserves_single_head_mode(monkeypatch):
    from unittest.mock import Mock

    from lerobot.cameras.zmq import image_server

    server = Mock()
    monkeypatch.setattr(image_server, "ImageServer", server)
    monkeypatch.setattr("sys.argv", ["image_server", "--device", "2"])
    image_server.main()
    assert server.call_args.args[0]["cameras"] == {
        "head_camera": {"device_id": 2, "shape": [480, 640], "fourcc": "MJPG"}
    }


@pytest.mark.parametrize(
    "options",
    [
        ["--camera", "head"],
        ["--camera", "=0"],
        ["--camera", "head="],
        ["--camera", "head=0", "--camera", "head=2"],
    ],
)
def test_publisher_cli_rejects_invalid_camera_mapping(monkeypatch, options):
    from unittest.mock import Mock

    from lerobot.cameras.zmq import image_server

    server = Mock()
    monkeypatch.setattr(image_server, "ImageServer", server)
    monkeypatch.setattr("sys.argv", ["image_server", *options])
    with pytest.raises(SystemExit) as error:
        image_server.main()
    assert error.value.code == 2
    server.assert_not_called()


def test_duplicate_frames_stay_stale_and_new_session_recovers(publisher):
    port, message = publisher
    camera = make_camera(port)
    try:
        camera.connect()
        camera.async_read(timeout_ms=1000)
        jpeg = message[0]["images"]["head_camera"]
        message[0] = {"images": {"head_camera": jpeg}, "sequence": 100000}
        camera.async_read(timeout_ms=1000)
        time.sleep(0.15)
        with pytest.raises(TimeoutError, match="latest frame is too old"):
            camera.read_latest(max_age_ms=50)
        # A restarted publisher may reset its sequence to a smaller value.
        message[0] = {"images": {"head_camera": jpeg}, "sequence": 1, "session": b"b" * 16}
        camera.new_frame_event.clear()
        assert camera.async_read(timeout_ms=1000).shape == (24, 32, 3)
    finally:
        camera.disconnect()


def test_camera_topics_are_exact_and_conflation_keeps_latest_own_frame():
    """Pause consumers while other camera messages arrive after their own frame."""
    commands = Queue()
    endpoints = Queue()
    ready = Event()
    finished = Event()
    stopped = Event()

    def publish():
        with zmq.Context() as context, context.socket(zmq.PUB) as socket:
            endpoints.put(socket.bind_to_random_port("tcp://127.0.0.1"))
            while not stopped.wait(0.01):
                if not ready.is_set():
                    for name in ("head", "wrist"):
                        socket.send(pack_frame(name, b"a" * 16, 0, 0.0, b"ready"))
                elif not commands.empty():
                    for name, sequence in commands.get():
                        socket.send(pack_frame(name, b"a" * 16, sequence, 0.0, b"jpeg"))
                    finished.set()

    thread = Thread(target=publish)
    thread.start()
    try:
        port = endpoints.get(timeout=5)
        with zmq.Context() as context, ExitStack() as stack:
            subscribers = []
            for name in ("head", "wrist"):
                socket = stack.enter_context(context.socket(zmq.SUB))
                socket.setsockopt(zmq.CONFLATE, 1)
                socket.setsockopt(zmq.SUBSCRIBE, camera_topic(name))
                socket.setsockopt(zmq.RCVTIMEO, 2000)
                socket.connect(f"tcp://127.0.0.1:{port}")
                subscribers.append(socket)
            for socket in subscribers:
                socket.recv()  # Confirm subscription propagation before the burst.
            ready.set()
            commands.put([("head", 1), ("head", 2), ("wrist", 1), ("head_extra", 99)])
            assert finished.wait(2)
            time.sleep(0.1)  # Let the burst reach the paused subscribers.
            for socket, name, sequence in zip(subscribers, ("head", "wrist"), (2, 1), strict=True):
                assert unpack_frame(socket.recv(), name)[1] == sequence
                assert not socket.getsockopt(zmq.RCVMORE)
    finally:
        stopped.set()
        thread.join(timeout=2)
        assert not thread.is_alive()


@pytest.mark.parametrize("name", ["", "head\0extra"])
def test_invalid_camera_topics_are_rejected(name):
    with pytest.raises(ValueError, match="Camera name"):
        camera_topic(name)


def test_invalid_protocol_frames_are_rejected():
    with pytest.raises(ValueError, match="Truncated"):
        unpack_frame(camera_topic("head") + b"short", "head")
    message = pack_frame("head", b"a" * 16, 1, 1.0, b"jpeg")
    with pytest.raises(ValueError, match="Unexpected camera"):
        unpack_frame(message, "wrist")
    with pytest.raises(ValueError, match="Unsupported"):
        unpack_frame(message.replace(b"LZQ1", b"LZQ0", 1), "head")


def test_server_publishes_each_camera_once_per_sequence(monkeypatch):
    from unittest.mock import Mock

    from lerobot.cameras.zmq import image_server

    server = image_server.ImageServer.__new__(image_server.ImageServer)
    server.fps = 1  # Publication must not wait for a periodic FPS timer.
    server.session = b"a" * 16
    server.socket = Mock()
    server.context = Mock()
    names = ("head", "left_wrist", "right_wrist")
    server.cameras = {name: Mock() for name in names}
    server.capture_threads = {name: Mock() for name in names}
    for capture in server.capture_threads.values():
        capture.get_latest.return_value = (b"jpeg", 100.0, 1)

    class WakeEvent(Event):
        calls = 0

        def wait(self, timeout=None):
            self.calls += 1
            if self.calls == 2:
                # The new head arrived after its slot had already been scanned.
                # Its notification must survive until the next publisher wait.
                assert self.is_set()
            if self.calls == 4:
                raise KeyboardInterrupt
            return self.is_set()

    server.frame_ready = WakeEvent()
    server.frame_ready.set()

    def send(message, flags):
        if message.startswith(camera_topic("right_wrist")):
            # New frame despite backwards wall time, arriving during publication.
            server.capture_threads["head"].get_latest.return_value = (b"new-jpeg", 50.0, 2)
            server.frame_ready.set()

    server.socket.send.side_effect = send

    def unexpected_sleep(_):
        pytest.fail("Publisher must wait for frame notifications, not an FPS timer")

    monkeypatch.setattr(image_server.time, "sleep", unexpected_sleep)
    server.run()
    messages = [call.args[0] for call in server.socket.send.call_args_list]
    assert len(messages) == 4  # Three initial frames, then only the changed head.
    for name, message in zip(names, messages[:3], strict=True):
        assert unpack_frame(message, name) == (server.session, 1, 100.0, b"jpeg")
    assert unpack_frame(messages[3], "head") == (server.session, 2, 50.0, b"new-jpeg")
    for capture in server.capture_threads.values():
        capture.stop.assert_called_once()
    server.socket.close.assert_called_once()


def test_native_bgr_encoding_preserves_colours_without_conversion(monkeypatch):
    import cv2

    bgr = np.zeros((24, 32, 3), dtype=np.uint8)
    bgr[:, :, 2] = 255

    def unexpected_conversion(*args, **kwargs):
        pytest.fail("BGR pixels must pass directly to JPEG encoding")

    monkeypatch.setattr(cv2, "cvtColor", unexpected_conversion)
    jpeg = encode_image(bgr, color_mode=ColorMode.BGR)
    decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert decoded[:, :, 2].mean() > 240
    assert decoded[:, :, :2].mean() < 10


def test_capture_uses_newest_pending_frame_and_encodes_each_capture_once(monkeypatch):
    from lerobot.cameras.zmq import image_server

    class BufferedCamera:
        frame_lock = Lock()
        latest_frame = np.full((2, 2, 3), 1, dtype=np.uint8)
        latest_timestamp = 4999.75
        calls = 0

        def async_read(self, timeout_ms):
            self.calls += 1
            assert timeout_ms <= 100
            if self.calls == 3:
                # Simulate a late notification for the image already encoded.
                worker.stop_event.set()
            return self.latest_frame

        def read(self):
            pytest.fail("Capture must preserve pending frame notifications")

    camera = BufferedCamera()
    ready = Event()
    worker = image_server.CameraCaptureThread(camera, "head", ready)
    encoded_frames = []

    def encode(frame, *, color_mode):
        assert color_mode == ColorMode.BGR
        encoded_frames.append(int(frame[0, 0, 0]))
        if len(encoded_frames) == 1:
            # Two hardware frames arrive while the first JPEG is being encoded.
            # The next iteration should encode only the newest one immediately.
            with camera.frame_lock:
                camera.latest_frame = np.full((2, 2, 3), 3, dtype=np.uint8)
                camera.latest_timestamp = 4999.85
        return bytes([int(frame[0, 0, 0])])

    monkeypatch.setattr(image_server, "encode_image", encode)
    monkeypatch.setattr(image_server.time, "time", lambda: 1000.0)
    monkeypatch.setattr(image_server.time, "perf_counter", lambda: 5000.0)
    worker._capture_loop()
    assert encoded_frames == [1, 3]
    assert worker.get_latest() == (b"\x03", pytest.approx(999.85), 2)
    assert ready.is_set()


def test_capture_stops_while_camera_is_stalled():
    from lerobot.cameras.zmq.image_server import CameraCaptureThread

    waiting = Event()

    class StalledCamera:
        def async_read(self, timeout_ms):
            waiting.set()
            Event().wait(timeout_ms / 1000)
            raise TimeoutError("No frame")

    worker = CameraCaptureThread(StalledCamera(), "head", Event())
    worker.start()
    try:
        assert waiting.wait(1)
    finally:
        worker.stop()
    assert not worker.thread.is_alive()


def test_server_publishes_healthy_camera_before_startup_timeout(monkeypatch):
    from unittest.mock import Mock

    from lerobot.cameras.zmq import image_server

    server = image_server.ImageServer.__new__(image_server.ImageServer)
    server.session = b"a" * 16
    server.socket = Mock()
    server.context = Mock()
    server.frame_ready = Mock()
    server.cameras = {"head": Mock(), "stalled": Mock()}
    server.capture_threads = {"head": Mock(), "stalled": Mock()}
    server.capture_threads["head"].get_latest.return_value = (b"jpeg", 100.0, 1)
    server.capture_threads["stalled"].get_latest.return_value = (None, 0.0, 0)
    monkeypatch.setattr(image_server.time, "monotonic", iter([0.0, 11.0]).__next__)
    with pytest.raises(TimeoutError, match="stalled.*within 10 seconds"):
        server.run()
    assert server.socket.send.call_count == 1
    assert unpack_frame(server.socket.send.call_args.args[0], "head")[1] == 1
    for capture in server.capture_threads.values():
        capture.stop.assert_called_once()
    for camera in server.cameras.values():
        camera.disconnect.assert_called_once()
    server.socket.close.assert_called_once()
