"""Single-part camera transport: NUL-terminated topic, versioned header, JPEG.

Every frame is one ZMQ message so SUB filtering precedes CONFLATE. Server and
client must both use this protocol; the previous JSON protocol is unsupported.
The timestamp is server wall time, not a clock-synchronized freshness guarantee.
"""

import struct

_HEADER = struct.Struct("!4s16sQd")
_MAGIC = b"LZQ1"


def camera_topic(name: str) -> bytes:
    """Include a delimiter so camera names cannot match longer name prefixes."""
    if not name or "\0" in name:
        raise ValueError("Camera name must be nonempty and contain no NUL characters")
    return name.encode("utf-8") + b"\0"


def pack_frame(name: str, session: bytes, sequence: int, timestamp: float, jpeg: bytes) -> bytes:
    """Serialize one frame, including a session ID that changes on server restart."""
    if len(session) != 16:
        raise ValueError("Session ID must contain 16 bytes")
    return camera_topic(name) + _HEADER.pack(_MAGIC, session, sequence, timestamp) + jpeg


def unpack_frame(message: bytes, name: str) -> tuple[bytes, int, float, bytes]:
    """Validate the topic/version and return session, sequence, time, and JPEG."""
    topic = camera_topic(name)
    if not message.startswith(topic):
        raise ValueError("Unexpected camera topic")
    offset = len(topic)
    if len(message) <= offset + _HEADER.size:
        raise ValueError("Truncated camera frame")
    magic, session, sequence, timestamp = _HEADER.unpack_from(message, offset)
    if magic != _MAGIC:
        raise ValueError("Unsupported camera protocol version")
    return session, sequence, timestamp, message[offset + _HEADER.size :]
