"""Binary stream parser

Incrementally parses byte streams into BinaryFrame instances using the Binary codec.
This avoids repeatedly attempting full JSON parsing during streaming decode.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from caramba.core.event_codec.binary_codec import BinaryFrame


@dataclass(slots=True)
class BinaryStreamParser:
    """Incremental binary frame parser.

    Feed bytes in chunks and collect fully parsed frames.
    """

    buffer: bytearray = field(default_factory=bytearray)

    def feed(self, chunk: bytes) -> list[BinaryFrame]:
        """Feed bytes and return any complete frames parsed."""
        if not isinstance(chunk, (bytes, bytearray)):
            raise TypeError(f"chunk must be bytes-like, got {type(chunk).__name__}")
        self.buffer.extend(chunk)
        frames: list[BinaryFrame] = []
        while True:
            if len(self.buffer) < 5:
                break
            type_id = int(self.buffer[0])
            length = int.from_bytes(self.buffer[1:5], byteorder="little", signed=False)
            total = 5 + int(length)
            if len(self.buffer) < total:
                break
            payload = bytes(self.buffer[5:total])
            frames.append(BinaryFrame(type_id=type_id, payload=payload))
            del self.buffer[:total]
        return frames

