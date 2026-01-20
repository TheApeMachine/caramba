"""Format oracle

Provides ground-truth encoder/decoder for unknown format samples.
This is used to generate supervision, tests, and scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lab.unknown_format.format_family import FormatSpec


@dataclass(frozen=True, slots=True)
class Record:
    """One decoded record."""

    payload: bytes

    def to_json(self) -> dict[str, Any]:
        return {"payload_hex": self.payload.hex()}


class FormatOracle:
    """Oracle encoder/decoder for FormatSpec."""

    def encode(self, *, spec: FormatSpec, records: list[Record]) -> bytes:
        spec.validate()
        if len(records) != int(spec.record_count):
            raise ValueError("records length must equal spec.record_count")
        out = bytearray()
        out.append(int(spec.version))
        out.append(int(spec.record_count))
        for rec in records:
            payload = bytes(rec.payload)
            if len(payload) > 255:
                raise ValueError("payload too long for u8 length")
            out.append(len(payload))
            out.extend(payload)
            out.append(self.checksum(payload))
        return bytes(out)

    def decode(self, *, spec: FormatSpec, raw: bytes) -> list[Record]:
        spec.validate()
        if not isinstance(raw, (bytes, bytearray)):
            raise TypeError("raw must be bytes-like")
        buf = bytes(raw)
        if len(buf) < 2:
            raise ValueError("raw too short")
        if buf[0] != int(spec.version):
            raise ValueError("version mismatch")
        if buf[1] != int(spec.record_count):
            raise ValueError("record_count mismatch")
        pos = 2
        records: list[Record] = []
        for _ in range(int(spec.record_count)):
            if pos >= len(buf):
                raise ValueError("truncated record")
            n = int(buf[pos])
            pos += 1
            if pos + n + 1 > len(buf):
                raise ValueError("truncated payload/checksum")
            payload = buf[pos : pos + n]
            pos += n
            chk = int(buf[pos])
            pos += 1
            if chk != self.checksum(payload):
                raise ValueError("checksum mismatch")
            records.append(Record(payload=payload))
        if pos != len(buf):
            raise ValueError("extra bytes after records")
        return records

    def checksum(self, payload: bytes) -> int:
        """Simple additive checksum over payload bytes."""
        return int(sum(int(b) for b in payload) % 256)

