"""Binary event codec

Defines a minimal binary envelope format for raw-byte experiments:

Frame layout (v1):
  [type: u8][length: u32 little-endian][payload: bytes(length)]

This avoids JSON overhead and preserves arbitrary raw binary payloads.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class BinaryFrame:
    """Binary frame.

    Used as an internal representation for framed binary streams.
    """

    type_id: int
    payload: bytes

    def validate(self) -> None:
        t = int(self.type_id)
        if t < 0 or t > 255:
            raise ValueError(f"BinaryFrame.type_id must be in [0,255], got {t}")
        if not isinstance(self.payload, (bytes, bytearray)):
            raise TypeError(f"BinaryFrame.payload must be bytes-like, got {type(self.payload).__name__}")


class BinaryEventEncoder:
    """Binary encoder.

    Encodes a BinaryFrame to a byte-level tensor.
    """

    def encode_frame(self, frame: BinaryFrame) -> Tensor:
        if not isinstance(frame, BinaryFrame):
            raise TypeError(f"frame must be a BinaryFrame, got {type(frame).__name__}")
        frame.validate()
        payload = bytes(frame.payload)
        n = len(payload)
        if n < 0 or n > 0xFFFFFFFF:
            raise ValueError(f"BinaryFrame payload length out of range: {n}")
        header = bytes([int(frame.type_id)]) + int(n).to_bytes(4, byteorder="little", signed=False)
        out = header + payload
        if not out:
            raise ValueError("Encoded frame produced empty bytes")
        return torch.tensor(list(out), dtype=torch.long)


class BinaryEventDecoder:
    """Binary decoder.

    Decodes a single complete frame from a byte-level tensor.
    """

    def decode_frame(self, ids: Tensor) -> BinaryFrame:
        if not isinstance(ids, Tensor):
            raise TypeError(f"ids must be a Tensor, got {type(ids).__name__}")
        if ids.ndim != 1:
            raise ValueError(f"ids must be 1D, got shape {tuple(ids.shape)}")
        if int(ids.numel()) < 5:
            raise ValueError("ids too short for binary frame header (need >= 5 bytes)")
        vals = ids.detach().cpu().to(dtype=torch.int64).tolist()
        if any((v < 0 or v > 255) for v in vals):
            raise ValueError("Binary frame decoding expects values in [0,255]")
        raw = bytes(int(v) for v in vals)
        type_id = raw[0]
        length = int.from_bytes(raw[1:5], byteorder="little", signed=False)
        if len(raw) != 5 + length:
            raise ValueError(f"Binary frame length mismatch: header says {length}, got {len(raw) - 5}")
        payload = raw[5:]
        return BinaryFrame(type_id=int(type_id), payload=payload)

