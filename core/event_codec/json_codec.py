"""JSON event codec

Encodes/decodes EventEnvelope as canonical JSON bytes. This is human-friendly and
useful for debugging, but does not support arbitrary raw binary payloads.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence

import torch
from torch import Tensor

from caramba.core.event import EventEnvelope


class JsonEventEncoder:
    """JSON event encoder.

    Converts EventEnvelope to UTF-8 bytes and returns them as a 1D int64 tensor.
    """

    def encode(self, event: EventEnvelope) -> Tensor:
        if not isinstance(event, EventEnvelope):
            raise TypeError(f"Expected EventEnvelope, got {type(event).__name__}")
        s = json.dumps(event.to_json_dict(), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        b = s.encode("utf-8")
        if not b:
            raise ValueError("Encoded event produced empty byte string")
        return torch.tensor(list(b), dtype=torch.long)

    def encode_many(self, events: Sequence[EventEnvelope]) -> list[Tensor]:
        return [self.encode(e) for e in events]

    def encode_padded(self, events: Sequence[EventEnvelope], *, pad_id: int = 0) -> tuple[Tensor, Tensor]:
        """Encode a batch to (ids, mask)."""
        if not isinstance(events, Sequence):
            raise TypeError(f"Expected a Sequence of events, got {type(events).__name__}")
        encoded = self.encode_many(events)
        if not encoded:
            raise ValueError("encode_padded requires at least one event")
        lens = [int(t.numel()) for t in encoded]
        max_len = max(lens)
        if max_len <= 0:
            raise ValueError("Encoded events have zero length")
        batch_size = len(encoded)
        pad = int(pad_id)
        ids = torch.full((batch_size, max_len), pad, dtype=torch.long)
        mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
        for index, tokens in enumerate(encoded):
            n = int(tokens.numel())
            ids[index, :n] = tokens
            mask[index, :n] = True
        return ids, mask


class JsonEventDecoder:
    """JSON event decoder.

    Converts 1D byte-level token tensors into EventEnvelope by decoding UTF-8 JSON.
    """

    def decode(self, ids: Tensor) -> EventEnvelope:
        if not isinstance(ids, Tensor):
            raise TypeError(f"Expected Tensor, got {type(ids).__name__}")
        if ids.ndim != 1:
            raise ValueError(f"Expected 1D tensor, got shape {tuple(ids.shape)}")
        if ids.numel() <= 0:
            raise ValueError("Cannot decode empty tensor")
        if ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise TypeError(f"Expected integer tensor dtype, got {ids.dtype}")

        vals = ids.detach().cpu().to(dtype=torch.int64).tolist()
        if any((v < 0 or v > 255) for v in vals):
            raise ValueError("Byte-level event decoding expects values in [0, 255]")
        raw = bytes(int(v) for v in vals)
        s = raw.decode("utf-8")
        obj = json.loads(s)
        if not isinstance(obj, dict):
            raise TypeError(f"Decoded JSON must be an object, got {type(obj).__name__}")
        return EventEnvelope.from_json_dict(obj)

    def decode_many(self, encoded: Iterable[Tensor]) -> list[EventEnvelope]:
        return [self.decode(t) for t in encoded]

    def decode_padded(self, ids: Tensor, mask: Tensor) -> list[EventEnvelope]:
        if not isinstance(ids, Tensor) or not isinstance(mask, Tensor):
            raise TypeError("decode_padded expects (Tensor ids, Tensor mask)")
        if ids.ndim != 2 or mask.ndim != 2:
            raise ValueError("decode_padded expects 2D (B,L) tensors")
        if ids.shape != mask.shape:
            raise ValueError(f"ids and mask shape mismatch: {tuple(ids.shape)} vs {tuple(mask.shape)}")
        if mask.dtype != torch.bool:
            raise TypeError(f"mask must be bool, got {mask.dtype}")

        out: list[EventEnvelope] = []
        batch_size = int(ids.size(0))
        for index in range(batch_size):
            sel = mask[index]
            out.append(self.decode(ids[index][sel]))
        return out

