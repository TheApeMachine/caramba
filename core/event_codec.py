"""Event transducers (JSON ↔ Tensor).

The encoder/decoder provide a minimal, reversible bridge between:
- external event envelopes (JSON)
- internal tensor representations (byte-level tokens)

This is intended as a building block for event-native training/inference where
"tokens" are VM time-steps but the *interface* is event-based.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from typing import Any

import torch
from torch import Tensor

from caramba.core.event import EventEnvelope


class EventEncoder:
    """Encode EventEnvelope → byte-level token tensor.

    Representation:
    - JSON is serialized with stable separators and sorted keys.
    - UTF-8 bytes are returned as int64 tensor values in [0, 255].
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

    def encode_padded(
        self,
        events: Sequence[EventEnvelope],
        *,
        pad_id: int = 0,
    ) -> tuple[Tensor, Tensor]:
        """Encode a batch to (ids, mask).

        Returns:
        - ids:  (B, L) int64
        - mask: (B, L) bool, True where ids are real (not padding)
        """
        if not isinstance(events, Sequence):
            raise TypeError(f"Expected a Sequence of events, got {type(events).__name__}")
        encoded = self.encode_many(events)
        if not encoded:
            raise ValueError("encode_padded requires at least one event")
        lens = [int(t.numel()) for t in encoded]
        L = max(lens)
        if L <= 0:
            raise ValueError("Encoded events have zero length")
        B = len(encoded)
        pad = int(pad_id)
        ids = torch.full((B, L), pad, dtype=torch.long)
        mask = torch.zeros((B, L), dtype=torch.bool)
        for i, t in enumerate(encoded):
            n = int(t.numel())
            ids[i, :n] = t
            mask[i, :n] = True
        return ids, mask


class EventDecoder:
    """Decode byte-level token tensor → EventEnvelope."""

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
        B = int(ids.size(0))
        for i in range(B):
            sel = mask[i]
            out.append(self.decode(ids[i][sel]))
        return out

