"""Cap'n Proto event codec

Encodes/decodes EventEnvelope using Cap'n Proto for zero-copy serialization.
This provides significant performance improvements over JSON for high-throughput
event processing.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import torch
from torch import Tensor
import capnp
from caramba.core.event import EventEnvelope


def _load_schema():
    """Load the Cap'n Proto schema file."""
    schema_path = Path(__file__).parent / "event.capnp"

    if not schema_path.exists():
        raise FileNotFoundError(f"Cap'n Proto schema not found: {schema_path}")

    return capnp.load(str(schema_path))


# Lazy-load schema to avoid import-time errors
_schema = None


def _get_schema():
    global _schema
    if _schema is None:
        _schema = _load_schema()
    return _schema


class CapnpEventEncoder:
    """Cap'n Proto event encoder.

    Converts EventEnvelope to Cap'n Proto bytes and returns them as a 1D int64 tensor.
    """

    def encode(self, event: EventEnvelope) -> Tensor:
        if not isinstance(event, EventEnvelope):
            raise TypeError(f"Expected EventEnvelope, got {type(event).__name__}")

        schema = _get_schema()

        # Create message
        msg = schema.EventEnvelope.new_message()
        msg.id = str(event.id)
        msg.ts = float(event.ts)
        msg.type = str(event.type)
        msg.sender = str(event.sender)
        msg.priority = int(event.priority)
        msg.budgetMs = int(event.budget_ms) if event.budget_ms is not None else -1
        msg.commitmentDelta = int(event.commitment_delta)
        msg.commitmentId = str(event.commitment_id) if event.commitment_id else ""

        # Serialize payload to JSON bytes
        payload_bytes = json.dumps(event.payload, ensure_ascii=False).encode("utf-8")
        msg.payload = payload_bytes

        # Serialize to bytes
        buf = msg.to_bytes()
        if not buf:
            raise ValueError("Cap'n Proto serialization produced empty bytes")

        return torch.tensor(list(buf), dtype=torch.long)

    def encode_many(self, events: Sequence[EventEnvelope]) -> list[Tensor]:
        return [self.encode(e) for e in events]

    def encode_padded(
        self, events: Sequence[EventEnvelope], *, pad_id: int = 0
    ) -> tuple[Tensor, Tensor]:
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


class CapnpEventDecoder:
    """Cap'n Proto event decoder.

    Converts 1D byte-level token tensors into EventEnvelope by deserializing Cap'n Proto.
    """

    def decode(self, ids: Tensor) -> EventEnvelope:
        if not isinstance(ids, Tensor):
            raise TypeError(f"Expected Tensor, got {type(ids).__name__}")
        if ids.ndim != 1:
            raise ValueError(f"Expected 1D tensor, got shape {tuple(ids.shape)}")
        if ids.numel() <= 0:
            raise ValueError("Cannot decode empty tensor")

        schema = _get_schema()

        # Convert tensor to bytes
        vals = ids.detach().cpu().to(dtype=torch.int64).tolist()
        raw = bytes(int(v) & 0xFF for v in vals)

        # Deserialize Cap'n Proto message (from_bytes returns context manager)
        with schema.EventEnvelope.from_bytes(raw) as msg:
            # Parse payload from JSON bytes
            payload_bytes = bytes(msg.payload)
            payload = json.loads(payload_bytes.decode("utf-8")) if payload_bytes else None

            # Handle optional fields
            budget_ms = int(msg.budgetMs) if msg.budgetMs >= 0 else None
            commitment_id = str(msg.commitmentId) if msg.commitmentId else None

            return EventEnvelope(
                id=str(msg.id),
                ts=float(msg.ts),
                type=str(msg.type),
                sender=str(msg.sender),
                priority=int(msg.priority),
                budget_ms=budget_ms,
                commitment_delta=int(msg.commitmentDelta),
                commitment_id=commitment_id,
                payload=payload,
            )

    def decode_many(self, encoded: Iterable[Tensor]) -> list[EventEnvelope]:
        return [self.decode(t) for t in encoded]

    def decode_padded(self, ids: Tensor, mask: Tensor) -> list[EventEnvelope]:
        if not isinstance(ids, Tensor) or not isinstance(mask, Tensor):
            raise TypeError("decode_padded expects (Tensor ids, Tensor mask)")
        if ids.ndim != 2 or mask.ndim != 2:
            raise ValueError("decode_padded expects 2D (B,L) tensors")
        if ids.shape != mask.shape:
            raise ValueError(
                f"ids and mask shape mismatch: {tuple(ids.shape)} vs {tuple(mask.shape)}"
            )
        if mask.dtype != torch.bool:
            raise TypeError(f"mask must be bool, got {mask.dtype}")

        out: list[EventEnvelope] = []
        batch_size = int(ids.size(0))
        for index in range(batch_size):
            sel = mask[index]
            out.append(self.decode(ids[index][sel]))
        return out
