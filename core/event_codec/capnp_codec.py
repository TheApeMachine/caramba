"""Cap'n Proto event codec

Encodes/decodes EventEnvelope using Cap'n Proto.

Cap'n Proto can enable zero-copy *buffer views* in principle. This codec is written
to avoid unnecessary copies when moving bytes around (for example, decoding via a
NumPy view instead of materializing a new `bytes` object).

However, some copies are unavoidable depending on usage:
- The schema stores payloads as opaque bytes. If `EventEnvelope.payload` is not
  already bytes-like, we serialize it to UTF-8 JSON (encoding allocates).
- If you need "token ids" as `torch.long` (common for embedding layers), converting
  raw bytes (`uint8`) to `int64` necessarily allocates/copies.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path

import torch
from torch import Tensor
import capnp
from typing import Any, cast
from caramba.core.event import EventEnvelope


def _load_schema():
    """Load the Cap'n Proto schema file."""
    schema_path = Path(__file__).parent / "event.capnp"

    if not schema_path.exists():
        raise FileNotFoundError(f"Cap'n Proto schema not found: {schema_path}")

    # capnp stubs are incomplete; cast to satisfy type checkers.
    capnp_any = cast(Any, capnp)
    return capnp_any.load(str(schema_path))


# Lazy-load schema to avoid import-time errors
_schema = None


def _get_schema():
    global _schema
    if _schema is None:
        _schema = _load_schema()
    return _schema


class CapnpEventEncoder:
    """Cap'n Proto event encoder.

    Converts EventEnvelope to Cap'n Proto bytes and returns them as a 1D token tensor.
    """

    def _payload_to_bytes(self, payload: Any) -> bytes:
        """Convert an EventEnvelope payload to schema bytes.

        This codec is Cap'n Proto-only: payloads must already be bytes-like.
        """
        if payload is None:
            return b""
        if isinstance(payload, (bytes, bytearray)):
            return bytes(payload)
        if isinstance(payload, memoryview):
            return payload.tobytes()
        raise TypeError(
            "CapnpEventEncoder: payload must be bytes-like (bytes|bytearray|memoryview). "
            f"Got {type(payload).__name__}."
        )

    def encode_bytes(self, event: EventEnvelope) -> bytes:
        """Encode an event to Cap'n Proto wire bytes."""
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

        # Payload is schema Data (opaque bytes)
        msg.payload = self._payload_to_bytes(event.payload)

        # Serialize to bytes
        buf = msg.to_bytes()
        if not buf:
            raise ValueError("Cap'n Proto serialization produced empty bytes")
        return buf

    def encode_uint8(self, event: EventEnvelope) -> Tensor:
        """Encode an event to a 1D byte tensor (`torch.uint8`)."""
        buf = self.encode_bytes(event)
        mv = memoryview(buf)
        # Avoid noisy global warnings; callers can `.clone()` if they need ownership.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"The given buffer is not writable.*",
                category=UserWarning,
            )
            return torch.frombuffer(mv, dtype=torch.uint8)

    def encode(self, event: EventEnvelope, *, dtype: torch.dtype = torch.long) -> Tensor:
        """Encode an event to a 1D token tensor.

        - `dtype=torch.uint8` returns a byte-level view (no dtype conversion copy).
        - `dtype=torch.long` (default) allocates/copies as required by most embedding layers.
        """
        u8 = self.encode_uint8(event)
        if dtype == torch.uint8:
            return u8
        return u8.to(dtype)

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

        # Convert tensor to a CPU uint8 contiguous buffer.
        cpu = ids.detach()
        if cpu.device.type != "cpu":
            cpu = cpu.cpu()
        if cpu.numel() <= 0:
            raise ValueError("Cannot decode empty tensor")
        if cpu.dtype == torch.uint8:
            u8 = cpu
        else:
            if cpu.min().item() < 0 or cpu.max().item() > 255:
                raise ValueError("Cap'n Proto byte tensor values must be in [0, 255]")
            u8 = cpu.to(dtype=torch.uint8)
        u8 = u8.contiguous()

        # Deserialize Cap'n Proto message (from_bytes returns context manager).
        # We pass a NumPy view to avoid an extra `.tobytes()` allocation.
        with schema.EventEnvelope.from_bytes(u8.numpy()) as msg:
            # msg.payload is a memoryview (zero-copy).
            payload = msg.payload.tobytes() if msg.payload else b""

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
