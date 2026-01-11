"""Event-driven inference runtime (EventBus ↔ model).

Implements the integration layer described in the MOSAIC meeting notes:
- Encode inbound EventEnvelope → byte tokens
- Run an autoregressive model to emit byte tokens
- Decode outbound bytes → EventEnvelope
- "Mode B" commitment injection: set commitment_delta from aux head logits
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import torch
from torch import Tensor, nn

from caramba.core.commitments import CommitmentLedger
from caramba.core.event import EventEnvelope
from caramba.core.event_bus import EventBus, EventHandler
from caramba.core.event_codec import EventDecoder, EventEncoder
from caramba.infer.context import InferContext
from caramba.infer.replay import ReplayBuffer


@dataclass(frozen=True, slots=True)
class ByteVocabulary:
    size: int = 256

    def slice_logits(self, logits: Tensor) -> Tensor:
        if not isinstance(logits, Tensor):
            raise TypeError(f"logits must be a Tensor, got {type(logits).__name__}")
        if logits.ndim != 2:
            raise ValueError(f"logits must have shape (B,V), got {tuple(logits.shape)}")
        V = int(logits.size(-1))
        n = int(self.size)
        if V < n:
            raise ValueError(f"Model vocab_size={V} is smaller than byte vocab size={n}")
        return logits[:, :n]


@dataclass(frozen=True, slots=True)
class GreedySampler:
    def sample(self, logits: Tensor) -> Tensor:
        if not isinstance(logits, Tensor):
            raise TypeError(f"logits must be a Tensor, got {type(logits).__name__}")
        if logits.ndim != 2:
            raise ValueError(f"logits must have shape (B,V), got {tuple(logits.shape)}")
        return logits.argmax(dim=-1)


@dataclass(frozen=True, slots=True)
class EventStreamCodec:
    encoder: EventEncoder = field(default_factory=EventEncoder)
    decoder: EventDecoder = field(default_factory=EventDecoder)
    delimiter: int = 0  # Cap'n Proto uses 0-byte as segment delimiter

    def encode_with_delimiter(self, event: EventEnvelope) -> Tensor:
        ids = self.encoder.encode(event).to(dtype=torch.long)
        d = int(self.delimiter)
        if d < 0 or d > 255:
            raise ValueError(f"delimiter must be a byte in [0,255], got {d}")
        return torch.cat([ids, torch.tensor([d], dtype=torch.long)], dim=0)

    def decode_bytes(self, ids: Tensor) -> EventEnvelope:
        return self.decoder.decode(ids)


@dataclass(slots=True)
class StreamModelRunner:
    """Autoregressive runner that keeps a persistent InferContext."""

    model: nn.Module
    ctx: InferContext
    pos: int = 0
    collect_aux: bool = True
    # Last aux outputs (cleaned) from the most recent forward_chunk call.
    last_aux: dict[str, Tensor] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model, nn.Module):
            raise TypeError(f"model must be an nn.Module, got {type(self.model).__name__}")
        if not isinstance(self.ctx, InferContext):
            raise TypeError(f"ctx must be an InferContext, got {type(self.ctx).__name__}")
        self.pos = int(self.pos)

    def forward_chunk(self, input_ids: Tensor, *, advance_pos: bool = True) -> tuple[Tensor, dict[str, Tensor] | None]:
        if not isinstance(input_ids, Tensor):
            raise TypeError(f"input_ids must be a Tensor, got {type(input_ids).__name__}")
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape (B,T), got {tuple(input_ids.shape)}")
        if input_ids.numel() <= 0:
            raise ValueError("input_ids must be non-empty")

        B, T = input_ids.shape
        self.ctx.begin(pos_offset=int(self.pos))
        self.ctx.input_ids = input_ids
        self.ctx.memblock_collect_aux = bool(self.collect_aux)
        self.ctx.memblock_aux_out = {} if bool(self.collect_aux) else None

        out = self.model(input_ids, ctx=self.ctx)  # type: ignore[call-arg]
        if not isinstance(out, Tensor):
            raise TypeError(f"Model forward must return a Tensor, got {type(out).__name__}")
        if out.ndim != 3:
            raise ValueError(f"Model forward must return (B,T,V) logits, got {tuple(out.shape)}")
        if out.shape[0] != B or out.shape[1] != T:
            raise ValueError(f"Model logits shape mismatch: expected (B,T,*)={(B,T)}, got {tuple(out.shape)}")

        self.ctx.ensure_consumed()
        if bool(advance_pos):
            self.pos += int(T)

        aux = self.ctx.memblock_aux_out
        if not bool(self.collect_aux):
            return out, None
        if aux is None:
            raise RuntimeError("collect_aux=True but ctx.memblock_aux_out is None after forward")
        if not isinstance(aux, dict):
            raise TypeError(f"ctx.memblock_aux_out must be a dict when set, got {type(aux).__name__}")
        # Keep only tensor entries.
        clean: dict[str, Tensor] = {}
        for k, v in aux.items():
            if isinstance(k, str) and isinstance(v, Tensor):
                clean[k] = v
        self.last_aux = clean
        return out, clean


@dataclass(frozen=True, slots=True)
class CommitmentModeB:
    """Inject commitment_delta from the commitment head logits."""

    key: str = "mosaic_commitment_logits"

    def inject(self, event: EventEnvelope, *, aux: dict[str, Tensor] | None) -> EventEnvelope:
        if aux is None or self.key not in aux:
            raise KeyError(f"Missing {self.key!r} in aux outputs (commitment_head_enabled must be true).")
        logits = aux[self.key]
        if not isinstance(logits, Tensor):
            raise TypeError(f"{self.key} must be a Tensor, got {type(logits).__name__}")
        if logits.ndim != 3 or int(logits.size(-1)) != 3:
            raise ValueError(f"{self.key} must have shape (B,T,3), got {tuple(logits.shape)}")
        if int(logits.size(0)) != 1:
            raise ValueError("Mode B injection currently expects batch_size=1")
        last = logits[0, int(logits.size(1)) - 1, :]
        cls = int(last.argmax(dim=-1).item())
        delta = cls - 1
        if delta not in (-1, 0, 1):
            raise RuntimeError(f"Invalid commitment delta derived from logits: {delta}")
        return replace(event, commitment_delta=int(delta))


@dataclass(slots=True)
class EventResponder:
    """Generate one response EventEnvelope for an inbound event."""

    runner: StreamModelRunner
    codec: EventStreamCodec = field(default_factory=EventStreamCodec)
    vocab: ByteVocabulary = field(default_factory=ByteVocabulary)
    sampler: GreedySampler = field(default_factory=GreedySampler)
    max_new_tokens: int = 1024
    message_end_byte: int = 0  # Cap'n Proto segment delimiter
    replay: ReplayBuffer | None = None
    replay_max_len: int = 4096

    def respond(self, event: EventEnvelope) -> tuple[EventEnvelope, dict[str, Tensor] | None]:
        if not isinstance(event, EventEnvelope):
            raise TypeError(f"event must be an EventEnvelope, got {type(event).__name__}")

        try:
            device = next(self.runner.model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        prompt = self.codec.encode_with_delimiter(event).to(device=device, dtype=torch.long).unsqueeze(0)
        logits, _ = self.runner.forward_chunk(prompt)
        next_logits = logits[:, -1, :]

        buf: list[int] = []
        aux_last: dict[str, Tensor] | None = None

        max_steps = int(self.max_new_tokens)
        if max_steps < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {max_steps}")
        end_b = int(self.message_end_byte)
        if end_b < 0 or end_b > 255:
            raise ValueError(f"message_end_byte must be a byte in [0,255], got {end_b}")

        for _ in range(max_steps):
            byte_logits = self.vocab.slice_logits(next_logits)
            nxt = self.sampler.sample(byte_logits)
            if nxt.ndim != 1 or int(nxt.size(0)) != 1:
                raise RuntimeError("GreedySampler must return shape (1,) for batch_size=1")
            tok = int(nxt[0].item())
            if tok < 0 or tok > 255:
                raise ValueError(f"Generated token is not a byte: {tok}")
            buf.append(tok)

            inp = torch.tensor([[tok]], dtype=torch.long, device=device)
            step_logits, aux = self.runner.forward_chunk(inp)
            aux_last = aux
            next_logits = step_logits[:, -1, :]

            try:
                out_ids = torch.tensor(buf, dtype=torch.long)
                ev = self.codec.decode_bytes(out_ids)
            except Exception:
                # Cap'n Proto decoding failed (e.g. incomplete message), continue generating
                continue

            # If we reach here, decoding succeeded.
            # Maintain the same delimiter convention as training traces.
            delim = int(self.codec.delimiter)
            _ = self.runner.forward_chunk(torch.tensor([[delim]], dtype=torch.long, device=device))
            # Record a replay sequence: prompt + generated bytes + delimiter.
            if self.replay is not None:
                try:
                    max_len = int(self.replay_max_len)
                    seq = torch.cat(
                        [
                            prompt[0].detach().to(dtype=torch.long, device="cpu"),
                            out_ids.detach().to(dtype=torch.long, device="cpu"),
                            torch.tensor([delim], dtype=torch.long),
                        ],
                        dim=0,
                    )
                    if int(seq.numel()) > max_len:
                        seq = seq[-max_len:]
                    self.replay.add(seq)
                except Exception:
                    # Best-effort: replay should never break inference.
                    pass
            return ev, aux_last

        raise RuntimeError("Failed to decode a complete EventEnvelope before max_new_tokens")


@dataclass(slots=True)
class ModelHandler(EventHandler):
    """EventBus handler that runs a model and republishes the response event."""

    bus: EventBus
    responder: EventResponder
    ledger: CommitmentLedger = field(default_factory=CommitmentLedger)
    mode_b: CommitmentModeB = field(default_factory=CommitmentModeB)

    def __post_init__(self) -> None:
        if not isinstance(self.bus, EventBus):
            raise TypeError(f"bus must be an EventBus, got {type(self.bus).__name__}")
        if not isinstance(self.responder, EventResponder):
            raise TypeError(f"responder must be an EventResponder, got {type(self.responder).__name__}")
        if not isinstance(self.ledger, CommitmentLedger):
            raise TypeError(f"ledger must be a CommitmentLedger, got {type(self.ledger).__name__}")

    def handle(self, event: EventEnvelope) -> None:
        # Track incoming events too (idle/open commitments metrics).
        event = self.ledger.apply(event)

        resp, aux = self.responder.respond(event)
        # Only inject commitment_delta if commitment head is enabled and aux contains the required key
        if aux is not None and self.mode_b.key in aux:
            try:
                resp = self.mode_b.inject(resp, aux=aux)
            except KeyError:
                # Commitment head not available, skip injection
                pass
        resp = self.ledger.apply(resp)
        self.bus.publish(resp)

