"""Event trace dataset

Converts Cap'n Proto event envelopes into token sequences for training, bridging the
gap between high-level events and low-level token processing. Produces
next-token pairs along with teacher signals for MOSAIC memory operations,
enabling models to learn when and how to interact with external memory.
"""

from __future__ import annotations

from dataclasses import dataclass
import random

import torch
from torch import Tensor
from torch.utils.data import Dataset

from caramba.core.event import EventEnvelope
from caramba.core.event_codec import EventEncoder
from caramba.core.event_codec.payloads import (
    encode_idle_payload,
    encode_memory_answer_payload,
    encode_memory_query_payload,
    encode_memory_write_payload,
    encode_message_payload,
    encode_noise_payload,
)
from caramba.layer.memory_block.isa import MemoryOpcode


class _EventTraceBuilder:
    """Event trace builder

    Converts event envelopes into token sequences with supervision signals,
    tracking which tokens correspond to memory operations, write gates, and
    other MOSAIC control mechanisms for teacher-forced training.
    """
    def __init__(self, *, mem_hashes: int, reg_slots: int = 0) -> None:
        """Initialize event trace builder

        Sets up storage for tokens and supervision signals, configuring the
        number of memory hash functions and register slots for the target
        architecture.
        """
        self.mem_hashes = int(mem_hashes)
        if self.mem_hashes < 1:
            raise ValueError(f"mem_hashes must be >= 1, got {self.mem_hashes}")
        self.reg_slots = int(reg_slots)
        if self.reg_slots < 0:
            raise ValueError(f"reg_slots must be >= 0, got {self.reg_slots}")

        self._encoder = EventEncoder()

        self.tokens: list[int] = []
        self.opcodes: list[int] = []
        self.write_gate: list[int] = []
        self.write_utility: list[int] = []
        self.read_bucket: list[list[int]] = []
        self.write_bucket: list[list[int]] = []
        self.drop_local: list[int] = []
        self.commitment_delta: list[int] = []
        self.reg_write_gate: list[int] = []
        self.reg_sel: list[int] = []

    def _append_token(
        self,
        tok: int,
        *,
        op: int = int(MemoryOpcode.NOP),
        wg: int = 0,
        wu: int = 0,
        rb: list[int] | None = None,
        wb: list[int] | None = None,
        dl: int = 0,
        cd: int = -100,
        rg: int = 0,
        rs: int = -1,
    ) -> None:
        """Append token with supervision

        Adds a token to the sequence along with its associated supervision
        signals, which tell the model what memory operations should happen
        at this position during teacher-forced training.
        """
        t = int(tok)
        if t < 0 or t > 255:
            raise ValueError(f"Event token must be a byte in [0, 255], got {t}")
        cd_i = int(cd)
        if cd_i not in (-100, -1, 0, 1):
            raise ValueError(f"commitment_delta teacher must be in {{-100,-1,0,1}}, got {cd_i}")
        self.tokens.append(t)
        self.opcodes.append(int(op))
        self.write_gate.append(int(wg))
        self.write_utility.append(int(wu))
        self.read_bucket.append(([-1] * self.mem_hashes) if rb is None else [int(x) for x in rb])
        self.write_bucket.append(([-1] * self.mem_hashes) if wb is None else [int(x) for x in wb])
        self.drop_local.append(int(dl))
        self.commitment_delta.append(cd_i)
        if self.reg_slots > 0:
            self.reg_write_gate.append(int(rg))
            if int(rg) > 0:
                sel = int(rs)
                if sel < 0 or sel >= int(self.reg_slots):
                    raise ValueError(f"reg_sel out of range: {sel} for reg_slots={self.reg_slots}")
                self.reg_sel.append(sel)
            else:
                self.reg_sel.append(-1)

    def _event_bytes(self, env: EventEnvelope) -> bytes:
        """Serialize an event envelope to bytes (Cap'n Proto)."""
        b = self._encoder.encode_bytes(env)
        if not b:
            raise ValueError("Event Cap'n Proto serialization produced empty bytes")
        return b

    def append_event(
        self,
        env: EventEnvelope,
        *,
        opcode: MemoryOpcode | None = None,
        write_gate: bool = False,
        reg_write_gate: bool = False,
        reg_slot: int | None = None,
        write_bucket: int | None = None,
        read_bucket: int | None = None,
        drop_local: int = 0,
        teacher_commitment_delta: int | None = None,
        delimiter: int = 10,  # '\n'
    ) -> None:
        """Append event with supervision

        Converts an event to tokens and adds supervision signals at a fixed
        per-event position (event-level supervision).
        """
        b = self._event_bytes(env)
        # Event-level supervision anchor: first byte of the event.
        # This keeps supervision stable under truncation/padding.
        pos = 0

        if teacher_commitment_delta is not None:
            cd_i = int(teacher_commitment_delta)
            if cd_i not in (-1, 0, 1):
                raise ValueError(f"teacher_commitment_delta must be in {{-1,0,1}}, got {cd_i}")
        if reg_write_gate:
            if int(self.reg_slots) <= 0:
                raise ValueError("reg_write_gate requested but reg_slots == 0")
            if reg_slot is None:
                raise ValueError("reg_write_gate requested but reg_slot is None")

        wb_vec = ([int(write_bucket)] * self.mem_hashes) if write_bucket is not None else None
        rb_vec = ([int(read_bucket)] * self.mem_hashes) if read_bucket is not None else None

        for i, bt in enumerate(b):
            active = (i == pos)
            op = int(opcode) if (active and opcode is not None) else int(MemoryOpcode.NOP)
            wg = 1 if (active and bool(write_gate)) else 0
            wu = wg
            wb = wb_vec if (active and bool(write_gate)) else None
            rb = rb_vec if (active and opcode is not None and rb_vec is not None) else None
            cd = int(teacher_commitment_delta) if (active and teacher_commitment_delta is not None) else -100
            rg = 1 if (active and bool(reg_write_gate)) else 0
            if rg > 0:
                if reg_slot is None:
                    raise ValueError("reg_slot must be provided when reg_write_gate is active")
                rs = int(reg_slot)
            else:
                rs = -1
            self._append_token(bt, op=op, wg=wg, wu=wu, rb=rb, wb=wb, dl=int(drop_local), cd=cd, rg=rg, rs=rs)

        # Add delimiter with no supervision.
        self._append_token(int(delimiter), op=int(MemoryOpcode.NOP), wg=0, wu=0, rb=None, wb=None, dl=0, cd=-100)


class _MosaicEventTraceTorchDataset(Dataset[dict[str, Tensor]]):
    """MOSAIC event trace dataset implementation

    Generates synthetic event sequences that require memory operations, creating
    write-then-query patterns with distractors to teach models when to use
    external memory versus local context.
    """
    def __init__(
        self,
        *,
        n_items: int,
        block_size: int,
        vocab_size: int,
        mem_buckets: int,
        mem_hashes: int,
        n_pairs: int,
        distractor_events: int,
        negotiation_pairs: int = 0,
        seed: int,
        reg_slots: int = 0,
        sleep_replay_per_pair: int = 0,
    ) -> None:
        """Initialize event trace dataset

        Sets up parameters for generating synthetic memory operations, including
        the number of key-value pairs, distractors between operations, and
        optional commitment tracking for multi-agent scenarios.
        """
        self.n_items = int(n_items)
        self.block_size = int(block_size)
        self.vocab_size = int(vocab_size)
        self.mem_buckets = int(mem_buckets)
        self.mem_hashes = int(mem_hashes)
        self.n_pairs = int(n_pairs)
        self.distractor_events = int(distractor_events)
        self.negotiation_pairs = int(negotiation_pairs)
        self.seed = int(seed)
        self.reg_slots = int(reg_slots)
        self.sleep_replay_per_pair = int(sleep_replay_per_pair)

        if self.n_items <= 0:
            raise ValueError(f"n_items must be > 0, got {self.n_items}")
        if self.block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {self.block_size}")
        if self.vocab_size < 256:
            raise ValueError(f"vocab_size must be >= 256 for byte-level event tokens, got {self.vocab_size}")
        if self.mem_buckets < 2:
            raise ValueError(f"mem_buckets must be >= 2, got {self.mem_buckets}")
        if self.mem_hashes < 1:
            raise ValueError(f"mem_hashes must be >= 1, got {self.mem_hashes}")
        if self.n_pairs < 1:
            raise ValueError(f"n_pairs must be >= 1, got {self.n_pairs}")
        if self.distractor_events < 0:
            raise ValueError(f"distractor_events must be >= 0, got {self.distractor_events}")
        if self.negotiation_pairs < 0:
            raise ValueError(f"negotiation_pairs must be >= 0, got {self.negotiation_pairs}")
        if self.reg_slots < 0:
            raise ValueError(f"reg_slots must be >= 0, got {self.reg_slots}")
        if self.sleep_replay_per_pair < 0:
            raise ValueError(f"sleep_replay_per_pair must be >= 0, got {self.sleep_replay_per_pair}")

        # Keep key/value tokens away from small control bytes.
        self.min_tok = 256
        if self.vocab_size <= self.min_tok + 8:
            raise ValueError("vocab_size too small for event trace curriculum")

    def __len__(self) -> int:
        return int(self.n_items)

    def _bucket_for_key(self, key_tok: int) -> int:
        return int(key_tok) % int(self.mem_buckets)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        rng = random.Random(int(self.seed) + int(idx))
        need = int(self.block_size) + 1
        if need < 2:
            raise ValueError("block_size must allow at least 2 tokens")

        keys: list[int] = []
        vals: list[int] = []
        for _ in range(int(self.n_pairs)):
            k = rng.randrange(int(self.min_tok), int(self.vocab_size))
            v = rng.randrange(int(self.min_tok), int(self.vocab_size))
            keys.append(k)
            vals.append(v)

        b = _EventTraceBuilder(mem_hashes=int(self.mem_hashes), reg_slots=int(self.reg_slots))

        # Deterministic timestamp base per sample.
        ts0 = float(1_700_000_000.0 + float(idx))

        # Write phase.
        for j, (k, v) in enumerate(zip(keys, vals, strict=True)):
            env = EventEnvelope(
                type="MemoryWrite",
                sender="dataset",
                payload=encode_memory_write_payload(key=int(k), value=int(v)),
                priority=0,
                id=f"{idx:08x}{j:02x}w",
                ts=ts0 + float(j) * 0.001,
            )
            bucket = self._bucket_for_key(k)
            b.append_event(
                env,
                opcode=MemoryOpcode.WRITE_MEM,
                write_gate=True,
                reg_write_gate=(int(self.reg_slots) > 0),
                reg_slot=(int(j) % int(self.reg_slots)) if int(self.reg_slots) > 0 else None,
                write_bucket=int(bucket),
                read_bucket=None,
                drop_local=0,
            )
            # Distractors.
            for dj in range(int(self.distractor_events)):
                noise = rng.randrange(0, 1_000_000)
                env_d = EventEnvelope(
                    type="Noise",
                    sender="dataset",
                    payload=encode_noise_payload(tok=int(noise)),
                    priority=0,
                    id=f"{idx:08x}{j:02x}d{dj:02x}",
                    ts=ts0 + 0.1 + float(j) * 0.001 + float(dj) * 1e-6,
                )
                b.append_event(env_d)

        # Negotiation phase (Phase 2): commitment open/neutral/close cycles.
        if int(self.negotiation_pairs) > 0:
            for j in range(int(self.negotiation_pairs)):
                cid = f"{idx:08x}c{j:02x}"

                txt_open = "I will look for that file"
                env_open = EventEnvelope(
                    type="Message",
                    sender="agent",
                    payload=encode_message_payload(text=txt_open),
                    priority=0,
                    commitment_delta=+1,
                    commitment_id=cid,
                    id=f"{idx:08x}{j:02x}co",
                    ts=ts0 + 0.5 + float(j) * 0.001,
                )
                b.append_event(env_open, teacher_commitment_delta=+1)

                txt_work = "Working on it"
                env_work = EventEnvelope(
                    type="Message",
                    sender="agent",
                    payload=encode_message_payload(text=txt_work),
                    priority=0,
                    commitment_delta=0,
                    commitment_id=cid,
                    id=f"{idx:08x}{j:02x}cw",
                    ts=ts0 + 0.6 + float(j) * 0.001,
                )
                b.append_event(env_work, teacher_commitment_delta=0)

                txt_close = "Here is the content"
                env_close = EventEnvelope(
                    type="Message",
                    sender="agent",
                    payload=encode_message_payload(text=txt_close),
                    priority=0,
                    commitment_delta=-1,
                    commitment_id=cid,
                    id=f"{idx:08x}{j:02x}cc",
                    ts=ts0 + 0.7 + float(j) * 0.001,
                )
                b.append_event(env_close, teacher_commitment_delta=-1)

        # Query phase.
        for j, (k, v) in enumerate(zip(keys, vals, strict=True)):
            env_q = EventEnvelope(
                type="MemoryQuery",
                sender="dataset",
                payload=encode_memory_query_payload(key=int(k)),
                priority=0,
                id=f"{idx:08x}{j:02x}q",
                ts=ts0 + 1.0 + float(j) * 0.001,
            )
            bucket = self._bucket_for_key(k)
            b.append_event(
                env_q,
                opcode=MemoryOpcode.READ_MEM,
                write_bucket=None,
                read_bucket=int(bucket),
                drop_local=1,
            )

            env_a = EventEnvelope(
                type="MemoryAnswer",
                sender="dataset",
                payload=encode_memory_answer_payload(value=int(v)),
                priority=0,
                id=f"{idx:08x}{j:02x}a",
                ts=ts0 + 1.5 + float(j) * 0.001,
            )
            b.append_event(env_a)

        # Sleep/replay: no query key, but force a memory read and emit the answer.
        if int(self.sleep_replay_per_pair) > 0:
            for j, (k, v) in enumerate(zip(keys, vals, strict=True)):
                bucket = self._bucket_for_key(k)
                for r in range(int(self.sleep_replay_per_pair)):
                    env_idle = EventEnvelope(
                        type="Idle",
                        sender="dataset",
                        payload=encode_idle_payload(
                            ts=ts0 + 2.0 + float(j) * 0.001 + float(r) * 1e-6,
                            metrics={"i": float(r)},
                        ),
                        priority=0,
                        id=f"{idx:08x}{j:02x}s{r:02x}",
                        ts=ts0 + 2.0 + float(j) * 0.001 + float(r) * 1e-6,
                    )
                    b.append_event(
                        env_idle,
                        opcode=MemoryOpcode.READ_MEM,
                        read_bucket=int(bucket),
                        drop_local=1,
                    )
                    env_a = EventEnvelope(
                        type="MemoryAnswer",
                        sender="dataset",
                        payload=encode_memory_answer_payload(value=int(v)),
                        priority=0,
                        id=f"{idx:08x}{j:02x}sa{r:02x}",
                        ts=ts0 + 2.5 + float(j) * 0.001 + float(r) * 1e-6,
                    )
                    b.append_event(env_a)

        # Pad/truncate to need.
        if len(b.tokens) < need:
            for _ in range(need - len(b.tokens)):
                b._append_token(0)
        else:
            b.tokens = b.tokens[:need]
            b.opcodes = b.opcodes[:need]
            b.write_gate = b.write_gate[:need]
            b.write_utility = b.write_utility[:need]
            b.read_bucket = b.read_bucket[:need]
            b.write_bucket = b.write_bucket[:need]
            b.drop_local = b.drop_local[:need]
            b.commitment_delta = b.commitment_delta[:need]
            if int(self.reg_slots) > 0:
                b.reg_write_gate = b.reg_write_gate[:need]
                b.reg_sel = b.reg_sel[:need]

        x = torch.tensor(b.tokens[:-1], dtype=torch.long)
        y = torch.tensor(b.tokens[1:], dtype=torch.long)

        wg = torch.tensor(b.write_gate[:-1], dtype=torch.float32)
        wu = torch.tensor(b.write_utility[:-1], dtype=torch.float32)
        rb = torch.tensor(b.read_bucket[:-1], dtype=torch.long)
        wb = torch.tensor(b.write_bucket[:-1], dtype=torch.long)
        dl = torch.tensor(b.drop_local[:-1], dtype=torch.float32)
        op = torch.tensor(b.opcodes[:-1], dtype=torch.long)
        cd = torch.tensor(b.commitment_delta[:-1], dtype=torch.long)

        out = {
            "input_ids": x,
            "target_ids": y,
            "memblock_teacher_write_gate": wg,
            "memblock_teacher_write_utility": wu,
            "memblock_teacher_read_bucket": rb,
            "memblock_teacher_write_bucket": wb,
            "mosaic_drop_local": dl,
            "memblock_teacher_opcode": op,
            "memblock_teacher_commitment_delta": cd,
        }
        if int(self.reg_slots) > 0:
            out["memblock_teacher_reg_write_gate"] = torch.tensor(b.reg_write_gate[:-1], dtype=torch.float32)
            out["memblock_teacher_reg_sel"] = torch.tensor(b.reg_sel[:-1], dtype=torch.long)
        return out


@dataclass(frozen=True, slots=True)
class MosaicEventTraceDataset:
    """MOSAIC event trace dataset component

    Manifest-level dataset that generates synthetic event sequences for training
    MOSAIC models. Creates write-then-query patterns with teacher signals,
    teaching models to use external memory for long-range dependencies.
    """

    block_size: int = 256
    vocab_size: int = 2048  # must be >= 256 (byte-level tokens)
    mem_buckets: int = 4096
    mem_hashes: int = 2
    n_pairs: int = 2
    distractor_events: int = 8
    negotiation_pairs: int = 0
    n_items: int = 100_000
    seed: int = 1337
    reg_slots: int = 0
    sleep_replay_per_pair: int = 0

    def build(self) -> Dataset[dict[str, Tensor]]:
        """Build event trace dataset

        Creates the PyTorch dataset that will generate synthetic event sequences
        with memory operation supervision, ready for training MOSAIC models.
        """
        return _MosaicEventTraceTorchDataset(
            n_items=int(self.n_items),
            block_size=int(self.block_size),
            vocab_size=int(self.vocab_size),
            mem_buckets=int(self.mem_buckets),
            mem_hashes=int(self.mem_hashes),
            n_pairs=int(self.n_pairs),
            distractor_events=int(self.distractor_events),
            negotiation_pairs=int(self.negotiation_pairs),
            seed=int(self.seed),
            reg_slots=int(self.reg_slots),
            sleep_replay_per_pair=int(self.sleep_replay_per_pair),
        )

