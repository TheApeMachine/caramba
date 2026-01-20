"""MOSAIC synthetic curriculum dataset

Generates synthetic sequences that require explicit memory operations over long
gaps, providing teacher signals for memory addressing and gating. Designed for
curriculum training where models learn to use external memory before handling
real-world tasks.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict

import torch
from torch.utils.data import Dataset

from runtime.tensordict_utils import TensorDictBase


class _MosaicMemoryCurriculumDataset(Dataset[TensorDictBase]):
    """MOSAIC memory curriculum dataset implementation

    Generates synthetic key-value memory tasks with write-then-query patterns,
    teaching models when to store and retrieve information from external memory.
    Includes teacher signals for all memory operations to enable supervised
    learning of memory mechanisms.
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
        distractor_len: int,
        seed: int,
        reg_slots: int = 0,
        sleep_replay_per_pair: int = 0,
    ) -> None:
        """Initialize memory curriculum dataset

        Sets up parameters for generating memory tasks, including the number
        of key-value pairs, distractor length between operations, and optional
        register slots for register-based memory architectures.
        """
        self.n_items = int(n_items)
        self.block_size = int(block_size)
        self.vocab_size = int(vocab_size)
        self.mem_buckets = int(mem_buckets)
        self.mem_hashes = int(mem_hashes)
        self.n_pairs = int(n_pairs)
        self.distractor_len = int(distractor_len)
        self.seed = int(seed)
        self.reg_slots = int(reg_slots)
        self.sleep_replay_per_pair = int(sleep_replay_per_pair)
        if self.reg_slots < 0:
            raise ValueError(f"reg_slots must be >= 0, got {self.reg_slots}")
        if self.sleep_replay_per_pair < 0:
            raise ValueError(f"sleep_replay_per_pair must be >= 0, got {self.sleep_replay_per_pair}")

        # Reserve a tiny synthetic "protocol" token set in the low ids.
        self.T_SET = 1
        self.T_IS = 2
        self.T_GET = 3
        self.T_Q = 4
        self.T_PAD = 0

        self.min_tok = 8
        if self.vocab_size <= self.min_tok + 8:
            raise ValueError("vocab_size too small for synthetic curriculum")

        # VM opcode convention (matches MemoryBlockLayerConfig.opcode_vocab default=4):
        # 0: NOP, 1: READ, 2: WRITE, 3: CLEAR
        self.OP_NOP = 0
        self.OP_READ = 1
        self.OP_WRITE = 2
        self.OP_CLEAR = 3

    def __len__(self) -> int:
        return self.n_items

    def _bucket_for_key(self, key_tok: int) -> int:
        """Compute memory bucket for key

        Maps a key token to a memory bucket using modulo hashing, providing
        deterministic addressing that models can learn. This creates a stable
        mapping so the same key always goes to the same bucket.
        """
        # Deterministic teacher addressing: stable mapping key->bucket.
        return int(key_tok) % int(self.mem_buckets)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = random.Random(self.seed + int(idx))

        # Create key-value bindings.
        keys: list[int] = []
        vals: list[int] = []
        for _ in range(max(1, int(self.n_pairs))):
            k = rng.randrange(self.min_tok, self.vocab_size)
            v = rng.randrange(self.min_tok, self.vocab_size)
            keys.append(k)
            vals.append(v)

        seq: list[int] = []
        write_gate: list[int] = []
        write_utility: list[int] = []
        read_bucket: list[list[int]] = []
        write_bucket: list[list[int]] = []
        drop_local: list[int] = []
        opcode: list[int] = []
        reg_write_gate: list[int] = []
        reg_sel: list[int] = []

        def _append(
            tok: int,
            *,
            # By default, supervise "do not write" and "not useful" everywhere.
            # This prevents the write gate from drifting to high write rates due to
            # only seeing positive supervision on a tiny subset of tokens.
            wg: int = 0,
            wu: int = 0,
            rb: list[int] | None = None,
            wb: list[int] | None = None,
            dl: int = 0,
            op: int = 0,
            rg: int = 0,
            rs: int = -1,
        ) -> None:
            seq.append(int(tok))
            write_gate.append(int(wg))
            write_utility.append(int(wu))
            if rb is None:
                read_bucket.append([-1] * int(self.mem_hashes))
            else:
                read_bucket.append([int(x) for x in rb])
            if wb is None:
                write_bucket.append([-1] * int(self.mem_hashes))
            else:
                write_bucket.append([int(x) for x in wb])
            drop_local.append(int(dl))
            opcode.append(int(op))
            if self.reg_slots > 0:
                reg_write_gate.append(int(rg))
                if int(rg) > 0:
                    sel = int(rs)
                    if sel < 0 or sel >= int(self.reg_slots):
                        raise ValueError(f"reg_sel out of range: {sel} for reg_slots={self.reg_slots}")
                    reg_sel.append(sel)
                else:
                    reg_sel.append(-1)

        # Write phase: "SET k IS v" with long distractors.
        for i, (k, v) in enumerate(zip(keys, vals, strict=True)):
            b = self._bucket_for_key(k)
            bvec = [b] * int(self.mem_hashes)
            _append(self.T_SET, wg=0, wu=0, op=self.OP_NOP)
            _append(k, wg=0, wu=0, op=self.OP_NOP)
            _append(self.T_IS, wg=0, wu=0, op=self.OP_NOP)
            # Write on value token (teacher-forced).
            slot = int(i) % int(self.reg_slots) if self.reg_slots > 0 else -1
            _append(v, wg=1, wu=1, wb=bvec, op=self.OP_WRITE, rg=1 if self.reg_slots > 0 else 0, rs=slot)
            # Distractors.
            for _ in range(int(self.distractor_len)):
                _append(rng.randrange(self.min_tok, self.vocab_size), wg=0, wu=0, op=self.OP_NOP)

        # Query phase: "GET k ? IS v"
        for k, v in zip(keys, vals, strict=True):
            b = self._bucket_for_key(k)
            bvec = [b] * int(self.mem_hashes)
            # Force reliance on memory/state throughout the query prefix.
            _append(self.T_GET, wg=0, wu=0, dl=1, op=self.OP_NOP)
            _append(k, wg=0, wu=0, dl=1, op=self.OP_NOP)
            _append(self.T_Q, wg=0, wu=0, dl=1, op=self.OP_NOP)
            # On the token before emitting v as the next token (here: T_IS),
            # we want a memory read to be available.
            _append(self.T_IS, wg=0, wu=0, rb=bvec, dl=1, op=self.OP_READ)
            _append(v, wg=0, wu=0, op=self.OP_NOP)
            # More distractors (small).
            for _ in range(max(0, int(self.distractor_len) // 4)):
                _append(rng.randrange(self.min_tok, self.vocab_size), wg=0, wu=0, op=self.OP_NOP)

        # Sleep/replay phase: lock external input (PAD), read memory, predict stored values.
        # Pattern: [PAD] -> target is v, with teacher read_bucket set on PAD positions.
        if int(self.sleep_replay_per_pair) > 0:
            for k, v in zip(keys, vals, strict=True):
                b = self._bucket_for_key(k)
                bvec = [b] * int(self.mem_hashes)
                for _ in range(int(self.sleep_replay_per_pair)):
                    _append(self.T_PAD, wg=0, wu=0, rb=bvec, dl=1, op=self.OP_READ)
                    _append(v, wg=0, wu=0, op=self.OP_NOP)

        # Pad/truncate to block_size+1 so we can create input/target pairs.
        # We need at least 2 tokens.
        need = int(self.block_size) + 1
        if len(seq) < need:
            for _ in range(need - len(seq)):
                _append(self.T_PAD, op=self.OP_NOP)
        else:
            seq = seq[:need]
            write_gate = write_gate[:need]
            write_utility = write_utility[:need]
            read_bucket = read_bucket[:need]
            write_bucket = write_bucket[:need]
            drop_local = drop_local[:need]
            opcode = opcode[:need]
            if self.reg_slots > 0:
                reg_write_gate = reg_write_gate[:need]
                reg_sel = reg_sel[:need]

        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)

        # Teacher signals align to input token positions (same T as x/y).
        wg = torch.tensor(write_gate[:-1], dtype=torch.float32)
        wu = torch.tensor(write_utility[:-1], dtype=torch.float32)
        rb = torch.tensor(read_bucket[:-1], dtype=torch.long)
        wb = torch.tensor(write_bucket[:-1], dtype=torch.long)
        dl = torch.tensor(drop_local[:-1], dtype=torch.float32)
        op = torch.tensor(opcode[:-1], dtype=torch.long)

        out = {
            "input_ids": x,
            "target_ids": y,
            "memblock_teacher_write_gate": wg,
            "memblock_teacher_write_utility": wu,
            "memblock_teacher_read_bucket": rb,
            "memblock_teacher_write_bucket": wb,
            "mosaic_drop_local": dl,
            "memblock_teacher_opcode": op,
        }
        if self.reg_slots > 0:
            out["memblock_teacher_reg_write_gate"] = torch.tensor(reg_write_gate[:-1], dtype=torch.float32)
            out["memblock_teacher_reg_sel"] = torch.tensor(reg_sel[:-1], dtype=torch.long)
        return out


@dataclass(frozen=True, slots=True)
class MosaicMemoryCurriculumDataset:
    """MOSAIC memory curriculum dataset component

    Manifest-level dataset for curriculum training of memory-augmented models.
    Generates synthetic tasks that require explicit memory operations, with
    teacher signals guiding models to learn proper memory usage patterns.
    """

    block_size: int = 512
    vocab_size: int = 8192
    mem_buckets: int = 16384
    mem_hashes: int = 2
    n_pairs: int = 1
    distractor_len: int = 64
    n_items: int = 100_000
    seed: int = 1337
    reg_slots: int = 0
    sleep_replay_per_pair: int = 0

    def build(self) -> Dataset[TensorDictBase]:
        """Build memory curriculum dataset

        Creates the PyTorch dataset that generates synthetic memory tasks with
        teacher signals, ready for curriculum training of MOSAIC models.
        """
        return _MosaicMemoryCurriculumDataset(
            n_items=int(self.n_items),
            block_size=int(self.block_size),
            vocab_size=int(self.vocab_size),
            mem_buckets=int(self.mem_buckets),
            mem_hashes=int(self.mem_hashes),
            n_pairs=int(self.n_pairs),
            distractor_len=int(self.distractor_len),
            seed=int(self.seed),
            reg_slots=int(self.reg_slots),
            sleep_replay_per_pair=int(self.sleep_replay_per_pair),
        )

