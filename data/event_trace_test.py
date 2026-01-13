from __future__ import annotations

import torch

from caramba.data.event_trace import MosaicEventTraceDataset
from caramba.layer.memory_block.isa import MemoryOpcode


def test_event_trace_shapes_and_basic_signal_presence() -> None:
    ds = MosaicEventTraceDataset(
        block_size=256,
        vocab_size=2048,
        mem_buckets=128,
        mem_hashes=2,
        n_pairs=1,
        distractor_events=0,
        negotiation_pairs=1,
        n_items=2,
        seed=1337,
        reg_slots=2,
        sleep_replay_per_pair=0,
    ).build()

    item = ds[0]
    assert item["input_ids"].shape == (256,)
    assert item["target_ids"].shape == (256,)
    assert item["memblock_teacher_opcode"].shape == (256,)
    assert item["memblock_teacher_commitment_delta"].shape == (256,)

    # Byte-level token stream (plus 0 padding).
    assert int(item["input_ids"].min().item()) >= 0
    assert int(item["input_ids"].max().item()) <= 255

    # We should teach at least one memory opcode.
    assert int((item["memblock_teacher_opcode"] != int(MemoryOpcode.NOP)).sum().item()) >= 1

    # Commitment delta teacher should include at least one labeled position.
    cd = item["memblock_teacher_commitment_delta"]
    assert bool(((cd == -1) | (cd == 0) | (cd == 1)).any().item())

    # Optional reg supervision present when reg_slots>0.
    assert "memblock_teacher_reg_write_gate" in item
    assert item["memblock_teacher_reg_write_gate"].dtype == torch.float32

