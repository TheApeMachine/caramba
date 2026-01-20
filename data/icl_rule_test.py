from __future__ import annotations

import torch

from data.icl_rule import IclRuleInductionDataset
from trainer.mosaic_table2 import Table2Telemetry
from runtime.tensordict_utils import as_tensordict


def test_icl_rule_emits_teacher_signals_when_enabled() -> None:
    ds_comp = IclRuleInductionDataset(
        block_size=64,
        vocab_size=256,
        n_items=8,
        seed=0,
        n_demos=2,
        gap_bins=[0, 8],
        demo_distractors=2,
        emit_mem_teacher=True,
        mem_buckets=32,
    )
    ds = ds_comp.build()
    ex = ds[0]
    assert "memblock_teacher_write_gate" in ex
    assert "memblock_teacher_write_bucket" in ex
    assert "memblock_teacher_read_bucket" in ex
    assert ex["memblock_teacher_write_gate"].dtype == torch.float32
    assert ex["memblock_teacher_write_bucket"].dtype == torch.long
    assert ex["memblock_teacher_read_bucket"].dtype == torch.long
    assert ex["memblock_teacher_write_gate"].shape == ex["input_ids"].shape


def test_table2_telemetry_memory_path_runs_with_teacher_signals() -> None:
    # Create a tiny synthetic batch by stacking examples.
    ds_comp = IclRuleInductionDataset(
        block_size=64,
        vocab_size=256,
        n_items=8,
        seed=0,
        n_demos=2,
        gap_bins=[0, 8],
        demo_distractors=2,
        emit_mem_teacher=True,
        mem_buckets=32,
    )
    ds = ds_comp.build()
    b0 = ds[0]
    b1 = ds[1]
    batch = {k: torch.stack([b0[k], b1[k]], dim=0) for k in b0.keys()}
    batch_td = as_tensordict(batch)

    # Fake logits predicting the targets (perfect model) so accuracy should be 1 at read positions.
    B, T = batch_td["target_ids"].shape
    V = 256
    logits = torch.zeros((B, T, V), dtype=torch.float32)
    logits.scatter_(2, batch_td["target_ids"].unsqueeze(-1), 1.0)
    out = Table2Telemetry().compute(outputs={"logits": logits}, batch=batch_td)
    assert "acc/worst_bin" in out
    assert "collision/wrong_item_read_rate" in out
    # ICL rule induction uses the generic path; collision proxy is not defined there.
    assert float(out["collision/wrong_item_read_rate"]) == -1.0


def test_table2_telemetry_memory_path_runs_when_no_bins_present() -> None:
    # Craft a tiny batch with no table2_bin, but with consistent teacher write->read.
    # Sequence length T=6: write bucket 3 at t=1, read bucket 3 at t=4.
    B, T, V = 2, 6, 16
    batch = {
        "input_ids": torch.zeros((B, T), dtype=torch.long),
        "target_ids": torch.zeros((B, T), dtype=torch.long),
        "memblock_teacher_write_gate": torch.zeros((B, T), dtype=torch.float32),
        "memblock_teacher_write_bucket": torch.full((B, T), -1, dtype=torch.long),
        "memblock_teacher_read_bucket": torch.full((B, T), -1, dtype=torch.long),
    }
    batch["memblock_teacher_write_gate"][:, 1] = 1.0
    batch["memblock_teacher_write_bucket"][:, 1] = 3
    batch["memblock_teacher_read_bucket"][:, 4] = 3
    batch_td = as_tensordict(batch)

    # Perfect logits matching targets.
    logits = torch.zeros((B, T, V), dtype=torch.float32)
    logits.scatter_(2, batch_td["target_ids"].unsqueeze(-1), 1.0)
    out = Table2Telemetry().compute(outputs={"logits": logits}, batch=batch_td)
    assert float(out["collision/wrong_item_read_rate"]) == 0.0
    assert float(out["acc/worst_bin"]) == 1.0

