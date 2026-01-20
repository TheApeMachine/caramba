from __future__ import annotations

import pytest

from trainer.checkpointers.default import DefaultCheckPointer


def test_validate_checkpoint_state_rejects_missing_keys() -> None:
    bad: dict[str, object] = {"run_id": "r", "phase": "global", "step": 1}
    with pytest.raises(ValueError):
        DefaultCheckPointer._validate(bad)


def test_validate_checkpoint_state_accepts_valid_state() -> None:
    """Test that a complete, valid checkpoint state dict is accepted."""
    valid_state: dict[str, object] = {
        "run_id": "test_run",
        "phase": "global",
        "step": 100,
        "student_state_dict": {},
    }
    try:
        DefaultCheckPointer._validate(valid_state)
    except ValueError as e:
        pytest.fail(f"Valid state should be accepted, but got ValueError: {e}")

