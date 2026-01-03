"""MOSAIC "Differentiable VM" instruction set (v0).

This module is MOSAIC-specific: Caramba is the substrate; MOSAIC is one
architecture built on it. Other architectures can define their own ISAs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class MosaicOpcode(IntEnum):
    """v0 opcode IDs (contiguous from 0).

    Notes:
    - IDs are stable and intended to be used as supervision targets.
    - Model configs may use a smaller `opcode_vocab` (prefix of this enum).
    """

    NOP = 0
    READ_MEM = 1
    WRITE_MEM = 2
    CLEAR_MEM = 3
    IDLE = 4
    GATE_UP = 5
    GATE_DOWN = 6
    SCAN = 7
    COMMIT = 8
    RESPOND = 9


@dataclass(frozen=True, slots=True)
class MosaicISAV0:
    """Typed helpers for the v0 opcode vocabulary."""

    def vocab_size(self) -> int:
        return len(MosaicOpcode)

    def validate_opcode_vocab(self, opcode_vocab: int) -> int:
        v = int(opcode_vocab)
        if v < 2:
            raise ValueError(f"opcode_vocab must be >= 2, got {v}")
        max_v = self.vocab_size()
        if v > max_v:
            raise ValueError(f"opcode_vocab must be <= {max_v} for MosaicISAV0, got {v}")
        return v

    def name_for_id(self, opcode_id: int, *, opcode_vocab: int) -> str:
        v = self.validate_opcode_vocab(opcode_vocab)
        oid = int(opcode_id)
        if oid < 0 or oid >= v:
            raise ValueError(f"opcode_id out of range for opcode_vocab={v}: {oid}")
        return str(MosaicOpcode(oid).name)

    def id_for_opcode(self, opcode: MosaicOpcode, *, opcode_vocab: int) -> int:
        v = self.validate_opcode_vocab(opcode_vocab)
        oid = int(opcode.value)
        if oid < 0 or oid >= v:
            raise ValueError(f"Opcode {opcode.name} (id={oid}) out of range for opcode_vocab={v}")
        return oid

