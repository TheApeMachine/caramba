"""Instruction set for the streaming memory block (v0)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class MemoryOpcode(IntEnum):
    """Opcode identifiers

    Stable opcode IDs make “control signals” learnable and supervisable; you can
    treat them like a tiny action vocabulary that the model can predict and
    execute during streaming.
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
class MemoryISAV0:
    """ISA helper utilities

    Keeping ISA helpers typed and centralized makes it easier to evolve the
    opcode set without sprinkling numeric constants across the codebase.
    """

    def vocab_size(self) -> int:
        return len(MemoryOpcode)

    def validate_opcode_vocab(self, opcode_vocab: int) -> int:
        v = int(opcode_vocab)
        if v < 2:
            raise ValueError(f"opcode_vocab must be >= 2, got {v}")
        max_v = self.vocab_size()
        if v > max_v:
            raise ValueError(f"opcode_vocab must be <= {max_v} for MemoryISAV0, got {v}")
        return v

    def name_for_id(self, opcode_id: int, *, opcode_vocab: int) -> str:
        v = self.validate_opcode_vocab(opcode_vocab)
        oid = int(opcode_id)
        if oid < 0 or oid >= v:
            raise ValueError(f"opcode_id out of range for opcode_vocab={v}: {oid}")
        return str(MemoryOpcode(oid).name)

    def id_for_opcode(self, opcode: MemoryOpcode, *, opcode_vocab: int) -> int:
        v = self.validate_opcode_vocab(opcode_vocab)
        oid = int(opcode.value)
        if oid < 0 or oid >= v:
            raise ValueError(f"Opcode {opcode.name} (id={oid}) out of range for opcode_vocab={v}")
        return oid

