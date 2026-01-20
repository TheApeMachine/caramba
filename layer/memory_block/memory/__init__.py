"""Hard-addressed memory subsystem

This treats memory as a first-class, explicit component: reads and writes are
hard-addressed and stateful, which makes “what is stored” and “how it is
retrieved” an architectural choice you can study and change.
"""

from layer.memory_block.memory.memory import MemoryBlockMemory

__all__ = ["MemoryBlockMemory"]

