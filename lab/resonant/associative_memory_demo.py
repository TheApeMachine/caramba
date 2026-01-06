"""Resonant Memory Framework (RMF) substrate demo (alias).

This module forwards/aliases to `caramba.lab.synthnn.associative_memory_demo` for backwards
compatibility after a module reorganization.

- Preferred import path: `caramba.lab.resonant.associative_memory_demo`
- Implementation path: `caramba.lab.synthnn.associative_memory_demo`

This forwarding is intended to be temporary; plan to remove the alias after a deprecation
window (suggested target: v0.2.0). New code should import from the preferred path.
"""

from caramba.lab.synthnn.associative_memory_demo import AssociativeMemoryDemoTrainer

__all__ = ["AssociativeMemoryDemoTrainer"]

