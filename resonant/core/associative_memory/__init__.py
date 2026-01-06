"""Associative memory substrate for RMF.

This currently re-exports the reference phasor-coded associative memory that
historically lived under the `synthnn` name.
"""

from synthnn.core.associative_memory import PhaseAssociativeMemory, RecallResult

__all__ = ["PhaseAssociativeMemory", "RecallResult"]

