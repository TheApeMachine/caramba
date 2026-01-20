"""Verifier component configuration.

This is *not* the same as `config/verify.py` (which defines the verification
thresholds + metrics). This config selects which Verifier implementation
orchestrates verification execution.
"""

from __future__ import annotations

import enum
from typing import Annotated, Literal, TypeAlias

from pydantic import Field

from config import Config


class VerifierType(str, enum.Enum):
    """Available verifier implementations."""

    DEFAULT = "DefaultVerifier"

    @staticmethod
    def module_name() -> str:
        return "caramba.trainer.verifiers"


class DefaultVerifierConfig(Config):
    """Default verifier: runs the existing compare/eval/fidelity/kvcache checks."""

    type: Literal[VerifierType.DEFAULT] = VerifierType.DEFAULT
    enabled: bool = True


VerifierConfig: TypeAlias = Annotated[
    DefaultVerifierConfig,
    Field(discriminator="type"),
]

