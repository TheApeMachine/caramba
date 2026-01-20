"""Checkpoint error types

Defines specific error types for checkpoint loading failures, making it easier to
distinguish between file not found, file is a directory, and other issues when
debugging checkpoint loading.
"""
from __future__ import annotations

from enum import Enum

from core.error import CoreError, ErrorType
from console.logger import Logger

logger: Logger = Logger()


class CheckpointErrorType(Enum):
    """Checkpoint error type

    Categorizes different failure modes in checkpoint loading, allowing error
    handlers to provide specific guidance based on what went wrong.
    """
    CHECKPOINT_NOT_FOUND = "checkpoint not found"
    CHECKPOINT_IS_DIRECTORY = "checkpoint is a directory"
    CHECKPOINT_UNSUPPORTED = "checkpoint unsupported"
    CHECKPOINT_INVALID_FORMAT = "checkpoint invalid format"
    CHECKPOINT_DUPLICATE_KEY = "checkpoint duplicate key"


class CheckpointError(CoreError):
    """Checkpoint error

    Specialized error class for checkpoint loading failures that integrates with the
    framework's error handling system while providing type-safe error checking.
    """
    def __init__(self, err: CheckpointErrorType) -> None:
        """Initialize checkpoint error

        Creates an error instance with a specific failure type, enabling callers
        to check the error category and respond appropriately.
        """
        super().__init__(ErrorType.CHECKPOINT_ERROR)

    def isError(self, err: CheckpointErrorType) -> bool:
        """Check error type

        Tests whether this error matches a specific failure mode, allowing
        conditional error handling without string comparisons or type checks.
        """
        return self.err == err