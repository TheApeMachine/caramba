"""Errors for the core module

Implements the custom errors for the core module.
"""
from __future__ import annotations
from enum import Enum

from console.logger import Logger
logger: Logger = Logger()


class ErrorType(Enum):
    """Type of error"""
    DATA_ERROR = "data error"
    CHECKPOINT_ERROR = "checkpoint error"
    UNKNOWN = "unknown"

class Error(Exception):
    """Base class for all caramba errors"""
    def __init__(self, err: ErrorType) -> None:
        self.err = err
        logger.error(self.err.value)


class CoreError(Error):
    """Base class for all caramba errors"""
    def __init__(self, err: ErrorType) -> None:
        super().__init__(err)

    def isError(self, err: type[Error]) -> bool:
        raise NotImplementedError("This method must be implemented by the subclass")