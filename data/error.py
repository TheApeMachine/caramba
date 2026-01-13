"""Data error types

Defines specific error types for data loading failures, making it easier to
distinguish between download problems, file corruption, and other issues when
debugging dataset preparation.
"""
from __future__ import annotations

from enum import Enum

from caramba.core.error import CoreError, ErrorType
from caramba.console.logger import Logger

logger: Logger = Logger()


class DataErrorType(Enum):
    """Data error type

    Categorizes different failure modes in dataset preparation, allowing error
    handlers to provide specific guidance based on what went wrong.
    """
    DATASET_BUILD_FAILED = "dataset build failed"
    DATASET_DOWNLOAD_FAILED = "dataset download failed"
    DATASET_LOAD_FAILED = "dataset load failed"
    DATASET_UNSUPPORTED = "dataset unsupported"
    DATASET_UNSUPPORTED_TOKENIZER = "dataset unsupported tokenizer"


class DataError(CoreError):
    """Data error

    Specialized error class for data-related failures that integrates with the
    framework's error handling system while providing type-safe error checking.
    """
    def __init__(self, err: DataErrorType) -> None:
        """Initialize data error

        Creates an error instance with a specific failure type, enabling callers
        to check the error category and respond appropriately.
        """
        super().__init__(ErrorType.DATA_ERROR)
        self.data_err = err

    def isError(self, err: DataErrorType) -> bool:
        """Check error type

        Tests whether this error matches a specific failure mode, allowing
        conditional error handling without string comparisons or type checks.
        """
        return self.data_err == err