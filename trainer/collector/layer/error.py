"""Layer stats error types

Defines specific error types for layer stats collection failures, making it easier to
distinguish between collection problems, file corruption, and other issues when
debugging layer stats collection.
"""
from __future__ import annotations

from enum import Enum

from caramba.core.error import CoreError, ErrorType
from caramba.console.logger import Logger

logger: Logger = Logger()


class LayerStatsErrorType(Enum):
    """Layer stats error type

    Categorizes different failure modes in layer stats collection, allowing error
    handlers to provide specific guidance based on what went wrong.
    """
    LAYER_STATS_COLLECTION_FAILED = "layer stats collection failed"
    LAYER_STATS_FINALIZATION_FAILED = "layer stats finalization failed"


class LayerStatsError(CoreError):
    """Layer stats error

    Specialized error class for layer stats collection failures that integrates with the
    framework's error handling system while providing type-safe error checking.
    """
    def __init__(self, err: LayerStatsErrorType, exc: Exception) -> None:
        """Initialize layer stats error

        Creates an error instance with a specific failure type, enabling callers
        to check the error category and respond appropriately.
        """
        super().__init__(ErrorType.LAYER_STATS_ERROR)
        self.layer_stats_err = err
        logger.error(self.layer_stats_err.value, exc=exc)

    def isError(self, err: LayerStatsErrorType) -> bool:
        """Check error type

        Tests whether this error matches a specific failure mode, allowing
        conditional error handling without string comparisons or type checks.
        """
        return self.layer_stats_err == err