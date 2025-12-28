"""Knowledge item sourced from a data source."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

@dataclass
class Knowledge:
    """Knowledge item sourced from a data source."""
    name: str
    content: str
    source: str