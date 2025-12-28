"""Message of an agent."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

@dataclass
class Message:
    """Message of an agent."""
    name: str
    role: str
    content: str