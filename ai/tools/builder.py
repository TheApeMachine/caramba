"""Native tool building (scaffold + registry).

This is a best-effort, local mechanism:
- The model can emit a ToolDefinition event (EventEnvelope.type == "ToolDefinition")
- We validate the definition and scaffold a new tool under ai/tools/generated/<name>/
- We publish ToolRegistered / ToolRejected events to the EventBus

This intentionally does NOT implement MCP itself; it produces a durable artifact
that can be wired as an MCP server later (or executed as local python).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from caramba.core.event import EventEnvelope
from caramba.core.event_bus import EventBus, EventHandler


_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    implementation: str
    requirements: list[str] | None = None

    @staticmethod
    def from_payload(payload: Any) -> "ToolDefinition":
        if not isinstance(payload, dict):
            raise TypeError(f"ToolDefinition payload must be a dict, got {type(payload).__name__}")
        name = payload.get("name")
        desc = payload.get("description", "")
        impl = payload.get("implementation", "")
        reqs = payload.get("requirements", None)
        if not isinstance(name, str):
            raise TypeError("ToolDefinition.name must be a string")
        if not isinstance(desc, str):
            raise TypeError("ToolDefinition.description must be a string")
        if not isinstance(impl, str):
            raise TypeError("ToolDefinition.implementation must be a string")
        if reqs is not None and not (isinstance(reqs, list) and all(isinstance(x, str) for x in reqs)):
            raise TypeError("ToolDefinition.requirements must be a list[str] when provided")
        return ToolDefinition(name=name.strip(), description=desc.strip(), implementation=impl, requirements=reqs)

    def validate(self) -> list[str]:
        errs: list[str] = []
        if not self.name or not _NAME_RE.match(self.name):
            errs.append("name must be a valid identifier (^[A-Za-z_][A-Za-z0-9_]*$)")
        if len(self.description) > 4000:
            errs.append("description too long")
        if not self.implementation.strip():
            errs.append("implementation must be non-empty python source")
        return errs


class ToolBuilderHandler(EventHandler):
    """Event handler that builds tools from ToolDefinition events."""

    def __init__(self, *, bus: EventBus, out_dir: Path | str = Path("ai/tools/generated")) -> None:
        self.bus = bus
        self.out_dir = Path(out_dir)

    def handle(self, event: EventEnvelope) -> None:
        if str(event.type) != "ToolDefinition":
            return

        try:
            td = ToolDefinition.from_payload(event.payload)
            errs = td.validate()
            if errs:
                raise ValueError("; ".join(errs))
            tool_dir = self.out_dir / td.name
            tool_dir.mkdir(parents=True, exist_ok=True)

            (tool_dir / "__init__.py").write_text("", encoding="utf-8")
            (tool_dir / "README.md").write_text(
                f"# {td.name}\n\n{td.description}\n", encoding="utf-8"
            )
            (tool_dir / "tool.py").write_text(td.implementation, encoding="utf-8")
            if td.requirements:
                (tool_dir / "requirements.txt").write_text("\n".join(td.requirements) + "\n", encoding="utf-8")

            # Update registry.json
            reg_path = self.out_dir / "registry.json"
            registry: dict[str, Any] = {"tools": []}
            if reg_path.exists():
                try:
                    registry = json.loads(reg_path.read_text(encoding="utf-8")) or registry
                except Exception:
                    registry = {"tools": []}
            tools = registry.get("tools")
            if not isinstance(tools, list):
                tools = []
            tools = [t for t in tools if not (isinstance(t, dict) and t.get("name") == td.name)]
            tools.append({"name": td.name, "description": td.description, "path": str(tool_dir)})
            registry["tools"] = tools
            reg_path.parent.mkdir(parents=True, exist_ok=True)
            reg_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            self.bus.publish(
                EventEnvelope(
                    type="ToolRegistered",
                    payload={"name": td.name, "path": str(tool_dir)},
                    sender=str(event.sender),
                    priority=int(event.priority),
                )
            )
        except Exception as e:
            self.bus.publish(
                EventEnvelope(
                    type="ToolRejected",
                    payload={"error": f"{type(e).__name__}: {e}"},
                    sender=str(event.sender),
                    priority=int(event.priority),
                )
            )

