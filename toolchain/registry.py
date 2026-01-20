"""Tool registry

Stores versioned tool artifacts on disk and exposes strict APIs for:
- registering a new tool version
- loading a tool version
- listing available tools/versions

This is designed to be deterministic and diff-friendly (JSON metadata + files).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from toolchain.events import ToolCapabilities, ToolDefinitionPayload


@dataclass(frozen=True, slots=True)
class ToolArtifact:
    """Tool artifact record."""

    name: str
    version: str
    path: Path
    sha256: str
    capabilities: ToolCapabilities

    def to_json(self) -> dict[str, Any]:
        return {
            "name": str(self.name),
            "version": str(self.version),
            "path": str(self.path),
            "sha256": str(self.sha256),
            "capabilities": self.capabilities.to_json(),
        }


class ToolRegistry:
    """Tool registry.

    Maintains a versioned on-disk layout:
      root/<name>/<version>/{tool.py, tests.py, requirements.txt, meta.json}
      root/registry.json
    """

    def __init__(self, *, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "registry.json"
        if not self.index_path.exists():
            self.index_path.write_text(json.dumps({"tools": []}, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def register(self, payload: ToolDefinitionPayload) -> ToolArtifact:
        """Register a tool definition as a new versioned artifact."""
        if not isinstance(payload, ToolDefinitionPayload):
            raise TypeError(f"payload must be ToolDefinitionPayload, got {type(payload).__name__}")
        payload.validate()
        tool_dir = self.root / payload.name / payload.version
        tool_dir.mkdir(parents=True, exist_ok=False)

        code_path = tool_dir / "tool.py"
        tests_path = tool_dir / "tests.py"
        req_path = tool_dir / "requirements.txt"
        meta_path = tool_dir / "meta.json"

        code_path.write_text(payload.code, encoding="utf-8")
        tests_path.write_text(payload.tests, encoding="utf-8")
        req_path.write_text("\n".join(payload.requirements) + "\n", encoding="utf-8")
        sha = self.compute_sha256(code_path)
        meta = {
            "name": payload.name,
            "version": payload.version,
            "description": payload.description,
            "entrypoint": payload.entrypoint,
            "sha256": sha,
            "capabilities": payload.capabilities.to_json(),
        }
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        artifact = ToolArtifact(
            name=payload.name,
            version=payload.version,
            path=tool_dir,
            sha256=sha,
            capabilities=payload.capabilities,
        )
        self.write_index(artifact)
        return artifact

    def compute_sha256(self, path: Path) -> str:
        """Compute sha256 of file content."""
        if not isinstance(path, Path):
            raise TypeError("path must be Path")
        if not path.exists():
            raise FileNotFoundError(f"File not found for sha256: {path}")
        h = hashlib.sha256()
        h.update(path.read_bytes())
        return h.hexdigest()

    def write_index(self, artifact: ToolArtifact) -> None:
        """Update registry.json with the artifact record (supersedes existing version entry)."""
        obj = json.loads(self.index_path.read_text(encoding="utf-8"))
        tools = obj.get("tools")
        if not isinstance(tools, list):
            raise TypeError("registry.json must contain list under 'tools'")
        tools = [t for t in tools if not (isinstance(t, dict) and t.get("name") == artifact.name and t.get("version") == artifact.version)]
        tools.append(artifact.to_json())
        obj["tools"] = tools
        self.index_path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")

