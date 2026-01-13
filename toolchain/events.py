"""Toolchain event payloads

Defines typed payloads for ToolDefinition and ToolTestResult events.
These payloads are carried inside `EventEnvelope.payload`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from caramba.core.event_codec.payloads import (
    decode_toolchain_definition_payload,
    decode_toolchain_test_result_payload,
    encode_toolchain_definition_payload,
    encode_toolchain_test_result_payload,
)


@dataclass(frozen=True, slots=True)
class ToolCapabilities:
    """Tool capability declaration.

    Capabilities define the sandbox allowances required to execute the tool.
    """

    filesystem: bool = False
    network: bool = False
    process: bool = False
    clock: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "filesystem": bool(self.filesystem),
            "network": bool(self.network),
            "process": bool(self.process),
            "clock": bool(self.clock),
        }

    @staticmethod
    def from_json(obj: Any) -> "ToolCapabilities":
        if not isinstance(obj, dict):
            raise TypeError(f"ToolCapabilities must be a dict, got {type(obj).__name__}")
        return ToolCapabilities(
            filesystem=bool(obj.get("filesystem", False)),
            network=bool(obj.get("network", False)),
            process=bool(obj.get("process", False)),
            clock=bool(obj.get("clock", False)),
        )

    def to_bytes(self) -> dict[str, bool]:
        # Used by ToolDefinitionPayload.to_bytes()
        return {
            "filesystem": bool(self.filesystem),
            "network": bool(self.network),
            "process": bool(self.process),
            "clock": bool(self.clock),
        }


@dataclass(frozen=True, slots=True)
class ToolDefinitionPayload:
    """Tool definition payload.

    Contains everything needed to materialize a tool as an artifact.
    """

    name: str
    version: str
    description: str
    entrypoint: str
    code: str
    tests: str
    capabilities: ToolCapabilities
    requirements: list[str]

    def validate(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("ToolDefinitionPayload.name must be a non-empty string")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("ToolDefinitionPayload.version must be a non-empty string")
        if not isinstance(self.entrypoint, str) or not self.entrypoint.strip():
            raise ValueError("ToolDefinitionPayload.entrypoint must be a non-empty string")
        if not isinstance(self.code, str) or not self.code.strip():
            raise ValueError("ToolDefinitionPayload.code must be a non-empty string")
        if not isinstance(self.tests, str) or not self.tests.strip():
            raise ValueError("ToolDefinitionPayload.tests must be a non-empty string")
        if not isinstance(self.requirements, list) or not all(isinstance(x, str) for x in self.requirements):
            raise TypeError("ToolDefinitionPayload.requirements must be list[str]")

    def to_json(self) -> dict[str, Any]:
        self.validate()
        return {
            "name": str(self.name),
            "version": str(self.version),
            "description": str(self.description),
            "entrypoint": str(self.entrypoint),
            "code": str(self.code),
            "tests": str(self.tests),
            "capabilities": self.capabilities.to_json(),
            "requirements": list(self.requirements),
        }

    @staticmethod
    def from_json(obj: Any) -> "ToolDefinitionPayload":
        if not isinstance(obj, dict):
            raise TypeError(f"ToolDefinitionPayload must be dict, got {type(obj).__name__}")
        caps = ToolCapabilities.from_json(obj.get("capabilities", {}))
        req = obj.get("requirements", [])
        if not isinstance(req, list):
            raise TypeError("ToolDefinitionPayload.requirements must be list")
        payload = ToolDefinitionPayload(
            name=str(obj.get("name", "")),
            version=str(obj.get("version", "")),
            description=str(obj.get("description", "")),
            entrypoint=str(obj.get("entrypoint", "")),
            code=str(obj.get("code", "")),
            tests=str(obj.get("tests", "")),
            capabilities=caps,
            requirements=[str(x) for x in req],
        )
        payload.validate()
        return payload

    def to_bytes(self) -> bytes:
        self.validate()
        return encode_toolchain_definition_payload(
            name=self.name,
            version=self.version,
            description=self.description,
            entrypoint=self.entrypoint,
            code=self.code,
            tests=self.tests,
            capabilities=self.capabilities.to_bytes(),
            requirements=self.requirements,
        )

    @staticmethod
    def from_bytes(payload: bytes) -> "ToolDefinitionPayload":
        name, version, desc, entrypoint, code, tests, caps, reqs = decode_toolchain_definition_payload(payload)
        out = ToolDefinitionPayload(
            name=name,
            version=version,
            description=desc,
            entrypoint=entrypoint,
            code=code,
            tests=tests,
            capabilities=ToolCapabilities(
                filesystem=bool(caps.get("filesystem", False)),
                network=bool(caps.get("network", False)),
                process=bool(caps.get("process", False)),
                clock=bool(caps.get("clock", False)),
            ),
            requirements=reqs,
        )
        out.validate()
        return out


@dataclass(frozen=True, slots=True)
class ToolTestResultPayload:
    """Tool test result payload."""

    name: str
    version: str
    ok: bool
    output: str

    def to_json(self) -> dict[str, Any]:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("ToolTestResultPayload.name must be non-empty")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("ToolTestResultPayload.version must be non-empty")
        return {
            "name": str(self.name),
            "version": str(self.version),
            "ok": bool(self.ok),
            "output": str(self.output),
        }

    def to_bytes(self) -> bytes:
        return encode_toolchain_test_result_payload(
            name=self.name, version=self.version, ok=bool(self.ok), output=self.output
        )

    @staticmethod
    def from_bytes(payload: bytes) -> "ToolTestResultPayload":
        name, version, ok, output = decode_toolchain_test_result_payload(payload)
        return ToolTestResultPayload(name=name, version=version, ok=ok, output=output)

