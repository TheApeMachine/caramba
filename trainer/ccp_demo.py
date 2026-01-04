"""CCP demo trainer

Runs the Unknown Format Decoder party-trick as a manifest-driven target:
- generate an unknown-format sample in the Lab
- propose a ToolDefinition (decoder tool)
- run tests via Toolchain
- simulate a version bump and repair the tool
- write a deterministic trace

This is intentionally deterministic and framework-first. A trained MOSAIC kernel
can later replace the tool proposal policy without changing the Toolchain/Evaluator/Lab.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from caramba.config.manifest import Manifest
from caramba.config.target import ExperimentTargetConfig
from caramba.core.event import EventEnvelope
from caramba.core.event_bus import EventBus
from caramba.evaluator.policy import PolicyConfig, PolicyGate
from caramba.evaluator.validity import ValidityGate
from caramba.lab.unknown_format.dataset import UnknownFormatLabDataset
from caramba.runtime.trace.writer import TraceWriter
from caramba.toolchain.events import ToolCapabilities, ToolDefinitionPayload
from caramba.toolchain.handler import ToolchainHandler
from caramba.toolchain.registry import ToolRegistry
from caramba.toolchain.sandbox import ToolSandbox
from caramba.toolchain.test_runner import ToolTestRunner


@dataclass(slots=True)
class CcpDemoTrainer:
    """CCP demo trainer.

    Implements the demo as a trainer so it can be run by the manifest engine.
    """

    output_dir: str = "artifacts/ccp_demo"
    seed: int = 1337
    n_items: int = 1
    tool_root: str = "artifacts/ccp_demo/tools"

    def run(self, *, manifest: Manifest, target: ExperimentTargetConfig, engine: Any, dry_run: bool = False) -> dict[str, Path]:
        if dry_run:
            return {}
        out_dir = Path(str(self.output_dir))
        out_dir.mkdir(parents=True, exist_ok=True)
        trace = TraceWriter(path=out_dir / "trace.jsonl")
        bus = EventBus()

        validity = ValidityGate()
        policy = PolicyGate(config=PolicyConfig())
        registry = ToolRegistry(root=Path(str(self.tool_root)))
        tester = ToolTestRunner(sandbox=ToolSandbox())
        toolchain = ToolchainHandler(bus=bus, registry=registry, tester=tester)
        bus.subscribe("ToolDefinition", toolchain)

        dataset = UnknownFormatLabDataset(seed=int(self.seed), n_items=int(self.n_items))
        sample = dataset.sample(0)
        self.append_trace(trace, kind="RawBytesChunk", payload={"raw_hex": sample.raw.hex()})

        tool_payload = self.propose_tool(sample_raw_hex=sample.raw.hex(), tests_src=sample.tests, version="v1")
        policy.validate_tool_definition(tool_payload, revision_index=0)
        self.publish_tool(bus, validity, trace, tool_payload)
        _ = bus.drain()

        bumped = self.bump_version(sample_raw_hex=sample.raw.hex())
        self.append_trace(trace, kind="FormatVersionBump", payload={"raw_hex": bumped.hex()})
        bad_payload = self.propose_tool(sample_raw_hex=bumped.hex(), tests_src=sample.tests, version="v2")
        policy.validate_tool_definition(bad_payload, revision_index=1)
        self.publish_tool(bus, validity, trace, bad_payload)
        _ = bus.drain()

        fixed_payload = self.propose_tool(sample_raw_hex=bumped.hex(), tests_src=sample.tests, version="v3")
        policy.validate_tool_definition(fixed_payload, revision_index=2)
        self.publish_tool(bus, validity, trace, fixed_payload)
        _ = bus.drain()

        return {"trace": out_dir / "trace.jsonl"}

    def publish_tool(self, bus: EventBus, validity: ValidityGate, trace: TraceWriter, payload: ToolDefinitionPayload) -> None:
        ev = EventEnvelope(type="ToolDefinition", payload=payload.to_json(), sender="ccp_demo")
        validity.validate_event(ev)
        self.append_trace(trace, kind="ToolDefinition", payload=payload.to_json())
        bus.publish(ev)

    def propose_tool(self, *, sample_raw_hex: str, tests_src: str, version: str) -> ToolDefinitionPayload:
        code = self.decoder_code()
        return ToolDefinitionPayload(
            name="unknown_format_decoder",
            version=str(version),
            description="Decode unknown-format records into payload bytes.",
            entrypoint="tool:decode",
            code=code,
            tests=str(tests_src),
            capabilities=ToolCapabilities(filesystem=False, network=False, process=False, clock=False),
            requirements=[],
        )

    def decoder_code(self) -> str:
        return (
            "def decode(raw: bytes):\n"
            "    if not isinstance(raw, (bytes, bytearray)):\n"
            "        raise TypeError('raw must be bytes-like')\n"
            "    buf = bytes(raw)\n"
            "    if len(buf) < 2:\n"
            "        raise ValueError('raw too short')\n"
            "    version = buf[0]\n"
            "    count = buf[1]\n"
            "    pos = 2\n"
            "    out = []\n"
            "    for _ in range(count):\n"
            "        if pos >= len(buf):\n"
            "            raise ValueError('truncated')\n"
            "        n = buf[pos]\n"
            "        pos += 1\n"
            "        if pos + n + 1 > len(buf):\n"
            "            raise ValueError('truncated')\n"
            "        payload = buf[pos:pos+n]\n"
            "        pos += n\n"
            "        chk = buf[pos]\n"
            "        pos += 1\n"
            "        if (sum(payload) % 256) != chk:\n"
            "            raise ValueError('checksum mismatch')\n"
            "        out.append(payload)\n"
            "    if pos != len(buf):\n"
            "        raise ValueError('extra bytes')\n"
            "    return out\n"
        )

    def bump_version(self, *, sample_raw_hex: str) -> bytes:
        raw = bytes.fromhex(str(sample_raw_hex))
        if len(raw) < 1:
            raise ValueError("raw too short to bump")
        b = bytearray(raw)
        b[0] = (int(b[0]) + 1) % 256
        return bytes(b)

    def append_trace(self, trace: TraceWriter, *, kind: str, payload: dict[str, Any]) -> None:
        trace.append(kind=str(kind), payload=dict(payload), ts=float(time.time()))

