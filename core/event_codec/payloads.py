"""Typed Cap'n Proto payload helpers.

`EventEnvelope.payload` is bytes-only and is expected to be a Cap'n Proto-encoded
payload struct appropriate for the event `type`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import capnp


def _load_payload_schema():
    schema_path = Path(__file__).parent / "event_payloads.capnp"
    if not schema_path.exists():
        raise FileNotFoundError(f"Cap'n Proto payload schema not found: {schema_path}")
    capnp_any = cast(Any, capnp)
    return capnp_any.load(str(schema_path))


_payload_schema = None


def _get_payload_schema():
    global _payload_schema
    if _payload_schema is None:
        _payload_schema = _load_payload_schema()
    return _payload_schema


def encode_message_payload(*, text: str) -> bytes:
    schema = _get_payload_schema()
    msg = schema.MessagePayload.new_message()
    msg.text = str(text)
    return msg.to_bytes()


def decode_message_payload(payload: bytes) -> str:
    schema = _get_payload_schema()
    with schema.MessagePayload.from_bytes(payload) as msg:
        return str(msg.text)


def encode_memory_write_payload(*, key: int, value: int) -> bytes:
    schema = _get_payload_schema()
    msg = schema.MemoryWritePayload.new_message()
    msg.key = int(key)
    msg.value = int(value)
    return msg.to_bytes()


def decode_memory_write_payload(payload: bytes) -> tuple[int, int]:
    schema = _get_payload_schema()
    with schema.MemoryWritePayload.from_bytes(payload) as msg:
        return int(msg.key), int(msg.value)


def encode_memory_query_payload(*, key: int) -> bytes:
    schema = _get_payload_schema()
    msg = schema.MemoryQueryPayload.new_message()
    msg.key = int(key)
    return msg.to_bytes()


def decode_memory_query_payload(payload: bytes) -> int:
    schema = _get_payload_schema()
    with schema.MemoryQueryPayload.from_bytes(payload) as msg:
        return int(msg.key)


def encode_memory_answer_payload(*, value: int) -> bytes:
    schema = _get_payload_schema()
    msg = schema.MemoryAnswerPayload.new_message()
    msg.value = int(value)
    return msg.to_bytes()


def decode_memory_answer_payload(payload: bytes) -> int:
    schema = _get_payload_schema()
    with schema.MemoryAnswerPayload.from_bytes(payload) as msg:
        return int(msg.value)


def encode_noise_payload(*, tok: int) -> bytes:
    schema = _get_payload_schema()
    msg = schema.NoisePayload.new_message()
    msg.tok = int(tok)
    return msg.to_bytes()


def decode_noise_payload(payload: bytes) -> int:
    schema = _get_payload_schema()
    with schema.NoisePayload.from_bytes(payload) as msg:
        return int(msg.tok)


def encode_idle_payload(*, ts: float, metrics: dict[str, float]) -> bytes:
    schema = _get_payload_schema()
    msg = schema.IdlePayload.new_message()
    msg.ts = float(ts)
    items = [(str(k), float(v)) for k, v in metrics.items()]
    mlist = msg.init("metrics", len(items))
    for i, (k, v) in enumerate(items):
        mlist[i].key = k
        mlist[i].value = float(v)
    return msg.to_bytes()


def decode_idle_payload(payload: bytes) -> tuple[float, dict[str, float]]:
    schema = _get_payload_schema()
    with schema.IdlePayload.from_bytes(payload) as msg:
        metrics = {str(m.key): float(m.value) for m in msg.metrics}
        return float(msg.ts), metrics


def encode_tool_definition_payload(
    *,
    name: str,
    description: str,
    implementation: str,
    requirements: list[str] | None = None,
) -> bytes:
    schema = _get_payload_schema()
    msg = schema.ToolBuilderDefinitionPayload.new_message()
    msg.name = str(name)
    msg.description = str(description)
    msg.implementation = str(implementation)
    reqs = [] if requirements is None else [str(r) for r in requirements]
    rlist = msg.init("requirements", len(reqs))
    for i, r in enumerate(reqs):
        rlist[i] = r
    return msg.to_bytes()


def decode_tool_definition_payload(
    payload: bytes,
) -> tuple[str, str, str, list[str]]:
    schema = _get_payload_schema()
    with schema.ToolBuilderDefinitionPayload.from_bytes(payload) as msg:
        reqs = [str(r) for r in msg.requirements]
        return str(msg.name), str(msg.description), str(msg.implementation), reqs


def encode_tool_registered_payload(*, name: str, path: str) -> bytes:
    schema = _get_payload_schema()
    msg = schema.ToolBuilderRegisteredPayload.new_message()
    msg.name = str(name)
    msg.path = str(path)
    return msg.to_bytes()


def decode_tool_registered_payload(payload: bytes) -> tuple[str, str]:
    schema = _get_payload_schema()
    with schema.ToolBuilderRegisteredPayload.from_bytes(payload) as msg:
        return str(msg.name), str(msg.path)


def encode_tool_rejected_payload(*, error: str) -> bytes:
    schema = _get_payload_schema()
    msg = schema.ToolBuilderRejectedPayload.new_message()
    msg.error = str(error)
    return msg.to_bytes()


def decode_tool_rejected_payload(payload: bytes) -> str:
    schema = _get_payload_schema()
    with schema.ToolBuilderRejectedPayload.from_bytes(payload) as msg:
        return str(msg.error)


def encode_impulse_payload(
    *,
    metrics: dict[str, float],
    signals: list[dict[str, Any]],
    max_urgency: float,
) -> bytes:
    schema = _get_payload_schema()
    msg = schema.ImpulsePayload.new_message()

    m_items = [(str(k), float(v)) for k, v in metrics.items()]
    mlist = msg.init("metrics", len(m_items))
    for i, (k, v) in enumerate(m_items):
        mlist[i].key = k
        mlist[i].value = float(v)

    slist = msg.init("signals", len(signals))
    for i, s in enumerate(signals):
        slist[i].name = str(s.get("name", ""))
        slist[i].metric = str(s.get("metric", ""))
        slist[i].value = float(s.get("value", 0.0))
        band = slist[i].band
        b = s.get("band", {}) if isinstance(s.get("band", {}), dict) else {}
        band.minValue = float(b.get("min", 0.0))
        band.maxValue = float(b.get("max", 0.0))
        slist[i].deviation = float(s.get("deviation", 0.0))
        slist[i].urgency = float(s.get("urgency", 0.0))

    msg.maxUrgency = float(max_urgency)
    return msg.to_bytes()


def decode_impulse_payload(payload: bytes) -> tuple[dict[str, float], list[dict[str, Any]], float]:
    schema = _get_payload_schema()
    with schema.ImpulsePayload.from_bytes(payload) as msg:
        metrics = {str(m.key): float(m.value) for m in msg.metrics}
        signals: list[dict[str, Any]] = []
        for s in msg.signals:
            signals.append(
                {
                    "name": str(s.name),
                    "metric": str(s.metric),
                    "value": float(s.value),
                    "band": {"min": float(s.band.minValue), "max": float(s.band.maxValue)},
                    "deviation": float(s.deviation),
                    "urgency": float(s.urgency),
                }
            )
        return metrics, signals, float(msg.maxUrgency)


def encode_toolchain_definition_payload(
    *,
    name: str,
    version: str,
    description: str,
    entrypoint: str,
    code: str,
    tests: str,
    capabilities: dict[str, bool] | None = None,
    requirements: list[str] | None = None,
) -> bytes:
    schema = _get_payload_schema()
    msg = schema.ToolchainDefinitionPayload.new_message()
    msg.name = str(name)
    msg.version = str(version)
    msg.description = str(description)
    msg.entrypoint = str(entrypoint)
    msg.code = str(code)
    msg.tests = str(tests)
    caps = capabilities or {}
    msg.capabilities.filesystem = bool(caps.get("filesystem", False))
    msg.capabilities.network = bool(caps.get("network", False))
    msg.capabilities.process = bool(caps.get("process", False))
    msg.capabilities.clock = bool(caps.get("clock", False))
    reqs = [] if requirements is None else [str(r) for r in requirements]
    rlist = msg.init("requirements", len(reqs))
    for i, r in enumerate(reqs):
        rlist[i] = r
    return msg.to_bytes()


def decode_toolchain_definition_payload(
    payload: bytes,
) -> tuple[str, str, str, str, str, str, dict[str, bool], list[str]]:
    schema = _get_payload_schema()
    with schema.ToolchainDefinitionPayload.from_bytes(payload) as msg:
        caps = {
            "filesystem": bool(msg.capabilities.filesystem),
            "network": bool(msg.capabilities.network),
            "process": bool(msg.capabilities.process),
            "clock": bool(msg.capabilities.clock),
        }
        reqs = [str(r) for r in msg.requirements]
        return (
            str(msg.name),
            str(msg.version),
            str(msg.description),
            str(msg.entrypoint),
            str(msg.code),
            str(msg.tests),
            caps,
            reqs,
        )


def encode_toolchain_test_result_payload(*, name: str, version: str, ok: bool, output: str) -> bytes:
    schema = _get_payload_schema()
    msg = schema.ToolchainTestResultPayload.new_message()
    msg.name = str(name)
    msg.version = str(version)
    msg.ok = bool(ok)
    msg.output = str(output)
    return msg.to_bytes()


def decode_toolchain_test_result_payload(payload: bytes) -> tuple[str, str, bool, str]:
    schema = _get_payload_schema()
    with schema.ToolchainTestResultPayload.from_bytes(payload) as msg:
        return str(msg.name), str(msg.version), bool(msg.ok), str(msg.output)

