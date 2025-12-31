"""Code/model graph sync process (Graphiti ingestion).

This process indexes the *current structural reality* of a manifest (model topology
and LayerConfig hyperparameters) into Graphiti so agents can query it before
proposing architecture changes.
"""

from __future__ import annotations

import json
import re
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING, Iterable

from caramba.agent.process import Process
from caramba.console import logger
from caramba.config.manifest import Manifest
from caramba.config.model import ModelConfig
from caramba.config.target import ExperimentTargetConfig

if TYPE_CHECKING:
    from caramba.agent import Researcher
    from caramba.config.topology import NodeConfig


@dataclass(frozen=True, slots=True)
class LayerRecord:
    name: str
    path: str
    properties: dict[str, Any]


def _collect_models(manifest: Manifest) -> list[tuple[str, ModelConfig]]:
    """Extract ModelConfig(s) referenced by experiment targets."""
    out: list[tuple[str, ModelConfig]] = []
    for t in manifest.targets:
        if not isinstance(t, ExperimentTargetConfig):
            continue
        if t.system.ref not in ("system.language_model", "system.generic"):
            continue
        model_payload = t.system.config.get("model", None)
        if not isinstance(model_payload, dict):
            continue
        try:
            cfg = ModelConfig.model_validate(model_payload)
        except Exception as e:
            logger.error(
                f"code_graph_sync: ModelConfig validation failed for target={getattr(t, 'name', None)!r}: {e}"
            )
            logger.error(
                f"code_graph_sync: model_payload for target={getattr(t, 'name', None)!r}: {model_payload!r}"
            )
            logger.error(traceback.format_exc())
            continue
        out.append((str(t.name), cfg))
    return out


def _iter_layers(node: "NodeConfig", *, path: str) -> Iterable[tuple["NodeConfig", str]]:
    """Yield (layer_node, path) in a deterministic, execution-ish order.

    We expand `repeat` by unrolling into `repeat[i]` path segments to make
    layer IDs stable and unique.
    """
    layers = getattr(node, "layers", None)
    if isinstance(layers, list):
        repeat = int(getattr(node, "repeat", 1) or 1)
        repeat = max(1, repeat)
        for r in range(repeat):
            for i, child in enumerate(layers):
                child_path = f"{path}.repeat[{r}].layers[{i}]"
                yield from _iter_layers(child, path=child_path)  # type: ignore[arg-type]
        return
    # Leaf node (assume it is a LayerConfig-like object).
    yield node, path


def _layer_props(layer: object) -> dict[str, Any]:
    """Extract a stable property dict from a LayerConfig-ish object."""
    props: dict[str, Any] = {}
    dump = getattr(layer, "model_dump", None)
    if callable(dump):
        try:
            props = dict(dump())  # type: ignore[misc]
        except Exception:
            props = {}

    # Normalize config type (enum -> string) for readability.
    t = getattr(layer, "type", None)
    if t is not None:
        props["type"] = getattr(t, "value", str(t))
    # Normalize attention mode if present.
    mode = getattr(layer, "mode", None)
    if mode is not None:
        props["mode"] = getattr(mode, "value", str(mode))
    return props


class CodeGraphSync(Process):
    """Sync manifest model topology into Graphiti memory."""

    def __init__(
        self,
        agents: dict[str, "Researcher"],
        *,
        agent_key: str,
        index_namespace: str = "main",
    ) -> None:
        super().__init__(agents)
        self.agent_key = str(agent_key)
        self.index_namespace = str(index_namespace or "main")

    async def run(self, *, manifest: Manifest, manifest_path: Path | None) -> dict[str, Any]:
        models = _collect_models(manifest)
        if not models:
            return {
                "ok": False,
                "error": "no_model_configs_found",
                "hint": "No experiment target contained system.config.model for system.language_model/system.generic.",
            }

        manifest_name = str(getattr(manifest, "name", "") or (manifest_path.stem if manifest_path else "manifest"))
        agent = self.agents[self.agent_key]

        indexed_targets: list[str] = []
        payloads: list[dict[str, Any]] = []

        for target_name, model_cfg in models:
            prefix = f"{self.index_namespace}:{manifest_name}:{target_name}"
            layers: list[LayerRecord] = []
            for layer_node, path in _iter_layers(model_cfg.topology, path="topology"):
                layer_id = f"{prefix}:{path}"
                layers.append(
                    LayerRecord(
                        name=layer_id,
                        path=path,
                        properties=_layer_props(layer_node),
                    )
                )

            # Sequential dependency edges between adjacent layer records.
            deps: list[dict[str, str]] = []
            for i in range(1, len(layers)):
                deps.append(
                    {
                        "dependent": layers[i].name,
                        "dependency": layers[i - 1].name,
                        "rel": "DEPENDS_ON",
                    }
                )

            payload = {
                "kind": "caramba_code_graph_sync",
                "manifest": manifest_name,
                "target": target_name,
                "index_namespace": self.index_namespace,
                "layer_id_prefix": prefix,
                "layers": [
                    {"name": lr.name, "path": lr.path, **lr.properties} for lr in layers
                ],
                "depends_on": deps,
            }
            payloads.append(payload)
            indexed_targets.append(target_name)

        # Ask the agent to ingest via Graphiti MCP tools.
        # We intentionally include the JSON payload directly to avoid any hallucinated structure.
        prompt = (
            "You are performing deterministic graph ingestion for Caramba.\n\n"
            "You MUST call the Graphiti tool `add_memory` once per payload below.\n"
            "For each payload:\n"
            "  - name: \"caramba_layer_index\"\n"
            "  - episode_body: the payload JSON\n"
            "  - source: \"json\"\n"
            f"  - group_id: \"{self.index_namespace}\"\n\n"
            "After ingesting all payloads, return STRICT JSON: {\"ok\": true, \"ingested\": <int>}.\n\n"
            "<payloads_json>\n"
            f"{json.dumps(payloads, ensure_ascii=False, indent=2)}\n"
            "</payloads_json>\n"
        )

        res = await agent.run(prompt, context=None)
        raw = str(getattr(res, "content", "") or "")

        logger.success(
            f"Requested Graphiti ingestion for {len(payloads)} target(s): {', '.join(indexed_targets)}"
        )

        ingested = 0
        ok = False
        err: str | None = None
        parsed: dict[str, Any] | None = None
        try:
            s = raw.strip()
            try:
                obj = json.loads(s)
                parsed = obj if isinstance(obj, dict) else None
            except Exception:
                m = re.search(r"\{[\s\S]*\}", s)
                if m:
                    obj2 = json.loads(m.group(0))
                    parsed = obj2 if isinstance(obj2, dict) else None

            if parsed is None:
                err = "invalid_agent_response_json"
            else:
                ok = bool(parsed.get("ok", False))
                try:
                    ingested = int(parsed.get("ingested", 0) or 0)
                except (TypeError, ValueError):
                    ingested = 0
                if not ok:
                    err = str(parsed.get("error", "") or "agent_reported_failure")
            if err or not ok:
                logger.error(
                    f"code_graph_sync: agent ingestion failed (ok={ok}, ingested={ingested}, error={err})"
                )
                logger.error(f"code_graph_sync: raw agent_response: {raw}")
        except Exception:
            ok = False
            ingested = 0
            err = "failed_to_parse_agent_response"
            logger.error("code_graph_sync: failed to parse agent response JSON")
            logger.error(traceback.format_exc())

        return {
            "ok": bool(ok),
            "process": "code_graph_sync",
            "manifest": manifest_name,
            "index_namespace": self.index_namespace,
            "targets": indexed_targets,
            "payload_count": len(payloads),
            "ingested": int(ingested),
            "error": err,
            "agent_response": raw,
        }

