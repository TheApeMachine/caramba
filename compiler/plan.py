"""Plan printer: human-readable IR for debugging.

After compilation, you may want to inspect what the config looks like.
The planner renders the lowered manifest as a structured text format,
making it easy to verify that repeats expanded correctly and layers
are configured as expected.
"""
from __future__ import annotations

from typing import Iterable

from config.layer import LayerConfig
from config.manifest import Manifest
from config.model import ModelConfig
from config.target import ExperimentTargetConfig, ProcessTargetConfig, TargetConfig
from config.topology import GraphTopologyConfig, NodeConfig, TopologyConfig


class Planner:
    """Renders human-readable execution plans from lowered manifests.

    Useful for debugging config issues by showing the fully expanded
    layer structure.
    """

    def format(self, manifest: Manifest) -> str:
        """Render a human-readable plan for a lowered manifest."""
        out: list[str] = []
        out.append(f"manifest.version={manifest.version}")
        out.append(f"targets.count={len(manifest.targets)}")
        for i, t in enumerate(manifest.targets):
            out.extend(self.format_target(t, indent=0, path=f"targets[{i}]"))
        return "\n".join(out)

    def format_target(self, target: TargetConfig, *, indent: int, path: str) -> list[str]:
        pad = " " * indent
        out: list[str] = []
        out.append(f"{pad}target.name={target.name} type={target.type} path={path}")

        if isinstance(target, ProcessTargetConfig):
            out.append(f"{pad}process.type={target.process.type} name={target.process.name}")
            out.append(f"{pad}process.leader={getattr(target.process, 'leader', '')}")
            out.append(f"{pad}process.topic={getattr(target.process, 'topic', '')}")
            out.append(f"{pad}team:")
            for k, v in sorted(target.team.root.items()):
                out.append(f"{pad}  - {k}: {v}")
            return out

        if not isinstance(target, ExperimentTargetConfig):
            return out

        out.append(
            f"{pad}components: task={target.task.ref} data={target.data.ref} "
            f"system={target.system.ref} objective={target.objective.ref} trainer={target.trainer.ref}"
        )
        out.append(f"{pad}runs.count={len(target.runs)} benchmarks.count={len(target.benchmarks or [])}")

        # If it's a ModelConfig-driven system, render the topology for debugging.
        if target.system.ref in ("system.language_model", "system.generic"):
            model_payload = target.system.config.get("model", None)
            if isinstance(model_payload, dict):
                try:
                    cfg = ModelConfig.model_validate(model_payload)
                    out.append(f"{pad}model.type={cfg.type.value}")
                    out.append(f"{pad}model.topology:")
                    out.extend(self.format_topology(cfg.topology, indent=indent + 2, path=f"{path}.system.model"))
                except Exception:
                    out.append(f"{pad}model=<invalid>")
            else:
                out.append(f"{pad}model=<missing>")

        return out

    def is_topology(self, node: NodeConfig) -> bool:
        """Check if node is a topology (has layers attribute)."""
        return isinstance(node, GraphTopologyConfig) or hasattr(node, "layers")

    def format_topology(
        self, config: TopologyConfig, *, indent: int, path: str
    ) -> Iterable[str]:
        """Format a topology node with its children."""
        pad = " " * indent
        yield (
            f"{pad}- topology={config.type.value} "
            f"repeat={getattr(config, 'repeat', None)} "
            f"path={path}.topology"
        )

        if isinstance(config, GraphTopologyConfig):
            for i, n in enumerate(config.nodes):
                ins = n.in_keys if isinstance(n.in_keys, str) else list(n.in_keys)
                outs = n.out_keys if isinstance(n.out_keys, str) else list(n.out_keys)
                yield (
                    f"{pad}  - node[{i}].id={n.id} op={n.op} in={ins} out={outs} "
                    f"repeat={getattr(n, 'repeat', 1)}"
                )
            return

        for i, node in enumerate(config.layers):
            yield from self.format_node(node, indent=indent + 2, path=f"{path}.topology.layers[{i}]")

    def format_node(
        self, config: NodeConfig, *, indent: int, path: str
    ) -> Iterable[str]:
        """Format a topology or layer node."""
        if self.is_topology(config):
            yield from self.format_topology(config, indent=indent, path=path)  # type: ignore[arg-type]
        else:
            yield from self.format_layer(config, indent=indent, path=path)  # type: ignore[arg-type]

    def format_layer(
        self, config: LayerConfig, *, indent: int, path: str
    ) -> Iterable[str]:
        """Format a single layer node."""
        pad = " " * indent
        yield f"{pad}- layer={config.type.value} path={path}"
