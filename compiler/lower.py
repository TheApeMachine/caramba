"""Lowering pass: expand user shortcuts to canonical configs.

Configs often use shortcuts for brevity (e.g., `repeat: 16` instead of
listing 16 layers). The lowerer expands these so downstream code sees
fully explicit, uniform structures.
"""
from __future__ import annotations

from config.manifest import Manifest
from config.model import ModelConfig
from config.target import ExperimentTargetConfig
from config.topology import NodeConfig, TopologyConfig


class Lowerer:
    """Expands repeat declarations at compile-time.

    Transforms `repeat: N` into N explicit copies of the repeated structure,
    making the config canonical for model construction.
    """

    def lower_manifest(self, manifest: Manifest) -> Manifest:
        """Lower a manifest into canonical form.

        In manifest v2 we lower component configs where we recognize canonical
        structures. Currently:
        - `system.language_model` with `config.model` (a ModelConfig payload)
        """
        lowered_targets = []
        for t in list(getattr(manifest, "targets", [])):
            if isinstance(t, ExperimentTargetConfig) and t.system.ref == "system.language_model":
                model_payload = t.system.config.get("model", None)
                if isinstance(model_payload, dict):
                    cfg = ModelConfig.model_validate(model_payload)
                    lowered = self.lower_model(cfg)
                    t2 = t.model_copy(deep=True)
                    t2.system.config["model"] = lowered.model_dump()
                    lowered_targets.append(t2)
                    continue
            lowered_targets.append(t)
        return manifest.model_copy(update={"targets": lowered_targets})

    def lower_model(self, model: ModelConfig) -> ModelConfig:
        """Lower a model config into canonical form."""
        lowered_topology = self.lower_topology(model.topology)
        return model.model_copy(update={"topology": lowered_topology})

    def lower_topology(self, config: TopologyConfig) -> TopologyConfig:
        """Expand topology-level repeat.

        Recursively lowers child nodes, then replicates the result
        according to the repeat count.
        """
        lowered = self.lower_nodes(list(config.layers))
        layers = self.repeat_nodes(lowered, repeat=int(config.repeat))
        return config.model_copy(update={"layers": layers, "repeat": 1})

    def lower_nodes(self, nodes: list[NodeConfig]) -> list[NodeConfig]:
        """Lower nested topology nodes recursively."""
        return [
            self.lower_topology(node) if self.is_topology(node) else node  # type: ignore[arg-type]
            for node in nodes
        ]

    def repeat_nodes(self, nodes: list[NodeConfig], *, repeat: int) -> list[NodeConfig]:
        """Repeat nodes by deep-copying each element.

        Deep copy ensures each layer has independent parameters.
        """
        return [node.model_copy(deep=True) for _ in range(repeat) for node in nodes]

    def is_topology(self, node: NodeConfig) -> bool:
        """Check if node is a topology (has layers list attribute)."""
        if not hasattr(node, "layers"):
            return False
        layers = getattr(node, "layers")
        return isinstance(layers, list)
