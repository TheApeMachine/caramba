"""
lower_test provides tests for the lowering pass.
"""
from __future__ import annotations

import unittest
import tempfile
import textwrap
import torch
from pathlib import Path
from typing import cast

from caramba.compiler import Compiler
from caramba.config.layer import LinearLayerConfig, LayerNormLayerConfig
from caramba.config.manifest import Manifest
from caramba.config.topology import (
    NestedTopologyConfig,
    SequentialTopologyConfig,
    StackedTopologyConfig,
)


class LowerTest(unittest.TestCase):
    """
    LowerTest provides tests for lowering.
    """

    def setUp(self) -> None:
        """Create a fresh Compiler instance for each test."""
        self.compiler = Compiler()

    def test_expands_repeat_on_stacked(self) -> None:
        """
        test expanding repeat on stacked topology.
        """
        linear = LinearLayerConfig(d_in=4, d_out=4, bias=True)
        topo = StackedTopologyConfig(layers=[linear], repeat=3)
        lowered = self.compiler.lowerer.lower_topology(topo)

        # `TopologyConfig` now includes GraphTopology, so narrow the type here.
        self.assertIsInstance(lowered, StackedTopologyConfig)
        lowered_stacked = cast(StackedTopologyConfig, lowered)
        self.assertEqual(lowered_stacked.repeat, 1)
        self.assertEqual(len(lowered_stacked.layers), 3)

    def test_expands_repeat_through_nested(self) -> None:
        """
        test expanding repeat through nested topology.
        """
        linear = LinearLayerConfig(d_in=4, d_out=4, bias=True)
        inner = StackedTopologyConfig(layers=[linear], repeat=2)
        outer = NestedTopologyConfig(layers=[inner], repeat=3)
        lowered = self.compiler.lowerer.lower_topology(outer)

        self.assertIsInstance(lowered, NestedTopologyConfig)
        lowered_nested = cast(NestedTopologyConfig, lowered)

        self.assertEqual(lowered_nested.repeat, 1)
        self.assertEqual(len(lowered_nested.layers), 3)
        for child in lowered_nested.layers:
            self.assertIsInstance(child, StackedTopologyConfig)
            child_stacked = cast(StackedTopologyConfig, child)
            self.assertEqual(child_stacked.repeat, 1)
            self.assertEqual(len(child_stacked.layers), 2)

    def test_builds_and_runs_after_lowering(self) -> None:
        """
        test model build/forward after lowering.
        """
        linear = LinearLayerConfig(d_in=4, d_out=4, bias=False)
        topo = StackedTopologyConfig(layers=[linear], repeat=2)
        lowered = self.compiler.lowerer.lower_topology(topo)

        model = lowered.build()
        x = torch.randn(2, 3, 4)
        _ = model.forward(x)

    def test_allows_topology_nodes_inside_layers(self) -> None:
        """
        test allowing topology nodes inside layer lists.
        """
        linear = LinearLayerConfig(d_in=4, d_out=4, bias=False)
        inner = SequentialTopologyConfig(layers=[linear], repeat=1)
        outer = StackedTopologyConfig(layers=[inner], repeat=1)
        lowered = self.compiler.lowerer.lower_topology(outer)

        model = lowered.build()
        x = torch.randn(2, 3, 4)
        _ = model.forward(x)

    def test_rejects_dim_mismatch(self) -> None:
        """
        test rejecting a simple d_model mismatch across layers.
        """
        topo = StackedTopologyConfig(
            layers=[
                LinearLayerConfig(d_in=4, d_out=4, bias=False),
                LayerNormLayerConfig(d_model=8),
            ]
        )
        with self.assertRaises(ValueError):
            self.compiler.validator.validate_topology(topo)

    def test_expands_multi_seed_runs(self) -> None:
        """Lowering expands seed: [..] into multiple runs with stable ids."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.yml"
            path.write_text(
                textwrap.dedent(
                    """\
                    version: 2
                    name: test
                    defaults:
                      logging: { wandb: false, wandb_project: '', wandb_entity: '' }
                      data: { tokenizer: tiktoken, val_frac: 0.1 }
                      runtime: { save_every: 100 }
                    targets:
                      - type: experiment
                        name: exp
                        backend: torch
                        task: task.language_modeling
                        data:
                          ref: dataset.tokens
                          config: { path: 'x.tokens', block_size: 4 }
                        system:
                          ref: system.language_model
                          config:
                            model:
                              type: TransformerModel
                              topology:
                                type: StackedTopology
                                layers:
                                  - type: LinearLayer
                                    d_in: 128
                                    d_out: 128
                                    bias: true
                        objective: objective.next_token_ce
                        trainer: trainer.standard
                        runs:
                          - id: r
                            mode: train
                            exp: e
                            seed: [1, 2, 7]
                            steps: 2
                            expected: {}
                            train:
                              phase: standard
                              batch_size: 1
                              block_size: 4
                              lr: 0.001
                              device: cpu
                              dtype: float32
                    """
                ),
                encoding="utf-8",
            )
            m = Manifest.from_path(path)
            lowered = self.compiler.lowerer.lower_manifest(m)
            self.assertEqual(len(lowered.targets), 1)
            t0 = lowered.targets[0]
            assert t0.type == "experiment"
            self.assertEqual(len(t0.runs), 3)
            ids = [r.id for r in t0.runs]
            seeds = [r.seed for r in t0.runs]
            self.assertEqual(ids, ["r__s1", "r__s2", "r__s7"])
            self.assertEqual(seeds, [1, 2, 7])
            exps = [r.exp for r in t0.runs]
            self.assertEqual(exps, ["e__s1", "e__s2", "e__s7"])


if __name__ == "__main__":
    unittest.main()
