"""Tests for manifest loading and typed config parsing.

The manifest schema is target-based: experiments and agent processes are both
targets.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from typing import Any, cast

from pydantic import ValidationError
from unittest.mock import patch

from caramba.config.manifest import Manifest


class ManifestTest(unittest.TestCase):
    """
    ManifestTest provides tests for the Manifest class.
    """
    def test_load_yaml_manifest(self) -> None:
        """
        test loading a YAML manifest.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.yml"
            path.write_text(
                "\n".join(
                    [
                        "version: 2",
                        "name: test",
                        'notes: "x"',
                        "defaults:",
                        "  logging:",
                        "    wandb: false",
                        "    wandb_project: \"\"",
                        "    wandb_entity: \"\"",
                        "  data:",
                        "    tokenizer: tiktoken",
                        "    val_frac: 0.1",
                        "  runtime:",
                        "    save_every: 100",
                        "targets:",
                        "  - type: experiment",
                        "    name: exp",
                        "    description: d",
                        "    backend: torch",
                        "    task: task.language_modeling",
                        "    data:",
                        "      ref: dataset.tokens",
                        "      config: { path: 'x.tokens', block_size: 4 }",
                        "    system:",
                        "      ref: system.language_model",
                        "      config:",
                        "        model:",
                        "          type: TransformerModel",
                        "          topology:",
                        "            type: StackedTopology",
                        "            layers:",
                        "              - type: LinearLayer",
                        "                d_in: 128",
                        "                d_out: 128",
                        "                bias: true",
                        "    objective: objective.next_token_ce",
                        "    trainer: trainer.train",
                        "    runs:",
                        "      - id: r",
                        "        mode: train",
                        "        exp: e",
                        "        seed: 1",
                        "        steps: 2",
                        "        expected: {}",
                        "        train:",
                        "          phase: standard",
                        "          batch_size: 1",
                        "          block_size: 4",
                        "          lr: 0.001",
                        "          device: cpu",
                        "          dtype: float32",
                    ]
                ),
                encoding="utf-8",
            )

            m = Manifest.from_path(path)
            self.assertEqual(m.version, 2)
            self.assertEqual(len(m.targets), 1)
            t0 = m.targets[0]
            assert t0.type == "experiment"
            self.assertEqual(t0.system.ref, "system.language_model")
            model_payload = t0.system.config["model"]
            # Vars already resolved at manifest load time, so payload is concrete.
            self.assertEqual(model_payload["topology"]["layers"][0]["type"], "LinearLayer")

    def test_load_yaml_manifest_with_agents(self) -> None:
        """test loading a YAML manifest with a process target."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.yml"
            path.write_text(
                "\n".join(
                    [
                        "version: 2",
                        "name: test",
                        'notes: "x"',
                        "defaults:",
                        "  logging:",
                        "    wandb: false",
                        "    wandb_project: \"\"",
                        "    wandb_entity: \"\"",
                        "  data: { tokenizer: tiktoken, val_frac: 0.1 }",
                        "  runtime: { save_every: 100 }",
                        "targets:",
                        "  - type: process",
                        "    name: platform_optimizations",
                        "    team:",
                        "      research_team_leader: research_lead",
                        "      developer: developer",
                        "    process:",
                        "      name: platform_optimizations",
                        "      type: discussion",
                        "      leader: research_team_leader",
                        "      topic: \"Optimizations for caramba\"",
                        "entrypoints:",
                        "  default: \"platform_optimizations\"",
                    ]
                ),
                encoding="utf-8",
            )

            m = Manifest.from_path(path)
            assert m.entrypoints is not None
            self.assertEqual(
                m.entrypoints["default"],
                "platform_optimizations",
            )

    def test_load_yaml_manifest_with_include(self) -> None:
        """test loading a YAML manifest that uses !include fragments."""
        with tempfile.TemporaryDirectory() as tmp:
            layer_path = Path(tmp) / "layer.yml"
            layer_path.write_text(
                "\n".join(
                    [
                        "type: LinearLayer",
                        "d_in: ${d_model}",
                        "d_out: ${d_model}",
                        "bias: true",
                    ]
                ),
                encoding="utf-8",
            )

            path = Path(tmp) / "m.yml"
            path.write_text(
                "\n".join(
                    [
                        "version: 2",
                        "name: test",
                        'notes: "x"',
                        "vars:",
                        "  d_model: 128",
                        "defaults:",
                        "  logging:",
                        "    wandb: false",
                        "    wandb_project: \"\"",
                        "    wandb_entity: \"\"",
                        "  data: { tokenizer: tiktoken, val_frac: 0.1 }",
                        "  runtime: { save_every: 100 }",
                        "targets:",
                        "  - type: experiment",
                        "    name: exp",
                        "    description: d",
                        "    backend: torch",
                        "    task: task.language_modeling",
                        "    data:",
                        "      ref: dataset.tokens",
                        "      config: { path: 'x.tokens', block_size: 4 }",
                        "    system:",
                        "      ref: system.language_model",
                        "      config:",
                        "        model:",
                        "          type: TransformerModel",
                        "          topology:",
                        "            type: StackedTopology",
                        "            layers:",
                        "              - !include layer.yml",
                        "    objective: objective.next_token_ce",
                        "    trainer: trainer.train",
                        "    runs:",
                        "      - id: r",
                        "        mode: train",
                        "        exp: e",
                        "        seed: 1",
                        "        steps: 2",
                        "        expected: {}",
                        "        train:",
                        "          phase: standard",
                        "          batch_size: 1",
                        "          block_size: 4",
                        "          lr: 0.001",
                        "          device: cpu",
                        "          dtype: float32",
                    ]
                ),
                encoding="utf-8",
            )

            m = Manifest.from_path(path)
            self.assertEqual(m.version, 2)
            self.assertEqual(len(m.targets), 1)
            t0 = m.targets[0]
            assert t0.type == "experiment"
            model_payload = t0.system.config["model"]
            layer0 = model_payload["topology"]["layers"][0]
            self.assertEqual(layer0["type"], "LinearLayer")
            self.assertEqual(layer0["d_in"], 128)
            self.assertEqual(layer0["d_out"], 128)

    def test_load_yaml_manifest_with_include_vars_override(self) -> None:
        """!include supports per-include local vars for reusable YAML blocks."""
        with tempfile.TemporaryDirectory() as tmp:
            layer_path = Path(tmp) / "layer.yml"
            layer_path.write_text(
                "\n".join(
                    [
                        "type: LinearLayer",
                        "d_in: ${d_in}",
                        "d_out: ${d_out}",
                        "bias: false",
                    ]
                ),
                encoding="utf-8",
            )

            path = Path(tmp) / "m.yml"
            path.write_text(
                "\n".join(
                    [
                        "version: 2",
                        "name: test",
                        'notes: \"x\"',
                        "vars:",
                        "  d_model: 128",
                        "defaults:",
                        "  logging:",
                        "    wandb: false",
                        "    wandb_project: \"\"",
                        "    wandb_entity: \"\"",
                        "  data: { tokenizer: tiktoken, val_frac: 0.1 }",
                        "  runtime: { save_every: 100 }",
                        "targets:",
                        "  - type: experiment",
                        "    name: exp",
                        "    description: d",
                        "    backend: torch",
                        "    task: task.language_modeling",
                        "    data:",
                        "      ref: dataset.tokens",
                        "      config: { path: 'x.tokens', block_size: 4 }",
                        "    system:",
                        "      ref: system.language_model",
                        "      config:",
                        "        model:",
                        "          type: TransformerModel",
                        "          topology:",
                        "            type: StackedTopology",
                        "            layers:",
                        "              - !include",
                        "                  path: layer.yml",
                        "                  vars:",
                        "                    d_in: ${d_model}",
                        "                    d_out: 256",
                        "    objective: objective.next_token_ce",
                        "    trainer: trainer.train",
                        "    runs:",
                        "      - id: r",
                        "        mode: train",
                        "        exp: e",
                        "        seed: 1",
                        "        steps: 2",
                        "        expected: {}",
                        "        train:",
                        "          phase: standard",
                        "          batch_size: 1",
                        "          block_size: 4",
                        "          lr: 0.001",
                        "          device: cpu",
                        "          dtype: float32",
                    ]
                ),
                encoding="utf-8",
            )

            m = Manifest.from_path(path)
            t0 = m.targets[0]
            assert t0.type == "experiment"
            layer0 = t0.system.config["model"]["topology"]["layers"][0]
            self.assertEqual(layer0["type"], "LinearLayer")
            self.assertEqual(layer0["d_in"], 128)
            self.assertEqual(layer0["d_out"], 256)

    def test_load_dba_paper_local_preset(self) -> None:
        """Preset manifests should load with OpGraph attention includes."""
        repo_root = Path(__file__).resolve().parents[1]
        path = repo_root / "config" / "presets" / "dba_paper_local.yml"
        with patch.dict(
            os.environ,
            {
                "wandb_project": "test",
                "wandb_entity": "",
                "wandb_mode": "offline",
                "eval_iters": "0",
            },
            clear=False,
        ):
            m = Manifest.from_path(path)
        self.assertEqual(m.version, 2)
        self.assertGreater(len(m.targets), 0)

    def test_load_yaml_manifest_with_paper_collect_artifacts(self) -> None:
        """Process targets can run non-agent utility workflows (paper artifact collection)."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.yml"
            path.write_text(
                "\n".join(
                    [
                        "version: 2",
                        "name: test_collect",
                        "notes: \"\"",
                        "defaults:",
                        "  logging:",
                        "    wandb: false",
                        "    wandb_project: \"\"",
                        "    wandb_entity: \"\"",
                        "  data: { tokenizer: llama, val_frac: 0.0 }",
                        "  runtime: { save_every: 1 }",
                        "targets:",
                        "  - type: process",
                        "    name: collect_paper_artifacts",
                        "    team: {}",
                        "    process:",
                        "      type: paper_collect_artifacts",
                        "      name: collect_paper_artifacts",
                        "      out_dir: artifacts/paper",
                        "      title: \"DBA Ablations\"",
                    ]
                ),
                encoding="utf-8",
            )
            m = Manifest.from_path(path)
            self.assertEqual(len(m.targets), 1)
            self.assertEqual(m.targets[0].type, "process")

    def test_load_yaml_manifest_with_compare_verify(self) -> None:
        """
        test loading a YAML manifest with a typed compare verify block.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.yml"
            path.write_text(
                "\n".join(
                    [
                        "version: 2",
                        "name: test",
                        'notes: "x"',
                        "defaults:",
                        "  logging: { wandb: false, wandb_project: \"\", wandb_entity: \"\", eval_iters: 50 }",
                        "  data: { tokenizer: tiktoken, val_frac: 0.1 }",
                        "  runtime: { save_every: 100 }",
                        "targets:",
                        "  - type: experiment",
                        "    name: exp",
                        "    backend: torch",
                        "    task: task.language_modeling",
                        "    data: { ref: dataset.tokens, config: { path: 'x.tokens', block_size: 4 } }",
                        "    system:",
                        "      ref: system.language_model",
                        "      config:",
                        "        model:",
                        "          type: TransformerModel",
                        "          topology:",
                        "            type: StackedTopology",
                        "            layers:",
                        "              - type: LinearLayer",
                        "                d_in: 128",
                        "                d_out: 128",
                        "                bias: true",
                        "    objective: objective.next_token_ce",
                        "    trainer: trainer.train",
                        "    runs:",
                        "      - id: r",
                        "        mode: train",
                        "        exp: e",
                        "        seed: 1",
                        "        steps: 2",
                        "        expected: {}",
                        "        train:",
                        "          phase: standard",
                        "          batch_size: 1",
                        "          block_size: 4",
                        "          lr: 0.001",
                        "          device: cpu",
                        "          dtype: float32",
                        "        verify:",
                        "          type: compare",
                        "          batches: 1",
                        "          logits:",
                        "            max_mean_l1: 1.0",
                        "            max_max_l1: 2.0",
                    ]
                ),
                encoding="utf-8",
            )

            m = Manifest.from_path(path)
            run = cast(Any, m.targets[0]).runs[0]
            self.assertIsNotNone(run.verify)

    def test_rejects_compare_verify_without_metrics(self) -> None:
        """
        test rejecting compare verify blocks without attention/logits.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.yml"
            path.write_text(
                "\n".join(
                    [
                        "version: 2",
                        "name: test",
                        'notes: "x"',
                        "defaults:",
                        "  logging: { wandb: false, wandb_project: \"\", wandb_entity: \"\", eval_iters: 50 }",
                        "  data: { tokenizer: tiktoken, val_frac: 0.1 }",
                        "  runtime: { save_every: 100 }",
                        "targets:",
                        "  - type: experiment",
                        "    name: exp",
                        "    backend: torch",
                        "    task: task.language_modeling",
                        "    data: { ref: dataset.tokens, config: { path: 'x.tokens', block_size: 4 } }",
                        "    system:",
                        "      ref: system.language_model",
                        "      config:",
                        "        model:",
                        "          type: TransformerModel",
                        "          topology:",
                        "            type: StackedTopology",
                        "            layers:",
                        "              - type: LinearLayer",
                        "                d_in: 128",
                        "                d_out: 128",
                        "                bias: true",
                        "    objective: objective.next_token_ce",
                        "    trainer: trainer.train",
                        "    runs:",
                        "      - id: r",
                        "        mode: train",
                        "        exp: e",
                        "        seed: 1",
                        "        steps: 2",
                        "        expected: {}",
                        "        train:",
                        "          phase: standard",
                        "          batch_size: 1",
                        "          block_size: 4",
                        "          lr: 0.001",
                        "          device: cpu",
                        "          dtype: float32",
                        "        verify:",
                        "          type: compare",
                        "          batches: 1",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ValidationError):
                _ = Manifest.from_path(path)

    def test_load_yaml_manifest_with_eval_verify(self) -> None:
        """
        test loading a YAML manifest with a typed eval verify block.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.yml"
            path.write_text(
                "\n".join(
                    [
                        "version: 2",
                        "name: test",
                        'notes: \"x\"',
                        "defaults:",
                        "  logging: { wandb: false, wandb_project: \"\", wandb_entity: \"\", eval_iters: 50 }",
                        "  data: { tokenizer: tiktoken, val_frac: 0.1 }",
                        "  runtime: { save_every: 100 }",
                        "targets:",
                        "  - type: experiment",
                        "    name: exp",
                        "    backend: torch",
                        "    task: task.language_modeling",
                        "    data: { ref: dataset.tokens, config: { path: 'x.tokens', block_size: 4 } }",
                        "    system:",
                        "      ref: system.language_model",
                        "      config:",
                        "        model:",
                        "          type: TransformerModel",
                        "          topology:",
                        "            type: StackedTopology",
                        "            layers:",
                        "              - type: LinearLayer",
                        "                d_in: 128",
                        "                d_out: 128",
                        "                bias: true",
                        "    objective: objective.next_token_ce",
                        "    trainer: trainer.train",
                        "    runs:",
                        "      - id: r",
                        "        mode: train",
                        "        exp: e",
                        "        seed: 1",
                        "        steps: 2",
                        "        expected: {}",
                        "        train:",
                        "          phase: standard",
                        "          batch_size: 1",
                        "          block_size: 4",
                        "          lr: 0.001",
                        "          device: cpu",
                        "          dtype: float32",
                        "        verify:",
                        "          type: eval",
                        "          tokenizer:",
                        "            type: tiktoken",
                        "            encoding: gpt2",
                        "          max_new_tokens: 4",
                        "          cases:",
                        "            - id: strawberry_r",
                        "              prompt: \"How many times do we find the letter r in the word strawberry?\"",
                        "              kind: choice_logprob",
                        "              choices: [\"1\", \"2\", \"3\", \"4\"]",
                        "              answer: \"3\"",
                        "          thresholds:",
                        "            min_student_accuracy: 0.0",
                        "            max_accuracy_drop: 1.0",
                    ]
                ),
                encoding="utf-8",
            )

            m = Manifest.from_path(path)
            run = cast(Any, m.targets[0]).runs[0]
            self.assertIsNotNone(run.verify)

    def test_load_yaml_manifest_with_kvcache_verify(self) -> None:
        """
        test loading a YAML manifest with a typed kvcache verify block.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.yml"
            path.write_text(
                "\n".join(
                    [
                        "version: 2",
                        "name: test",
                        'notes: "x"',
                        "defaults:",
                        "  logging: { wandb: false, wandb_project: \"\", wandb_entity: \"\", eval_iters: 50 }",
                        "  data: { tokenizer: tiktoken, val_frac: 0.1 }",
                        "  runtime: { save_every: 100 }",
                        "targets:",
                        "  - type: experiment",
                        "    name: exp",
                        "    backend: torch",
                        "    task: task.language_modeling",
                        "    data: { ref: dataset.tokens, config: { path: 'x.tokens', block_size: 4 } }",
                        "    system:",
                        "      ref: system.language_model",
                        "      config:",
                        "        model:",
                        "          type: TransformerModel",
                        "          topology:",
                        "            type: StackedTopology",
                        "            layers:",
                        "              - type: LinearLayer",
                        "                d_in: 128",
                        "                d_out: 128",
                        "                bias: true",
                        "    objective: objective.next_token_ce",
                        "    trainer: trainer.train",
                        "    runs:",
                        "      - id: r",
                        "        mode: train",
                        "        exp: e",
                        "        seed: 1",
                        "        steps: 2",
                        "        expected: {}",
                        "        train:",
                        "          phase: standard",
                        "          batch_size: 1",
                        "          block_size: 4",
                        "          lr: 0.001",
                        "          device: cpu",
                        "          dtype: float32",
                        "        verify:",
                        "          type: kvcache",
                        "          n_layers: 1",
                        "          batch_size: 1",
                        "          max_seq_len: 8",
                        "          teacher:",
                        "            k: { kind: fp16, qblock: 32, residual_len: 0 }",
                        "            v: { kind: fp16, qblock: 32, residual_len: 0 }",
                        "          student:",
                        "            k_sem: { kind: q4_0, qblock: 32, residual_len: 0 }",
                        "            k_geo: { kind: q8_0, qblock: 32, residual_len: 0 }",
                        "            v: { kind: q4_0, qblock: 32, residual_len: 0 }",
                    ]
                ),
                encoding="utf-8",
            )

            m = Manifest.from_path(path)
            run = cast(Any, m.targets[0]).runs[0]
            self.assertIsNotNone(run.verify)

    def test_load_json_manifest(self) -> None:
        """
        test loading a JSON manifest.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.json"
            payload = {
                "version": 2,
                "name": "test",
                "notes": "x",
                "defaults": {
                    "data": {"tokenizer": "tiktoken", "val_frac": 0.1},
                    "logging": {"wandb": False, "wandb_project": "", "wandb_entity": "", "eval_iters": 50},
                    "runtime": {"save_every": 100},
                },
                "targets": [
                    {
                        "type": "experiment",
                        "name": "exp",
                        "backend": "torch",
                        "task": "task.language_modeling",
                        "data": {"ref": "dataset.tokens", "config": {"path": "x.tokens", "block_size": 4}},
                        "system": {
                            "ref": "system.language_model",
                            "config": {
                                "model": {
                                    "type": "TransformerModel",
                                    "topology": {
                                        "type": "StackedTopology",
                                        "layers": [
                                            {"type": "LinearLayer", "d_in": 128, "d_out": 128, "bias": True}
                                        ],
                                    },
                                }
                            },
                        },
                        "objective": "objective.next_token_ce",
                        "trainer": "trainer.train",
                        "runs": [{"id": "r", "mode": "train", "exp": "e", "seed": 1, "steps": 2, "expected": {}}],
                    }
                ],
            }
            path.write_text(json.dumps(payload), encoding="utf-8")

            m = Manifest.from_path(path)
            self.assertEqual(m.targets[0].type, "experiment")

    def test_resolves_vars(self) -> None:
        """
        test resolving vars in a manifest payload.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.yml"
            path.write_text(
                "\n".join(
                    [
                        "version: 2",
                        "name: test",
                        'notes: "x"',
                        "vars:",
                        "  d_in: 16",
                        "  d_out: 32",
                        "defaults:",
                        "  logging: { wandb: false, wandb_project: \"\", wandb_entity: \"\", eval_iters: 50 }",
                        "  data: { tokenizer: tiktoken, val_frac: 0.1 }",
                        "  runtime: { save_every: 100 }",
                        "targets:",
                        "  - type: experiment",
                        "    name: exp",
                        "    backend: torch",
                        "    task: task.language_modeling",
                        "    data: { ref: dataset.tokens, config: { path: 'x.tokens', block_size: 4 } }",
                        "    system:",
                        "      ref: system.language_model",
                        "      config:",
                        "        model:",
                        "          type: TransformerModel",
                        "          topology:",
                        "            type: StackedTopology",
                        "            layers:",
                        "              - type: LinearLayer",
                        "                d_in: \"${d_in}\"",
                        "                d_out: \"${d_out}\"",
                        "                bias: true",
                        "    objective: objective.next_token_ce",
                        "    trainer: trainer.train",
                        "    runs: []",
                    ]
                ),
                encoding="utf-8",
            )

            m = Manifest.from_path(path)
            target = cast(Any, m.targets[0])
            model_payload = target.system.config["model"]
            layer = model_payload["topology"]["layers"][0]
            self.assertEqual(layer["type"], "LinearLayer")
            self.assertEqual(layer["d_in"], 16)
            self.assertEqual(layer["d_out"], 32)

    def test_rejects_invalid_layer_shape(self) -> None:
        """
        test rejecting an invalid layer shape.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yml"
            # Old preset style (layer params under `config:`) should fail validation.
            path.write_text(
                "\n".join(
                    [
                        "version: 2",
                        "notes: x",
                        "defaults:",
                        "  logging: { wandb: false, wandb_project: \"\", wandb_entity: \"\", eval_iters: 50 }",
                        "  data: { tokenizer: tiktoken, val_frac: 0.1 }",
                        "  runtime: { save_every: 100 }",
                        "targets:",
                        "  - type: experiment",
                        "    name: exp",
                        "    backend: torch",
                        "    task: task.language_modeling",
                        "    data: { ref: dataset.tokens, config: { path: 'x.tokens', block_size: 4 } }",
                        "    system:",
                        "      ref: system.language_model",
                        "      config:",
                        "        model:",
                        "          type: TransformerModel",
                        "          topology:",
                        "            type: StackedTopology",
                        "            layers:",
                        "              - type: LinearLayer",
                        "                config: { d_in: 128, d_out: 128 }",
                        "    objective: objective.next_token_ce",
                        "    trainer: trainer.train",
                        "    runs: []",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ValidationError):
                _ = Manifest.from_path(path)
