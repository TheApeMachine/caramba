"""Tests for template-manifest loading.

Template manifests add `variables` and `instrumentation` sections which compile
into the stable config-manifest schema.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from caramba.config.manifest import Manifest


class TemplateManifestTest(unittest.TestCase):
    """TemplateManifestTest validates template compilation and interpolation."""

    def test_load_template_manifest(self) -> None:
        """Load a template manifest and validate interpolation + defaults."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.yml"
            path.write_text(
                "\n".join(
                    [
                        "version: 2",
                        "name: test_template",
                        "instrumentation:",
                        "  logger: { type: rich }",
                        "  metrics:",
                        "    hdf5: { enabled: false }",
                        "    tensorboard: { enabled: false }",
                        "    wandb:",
                        "      enabled: true",
                        "      project: ${ENV:wandb_project}",
                        "      entity: ''",
                        "      mode: offline",
                        "      eval_iters: ${ENV:eval_iters}",
                        "    liveplot: { enabled: false }",
                        "    jsonl: { enabled: false }",
                        "variables:",
                        "  datasets:",
                        "    - repo: example/dataset",
                        "      tokens: 1b",
                        "      tokenizer: tiktoken:gpt2",
                        "      value_fraction: 0.0",
                        "      block_size: 4",
                        "  model:",
                        "    d_model: 128",
                        "    n_layers: 1",
                        "    n_heads: 2",
                        "    n_kv_heads_gqa: 1",
                        "    d_ff: 256",
                        "    vocab_size: 100",
                        "    rope_base: 10000.0",
                        "  sem_dim: 8",
                        "  geo_dim: 16",
                        "  attn_dim: 24",
                        "  trainer:",
                        "    type: stepwise",
                        "    steps: 2",
                        "    steps_extended: 2",
                        "    device: cpu",
                        "    dtype: float32",
                        "    batch_size: 1",
                        "    grad_accum: 1",
                        "    lr: 0.001",
                        "    lr_decoupled: 0.0002",
                        "    lr_2e4: 0.0002",
                        "    lr_4e4: 0.0004",
                        "    optimizer:",
                        "      type: adamw",
                        "      betas: [0.9, 0.999]",
                        "      eps: 1e-8",
                        "      weight_decay: 0.0",
                        "      fused: false",
                        "    scheduler:",
                        "      type: cosine",
                        "      warmup_steps: 0",
                        "      min_lr_ratio: 0.0",
                        "      auto_resume: false",
                        "      total_steps: 2",
                        "    save_every: 1",
                        "targets:",
                        "  - type: experiment",
                        "    name: exp",
                        "    description: d",
                        "    backend: torch",
                        "    task: task.language_modeling",
                        "    data:",
                        "      ref: dataset.tokens",
                        "      config:",
                        "        dataset: ${dataset}",
                        "        tokens: ${tokens}",
                        "        tokenizer: ${tokenizer}",
                        "        block_size: ${block_size}",
                        "    system:",
                        "      ref: system.language_model",
                        "      config:",
                        "        model:",
                        "          type: TransformerModel",
                        "          topology:",
                        "            type: StackedTopology",
                        "            layers:",
                        "              - type: LinearLayer",
                        "                d_in: ${d_model}",
                        "                d_out: ${d_model}",
                        "                bias: true",
                        "    objective: objective.next_token_ce",
                        "    trainer: trainer.train",
                        "    runs:",
                        "      - id: r",
                        "        mode: train",
                        "        exp: e",
                        "        seed: 1",
                        "        steps: ${steps}",
                        "        expected: {}",
                        "        train:",
                        "          phase: standard",
                        "          batch_size: ${batch_size}",
                        "          block_size: ${block_size}",
                        "          lr: ${lr}",
                        "          device: ${device}",
                        "          dtype: ${dtype}",
                    ]
                ),
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {"wandb_project": "proj", "eval_iters": "0"},
                clear=False,
            ):
                m = Manifest.from_path(path)

            self.assertEqual(m.version, 2)
            self.assertEqual(m.defaults.data.tokenizer, "tiktoken:gpt2")
            self.assertEqual(float(m.defaults.data.val_frac), 0.0)
            self.assertEqual(m.defaults.runtime.save_every, 1)
            self.assertEqual(m.defaults.logging.wandb_project, "proj")
            self.assertEqual(m.defaults.logging.eval_iters, 0)

            self.assertEqual(len(m.targets), 1)
            t0 = m.targets[0]
            assert t0.type == "experiment"
            run0 = t0.runs[0]
            self.assertEqual(run0.steps, 2)
            assert run0.train is not None
            self.assertEqual(run0.train.device, "cpu")
            self.assertEqual(run0.train.batch_size, 1)
