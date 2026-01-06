from __future__ import annotations

from pathlib import Path

from caramba.config.component import ComponentSpec
from caramba.config.defaults import Defaults
from caramba.config.manifest import Manifest
from caramba.config.mode import Mode
from caramba.config.run import Run
from caramba.config.target import ExperimentTargetConfig
from caramba.config.train import TrainConfig, TrainPhase
from caramba.runtime.engine.torch_engine import TorchEngine


class TestDiffusionCodegenTrainerSmoke:
    """End-to-end smoke test for diffusion codegen trainer."""

    def test_train_then_generate_cpu(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "src"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "a.py").write_text("x = 1\n" * 200, encoding="utf-8")

        tokenizer_file = tmp_path / "tok.json"
        ckpt_dir = tmp_path / "runs"

        defaults = Defaults()
        defaults.runtime.save_every = 10  # type: ignore[attr-defined]

        train = TrainConfig(phase=TrainPhase.STANDARD, batch_size=2, block_size=1, lr=1e-3, device="cpu", dtype="float32")
        run = Run(id="train", mode=Mode.TRAIN, exp="e", seed=0, steps=2, expected={}, train=train)

        train_target = ExperimentTargetConfig(
            name="codegen_train",
            backend="torch",
            task=ComponentSpec(ref="task.diffusion_codegen"),
            data=ComponentSpec(ref="dataset.codegen_chunks", config={"data_dir": str(data_dir), "seq_len": 32}),
            system=ComponentSpec(
                ref="system.diffusion_codegen",
                config={
                    "vocab_size": 128,
                    "hidden_size": 32,
                    "num_layers": 1,
                    "num_heads": 4,
                    "dim_feedforward": 64,
                    "max_len": 32,
                },
            ),
            objective=ComponentSpec(ref="objective.mse"),
            trainer=ComponentSpec(
                ref="trainer.diffusion_codegen",
                config={
                    "action": "train",
                    "checkpoint_dir": str(ckpt_dir),
                    "timesteps": 10,
                    "schedule": "linear",
                    "mse_lambda": 0.5,
                    "unconditional_prob": 0.1,
                    "self_condition_prob": 0.5,
                    "grad_clip": 1.0,
                    "use_ema": False,
                    "tokenizer": {
                        "tokenizer_file": str(tokenizer_file),
                        "data_dir": str(data_dir),
                        "vocab_size": 128,
                        "special_tokens": ["<unk>", "<pad>", "<s>", "</s>"],
                        "file_extensions": [".py"],
                        "train_if_missing": True,
                    },
                    "sampler": {"every": 0},
                },
            ),
            runs=[run],
        )

        gen_target = ExperimentTargetConfig(
            name="codegen_generate",
            backend="torch",
            task=ComponentSpec(ref="task.diffusion_codegen"),
            data=ComponentSpec(ref="dataset.codegen_chunks", config={"data_dir": str(data_dir), "seq_len": 32}),
            system=train_target.system,
            objective=ComponentSpec(ref="objective.mse"),
            trainer=ComponentSpec(
                ref="trainer.diffusion_codegen",
                config={
                    "action": "generate",
                    "checkpoint_dir": str(ckpt_dir),
                    "timesteps": 10,
                    "schedule": "linear",
                    "tokenizer": {"tokenizer_file": str(tokenizer_file), "train_if_missing": False},
                    "sampler": {"kind": "ddim", "ddim_steps": 3, "ddim_eta": 0.0, "guidance_scale": 1.0},
                    "generate": {"run_id": "train", "device": "cpu", "num_samples": 2, "sequence_length": 32, "prompt": ""},
                },
            ),
            runs=[],
        )

        engine = TorchEngine()
        manifest = Manifest(version=2, defaults=defaults, targets=[train_target, gen_target])
        engine.run_experiment(manifest, train_target)
        out = engine.run_experiment(manifest, gen_target)
        assert isinstance(out, dict)

