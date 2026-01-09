"""Generation for diffusion codegen

Loads a trained checkpoint and runs DDIM/DDPM sampling to produce decoded text
samples written as artifacts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from caramba.console import logger
from caramba.data.tokenizers.hf_json import HfJsonTokenizer
from caramba.data.tokenizers.training import TrainingTokenizer
from caramba.diffusion.samplers import DdimSampler, DdpmSampler, GuidanceConfig
from caramba.diffusion.schedule import NoiseSchedule
from caramba.trainer.diffusion_codegen.checkpoints import CheckpointManager
from caramba.trainer.diffusion_codegen.prompt import PromptEncoder


@dataclass(frozen=True, slots=True)
class GenerationSettings:
    """Generation settings for diffusion codegen."""

    timesteps: int
    schedule: str
    beta_start: float
    beta_end: float
    sampler: dict[str, Any] | None
    generate: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class GenerationRunner:
    """Run generation from a checkpoint."""

    settings: GenerationSettings

    def run(
        self,
        *,
        target: Any,
        engine: Any,
        checkpoint_dir: Path,
        tokenizer_file: str,
    ) -> list[Path]:
        cfg = self.settings.generate or {}
        run_id = str(cfg.get("run_id", "train"))
        device = torch.device(str(cfg.get("device", "cpu")))

        manager = CheckpointManager(checkpoint_dir=str(checkpoint_dir))
        checkpoint_path = cfg.get("checkpoint_path", None)
        ckpt_path = Path(str(checkpoint_path)) if checkpoint_path else manager.findLatest(run_id=run_id)
        ckpt = manager.load(checkpoint_path=str(ckpt_path), map_location=device)

        system = engine.registry.build(target.system, backend=str(target.backend))
        system = self.moveSystem(system=system, device=device, dtype=torch.float32)
        self.loadModelState(system=system, state=ckpt.get("model_state", {}))
        model = self.unwrapModel(system=system)

        tokenizer = HfJsonTokenizer.from_file(tokenizer_file=str(Path(tokenizer_file)))
        pad_id = tokenizer.token_to_id("<pad>")
        if pad_id is None:
            raise ValueError("Tokenizer must define <pad>.")

        seq_len = int(cfg.get("sequence_length", int(getattr(model, "max_len", 128))))
        prompt_emb, prompt_pad = self.promptEmbeddings(cfg=cfg, tokenizer=tokenizer, model=model, pad_id=int(pad_id), seq_len=seq_len, device=device)

        schedule = NoiseSchedule(kind=str(self.settings.schedule), beta_start=float(self.settings.beta_start), beta_end=float(self.settings.beta_end))
        alpha_bar = schedule.alphasCumprod(timesteps=int(self.settings.timesteps), device=device)
        tokens = self.sampleTokens(cfg=cfg, model=model, alpha_bar=alpha_bar, device=device, seq_len=seq_len, prompt_emb=prompt_emb, prompt_pad=prompt_pad)

        texts = [tokenizer.decode(row.tolist(), skip_special_tokens=True) for row in tokens]
        out = checkpoint_dir / f"generated_{int(time.time())}.txt"
        out.write_text("\\n\\n".join(texts), encoding="utf-8")
        logger.path(str(out), label="generated")
        return [out]

    def sampleTokens(
        self,
        *,
        cfg: dict[str, Any],
        model: nn.Module,
        alpha_bar: Tensor,
        device: torch.device,
        seq_len: int,
        prompt_emb: Tensor | None,
        prompt_pad: Tensor | None,
    ) -> Tensor:
        sampler_cfg = self.settings.sampler or {}
        kind = str(cfg.get("sampler", sampler_cfg.get("kind", "ddim"))).lower().strip()
        num_samples = int(cfg.get("num_samples", 5))
        guidance = float(cfg.get("guidance_scale", sampler_cfg.get("guidance_scale", 7.5)))

        if kind == "ddim":
            sampler = DdimSampler(
                model=model,  # type: ignore[arg-type]
                alpha_bar=alpha_bar,
                timesteps=int(self.settings.timesteps),
                device=device,
                hidden_size=int(getattr(model, "hidden_size", 512)),
                steps=int(cfg.get("ddim_steps", sampler_cfg.get("ddim_steps", 50))),
                eta=float(cfg.get("ddim_eta", sampler_cfg.get("ddim_eta", 0.0))),
            )
            return sampler.sample(
                batch_size=int(num_samples),
                seq_len=int(seq_len),
                target_pad_mask=None,
                prompt_emb=prompt_emb,
                prompt_pad_mask=prompt_pad,
                cfg=GuidanceConfig(guidance_scale=float(guidance)),
                embedding_weight=model.embedding.weight,  # type: ignore[attr-defined]
            )

        if kind == "ddpm":
            sampler = DdpmSampler(
                model=model,  # type: ignore[arg-type]
                alpha_bar=alpha_bar,
                timesteps=int(self.settings.timesteps),
                device=device,
                hidden_size=int(getattr(model, "hidden_size", 512)),
            )
            return sampler.sample(
                batch_size=int(num_samples),
                seq_len=int(seq_len),
                target_pad_mask=None,
                prompt_emb=prompt_emb,
                prompt_pad_mask=prompt_pad,
                cfg=GuidanceConfig(guidance_scale=float(guidance)),
                embedding_weight=model.embedding.weight,  # type: ignore[attr-defined]
            )

        raise ValueError(f"Unknown sampler kind: {kind!r}")

    def promptEmbeddings(
        self,
        *,
        cfg: dict[str, Any],
        tokenizer: TrainingTokenizer,
        model: nn.Module,
        pad_id: int,
        seq_len: int,
        device: torch.device,
    ) -> tuple[Tensor | None, Tensor | None]:
        prompt = cfg.get("prompt", None)
        if not isinstance(prompt, str) or not prompt:
            return None, None
        encoder = PromptEncoder(tokenizer=tokenizer, embedding=model.embedding, pad_id=int(pad_id), seq_len=int(seq_len))  # type: ignore[arg-type]
        return encoder.encode(prompt=str(prompt), device=device)

    def moveSystem(self, *, system: object, device: torch.device, dtype: torch.dtype) -> object:
        if not hasattr(system, "to"):
            raise TypeError("System component does not expose to(device=..., dtype=...)")
        return system.to(device=device, dtype=dtype)  # type: ignore[no-any-return, attr-defined]

    def unwrapModel(self, *, system: object) -> nn.Module:
        module = getattr(system, "module", None)
        if isinstance(module, nn.Module):
            return module
        if isinstance(system, nn.Module):
            return system
        raise TypeError(f"Expected system to expose .module nn.Module, got {type(system).__name__}")

    def loadModelState(self, *, system: object, state: dict[str, Any]) -> None:
        if not hasattr(system, "load_state_dict"):
            raise TypeError("System component does not expose load_state_dict().")
        system.load_state_dict(state)  # type: ignore[attr-defined]

