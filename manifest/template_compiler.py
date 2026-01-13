"""Template manifest compiler

Template manifests extend the core config manifest schema with two sections:

- `variables`: a structured set of values used for `${var}` interpolation
- `instrumentation`: a structured config that compiles into `defaults.logging`

The compiler resolves placeholders and produces a payload compatible with
`caramba.config.manifest.Manifest`.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from caramba.manifest.intrumentation import Instrumentation
from caramba.manifest.template_resolver import TemplateResolver
from caramba.manifest.variables import Variables


class TemplateManifestCompiler:
    """Compile a template manifest into a config-manifest payload."""

    def compile(self, payload: dict[str, object], *, env: Mapping[str, str] | None = None) -> dict[str, object]:
        """Compile a template manifest payload.

        Args:
            payload: Raw YAML payload (post-`!include` expansion).
            env: Environment mapping for `${ENV:NAME}` placeholders.

        Returns:
            A dict compatible with `caramba.config.manifest.Manifest`.
        """
        if not isinstance(payload, dict):
            raise TypeError(f"Template manifest payload must be a dict, got {type(payload)!r}")

        variables_raw = payload.get("variables", None)
        if not isinstance(variables_raw, dict):
            raise ValueError("Template manifest requires a top-level 'variables' mapping.")
        variables = Variables.model_validate(variables_raw)

        var_map = self.build_variable_map(variables)
        resolver = TemplateResolver(var_map, env=env or os.environ)
        resolved = resolver.resolve(payload)
        if not isinstance(resolved, dict):
            raise TypeError("Resolved template manifest must be a dict.")

        instrumentation = self.load_instrumentation(resolved)
        variables = Variables.model_validate(resolved.get("variables", {}))

        out = dict(resolved)
        out.pop("variables", None)
        out.pop("instrumentation", None)
        out.pop("vars", None)

        out["defaults"] = self.build_defaults(variables=variables, instrumentation=instrumentation)
        return out

    def load_instrumentation(self, payload: Mapping[str, object]) -> Instrumentation | None:
        """Load instrumentation from a resolved payload, if present."""
        raw = payload.get("instrumentation", None)
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise TypeError("instrumentation must be a mapping when provided.")
        return Instrumentation.model_validate(raw)

    def build_defaults(
        self, *, variables: Variables, instrumentation: Instrumentation | None
    ) -> dict[str, object]:
        """Compile template sections into config-manifest defaults."""
        if not variables.datasets:
            raise ValueError("variables.datasets must contain at least one dataset.")
        ds0 = variables.datasets[0]

        instrument = "rich"
        wandb_enabled = True
        wandb_project = ""
        wandb_entity = ""
        wandb_mode = "online"
        eval_iters: object = 0

        if instrumentation is not None:
            instrument = str(instrumentation.logger.type.value)
            wandb_enabled = bool(instrumentation.metrics.wandb.enabled)
            wandb_project = str(instrumentation.metrics.wandb.project)
            wandb_entity = str(instrumentation.metrics.wandb.entity)
            wandb_mode = str(instrumentation.metrics.wandb.mode)
            eval_iters = instrumentation.metrics.wandb.eval_iters

        return {
            "data": {
                "tokenizer": str(ds0.tokenizer),
                "val_frac": float(ds0.value_fraction),
            },
            "logging": {
                "instrument": str(instrument),
                "wandb": bool(wandb_enabled),
                "wandb_project": str(wandb_project),
                "wandb_entity": str(wandb_entity),
                "wandb_mode": str(wandb_mode),
                "eval_iters": eval_iters,
            },
            "runtime": {
                "save_every": int(variables.trainer.save_every),
            },
        }

    def build_variable_map(self, variables: Variables) -> dict[str, object]:
        """Flatten structured variables into a `${name}`-addressable mapping."""
        if not variables.datasets:
            raise ValueError("variables.datasets must contain at least one dataset.")
        ds0 = variables.datasets[0]
        trainer = variables.trainer
        optimizer = trainer.optimizer
        scheduler = trainer.scheduler

        out: dict[str, object] = {
            # Dataset convenience variables
            "dataset": str(ds0.repo),
            "tokens": ds0.tokens,
            "tokenizer": str(ds0.tokenizer),
            "block_size": int(ds0.block_size),
            # Model variables
            "d_model": int(variables.model.d_model),
            "n_layers": int(variables.model.n_layers),
            "n_heads": int(variables.model.n_heads),
            "n_kv_heads_gqa": int(variables.model.n_kv_heads_gqa),
            "d_ff": int(variables.model.d_ff),
            "vocab_size": int(variables.model.vocab_size),
            "rope_base": float(variables.model.rope_base),
            # DBA dimensions
            "sem_dim": int(variables.sem_dim),
            "geo_dim": int(variables.geo_dim),
            "attn_dim": int(variables.attn_dim),
            # Training variables
            "steps": int(trainer.steps),
            "device": str(trainer.device.type.value),
            "dtype": str(trainer.dtype.value),
            "batch_size": int(trainer.batch_size),
            "grad_accum": int(trainer.grad_accum),
            "lr": float(trainer.lr),
            # Derived common placeholders used by train configs
            "optimizer": str(optimizer.type.value),
            "weight_decay": float(optimizer.weight_decay),
            "fused_optimizer": bool(optimizer.fused),
            "scheduler": str(scheduler.type.value),
            "warmup_steps": int(scheduler.warmup_steps),
            "min_lr_ratio": float(scheduler.min_lr_ratio),
        }

        steps_extended = trainer.steps_extended if trainer.steps_extended is not None else scheduler.total_steps
        out["steps_extended"] = int(steps_extended)

        if trainer.lr_decoupled is not None:
            out["lr_decoupled"] = float(trainer.lr_decoupled)
        if trainer.lr_2e4 is not None:
            out["lr_2e4"] = float(trainer.lr_2e4)
        if trainer.lr_4e4 is not None:
            out["lr_4e4"] = float(trainer.lr_4e4)

        return out

