"""Unsloth-backed finetuning trainer.

Uses Unsloth's FastLanguageModel for faster HF-based finetuning.
This trainer expects a Hugging Face model id/path (or local HF checkpoint).
"""
from __future__ import annotations

import sys
from pathlib import Path

# These training scripts are often invoked as plain files under e.g.
# `accelerate launch trainer/finetune_unsloth.py`. In that mode, Python puts
# `trainer/` on sys.path but not the repo root, so imports like `data.*` fail.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from typing import Any, cast
from datetime import datetime
import inspect
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

if torch.cuda.is_available():
    from trl.trainer.sft_trainer import SFTTrainer
    from unsloth import FastLanguageModel
    from unsloth.tokenizer_utils import SFTConfig  # type: ignore[import-not-found]
else:
    from unsloth_mlx import FastLanguageModel, SFTTrainer, SFTConfig

from data.datasets.finetune_text import build_finetune_text_datasets
from console import logger

DEFAULT_SEQ_LENGTH = 2048


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class FinetuneUnslothTrainer:
    """Minimal HF finetuning route using Unsloth kernels."""

    def __init__(self, *, checkpoint_dir: str | None = None, **_: Any) -> None:
        # checkpoint_dir is accessed from target.trainer.config in run()
        # Accept extra kwargs for registry compatibility (config-driven construction).
        pass

    def run(
        self,
        *,
        manifest: Any,
        target: Any,
        engine: Any,
        dry_run: bool = False,
    ) -> dict[str, Any] | None:
        device = get_device()

        if dry_run:
            return None

        trainer_config = getattr(target.trainer, "config", {}) or {}

        seq_lengths = [
            int(run.train.block_size)
            for run in target.runs
            if run.train is not None and getattr(run.train, "block_size", None)
        ]
        max_seq_length = seq_lengths[0] if seq_lengths else DEFAULT_SEQ_LENGTH
        if seq_lengths and any(s != max_seq_length for s in seq_lengths):
            logger.warning(
                "Multiple block_size values found across runs; "
                f"using {max_seq_length} for tokenization"
            )

        # Resolve HF model name/path (prefer trainer config, fall back to run checkpoint).
        hf_model = (
            trainer_config.get("hf_model")
            or trainer_config.get("model_name")
            or trainer_config.get("model")
        )
        if not hf_model:
            # Fall back to run.train.load_checkpoint if present.
            for run in target.runs:
                if run.train is not None:
                    hf_model = getattr(run.train, "load_checkpoint", None)
                    if hf_model:
                        break
        if not hf_model:
            raise ValueError(
                "Unsloth finetuning requires a Hugging Face model id/path. "
                "Set trainer.config.hf_model (or run.train.load_checkpoint)."
            )

        trust_remote_code = bool(trainer_config.get("trust_remote_code", False))
        hf_token = trainer_config.get("hf_token")
        full_finetuning = bool(trainer_config.get("full_finetuning", False))

        load_in_4bit = bool(trainer_config.get("load_in_4bit", False))
        load_in_8bit = bool(trainer_config.get("load_in_8bit", False))
        load_in_16bit = bool(trainer_config.get("load_in_16bit", False))
        allow_non_cuda_quantization = bool(trainer_config.get("allow_non_cuda_quantization", False))
        if device.type != "cuda" and (load_in_4bit or load_in_8bit) and not allow_non_cuda_quantization:
            # Keep this conservative by default; opt in if your Unsloth build supports Mac/MLX.
            logger.warning(
                "Disabling 4-bit/8-bit loading on non-CUDA device. "
                "Set trainer.config.allow_non_cuda_quantization=true to try anyway."
            )
            load_in_4bit = False
            load_in_8bit = False

        logger.info(f"Loading HF model via Unsloth: {hf_model}")
        from_pretrained_kwargs: dict[str, Any] = {
            "model_name": hf_model,
            "max_seq_length": max_seq_length,
            "load_in_4bit": load_in_4bit,
            "load_in_8bit": load_in_8bit,
            "trust_remote_code": trust_remote_code,
            "token": hf_token,
        }
        if device.type == "cuda":
            from_pretrained_kwargs["load_in_16bit"] = load_in_16bit
            from_pretrained_kwargs["full_finetuning"] = full_finetuning

        model, tokenizer = FastLanguageModel.from_pretrained(**from_pretrained_kwargs)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.model_max_length = max_seq_length

        # Process each run
        for run in target.runs:
            if run.train is None:
                raise ValueError(f"Run {run.id} has no train config")

            train_config = run.train if run.train else None
            dataset_ids = getattr(train_config, "finetune_dataset_ids", None) if train_config else None
            dataset_probs = getattr(train_config, "finetune_dataset_probs", None) if train_config else None
            streaming = bool(getattr(train_config, "finetune_streaming", False)) if train_config else False
            streaming_shuffle_buffer = int(
                getattr(train_config, "finetune_streaming_shuffle_buffer", 10_000)
            ) if train_config else 10_000
            append_eos = bool(getattr(train_config, "append_eos", True)) if train_config else True

            if not dataset_ids:
                raise ValueError(
                    "finetune_dataset_ids must be set in run.train for finetuning; refusing to use defaults."
                )

            train_dataset, _ = build_finetune_text_datasets(
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                dataset_ids=dataset_ids,
                dataset_probs=dataset_probs,
                streaming=streaming,
                streaming_shuffle_buffer=streaming_shuffle_buffer,
                append_eos=append_eos,
            )
            eval_dataset = None

            # Apply LoRA unless full finetuning is requested.
            if not full_finetuning:
                peft_cfg = getattr(run.train, "peft", None)
                lora_cfg = trainer_config.get("lora", {}) if peft_cfg is None else {}
                lora_enabled = True
                if peft_cfg is not None and hasattr(peft_cfg, "enabled"):
                    lora_enabled = bool(peft_cfg.enabled)
                if "enabled" in lora_cfg:
                    lora_enabled = bool(lora_cfg.get("enabled"))

                if lora_enabled:
                    r = int(getattr(peft_cfg, "r", 8) if peft_cfg is not None else lora_cfg.get("r", 8))
                    alpha = int(getattr(peft_cfg, "alpha", 16) if peft_cfg is not None else lora_cfg.get("alpha", 16))
                    dropout = float(
                        getattr(peft_cfg, "dropout", 0.0) if peft_cfg is not None else lora_cfg.get("dropout", 0.0)
                    )
                    bias = str(
                        getattr(peft_cfg, "bias", "none") if peft_cfg is not None else lora_cfg.get("bias", "none")
                    )
                    target_modules = (
                        getattr(peft_cfg, "target_modules", None)
                        if peft_cfg is not None
                        else lora_cfg.get("target_modules")
                    )
                    use_gradient_checkpointing = lora_cfg.get("use_gradient_checkpointing", "unsloth")
                    random_state = int(lora_cfg.get("random_state", 3407))
                    use_rslora = bool(lora_cfg.get("use_rslora", False))
                    loftq_config = lora_cfg.get("loftq_config", None)

                    logger.info("Applying Unsloth LoRA adapters...")
                    model = FastLanguageModel.get_peft_model(
                        model,
                        r=r,
                        lora_alpha=alpha,
                        lora_dropout=dropout,
                        bias=bias,
                        target_modules=target_modules,
                        use_gradient_checkpointing=use_gradient_checkpointing,
                        random_state=random_state,
                        max_seq_length=max_seq_length,
                        use_rslora=use_rslora,
                        loftq_config=loftq_config,
                    )

            if hasattr(model, "print_trainable_parameters"):
                try:
                    model.print_trainable_parameters()
                except Exception as e:
                    logger.warning(f"Failed to print trainable parameters: {e}")

            # Create data collator
            collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

            # Get training config from run.train if available
            train_config = run.train if run.train else None
            max_steps = None
            num_epochs = None
            learning_rate = 2e-5
            batch_size = 4
            grad_accum = 1

            # Steps is on run, not run.train
            if hasattr(run, "steps") and run.steps is not None:
                max_steps = int(run.steps)
                logger.info(f"Using steps from run.steps: {max_steps}")
            else:
                logger.warning(f"run.steps not found or None. Available run attributes: {dir(run)}")

            if train_config:
                if hasattr(train_config, "lr") and train_config.lr:
                    learning_rate = float(train_config.lr)
                if hasattr(train_config, "batch_size") and train_config.batch_size:
                    batch_size = int(train_config.batch_size)
                if hasattr(train_config, "gradient_accumulation_steps") and train_config.gradient_accumulation_steps:
                    grad_accum = int(train_config.gradient_accumulation_steps)
                if hasattr(train_config, "block_size") and train_config.block_size:
                    if int(train_config.block_size) != max_seq_length:
                        logger.warning(
                            f"Run block_size {train_config.block_size} does not match "
                            f"tokenization length {max_seq_length}; using tokenized length"
                        )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"research/dba/100k_finetuned/{run.id}/{timestamp}"

            if max_steps is None and num_epochs is None:
                num_epochs = 3
                logger.warning("No steps or epochs specified, defaulting to 3 epochs")

            training_args = _build_training_args(
                output_dir=output_dir,
                evaluation_strategy="epoch",
                eval_strategy="epoch",
                save_strategy="epoch",
                learning_rate=learning_rate,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                gradient_accumulation_steps=grad_accum,
                # HF requires at most one of fp16/bf16.
                bf16=device.type == "cuda" and torch.cuda.is_bf16_supported(),
                fp16=device.type == "cuda" and not torch.cuda.is_bf16_supported(),
                push_to_hub=False,
                save_total_limit=3,
                logging_steps=10,
                max_steps=max_steps if max_steps is not None else -1,
                num_train_epochs=float(num_epochs) if num_epochs is not None else 1.0,
            )

            sft_args = SFTConfig(
                max_length = max_seq_length,
                per_device_train_batch_size = 2,
                gradient_accumulation_steps = 4,
                warmup_steps = 10,
                max_steps = 60,
                logging_steps = 1,
                output_dir = "outputs",
                optim = "adamw_8bit",
                seed = 3407,
            )
            trainer = SFTTrainer(
                model = model,
                train_dataset = train_dataset,
                eval_dataset = eval_dataset,
                data_collator = collator,
                # MPS uses a different SFTConfig type; cast keeps typing clean.
                args = cast(Any, sft_args),
            )
            trainer.train()

        return None


def _build_training_args(**kwargs: Any) -> TrainingArguments:
    signature = inspect.signature(TrainingArguments.__init__)
    allowed = set(signature.parameters.keys())
    allowed.discard("self")
    filtered = {key: value for key, value in kwargs.items() if key in allowed}
    return TrainingArguments(**filtered)
