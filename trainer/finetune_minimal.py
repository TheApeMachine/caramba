"""Minimal finetuning trainer.

Focused on straightforward LM finetuning: dataset → forward → loss → step.
No extra instrumentation, no special research features.
"""
from __future__ import annotations

import sys
from pathlib import Path
import os

# Reduce CUDA allocator fragmentation by default (helps long runs / near-OOM).
# Must be set before CUDA allocations begin; this script is commonly launched
# directly via `accelerate launch trainer/finetune_minimal.py`.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# These training scripts are often invoked as plain files under e.g.
# `accelerate launch trainer/finetune_minimal.py`. In that mode, Python puts
# `trainer/` on sys.path but not the repo root, so imports like `data.*` fail.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from typing import Any, Literal, cast
from datetime import datetime
import inspect
import torch
from data.datasets.finetune_text import build_finetune_text_datasets
from transformers import (
    Trainer,
    TrainingArguments,
    default_data_collator,
    GPT2Tokenizer,
)
from peft.tuners.p_tuning import PromptEncoderConfig  # type: ignore[import-not-found]
from peft import LoraConfig, get_peft_model  # type: ignore[import-not-found]
from peft import TaskType  # type: ignore[import-not-found]

from adapter.model import CompatibleWrapper, HFConfigShim
from console import logger
from model import Model
from config.embedder import TokenEmbedderConfig
from config.model import ModelConfig
from config.layer import AttentionLayerConfig
from config.topology import (
    BranchingTopologyConfig,
    CyclicTopologyConfig,
    NestedTopologyConfig,
    ParallelTopologyConfig,
    RecurrentTopologyConfig,
    ResidualTopologyConfig,
    SequentialTopologyConfig,
    StackedTopologyConfig,
    GraphTopologyConfig,
)
from compiler.lower import Lowerer
from compiler.validate import Validator

DEFAULT_SEQ_LENGTH = 2048


def _print_distributed_env() -> None:
    """Print lightweight diagnostics for accelerate/torchrun launches."""
    keys = (
        "CUDA_VISIBLE_DEVICES",
        "LOCAL_RANK",
        "RANK",
        "WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "NCCL_P2P_DISABLE",
        "NCCL_IB_DISABLE",
    )
    logger.info("Distributed env: " + " ".join(f"{k}={os.getenv(k)!r}" for k in keys))
    try:
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            names = [torch.cuda.get_device_name(i) for i in range(count)]
            logger.info(f"torch.cuda: device_count={count} devices={names}")
        else:
            logger.info("torch.cuda: not available")
    except Exception as e:
        logger.warning(f"torch.cuda: failed to query devices: {e}")


def _cleanup_distributed() -> None:
    """Avoid NCCL resource-leak warnings on exit (accelerate/torchrun)."""
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    except Exception:
        # Best-effort cleanup; never mask a real training error.
        pass


def get_device():
    # In distributed (accelerate/torchrun), each process must bind to its LOCAL_RANK GPU.
    # Also, when using Accelerate/FSDP we must NOT move the model to CUDA manually.
    if torch.cuda.is_available():
        try:
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        except ValueError:
            local_rank = 0
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def _is_distributed() -> bool:
    try:
        return int(os.environ.get("WORLD_SIZE", "1")) > 1
    except ValueError:
        return False

class FinetuneMinimalTrainer:
    def __init__(self, *, checkpoint_dir: str | None = None) -> None:
        # checkpoint_dir is accessed from target.trainer.config in run()
        # This __init__ exists for registry compatibility
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
        distributed = _is_distributed()

        if dry_run:
            return None
        
        # Get model config from target.system.config.model
        system_config = target.system.config
        
        if not isinstance(system_config, dict):
            raise ValueError("target.system.config must be a dict")
        
        model_config_dict = system_config.get("model")
        
        if not isinstance(model_config_dict, dict):
            raise ValueError("target.system.config.model must be a dict")
        
        # Parse, lower, and validate model config
        cfg = ModelConfig.model_validate(model_config_dict)
        cfg = Lowerer().lower_model(cfg)
        Validator().validate_model_config(cfg)

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

        logger.info("Building base model from config...")
        dtype = torch.float16 if device.type != "cpu" else torch.float32

        # Load tokenizer (fallback to GPT-2; prefer manifest overrides if provided)
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2", trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.model_max_length = max_seq_length

        dims = _extract_model_dims(cfg)
        tokenizer_vocab = _tokenizer_vocab_size(tokenizer)
        _validate_tokenizer_vocab(tokenizer_vocab, dims["vocab_size"])
        
        # Process each run
        for run in target.runs:
            if run.train is None:
                raise ValueError(f"Run {run.id} has no train config")
            
            checkpoint_dir = getattr(run.train, "load_checkpoint", None)
            if not checkpoint_dir:
                raise ValueError(f"Run {run.id} has no load_checkpoint")
            
            # Check if variable wasn't resolved
            if isinstance(checkpoint_dir, str) and "${" in checkpoint_dir:
                logger.error(
                    f"Variable substitution failed for load_checkpoint in run {run.id}: {checkpoint_dir}\n"
                    f"This suggests the variable wasn't resolved by the compiler. "
                    f"Check that the variable is defined in the manifest's 'vars' section."
                )
                raise ValueError(f"Unresolved variable in load_checkpoint: {checkpoint_dir}")
            
            logger.info(f"Processing run {run.id} with checkpoint {checkpoint_dir}")
            
            # Build fresh model for this run
            # IMPORTANT:
            # - If distributed/FSDP, do NOT move to CUDA here; Accelerate will place shards per-rank.
            # - Keep model on CPU but set dtype (fp16 params are ok; they won't execute on CPU before wrapping).
            if distributed and device.type == "cuda":
                base_model = Model(cfg).to(dtype=dtype)
            else:
                base_model = Model(cfg).to(device=device, dtype=dtype)
            
            # Verify model has parameters (safety check)
            param_count = sum(p.numel() for p in base_model.parameters())
            if param_count == 0:
                raise RuntimeError("Model has no parameters after initialization")
            
            # Load checkpoint weights
            logger.info(f"Loading checkpoint weights from {checkpoint_dir}")
            # In distributed, load to CPU to avoid mismatched device placement across ranks.
            map_loc = "cpu" if (distributed and device.type == "cuda") else device
            payload = torch.load(str(checkpoint_dir), map_location=map_loc, weights_only=False)
            
            # Extract state_dict
            state_dict = payload
            if isinstance(payload, dict):
                for key in ("system_state_dict", "model_state_dict", "state_dict", "model"):
                    if key in payload:
                        state_dict = payload[key]
                        break
            
            # Strip torch.compile prefix if present
            if isinstance(state_dict, dict) and any(k.startswith("_orig_mod.") for k in state_dict.keys()):
                state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
            
            # Load weights
            missing, unexpected = base_model.load_state_dict(state_dict, strict=False)

            if missing:
                logger.warning(f"Missing keys: {len(missing)}\n{missing}")
            if unexpected:
                logger.warning(f"Unexpected keys: {len(unexpected)}\n{unexpected}")
            
            logger.success(f"Loaded checkpoint with {len(state_dict)} keys")
            
            # Verify model still has parameters after loading
            param_count_after = sum(p.numel() for p in base_model.parameters())
            if param_count_after == 0:
                raise RuntimeError("Model has no parameters after loading checkpoint")
            
            # Get training config from run.train if available (used for PEFT + args)
            train_config = run.train if run.train else None

            dataset_ids = getattr(train_config, "finetune_dataset_ids", None) if train_config else None
            dataset_probs = getattr(train_config, "finetune_dataset_probs", None) if train_config else None
            streaming = bool(getattr(train_config, "finetune_streaming", False)) if train_config else False
            streaming_shuffle_buffer = int(
                getattr(train_config, "finetune_streaming_shuffle_buffer", 10_000)
            ) if train_config else 10_000
            append_eos = bool(getattr(train_config, "append_eos", True)) if train_config else True
            pin_memory = bool(getattr(train_config, "pin_memory", False)) if train_config else False

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

            # Configure PEFT (LoRA if requested, else p-tuning)
            if dims["vocab_size"] != tokenizer_vocab:
                logger.warning(
                    "Tokenizer vocab_size does not match model vocab_size "
                    f"({tokenizer_vocab} vs {dims['vocab_size']}). "
                    "Logits will be sliced to tokenizer vocab size for loss."
                )
            
            # Create HF config shim (based on PretrainedConfig for full compatibility)
            hf_config = HFConfigShim(
                hidden_size=dims["hidden_size"],
                num_attention_heads=dims["num_attention_heads"],
                num_hidden_layers=dims["num_hidden_layers"],
                vocab_size=dims["vocab_size"],
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            # Wrap model and apply PEFT
            wrapped = CompatibleWrapper(base_model, hf_config)
            
            # Verify wrapper has parameters before PEFT
            wrapper_param_count = sum(p.numel() for p in wrapped.parameters())
            if wrapper_param_count == 0:
                raise RuntimeError("Wrapper has no parameters before PEFT application")
            
            # Verify base_model attribute exists
            if not hasattr(wrapped, "base_model") or wrapped.base_model is None:
                raise RuntimeError("Wrapper missing base_model attribute")
            
            # Verify config attribute exists
            if not hasattr(wrapped, "config") or wrapped.config is None:
                raise RuntimeError("Wrapper missing config attribute")
            
            use_lora = False
            peft_cfg = getattr(train_config, "peft", None) if train_config else None
            if peft_cfg is not None:
                enabled = getattr(peft_cfg, "enabled", True)
                peft_type = str(getattr(peft_cfg, "type", "lora")).lower()
                use_lora = bool(enabled) and peft_type == "lora"

            logger.info("Applying PEFT adapters...")
            
            # Verify input embeddings exist before PEFT
            # Note: We skip this check if get_input_embeddings() raises an error,
            # as PEFT will handle it internally and provide better error messages
            try:
                input_emb = wrapped.get_input_embeddings()
                if input_emb is not None:
                    logger.info(f"Input embeddings found: {type(input_emb).__name__}")
                else:
                    logger.warning("get_input_embeddings() returned None - PEFT may handle this")
            except Exception as e:
                logger.warning(f"Could not verify input embeddings (PEFT will handle): {e}")
                # Don't raise - let PEFT handle the error with better context
            
            try:
                import traceback

                logger.info(f"Wrapped model type: {type(wrapped).__name__}")
                logger.info(f"Wrapped base_model type: {type(wrapped.base_model).__name__}")

                if use_lora:
                    r = int(getattr(peft_cfg, "r", 8))
                    alpha = int(getattr(peft_cfg, "alpha", 16))
                    dropout = float(getattr(peft_cfg, "dropout", 0.0))
                    bias = str(getattr(peft_cfg, "bias", "none")).lower()
                    if bias not in ("none", "all", "lora_only"):
                        bias = "none"
                    bias = cast(Literal["none", "all", "lora_only"], bias)
                    target_modules = getattr(peft_cfg, "target_modules", None)
                    modules_to_save = getattr(peft_cfg, "modules_to_save", None)
                    if not target_modules:
                        target_modules = _infer_lora_target_modules(wrapped)
                        if target_modules is None:
                            logger.warning(
                                "Could not infer LoRA target modules; falling back to p-tuning."
                            )
                            use_lora = False

                if use_lora:
                    lora_config = LoraConfig(  # type: ignore[call-arg]
                        r=r,
                        lora_alpha=alpha,
                        lora_dropout=dropout,
                        bias=bias,
                        target_modules=target_modules,
                        modules_to_save=modules_to_save,
                        task_type=TaskType.CAUSAL_LM,
                    )
                    logger.info(f"LoRA config: {lora_config}")
                    peft_model = get_peft_model(wrapped, lora_config)  # type: ignore[arg-type]
                else:
                    # Set base_model_name_or_path to avoid None issues in PEFT
                    # Using a generic name since we're using a custom model
                    p_config = PromptEncoderConfig(  # type: ignore[call-arg]
                        base_model_name_or_path="custom-caramba-model",
                        encoder_reparameterization_type="MLP",
                        encoder_hidden_size=128,
                        encoder_num_layers=2,
                        num_attention_heads=dims["num_attention_heads"],
                        num_layers=dims["num_hidden_layers"],
                        num_transformer_submodules=1,
                        num_virtual_tokens=20,
                        token_dim=dims["hidden_size"],
                        task_type=TaskType.CAUSAL_LM,
                    )
                    logger.info(f"P-tuning config: {p_config}")
                    peft_model = get_peft_model(wrapped, p_config)  # type: ignore[arg-type]
            except Exception as e:
                logger.error(f"Failed to apply PEFT: {e}")
                logger.error(f"Exception type: {type(e).__name__}")
                logger.error(f"Traceback:\n{traceback.format_exc()}")
                raise
            
            # Verify PEFT model was created successfully
            if peft_model is None:
                raise RuntimeError("get_peft_model returned None")
            
            # Verify PEFT model has required attributes
            if not hasattr(peft_model, "base_model") or peft_model.base_model is None:
                raise RuntimeError("PEFT model missing base_model attribute")
            
            # Verify PEFT model has parameters
            try:
                peft_param_count = sum(p.numel() for p in peft_model.parameters())
                logger.info(f"PEFT model has {peft_param_count} total parameters")
            except Exception as e:
                logger.error(f"Failed to count PEFT model parameters: {e}")
                raise
            
            # Try to print trainable parameters (this might fail)
            try:
                peft_model.print_trainable_parameters()
            except Exception as e:
                logger.warning(f"Failed to print trainable parameters: {e}")
                # Try to manually count trainable params
                try:
                    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
                    total = sum(p.numel() for p in peft_model.parameters())
                    logger.info(f"Trainable params: {trainable:,} || All params: {total:,} || Trainable%: {100 * trainable / total:.4f}")
                except Exception as e2:
                    logger.error(f"Failed to manually count trainable parameters: {e2}")
                    raise
            
            # IMPORTANT: Use a collator that preserves dataset-provided `labels`.
            # `DataCollatorForLanguageModeling(mlm=False)` overwrites labels with input_ids,
            # which breaks assistant-only masking in supervised finetuning.
            collator = default_data_collator
            
            max_steps = None
            num_epochs = None
            learning_rate = 1e-5
            batch_size = 1
            grad_accum_steps = 1
            warmup_steps = 0
            weight_decay = 0.0
            lr_scheduler_type = "constant"
            optim = "lion"
            max_grad_norm = 0.0
            
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
                if (
                    hasattr(train_config, "gradient_accumulation_steps")
                    and train_config.gradient_accumulation_steps
                ):
                    grad_accum_steps = int(train_config.gradient_accumulation_steps)
                if hasattr(train_config, "block_size") and train_config.block_size:
                    if int(train_config.block_size) != max_seq_length:
                        logger.warning(
                            f"Run block_size {train_config.block_size} does not match "
                            f"tokenization length {max_seq_length}; using tokenized length"
                        )
                if hasattr(train_config, "warmup_steps") and train_config.warmup_steps:
                    warmup_steps = int(train_config.warmup_steps)
                if hasattr(train_config, "weight_decay") and train_config.weight_decay is not None:
                    weight_decay = float(train_config.weight_decay)
                if hasattr(train_config, "grad_clip_norm") and train_config.grad_clip_norm is not None:
                    max_grad_norm = float(train_config.grad_clip_norm)
                if hasattr(train_config, "scheduler") and train_config.scheduler:
                    scheduler = str(train_config.scheduler).lower()
                    if scheduler in ("linear", "cosine", "constant"):
                        lr_scheduler_type = scheduler
                if hasattr(train_config, "optimizer") and train_config.optimizer:
                    optimizer = str(train_config.optimizer).lower()
                    if optimizer == "adamw":
                        fused = bool(getattr(train_config, "fused_optimizer", False))
                        optim = "adamw_torch_fused" if fused else "adamw_torch"
                    elif optimizer == "sgd":
                        optim = "sgd"
                    elif optimizer == "lion":
                        optim = "lion"

            # On pre-Ampere CUDA GPUs, full-sequence attention training at 2048 tokens
            # can OOM easily even with FSDP because activations/attention buffers are per-rank.
            # If the user requested a microbatch > 1, downshift to 1 and compensate with
            # gradient accumulation to preserve the effective global batch size.
            if device.type == "cuda":
                try:
                    cc_major, _cc_minor = torch.cuda.get_device_capability(device)
                except Exception:
                    cc_major = 0
                if int(cc_major) < 8 and int(max_seq_length) >= 1024 and int(batch_size) > 1:
                    old_bs = int(batch_size)
                    batch_size = 1
                    grad_accum_steps = max(1, int(grad_accum_steps) * old_bs)
                    logger.warning(
                        "Auto-adjusting microbatch for pre-Ampere CUDA: "
                        f"per_device_train_batch_size {old_bs} -> {batch_size}, "
                        f"gradient_accumulation_steps -> {grad_accum_steps} "
                        f"(seq_len={int(max_seq_length)} cc_major={int(cc_major)})."
                    )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"research/dba/100k_finetuned/{run.id}/{timestamp}"
            
            if max_steps is None and num_epochs is None:
                num_epochs = 3
                logger.warning("No steps or epochs specified, defaulting to 3 epochs")

            eval_strategy = "no" if streaming else "epoch"
            save_strategy = "steps" if streaming else "epoch"
            save_steps = None
            if streaming:
                if max_steps is not None and max_steps > 0:
                    save_steps = max(100, max_steps // 5)
                else:
                    save_steps = 500

            # HuggingFace Trainer uses its own DataLoader pinning flag
            # (`TrainingArguments.dataloader_pin_memory`) which defaults to True.
            # Mirror our training config and disable it for non-CUDA devices (MPS/CPU).
            hf_dataloader_pin_memory = bool(pin_memory) and device.type == "cuda"

            training_args = _build_training_args(
                output_dir=output_dir,
                evaluation_strategy=eval_strategy,
                eval_strategy=eval_strategy,
                save_strategy=save_strategy,
                save_steps=save_steps,
                learning_rate=learning_rate,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                gradient_accumulation_steps=grad_accum_steps,
                warmup_steps=warmup_steps,
                weight_decay=weight_decay,
                lr_scheduler_type=lr_scheduler_type,
                optim=optim,
                max_grad_norm=max_grad_norm,
                # HF requires at most one of fp16/bf16.
                bf16=device.type == "cuda" and torch.cuda.is_bf16_supported(),
                fp16=device.type == "cuda" and not torch.cuda.is_bf16_supported(),
                dataloader_pin_memory=hf_dataloader_pin_memory,
                push_to_hub=False,
                save_total_limit=3,
                logging_steps=10,
                max_steps=max_steps if max_steps is not None else -1,
                num_train_epochs=float(num_epochs) if num_epochs is not None else 1.0,
            )

            trainer = _FinetuneTrainer(
                model=peft_model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=collator,
                valid_vocab_size=tokenizer_vocab,
            )

            try:
                trainer.train()
            except torch.OutOfMemoryError:
                # Ensure the rank tears down cleanly; avoid cascading crashes in
                # torchrun/accelerate shutdown paths after OOM.
                if device.type == "cuda":
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                raise
                    
        return None


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint to run this trainer via a manifest."""
    import argparse
    from pathlib import Path

    from experiment.runner import run_from_manifest_path

    parser = argparse.ArgumentParser(
        prog="finetune_minimal",
        description="Run Caramba manifest targets (finetune_minimal trainer)",
    )
    parser.add_argument("--manifest", required=True, help="Path to a manifest YAML")
    parser.add_argument("--target", default=None, help="Target name (or omit to run all targets)")
    parser.add_argument("--dry-run", action="store_true", help="Validate + print plan only (no training)")
    args = parser.parse_args(argv)

    _print_distributed_env()
    try:
        run_from_manifest_path(Path(args.manifest), target=args.target, dry_run=bool(args.dry_run))
        return 0
    finally:
        _cleanup_distributed()


if __name__ == "__main__":
    raise SystemExit(main())


def _extract_model_dims(cfg: ModelConfig) -> dict[str, int]:
    def walk(node: object) -> tuple[int, int | None, int | None]:
        if isinstance(node, AttentionLayerConfig):
            return 1, int(node.n_heads), int(node.d_model)
        if isinstance(
            node,
            (
                NestedTopologyConfig,
                StackedTopologyConfig,
                ResidualTopologyConfig,
                SequentialTopologyConfig,
                ParallelTopologyConfig,
                BranchingTopologyConfig,
                CyclicTopologyConfig,
                RecurrentTopologyConfig,
            ),
        ):
            total = 0
            heads: int | None = None
            d_model: int | None = None
            for layer in node.layers:
                count, layer_heads, layer_d_model = walk(layer)
                total += count
                if heads is None and layer_heads is not None:
                    heads = layer_heads
                if d_model is None and layer_d_model is not None:
                    d_model = layer_d_model
            total *= int(node.repeat)
            return total, heads, d_model
        if isinstance(node, GraphTopologyConfig):
            return 0, None, None
        return 0, None, None

    attn_layers, attn_heads, attn_d_model = walk(cfg.topology)

    hidden_size = None
    vocab_size = None
    if isinstance(cfg.embedder, TokenEmbedderConfig):
        hidden_size = int(cfg.embedder.d_model)
        vocab_size = int(cfg.embedder.vocab_size)

    if hidden_size is None:
        hidden_size = attn_d_model
    if vocab_size is None and cfg.vocab_size is not None:
        vocab_size = int(cfg.vocab_size)

    if hidden_size is None:
        raise ValueError("Could not infer hidden_size from model config")
    if vocab_size is None:
        raise ValueError("Could not infer vocab_size from model config")

    num_hidden_layers = int(attn_layers) if attn_layers > 0 else int(getattr(cfg.weight_init, "n_layers", 1))
    num_attention_heads = int(attn_heads) if attn_heads is not None else max(1, hidden_size // 64)

    return {
        "hidden_size": int(hidden_size),
        "num_attention_heads": int(num_attention_heads),
        "num_hidden_layers": int(num_hidden_layers),
        "vocab_size": int(vocab_size),
    }


def _build_training_args(**kwargs: Any) -> TrainingArguments:
    signature = inspect.signature(TrainingArguments.__init__)
    allowed = set(signature.parameters.keys())
    allowed.discard("self")
    filtered = {key: value for key, value in kwargs.items() if key in allowed}
    return TrainingArguments(**filtered)


def _tokenizer_vocab_size(tokenizer: Any) -> int:
    if hasattr(tokenizer, "vocab_size") and tokenizer.vocab_size is not None:
        return int(tokenizer.vocab_size)
    if hasattr(tokenizer, "get_vocab"):
        vocab = tokenizer.get_vocab()
        if isinstance(vocab, dict):
            return int(len(vocab))
    return int(len(tokenizer))


def _validate_tokenizer_vocab(tokenizer_vocab_size: int, model_vocab_size: int) -> None:
    if tokenizer_vocab_size > model_vocab_size:
        raise ValueError(
            "Tokenizer vocab_size is larger than model vocab_size "
            f"({tokenizer_vocab_size} vs {model_vocab_size}). "
            "Use a tokenizer that matches the model vocab."
        )


def _infer_lora_target_modules(model: torch.nn.Module) -> list[str] | None:
    """Infer LoRA target module names from Linear layers."""
    names: set[str] = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            leaf = name.split(".")[-1]
            if leaf:
                names.add(leaf)
    return sorted(names) if names else None


class _FinetuneTrainer(Trainer):
    def __init__(self, *args: Any, valid_vocab_size: int, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._valid_vocab_size = int(valid_vocab_size)

    def compute_loss(  # type: ignore[override]
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        **_: Any,
    ):
        labels = inputs.get("labels")
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        # Guard against CUDA device-side asserts from out-of-range token IDs
        # (these typically surface later and look unrelated, e.g. during RMSNorm).
        input_ids = model_inputs.get("input_ids")
        if isinstance(input_ids, torch.Tensor):
            # Avoid reductions on non-integer tensors if something upstream casted IDs.
            if input_ids.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8):
                # We still accept it (embedder will .long()), but this is a strong smell.
                input_ids_i64 = input_ids.to(dtype=torch.int64)
            else:
                input_ids_i64 = input_ids.to(dtype=torch.int64)
            try:
                min_id = int(input_ids_i64.min().item())
                max_id = int(input_ids_i64.max().item())
            except Exception as e:
                raise RuntimeError(f"Failed to validate input_ids range: {type(e).__name__}: {e}") from e
            if min_id < 0 or max_id >= int(self._valid_vocab_size):
                raise ValueError(
                    "Found out-of-range token IDs in input_ids. "
                    f"min={min_id} max={max_id} valid=[0,{int(self._valid_vocab_size) - 1}] "
                    f"shape={tuple(input_ids.shape)} dtype={input_ids.dtype} device={input_ids.device}"
                )
        outputs = model(**model_inputs)
        logits = outputs.get("logits") if isinstance(outputs, dict) else outputs.logits
        if logits is None:
            raise ValueError("Model did not return logits for loss computation")

        if int(logits.shape[-1]) > self._valid_vocab_size:
            logits = logits[..., : self._valid_vocab_size]

        if labels is None:
            raise ValueError("Labels are required for loss computation")
        assert labels is not None

        seq_len = min(int(logits.shape[-2]), int(labels.shape[-1]))
        if seq_len <= 1:
            raise ValueError("Sequence length too short for loss computation")

        logits = logits[..., :seq_len, :]
        labels = labels[..., :seq_len]

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = torch.nn.CrossEntropyLoss(ignore_index=-100)(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        return (loss, outputs) if return_outputs else loss
