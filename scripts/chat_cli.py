"""Simple console chat for Caramba checkpoints + LoRA adapters."""
from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import GPT2Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adapter.model import CompatibleWrapper, HFConfigShim
from config.target import ExperimentTargetConfig
from compiler import Compiler
from compiler.lower import Lowerer
from compiler.validate import Validator
from config.manifest import Manifest
from config.model import ModelConfig
from console import logger
from infer.generate import GenerateConfig, Generator, _apply_repetition_penalty_, sample_next_token
from model import Model
from model.prompt_adapter import PromptTuningAdapter, load_prompt_embeddings

try:
    from peft import PeftModel  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    PeftModel = None  # type: ignore[assignment]


def _get_device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_dtype(dtype: str | None, device: torch.device) -> torch.dtype:
    if dtype:
        s = dtype.strip().lower()
        if s in ("fp16", "float16"):
            return torch.float16
        if s in ("bf16", "bfloat16"):
            return torch.bfloat16
        return torch.float32
    return torch.float16 if device.type in ("cuda", "mps") else torch.float32


def _load_state_dict(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    payload = torch.load(str(path), map_location=device, weights_only=False)
    state_dict: dict[str, torch.Tensor] = payload
    if isinstance(payload, dict):
        for key in ("system_state_dict", "model_state_dict", "state_dict", "model"):
            if key in payload:
                state_dict = payload[key]
                break
    if isinstance(state_dict, dict) and any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def _extract_logits(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        output = output[0]
    if isinstance(output, dict):
        output = output.get("logits")
    if hasattr(output, "logits"):
        output = output.logits
    if output is None:
        raise ValueError("Model did not return logits.")
    return output


def _format_prompt(history: list[tuple[str, str]], user_text: str, system: str | None) -> str:
    """Match finetune_minimal prompt format."""
    parts: list[str] = []
    if system:
        parts.append(f"System: {system}")
    for role, content in history:
        if role.lower() == "user":
            parts.append(f"User: {content}")
        else:
            parts.append(f"Assistant: {content}")
    parts.append(f"User: {user_text}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


def _format_turn(user_text: str, system: str | None, has_history: bool) -> str:
    """Format only the new turn for cache reuse."""
    prefix = ""
    if not has_history and system:
        prefix = f"System: {system}\n\n"
    return f"{prefix}User: {user_text}\n\nAssistant:"


def _load_from_manifest(manifest_path: Path, target: str) -> dict[str, Any]:
    manifest = Manifest.from_path(manifest_path)
    compiler = Compiler()
    lowered = compiler.lowerer.lower_manifest(manifest)
    compiler.validator.validate_manifest(lowered, print_plan=False)

    target_name = target
    if lowered.entrypoints and target_name in lowered.entrypoints:
        target_name = lowered.entrypoints[target_name]
    if ":" in target_name:
        _, target_name = target_name.split(":", 1)
        target_name = target_name.strip()

    match = next((t for t in lowered.targets if t.name == target_name), None)
    if match is None:
        raise ValueError(f"Target not found in manifest: {target_name}")
    if not isinstance(match, ExperimentTargetConfig):
        raise ValueError(f"Target '{target_name}' is not an experiment.")
    if match.system is None or not isinstance(match.system.config, dict):
        raise ValueError("Target has no system.config model definition.")
    model_config_dict = match.system.config.get("model")
    if not isinstance(model_config_dict, dict):
        raise ValueError("Target system.config.model must be a dict.")

    cfg = ModelConfig.model_validate(model_config_dict)
    cfg = Lowerer().lower_model(cfg)
    Validator().validate_model_config(cfg)

    ckpt = None
    if match.runs:
        run0 = match.runs[0]
        if run0.train is not None:
            ckpt = getattr(run0.train, "load_checkpoint", None)
    return {"model_config": cfg, "checkpoint": ckpt}


def _apply_adapter(
    *,
    base_model: Model,
    cfg: ModelConfig,
    adapter_path: Path | None,
) -> torch.nn.Module:
    if adapter_path is None:
        return base_model

    adapter_dir = adapter_path.parent if adapter_path.is_file() else adapter_path
    cfg_path = adapter_dir / "adapter_config.json"
    peft_type = None
    if cfg_path.exists():
        try:
            import json

            payload = json.loads(cfg_path.read_text(encoding="utf-8"))
            peft_type = str(payload.get("peft_type", "")).lower()
        except Exception:
            peft_type = None

    if peft_type == "lora":
        if PeftModel is None:
            raise RuntimeError("peft is required to load LoRA adapters.")
        dims = _extract_model_dims(cfg)
        hf_config = HFConfigShim(
            hidden_size=dims["hidden_size"],
            num_attention_heads=dims["num_attention_heads"],
            num_hidden_layers=dims["num_hidden_layers"],
            vocab_size=dims["vocab_size"],
        )
        wrapped = CompatibleWrapper(base_model, hf_config)
        return PeftModel.from_pretrained(wrapped, str(adapter_dir))  # type: ignore[arg-type]

    prompt = load_prompt_embeddings(adapter_path)
    if prompt.ndim != 2:
        raise ValueError(f"prompt_embeddings must be rank-2, got {prompt.shape}")
    return PromptTuningAdapter(base_model, prompt)


def _supports_ctx(model: torch.nn.Module) -> bool:
    try:
        sig = inspect.signature(model.forward)
    except Exception:
        return False
    return "ctx" in sig.parameters


def _extract_model_dims(cfg: ModelConfig) -> dict[str, int]:
    from config.layer import AttentionLayerConfig
    from config.embedder import TokenEmbedderConfig
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
    if hidden_size is None or vocab_size is None:
        raise ValueError("Could not infer hidden_size/vocab_size from model config")
    num_hidden_layers = int(attn_layers) if attn_layers > 0 else int(getattr(cfg.weight_init, "n_layers", 1))
    num_attention_heads = int(attn_heads) if attn_heads is not None else max(1, hidden_size // 64)
    return {
        "hidden_size": int(hidden_size),
        "num_attention_heads": int(num_attention_heads),
        "num_hidden_layers": int(num_hidden_layers),
        "vocab_size": int(vocab_size),
    }


def _generate_no_cache(
    *,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    repetition_penalty: float,
    eos_token_id: int | None,
    valid_vocab_size: int | None,
    tokenizer: GPT2Tokenizer,
    stream: bool,
) -> list[int]:
    generated: list[int] = []
    if stream:
        print("\nAssistant> ", end="", flush=True)
    for _ in range(int(max_new_tokens)):
        outputs = model(input_ids)
        logits = _extract_logits(outputs)
        logits = logits[:, -1, :]
        if valid_vocab_size is not None and int(logits.size(-1)) > int(valid_vocab_size):
            logits = logits[..., : int(valid_vocab_size)]
        if repetition_penalty and repetition_penalty != 1.0:
            logits = _apply_repetition_penalty_(logits, token_ids=input_ids, penalty=float(repetition_penalty))
        next_token = sample_next_token(
            logits,
            temperature=float(temperature),
            top_k=int(top_k) if top_k else None,
            top_p=float(top_p) if top_p else None,
        )
        input_ids = torch.cat([input_ids, next_token[:, None]], dim=1)
        tok = int(next_token.item())
        generated.append(tok)
        if stream:
            chunk = tokenizer.decode([tok], skip_special_tokens=True)
            if chunk:
                print(chunk, end="", flush=True)
        if eos_token_id is not None and tok == int(eos_token_id):
            break
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Console chat for Caramba checkpoints.")
    parser.add_argument("--manifest", type=Path, required=True, help="Manifest path (YAML).")
    parser.add_argument("--target", type=str, required=True, help="Target name (or entrypoint).")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Base model checkpoint (.pt).")
    parser.add_argument("--adapter", type=Path, default=None, help="Adapter path or directory.")
    parser.add_argument("--device", type=str, default=None, help="cuda|mps|cpu (default: auto).")
    parser.add_argument("--dtype", type=str, default=None, help="float16|bfloat16|float32 (default: auto).")
    parser.add_argument("--max-new", type=int, default=256, help="Max new tokens per turn.")
    parser.add_argument("--context", type=int, default=2048, help="Context window.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling.")
    parser.add_argument("--repetition-penalty", type=float, default=1.05, help="Repetition penalty.")
    parser.add_argument("--system", type=str, default=None, help="Optional system prompt.")
    parser.add_argument(
        "--stop",
        action="append",
        default=[],
        help="Optional stop string (repeatable). Generation stops when the token stream ends with any stop string.",
    )
    parser.add_argument("--stream", action="store_true", help="Stream tokens as they are generated.")
    parser.add_argument(
        "--reuse-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse KV cache across turns when supported.",
    )
    args = parser.parse_args()

    resolved = _load_from_manifest(args.manifest, args.target)
    cfg: ModelConfig = resolved["model_config"]
    ckpt = args.checkpoint or (Path(str(resolved["checkpoint"])) if resolved["checkpoint"] else None)
    if ckpt is None:
        raise ValueError("No checkpoint provided and none found in manifest run config.")

    device = _get_device(args.device)
    dtype = _resolve_dtype(args.dtype, device)

    logger.info(f"Loading base model: {ckpt}")
    model = Model(cfg).to(device=device, dtype=dtype)
    state_dict = _load_state_dict(ckpt, device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        logger.warning(f"load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")

    if args.adapter:
        logger.info(f"Loading adapter: {args.adapter}")
    model = _apply_adapter(base_model=model, cfg=cfg, adapter_path=args.adapter)
    model.eval()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    eos_token_id = tokenizer.eos_token_id
    valid_vocab_size = int(getattr(tokenizer, "vocab_size", 50257))

    # Tokenize stop strings into token-id suffix sequences.
    stop_sequences: list[list[int]] = []
    stop_token_ids: list[int] = []
    for s in (args.stop or []):
        ss = str(s)
        if not ss:
            continue
        ids = tokenizer.encode(ss)
        if not ids:
            continue
        stop_sequences.append([int(x) for x in ids])
        if len(ids) == 1:
            stop_token_ids.append(int(ids[0]))

    use_cache = _supports_ctx(model)
    gen_cfg = GenerateConfig(
        max_new_tokens=int(args.max_new),
        temperature=float(args.temperature),
        top_k=int(args.top_k) if args.top_k else None,
        top_p=float(args.top_p) if args.top_p else None,
        repetition_penalty=float(args.repetition_penalty),
        eos_token_id=eos_token_id,
        max_seq_len=int(args.context),
        stop_sequences=stop_sequences,
        stop_token_ids=stop_token_ids,
    )
    generator = Generator(model, config=gen_cfg, device=device) if use_cache else None

    history: list[tuple[str, str]] = []
    conversation_ids: list[int] = []
    print("Chat ready. Type 'exit' or 'quit' to stop.")
    while True:
        user_text = input("\nUser> ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            break

        if generator is not None and args.reuse_cache:
            turn_text = _format_turn(user_text, args.system, has_history=bool(conversation_ids))
            turn_ids = tokenizer.encode(turn_text)
            if not turn_ids:
                continue
            if len(conversation_ids) + len(turn_ids) >= int(args.context):
                generator.reset()
                conversation_ids.clear()
                prompt = _format_prompt(history, user_text, args.system)
                turn_ids = tokenizer.encode(prompt)
            input_tensor = torch.tensor([turn_ids], device=device)
            if not conversation_ids:
                logits = generator.prefill(input_tensor)
            else:
                logits = generator.decode_step(input_tensor)
            conversation_ids.extend(turn_ids)
            generated: list[int] = []
            if args.stream:
                print("\nAssistant> ", end="", flush=True)
            for _ in range(int(args.max_new)):
                if valid_vocab_size and int(logits.size(-1)) > int(valid_vocab_size):
                    logits = logits[..., : int(valid_vocab_size)]
                if args.repetition_penalty and args.repetition_penalty != 1.0:
                    token_ids = torch.tensor(
                        [conversation_ids + generated], device=device, dtype=torch.long
                    )
                    logits = _apply_repetition_penalty_(
                        logits,
                        token_ids=token_ids,
                        penalty=float(args.repetition_penalty),
                    )
                next_token = sample_next_token(
                    logits,
                    temperature=float(args.temperature),
                    top_k=int(args.top_k) if args.top_k else None,
                    top_p=float(args.top_p) if args.top_p else None,
                )
                tok = int(next_token.item())
                generated.append(tok)
                conversation_ids.append(tok)
                if args.stream:
                    chunk = tokenizer.decode([tok], skip_special_tokens=True)
                    if chunk:
                        print(chunk, end="", flush=True)
                if eos_token_id is not None and tok == int(eos_token_id):
                    break
                if stop_token_ids and tok in set(stop_token_ids):
                    break
                if stop_sequences:
                    full = conversation_ids
                    hit = False
                    for seq in stop_sequences:
                        k = len(seq)
                        if k > 0 and len(full) >= k and full[-k:] == seq:
                            hit = True
                            break
                    if hit:
                        break
                logits = generator.decode_step(next_token)
        else:
            prompt = _format_prompt(history, user_text, args.system)
            input_ids = tokenizer.encode(prompt)
            input_tensor = torch.tensor([input_ids], device=device)
            if generator is not None:
                generator.reset()
                logits = generator.prefill(input_tensor)
                generated: list[int] = []
                if args.stream:
                    print("\nAssistant> ", end="", flush=True)
                for _ in range(int(args.max_new)):
                    if valid_vocab_size and int(logits.size(-1)) > int(valid_vocab_size):
                        logits = logits[..., : int(valid_vocab_size)]
                    if args.repetition_penalty and args.repetition_penalty != 1.0:
                        token_ids = torch.tensor(
                            [input_ids + generated], device=device, dtype=torch.long
                        )
                        logits = _apply_repetition_penalty_(
                            logits,
                            token_ids=token_ids,
                            penalty=float(args.repetition_penalty),
                        )
                    next_token = sample_next_token(
                        logits,
                        temperature=float(args.temperature),
                        top_k=int(args.top_k) if args.top_k else None,
                        top_p=float(args.top_p) if args.top_p else None,
                    )
                    tok = int(next_token.item())
                    generated.append(tok)
                    if args.stream:
                        chunk = tokenizer.decode([tok], skip_special_tokens=True)
                        if chunk:
                            print(chunk, end="", flush=True)
                    if eos_token_id is not None and tok == int(eos_token_id):
                        break
                    if stop_token_ids and tok in set(stop_token_ids):
                        break
                    if stop_sequences:
                        full = input_ids + generated
                        hit = False
                        for seq in stop_sequences:
                            k = len(seq)
                            if k > 0 and len(full) >= k and full[-k:] == seq:
                                hit = True
                                break
                        if hit:
                            break
                    logits = generator.decode_step(next_token)
            else:
                generated = _generate_no_cache(
                    model=model,
                    input_ids=input_tensor,
                    max_new_tokens=int(args.max_new),
                    temperature=float(args.temperature),
                    top_k=args.top_k,
                    top_p=args.top_p,
                    repetition_penalty=float(args.repetition_penalty),
                    eos_token_id=eos_token_id,
                    valid_vocab_size=valid_vocab_size,
                    tokenizer=tokenizer,
                    stream=bool(args.stream),
                )

        assistant_text = tokenizer.decode(generated, skip_special_tokens=True)
        if not args.stream:
            print(f"\nAssistant> {assistant_text.strip()}")
        else:
            print("")
        history.append(("User", user_text))
        history.append(("Assistant", assistant_text.strip()))


if __name__ == "__main__":
    main()
