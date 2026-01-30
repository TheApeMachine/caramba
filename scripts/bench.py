import os
import sys
import gc
import time
from pathlib import Path
from typing import cast

# Opik OSS runs on localhost and only supports the `default` workspace.
# Force the local URL + workspace so a user-level config (or env var) doesn't break runs.
os.environ.setdefault("OPIK_URL_OVERRIDE", "http://localhost:5173/api/")
if os.environ["OPIK_URL_OVERRIDE"].startswith(("http://localhost", "http://127.0.0.1")):
    # Override (not setdefault): a user config/env can otherwise point to a non-existent
    # workspace and Opik OSS will hard-fail with "Workspace not found".
    os.environ["OPIK_WORKSPACE"] = "default"

from opik import Opik
from opik.evaluation import evaluate
from opik.evaluation.metrics import (
    AnswerRelevance,
    Contains,
    ContextPrecision,
    ContextRecall,
    Equals,
    Hallucination,
    LevenshteinRatio,
    Moderation,
    RegexMatch,
)

# Allow running this script directly (so it can import Caramba modules).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.dba.benchmark_loader import (  # noqa: E402
    load_models_from_benchmark_manifest,
    read_multi_checkpoint_specs,
)

# Caramba models don't implement HuggingFace's `.generate(prompt: str, ...)`.
# Use Caramba's native generation loop instead.
from config.eval import TiktokenTokenizerConfig
from data.tokenizers.builder import TokenizerBuilder
from benchmark.behavior.inference import tokenizer_vocab_size
from infer.generate import Generator, GenerateConfig, sample_next_token
import torch
import torch.nn as nn


def scored_response(*, output_text: str, expected_output: object) -> str:
    """Reduce a raw model output to a single line for scoring.

    Rules:
    1) If `expected_output` is found as a substring of `output_text`, return the full
       line containing it.
    2) Otherwise, return the first line of `output_text`.
    """
    text = str(output_text or "")
    # If the model printed literal "\n" sequences (two chars), treat them as line breaks.
    if "\n" not in text and "\\n" in text:
        text = text.replace("\\n", "\n")

    lines = text.splitlines()
    # Default: first non-empty line (or empty string if nothing).
    out = next((ln for ln in lines if ln.strip() != ""), lines[0] if lines else "")

    exp = "" if expected_output is None else str(expected_output)
    if exp:
        for ln in lines:
            if exp.lower() in ln.lower():
                out = ln
                break

    # Guarantee single-line return (defensive; splitlines() already removes newlines).
    return str(out).replace("\r", "").replace("\n", "").strip()


client = Opik()
DATASET_NAME = os.getenv("OPIK_DATASET_NAME", "In-House")
dataset = client.get_dataset(name=DATASET_NAME)

# Runtime knobs for quick local smoke-runs.
TASK_THREADS = int(os.getenv("OPIK_TASK_THREADS", "1"))
NB_SAMPLES = int(os.getenv("OPIK_NB_SAMPLES", "600"))
MAX_NEW_TOKENS = int(os.getenv("BENCH_MAX_NEW_TOKENS", "64"))
BENCH_DEVICE = os.getenv("BENCH_DEVICE") or None
BENCH_DTYPE = os.getenv("BENCH_DTYPE") or None

MANIFEST_PATH = os.getenv("BENCH_MANIFEST", "research/dba/benchmark-gated.yml")
MANIFEST_TARGET = os.getenv("BENCH_TARGET", "multi_checkpoint_compare")
OPIK_PROJECT_NAME = os.getenv("OPIK_PROJECT_NAME", "bench")
EXPERIMENT_PREFIX = os.getenv("OPIK_EXPERIMENT_PREFIX", "bench-full")

metrics = [
    Hallucination(),
    Moderation(),
    AnswerRelevance(),
    ContextRecall(),
    ContextPrecision(),
    LevenshteinRatio(),
    Contains(),
    RegexMatch(regex="d{3}-d{2}-d{4}"),
    Equals(),
]

# Parse model specs without instantiating models.
specs, _meta = read_multi_checkpoint_specs(manifest_path=MANIFEST_PATH, target=MANIFEST_TARGET)

for spec in specs:
    # Load exactly one model, run benchmark, then free it before the next iteration.
    models = cast(
        dict[str, nn.Module],
        load_models_from_benchmark_manifest(
            manifest_path=MANIFEST_PATH,
            target=MANIFEST_TARGET,
            models=[spec.name],
            with_adapter=False,
            device=BENCH_DEVICE,
            dtype=BENCH_DTYPE,
        ),
    )
    model = models[spec.name]

    tokenizer = TokenizerBuilder().build(TiktokenTokenizerConfig(encoding="gpt2"))
    valid_vocab_size = tokenizer_vocab_size(tokenizer)
    raw_device = getattr(model, "device", None)
    if raw_device is None:
        # Guard for static typing (and safety): only nn.Module has `.parameters()`.
        raw_device = next(model.parameters()).device if isinstance(model, nn.Module) else torch.device("cpu")
    device: torch.device = raw_device if isinstance(raw_device, torch.device) else torch.device(str(raw_device))

    # IMPORTANT: `benchmark.behavior.inference.generate_greedy()` creates a new Generator
    # each call, which reallocates KV-caches every question (very expensive).
    # Instead, allocate once per model and truncate caches between questions.
    gen = Generator(
        model,
        config=GenerateConfig(
            max_new_tokens=int(MAX_NEW_TOKENS),
            temperature=0.0,
            max_seq_len=2048,
        ),
        device=device,
    )

    def evaluation_task(dataset_item):
        prompt_text = dataset_item.get("prompt")
        prompt_ids = tokenizer.encode(str(prompt_text))

        if gen._caches is not None:
            for c in gen._caches:
                c.truncate(0)
            gen._pos = 0

        t0 = time.perf_counter()
        input_ids = torch.tensor([list(prompt_ids)], device=device, dtype=torch.long)
        logits = gen.prefill(input_ids)

        vv = int(valid_vocab_size) if valid_vocab_size is not None else None
        generated: list[int] = []
        for _ in range(int(MAX_NEW_TOKENS)):
            if vv is not None and int(getattr(logits, "shape", [0])[-1]) > int(vv):
                logits = logits[..., : int(vv)]
            next_token = sample_next_token(logits, temperature=0.0)
            generated.append(int(next_token.item()))
            logits = gen.decode_step(next_token)
            if (len(prompt_ids) + len(generated)) >= 2048:
                break

        llm_text = str(tokenizer.decode(generated))
        gen_s = float(time.perf_counter() - t0)

        expected_output = dataset_item.get("expected")
        output = scored_response(output_text=llm_text, expected_output=expected_output)
        return {
            "model": spec.name,
            "input": dataset_item.get("prompt"),
            "output": output,
            "context": [
                dataset_item.get("task_type"),
                dataset_item.get("category"),
                dataset_item.get("difficulty"),
                dataset_item.get("kind"),
                dataset_item.get("prompt"),
                dataset_item.get("expected"),
            ],
            "expected_output": expected_output,
            "reference": expected_output,
            "meta": {
                "model": spec.name,
                "prompt_tokens": int(len(prompt_ids)),
                "generated_tokens": int(len(generated)),
                "generation_seconds": gen_s,
                "raw_output": llm_text,
            },
        }

    _ = evaluate(
        dataset=dataset,
        task=evaluation_task,
        scoring_metrics=metrics,
        experiment_name=f"{EXPERIMENT_PREFIX}:{spec.name}",
        project_name=OPIK_PROJECT_NAME,
        experiment_config={
            "manifest_path": str(MANIFEST_PATH),
            "manifest_target": str(MANIFEST_TARGET),
            "model": str(spec.name),
            "max_new_tokens": int(MAX_NEW_TOKENS),
            "device": str(BENCH_DEVICE) if BENCH_DEVICE is not None else None,
            "dtype": str(BENCH_DTYPE) if BENCH_DTYPE is not None else None,
        },
        verbose=1,
        nb_samples=NB_SAMPLES,
        task_threads=TASK_THREADS,
    )

    # Explicitly release references + clear accelerator caches.
    del gen
    del model
    del models
    del tokenizer
    del evaluation_task
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass