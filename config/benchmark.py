"""Benchmark configuration for measuring model quality and performance.

Benchmarks run after training to measure what matters:
- Perplexity: language modeling quality
- Latency: tokens per second
- Memory: KV-cache and peak usage
- Accuracy: downstream task performance
- Generation: text quality assessment
"""
from __future__ import annotations

import enum
from typing import Literal

from pydantic import BaseModel, Field

from config import NonNegativeInt, PositiveFloat, PositiveInt, Probability
from config.eval import TiktokenTokenizerConfig, TokenizerConfig
from config.kvcache import KVCacheConfig


class BenchmarkType(str, enum.Enum):
    """Types of benchmarks available."""

    PERPLEXITY = "perplexity"
    LATENCY = "latency"
    MEMORY = "memory"
    ACCURACY = "accuracy"
    GENERATION = "generation"
    BEHAVIOR = "behavior"
    BEHAVIOR_INSTRUCT = "behavior_instruct"
    CONTEXT = "context"


class PerplexityBenchmarkConfig(BaseModel):
    """Measure language modeling perplexity on a dataset.

    Lower perplexity = better language modeling. This is the core
    metric for comparing model quality after upcycling.
    """

    type: Literal[BenchmarkType.PERPLEXITY] = BenchmarkType.PERPLEXITY
    dataset: str
    block_size: PositiveInt = 2048
    batch_size: PositiveInt = 1
    num_batches: PositiveInt | None = None
    stride: PositiveInt | None = None
    # Optional "effective" vocab size (tokenizer vocab) to ensure padded model
    # vocabs (e.g. 50304) are scored fairly against real token IDs (e.g. 50257).
    # When set, logits are sliced/masked to this size before CE/log-softmax.
    valid_vocab_size: PositiveInt | None = None


class LatencyBenchmarkConfig(BaseModel):
    """Measure generation speed (tokens per second).

    Tests different prompt lengths, generation lengths, and batch sizes
    to characterize throughput across usage patterns.
    """

    type: Literal[BenchmarkType.LATENCY] = BenchmarkType.LATENCY
    # REQUIRED: Seed used to generate deterministic synthetic inputs for latency.
    # This is part of the benchmark definition and must be explicit in the manifest.
    seed: PositiveInt
    prompt_lengths: list[PositiveInt] = Field(
        default_factory=lambda: [128, 512, 1024, 2048]
    )
    generation_lengths: list[PositiveInt] = Field(
        default_factory=lambda: [128, 256, 512]
    )
    batch_sizes: list[PositiveInt] = Field(default_factory=lambda: [1, 4, 8])
    warmup_runs: PositiveInt = 3
    timed_runs: PositiveInt = 10
    use_cache: bool = True
    cache_kind: str = "fp16"
    # Optional explicit KV-cache policy (enables heterogeneous DBA caching).
    # If provided, it overrides cache_kind/qblock/residual_len behavior in the generator.
    cache_policy: KVCacheConfig | None = None
    # Optional "effective" vocab size to sample test tokens from. This helps keep
    # inputs comparable when model vocab is padded beyond tokenizer vocab.
    valid_vocab_size: PositiveInt | None = None


class MemoryBenchmarkConfig(BaseModel):
    """Measure memory usage (KV-cache and peak).

    For DBA upcycling, we expect significant KV-cache reduction due to
    the compressed attention dimensions.
    """

    type: Literal[BenchmarkType.MEMORY] = BenchmarkType.MEMORY
    # REQUIRED: Seed used to generate deterministic synthetic inputs for memory measurement.
    # This is part of the benchmark definition and must be explicit in the manifest.
    seed: PositiveInt
    sequence_lengths: list[PositiveInt] = Field(
        default_factory=lambda: [512, 1024, 2048, 4096]
    )
    batch_sizes: list[PositiveInt] = Field(default_factory=lambda: [1, 4, 8])
    measure_peak: bool = True
    measure_kvcache: bool = True
    quantization_modes: list[str] = Field(
        default_factory=lambda: ["fp16", "q8", "q4"]
    )
    # Optional "effective" vocab size to sample test tokens from. This helps keep
    # inputs comparable when model vocab is padded beyond tokenizer vocab.
    valid_vocab_size: PositiveInt | None = None


class AccuracyBenchmarkConfig(BaseModel):
    """Measure accuracy on downstream tasks.

    Uses standard evaluation benchmarks like HellaSwag, WinoGrande, etc.
    to assess whether model capabilities are preserved after upcycling.
    """

    type: Literal[BenchmarkType.ACCURACY] = BenchmarkType.ACCURACY
    tasks: list[str]
    # Text tokenizer used to encode prompts/choices for scoring.
    # Default matches the paper's GPT-style experiments.
    tokenizer: TokenizerConfig = Field(default_factory=lambda: TiktokenTokenizerConfig(encoding="gpt2"))
    # 0 = zero-shot evaluation.
    num_fewshot: NonNegativeInt = 0
    limit: PositiveInt | None = None
    # Optional sliding context window for scoring long prompts.
    context_window: PositiveInt | None = None
    # Optional: print a small number of examples per task for insight.
    # This does not affect scoring, only console output.
    # 0 = don't print examples.
    print_examples: NonNegativeInt = 0
    print_only_incorrect: bool = True
    print_max_chars: PositiveInt = 240
    # Stream examples to console as they are evaluated (live progress).
    # Shows prompt snippet, model choice, and correct/incorrect status in real-time.
    stream_live: bool = True
    # How often to print live progress (every N examples). Set to 1 for all.
    stream_every: PositiveInt = 1
    # Write full untruncated details to a log file for later analysis.
    # If set, writes to this path (relative to output_dir or absolute).
    log_file: str | None = "accuracy_log.txt"


class GenerationBenchmarkConfig(BaseModel):
    """Assess text generation quality.

    Runs generation on curated prompts to qualitatively evaluate
    the model's output quality.
    """

    type: Literal[BenchmarkType.GENERATION] = BenchmarkType.GENERATION
    prompts_file: str
    max_new_tokens: PositiveInt = 256
    temperature: PositiveFloat = 1.0
    top_p: Probability = 1.0
    repetition_penalty: PositiveFloat = 1.0


class BehaviorBenchmarkConfig(BaseModel):
    """Run the unified YAML-driven behavior suite (single source of truth).

    This replaces the legacy split between `behavior` and `behavioral_v2` by
    providing one benchmark that:
    - materializes a fixed number of cases per category (default 30)
    - randomizes/shuffles deterministically (seeded) to reduce bias
    - supports both greedy generation and choice-logprob evaluation
    - produces complete artifacts (raw logs + tables + plots + attention dumps)

    IMPORTANT: The suite is intentionally "fail fast": if the YAML spec is
    malformed or expectations cannot be met (e.g. missing pools, target spans),
    execution should raise and stop.
    """

    type: Literal[BenchmarkType.BEHAVIOR] = BenchmarkType.BEHAVIOR
    tokenizer: TokenizerConfig
    # The unified suite spec (YAML). This file defines categories + templates.
    suite_file: str = "benchmark/behavior/cases.yml"
    # REQUIRED: seed for deterministic randomization across models.
    seed: PositiveInt

    max_new_tokens: PositiveInt = 32
    context_window: PositiveInt | None = None
    # Optional: print per-case outputs to console for debugging/insight.
    print_outputs: bool = False
    # If true, only print cases where either model is wrong or they disagree.
    print_only_failures: bool = True
    # Truncate printed outputs to keep logs readable.
    print_max_chars: PositiveInt = 160
    # Stream each case to console in real-time as it's evaluated.
    # Shows prompt, teacher/student outputs, and pass/fail status live.
    stream_live: bool = True
    # Write full untruncated details to a log file for later analysis.
    # If set, writes to this path (relative to output_dir or absolute).
    log_file: str | None = "behavior_log.txt"

    # ---- Optional attention introspection (paper/debug) ----
    # If true, run an additional forward pass on selected prompts with a viz ctx
    # and dump small attention matrices + summary stats to output_dir.
    dump_attention: bool = False
    # Which case IDs to dump. If empty/None, dumps only cases where either model is wrong.
    dump_attention_case_ids: list[str] | None = None
    # Downsample controls (these bounds apply per attention layer).
    dump_attention_max_tokens: PositiveInt = 96
    dump_attention_max_heads: PositiveInt = 4
    # Substring used to split exemplar vs target regions (for mass metrics).
    # For the copy probes, "A7" is a stable anchor.
    dump_attention_anchor: str = "A7"
    # Optional: also copy rendered PNGs into a stable "paper figures" directory.
    # This makes paper.tex inclusion deterministic (no timestamped run dirs).
    dump_attention_paper_dir: str | None = None
    # Tag used in filenames when copying to paper dir (e.g. "dba_decoupled" or "dba_sem8geo32v40").
    dump_attention_paper_tag: str | None = None


class BehaviorInstructBenchmarkConfig(BaseModel):
    """Instruction-formatted behavioral suite (generation-only).

    Renders each prompt exactly as:
        User: <instruction>

        Assistant:

    It reuses the v2 template generator to keep the benchmark randomized
    (slot-based) and reduce prompt-specific bias, but always evaluates via
    free generation (no logprob-only choice scoring).
    """

    type: Literal[BenchmarkType.BEHAVIOR_INSTRUCT] = BenchmarkType.BEHAVIOR_INSTRUCT
    tokenizer: TokenizerConfig = Field(default_factory=lambda: TiktokenTokenizerConfig(encoding="gpt2"))

    # Test generation settings (v2 suite generator)
    seed: PositiveInt
    tests_per_category: PositiveInt = 30
    category_counts: dict[str, int] | None = None
    categories: list[str] | None = None
    # Optional finer filter within a category.
    subcategories: list[str] | None = None
    # If true, allow v2 template metadata to override generation settings per test
    # (e.g. adversarial dread induction wants repetition_penalty > 1).
    honor_recommended_settings: bool = False

    # Generation settings
    max_new_tokens: PositiveInt = 32
    context_window: PositiveInt | None = None
    repetition_penalty: PositiveFloat = 1.0

    # Output settings
    stream_live: bool = True
    stream_every: PositiveInt = 10
    log_file: str | None = "behavior_instruct_log.txt"
    # Raw transcript log (exact prompt + decoded model output, no scoring/stripping).
    transcript_file: str | None = "behavior_instruct_transcript.txt"
    # Optional: write degeneration/chaos diagnostics (token/word repetition stats).
    degeneration_metrics_file: str | None = "behavior_instruct_degeneration.csv"

    # ---- Optional attention introspection (paper/debug) ----
    # If true, run additional forward passes on selected prompts with a viz ctx
    # and dump small attention matrices + summary stats to output_dir.
    dump_attention: bool = False
    # Which test IDs to dump. If empty/None, dumps all tests (can be large).
    dump_attention_case_ids: list[str] | None = None
    # Downsample controls (these bounds apply per attention layer).
    dump_attention_max_tokens: PositiveInt = 96
    dump_attention_max_heads: PositiveInt = 4
    # Substring used to split exemplar vs target regions (for mass metrics).
    # For instruction-style prompts, "Assistant:" is a stable anchor.
    dump_attention_anchor: str = "Assistant:"
    # Optional: also copy rendered PNGs into a stable "paper figures" directory.
    dump_attention_paper_dir: str | None = None
    # Tag used in filenames when copying to paper dir (e.g. "dba_decoupled").
    dump_attention_paper_tag: str | None = None


class ContextBenchmarkConfig(BaseModel):
    """Long-context stability + decode-at-context throughput.

    Uses KV caches with chunked prefill to reach contexts beyond the training
    block size. Intended for the DBA paper's "stability up to 128k" claim and
    decode-at-context throughput curves.
    """

    type: Literal[BenchmarkType.CONTEXT] = BenchmarkType.CONTEXT
    # REQUIRED dataset used to source a deterministic token prefix.
    # This benchmark intentionally does not fall back to random tokens: without a
    # manifest-specified dataset, the benchmark definition is incomplete.
    dataset: str
    # Context lengths to test.
    context_lengths: list[PositiveInt] = Field(
        default_factory=lambda: [2048, 4096, 8192, 16384, 32768]
    )
    # Chunk size for prefill. Actual chunk size may be reduced dynamically to
    # keep attention mask materialization bounded.
    chunk_size: PositiveInt = 1024
    # Upper bound on (t_q * t_k) for any attention mask block (conservative bound).
    max_mask_elems: PositiveInt = 16_000_000
    batch_size: PositiveInt = 1
    # Decode benchmark after prefill.
    decode_len: PositiveInt = 128
    decode_warmup: PositiveInt = 8
    # KV cache config for the benchmark run.
    cache_kind: str = "fp16"
    # Optional explicit KV-cache policy (enables heterogeneous DBA caching).
    # If provided, it overrides cache_kind/qblock/residual_len behavior in the generator.
    cache_policy: KVCacheConfig | None = None
    # Optional "effective" vocab size (tokenizer vocab) used for:
    # - random token fallback (if dataset is missing)
    # - masking logits for loss/ppl
    valid_vocab_size: PositiveInt | None = None


# Union of all benchmark config types
BenchmarkConfig = (
    PerplexityBenchmarkConfig
    | LatencyBenchmarkConfig
    | MemoryBenchmarkConfig
    | AccuracyBenchmarkConfig
    | GenerationBenchmarkConfig
    | BehaviorBenchmarkConfig
    | BehaviorInstructBenchmarkConfig
    | ContextBenchmarkConfig
)


class BenchmarkSpec(BaseModel):
    """Specification for running a benchmark.

    Wraps a benchmark config with metadata like which models to test
    and how many times to repeat for statistical confidence.
    """

    id: str
    config: BenchmarkConfig = Field(discriminator="type")
    # If true, show live matplotlib plots for "raw" benchmark metrics while running,
    # and save those plots to disk at the end of each benchmark.
    realtime: bool = False
    models: list[str] = Field(
        default_factory=lambda: ["teacher", "student"],
        description="Which models to benchmark",
    )
    repeats: PositiveInt = 1


class BenchmarkSuite(BaseModel):
    """Complete benchmark suite for an experiment.

    Collects multiple benchmarks and configures output formats.
    """

    benchmarks: list[BenchmarkSpec]
    output_dir: str = "artifacts"
    formats: list[str] = Field(
        default_factory=lambda: ["csv", "json", "png", "latex"],
        description="Output formats: csv, json, png, latex",
    )
    comparison_baseline: str | None = "teacher"
