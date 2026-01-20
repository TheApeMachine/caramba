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

from config import PositiveFloat, PositiveInt, Probability
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
    BEHAVIORAL_V2 = "behavioral_v2"
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
    num_fewshot: PositiveInt = 0
    limit: PositiveInt | None = None
    # Optional sliding context window for scoring long prompts.
    context_window: PositiveInt | None = None
    # Optional: print a small number of examples per task for insight.
    # This does not affect scoring, only console output.
    print_examples: PositiveInt = 0
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
    """Run a small behavioral prompt suite against teacher/student.

    This is intended for paper workflows where we want deterministic,
    structured sanity checks (e.g. copy/format fidelity, simple arithmetic)
    in addition to perplexity/latency/memory.
    """

    type: Literal[BenchmarkType.BEHAVIOR] = BenchmarkType.BEHAVIOR
    tokenizer: TokenizerConfig
    cases_file: str
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


class BehavioralV2BenchmarkConfig(BaseModel):
    """Run the v2 behavioral test suite with template-based generation.

    This is the expanded behavioral evaluation framework supporting:
    - 500+ parameterized test cases across 18 categories
    - Template-based generation with randomization
    - Teacher vs student comparison with match quality tracking
    - Optional downstream HF tasks (winogrande, arc_easy, etc.)
    """

    type: Literal[BenchmarkType.BEHAVIORAL_V2] = BenchmarkType.BEHAVIORAL_V2
    tokenizer: TokenizerConfig = Field(default_factory=lambda: TiktokenTokenizerConfig(encoding="gpt2"))

    # Test generation settings
    seed: PositiveInt = 42
    tests_per_category: PositiveInt = 30
    # Override counts for specific categories (e.g., {"copy_tasks": 50, "reasoning": 20})
    category_counts: dict[str, int] | None = None
    # Categories to include (None = all). Use to focus on specific test types.
    categories: list[str] | None = None

    # Downstream HF tasks (optional, integrated with behavioral suite)
    # Available: winogrande, arc_easy, arc_challenge, hellaswag, piqa, boolq
    downstream_tasks: list[str] | None = None
    downstream_limit: PositiveInt | None = None  # Limit samples per downstream task

    # Generation settings
    max_new_tokens: PositiveInt = 32
    context_window: PositiveInt | None = None

    # Output settings
    stream_live: bool = True
    stream_every: PositiveInt = 10
    log_file: str | None = "behavioral_v2_log.txt"


class ContextBenchmarkConfig(BaseModel):
    """Long-context stability + decode-at-context throughput.

    Uses KV caches with chunked prefill to reach contexts beyond the training
    block size. Intended for the DBA paper's "stability up to 128k" claim and
    decode-at-context throughput curves.
    """

    type: Literal[BenchmarkType.CONTEXT] = BenchmarkType.CONTEXT
    # Optional dataset used only to source a deterministic token prefix.
    # If missing/unavailable, falls back to random tokens.
    dataset: str | None = None
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
    | BehavioralV2BenchmarkConfig
    | ContextBenchmarkConfig
)


class BenchmarkSpec(BaseModel):
    """Specification for running a benchmark.

    Wraps a benchmark config with metadata like which models to test
    and how many times to repeat for statistical confidence.
    """

    id: str
    config: BenchmarkConfig = Field(discriminator="type")
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
