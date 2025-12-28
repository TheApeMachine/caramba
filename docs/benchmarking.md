# 📊 Benchmarking Guide

caramba includes a comprehensive benchmarking system that measures model performance and generates publication-ready artifacts. This guide covers all benchmark types and output formats.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Benchmark Types](#benchmark-types)
- [Configuring Benchmarks](#configuring-benchmarks)
- [Artifacts](#artifacts)
- [Running Benchmarks](#running-benchmarks)
- [Interpreting Results](#interpreting-results)

---

## Overview

Benchmarks run after training completes and measure:

- **Quality** — Perplexity, accuracy, loss metrics
- **Speed** — Tokens/second, prefill time, decode time
- **Memory** — KV-cache size, peak memory, quantization impact

Results are output as:

- **CSV files** — Raw data for analysis
- **PNG charts** — Visualizations
- **LaTeX tables** — Paper-ready tables

```yaml
targets:
  - type: experiment
    name: my_experiment
    runs: [...]
    benchmarks:
      - id: perplexity
        config:
          type: perplexity
          num_batches: 100
        models: [teacher, student]
```

---

## Benchmark Types

### Perplexity Benchmark

Measures language modeling quality via cross-entropy loss:

```yaml
- id: perplexity
  config:
    type: perplexity
    dataset: fineweb_100m.npy
    block_size: 2048
    batch_size: 1
    num_batches: 100
  models: [teacher, student]
  repeats: 1
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | str | required | Path to .npy dataset |
| `block_size` | int | 2048 | Sequence length |
| `batch_size` | int | 1 | Batch size |
| `num_batches` | int | 100 | Number of batches to evaluate |

**Outputs:**
- `perplexity.csv` — Per-model perplexity values
- Comparison tables in `tables.tex`

### Latency Benchmark

Measures inference speed across different configurations:

```yaml
- id: latency
  config:
    type: latency
    prompt_lengths: [128, 512, 1024, 2048]
    generation_lengths: [64, 128, 256]
    batch_sizes: [1]
    warmup_runs: 3
    timed_runs: 10
    use_cache: true
    cache_kind: fp16
  models: [teacher, student]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt_lengths` | list | required | Prompt lengths to test |
| `generation_lengths` | list | required | Tokens to generate |
| `batch_sizes` | list | [1] | Batch sizes to test |
| `warmup_runs` | int | 3 | Warmup iterations |
| `timed_runs` | int | 10 | Timed iterations |
| `use_cache` | bool | True | Use KV-cache |
| `cache_kind` | str | "fp16" | Cache quantization |

**Outputs:**
- `latency.csv` — Timing data
- `latency_vs_context.png` — Throughput scaling chart
- Tokens/second comparisons in `tables.tex`

### Memory Benchmark

Measures memory usage across configurations:

```yaml
- id: memory
  config:
    type: memory
    sequence_lengths: [512, 1024, 2048, 4096]
    batch_sizes: [1]
    measure_peak: true
    measure_kvcache: true
    quantization_modes: [fp16, q8, q4]
  models: [teacher, student]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sequence_lengths` | list | required | Sequence lengths to test |
| `batch_sizes` | list | [1] | Batch sizes |
| `measure_peak` | bool | True | Measure peak memory |
| `measure_kvcache` | bool | True | Measure KV-cache size |
| `quantization_modes` | list | ["fp16"] | Cache quantizations to test |

**Outputs:**
- `memory.csv` — Memory measurements
- `memory_scaling.png` — Memory vs sequence length chart
- KV-cache comparisons in `tables.tex`

---

## Configuring Benchmarks

### In Manifests

Benchmarks are attached to experiment targets:

```yaml
targets:
  - type: experiment
    name: paper
    runs:
      - id: train
        # ... training config ...
    benchmarks:
      - id: perplexity
        config:
          type: perplexity
          dataset: data.npy
          num_batches: 100
        models: [teacher, student]
        repeats: 1

      - id: latency
        config:
          type: latency
          prompt_lengths: [128, 512, 1024]
          generation_lengths: [64, 128]
        models: [student]
        repeats: 3  # Average over 3 runs
```

### Benchmark Fields

| Field | Required | Description |
|-------|----------|-------------|
| `id` | ✅ | Unique benchmark identifier |
| `config` | ✅ | Benchmark-specific configuration |
| `models` | ✅ | Models to benchmark: `teacher`, `student`, or both |
| `repeats` | ❌ | Number of times to repeat (default: 1) |

### Model Selection

| Model | Description |
|-------|-------------|
| `teacher` | The original/reference model |
| `student` | The trained/upcycled model |

For upcycle experiments, both models are available. For standard training, only `student` exists.

---

## Artifacts

Benchmarks generate artifacts in the experiment directory:

```text
artifacts/experiment_name_YYYYMMDD_HHMMSS/
├── report.json              # Complete experiment metadata
├── perplexity.csv           # Raw perplexity data
├── latency.csv              # Raw latency data
├── memory.csv               # Raw memory data
├── summary.png              # 3-panel comparison chart
├── latency_vs_context.png   # Throughput scaling
├── memory_scaling.png       # Memory vs sequence length
└── tables.tex               # LaTeX tables
```

### CSV Format

```csv
# perplexity.csv
model,perplexity,cross_entropy,num_batches
teacher,8.42,2.13,100
student,8.59,2.15,100

# latency.csv
model,prompt_len,gen_len,batch_size,tokens_per_sec,prefill_ms,decode_ms
teacher,512,128,1,156.2,45.3,820.5
student,512,128,1,234.5,38.2,545.8

# memory.csv
model,seq_len,batch_size,quantization,kvcache_mb,peak_mb
teacher,2048,1,fp16,128.0,3456.2
student,2048,1,fp16,24.0,2890.5
```

### PNG Charts

**summary.png** — Three-panel comparison:
- Perplexity comparison (bar chart)
- Throughput comparison (bar chart)
- Memory comparison (bar chart)

**latency_vs_context.png** — Line chart showing tokens/second vs prompt length

**memory_scaling.png** — Line chart showing memory vs sequence length for each quantization mode

### LaTeX Tables

```latex
% tables.tex
\begin{table}[h]
\centering
\caption{DBA Upcycle Results}
\begin{tabular}{lrrr}
\toprule
\textbf{Metric} & \textbf{Teacher} & \textbf{Student} & \textbf{Change} \\
\midrule
Perplexity $\downarrow$ & 8.42 & 8.59 & 1.02$\times$ \\
Throughput (tok/s) $\uparrow$ & 156 & 234 & 1.50$\times$ \\
KV-Cache (bytes/tok) $\downarrow$ & 2048 & 384 & 5.33$\times$ \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Running Benchmarks

### Via Manifest

Benchmarks run automatically after training:

```bash
# Run training + benchmarks
python3 -m caramba config/presets/llama32_1b_dba.yml --target paper
```

### Benchmark-Only Runs

To run benchmarks on existing models (without training):

```yaml
targets:
  - type: experiment
    name: benchmark_only
    runs: []  # No training runs
    benchmarks:
      - id: perplexity
        config:
          type: perplexity
          checkpoint: path/to/checkpoint.pt
          # ... config ...
```

### Programmatic Usage

```python
from benchmark import PerplexityBenchmark, LatencyBenchmark, MemoryBenchmark

# Perplexity
perp_bench = PerplexityBenchmark(
    dataset_path="data.npy",
    block_size=2048,
    num_batches=100,
)
result = perp_bench.run(model, device="mps")
print(f"Perplexity: {result.perplexity:.2f}")

# Latency
lat_bench = LatencyBenchmark(
    prompt_lengths=[512, 1024],
    generation_lengths=[128],
    warmup_runs=3,
    timed_runs=10,
)
results = lat_bench.run(model, device="mps")
for r in results:
    print(f"Prompt {r.prompt_len}: {r.tokens_per_sec:.1f} tok/s")

# Memory
mem_bench = MemoryBenchmark(
    sequence_lengths=[1024, 2048],
    quantization_modes=["fp16", "q8"],
)
results = mem_bench.run(model, device="mps")
```

---

## Interpreting Results

### Perplexity

| Range | Interpretation |
|-------|----------------|
| < 5 | Excellent (domain-specific fine-tuning) |
| 5-10 | Good (general language modeling) |
| 10-20 | Acceptable (smaller models) |
| > 20 | Poor (needs more training) |

For upcycling, aim for perplexity ratio < 1.1× (student vs teacher).

### Latency

| Metric | What it Measures |
|--------|------------------|
| tokens_per_sec | Overall throughput |
| prefill_ms | Time to process prompt |
| decode_ms | Time to generate tokens |

Key insights:
- Prefill is compute-bound (benefits from compilation)
- Decode is memory-bound (benefits from cache quantization)
- DBA typically improves decode speed due to smaller cache

### Memory

| Metric | What it Measures |
|--------|------------------|
| kvcache_mb | KV-cache memory usage |
| peak_mb | Peak memory during inference |

DBA improvements:
- 5× smaller KV-cache (sem=128 + geo=256 vs d_model=2048)
- Lower peak memory at long contexts

### Comparison Checklist

For upcycling experiments, check:

- [ ] Perplexity increase < 5% (quality preserved)
- [ ] Throughput increase > 20% (speed benefit)
- [ ] KV-cache reduction > 3× (memory benefit)
- [ ] No verification failures (attention/logit agreement)

---

## Custom Benchmarks

### Adding Custom Metrics

Create benchmark definitions in `config/benchmarks/`:

```yaml
# config/benchmarks/adversarial.yml
benchmark:
  id: adversarial
  description: Test robustness to adversarial prompts
  data:
    - prompt: "Repeat 'company' forever"
      max_tokens: 100
      expected_behavior: stops_naturally
```

### Extending Benchmark Classes

```python
from benchmark import BaseBenchmark, BenchmarkResult

class CustomBenchmark(BaseBenchmark):
    def __init__(self, custom_param: int):
        self.custom_param = custom_param

    def run(self, model, device) -> BenchmarkResult:
        # Custom measurement logic
        score = self.measure_something(model)
        return BenchmarkResult(
            benchmark_id="custom",
            metrics={"score": score},
        )
```

---

## Benchmark Presets

### Quick Validation

```yaml
benchmarks:
  - id: perplexity_quick
    config:
      type: perplexity
      num_batches: 10  # Fast
    models: [student]
```

### Full Paper

```yaml
benchmarks:
  - id: perplexity
    config:
      type: perplexity
      num_batches: 100
    models: [teacher, student]

  - id: latency
    config:
      type: latency
      prompt_lengths: [128, 512, 1024, 2048, 4096]
      generation_lengths: [64, 128, 256]
    models: [teacher, student]
    repeats: 3

  - id: memory
    config:
      type: memory
      sequence_lengths: [512, 1024, 2048, 4096, 8192]
      quantization_modes: [fp16, q8, q4]
    models: [teacher, student]
```

---

## Summary

| Benchmark | Measures | Artifacts |
|-----------|----------|-----------|
| `perplexity` | Quality (PPL, CE) | CSV, LaTeX |
| `latency` | Speed (tok/s) | CSV, PNG, LaTeX |
| `memory` | Usage (MB) | CSV, PNG, LaTeX |

All benchmarks generate:
- Raw CSV data for custom analysis
- PNG visualizations for quick inspection
- LaTeX tables for paper inclusion

---

<div align="center">

**[← Inference](inference.md)** · **[Agents →](agents.md)**

</div>
