# DBA Behavioral Test Suite v2

## Overview

This is a comprehensive behavioral evaluation framework for comparing attention architectures.
The suite provides:

1. **500+ parameterized test cases** across 18 categories
2. **Template-based generation** with randomization
3. **Adversarial prompt attack suite**
4. **Multi-model comparison** (arbitrary number of models)
5. **Rich scoring**: binary, soft, and attention diagnostics
6. **Visualization dashboard** with normalized attention heatmaps

## Architecture

```
behavioral_suite_v2/
├── README.md
├── config.yml                    # Suite configuration
├── templates/                    # Template definitions per category
│   ├── copy_tasks.py
│   ├── fewshot_learning.py
│   ├── distractor_tests.py
│   ├── reasoning.py
│   ├── arithmetic.py
│   ├── sequences.py
│   ├── world_knowledge.py
│   ├── semantic.py
│   ├── format_preservation.py
│   ├── long_context.py
│   ├── robustness.py
│   ├── edge_cases.py
│   ├── attention_probes.py
│   ├── instruction_following.py
│   ├── consistency_checks.py
│   ├── adversarial.py           # Prompt injection attacks
│   ├── binding_tests.py         # Entity-attribute binding
│   └── multi_hop.py             # Multi-hop reasoning
├── generator.py                  # Template instantiation engine
├── runner.py                     # Multi-model evaluation runner
├── scoring.py                    # Scoring framework
├── visualizer.py                 # Attention & results visualization
├── attention_extractor.py        # Hook-based attention capture
└── cli.py                        # Command-line interface
```

## Design Principles

1. **No positional bias**: Copy tokens appear in all positions (start/middle/end)
2. **Randomized parameters**: Each template generates many variants
3. **Stratified sampling**: Equal representation across difficulty levels
4. **Reproducible**: Seeded random generation for deterministic test sets
5. **Extensible**: Easy to add new categories and templates

## Categories

| Category | Tests | Description |
|----------|-------|-------------|
| copy_tasks | 30 | Raw memorization fidelity |
| fewshot_learning | 35 | In-context pattern recognition |
| distractor_tests | 40 | Attention focus under interference |
| reasoning | 30 | Logical inference |
| arithmetic | 25 | Numerical reasoning |
| sequences | 25 | Pattern extrapolation |
| world_knowledge | 20 | Factual recall |
| semantic | 25 | Meaning comprehension |
| format_preservation | 25 | Structural fidelity |
| long_context | 30 | Context window utilization |
| robustness | 20 | Consistency under variation |
| edge_cases | 20 | Boundary conditions |
| attention_probes | 40 | High/low-rank attention diagnostics |
| instruction_following | 25 | Instruction adherence |
| consistency_checks | 20 | Same capability, different format |
| **adversarial** | **50** | Prompt injection resistance |
| **binding_tests** | **30** | Entity-attribute binding |
| **multi_hop** | **25** | Multi-step reasoning chains |

**Total: ~535 tests**

## Scoring System

### Binary Scores
- `exact_match`: Output exactly equals expected
- `content_match`: Expected content present (case-insensitive)
- `prefix_match`: Output starts with expected
- `choice_correct`: For multiple choice, correct option has highest logprob

### Soft Scores (0-3 scale)
- `3`: Exact match
- `2`: Content correct, minor format differences
- `1`: Content present but buried in noise
- `0`: Wrong content or distractor contamination
- `-1`: Radical failure (loops, garbage)

### Diagnostic Flags
- `repetition_loop`: Pathological repetition detected
- `distractor_contamination`: Earlier example content in output
- `format_continuation`: Model continued prompt structure
- `attention_entropy`: Entropy of attention distribution
- `attention_sparsity`: Sparsity of attention pattern

## Visualization Outputs

1. **Attention heatmaps** (shared colorbar, vmin/vmax fixed across models)
2. **Per-category accuracy bar charts**
3. **Head-to-head comparison matrices**
4. **Failure mode breakdown**
5. **Soft score distributions**
6. **Attention pattern comparisons** (teacher vs student per test)
7. **Pareto curves** (perplexity vs behavioral accuracy)

## Usage

### Command Line (Recommended)

The suite uses caramba's manifest system to load models. Point it to your `benchmark.yml`:

```bash
# Run with caramba manifest (loads teacher vs student models)
python research/dba/run_eval.py --manifest research/dba/benchmark.yml

# Customize test count and output directory
python research/dba/run_eval.py --manifest benchmark.yml -n 50 -o ./my_results

# Skip attention capture for faster runs
python research/dba/run_eval.py --manifest benchmark.yml --no-attention

# Use mock models for testing the pipeline (no GPU required)
python research/dba/run_eval.py --use-mock -n 10

# Don't open browser automatically
python research/dba/run_eval.py --manifest benchmark.yml --no-browser

# Add perplexity data for Pareto curve visualization
python research/dba/run_eval.py --manifest benchmark.yml \
    --perplexities '{"teacher": 15.2, "student": 15.8}'

# Full options
python research/dba/run_eval.py --help
```

The CLI will:
1. Load teacher/student models using the manifest configuration
2. Run the full behavioral test suite
3. Generate visualizations and HTML report
4. **Open the results dashboard in your browser**

### Python API

```python
from behavioral_suite_v2 import (
    generate_suite,
    EvalRunner,
    EvalConfig,
)
from behavioral_suite_v2.cli import load_models_from_manifest, CarambaModelWrapper
import tiktoken

# Load tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Load models from manifest
raw_models, metadata = load_models_from_manifest(
    "research/dba/benchmark.yml",
    device="cuda",
)

# Wrap models with evaluable interface
models = {
    name: CarambaModelWrapper(model, tokenizer, metadata["device"])
    for name, model in raw_models.items()
}

# Generate test suite with seed for reproducibility
suite = generate_suite(seed=42, tests_per_category=30)

# Run evaluation
runner = EvalRunner(models, EvalConfig(capture_attention=True))
results = runner.run(suite.tests)

# Save results (includes HTML dashboard)
results.save(Path('./results/'))
```

## Manifest Format

The suite expects a caramba manifest with a `trainer.checkpoint_compare` target:

```yaml
targets:
  - type: experiment
    trainer:
      ref: trainer.checkpoint_compare
      config:
        teacher_ckpt: path/to/teacher.pt
        student_ckpt: path/to/student.pt
        teacher_model:
          type: TransformerModel
          # ... model config
        device: cuda
        dtype: float16
    system:
      ref: system.language_model
      config:
        model:
          type: TransformerModel
          # ... student model config (if different from teacher)
```

See `research/dba/benchmark.yml` for a complete example.
