# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Commands

### Installation
```bash
pip install -r requirements.txt
```

### Running Experiments
```bash
# Run full experiment (upcycle + benchmarks + artifacts)
python3 -m caramba run caramba/config/presets/llama32_1b_dba.yml --group paper

# Quick validation run
python3 -m caramba run caramba/config/presets/llama32_1b_dba.yml --group quick

# Compile only (with optional plan output)
python3 -m caramba compile caramba/config/presets/llama32_1b_dba.yml --print-plan
```

### Testing
```bash
# Run everything (repo-wide)
python -m pytest -q

# Run only caramba unit tests
python -m pytest caramba -q

# Run with coverage
coverage run -m pytest
coverage report -m
```

## Code Architecture

The project is a framework for neural network architecture research. The core idea is to define experiments using YAML manifests, which are then compiled and run.

- **`config/`**: Contains typed configuration models and preset manifests for experiments.
- **`compiler/`**: Handles parsing, lowering (variable substitution), and validation of manifests.
- **`topology/`**: Defines how layers are composed (e.g., `StackedTopology`, `ResidualTopology`). These are the building blocks of the neural network graph.
- **`layer/`**: Contains the actual neural network layers as thin PyTorch modules (e.g., `AttentionLayer`, `RMSNormLayer`).
- **`model/`**: Responsible for building the full model from the defined topology and layers.
- **`trainer/`**: Manages the training process, including blockwise distillation and upcycling.
- **`infer/`**: Handles inference, including standard and speculative decoding.
- **`benchmark/`**: Provides tools for measuring perplexity, latency, and memory.
- **`experiment/`**: Orchestrates the entire experiment pipeline.
