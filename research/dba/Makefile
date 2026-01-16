# DBA Research Makefile
#
# Run from research/dba directory:
#   cd research/dba && make benchmark
#
# Usage:
#   make benchmark        - Run behavioral evaluation on all 10k_runs checkpoints
#   make benchmark-quick  - Run with reduced test count (5 per category)
#   make benchmark-type TYPE=baseline  - Run on specific model type
#   make clean            - Clean generated results
#   make help             - Show this help

PYTHON ?= python3
CHECKPOINTS_DIR ?= 10k_runs
OUTPUT_DIR ?= behavioral_results
TESTS_PER_CATEGORY ?= 30
SEED ?= 42

# Set PYTHONPATH to caramba root (two levels up from research/dba)
CARAMBA_ROOT ?= $(shell cd ../.. && pwd)
export PYTHONPATH := $(CARAMBA_ROOT):$(PYTHONPATH)

# Detect device
DEVICE ?= $(shell $(PYTHON) -c "import torch; print('cuda' if torch.cuda.is_available() else ('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'))" 2>/dev/null || echo "cpu")

.PHONY: benchmark benchmark-quick benchmark-type clean help test-suite colab-dispatch colab-notebook

help:
	@echo "DBA Research Makefile"
	@echo ""
	@echo "Usage (run from research/dba directory):"
	@echo "  make benchmark           Run behavioral evaluation on all 10k_runs checkpoints"
	@echo "  make benchmark-quick     Run with reduced test count (5 per category)"
	@echo "  make benchmark-type TYPE=baseline   Run on specific model type"
	@echo "  make benchmark-seed FILTER_SEED=1337  Run on specific seed"
	@echo "  make benchmark-100k      Run on 100k step checkpoints"
	@echo "  make test-suite          Generate test suite JSON only (no model eval)"
	@echo "  make clean               Clean generated results"
	@echo ""
	@echo "Colab Dispatch (run on Google Colab's free GPU):"
	@echo "  make colab-dispatch      Dispatch benchmark to Colab"
	@echo "  make colab-dispatch-quick  Dispatch quick benchmark (5 tests/category)"
	@echo "  make colab-notebook      Generate notebook for manual upload"
	@echo ""
	@echo "Configuration:"
	@echo "  CHECKPOINTS_DIR=$(CHECKPOINTS_DIR)"
	@echo "  OUTPUT_DIR=$(OUTPUT_DIR)"
	@echo "  TESTS_PER_CATEGORY=$(TESTS_PER_CATEGORY)"
	@echo "  SEED=$(SEED)"
	@echo "  DEVICE=$(DEVICE)"
	@echo ""
	@echo "Colab Configuration:"
	@echo "  COLAB_CHECKPOINT_DIR=$(COLAB_CHECKPOINT_DIR)"
	@echo "  COLAB_RESULTS_DIR=$(COLAB_RESULTS_DIR)"
	@echo "  COLAB_TIMEOUT=$(COLAB_TIMEOUT)"
	@echo ""
	@echo "Examples:"
	@echo "  make benchmark DEVICE=mps"
	@echo "  make benchmark-quick OUTPUT_DIR=./quick_results"
	@echo "  make benchmark-type TYPE=decoupled"
	@echo "  make colab-dispatch COLAB_CHECKPOINT_DIR=/DBA/checkpoints/100k"

# Generate test suite only (no model evaluation, no torch required)
test-suite:
	@echo "Generating test suite..."
	$(PYTHON) generate_test_suite.py \
		--tests-per-category $(TESTS_PER_CATEGORY) \
		--seed $(SEED) \
		--output $(OUTPUT_DIR)/test_suite.json

# Run full behavioral benchmark on all checkpoints
benchmark:
	@echo "Running behavioral benchmark on all 10k_runs checkpoints..."
	@echo "Device: $(DEVICE)"
	@echo "Checkpoints: $(CHECKPOINTS_DIR)"
	@echo "Output: $(OUTPUT_DIR)"
	@echo ""
	$(PYTHON) -m behavioral_suite_v2.multi_checkpoint_eval \
		--checkpoints-dir $(CHECKPOINTS_DIR) \
		--output-dir $(OUTPUT_DIR) \
		--tests-per-category $(TESTS_PER_CATEGORY) \
		--seed $(SEED) \
		--device $(DEVICE) \
		--verbose

# Quick benchmark with reduced test count
benchmark-quick:
	@echo "Running quick behavioral benchmark (5 tests per category)..."
	$(PYTHON) -m behavioral_suite_v2.multi_checkpoint_eval \
		--checkpoints-dir $(CHECKPOINTS_DIR) \
		--output-dir $(OUTPUT_DIR)_quick \
		--tests-per-category 5 \
		--seed $(SEED) \
		--device $(DEVICE) \
		--verbose

# Benchmark specific model type
benchmark-type:
ifndef TYPE
	$(error TYPE is not set. Usage: make benchmark-type TYPE=baseline)
endif
	@echo "Running behavioral benchmark on $(TYPE) models..."
	$(PYTHON) -m behavioral_suite_v2.multi_checkpoint_eval \
		--checkpoints-dir $(CHECKPOINTS_DIR) \
		--output-dir $(OUTPUT_DIR)_$(TYPE) \
		--tests-per-category $(TESTS_PER_CATEGORY) \
		--seed $(SEED) \
		--device $(DEVICE) \
		--filter-type $(TYPE) \
		--verbose

# Benchmark specific seed
benchmark-seed:
ifndef FILTER_SEED
	$(error FILTER_SEED is not set. Usage: make benchmark-seed FILTER_SEED=1337)
endif
	@echo "Running behavioral benchmark on seed $(FILTER_SEED) models..."
	$(PYTHON) -m behavioral_suite_v2.multi_checkpoint_eval \
		--checkpoints-dir $(CHECKPOINTS_DIR) \
		--output-dir $(OUTPUT_DIR)_s$(FILTER_SEED) \
		--tests-per-category $(TESTS_PER_CATEGORY) \
		--seed $(SEED) \
		--device $(DEVICE) \
		--filter-seed $(FILTER_SEED) \
		--verbose

# Compare baseline vs decoupled only (seed 1337)
benchmark-compare:
	@echo "Running comparison between baseline and decoupled..."
	$(PYTHON) -m behavioral_suite_v2.multi_checkpoint_eval \
		--checkpoints-dir $(CHECKPOINTS_DIR) \
		--output-dir $(OUTPUT_DIR)_compare \
		--tests-per-category $(TESTS_PER_CATEGORY) \
		--seed $(SEED) \
		--device $(DEVICE) \
		--filter-seed 1337 \
		--verbose

# Clean generated results
clean:
	@echo "Cleaning generated results..."
	rm -rf $(OUTPUT_DIR)
	rm -rf $(OUTPUT_DIR)_*
	rm -rf ./results/behavioral_*
	@echo "Done."

# Run with mock models (for testing the pipeline without real models)
benchmark-mock:
	@echo "Running benchmark with mock models..."
	$(PYTHON) -m behavioral_suite_v2.multi_checkpoint_eval \
		--checkpoints-dir $(CHECKPOINTS_DIR) \
		--output-dir $(OUTPUT_DIR)_mock \
		--tests-per-category $(TESTS_PER_CATEGORY) \
		--seed $(SEED) \
		--device cpu \
		--use-mock \
		--verbose

# Run benchmark on 100k checkpoints
benchmark-100k:
	@echo "Running behavioral benchmark on 100k checkpoints..."
	@echo "Device: $(DEVICE)"
	$(PYTHON) -m behavioral_suite_v2.multi_checkpoint_eval \
		--checkpoint-files \
			100k_checkpoints/baseline/a100_fw1b_l22_baseline_s42_100k.pt \
			100k_checkpoints/sem8geo32v40/a100_fw1b_l22_dba_s42_100k.pt \
		--output-dir $(OUTPUT_DIR)_100k \
		--tests-per-category $(TESTS_PER_CATEGORY) \
		--seed $(SEED) \
		--device $(DEVICE) \
		--verbose

# Run benchmark on 100k checkpoints (quick mode)
benchmark-100k-quick:
	@echo "Running quick behavioral benchmark on 100k checkpoints..."
	$(PYTHON) -m behavioral_suite_v2.multi_checkpoint_eval \
		--checkpoint-files \
			100k_checkpoints/baseline/a100_fw1b_l22_baseline_s42_100k.pt \
			100k_checkpoints/sem8geo32v40/a100_fw1b_l22_dba_s42_100k.pt \
		--output-dir $(OUTPUT_DIR)_100k_quick \
		--tests-per-category 5 \
		--seed $(SEED) \
		--device $(DEVICE) \
		--verbose

# ============================================================================
# COLAB DISPATCHER - Run benchmarks on Google Colab's free GPU
# ============================================================================

# Google Drive paths (without /content/drive/MyDrive prefix)
COLAB_CHECKPOINT_DIR ?= /DBA/checkpoints/100k
COLAB_RESULTS_DIR ?= /DBA/results
COLAB_TIMEOUT ?= 7200
COLAB_WEBHOOK ?=

# Dispatch benchmark to Google Colab
colab-dispatch:
ifndef COLAB_CHECKPOINT_DIR
	$(error COLAB_CHECKPOINT_DIR is not set. Usage: make colab-dispatch COLAB_CHECKPOINT_DIR="/DBA/checkpoints/100k")
endif
	@echo "Dispatching benchmark to Google Colab..."
	@echo "Checkpoint dir: $(COLAB_CHECKPOINT_DIR)"
	@echo "Results dir: $(COLAB_RESULTS_DIR)"
	@echo ""
	$(PYTHON) colab_runner/dispatch.py \
		--checkpoint-dir "$(COLAB_CHECKPOINT_DIR)" \
		--results-dir "$(COLAB_RESULTS_DIR)" \
		--tests-per-category $(TESTS_PER_CATEGORY) \
		--seed $(SEED) \
		--timeout $(COLAB_TIMEOUT) \
		$(if $(COLAB_WEBHOOK),--webhook "$(COLAB_WEBHOOK)",)

# Dispatch quick benchmark to Colab (5 tests per category)
colab-dispatch-quick:
	@echo "Dispatching quick benchmark to Google Colab..."
	$(PYTHON) colab_runner/dispatch.py \
		--checkpoint-dir "$(COLAB_CHECKPOINT_DIR)" \
		--results-dir "$(COLAB_RESULTS_DIR)" \
		--tests-per-category 5 \
		--seed $(SEED) \
		--timeout 1800

# Generate Colab notebook only (for manual upload)
colab-notebook:
	@echo "Generating Colab notebook..."
	$(PYTHON) colab_runner/dispatch.py \
		--checkpoint-dir "$(COLAB_CHECKPOINT_DIR)" \
		--results-dir "$(COLAB_RESULTS_DIR)" \
		--tests-per-category $(TESTS_PER_CATEGORY) \
		--seed $(SEED) \
		--notebook-only
