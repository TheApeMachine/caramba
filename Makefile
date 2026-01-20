.PHONY: install install-all test paper discussion brainstorm brainstorm-full ingest platform tui mosaic-smoke benchmark local surgery distill dual prepare-llama-data convert-safetensors verify-surgery infer colab-dispatch colab-dispatch-quick colab-notebook benchmark10k benchmark100k benchmark_ablations benchmark_gated

install:
	python3.12 -m venv .venv
	. .venv/bin/activate && uv sync

install-all:
	python3.12 -m venv .venv
	. .venv/bin/activate && python3.12 -m pip install -U pip setuptools wheel
	. .venv/bin/activate && python3.12 -m pip install -c constraints.txt -e "."

test:
	. .venv/bin/activate && python3.12 -m pytest -q

coverage:
	. .venv/bin/activate \
	&& coverage run --source=caramba -m pytest \
	&& coverage report -m --ignore-errors

paper:
	. .venv/bin/activate \
	&& python3.12 -m caramba run config/presets/llama32_1b_dba_paper.yml

local:
	. .venv/bin/activate \
	&& python3.12 -m caramba run config/presets/dba_paper_local.yml

surgery:
	. .venv/bin/activate \
	&& python3.12 -m caramba run config/presets/llama32_1b_dba_routing_hypothesis.yml --target exp_routing_fresh

distill:
	. .venv/bin/activate \
	&& python3.12 -m caramba run config/presets/llama32_1b_dba_attention_distillation.yml --target exp_attention_distill

dual:
	. .venv/bin/activate \
	&& python3.12 -m caramba run config/presets/llama32_1b_dba_attention_distillation.yml --target exp_dual_attention

# Prepare Llama-tokenized FineWeb dataset (1B tokens by default)
prepare-llama-data:
	. .venv/bin/activate \
	&& python3.12 scripts/prepare_fineweb_llama.py --tokens 1B --output artifacts/datasets/fineweb_llama

# Prepare larger dataset (10B tokens)
prepare-llama-data-10b:
	. .venv/bin/activate \
	&& python3.12 scripts/prepare_fineweb_llama.py --tokens 10B --output artifacts/datasets/fineweb_llama_10b

# Convert DBA checkpoint to SafeTensors format
# Usage: make convert-safetensors CHECKPOINT=checkpoints/checkpoint_5000.npz OUTPUT=models/dba.safetensors
convert-safetensors:
	. .venv/bin/activate \
	&& python3.12 scripts/convert_to_safetensors.py \
		--checkpoint $(CHECKPOINT) \
		--output $(OUTPUT)

# Export only DBA attention weights (not merged with Llama)
convert-dba-only:
	. .venv/bin/activate \
	&& python3.12 scripts/convert_to_safetensors.py \
		--checkpoint $(CHECKPOINT) \
		--output $(OUTPUT) \
		--dba-only

# Verify surgery correctly loads FFN/embedding weights
verify-surgery:
	. .venv/bin/activate \
	&& python3.12 scripts/verify_surgery.py

# Run inference with trained DBA model
# Usage: make infer CHECKPOINT=checkpoints/checkpoint_5000.npz PROMPT="Hello world"
# Add COMPARE=1 to compare teacher vs student output
infer:
	. .venv/bin/activate \
	&& python3.12 scripts/infer_mlx.py \
		--checkpoint $(CHECKPOINT) \
		$(if $(PROMPT),--prompt "$(PROMPT)",)

# Fine-tune only DBA semantic path (q_sem/k_sem and optionally gate) starting from
# a behavior-preserving retrofit of pretrained Llama weights.
# Usage:
#   make finetune-semantic DATA=artifacts/datasets/fineweb_llama/train.npy STEPS=1000
finetune-semantic:
	. .venv/bin/activate \
	&& python3.12 scripts/finetune_semantic_mlx.py \
		--data $(DATA) \
		$(if $(TEACHER_WEIGHTS),--teacher-weights $(TEACHER_WEIGHTS),) \
		--max-steps $(if $(STEPS),$(STEPS),1000) \
		--block-size $(if $(BLOCK_SIZE),$(BLOCK_SIZE),1024) \
		--batch-size $(if $(BATCH_SIZE),$(BATCH_SIZE),1) \
		--lr $(if $(LR),$(LR),1e-4) \
		--sem-head-dim $(if $(SEM_HEAD_DIM),$(SEM_HEAD_DIM),8)

real:
	. .venv/bin/activate \
	&& python3.12 scripts/real_version.py \
		$(CHECKPOINT) \
		tokenizer.model

benchmark10k:
	. .venv/bin/activate \
	&& python3.12 -m caramba research/dba/benchmark10k.yml

benchmark100k:
	. .venv/bin/activate \
	&& python3.12 -m caramba research/dba/benchmark100k.yml

benchmark_ablations:
	. .venv/bin/activate \
	&& python3.12 -m caramba research/dba/benchmark_ablations.yml

benchmark_gated:
	. .venv/bin/activate \
	&& python3.12 -m caramba research/dba/benchmark-gated.yml

benchmark:
	@echo "Running behavioral benchmark..."
	@echo "NOTE: For mock testing, use 'make benchmark-mock' instead"
	cd research/dba \
	&& PYTHONPATH=$(CURDIR) python3 -m behavioral_suite_v2.multi_checkpoint_eval \
		--checkpoints-dir 10k_runs \
		--output-dir behavioral_results \
		--tests-per-category 30 \
		--seed 42 \
		--verbose

# Run benchmark with mock models (for testing pipeline without Python 3.12+)
benchmark-mock:
	cd research/dba \
	&& python3 -m behavioral_suite_v2.multi_checkpoint_eval \
		--checkpoints-dir 10k_runs \
		--output-dir behavioral_results_mock \
		--tests-per-category 30 \
		--seed 42 \
		--device cpu \
		--use-mock \
		--verbose \
		--no-browser

resume:
	. .venv/bin/activate \
	&& python3.12 -m caramba run \
	--benchmarks-only \
	--resume-from runs/paper/finetune_global_final.pt \
	config/presets/llama32_1b_dba_paper.yml \
	--group paper

fixed:
	. .venv/bin/activate \
	&& python3.12 -m caramba run config/presets/llama32_1b_dba_paper_efficiency.yml --group paper

aggressive:
	. .venv/bin/activate \
	&& python3.12 -m caramba run config/presets/llama32_1b_dba_paper_efficiency_aggressive.yml --group paper

# Multi-agent discussion process
discussion:
	. .venv/bin/activate \
	&& python3.12 -m caramba brainstorm config/presets/llama32_1b_dba_paper.yml \
	--topic "$(topic)"

# Basic brainstorm with web search and reasoning (no graph)
brainstorm:
	. .venv/bin/activate \
	&& python3.12 -m caramba brainstorm config/presets/llama32_1b_dba_paper.yml \
	--topic "How can we improve KV-cache compression efficiency?"

# Full brainstorm with FalkorDB graph memory (requires: docker run -p 6379:6379 falkordb/falkordb)
brainstorm-full:
	. .venv/bin/activate \
	&& python3.12 -m caramba brainstorm config/presets/llama32_1b_dba_paper.yml \
	--topic "How can we improve KV-cache compression efficiency?" \
	--falkordb-host localhost \
	--falkordb-port 6379

platform:
	. .venv/bin/activate \
	&& python3.12 -m caramba run config/presets/platform_improve.yml

# First MOSAIC end-to-end smoke run (CPU by default).
mosaic-smoke:
	. .venv/bin/activate \
	&& python3.12 -m caramba run config/presets/mosaic_memory_smoke.yml --target mosaic_smoke

# Ingest the caramba codebase into FalkorDB graph (requires: make falkordb first)
# Note: .gitignore patterns are automatically respected
ingest:
	. .venv/bin/activate \
	&& python3.12 -m caramba ingest . \
	--falkordb-host localhost \
	--falkordb-port 6379 \
	--include "**/*.py"

tui:
	. .venv/bin/activate \
	&& ROOT_AGENT_URL=http://localhost:9000 python3.12 -m caramba.tui.app

# ============================================================================
# COLAB DISPATCHER - Run DBA benchmarks on Google Colab's free GPU
# ============================================================================
#
# First time setup:
#   make colab-install
#
# Usage:
#   make colab-dispatch COLAB_CHECKPOINT_DIR="/DBA/checkpoints/100k"
#   make colab-dispatch-quick COLAB_CHECKPOINT_DIR="/DBA/checkpoints/100k"
#   make colab-notebook COLAB_CHECKPOINT_DIR="/DBA/checkpoints/100k"

COLAB_CHECKPOINT_DIR ?= /DBA/checkpoints/100k
COLAB_RESULTS_DIR ?= /DBA/results
COLAB_TIMEOUT ?= 7200
COLAB_TESTS_PER_CATEGORY ?= 30
COLAB_SEED ?= 42
COLAB_FOLDER_ID ?=

# Install Playwright for Colab automation
colab-install:
	. .venv/bin/activate \
	&& pip install playwright \
	&& python -m playwright install chromium

# Dispatch full benchmark to Google Colab (using MCP API)
# Usage: make colab-dispatch COLAB_CHECKPOINT_DIR="/DBA/checkpoints/100k"
# Optional: COLAB_FOLDER_ID="15OkxtAVtKPcfh768CjlGm5zucqiR5SUT" to save to specific folder
colab-dispatch:
	. .venv/bin/activate \
	&& python research/dba/colab_runner/dispatch.py \
		--checkpoint-dir "$(COLAB_CHECKPOINT_DIR)" \
		--results-dir "$(COLAB_RESULTS_DIR)" \
		--tests-per-category $(COLAB_TESTS_PER_CATEGORY) \
		--seed $(COLAB_SEED) \
		--timeout $(COLAB_TIMEOUT) \
		$(if $(COLAB_FOLDER_ID),--folder-id "$(COLAB_FOLDER_ID)",)

# Dispatch quick benchmark (5 tests per category)
colab-dispatch-quick:
	. .venv/bin/activate \
	&& python research/dba/colab_runner/dispatch.py \
		--checkpoint-dir "$(COLAB_CHECKPOINT_DIR)" \
		--results-dir "$(COLAB_RESULTS_DIR)" \
		--tests-per-category 5 \
		--seed $(COLAB_SEED) \
		--timeout 1800 \
		$(if $(COLAB_FOLDER_ID),--folder-id "$(COLAB_FOLDER_ID)",)

# Generate Colab notebook only (for manual upload)
colab-notebook:
	. .venv/bin/activate \
	&& python research/dba/colab_runner/dispatch.py \
		--checkpoint-dir "$(COLAB_CHECKPOINT_DIR)" \
		--results-dir "$(COLAB_RESULTS_DIR)" \
		--tests-per-category $(COLAB_TESTS_PER_CATEGORY) \
		--seed $(COLAB_SEED) \
		--notebook-only