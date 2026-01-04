.PHONY: install install-all test paper discussion brainstorm brainstorm-full ingest platform tui mosaic-smoke

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