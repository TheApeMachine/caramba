.PHONY: install install-all test paper brainstorm

install:
	python3 -m venv .venv
	. .venv/bin/activate && pip3 install -e ".[dev]"

install-all:
	python3 -m venv .venv
	. .venv/bin/activate && pip3 install -e ".[all]"

test:
	. .venv/bin/activate && python3 -m pytest -q

paper:
	. .venv/bin/activate && python3 -m caramba paper config/presets/llama32_1b_dba_paper.yml --output-dir artifacts/llama32_1b_dba/paper

brainstorm:
	. .venv/bin/activate \
	&& python3 -m caramba brainstorm config/presets/llama32_1b_dba_paper.yml \
	--topic "How can we improve KV-cache compression efficiency?"
