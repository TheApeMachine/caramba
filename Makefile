.PHONY: build dump metal cuda

DUMP ?= caramba.txt

dump:
	python3 "$(CURDIR)/scripts/dump-repo.py" "$(DUMP)"

metal:
	cd pkg/backend/device/metal && go generate

cuda:
	@if command -v nvcc >/dev/null 2>&1; then \
		go generate -tags cuda ./pkg/backend/device/cuda; \
	else \
		echo "Skipping CUDA generation: nvcc not found (run make cuda on a CUDA host)"; \
	fi

build: metal cuda
	go build .
