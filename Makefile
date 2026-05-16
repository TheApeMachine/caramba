.PHONY: build dump

DUMP ?= caramba.txt

dump:
	python3 "$(CURDIR)/scripts/dump-repo.py" "$(DUMP)"

build:
	cd pkg/backend/compute/metal \
	&& xcrun -sdk macosx metal -std=metal3.1 -mmacosx-version-min=14.0 -fmodules-cache-path=/tmp/six-metal-module-cache -I. -c vsa.metal -o vsa.air \
	&& xcrun -sdk macosx metallib vsa.air -o vsa.metallib \
	&& xcrun -sdk macosx metal -std=metal3.1 -mmacosx-version-min=14.0 -fmodules-cache-path=/tmp/six-metal-module-cache -I. -c active_inference.metal -o active_inference.air \
	&& xcrun -sdk macosx metallib active_inference.air -o active_inference.metallib \
	&& xcrun -sdk macosx metal -std=metal3.1 -mmacosx-version-min=14.0 -fmodules-cache-path=/tmp/six-metal-module-cache -I. -c hawkes.metal -o hawkes.air \
	&& xcrun -sdk macosx metallib hawkes.air -o hawkes.metallib \
	&& xcrun -sdk macosx metal -std=metal3.1 -mmacosx-version-min=14.0 -fmodules-cache-path=/tmp/six-metal-module-cache -I. -c markov_blanket.metal -o markov_blanket.air \
	&& xcrun -sdk macosx metallib markov_blanket.air -o markov_blanket.metallib \
	&& xcrun -sdk macosx metal -std=metal3.1 -mmacosx-version-min=14.0 -fmodules-cache-path=/tmp/six-metal-module-cache -I. -c causal.metal -o causal.air \
	&& xcrun -sdk macosx metallib causal.air -o causal.metallib \
	&& xcrun -sdk macosx metal -std=metal3.1 -mmacosx-version-min=14.0 -fmodules-cache-path=/tmp/six-metal-module-cache -I. -c predictive_coding.metal -o predictive_coding.air \
	&& xcrun -sdk macosx metallib predictive_coding.air -o predictive_coding.metallib \
	&& xcrun -sdk macosx metal -std=metal3.1 -mmacosx-version-min=14.0 -fmodules-cache-path=/tmp/six-metal-module-cache -I. -c activation.metal -o activation.air \
	&& xcrun -sdk macosx metallib activation.air -o activation.metallib \
	&& xcrun -sdk macosx metal -std=metal3.1 -mmacosx-version-min=14.0 -fmodules-cache-path=/tmp/six-metal-module-cache -I. -c embedding.metal -o embedding.air \
	&& xcrun -sdk macosx metallib embedding.air -o embedding.metallib \
	&& xcrun -sdk macosx metal -std=metal3.1 -mmacosx-version-min=14.0 -fmodules-cache-path=/tmp/six-metal-module-cache -I. -c attention.metal -o attention.air \
	&& xcrun -sdk macosx metallib attention.air -o attention.metallib \
	&& xcrun -sdk macosx metal -std=metal3.1 -mmacosx-version-min=14.0 -fmodules-cache-path=/tmp/six-metal-module-cache -I. -c masking.metal -o masking.air \
	&& xcrun -sdk macosx metallib masking.air -o masking.metallib \
	&& xcrun -sdk macosx metal -std=metal3.1 -mmacosx-version-min=14.0 -fmodules-cache-path=/tmp/six-metal-module-cache -I. -c math.metal -o math.air \
	&& xcrun -sdk macosx metallib math.air -o math.metallib \
	&& xcrun -sdk macosx metal -std=metal3.1 -mmacosx-version-min=14.0 -fmodules-cache-path=/tmp/six-metal-module-cache -I. -c shape.metal -o shape.air \
	&& xcrun -sdk macosx metallib shape.air -o shape.metallib \
	&& xcrun -sdk macosx metal -std=metal3.1 -mmacosx-version-min=14.0 -fmodules-cache-path=/tmp/six-metal-module-cache -I. -c positional.metal -o positional.air \
	&& xcrun -sdk macosx metallib positional.air -o positional.metallib \
	&& xcrun -sdk macosx metal -std=metal3.1 -mmacosx-version-min=14.0 -fmodules-cache-path=/tmp/six-metal-module-cache -I. -c pooling.metal -o pooling.air \
	&& xcrun -sdk macosx metallib pooling.air -o pooling.metallib \
	&& xcrun -sdk macosx metal -std=metal3.1 -mmacosx-version-min=14.0 -fmodules-cache-path=/tmp/six-metal-module-cache -I. -c convolution.metal -o convolution.air \
	&& xcrun -sdk macosx metallib convolution.air -o convolution.metallib \
	&& xcrun -sdk macosx metal -std=metal3.1 -mmacosx-version-min=14.0 -fmodules-cache-path=/tmp/six-metal-module-cache -I. -c projection.metal -o projection.air \
	&& xcrun -sdk macosx metallib projection.air -o projection.metallib

	@if command -v nvcc >/dev/null 2>&1; then \
		go generate -tags cuda ./pkg/backend/compute/cuda; \
	else \
		echo "Skipping CUDA generation: nvcc not found (run make cuda on a CUDA host)"; \
	fi

	go build .
