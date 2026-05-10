.PHONY: build

build:
	cd pkg/backend/compute/metal \
		&& xcrun -sdk macosx metal -std=metal3.1 -mmacosx-version-min=14.0 -fmodules-cache-path=/tmp/six-metal-module-cache -I. -c activation.metal -o activation.air \
		&& xcrun -sdk macosx metallib activation.air -o activation.metallib \
		&& xcrun -sdk macosx metal -std=metal3.1 -mmacosx-version-min=14.0 -fmodules-cache-path=/tmp/six-metal-module-cache -I. -c attention.metal -o attention.air \
		&& xcrun -sdk macosx metallib attention.air -o attention.metallib \
		&& xcrun -sdk macosx metal -std=metal3.1 -mmacosx-version-min=14.0 -fmodules-cache-path=/tmp/six-metal-module-cache -I. -c math.metal -o math.air \
		&& xcrun -sdk macosx metallib math.air -o math.metallib \
		&& xcrun -sdk macosx metal -std=metal3.1 -mmacosx-version-min=14.0 -fmodules-cache-path=/tmp/six-metal-module-cache -I. -c shape.metal -o shape.air \
		&& xcrun -sdk macosx metallib shape.air -o shape.metallib \
		&& xcrun -sdk macosx metal -std=metal3.1 -mmacosx-version-min=14.0 -fmodules-cache-path=/tmp/six-metal-module-cache -I. -c positional.metal -o positional.air \
		&& xcrun -sdk macosx metallib positional.air -o positional.metallib \
		&& xcrun -sdk macosx metal -std=metal3.1 -mmacosx-version-min=14.0 -fmodules-cache-path=/tmp/six-metal-module-cache -I. -c pooling.metal -o pooling.air \
		&& xcrun -sdk macosx metallib pooling.air -o pooling.metallib \
		&& xcrun -sdk macosx metal -std=metal3.1 -mmacosx-version-min=14.0 -fmodules-cache-path=/tmp/six-metal-module-cache -I. -c convolution.metal -o convolution.air \
		&& xcrun -sdk macosx metallib convolution.air -o convolution.metallib

	@if command -v nvcc >/dev/null 2>&1; then \
		go generate -tags cuda ./pkg/backend/compute/cuda; \
	else \
		echo "Skipping CUDA generation: nvcc not found (run make cuda on a CUDA host)"; \
	fi

	go build .