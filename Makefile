.PHONY: build test generate clean chat image research serve

DUMP ?= caramba.txt

# The pool package uses go:linkname to access runtime scheduling
# primitives (dropg, readgstatus) for zero-overhead goroutine parking.
# Go 1.26 restricts these by default; -checklinkname=0 preserves access.
LDFLAGS := -ldflags='-checklinkname=0'

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
	go build $(LDFLAGS) .

test:
	go test $(LDFLAGS) ./...

generate:
	go generate $(LDFLAGS) ./...

clean:
	go clean $(LDFLAGS) ./...

chat:
	go run $(LDFLAGS) . chat

image:
	go run $(LDFLAGS) . image "An elephant playing chess"

research:
	go run $(LDFLAGS) . research

serve:
	go run $(LDFLAGS) . serve