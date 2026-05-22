.PHONY: build test generate clean chat diffusion diffusion-diagnose image research serve

DUMP ?= caramba.txt

# Metal kernels and metallib generation live in the puter module (go.mod replace).
PUTER_ROOT := ../puter

# The pool package uses go:linkname to access runtime scheduling
# primitives (dropg, readgstatus) for zero-overhead goroutine parking.
# Go 1.26 restricts these by default; -checklinkname=0 preserves access.
LDFLAGS := -ldflags='-checklinkname=0'

dump:
	python3 "$(CURDIR)/scripts/dump-repo.py" "$(DUMP)"

metal:
	cd $(PUTER_ROOT)/device/metal && CGO_ENABLED=1 go generate

cuda:
	@echo "Skipping CUDA generation: device/cuda in puter has no go:generate step"

build: metal
	go build $(LDFLAGS) .

test:
	go test $(LDFLAGS) ./...

generate:
	go generate $(LDFLAGS) ./...

clean:
	go clean $(LDFLAGS) ./...

chat:
	go run $(LDFLAGS) . chat

diffusion:
	go run $(LDFLAGS) . diffusion "An elephant playing chess"

diffusion-diagnose: build
	./caramba diffusion-diagnose "An elephant playing chess"

image: diffusion

research:
	go run $(LDFLAGS) . research

serve:
	go run $(LDFLAGS) . serve