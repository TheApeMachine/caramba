.PHONY: build test check verify generate clean chat diffusion diffusion-diagnose image research serve dump tag

# Positional args after `tag` become dummy goals; see scripts/tag-release.sh.
ifeq (tag,$(firstword $(MAKECMDGOALS)))
  TAG_RUN_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  $(foreach arg,$(TAG_RUN_ARGS),$(eval $(arg):;@:))
endif

DUMP_SCRIPT := $(CURDIR)/scripts/dump-repo.py
DUMP_OUTPUT_DIR := $(CURDIR)
DUMP_TARGETS := \
	caramba:. \
	manifesto:../manifesto \
	hf:../hf \
	alcatraz:../alcatraz \
	qpool:../qpool \
	errnie:../errnie

# Metal kernels and metallib generation live in the puter module (go.mod replace).
PUTER_ROOT := ../puter
PUTER_BACKEND_DUMPS := cpu metal cuda xla

# The pool package uses go:linkname to access runtime scheduling
# primitives (dropg, readgstatus) for zero-overhead goroutine parking.
# Go 1.26 restricts these by default; -checklinkname=0 preserves access.
LDFLAGS := -ldflags='-checklinkname=0'

dump:
	@set -e; \
	for spec in $(DUMP_TARGETS); do \
		name=$${spec%%:*}; \
		root=$${spec#*:}; \
		python3 "$(DUMP_SCRIPT)" "$(DUMP_OUTPUT_DIR)/$$name.txt" "$(DUMP_OUTPUT_DIR)/$$root"; \
	done; \
	for backend in $(PUTER_BACKEND_DUMPS); do \
		python3 "$(DUMP_SCRIPT)" "$(DUMP_OUTPUT_DIR)/puter-$$backend.txt" "$(DUMP_OUTPUT_DIR)/$(PUTER_ROOT)" \
			--include-prefix "device/$$backend/"; \
	done; \
	python3 "$(DUMP_SCRIPT)" "$(DUMP_OUTPUT_DIR)/puter.txt" "$(DUMP_OUTPUT_DIR)/$(PUTER_ROOT)" \
		--exclude-prefix device/cpu/ \
		--exclude-prefix device/metal/ \
		--exclude-prefix device/cuda/ \
		--exclude-prefix device/xla/

metal:
	cd $(PUTER_ROOT)/device/metal && CGO_ENABLED=1 go generate

cuda:
	@echo "Skipping CUDA generation: device/cuda in puter has no go:generate step"

build: metal
	go build $(LDFLAGS) .

test:
	go test $(LDFLAGS) ./...

# check runs mechanical enforcement of the manifest-first contract.
# See AGENTS.md and ../puter/GAPS.md §6.5 for the rules.
check:
	@bash "$(CURDIR)/scripts/check_banned.sh"

# verify is the gate: banned-pattern check first, then tests.
verify: check test

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
	go run $(LDFLAGS) main.go serve

tag:
	@bash "$(CURDIR)/scripts/tag-release.sh" \
		"$(if $(TAG),$(TAG),$(word 2,$(MAKECMDGOALS)))" \
		"$(if $(MESSAGE),$(MESSAGE),$(wordlist 3,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS)))"