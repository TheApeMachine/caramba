#!/usr/bin/env bash
# scripts/check_banned.sh — mechanical enforcement for caramba.
#
# caramba is the orchestrator. It wires puter + manifesto + hf together
# and exposes the user-facing CLI/HTTP. Same closed-world rule as the
# rest of the stack: no per-model Go fast-paths in orchestration code.
# The diffusion CLI subcommand survives only as a thin wrapper that
# loads a YAML manifest and dispatches it through the standard executor.
#
# Exits 0 if clean, 1 if any violation found.

set -u
cd "$(git rev-parse --show-toplevel 2>/dev/null || dirname "$0"/..)" || exit 2

violations=0
fail() { printf '  %s\n' "$1" >&2; violations=$((violations + 1)); }
section() { printf '\n=== %s ===\n' "$1"; }

# -----------------------------------------------------------------------------
# 1. No imports of doomed model-specific packages.
# After GAPS.md §6.5 cleanup, manifesto/diffusion will be deleted.
# Catch any new code that still imports it (or new model-specific
# packages that get added).
# -----------------------------------------------------------------------------
section "1. Imports of model-specific manifesto subpackages"

model_pkgs='manifesto/(diffusion|llama|bert|sd3|sdxl|flux|dit|stable_diffusion|stablediffusion|unet|vae)'
while IFS= read -r line; do
    fail "model-specific manifesto import (should compile from YAML): $line"
done < <(grep -rnE --include='*.go' --exclude-dir=vendor --exclude-dir=.git \
    "\"github\\.com/theapemachine/$model_pkgs\"" . 2>/dev/null || true)

# -----------------------------------------------------------------------------
# 2. No model-named orchestration files.
# CLI subcommands and HTTP handlers route to YAML manifests, not to
# model-specific Go. cmd/diffusion.go is OK only if it's a manifest
# loader; it is NOT OK if it imports diffusion-specific logic.
# -----------------------------------------------------------------------------
section "2. Model-named orchestration files importing diffusion logic"

# Find files named after model concepts in cmd/ and check what they import.
while IFS= read -r path; do
    if grep -qE 'manifesto/diffusion|FlowMatchEulerDiscrete|PackLatents|prepare_latents' "$path" 2>/dev/null; then
        fail "$path: imports diffusion-specific logic; must load YAML manifest instead"
    fi
done < <(find cmd pkg -type f -name '*.go' 2>/dev/null \
    | grep -iE '/(diffusion|denoise|flux|sd3|sdxl|llama|unet|vae)' \
    || true)

# -----------------------------------------------------------------------------
# 3. Hot-path map lookups (ARCHITECTURE.md §7)
# Map lookups in execution-DAG hot paths violate the no-allocation
# contract. Flag obvious patterns; reviewer judges scope.
# -----------------------------------------------------------------------------
section "3. Hot-path map lookups (heuristic)"

# Look for map[K]V field declarations in execution paths. This is a
# coarse heuristic — the fix is pre-resolution into a flat slice.
while IFS= read -r line; do
    fail "potential hot-path map (verify not in execution loop): $line"
done < <(grep -rnE --include='*.go' --exclude-dir=vendor --exclude-dir=.git \
    'devices\s+map\[' pkg/backend/compute 2>/dev/null || true)

# -----------------------------------------------------------------------------
# 4. Fusion entries without parity tests (caramba-specific).
# Per AGENTS.md §2: every fusion entry must have a parity test at
# N ∈ {1, 7, 64, 1024, 8192} against the unfused reference.
# -----------------------------------------------------------------------------
section "4. Fusion entries without parity tests"

if [ -d pkg/backend/compute/fusion ]; then
    entries=$(grep -rlE 'Entry\{|FusedOp:' pkg/backend/compute/fusion 2>/dev/null \
        | grep -v _test.go || true)
    parity=$(find pkg/backend/compute/fusion -name '*_parity_test.go' 2>/dev/null || true)
    if [ -n "$entries" ] && [ -z "$parity" ]; then
        fail "fusion catalog has entries but no *_parity_test.go files"
    fi
fi

# -----------------------------------------------------------------------------
# 5. Banned phrases (mirror puter/AGENTS.md §1)
# -----------------------------------------------------------------------------
section "5. Banned phrases"

phrases='for now|approximation acceptable|required vs optional backend|fallback to Go|TODO.*later|will implement.*later|placeholder.*until'
while IFS= read -r line; do
    fail "banned phrase: $line"
done < <(grep -rniE --include='*.go' --exclude-dir=vendor --exclude-dir=.git \
    "(//|/\\*).*($phrases)" . 2>/dev/null || true)

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
printf '\n'
if [ "$violations" -gt 0 ]; then
    printf 'FAILED: %d banned-pattern violation(s)\n' "$violations" >&2
    printf 'See AGENTS.md, puter/ARCHITECTURE.md, puter/GAPS.md.\n' >&2
    exit 1
fi
printf 'OK: no banned-pattern violations\n'
