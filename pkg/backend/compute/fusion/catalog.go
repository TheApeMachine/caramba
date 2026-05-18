/*
Package fusion is the explicit fusion catalog for the orchestrator's
graph-level fusion pass (TENSOR_BACKEND_REWRITE.md §2.19, §3.12).
Every fusion entry names: the source op sequence, the fused op, the
dtype combination, the parity bound vs. unfused execution, and the
target backends.

Per AGENTS.md §1, fusion across a dtype-changing op (cast, convert)
is forbidden unless the resulting numeric path is in the catalog and
parity-tested. The orchestrator's fusion pass walks the IR, matches
sequences against catalog entries, and emits the fused op only when
the entire (op-sequence, dtype-combination) tuple matches.

Per the spray-and-pray contract, this file establishes the catalog
shape and seeds it with the most common transformer fusions:
matmul+bias+gelu, layernorm+residual, int4_dequant+matmul. The
parity tests for these fusions live alongside (Phase 12 work) and
verify against unfused execution at N ∈ {1, 7, 64, 1024, 8192}.
*/
package fusion

import (
	"sync"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
Entry describes a single fusion in the catalog.
*/
type Entry struct {
	// SourceOps is the sequence of forward-graph ops that the
	// fusion replaces. Match is positional and exact.
	SourceOps []string

	// FusedOp is the name of the kernel registered in
	// pkg/backend/compute/kernels that implements the fused
	// computation.
	FusedOp string

	// InputDTypes lists the dtype of each input to the source-op
	// sequence. The fused kernel must accept this combination.
	InputDTypes []dtype.DType

	// OutputDType is the dtype of the fused output.
	OutputDType dtype.DType

	// Layout is the tensor layout the fusion applies to. Most
	// fusions are LayoutDense; sparse fusions go here too.
	Layout tensor.Layout

	// Backends names the Locations this fusion is verified on. A
	// fusion missing from a backend's list is not applied even if
	// the catalog otherwise matches.
	Backends []tensor.Location

	// ParityULPBound is the tolerated ULP difference vs. unfused
	// execution. The orchestrator's parity test will fail if the
	// fused output exceeds this bound.
	ParityULPBound int
}

/*
Catalog is a thread-safe registry of fusion entries.
*/
type Catalog struct {
	mu      sync.RWMutex
	entries []Entry
}

/*
NewCatalog returns an empty catalog.
*/
func NewCatalog() *Catalog {
	return &Catalog{}
}

/*
Register adds an entry. Duplicate (SourceOps, InputDTypes, Layout,
Backend) combinations are programmer errors and panic.
*/
func (catalog *Catalog) Register(entry Entry) {
	catalog.mu.Lock()
	defer catalog.mu.Unlock()

	for _, existing := range catalog.entries {
		if entriesEqual(existing, entry) {
			panic("fusion: duplicate entry in catalog")
		}
	}

	catalog.entries = append(catalog.entries, entry)
}

/*
Lookup returns the catalog entry matching the given source-op
sequence, input dtypes, layout, and backend. Returns nil if no entry
matches; the orchestrator pass passes the sequence through unfused
in that case.
*/
func (catalog *Catalog) Lookup(
	sourceOps []string,
	inputDTypes []dtype.DType,
	layout tensor.Layout,
	backend tensor.Location,
) *Entry {
	catalog.mu.RLock()
	defer catalog.mu.RUnlock()

	for index, entry := range catalog.entries {
		if entry.Layout != layout {
			continue
		}

		if !stringSliceEqual(entry.SourceOps, sourceOps) {
			continue
		}

		if !dtypeSliceEqual(entry.InputDTypes, inputDTypes) {
			continue
		}

		if !containsLocation(entry.Backends, backend) {
			continue
		}

		return &catalog.entries[index]
	}

	return nil
}

/*
Entries returns a snapshot of all registered entries. Used by the
orchestrator's catalog-dump diagnostic and by the parity test suite.
*/
func (catalog *Catalog) Entries() []Entry {
	catalog.mu.RLock()
	defer catalog.mu.RUnlock()

	out := make([]Entry, len(catalog.entries))
	copy(out, catalog.entries)

	return out
}

/*
Default is the package-level fusion catalog, pre-populated with the
standard transformer fusions on host.
*/
var Default = NewCatalog()

func init() {
	registerStandardTransformerFusions()
}

func registerStandardTransformerFusions() {
	// MatMul + Bias + GELU is the most common feed-forward block.
	Default.Register(Entry{
		SourceOps:      []string{"matmul", "add", "gelu"},
		FusedOp:        "matmul_bias_gelu",
		InputDTypes:    []dtype.DType{dtype.BFloat16, dtype.BFloat16, dtype.BFloat16},
		OutputDType:    dtype.BFloat16,
		Layout:         tensor.LayoutDense,
		Backends:       []tensor.Location{tensor.Host, tensor.Metal, tensor.CUDA},
		ParityULPBound: 2,
	})

	Default.Register(Entry{
		SourceOps:      []string{"matmul", "add", "gelu"},
		FusedOp:        "matmul_bias_gelu",
		InputDTypes:    []dtype.DType{dtype.Float32, dtype.Float32, dtype.Float32},
		OutputDType:    dtype.Float32,
		Layout:         tensor.LayoutDense,
		Backends:       []tensor.Location{tensor.Host, tensor.Metal, tensor.CUDA},
		ParityULPBound: 1,
	})

	// LayerNorm + Residual fuses the post-attention norm with the
	// residual connection that follows it. Common in transformer
	// blocks.
	Default.Register(Entry{
		SourceOps:   []string{"layernorm", "add"},
		FusedOp:     "layernorm_residual",
		InputDTypes: []dtype.DType{
			dtype.Float16, dtype.Float16, dtype.Float16, dtype.Float16,
		},
		OutputDType:    dtype.Float16,
		Layout:         tensor.LayoutDense,
		Backends:       []tensor.Location{tensor.Host, tensor.CUDA},
		ParityULPBound: 2,
	})

	// Int4 dequant + bf16 matmul: the inference fast path for
	// GPTQ/AWQ-quantized models. The fused kernel keeps the
	// dequantization output in registers and never writes a
	// full bf16 intermediate.
	Default.Register(Entry{
		SourceOps:      []string{"int4_dequant", "matmul"},
		FusedOp:        "int4_dequant_matmul",
		InputDTypes:    []dtype.DType{dtype.Int4, dtype.Float32, dtype.BFloat16},
		OutputDType:    dtype.BFloat16,
		Layout:         tensor.LayoutDense,
		Backends:       []tensor.Location{tensor.CUDA, tensor.Metal},
		ParityULPBound: 4,
	})
}

func entriesEqual(left, right Entry) bool {
	if left.Layout != right.Layout {
		return false
	}

	if !stringSliceEqual(left.SourceOps, right.SourceOps) {
		return false
	}

	if !dtypeSliceEqual(left.InputDTypes, right.InputDTypes) {
		return false
	}

	if left.OutputDType != right.OutputDType {
		return false
	}

	for _, backend := range left.Backends {
		if !containsLocation(right.Backends, backend) {
			return false
		}
	}

	for _, backend := range right.Backends {
		if !containsLocation(left.Backends, backend) {
			return false
		}
	}

	return true
}

func stringSliceEqual(left, right []string) bool {
	if len(left) != len(right) {
		return false
	}

	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}

	return true
}

func dtypeSliceEqual(left, right []dtype.DType) bool {
	if len(left) != len(right) {
		return false
	}

	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}

	return true
}

func containsLocation(set []tensor.Location, target tensor.Location) bool {
	for _, candidate := range set {
		if candidate == target {
			return true
		}
	}

	return false
}
