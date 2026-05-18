/*
Package kernels is the dispatch home for every compute operation in
Caramba. Kernels are keyed on (operation name, Layout, DType
combination) and resolved at call time; backends register kernel
implementations during their package init.

Per AGENTS.md §1 and the spray-and-pray contract, this package
establishes the kernel-registry shape that Phase 8 fills in across
math, attention, convolution, optimizer, quantized, FP8, and sparse
families. Each kernel family ships scalar Go + AVX-512 + AVX2 + SSE2
+ NEON variants plus Metal / CUDA / XLA paths.

The example kernel in add.go shows the pattern: a Kernel value
declares its supported dtype combinations, the registry routes calls
by inputs' DType signatures, and the implementation chooses the best
available ISA path at runtime.
*/
package kernels

import (
	"fmt"
	"sync"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
Signature identifies a kernel's input/output dtype combination. The
registry uses Signature as the secondary key (operation name is the
primary).
*/
type Signature struct {
	Layout  tensor.Layout
	Inputs  []dtype.DType
	Outputs []dtype.DType
}

/*
Kernel is the registered unit. Run is the entry point; the body
dispatches on Location and ISA at call time.
*/
type Kernel struct {
	Name      string
	Signature Signature
	Locations []tensor.Location
	Run       func(args ...tensor.Tensor) error
}

/*
Registry maps (op-name, signature) to a Kernel. Phase 8 populates it
across kernel families; this skeleton supports registration plus
lookup by name + input dtype list.
*/
type Registry struct {
	mu      sync.RWMutex
	kernels map[string][]Kernel
}

/*
NewRegistry returns an empty registry.
*/
func NewRegistry() *Registry {
	return &Registry{kernels: map[string][]Kernel{}}
}

/*
Register adds a kernel to the registry. Duplicate (name, signature)
pairs panic — kernel-table conflicts are programmer errors caught at
init time.
*/
func (registry *Registry) Register(kernel Kernel) {
	registry.mu.Lock()
	defer registry.mu.Unlock()

	for _, existing := range registry.kernels[kernel.Name] {
		if signaturesEqual(existing.Signature, kernel.Signature) {
			panic(fmt.Sprintf(
				"kernels: duplicate registration for %q with signature %v",
				kernel.Name, kernel.Signature,
			))
		}
	}

	registry.kernels[kernel.Name] = append(registry.kernels[kernel.Name], kernel)
}

/*
Lookup returns the kernel registered for the given (name, signature)
plus an ok flag. The returned Kernel is a copy of the registered
entry, not a pointer into the registry's internal slice — that would
dangle if Register later appends and the slice grows. Phase 8 work
refines this to honor Location preferences.
*/
func (registry *Registry) Lookup(name string, signature Signature) (Kernel, bool) {
	registry.mu.RLock()
	defer registry.mu.RUnlock()

	for _, kernel := range registry.kernels[name] {
		if signaturesEqual(kernel.Signature, signature) {
			return kernel, true
		}
	}

	return Kernel{}, false
}

/*
Default is the package-level registry. Forward kernels register here
at init time; callers look up by name.
*/
var Default = NewRegistry()

func signaturesEqual(left, right Signature) bool {
	if left.Layout != right.Layout {
		return false
	}

	if len(left.Inputs) != len(right.Inputs) || len(left.Outputs) != len(right.Outputs) {
		return false
	}

	for index := range left.Inputs {
		if left.Inputs[index] != right.Inputs[index] {
			return false
		}
	}

	for index := range left.Outputs {
		if left.Outputs[index] != right.Outputs[index] {
			return false
		}
	}

	return true
}
