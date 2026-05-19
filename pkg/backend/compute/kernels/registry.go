package kernels

import (
	"fmt"
	"slices"
	"sync"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
Signature identifies a kernel's tensor layout and dtype contract.
*/
type Signature struct {
	Layout  tensor.Layout
	Inputs  []dtype.DType
	Outputs []dtype.DType
}

/*
Kernel is the registered executable unit.
*/
type Kernel struct {
	Name      string
	Signature Signature
	Locations []tensor.Location
	Run       func(args ...tensor.Tensor) error
}

/*
Registry stores kernels by operation name and signature.
*/
type Registry struct {
	mu      sync.RWMutex
	kernels map[string][]Kernel
}

/*
NewRegistry returns an empty kernel registry.
*/
func NewRegistry() *Registry {
	return &Registry{kernels: map[string][]Kernel{}}
}

/*
Register inserts a kernel. Duplicate name/signature/location entries panic.
*/
func (registry *Registry) Register(kernel Kernel) {
	registry.mu.Lock()
	defer registry.mu.Unlock()

	for _, existing := range registry.kernels[kernel.Name] {
		if !existing.Signature.Equal(kernel.Signature) {
			continue
		}

		if !locationsOverlap(existing.Locations, kernel.Locations) {
			continue
		}

		panic(fmt.Sprintf(
			"kernels: duplicate registration for %q with signature %v",
			kernel.Name,
			kernel.Signature,
		))
	}

	registry.kernels[kernel.Name] = append(registry.kernels[kernel.Name], kernel)
}

/*
Lookup returns the first kernel matching name and signature.
*/
func (registry *Registry) Lookup(name string, signature Signature) (Kernel, bool) {
	registry.mu.RLock()
	defer registry.mu.RUnlock()

	for _, kernel := range registry.kernels[name] {
		if kernel.Signature.Equal(signature) {
			return kernel, true
		}
	}

	return Kernel{}, false
}

/*
LookupLocation returns the kernel matching name, signature, and location.
*/
func (registry *Registry) LookupLocation(
	name string,
	signature Signature,
	location tensor.Location,
) (Kernel, bool) {
	registry.mu.RLock()
	defer registry.mu.RUnlock()

	for _, kernel := range registry.kernels[name] {
		if !kernel.Signature.Equal(signature) {
			continue
		}

		if slices.Contains(kernel.Locations, location) {
			return kernel, true
		}
	}

	return Kernel{}, false
}

/*
Equal reports whether two signatures describe the same tensor contract.
*/
func (signature Signature) Equal(other Signature) bool {
	if signature.Layout != other.Layout {
		return false
	}

	if !slices.Equal(signature.Inputs, other.Inputs) {
		return false
	}

	return slices.Equal(signature.Outputs, other.Outputs)
}

/*
Default is the process-wide kernel registry used by device packages.
*/
var Default = NewRegistry()

func locationsOverlap(left []tensor.Location, right []tensor.Location) bool {
	if len(left) == 0 || len(right) == 0 {
		return true
	}

	for _, location := range left {
		if slices.Contains(right, location) {
			return true
		}
	}

	return false
}
