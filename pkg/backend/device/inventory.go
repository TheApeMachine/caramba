package device

import (
	"fmt"
	"reflect"
	"sort"

	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
)

/*
CrossLinkKind classifies how an ir.RequiredOperationIDs entry relates to
device.Backend surfaces.
*/
type CrossLinkKind string

const (
	CrossLinkDirect         CrossLinkKind = "direct"
	CrossLinkComposite      CrossLinkKind = "composite"
	CrossLinkKernelRegistry CrossLinkKind = "kernel_registry"
	CrossLinkGraphOnly      CrossLinkKind = "graph_only"
)

/*
MethodRef names one method on an embedded Backend surface interface.
*/
type MethodRef struct {
	Surface string
	Method  string
}

/*
BackendMethodEntry is one method on device.Backend (via an embedded interface).
*/
type BackendMethodEntry struct {
	Surface string
	Method  string
}

/*
OperationCrossLink records how a required IR operation ID maps to Backend
methods and/or other execution surfaces.
*/
type OperationCrossLink struct {
	OperationID  ir.OpType
	Kind         CrossLinkKind
	Methods      []MethodRef
	RegistryNote string
}

/*
EnumerateBackendMethods lists every method on the embedded interfaces that
compose device.Backend, discovered via reflection.
*/
func EnumerateBackendMethods() []BackendMethodEntry {
	surfaces := backendSurfaceTypes()
	entries := make([]BackendMethodEntry, 0, 160)

	for _, surface := range surfaces {
		for methodIndex := 0; methodIndex < surface.NumMethod(); methodIndex++ {
			entries = append(entries, BackendMethodEntry{
				Surface: surface.Name(),
				Method:  surface.Method(methodIndex).Name,
			})
		}
	}

	sort.Slice(entries, func(leftIndex, rightIndex int) bool {
		leftEntry := entries[leftIndex]
		rightEntry := entries[rightIndex]

		if leftEntry.Surface != rightEntry.Surface {
			return leftEntry.Surface < rightEntry.Surface
		}

		return leftEntry.Method < rightEntry.Method
	})

	return entries
}

/*
BuildOperationCrossLinkIndex maps each required operation ID to its cross-link.
*/
func BuildOperationCrossLinkIndex() map[ir.OpType]OperationCrossLink {
	links := operationCrossLinks()
	index := make(map[ir.OpType]OperationCrossLink, len(links))

	for _, link := range links {
		index[link.OperationID] = link
	}

	return index
}

/*
ValidateBackendInventory checks enumeration and cross-link consistency.
*/
func ValidateBackendInventory() error {
	methodSet := methodRefSet(EnumerateBackendMethods())
	links := operationCrossLinks()
	seenOperations := make(map[ir.OpType]bool, len(links))

	for _, link := range links {
		if seenOperations[link.OperationID] {
			return fmt.Errorf("duplicate cross-link for operation %q", link.OperationID)
		}

		seenOperations[link.OperationID] = true

		if err := validateOperationCrossLink(link, methodSet); err != nil {
			return err
		}
	}

	for _, operationID := range ir.RequiredOperationIDs() {
		if !seenOperations[operationID] {
			return fmt.Errorf("required operation %q has no cross-link", operationID)
		}
	}

	return nil
}

/*
BackendMethodsWithoutRequiredOperation lists Backend methods not referenced by
any required-operation cross-link (direct or composite).
*/
func BackendMethodsWithoutRequiredOperation() []BackendMethodEntry {
	referenced := make(map[BackendMethodEntry]bool)

	for _, link := range operationCrossLinks() {
		if link.Kind != CrossLinkDirect && link.Kind != CrossLinkComposite {
			continue
		}

		for _, methodRef := range link.Methods {
			referenced[BackendMethodEntry{Surface: methodRef.Surface, Method: methodRef.Method}] = true
		}
	}

	unmapped := make([]BackendMethodEntry, 0)

	for _, entry := range EnumerateBackendMethods() {
		if referenced[entry] {
			continue
		}

		unmapped = append(unmapped, entry)
	}

	return unmapped
}

func backendSurfaceTypes() []reflect.Type {
	interfaceValues := []any{
		(*PosPop)(nil),
		(*Activation)(nil),
		(*Elementwise)(nil),
		(*Reduction)(nil),
		(*Dot)(nil),
		(*Matmul)(nil),
		(*Pool)(nil),
		(*Convolution)(nil),
		(*Dropout)(nil),
		(*Losses)(nil),
		(*Sampling)(nil),
		(*Embedding)(nil),
		(*Normalization)(nil),
		(*LayerNorm)(nil),
		(*RoPE)(nil),
		(*Hawkes)(nil),
		(*Physics)(nil),
		(*Causal)(nil),
		(*Masking)(nil),
		(*Attention)(nil),
		(*VSA)(nil),
		(*ActiveInference)(nil),
		(*PredictiveCoding)(nil),
		(*Dequant)(nil),
		(*Quant)(nil),
	}

	surfaces := make([]reflect.Type, 0, len(interfaceValues))

	for _, interfaceValue := range interfaceValues {
		surfaces = append(surfaces, reflect.TypeOf(interfaceValue).Elem())
	}

	return surfaces
}

func methodRefSet(entries []BackendMethodEntry) map[BackendMethodEntry]bool {
	set := make(map[BackendMethodEntry]bool, len(entries))

	for _, entry := range entries {
		set[entry] = true
	}

	return set
}

func validateOperationCrossLink(
	link OperationCrossLink,
	methodSet map[BackendMethodEntry]bool,
) error {
	switch link.Kind {
	case CrossLinkDirect, CrossLinkComposite:
		if len(link.Methods) == 0 {
			return fmt.Errorf("operation %q: %s link has no methods", link.OperationID, link.Kind)
		}

		for _, methodRef := range link.Methods {
			entry := BackendMethodEntry{Surface: methodRef.Surface, Method: methodRef.Method}

			if !methodSet[entry] {
				return fmt.Errorf(
					"operation %q: unknown Backend method %s.%s",
					link.OperationID,
					methodRef.Surface,
					methodRef.Method,
				)
			}
		}

	case CrossLinkKernelRegistry, CrossLinkGraphOnly:
		if link.RegistryNote == "" {
			return fmt.Errorf("operation %q: %s link needs RegistryNote", link.OperationID, link.Kind)
		}

	default:
		return fmt.Errorf("operation %q: unknown cross-link kind %q", link.OperationID, link.Kind)
	}

	return nil
}

func directLink(operationID ir.OpType, surface, method string) OperationCrossLink {
	return OperationCrossLink{
		OperationID: operationID,
		Kind:        CrossLinkDirect,
		Methods:     []MethodRef{{Surface: surface, Method: method}},
	}
}

func compositeLink(operationID ir.OpType, methods ...MethodRef) OperationCrossLink {
	return OperationCrossLink{
		OperationID: operationID,
		Kind:        CrossLinkComposite,
		Methods:     methods,
	}
}

func registryLink(operationID ir.OpType, registryNote string) OperationCrossLink {
	return OperationCrossLink{
		OperationID:  operationID,
		Kind:         CrossLinkKernelRegistry,
		RegistryNote: registryNote,
	}
}

func graphLink(operationID ir.OpType, note string) OperationCrossLink {
	return OperationCrossLink{
		OperationID:  operationID,
		Kind:         CrossLinkGraphOnly,
		RegistryNote: note,
	}
}
