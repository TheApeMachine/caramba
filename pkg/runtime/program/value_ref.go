package program

import (
	"fmt"
	"strings"
)

/*
ValueRef is a typed reference to a runtime value.
The namespace selects the lookup table (state, graph, sampler,
scheduler, asset, tokenizer, dataset, literal) and Name selects the
object within that table. Path navigates nested fields (e.g.
"sampler.main.stop_matched" → Namespace=sampler, Name=main,
Path=[stop_matched]).
*/
type ValueRef struct {
	Namespace string
	Name      string
	Path      []string
}

/*
ParseValueRef parses a dotted runtime reference. Known namespaces
are enumerated in knownNamespaces so a typo on "state" never silently
falls through to the local variable scope.
*/
func ParseValueRef(raw string) (ValueRef, error) {
	trimmed := strings.TrimSpace(raw)

	if trimmed == "" {
		return ValueRef{}, fmt.Errorf("runtime/program: empty value reference")
	}

	parts := strings.Split(trimmed, ".")

	if len(parts) == 1 {
		return ValueRef{Namespace: NamespaceLocal, Name: parts[0]}, nil
	}

	head := parts[0]

	if !knownNamespaces[head] {
		return ValueRef{Namespace: NamespaceLocal, Name: trimmed}, nil
	}

	if len(parts) < 2 {
		return ValueRef{}, fmt.Errorf("runtime/program: %q is missing a name after namespace", raw)
	}

	return ValueRef{
		Namespace: head,
		Name:      parts[1],
		Path:      append([]string(nil), parts[2:]...),
	}, nil
}

/*
String formats a ValueRef back into its dotted form.
*/
func (valueRef ValueRef) String() string {
	if valueRef.Namespace == NamespaceLocal {
		if len(valueRef.Path) == 0 {
			return valueRef.Name
		}

		return valueRef.Name + "." + strings.Join(valueRef.Path, ".")
	}

	pieces := []string{valueRef.Namespace, valueRef.Name}
	pieces = append(pieces, valueRef.Path...)

	return strings.Join(pieces, ".")
}

/*
IsLocal reports whether the reference targets a step-scoped local
value rather than a declared state object, graph, sampler, or asset.
*/
func (valueRef ValueRef) IsLocal() bool {
	return valueRef.Namespace == NamespaceLocal
}

/*
Equal returns true when two refs target the same object and field.
*/
func (valueRef ValueRef) Equal(other ValueRef) bool {
	if valueRef.Namespace != other.Namespace || valueRef.Name != other.Name {
		return false
	}

	if len(valueRef.Path) != len(other.Path) {
		return false
	}

	for index, segment := range valueRef.Path {
		if other.Path[index] != segment {
			return false
		}
	}

	return true
}

const (
	NamespaceLocal     = "local"
	NamespaceState     = "state"
	NamespaceGraph     = "graph"
	NamespaceSampler   = "sampler"
	NamespaceScheduler = "scheduler"
	NamespaceAsset     = "asset"
	NamespaceTokenizer = "tokenizer"
	NamespaceDataset   = "dataset"
	NamespaceLiteral   = "literal"
)

var knownNamespaces = map[string]bool{
	NamespaceState:     true,
	NamespaceGraph:     true,
	NamespaceSampler:   true,
	NamespaceScheduler: true,
	NamespaceAsset:     true,
	NamespaceTokenizer: true,
	NamespaceDataset:   true,
	NamespaceLiteral:   true,
}
