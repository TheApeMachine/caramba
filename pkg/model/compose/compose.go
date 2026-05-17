// Package compose turns a safetensors weight store into a runtime
// manifest.Graph by walking tensor names and shapes and emitting
// primitive IR nodes. The same compiler runs on any model whose
// tensors follow PyTorch naming conventions (HuggingFace transformers,
// diffusers, etc.) — there is no per-model code path, only pattern
// registrations.
//
// The "atom layer" the platform commits to is: every higher-level
// concept (AdaLayerNorm, MHA, MLP, sinusoidal-timestep-embedding…)
// must be expressible as a composition of registered primitives
// (projection.linear, math.{add,mul,layernorm,rmsnorm,sin,cos,exp,…},
// shape.{split,concat,slice}, activation.{gelu,silu,…}). If a model
// needs something the primitive set does not cover, the gap is an
// op-level addition, not an architecture-level special case.
//
// Usage:
//
//	store, _ := modelweights.Open(path)
//	graph, err := compose.FromSafetensors(store, compose.Hints{
//	    Inputs: []compose.InputSpec{{Name: "input_ids"}},
//	    Output: "logits",
//	})
//
// Patterns are discovered automatically: the compiler walks tensor
// names, groups them by hierarchical prefix, and asks each registered
// pattern in priority order whether it claims the group. Higher-level
// patterns (a full transformer block) can claim a whole prefix and
// emit several primitive nodes in one shot; leaf patterns (a single
// linear projection) claim individual tensor pairs.
package compose

import (
	"fmt"
	"sort"
	"strings"

	"github.com/theapemachine/caramba/pkg/manifest"
	modelweights "github.com/theapemachine/caramba/pkg/model/weights"
)

/*
TensorCatalog is the minimal weights-store view compose needs:
enumerate tensor names and look up their shape/dtype. The production
*modelweights.Store satisfies this interface; tests can pass an
in-memory implementation without touching disk.
*/
type TensorCatalog interface {
	Names() []string
	Info(name string) (modelweights.TensorInfo, bool)
}

/*
InputSpec describes a graph-level input the compiled topology needs
to declare. Kind is an optional hint that helps patterns pick the
right entry node (e.g. "tokens" for an embedding lookup, "latent"
for a diffusion noise tensor).
*/
type InputSpec struct {
	Name string
	Kind string
}

/*
Hints carry the small amount of context that cannot be inferred from
the safetensors file alone — what the graph's external inputs are
called and which tensor name the runtime should treat as the final
output. The compiler chooses among recognized architectures using
these hints plus what it finds in the tensor list; there is no
"architecture: flux2" switch.
*/
type Hints struct {
	Inputs []InputSpec
	Output string
}

/*
TensorRef points a node back at the safetensors entry whose values
should be bound to its weight slot. Patterns set this on the nodes
they emit; the runtime's WeightBinder uses it to load values at
execute time.
*/
type TensorRef struct {
	Weight string // tensor name in the store (e.g. "transformer.h.0.attn.c_attn.weight")
	Bias   string // optional, "" when the layer has no bias
	// Transpose records whether the stored weight matrix needs to be
	// transposed before use. Most PyTorch linear layers store the
	// weight as [out_features, in_features] and need a transpose to
	// reach the [in, out] layout our projection.linear consumes.
	Transpose bool
}

/*
Pattern claims a group of tensors and emits the IR nodes that
implement them. Patterns are stateless and idempotent — given the
same input tensor group, they must emit the same nodes. The compiler
calls Match first; only patterns that return true have Emit invoked.
*/
type Pattern interface {
	// Name is the human-readable pattern identifier used in errors
	// and debugging.
	Name() string

	// Priority orders patterns: higher numbers run first. A
	// "transformer-block" pattern (priority 100) gets first crack at
	// a prefix group; if it declines, lower-priority leaf patterns
	// (linear, layernorm…) try to claim the remaining tensors.
	Priority() int

	// Match decides whether this pattern handles group. group.Names
	// is the set of tensor names sharing the same hierarchical prefix
	// (e.g. all tensors under "transformer.h.0."). Returning true
	// commits the pattern to handling every tensor in group.Claimed.
	Match(group TensorGroup) bool

	// Emit produces the IR nodes that implement the matched group.
	// It receives the BuilderContext so it can append nodes, allocate
	// fresh intermediate-binding names, and resolve cross-group
	// dependencies (e.g. an attention block depending on the previous
	// block's residual stream).
	Emit(group TensorGroup, ctx *BuilderContext) error
}

/*
TensorGroup is a set of tensors that share a hierarchical prefix.
The compiler builds groups by splitting tensor names on dots and
grouping tensors whose first N components match.
*/
type TensorGroup struct {
	Prefix string                 // e.g. "transformer.h.0"
	Names  []string               // full tensor names in the group, sorted
	Info   func(string) (modelweights.TensorInfo, bool)
}

/*
BuilderContext gives patterns the entry points they need to emit IR.
Patterns hold no state of their own — everything they need flows
through this context. Concurrent calls into a BuilderContext are
not safe; the compiler calls patterns serially.

Block-level patterns (e.g. a whole transformer block) use Catalog()
to peek at sibling tensors and ConsumeTensors() to mark every
tensor they claim so the leaf walker does not re-emit nodes for
sub-components the block already handled.
*/
type BuilderContext struct {
	graph     *manifest.Graph
	bindings  map[string]string // logical name → producing binding (out port)
	nextLocal int
	catalog   TensorCatalog
	consumed  map[string]bool
}

/*
Catalog returns the full tensor catalog so a pattern can peek at
sibling or descendant tensors before deciding how to emit. Used
mainly by block-level patterns that need to confirm every expected
sub-component is present before claiming the prefix.
*/
func (ctx *BuilderContext) Catalog() TensorCatalog {
	return ctx.catalog
}

/*
ConsumeTensors marks the named tensors as already handled by the
caller. The leaf walker skips any group whose tensors are all
consumed. Block patterns call this for every tensor they emitted
nodes for, including sub-prefix tensors that leaf patterns would
otherwise claim a second time.
*/
func (ctx *BuilderContext) ConsumeTensors(names ...string) {
	if ctx.consumed == nil {
		ctx.consumed = make(map[string]bool, len(names))
	}

	for _, name := range names {
		ctx.consumed[name] = true
	}
}

/*
Bind records that `producerNode` writes a value named `binding`.
Subsequent calls to ResolveBinding(binding) return the producing
node id so downstream patterns can wire their inputs.
*/
func (ctx *BuilderContext) Bind(binding, producerNode string) {
	if ctx.bindings == nil {
		ctx.bindings = make(map[string]string)
	}

	ctx.bindings[binding] = producerNode
}

/*
ResolveBinding returns the node id that produces binding, or ""
when no such binding has been registered yet. Returning the empty
string signals an external graph input that the caller has to mark
via Graph's input map.
*/
func (ctx *BuilderContext) ResolveBinding(binding string) string {
	return ctx.bindings[binding]
}

/*
LocalName returns a fresh binding name unique within this graph,
prefixed with the pattern's identifier. Patterns use it for the
intermediate values they need between emitted nodes (e.g. q/k/v
projections inside an attention block).
*/
func (ctx *BuilderContext) LocalName(prefix string) string {
	ctx.nextLocal++

	return fmt.Sprintf("%s.local.%d", prefix, ctx.nextLocal)
}

/*
AddNode appends node to the underlying graph. Patterns call this for
every primitive operation they emit. The compiler validates the
graph at the end of the FromSafetensors call.
*/
func (ctx *BuilderContext) AddNode(node *manifest.Node) error {
	return manifest.AddComposedNode(ctx.graph, node)
}

/*
Registry holds the set of Patterns the compiler will try, ordered
by descending priority. Patterns can be added at init time from
sub-packages without modifying the compose package itself.
*/
type Registry struct {
	patterns []Pattern
}

func NewRegistry() *Registry {
	return &Registry{}
}

func (registry *Registry) Register(pattern Pattern) {
	registry.patterns = append(registry.patterns, pattern)

	sort.SliceStable(registry.patterns, func(i, j int) bool {
		return registry.patterns[i].Priority() > registry.patterns[j].Priority()
	})
}

func (registry *Registry) Patterns() []Pattern {
	out := make([]Pattern, len(registry.patterns))
	copy(out, registry.patterns)

	return out
}

/*
DefaultRegistry is populated by sub-packages via init(). The
FromSafetensors entry point uses it unless the caller overrides via
FromSafetensorsWithRegistry.
*/
var DefaultRegistry = NewRegistry()

/*
FromSafetensors compiles catalog + hints into a manifest.Graph using
the DefaultRegistry. See FromSafetensorsWithRegistry for the form
that accepts a custom pattern set (used by tests).
*/
func FromSafetensors(catalog TensorCatalog, hints Hints) (*manifest.Graph, error) {
	return FromSafetensorsWithRegistry(catalog, hints, DefaultRegistry)
}

/*
FromSafetensorsWithRegistry is the explicit form that accepts a
caller-provided pattern set. Tests and embedded use cases that want
a known-good pattern subset call this directly.
*/
func FromSafetensorsWithRegistry(
	catalog TensorCatalog, hints Hints, registry *Registry,
) (*manifest.Graph, error) {
	if catalog == nil {
		return nil, fmt.Errorf("compose: tensor catalog is required")
	}

	if registry == nil {
		return nil, fmt.Errorf("compose: pattern registry is required")
	}

	tensorNames := catalog.Names()
	sort.Strings(tensorNames)

	groups := groupByPrefix(tensorNames)

	graph := manifest.NewComposedGraph()
	ctx := &BuilderContext{
		graph:    graph,
		catalog:  catalog,
		consumed: make(map[string]bool, len(tensorNames)),
	}

	for _, input := range hints.Inputs {
		ctx.Bind(input.Name, "")
		manifest.MarkComposedInput(graph, input.Name)
	}

	for _, group := range groups {
		group.Info = catalog.Info

		filtered := TensorGroup{Prefix: group.Prefix, Names: filterUnconsumed(group.Names, ctx.consumed), Info: group.Info}

		if len(filtered.Names) == 0 {
			continue
		}

		matched := false

		for _, pattern := range registry.Patterns() {
			if !pattern.Match(filtered) {
				continue
			}

			if err := pattern.Emit(filtered, ctx); err != nil {
				return nil, fmt.Errorf("compose: pattern %q on prefix %q: %w", pattern.Name(), filtered.Prefix, err)
			}

			ctx.ConsumeTensors(filtered.Names...)
			matched = true

			break
		}

		if !matched {
			// Surface uncovered prefixes loudly rather than emitting
			// a half-graph. Adding a missing pattern is the right fix.
			return nil, fmt.Errorf(
				"compose: no pattern claims prefix %q (tensors: %s)",
				filtered.Prefix,
				strings.Join(headStrings(filtered.Names, 4), ", "),
			)
		}
	}

	if hints.Output != "" {
		if producer := ctx.ResolveBinding(hints.Output); producer == "" {
			return nil, fmt.Errorf(
				"compose: declared output %q is not produced by any pattern",
				hints.Output,
			)
		}
	}

	return graph, nil
}

/*
groupByPrefix returns one TensorGroup per leaf-or-near-leaf prefix.
"Near-leaf" here means: every tensor name is split on dots, and the
group prefix is everything up to (but not including) the final
component (e.g. tensors "a.b.weight" and "a.b.bias" share prefix
"a.b"). Tensors with no dots become their own group with prefix "".
This is the shape patterns expect: each group is roughly one layer
or one tightly-coupled set of parameters (weight + bias, q/k/v).
*/
func groupByPrefix(names []string) []TensorGroup {
	buckets := make(map[string][]string)

	for _, name := range names {
		prefix := prefixOf(name)
		buckets[prefix] = append(buckets[prefix], name)
	}

	prefixes := make([]string, 0, len(buckets))

	for prefix := range buckets {
		prefixes = append(prefixes, prefix)
	}

	sort.Strings(prefixes)

	groups := make([]TensorGroup, 0, len(prefixes))

	for _, prefix := range prefixes {
		names := buckets[prefix]
		sort.Strings(names)
		groups = append(groups, TensorGroup{Prefix: prefix, Names: names})
	}

	return groups
}

func prefixOf(name string) string {
	index := strings.LastIndex(name, ".")

	if index < 0 {
		return ""
	}

	return name[:index]
}

func filterUnconsumed(names []string, consumed map[string]bool) []string {
	out := make([]string, 0, len(names))

	for _, name := range names {
		if consumed[name] {
			continue
		}

		out = append(out, name)
	}

	return out
}

func headStrings(items []string, n int) []string {
	if len(items) <= n {
		return items
	}

	out := make([]string, 0, n+1)
	out = append(out, items[:n]...)
	out = append(out, fmt.Sprintf("…%d more", len(items)-n))

	return out
}
