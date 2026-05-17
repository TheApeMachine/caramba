package state

import (
	"context"
	"fmt"
	"sort"
	"sync"
)

/*
Snapshot is a serializable capture of a state object. The runtime
uses it for checkpointing and for branching (beam search, speculative
decoding, ablation forks). Payload is the implementation-specific
encoded body; Schema names the encoding so a future loader can pick
the right decoder.
*/
type Snapshot struct {
	StateID string
	Type    string
	Schema  string
	Payload []byte
	Meta    map[string]string
}

/*
Inspection is a human-readable summary used by the researcher
inspection surface. Values is a flat map of named scalars or short
strings; LargePayloadHint points at where a larger payload would be
streamed from.
*/
type Inspection struct {
	StateID          string
	Type             string
	Values           map[string]any
	LargePayloadHint string
}

/*
State is the runtime contract every state object satisfies. The
executor owns the lifetime; ops mutate state through typed helper
methods on the concrete type once they downcast via the registry.
*/
type State interface {
	ID() string
	Type() string
	Reset(ctx context.Context) error
	Snapshot(ctx context.Context) (Snapshot, error)
	Restore(ctx context.Context, snapshot Snapshot) error
	Inspect(ctx context.Context) (Inspection, error)
}

/*
Branchable is the optional contract for state objects that can fork.
Beam search, speculative decoding, classifier-free guidance, and
ablation runs all rely on this.
*/
type Branchable interface {
	State
	Branch(ctx context.Context) (State, error)
	Commit(ctx context.Context) error
}

/*
Factory builds a state object from a typed configuration map. The
runtime registry holds one Factory per state type name.
*/
type Factory func(id string, config map[string]any) (State, error)

/*
Registry is the catalog of state-type factories. The runtime always
goes through a registry so the set of legal state types is explicit
and discoverable.
*/
type Registry struct {
	mu        sync.RWMutex
	factories map[string]Factory
}

func NewRegistry() *Registry {
	return &Registry{factories: map[string]Factory{}}
}

/*
Register binds a state-type name to its factory. It is an error to
register the same name twice — the runtime's set of state kinds is
a typed contract, not a mutable lookup.
*/
func (registry *Registry) Register(typeName string, factory Factory) error {
	registry.mu.Lock()
	defer registry.mu.Unlock()

	if _, exists := registry.factories[typeName]; exists {
		return fmt.Errorf("runtime/state: type %q already registered", typeName)
	}

	registry.factories[typeName] = factory

	return nil
}

/*
MustRegister panics on conflict. Used at package init for the
built-in state types.
*/
func (registry *Registry) MustRegister(typeName string, factory Factory) {
	if err := registry.Register(typeName, factory); err != nil {
		panic(err)
	}
}

/*
Build instantiates a state object by type. The id is the runtime
identifier the executor will refer to it by.
*/
func (registry *Registry) Build(typeName, id string, config map[string]any) (State, error) {
	registry.mu.RLock()
	factory, ok := registry.factories[typeName]
	registry.mu.RUnlock()

	if !ok {
		return nil, fmt.Errorf("runtime/state: unknown type %q", typeName)
	}

	if id == "" {
		return nil, fmt.Errorf("runtime/state: empty id for type %q", typeName)
	}

	return factory(id, config)
}

/*
Types lists every registered state type in lexical order. Used by
diagnostics and by the runtime capability report.
*/
func (registry *Registry) Types() []string {
	registry.mu.RLock()
	defer registry.mu.RUnlock()

	types := make([]string, 0, len(registry.factories))

	for typeName := range registry.factories {
		types = append(types, typeName)
	}

	sort.Strings(types)

	return types
}

/*
Default is the package-level registry pre-populated with the built-in
state types. Callers can register additional types onto it or build
a private registry.
*/
var Default = NewRegistry()

func init() {
	Default.MustRegister("token_buffer", newTokenBufferFromConfig)
	Default.MustRegister("counter", newCounterFromConfig)
	Default.MustRegister("kv_cache", newKVCacheFromConfig)
	Default.MustRegister("rng", newRNGFromConfig)
	Default.MustRegister("tensor", newTensorFromConfig)
}
