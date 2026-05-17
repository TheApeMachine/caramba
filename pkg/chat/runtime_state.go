package chat

import (
	"context"
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/kv"
	"github.com/theapemachine/caramba/pkg/runtime/state"
)

/*
KVCacheState wraps the decoder KV cache as a runtime state object so
the runtime program can declare it alongside other state and pass it
to graph.call through the standard inputs map. The full KV strategy
system the platform requirements call for (paged/quantized/branchable
stores etc.) will land as additional state types implementing the
same wrapping pattern.
*/
type KVCacheState struct {
	id    string
	cache *kv.Cache
}

func newKVCacheState(id string, config map[string]any) (state.State, error) {
	store := &KVCacheState{id: id, cache: kv.NewCache()}

	if raw, ok := config["capacity"]; ok {
		capacity, err := asInt(raw)

		if err != nil {
			return nil, fmt.Errorf("state/kv_cache: capacity: %w", err)
		}

		if err := store.cache.SetCapacity(capacity); err != nil {
			return nil, err
		}
	}

	return store, nil
}

func (kvState *KVCacheState) ID() string {
	return kvState.id
}

func (kvState *KVCacheState) Type() string {
	return "kv_cache"
}

func (kvState *KVCacheState) Reset(ctx context.Context) error {
	kvState.cache.Reset()

	return nil
}

/*
Cache exposes the underlying KV cache pointer. The chat PreExecute
hook reads this and attaches it to IR attention nodes as metadata.
*/
func (kvState *KVCacheState) Cache() *kv.Cache {
	return kvState.cache
}

func (kvState *KVCacheState) Snapshot(ctx context.Context) (state.Snapshot, error) {
	return state.Snapshot{}, fmt.Errorf("state/kv_cache: snapshot is not implemented yet")
}

func (kvState *KVCacheState) Restore(ctx context.Context, snapshot state.Snapshot) error {
	return fmt.Errorf("state/kv_cache: restore is not implemented yet")
}

func (kvState *KVCacheState) Inspect(ctx context.Context) (state.Inspection, error) {
	return state.Inspection{
		StateID: kvState.id,
		Type:    kvState.Type(),
		Values: map[string]any{
			"epoch": kvState.cache.Epoch(),
		},
	}, nil
}

func asInt(value any) (int, error) {
	switch typed := value.(type) {
	case int:
		return typed, nil
	case int32:
		return int(typed), nil
	case int64:
		return int(typed), nil
	case float64:
		return int(typed), nil
	}

	return 0, fmt.Errorf("expected integer, got %T", value)
}

func init() {
	state.Default.MustRegister("kv_cache", newKVCacheState)
}
