package state

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/kv"
)

/*
KVCache wraps the decoder key/value store as a runtime state object.
Runtime manifests declare it and graph.call passes it through the
standard state namespace.
*/
type KVCache struct {
	id    string
	cache *kv.Cache
}

func newKVCacheFromConfig(id string, config map[string]any) (State, error) {
	state := &KVCache{id: id, cache: kv.NewCache()}

	if raw, ok := config["capacity"]; ok {
		capacity, err := intFromAny(raw)

		if err != nil {
			return nil, fmt.Errorf("kv_cache: capacity: %w", err)
		}

		if err := state.cache.SetCapacity(capacity); err != nil {
			return nil, err
		}
	}

	return state, nil
}

func (state *KVCache) ID() string {
	return state.id
}

func (state *KVCache) Type() string {
	return "kv_cache"
}

func (state *KVCache) Reset(ctx context.Context) error {
	state.cache.Reset()

	return nil
}

/*
Cache exposes the backend KV store to graph binding.
*/
func (state *KVCache) Cache() *kv.Cache {
	return state.cache
}

func (state *KVCache) Snapshot(ctx context.Context) (Snapshot, error) {
	snapshot, err := state.cache.Snapshot()

	if err != nil {
		return Snapshot{}, err
	}

	payload, err := json.Marshal(snapshot)

	if err != nil {
		return Snapshot{}, fmt.Errorf("kv_cache: encode snapshot: %w", err)
	}

	return Snapshot{
		StateID: state.id,
		Type:    state.Type(),
		Schema:  "kv-cache-json-v1",
		Payload: payload,
	}, nil
}

func (state *KVCache) Restore(ctx context.Context, snapshot Snapshot) error {
	if snapshot.Schema != "kv-cache-json-v1" {
		return fmt.Errorf("kv_cache: unsupported snapshot schema %q", snapshot.Schema)
	}

	var decoded kv.Snapshot

	if err := json.Unmarshal(snapshot.Payload, &decoded); err != nil {
		return fmt.Errorf("kv_cache: decode snapshot: %w", err)
	}

	return state.cache.Restore(decoded)
}

func (state *KVCache) Inspect(ctx context.Context) (Inspection, error) {
	return Inspection{
		StateID: state.id,
		Type:    state.Type(),
		Values: map[string]any{
			"capacity": state.cache.Capacity(),
			"entries":  state.cache.EntryCount(),
			"epoch":    state.cache.Epoch(),
		},
	}, nil
}
