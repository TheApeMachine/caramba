package backend

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/kv"
	"github.com/theapemachine/caramba/pkg/runtime/state"
)

const (
	opIDAttentionGQA      ir.OpID = "attention.gqa"
	opIDAttentionSDPA     ir.OpID = "attention.sdpa"
	opIDRoPEPositionStart ir.OpID = "positional.rope"
)

type kvCacheProvider interface {
	Cache() *kv.Cache
}

/*
NewStateBindingHook attaches runtime state supplied through graph.call
inputs to backend IR nodes that still consume state through metadata.
The values themselves are manifest-declared graph inputs such as
kv_cache, position_start, and position_ids.
*/
func NewStateBindingHook() PreExecuteHook {
	return func(irGraph *ir.Graph, inputs map[string]any) error {
		if cache, ok := kvCacheFromInputs(inputs); ok {
			injectKVCache(irGraph, cache)
		}

		positionStart, ok, err := positionStartFromInputs(inputs)

		if err != nil {
			return err
		}

		if ok {
			injectRoPEPositionStart(irGraph, positionStart)
		}

		return nil
	}
}

func kvCacheFromInputs(inputs map[string]any) (*kv.Cache, bool) {
	value, ok := inputs["kv_cache"]

	if !ok {
		return nil, false
	}

	switch typed := value.(type) {
	case kvCacheProvider:
		return typed.Cache(), true
	case state.State:
		provider, ok := typed.(kvCacheProvider)

		if ok {
			return provider.Cache(), true
		}
	}

	return nil, false
}

func positionStartFromInputs(inputs map[string]any) (int, bool, error) {
	if value, ok := inputs["position_start"]; ok {
		position, err := intFromRuntimeValue(value)

		if err != nil {
			return 0, true, fmt.Errorf("position_start: %w", err)
		}

		return position, true, nil
	}

	tokenCount, hasTokens := tokenCountFromInputs(inputs)
	historyLength, hasHistory := historyLengthFromInputs(inputs)

	if !hasTokens || !hasHistory {
		return 0, false, nil
	}

	position := historyLength - tokenCount

	if position < 0 {
		position = 0
	}

	return position, true, nil
}

func historyLengthFromInputs(inputs map[string]any) (int, bool) {
	value, ok := inputs["history"]

	if !ok {
		return 0, false
	}

	switch typed := value.(type) {
	case *state.TokenBuffer:
		return typed.Length(), true
	case []int:
		return len(typed), true
	}

	return 0, false
}

func tokenCountFromInputs(inputs map[string]any) (int, bool) {
	value, ok := inputs["input_ids"]

	if !ok {
		return 0, false
	}

	switch typed := value.(type) {
	case []int:
		return len(typed), true
	case []float64:
		return len(typed), true
	case int, int32, int64:
		return 1, true
	}

	return 0, false
}

func intFromRuntimeValue(value any) (int, error) {
	switch typed := value.(type) {
	case int:
		return typed, nil
	case int32:
		return int(typed), nil
	case int64:
		return int(typed), nil
	case float64:
		return int(typed), nil
	case *state.Counter:
		return typed.Value(), nil
	}

	return 0, fmt.Errorf("expected integer, got %T", value)
}

func injectKVCache(irGraph *ir.Graph, cache *kv.Cache) {
	for _, node := range irGraph.Nodes() {
		operationID := node.OperationID()

		if operationID != opIDAttentionSDPA && operationID != opIDAttentionGQA {
			continue
		}

		causal, _ := node.Metadata()["causal"].(bool)

		if !causal {
			continue
		}

		node.SetMetadata("kv_cache", cache)
	}
}

func injectRoPEPositionStart(irGraph *ir.Graph, positionStart int) {
	for _, node := range irGraph.Nodes() {
		if node.OperationID() != opIDRoPEPositionStart {
			continue
		}

		node.SetMetadata("position_start", positionStart)
	}
}
