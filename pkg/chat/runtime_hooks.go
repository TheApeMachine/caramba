package chat

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	modelweights "github.com/theapemachine/caramba/pkg/model/weights"
	"github.com/theapemachine/caramba/pkg/runtime/backend"
	"github.com/theapemachine/caramba/pkg/runtime/program"
	"github.com/theapemachine/caramba/pkg/runtime/state"
)

// IR operation IDs the pre-execute hook scans for. The chat path
// injects KV cache pointers into causal attention nodes and position
// offsets into RoPE nodes; the IDs match what pkg/manifest emits
// during topology lowering for the standard transformer manifests.
const (
	opIDAttentionGQA      ir.OpID = "attention.gqa"
	opIDAttentionSDPA     ir.OpID = "attention.sdpa"
	opIDRoPEPositionStart ir.OpID = "positional.rope"
)

/*
NewWeightBinder returns a backend.WeightBinder that binds every IR
parameter node to its tensor in the supplied modelweights.Store. The
hook is registered on the GraphRunner once and runs per-call so
freshly lowered IR graphs get rebound on every decode iteration.
*/
func NewWeightBinder(store *modelweights.Store) backend.WeightBinder {
	if store == nil {
		return nil
	}

	return func(irGraph *ir.Graph, module program.GraphModule) error {
		return modelweights.BindIR(irGraph, store)
	}
}

/*
NewPreExecuteHook returns a backend.PreExecuteHook that attaches KV
cache pointers, RoPE position offsets, and position_id values to the
right IR nodes before execution. The hook reads these input names:

  inputs["kv_cache"]: *KVCacheState  → causal SDPA/GQA nodes
  inputs["history"]:  *state.TokenBuffer (or []int) → derives positionStart
  inputs["input_ids"]: []int                       → derives input length

position is `history.Length() - len(input_ids)`: the position of the
first token in the forward pass's input sequence. For prefill, where
input_ids contains the full prompt and history already holds it, that
yields 0; for decode, where input_ids is one token appended after a
length-N history, that yields N. This is the same arithmetic the
legacy ModelGenerator does via positionStart = len(history) - 1
followed by single-token decode inputs.
*/
func NewPreExecuteHook() backend.PreExecuteHook {
	return func(irGraph *ir.Graph, inputs map[string]any) error {
		if cache, ok := kvCacheFromInputs(inputs); ok {
			injectKVCache(irGraph, cache)
		}

		tokenCount, hasTokens := tokenCountFromInputs(inputs)
		historyLength, hasHistory := historyLengthFromInputs(inputs)

		if hasTokens && hasHistory {
			positionStart := historyLength - tokenCount

			if positionStart < 0 {
				positionStart = 0
			}

			injectRoPEPositionStart(irGraph, positionStart)
			injectPositionIDs(irGraph, positionStart, tokenCount)
		}

		return nil
	}
}

func kvCacheFromInputs(inputs map[string]any) (any, bool) {
	value, ok := inputs["kv_cache"]

	if !ok {
		return nil, false
	}

	switch typed := value.(type) {
	case *KVCacheState:
		return typed.Cache(), true
	case state.State:
		if kvState, ok := typed.(*KVCacheState); ok {
			return kvState.Cache(), true
		}
	}

	return nil, false
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
	}

	return 0, false
}

func injectKVCache(irGraph *ir.Graph, cache any) {
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

func injectPositionIDs(irGraph *ir.Graph, positionStart int, tokenCount int) {
	index, err := irGraph.Index()

	if err != nil {
		return
	}

	node := index.Node("position_ids")

	if node == nil {
		return
	}

	if node.OpType() != ir.OpInput {
		return
	}

	values := make([]float64, tokenCount)

	for offset := range values {
		values[offset] = float64(positionStart + offset)
	}

	node.SetMetadata("values", values)
}

// formatHookFlowError centralizes error formatting for diagnostics.
func formatHookFlowError(stage string, err error) error {
	return fmt.Errorf("chat/runtime: %s: %w", stage, err)
}
