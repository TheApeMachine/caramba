package chat

import (
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"

	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/runtime/state"
)

func TestPreExecuteHookKVCacheInjection(t *testing.T) {
	Convey("Given an IR graph with a causal SDPA node", t, func() {
		shape, err := tensor.NewShape([]int{1, 1, 1, 4})
		So(err, ShouldBeNil)

		graph := ir.NewGraph()
		sdpa := ir.NewNode("attn", ir.OpType(opIDAttentionSDPA), shape)
		sdpa.SetOperationID(opIDAttentionSDPA)
		sdpa.SetMetadata("causal", true)
		graph.AddNode(sdpa)

		kvState, err := newKVCacheState("kv", nil)
		So(err, ShouldBeNil)

		hook := NewPreExecuteHook()

		Convey("Running the hook should attach the cache pointer to the SDPA node", func() {
			err := hook(graph, map[string]any{
				"kv_cache": kvState,
			})
			So(err, ShouldBeNil)

			injected, ok := sdpa.Metadata()["kv_cache"]
			So(ok, ShouldBeTrue)
			So(injected, ShouldEqual, kvState.(*KVCacheState).Cache())
		})

		Convey("Non-causal SDPA nodes should not receive a cache pointer", func() {
			sdpa.SetMetadata("causal", false)

			err := hook(graph, map[string]any{"kv_cache": kvState})
			So(err, ShouldBeNil)

			_, ok := sdpa.Metadata()["kv_cache"]
			So(ok, ShouldBeFalse)
		})
	})
}

func TestPreExecuteHookRoPEAndPositionIDs(t *testing.T) {
	Convey("Given an IR graph with a RoPE node and a position_ids input", t, func() {
		shape, err := tensor.NewShape([]int{1, 3})
		So(err, ShouldBeNil)

		graph := ir.NewGraph()
		rope := ir.NewNode("rope", ir.OpType(opIDRoPEPositionStart), shape)
		rope.SetOperationID(opIDRoPEPositionStart)
		positions := ir.NewNode("position_ids", ir.OpInput, shape)
		graph.AddNode(rope)
		graph.AddNode(positions)

		hook := NewPreExecuteHook()

		Convey("Prefill should set position_start=0 and bind 0..N-1", func() {
			history, err := state.Default.Build("token_buffer", "history", nil)
			So(err, ShouldBeNil)
			history.(*state.TokenBuffer).Extend([]int{10, 20, 30})

			err = hook(graph, map[string]any{
				"history":   history,
				"input_ids": []int{10, 20, 30},
			})
			So(err, ShouldBeNil)

			So(rope.Metadata()["position_start"], ShouldEqual, 0)

			values, ok := positions.Metadata()["values"].([]float64)
			So(ok, ShouldBeTrue)
			So(values, ShouldResemble, []float64{0, 1, 2})
		})

		Convey("Decode should set position_start to history.Length() - 1", func() {
			history, err := state.Default.Build("token_buffer", "history", nil)
			So(err, ShouldBeNil)
			history.(*state.TokenBuffer).Extend([]int{10, 20, 30, 99})

			err = hook(graph, map[string]any{
				"history":   history,
				"input_ids": []int{99},
			})
			So(err, ShouldBeNil)

			So(rope.Metadata()["position_start"], ShouldEqual, 3)

			values, ok := positions.Metadata()["values"].([]float64)
			So(ok, ShouldBeTrue)
			So(values, ShouldResemble, []float64{3})
		})
	})
}

func TestKVCacheStateLifecycle(t *testing.T) {
	Convey("Given a kv_cache state object", t, func() {
		stateObject, err := state.Default.Build(
			"kv_cache", "main", map[string]any{"capacity": 128},
		)
		So(err, ShouldBeNil)
		kvState := stateObject.(*KVCacheState)

		Convey("It should expose a usable underlying KV cache", func() {
			So(kvState.Cache(), ShouldNotBeNil)
		})

		Convey("Reset should bump the epoch", func() {
			before := kvState.Cache().Epoch()
			So(kvState.Reset(context.Background()), ShouldBeNil)
			So(kvState.Cache().Epoch(), ShouldBeGreaterThan, before)
		})

		Convey("Inspect should expose the current epoch", func() {
			inspection, err := kvState.Inspect(context.Background())
			So(err, ShouldBeNil)
			So(inspection.Type, ShouldEqual, "kv_cache")
			So(inspection.Values["epoch"], ShouldNotBeNil)
		})

		Convey("Snapshot should report not-implemented for now", func() {
			_, err := kvState.Snapshot(context.Background())
			So(err, ShouldNotBeNil)
		})
	})
}
