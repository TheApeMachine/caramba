package backend

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"

	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/runtime/state"
)

func TestNewStateBindingHook(t *testing.T) {
	Convey("Given an IR graph with state-bound attention and RoPE nodes", t, func() {
		shape, err := tensor.NewShape([]int{1, 1, 1, 4})
		So(err, ShouldBeNil)

		graph := ir.NewGraph()
		sdpa := ir.NewNode("attn", ir.OpType(opIDAttentionSDPA), shape)
		sdpa.SetOperationID(opIDAttentionSDPA)
		sdpa.SetMetadata("causal", true)
		rope := ir.NewNode("rope", ir.OpType(opIDRoPEPositionStart), shape)
		rope.SetOperationID(opIDRoPEPositionStart)
		graph.AddNode(sdpa)
		graph.AddNode(rope)

		hook := NewStateBindingHook()

		Convey("It should attach a manifest-declared KV cache input", func() {
			kvState, err := state.Default.Build("kv_cache", "kv", nil)
			So(err, ShouldBeNil)

			err = hook(graph, map[string]any{"kv_cache": kvState})
			So(err, ShouldBeNil)

			injected, ok := sdpa.Metadata()["kv_cache"]
			So(ok, ShouldBeTrue)
			So(injected, ShouldEqual, kvState.(*state.KVCache).Cache())
		})

		Convey("It should ignore non-causal attention nodes", func() {
			sdpa.SetMetadata("causal", false)
			kvState, err := state.Default.Build("kv_cache", "kv", nil)
			So(err, ShouldBeNil)

			err = hook(graph, map[string]any{"kv_cache": kvState})
			So(err, ShouldBeNil)

			_, ok := sdpa.Metadata()["kv_cache"]
			So(ok, ShouldBeFalse)
		})

		Convey("It should bind an explicit position_start counter", func() {
			position, err := state.Default.Build("counter", "position", map[string]any{
				"initial": 11,
			})
			So(err, ShouldBeNil)

			err = hook(graph, map[string]any{"position_start": position})
			So(err, ShouldBeNil)
			So(rope.Metadata()["position_start"], ShouldEqual, 11)
		})

		Convey("It should derive position_start from history and input length", func() {
			history, err := state.Default.Build("token_buffer", "history", nil)
			So(err, ShouldBeNil)
			history.(*state.TokenBuffer).Extend([]int{10, 20, 30, 40})

			err = hook(graph, map[string]any{
				"history":   history,
				"input_ids": []int{30, 40},
			})
			So(err, ShouldBeNil)
			So(rope.Metadata()["position_start"], ShouldEqual, 2)
		})
	})
}

func BenchmarkNewStateBindingHook(benchmark *testing.B) {
	shape, err := tensor.NewShape([]int{1, 1, 1, 4})

	if err != nil {
		benchmark.Fatal(err)
	}

	graph := ir.NewGraph()
	sdpa := ir.NewNode("attn", ir.OpType(opIDAttentionSDPA), shape)
	sdpa.SetOperationID(opIDAttentionSDPA)
	sdpa.SetMetadata("causal", true)
	graph.AddNode(sdpa)

	kvState, err := state.Default.Build("kv_cache", "kv", nil)

	if err != nil {
		benchmark.Fatal(err)
	}

	hook := NewStateBindingHook()
	inputs := map[string]any{"kv_cache": kvState}

	for benchmark.Loop() {
		if err := hook(graph, inputs); err != nil {
			benchmark.Fatal(err)
		}
	}
}
