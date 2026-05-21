package runtime

import (
	"context"
	"encoding/binary"
	"math"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/metal"
)

func TestGraphBackendMaterializeInputs(testingObject *testing.T) {
	convey.Convey("Given a scalar float32 graph input", testingObject, func() {
		memory, err := metal.NewBackend(context.Background(), nil)
		convey.So(err, convey.ShouldBeNil)

		backend := &GraphBackend{}
		graph := &ast.Graph{Inputs: []string{"timestep"}}

		convey.Convey("It should upload a one element float32 tensor", func() {
			inputs, err := backend.materializeInputs(memory, graph, map[string]any{
				"timestep": float32(0.5),
			})

			convey.So(err, convey.ShouldBeNil)
			convey.So(inputs["timestep"].Shape().Dims(), convey.ShouldResemble, []int{1})
			convey.So(inputs["timestep"].DType(), convey.ShouldEqual, dtype.Float32)

			_, rawBytes, err := inputs["timestep"].RawBytes()
			convey.So(err, convey.ShouldBeNil)
			convey.So(math.Float32frombits(binary.LittleEndian.Uint32(rawBytes)), convey.ShouldEqual, float32(0.5))
		})

		convey.Reset(func() {
			if memory != nil {
				memory.Close()
			}
		})
	})

	convey.Convey("Given a flat float32 vector feeding a linear graph input", testingObject, func() {
		memory, err := metal.NewBackend(context.Background(), nil)
		convey.So(err, convey.ShouldBeNil)

		backend := &GraphBackend{}
		graph := &ast.Graph{
			Inputs: []string{"hidden_states"},
			Nodes: []*ast.GraphNode{
				{
					ID:     "x_embedder",
					Op:     "projection.linear",
					Inputs: []string{"hidden_states"},
					Attributes: map[string]any{
						"in_features": int64(128),
					},
				},
			},
		}

		convey.Convey("It should preserve the inferred feature width", func() {
			inputs, err := backend.materializeInputs(memory, graph, map[string]any{
				"hidden_states": make([]float32, 4096*128),
			})

			convey.So(err, convey.ShouldBeNil)
			convey.So(inputs["hidden_states"].Shape().Dims(), convey.ShouldResemble, []int{1, 4096, 128})
		})

		convey.Reset(func() {
			if memory != nil {
				memory.Close()
			}
		})
	})
}
