package compute

import (
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func TestNewBackend(test *testing.T) {
	Convey("Given a compute backend", test, func() {
		backend := NewBackend(CPU)
		defer func() {
			So(backend.Close(), ShouldBeNil)
		}()

		Convey("It should expose runner execution through the backend facade", func() {
			shape, err := tensor.NewShape([]int{2})
			So(err, ShouldBeNil)

			graph := ir.NewGraph()
			input := ir.NewNode("input", ir.OpInput, shape)
			input.SetMetadata("values", []float64{1, 2})
			graph.AddNode(input)

			outputs, err := backend.Execute(context.Background(), graph, []*ir.Node{input})
			So(err, ShouldBeNil)

			values, err := outputs["input"].CloneFloat64()
			So(err, ShouldBeNil)
			So(values, ShouldResemble, []float64{1, 2})
			So(outputs["input"].Close(), ShouldBeNil)
			So(backend.Location(), ShouldEqual, tensor.Host)
		})

		Convey("It should reject unsupported backend types instead of silently using CPU", func() {
			So(func() { NewBackend(BackendType(255)) }, ShouldPanic)
		})
	})
}

func BenchmarkNewBackend(benchmark *testing.B) {
	for benchmark.Loop() {
		backend := NewBackend(CPU)
		_ = backend.Close()
	}
}
