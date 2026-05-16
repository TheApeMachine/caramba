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
		backend, err := NewBackend(CPU)
		So(err, ShouldBeNil)
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
			backend, err := NewBackend(BackendType(255))

			So(backend, ShouldBeNil)
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "unsupported backend type")
		})
	})
}

func BenchmarkNewBackend(benchmark *testing.B) {
	for benchmark.Loop() {
		backend, err := NewBackend(CPU)
		if err != nil {
			benchmark.Fatal(err)
		}

		_ = backend.Close()
	}
}
