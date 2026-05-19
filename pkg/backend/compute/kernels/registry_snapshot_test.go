package kernels

import (
	"testing"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
	. "github.com/smartystreets/goconvey/convey"
)

func TestSnapshot(t *testing.T) {
	Convey("Given an isolated registry with one kernel", t, func() {
		registry := NewRegistry()

		registry.Register(Kernel{
			Name: "add",
			Signature: Signature{
				Layout:  tensor.LayoutDense,
				Inputs:  []dtype.DType{dtype.Float32, dtype.Float32},
				Outputs: []dtype.DType{dtype.Float32},
			},
			Locations: []tensor.Location{tensor.Metal},
			Run:       func(args ...tensor.Tensor) error { return nil },
		})

		Convey("It should return a snapshot containing the kernel", func() {
			entries := registry.Snapshot()

			So(len(entries), ShouldEqual, 1)
			So(entries[0].Name, ShouldEqual, "add")
		})
	})
}

func BenchmarkSnapshot(b *testing.B) {
	registry := NewRegistry()

	registry.Register(Kernel{
		Name: "add",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       func(args ...tensor.Tensor) error { return nil },
	})

	for b.Loop() {
		_ = registry.Snapshot()
	}
}
