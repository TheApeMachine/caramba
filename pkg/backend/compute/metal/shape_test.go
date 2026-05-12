//go:build darwin && cgo

package metal

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	cpushape "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/shape"
)

func TestMetalShapeOpsForward(t *testing.T) {
	Convey("Given MetalShapeOps Forward dispatch", t, func() {
		ops := &MetalShapeOps{}

		Convey("It should reject unsupported operation metadata", func() {
			output, err := ops.Forward(
				ShapeForwardRequest{Op: "shape.unknown"},
				[]int{2, 2},
				[]float64{1, 2, 3, 4},
			)

			So(err, ShouldNotBeNil)
			So(output, ShouldBeNil)
		})
	})
}

func TestMetalShapeOpsForwardParity(t *testing.T) {
	lib := metallibPathOrSkip(t, "shape.metallib")

	Convey("Given initialized MetalShapeOps Forward dispatch", t, func() {
		ops, err := NewShapeOps(lib)
		So(err, ShouldBeNil)

		Convey("It should route transpose metadata to the Metal transpose kernel", func() {
			input := []float64{1, 2, 3, 4, 5, 6}
			expected := cpushape.NewTranspose(0, 1).Forward([]int{2, 3}, input)

			output, err := ops.Forward(
				ShapeForwardRequest{
					Op: "shape.transpose",
					Metadata: map[string]any{
						"dim0": 0,
						"dim1": 1,
					},
				},
				[]int{2, 3},
				input,
			)

			So(err, ShouldBeNil)
			So(output, ShouldResemble, expected)
		})

		Convey("It should route view_as_heads metadata to the Metal head-view kernel", func() {
			input := []float64{1, 2, 3, 4, 5, 6, 7, 8}
			expected := cpushape.NewViewAsHeads(2).Forward([]int{1, 2, 4}, input)

			output, err := ops.Forward(
				ShapeForwardRequest{
					Op: "shape.view_as_heads",
					Metadata: map[string]any{
						"num_heads": 2,
					},
				},
				[]int{1, 2, 4},
				input,
			)

			So(err, ShouldBeNil)
			So(output, ShouldResemble, expected)
		})

		Convey("It should route merge_heads metadata to the Metal merge-heads kernel", func() {
			input := []float64{1, 3, 5, 7, 2, 4, 6, 8}
			expected := cpushape.NewMergeHeads().Forward([]int{1, 2, 2, 2}, input)

			output, err := ops.Forward(
				ShapeForwardRequest{Op: "shape.merge_heads"},
				[]int{1, 2, 2, 2},
				input,
			)

			So(err, ShouldBeNil)
			So(output, ShouldResemble, expected)
		})
	})
}

func BenchmarkMetalShapeOpsForwardValidation(b *testing.B) {
	ops := &MetalShapeOps{}
	input := []float64{1, 2, 3, 4}

	for b.Loop() {
		_, _ = ops.Forward(ShapeForwardRequest{Op: "shape.unknown"}, []int{2, 2}, input)
	}
}
