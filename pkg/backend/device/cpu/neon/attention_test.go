package neon

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func TestAttentionFloat32(t *testing.T) {
	convey.Convey("Given a 2x3 query/key with identity-like keys", t, func() {
		queryShape, _ := tensor.NewShape([]int{2, 3})
		keyShape, _ := tensor.NewShape([]int{2, 3})
		valueShape, _ := tensor.NewShape([]int{2, 4})
		outShape, _ := tensor.NewShape([]int{2, 4})

		query, _ := tensor.NewZeroed(queryShape, dtype.Float32)
		key, _ := tensor.NewZeroed(keyShape, dtype.Float32)
		value, _ := tensor.NewZeroed(valueShape, dtype.Float32)
		out, _ := tensor.NewZeroed(outShape, dtype.Float32)

		queryView, _ := query.Float32Native()
		keyView, _ := key.Float32Native()
		valueView, _ := value.Float32Native()

		// Query row 0 strongly matches key 0; row 1 strongly matches key 1.
		copy(queryView, []float32{
			1, 0, 0,
			0, 1, 0,
		})

		copy(keyView, []float32{
			10, 0, 0,
			0, 10, 0,
		})

		// Distinct values per key.
		copy(valueView, []float32{
			1, 2, 3, 4,
			5, 6, 7, 8,
		})

		kernel, _ := Default.Lookup("attention", Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Float32, dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		})

		err := kernel.Run(query, key, value, out)

		convey.Convey("Output rows should be close to the matching value rows", func() {
			convey.So(err, convey.ShouldBeNil)

			outView, _ := out.Float32Native()

			// Row 0 should be close to [1, 2, 3, 4] (value[0]).
			// Row 1 should be close to [5, 6, 7, 8] (value[1]).
			// "Close" because softmax is sharp but not exact at this scale.
			for index, expected := range []float32{1, 2, 3, 4} {
				delta := outView[index] - expected

				if delta < 0 {
					delta = -delta
				}

				convey.So(delta, convey.ShouldBeLessThan, float32(0.1))
			}
		})
	})
}

func TestAdamStepFloat32(t *testing.T) {
	convey.Convey("Given a one-step Adam update with default config", t, func() {
		shape, _ := tensor.NewShape([]int{2})

		params, _ := tensor.NewZeroed(shape, dtype.Float32)
		grads, _ := tensor.NewZeroed(shape, dtype.Float32)
		firstMoment, _ := tensor.NewZeroed(shape, dtype.Float32)
		secondMoment, _ := tensor.NewZeroed(shape, dtype.Float32)
		out, _ := tensor.NewZeroed(shape, dtype.Float32)

		paramsView, _ := params.Float32Native()
		gradsView, _ := grads.Float32Native()

		paramsView[0] = 1.0
		paramsView[1] = -1.0
		gradsView[0] = 0.1
		gradsView[1] = -0.1

		err := AdamStepFloat32(
			DefaultAdamConfig(),
			params, grads, firstMoment, secondMoment, out,
		)

		convey.Convey("Output should be a small step against the gradient direction", func() {
			convey.So(err, convey.ShouldBeNil)

			outView, _ := out.Float32Native()

			// With lr=1e-4 the update is ~1e-4 in magnitude in the
			// direction opposite to the gradient.
			convey.So(outView[0] < paramsView[0], convey.ShouldBeTrue)
			convey.So(outView[1] > paramsView[1], convey.ShouldBeTrue)
		})
	})
}
