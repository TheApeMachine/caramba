package attention

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
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

		err := RunAttentionFloat32(query, key, value, out)

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
