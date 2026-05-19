package matmul

import (
	"fmt"
	"math"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func TestMatMulFloat32(t *testing.T) {
	for _, n := range parityNs {
		n := n

		t.Run(fmt.Sprintf("inner=%d", n), func(t *testing.T) {
			convey.Convey("Output should equal the reference matmul", t, func() {
				leftShape, _ := tensor.NewShape([]int{2, n})
				rightShape, _ := tensor.NewShape([]int{n, 2})
				outShape, _ := tensor.NewShape([]int{2, 2})

				left, _ := tensor.NewZeroed(leftShape, dtype.Float32)
				right, _ := tensor.NewZeroed(rightShape, dtype.Float32)
				out, _ := tensor.NewZeroed(outShape, dtype.Float32)

				leftView, _ := left.Float32Native()
				rightView, _ := right.Float32Native()

				for index := range leftView {
					leftView[index] = float32((index%5)+1) * 0.25
				}

				for index := range rightView {
					rightView[index] = float32((index%7)+1) * 0.5
				}

				err := RunMatMulFloat32(left, right, out)
				convey.So(err, convey.ShouldBeNil)

				outView, _ := out.Float32Native()

				expected := make([]float32, 4)

				for rowIndex := 0; rowIndex < 2; rowIndex++ {
					for colIndex := 0; colIndex < 2; colIndex++ {
						var sum float32

						for innerIndex := 0; innerIndex < n; innerIndex++ {
							sum += leftView[rowIndex*n+innerIndex] *
								rightView[innerIndex*2+colIndex]
						}

						expected[rowIndex*2+colIndex] = sum
					}
				}

				for index, want := range expected {
					convey.So(outView[index], convey.ShouldAlmostEqual, want, math.Max(1e-3, float64(want)*1e-4))
				}
			})
		})
	}
}
