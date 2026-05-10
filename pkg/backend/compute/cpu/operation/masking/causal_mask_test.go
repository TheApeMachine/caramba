package masking

import (
	"math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestCausalMask(t *testing.T) {
	Convey("Given a CausalMask operation", t, func() {
		op := NewCausalMask()

		Convey("Forward", func() {
			Convey("It should produce a lower-triangular mask of zeros with -inf above diagonal", func() {
				seq := 4
				out := op.Forward([]int{seq})
				So(out, ShouldHaveLength, seq*seq)
				for i := 0; i < seq; i++ {
					for j := 0; j < seq; j++ {
						v := out[i*seq+j]
						if j <= i {
							So(v, ShouldEqual, 0.0)
						} else {
							So(math.IsInf(v, -1), ShouldBeTrue)
						}
					}
				}
			})

			Convey("It should produce a 1x1 mask of zero for seq_len=1", func() {
				out := op.Forward([]int{1})
				So(out, ShouldResemble, []float64{0.0})
			})
		})
	})
}

func BenchmarkCausalMask_Forward(b *testing.B) {
	op := NewCausalMask()
	shape := []int{512}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		op.Forward(shape)
	}
}
