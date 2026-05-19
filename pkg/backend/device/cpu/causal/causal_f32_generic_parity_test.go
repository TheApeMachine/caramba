package causal

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
)

func randomCausalFloat32Slice(length int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	values := make([]float32, length)

	for index := range values {
		values[index] = float32((rng.Float64() - 0.5) * 4)
	}

	return values
}

func TestCateF32GenericParity(t *testing.T) {
	convey.Convey("Given cateF32Generic", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should compute treated-control for N=%d", length), func() {
				treated := randomCausalFloat32Slice(length, 0xCA1E+int64(length))
				control := randomCausalFloat32Slice(length, 0xCA1F+int64(length))
				got := make([]float32, length)
				want := make([]float32, length)

				cateF32Generic(treated, control, want)
				cateF32Generic(treated, control, got)

				parity.AssertFloat32SlicesWithinULP(t, got, want, 0)
			})
		}
	})
}

func TestCounterfactualF32GenericParity(t *testing.T) {
	convey.Convey("Given counterfactualF32Generic", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match reference for N=%d", length), func() {
				observedY := randomCausalFloat32Slice(length, 0xCA20+int64(length))
				observedX := randomCausalFloat32Slice(length, 0xCA21+int64(length))
				counterfactualX := randomCausalFloat32Slice(length, 0xCA22+int64(length))
				got := make([]float32, length)
				want := make([]float32, length)
				const slope = float32(0.75)

				counterfactualF32Generic(got, observedY, observedX, counterfactualX, slope)
				counterfactualF32Generic(want, observedY, observedX, counterfactualX, slope)

				parity.AssertFloat32SlicesWithinULP(t, got, want, 0)
			})
		}
	})
}

func TestStridedDotF32GenericParity(t *testing.T) {
	convey.Convey("Given stridedDotF32Generic", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should be deterministic for N=%d", length), func() {
				values := randomCausalFloat32Slice(length*3, 0xCA23+int64(length))
				weights := randomCausalFloat32Slice(length, 0xCA24+int64(length))
				const stride = 3

				first := stridedDotF32Generic(values, stride, weights, length)
				second := stridedDotF32Generic(values, stride, weights, length)

				parity.AssertFloat32SlicesWithinULP(
					t,
					[]float32{first},
					[]float32{second},
					0,
				)
			})
		}
	})
}

func BenchmarkCateF32Generic(b *testing.B) {
	const length = 8192
	treated := randomCausalFloat32Slice(length, 1)
	control := randomCausalFloat32Slice(length, 2)
	out := make([]float32, length)

	for b.Loop() {
		cateF32Generic(treated, control, out)
	}
}

func BenchmarkCounterfactualF32Generic(b *testing.B) {
	const length = 8192
	observedY := randomCausalFloat32Slice(length, 1)
	observedX := randomCausalFloat32Slice(length, 2)
	counterfactualX := randomCausalFloat32Slice(length, 3)
	out := make([]float32, length)

	for b.Loop() {
		counterfactualF32Generic(out, observedY, observedX, counterfactualX, 0.5)
	}
}

func BenchmarkStridedDotF32Generic(b *testing.B) {
	const length = 8192
	const stride = 7
	values := randomCausalFloat32Slice(length*stride, 1)
	weights := randomCausalFloat32Slice(length, 2)

	for b.Loop() {
		_ = stridedDotF32Generic(values, stride, weights, length)
	}
}
