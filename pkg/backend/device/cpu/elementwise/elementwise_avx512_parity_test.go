//go:build amd64

package elementwise

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
	"golang.org/x/sys/cpu"
)

func avx512ElementwiseAvailable() bool {
	return cpu.X86.HasAVX512F
}

func TestAddF32AVX512Parity(t *testing.T) {
	if !avx512ElementwiseAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given AddF32AVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match AddF32Generic for N=%d", length), func() {
				left := parityRandomFloat32Slice(length, 0xA11CE+int64(length))
				right := parityRandomFloat32Slice(length, 0xB0B+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				AddF32Generic(&want[0], &left[0], &right[0], length)
				AddF32AVX512(&got[0], &left[0], &right[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, 0)
			})
		}
	})
}

func TestSubF32AVX512Parity(t *testing.T) {
	if !avx512ElementwiseAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given SubF32AVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match SubF32Generic for N=%d", length), func() {
				left := parityRandomFloat32Slice(length, 0xC0FFEE+int64(length))
				right := parityRandomFloat32Slice(length, 0xDEAF+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				SubF32Generic(&want[0], &left[0], &right[0], length)
				SubF32AVX512(&got[0], &left[0], &right[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, 0)
			})
		}
	})
}

func TestMulF32AVX512Parity(t *testing.T) {
	if !avx512ElementwiseAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given MulF32AVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match MulF32Generic for N=%d", length), func() {
				left := parityRandomFloat32Slice(length, 0xFEED+int64(length))
				right := parityRandomFloat32Slice(length, 0xBEEF+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				MulF32Generic(&want[0], &left[0], &right[0], length)
				MulF32AVX512(&got[0], &left[0], &right[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, 0)
			})
		}
	})
}

func TestDivF32AVX512Parity(t *testing.T) {
	if !avx512ElementwiseAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given DivF32AVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match DivF32Generic for N=%d", length), func() {
				left := parityRandomFloat32Slice(length, 0x1234+int64(length))
				right := parityRandomNonZeroFloat32Slice(length, 0x5678+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				DivF32Generic(&want[0], &left[0], &right[0], length)
				DivF32AVX512(&got[0], &left[0], &right[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, 0)
			})
		}
	})
}

func TestMaxF32AVX512Parity(t *testing.T) {
	if !avx512ElementwiseAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given MaxF32AVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match MaxF32Generic for N=%d", length), func() {
				left := parityRandomFloat32Slice(length, 0xAAAA+int64(length))
				right := parityRandomFloat32Slice(length, 0xBBBB+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				MaxF32Generic(&want[0], &left[0], &right[0], length)
				MaxF32AVX512(&got[0], &left[0], &right[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, 0)
			})
		}
	})
}

func TestMinF32AVX512Parity(t *testing.T) {
	if !avx512ElementwiseAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given MinF32AVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match MinF32Generic for N=%d", length), func() {
				left := parityRandomFloat32Slice(length, 0xCCCC+int64(length))
				right := parityRandomFloat32Slice(length, 0xDDDD+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				MinF32Generic(&want[0], &left[0], &right[0], length)
				MinF32AVX512(&got[0], &left[0], &right[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, 0)
			})
		}
	})
}

func TestAbsF32AVX512Parity(t *testing.T) {
	if !avx512ElementwiseAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given AbsF32AVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match math.Abs scalar for N=%d", length), func() {
				source := parityRandomFloat32Slice(length, 0xABBA+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				for index, value := range source {
					want[index] = float32(math.Abs(float64(value)))
				}

				AbsF32AVX512(&got[0], &source[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, 0)
			})
		}
	})
}

func TestNegF32AVX512Parity(t *testing.T) {
	if !avx512ElementwiseAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given NegF32AVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match NegF32Generic for N=%d", length), func() {
				source := parityRandomFloat32Slice(length, 0xBADBAD+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				NegF32Generic(&want[0], &source[0], length)
				NegF32AVX512(&got[0], &source[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, 0)
			})
		}
	})
}

func TestSqrtF32AVX512Parity(t *testing.T) {
	if !avx512ElementwiseAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given SqrtF32AVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match SqrtF32Generic for N=%d", length), func() {
				source := parityRandomNonNegativeFloat32Slice(length, 0xC0DE+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				SqrtF32Generic(&want[0], &source[0], length)
				SqrtF32AVX512(&got[0], &source[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, 0)
			})
		}
	})
}

func TestReluF32AVX512Parity(t *testing.T) {
	if !avx512ElementwiseAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given ReluF32AVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match ReluF32Generic for N=%d", length), func() {
				source := parityRandomFloat32Slice(length, 0xDEADBEEF+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				ReluF32Generic(&want[0], &source[0], length)
				ReluF32AVX512(&got[0], &source[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, 0)
			})
		}
	})
}

func TestAxpyF32AVX512Parity(t *testing.T) {
	if !avx512ElementwiseAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given AxpyF32AVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match FMA scalar reference within 2 ULP for N=%d", length), func() {
				yInit := parityRandomFloat32Slice(length, 0xAA00+int64(length))
				x := parityRandomFloat32Slice(length, 0xBB00+int64(length))
				alpha := float32(0.7654321)
				want := make([]float32, length)
				got := append([]float32(nil), yInit...)

				for index := range yInit {
					want[index] = float32(math.FMA(float64(alpha), float64(x[index]), float64(yInit[index])))
				}

				AxpyF32AVX512(&got[0], &x[0], alpha, length)
				parity.AssertFloat32SlicesWithinULP(t, want, got, 2)
			})
		}
	})
}

func parityRandomFloat32Slice(length int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	out := make([]float32, length)

	for index := range out {
		out[index] = float32((rng.Float64() - 0.5) * math.Pow(10, rng.Float64()*4-2))
	}

	return out
}

func parityRandomNonZeroFloat32Slice(length int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	out := make([]float32, length)

	for index := range out {
		value := float32((rng.Float64() - 0.5) * math.Pow(10, rng.Float64()*4-2))

		if value == 0 {
			value = 1
		}

		out[index] = value
	}

	return out
}

func parityRandomNonNegativeFloat32Slice(length int, seed int64) []float32 {
	out := parityRandomFloat32Slice(length, seed)

	for index, value := range out {
		if value < 0 {
			out[index] = -value
		}
	}

	return out
}
