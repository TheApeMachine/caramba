//go:build amd64

package reduction

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
	"golang.org/x/sys/cpu"
)

func avx512ReductionAvailable() bool {
	return cpu.X86.HasAVX512F
}

func randomReductionFloat32Slice(length int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	slice := make([]float32, length)

	for index := range slice {
		slice[index] = float32((rng.Float64() - 0.5) * math.Pow(10, rng.Float64()*4-2))
	}

	return slice
}

func assertSumF32Parity(
	testingTB *testing.T,
	got, want float32,
	length int,
) {
	testingTB.Helper()

	tolerance := math.Max(math.Abs(float64(want)), 1.0) * float64(length) * 0x1p-50

	if math.Abs(float64(got-want)) > tolerance {
		testingTB.Fatalf(
			"N=%d got=%g want=%g diff=%g tol=%g",
			length, got, want, got-want, tolerance,
		)
	}
}

func assertProdF32Parity(
	testingTB *testing.T,
	got, want float32,
) {
	testingTB.Helper()

	leftBits := math.Float32bits(got)
	rightBits := math.Float32bits(want)

	if leftBits == rightBits {
		return
	}

	if leftBits > rightBits {
		leftBits, rightBits = rightBits, leftBits
	}

	if int64(rightBits)-int64(leftBits) > 16 {
		testingTB.Fatalf("got=%g want=%g ulp gap > 16", got, want)
	}
}

func TestSumF32AVX512Parity(t *testing.T) {
	if !avx512ReductionAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given SumF32AVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match SumF32Generic for N=%d", length), func() {
				values := randomReductionFloat32Slice(length, 0x510+int64(length))
				want := SumF32Generic(&values[0], length)
				got := SumF32AVX512(&values[0], length)

				assertSumF32Parity(t, got, want, length)
			})
		}
	})
}

func TestProdF32AVX512Parity(t *testing.T) {
	if !avx512ReductionAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given ProdF32AVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match ProdF32Generic for N=%d", length), func() {
				values := randomReductionFloat32Slice(length, 0xB10+int64(length))

				for index := range values {
					values[index] = float32(0.5 + float64(index%17)*0.01)
				}

				want := ProdF32Generic(&values[0], length)
				got := ProdF32AVX512(&values[0], length)

				assertProdF32Parity(t, got, want)
			})
		}
	})
}

func TestMaxF32AVX512Parity(t *testing.T) {
	if !avx512ReductionAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given MaxF32AVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match MaxF32Generic for N=%d", length), func() {
				values := randomReductionFloat32Slice(length, 0xA22+int64(length))
				want := MaxF32Generic(&values[0], length)
				got := MaxF32AVX512(&values[0], length)

				if math.Float32bits(got) != math.Float32bits(want) {
					t.Fatalf("N=%d got=%g want=%g", length, got, want)
				}
			})
		}
	})
}

func TestMinF32AVX512Parity(t *testing.T) {
	if !avx512ReductionAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given MinF32AVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match MinF32Generic for N=%d", length), func() {
				values := randomReductionFloat32Slice(length, 0xB22+int64(length))
				want := MinF32Generic(&values[0], length)
				got := MinF32AVX512(&values[0], length)

				if math.Float32bits(got) != math.Float32bits(want) {
					t.Fatalf("N=%d got=%g want=%g", length, got, want)
				}
			})
		}
	})
}

func TestL1NormF32AVX512Parity(t *testing.T) {
	if !avx512ReductionAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given L1NormF32AVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match L1NormF32Generic for N=%d", length), func() {
				values := randomReductionFloat32Slice(length, 0xC10+int64(length))
				want := L1NormF32Generic(&values[0], length)
				got := L1NormF32AVX512(&values[0], length)

				assertSumF32Parity(t, got, want, length)
			})
		}
	})
}
