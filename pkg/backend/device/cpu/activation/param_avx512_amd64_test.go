//go:build amd64

package activation

import (
	"math"
	"math/rand"
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"golang.org/x/sys/cpu"
)

func ulpDiffF32(a, b float32) uint32 {
	if math.IsNaN(float64(a)) && math.IsNaN(float64(b)) {
		return 0
	}
	ia := math.Float32bits(a)
	ib := math.Float32bits(b)
	if int32(ia) < 0 {
		ia = 0x80000000 - ia
	}
	if int32(ib) < 0 {
		ib = 0x80000000 - ib
	}
	if ia > ib {
		return ia - ib
	}
	return ib - ia
}

func TestLeakyReLUSlopeF32AVX512(t *testing.T) {
	if !cpu.X86.HasAVX512F {
		t.Skip("AVX512F not supported")
	}

	convey.Convey("Given LeakyReLUSlopeF32AVX512", t, func() {
		sizes := []int{1, 7, 64, 1024, 8192}
		negativeSlope := float32(0.1)

		for _, n := range sizes {
			convey.Convey("It should match scalar reference for N="+fmt.Sprintf("%d", n), func() {
				src := make([]float32, n)
				expected := make([]float32, n)
				actual := make([]float32, n)

				for i := 0; i < n; i++ {
					src[i] = rand.Float32()*2.0 - 1.0
				}

				LeakyReLUSlopeF32Generic(&expected[0], &src[0], n, negativeSlope)
				LeakyReLUSlopeF32AVX512(&actual[0], &src[0], n, negativeSlope)

				for i := 0; i < n; i++ {
					diff := ulpDiffF32(expected[i], actual[i])
					convey.So(diff, convey.ShouldBeLessThanOrEqualTo, uint32(1))
				}
			})
		}
	})
}

func BenchmarkLeakyReLUSlopeF32AVX512(b *testing.B) {
	if !cpu.X86.HasAVX512F {
		b.Skip("AVX512F not supported")
	}

	n := 8192
	src := make([]float32, n)
	dst := make([]float32, n)
	negativeSlope := float32(0.1)

	for i := 0; i < n; i++ {
		src[i] = rand.Float32()*2.0 - 1.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		LeakyReLUSlopeF32AVX512(&dst[0], &src[0], n, negativeSlope)
	}
}

func BenchmarkLeakyReLUSlopeF32Generic(b *testing.B) {
	n := 8192
	src := make([]float32, n)
	dst := make([]float32, n)
	negativeSlope := float32(0.1)

	for i := 0; i < n; i++ {
		src[i] = rand.Float32()*2.0 - 1.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		LeakyReLUSlopeF32Generic(&dst[0], &src[0], n, negativeSlope)
	}
}