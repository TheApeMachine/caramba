//go:build amd64

package activation

import (
	"math/rand"
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"golang.org/x/sys/cpu"
)

func TestReLUF32AVX512(t *testing.T) {
	if !cpu.X86.HasAVX512F {
		t.Skip("AVX512F not supported")
	}

	convey.Convey("Given ReLUF32AVX512", t, func() {
		sizes := []int{1, 7, 64, 1024, 8192}

		for _, n := range sizes {
			convey.Convey("It should match scalar reference for N="+fmt.Sprintf("%d", n), func() {
				src := make([]float32, n)
				expected := make([]float32, n)
				actual := make([]float32, n)

				for i := 0; i < n; i++ {
					src[i] = rand.Float32()*4.0 - 2.0
				}

				ReLUF32Generic(&expected[0], &src[0], n)
				ReLUF32AVX512(&actual[0], &src[0], n)

				for i := 0; i < n; i++ {
					diff := ulpDiffF32(expected[i], actual[i])
					convey.So(diff, convey.ShouldBeLessThanOrEqualTo, uint32(1))
				}
			})
		}
	})
}

func TestExpF32AVX512(t *testing.T) {
	if !cpu.X86.HasAVX512F {
		t.Skip("AVX512F not supported")
	}

	convey.Convey("Given ExpF32AVX512", t, func() {
		sizes := []int{1, 7, 64, 1024, 8192}

		for _, n := range sizes {
			convey.Convey("It should match scalar reference for N="+fmt.Sprintf("%d", n), func() {
				src := make([]float32, n)
				expected := make([]float32, n)
				actual := make([]float32, n)

				for i := 0; i < n; i++ {
					src[i] = rand.Float32()*4.0 - 2.0
				}

				ExpF32Generic(&expected[0], &src[0], n)
				ExpF32AVX512(&actual[0], &src[0], n)

				for i := 0; i < n; i++ {
					diff := ulpDiffF32(expected[i], actual[i])
					convey.So(diff, convey.ShouldBeLessThanOrEqualTo, uint32(2))
				}
			})
		}
	})
}

func TestHardTanhF32AVX512(t *testing.T) {
	if !cpu.X86.HasAVX512F {
		t.Skip("AVX512F not supported")
	}

	convey.Convey("Given HardTanhF32AVX512", t, func() {
		sizes := []int{1, 7, 64, 1024, 8192}

		for _, n := range sizes {
			convey.Convey("It should match scalar reference for N="+fmt.Sprintf("%d", n), func() {
				src := make([]float32, n)
				expected := make([]float32, n)
				actual := make([]float32, n)

				for i := 0; i < n; i++ {
					src[i] = rand.Float32()*4.0 - 2.0
				}

				HardTanhF32Generic(&expected[0], &src[0], n)
				HardTanhF32AVX512(&actual[0], &src[0], n)

				for i := 0; i < n; i++ {
					diff := ulpDiffF32(expected[i], actual[i])
					convey.So(diff, convey.ShouldBeLessThanOrEqualTo, uint32(1))
				}
			})
		}
	})
}

func BenchmarkReLUF32AVX512(b *testing.B) {
	if !cpu.X86.HasAVX512F {
		b.Skip("AVX512F not supported")
	}

	n := 8192
	src := make([]float32, n)
	dst := make([]float32, n)

	for i := 0; i < n; i++ {
		src[i] = rand.Float32()*4.0 - 2.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ReLUF32AVX512(&dst[0], &src[0], n)
	}
}

func BenchmarkExpF32AVX512(b *testing.B) {
	if !cpu.X86.HasAVX512F {
		b.Skip("AVX512F not supported")
	}

	n := 8192
	src := make([]float32, n)
	dst := make([]float32, n)

	for i := 0; i < n; i++ {
		src[i] = rand.Float32()*4.0 - 2.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ExpF32AVX512(&dst[0], &src[0], n)
	}
}

func BenchmarkHardTanhF32AVX512(b *testing.B) {
	if !cpu.X86.HasAVX512F {
		b.Skip("AVX512F not supported")
	}

	n := 8192
	src := make([]float32, n)
	dst := make([]float32, n)

	for i := 0; i < n; i++ {
		src[i] = rand.Float32()*4.0 - 2.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		HardTanhF32AVX512(&dst[0], &src[0], n)
	}
}
