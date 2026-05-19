//go:build amd64

package activation

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"golang.org/x/sys/cpu"
)

func TestParamExtraAVX512Parity(t *testing.T) {
	if !cpu.X86.HasAVX512F {
		t.Skip("AVX512F not supported")
	}

	sizes := []int{1, 7, 64, 1024, 8192}

	testCases := []struct {
		name    string
		generic func(dst, src *float32, count int)
		avx512  func(dst, src *float32, count int)
	}{
		{
			name: "CELUAlphaF32",
			generic: func(dst, src *float32, count int) {
				CELUAlphaF32Generic(dst, src, count, 1.5)
			},
			avx512: func(dst, src *float32, count int) {
				CELUAlphaF32AVX512(dst, src, count, 1.5)
			},
		},
		{
			name: "HardShrinkF32",
			generic: func(dst, src *float32, count int) {
				HardShrinkF32Generic(dst, src, count, 0.5)
			},
			avx512: func(dst, src *float32, count int) {
				HardShrinkF32AVX512(dst, src, count, 0.5)
			},
		},
		{
			name: "SoftShrinkF32",
			generic: func(dst, src *float32, count int) {
				SoftShrinkF32Generic(dst, src, count, 0.5)
			},
			avx512: func(dst, src *float32, count int) {
				SoftShrinkF32AVX512(dst, src, count, 0.5)
			},
		},
		{
			name: "SnakeF32",
			generic: func(dst, src *float32, count int) {
				SnakeF32Generic(dst, src, count, 0.5)
			},
			avx512: func(dst, src *float32, count int) {
				SnakeF32AVX512(dst, src, count, 0.5)
			},
		},
		{
			name: "SnakeParametricF32",
			generic: func(dst, src *float32, count int) {
				SnakeParametricF32Generic(dst, src, count, 0.5, 0.2)
			},
			avx512: func(dst, src *float32, count int) {
				SnakeParametricF32AVX512(dst, src, count, 0.5, 0.2)
			},
		},
		{
			name: "RReLUF32",
			generic: func(dst, src *float32, count int) {
				RReLUF32Generic(dst, src, count, 0.1, 0.3)
			},
			avx512: func(dst, src *float32, count int) {
				RReLUF32AVX512(dst, src, count, 0.1, 0.3)
			},
		},
	}

	for _, tc := range testCases {
		convey.Convey(fmt.Sprintf("Given %sAVX512", tc.name), t, func() {
			for _, n := range sizes {
				convey.Convey(fmt.Sprintf("It should match scalar reference for N=%d", n), func() {
					src := make([]float32, n)
					expected := make([]float32, n)
					actual := make([]float32, n)

					for i := 0; i < n; i++ {
						src[i] = rand.Float32()*10.0 - 5.0
					}

					tc.generic(&expected[0], &src[0], n)
					tc.avx512(&actual[0], &src[0], n)

					for i := 0; i < n; i++ {
						diff := ulpDiffF32(expected[i], actual[i])
						if diff > 10 {
							convey.So(diff, convey.ShouldBeLessThanOrEqualTo, uint32(10))
						}
					}
				})
			}
		})
	}
}

func benchmarkParamExtraKernel(b *testing.B, name string, kernel func(dst, src *float32, count int)) {
	if !cpu.X86.HasAVX512F {
		b.Skip("AVX512F not supported")
	}
	b.Run(name, func(b *testing.B) {
		n := 8192
		src := make([]float32, n)
		dst := make([]float32, n)
		for i := 0; i < n; i++ {
			src[i] = rand.Float32()*10.0 - 5.0
		}
		b.ResetTimer()
		for b.Loop() {
			kernel(&dst[0], &src[0], n)
		}
	})
}

func BenchmarkParamExtraAVX512(b *testing.B) {
	benchmarkParamExtraKernel(b, "CELUAlphaF32_Generic", func(dst, src *float32, count int) { CELUAlphaF32Generic(dst, src, count, 1.5) })
	benchmarkParamExtraKernel(b, "CELUAlphaF32_AVX512", func(dst, src *float32, count int) { CELUAlphaF32AVX512(dst, src, count, 1.5) })
	benchmarkParamExtraKernel(b, "HardShrinkF32_Generic", func(dst, src *float32, count int) { HardShrinkF32Generic(dst, src, count, 0.5) })
	benchmarkParamExtraKernel(b, "HardShrinkF32_AVX512", func(dst, src *float32, count int) { HardShrinkF32AVX512(dst, src, count, 0.5) })
	benchmarkParamExtraKernel(b, "SoftShrinkF32_Generic", func(dst, src *float32, count int) { SoftShrinkF32Generic(dst, src, count, 0.5) })
	benchmarkParamExtraKernel(b, "SoftShrinkF32_AVX512", func(dst, src *float32, count int) { SoftShrinkF32AVX512(dst, src, count, 0.5) })
	benchmarkParamExtraKernel(b, "SnakeF32_Generic", func(dst, src *float32, count int) { SnakeF32Generic(dst, src, count, 0.5) })
	benchmarkParamExtraKernel(b, "SnakeF32_AVX512", func(dst, src *float32, count int) { SnakeF32AVX512(dst, src, count, 0.5) })
	benchmarkParamExtraKernel(b, "SnakeParametricF32_Generic", func(dst, src *float32, count int) { SnakeParametricF32Generic(dst, src, count, 0.5, 0.2) })
	benchmarkParamExtraKernel(b, "SnakeParametricF32_AVX512", func(dst, src *float32, count int) { SnakeParametricF32AVX512(dst, src, count, 0.5, 0.2) })
	benchmarkParamExtraKernel(b, "RReLUF32_Generic", func(dst, src *float32, count int) { RReLUF32Generic(dst, src, count, 0.1, 0.3) })
	benchmarkParamExtraKernel(b, "RReLUF32_AVX512", func(dst, src *float32, count int) { RReLUF32AVX512(dst, src, count, 0.1, 0.3) })
}
