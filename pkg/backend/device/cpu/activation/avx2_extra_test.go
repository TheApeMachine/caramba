//go:build amd64

package activation

import (
	"fmt"
	"math"
	"testing"

	"github.com/smartystreets/goconvey/convey"
)

func abs(a int) int {
	if a < 0 {
		return -a
	}
	return a
}

func ulpDiff(a, b float32) int {
	if math.IsNaN(float64(a)) && math.IsNaN(float64(b)) {
		return 0
	}
	if a == b {
		return 0
	}
	aBits := math.Float32bits(a)
	bBits := math.Float32bits(b)
	if (aBits >> 31) != (bBits >> 31) {
		if a == 0 && b == 0 {
			return 0
		}
		return 1000000
	}
	return abs(int(aBits) - int(bBits))
}

func testParity(t *testing.T, name string, generic, avx2 func(dst, src *float32, count int)) {
	convey.Convey(fmt.Sprintf("Given %s", name), t, func() {
		sizes := []int{1, 7, 64, 1024, 8192}
		for _, n := range sizes {
			convey.Convey(fmt.Sprintf("It should match generic at N=%d", n), func() {
				src := make([]float32, n)
				dstRef := make([]float32, n)
				dstAVX := make([]float32, n)

				for i := 0; i < n; i++ {
					src[i] = float32(i)/100.0 - 5.0
				}

				generic(&dstRef[0], &src[0], n)
				avx2(&dstAVX[0], &src[0], n)

				maxUlp := 0
				for i := 0; i < n; i++ {
					diff := ulpDiff(dstRef[i], dstAVX[i])
					if diff > maxUlp {
						maxUlp = diff
					}
					if diff > 10 { // Allow tight 10 ULP
						convey.So(diff, convey.ShouldBeLessThanOrEqualTo, 10)
					}
				}
				convey.So(maxUlp, convey.ShouldBeLessThanOrEqualTo, 10)
			})
		}
	})
}

func TestAVX2Parity(t *testing.T) {
	testParity(t, "ExpF32", ExpF32Generic, ExpF32AVX2)
	testParity(t, "SoftsignF32", SoftsignF32Generic, SoftsignF32AVX2)
	testParity(t, "QuickGeluF32", QuickGeluF32Generic, QuickGeluF32AVX2)
}

func benchmarkKernel(b *testing.B, name string, kernel func(dst, src *float32, count int)) {
	b.Run(name, func(b *testing.B) {
		n := 8192
		src := make([]float32, n)
		dst := make([]float32, n)
		for i := 0; i < n; i++ {
			src[i] = float32(i)/100.0 - 5.0
		}
		b.ResetTimer()
		for b.Loop() {
			kernel(&dst[0], &src[0], n)
		}
	})
}

func BenchmarkAVX2(b *testing.B) {
	benchmarkKernel(b, "ExpF32_Generic", ExpF32Generic)
	benchmarkKernel(b, "ExpF32_AVX2", ExpF32AVX2)
	benchmarkKernel(b, "SoftsignF32_Generic", SoftsignF32Generic)
	benchmarkKernel(b, "SoftsignF32_AVX2", SoftsignF32AVX2)
	benchmarkKernel(b, "QuickGeluF32_Generic", QuickGeluF32Generic)
	benchmarkKernel(b, "QuickGeluF32_AVX2", QuickGeluF32AVX2)
}
