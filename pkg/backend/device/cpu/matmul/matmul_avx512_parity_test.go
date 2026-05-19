//go:build amd64

package matmul

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
	"golang.org/x/sys/cpu"
)

func avx512MatmulAvailable() bool {
	return cpu.X86.HasAVX512F
}

func randomMatmulFloat32Slice(length int, seed int64) []float32 {
	slice := make([]float32, length)
	state := uint64(seed)

	for index := range slice {
		state = state*6364136223846793005 + 1
		slice[index] = float32(int32(state>>33)%1000) * 0.001
	}

	return slice
}

func TestMatMulF32AVX512Parity(t *testing.T) {
	if !avx512MatmulAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given MatmulFloat32AVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match scalar reference for N=%d", length), func() {
				rows := 7
				inner := length
				cols := length

				left := randomMatmulFloat32Slice(rows*inner, 0xA11+int64(length))
				right := randomMatmulFloat32Slice(inner*cols, 0xB22+int64(length))
				want := make([]float32, rows*cols)
				got := make([]float32, rows*cols)

				matmulFloat32Scalar(want, left, right, rows, inner, cols)
				MatmulFloat32AVX512(got, left, right, rows, inner, cols)

				parity.AssertFloat32SlicesWithinULP(t, got, want, 1)
			})
		}
	})
}
