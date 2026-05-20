//go:build amd64

package model_editing

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
	"golang.org/x/sys/cpu"
)

func avx512ModelEditingAvailable() bool {
	return cpu.X86.HasAVX512F
}

func TestWeightGraftAddFloat32AVX512Parity(t *testing.T) {
	if !avx512ModelEditingAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given WeightGraftAddFloat32AVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match WeightGraftAddFloat32Scalar for N=%d", length), func() {
				weights, injection := randomGraftVectors(length, 0x2B40+int64(length))
				got := append([]float32(nil), weights...)
				want := append([]float32(nil), weights...)

				WeightGraftAddFloat32Scalar(want, injection)
				WeightGraftAddFloat32AVX512(&got[0], &injection[0], length)

				assertFloat32SliceEqual(t, got, want)
			})
		}

		convey.Convey("It should match scalar via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				weights, injection := randomGraftVectors(length, 0x2B41+int64(length))
				got := append([]float32(nil), weights...)
				want := append([]float32(nil), weights...)

				WeightGraftAddFloat32Scalar(want, injection)
				WeightGraftAddFloat32AVX512Asm(&got[0], &injection[0], length)

				assertFloat32SliceEqual(t, got, want)
			}
		})
	})
}
