//go:build amd64

package interpretability

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
	"golang.org/x/sys/cpu"
)

func avx512InterpretabilityAvailable() bool {
	return cpu.X86.HasAVX512F
}

func TestActivationSteerFloat32AVX512Parity(t *testing.T) {
	if !avx512InterpretabilityAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given ActivationSteerFloat32AVX512", t, func() {
		const coefficient = float32(0.25)

		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match ActivationSteerFloat32Scalar for N=%d", length), func() {
				base, direction := randomSteerVectors(length, 0x2A40+int64(length))
				got := make([]float32, length)
				want := make([]float32, length)

				ActivationSteerFloat32Scalar(want, base, direction, coefficient)
				ActivationSteerFloat32AVX512(&got[0], &base[0], &direction[0], coefficient, length)

				assertFloat32SliceEqual(t, got, want)
			})
		}

		convey.Convey("It should match scalar via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				base, direction := randomSteerVectors(length, 0x2A41+int64(length))
				got := make([]float32, length)
				want := make([]float32, length)

				ActivationSteerFloat32Scalar(want, base, direction, coefficient)
				ActivationSteerFloat32AVX512Asm(&got[0], &base[0], &direction[0], coefficient, length)

				assertFloat32SliceEqual(t, got, want)
			}
		})
	})
}
