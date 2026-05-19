//go:build amd64

package activation

import (
	"fmt"
	"testing"

	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
	. "github.com/smartystreets/goconvey/convey"
)

func TestSwiGLUTensorsF32SSE2(t *testing.T) {
	Convey("Given SwiGLU SSE2 tensors", t, func() {
		for _, length := range parity.Lengths {
			gate := make([]float32, length)
			up := make([]float32, length)
			dst := make([]float32, length)
			reference := make([]float32, length)

			for index := range gate {
				gate[index] = float32(index)*0.1 - 0.5
				up[index] = float32(index)*0.1 + 0.5
			}

			SwiGLUTensorsF32Generic(&reference[0], &gate[0], &up[0], length)
			SwiGLUTensorsF32SSE2(&dst[0], &gate[0], &up[0], length)

			Convey(fmt.Sprintf("It should match generic within 2 ULP at N=%d", length), func() {
				parity.AssertFloat32SlicesWithinULP(t, dst, reference, 2)
			})
		}
	})
}

func TestLinGLUTensorsF32SSE2(t *testing.T) {
	Convey("Given LinGLU SSE2 tensors", t, func() {
		for _, length := range parity.Lengths {
			gate := make([]float32, length)
			up := make([]float32, length)
			dst := make([]float32, length)
			reference := make([]float32, length)

			for index := range gate {
				gate[index] = float32(index)*0.1 - 0.5
				up[index] = float32(index)*0.1 + 0.5
			}

			LinGLUTensorsF32Generic(&reference[0], &gate[0], &up[0], length)
			LinGLUTensorsF32SSE2(&dst[0], &gate[0], &up[0], length)

			Convey("It should match generic exactly", func() {
				parity.AssertFloat32SlicesWithinULP(t, dst, reference, 0)
			})
		}
	})
}

func TestSiGLUTensorsF32SSE2(t *testing.T) {
	Convey("Given SiGLU SSE2 tensors", t, func() {
		for _, length := range parity.Lengths {
			gate := make([]float32, length)
			up := make([]float32, length)
			dst := make([]float32, length)
			reference := make([]float32, length)

			for index := range gate {
				gate[index] = float32(index)*0.1 - 0.5
				up[index] = float32(index)*0.1 + 0.5
			}

			SiGLUTensorsF32Generic(&reference[0], &gate[0], &up[0], length)
			SiGLUTensorsF32SSE2(&dst[0], &gate[0], &up[0], length)

			Convey("It should match generic within 2 ULP", func() {
				parity.AssertFloat32SlicesWithinULP(t, dst, reference, 2)
			})
		}
	})
}

func TestSeGLUTensorsF32SSE2(t *testing.T) {
	Convey("Given SeGLU SSE2 tensors", t, func() {
		for _, length := range parity.Lengths {
			gate := make([]float32, length)
			up := make([]float32, length)
			dst := make([]float32, length)
			reference := make([]float32, length)

			for index := range gate {
				gate[index] = float32(index)*0.1 - 0.5
				up[index] = float32(index)*0.1 + 0.5
			}

			SeGLUTensorsF32Generic(&reference[0], &gate[0], &up[0], length)
			SeGLUTensorsF32SSE2(&dst[0], &gate[0], &up[0], length)

			Convey("It should match generic within 2 ULP", func() {
				parity.AssertFloat32SlicesWithinULP(t, dst, reference, 2)
			})
		}
	})
}

func BenchmarkSwiGLUTensorsF32SSE2(b *testing.B) {
	length := 8192
	gate := make([]float32, length)
	up := make([]float32, length)
	dst := make([]float32, length)

	for b.Loop() {
		SwiGLUTensorsF32SSE2(&dst[0], &gate[0], &up[0], length)
	}
}
