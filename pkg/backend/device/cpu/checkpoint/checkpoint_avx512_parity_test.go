//go:build amd64

package checkpoint

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
	"golang.org/x/sys/cpu"
)

func avx512CheckpointAvailable() bool {
	return cpu.X86.HasAVX512F
}

func TestCheckpointEncodeFloat32DataAVX512Parity(t *testing.T) {
	if !avx512CheckpointAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given CheckpointEncodeFloat32DataAVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match encodeFloat32DataScalar for N=%d", length), func() {
				source := randomFloat32Vector(length, 0x2940+int64(length))
				want := make([]uint8, length*4)
				got := make([]uint8, length*4)

				encodeFloat32DataScalar(want, source)
				CheckpointEncodeFloat32DataAVX512(&got[0], &source[0], length)

				assertUint8PayloadEqual(t, got, want)
			})
		}

		convey.Convey("It should match encodeFloat32DataScalar via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				source := randomFloat32Vector(length, 0x2941+int64(length))
				want := make([]uint8, length*4)
				got := make([]uint8, length*4)

				encodeFloat32DataScalar(want, source)
				CheckpointEncodeFloat32DataAVX512Asm(&got[0], &source[0], length)

				assertUint8PayloadEqual(t, got, want)
			}
		})
	})
}

func TestCheckpointDecodeFloat32DataAVX512Parity(t *testing.T) {
	if !avx512CheckpointAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given CheckpointDecodeFloat32DataAVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match decodeFloat32DataScalar for N=%d", length), func() {
				source := randomFloat32Vector(length, 0x2950+int64(length))
				payload := make([]uint8, length*4)
				encodeFloat32DataScalar(payload, source)

				want := make([]float32, length)
				got := make([]float32, length)

				decodeFloat32DataScalar(want, payload)
				CheckpointDecodeFloat32DataAVX512(&got[0], &payload[0], length)

				assertFloat32SliceEqual(t, got, want)
			})
		}

		convey.Convey("It should match decodeFloat32DataScalar via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				source := randomFloat32Vector(length, 0x2951+int64(length))
				payload := make([]uint8, length*4)
				encodeFloat32DataScalar(payload, source)

				want := make([]float32, length)
				got := make([]float32, length)

				decodeFloat32DataScalar(want, payload)
				CheckpointDecodeFloat32DataAVX512Asm(&got[0], &payload[0], length)

				assertFloat32SliceEqual(t, got, want)
			}
		})
	})
}
