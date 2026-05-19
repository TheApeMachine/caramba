//go:build arm64

package convolution

import (
	"fmt"
	"testing"

	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
)

func TestConv3dPatchDotNEONAsmParity(t *testing.T) {
	for _, patchLength := range parity.Lengths {
		t.Run(fmt.Sprintf("N=%d", patchLength), func(t *testing.T) {
			weight := randFloat32Slice(patchLength, 0x3D0+int64(patchLength))
			patch := randFloat32Slice(patchLength, 0x3D1+int64(patchLength))

			want := ConvPatchDotScalar(weight, patch, patchLength)
			got := Conv3dPatchDotNEONAsm(&weight[0], &patch[0], patchLength)
			parity.AssertFloat32SlicesWithinULP(t, []float32{got}, []float32{want}, 2)
		})
	}
}

func BenchmarkConv3dPatchDotNEONAsm(b *testing.B) {
	weight := randFloat32Slice(8192, 0x3DB)
	patch := randFloat32Slice(8192, 0x3DC)

	for b.Loop() {
		_ = Conv3dPatchDotNEONAsm(&weight[0], &patch[0], len(weight))
	}
}
