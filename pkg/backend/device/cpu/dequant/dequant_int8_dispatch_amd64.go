//go:build amd64

package dequant

/*
amd64 dispatcher for int8 dequantization. AVX-512-VNNI / AVX2 paths
(VPMOVSXBD widen, VPSUBD subtract zero point, VCVTDQ2PS, VMULPS) land
in .s files in a hardware-verified session; today this routes through
the scalar reference.
*/

func DequantInt8Native(dst []float32, src []int8, scale float32, zeroPoint int8) {
	for index := range src {
		dst[index] = float32(int32(src[index])-int32(zeroPoint)) * scale
	}
}
