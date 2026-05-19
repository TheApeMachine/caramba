//go:build arm64

package dropout

import "unsafe"

func DropoutF32NEON(
	dst, src *float32,
	count int,
	seedState *[4]uint32,
	keepProb float32,
) {
	blockCount := count &^ 3
	scale := float32(1.0 / keepProb)
	threshold := dropoutThreshold(keepProb)

	if blockCount > 0 {
		DropoutFloat32NEONAsm(dst, src, blockCount, &seedState[0], scale, threshold)
	}

	destination := unsafe.Slice(dst, count)
	source := unsafe.Slice(src, count)

	for index := blockCount; index < count; index++ {
		destination[index] = dropoutFloat32ScalarLane(
			source[index], seedState, scale, threshold,
		)
	}
}

func DropoutFloat32Native(
	dst, src []float32,
	seedState *[4]uint32,
	keepProb float32,
) {
	if len(src) == 0 {
		return
	}

	dropoutF32Kernel(&dst[0], &src[0], len(src), seedState, keepProb)
}

var dropoutF32Funcs = []f32DropoutKernelImpl{
	{DropoutF32NEON, "neon", true},
	{DropoutF32Generic, "generic", true},
}
