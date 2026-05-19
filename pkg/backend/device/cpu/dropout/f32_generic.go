package dropout

import "unsafe"

func DropoutF32Generic(
	dst, src *float32,
	count int,
	seedState *[4]uint32,
	keepProb float32,
) {
	destination := unsafe.Slice(dst, count)
	source := unsafe.Slice(src, count)
	scale := float32(1.0 / keepProb)
	threshold := dropoutThreshold(keepProb)

	for index := 0; index < count; index++ {
		destination[index] = dropoutFloat32ScalarLane(
			source[index], seedState, scale, threshold,
		)
	}
}
