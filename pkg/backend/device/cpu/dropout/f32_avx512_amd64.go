//go:build amd64

package dropout

import "unsafe"

//go:noescape
func DropoutFloat32AVX512Asm(
	dst, src *float32,
	count int,
	seedLane *uint32,
	scale, threshold float32,
)

/*
DropoutF32AVX512 applies inverted dropout with the same sequential
xorshift32 stream on seedState[0] as DropoutF32Generic.
*/
func DropoutF32AVX512(
	dst, src *float32,
	count int,
	seedState *[4]uint32,
	keepProb float32,
) {
	if count == 0 {
		return
	}

	scale := float32(1.0 / keepProb)
	threshold := dropoutThreshold(keepProb)
	blockCount := count &^ 15

	if blockCount > 0 {
		DropoutFloat32AVX512Asm(
			dst, src, blockCount,
			&seedState[0], scale, threshold,
		)
	}

	destination := unsafe.Slice(dst, count)
	source := unsafe.Slice(src, count)

	for index := blockCount; index < count; index++ {
		destination[index] = dropoutFloat32ScalarLane(
			source[index], seedState, scale, threshold,
		)
	}
}
