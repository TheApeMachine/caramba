package vsa

import "unsafe"

func Bind(left, right, output unsafe.Pointer, count int) {
	if count == 0 {
		return
	}

	leftView := unsafe.Slice((*float32)(left), count)
	rightView := unsafe.Slice((*float32)(right), count)
	outputView := unsafe.Slice((*float32)(output), count)

	VsaBindFloat32Native(outputView, leftView, rightView)
}

func Bundle(left, right, output unsafe.Pointer, count int) {
	if count == 0 {
		return
	}

	leftView := unsafe.Slice((*float32)(left), count)
	rightView := unsafe.Slice((*float32)(right), count)
	outputView := unsafe.Slice((*float32)(output), count)

	VsaBundleFloat32Native(outputView, leftView, rightView)
}

func Permute(config VSAConfig, input, output unsafe.Pointer, count int) {
	if count == 0 {
		return
	}

	inputView := unsafe.Slice((*float32)(input), count)
	outputView := unsafe.Slice((*float32)(output), count)
	shift := config.Shift % count

	if shift < 0 {
		shift += count
	}

	VsaPermuteFloat32Native(outputView, inputView, shift)
}

func InversePermute(config VSAConfig, input, output unsafe.Pointer, count int) {
	config.Shift = -config.Shift
	Permute(config, input, output, count)
}

func Similarity(left, right unsafe.Pointer, count int) float32 {
	if count == 0 {
		return 0
	}

	leftView := unsafe.Slice((*float32)(left), count)
	rightView := unsafe.Slice((*float32)(right), count)

	return VsaSimilarityFloat32Native(leftView, rightView)
}
