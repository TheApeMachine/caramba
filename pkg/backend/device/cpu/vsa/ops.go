package vsa

import (
	"unsafe"

	"github.com/theapemachine/caramba/pkg/dtype"
)

func Bind(left, right, output unsafe.Pointer, count int, format dtype.DType) {
	if count == 0 {
		return
	}

	requireVSAFloat32(format)

	leftView := unsafe.Slice((*float32)(left), count)
	rightView := unsafe.Slice((*float32)(right), count)
	outputView := unsafe.Slice((*float32)(output), count)

	VsaBindFloat32Native(outputView, leftView, rightView)
}

func Bundle(left, right, output unsafe.Pointer, count int, format dtype.DType) {
	if count == 0 {
		return
	}

	requireVSAFloat32(format)

	leftView := unsafe.Slice((*float32)(left), count)
	rightView := unsafe.Slice((*float32)(right), count)
	outputView := unsafe.Slice((*float32)(output), count)

	VsaBundleFloat32Native(outputView, leftView, rightView)
}

func Permute(config VSAConfig, input, output unsafe.Pointer, count int, format dtype.DType) {
	if count == 0 {
		return
	}

	requireVSAFloat32(format)

	inputView := unsafe.Slice((*float32)(input), count)
	outputView := unsafe.Slice((*float32)(output), count)
	shift := config.Shift % count

	if shift < 0 {
		shift += count
	}

	VsaPermuteFloat32Native(outputView, inputView, shift)
}

func InversePermute(config VSAConfig, input, output unsafe.Pointer, count int, format dtype.DType) {
	config.Shift = -config.Shift
	Permute(config, input, output, count, format)
}

func Similarity(left, right unsafe.Pointer, count int, format dtype.DType) float32 {
	if count == 0 {
		return 0
	}

	requireVSAFloat32(format)

	leftView := unsafe.Slice((*float32)(left), count)
	rightView := unsafe.Slice((*float32)(right), count)

	return VsaSimilarityFloat32Native(leftView, rightView)
}

func requireVSAFloat32(format dtype.DType) {
	if format != dtype.Float32 {
		panic("vsa: unsupported dtype")
	}
}
