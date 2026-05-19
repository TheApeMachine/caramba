package layernorm

import (
	"unsafe"

	"github.com/theapemachine/caramba/pkg/dtype"
)

func LayerNorm(
	input, scale, bias, output unsafe.Pointer,
	rows, lastDim int,
	format dtype.DType,
) {
	dispatchLayerNorm(input, scale, bias, output, rows, lastDim, format)
}

func RMSNorm(
	input, scale, output unsafe.Pointer,
	rows, lastDim int,
	format dtype.DType,
) {
	dispatchRMSNorm(input, scale, output, rows, lastDim, format)
}
