package normalization

import (
	"unsafe"

	"github.com/theapemachine/caramba/pkg/dtype"
)

func GroupNorm(
	config GroupNormConfig,
	input, scale, bias, output unsafe.Pointer,
	batch, channels, spatial int,
	format dtype.DType,
) {
	dispatchGroupNorm(config, input, scale, bias, output, batch, channels, spatial, format)
}

func InstanceNorm(
	input, scale, bias, output unsafe.Pointer,
	batch, channels, spatial int,
	format dtype.DType,
) {
	dispatchInstanceNorm(input, scale, bias, output, batch, channels, spatial, format)
}

func BatchNormEval(
	input, scale, bias, mean, variance, output unsafe.Pointer,
	batch, channels, spatial int,
	format dtype.DType,
) {
	dispatchBatchNormEval(input, scale, bias, mean, variance, output, batch, channels, spatial, format)
}
