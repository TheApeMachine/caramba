package pooling

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

type poolingConfiguration struct {
	kernelH   int
	kernelW   int
	strideH   int
	strideW   int
	padH      int
	padW      int
	dilationH int
	dilationW int
}

func poolingShape4(name string, stateDict *state.Dict) (int, int, int, int, error) {
	shape := stateDict.OperationShape()

	if len(shape) != 4 {
		return 0, 0, 0, 0, fmt.Errorf("%s: expected rank 4, got %d", name, len(shape))
	}

	for index, dimension := range shape {
		if dimension <= 0 {
			return 0, 0, 0, 0, fmt.Errorf("%s: shape[%d] must be positive", name, index)
		}
	}

	return shape[0], shape[1], shape[2], shape[3], nil
}

func poolingConfig(name string, stateDict *state.Dict) (poolingConfiguration, error) {
	config := poolingConfiguration{
		kernelH:   stateDict.KernelH,
		kernelW:   stateDict.KernelW,
		strideH:   stateDict.StrideH,
		strideW:   stateDict.StrideW,
		padH:      stateDict.PadH,
		padW:      stateDict.PadW,
		dilationH: stateDict.DilationH,
		dilationW: stateDict.DilationW,
	}

	if config.kernelH <= 0 || config.kernelW <= 0 {
		return config, fmt.Errorf("%s: kernel dimensions must be positive", name)
	}

	if config.strideH <= 0 || config.strideW <= 0 {
		return config, fmt.Errorf("%s: stride dimensions must be positive", name)
	}

	if config.dilationH <= 0 || config.dilationW <= 0 {
		return config, fmt.Errorf("%s: dilation dimensions must be positive", name)
	}

	if config.padH < 0 || config.padW < 0 {
		return config, fmt.Errorf("%s: padding dimensions must be non-negative", name)
	}

	if config.kernelH*config.kernelW > maxKernelElems {
		return config, fmt.Errorf("%s: kernel exceeds maxKernelElems", name)
	}

	return config, nil
}
