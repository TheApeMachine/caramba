package dropout

import (
	"unsafe"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func DropoutFloat32(config DropoutConfig, input, output tensor.Tensor) error {
	inView, err := input.Float32Native()

	if err != nil {
		return err
	}

	outView, err := output.Float32Native()

	if err != nil {
		return err
	}

	if len(outView) != len(inView) {
		return tensor.ErrShapeMismatch
	}

	if len(inView) == 0 {
		return nil
	}

	Dropout(
		unsafe.Pointer(&outView[0]),
		unsafe.Pointer(&inView[0]),
		len(inView),
		config,
		dtype.Float32,
	)

	return nil
}

func RunDropoutDefault(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	return DropoutFloat32(DefaultDropoutConfig(), args[0], args[1])
}
