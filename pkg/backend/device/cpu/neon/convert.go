package neon

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func widenToF32(arg tensor.Tensor, kind dtype.DType, dst []float32) error {
	switch kind {
	case dtype.BFloat16:
		view, err := arg.BFloat16Native()

		if err != nil {
			return err
		}

		if len(view) != len(dst) {
			return tensor.ErrShapeMismatch
		}

		Bfloat16BulkToFloat32(dst, view)
	case dtype.Float16:
		view, err := arg.Float16Native()

		if err != nil {
			return err
		}

		if len(view) != len(dst) {
			return tensor.ErrShapeMismatch
		}

		Float16BulkToFloat32(dst, view)
	}

	return nil
}

func narrowFromF32(arg tensor.Tensor, kind dtype.DType, src []float32) error {
	switch kind {
	case dtype.BFloat16:
		view, err := arg.BFloat16Native()

		if err != nil {
			return err
		}

		if len(view) != len(src) {
			return tensor.ErrShapeMismatch
		}

		Float32BulkToBFloat16(view, src)
	case dtype.Float16:
		view, err := arg.Float16Native()

		if err != nil {
			return err
		}

		if len(view) != len(src) {
			return tensor.ErrShapeMismatch
		}

		Float32BulkToFloat16(view, src)
	}

	return nil
}
