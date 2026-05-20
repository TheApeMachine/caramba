package losses

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/convert"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func Bfloat16BulkToFloat32(dst []float32, src []dtype.BF16) {
	if len(src) == 0 {
		return
	}

	_ = convert.BFloat16ToFloat32(dst, src)
}

func Float32BulkToBFloat16(dst []dtype.BF16, src []float32) {
	if len(src) == 0 {
		return
	}

	_ = convert.Float32ToBFloat16(dst, src)
}

func Float16BulkToFloat32(dst []float32, src []dtype.F16) {
	if len(src) == 0 {
		return
	}

	_ = convert.Float16ToFloat32(dst, src)
}

func Float32BulkToFloat16(dst []dtype.F16, src []float32) {
	if len(src) == 0 {
		return
	}

	_ = convert.Float32ToFloat16(dst, src)
}

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

func argLen(arg tensor.Tensor, kind dtype.DType) (int, error) {
	switch kind {
	case dtype.BFloat16:
		view, err := arg.BFloat16Native()

		if err != nil {
			return 0, err
		}

		return len(view), nil
	case dtype.Float16:
		view, err := arg.Float16Native()

		if err != nil {
			return 0, err
		}

		return len(view), nil
	}

	return 0, tensor.ErrShapeMismatch
}
