package tokenizer

import "github.com/theapemachine/caramba/pkg/backend/compute/tensor"

func RunTokenizerPackInt32(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	in, err := args[0].Int32Native()

	if err != nil {
		return err
	}

	out, err := args[1].Int32Native()

	if err != nil {
		return err
	}

	if len(out) < len(in) {
		return tensor.ErrShapeMismatch
	}

	copy(out, in)
	return nil
}
