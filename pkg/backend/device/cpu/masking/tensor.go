package masking

import "github.com/theapemachine/manifesto/tensor"

func RunApplyMask(args ...tensor.Tensor) error {
	return runApplyMask(args...)
}

func RunCausalMask(args ...tensor.Tensor) error {
	return runCausalMask(args...)
}

func RunALiBiBias(args ...tensor.Tensor) error {
	return runALiBiBias(args...)
}

func RunApplyMaskBFloat16(args ...tensor.Tensor) error {
	return runApplyMaskBFloat16(args...)
}

func RunApplyMaskFloat16(args ...tensor.Tensor) error {
	return runApplyMaskFloat16(args...)
}

func RunCausalMaskBFloat16(args ...tensor.Tensor) error {
	return runCausalMaskBFloat16(args...)
}

func RunCausalMaskFloat16(args ...tensor.Tensor) error {
	return runCausalMaskFloat16(args...)
}

func RunALiBiBiasBFloat16(args ...tensor.Tensor) error {
	return runALiBiBiasBFloat16(args...)
}

func RunALiBiBiasFloat16(args ...tensor.Tensor) error {
	return runALiBiBiasFloat16(args...)
}
