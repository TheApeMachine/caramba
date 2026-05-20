package vsa

import "github.com/theapemachine/manifesto/tensor"

func RunVSABind(args ...tensor.Tensor) error {
	return runVSABind(args...)
}

func RunVSABundle(args ...tensor.Tensor) error {
	return runVSABundle(args...)
}

func RunVSAPermuteDefault(args ...tensor.Tensor) error {
	return runVSAPermuteDefault(args...)
}

func RunVSAInversePermuteDefault(args ...tensor.Tensor) error {
	return runVSAInversePermuteDefault(args...)
}

func RunVSASimilarity(args ...tensor.Tensor) error {
	return runVSASimilarity(args...)
}
