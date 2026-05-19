package attention

import "github.com/theapemachine/caramba/pkg/backend/compute/tensor"

func RunAttentionFloat32(args ...tensor.Tensor) error {
	return runAttentionFloat32(args...)
}

func RunAttentionBFloat16(args ...tensor.Tensor) error {
	return runAttentionBFloat16(args...)
}

func RunAttentionFloat16(args ...tensor.Tensor) error {
	return runAttentionFloat16(args...)
}

func RunFlashAttentionFloat32Default(args ...tensor.Tensor) error {
	return runFlashAttentionFloat32Default(args...)
}

func RunFlashAttentionBFloat16(args ...tensor.Tensor) error {
	return runFlashAttentionBFloat16(args...)
}

func RunFlashAttentionFloat16(args ...tensor.Tensor) error {
	return runFlashAttentionFloat16(args...)
}

func RunMultiHeadAttentionDefault(args ...tensor.Tensor) error {
	return runMultiHeadAttentionDefault(args...)
}

func RunGroupedQueryAttentionDefault(args ...tensor.Tensor) error {
	return runGroupedQueryAttentionDefault(args...)
}

func RunSlidingWindowAttentionDefault(args ...tensor.Tensor) error {
	return runSlidingWindowAttentionDefault(args...)
}

func RunMultiHeadAttentionVariantBFloat16(variantName string, args ...tensor.Tensor) error {
	return runMultiHeadAttentionBFloat16(args, configForVariant(variantName))
}

func RunMultiHeadAttentionVariantFloat16(variantName string, args ...tensor.Tensor) error {
	return runMultiHeadAttentionFloat16(args, configForVariant(variantName))
}
