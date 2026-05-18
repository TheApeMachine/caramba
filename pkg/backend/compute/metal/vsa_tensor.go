//go:build darwin && cgo

package metal

// #include "vsa.h"
import "C"

import (
	"fmt"

	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

/*
BindTensor computes resident elementwise VSA binding.
*/
func (metalVSAOps *MetalVSAOps) BindTensor(
	left, right computetensor.Tensor,
) (computetensor.Tensor, error) {
	metalLeft, metalRight, err := requireVSABinary(left, right)
	if err != nil {
		return nil, err
	}

	output, err := metalVSAOps.runtime.NewFloat32Tensor(metalLeft.shape, MetalAllocationTensor)
	if err != nil {
		return nil, err
	}

	rc := C.metal_vsa_bind_tensor(
		metalLeft.buffer,
		metalRight.buffer,
		output.buffer,
		C.int(metalLeft.Len()),
	)
	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal tensor: vsa.bind launch failed")
	}

	return output, nil
}

/*
BundleTensor sums count resident vectors and normalises the result.
*/
func (metalVSAOps *MetalVSAOps) BundleTensor(
	vectors computetensor.Tensor,
	outputShape computetensor.Shape,
	count int,
) (computetensor.Tensor, error) {
	metalVectors, err := requireMetalTensor(vectors)
	if err != nil {
		return nil, err
	}

	if count <= 0 || outputShape.Len() <= 0 || metalVectors.Len() != count*outputShape.Len() {
		return nil, fmt.Errorf("metal tensor: vsa.bundle shape mismatch")
	}

	output, err := metalVSAOps.runtime.NewFloat32Tensor(outputShape, MetalAllocationTensor)
	if err != nil {
		return nil, err
	}

	rc := C.metal_vsa_bundle_tensor(
		metalVectors.buffer,
		output.buffer,
		C.int(count),
		C.int(outputShape.Len()),
	)
	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal tensor: vsa.bundle launch failed")
	}

	return output, nil
}

/*
SimilarityTensor computes resident dot-product similarity.
*/
func (metalVSAOps *MetalVSAOps) SimilarityTensor(
	left, right computetensor.Tensor,
	outputShape computetensor.Shape,
) (computetensor.Tensor, error) {
	metalLeft, metalRight, err := requireVSABinary(left, right)
	if err != nil {
		return nil, err
	}

	if outputShape.Len() != 1 {
		return nil, fmt.Errorf("metal tensor: vsa.similarity output length must be 1")
	}

	output, err := metalVSAOps.runtime.NewFloat32Tensor(outputShape, MetalAllocationTensor)
	if err != nil {
		return nil, err
	}

	rc := C.metal_vsa_dot_tensor(
		metalLeft.buffer,
		metalRight.buffer,
		output.buffer,
		C.int(metalLeft.Len()),
	)
	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal tensor: vsa.similarity launch failed")
	}

	return output, nil
}

/*
PermuteTensor cyclically shifts a resident vector.
*/
func (metalVSAOps *MetalVSAOps) PermuteTensor(
	input computetensor.Tensor,
	shift int,
) (computetensor.Tensor, error) {
	return metalVSAOps.permuteTensor(input, shift, false)
}

/*
InversePermuteTensor applies the inverse resident VSA permutation.
*/
func (metalVSAOps *MetalVSAOps) InversePermuteTensor(
	input computetensor.Tensor,
	shift int,
) (computetensor.Tensor, error) {
	return metalVSAOps.permuteTensor(input, shift, true)
}

func (metalVSAOps *MetalVSAOps) permuteTensor(
	input computetensor.Tensor,
	shift int,
	inverse bool,
) (computetensor.Tensor, error) {
	metalInput, err := requireMetalTensor(input)
	if err != nil {
		return nil, err
	}

	if metalInput.Len() <= 0 {
		return nil, fmt.Errorf("metal tensor: vsa.permute requires non-empty input")
	}

	output, err := metalVSAOps.runtime.NewFloat32Tensor(metalInput.shape, MetalAllocationTensor)
	if err != nil {
		return nil, err
	}

	rc := C.metal_vsa_permute_tensor(
		metalInput.buffer,
		output.buffer,
		C.int(metalInput.Len()),
		C.int(shift),
	)

	if inverse {
		rc = C.metal_vsa_inverse_permute_tensor(
			metalInput.buffer,
			output.buffer,
			C.int(metalInput.Len()),
			C.int(shift),
		)
	}

	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal tensor: vsa.permute launch failed")
	}

	return output, nil
}

func requireVSABinary(
	left computetensor.Tensor,
	right computetensor.Tensor,
) (*Tensor, *Tensor, error) {
	metalLeft, err := requireMetalTensor(left)
	if err != nil {
		return nil, nil, err
	}

	metalRight, err := requireMetalTensor(right)
	if err != nil {
		return nil, nil, err
	}

	if !metalLeft.shape.Equal(metalRight.shape) {
		return nil, nil, fmt.Errorf("metal tensor: VSA input shape mismatch")
	}

	return metalLeft, metalRight, nil
}
