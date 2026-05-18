//go:build !darwin || !cgo

package metal

import (
	"context"

	"github.com/theapemachine/caramba/pkg/backend/compute/executor"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

var _ executor.Backend = (*TensorBackend)(nil)

func (tensorBackend *TensorBackend) Apply(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []computetensor.Tensor,
) (computetensor.Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	return nil, metalUnavailable()
}

func (tensorBackend *TensorBackend) ReLU(
	input computetensor.Tensor,
) (computetensor.Tensor, error) {
	return nil, metalUnavailable()
}

func (tensorBackend *TensorBackend) LeakyReLU(
	input computetensor.Tensor, alpha float64,
) (computetensor.Tensor, error) {
	return nil, metalUnavailable()
}

func (tensorBackend *TensorBackend) GELU(
	input computetensor.Tensor,
) (computetensor.Tensor, error) {
	return nil, metalUnavailable()
}

func (tensorBackend *TensorBackend) Tanh(
	input computetensor.Tensor,
) (computetensor.Tensor, error) {
	return nil, metalUnavailable()
}

func (tensorBackend *TensorBackend) Sigmoid(
	input computetensor.Tensor,
) (computetensor.Tensor, error) {
	return nil, metalUnavailable()
}

func (tensorBackend *TensorBackend) SwiGLU(
	input computetensor.Tensor,
) (computetensor.Tensor, error) {
	return nil, metalUnavailable()
}

func (tensorBackend *TensorBackend) Add(
	left, right computetensor.Tensor,
) (computetensor.Tensor, error) {
	return nil, metalUnavailable()
}

func (tensorBackend *TensorBackend) Mul(
	left, right computetensor.Tensor,
) (computetensor.Tensor, error) {
	return nil, metalUnavailable()
}

func (tensorBackend *TensorBackend) Matmul(
	left, right computetensor.Tensor,
) (computetensor.Tensor, error) {
	return nil, metalUnavailable()
}

func (tensorBackend *TensorBackend) MatmulAdd(
	left, right, bias computetensor.Tensor,
) (computetensor.Tensor, error) {
	return nil, metalUnavailable()
}

func (tensorBackend *TensorBackend) MatmulAddGELU(
	left, right, bias computetensor.Tensor,
) (computetensor.Tensor, error) {
	return nil, metalUnavailable()
}
