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
	inputs []computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	return nil, metalUnavailable()
}

func (tensorBackend *TensorBackend) ReLU(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return nil, metalUnavailable()
}

func (tensorBackend *TensorBackend) LeakyReLU(
	input computetensor.Float64Tensor, alpha float64,
) (computetensor.Float64Tensor, error) {
	return nil, metalUnavailable()
}

func (tensorBackend *TensorBackend) GELU(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return nil, metalUnavailable()
}

func (tensorBackend *TensorBackend) Tanh(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return nil, metalUnavailable()
}

func (tensorBackend *TensorBackend) Sigmoid(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return nil, metalUnavailable()
}

func (tensorBackend *TensorBackend) SwiGLU(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return nil, metalUnavailable()
}

func (tensorBackend *TensorBackend) Add(
	left, right computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return nil, metalUnavailable()
}

func (tensorBackend *TensorBackend) Mul(
	left, right computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return nil, metalUnavailable()
}

func (tensorBackend *TensorBackend) Matmul(
	left, right computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return nil, metalUnavailable()
}

func (tensorBackend *TensorBackend) MatmulAdd(
	left, right, bias computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return nil, metalUnavailable()
}

func (tensorBackend *TensorBackend) MatmulAddGELU(
	left, right, bias computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return nil, metalUnavailable()
}
