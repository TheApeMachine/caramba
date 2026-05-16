//go:build !cgo || !xla

package xla

import (
	"context"
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/executor"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

var _ executor.Backend = (*TensorBackend)(nil)

/*
TensorBackend is unavailable without the cgo and xla build tags.
*/
type TensorBackend struct{}

/*
NewTensorBackend reports that resident XLA tensors require cgo and xla.

platform is accepted so call signatures match the cgo+xla implementation (PJRT plugin selection).
*/
func NewTensorBackend(platform string) (*TensorBackend, error) {
	_ = platform

	return &TensorBackend{}, fmt.Errorf("xla tensor: unavailable without cgo and xla build tags")
}

func (tensorBackend *TensorBackend) Location() computetensor.Location {
	return computetensor.XLA
}

func (tensorBackend *TensorBackend) UploadFloat64(
	shape computetensor.Shape, values []float64,
) (computetensor.Float64Tensor, error) {
	return nil, fmt.Errorf("xla tensor: unavailable without cgo and xla build tags")
}

func (tensorBackend *TensorBackend) DownloadFloat64(
	input computetensor.Float64Tensor,
) ([]float64, error) {
	return nil, fmt.Errorf("xla tensor: unavailable without cgo and xla build tags")
}

func (tensorBackend *TensorBackend) Close() error {
	return nil
}

func (tensorBackend *TensorBackend) Apply(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	return nil, fmt.Errorf("xla tensor: unavailable without cgo and xla build tags")
}

func (tensorBackend *TensorBackend) ReLU(input computetensor.Float64Tensor) (computetensor.Float64Tensor, error) {
	return nil, fmt.Errorf("xla tensor: unavailable without cgo and xla build tags")
}

func (tensorBackend *TensorBackend) LeakyReLU(input computetensor.Float64Tensor, alpha float64) (computetensor.Float64Tensor, error) {
	return nil, fmt.Errorf("xla tensor: unavailable without cgo and xla build tags")
}

func (tensorBackend *TensorBackend) GELU(input computetensor.Float64Tensor) (computetensor.Float64Tensor, error) {
	return nil, fmt.Errorf("xla tensor: unavailable without cgo and xla build tags")
}

func (tensorBackend *TensorBackend) Tanh(input computetensor.Float64Tensor) (computetensor.Float64Tensor, error) {
	return nil, fmt.Errorf("xla tensor: unavailable without cgo and xla build tags")
}

func (tensorBackend *TensorBackend) Sigmoid(input computetensor.Float64Tensor) (computetensor.Float64Tensor, error) {
	return nil, fmt.Errorf("xla tensor: unavailable without cgo and xla build tags")
}

func (tensorBackend *TensorBackend) Swish(input computetensor.Float64Tensor) (computetensor.Float64Tensor, error) {
	return nil, fmt.Errorf("xla tensor: unavailable without cgo and xla build tags")
}

func (tensorBackend *TensorBackend) SELU(input computetensor.Float64Tensor) (computetensor.Float64Tensor, error) {
	return nil, fmt.Errorf("xla tensor: unavailable without cgo and xla build tags")
}

func (tensorBackend *TensorBackend) SwiGLU(input computetensor.Float64Tensor) (computetensor.Float64Tensor, error) {
	return nil, fmt.Errorf("xla tensor: unavailable without cgo and xla build tags")
}

func (tensorBackend *TensorBackend) Add(left, right computetensor.Float64Tensor) (computetensor.Float64Tensor, error) {
	return nil, fmt.Errorf("xla tensor: unavailable without cgo and xla build tags")
}

func (tensorBackend *TensorBackend) Mul(left, right computetensor.Float64Tensor) (computetensor.Float64Tensor, error) {
	return nil, fmt.Errorf("xla tensor: unavailable without cgo and xla build tags")
}

func (tensorBackend *TensorBackend) Matmul(left, right computetensor.Float64Tensor) (computetensor.Float64Tensor, error) {
	return nil, fmt.Errorf("xla tensor: unavailable without cgo and xla build tags")
}

func (tensorBackend *TensorBackend) MatmulAdd(left, right, bias computetensor.Float64Tensor) (computetensor.Float64Tensor, error) {
	return nil, fmt.Errorf("xla tensor: unavailable without cgo and xla build tags")
}

func (tensorBackend *TensorBackend) MatmulAddGELU(left, right, bias computetensor.Float64Tensor) (computetensor.Float64Tensor, error) {
	return nil, fmt.Errorf("xla tensor: unavailable without cgo and xla build tags")
}
