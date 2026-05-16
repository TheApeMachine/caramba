//go:build !linux || !cgo || !cuda

package cuda

import (
	"context"
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/executor"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

const cudaTensorUnavailableMsg = "CUDA tensor unavailable: rebuild on Linux with CGO enabled and build tags linux,cgo,cuda"

var _ computetensor.Backend = (*TensorBackend)(nil)
var _ executor.Backend = (*TensorBackend)(nil)

/*
TensorBackend reports unavailable CUDA tensor support when CUDA is not built in.
*/
type TensorBackend struct{}

/*
NewTensorBackend creates a CUDA tensor backend stub.
*/
func NewTensorBackend() *TensorBackend {
	return &TensorBackend{}
}

/*
Location identifies the intended CUDA storage location.
*/
func (tensorBackend *TensorBackend) Location() computetensor.Location {
	return computetensor.CUDA
}

/*
UploadFloat64 rejects uploads when CUDA support is not built in.
*/
func (tensorBackend *TensorBackend) UploadFloat64(
	shape computetensor.Shape, values []float64,
) (computetensor.Float64Tensor, error) {
	return nil, fmt.Errorf("%s", cudaTensorUnavailableMsg)
}

/*
DownloadFloat64 rejects downloads when CUDA support is not built in.
*/
func (tensorBackend *TensorBackend) DownloadFloat64(
	input computetensor.Float64Tensor,
) ([]float64, error) {
	return nil, fmt.Errorf("%s", cudaTensorUnavailableMsg)
}

/*
Close is a no-op for the CUDA tensor backend stub.
*/
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

	return nil, fmt.Errorf("%s", cudaTensorUnavailableMsg)
}

func (tensorBackend *TensorBackend) ReLU(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return nil, fmt.Errorf("%s", cudaTensorUnavailableMsg)
}

func (tensorBackend *TensorBackend) LeakyReLU(
	input computetensor.Float64Tensor, alpha float64,
) (computetensor.Float64Tensor, error) {
	return nil, fmt.Errorf("%s", cudaTensorUnavailableMsg)
}

func (tensorBackend *TensorBackend) GELU(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return nil, fmt.Errorf("%s", cudaTensorUnavailableMsg)
}

func (tensorBackend *TensorBackend) Tanh(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return nil, fmt.Errorf("%s", cudaTensorUnavailableMsg)
}

func (tensorBackend *TensorBackend) Sigmoid(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return nil, fmt.Errorf("%s", cudaTensorUnavailableMsg)
}

func (tensorBackend *TensorBackend) Swish(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return nil, fmt.Errorf("%s", cudaTensorUnavailableMsg)
}

func (tensorBackend *TensorBackend) SELU(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return nil, fmt.Errorf("%s", cudaTensorUnavailableMsg)
}

func (tensorBackend *TensorBackend) SwiGLU(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return nil, fmt.Errorf("%s", cudaTensorUnavailableMsg)
}

func (tensorBackend *TensorBackend) Add(
	left, right computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return nil, fmt.Errorf("%s", cudaTensorUnavailableMsg)
}

func (tensorBackend *TensorBackend) Mul(
	left, right computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return nil, fmt.Errorf("%s", cudaTensorUnavailableMsg)
}

func (tensorBackend *TensorBackend) Matmul(
	left, right computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return nil, fmt.Errorf("%s", cudaTensorUnavailableMsg)
}

func (tensorBackend *TensorBackend) MatmulAdd(
	left, right, bias computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return nil, fmt.Errorf("%s", cudaTensorUnavailableMsg)
}

func (tensorBackend *TensorBackend) MatmulAddGELU(
	left, right, bias computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return nil, fmt.Errorf("%s", cudaTensorUnavailableMsg)
}
