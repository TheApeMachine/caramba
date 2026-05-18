package metal

import (
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
	dtypeconvert "github.com/theapemachine/caramba/pkg/dtype/convert"
)

func uploadTensorValues(
	tensorBackend *TensorBackend,
	shape computetensor.Shape,
	values []float64,
) (computetensor.Tensor, error) {
	return tensorBackend.Upload(shape, dtype.Float64, dtypeconvert.Float64ToBytes(values))
}

func tensorFloat64Values(input computetensor.Tensor) ([]float64, error) {
	sourceDType, bytes, err := input.RawBytes()
	if err != nil {
		return nil, err
	}

	return dtypeconvert.BytesToFloat64(sourceDType, bytes)
}
