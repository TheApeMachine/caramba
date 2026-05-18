package orchestrator

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	dtypeconvert "github.com/theapemachine/caramba/pkg/dtype/convert"
)

func tensorFloat64Values(value tensor.Tensor) ([]float64, error) {
	sourceDType, bytes, err := value.RawBytes()

	if err != nil {
		return nil, err
	}

	return dtypeconvert.BytesToFloat64(sourceDType, bytes)
}
