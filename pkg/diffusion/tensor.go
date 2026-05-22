package diffusion

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/dtype/convert"
	"github.com/theapemachine/manifesto/tensor"
)

/*
CloneTensor copies a resident tensor through download and upload on the same backend.
*/
func CloneTensor(memory tensor.Backend, source tensor.Tensor) (tensor.Tensor, error) {
	if memory == nil {
		return nil, fmt.Errorf("diffusion: tensor backend is required")
	}

	if source == nil {
		return nil, fmt.Errorf("diffusion: source tensor is required")
	}

	storageDType, raw, err := memory.Download(source)

	if err != nil {
		return nil, err
	}

	return memory.Upload(source.Shape(), storageDType, raw)
}

/*
TensorToFloat32 downloads a tensor and decodes it to float32 elements.
*/
func TensorToFloat32(memory tensor.Backend, value tensor.Tensor) ([]float32, error) {
	if memory == nil {
		return nil, fmt.Errorf("diffusion: tensor backend is required")
	}

	if value == nil {
		return nil, fmt.Errorf("diffusion: tensor is required")
	}

	storageDType, raw, err := memory.Download(value)

	if err != nil {
		return nil, err
	}

	if storageDType == dtype.Float32 {
		elementCount := len(raw) / 4
		elements := make([]float32, elementCount)

		for index := range elementCount {
			elements[index] = math.Float32frombits(
				binary.LittleEndian.Uint32(raw[index*4 : index*4+4]),
			)
		}

		return elements, nil
	}

	if storageDType.IsFloat() {
		return convert.BytesToFloat32(storageDType, raw)
	}

	return nil, fmt.Errorf("diffusion: unsupported tensor dtype %s", storageDType)
}
