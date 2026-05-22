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
TensorFloat32L2Norm downloads a tensor and returns its L2 norm without retaining the full vector.
*/
func TensorFloat32L2Norm(memory tensor.Backend, value tensor.Tensor) (float64, error) {
	storageDType, raw, err := memory.Download(value)

	if err != nil {
		return 0, err
	}

	if storageDType == dtype.Float32 {
		return l2NormFloat32Bytes(raw), nil
	}

	if storageDType.IsFloat() {
		elements, err := convert.BytesToFloat32(storageDType, raw)

		if err != nil {
			return 0, err
		}

		return l2NormFloat32(elements), nil
	}

	return 0, fmt.Errorf("diffusion: unsupported tensor dtype %s", storageDType)
}

func l2NormFloat32Bytes(raw []byte) float64 {
	elementCount := len(raw) / 4
	var sumSquares float64

	for index := range elementCount {
		value := math.Float32frombits(binary.LittleEndian.Uint32(raw[index*4 : index*4+4]))

		sumSquares += float64(value) * float64(value)
	}

	return math.Sqrt(sumSquares)
}

func l2NormFloat32(values []float32) float64 {
	var sumSquares float64

	for _, value := range values {
		sumSquares += float64(value) * float64(value)
	}

	return math.Sqrt(sumSquares)
}
